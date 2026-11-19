"""Separate procedural natural-draping garments (Top, Bottom, Shoes) fitted to CharacterSurface."""

from __future__ import annotations

import numpy as np

from frfruit.character.surface import CharacterSurface
from frfruit.geometry.mesh import MeshData


def _lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * t


def _circular_smooth(arr: np.ndarray, passes: int = 2) -> np.ndarray:
    """Apply periodic [0.25, 0.5, 0.25] circular smoothing along the last axis."""
    out = arr.copy()
    for _ in range(passes):
        out = 0.50 * out + 0.25 * np.roll(out, 1, axis=-1) + 0.25 * np.roll(out, -1, axis=-1)
    return out


def _push_clear_of_skin(
    surface: CharacterSurface,
    verts: np.ndarray,
    min_clear: float,
    iters: int = 2,
) -> np.ndarray:
    """Vectorized KDTree skin clearance push guaranteeing zero skin poke-through."""
    p = verts.copy()
    for _ in range(iters):
        p = surface.push_clear_of_skin(p, min_clear=min_clear)
    return p


def _radial_march_rings(
    surface: CharacterSurface,
    centers: np.ndarray,
    dirs: np.ndarray,
    r_lo: float,
    r_hi: float,
    iters: int = 9,
) -> np.ndarray:
    """Vectorized bisection along radial rays `centers[i] + r * dirs[i, j]` to find `sd == 0`.

    Args:
        centers: (M, 3) ring centers.
        dirs: (M, K, 3) unit ray directions per ring.
        r_lo: initial inner radius (typically inside body).
        r_hi: initial outer radius (strictly outside body).
    Returns:
        (M, K) surface radii where the skin signed distance crosses zero.
    """
    m, k, _ = dirs.shape
    lo = np.full((m, k), float(r_lo), dtype=np.float64)
    hi = np.full((m, k), float(r_hi), dtype=np.float64)

    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        pts = centers[:, None, :] + dirs * mid[:, :, None]
        d = surface.signed_distance(pts.reshape(-1, 3)).reshape(m, k)
        lo = np.where(d < 0.0, mid, lo)
        hi = np.where(d < 0.0, hi, mid)

    return 0.5 * (lo + hi)


def _loft_closed_rings(
    rings: np.ndarray,
    u_coords: np.ndarray,
    v_coords: np.ndarray,
    *,
    flip_winding: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Loft (M, K, 3) periodic rings into a clean triangulated mesh with UVs."""
    m, k, _ = rings.shape
    verts = rings.reshape(-1, 3)
    uu = np.broadcast_to(u_coords[None, :], (m, k))
    vv = np.broadcast_to(v_coords[:, None], (m, k))
    uvs = np.stack([uu, vv], axis=2).reshape(-1, 2)

    faces_list: list[tuple[int, int, int]] = []
    for ri in range(m - 1):
        r0 = ri * k
        r1 = (ri + 1) * k
        for j in range(k):
            jn = (j + 1) % k
            a, b, c, d = r0 + j, r0 + jn, r1 + jn, r1 + j
            if flip_winding:
                faces_list.append((a, c, b))
                faces_list.append((a, d, c))
            else:
                faces_list.append((a, b, c))
                faces_list.append((a, c, d))

    return verts, np.asarray(faces_list, dtype=np.int64), uvs


def _verify_outward_tube(
    verts: np.ndarray,
    faces: np.ndarray,
    axis_pts: np.ndarray,
    n_k: int,
) -> np.ndarray:
    """Ensure tube triangle normals point outward away from ring centers `axis_pts`."""
    if len(faces) == 0:
        return faces
    sample = faces[: min(len(faces), n_k * 6)]
    v0 = verts[sample[:, 0]]
    v1 = verts[sample[:, 1]]
    v2 = verts[sample[:, 2]]
    fn = np.cross(v1 - v0, v2 - v0)
    tri_c = (v0 + v1 + v2) * (1.0 / 3.0)
    ring_idx = np.clip(sample[:, 0] // n_k, 0, len(axis_pts) - 1)
    radial = tri_c - axis_pts[ring_idx]
    if float(np.mean(np.sum(fn * radial, axis=1))) < 0.0:
        return faces[:, [0, 2, 1]]
    return faces


def _build_arm_sleeve(
    surface: CharacterSurface,
    sh: np.ndarray,
    el: np.ndarray,
    side_sign: float,
    sleeve_frac: float,
    n_rings: int,
    n_segs: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build a smooth tubular sleeve along an upper arm with turned cuff hem."""
    body_scale = surface.lm.height / 1.75
    arm_vec = el - sh
    arm_len = float(np.linalg.norm(arm_vec))
    a_dir = arm_vec / max(1e-6, arm_len)

    # Orthonormal basis around arm bone axis
    ref = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    if abs(float(np.dot(a_dir, ref))) > 0.92:
        ref = np.array([0.0, -1.0, 0.0], dtype=np.float64)
    u_ax = np.cross(ref, a_dir)
    u_ax /= np.linalg.norm(u_ax)
    v_ax = np.cross(a_dir, u_ax)
    v_ax /= np.linalg.norm(v_ax)

    # Start slightly inward from shoulder joint so sleeve blends seamlessly with torso shoulder yoke
    t_vals = np.linspace(-0.14, sleeve_frac, n_rings)
    centers = sh[None, :] + (t_vals * arm_len)[:, None] * a_dir[None, :]

    thetas = np.linspace(0.0, 2.0 * np.pi, n_segs, endpoint=False)
    ring_dirs_1 = np.cos(thetas)[:, None] * u_ax[None, :] + np.sin(thetas)[:, None] * v_ax[None, :]
    dirs = np.broadcast_to(ring_dirs_1[None, :, :], (n_rings, n_segs, 3)).copy()

    r_skin_raw = _radial_march_rings(
        surface, centers, dirs, r_lo=0.015 * body_scale, r_hi=0.115 * body_scale, iters=8
    )
    med_r = np.median(r_skin_raw, axis=1, keepdims=True)
    r_skin = np.minimum(r_skin_raw, med_r * 1.28)
    r_skin = _circular_smooth(r_skin, passes=2)

    ease = np.linspace(0.0110, 0.0165, n_rings)[:, None] * body_scale
    fold = (0.0014 * body_scale) * np.sin(4.0 * thetas[None, :] + t_vals[:, None] * 6.0)
    r_cloth = r_skin + ease + fold * np.clip(t_vals[:, None] / max(0.2, sleeve_frac), 0.0, 1.0)
    r_cloth = np.maximum(r_cloth, r_skin + 0.0085 * body_scale)

    cuff_outer = r_cloth[-1:]
    cuff_lip = np.maximum(r_skin[-1:] + 0.002 * body_scale, cuff_outer - 0.006 * body_scale)
    centers_ext = np.vstack(
        [centers, centers[-1:] + 0.003 * body_scale * a_dir[None, :], centers[-1:]]
    )
    r_ext = np.vstack([r_cloth, cuff_outer - 0.002 * body_scale, cuff_lip])
    dirs_ext = np.broadcast_to(ring_dirs_1[None, :, :], (len(centers_ext), n_segs, 3))

    rings = centers_ext[:, None, :] + dirs_ext * r_ext[:, :, None]
    # Exact KDTree clearance on outer sleeve rings (skipping only innermost tucked cuff lip)
    outer_pts = rings[:-1].reshape(-1, 3)
    rings[:-1] = _push_clear_of_skin(
        surface, outer_pts, min_clear=0.0085 * body_scale, iters=2
    ).reshape(len(centers_ext) - 1, n_segs, 3)

    u_c = np.linspace(0.0, 1.0, n_segs, endpoint=False)
    v_c = np.linspace(0.0, 1.0, len(centers_ext))
    verts, faces, uvs = _loft_closed_rings(rings, u_c, v_c)
    faces = _verify_outward_tube(verts, faces, centers_ext, n_segs)
    return verts, faces, uvs


def build_top_mesh(surface: CharacterSurface) -> MeshData:
    """Build a natural-draping tailored Top with shoulder yoke, crew-neck collar & sleeves."""
    cstyle = surface.spec.materials.clothing_style
    if cstyle == "none":
        return MeshData(vertices=np.zeros((0, 3)), faces=np.zeros((0, 3), dtype=np.int64))

    lm = surface.lm
    b = surface.spec.body
    body_scale = lm.height / 1.75
    g = float(b.gender)
    m = float(b.muscularity)
    fat = float(b.body_fat)
    fem_factor = max(0.0, -g)
    q = surface.spec.quality

    n_body = 16 if q == "preview" else 22
    n_segs = 32 if q == "preview" else 48
    thetas = np.linspace(0.0, 2.0 * np.pi, n_segs, endpoint=False)
    # Convention: theta=0 is +X, theta=3*pi/2 (-pi/2) is front (-Y), theta=pi/2 is back (+Y)
    cos_t = np.cos(thetas)
    sin_t = np.sin(thetas)

    z_hem = (
        float(lm.pelvis[2]) - 0.075 * body_scale
        if cstyle == "tunic"
        else float(lm.pelvis[2]) + 0.015 * body_scale
    )
    neck_base_z = float(lm.neck_base[2])
    z_sh = float(lm.shoulder_left[2])
    sh_x = abs(float(lm.shoulder_left[0]))

    front_weight = np.clip(-sin_t, 0.0, 1.0) ** 1.6
    back_weight = np.clip(sin_t, 0.0, 1.0) ** 1.6

    # Top of trunk section aligns cleanly with shoulder crest on sides and upper chest/back
    scoop_depth = (
        0.044 + 0.012 * fem_factor if cstyle == "athletic" else 0.026 + 0.008 * fem_factor
    ) * body_scale
    z_side_crest = max(z_sh + 0.038 * body_scale, neck_base_z - 0.018 * body_scale)
    z_sh_crest = (
        neck_base_z
        + (z_side_crest - neck_base_z) * (cos_t**2)
        - scoop_depth * front_weight
        + (0.006 * body_scale) * back_weight
    )

    s_frac = np.linspace(0.0, 1.0, n_body)
    z_grid = np.empty((n_body, n_segs), dtype=np.float64)
    z_top_mean = float(np.mean(z_sh_crest))
    for ri in range(n_body):
        sf = float(s_frac[ri])
        crest_blend = max(0.0, (sf - 0.65) / 0.35) ** 1.4
        z_plain = _lerp(z_hem, z_top_mean, sf)
        z_grid[ri] = _lerp(z_plain, z_sh_crest, crest_blend)

    centers = np.zeros((n_body, 3), dtype=np.float64)
    centers[:, 1] = 0.006 * body_scale
    centers[:, 2] = np.mean(z_grid, axis=1)

    lo = np.full((n_body, n_segs), 0.035 * body_scale, dtype=np.float64)
    lat_cap = sh_x * (0.94 if cstyle == "athletic" else 1.06)
    r_max_xy = np.where(
        np.abs(cos_t)[None, :] > 0.35,
        np.minimum(0.28 * body_scale, lat_cap / np.maximum(np.abs(cos_t)[None, :], 0.35)),
        0.24 * body_scale,
    )
    armpit_z = z_sh - 0.102 * body_scale
    below_armpit = z_grid < armpit_z
    r_max_xy = np.where(below_armpit, np.minimum(r_max_xy, sh_x * 0.92), r_max_xy)
    hi = r_max_xy.copy()

    for _ in range(9):
        mid = 0.5 * (lo + hi)
        px = mid * cos_t[None, :]
        py = centers[:, 1:2] + mid * sin_t[None, :]
        pts = np.stack([px, py, z_grid], axis=2).reshape(-1, 3)
        d = surface.signed_distance(pts).reshape(n_body, n_segs)
        lo = np.where(d < 0.0, mid, lo)
        hi = np.where(d < 0.0, hi, mid)

    r_skin_raw = 0.5 * (lo + hi)
    r_skin = _circular_smooth(r_skin_raw, passes=2)

    # Convert to Cartesian (x, y) for Front-Chest Convex Bridging & Vertical Gravity Drape
    x_cloth = r_skin * cos_t[None, :]
    y_cloth = centers[:, 1:2] + r_skin * sin_t[None, :]

    # 1) Front-Chest Convex Bridge: eliminate vacuum-cling between pectorals / bust
    front_mask = sin_t < -0.15
    for ri in range(n_body):
        if z_grid[ri, 0] < float(lm.waist[2]) - 0.02 * body_scale:
            continue
        f_idx = np.nonzero(front_mask)[0]
        xf = x_cloth[ri, f_idx]
        yf = y_cloth[ri, f_idx]
        l_pts = xf >= 0.01 * body_scale
        r_pts = xf <= -0.01 * body_scale
        if np.any(l_pts) and np.any(r_pts):
            l_i = np.argmin(yf[l_pts])
            r_i = np.argmin(yf[r_pts])
            xl_pk, yl_pk = float(xf[l_pts][l_i]), float(yf[l_pts][l_i])
            xr_pk, yr_pk = float(xf[r_pts][r_i]), float(yf[r_pts][r_i])
            if xl_pk - xr_pk > 0.02 * body_scale:
                between = (xf >= xr_pk) & (xf <= xl_pk)
                if np.any(between):
                    t_ch = (xf[between] - xr_pk) / (xl_pk - xr_pk)
                    y_chord = _lerp(yr_pk, yl_pk, t_ch) - 0.003 * body_scale * np.sin(np.pi * t_ch)
                    y_cloth[ri, f_idx[between]] = np.minimum(y_cloth[ri, f_idx[between]], y_chord)

    # 2) Vertical Gravity Drape from Chest Apex down to Hem
    chest_ri = int(np.argmin(np.abs(centers[:, 2] - float(lm.chest[2]))))
    drape_strength = 0.72 if cstyle == "tunic" else (0.58 if fem_factor > 0.2 else 0.45)
    for ri in range(chest_ri - 1, -1, -1):
        front_w = np.clip(-sin_t, 0.0, 1.0)
        y_hang = np.minimum(
            y_cloth[ri], _lerp(y_cloth[ri], y_cloth[ri + 1] + 0.0028 * body_scale, drape_strength)
        )
        y_cloth[ri] = _lerp(y_cloth[ri], y_hang, front_w)

    dx = x_cloth
    dy = y_cloth - centers[:, 1:2]
    r_draped = np.sqrt(dx * dx + dy * dy)
    r_draped = _circular_smooth(r_draped, passes=2)

    ease_hem = (
        0.028 if cstyle == "tunic" else (0.022 if cstyle == "casual" else 0.017)
    ) * body_scale
    ease_top = 0.012 * body_scale
    ease_v = _lerp(ease_hem, ease_top, s_frac**0.85)[:, None]
    fold_env = (1.0 - s_frac) ** 1.3
    folds = (
        (0.0028 if cstyle == "tunic" else 0.0018)
        * body_scale
        * fold_env[:, None]
        * (0.65 * np.cos(5.0 * thetas[None, :] + 0.4) + 0.35 * np.sin(8.0 * thetas[None, :]))
    )
    flare = (0.020 if cstyle == "tunic" else 0.006) * body_scale * ((1.0 - s_frac) ** 2.0)[:, None]
    r_final = r_draped + ease_v + folds + flare

    # Never clip unsmoothed raw skin peaks!
    r_final = np.maximum(r_final, np.maximum(r_skin_raw, r_skin) + 0.010 * body_scale)

    # Assemble body rings (n_body, n_segs, 3)
    body_rings = np.stack(
        [
            r_final * cos_t[None, :],
            centers[:, 1:2] + r_final * sin_t[None, :],
            z_grid,
        ],
        axis=2,
    )

    # --- 3. Anatomical Shoulder-Yoke Cap & Snug Crew-Neck Collar ---
    neck_r_est = 0.053 * (1.0 + 0.11 * g + 0.07 * m + 0.08 * fat) * body_scale * 1.15
    neck_cy = float(lm.neck_base[1])
    if cstyle == "athletic":
        r_col_x = max(neck_r_est + 0.030 * body_scale, sh_x * 0.52)
        r_col_y = neck_r_est + 0.018 * body_scale
    else:
        r_col_x = neck_r_est + 0.013 * body_scale
        r_col_y = neck_r_est + 0.015 * body_scale

    z_col_target = (
        (neck_base_z - 0.004 * body_scale)
        + (0.005 * body_scale) * (cos_t**2)
        - front_weight * (scoop_depth + 0.008 * body_scale)
        + back_weight * (0.007 * body_scale)
    )
    collar_ring = np.column_stack(
        [
            r_col_x * cos_t,
            neck_cy + r_col_y * sin_t,
            z_col_target,
        ]
    )

    # Interpolate 6 smooth shoulder-yoke rings from outer shoulder crest inward to crew-neck collar
    top_body_ring = body_rings[-1]
    yoke_fracs = [0.14, 0.32, 0.52, 0.72, 0.88, 1.0]
    yoke_rings = np.empty((len(yoke_fracs), n_segs, 3), dtype=np.float64)
    z_collar_cap = neck_base_z + 0.056 * body_scale
    for yi, yf in enumerate(yoke_fracs):
        # Smooth ease-in-out curve over trapezius dome
        smooth_f = yf * yf * (3.0 - 2.0 * yf)
        pts_y = top_body_ring * (1.0 - smooth_f) + collar_ring * smooth_f
        # Conform snugly onto trapezius/clavicle surface (+0.011m clearance, zero clipping!)
        target_clr = 0.011 * body_scale
        for _ in range(2):
            surf_p, surf_n, dists = surface.query_nearest(pts_y)
            d_y = np.where(np.sum((pts_y - surf_p) * surf_n, axis=1) >= 0.0, dists, -dists)
            diff = d_y - target_clr
            step = np.where(
                diff < 0.0,
                np.clip(diff, -0.028, 0.0),
                np.where(yf < 0.90, np.clip(diff - 0.004 * body_scale, 0.0, 0.014), 0.0),
            )
            pts_y = pts_y - step[:, None] * surf_n
        pts_y[:, 2] = np.minimum(pts_y[:, 2], z_collar_cap)
        yoke_rings[yi] = pts_y

    # Rolled crew-neck collar rib (2 rings) + turned bottom hem lip (1 ring)
    hem_in = body_rings[0].copy()
    hem_rad = np.sqrt(hem_in[:, 0] ** 2 + (hem_in[:, 1] - centers[0, 1]) ** 2)
    hem_shrink = np.maximum(0.88, (hem_rad - 0.006 * body_scale) / np.maximum(hem_rad, 1e-6))
    hem_in[:, 0] *= hem_shrink
    hem_in[:, 1] = centers[0, 1] + (hem_in[:, 1] - centers[0, 1]) * hem_shrink
    hem_in[:, 2] += 0.002 * body_scale

    col_rim = yoke_rings[-1].copy()
    col_rim[:, 2] = np.minimum(
        col_rim[:, 2] + 0.0025 * body_scale, z_collar_cap + 0.003 * body_scale
    )
    col_in = yoke_rings[-1].copy()
    col_dx = col_in[:, 0]
    col_dy = col_in[:, 1] - neck_cy
    col_r = np.maximum(np.sqrt(col_dx * col_dx + col_dy * col_dy), 1e-6)
    col_shrink = np.maximum(0.84, (col_r - 0.005 * body_scale) / col_r)
    col_in[:, 0] = col_dx * col_shrink
    col_in[:, 1] = neck_cy + col_dy * col_shrink
    col_in[:, 2] -= 0.001 * body_scale

    all_rings = np.vstack(
        [hem_in[None, :, :], body_rings, yoke_rings, col_rim[None, :, :], col_in[None, :, :]]
    )
    # Final exact clearance push on all outer rings (0.0% chest/shoulder poke-through)
    outer_slice = all_rings[1:-1]
    n_out_r = len(outer_slice)
    all_rings[1:-1] = _push_clear_of_skin(
        surface, outer_slice.reshape(-1, 3), min_clear=0.0082 * body_scale, iters=2
    ).reshape(n_out_r, n_segs, 3)

    u_c = np.linspace(0.0, 1.0, n_segs, endpoint=False)
    v_c = np.linspace(0.0, 1.0, len(all_rings))
    v_torso, f_torso, uv_torso = _loft_closed_rings(all_rings, u_c, v_c)
    axis_all = np.mean(all_rings, axis=1)
    f_torso = _verify_outward_tube(v_torso, f_torso, axis_all, n_segs)

    all_v = [v_torso]
    all_f = [f_torso]
    all_uv = [uv_torso]
    v_off = len(v_torso)

    # Add smooth tubular sleeves for casual (short-sleeve) and tunic (3/4 sleeve)
    if cstyle != "athletic":
        sleeve_frac = 0.80 if cstyle == "tunic" else 0.44
        n_sl_r = 7 if q == "preview" else 10
        n_sl_s = 20 if q == "preview" else 28
        for sh_pt, el_pt, side_s in (
            (lm.shoulder_left, lm.elbow_left, 1.0),
            (lm.shoulder_right, lm.elbow_right, -1.0),
        ):
            vs, fs, uvs = _build_arm_sleeve(
                surface, sh_pt, el_pt, side_s, sleeve_frac, n_rings=n_sl_r, n_segs=n_sl_s
            )
            all_v.append(vs)
            all_f.append(fs + v_off)
            all_uv.append(uvs)
            v_off += len(vs)

    return MeshData(
        vertices=np.vstack(all_v),
        faces=np.vstack(all_f),
        uvs=np.vstack(all_uv),
    )


def build_bottom_mesh(surface: CharacterSurface) -> MeshData:
    """Build natural tailored straight-cut Trousers or Athletic Shorts with waistband & cuffs."""
    cstyle = surface.spec.materials.clothing_style
    if cstyle == "none":
        return MeshData(vertices=np.zeros((0, 3)), faces=np.zeros((0, 3), dtype=np.int64))

    lm = surface.lm
    body_scale = lm.height / 1.75
    q = surface.spec.quality

    all_v: list[np.ndarray] = []
    all_f: list[np.ndarray] = []
    all_uv: list[np.ndarray] = []
    v_off = 0

    def _append(v: np.ndarray, f: np.ndarray, uv: np.ndarray) -> None:
        nonlocal v_off
        all_v.append(v)
        all_f.append(f + v_off)
        all_uv.append(uv)
        v_off += len(v)

    # --- 1. Upper Pelvis / Waistband Shell ---
    n_p_rings = 10 if q == "preview" else 14
    n_p_segs = 30 if q == "preview" else 44
    thetas = np.linspace(0.0, 2.0 * np.pi, n_p_segs, endpoint=False)
    cos_t = np.cos(thetas)
    sin_t = np.sin(thetas)

    z_waist = float(lm.waist[2]) + 0.012 * body_scale
    z_crotch = float(lm.hip_left[2]) - 0.068 * body_scale
    z_p = np.linspace(z_crotch, z_waist, n_p_rings)

    centers_p = np.column_stack([np.zeros(n_p_rings), np.full(n_p_rings, 0.008 * body_scale), z_p])
    dirs_p = np.broadcast_to(
        np.column_stack([cos_t, sin_t, np.zeros(n_p_segs)])[None, :, :],
        (n_p_rings, n_p_segs, 3),
    ).copy()

    hip_x = abs(float(lm.hip_left[0]))
    r_cap = max(hip_x * 2.10, 0.220 * body_scale)
    r_skin_p_raw = _radial_march_rings(
        surface, centers_p, dirs_p, r_lo=0.03 * body_scale, r_hi=r_cap, iters=9
    )
    r_skin_p = _circular_smooth(r_skin_p_raw, passes=2)

    ease_p = np.linspace(0.014, 0.008, n_p_rings)[:, None] * body_scale
    r_pelv = np.maximum(r_skin_p + ease_p, r_skin_p_raw + 0.0065 * body_scale)
    r_wb_in = np.maximum(r_skin_p[-1:] + 0.001 * body_scale, r_pelv[-1:] - 0.006 * body_scale)
    r_p_all = np.vstack([r_pelv, r_wb_in])
    z_p_all = np.concatenate([z_p, [z_waist + 0.002 * body_scale]])
    cy_p_all = np.full(len(z_p_all), 0.008 * body_scale)

    rings_p = np.stack(
        [
            r_p_all * cos_t[None, :],
            cy_p_all[:, None] + r_p_all * sin_t[None, :],
            np.broadcast_to(z_p_all[:, None], r_p_all.shape),
        ],
        axis=2,
    )
    rings_p[:-1] = _push_clear_of_skin(
        surface, rings_p[:-1].reshape(-1, 3), min_clear=0.0058 * body_scale, iters=2
    ).reshape(n_p_rings, n_p_segs, 3)

    vp, fp, uvp = _loft_closed_rings(
        rings_p,
        np.linspace(0.0, 1.0, n_p_segs, endpoint=False),
        np.linspace(0.0, 0.35, len(rings_p)),
    )
    fp = _verify_outward_tube(
        vp, fp, np.column_stack([np.zeros(len(rings_p)), cy_p_all, z_p_all]), n_p_segs
    )
    _append(vp, fp, uvp)

    # --- 2. Left & Right Straight/Tapered Trouser Legs (or Athletic Shorts) ---
    is_shorts = cstyle == "athletic"
    z_leg_top = float(lm.hip_left[2]) + 0.042 * body_scale
    z_leg_bot = (
        float(lm.knee_left[2]) + 0.045 * body_scale
        if is_shorts
        else float(lm.ankle_left[2]) + 0.008 * body_scale
    )
    n_l_rings = 10 if (is_shorts or q == "preview") else 18
    n_l_segs = 22 if q == "preview" else 30
    l_thetas = np.linspace(0.0, 2.0 * np.pi, n_l_segs, endpoint=False)
    l_cos = np.cos(l_thetas)
    l_sin = np.sin(l_thetas)

    z_leg = np.linspace(z_leg_bot, z_leg_top, n_l_rings)
    for hp, kn, ak in (
        (lm.hip_left, lm.knee_left, lm.ankle_left),
        (lm.hip_right, lm.knee_right, lm.ankle_right),
    ):
        cx_l = np.empty(n_l_rings, dtype=np.float64)
        cy_l = np.empty(n_l_rings, dtype=np.float64)
        for ri, zv in enumerate(z_leg):
            if zv >= float(kn[2]):
                t = (zv - float(kn[2])) / max(1e-5, float(hp[2] - kn[2]))
                t = float(np.clip(t, 0.0, 1.15))
                cx_l[ri] = _lerp(float(kn[0]), float(hp[0]), t)
                cy_l[ri] = _lerp(float(kn[1]), float(hp[1]), t)
            else:
                t = (zv - float(ak[2])) / max(1e-5, float(kn[2] - ak[2]))
                t = float(np.clip(t, 0.0, 1.0))
                cx_l[ri] = _lerp(float(ak[0]), float(kn[0]), t)
                cy_l[ri] = _lerp(float(ak[1]), float(kn[1]), t)

        centers_l = np.column_stack([cx_l, cy_l, z_leg])
        dirs_l = np.broadcast_to(
            np.column_stack([l_cos, l_sin, np.zeros(n_l_segs)])[None, :, :],
            (n_l_rings, n_l_segs, 3),
        ).copy()

        r_hi_max = np.full((n_l_rings, n_l_segs), 0.138 * body_scale, dtype=np.float64)
        pointing_in = (l_cos[None, :] * np.sign(cx_l[:, None])) < -0.15
        dist_to_mid = np.abs(cx_l[:, None]) / np.maximum(np.abs(l_cos[None, :]), 0.15)
        r_hi_max = np.where(
            pointing_in, np.minimum(r_hi_max, dist_to_mid + 0.006 * body_scale), r_hi_max
        )

        lo_l = np.full((n_l_rings, n_l_segs), 0.016 * body_scale, dtype=np.float64)
        hi_l = r_hi_max.copy()
        for _ in range(9):
            mid = 0.5 * (lo_l + hi_l)
            pts = centers_l[:, None, :] + dirs_l * mid[:, :, None]
            d = surface.signed_distance(pts.reshape(-1, 3)).reshape(n_l_rings, n_l_segs)
            lo_l = np.where(d < 0.0, mid, lo_l)
            hi_l = np.where(d < 0.0, hi_l, mid)

        r_skin_l_raw = 0.5 * (lo_l + hi_l)
        r_skin_l = _circular_smooth(r_skin_l_raw, passes=2)

        s_leg = np.linspace(0.0, 1.0, n_l_rings)[:, None]  # 0 = cuff, 1 = hip
        min_tailored_r = _lerp(0.058 if not is_shorts else 0.068, 0.082, s_leg) * body_scale
        ease_l = _lerp(0.017, 0.012, s_leg) * body_scale
        r_cloth_l = np.maximum(r_skin_l + ease_l, min_tailored_r)

        below_crotch = z_leg[:, None] < z_crotch - 0.008 * body_scale
        inner_lim = (
            np.abs(cx_l[:, None]) / np.maximum(np.abs(l_cos[None, :]), 0.25) - 0.0005 * body_scale
        )
        r_cloth_l = np.where(
            pointing_in & below_crotch,
            np.minimum(r_cloth_l, np.maximum(r_skin_l_raw + 0.005 * body_scale, inner_lim)),
            r_cloth_l,
        )

        fold_l = (0.0018 * body_scale) * np.cos(4.0 * l_thetas[None, :] + s_leg * 4.0)
        r_cloth_l = np.maximum(r_skin_l_raw + 0.0055 * body_scale, r_cloth_l + fold_l)

        r_cuff_in = np.maximum(
            r_skin_l[0:1] + 0.001 * body_scale, r_cloth_l[0:1] - 0.007 * body_scale
        )
        r_l_all = np.vstack([r_cuff_in, r_cloth_l])
        c_l_all = np.vstack(
            [centers_l[0:1] + np.array([[0.0, 0.0, 0.003 * body_scale]]), centers_l]
        )
        dirs_l_all = np.broadcast_to(dirs_l[0:1], (len(c_l_all), n_l_segs, 3))

        rings_l = c_l_all[:, None, :] + dirs_l_all * r_l_all[:, :, None]
        rings_l[1:] = _push_clear_of_skin(
            surface, rings_l[1:].reshape(-1, 3), min_clear=0.0055 * body_scale, iters=2
        ).reshape(n_l_rings, n_l_segs, 3)

        vl, fl, uvl = _loft_closed_rings(
            rings_l,
            np.linspace(0.0, 1.0, n_l_segs, endpoint=False),
            np.linspace(0.35, 1.0, len(rings_l)),
        )
        fl = _verify_outward_tube(vl, fl, c_l_all, n_l_segs)
        _append(vl, fl, uvl)

    return MeshData(
        vertices=np.vstack(all_v),
        faces=np.vstack(all_f),
        uvs=np.vstack(all_uv),
    )


def build_shoes_mesh(surface: CharacterSurface) -> MeshData:
    """Build clean sculpted enclosed parametric sneakers fitted to both feet (zero poke-through)."""
    if surface.spec.materials.clothing_style == "none":
        return MeshData(vertices=np.zeros((0, 3)), faces=np.zeros((0, 3), dtype=np.int64))

    lm = surface.lm
    body_scale = lm.height / 1.75
    lt = float(surface.spec.body.limb_thickness) * body_scale
    q = surface.spec.quality

    n_segs = 20 if q == "preview" else 28
    thetas = np.linspace(0.0, 2.0 * np.pi, n_segs, endpoint=False)
    cos_t = np.cos(thetas)
    sin_t = np.sin(thetas)

    all_v: list[np.ndarray] = []
    all_f: list[np.ndarray] = []
    all_uv: list[np.ndarray] = []
    v_off = 0

    for ak, to in ((lm.ankle_left, lm.toe_left), (lm.ankle_right, lm.toe_right)):
        x_mid = 0.5 * (float(ak[0]) + float(to[0]))
        y_heel = float(ak[1]) + 0.058 * body_scale
        y_toe = float(to[1]) - 0.044 * body_scale
        y_mid = 0.5 * (y_heel + y_toe)
        half_len = 0.5 * (y_heel - y_toe)
        half_w = 0.056 * lt
        ankle_z = float(ak[2]) + 0.028 * body_scale

        # Lofted enclosed sneaker profile clearing the entire foot mesh + ball + instep
        levels = [
            (0.001, 0.90, 0.95, 0.93, 0.0),
            (0.005, 1.02, 1.02, 1.01, 0.0),
            (0.024 * body_scale, 1.04, 1.02, 1.02, 0.0),  # Top of rubber sole
            (0.046 * body_scale, 1.00, 0.98, 0.98, 0.004 * body_scale),  # Enclosed toe box
            (0.068 * body_scale, 0.95, 0.86, 0.95, 0.016 * body_scale),  # Vamp / forefoot dome
            (0.082 * body_scale, 0.88, 0.62, 0.90, 0.032 * body_scale),  # Tongue / instep
            (ankle_z, 0.82, 0.44, 0.86, 0.044 * body_scale),  # Padded ankle collar
            (ankle_z - 0.004 * body_scale, 0.70, 0.34, 0.74, 0.044 * body_scale),  # Inner collar
        ]

        rings = np.empty((len(levels), n_segs, 3), dtype=np.float64)
        axis_pts = np.empty((len(levels), 3), dtype=np.float64)
        for li, (zv, xs, yf_s, yb_s, y_sh) in enumerate(levels):
            rx = half_w * xs
            ry = np.where(sin_t < 0.0, half_len * yf_s, half_len * yb_s)
            w_mod = np.where(sin_t > 0.2, 0.84, 1.0)
            px = x_mid + rx * w_mod * cos_t
            py = (y_mid + y_sh) + ry * sin_t
            rings[li, :, 0] = px
            rings[li, :, 1] = py
            rings[li, :, 2] = zv
            axis_pts[li] = (x_mid, y_mid + y_sh, zv)

        # Guarantee 0.0% foot/instep skin poke-through on upper sneaker shell (levels 2..-1)
        upper_slice = rings[2:-1]
        n_up = len(upper_slice)
        rings[2:-1] = _push_clear_of_skin(
            surface, upper_slice.reshape(-1, 3), min_clear=0.0075 * body_scale, iters=2
        ).reshape(n_up, n_segs, 3)

        vs, fs, uvs = _loft_closed_rings(
            rings,
            np.linspace(0.0, 1.0, n_segs, endpoint=False),
            np.linspace(0.0, 1.0, len(levels)),
        )
        fs = _verify_outward_tube(vs, fs, axis_pts, n_segs)

        # Cap flat bottom sole cleanly
        sole_c_idx = len(vs)
        vs = np.vstack([vs, np.array([[x_mid, y_mid, 0.001]])])
        uvs = np.vstack([uvs, np.array([[0.5, 0.0]])])
        sole_tris = np.array(
            [(sole_c_idx, (j + 1) % n_segs, j) for j in range(n_segs)], dtype=np.int64
        )
        fs = np.vstack([fs, sole_tris])

        all_v.append(vs)
        all_f.append(fs + v_off)
        all_uv.append(uvs)
        v_off += len(vs)

    return MeshData(
        vertices=np.vstack(all_v),
        faces=np.vstack(all_f),
        uvs=np.vstack(all_uv),
    )
