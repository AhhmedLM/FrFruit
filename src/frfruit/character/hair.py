"""High-performance vectorized parametric 3D hair groom system (Scalp + Layered Locks)."""

from __future__ import annotations

import numpy as np

from frfruit.character.surface import CharacterSurface
from frfruit.geometry.mesh import MeshData


def _lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * t


def _hairline_theta_max(phi: np.ndarray, style: str, gender: float) -> np.ndarray:
    """Anatomical hairline polar angle from crown (+Z) as a function of azimuth phi=atan2(x,-y).

    phi = 0 is center forehead, +/-pi/2 is temples/ears, +/-pi is back nape.
    """
    ap = np.abs(phi)
    # Center forehead hairline sits slightly higher on males (gentle natural temple contour)
    forehead_th = 0.380 - 0.012 * max(0.0, gender)
    if style == "short":
        forehead_th -= 0.006
    # Subtle natural temple contour (fuller coverage on sides & temples)
    temple_recess = -0.012 * max(0.0, gender + 0.1) * np.exp(-(((ap - 0.21 * np.pi) / 0.14) ** 2))
    # Neat sideburn in front of ear, gentle arch over ear, full nape drop behind ear
    side_curve = 0.145 * np.exp(-(((ap - 0.39 * np.pi) / 0.17) ** 2))
    ear_arch = -0.018 * np.exp(-(((ap - 0.53 * np.pi) / 0.09) ** 2))
    nape_drop = 0.42 * np.power(np.clip((ap - 0.34 * np.pi) / (0.66 * np.pi), 0.0, 1.0), 1.05)
    th_clip = np.clip(forehead_th + temple_recess + side_curve + ear_arch + nape_drop, 0.34, 0.80)
    return th_clip * np.pi


def _build_scalp_shell(
    surface: CharacterSurface,
    n_rings: int = 12,
    n_segs: int = 32,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build a close-fitting scalp underlayer shell projected onto the cranium surface."""
    lm = surface.lm
    hh = lm.head_half_h
    style = surface.spec.materials.hair_style
    g = float(surface.spec.body.gender)

    # Center inside upper cranium
    c_scalp = lm.head_center + np.array([0.0, 0.10 * hh, 0.20 * hh], dtype=np.float64)
    r_init = np.array([0.74 * hh, 0.85 * hh, 0.78 * hh], dtype=np.float64)

    phis = np.linspace(-np.pi, np.pi, n_segs, endpoint=False)
    th_max_all = _hairline_theta_max(phis, style, g)

    # Vertices: 1 crown pole + n_rings * n_segs ring points
    raw_pts = np.empty((1 + n_rings * n_segs, 3), dtype=np.float64)
    uvs = np.empty((1 + n_rings * n_segs, 2), dtype=np.float64)

    raw_pts[0] = c_scalp + np.array([0.0, 0.0, r_init[2]])
    uvs[0] = (0.5, 0.05)

    for ri in range(1, n_rings + 1):
        frac = ri / n_rings
        th = th_max_all * frac
        st = np.sin(th)
        ct = np.cos(th)
        dx = st * np.sin(phis)
        dy = -st * np.cos(phis)
        dz = ct
        idx0 = 1 + (ri - 1) * n_segs
        raw_pts[idx0 : idx0 + n_segs, 0] = c_scalp[0] + dx * r_init[0]
        raw_pts[idx0 : idx0 + n_segs, 1] = c_scalp[1] + dy * r_init[1]
        raw_pts[idx0 : idx0 + n_segs, 2] = c_scalp[2] + dz * r_init[2]
        # Map scalp to root-tone lower V band so it darkens nicely underneath clumps
        uvs[idx0 : idx0 + n_segs, 0] = (phis + np.pi) / (2.0 * np.pi)
        uvs[idx0 : idx0 + n_segs, 1] = 0.04 + 0.22 * frac

    offset = 0.0045 * (hh / 0.12)
    verts, _ = surface.project_to_surface(raw_pts, offset=offset)

    # Taper outer hairline rim slightly inward so scalp edge blends smoothly into skin
    rim_start = 1 + (n_rings - 1) * n_segs
    rim_dir = verts[rim_start:] - c_scalp[None, :]
    rim_dir /= np.maximum(np.linalg.norm(rim_dir, axis=1, keepdims=True), 1e-9)
    verts[rim_start:] -= rim_dir * (0.0020 * (hh / 0.12))

    faces_list: list[tuple[int, int, int]] = []
    # Crown fan (CCW outward viewed from +Z)
    for j in range(n_segs):
        jn = (j + 1) % n_segs
        faces_list.append((0, 1 + j, 1 + jn))

    for ri in range(n_rings - 1):
        r0 = 1 + ri * n_segs
        r1 = 1 + (ri + 1) * n_segs
        for j in range(n_segs):
            jn = (j + 1) % n_segs
            a, b, c, d = r0 + j, r0 + jn, r1 + jn, r1 + j
            faces_list.append((a, d, c))
            faces_list.append((a, c, b))

    faces = np.asarray(faces_list, dtype=np.int64)
    # Verify outward orientation
    v0 = verts[faces[:, 0]]
    v1 = verts[faces[:, 1]]
    v2 = verts[faces[:, 2]]
    fn = np.cross(v1 - v0, v2 - v0)
    dots = np.sum(fn * ((v0 + v1 + v2) * (1.0 / 3.0) - c_scalp[None, :]), axis=1)
    if float(np.mean(dots)) < 0.0:
        faces = faces[:, [0, 2, 1]]
    return verts, faces, uvs


def _build_clump_meshes_batch(
    spines: np.ndarray,
    nrms: np.ndarray,
    half_widths: np.ndarray,
    thicknesses: np.ndarray,
    u_offsets: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Vectorized extrusion of N volumetric 4-sided diamond/arched 3D hair locks.

    Args:
        spines: (N, S, 3) world positions along each lock spine.
        nrms: (N, S, 3) outward surface/clump normal hints.
        half_widths: (N, S) half-width of each lock along spine.
        thicknesses: (N, S) arch thickness of each lock along spine.
        u_offsets: (N,) per-lock horizontal texture offset for strand diversity.
    """
    n_locks, n_s, _ = spines.shape
    # Compute smooth tangents along axis=1
    tang = np.empty_like(spines)
    tang[:, 0, :] = spines[:, 1, :] - spines[:, 0, :]
    tang[:, -1, :] = spines[:, -1, :] - spines[:, -2, :]
    if n_s > 2:
        tang[:, 1:-1, :] = spines[:, 2:, :] - spines[:, :-2, :]
    t_len = np.linalg.norm(tang, axis=2, keepdims=True)
    tang = tang / np.where(t_len < 1e-9, 1.0, t_len)

    # Orthonormalize outward normal relative to tangent
    nrm = nrms - np.sum(nrms * tang, axis=2, keepdims=True) * tang
    n_len = np.linalg.norm(nrm, axis=2, keepdims=True)
    fallback = np.zeros_like(nrm)
    fallback[:, :, 1] = -1.0
    nrm = np.where(n_len < 1e-6, fallback, nrm / np.where(n_len < 1e-6, 1.0, n_len))

    side = np.cross(nrm, tang)
    s_len = np.linalg.norm(side, axis=2, keepdims=True)
    side = side / np.where(s_len < 1e-9, 1.0, s_len)

    # 4 cross-section vertices per ring: left, outer ridge, right, inner ridge
    hw = half_widths[:, :, None]
    th = thicknesses[:, :, None]
    v_left = spines - side * hw
    v_top = spines + nrm * th
    v_right = spines + side * hw
    v_bot = spines - nrm * (th * 0.42)

    # Stack to shape (N, S, 4, 3) -> reshape to (N * S * 4, 3)
    verts = np.stack([v_left, v_top, v_right, v_bot], axis=2).reshape(-1, 3)

    # Continuous UVs: u across clump width + per-lock jitter, v from 0 (root) to 1 (tip)
    v_lin = np.linspace(0.0, 1.0, n_s, dtype=np.float64)[None, :, None]
    u_base = u_offsets[:, None, None]
    u_pat = np.array([0.0, 0.14, 0.28, 0.14], dtype=np.float64)[None, None, :]
    uu = np.mod(u_base + u_pat, 1.0)
    vv = np.broadcast_to(v_lin, (n_locks, n_s, 4))
    uvs = np.stack([np.broadcast_to(uu, (n_locks, n_s, 4)), vv], axis=3).reshape(-1, 2)

    # Build triangle indices for all locks in one vectorized pass
    lock_base = (np.arange(n_locks, dtype=np.int64) * (n_s * 4))[:, None, None]
    ring_i = np.arange(n_s - 1, dtype=np.int64)[None, :, None]
    r0 = lock_base + ring_i * 4
    r1 = r0 + 4

    quads_t1 = []
    quads_t2 = []
    for k in range(4):
        kn = (k + 1) % 4
        a = r0 + k
        b = r0 + kn
        c = r1 + kn
        d = r1 + k
        quads_t1.append(np.concatenate([a, d, c], axis=2))
        quads_t2.append(np.concatenate([a, c, b], axis=2))

    faces = np.vstack([q.reshape(-1, 3) for q in quads_t1] + [q.reshape(-1, 3) for q in quads_t2])
    return verts, faces, uvs


def build_hair_mesh(surface: CharacterSurface) -> MeshData:
    """Synthesize a layered volumetric parametric 3D hair groom fitted to `surface`."""
    mat = surface.spec.materials
    style = mat.hair_style
    if style == "none":
        return MeshData(vertices=np.zeros((0, 3)), faces=np.zeros((0, 3), dtype=np.int64))

    lm = surface.lm
    hh = lm.head_half_h
    scale = hh / 0.12
    g = float(surface.spec.body.gender)
    q = surface.spec.quality

    h_len = float(mat.hair_length)
    h_vol = float(mat.hair_volume)
    h_curl = float(mat.hair_curl)
    h_part = float(mat.hair_part)

    # 1. Close-fitting scalp base shell
    scalp_v, scalp_f, scalp_uv = _build_scalp_shell(
        surface,
        n_rings=9 if q == "preview" else 13,
        n_segs=24 if q == "preview" else 34,
    )

    # 2. Seed lock root positions across anatomical scalp tiers
    c_scalp = lm.head_center + np.array([0.0, 0.10 * hh, 0.20 * hh], dtype=np.float64)
    r_init = np.array([0.74 * hh, 0.85 * hh, 0.78 * hh], dtype=np.float64)

    rng = np.random.default_rng(4219)
    # Tier counts scale smoothly by quality tier while maintaining rich full silhouette
    q_mult = 0.68 if q == "preview" else (1.15 if q == "high" else 1.0)

    if style == "short":
        tier_specs = [
            (0.16, int(18 * q_mult), 0.90),
            (0.40, int(28 * q_mult), 0.94),
            (0.66, int(36 * q_mult), 0.84),
            (0.90, int(36 * q_mult), 0.66),
        ]
        n_seg = 7 if q == "preview" else 9
    elif style == "swept":
        tier_specs = [
            (0.15, int(20 * q_mult), 1.02),
            (0.38, int(30 * q_mult), 0.98),
            (0.64, int(38 * q_mult), 0.88),
            (0.88, int(34 * q_mult), 0.70),
        ]
        n_seg = 8 if q == "preview" else 10
    elif style == "bob":
        tier_specs = [
            (0.14, int(24 * q_mult), 1.06),
            (0.35, int(34 * q_mult), 1.00),
            (0.60, int(40 * q_mult), 0.92),
            (0.84, int(36 * q_mult), 0.82),
        ]
        n_seg = 10 if q == "preview" else 13
    else:  # long_wavy
        tier_specs = [
            (0.14, int(28 * q_mult), 1.08),
            (0.34, int(38 * q_mult), 1.02),
            (0.58, int(44 * q_mult), 0.94),
            (0.82, int(42 * q_mult), 0.86),
        ]
        n_seg = 12 if q == "preview" else 16

    root_phis_l: list[np.ndarray] = []
    root_fracs_l: list[np.ndarray] = []
    root_len_mul_l: list[np.ndarray] = []
    root_layer_l: list[np.ndarray] = []

    for tier_idx, (frac_base, count, len_mul) in enumerate(tier_specs):
        count = max(10, count)
        p_arr = np.linspace(-np.pi, np.pi, count, endpoint=False)
        p_arr += rng.uniform(-0.45, 0.45, size=count) * (2.0 * np.pi / count)
        p_arr = (p_arr + np.pi) % (2.0 * np.pi) - np.pi
        f_arr = np.clip(frac_base + rng.uniform(-0.07, 0.07, size=count), 0.06, 0.94)
        l_arr = len_mul * rng.uniform(0.76, 1.15, size=count)
        root_phis_l.append(p_arr)
        root_fracs_l.append(f_arr)
        root_len_mul_l.append(l_arr)
        root_layer_l.append(np.full(count, float(tier_idx) / max(1, len(tier_specs) - 1)))

    phis = np.concatenate(root_phis_l)
    fracs = np.concatenate(root_fracs_l)
    len_muls = np.concatenate(root_len_mul_l)
    layers = np.concatenate(root_layer_l)  # 0 = crown outer layer, 1 = lower nape/rim layer
    n_locks = len(phis)

    th_max = _hairline_theta_max(phis, style, g)
    th = th_max * fracs
    dx = np.sin(th) * np.sin(phis)
    dy = -np.sin(th) * np.cos(phis)
    dz = np.cos(th)
    raw_roots = np.column_stack(
        [
            c_scalp[0] + dx * r_init[0],
            c_scalp[1] + dy * r_init[1],
            c_scalp[2] + dz * r_init[2],
        ]
    )

    # Project all lock roots onto cranium surface in one vectorized KDTree batch
    roots, root_nrms = surface.project_to_surface(raw_roots, offset=0.0018 * scale)

    # Determine part line X position and per-lock flow sign
    part_x = h_part * 0.40 * hh
    part_side = np.where(roots[:, 0] >= part_x, 1.0, -1.0)
    sweep_sign = -1.0 if h_part >= 0.0 else 1.0
    is_front = np.abs(phis) < 0.34 * np.pi
    is_front_wide = np.abs(phis) < 0.44 * np.pi
    is_side = (np.abs(phis) >= 0.34 * np.pi) & (np.abs(phis) < 0.65 * np.pi)

    # Compute per-lock total physical length in meters
    if style == "short":
        # Sleek layered side-swept short hair: longer smooth top locks, tapered neat sides
        base_len = (
            np.where(
                is_front,
                np.where(fracs > 0.70, 0.062, 0.084),
                np.where(is_side, 0.056, 0.068),
            )
            * scale
            * h_len
        )
    elif style == "swept":
        # Lively swept fringe lifted cleanly above eye line
        base_len = np.where(is_front, 0.088, np.where(is_side, 0.068, 0.082)) * scale * h_len
    elif style == "bob":
        target_hem_z = float(lm.chin[2]) - 0.015 * scale * h_len
        vert_drop = np.clip(roots[:, 2] - target_hem_z, 0.085 * scale, 0.25 * scale)
        base_len = np.where(is_front & (fracs > 0.68), vert_drop * 0.85, vert_drop * 1.18)
    else:  # long_wavy
        target_hem_z = float(lm.chest[2]) + 0.010 * scale - 0.075 * scale * (h_len - 1.0)
        vert_drop = np.clip(roots[:, 2] - target_hem_z, 0.16 * scale, 0.44 * scale)
        base_len = np.where(is_front & (fracs > 0.72), vert_drop * 0.92, vert_drop * 1.18)

    lock_lens = base_len * len_muls
    step_lens = lock_lens / float(n_seg - 1)

    # Tangent direction down the skull surface away from crown/part
    down_vec = np.array([0.0, 0.0, -1.0], dtype=np.float64)[None, :]
    tang_down = down_vec - np.sum(down_vec * root_nrms, axis=1, keepdims=True) * root_nrms
    td_len = np.linalg.norm(tang_down, axis=1, keepdims=True)
    # Fallback flow direction for top-crown locks near pole
    crown_flow = np.column_stack(
        [
            part_side * (0.78 if style in ("long_wavy", "bob") else 0.72),
            np.where(is_front, 0.18 if style in ("long_wavy", "bob") else -0.35, 0.48),
            np.full(n_locks, -0.22),
        ]
    )
    tang_down = np.where(
        td_len < 0.25, crown_flow, tang_down / np.where(td_len < 1e-6, 1.0, td_len)
    )

    # Build style-specific initial flow velocity
    flow_dir = tang_down.copy()
    if style == "short":
        # Sleek combed diagonal side-sweep across crown/forehead, neat tapered sides/back
        top_mask = layers < 0.72
        flow_dir[:, 0] += np.where(
            top_mask & is_front_wide,
            sweep_sign * 0.86,
            np.where(top_mask, sweep_sign * 0.42, part_side * 0.10),
        )
        flow_dir[:, 1] += np.where(top_mask & is_front, -0.26, 0.18)
        flow_dir[:, 2] += np.where(top_mask & is_front, 0.06 * h_vol, -0.58)
    elif style == "swept":
        # Dynamic swept volume curving cleanly across forehead ABOVE eye level
        top_mask = layers < 0.70
        flow_dir[:, 0] += np.where(top_mask, sweep_sign * 0.96, part_side * 0.22)
        flow_dir[:, 1] += np.where(top_mask & is_front, -0.38, 0.14)
        flow_dir[:, 2] += np.where(top_mask & is_front, 0.32 * h_vol, -0.58)
    elif style == "bob":
        # Parted front sweeps smoothly past temples before framing cheeks
        flow_dir[:, 0] += part_side * np.where(is_front_wide, 0.62, 0.14) * (1.0 - 0.30 * layers)
        flow_dir[:, 1] += np.where(is_front_wide, 0.24, 0.18)
        flow_dir[:, 2] -= np.where(is_front_wide, 0.68, 0.98)
    else:  # long_wavy
        # Sleek parted crown hugging temples & cheeks closely, then cascading over shoulders
        flow_dir[:, 0] += part_side * np.where(is_front_wide, 0.56, 0.08) * (1.0 - 0.30 * layers)
        flow_dir[:, 1] += np.where(is_front_wide, 0.26, 0.20)
        flow_dir[:, 2] -= np.where(is_front_wide, 0.72, 1.12)

    # Add root volume lift along scalp normal
    lift_base = (
        0.16
        if style == "short"
        else (0.14 if style == "long_wavy" else (0.34 if style == "swept" else 0.20))
    )
    lift_amt = lift_base * h_vol * (1.0 - 0.48 * layers)
    vel = flow_dir / np.maximum(np.linalg.norm(flow_dir, axis=1, keepdims=True), 1e-6)
    vel = vel + root_nrms * lift_amt[:, None]
    vel /= np.maximum(np.linalg.norm(vel, axis=1, keepdims=True), 1e-6)

    # Organic per-lock curl phase & gentle natural wave amplitude
    curl_phase = rng.uniform(0.0, 2.0 * np.pi, size=n_locks)
    curl_freq = rng.uniform(1.6, 2.4, size=n_locks) * np.pi
    curl_amp = (0.005 + 0.013 * h_curl) * scale * (1.0 if style == "long_wavy" else 0.42)

    spines = np.empty((n_locks, n_seg, 3), dtype=np.float64)
    nrms_out = np.empty((n_locks, n_seg, 3), dtype=np.float64)
    spines[:, 0, :] = roots
    nrms_out[:, 0, :] = root_nrms

    # Outer crown layers ride sleekly over inner layers
    layer_clear_base = (0.0045 + 0.0075 * (1.0 - layers)) * h_vol * scale
    if style == "short":
        layer_clear_base *= 0.56
    elif style == "swept":
        layer_clear_base *= 1.10
    elif style == "long_wavy":
        layer_clear_base *= 0.54

    sh_x = abs(float(lm.shoulder_left[0]))
    eye_z_lo = float(lm.eye_left[2]) - float(lm.eye_radius) * 1.4
    eye_z_hi = float(lm.eye_left[2]) + float(lm.eye_radius) * 1.5
    face_z_hi = float(lm.head_center[2]) + 0.54 * hh
    face_z_lo = float(lm.chin[2]) - 0.08 * hh
    face_y_cut = float(lm.head_center[1]) - 0.08 * hh

    # Vectorized spine integration with KDTree skin collision pushing & sleek gravity drape
    cur_p = roots.copy()
    cur_v = vel.copy()
    for si in range(1, n_seg):
        t = float(si) / float(n_seg - 1)
        if style == "long_wavy":
            grav_w = 0.34 + 1.10 * t
        elif style == "bob":
            grav_w = 0.24 + 0.85 * t
        else:
            grav_w = 0.18 + 0.34 * t
        cur_v[:, 2] -= grav_w * 0.46

        if style == "bob" and t > 0.65:
            # Gentle inward bob curl toward jawline at tips
            to_center = lm.head_center[None, :] - cur_p
            to_center[:, 2] = 0.0
            tc_len = np.maximum(np.linalg.norm(to_center, axis=1, keepdims=True), 1e-6)
            cur_v += (to_center / tc_len) * 0.28

        if style == "long_wavy" and t > 0.26:
            # Keep hanging curtain sleek & close to cheeks/neck/shoulders (no bell flare!)
            max_w = min(sh_x * 0.86, hh * (1.02 + 0.22 * t))
            lateral_excess = np.abs(cur_p[:, 0]) - max_w
            too_wide = lateral_excess > 0.0
            if np.any(too_wide):
                cur_v[too_wide, 0] -= np.sign(cur_p[too_wide, 0]) * np.minimum(
                    0.55, lateral_excess[too_wide] * 8.0
                )
            # Soft inward drape damping after clearing upper forehead
            cur_v[:, 0] *= 0.84

        # Keep swept bangs lifted above eyes
        if style == "swept":
            eye_zone = (
                (cur_p[:, 1] < float(lm.eye_left[1]) + 0.015 * scale)
                & (np.abs(cur_p[:, 0]) < abs(float(lm.eye_left[0])) * 1.38)
                & (cur_p[:, 2] > eye_z_lo)
                & (cur_p[:, 2] < eye_z_hi)
            )
            if np.any(eye_zone):
                cur_v[eye_zone, 0] += sweep_sign * 0.42
                cur_v[eye_zone, 2] += 0.26

        cur_v /= np.maximum(np.linalg.norm(cur_v, axis=1, keepdims=True), 1e-6)
        nxt_p = cur_p + cur_v * step_lens[:, None]

        # Apply smooth sinusoidal wave displacement perpendicular to lock flow
        if h_curl > 0.02 and t > 0.20:
            wave_env = np.sin(np.pi * np.clip((t - 0.16) / 0.84, 0.0, 1.0))
            w_val = (
                np.sin(t * curl_freq + curl_phase)
                * curl_amp
                * wave_env
                * (1.0 / max(4, n_seg - 1))
                * 2.2
            )
            nxt_p[:, 0] += w_val * np.cos(phis)
            nxt_p[:, 1] += w_val * np.sin(phis) * 0.55

        # Face-Window Repulsor for parted styles: smooth sleek oval frame around face
        if style in ("bob", "long_wavy"):
            z_norm = np.clip((nxt_p[:, 2] - face_z_lo) / max(1e-6, face_z_hi - face_z_lo), 0.0, 1.0)
            # Sleek oval framing just outside brows/eyes/cheeks (max ~0.64*hh)
            min_face_x = (0.48 + 0.16 * np.sin(np.pi * np.clip(z_norm, 0.08, 0.94))) * hh
            in_face = (
                (nxt_p[:, 2] < face_z_hi)
                & (nxt_p[:, 2] > face_z_lo)
                & (nxt_p[:, 1] < face_y_cut)
                & (np.abs(nxt_p[:, 0]) < min_face_x)
            )
            if np.any(in_face):
                target_x = part_side[in_face] * (min_face_x[in_face] + 0.002 * scale)
                # Smooth progressive glide around face contour (no sharp kink at forehead!)
                nxt_p[in_face, 0] = _lerp(nxt_p[in_face, 0], target_x, 0.52)
                nxt_p[in_face, 1] += 0.006 * scale
                cur_v[in_face, 0] += part_side[in_face] * 0.34
                cur_v[in_face, 1] += 0.16

        # Collision push against pure-skin human mesh (plus shirt clearance below neck!)
        below_neck = np.clip((float(lm.neck_base[2]) - nxt_p[:, 2]) / (0.08 * scale), 0.0, 1.0)
        cloth_pad = (0.020 * scale) * below_neck
        clearance = layer_clear_base * (0.55 + 0.55 * np.sin(np.pi * min(0.85, t))) + cloth_pad
        surf_p, surf_n, dists = surface.query_nearest(nxt_p)
        signed_d = np.where(np.sum((nxt_p - surf_p) * surf_n, axis=1) >= 0.0, dists, -dists)
        pen = clearance - signed_d
        hit = pen > 0.0
        if np.any(hit):
            grad_hit = surf_n[hit]
            nxt_p[hit] = surf_p[hit] + (clearance[hit] + 0.0012 * scale)[:, None] * grad_hit
            # Deflect velocity tangentially so next segment flows smoothly around skin
            vn = np.sum(cur_v[hit] * grad_hit, axis=1, keepdims=True)
            cur_v[hit] = cur_v[hit] - np.minimum(0.0, vn) * grad_hit

        # Outward normal points away from head/neck spine center
        ref_c = np.column_stack(
            [
                np.zeros(n_locks),
                np.full(n_locks, float(lm.head_center[1]) + 0.04 * hh),
                np.minimum(nxt_p[:, 2], float(lm.head_center[2]) + 0.22 * hh),
            ]
        )
        out_n = nxt_p - ref_c
        out_n /= np.maximum(np.linalg.norm(out_n, axis=1, keepdims=True), 1e-6)

        spines[:, si, :] = nxt_p
        nrms_out[:, si, :] = out_n
        cur_p = nxt_p

    # Width & thickness taper profiles per lock (sleeker flatter ribbons for short hair)
    ts = np.linspace(0.0, 1.0, n_seg, dtype=np.float64)[None, :]
    w_taper = np.power(np.clip(1.0 - np.power(ts, 1.65), 0.04, 1.0), 0.72)
    w_taper[:, 0] *= 0.85
    w_taper[:, -1] = 0.02

    base_w_m = (
        (
            0.0205
            if style == "short"
            else (0.0205 if style == "swept" else (0.022 if style == "bob" else 0.023))
        )
        * scale
        * (0.88 + 0.14 * h_vol)
    )
    base_w_m *= 1.16 if q == "preview" else (0.94 if q == "high" else 1.0)
    per_lock_w = base_w_m * rng.uniform(0.78, 1.20, size=(n_locks, 1))

    half_widths = per_lock_w * w_taper
    thick_ratio = 0.34 if style == "short" else 0.46
    thicknesses = half_widths * (thick_ratio + 0.12 * h_vol)
    u_offsets = rng.uniform(0.0, 1.0, size=n_locks)

    clump_v, clump_f, clump_uv = _build_clump_meshes_batch(
        spines, nrms_out, half_widths, thicknesses, u_offsets
    )

    verts = np.vstack([scalp_v, clump_v])
    faces = np.vstack([scalp_f, clump_f + len(scalp_v)])
    uvs = np.vstack([scalp_uv, clump_uv])
    return MeshData(vertices=verts, faces=faces, uvs=uvs)
