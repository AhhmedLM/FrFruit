"""Analytical procedural eye, eyebrow, and eyelash child meshes fitted to CharacterSurface."""

from __future__ import annotations

import numpy as np

from frfruit.character.surface import CharacterSurface
from frfruit.geometry.mesh import MeshData


def _ensure_outward_winding(
    verts: np.ndarray,
    faces: np.ndarray,
    center: np.ndarray,
) -> np.ndarray:
    """Ensure triangle winding faces outward relative to `center`."""
    v0 = verts[faces[:, 0]]
    v1 = verts[faces[:, 1]]
    v2 = verts[faces[:, 2]]
    fn = np.cross(v1 - v0, v2 - v0)
    tri_c = (v0 + v1 + v2) * (1.0 / 3.0) - center[None, :]
    dots = np.sum(fn * tri_c, axis=1)
    if float(np.mean(dots)) < 0.0:
        return faces[:, [0, 2, 1]]
    return faces


def _resample_polyline(pts: np.ndarray, n_out: int) -> np.ndarray:
    """Resample an ordered 3D polyline smoothly to `n_out` arc-length spaced points."""
    if len(pts) < 2:
        return np.repeat(pts[:1], n_out, axis=0)
    segs = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    s = np.concatenate([[0.0], np.cumsum(segs)])
    total = float(s[-1])
    if total < 1e-9:
        return np.repeat(pts[:1], n_out, axis=0)
    s_norm = s / total
    t_out = np.linspace(0.0, 1.0, n_out)
    out = np.empty((n_out, 3), dtype=np.float64)
    for ax in range(3):
        out[:, ax] = np.interp(t_out, s_norm, pts[:, ax])
    # One gentle Laplacian smoothing pass keeping endpoints anchored
    if n_out >= 5:
        sm = out.copy()
        sm[1:-1] = 0.25 * out[:-2] + 0.50 * out[1:-1] + 0.25 * out[2:]
        out = sm
    return out


def _build_single_eyeball(
    center: np.ndarray,
    radius: float,
    n_rings: int = 22,
    n_segs: int = 32,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate a smooth UV-mapped eyeball sphere with front-pole (-Y) azimuthal UVs."""
    # Parameterize polar angle theta from front pole (-Y) to back pole (+Y)
    # Using concentric rings around the optical axis (-Y) gives razor-sharp iris & pupil sampling!
    thetas = np.linspace(0.0, np.pi, n_rings + 1)
    phis = np.linspace(0.0, 2.0 * np.pi, n_segs, endpoint=False)

    verts_list: list[np.ndarray] = []
    uvs_list: list[np.ndarray] = []

    # Front pole vertex (index 0)
    verts_list.append(center + np.array([0.0, -radius, 0.0]))
    uvs_list.append(np.array([0.5, 0.5]))

    # Intermediate rings (1 .. n_rings - 1)
    for ti in range(1, n_rings):
        th = float(thetas[ti])
        st = float(np.sin(th))
        ct = float(np.cos(th))
        rad_uv = (th / np.pi) * 0.48
        for phi in phis:
            cp = float(np.cos(phi))
            sp = float(np.sin(phi))
            dx = st * cp
            dy = -ct
            dz = st * sp
            verts_list.append(center + radius * np.array([dx, dy, dz]))
            uvs_list.append(np.array([0.5 + rad_uv * cp, 0.5 + rad_uv * sp]))

    # Back pole vertex (last index)
    back_idx = len(verts_list)
    verts_list.append(center + np.array([0.0, radius, 0.0]))
    uvs_list.append(np.array([0.98, 0.5]))

    verts = np.asarray(verts_list, dtype=np.float64)
    uvs = np.asarray(uvs_list, dtype=np.float64)

    faces_list: list[tuple[int, int, int]] = []
    # Front cap
    for j in range(n_segs):
        jn = (j + 1) % n_segs
        faces_list.append((0, 1 + jn, 1 + j))

    # Middle quads
    for ri in range(n_rings - 2):
        row0 = 1 + ri * n_segs
        row1 = 1 + (ri + 1) * n_segs
        for j in range(n_segs):
            jn = (j + 1) % n_segs
            a = row0 + j
            b = row0 + jn
            c = row1 + jn
            d = row1 + j
            faces_list.append((a, b, c))
            faces_list.append((a, c, d))

    # Back cap
    last_row = 1 + (n_rings - 2) * n_segs
    for j in range(n_segs):
        jn = (j + 1) % n_segs
        faces_list.append((back_idx, last_row + j, last_row + jn))

    faces = _ensure_outward_winding(verts, np.asarray(faces_list, dtype=np.int64), center)
    return verts, faces, uvs


def build_eyeballs_mesh(surface: CharacterSurface) -> MeshData:
    """Build dual smooth analytical UV-mapped eyeballs seated precisely in the orbital sockets."""
    lm = surface.lm
    r = float(lm.eye_radius) * 0.992
    n_rings = 16 if surface.spec.quality == "preview" else 24
    n_segs = 24 if surface.spec.quality == "preview" else 36

    vl, fl, uvl = _build_single_eyeball(lm.eye_left, r, n_rings=n_rings, n_segs=n_segs)
    vr, fr, uvr = _build_single_eyeball(lm.eye_right, r, n_rings=n_rings, n_segs=n_segs)

    verts = np.vstack([vl, vr])
    faces = np.vstack([fl, fr + len(vl)])
    uvs = np.vstack([uvl, uvr])
    return MeshData(vertices=verts, faces=faces, uvs=uvs)


def _build_profiled_ribbon(
    spine: np.ndarray,
    normals: np.ndarray,
    half_widths: np.ndarray,
    heights: np.ndarray,
    *,
    u_offset: float = 0.0,
    u_scale: float = 1.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Extrude a closed 3D diamond/arched ribbon along `spine` with smooth UVs and capped tips."""
    m = len(spine)
    # Compute smooth tangents along spine
    tang = np.zeros_like(spine)
    tang[0] = spine[1] - spine[0]
    tang[-1] = spine[-1] - spine[-2]
    if m > 2:
        tang[1:-1] = spine[2:] - spine[:-2]
    t_len = np.linalg.norm(tang, axis=1, keepdims=True)
    tang = tang / np.where(t_len < 1e-9, 1.0, t_len)

    # Orthonormalize surface normal & lateral binormal
    nrm = normals - np.sum(normals * tang, axis=1, keepdims=True) * tang
    n_len = np.linalg.norm(nrm, axis=1, keepdims=True)
    nrm = nrm / np.where(n_len < 1e-9, 1.0, n_len)
    side = np.cross(nrm, tang)
    s_len = np.linalg.norm(side, axis=1, keepdims=True)
    side = side / np.where(s_len < 1e-9, 1.0, s_len)

    # 4-vertex closed cross-section per ring: left edge, top ridge, right edge, bottom seat
    verts = np.empty((m * 4, 3), dtype=np.float64)
    uvs = np.empty((m * 4, 2), dtype=np.float64)
    v_coords = np.linspace(0.0, 1.0, m)

    for i in range(m):
        c = spine[i]
        hw = float(half_widths[i])
        ht = float(heights[i])
        s_vec = side[i] * hw
        n_vec = nrm[i] * ht
        back_vec = nrm[i] * (-ht * 0.35)
        base = i * 4
        verts[base + 0] = c - s_vec
        verts[base + 1] = c + n_vec
        verts[base + 2] = c + s_vec
        verts[base + 3] = c + back_vec

        vc = float(v_coords[i])
        uvs[base + 0] = (u_offset, vc)
        uvs[base + 1] = (u_offset + 0.5 * u_scale, vc)
        uvs[base + 2] = (u_offset + u_scale, vc)
        uvs[base + 3] = (u_offset + 0.5 * u_scale, vc)

    faces_list: list[tuple[int, int, int]] = []
    for i in range(m - 1):
        r0 = i * 4
        r1 = (i + 1) * 4
        for k in range(4):
            kn = (k + 1) % 4
            a, b, c, d = r0 + k, r0 + kn, r1 + kn, r1 + k
            faces_list.append((a, d, c))
            faces_list.append((a, c, b))

    # Cap ends
    faces_list.append((0, 1, 2))
    faces_list.append((0, 2, 3))
    end = (m - 1) * 4
    faces_list.append((end + 0, end + 2, end + 1))
    faces_list.append((end + 0, end + 3, end + 2))

    faces = np.asarray(faces_list, dtype=np.int64)
    return verts, faces, uvs


def build_brows_and_lashes_mesh(surface: CharacterSurface) -> MeshData:
    """Build sculpted 3D eyebrows and curved upper/lower eyelash ribbons fitted to the skin mesh."""
    lm = surface.lm
    f = surface.spec.face
    b = surface.spec.body
    expr = surface.spec.expressions
    g = float(b.gender)
    hh = lm.head_half_h

    eye_x = abs(float(lm.eye_left[0])) / hh
    eye_y = (float(lm.eye_left[1]) - lm.head_center[1]) / hh
    eye_z = (float(lm.eye_left[2]) - lm.head_center[2]) / hh
    eye_r = float(lm.eye_radius) / hh
    brow_arch = 0.042 * f.brow_arch + 0.020 * max(0.0, -g) + 0.032 * float(expr.brow_raise)
    brow_lift = 0.018 * float(f.brow_ridge) + 0.048 * float(expr.brow_raise)

    all_v: list[np.ndarray] = []
    all_f: list[np.ndarray] = []
    all_uv: list[np.ndarray] = []
    v_off = 0

    def _append_part(v: np.ndarray, fa: np.ndarray, uv: np.ndarray) -> None:
        nonlocal v_off
        all_v.append(v)
        all_f.append(fa + v_off)
        all_uv.append(uv)
        v_off += len(v)

    # --- 1. Sculpted Realistic Eyebrows (Left & Right, projected cleanly onto morphed skin) ---
    brow_thick_scale = 1.0 + 0.20 * g - 0.16 * max(0.0, -g)
    n_pts = 15
    ts = np.linspace(0.0, 1.0, n_pts)

    # Quadratic Bezier in normalized head space for brow centerline
    p0 = np.array([eye_x - eye_r * 1.24, eye_y - 0.14, eye_z + eye_r * 1.32 + brow_lift])
    p1 = np.array(
        [eye_x + eye_r * 0.22, eye_y - 0.12, eye_z + eye_r * 1.66 + brow_arch + brow_lift]
    )
    p2 = np.array(
        [eye_x + eye_r * 1.40, eye_y + 0.03, eye_z + eye_r * 1.28 + 0.45 * brow_arch + brow_lift]
    )

    for side_sign in (1.0, -1.0):
        mt = 1.0 - ts
        spine_q = (
            (mt * mt)[:, None] * p0[None, :]
            + (2.0 * mt * ts)[:, None] * p1[None, :]
            + (ts * ts)[:, None] * p2[None, :]
        )
        spine_q[:, 0] *= side_sign
        spine_w = lm.head_center[None, :] + spine_q * hh
        # Snap snugly onto exact morphed brow ridge surface via KDTree, then smooth to prevent kinks
        proj_pts, nrms = surface.project_to_surface(spine_w, offset=0.0011 * (hh / 0.12))
        for _ in range(3):
            proj_pts[1:-1] = 0.25 * proj_pts[:-2] + 0.50 * proj_pts[1:-1] + 0.25 * proj_pts[2:]
            nrms[1:-1] = 0.25 * nrms[:-2] + 0.50 * nrms[1:-1] + 0.25 * nrms[2:]
        nrms /= np.maximum(np.linalg.norm(nrms, axis=1, keepdims=True), 1e-9)

        # Taper gracefully: soft rounded inner head, refined body, fine pointed outer tail
        profile_w = np.sin(np.pi * np.power(ts, 0.70)) ** 0.80
        hw = (0.0031 * brow_thick_scale * (hh / 0.12)) * (0.24 + 0.82 * profile_w)
        ht = (0.0015 * brow_thick_scale * (hh / 0.12)) * (0.18 + 0.78 * profile_w)
        hw[0] *= 0.38
        hw[-1] *= 0.12
        ht[0] *= 0.30
        ht[-1] *= 0.12

        if side_sign < 0.0:
            # Reverse order on right side so ribbon side-normal points consistently outward
            proj_pts = proj_pts[::-1]
            nrms = nrms[::-1]
            hw = hw[::-1]
            ht = ht[::-1]

        v_b, f_b, uv_b = _build_profiled_ribbon(proj_pts, nrms, hw, ht, u_offset=0.1, u_scale=0.8)
        _append_part(v_b, f_b, uv_b)

    # --- 2. Crisp Curved Upper & Lower Eyelash Ribbons Anchored to Morphed Eyelid Loops ---
    n_lash = 13
    u_l = np.linspace(-1.0, 1.0, n_lash)
    fem_lash = 1.0 + 0.36 * max(0.0, -g)

    for side_sign in (1.0, -1.0):
        ec = lm.eye_left if side_sign > 0.0 else lm.eye_right

        for is_upper in (True, False):
            # Sample directly from the morphed quad-base eyelid margin vertices so blinking tracks!
            raw_margin = surface.eyelid_margin_points(side_sign, upper=is_upper)
            pts_lash = _resample_polyline(raw_margin, n_lash)
            nrm_lash = pts_lash - ec[None, :]
            nrm_lash /= np.maximum(np.linalg.norm(nrm_lash, axis=1, keepdims=True), 1e-9)

            # Seat slightly proud of the eyelid crease with natural forward lash curl on upper lid
            taper = np.clip(1.0 - (u_l * u_l) ** 1.6, 0.08, 1.0)
            pts_lash = pts_lash + nrm_lash * (0.0004 * (hh / 0.12))
            if is_upper:
                pts_lash[:, 1] -= 0.0013 * fem_lash * taper * (hh / 0.12)
                hw_l = (0.0015 * fem_lash * (hh / 0.12)) * taper
                ht_l = (0.0014 * fem_lash * (hh / 0.12)) * taper
            else:
                low_s = 0.62 + 0.25 * max(0.0, -g)
                hw_l = (0.00075 * low_s * (hh / 0.12)) * taper
                ht_l = (0.00075 * low_s * (hh / 0.12)) * taper

            if side_sign < 0.0:
                pts_lash = pts_lash[::-1]
                nrm_lash = nrm_lash[::-1]
                hw_l = hw_l[::-1]
                ht_l = ht_l[::-1]

            v_l, f_l, uv_l = _build_profiled_ribbon(
                pts_lash, nrm_lash, hw_l, ht_l, u_offset=0.2, u_scale=0.6
            )
            _append_part(v_l, f_l, uv_l)

    return MeshData(
        vertices=np.vstack(all_v),
        faces=np.vstack(all_f),
        uvs=np.vstack(all_uv),
    )
