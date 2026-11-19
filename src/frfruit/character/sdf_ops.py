"""Vectorized NumPy Signed Distance Function primitives, smooth CSG, and Wendland C2 RBF."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from frfruit.character.schema import RBFDeltaNode


def smin_poly(a: np.ndarray, b: np.ndarray, k: float) -> np.ndarray:
    """Quadratic polynomial smooth minimum (Inigo Quilez)."""
    if k <= 1e-7:
        return np.minimum(a, b)
    h = np.maximum(k - np.abs(a - b), 0.0) / k
    return np.minimum(a, b) - 0.25 * h * h * k


def smin_cubic(a: np.ndarray, b: np.ndarray, k: float) -> np.ndarray:
    """Cubic polynomial smooth minimum (C2 continuous, eliminates specular seams)."""
    if k <= 1e-7:
        return np.minimum(a, b)
    h = np.maximum(k - np.abs(a - b), 0.0) / k
    return np.minimum(a, b) - (1.0 / 6.0) * h * h * h * k


def smax_poly(a: np.ndarray, b: np.ndarray, k: float) -> np.ndarray:
    """Quadratic polynomial smooth maximum (intersection)."""
    return -smin_poly(-a, -b, k)


def ssub_poly(d_base: np.ndarray, d_cut: np.ndarray, k: float) -> np.ndarray:
    """Smooth boolean subtraction (carve d_cut out of d_base)."""
    return -smin_poly(-d_base, d_cut, k)


def sd_sphere(p: np.ndarray, center: np.ndarray, radius: float) -> np.ndarray:
    """Exact SDF of a sphere."""
    return np.linalg.norm(p - center[None, :], axis=1) - radius


def sd_ellipsoid(p: np.ndarray, center: np.ndarray, radii: np.ndarray) -> np.ndarray:
    """Fast smooth SDF estimate of an axis-aligned ellipsoid."""
    q = p - center[None, :]
    r = np.maximum(radii[None, :], 1e-6)
    qr = q / r
    k0 = np.linalg.norm(qr, axis=1)
    k1 = np.linalg.norm(qr / r, axis=1)
    safe_k1 = np.where(k1 < 1e-9, 1e-9, k1)
    return np.where(k0 < 1e-9, -float(np.min(radii)), k0 * (k0 - 1.0) / safe_k1)


def sd_oriented_ellipsoid(
    p: np.ndarray,
    center: np.ndarray,
    radii: np.ndarray,
    rot_mat: np.ndarray,
) -> np.ndarray:
    """SDF of an oriented ellipsoid; `rot_mat` maps local ellipsoid axes to world."""
    q = (p - center[None, :]) @ rot_mat  # inverse rotation = rot_mat.T on column vecs
    r = np.maximum(radii[None, :], 1e-6)
    qr = q / r
    k0 = np.linalg.norm(qr, axis=1)
    k1 = np.linalg.norm(qr / r, axis=1)
    safe_k1 = np.where(k1 < 1e-9, 1e-9, k1)
    return np.where(k0 < 1e-9, -float(np.min(radii)), k0 * (k0 - 1.0) / safe_k1)


def sd_capsule(p: np.ndarray, a: np.ndarray, b: np.ndarray, r: float) -> np.ndarray:
    """Exact SDF of a capsule with constant radius `r` between endpoints `a` and `b`."""
    pa = p - a[None, :]
    ba = b - a
    len2 = float(np.dot(ba, ba))
    if len2 < 1e-12:
        return np.linalg.norm(pa, axis=1) - r
    h = np.clip(np.sum(pa * ba[None, :], axis=1) / len2, 0.0, 1.0)
    return np.linalg.norm(pa - h[:, None] * ba[None, :], axis=1) - r


def sd_round_cone(
    p: np.ndarray,
    a: np.ndarray,
    b: np.ndarray,
    r1: float,
    r2: float,
) -> np.ndarray:
    """Exact C1 SDF of an asymmetric round cone between `a` (radius r1) and `b` (radius r2)."""
    ba = b - a
    length = float(np.linalg.norm(ba))
    if length < 1e-9:
        return np.linalg.norm(p - a[None, :], axis=1) - max(r1, r2)

    u = ba / length
    pa = p - a[None, :]
    s = np.sum(pa * u[None, :], axis=1)
    perp = pa - s[:, None] * u[None, :]
    r = np.linalg.norm(perp, axis=1)

    sin_a = max(-0.995, min(0.995, (r1 - r2) / length))
    cos_a = float(np.sqrt(max(1e-6, 1.0 - sin_a * sin_a)))

    qt = -r * sin_a + s * cos_a
    d_cap_a = np.sqrt(r * r + s * s) - r1
    d_cap_b = np.sqrt(r * r + (s - length) ** 2) - r2
    d_slant = r * cos_a + s * sin_a - r1

    q_max = length * cos_a
    return np.where(qt < 0.0, d_cap_a, np.where(qt > q_max, d_cap_b, d_slant))


def sd_rounded_box(
    p: np.ndarray,
    center: np.ndarray,
    half_extents: np.ndarray,
    radius: float = 0.0,
) -> np.ndarray:
    """Exact SDF of an axis-aligned rounded box."""
    q = np.abs(p - center[None, :]) - (half_extents[None, :] - radius)
    outside = np.linalg.norm(np.maximum(q, 0.0), axis=1)
    inside = np.minimum(np.maximum(q[:, 0], np.maximum(q[:, 1], q[:, 2])), 0.0)
    return outside + inside - radius


def sd_bezier_round_cones(
    p: np.ndarray,
    a: np.ndarray,
    b: np.ndarray,
    c: np.ndarray,
    r_a: float,
    r_b: float,
    r_c: float,
    segments: int = 8,
) -> np.ndarray:
    """Smooth quadratic Bezier tube with variable radius evaluated via chained round cones."""
    ts = np.linspace(0.0, 1.0, segments + 1)
    pts = []
    rads = []
    for t in ts:
        mt = 1.0 - t
        pts.append((mt * mt) * a + (2.0 * mt * t) * b + (t * t) * c)
        rads.append((mt * mt) * r_a + (2.0 * mt * t) * r_b + (t * t) * r_c)

    d_min = sd_round_cone(p, pts[0], pts[1], rads[0], rads[1])
    for i in range(1, segments):
        d_seg = sd_round_cone(p, pts[i], pts[i + 1], rads[i], rads[i + 1])
        d_min = np.minimum(d_min, d_seg)
    return d_min


def wendland_c2(q: np.ndarray) -> np.ndarray:
    """Compactly supported Wendland C2 radial basis kernel: (1-q)^4 * (4q+1) for q < 1."""
    t = np.clip(1.0 - q, 0.0, 1.0)
    return (t * t * t * t) * (4.0 * q + 1.0)


def apply_rbf_nodes(
    pts: np.ndarray,
    nodes: list[RBFDeltaNode],
    head_center: np.ndarray,
    head_half_height: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Evaluate Wendland C2 RBF domain warp & scalar SDF offset across all nodes.

    Returns (warped_pts, scalar_offset) where `warped_pts` has shape (N, 3) and
    `scalar_offset` has shape (N,).
    """
    if not nodes:
        return pts, np.zeros((len(pts),), dtype=np.float64)

    warped = pts.copy()
    scalar_off = np.zeros((len(pts),), dtype=np.float64)

    for node in nodes:
        if node.space == "head_norm":
            c_base = head_center + np.asarray(node.center, dtype=np.float64) * head_half_height
            rad = float(node.radius) * head_half_height
            v_delta = np.asarray(node.vector_delta, dtype=np.float64) * head_half_height
            w_val = float(node.weight) * head_half_height
        else:
            c_base = np.asarray(node.center, dtype=np.float64)
            rad = float(node.radius)
            v_delta = np.asarray(node.vector_delta, dtype=np.float64)
            w_val = float(node.weight)

        centers = [c_base]
        v_deltas = [v_delta]
        if node.mirror_x and abs(float(c_base[0])) > 1e-5:
            c_mir = c_base.copy()
            c_mir[0] = -c_mir[0]
            v_mir = v_delta.copy()
            v_mir[0] = -v_mir[0]
            centers.append(c_mir)
            v_deltas.append(v_mir)

        inv_r = 1.0 / max(1e-5, rad)
        for c_pt, vd in zip(centers, v_deltas, strict=True):
            diff = pts - c_pt[None, :]
            dist = np.linalg.norm(diff, axis=1)
            q = dist * inv_r
            mask = q < 1.0
            if not np.any(mask):
                continue
            phi = wendland_c2(q[mask])
            if abs(w_val) > 1e-8:
                scalar_off[mask] += w_val * phi
            if np.any(np.abs(vd) > 1e-8):
                # Subtract displacement from query point to pull surface in +vd direction
                warped[mask] -= phi[:, None] * vd[None, :]

    return warped, scalar_off


def eval_sdf_gradient(
    sdf_fn: Callable[[np.ndarray], np.ndarray],
    pts: np.ndarray,
    eps: float = 1e-4,
) -> np.ndarray:
    """Compute central-difference SDF gradient (N, 3) normalized to unit vectors."""
    n = len(pts)
    if n == 0:
        return np.zeros((0, 3), dtype=np.float64)
    grad = np.empty((n, 3), dtype=np.float64)
    for ax in range(3):
        off = np.zeros((1, 3), dtype=np.float64)
        off[0, ax] = eps
        fp = sdf_fn(pts + off)
        fm = sdf_fn(pts - off)
        grad[:, ax] = (fp - fm) / (2.0 * eps)
    lens = np.linalg.norm(grad, axis=1, keepdims=True)
    return grad / np.where(lens < 1e-12, 1.0, lens)


_TETRA_K = np.array(
    [
        [1.0, -1.0, -1.0],
        [-1.0, -1.0, 1.0],
        [-1.0, 1.0, -1.0],
        [1.0, 1.0, 1.0],
    ],
    dtype=np.float64,
)


def eval_sdf_gradient_tetra(
    sdf_fn: Callable[[np.ndarray], np.ndarray],
    pts: np.ndarray,
    eps: float = 1e-4,
) -> np.ndarray:
    """Compute 4-point tetrahedral SDF gradient (N, 3) normalized to unit vectors."""
    n = len(pts)
    if n == 0:
        return np.zeros((0, 3), dtype=np.float64)
    # Single batched call of shape (4*N, 3) reduces Python/NumPy dispatch overhead
    batched = np.concatenate([pts + (_TETRA_K[i][None, :] * eps) for i in range(4)], axis=0)
    vals = sdf_fn(batched).reshape(4, n)
    grad = (_TETRA_K.T @ vals).T  # (N, 3)
    lens = np.linalg.norm(grad, axis=1, keepdims=True)
    return grad / np.where(lens < 1e-12, 1.0, lens)


def rotation_matrix_x(deg: float) -> np.ndarray:
    r = np.deg2rad(deg)
    c, s = float(np.cos(r)), float(np.sin(r))
    return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]], dtype=np.float64)


def rotation_matrix_y(deg: float) -> np.ndarray:
    r = np.deg2rad(deg)
    c, s = float(np.cos(r)), float(np.sin(r))
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]], dtype=np.float64)


def rotation_matrix_z(deg: float) -> np.ndarray:
    r = np.deg2rad(deg)
    c, s = float(np.cos(r)), float(np.sin(r))
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)
