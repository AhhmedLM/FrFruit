"""Deterministic 3D linear algebra and intersection helpers."""

from __future__ import annotations

import math

import numpy as np

from frfruit.core.types import BoundingBox, Vec3


def euler_deg_to_matrix(rx_deg: float, ry_deg: float, rz_deg: float) -> np.ndarray:
    """Convert XYZ Euler angles in degrees to a 3x3 rotation matrix (Rz @ Ry @ Rx)."""
    rx = math.radians(rx_deg)
    ry = math.radians(ry_deg)
    rz = math.radians(rz_deg)

    cx, sx = math.cos(rx), math.sin(rx)
    cy, sy = math.cos(ry), math.sin(ry)
    cz, sz = math.cos(rz), math.sin(rz)

    rx_m = np.array([[1.0, 0.0, 0.0], [0.0, cx, -sx], [0.0, sx, cx]], dtype=np.float64)
    ry_m = np.array([[cy, 0.0, sy], [0.0, 1.0, 0.0], [-sy, 0.0, cy]], dtype=np.float64)
    rz_m = np.array([[cz, -sz, 0.0], [sz, cz, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)
    return rz_m @ ry_m @ rx_m


def compose_matrix(
    location: Vec3 = (0.0, 0.0, 0.0),
    rotation_deg: Vec3 = (0.0, 0.0, 0.0),
    scale: Vec3 = (1.0, 1.0, 1.0),
) -> np.ndarray:
    """Compose a 4x4 TRS affine transformation matrix."""
    rot = euler_deg_to_matrix(*rotation_deg)
    s = np.diag([float(scale[0]), float(scale[1]), float(scale[2])])
    mat = np.eye(4, dtype=np.float64)
    mat[:3, :3] = rot @ s
    mat[:3, 3] = np.asarray(location, dtype=np.float64)
    return mat


def decompose_matrix(mat: np.ndarray) -> tuple[Vec3, Vec3, Vec3]:
    """Decompose a 4x4 TRS matrix into (location, rotation_deg, scale)."""
    loc = (float(mat[0, 3]), float(mat[1, 3]), float(mat[2, 3]))
    m3 = mat[:3, :3]
    sx = float(np.linalg.norm(m3[:, 0]))
    sy = float(np.linalg.norm(m3[:, 1]))
    sz = float(np.linalg.norm(m3[:, 2]))
    if np.linalg.det(m3) < 0:
        sx = -sx
    sx_safe = sx if abs(sx) > 1e-12 else 1.0
    sy_safe = sy if abs(sy) > 1e-12 else 1.0
    sz_safe = sz if abs(sz) > 1e-12 else 1.0
    r = np.column_stack([m3[:, 0] / sx_safe, m3[:, 1] / sy_safe, m3[:, 2] / sz_safe])

    sy_val = -float(r[2, 0])
    sy_val = max(-1.0, min(1.0, sy_val))
    ry = math.asin(sy_val)
    if abs(math.cos(ry)) > 1e-6:
        rx = math.atan2(float(r[2, 1]), float(r[2, 2]))
        rz = math.atan2(float(r[1, 0]), float(r[0, 0]))
    else:
        rx = math.atan2(-float(r[1, 2]), float(r[1, 1]))
        rz = 0.0
    rot_deg = (math.degrees(rx), math.degrees(ry), math.degrees(rz))
    return loc, rot_deg, (sx, sy, sz)


def transform_points(mat: np.ndarray, pts: np.ndarray) -> np.ndarray:
    """Apply a 4x4 matrix to (N, 3) points."""
    if len(pts) == 0:
        return pts.copy()
    pts_f = np.asarray(pts, dtype=np.float64)
    ones = np.ones((pts_f.shape[0], 1), dtype=np.float64)
    hom = np.hstack([pts_f, ones])
    res = hom @ mat.T
    return res[:, :3]


def transform_normals(mat: np.ndarray, normals: np.ndarray) -> np.ndarray:
    """Transform (N, 3) surface normals using inverse-transpose upper 3x3."""
    if len(normals) == 0:
        return normals.copy()
    inv_t = np.linalg.inv(mat[:3, :3]).T
    res = np.asarray(normals, dtype=np.float64) @ inv_t.T
    norms = np.linalg.norm(res, axis=1, keepdims=True)
    norms = np.where(norms < 1e-12, 1.0, norms)
    return res / norms


def compute_aabb(vertices: np.ndarray) -> BoundingBox:
    """Compute world/local AABB from an (N, 3) array of vertices."""
    if len(vertices) == 0:
        return BoundingBox(min_corner=(0.0, 0.0, 0.0), max_corner=(0.0, 0.0, 0.0))
    mn = np.min(vertices, axis=0)
    mx = np.max(vertices, axis=0)
    return BoundingBox(
        min_corner=(float(mn[0]), float(mn[1]), float(mn[2])),
        max_corner=(float(mx[0]), float(mx[1]), float(mx[2])),
    )


def look_at_matrix(
    eye: Vec3,
    target: Vec3,
    up: Vec3 = (0.0, 0.0, 1.0),
) -> np.ndarray:
    """Return 4x4 world-to-camera view matrix (camera looks along -Z, +Y is up)."""
    eye_v = np.asarray(eye, dtype=np.float64)
    target_v = np.asarray(target, dtype=np.float64)
    up_v = np.asarray(up, dtype=np.float64)

    forward = target_v - eye_v
    f_len = float(np.linalg.norm(forward))
    if f_len < 1e-9:
        forward = np.array([0.0, 1.0, 0.0], dtype=np.float64)
    else:
        forward = forward / f_len

    # Handle collinear up vector (e.g., top-down view along Z)
    up_norm = np.linalg.norm(up_v)
    up_v = up_v / up_norm if up_norm > 1e-9 else np.array([0.0, 0.0, 1.0], dtype=np.float64)
    if abs(float(np.dot(forward, up_v))) > 0.999:
        up_v = np.array([0.0, 1.0, 0.0], dtype=np.float64)
        if abs(float(np.dot(forward, up_v))) > 0.999:
            up_v = np.array([1.0, 0.0, 0.0], dtype=np.float64)

    right = np.cross(forward, up_v)
    right = right / np.linalg.norm(right)
    true_up = np.cross(right, forward)

    view = np.eye(4, dtype=np.float64)
    view[0, :3] = right
    view[1, :3] = true_up
    view[2, :3] = -forward
    view[0, 3] = -float(np.dot(right, eye_v))
    view[1, 3] = -float(np.dot(true_up, eye_v))
    view[2, 3] = float(np.dot(forward, eye_v))
    return view


def perspective_matrix(fov_deg: float, aspect: float, near: float, far: float) -> np.ndarray:
    """Return 4x4 OpenGL-style perspective projection matrix."""
    f = 1.0 / math.tan(math.radians(fov_deg) * 0.5)
    nf = 1.0 / (near - far)
    proj = np.zeros((4, 4), dtype=np.float64)
    proj[0, 0] = f / max(aspect, 1e-6)
    proj[1, 1] = f
    proj[2, 2] = (far + near) * nf
    proj[2, 3] = 2.0 * far * near * nf
    proj[3, 2] = -1.0
    return proj


def orthographic_matrix(ortho_scale: float, aspect: float, near: float, far: float) -> np.ndarray:
    """Return 4x4 orthographic projection matrix."""
    half_h = max(ortho_scale * 0.5, 1e-6)
    half_w = half_h * max(aspect, 1e-6)
    proj = np.zeros((4, 4), dtype=np.float64)
    proj[0, 0] = 1.0 / half_w
    proj[1, 1] = 1.0 / half_h
    proj[2, 2] = -2.0 / (far - near)
    proj[2, 3] = -(far + near) / (far - near)
    proj[3, 3] = 1.0
    return proj


def ray_aabb_intersect(
    origin: np.ndarray,
    direction: np.ndarray,
    bbox: BoundingBox,
) -> float | None:
    """Slab test for ray-AABB intersection; returns closest positive distance or None."""
    bmin = np.asarray(bbox.min_corner, dtype=np.float64)
    bmax = np.asarray(bbox.max_corner, dtype=np.float64)
    tmin = -np.inf
    tmax = np.inf
    for i in range(3):
        if abs(direction[i]) < 1e-12:
            if origin[i] < bmin[i] or origin[i] > bmax[i]:
                return None
        else:
            inv_d = 1.0 / direction[i]
            t1 = (bmin[i] - origin[i]) * inv_d
            t2 = (bmax[i] - origin[i]) * inv_d
            if t1 > t2:
                t1, t2 = t2, t1
            tmin = max(tmin, t1)
            tmax = min(tmax, t2)
            if tmin > tmax:
                return None
    if tmax < 0.0:
        return None
    return float(max(0.0, tmin))


def ray_triangles_intersect(
    origin: np.ndarray,
    direction: np.ndarray,
    vertices: np.ndarray,
    faces: np.ndarray,
) -> tuple[int, float, np.ndarray] | None:
    """Vectorized Moller-Trumbore ray-triangle intersection.

    Returns (face_index, distance, face_normal) for the closest hit with t >= 1e-6.
    """
    if len(faces) == 0:
        return None
    eps = 1e-7
    v0 = vertices[faces[:, 0]]
    v1 = vertices[faces[:, 1]]
    v2 = vertices[faces[:, 2]]

    edge1 = v1 - v0
    edge2 = v2 - v0
    h = np.cross(direction[None, :], edge2)
    a = np.einsum("ij,ij->i", edge1, h)
    valid = np.abs(a) > eps

    f = np.zeros_like(a)
    f[valid] = 1.0 / a[valid]
    s = origin[None, :] - v0
    u = f * np.einsum("ij,ij->i", s, h)
    valid &= (u >= -eps) & (u <= 1.0 + eps)

    q = np.cross(s, edge1)
    v = f * np.einsum("ij,j->i", q, direction)
    valid &= (v >= -eps) & (u + v <= 1.0 + eps)

    t = f * np.einsum("ij,ij->i", edge2, q)
    valid &= t > 1e-6

    if not np.any(valid):
        return None

    valid_idx = np.nonzero(valid)[0]
    best_local = int(np.argmin(t[valid_idx]))
    face_idx = int(valid_idx[best_local])
    dist = float(t[face_idx])
    normal = np.cross(edge1[face_idx], edge2[face_idx])
    n_len = float(np.linalg.norm(normal))
    if n_len > 1e-12:
        normal = normal / n_len
    return face_idx, dist, normal
