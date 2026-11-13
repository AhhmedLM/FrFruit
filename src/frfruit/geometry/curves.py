"""2D/3D curves, extrusion, lathe/revolve, and parallel-transport sweeps."""

from __future__ import annotations

import math

import manifold3d
import numpy as np

from frfruit.core.types import Vec2, Vec3
from frfruit.geometry.mesh import GeometryError, MeshData


def eval_cubic_bezier(
    p0: np.ndarray,
    p1: np.ndarray,
    p2: np.ndarray,
    p3: np.ndarray,
    steps: int = 16,
) -> np.ndarray:
    """Evaluate a cubic Bezier curve segment."""
    t = np.linspace(0.0, 1.0, max(2, steps))[:, None]
    omt = 1.0 - t
    return (omt**3) * p0 + 3.0 * (omt**2) * t * p1 + 3.0 * omt * (t**2) * p2 + (t**3) * p3


def eval_catmull_rom(
    points: list[Vec3] | np.ndarray,
    steps_per_segment: int = 8,
    closed: bool = False,
) -> np.ndarray:
    """Evaluate a smooth Catmull-Rom spline through 3D control points."""
    pts = np.asarray(points, dtype=np.float64)
    if len(pts) < 2:
        raise GeometryError("Curve requires at least 2 control points.")
    if len(pts) == 2:
        t = np.linspace(0.0, 1.0, max(2, steps_per_segment))[:, None]
        return pts[0] * (1.0 - t) + pts[1] * t

    if closed:
        ext = np.vstack([pts[-1:], pts, pts[:2]])
        seg_count = len(pts)
    else:
        p_start = 2.0 * pts[0] - pts[1]
        p_end = 2.0 * pts[-1] - pts[-2]
        ext = np.vstack([p_start[None, :], pts, p_end[None, :]])
        seg_count = len(pts) - 1

    out: list[np.ndarray] = []
    for i in range(seg_count):
        p0, p1, p2, p3 = ext[i], ext[i + 1], ext[i + 2], ext[i + 3]
        n = max(2, steps_per_segment)
        ts = np.linspace(0.0, 1.0, n, endpoint=(i == seg_count - 1 and not closed))[:, None]
        t2 = ts * ts
        t3 = t2 * ts
        seg = 0.5 * (
            (2.0 * p1)
            + (-p0 + p2) * ts
            + (2.0 * p0 - 5.0 * p1 + 4.0 * p2 - p3) * t2
            + (-p0 + 3.0 * p1 - 3.0 * p2 + p3) * t3
        )
        out.append(seg)
    return np.vstack(out)


def _ensure_ccw(points_2d: list[Vec2]) -> list[list[float]]:
    pts = np.asarray(points_2d, dtype=np.float64)
    if len(pts) < 3:
        raise GeometryError("2D profile requires at least 3 points.")
    # Shoelace signed area
    x, y = pts[:, 0], pts[:, 1]
    area = 0.5 * float(np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y))
    if abs(area) < 1e-10:
        raise GeometryError("2D profile polygon has zero area.")
    if area < 0:
        pts = pts[::-1]
    return pts.tolist()


def extrude_2d_profile(
    points_2d: list[Vec2],
    height: float = 1.0,
    twist_deg: float = 0.0,
    scale_top: Vec2 = (1.0, 1.0),
    divisions: int = 8,
    center: bool = False,
) -> MeshData:
    """Extrude a closed 2D polygon profile into a watertight 3D mesh."""
    poly = _ensure_ccw(points_2d)
    cs = manifold3d.CrossSection([poly])
    mf = manifold3d.Manifold.extrude(
        cs,
        height=float(height),
        n_divisions=max(1, int(divisions)),
        twist_degrees=float(twist_deg),
        scale_top=[float(scale_top[0]), float(scale_top[1])],
    )
    if center:
        mf = mf.translate([0.0, 0.0, -0.5 * float(height)])
    return MeshData.from_manifold(mf)


def revolve_profile(
    points_2d: list[Vec2],
    angle_deg: float = 360.0,
    segments: int = 32,
) -> MeshData:
    """Revolve a 2D (radius >= 0, height) profile around the Z axis (Lathe)."""
    cleaned: list[Vec2] = []
    for r, z in points_2d:
        cleaned.append((max(0.0, float(r)), float(z)))
    poly = _ensure_ccw(cleaned)
    cs = manifold3d.CrossSection([poly])
    mf = manifold3d.Manifold.revolve(
        cs,
        circular_segments=max(6, int(segments)),
        revolve_degrees=float(angle_deg),
    )
    if mf.is_empty():
        raise GeometryError("Revolve generated an empty mesh; verify profile coordinates (r >= 0).")
    return MeshData.from_manifold(mf)


def sweep_curve(
    path_points: list[Vec3],
    radius: float = 0.08,
    radial_segments: int = 16,
    closed: bool = False,
    taper_end: float = 1.0,
    smooth_spline: bool = True,
    steps_per_segment: int = 8,
) -> MeshData:
    """Sweep a circular cross-section along a 3D path using parallel transport frames."""
    pts = np.asarray(path_points, dtype=np.float64)
    if len(pts) < 2:
        raise GeometryError("Sweep path requires at least 2 points.")

    if smooth_spline and len(pts) >= 3:
        spine = eval_catmull_rom(pts, steps_per_segment=steps_per_segment, closed=closed)
    else:
        spine = pts

    n_pts = len(spine)
    if n_pts < 2:
        raise GeometryError("Evaluated sweep path has fewer than 2 points.")

    # Compute tangents
    tangents = np.zeros_like(spine)
    if closed:
        for i in range(n_pts):
            t = spine[(i + 1) % n_pts] - spine[(i - 1) % n_pts]
            ln = float(np.linalg.norm(t))
            tangents[i] = t / ln if ln > 1e-12 else np.array([0.0, 0.0, 1.0])
    else:
        tangents[0] = spine[1] - spine[0]
        tangents[-1] = spine[-1] - spine[-2]
        for i in range(1, n_pts - 1):
            tangents[i] = spine[i + 1] - spine[i - 1]
        for i in range(n_pts):
            ln = float(np.linalg.norm(tangents[i]))
            tangents[i] = tangents[i] / ln if ln > 1e-12 else np.array([0.0, 0.0, 1.0])

    # Initial frame via non-collinear seed
    t0 = tangents[0]
    seed = np.array([0.0, 0.0, 1.0]) if abs(t0[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
    normal = np.cross(t0, seed)
    normal /= max(1e-12, float(np.linalg.norm(normal)))
    binormal = np.cross(t0, normal)

    seg = max(6, int(radial_segments))
    angles = np.linspace(0.0, 2.0 * math.pi, seg, endpoint=False)
    cos_a = np.cos(angles)
    sin_a = np.sin(angles)

    rings: list[np.ndarray] = []
    for i in range(n_pts):
        if i > 0:
            t_prev = tangents[i - 1]
            t_curr = tangents[i]
            axis = np.cross(t_prev, t_curr)
            axis_len = float(np.linalg.norm(axis))
            if axis_len > 1e-9:
                axis /= axis_len
                dot = max(-1.0, min(1.0, float(np.dot(t_prev, t_curr))))
                ang = math.acos(dot)
                # Rodrigues rotation of normal
                ca, sa = math.cos(ang), math.sin(ang)
                normal = (
                    normal * ca
                    + np.cross(axis, normal) * sa
                    + axis * float(np.dot(axis, normal)) * (1.0 - ca)
                )
                normal -= t_curr * float(np.dot(normal, t_curr))
                normal /= max(1e-12, float(np.linalg.norm(normal)))
            binormal = np.cross(t_curr, normal)

        frac = i / max(1, n_pts - 1)
        r = radius * (1.0 + (taper_end - 1.0) * frac)
        ring = spine[i][None, :] + r * (
            cos_a[:, None] * normal[None, :] + sin_a[:, None] * binormal[None, :]
        )
        rings.append(ring)

    verts = np.vstack(rings)
    faces: list[list[int]] = []
    ring_count = n_pts if closed else n_pts - 1
    for i in range(ring_count):
        i_next = (i + 1) % n_pts
        b0 = i * seg
        b1 = i_next * seg
        for j in range(seg):
            jn = (j + 1) % seg
            a = b0 + j
            b = b0 + jn
            c = b1 + jn
            d = b1 + j
            faces.append([a, b, c])
            faces.append([a, c, d])

    if not closed:
        # Add start and end cap centers so tube is watertight
        start_center_idx = len(verts)
        end_center_idx = len(verts) + 1
        verts = np.vstack([verts, spine[0][None, :], spine[-1][None, :]])
        last_base = (n_pts - 1) * seg
        for j in range(seg):
            jn = (j + 1) % seg
            faces.append([start_center_idx, jn, j])
            faces.append([end_center_idx, last_base + j, last_base + jn])

    return MeshData(vertices=verts, faces=np.asarray(faces, dtype=np.int64))
