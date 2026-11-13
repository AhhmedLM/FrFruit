"""3D primitive mesh generators."""

from __future__ import annotations

from typing import Literal

import manifold3d
import numpy as np
import trimesh

from frfruit.core.types import Vec2, Vec3
from frfruit.geometry.mesh import MeshData


def _apply_origin(mesh: MeshData, origin: Literal["center", "bottom"]) -> MeshData:
    if origin == "bottom" and not mesh.is_empty:
        min_z = float(np.min(mesh.vertices[:, 2]))
        mesh.vertices[:, 2] -= min_z
    return mesh


def make_cube(
    size: float | Vec3 = 1.0,
    origin: Literal["center", "bottom"] = "center",
) -> MeshData:
    """Create an axis-aligned box/cube primitive."""
    dims = (float(size), float(size), float(size)) if isinstance(size, (int, float)) else size
    mf = manifold3d.Manifold.cube(list(dims), center=True)
    return _apply_origin(MeshData.from_manifold(mf), origin)


def make_sphere(
    radius: float = 0.5,
    segments: int = 24,
    origin: Literal["center", "bottom"] = "center",
) -> MeshData:
    """Create a UV sphere primitive."""
    tm = trimesh.creation.uv_sphere(radius=radius, count=[max(8, segments // 2), segments])
    return _apply_origin(MeshData.from_trimesh(tm), origin)


def make_icosphere(
    radius: float = 0.5,
    subdivisions: int = 2,
    origin: Literal["center", "bottom"] = "center",
) -> MeshData:
    """Create a geodesic icosphere primitive."""
    tm = trimesh.creation.icosphere(subdivisions=max(0, min(4, subdivisions)), radius=radius)
    return _apply_origin(MeshData.from_trimesh(tm), origin)


def make_cylinder(
    radius: float = 0.5,
    height: float = 1.0,
    radius_top: float | None = None,
    segments: int = 24,
    origin: Literal["center", "bottom"] = "center",
) -> MeshData:
    """Create a cylinder or frustum primitive."""
    r_top = radius if radius_top is None else radius_top
    mf = manifold3d.Manifold.cylinder(
        height=float(height),
        radius_low=float(radius),
        radius_high=float(r_top),
        circular_segments=max(3, segments),
        center=True,
    )
    return _apply_origin(MeshData.from_manifold(mf), origin)


def make_cone(
    radius: float = 0.5,
    height: float = 1.0,
    segments: int = 24,
    origin: Literal["center", "bottom"] = "center",
) -> MeshData:
    """Create a cone primitive."""
    return make_cylinder(
        radius=radius,
        height=height,
        radius_top=0.0,
        segments=segments,
        origin=origin,
    )


def make_torus(
    major_radius: float = 0.5,
    minor_radius: float = 0.15,
    major_segments: int = 32,
    minor_segments: int = 16,
    origin: Literal["center", "bottom"] = "center",
) -> MeshData:
    """Create a torus primitive via trimesh."""
    tm = trimesh.creation.torus(
        major_radius=major_radius,
        minor_radius=minor_radius,
        major_sections=max(8, major_segments),
        minor_sections=max(6, minor_segments),
    )
    return _apply_origin(MeshData.from_trimesh(tm), origin)


def make_capsule(
    radius: float = 0.3,
    height: float = 1.0,
    segments: int = 24,
    origin: Literal["center", "bottom"] = "center",
) -> MeshData:
    """Create a capsule primitive."""
    cyl_h = max(0.01, float(height) - 2.0 * float(radius))
    tm = trimesh.creation.capsule(
        height=cyl_h,
        radius=radius,
        count=[max(8, segments // 2), segments],
    )
    return _apply_origin(MeshData.from_trimesh(tm), origin)


def make_plane(
    size: float | Vec2 = 2.0,
    subdivisions: int = 1,
) -> MeshData:
    """Create a subdivided horizontal grid plane on Z=0."""
    sx, sy = (float(size), float(size)) if isinstance(size, (int, float)) else size
    nx = max(1, int(subdivisions))
    ny = max(1, int(subdivisions))
    xs = np.linspace(-0.5 * sx, 0.5 * sx, nx + 1)
    ys = np.linspace(-0.5 * sy, 0.5 * sy, ny + 1)
    gx, gy = np.meshgrid(xs, ys, indexing="xy")
    verts = np.column_stack([gx.ravel(), gy.ravel(), np.zeros(gx.size, dtype=np.float64)])
    uvs = np.column_stack(
        [
            np.tile(np.linspace(0.0, 1.0, nx + 1), ny + 1),
            np.repeat(np.linspace(0.0, 1.0, ny + 1), nx + 1),
        ]
    )
    faces: list[list[int]] = []
    stride = nx + 1
    for j in range(ny):
        for i in range(nx):
            a = j * stride + i
            b = a + 1
            c = a + stride + 1
            d = a + stride
            faces.append([a, b, c])
            faces.append([a, c, d])
    return MeshData(vertices=verts, faces=np.asarray(faces, dtype=np.int64), uvs=uvs)
