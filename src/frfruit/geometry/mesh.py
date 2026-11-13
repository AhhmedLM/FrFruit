"""Core MeshData container and interop with trimesh and manifold3d."""

from __future__ import annotations

from dataclasses import dataclass

import manifold3d
import numpy as np
import trimesh

from frfruit.core.math3d import compute_aabb
from frfruit.core.types import BoundingBox, MeshPayload


class GeometryError(ValueError):
    """Raised when a geometry operation cannot proceed on the given mesh."""


@dataclass
class MeshData:
    """In-memory triangle mesh representation."""

    vertices: np.ndarray  # (N, 3) float64
    faces: np.ndarray  # (M, 3) int64
    uvs: np.ndarray | None = None  # (N, 2) float64
    face_groups: np.ndarray | None = None  # (M,) int32
    vertex_colors: np.ndarray | None = None  # (N, 4) float64

    def __post_init__(self) -> None:
        self.vertices = np.asarray(self.vertices, dtype=np.float64)
        self.faces = np.asarray(self.faces, dtype=np.int64)
        if self.vertices.ndim != 2 or (len(self.vertices) > 0 and self.vertices.shape[1] != 3):
            self.vertices = self.vertices.reshape((-1, 3))
        if self.faces.ndim != 2 or (len(self.faces) > 0 and self.faces.shape[1] != 3):
            self.faces = self.faces.reshape((-1, 3))
        if self.uvs is not None:
            self.uvs = np.asarray(self.uvs, dtype=np.float64).reshape((-1, 2))
        if self.face_groups is not None:
            self.face_groups = np.asarray(self.face_groups, dtype=np.int32).reshape((-1,))
        if self.vertex_colors is not None:
            self.vertex_colors = np.asarray(self.vertex_colors, dtype=np.float64).reshape((-1, 4))

    def copy(self) -> MeshData:
        return MeshData(
            vertices=self.vertices.copy(),
            faces=self.faces.copy(),
            uvs=None if self.uvs is None else self.uvs.copy(),
            face_groups=None if self.face_groups is None else self.face_groups.copy(),
            vertex_colors=None if self.vertex_colors is None else self.vertex_colors.copy(),
        )

    @property
    def is_empty(self) -> bool:
        return len(self.vertices) == 0 or len(self.faces) == 0

    def to_payload(self) -> MeshPayload:
        return MeshPayload(
            vertices=np.round(self.vertices, 7).tolist(),
            faces=self.faces.tolist(),
            uvs=None if self.uvs is None else np.round(self.uvs, 6).tolist(),
            face_groups=None if self.face_groups is None else self.face_groups.tolist(),
            vertex_colors=(
                None if self.vertex_colors is None else np.round(self.vertex_colors, 5).tolist()
            ),
        )

    @classmethod
    def from_payload(cls, payload: MeshPayload) -> MeshData:
        verts = (
            np.asarray(payload.vertices, dtype=np.float64)
            if payload.vertices
            else np.zeros((0, 3), dtype=np.float64)
        )
        faces = (
            np.asarray(payload.faces, dtype=np.int64)
            if payload.faces
            else np.zeros((0, 3), dtype=np.int64)
        )
        uvs = None if payload.uvs is None else np.asarray(payload.uvs, dtype=np.float64)
        fg = (
            None if payload.face_groups is None else np.asarray(payload.face_groups, dtype=np.int32)
        )
        vc = (
            None
            if payload.vertex_colors is None
            else np.asarray(payload.vertex_colors, dtype=np.float64)
        )
        return cls(vertices=verts, faces=faces, uvs=uvs, face_groups=fg, vertex_colors=vc)

    def to_trimesh(self, process: bool = False) -> trimesh.Trimesh:
        return trimesh.Trimesh(vertices=self.vertices, faces=self.faces, process=process)

    @classmethod
    def from_trimesh(cls, tm: trimesh.Trimesh) -> MeshData:
        uvs = None
        if hasattr(tm.visual, "uv") and tm.visual.uv is not None:
            uv_arr = np.asarray(tm.visual.uv, dtype=np.float64)
            if len(uv_arr) == len(tm.vertices):
                uvs = uv_arr[:, :2]
        return cls(
            vertices=np.asarray(tm.vertices, dtype=np.float64),
            faces=np.asarray(tm.faces, dtype=np.int64),
            uvs=uvs,
        )

    def to_manifold(self, merge_tolerance: float = 1e-6) -> manifold3d.Manifold:
        """Convert MeshData into a manifold3d.Manifold solid."""
        if self.is_empty:
            return manifold3d.Manifold()
        # Merge nearly coincident vertices cleanly so CSG stays watertight
        tm = trimesh.Trimesh(vertices=self.vertices, faces=self.faces, process=False)
        tm.merge_vertices(digits_vertex=6)
        trimesh.repair.fix_normals(tm)
        verts = np.asarray(tm.vertices, dtype=np.float32)
        tris = np.asarray(tm.faces, dtype=np.uint32)
        m = manifold3d.Mesh(vert_properties=verts, tri_verts=tris)
        m.merge()
        mf = manifold3d.Manifold(m)
        if mf.status() != manifold3d.Error.NoError or mf.is_empty():
            raise GeometryError(
                f"Mesh is not a valid closed manifold solid (status={mf.status()}). "
                "For open sheets/planes, apply Solidify modifier or repair_manifold first."
            )
        return mf

    @classmethod
    def from_manifold(cls, mf: manifold3d.Manifold) -> MeshData:
        if mf.is_empty():
            return cls(
                vertices=np.zeros((0, 3), dtype=np.float64),
                faces=np.zeros((0, 3), dtype=np.int64),
            )
        m = mf.to_mesh()
        verts = np.asarray(m.vert_properties[:, :3], dtype=np.float64)
        faces = np.asarray(m.tri_verts, dtype=np.int64)
        return cls(vertices=verts, faces=faces)

    def face_normals(self) -> np.ndarray:
        if len(self.faces) == 0:
            return np.zeros((0, 3), dtype=np.float64)
        v0 = self.vertices[self.faces[:, 0]]
        v1 = self.vertices[self.faces[:, 1]]
        v2 = self.vertices[self.faces[:, 2]]
        cross = np.cross(v1 - v0, v2 - v0)
        lens = np.linalg.norm(cross, axis=1, keepdims=True)
        lens = np.where(lens < 1e-12, 1.0, lens)
        return cross / lens

    def vertex_normals(self) -> np.ndarray:
        if len(self.vertices) == 0:
            return np.zeros((0, 3), dtype=np.float64)
        v0 = self.vertices[self.faces[:, 0]]
        v1 = self.vertices[self.faces[:, 1]]
        v2 = self.vertices[self.faces[:, 2]]
        area_normals = np.cross(v1 - v0, v2 - v0)
        vn = np.zeros_like(self.vertices)
        np.add.at(vn, self.faces[:, 0], area_normals)
        np.add.at(vn, self.faces[:, 1], area_normals)
        np.add.at(vn, self.faces[:, 2], area_normals)
        lens = np.linalg.norm(vn, axis=1, keepdims=True)
        lens = np.where(lens < 1e-12, 1.0, lens)
        return vn / lens

    def bounds(self) -> BoundingBox:
        return compute_aabb(self.vertices)

    def stats(self) -> dict[str, object]:
        if self.is_empty:
            return {
                "vertex_count": 0,
                "face_count": 0,
                "edge_count": 0,
                "is_watertight": False,
                "is_manifold": False,
                "volume": 0.0,
                "surface_area": 0.0,
            }
        tm = self.to_trimesh(process=False)
        edges_sorted = np.sort(
            np.vstack([self.faces[:, [0, 1]], self.faces[:, [1, 2]], self.faces[:, [2, 0]]]),
            axis=1,
        )
        unique_edges = len(np.unique(edges_sorted, axis=0))
        is_manifold = False
        try:
            mf = self.to_manifold()
            is_manifold = mf.status() == manifold3d.Error.NoError and not mf.is_empty()
        except GeometryError:
            is_manifold = False
        bbox = self.bounds()
        return {
            "vertex_count": int(len(self.vertices)),
            "face_count": int(len(self.faces)),
            "edge_count": int(unique_edges),
            "is_watertight": bool(tm.is_watertight),
            "is_manifold": bool(is_manifold),
            "volume": float(abs(tm.volume)) if tm.is_watertight else 0.0,
            "surface_area": float(tm.area),
            "bounds_min": list(bbox.min_corner),
            "bounds_max": list(bbox.max_corner),
            "size": list(bbox.size),
        }


def concat_meshes(meshes: list[MeshData]) -> MeshData:
    """Merge multiple MeshData objects into one single MeshData."""
    non_empty = [m for m in meshes if not m.is_empty]
    if not non_empty:
        return MeshData(
            vertices=np.zeros((0, 3), dtype=np.float64),
            faces=np.zeros((0, 3), dtype=np.int64),
        )
    all_verts: list[np.ndarray] = []
    all_faces: list[np.ndarray] = []
    all_groups: list[np.ndarray] = []
    offset = 0
    for idx, m in enumerate(non_empty):
        all_verts.append(m.vertices)
        all_faces.append(m.faces + offset)
        if m.face_groups is not None and len(m.face_groups) == len(m.faces):
            all_groups.append(m.face_groups)
        else:
            all_groups.append(np.full((len(m.faces),), idx, dtype=np.int32))
        offset += len(m.vertices)
    return MeshData(
        vertices=np.vstack(all_verts),
        faces=np.vstack(all_faces),
        face_groups=np.concatenate(all_groups),
    )
