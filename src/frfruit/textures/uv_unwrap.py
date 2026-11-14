"""Smart angle-based UV chart splitting, tangent-plane PCA projection, and atlas bin packing."""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass

import numpy as np

from frfruit.core.types import Vec2
from frfruit.geometry.mesh import MeshData


@dataclass
class _UVIsland:
    face_indices: list[int]
    corner_uvs: list[np.ndarray]  # per-face (3, 2) local UVs normalized to min=(0, 0)
    width: float
    height: float
    placed_x: float = 0.0
    placed_y: float = 0.0


def _basis_for_island(
    verts: np.ndarray, faces: np.ndarray, f_indices: list[int], avg_n: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute centered origin and PCA tangent axes (T, B) orthogonal to `avg_n`."""
    unique_vi = sorted({int(v) for fi in f_indices for v in faces[fi]})
    pts = verts[unique_vi]
    origin = np.mean(pts, axis=0)
    rel = pts - origin[None, :]
    # Project points into plane orthogonal to avg_n
    rel_proj = rel - (rel @ avg_n)[:, None] * avg_n[None, :]

    up_ref = np.array([1.0, 0.0, 0.0]) if abs(float(avg_n[2])) > 0.92 else np.array([0.0, 0.0, 1.0])
    t0 = np.cross(up_ref, avg_n)
    t0 /= max(1e-12, float(np.linalg.norm(t0)))
    b0 = np.cross(avg_n, t0)
    b0 /= max(1e-12, float(np.linalg.norm(b0)))

    if len(rel_proj) >= 3:
        coords_2d = np.column_stack([rel_proj @ t0, rel_proj @ b0])
        cov = np.cov(coords_2d.T) if len(coords_2d) > 1 else np.eye(2)
        if cov.shape == (2, 2) and np.all(np.isfinite(cov)):
            vals, vecs = np.linalg.eigh(cov)
            major = vecs[:, int(np.argmax(vals))]
            t_vec = major[0] * t0 + major[1] * b0
            t_len = float(np.linalg.norm(t_vec))
            if t_len > 1e-12:
                t_vec /= t_len
                b_vec = np.cross(avg_n, t_vec)
                b_vec /= max(1e-12, float(np.linalg.norm(b_vec)))
                return origin, t_vec, b_vec
    return origin, t0, b0


def smart_uv_unwrap(
    mesh: MeshData,
    angle_limit_deg: float = 66.0,
    island_margin: float = 0.02,
    uv_scale: Vec2 = (1.0, 1.0),
) -> tuple[MeshData, dict[str, float | int]]:
    """Unwrap mesh into non-overlapping `[0, 1]^2` UV islands with seam vertex splitting.

    Partitions triangles into dihedral-angle charts, projects each island onto its optimal
    PCA tangent frame, packs islands cleanly into `[0, 1]^2` with configurable margin, and
    splits seam vertices so textured sampling produces zero stretching across hard edges.
    """
    if mesh.is_empty:
        return mesh.copy(), {"uv_island_count": 0, "uv_coverage": 0.0}

    verts = mesh.vertices
    faces = mesh.faces
    n_faces = len(faces)
    fnormals = mesh.face_normals()

    # Face triangle areas for area-weighted island normals
    v0 = verts[faces[:, 0]]
    v1 = verts[faces[:, 1]]
    v2 = verts[faces[:, 2]]
    areas = 0.5 * np.linalg.norm(np.cross(v1 - v0, v2 - v0), axis=1)

    # Undirected edge adjacency (only pair across matching winding / shared vertices)
    edge_faces: dict[tuple[int, int], list[int]] = defaultdict(list)
    for fi, (a, b, c) in enumerate(faces):
        ia, ib, ic = int(a), int(b), int(c)
        edge_faces[(min(ia, ib), max(ia, ib))].append(fi)
        edge_faces[(min(ib, ic), max(ib, ic))].append(fi)
        edge_faces[(min(ic, ia), max(ic, ia))].append(fi)

    adj: list[list[int]] = [[] for _ in range(n_faces)]
    for flist in edge_faces.values():
        if len(flist) == 2:
            f1, f2 = flist
            adj[f1].append(f2)
            adj[f2].append(f1)

    step_cos = math.cos(math.radians(max(1.0, min(179.0, angle_limit_deg))))
    seed_cos = math.cos(math.radians(max(1.0, min(88.0, angle_limit_deg * 1.15))))

    visited = np.zeros((n_faces,), dtype=bool)
    raw_islands: list[list[int]] = []
    for seed in range(n_faces):
        if visited[seed]:
            continue
        visited[seed] = True
        n_seed = fnormals[seed]
        comp: list[int] = [seed]
        queue = [seed]
        while queue:
            cur = queue.pop(0)
            n_cur = fnormals[cur]
            for nb in adj[cur]:
                if visited[nb]:
                    continue
                if (
                    float(np.dot(n_cur, fnormals[nb])) >= step_cos
                    and float(np.dot(n_seed, fnormals[nb])) >= seed_cos
                ):
                    visited[nb] = True
                    comp.append(nb)
                    queue.append(nb)
        raw_islands.append(comp)

    islands: list[_UVIsland] = []
    total_uv_area = 0.0
    for f_indices in raw_islands:
        n_weighted = np.sum(
            fnormals[f_indices] * np.maximum(areas[f_indices], 1e-8)[:, None], axis=0
        )
        nw_len = float(np.linalg.norm(n_weighted))
        avg_n = n_weighted / nw_len if nw_len > 1e-12 else fnormals[f_indices[0]]

        origin, t_vec, b_vec = _basis_for_island(verts, faces, f_indices, avg_n)

        tri_uvs: list[np.ndarray] = []
        min_u = float("inf")
        min_v = float("inf")
        max_u = float("-inf")
        max_v = float("-inf")
        for fi in f_indices:
            tri_pts = verts[faces[fi]] - origin[None, :]
            uv_tri = np.column_stack([tri_pts @ t_vec, tri_pts @ b_vec])
            min_u = min(min_u, float(np.min(uv_tri[:, 0])))
            min_v = min(min_v, float(np.min(uv_tri[:, 1])))
            max_u = max(max_u, float(np.max(uv_tri[:, 0])))
            max_v = max(max_v, float(np.max(uv_tri[:, 1])))
            tri_uvs.append(uv_tri)

        w_i = max(1e-5, max_u - min_u)
        h_i = max(1e-5, max_v - min_v)
        shifted = [uv - np.array([[min_u, min_v]], dtype=np.float64) for uv in tri_uvs]
        total_uv_area += w_i * h_i
        islands.append(
            _UVIsland(
                face_indices=f_indices,
                corner_uvs=shifted,
                width=w_i,
                height=h_i,
            )
        )

    # Shelf bin packing into near-square atlas normalized to [0, 1]^2
    max_w = max(isl.width for isl in islands)
    max_h = max(isl.height for isl in islands)
    margin_ratio = max(0.0, min(0.15, float(island_margin)))
    est_side = max(max_w, max_h, math.sqrt(total_uv_area * 1.28))
    pad = est_side * margin_ratio
    row_limit = max(max_w + 2.0 * pad, est_side * (1.0 + margin_ratio * 2.0))

    islands.sort(key=lambda isl: (isl.height, isl.width), reverse=True)
    cur_x = pad
    cur_y = pad
    shelf_h = 0.0
    used_w = pad
    for isl in islands:
        if cur_x > pad and (cur_x + isl.width + pad) > row_limit:
            cur_x = pad
            cur_y += shelf_h + pad
            shelf_h = 0.0
        isl.placed_x = cur_x
        isl.placed_y = cur_y
        cur_x += isl.width + pad
        used_w = max(used_w, cur_x)
        shelf_h = max(shelf_h, isl.height)

    atlas_side = max(1e-6, max(used_w, cur_y + shelf_h + pad))

    # Split seam vertices cleanly per (original_vertex_idx, island_idx)
    new_verts: list[np.ndarray] = []
    new_uvs: list[np.ndarray] = []
    new_faces = np.empty_like(faces)
    vert_map: dict[tuple[int, int], int] = {}

    uv_area_sum = 0.0
    for isl_id, isl in enumerate(islands):
        for fi, tri_uv in zip(isl.face_indices, isl.corner_uvs, strict=True):
            uv_norm = (
                tri_uv + np.array([[isl.placed_x, isl.placed_y]], dtype=np.float64)
            ) / atlas_side
            uv_norm[:, 0] *= float(uv_scale[0])
            uv_norm[:, 1] *= float(uv_scale[1])

            # Track normalized UV triangle area
            e1 = uv_norm[1] - uv_norm[0]
            e2 = uv_norm[2] - uv_norm[0]
            uv_area_sum += 0.5 * abs(float(e1[0] * e2[1] - e1[1] * e2[0]))

            for ci in range(3):
                orig_vi = int(faces[fi, ci])
                key = (orig_vi, isl_id)
                nvi = vert_map.get(key)
                if nvi is None:
                    nvi = len(new_verts)
                    vert_map[key] = nvi
                    new_verts.append(verts[orig_vi])
                    new_uvs.append(uv_norm[ci])
                new_faces[fi, ci] = nvi

    out_mesh = MeshData(
        vertices=np.asarray(new_verts, dtype=np.float64),
        faces=new_faces,
        uvs=np.asarray(new_uvs, dtype=np.float64),
        face_groups=mesh.face_groups,
    )
    coverage = round(uv_area_sum / max(1e-6, float(uv_scale[0] * uv_scale[1])), 4)
    return out_mesh, {"uv_island_count": len(islands), "uv_coverage": coverage}
