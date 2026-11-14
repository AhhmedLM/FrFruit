"""AI-friendly semantic Edit Mesh & full Edit Mode modeling operations."""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np
import trimesh

from frfruit.core.math3d import compose_matrix, transform_points
from frfruit.core.types import SelectionQuery, Vec2, Vec3
from frfruit.geometry.mesh import GeometryError, MeshData, concat_meshes, safe_fix_normals


@dataclass
class EditOpOutput:
    """Structured result of an Edit Mode modeling operation with per-face material remapping."""

    mesh: MeshData
    face_material_ids: dict[int, str] = field(default_factory=dict)
    separated_mesh: MeshData | None = None
    separated_face_material_ids: dict[int, str] = field(default_factory=dict)
    extra: dict[str, Any] = field(default_factory=dict)


def _remap_face_materials(
    old_mats: dict[int, str] | None,
    face_sources: list[int] | np.ndarray,
    *,
    override_indices: set[int] | None = None,
    override_material_id: str | None = None,
) -> dict[int, str]:
    """Remap per-face material overrides cleanly across topology changes."""
    out: dict[int, str] = {}
    old = old_mats or {}
    ov_set = override_indices or set()
    for new_idx, src_idx in enumerate(face_sources):
        if override_material_id is not None and new_idx in ov_set:
            out[int(new_idx)] = override_material_id
        elif int(src_idx) >= 0 and int(src_idx) in old:
            out[int(new_idx)] = old[int(src_idx)]
    return out


def _interp_uvs(
    uvs: np.ndarray | None,
    idx_a: int,
    idx_b: int,
    t: float,
) -> list[float] | None:
    if uvs is None:
        return None
    val = uvs[idx_a] * (1.0 - t) + uvs[idx_b] * t
    return [float(val[0]), float(val[1])]


def _build_directed_edge_map(faces: np.ndarray) -> dict[tuple[int, int], list[int]]:
    d2f: dict[tuple[int, int], list[int]] = defaultdict(list)
    for fi, (a, b, c) in enumerate(faces):
        ia, ib, ic = int(a), int(b), int(c)
        d2f[(ia, ib)].append(fi)
        d2f[(ib, ic)].append(fi)
        d2f[(ic, ia)].append(fi)
    return d2f


def _patch_boundary_edges(faces: np.ndarray, sel_mask: np.ndarray) -> list[tuple[int, int]]:
    """Return directed boundary edges (u, v) of the selected face patch."""
    selected_faces = faces[sel_mask]
    directed: set[tuple[int, int]] = set()
    for a, b, c in selected_faces:
        ia, ib, ic = int(a), int(b), int(c)
        directed.add((ia, ib))
        directed.add((ib, ic))
        directed.add((ic, ia))
    return [(u, v) for (u, v) in directed if (v, u) not in directed]


def _extract_open_boundary_edges(faces: np.ndarray) -> list[tuple[int, int]]:
    """Return all directed half-edges (u, v) on open boundaries of a mesh."""
    directed: set[tuple[int, int]] = set()
    for a, b, c in faces:
        ia, ib, ic = int(a), int(b), int(c)
        directed.add((ia, ib))
        directed.add((ib, ic))
        directed.add((ic, ia))
    return [(u, v) for (u, v) in directed if (v, u) not in directed]


def _chain_boundary_loops(edges: list[tuple[int, int]]) -> list[list[int]]:
    """Chain directed open boundary edges (u -> v) into ordered vertex loops."""
    if not edges:
        return []
    adj: dict[int, list[int]] = defaultdict(list)
    remaining: set[tuple[int, int]] = set(edges)
    for u, v in edges:
        adj[u].append(v)

    loops: list[list[int]] = []
    while remaining:
        u0, v0 = min(remaining)
        remaining.remove((u0, v0))
        loop = [u0, v0]
        cur = v0
        guard = 0
        while cur != u0 and guard < len(edges) + 4:
            guard += 1
            nxt_cands = [n for n in adj.get(cur, []) if (cur, n) in remaining]
            if not nxt_cands:
                break
            nxt = nxt_cands[0]
            remaining.remove((cur, nxt))
            if nxt == u0:
                break
            loop.append(nxt)
            cur = nxt
        if len(loop) >= 3:
            loops.append(loop)
    return loops


def resolve_face_selection(
    mesh: MeshData,
    query: SelectionQuery,
    face_material_ids: dict[int, str] | None = None,
    base_material_id: str | None = None,
) -> np.ndarray:
    """Resolve a semantic SelectionQuery into a boolean mask over faces."""
    m_count = len(mesh.faces)
    if m_count == 0:
        return np.zeros((0,), dtype=bool)

    mode = query.mode
    if mode == "all":
        mask = np.ones((m_count,), dtype=bool)
        return ~mask if query.invert else mask

    dir_map: dict[str, Vec3] = {
        "top": (0.0, 0.0, 1.0),
        "bottom": (0.0, 0.0, -1.0),
        "front": (0.0, -1.0, 0.0),
        "back": (0.0, 1.0, 0.0),
        "left": (-1.0, 0.0, 0.0),
        "right": (1.0, 0.0, 0.0),
    }
    if mode in dir_map or mode == "normal_dir":
        target = dir_map[mode] if mode in dir_map else (query.direction or (0.0, 0.0, 1.0))
        t_vec = np.asarray(target, dtype=np.float64)
        t_len = float(np.linalg.norm(t_vec))
        if t_len < 1e-12:
            raise GeometryError("Selection direction vector cannot be zero.")
        t_vec /= t_len
        normals = mesh.face_normals()
        cos_thresh = math.cos(math.radians(query.angle_threshold_deg))
        mask = (normals @ t_vec) >= cos_thresh
    elif mode == "bbox":
        if query.bbox_min is None or query.bbox_max is None:
            raise GeometryError("bbox selection requires both bbox_min and bbox_max.")
        bmin = np.asarray(query.bbox_min, dtype=np.float64)
        bmax = np.asarray(query.bbox_max, dtype=np.float64)
        centroids = np.mean(mesh.vertices[mesh.faces], axis=1)
        mask = np.all((centroids >= bmin - 1e-6) & (centroids <= bmax + 1e-6), axis=1)
    elif mode == "indices":
        mask = np.zeros((m_count,), dtype=bool)
        if query.indices:
            valid = [int(i) for i in query.indices if 0 <= int(i) < m_count]
            mask[valid] = True
    elif mode == "vertex_indices":
        mask = np.zeros((m_count,), dtype=bool)
        if query.vertex_indices:
            vset = {int(v) for v in query.vertex_indices}
            for fi, (a, b, c) in enumerate(mesh.faces):
                if int(a) in vset or int(b) in vset or int(c) in vset:
                    mask[fi] = True
    elif mode == "edge_indices":
        mask = np.zeros((m_count,), dtype=bool)
        if query.edge_indices:
            eset = {(min(int(u), int(v)), max(int(u), int(v))) for u, v in query.edge_indices}
            for fi, (a, b, c) in enumerate(mesh.faces):
                for u, v in ((int(a), int(b)), (int(b), int(c)), (int(c), int(a))):
                    if (min(u, v), max(u, v)) in eset:
                        mask[fi] = True
                        break
    elif mode == "coplanar":
        normals = mesh.face_normals()
        if query.seed_face is not None and 0 <= query.seed_face < m_count:
            seeds = [int(query.seed_face)]
        elif query.indices:
            seeds = [int(i) for i in query.indices if 0 <= int(i) < m_count]
        else:
            t_vec = np.asarray(query.direction or (0.0, 0.0, 1.0), dtype=np.float64)
            t_vec /= max(1e-12, float(np.linalg.norm(t_vec)))
            seeds = [int(np.argmax(normals @ t_vec))]
        if not seeds:
            mask = np.zeros((m_count,), dtype=bool)
        else:
            d2f = _build_directed_edge_map(mesh.faces)
            adj: list[list[int]] = [[] for _ in range(m_count)]
            for (u, v), f1s in d2f.items():
                f2s = d2f.get((v, u), [])
                for f1 in f1s:
                    for f2 in f2s:
                        if f1 != f2:
                            adj[f1].append(f2)
            thresh = math.cos(math.radians(min(query.angle_threshold_deg, 25.0)))
            mask = np.zeros((m_count,), dtype=bool)
            stack = list(seeds)
            for s in seeds:
                mask[s] = True
            while stack:
                cur = stack.pop()
                for nxt in adj[cur]:
                    if not mask[nxt] and float(np.dot(normals[cur], normals[nxt])) >= thresh:
                        mask[nxt] = True
                        stack.append(nxt)
    elif mode == "boundary":
        open_edges = _extract_open_boundary_edges(mesh.faces)
        b_set = {(min(u, v), max(u, v)) for u, v in open_edges}
        mask = np.zeros((m_count,), dtype=bool)
        for fi, (a, b, c) in enumerate(mesh.faces):
            for u, v in ((int(a), int(b)), (int(b), int(c)), (int(c), int(a))):
                if (min(u, v), max(u, v)) in b_set:
                    mask[fi] = True
                    break
    elif mode == "material":
        target_mat = query.material_id
        fm = face_material_ids or {}
        mask = np.zeros((m_count,), dtype=bool)
        for fi in range(m_count):
            eff_mat = fm.get(fi, base_material_id)
            if eff_mat == target_mat:
                mask[fi] = True
    else:
        raise GeometryError(f"Unsupported selection mode: {mode}")

    return ~mask if query.invert else mask


def resolve_vertex_selection(
    mesh: MeshData,
    query: SelectionQuery,
    face_material_ids: dict[int, str] | None = None,
    base_material_id: str | None = None,
) -> np.ndarray:
    """Resolve a SelectionQuery into a boolean mask over vertices."""
    n_verts = len(mesh.vertices)
    if n_verts == 0:
        return np.zeros((0,), dtype=bool)

    if query.mode == "vertex_indices":
        mask = np.zeros((n_verts,), dtype=bool)
        if query.vertex_indices:
            valid = [int(v) for v in query.vertex_indices if 0 <= int(v) < n_verts]
            mask[valid] = True
        return ~mask if query.invert else mask

    if query.mode == "edge_indices":
        mask = np.zeros((n_verts,), dtype=bool)
        if query.edge_indices:
            for u, v in query.edge_indices:
                if 0 <= int(u) < n_verts:
                    mask[int(u)] = True
                if 0 <= int(v) < n_verts:
                    mask[int(v)] = True
        return ~mask if query.invert else mask

    if query.mode == "boundary":
        open_edges = _extract_open_boundary_edges(mesh.faces)
        mask = np.zeros((n_verts,), dtype=bool)
        for u, v in open_edges:
            if 0 <= u < n_verts:
                mask[u] = True
            if 0 <= v < n_verts:
                mask[v] = True
        return ~mask if query.invert else mask

    f_mask = resolve_face_selection(
        mesh, query, face_material_ids=face_material_ids, base_material_id=base_material_id
    )
    mask = np.zeros((n_verts,), dtype=bool)
    if np.any(f_mask):
        mask[np.unique(mesh.faces[f_mask].ravel())] = True
    return mask


def resolve_edge_selection(
    mesh: MeshData,
    query: SelectionQuery,
    face_material_ids: dict[int, str] | None = None,
    base_material_id: str | None = None,
) -> list[tuple[int, int]]:
    """Resolve a SelectionQuery into unique undirected edges (u, v) with u < v."""
    if mesh.is_empty:
        return []

    if query.mode == "edge_indices":
        n_v = len(mesh.vertices)
        out: set[tuple[int, int]] = set()
        for u, v in query.edge_indices or []:
            iu, iv = int(u), int(v)
            if 0 <= iu < n_v and 0 <= iv < n_v and iu != iv:
                out.add((min(iu, iv), max(iu, iv)))
        if not query.invert:
            return sorted(out)
        all_edges = {
            (min(int(x), int(y)), max(int(x), int(y)))
            for a, b, c in mesh.faces
            for x, y in ((a, b), (b, c), (c, a))
        }
        return sorted(all_edges - out)

    if query.mode == "boundary" and not query.invert:
        open_e = _extract_open_boundary_edges(mesh.faces)
        return sorted({(min(u, v), max(u, v)) for u, v in open_e})

    f_mask = resolve_face_selection(
        mesh, query, face_material_ids=face_material_ids, base_material_id=base_material_id
    )
    if not np.any(f_mask):
        return []
    # If partial selection and not "all", prefer boundary edges if interior is contiguous;
    # otherwise include all edges of selected faces when explicit vertex/edge selection used.
    if query.mode in ("top", "bottom", "front", "back", "left", "right", "normal_dir", "coplanar"):
        b_edges = _patch_boundary_edges(mesh.faces, f_mask)
        if b_edges:
            return sorted({(min(u, v), max(u, v)) for u, v in b_edges})
    edges: set[tuple[int, int]] = set()
    for a, b, c in mesh.faces[f_mask]:
        ia, ib, ic = int(a), int(b), int(c)
        edges.add((min(ia, ib), max(ia, ib)))
        edges.add((min(ib, ic), max(ib, ic)))
        edges.add((min(ic, ia), max(ic, ia)))
    return sorted(edges)


def inspect_mesh_elements(
    mesh: MeshData,
    *,
    detail: Literal["summary", "faces", "edges", "vertices", "selection", "materials"] = "summary",
    query: SelectionQuery | None = None,
    face_material_ids: dict[int, str] | None = None,
    base_material_id: str | None = None,
    offset: int = 0,
    limit: int = 64,
) -> dict[str, Any]:
    """Inspect Edit Mode elements (vertices, edges, faces, per-face materials, selection)."""
    fm = face_material_ids or {}
    out: dict[str, Any] = {
        "vertex_count": int(len(mesh.vertices)),
        "edge_count": 0,
        "face_count": int(len(mesh.faces)),
        "base_material_id": base_material_id,
        "face_material_override_count": len(fm),
    }
    if mesh.is_empty:
        return out

    fnormals = mesh.face_normals()
    d2f = _build_directed_edge_map(mesh.faces)
    undirected_edges: dict[tuple[int, int], list[int]] = defaultdict(list)
    for fi, (a, b, c) in enumerate(mesh.faces):
        for u, v in ((int(a), int(b)), (int(b), int(c)), (int(c), int(a))):
            undirected_edges[(min(u, v), max(u, v))].append(fi)
    edge_list = sorted(undirected_edges.keys())
    out["edge_count"] = len(edge_list)

    if query is not None:
        f_mask = resolve_face_selection(
            mesh, query, face_material_ids=fm, base_material_id=base_material_id
        )
        v_mask = resolve_vertex_selection(
            mesh, query, face_material_ids=fm, base_material_id=base_material_id
        )
        sel_edges = resolve_edge_selection(
            mesh, query, face_material_ids=fm, base_material_id=base_material_id
        )
        out["selection"] = {
            "selected_face_indices": np.nonzero(f_mask)[0].tolist(),
            "selected_vertex_indices": np.nonzero(v_mask)[0].tolist(),
            "selected_edges": [list(e) for e in sel_edges],
        }

    off = max(0, int(offset))
    lim = max(1, min(512, int(limit)))

    if detail == "faces":
        v0 = mesh.vertices[mesh.faces[:, 0]]
        v1 = mesh.vertices[mesh.faces[:, 1]]
        v2 = mesh.vertices[mesh.faces[:, 2]]
        centroids = np.mean(mesh.vertices[mesh.faces], axis=1)
        areas = 0.5 * np.linalg.norm(np.cross(v1 - v0, v2 - v0), axis=1)
        items = []
        for fi in range(off, min(len(mesh.faces), off + lim)):
            items.append(
                {
                    "index": fi,
                    "vertices": [int(x) for x in mesh.faces[fi]],
                    "centroid": [round(float(x), 6) for x in centroids[fi]],
                    "normal": [round(float(x), 5) for x in fnormals[fi]],
                    "area": round(float(areas[fi]), 6),
                    "material_id": fm.get(fi, base_material_id),
                    "has_face_override": fi in fm,
                }
            )
        out["faces"] = items
        out["offset"] = off
        out["limit"] = lim
    elif detail == "vertices":
        vn = mesh.vertex_normals()
        items = []
        for vi in range(off, min(len(mesh.vertices), off + lim)):
            items.append(
                {
                    "index": vi,
                    "position": [round(float(x), 6) for x in mesh.vertices[vi]],
                    "normal": [round(float(x), 5) for x in vn[vi]],
                    "uv": (
                        [round(float(x), 5) for x in mesh.uvs[vi]]
                        if mesh.uvs is not None and len(mesh.uvs) == len(mesh.vertices)
                        else None
                    ),
                }
            )
        out["vertices"] = items
        out["offset"] = off
        out["limit"] = lim
    elif detail == "edges":
        items = []
        for ei in range(off, min(len(edge_list), off + lim)):
            u, v = edge_list[ei]
            adj_f = undirected_edges[(u, v)]
            is_boundary = (u, v) not in d2f or (v, u) not in d2f
            dihedral_deg = 0.0
            if len(adj_f) == 2:
                dot = float(np.clip(np.dot(fnormals[adj_f[0]], fnormals[adj_f[1]]), -1.0, 1.0))
                dihedral_deg = math.degrees(math.acos(dot))
            items.append(
                {
                    "index": ei,
                    "vertices": [u, v],
                    "length": round(float(np.linalg.norm(mesh.vertices[v] - mesh.vertices[u])), 6),
                    "faces": adj_f,
                    "is_boundary": is_boundary,
                    "dihedral_deg": round(dihedral_deg, 3),
                }
            )
        out["edges"] = items
        out["offset"] = off
        out["limit"] = lim
    elif detail == "materials":
        by_mat: dict[str, list[int]] = defaultdict(list)
        for fi in range(len(mesh.faces)):
            eff = fm.get(fi, base_material_id) or "(default)"
            by_mat[eff].append(fi)
        out["face_materials"] = {str(k): v for k, v in fm.items()}
        out["faces_by_material"] = {k: v for k, v in by_mat.items()}

    return out


# ---------------------------------------------------------------------------
# Internal Implementations Returning (MeshData, face_sources)
# ---------------------------------------------------------------------------


def _extrude_faces_impl(
    mesh: MeshData,
    query: SelectionQuery,
    distance: float = 0.5,
    scale: float = 1.0,
    direction: Vec3 | None = None,
    individual: bool = False,
    along_normals: bool = False,
    face_material_ids: dict[int, str] | None = None,
    base_material_id: str | None = None,
) -> tuple[MeshData, list[int], set[int]]:
    if mesh.is_empty:
        return mesh.copy(), list(range(len(mesh.faces))), set()
    sel_mask = resolve_face_selection(
        mesh, query, face_material_ids=face_material_ids, base_material_id=base_material_id
    )
    if not np.any(sel_mask):
        raise GeometryError(f"Selection query ({query.mode}) matched 0 faces.")

    verts = mesh.vertices.copy()
    faces = mesh.faces.copy()
    normals = mesh.face_normals()
    uvs = None if mesh.uvs is None or len(mesh.uvs) != len(verts) else mesh.uvs.copy()
    sel_idx = np.nonzero(sel_mask)[0]
    m_orig = len(faces)
    face_sources = list(range(m_orig))

    custom_dir: np.ndarray | None = None
    if direction is not None:
        d_arr = np.asarray(direction, dtype=np.float64)
        d_len = float(np.linalg.norm(d_arr))
        if d_len < 1e-12:
            raise GeometryError("Extrude direction cannot be zero.")
        custom_dir = d_arr / d_len

    if individual:
        new_verts_list = verts.tolist()
        new_uvs_list = uvs.tolist() if uvs is not None else None
        wall_faces: list[list[int]] = []
        wall_sources: list[int] = []
        cap_and_wall: set[int] = set(int(i) for i in sel_idx)
        for fi in sel_idx:
            a, b, c = int(faces[fi, 0]), int(faces[fi, 1]), int(faces[fi, 2])
            f_dir = custom_dir if custom_dir is not None else normals[fi]
            f_cent = np.mean(verts[[a, b, c]], axis=0)
            va = len(new_verts_list)
            vb = va + 1
            vc = va + 2
            for vi in (a, b, c):
                pos = f_cent + (verts[vi] - f_cent) * scale + f_dir * distance
                new_verts_list.append(pos.tolist())
                if new_uvs_list is not None:
                    new_uvs_list.append(uvs[vi].tolist())  # type: ignore[index]
            faces[fi] = [va, vb, vc]
            for u, v, ue, ve in ((a, b, va, vb), (b, c, vb, vc), (c, a, vc, va)):
                wall_faces.append([u, v, ve])
                wall_faces.append([u, ve, ue])
                wall_sources.extend([int(fi), int(fi)])
        wall_start = len(faces)
        all_faces = np.vstack([faces, np.asarray(wall_faces, dtype=np.int64)])
        face_sources.extend(wall_sources)
        cap_and_wall.update(range(wall_start, len(all_faces)))
        out_mesh = MeshData(
            vertices=np.asarray(new_verts_list, dtype=np.float64),
            faces=all_faces,
            uvs=None if new_uvs_list is None else np.asarray(new_uvs_list, dtype=np.float64),
        )
        return out_mesh, face_sources, cap_and_wall

    if custom_dir is not None:
        avg_normal = custom_dir
    else:
        avg_normal = np.sum(normals[sel_idx], axis=0)
        n_len = float(np.linalg.norm(avg_normal))
        avg_normal = avg_normal / n_len if n_len > 1e-12 else np.array([0.0, 0.0, 1.0])

    unique_v = np.unique(faces[sel_idx].ravel())
    centroid = np.mean(verts[unique_v], axis=0)

    # Per-vertex normal across selected faces for along_normals mode
    per_v_normal: dict[int, np.ndarray] = {}
    if along_normals:
        v_acc: dict[int, np.ndarray] = defaultdict(lambda: np.zeros(3, dtype=np.float64))
        for fi in sel_idx:
            for vi in faces[fi]:
                v_acc[int(vi)] += normals[fi]
        for vi, acc in v_acc.items():
            ln = float(np.linalg.norm(acc))
            per_v_normal[vi] = acc / ln if ln > 1e-12 else avg_normal

    boundary_edges = _patch_boundary_edges(faces, sel_mask)
    if not boundary_edges:
        for vi in unique_v:
            n_dir = per_v_normal[int(vi)] if along_normals else avg_normal
            verts[vi] = centroid + (verts[vi] - centroid) * scale + n_dir * distance
        return (
            MeshData(vertices=verts, faces=faces, uvs=uvs),
            face_sources,
            {int(i) for i in sel_idx},
        )

    v_map: dict[int, int] = {}
    new_verts_list = verts.tolist()
    new_uvs_list = uvs.tolist() if uvs is not None else None
    for vi in unique_v:
        n_dir = per_v_normal[int(vi)] if along_normals else avg_normal
        pos = centroid + (verts[vi] - centroid) * scale + n_dir * distance
        new_idx = len(new_verts_list)
        new_verts_list.append(pos.tolist())
        if new_uvs_list is not None:
            new_uvs_list.append(uvs[vi].tolist())  # type: ignore[index]
        v_map[int(vi)] = new_idx

    # Track adjacent face for each boundary edge so side wall inherits neighbor material
    d2f = _build_directed_edge_map(faces)
    sel_set = {int(i) for i in sel_idx}
    for fi in sel_idx:
        a, b, c = int(faces[fi, 0]), int(faces[fi, 1]), int(faces[fi, 2])
        faces[fi] = [v_map[a], v_map[b], v_map[c]]

    wall_faces = []
    wall_sources = []
    for u, v in boundary_edges:
        u_ext = v_map[u]
        v_ext = v_map[v]
        cand = [fi for fi in d2f.get((u, v), []) if fi in sel_set]
        src_fi = cand[0] if cand else int(sel_idx[0])
        wall_faces.append([u, v, v_ext])
        wall_faces.append([u, v_ext, u_ext])
        wall_sources.extend([src_fi, src_fi])

    wall_start = len(faces)
    all_faces = np.vstack([faces, np.asarray(wall_faces, dtype=np.int64)])
    face_sources.extend(wall_sources)
    affected = set(sel_set)
    affected.update(range(wall_start, len(all_faces)))
    out_mesh = MeshData(
        vertices=np.asarray(new_verts_list, dtype=np.float64),
        faces=all_faces,
        uvs=None if new_uvs_list is None else np.asarray(new_uvs_list, dtype=np.float64),
    )
    return out_mesh, face_sources, affected


def extrude_faces(
    mesh: MeshData,
    query: SelectionQuery,
    distance: float = 0.5,
    scale: float = 1.0,
    direction: Vec3 | None = None,
    individual: bool = False,
    along_normals: bool = False,
) -> MeshData:
    """Extrude semantically selected faces (region, individual, along normals, or axis vector)."""
    out, _, _ = _extrude_faces_impl(
        mesh,
        query,
        distance=distance,
        scale=scale,
        direction=direction,
        individual=individual,
        along_normals=along_normals,
    )
    return out


def extrude_manifold(
    mesh: MeshData,
    query: SelectionQuery,
    distance: float = 0.5,
    scale: float = 1.0,
    direction: Vec3 | None = None,
) -> MeshData:
    """Extrude selected faces cleanly dissolving coplanar side walls on inward/manifold cuts."""
    ext = extrude_faces(mesh, query, distance=distance, scale=scale, direction=direction)
    try:
        mf = ext.to_manifold()
        return MeshData.from_manifold(mf)
    except GeometryError:
        return repair_manifold(ext)


def extrude_edges(
    mesh: MeshData,
    query: SelectionQuery,
    distance: float = 0.3,
    direction: Vec3 | None = None,
) -> MeshData:
    """Extrude selected/boundary edges into quad strips."""
    if mesh.is_empty:
        return mesh.copy()
    edges = resolve_edge_selection(mesh, query)
    if not edges:
        raise GeometryError(f"Edge selection ({query.mode}) matched 0 edges.")

    verts = mesh.vertices.copy()
    faces = mesh.faces.copy()
    uvs = None if mesh.uvs is None or len(mesh.uvs) != len(verts) else mesh.uvs.copy()
    fnormals = mesh.face_normals()

    custom_dir: np.ndarray | None = None
    if direction is not None:
        d = np.asarray(direction, dtype=np.float64)
        ln = float(np.linalg.norm(d))
        if ln < 1e-12:
            raise GeometryError("Extrude edge direction cannot be zero.")
        custom_dir = d / ln

    # Average adjacent face normals per vertex as default edge extrude direction
    v_norms = np.zeros_like(verts)
    for fi, (a, b, c) in enumerate(faces):
        v_norms[int(a)] += fnormals[fi]
        v_norms[int(b)] += fnormals[fi]
        v_norms[int(c)] += fnormals[fi]

    d2f = _build_directed_edge_map(faces)
    sel_verts = sorted({u for u, _ in edges} | {v for _, v in edges})
    v_map: dict[int, int] = {}
    new_verts = verts.tolist()
    new_uvs = uvs.tolist() if uvs is not None else None
    for vi in sel_verts:
        if custom_dir is not None:
            ndir = custom_dir
        else:
            ln = float(np.linalg.norm(v_norms[vi]))
            ndir = v_norms[vi] / ln if ln > 1e-12 else np.array([0.0, 0.0, 1.0])
        pos = verts[vi] + ndir * distance
        v_map[vi] = len(new_verts)
        new_verts.append(pos.tolist())
        if new_uvs is not None:
            new_uvs.append(uvs[vi].tolist())  # type: ignore[index]

    added_faces: list[list[int]] = []
    for u, v in edges:
        # Match directed boundary winding when edge exists in faces
        if (u, v) in d2f and (v, u) not in d2f:
            a, b = v, u
        else:
            a, b = u, v
        ae, be = v_map[a], v_map[b]
        added_faces.append([a, b, be])
        added_faces.append([a, be, ae])

    all_faces = np.vstack([faces, np.asarray(added_faces, dtype=np.int64)])
    return MeshData(
        vertices=np.asarray(new_verts, dtype=np.float64),
        faces=all_faces,
        uvs=None if new_uvs is None else np.asarray(new_uvs, dtype=np.float64),
    )


def extrude_vertices(
    mesh: MeshData,
    query: SelectionQuery,
    distance: float = 0.3,
    direction: Vec3 | None = None,
    width: float = 0.03,
) -> MeshData:
    """Extrude selected vertices into solid triangular-prism struts along direction/normal."""
    if mesh.is_empty:
        return mesh.copy()
    v_mask = resolve_vertex_selection(mesh, query)
    sel_verts = np.nonzero(v_mask)[0]
    if len(sel_verts) == 0:
        raise GeometryError(f"Vertex selection ({query.mode}) matched 0 vertices.")

    vnorms = mesh.vertex_normals()
    custom_dir: np.ndarray | None = None
    if direction is not None:
        d = np.asarray(direction, dtype=np.float64)
        ln = float(np.linalg.norm(d))
        if ln < 1e-12:
            raise GeometryError("Extrude vertex direction cannot be zero.")
        custom_dir = d / ln

    parts: list[MeshData] = [mesh]
    r = max(0.004, float(width))
    for vi in sel_verts:
        p0 = mesh.vertices[int(vi)]
        ndir = custom_dir if custom_dir is not None else vnorms[int(vi)]
        ln = float(np.linalg.norm(ndir))
        ndir = ndir / ln if ln > 1e-12 else np.array([0.0, 0.0, 1.0])
        seed = np.array([1.0, 0.0, 0.0]) if abs(ndir[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
        t1 = np.cross(ndir, seed)
        t1 /= max(1e-12, float(np.linalg.norm(t1)))
        t2 = np.cross(ndir, t1)
        p1 = p0 + ndir * distance
        # Build watertight 3-sided prism between base p0 and tip p1
        ring0 = [
            p0
            + (math.cos(k * 2.0 * math.pi / 3.0) * t1 + math.sin(k * 2.0 * math.pi / 3.0) * t2) * r
            for k in range(3)
        ]
        ring1 = [
            p1
            + (math.cos(k * 2.0 * math.pi / 3.0) * t1 + math.sin(k * 2.0 * math.pi / 3.0) * t2)
            * (r * 0.7)
            for k in range(3)
        ]
        sv = np.asarray(ring0 + ring1, dtype=np.float64)
        sf = np.asarray(
            [
                [0, 2, 1],
                [3, 4, 5],
                [0, 1, 4],
                [0, 4, 3],
                [1, 2, 5],
                [1, 5, 4],
                [2, 0, 3],
                [2, 3, 5],
            ],
            dtype=np.int64,
        )
        parts.append(MeshData(vertices=sv, faces=sf))
    return concat_meshes(parts)


def _inset_faces_impl(
    mesh: MeshData,
    query: SelectionQuery,
    thickness: float = 0.1,
    depth: float = 0.0,
    individual: bool = False,
    face_material_ids: dict[int, str] | None = None,
    base_material_id: str | None = None,
) -> tuple[MeshData, list[int], set[int]]:
    if mesh.is_empty:
        return mesh.copy(), list(range(len(mesh.faces))), set()
    sel_mask = resolve_face_selection(
        mesh, query, face_material_ids=face_material_ids, base_material_id=base_material_id
    )
    if not np.any(sel_mask):
        raise GeometryError(f"Selection query ({query.mode}) matched 0 faces.")

    verts = mesh.vertices.copy()
    faces = mesh.faces.copy()
    normals = mesh.face_normals()
    uvs = None if mesh.uvs is None or len(mesh.uvs) != len(verts) else mesh.uvs.copy()
    sel_idx = np.nonzero(sel_mask)[0]
    face_sources = list(range(len(faces)))

    if individual:
        new_verts_list = verts.tolist()
        new_uvs_list = uvs.tolist() if uvs is not None else None
        rim_faces: list[list[int]] = []
        rim_sources: list[int] = []
        sel_set = {int(i) for i in sel_idx}
        for fi in sel_idx:
            a, b, c = int(faces[fi, 0]), int(faces[fi, 1]), int(faces[fi, 2])
            f_cent = np.mean(verts[[a, b, c]], axis=0)
            radii = np.linalg.norm(verts[[a, b, c]] - f_cent[None, :], axis=1)
            avg_r = max(1e-6, float(np.mean(radii)))
            sc = max(0.05, min(0.98, 1.0 - thickness / avg_r))
            va = len(new_verts_list)
            vb = va + 1
            vc = va + 2
            for vi in (a, b, c):
                pos = f_cent + (verts[vi] - f_cent) * sc + normals[fi] * depth
                new_verts_list.append(pos.tolist())
                if new_uvs_list is not None:
                    uv_c = np.mean(uvs[[a, b, c]], axis=0)  # type: ignore[index]
                    new_uvs_list.append((uv_c + (uvs[vi] - uv_c) * sc).tolist())  # type: ignore[index]
            faces[fi] = [va, vb, vc]
            for u, v, ui, vi_in in ((a, b, va, vb), (b, c, vb, vc), (c, a, vc, va)):
                rim_faces.append([u, v, vi_in])
                rim_faces.append([u, vi_in, ui])
                rim_sources.extend([int(fi), int(fi)])
        all_faces = np.vstack([faces, np.asarray(rim_faces, dtype=np.int64)])
        face_sources.extend(rim_sources)
        return (
            MeshData(
                vertices=np.asarray(new_verts_list, dtype=np.float64),
                faces=all_faces,
                uvs=None if new_uvs_list is None else np.asarray(new_uvs_list, dtype=np.float64),
            ),
            face_sources,
            sel_set,
        )

    avg_normal = np.sum(normals[sel_idx], axis=0)
    n_len = float(np.linalg.norm(avg_normal))
    avg_normal = avg_normal / n_len if n_len > 1e-12 else np.array([0.0, 0.0, 1.0])

    unique_v = np.unique(faces[sel_idx].ravel())
    centroid = np.mean(verts[unique_v], axis=0)

    boundary_edges = _patch_boundary_edges(faces, sel_mask)
    if not boundary_edges:
        raise GeometryError(
            "Inset requires a partial face selection with boundary edges (or set individual=True)."
        )

    b_verts = np.unique([u for u, _ in boundary_edges])
    radii = np.linalg.norm(verts[b_verts] - centroid[None, :], axis=1)
    avg_r = max(1e-6, float(np.mean(radii)))
    inset_scale = max(0.05, min(0.98, 1.0 - thickness / avg_r))

    v_map: dict[int, int] = {}
    new_verts_list = verts.tolist()
    new_uvs_list = uvs.tolist() if uvs is not None else None
    uv_cent = np.mean(uvs[unique_v], axis=0) if uvs is not None else None
    for vi in unique_v:
        pos = centroid + (verts[vi] - centroid) * inset_scale + avg_normal * depth
        new_idx = len(new_verts_list)
        new_verts_list.append(pos.tolist())
        if new_uvs_list is not None and uv_cent is not None:
            new_uvs_list.append((uv_cent + (uvs[vi] - uv_cent) * inset_scale).tolist())  # type: ignore[index]
        v_map[int(vi)] = new_idx

    d2f = _build_directed_edge_map(faces)
    sel_set = {int(i) for i in sel_idx}
    for fi in sel_idx:
        a, b, c = int(faces[fi, 0]), int(faces[fi, 1]), int(faces[fi, 2])
        faces[fi] = [v_map[a], v_map[b], v_map[c]]

    rim_faces = []
    rim_sources = []
    for u, v in boundary_edges:
        u_in = v_map[u]
        v_in = v_map[v]
        cand = [fi for fi in d2f.get((u, v), []) if fi in sel_set]
        src_fi = cand[0] if cand else int(sel_idx[0])
        rim_faces.append([u, v, v_in])
        rim_faces.append([u, v_in, u_in])
        rim_sources.extend([src_fi, src_fi])

    all_faces = np.vstack([faces, np.asarray(rim_faces, dtype=np.int64)])
    face_sources.extend(rim_sources)
    return (
        MeshData(
            vertices=np.asarray(new_verts_list, dtype=np.float64),
            faces=all_faces,
            uvs=None if new_uvs_list is None else np.asarray(new_uvs_list, dtype=np.float64),
        ),
        face_sources,
        sel_set,
    )


def inset_faces(
    mesh: MeshData,
    query: SelectionQuery,
    thickness: float = 0.1,
    depth: float = 0.0,
    individual: bool = False,
) -> MeshData:
    """Inset semantically selected faces inward (patch or individual per-face)."""
    out, _, _ = _inset_faces_impl(
        mesh, query, thickness=thickness, depth=depth, individual=individual
    )
    return out


def _subdivide_once(
    cur: MeshData,
    smooth: bool,
    sel_mask: np.ndarray | None = None,
    face_sources: list[int] | None = None,
) -> tuple[MeshData, list[int]]:
    verts = cur.vertices
    faces = cur.faces
    uvs = None if cur.uvs is None or len(cur.uvs) != len(verts) else cur.uvs
    edge_mid: dict[tuple[int, int], int] = {}
    new_verts = verts.tolist()
    new_uvs = uvs.tolist() if uvs is not None else None

    neighbors: dict[int, set[int]] = defaultdict(set)
    edge_opp: dict[tuple[int, int], list[int]] = defaultdict(list)
    if smooth:
        for a, b, c in faces:
            ia, ib, ic = int(a), int(b), int(c)
            neighbors[ia].update((ib, ic))
            neighbors[ib].update((ia, ic))
            neighbors[ic].update((ia, ib))
            edge_opp[(min(ia, ib), max(ia, ib))].append(ic)
            edge_opp[(min(ib, ic), max(ib, ic))].append(ia)
            edge_opp[(min(ic, ia), max(ic, ia))].append(ib)

    # Which edges should be split? If sel_mask is None or all True -> all edges.
    split_edges: set[tuple[int, int]] | None = None
    if sel_mask is not None and not np.all(sel_mask):
        split_edges = set()
        for fi in np.nonzero(sel_mask)[0]:
            a, b, c = int(faces[fi, 0]), int(faces[fi, 1]), int(faces[fi, 2])
            split_edges.add((min(a, b), max(a, b)))
            split_edges.add((min(b, c), max(b, c)))
            split_edges.add((min(c, a), max(c, a)))

    def get_mid(u: int, v: int) -> int:
        key = (u, v) if u < v else (v, u)
        if key in edge_mid:
            return edge_mid[key]
        if smooth and len(edge_opp[key]) == 2 and split_edges is None:
            o1, o2 = edge_opp[key]
            pos = 0.375 * (verts[u] + verts[v]) + 0.125 * (verts[o1] + verts[o2])
        else:
            pos = 0.5 * (verts[u] + verts[v])
        idx = len(new_verts)
        new_verts.append(pos.tolist())
        if new_uvs is not None:
            new_uvs.append((0.5 * (uvs[u] + uvs[v])).tolist())  # type: ignore[index]
        edge_mid[key] = idx
        return idx

    new_faces: list[list[int]] = []
    out_sources: list[int] = []
    src_in = face_sources if face_sources is not None else list(range(len(faces)))

    for fi, (a, b, c) in enumerate(faces):
        ia, ib, ic = int(a), int(b), int(c)
        orig_src = src_in[fi]
        if split_edges is None or sel_mask[fi]:  # type: ignore[index]
            ab = get_mid(ia, ib)
            bc = get_mid(ib, ic)
            ca = get_mid(ic, ia)
            new_faces.extend([[ia, ab, ca], [ab, ib, bc], [ca, bc, ic], [ab, bc, ca]])
            out_sources.extend([orig_src] * 4)
        else:
            # Conformal 1-to-2 or 1-to-3 split on boundary-adjacent unselected faces
            e_ab = (min(ia, ib), max(ia, ib)) in split_edges
            e_bc = (min(ib, ic), max(ib, ic)) in split_edges
            e_ca = (min(ic, ia), max(ic, ia)) in split_edges
            cnt = int(e_ab) + int(e_bc) + int(e_ca)
            if cnt == 0:
                new_faces.append([ia, ib, ic])
                out_sources.append(orig_src)
            elif cnt == 1:
                if e_ab:
                    m = get_mid(ia, ib)
                    new_faces.extend([[ia, m, ic], [m, ib, ic]])
                elif e_bc:
                    m = get_mid(ib, ic)
                    new_faces.extend([[ib, m, ia], [m, ic, ia]])
                else:
                    m = get_mid(ic, ia)
                    new_faces.extend([[ic, m, ib], [m, ia, ib]])
                out_sources.extend([orig_src, orig_src])
            elif cnt == 2:
                # Rotate so the two split edges touch vertex `p`
                if e_ab and e_bc:
                    p, q, r = ib, ic, ia
                elif e_bc and e_ca:
                    p, q, r = ic, ia, ib
                else:
                    p, q, r = ia, ib, ic
                mpq = get_mid(p, q)
                mrp = get_mid(r, p)
                new_faces.extend([[p, mpq, mrp], [mpq, q, r], [mpq, r, mrp]])
                out_sources.extend([orig_src] * 3)
            else:
                ab = get_mid(ia, ib)
                bc = get_mid(ib, ic)
                ca = get_mid(ic, ia)
                new_faces.extend([[ia, ab, ca], [ab, ib, bc], [ca, bc, ic], [ab, bc, ca]])
                out_sources.extend([orig_src] * 4)

    out_verts = np.asarray(new_verts, dtype=np.float64)
    if smooth and split_edges is None:
        orig_updated = verts.copy()
        for vi, nbrs in neighbors.items():
            n = len(nbrs)
            if n >= 3:
                beta = 3.0 / 16.0 if n == 3 else 3.0 / (8.0 * n)
                nbr_sum = np.sum(verts[list(nbrs)], axis=0)
                orig_updated[vi] = (1.0 - n * beta) * verts[vi] + beta * nbr_sum
        out_verts[: len(verts)] = orig_updated

    return (
        MeshData(
            vertices=out_verts,
            faces=np.asarray(new_faces, dtype=np.int64),
            uvs=None if new_uvs is None else np.asarray(new_uvs, dtype=np.float64),
        ),
        out_sources,
    )


def _subdivide_mesh_impl(
    mesh: MeshData,
    levels: int = 1,
    smooth: bool = False,
    query: SelectionQuery | None = None,
    face_material_ids: dict[int, str] | None = None,
    base_material_id: str | None = None,
    algorithm: Literal["catmull_clark", "loop", "simple"] = "catmull_clark",
    crease_angle_deg: float = 180.0,
) -> tuple[MeshData, list[int]]:
    if mesh.is_empty:
        return mesh.copy(), list(range(len(mesh.faces)))

    cur = mesh.copy()
    sources = list(range(len(mesh.faces)))
    partial = query is not None and query.mode != "all"

    if partial and query is not None:
        sel_mask = resolve_face_selection(
            cur, query, face_material_ids=face_material_ids, base_material_id=base_material_id
        )
        if not np.any(sel_mask):
            raise GeometryError(f"Selection query ({query.mode}) matched 0 faces.")
        for step in range(max(1, min(3, levels))):
            if step == 0:
                cur, sources = _subdivide_once(
                    cur, smooth=False, sel_mask=sel_mask, face_sources=sources
                )
                # Track expanded selected faces across step if levels > 1
                sel_set = set(int(i) for i in np.nonzero(sel_mask)[0])
                sel_mask = np.asarray([s in sel_set for s in sources], dtype=bool)
            else:
                cur, sources = _subdivide_once(
                    cur, smooth=False, sel_mask=sel_mask, face_sources=sources
                )
        return cur, sources

    if algorithm == "catmull_clark" and smooth:
        from frfruit.geometry.bmesh import catmull_clark_subdivide

        cc_mesh, cc_sources, _ = catmull_clark_subdivide(
            cur,
            levels=levels,
            smooth=True,
            crease_angle_deg=crease_angle_deg,
            face_sources=sources,
        )
        return cc_mesh, cc_sources

    use_smooth = smooth and (algorithm != "simple")
    for _ in range(max(1, min(3, levels))):
        cur, sources = _subdivide_once(cur, smooth=use_smooth, sel_mask=None, face_sources=sources)
    return cur, sources


def subdivide_mesh(
    mesh: MeshData,
    levels: int = 1,
    smooth: bool = False,
    query: SelectionQuery | None = None,
    algorithm: Literal["catmull_clark", "loop", "simple"] = "catmull_clark",
    crease_angle_deg: float = 180.0,
) -> MeshData:
    """Subdivide mesh (Catmull-Clark quad or Loop triangle; conformal on partial selection)."""
    out, _ = _subdivide_mesh_impl(
        mesh,
        levels=levels,
        smooth=smooth,
        query=query,
        algorithm=algorithm,
        crease_angle_deg=crease_angle_deg,
    )
    return out


def bevel_edges(
    mesh: MeshData,
    angle_threshold_deg: float = 35.0,
    width: float = 0.04,
    segments: int = 1,
    query: SelectionQuery | None = None,
) -> MeshData:
    """Bevel sharp edges (or selected edges/patch boundary) while keeping topology watertight."""
    if mesh.is_empty:
        return mesh.copy()

    tm = mesh.to_trimesh(process=False)
    tm.merge_vertices(digits_vertex=6)
    verts = np.asarray(tm.vertices, dtype=np.float64)
    faces = np.asarray(tm.faces, dtype=np.int64)
    cur = MeshData(vertices=verts, faces=faces)
    fnormals = cur.face_normals()
    m_count = len(faces)
    if m_count == 0:
        return cur

    # Determine optional subset of edges to bevel when query != all
    target_edges: set[tuple[int, int]] | None = None
    if query is not None and query.mode != "all":
        e_list = resolve_edge_selection(cur, query)
        if not e_list:
            raise GeometryError(f"Bevel selection ({query.mode}) matched 0 edges.")
        target_edges = set(e_list)

    directed_to_face: dict[tuple[int, int], int] = {}
    for fi, (a, b, c) in enumerate(faces):
        directed_to_face[(int(a), int(b))] = fi
        directed_to_face[(int(b), int(c))] = fi
        directed_to_face[(int(c), int(a))] = fi

    cos_thresh = math.cos(math.radians(angle_threshold_deg))
    adj_smooth: list[list[int]] = [[] for _ in range(m_count)]
    for (u, v), f1 in directed_to_face.items():
        f2 = directed_to_face.get((v, u))
        if f2 is not None and f1 < f2:
            ek = (min(u, v), max(u, v))
            if target_edges is not None:
                is_sharp = ek in target_edges
            else:
                is_sharp = float(np.dot(fnormals[f1], fnormals[f2])) < cos_thresh
            if not is_sharp:
                adj_smooth[f1].append(f2)
                adj_smooth[f2].append(f1)

    patch_of_face = [-1] * m_count
    patches: list[list[int]] = []
    for fi in range(m_count):
        if patch_of_face[fi] != -1:
            continue
        pid = len(patches)
        stack = [fi]
        patch_of_face[fi] = pid
        comp: list[int] = []
        while stack:
            cur_f = stack.pop()
            comp.append(cur_f)
            for nxt in adj_smooth[cur_f]:
                if patch_of_face[nxt] == -1:
                    patch_of_face[nxt] = pid
                    stack.append(nxt)
        patches.append(comp)

    if len(patches) <= 1:
        return cur

    patch_normals: list[np.ndarray] = []
    for comp in patches:
        n_sum = np.sum(fnormals[comp], axis=0)
        ln = float(np.linalg.norm(n_sum))
        patch_normals.append(n_sum / ln if ln > 1e-12 else np.array([0.0, 0.0, 1.0]))

    min_edge = np.full((len(verts),), np.inf, dtype=np.float64)
    for a, b, c in faces:
        for u, v in ((a, b), (b, c), (c, a)):
            d = float(np.linalg.norm(verts[v] - verts[u]))
            if d > 1e-9:
                min_edge[u] = min(min_edge[u], d)
                min_edge[v] = min(min_edge[v], d)
    min_edge = np.where(np.isfinite(min_edge), min_edge, width * 3.0)

    patch_boundary_edges: list[list[tuple[int, int]]] = [[] for _ in patches]
    sharp_edge_pairs: list[tuple[int, int, int, int]] = []
    for (u, v), f1 in directed_to_face.items():
        p1 = patch_of_face[f1]
        f2 = directed_to_face.get((v, u))
        if f2 is None:
            patch_boundary_edges[p1].append((u, v))
        else:
            p2 = patch_of_face[f2]
            if p1 != p2:
                patch_boundary_edges[p1].append((u, v))
                if p1 < p2:
                    sharp_edge_pairs.append((u, v, p1, p2))

    pv_map: dict[tuple[int, int], int] = {}
    new_verts: list[list[float]] = []

    for pid, comp in enumerate(patches):
        pn = patch_normals[pid]
        inward_acc: dict[int, list[np.ndarray]] = defaultdict(list)
        for u, v in patch_boundary_edges[pid]:
            edge_vec = verts[v] - verts[u]
            e_len = float(np.linalg.norm(edge_vec))
            if e_len < 1e-9:
                continue
            edge_dir = edge_vec / e_len
            inward = np.cross(pn, edge_dir)
            in_len = float(np.linalg.norm(inward))
            if in_len > 1e-9:
                inward /= in_len
                inward_acc[u].append(inward)
                inward_acc[v].append(inward)

        patch_verts = set()
        for fi in comp:
            a, b, c = faces[fi]
            patch_verts.update((int(a), int(b), int(c)))

        for vi in patch_verts:
            dirs = inward_acc.get(vi)
            if not dirs:
                pos = verts[vi]
            else:
                avg_in = np.sum(dirs, axis=0)
                avg_len = float(np.linalg.norm(avg_in))
                if avg_len > 1e-9:
                    avg_in /= avg_len
                    local_w = min(float(width), 0.38 * float(min_edge[vi]))
                    cos_half = max(0.35, float(np.dot(avg_in, dirs[0])))
                    pos = verts[vi] + avg_in * (local_w / cos_half)
                else:
                    pos = verts[vi]
            pv_map[(pid, vi)] = len(new_verts)
            new_verts.append(pos.tolist())

    new_faces: list[list[int]] = []
    for fi, (a, b, c) in enumerate(faces):
        pid = patch_of_face[fi]
        new_faces.append([pv_map[(pid, int(a))], pv_map[(pid, int(b))], pv_map[(pid, int(c))]])

    for u, v, p1, p2 in sharp_edge_pairs:
        u1 = pv_map[(p1, u)]
        v1 = pv_map[(p1, v)]
        u2 = pv_map[(p2, u)]
        v2 = pv_map[(p2, v)]
        new_faces.append([v1, u1, u2])
        new_faces.append([v1, u2, v2])

    vert_patches: dict[int, set[int]] = defaultdict(set)
    for pid, vi in pv_map:
        vert_patches[vi].add(pid)

    for vi, pids in vert_patches.items():
        if len(pids) < 3:
            continue
        avg_n = np.sum([patch_normals[p] for p in pids], axis=0)
        ln = float(np.linalg.norm(avg_n))
        avg_n = avg_n / ln if ln > 1e-12 else np.array([0.0, 0.0, 1.0])
        seed = np.array([1.0, 0.0, 0.0]) if abs(avg_n[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
        u_ax = np.cross(avg_n, seed)
        u_ax /= max(1e-12, float(np.linalg.norm(u_ax)))
        v_ax = np.cross(avg_n, u_ax)

        nv_arr = np.asarray(new_verts, dtype=np.float64)
        pts = np.asarray([nv_arr[pv_map[(p, vi)]] for p in pids])
        center = np.mean(pts, axis=0)
        ordered = sorted(
            pids,
            key=lambda p: math.atan2(
                float(np.dot(nv_arr[pv_map[(p, vi)]] - center, v_ax)),
                float(np.dot(nv_arr[pv_map[(p, vi)]] - center, u_ax)),
            ),
        )
        c_idx = pv_map[(ordered[0], vi)]
        for k in range(1, len(ordered) - 1):
            b_idx = pv_map[(ordered[k], vi)]
            d_idx = pv_map[(ordered[k + 1], vi)]
            new_faces.append([c_idx, b_idx, d_idx])

    out = MeshData(
        vertices=np.asarray(new_verts, dtype=np.float64),
        faces=np.asarray(new_faces, dtype=np.int64),
    )
    out = repair_manifold(out)

    if segments > 1 and not out.is_empty:
        try:
            mf = out.to_manifold()
            mf = mf.smooth_out(min_sharp_angle=15.0, min_smoothness=0.35).refine(min(3, segments))
            out = MeshData.from_manifold(mf)
        except GeometryError:
            pass
    return out


def bevel_vertices(
    mesh: MeshData,
    query: SelectionQuery,
    width: float = 0.08,
) -> MeshData:
    """Chamfer/bevel selected vertices along all incident edges, keeping mesh watertight."""
    if mesh.is_empty:
        return mesh.copy()
    tm = mesh.to_trimesh(process=False)
    tm.merge_vertices(digits_vertex=6)
    cur = MeshData(
        vertices=np.asarray(tm.vertices, dtype=np.float64),
        faces=np.asarray(tm.faces, dtype=np.int64),
    )
    v_mask = resolve_vertex_selection(cur, query)
    sel_verts = set(np.nonzero(v_mask)[0].tolist())
    if not sel_verts:
        raise GeometryError(f"Vertex bevel selection ({query.mode}) matched 0 vertices.")

    verts = cur.vertices
    faces = cur.faces
    vnorms = cur.vertex_normals()

    incident_faces: dict[int, list[int]] = defaultdict(list)
    nbr_verts: dict[int, set[int]] = defaultdict(set)
    for fi, (a, b, c) in enumerate(faces):
        ia, ib, ic = int(a), int(b), int(c)
        incident_faces[ia].append(fi)
        incident_faces[ib].append(fi)
        incident_faces[ic].append(fi)
        nbr_verts[ia].update((ib, ic))
        nbr_verts[ib].update((ia, ic))
        nbr_verts[ic].update((ia, ib))

    new_verts = verts.tolist()
    edge_split: dict[tuple[int, int], int] = {}

    def get_split(vi: int, nbr: int) -> int:
        key = (vi, nbr)
        if key in edge_split:
            return edge_split[key]
        vec = verts[nbr] - verts[vi]
        elen = float(np.linalg.norm(vec))
        step = min(float(width), 0.42 * elen)
        pos = verts[vi] + (vec / max(1e-12, elen)) * step
        idx = len(new_verts)
        new_verts.append(pos.tolist())
        edge_split[key] = idx
        return idx

    # For faces touching selected vertices, replace each corner cleanly
    new_faces: list[list[int]] = []
    for _fi, (a, b, c) in enumerate(faces):
        tri = [int(a), int(b), int(c)]
        sel_pos = [k for k, v in enumerate(tri) if v in sel_verts]
        if not sel_pos:
            new_faces.append(tri)
        elif len(sel_pos) == 1:
            k = sel_pos[0]
            v = tri[k]
            n1 = tri[(k + 1) % 3]
            n2 = tri[(k + 2) % 3]
            s1 = get_split(v, n1)
            s2 = get_split(v, n2)
            new_faces.append([s1, n1, n2])
            new_faces.append([s1, n2, s2])
        elif len(sel_pos) == 2:
            un = [k for k in range(3) if k not in sel_pos][0]
            u = tri[un]
            v1 = tri[(un + 1) % 3]
            v2 = tri[(un + 2) % 3]
            sv1_u = get_split(v1, u)
            sv1_2 = get_split(v1, v2)
            sv2_1 = get_split(v2, v1)
            sv2_u = get_split(v2, u)
            new_faces.append([u, sv1_u, sv2_u])
            new_faces.append([sv1_u, sv1_2, sv2_1])
            new_faces.append([sv1_u, sv2_1, sv2_u])
        else:
            va, vb, vc = tri
            sab = get_split(va, vb)
            sac = get_split(va, vc)
            sba = get_split(vb, va)
            sbc = get_split(vb, vc)
            sca = get_split(vc, va)
            scb = get_split(vc, vb)
            new_faces.append([sab, sba, sbc])
            new_faces.append([sab, sbc, scb])
            new_faces.append([sab, scb, sca])
            new_faces.append([sab, sca, sac])

    # Cap each beveled corner vertex with an outward fan polygon
    nv_arr = np.asarray(new_verts, dtype=np.float64)
    for vi in sel_verts:
        ring_ids = [edge_split[(vi, nbr)] for nbr in nbr_verts[vi] if (vi, nbr) in edge_split]
        if len(ring_ids) < 3:
            continue
        vn = vnorms[vi]
        seed = np.array([1.0, 0.0, 0.0]) if abs(vn[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
        u_ax = np.cross(vn, seed)
        u_ax /= max(1e-12, float(np.linalg.norm(u_ax)))
        v_ax = np.cross(vn, u_ax)
        cent = np.mean(nv_arr[ring_ids], axis=0)
        ordered = sorted(
            ring_ids,
            key=lambda idx: math.atan2(
                float(np.dot(nv_arr[idx] - cent, v_ax)),
                float(np.dot(nv_arr[idx] - cent, u_ax)),
            ),
        )
        c0 = ordered[0]
        for k in range(1, len(ordered) - 1):
            new_faces.append([c0, ordered[k], ordered[k + 1]])

    out = MeshData(
        vertices=nv_arr,
        faces=np.asarray(new_faces, dtype=np.int64),
    )
    return repair_manifold(out)


def _slice_mesh_by_plane_impl(
    mesh: MeshData,
    plane_point: Vec3,
    plane_normal: Vec3,
    face_sources: list[int] | None = None,
) -> tuple[MeshData, list[int], np.ndarray, list[tuple[int, int]]]:
    """Split all triangles crossing plane cleanly, sharing exact edge split vertices.

    Returns `(sliced_mesh, face_sources, side_tags (+1/-1 per face), cut_segments)`.
    """
    verts = mesh.vertices
    faces = mesh.faces
    uvs = None if mesh.uvs is None or len(mesh.uvs) != len(verts) else mesh.uvs
    p0 = np.asarray(plane_point, dtype=np.float64)
    pn = np.asarray(plane_normal, dtype=np.float64)
    pn /= max(1e-12, float(np.linalg.norm(pn)))

    dists = (verts - p0[None, :]) @ pn
    eps = 1e-7
    dists = np.where(np.abs(dists) < eps, 0.0, dists)

    new_verts = verts.tolist()
    new_uvs = uvs.tolist() if uvs is not None else None
    edge_split: dict[tuple[int, int], int] = {}

    def split_edge(u: int, v: int) -> int:
        key = (min(u, v), max(u, v))
        if key in edge_split:
            return edge_split[key]
        du, dv = float(dists[u]), float(dists[v])
        denom = du - dv
        t = 0.5 if abs(denom) < 1e-12 else du / denom
        t = max(0.0, min(1.0, t))
        pos = verts[u] * (1.0 - t) + verts[v] * t
        idx = len(new_verts)
        new_verts.append(pos.tolist())
        if new_uvs is not None and uvs is not None:
            uv_p = _interp_uvs(uvs, u, v, t)
            if uv_p is not None:
                new_uvs.append(uv_p)
        edge_split[key] = idx
        return idx

    out_faces: list[list[int]] = []
    out_src: list[int] = []
    out_sides: list[int] = []
    cut_segments: list[tuple[int, int]] = []
    src_in = face_sources if face_sources is not None else list(range(len(faces)))

    for fi, (a, b, c) in enumerate(faces):
        tri = [int(a), int(b), int(c)]
        da, db, dc = float(dists[tri[0]]), float(dists[tri[1]]), float(dists[tri[2]])
        src = src_in[fi]
        if da >= 0.0 and db >= 0.0 and dc >= 0.0:
            out_faces.append(tri)
            out_src.append(src)
            out_sides.append(1)
            continue
        if da <= 0.0 and db <= 0.0 and dc <= 0.0:
            out_faces.append(tri)
            out_src.append(src)
            out_sides.append(-1)
            continue

        # Classify strictly positive vs negative, allowing zero on either
        ds = [da, db, dc]
        pos_i = [k for k in range(3) if ds[k] > 0.0]
        neg_i = [k for k in range(3) if ds[k] < 0.0]
        if len(pos_i) == 1 and len(neg_i) == 1:
            # One vertex on plane, one +, one -
            k_z = [k for k in range(3) if ds[k] == 0.0][0]
            k1 = (k_z + 1) % 3
            k2 = (k_z + 2) % 3
            vz, v1, v2 = tri[k_z], tri[k1], tri[k2]
            m = split_edge(v1, v2)
            out_faces.append([vz, v1, m])
            out_src.append(src)
            out_sides.append(1 if ds[k1] > 0.0 else -1)
            out_faces.append([vz, m, v2])
            out_src.append(src)
            out_sides.append(1 if ds[k2] > 0.0 else -1)
            cut_segments.append((vz, m))
            continue

        # 1 vertex on one side (lone), 2 vertices on the other side
        lone_is_pos = len(pos_i) == 1
        k0 = pos_i[0] if lone_is_pos else neg_i[0]
        k1 = (k0 + 1) % 3
        k2 = (k0 + 2) % 3
        v0, v1, v2 = tri[k0], tri[k1], tri[k2]
        m01 = split_edge(v0, v1)
        m02 = split_edge(v0, v2)
        side_lone = 1 if lone_is_pos else -1
        side_pair = -side_lone

        out_faces.append([v0, m01, m02])
        out_src.append(src)
        out_sides.append(side_lone)

        out_faces.append([m01, v1, v2])
        out_src.append(src)
        out_sides.append(side_pair)

        out_faces.append([m01, v2, m02])
        out_src.append(src)
        out_sides.append(side_pair)

        cut_segments.append((m01, m02))

    out_mesh = MeshData(
        vertices=np.asarray(new_verts, dtype=np.float64),
        faces=np.asarray(out_faces, dtype=np.int64),
        uvs=None if new_uvs is None else np.asarray(new_uvs, dtype=np.float64),
    )
    return out_mesh, out_src, np.asarray(out_sides, dtype=np.int32), cut_segments


def bisect_plane(
    mesh: MeshData,
    plane_point: Vec3 = (0.0, 0.0, 0.0),
    plane_normal: Vec3 = (0.0, 0.0, 1.0),
    clear_inner: bool = False,
    clear_outer: bool = False,
    fill_caps: bool = True,
) -> MeshData:
    """Slice mesh by a 3D plane (Knife/Bisect), optionally removing inner/outer half and capping."""
    out, _, _ = _bisect_plane_impl(
        mesh,
        plane_point=plane_point,
        plane_normal=plane_normal,
        clear_inner=clear_inner,
        clear_outer=clear_outer,
        fill_caps=fill_caps,
    )
    return out


def _bisect_plane_impl(
    mesh: MeshData,
    plane_point: Vec3 = (0.0, 0.0, 0.0),
    plane_normal: Vec3 = (0.0, 0.0, 1.0),
    clear_inner: bool = False,
    clear_outer: bool = False,
    fill_caps: bool = True,
    face_sources: list[int] | None = None,
) -> tuple[MeshData, list[int], set[int]]:
    if mesh.is_empty:
        return mesh.copy(), list(range(len(mesh.faces))), set()
    sliced, src, sides, _ = _slice_mesh_by_plane_impl(
        mesh, plane_point=plane_point, plane_normal=plane_normal, face_sources=face_sources
    )
    if not clear_inner and not clear_outer:
        return sliced, src, set()

    keep_mask = np.ones((len(sliced.faces),), dtype=bool)
    if clear_inner:
        keep_mask &= sides >= 0
    if clear_outer:
        keep_mask &= sides <= 0
    if not np.any(keep_mask):
        raise GeometryError("Bisect cleared all faces of the mesh.")

    kept_faces = sliced.faces[keep_mask]
    kept_src = [src[i] for i in np.nonzero(keep_mask)[0]]
    cap_indices: set[int] = set()

    if fill_caps:
        open_edges = _extract_open_boundary_edges(kept_faces)
        loops = _chain_boundary_loops(open_edges)
        verts_list = sliced.vertices.tolist()
        uvs_list = sliced.uvs.tolist() if sliced.uvs is not None else None
        cap_faces: list[list[int]] = []
        pn = np.asarray(plane_normal, dtype=np.float64)
        pn /= max(1e-12, float(np.linalg.norm(pn)))
        # Outward cap normal should point toward cleared side
        cap_target_n = pn if clear_outer else -pn
        for lp in loops:
            pts = np.asarray([verts_list[vi] for vi in lp], dtype=np.float64)
            cent = np.mean(pts, axis=0)
            c_idx = len(verts_list)
            verts_list.append(cent.tolist())
            if uvs_list is not None and sliced.uvs is not None:
                uv_c = np.mean(sliced.uvs[lp], axis=0)
                uvs_list.append(uv_c.tolist())
            # Determine loop orientation relative to cap_target_n
            area_vec = np.zeros(3, dtype=np.float64)
            for k in range(len(lp)):
                u_i, v_i = lp[k], lp[(k + 1) % len(lp)]
                area_vec += np.cross(
                    np.asarray(verts_list[u_i]) - cent,
                    np.asarray(verts_list[v_i]) - cent,
                )
            flip = float(np.dot(area_vec, cap_target_n)) < 0.0
            for k in range(len(lp)):
                u_i, v_i = lp[k], lp[(k + 1) % len(lp)]
                tri = [u_i, v_i, c_idx] if not flip else [v_i, u_i, c_idx]
                cap_faces.append(tri)
        if cap_faces:
            c_start = len(kept_faces)
            kept_faces = np.vstack([kept_faces, np.asarray(cap_faces, dtype=np.int64)])
            kept_src.extend([-1] * len(cap_faces))
            cap_indices.update(range(c_start, len(kept_faces)))
        sliced = MeshData(
            vertices=np.asarray(verts_list, dtype=np.float64),
            faces=kept_faces,
            uvs=None if uvs_list is None else np.asarray(uvs_list, dtype=np.float64),
        )
    else:
        sliced = MeshData(vertices=sliced.vertices, faces=kept_faces, uvs=sliced.uvs)

    # Prune unreferenced vertices cleanly
    cleaned, src_clean = _prune_unreferenced_vertices_impl(sliced, kept_src)
    return cleaned, src_clean, cap_indices


def loop_cut(
    mesh: MeshData,
    axis: Literal["x", "y", "z"] = "z",
    cuts: int = 1,
    slide: float = 0.0,
    plane_normal: Vec3 | None = None,
    plane_point: Vec3 | None = None,
) -> MeshData:
    """Insert `cuts` edge loops across mesh along axis (or custom normal) with optional slide."""
    out, _ = _loop_cut_impl(
        mesh,
        axis=axis,
        cuts=cuts,
        slide=slide,
        plane_normal=plane_normal,
        plane_point=plane_point,
    )
    return out


def _loop_cut_impl(
    mesh: MeshData,
    axis: Literal["x", "y", "z"] = "z",
    cuts: int = 1,
    slide: float = 0.0,
    plane_normal: Vec3 | None = None,
    plane_point: Vec3 | None = None,
    face_sources: list[int] | None = None,
) -> tuple[MeshData, list[int]]:
    if mesh.is_empty:
        return mesh.copy(), list(range(len(mesh.faces)))

    ax_vecs = {
        "x": (1.0, 0.0, 0.0),
        "y": (0.0, 1.0, 0.0),
        "z": (0.0, 0.0, 1.0),
    }
    nrm = np.asarray(plane_normal if plane_normal is not None else ax_vecs[axis], dtype=np.float64)
    n_len = float(np.linalg.norm(nrm))
    if n_len < 1e-12:
        raise GeometryError("Loop cut normal cannot be zero.")
    nrm /= n_len

    proj = mesh.vertices @ nrm
    pmin = float(np.min(proj))
    pmax = float(np.max(proj))
    span = pmax - pmin
    if span < 1e-7:
        return mesh.copy(), list(range(len(mesh.faces)))

    cur = mesh.copy()
    src = face_sources if face_sources is not None else list(range(len(mesh.faces)))
    n_cuts = max(1, min(12, int(cuts)))
    s_clamped = max(-0.92, min(0.92, float(slide)))

    if plane_point is not None and n_cuts == 1:
        cur, src, _, _ = _slice_mesh_by_plane_impl(
            cur,
            plane_point=plane_point,
            plane_normal=(float(nrm[0]), float(nrm[1]), float(nrm[2])),
            face_sources=src,
        )
        return cur, src

    for k in range(1, n_cuts + 1):
        frac = k / (n_cuts + 1)
        if n_cuts == 1:
            frac = 0.5 + 0.45 * s_clamped
        else:
            step = 1.0 / (n_cuts + 1)
            frac = max(0.03, min(0.97, frac + s_clamped * 0.45 * step))
        coord = pmin + frac * span
        pt = nrm * coord
        cur, src, _, _ = _slice_mesh_by_plane_impl(
            cur,
            plane_point=(float(pt[0]), float(pt[1]), float(pt[2])),
            plane_normal=(float(nrm[0]), float(nrm[1]), float(nrm[2])),
            face_sources=src,
        )
    return cur, src


def poke_faces(
    mesh: MeshData,
    query: SelectionQuery,
    offset: float = 0.0,
) -> MeshData:
    """Insert a center vertex on each selected face (displaced by offset along normal)."""
    out, _ = _poke_faces_impl(mesh, query, offset=offset)
    return out


def _poke_faces_impl(
    mesh: MeshData,
    query: SelectionQuery,
    offset: float = 0.0,
    face_material_ids: dict[int, str] | None = None,
    base_material_id: str | None = None,
) -> tuple[MeshData, list[int]]:
    if mesh.is_empty:
        return mesh.copy(), list(range(len(mesh.faces)))
    sel_mask = resolve_face_selection(
        mesh, query, face_material_ids=face_material_ids, base_material_id=base_material_id
    )
    if not np.any(sel_mask):
        raise GeometryError(f"Selection query ({query.mode}) matched 0 faces.")

    verts = mesh.vertices.tolist()
    uvs = None if mesh.uvs is None or len(mesh.uvs) != len(verts) else mesh.uvs.tolist()
    fnormals = mesh.face_normals()
    out_faces: list[list[int]] = []
    out_src: list[int] = []

    for fi, (a, b, c) in enumerate(mesh.faces):
        ia, ib, ic = int(a), int(b), int(c)
        if not sel_mask[fi]:
            out_faces.append([ia, ib, ic])
            out_src.append(fi)
            continue
        cent = (mesh.vertices[ia] + mesh.vertices[ib] + mesh.vertices[ic]) / 3.0
        pos = cent + fnormals[fi] * float(offset)
        c_idx = len(verts)
        verts.append(pos.tolist())
        if uvs is not None and mesh.uvs is not None:
            uv_c = (mesh.uvs[ia] + mesh.uvs[ib] + mesh.uvs[ic]) / 3.0
            uvs.append(uv_c.tolist())
        out_faces.extend([[ia, ib, c_idx], [ib, ic, c_idx], [ic, ia, c_idx]])
        out_src.extend([fi, fi, fi])

    return (
        MeshData(
            vertices=np.asarray(verts, dtype=np.float64),
            faces=np.asarray(out_faces, dtype=np.int64),
            uvs=None if uvs is None else np.asarray(uvs, dtype=np.float64),
        ),
        out_src,
    )


def fill_holes(mesh: MeshData, query: SelectionQuery | None = None) -> MeshData:
    """Fill open boundary loops with clean oriented triangle fans."""
    out, _, _ = _fill_holes_impl(mesh, query=query)
    return out


def _fill_holes_impl(
    mesh: MeshData,
    query: SelectionQuery | None = None,
) -> tuple[MeshData, list[int], set[int]]:
    if mesh.is_empty:
        return mesh.copy(), list(range(len(mesh.faces))), set()
    open_edges = _extract_open_boundary_edges(mesh.faces)
    if not open_edges:
        return mesh.copy(), list(range(len(mesh.faces))), set()

    loops = _chain_boundary_loops(open_edges)
    if query is not None and query.mode != "all":
        v_sel = resolve_vertex_selection(mesh, query)
        sel_set = set(np.nonzero(v_sel)[0].tolist())
        loops = [lp for lp in loops if any(v in sel_set for v in lp)]
    if not loops:
        return mesh.copy(), list(range(len(mesh.faces))), set()

    verts = mesh.vertices.tolist()
    uvs = None if mesh.uvs is None or len(mesh.uvs) != len(verts) else mesh.uvs.tolist()
    added: list[list[int]] = []
    # Map directed edge (u -> v) to adjacent face normal for outward cap orientation
    d2f = _build_directed_edge_map(mesh.faces)
    fnormals = mesh.face_normals()

    for lp in loops:
        pts = mesh.vertices[lp]
        cent = np.mean(pts, axis=0)
        c_idx = len(verts)
        verts.append(cent.tolist())
        if uvs is not None and mesh.uvs is not None:
            uvs.append(np.mean(mesh.uvs[lp], axis=0).tolist())

        # Reference normal: average of adjacent faces around the loop
        ref_n = np.zeros(3, dtype=np.float64)
        for k in range(len(lp)):
            u_i, v_i = lp[k], lp[(k + 1) % len(lp)]
            for fi in d2f.get((u_i, v_i), []):
                ref_n += fnormals[fi]
        rn_len = float(np.linalg.norm(ref_n))
        ref_n = ref_n / rn_len if rn_len > 1e-12 else np.array([0.0, 0.0, 1.0])

        # Open edge (u -> v) in existing mesh must be closed by reverse edge (v -> u)
        area_vec = np.zeros(3, dtype=np.float64)
        for k in range(len(lp)):
            u_i, v_i = lp[k], lp[(k + 1) % len(lp)]
            area_vec += np.cross(mesh.vertices[v_i] - cent, mesh.vertices[u_i] - cent)
        keep_rev = float(np.dot(area_vec, ref_n)) >= 0.0
        for k in range(len(lp)):
            u_i, v_i = lp[k], lp[(k + 1) % len(lp)]
            added.append([v_i, u_i, c_idx] if keep_rev else [u_i, v_i, c_idx])

    start_idx = len(mesh.faces)
    all_faces = np.vstack([mesh.faces, np.asarray(added, dtype=np.int64)])
    sources = list(range(start_idx)) + [-1] * len(added)
    new_caps = set(range(start_idx, len(all_faces)))
    return (
        MeshData(
            vertices=np.asarray(verts, dtype=np.float64),
            faces=all_faces,
            uvs=None if uvs is None else np.asarray(uvs, dtype=np.float64),
        ),
        sources,
        new_caps,
    )


def bridge_edge_loops(
    mesh: MeshData,
    query: SelectionQuery,
    target_query: SelectionQuery | None = None,
    segments: int = 1,
    twist: int = 0,
) -> MeshData:
    """Bridge two boundary loops (or two selected face patches after opening caps)."""
    out, _, _ = _bridge_edge_loops_impl(
        mesh, query=query, target_query=target_query, segments=segments, twist=twist
    )
    return out


def _bridge_edge_loops_impl(
    mesh: MeshData,
    query: SelectionQuery,
    target_query: SelectionQuery | None = None,
    segments: int = 1,
    twist: int = 0,
) -> tuple[MeshData, list[int], set[int]]:
    if mesh.is_empty:
        return mesh.copy(), list(range(len(mesh.faces))), set()

    cur = mesh.copy()
    kept_src = list(range(len(cur.faces)))

    # If two face patches or partial faces exist, remove selected caps to expose bridge loops
    if target_query is not None:
        m1 = resolve_face_selection(cur, query)
        m2 = resolve_face_selection(cur, target_query)
        rem_mask = ~(m1 | m2)
        if np.any(m1 | m2) and np.any(rem_mask):
            cur = MeshData(vertices=cur.vertices, faces=cur.faces[rem_mask], uvs=cur.uvs)
            kept_src = [kept_src[i] for i in np.nonzero(rem_mask)[0]]
    elif query.mode != "all" and query.mode != "boundary":
        m1 = resolve_face_selection(cur, query)
        if np.any(m1) and not np.all(m1):
            cur = MeshData(vertices=cur.vertices, faces=cur.faces[~m1], uvs=cur.uvs)
            kept_src = [kept_src[i] for i in np.nonzero(~m1)[0]]

    open_edges = _extract_open_boundary_edges(cur.faces)
    loops = _chain_boundary_loops(open_edges)
    if len(loops) < 2:
        raise GeometryError(f"Bridge requires at least 2 open boundary loops (found {len(loops)}).")

    # Pick the two largest/closest loops (or matching query if applicable)
    loops.sort(key=len, reverse=True)
    l1, l2 = loops[0], loops[1]

    # Resample or equalize counts simply by repeating/matching angular order if needed
    pts1 = cur.vertices[l1]
    pts2 = cur.vertices[l2]
    c1 = np.mean(pts1, axis=0)
    c2 = np.mean(pts2, axis=0)
    axis = c2 - c1
    ax_len = float(np.linalg.norm(axis))
    axis = axis / ax_len if ax_len > 1e-12 else np.array([0.0, 0.0, 1.0])
    seed = np.array([1.0, 0.0, 0.0]) if abs(axis[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    u_ax = np.cross(axis, seed)
    u_ax /= max(1e-12, float(np.linalg.norm(u_ax)))
    v_ax = np.cross(axis, u_ax)

    def _order_loop(lp: list[int], cent: np.ndarray) -> list[int]:
        return sorted(
            lp,
            key=lambda idx: math.atan2(
                float(np.dot(cur.vertices[idx] - cent, v_ax)),
                float(np.dot(cur.vertices[idx] - cent, u_ax)),
            ),
        )

    o1 = _order_loop(l1, c1)
    o2 = _order_loop(l2, c2)
    n = max(len(o1), len(o2))
    if len(o1) != n:
        o1 = [o1[int(round(i * len(o1) / n)) % len(o1)] for i in range(n)]
    if len(o2) != n:
        o2 = [o2[int(round(i * len(o2) / n)) % len(o2)] for i in range(n)]

    # Align start vertex of o2 to nearest point to o1[0], plus optional twist
    d2 = [float(np.linalg.norm(cur.vertices[idx] - cur.vertices[o1[0]])) for idx in o2]
    best_shift = (int(np.argmin(d2)) + int(twist)) % n
    o2 = o2[best_shift:] + o2[:best_shift]

    segs = max(1, min(16, int(segments)))
    verts = cur.vertices.tolist()
    uvs = None if cur.uvs is None or len(cur.uvs) != len(verts) else cur.uvs.tolist()

    rings: list[list[int]] = [o1]
    for s in range(1, segs):
        t = s / segs
        ring: list[int] = []
        for k in range(n):
            p = cur.vertices[o1[k]] * (1.0 - t) + cur.vertices[o2[k]] * t
            idx = len(verts)
            verts.append(p.tolist())
            if uvs is not None and cur.uvs is not None:
                uv = cur.uvs[o1[k]] * (1.0 - t) + cur.uvs[o2[k]] * t
                uvs.append(uv.tolist())
            ring.append(idx)
        rings.append(ring)
    rings.append(o2)

    bridge_faces: list[list[int]] = []
    for r in range(segs):
        ra = rings[r]
        rb = rings[r + 1]
        for k in range(n):
            kn = (k + 1) % n
            a0, a1 = ra[k], ra[kn]
            b0, b1 = rb[k], rb[kn]
            bridge_faces.append([a0, b0, b1])
            bridge_faces.append([a0, b1, a1])

    start_b = len(cur.faces)
    all_faces = np.vstack([cur.faces, np.asarray(bridge_faces, dtype=np.int64)])
    sources = kept_src + [-1] * len(bridge_faces)
    added_set = set(range(start_b, len(all_faces)))
    out = MeshData(
        vertices=np.asarray(verts, dtype=np.float64),
        faces=all_faces,
        uvs=None if uvs is None else np.asarray(uvs, dtype=np.float64),
    )
    out = recompute_normals(out)
    return out, sources, added_set


def solidify_faces(
    mesh: MeshData,
    query: SelectionQuery,
    thickness: float = 0.05,
    offset: float = 0.0,
) -> MeshData:
    """Add solid thickness to selected faces (or whole open/closed mesh) with rim walls."""
    if mesh.is_empty:
        return mesh.copy()
    sel_mask = resolve_face_selection(mesh, query)
    if not np.any(sel_mask):
        raise GeometryError(f"Solidify selection ({query.mode}) matched 0 faces.")

    sub_faces = mesh.faces[sel_mask]
    vn = mesh.vertex_normals()
    unique_v = np.unique(sub_faces.ravel())
    v_outer_map = {int(vi): idx for idx, vi in enumerate(unique_v)}
    inner_off = (offset - 0.5) * thickness
    outer_off = (offset + 0.5) * thickness

    v_outer = mesh.vertices[unique_v] + vn[unique_v] * outer_off
    v_inner = mesh.vertices[unique_v] + vn[unique_v] * inner_off
    verts = np.vstack([v_outer, v_inner])
    n_u = len(unique_v)

    faces_out = np.asarray(
        [[v_outer_map[int(a)], v_outer_map[int(b)], v_outer_map[int(c)]] for a, b, c in sub_faces],
        dtype=np.int64,
    )
    faces_in = faces_out[:, [0, 2, 1]] + n_u

    directed: set[tuple[int, int]] = set()
    for a, b, c in faces_out:
        directed.add((int(a), int(b)))
        directed.add((int(b), int(c)))
        directed.add((int(c), int(a)))
    boundary = [(u, v) for (u, v) in directed if (v, u) not in directed]
    rim = []
    for u, v in boundary:
        ui, vi = u + n_u, v + n_u
        rim.append([v, u, ui])
        rim.append([v, ui, vi])

    slab_faces = [faces_out, faces_in]
    if rim:
        slab_faces.append(np.asarray(rim, dtype=np.int64))
    slab = MeshData(vertices=verts, faces=np.vstack(slab_faces))

    if np.all(sel_mask):
        return slab
    rest = MeshData(vertices=mesh.vertices.copy(), faces=mesh.faces[~sel_mask], uvs=mesh.uvs)
    return concat_meshes([rest, slab])


def wireframe_edges(
    mesh: MeshData,
    query: SelectionQuery,
    thickness: float = 0.02,
) -> MeshData:
    """Convert selected (or all) edges into solid 4-sided strut tubes."""
    if mesh.is_empty:
        return mesh.copy()
    edges = resolve_edge_selection(mesh, query)
    if not edges:
        raise GeometryError(f"Wireframe selection ({query.mode}) matched 0 edges.")

    r = max(0.002, float(thickness) * 0.5)
    parts: list[MeshData] = []
    for u, v in edges:
        p0 = mesh.vertices[u]
        p1 = mesh.vertices[v]
        d = p1 - p0
        ln = float(np.linalg.norm(d))
        if ln < 1e-8:
            continue
        nd = d / ln
        seed = np.array([1.0, 0.0, 0.0]) if abs(nd[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
        t1 = np.cross(nd, seed)
        t1 /= max(1e-12, float(np.linalg.norm(t1)))
        t2 = np.cross(nd, t1)
        offs = [
            (t1 * math.cos(k * math.pi / 2.0) + t2 * math.sin(k * math.pi / 2.0)) * r
            for k in range(4)
        ]
        sv = np.asarray([p0 + o for o in offs] + [p1 + o for o in offs], dtype=np.float64)
        sf = np.asarray(
            [
                [0, 2, 1],
                [0, 3, 2],
                [4, 5, 6],
                [4, 6, 7],
                [0, 1, 5],
                [0, 5, 4],
                [1, 2, 6],
                [1, 6, 5],
                [2, 3, 7],
                [2, 7, 6],
                [3, 0, 4],
                [3, 4, 7],
            ],
            dtype=np.int64,
        )
        parts.append(MeshData(vertices=sv, faces=sf))
    return concat_meshes(parts)


def _prune_unreferenced_vertices_impl(
    mesh: MeshData,
    face_sources: list[int] | None = None,
) -> tuple[MeshData, list[int]]:
    src = face_sources if face_sources is not None else list(range(len(mesh.faces)))
    if len(mesh.faces) == 0:
        return MeshData(vertices=np.zeros((0, 3)), faces=np.zeros((0, 3), dtype=np.int64)), []
    used = np.unique(mesh.faces.ravel())
    remap = np.full((len(mesh.vertices),), -1, dtype=np.int64)
    remap[used] = np.arange(len(used), dtype=np.int64)
    new_verts = mesh.vertices[used]
    new_faces = remap[mesh.faces]
    new_uvs = (
        mesh.uvs[used] if mesh.uvs is not None and len(mesh.uvs) == len(mesh.vertices) else None
    )
    return MeshData(vertices=new_verts, faces=new_faces, uvs=new_uvs), src


def merge_vertices(
    mesh: MeshData,
    query: SelectionQuery | None = None,
    merge_mode: Literal["by_distance", "at_center"] = "by_distance",
    distance: float = 1e-4,
) -> MeshData:
    """Merge vertices by distance threshold (Remove Doubles) or collapse selection at center."""
    out, _ = _merge_vertices_impl(mesh, query=query, merge_mode=merge_mode, distance=distance)
    return out


def _merge_vertices_impl(
    mesh: MeshData,
    query: SelectionQuery | None = None,
    merge_mode: Literal["by_distance", "at_center"] = "by_distance",
    distance: float = 1e-4,
    face_sources: list[int] | None = None,
) -> tuple[MeshData, list[int]]:
    if mesh.is_empty:
        return mesh.copy(), list(range(len(mesh.faces)))
    src = face_sources if face_sources is not None else list(range(len(mesh.faces)))
    verts = mesh.vertices.copy()
    faces = mesh.faces.copy()
    uvs = None if mesh.uvs is None or len(mesh.uvs) != len(verts) else mesh.uvs.copy()

    if query is not None and query.mode != "all":
        v_mask = resolve_vertex_selection(mesh, query)
        sel_idx = np.nonzero(v_mask)[0]
    else:
        sel_idx = np.arange(len(verts), dtype=np.int64)

    if len(sel_idx) == 0:
        raise GeometryError("Merge selection matched 0 vertices.")

    remap = np.arange(len(verts), dtype=np.int64)
    if merge_mode == "at_center":
        center = np.mean(verts[sel_idx], axis=0)
        keep = int(sel_idx[0])
        verts[keep] = center
        if uvs is not None:
            uvs[keep] = np.mean(uvs[sel_idx], axis=0)
        remap[sel_idx] = keep
    else:
        tol = max(1e-8, float(distance))
        pts = verts[sel_idx]
        # Union-find clustering within tolerance
        parent = list(range(len(sel_idx)))

        def find(x: int) -> int:
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        for i in range(len(sel_idx)):
            diffs = np.linalg.norm(pts[i + 1 :] - pts[i], axis=1)
            hits = np.nonzero(diffs <= tol)[0]
            for h in hits:
                j = i + 1 + int(h)
                ri, rj = find(i), find(j)
                if ri != rj:
                    if ri < rj:
                        parent[rj] = ri
                    else:
                        parent[ri] = rj

        clusters: dict[int, list[int]] = defaultdict(list)
        for local_i, orig_v in enumerate(sel_idx):
            clusters[find(local_i)].append(int(orig_v))
        for _root, members in clusters.items():
            if len(members) > 1:
                keep = min(members)
                verts[keep] = np.mean(verts[members], axis=0)
                if uvs is not None:
                    uvs[keep] = np.mean(uvs[members], axis=0)
                for m in members:
                    remap[m] = keep

    new_faces = remap[faces]
    valid = (
        (new_faces[:, 0] != new_faces[:, 1])
        & (new_faces[:, 1] != new_faces[:, 2])
        & (new_faces[:, 2] != new_faces[:, 0])
    )
    kept_faces = new_faces[valid]
    kept_src = [src[i] for i in np.nonzero(valid)[0]]
    return _prune_unreferenced_vertices_impl(
        MeshData(vertices=verts, faces=kept_faces, uvs=uvs), kept_src
    )


def delete_elements(
    mesh: MeshData,
    query: SelectionQuery,
    delete_mode: Literal["faces", "vertices", "edges", "only_faces"] = "faces",
) -> MeshData:
    """Delete selected faces, vertices, or edges cleanly."""
    out, _ = _delete_elements_impl(mesh, query=query, delete_mode=delete_mode)
    return out


def _delete_elements_impl(
    mesh: MeshData,
    query: SelectionQuery,
    delete_mode: Literal["faces", "vertices", "edges", "only_faces"] = "faces",
    face_material_ids: dict[int, str] | None = None,
    base_material_id: str | None = None,
) -> tuple[MeshData, list[int]]:
    if mesh.is_empty:
        return mesh.copy(), []
    if delete_mode in ("faces", "only_faces"):
        f_mask = resolve_face_selection(
            mesh, query, face_material_ids=face_material_ids, base_material_id=base_material_id
        )
        if not np.any(f_mask):
            raise GeometryError(f"Delete selection ({query.mode}) matched 0 faces.")
        keep = ~f_mask
    elif delete_mode == "vertices":
        v_mask = resolve_vertex_selection(
            mesh, query, face_material_ids=face_material_ids, base_material_id=base_material_id
        )
        if not np.any(v_mask):
            raise GeometryError(f"Delete selection ({query.mode}) matched 0 vertices.")
        v_del = set(np.nonzero(v_mask)[0].tolist())
        keep = np.asarray(
            [not (int(a) in v_del or int(b) in v_del or int(c) in v_del) for a, b, c in mesh.faces],
            dtype=bool,
        )
    else:
        edges = set(
            resolve_edge_selection(
                mesh, query, face_material_ids=face_material_ids, base_material_id=base_material_id
            )
        )
        if not edges:
            raise GeometryError(f"Delete selection ({query.mode}) matched 0 edges.")
        keep = np.asarray(
            [
                not any(
                    (min(int(u), int(v)), max(int(u), int(v))) in edges
                    for u, v in ((a, b), (b, c), (c, a))
                )
                for a, b, c in mesh.faces
            ],
            dtype=bool,
        )

    kept_idx = np.nonzero(keep)[0]
    new_faces = mesh.faces[kept_idx]
    kept_src = [int(i) for i in kept_idx]
    if delete_mode == "only_faces":
        return MeshData(vertices=mesh.vertices.copy(), faces=new_faces, uvs=mesh.uvs), kept_src
    return _prune_unreferenced_vertices_impl(
        MeshData(vertices=mesh.vertices.copy(), faces=new_faces, uvs=mesh.uvs), kept_src
    )


def duplicate_faces(
    mesh: MeshData,
    query: SelectionQuery,
    translate: Vec3 = (0.0, 0.0, 0.0),
    distance: float = 0.0,
    scale: float = 1.0,
    flip_normals: bool = False,
) -> MeshData:
    """Duplicate selected faces inside the same mesh (Shift+D in Edit Mode)."""
    out, _, _ = _duplicate_faces_impl(
        mesh,
        query,
        translate=translate,
        distance=distance,
        scale=scale,
        flip_normals=flip_normals,
    )
    return out


def _duplicate_faces_impl(
    mesh: MeshData,
    query: SelectionQuery,
    translate: Vec3 = (0.0, 0.0, 0.0),
    distance: float = 0.0,
    scale: float = 1.0,
    flip_normals: bool = False,
    face_material_ids: dict[int, str] | None = None,
    base_material_id: str | None = None,
) -> tuple[MeshData, list[int], set[int]]:
    if mesh.is_empty:
        return mesh.copy(), [], set()
    sel_mask = resolve_face_selection(
        mesh, query, face_material_ids=face_material_ids, base_material_id=base_material_id
    )
    sel_idx = np.nonzero(sel_mask)[0]
    if len(sel_idx) == 0:
        raise GeometryError(f"Duplicate selection ({query.mode}) matched 0 faces.")

    verts = mesh.vertices.tolist()
    uvs = None if mesh.uvs is None or len(mesh.uvs) != len(verts) else mesh.uvs.tolist()
    fnormals = mesh.face_normals()
    avg_n = np.sum(fnormals[sel_idx], axis=0)
    avg_n /= max(1e-12, float(np.linalg.norm(avg_n)))
    unique_v = np.unique(mesh.faces[sel_idx].ravel())
    cent = np.mean(mesh.vertices[unique_v], axis=0)
    tr = np.asarray(translate, dtype=np.float64)

    v_map: dict[int, int] = {}
    for vi in unique_v:
        pos = cent + (mesh.vertices[int(vi)] - cent) * scale + avg_n * distance + tr
        v_map[int(vi)] = len(verts)
        verts.append(pos.tolist())
        if uvs is not None and mesh.uvs is not None:
            uvs.append(mesh.uvs[int(vi)].tolist())

    dup_faces: list[list[int]] = []
    dup_src: list[int] = []
    for fi in sel_idx:
        a, b, c = int(mesh.faces[fi, 0]), int(mesh.faces[fi, 1]), int(mesh.faces[fi, 2])
        tri = [v_map[a], v_map[c], v_map[b]] if flip_normals else [v_map[a], v_map[b], v_map[c]]
        dup_faces.append(tri)
        dup_src.append(int(fi))

    start_idx = len(mesh.faces)
    all_faces = np.vstack([mesh.faces, np.asarray(dup_faces, dtype=np.int64)])
    sources = list(range(start_idx)) + dup_src
    added = set(range(start_idx, len(all_faces)))
    return (
        MeshData(
            vertices=np.asarray(verts, dtype=np.float64),
            faces=all_faces,
            uvs=None if uvs is None else np.asarray(uvs, dtype=np.float64),
        ),
        sources,
        added,
    )


def split_selection(mesh: MeshData, query: SelectionQuery) -> MeshData:
    """Disconnect selected faces from surrounding vertices inside the same mesh (Split/Y)."""
    out, _ = _split_selection_impl(mesh, query)
    return out


def _split_selection_impl(
    mesh: MeshData,
    query: SelectionQuery,
    face_material_ids: dict[int, str] | None = None,
    base_material_id: str | None = None,
) -> tuple[MeshData, list[int]]:
    if mesh.is_empty:
        return mesh.copy(), []
    sel_mask = resolve_face_selection(
        mesh, query, face_material_ids=face_material_ids, base_material_id=base_material_id
    )
    if not np.any(sel_mask) or np.all(sel_mask):
        return mesh.copy(), list(range(len(mesh.faces)))

    verts = mesh.vertices.tolist()
    uvs = None if mesh.uvs is None or len(mesh.uvs) != len(verts) else mesh.uvs.tolist()
    faces = mesh.faces.copy()
    sel_idx = np.nonzero(sel_mask)[0]
    unique_v = np.unique(faces[sel_idx].ravel())
    v_map: dict[int, int] = {}
    for vi in unique_v:
        v_map[int(vi)] = len(verts)
        verts.append(mesh.vertices[int(vi)].tolist())
        if uvs is not None and mesh.uvs is not None:
            uvs.append(mesh.uvs[int(vi)].tolist())
    for fi in sel_idx:
        a, b, c = int(faces[fi, 0]), int(faces[fi, 1]), int(faces[fi, 2])
        faces[fi] = [v_map[a], v_map[b], v_map[c]]
    return _prune_unreferenced_vertices_impl(
        MeshData(
            vertices=np.asarray(verts, dtype=np.float64),
            faces=faces,
            uvs=None if uvs is None else np.asarray(uvs, dtype=np.float64),
        ),
        list(range(len(faces))),
    )


def separate_selection(
    mesh: MeshData,
    query: SelectionQuery,
    face_material_ids: dict[int, str] | None = None,
    base_material_id: str | None = None,
) -> tuple[MeshData, MeshData, dict[int, str], dict[int, str]]:
    """Detach selected faces into a new separate mesh object, preserving per-face materials."""
    if mesh.is_empty:
        raise GeometryError("Cannot separate from an empty mesh.")
    sel_mask = resolve_face_selection(
        mesh, query, face_material_ids=face_material_ids, base_material_id=base_material_id
    )
    sel_idx = np.nonzero(sel_mask)[0]
    rem_idx = np.nonzero(~sel_mask)[0]
    if len(sel_idx) == 0:
        raise GeometryError(f"Separate selection ({query.mode}) matched 0 faces.")
    if len(rem_idx) == 0:
        raise GeometryError("Separate selection includes all faces; leave at least 1 face behind.")

    sep_mesh, sep_src = _prune_unreferenced_vertices_impl(
        MeshData(vertices=mesh.vertices.copy(), faces=mesh.faces[sel_idx], uvs=mesh.uvs),
        [int(i) for i in sel_idx],
    )
    rem_mesh, rem_src = _prune_unreferenced_vertices_impl(
        MeshData(vertices=mesh.vertices.copy(), faces=mesh.faces[rem_idx], uvs=mesh.uvs),
        [int(i) for i in rem_idx],
    )
    sep_mats = _remap_face_materials(face_material_ids, sep_src)
    rem_mats = _remap_face_materials(face_material_ids, rem_src)
    return rem_mesh, sep_mesh, rem_mats, sep_mats


def dissolve_faces(mesh: MeshData, angle_threshold_deg: float = 5.0) -> MeshData:
    """Clean/merge degenerate or near-zero vertices and coplanar redundant facets."""
    if mesh.is_empty:
        return mesh.copy()
    # Clean degenerate triangles & nearly coincident vertices first
    cleaned = merge_vertices(mesh, merge_mode="by_distance", distance=1e-5)
    # Remove zero-area faces if any remain
    if len(cleaned.faces) > 0:
        v0 = cleaned.vertices[cleaned.faces[:, 0]]
        v1 = cleaned.vertices[cleaned.faces[:, 1]]
        v2 = cleaned.vertices[cleaned.faces[:, 2]]
        areas = 0.5 * np.linalg.norm(np.cross(v1 - v0, v2 - v0), axis=1)
        valid = areas > 1e-10
        if not np.all(valid):
            cleaned = MeshData(
                vertices=cleaned.vertices, faces=cleaned.faces[valid], uvs=cleaned.uvs
            )
    return cleaned


def transform_selection(
    mesh: MeshData,
    query: SelectionQuery,
    translate: Vec3 = (0.0, 0.0, 0.0),
    rotate_deg: Vec3 = (0.0, 0.0, 0.0),
    scale: Vec3 = (1.0, 1.0, 1.0),
    proportional_radius: float = 0.0,
    proportional_falloff: Literal["smooth", "sphere", "linear", "sharp"] = "smooth",
) -> MeshData:
    """Transform selected vertices/faces around pivot with optional Proportional Editing."""
    if mesh.is_empty:
        return mesh.copy()
    v_mask = resolve_vertex_selection(mesh, query)
    v_idx = np.nonzero(v_mask)[0]
    if len(v_idx) == 0:
        raise GeometryError(f"Selection query ({query.mode}) matched 0 elements.")

    verts = mesh.vertices.copy()
    pivot = np.mean(verts[v_idx], axis=0)
    mat = compose_matrix(location=translate, rotation_deg=rotate_deg, scale=scale)

    if proportional_radius > 1e-6 and not np.all(v_mask):
        sel_pts = verts[v_idx]
        # Compute minimum distance from each vertex to nearest selected vertex
        diffs = verts[:, None, :] - sel_pts[None, :, :]
        dists = np.min(np.linalg.norm(diffs, axis=2), axis=1)
        u = np.clip(1.0 - dists / float(proportional_radius), 0.0, 1.0)
        if proportional_falloff == "smooth":
            weights = u * u * (3.0 - 2.0 * u)
        elif proportional_falloff == "sphere":
            weights = np.sqrt(np.clip(1.0 - (1.0 - u) ** 2, 0.0, 1.0))
        elif proportional_falloff == "sharp":
            weights = u * u
        else:
            weights = u
        weights[v_idx] = 1.0
        aff = np.nonzero(weights > 1e-6)[0]
        local_pts = verts[aff] - pivot[None, :]
        moved = transform_points(mat, local_pts) + pivot[None, :]
        w_col = weights[aff][:, None]
        verts[aff] = verts[aff] * (1.0 - w_col) + moved * w_col
    else:
        local_pts = verts[v_idx] - pivot[None, :]
        verts[v_idx] = transform_points(mat, local_pts) + pivot[None, :]

    return MeshData(
        vertices=verts, faces=mesh.faces.copy(), uvs=mesh.uvs, face_groups=mesh.face_groups
    )


def shrink_fatten(mesh: MeshData, query: SelectionQuery, distance: float = 0.05) -> MeshData:
    """Move selected vertices along their vertex normals (Alt+S Shrink/Fatten)."""
    if mesh.is_empty:
        return mesh.copy()
    v_mask = resolve_vertex_selection(mesh, query)
    v_idx = np.nonzero(v_mask)[0]
    if len(v_idx) == 0:
        raise GeometryError(f"Shrink/Fatten selection ({query.mode}) matched 0 vertices.")
    verts = mesh.vertices.copy()
    vn = mesh.vertex_normals()
    verts[v_idx] += vn[v_idx] * float(distance)
    return MeshData(
        vertices=verts, faces=mesh.faces.copy(), uvs=mesh.uvs, face_groups=mesh.face_groups
    )


def to_sphere(
    mesh: MeshData,
    query: SelectionQuery,
    factor: float = 1.0,
    radius: float | None = None,
) -> MeshData:
    """Morph selected vertices toward a sphere around selection centroid (Shift+Alt+S)."""
    if mesh.is_empty:
        return mesh.copy()
    v_mask = resolve_vertex_selection(mesh, query)
    v_idx = np.nonzero(v_mask)[0]
    if len(v_idx) == 0:
        raise GeometryError(f"To Sphere selection ({query.mode}) matched 0 vertices.")
    verts = mesh.vertices.copy()
    pts = verts[v_idx]
    cent = np.mean(pts, axis=0)
    vecs = pts - cent[None, :]
    lens = np.linalg.norm(vecs, axis=1)
    target_r = float(radius) if radius is not None else float(np.mean(lens))
    safe_lens = np.where(lens > 1e-12, lens, 1.0)[:, None]
    sph_pts = cent[None, :] + (vecs / safe_lens) * target_r
    t = max(0.0, min(1.0, float(factor)))
    verts[v_idx] = pts * (1.0 - t) + sph_pts * t
    return MeshData(
        vertices=verts, faces=mesh.faces.copy(), uvs=mesh.uvs, face_groups=mesh.face_groups
    )


def shear(
    mesh: MeshData,
    query: SelectionQuery,
    axis: Literal["x", "y", "z"] = "x",
    shear_axis: Literal["x", "y", "z"] = "z",
    factor: float = 0.5,
) -> MeshData:
    """Shear selected vertices along `axis` proportional to offset along `shear_axis`."""
    if mesh.is_empty:
        return mesh.copy()
    v_mask = resolve_vertex_selection(mesh, query)
    v_idx = np.nonzero(v_mask)[0]
    if len(v_idx) == 0:
        raise GeometryError(f"Shear selection ({query.mode}) matched 0 vertices.")
    ax_i = {"x": 0, "y": 1, "z": 2}[axis]
    sh_i = {"x": 0, "y": 1, "z": 2}[shear_axis]
    if ax_i == sh_i:
        sh_i = (ax_i + 2) % 3
    verts = mesh.vertices.copy()
    pivot = np.mean(verts[v_idx], axis=0)
    verts[v_idx, ax_i] += (verts[v_idx, sh_i] - pivot[sh_i]) * float(factor)
    return MeshData(
        vertices=verts, faces=mesh.faces.copy(), uvs=mesh.uvs, face_groups=mesh.face_groups
    )


def randomize_vertices(
    mesh: MeshData,
    query: SelectionQuery,
    strength: float = 0.05,
    seed: int = 42,
    normal_only: bool = False,
) -> MeshData:
    """Displace selected vertices deterministically with seeded noise."""
    if mesh.is_empty:
        return mesh.copy()
    v_mask = resolve_vertex_selection(mesh, query)
    v_idx = np.nonzero(v_mask)[0]
    if len(v_idx) == 0:
        raise GeometryError(f"Randomize selection ({query.mode}) matched 0 vertices.")
    verts = mesh.vertices.copy()
    rng = np.random.default_rng(int(seed))
    if normal_only:
        vn = mesh.vertex_normals()
        jitter = rng.uniform(-1.0, 1.0, size=(len(v_idx), 1)) * float(strength)
        verts[v_idx] += vn[v_idx] * jitter
    else:
        verts[v_idx] += rng.uniform(-1.0, 1.0, size=(len(v_idx), 3)) * float(strength)
    return MeshData(
        vertices=verts, faces=mesh.faces.copy(), uvs=mesh.uvs, face_groups=mesh.face_groups
    )


def symmetrize(
    mesh: MeshData,
    direction: Literal[
        "-x_to_+x", "+x_to_-x", "-y_to_+y", "+y_to_-y", "-z_to_+z", "+z_to_-z"
    ] = "-x_to_+x",
) -> MeshData:
    """Slice along axis=0 plane, keep source half, mirror and weld seamlessly."""
    if mesh.is_empty:
        return mesh.copy()
    ax_name = direction[-1]
    ax_idx = {"x": 0, "y": 1, "z": 2}[ax_name]
    normal = [0.0, 0.0, 0.0]
    normal[ax_idx] = 1.0
    # If "-x_to_+x", source is x <= 0 so clear outer (+x); else source is x >= 0 so clear inner (-x)
    from_neg = direction.startswith("-")
    half = bisect_plane(
        mesh,
        plane_point=(0.0, 0.0, 0.0),
        plane_normal=(normal[0], normal[1], normal[2]),
        clear_inner=not from_neg,
        clear_outer=from_neg,
        fill_caps=False,
    )
    verts_m = half.vertices.copy()
    verts_m[:, ax_idx] = -verts_m[:, ax_idx]
    faces_m = half.faces[:, [0, 2, 1]].copy()
    mirrored = MeshData(vertices=verts_m, faces=faces_m, uvs=half.uvs)
    combined = concat_meshes([half, mirrored])
    return repair_manifold(combined)


def smooth_laplacian(
    mesh: MeshData,
    iterations: int = 2,
    factor: float = 0.5,
    query: SelectionQuery | None = None,
) -> MeshData:
    """Apply Laplacian mesh smoothing (whole mesh or selected vertices only)."""
    if mesh.is_empty:
        return mesh.copy()
    verts = mesh.vertices.copy()
    allowed: set[int] | None = None
    if query is not None and query.mode != "all":
        v_mask = resolve_vertex_selection(mesh, query)
        allowed = set(np.nonzero(v_mask)[0].tolist())
        if not allowed:
            raise GeometryError(f"Smooth selection ({query.mode}) matched 0 vertices.")

    neighbors: dict[int, set[int]] = defaultdict(set)
    for a, b, c in mesh.faces:
        ia, ib, ic = int(a), int(b), int(c)
        neighbors[ia].update((ib, ic))
        neighbors[ib].update((ia, ic))
        neighbors[ic].update((ia, ib))
    for _ in range(max(1, iterations)):
        nxt = verts.copy()
        for vi, nbrs in neighbors.items():
            if allowed is not None and vi not in allowed:
                continue
            if nbrs:
                avg = np.mean(verts[list(nbrs)], axis=0)
                nxt[vi] = verts[vi] + factor * (avg - verts[vi])
        verts = nxt
    return MeshData(
        vertices=verts, faces=mesh.faces.copy(), uvs=mesh.uvs, face_groups=mesh.face_groups
    )


def flip_normals(mesh: MeshData, query: SelectionQuery | None = None) -> MeshData:
    """Flip triangle winding on selected faces (or all faces)."""
    if mesh.is_empty:
        return mesh.copy()
    faces = mesh.faces.copy()
    if query is None or query.mode == "all":
        faces = faces[:, [0, 2, 1]]
    else:
        f_mask = resolve_face_selection(mesh, query)
        if not np.any(f_mask):
            raise GeometryError(f"Flip normals selection ({query.mode}) matched 0 faces.")
        faces[f_mask] = faces[f_mask][:, [0, 2, 1]]
    return MeshData(
        vertices=mesh.vertices.copy(), faces=faces, uvs=mesh.uvs, face_groups=mesh.face_groups
    )


def recompute_normals(mesh: MeshData, inside: bool = False) -> MeshData:
    """Fix inconsistent triangle winding (outward by default, or inward if inside=True)."""
    if mesh.is_empty:
        return mesh.copy()
    tm = mesh.to_trimesh(process=False)
    safe_fix_normals(tm)
    faces = np.asarray(tm.faces, dtype=np.int64)
    if inside and len(faces) > 0:
        faces = faces[:, [0, 2, 1]]
    return MeshData(
        vertices=np.asarray(tm.vertices, dtype=np.float64),
        faces=faces,
        uvs=mesh.uvs if len(tm.vertices) == len(mesh.vertices) else None,
    )


def project_uvs(
    mesh: MeshData,
    projection: Literal[
        "box", "smart_uv", "planar_x", "planar_y", "planar_z", "cylindrical", "spherical"
    ] = "box",
    uv_scale: Vec2 = (1.0, 1.0),
    query: SelectionQuery | None = None,
    angle_limit_deg: float = 66.0,
    island_margin: float = 0.02,
) -> MeshData:
    """Compute/unwrap UV coordinates in Edit Mode (whole mesh or selected vertices)."""
    if mesh.is_empty:
        return mesh.copy()
    if projection == "smart_uv":
        from frfruit.textures.uv_unwrap import smart_uv_unwrap

        out_m, _ = smart_uv_unwrap(
            mesh,
            angle_limit_deg=angle_limit_deg,
            island_margin=island_margin,
            uv_scale=uv_scale,
        )
        return out_m
    verts = mesh.vertices
    vn = mesh.vertex_normals()
    bmin = np.min(verts, axis=0)
    span = np.maximum(np.max(verts, axis=0) - bmin, 1e-6)
    norm_p = (verts - bmin[None, :]) / span[None, :]
    cent = np.mean(verts, axis=0)
    rel = verts - cent[None, :]

    new_uv = np.zeros((len(verts), 2), dtype=np.float64)
    if projection == "planar_z":
        new_uv = norm_p[:, [0, 1]]
    elif projection == "planar_y":
        new_uv = norm_p[:, [0, 2]]
    elif projection == "planar_x":
        new_uv = norm_p[:, [1, 2]]
    elif projection == "cylindrical":
        theta = np.arctan2(rel[:, 1], rel[:, 0])
        new_uv[:, 0] = (theta + math.pi) / (2.0 * math.pi)
        new_uv[:, 1] = norm_p[:, 2]
    elif projection == "spherical":
        r = np.maximum(np.linalg.norm(rel, axis=1), 1e-8)
        theta = np.arctan2(rel[:, 1], rel[:, 0])
        phi = np.arccos(np.clip(rel[:, 2] / r, -1.0, 1.0))
        new_uv[:, 0] = (theta + math.pi) / (2.0 * math.pi)
        new_uv[:, 1] = 1.0 - phi / math.pi
    else:
        # Box/Triplanar projection by dominant vertex normal
        ax = np.abs(vn)
        dom = np.argmax(ax, axis=1)
        new_uv[dom == 0] = norm_p[dom == 0][:, [1, 2]]
        new_uv[dom == 1] = norm_p[dom == 1][:, [0, 2]]
        new_uv[dom == 2] = norm_p[dom == 2][:, [0, 1]]

    new_uv[:, 0] *= float(uv_scale[0])
    new_uv[:, 1] *= float(uv_scale[1])

    if (
        query is not None
        and query.mode != "all"
        and mesh.uvs is not None
        and len(mesh.uvs) == len(verts)
    ):
        out_uv = mesh.uvs.copy()
        v_mask = resolve_vertex_selection(mesh, query)
        out_uv[v_mask] = new_uv[v_mask]
    else:
        out_uv = new_uv

    return MeshData(
        vertices=mesh.vertices.copy(),
        faces=mesh.faces.copy(),
        uvs=out_uv,
        face_groups=mesh.face_groups,
    )


def repair_manifold(mesh: MeshData) -> MeshData:
    """Merge coincident vertices, remove degenerate faces, fill small holes, fix normals."""
    if mesh.is_empty:
        return mesh.copy()
    tm = trimesh.Trimesh(vertices=mesh.vertices, faces=mesh.faces, process=True)
    tm.merge_vertices(digits_vertex=6)
    trimesh.repair.fill_holes(tm)
    safe_fix_normals(tm)
    return MeshData.from_trimesh(tm)


def boolean_meshes(
    mesh_a: MeshData,
    mesh_b: MeshData,
    operation: str,
) -> MeshData:
    """Perform exact watertight CSG boolean (union, difference, intersection)."""
    mf_a = mesh_a.to_manifold()
    mf_b = mesh_b.to_manifold()
    if operation == "union":
        res = mf_a + mf_b
    elif operation == "difference":
        res = mf_a - mf_b
    elif operation == "intersection":
        res = mf_a ^ mf_b
    else:
        raise GeometryError(f"Unknown boolean operation: {operation!r}")
    return MeshData.from_manifold(res)


# ---------------------------------------------------------------------------
# Unified Edit Mode Dispatcher (Preserving Per-Face Material IDs)
# ---------------------------------------------------------------------------


def run_edit_operation(
    mesh: MeshData,
    operation: str,
    query: SelectionQuery,
    *,
    face_material_ids: dict[int, str] | None = None,
    base_material_id: str | None = None,
    distance: float = 0.3,
    thickness: float = 0.1,
    depth: float = 0.0,
    width: float = 0.04,
    segments: int = 1,
    levels: int = 1,
    cuts: int = 1,
    slide: float = 0.0,
    smooth_subdiv: bool = True,
    individual: bool = False,
    along_normals: bool = False,
    direction: Vec3 | None = None,
    scale: float = 1.0,
    translate: Vec3 = (0.0, 0.0, 0.0),
    rotate_deg: Vec3 = (0.0, 0.0, 0.0),
    scale_vec: Vec3 | None = None,
    iterations: int = 2,
    factor: float = 0.5,
    axis: Literal["x", "y", "z"] = "z",
    shear_axis: Literal["x", "y", "z"] = "z",
    plane_point: Vec3 = (0.0, 0.0, 0.0),
    plane_normal: Vec3 = (0.0, 0.0, 1.0),
    clear_inner: bool = False,
    clear_outer: bool = False,
    fill_caps: bool = True,
    merge_mode: Literal["by_distance", "at_center"] = "by_distance",
    merge_distance: float = 1e-4,
    delete_mode: Literal["faces", "vertices", "edges", "only_faces"] = "faces",
    proportional_radius: float = 0.0,
    proportional_falloff: Literal["smooth", "sphere", "linear", "sharp"] = "smooth",
    symmetrize_direction: Literal[
        "-x_to_+x", "+x_to_-x", "-y_to_+y", "+y_to_-y", "-z_to_+z", "+z_to_-z"
    ] = "-x_to_+x",
    projection: Literal[
        "box", "smart_uv", "planar_x", "planar_y", "planar_z", "cylindrical", "spherical"
    ] = "box",
    uv_scale: Vec2 = (1.0, 1.0),
    uv_angle_limit_deg: float = 66.0,
    island_margin: float = 0.02,
    subdiv_mode: Literal["catmull_clark", "loop", "simple"] = "catmull_clark",
    crease_angle_deg: float = 180.0,
    flip_normals_flag: bool = False,
    inside: bool = False,
    offset: float = 0.0,
    seed: int = 42,
    normal_only: bool = False,
    target_query: SelectionQuery | None = None,
    twist: int = 0,
    material_id: str | None = None,
) -> EditOpOutput:
    """Execute any Edit Mode modeling operation while preserving/updating per-face materials."""
    fm = dict(face_material_ids or {})

    if operation in ("extrude_faces", "extrude_individual"):
        is_ind = individual or (operation == "extrude_individual")
        out_m, src, affected = _extrude_faces_impl(
            mesh,
            query,
            distance=distance,
            scale=scale,
            direction=direction,
            individual=is_ind,
            along_normals=along_normals,
            face_material_ids=fm,
            base_material_id=base_material_id,
        )
        new_fm = _remap_face_materials(
            fm, src, override_indices=affected, override_material_id=material_id
        )
        return EditOpOutput(mesh=out_m, face_material_ids=new_fm)

    if operation == "extrude_along_normals":
        out_m, src, affected = _extrude_faces_impl(
            mesh,
            query,
            distance=distance,
            scale=scale,
            direction=None,
            individual=False,
            along_normals=True,
            face_material_ids=fm,
            base_material_id=base_material_id,
        )
        new_fm = _remap_face_materials(
            fm, src, override_indices=affected, override_material_id=material_id
        )
        return EditOpOutput(mesh=out_m, face_material_ids=new_fm)

    if operation == "extrude_manifold":
        out_m = extrude_manifold(mesh, query, distance=distance, scale=scale, direction=direction)
        return EditOpOutput(mesh=out_m, face_material_ids={})

    if operation == "extrude_edges":
        orig_n = len(mesh.faces)
        out_m = extrude_edges(mesh, query, distance=distance, direction=direction)
        src = list(range(orig_n)) + [-1] * (len(out_m.faces) - orig_n)
        added = set(range(orig_n, len(out_m.faces)))
        new_fm = _remap_face_materials(
            fm, src, override_indices=added, override_material_id=material_id
        )
        return EditOpOutput(mesh=out_m, face_material_ids=new_fm)

    if operation == "extrude_vertices":
        orig_n = len(mesh.faces)
        out_m = extrude_vertices(mesh, query, distance=distance, direction=direction, width=width)
        src = list(range(orig_n)) + [-1] * (len(out_m.faces) - orig_n)
        added = set(range(orig_n, len(out_m.faces)))
        new_fm = _remap_face_materials(
            fm, src, override_indices=added, override_material_id=material_id
        )
        return EditOpOutput(mesh=out_m, face_material_ids=new_fm)

    if operation == "inset_faces":
        out_m, src, inset_set = _inset_faces_impl(
            mesh,
            query,
            thickness=thickness,
            depth=depth,
            individual=individual,
            face_material_ids=fm,
            base_material_id=base_material_id,
        )
        new_fm = _remap_face_materials(
            fm, src, override_indices=inset_set, override_material_id=material_id
        )
        return EditOpOutput(mesh=out_m, face_material_ids=new_fm)

    if operation == "bevel_edges":
        out_m = bevel_edges(
            mesh,
            angle_threshold_deg=query.angle_threshold_deg,
            width=width,
            segments=segments,
            query=query,
        )
        return EditOpOutput(mesh=out_m, face_material_ids={})

    if operation == "bevel_vertices":
        out_m = bevel_vertices(mesh, query, width=width)
        return EditOpOutput(mesh=out_m, face_material_ids={})

    if operation == "subdivide":
        eff_levels = max(levels, min(3, cuts))
        out_m, src = _subdivide_mesh_impl(
            mesh,
            levels=eff_levels,
            smooth=smooth_subdiv,
            query=query,
            face_material_ids=fm,
            base_material_id=base_material_id,
            algorithm=subdiv_mode,
            crease_angle_deg=crease_angle_deg,
        )
        return EditOpOutput(mesh=out_m, face_material_ids=_remap_face_materials(fm, src))

    if operation == "loop_cut":
        out_m, src = _loop_cut_impl(
            mesh,
            axis=axis,
            cuts=cuts,
            slide=slide,
            plane_normal=direction,
            plane_point=plane_point if any(abs(v) > 1e-9 for v in plane_point) else None,
        )
        return EditOpOutput(mesh=out_m, face_material_ids=_remap_face_materials(fm, src))

    if operation == "bisect_plane":
        out_m, src, cap_set = _bisect_plane_impl(
            mesh,
            plane_point=plane_point,
            plane_normal=plane_normal,
            clear_inner=clear_inner,
            clear_outer=clear_outer,
            fill_caps=fill_caps,
        )
        new_fm = _remap_face_materials(
            fm, src, override_indices=cap_set, override_material_id=material_id
        )
        return EditOpOutput(mesh=out_m, face_material_ids=new_fm)

    if operation == "poke_faces":
        out_m, src = _poke_faces_impl(
            mesh,
            query,
            offset=offset if abs(offset) > 1e-9 else depth,
            face_material_ids=fm,
            base_material_id=base_material_id,
        )
        return EditOpOutput(mesh=out_m, face_material_ids=_remap_face_materials(fm, src))

    if operation == "bridge_edge_loops":
        out_m, src, added = _bridge_edge_loops_impl(
            mesh, query=query, target_query=target_query, segments=segments, twist=twist
        )
        new_fm = _remap_face_materials(
            fm, src, override_indices=added, override_material_id=material_id
        )
        return EditOpOutput(mesh=out_m, face_material_ids=new_fm)

    if operation == "fill_holes":
        out_m, src, caps = _fill_holes_impl(mesh, query=query)
        new_fm = _remap_face_materials(
            fm, src, override_indices=caps, override_material_id=material_id
        )
        return EditOpOutput(mesh=out_m, face_material_ids=new_fm)

    if operation == "solidify_faces":
        out_m = solidify_faces(mesh, query, thickness=thickness, offset=offset)
        return EditOpOutput(mesh=out_m, face_material_ids={})

    if operation == "wireframe_edges":
        out_m = wireframe_edges(mesh, query, thickness=width)
        return EditOpOutput(mesh=out_m, face_material_ids={})

    if operation == "merge_vertices":
        out_m, src = _merge_vertices_impl(
            mesh, query=query, merge_mode=merge_mode, distance=merge_distance
        )
        return EditOpOutput(mesh=out_m, face_material_ids=_remap_face_materials(fm, src))

    if operation == "delete_elements":
        out_m, src = _delete_elements_impl(
            mesh,
            query,
            delete_mode=delete_mode,
            face_material_ids=fm,
            base_material_id=base_material_id,
        )
        return EditOpOutput(mesh=out_m, face_material_ids=_remap_face_materials(fm, src))

    if operation == "dissolve_faces":
        out_m = dissolve_faces(mesh, angle_threshold_deg=query.angle_threshold_deg)
        return EditOpOutput(mesh=out_m, face_material_ids={})

    if operation == "duplicate_faces":
        out_m, src, added = _duplicate_faces_impl(
            mesh,
            query,
            translate=translate,
            distance=distance,
            scale=scale,
            flip_normals=flip_normals_flag,
            face_material_ids=fm,
            base_material_id=base_material_id,
        )
        new_fm = _remap_face_materials(
            fm, src, override_indices=added, override_material_id=material_id
        )
        return EditOpOutput(mesh=out_m, face_material_ids=new_fm)

    if operation == "split_selection":
        out_m, src = _split_selection_impl(
            mesh, query, face_material_ids=fm, base_material_id=base_material_id
        )
        return EditOpOutput(mesh=out_m, face_material_ids=_remap_face_materials(fm, src))

    if operation == "separate_selection":
        rem_m, sep_m, rem_fm, sep_fm = separate_selection(
            mesh, query, face_material_ids=fm, base_material_id=base_material_id
        )
        return EditOpOutput(
            mesh=rem_m,
            face_material_ids=rem_fm,
            separated_mesh=sep_m,
            separated_face_material_ids=sep_fm,
        )

    if operation == "transform_selection":
        sv = scale_vec if scale_vec is not None else (scale, scale, scale)
        out_m = transform_selection(
            mesh,
            query,
            translate=translate,
            rotate_deg=rotate_deg,
            scale=sv,
            proportional_radius=proportional_radius,
            proportional_falloff=proportional_falloff,
        )
        return EditOpOutput(mesh=out_m, face_material_ids=fm)

    if operation == "shrink_fatten":
        out_m = shrink_fatten(mesh, query, distance=distance)
        return EditOpOutput(mesh=out_m, face_material_ids=fm)

    if operation == "to_sphere":
        out_m = to_sphere(mesh, query, factor=factor)
        return EditOpOutput(mesh=out_m, face_material_ids=fm)

    if operation == "shear":
        out_m = shear(mesh, query, axis=axis, shear_axis=shear_axis, factor=factor)
        return EditOpOutput(mesh=out_m, face_material_ids=fm)

    if operation == "randomize_vertices":
        out_m = randomize_vertices(mesh, query, strength=factor, seed=seed, normal_only=normal_only)
        return EditOpOutput(mesh=out_m, face_material_ids=fm)

    if operation == "symmetrize":
        out_m = symmetrize(mesh, direction=symmetrize_direction)
        return EditOpOutput(mesh=out_m, face_material_ids={})

    if operation == "smooth":
        out_m = smooth_laplacian(mesh, iterations=iterations, factor=factor, query=query)
        return EditOpOutput(mesh=out_m, face_material_ids=fm)

    if operation == "flip_normals":
        out_m = flip_normals(mesh, query=query)
        return EditOpOutput(mesh=out_m, face_material_ids=fm)

    if operation == "recompute_normals":
        out_m = recompute_normals(mesh, inside=inside)
        return EditOpOutput(mesh=out_m, face_material_ids=fm)

    if operation == "project_uvs":
        if projection == "smart_uv":
            from frfruit.textures.uv_unwrap import smart_uv_unwrap

            out_m, uv_extra = smart_uv_unwrap(
                mesh,
                angle_limit_deg=uv_angle_limit_deg,
                island_margin=island_margin,
                uv_scale=uv_scale,
            )
            return EditOpOutput(mesh=out_m, face_material_ids=fm, extra=uv_extra)
        out_m = project_uvs(
            mesh,
            projection=projection,
            uv_scale=uv_scale,
            query=query,
            angle_limit_deg=uv_angle_limit_deg,
            island_margin=island_margin,
        )
        return EditOpOutput(mesh=out_m, face_material_ids=fm)

    if operation == "assign_face_material":
        if not material_id:
            raise GeometryError("material_id is required for assign_face_material.")
        sel_mask = resolve_face_selection(
            mesh, query, face_material_ids=fm, base_material_id=base_material_id
        )
        sel_idx = np.nonzero(sel_mask)[0]
        if len(sel_idx) == 0:
            raise GeometryError(f"Selection query ({query.mode}) matched 0 faces.")
        for fi in sel_idx:
            fm[int(fi)] = material_id
        return EditOpOutput(
            mesh=mesh.copy(),
            face_material_ids=fm,
            extra={"assigned_face_count": int(len(sel_idx)), "assigned_faces": sel_idx.tolist()},
        )

    if operation == "clear_face_materials":
        if query.mode == "all":
            fm.clear()
        else:
            sel_mask = resolve_face_selection(
                mesh, query, face_material_ids=fm, base_material_id=base_material_id
            )
            for fi in np.nonzero(sel_mask)[0]:
                fm.pop(int(fi), None)
        return EditOpOutput(mesh=mesh.copy(), face_material_ids=fm)

    if operation == "repair_manifold":
        out_m = repair_manifold(mesh)
        return EditOpOutput(mesh=out_m, face_material_ids={})

    raise GeometryError(f"Unsupported Edit Mesh operation: {operation!r}")
