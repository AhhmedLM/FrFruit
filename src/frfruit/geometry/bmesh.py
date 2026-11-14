"""Quad-friendly B-Mesh reconstruction and crease-preserving Catmull-Clark subdivision."""

from __future__ import annotations

import math
from collections import defaultdict

import numpy as np

from frfruit.geometry.mesh import MeshData


def reconstruct_quads(
    mesh: MeshData,
    angle_threshold_deg: float = 15.0,
) -> tuple[list[list[int]], list[int]]:
    """Pair coplanar adjacent triangles into clean convex quads while keeping stray tris.

    Returns `(polygons, poly_sources)` where `polygons` contains 3- or 4-index CCW cycles
    and `poly_sources` records the primary input triangle index for material slot propagation.
    """
    faces = mesh.faces
    n_faces = len(faces)
    if n_faces == 0:
        return [], []

    verts = mesh.vertices
    fnormals = mesh.face_normals()
    cos_thresh = math.cos(math.radians(max(0.1, min(89.0, angle_threshold_deg))))

    # Map directed half-edge (u, v) -> list[(face_idx, opp_vertex)]
    half_edges: dict[tuple[int, int], list[tuple[int, int]]] = defaultdict(list)
    for fi, (a, b, c) in enumerate(faces):
        ia, ib, ic = int(a), int(b), int(c)
        half_edges[(ia, ib)].append((fi, ic))
        half_edges[(ib, ic)].append((fi, ia))
        half_edges[(ic, ia)].append((fi, ib))

    # Candidate pairs scored by coplanarity and quad diagonal balance
    candidates: list[tuple[float, int, int, list[int]]] = []
    seen_pairs: set[tuple[int, int]] = set()
    for (u, v), list_uv in half_edges.items():
        if len(list_uv) != 1:
            continue
        list_vu = half_edges.get((v, u))
        if not list_vu or len(list_vu) != 1:
            continue
        f1, a = list_uv[0]
        f2, b = list_vu[0]
        if f1 == f2:
            continue
        pair_key = (min(f1, f2), max(f1, f2))
        if pair_key in seen_pairs:
            continue
        seen_pairs.add(pair_key)

        dot_n = float(np.dot(fnormals[f1], fnormals[f2]))
        if dot_n < cos_thresh:
            continue

        # Check strict convexity of the combined CCW quad [u, b, v, a]
        quad = [u, b, v, a]
        if len(set(quad)) != 4:
            continue
        n_avg = fnormals[f1] + fnormals[f2]
        n_len = float(np.linalg.norm(n_avg))
        if n_len < 1e-12:
            continue
        n_avg /= n_len

        pts = verts[quad]
        convex = True
        for k in range(4):
            e1 = pts[(k + 1) % 4] - pts[k]
            e2 = pts[(k + 2) % 4] - pts[(k + 1) % 4]
            if float(np.dot(np.cross(e1, e2), n_avg)) <= 1e-9:
                convex = False
                break
        if not convex:
            continue

        # Prefer pairing hypotenuse diagonals (diagonal ab close in length to shared uv)
        d_shared = float(np.linalg.norm(verts[v] - verts[u]))
        d_cross = float(np.linalg.norm(verts[b] - verts[a]))
        diag_balance = min(d_shared, d_cross) / max(d_shared, d_cross, 1e-9)
        score = dot_n * 2.0 + diag_balance
        candidates.append((score, f1, f2, quad))

    candidates.sort(key=lambda item: item[0], reverse=True)

    used = np.zeros((n_faces,), dtype=bool)
    polygons: list[list[int]] = []
    poly_sources: list[int] = []

    for _, f1, f2, quad in candidates:
        if used[f1] or used[f2]:
            continue
        used[f1] = True
        used[f2] = True
        polygons.append(quad)
        poly_sources.append(f1)

    for fi in range(n_faces):
        if not used[fi]:
            polygons.append([int(faces[fi, 0]), int(faces[fi, 1]), int(faces[fi, 2])])
            poly_sources.append(fi)

    return polygons, poly_sources


def _poly_normal(verts: np.ndarray, poly: list[int]) -> np.ndarray:
    p0 = verts[poly[0]]
    acc = np.zeros((3,), dtype=np.float64)
    for i in range(1, len(poly) - 1):
        acc += np.cross(verts[poly[i]] - p0, verts[poly[i + 1]] - p0)
    ln = float(np.linalg.norm(acc))
    return acc / ln if ln > 1e-12 else np.array([0.0, 0.0, 1.0], dtype=np.float64)


def _catmull_clark_step(
    verts: np.ndarray,
    polygons: list[list[int]],
    uvs: np.ndarray | None,
    poly_sources: list[int],
    *,
    smooth: bool,
    crease_cos: float,
) -> tuple[np.ndarray, list[list[int]], np.ndarray | None, list[int]]:
    n_v = len(verts)
    n_p = len(polygons)
    has_uv = uvs is not None and len(uvs) == n_v

    # 1. Face points & polygon normals
    face_pts = np.empty((n_p, 3), dtype=np.float64)
    face_uvs = np.empty((n_p, 2), dtype=np.float64) if has_uv else None
    poly_normals = np.empty((n_p, 3), dtype=np.float64)

    edge_to_polys: dict[tuple[int, int], list[int]] = defaultdict(list)
    vert_to_polys: list[list[int]] = [[] for _ in range(n_v)]
    vert_neighbors: list[set[int]] = [set() for _ in range(n_v)]

    for pi, poly in enumerate(polygons):
        idx = np.asarray(poly, dtype=np.int64)
        face_pts[pi] = np.mean(verts[idx], axis=0)
        if face_uvs is not None and uvs is not None:
            face_uvs[pi] = np.mean(uvs[idx], axis=0)
        poly_normals[pi] = _poly_normal(verts, poly)
        k = len(poly)
        for j in range(k):
            u = poly[j]
            v = poly[(j + 1) % k]
            vert_to_polys[u].append(pi)
            vert_neighbors[u].add(v)
            vert_neighbors[v].add(u)
            edge_to_polys[(min(u, v), max(u, v))].append(pi)

    # 2. Identify crease / boundary edges and compute edge points
    edge_keys = list(edge_to_polys.keys())
    edge_pt_idx: dict[tuple[int, int], int] = {}
    edge_is_sharp: dict[tuple[int, int], bool] = {}

    new_verts_list: list[np.ndarray] = [verts.copy()]
    new_uvs_list: list[np.ndarray] | None = [uvs.copy()] if has_uv and uvs is not None else None

    face_pt_offset = n_v
    new_verts_list.append(face_pts)
    if new_uvs_list is not None and face_uvs is not None:
        new_uvs_list.append(face_uvs)

    edge_pts_arr = np.empty((len(edge_keys), 3), dtype=np.float64)
    edge_uvs_arr = np.empty((len(edge_keys), 2), dtype=np.float64) if has_uv else None
    edge_offset = n_v + n_p

    for ei, ek in enumerate(edge_keys):
        u, v = ek
        inc = edge_to_polys[ek]
        is_boundary = len(inc) != 2
        is_crease = False
        if len(inc) == 2 and crease_cos > -0.9999:
            dot_n = float(np.dot(poly_normals[inc[0]], poly_normals[inc[1]]))
            if dot_n < crease_cos:
                is_crease = True
        sharp = (not smooth) or is_boundary or is_crease
        edge_is_sharp[ek] = sharp
        edge_pt_idx[ek] = edge_offset + ei

        if sharp:
            edge_pts_arr[ei] = 0.5 * (verts[u] + verts[v])
        else:
            edge_pts_arr[ei] = 0.25 * (verts[u] + verts[v] + face_pts[inc[0]] + face_pts[inc[1]])
        if edge_uvs_arr is not None and uvs is not None:
            edge_uvs_arr[ei] = 0.5 * (uvs[u] + uvs[v])

    new_verts_list.append(edge_pts_arr)
    if new_uvs_list is not None and edge_uvs_arr is not None:
        new_uvs_list.append(edge_uvs_arr)

    # 3. Update original vertex positions (smooth Catmull-Clark / crease & boundary rules)
    if smooth:
        updated = verts.copy()
        for vi in range(n_v):
            nbrs = vert_neighbors[vi]
            inc_polys = vert_to_polys[vi]
            if not nbrs or not inc_polys:
                continue
            sharp_nbrs = [nb for nb in nbrs if edge_is_sharp.get((min(vi, nb), max(vi, nb)), False)]
            if len(sharp_nbrs) >= 3:
                # Corner vertex stays fixed
                updated[vi] = verts[vi]
            elif len(sharp_nbrs) == 2:
                # Smooth crease / boundary curve rule
                updated[vi] = (
                    0.75 * verts[vi] + 0.125 * verts[sharp_nbrs[0]] + 0.125 * verts[sharp_nbrs[1]]
                )
            elif len(sharp_nbrs) == 1:
                updated[vi] = verts[vi]
            else:
                n = len(inc_polys)
                if n < 3 or len(nbrs) < 3:
                    continue
                q_pt = np.mean(face_pts[inc_polys], axis=0)
                r_pt = np.mean([0.5 * (verts[vi] + verts[nb]) for nb in nbrs], axis=0)
                updated[vi] = (q_pt + 2.0 * r_pt + (n - 3.0) * verts[vi]) / float(n)
        new_verts_list[0] = updated

    out_verts = np.vstack(new_verts_list)
    out_uvs = np.vstack(new_uvs_list) if new_uvs_list is not None else None

    # 4. Build pure-quad subdivided faces
    new_polys: list[list[int]] = []
    new_sources: list[int] = []
    for pi, poly in enumerate(polygons):
        k = len(poly)
        fp = face_pt_offset + pi
        src = poly_sources[pi]
        for j in range(k):
            v = poly[j]
            v_next = poly[(j + 1) % k]
            v_prev = poly[(j - 1) % k]
            e_next = edge_pt_idx[(min(v, v_next), max(v, v_next))]
            e_prev = edge_pt_idx[(min(v_prev, v), max(v_prev, v))]
            new_polys.append([v, e_next, fp, e_prev])
            new_sources.append(src)

    return out_verts, new_polys, out_uvs, new_sources


def _triangulate_polygons(
    verts: np.ndarray,
    polygons: list[list[int]],
    poly_sources: list[int],
) -> tuple[np.ndarray, list[int]]:
    """Split quad/n-gon polygons cleanly into triangles picking the better diagonal."""
    tris: list[list[int]] = []
    tri_sources: list[int] = []
    for poly, src in zip(polygons, poly_sources, strict=True):
        if len(poly) == 3:
            tris.append(poly)
            tri_sources.append(src)
        elif len(poly) == 4:
            a, b, c, d = poly
            # Choose shorter diagonal to prevent thin sliver triangles on curved quads
            d_ac = float(np.sum((verts[a] - verts[c]) ** 2))
            d_bd = float(np.sum((verts[b] - verts[d]) ** 2))
            if d_ac <= d_bd:
                tris.append([a, b, c])
                tris.append([a, c, d])
            else:
                tris.append([a, b, d])
                tris.append([b, c, d])
            tri_sources.extend([src, src])
        else:
            p0 = poly[0]
            for i in range(1, len(poly) - 1):
                tris.append([p0, poly[i], poly[i + 1]])
                tri_sources.append(src)
    return np.asarray(tris, dtype=np.int64), tri_sources


def catmull_clark_subdivide(
    mesh: MeshData,
    levels: int = 1,
    smooth: bool = True,
    crease_angle_deg: float = 180.0,
    quad_angle_deg: float = 15.0,
    face_sources: list[int] | None = None,
) -> tuple[MeshData, list[int], int]:
    """Reconstruct quads and apply Catmull-Clark subdivision preserving boundaries & creases.

    Returns `(subdivided_mesh, tri_face_sources, quad_count)`.
    """
    if mesh.is_empty:
        return mesh.copy(), list(range(len(mesh.faces))), 0

    src_in = face_sources if face_sources is not None else list(range(len(mesh.faces)))
    polygons, poly_sources_raw = reconstruct_quads(mesh, angle_threshold_deg=quad_angle_deg)
    poly_sources = [src_in[s] for s in poly_sources_raw]
    quad_count = sum(1 for p in polygons if len(p) == 4)

    verts = mesh.vertices.copy()
    uvs = mesh.uvs.copy() if mesh.uvs is not None and len(mesh.uvs) == len(verts) else None
    crease_cos = math.cos(math.radians(crease_angle_deg)) if crease_angle_deg < 179.9 else -1.0

    for _ in range(max(1, min(3, int(levels)))):
        verts, polygons, uvs, poly_sources = _catmull_clark_step(
            verts,
            polygons,
            uvs,
            poly_sources,
            smooth=smooth,
            crease_cos=crease_cos,
        )

    final_quad_count = len(polygons)
    faces, tri_sources = _triangulate_polygons(verts, polygons, poly_sources)
    out = MeshData(vertices=verts, faces=faces, uvs=uvs)
    return out, tri_sources, max(quad_count, final_quad_count)
