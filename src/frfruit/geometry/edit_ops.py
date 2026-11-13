"""AI-friendly semantic Edit Mesh operations preserving topology and watertightness."""

from __future__ import annotations

import math
from collections import defaultdict

import numpy as np
import trimesh

from frfruit.core.math3d import compose_matrix, transform_points
from frfruit.core.types import SelectionQuery, Vec3
from frfruit.geometry.mesh import GeometryError, MeshData


def resolve_face_selection(mesh: MeshData, query: SelectionQuery) -> np.ndarray:
    """Resolve a semantic SelectionQuery into a boolean mask over faces."""
    m_count = len(mesh.faces)
    if m_count == 0:
        return np.zeros((0,), dtype=bool)

    mode = query.mode
    if mode == "all":
        return np.ones((m_count,), dtype=bool)

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
        dots = normals @ t_vec
        return dots >= cos_thresh

    if mode == "bbox":
        if query.bbox_min is None or query.bbox_max is None:
            raise GeometryError("bbox selection requires both bbox_min and bbox_max.")
        bmin = np.asarray(query.bbox_min, dtype=np.float64)
        bmax = np.asarray(query.bbox_max, dtype=np.float64)
        centroids = np.mean(mesh.vertices[mesh.faces], axis=1)
        return np.all((centroids >= bmin - 1e-6) & (centroids <= bmax + 1e-6), axis=1)

    if mode == "indices":
        mask = np.zeros((m_count,), dtype=bool)
        if query.indices:
            valid = [i for i in query.indices if 0 <= i < m_count]
            mask[valid] = True
        return mask

    raise GeometryError(f"Unsupported selection mode: {mode}")


def _patch_boundary_edges(faces: np.ndarray, sel_mask: np.ndarray) -> list[tuple[int, int]]:
    """Return directed boundary edges (u, v) of the selected face patch."""
    selected_faces = faces[sel_mask]
    directed: set[tuple[int, int]] = set()
    for a, b, c in selected_faces:
        directed.add((int(a), int(b)))
        directed.add((int(b), int(c)))
        directed.add((int(c), int(a)))
    return [(u, v) for (u, v) in directed if (v, u) not in directed]


def extrude_faces(
    mesh: MeshData,
    query: SelectionQuery,
    distance: float = 0.5,
    scale: float = 1.0,
) -> MeshData:
    """Extrude semantically selected faces while keeping closed meshes watertight."""
    if mesh.is_empty:
        return mesh.copy()
    sel_mask = resolve_face_selection(mesh, query)
    if not np.any(sel_mask):
        raise GeometryError(f"Selection query ({query.mode}) matched 0 faces.")

    verts = mesh.vertices.copy()
    faces = mesh.faces.copy()
    normals = mesh.face_normals()

    sel_idx = np.nonzero(sel_mask)[0]
    avg_normal = np.sum(normals[sel_idx], axis=0)
    n_len = float(np.linalg.norm(avg_normal))
    avg_normal = avg_normal / n_len if n_len > 1e-12 else np.array([0.0, 0.0, 1.0])

    sel_faces = faces[sel_idx]
    unique_v = np.unique(sel_faces.ravel())
    centroid = np.mean(verts[unique_v], axis=0)

    boundary_edges = _patch_boundary_edges(faces, sel_mask)
    if not boundary_edges:
        # Whole mesh or closed shell selected: scale & translate selected vertices directly
        for vi in unique_v:
            verts[vi] = centroid + (verts[vi] - centroid) * scale + avg_normal * distance
        return MeshData(vertices=verts, faces=faces)

    # Duplicate vertices used by selected faces so unselected neighbors stay anchored
    v_map: dict[int, int] = {}
    new_verts_list = verts.tolist()
    for vi in unique_v:
        pos = centroid + (verts[vi] - centroid) * scale + avg_normal * distance
        new_idx = len(new_verts_list)
        new_verts_list.append(pos.tolist())
        v_map[int(vi)] = new_idx

    # Update selected faces to point to extruded vertices
    for fi in sel_idx:
        a, b, c = int(faces[fi, 0]), int(faces[fi, 1]), int(faces[fi, 2])
        faces[fi] = [v_map[a], v_map[b], v_map[c]]

    # Stitch boundary quads (u, v, v_ext, u_ext)
    wall_faces: list[list[int]] = []
    for u, v in boundary_edges:
        u_ext = v_map[u]
        v_ext = v_map[v]
        wall_faces.append([u, v, v_ext])
        wall_faces.append([u, v_ext, u_ext])

    all_faces = np.vstack([faces, np.asarray(wall_faces, dtype=np.int64)])
    return MeshData(vertices=np.asarray(new_verts_list, dtype=np.float64), faces=all_faces)


def inset_faces(
    mesh: MeshData,
    query: SelectionQuery,
    thickness: float = 0.1,
    depth: float = 0.0,
) -> MeshData:
    """Inset semantically selected faces inward (with optional depth offset)."""
    if mesh.is_empty:
        return mesh.copy()
    sel_mask = resolve_face_selection(mesh, query)
    if not np.any(sel_mask):
        raise GeometryError(f"Selection query ({query.mode}) matched 0 faces.")

    verts = mesh.vertices.copy()
    faces = mesh.faces.copy()
    normals = mesh.face_normals()
    sel_idx = np.nonzero(sel_mask)[0]

    avg_normal = np.sum(normals[sel_idx], axis=0)
    n_len = float(np.linalg.norm(avg_normal))
    avg_normal = avg_normal / n_len if n_len > 1e-12 else np.array([0.0, 0.0, 1.0])

    sel_faces = faces[sel_idx]
    unique_v = np.unique(sel_faces.ravel())
    centroid = np.mean(verts[unique_v], axis=0)

    boundary_edges = _patch_boundary_edges(faces, sel_mask)
    if not boundary_edges:
        raise GeometryError("Inset requires a partial face selection with boundary edges.")

    # Determine safe scale ratio from thickness vs average boundary radius
    b_verts = np.unique([u for u, _ in boundary_edges])
    radii = np.linalg.norm(verts[b_verts] - centroid[None, :], axis=1)
    avg_r = max(1e-6, float(np.mean(radii)))
    inset_scale = max(0.05, min(0.98, 1.0 - thickness / avg_r))

    v_map: dict[int, int] = {}
    new_verts_list = verts.tolist()
    for vi in unique_v:
        pos = centroid + (verts[vi] - centroid) * inset_scale + avg_normal * depth
        new_idx = len(new_verts_list)
        new_verts_list.append(pos.tolist())
        v_map[int(vi)] = new_idx

    for fi in sel_idx:
        a, b, c = int(faces[fi, 0]), int(faces[fi, 1]), int(faces[fi, 2])
        faces[fi] = [v_map[a], v_map[b], v_map[c]]

    rim_faces: list[list[int]] = []
    for u, v in boundary_edges:
        u_in = v_map[u]
        v_in = v_map[v]
        rim_faces.append([u, v, v_in])
        rim_faces.append([u, v_in, u_in])

    all_faces = np.vstack([faces, np.asarray(rim_faces, dtype=np.int64)])
    return MeshData(vertices=np.asarray(new_verts_list, dtype=np.float64), faces=all_faces)


def _subdivide_once(cur: MeshData, smooth: bool) -> MeshData:
    verts = cur.vertices
    faces = cur.faces
    edge_mid: dict[tuple[int, int], int] = {}
    new_verts = verts.tolist()

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

    def get_mid(u: int, v: int) -> int:
        key = (u, v) if u < v else (v, u)
        if key in edge_mid:
            return edge_mid[key]
        if smooth and len(edge_opp[key]) == 2:
            o1, o2 = edge_opp[key]
            pos = 0.375 * (verts[u] + verts[v]) + 0.125 * (verts[o1] + verts[o2])
        else:
            pos = 0.5 * (verts[u] + verts[v])
        idx = len(new_verts)
        new_verts.append(pos.tolist())
        edge_mid[key] = idx
        return idx

    new_faces: list[list[int]] = []
    for a, b, c in faces:
        ia, ib, ic = int(a), int(b), int(c)
        ab = get_mid(ia, ib)
        bc = get_mid(ib, ic)
        ca = get_mid(ic, ia)
        new_faces.append([ia, ab, ca])
        new_faces.append([ab, ib, bc])
        new_faces.append([ca, bc, ic])
        new_faces.append([ab, bc, ca])

    out_verts = np.asarray(new_verts, dtype=np.float64)
    if smooth:
        orig_updated = verts.copy()
        for vi, nbrs in neighbors.items():
            n = len(nbrs)
            if n >= 3:
                beta = 3.0 / 16.0 if n == 3 else 3.0 / (8.0 * n)
                nbr_sum = np.sum(verts[list(nbrs)], axis=0)
                orig_updated[vi] = (1.0 - n * beta) * verts[vi] + beta * nbr_sum
        out_verts[: len(verts)] = orig_updated

    return MeshData(vertices=out_verts, faces=np.asarray(new_faces, dtype=np.int64))


def subdivide_mesh(mesh: MeshData, levels: int = 1, smooth: bool = False) -> MeshData:
    """Subdivide triangle mesh (1-to-4 midpoint split, optional Loop-like smoothing)."""
    if mesh.is_empty:
        return mesh.copy()
    cur = mesh.copy()
    for _ in range(max(1, min(3, levels))):
        cur = _subdivide_once(cur, smooth=smooth)
    return cur


def bevel_edges(
    mesh: MeshData,
    angle_threshold_deg: float = 35.0,
    width: float = 0.04,
    segments: int = 1,
) -> MeshData:
    """Bevel sharp edges while keeping volume/bounds tight and topology watertight.

    Groups coplanar/smooth triangles into patches across dihedral angles below
    `angle_threshold_deg`, insets each patch's sharp boundary vertices along its
    tangent plane, stitches chamfer quads along sharp edges and corner polygons
    at vertices, and optionally subdivides chamfer regions for multi-segment bevels.
    """
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

    # Build half-edge adjacency
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
            dot = float(np.dot(fnormals[f1], fnormals[f2]))
            if dot >= cos_thresh:
                adj_smooth[f1].append(f2)
                adj_smooth[f2].append(f1)

    # Connected components of smooth/coplanar triangles -> patches
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

    # Compute each patch's normal and boundary inward offsets
    patch_normals: list[np.ndarray] = []
    for comp in patches:
        n_sum = np.sum(fnormals[comp], axis=0)
        ln = float(np.linalg.norm(n_sum))
        patch_normals.append(n_sum / ln if ln > 1e-12 else np.array([0.0, 0.0, 1.0]))

    # Compute shortest incident edge per vertex to clamp width safely
    min_edge = np.full((len(verts),), np.inf, dtype=np.float64)
    for a, b, c in faces:
        for u, v in ((a, b), (b, c), (c, a)):
            d = float(np.linalg.norm(verts[v] - verts[u]))
            if d > 1e-9:
                min_edge[u] = min(min_edge[u], d)
                min_edge[v] = min(min_edge[v], d)
    min_edge = np.where(np.isfinite(min_edge), min_edge, width * 3.0)

    # For each patch, gather its boundary directed edges (u -> v) where neighbor is another patch
    patch_boundary_edges: list[list[tuple[int, int]]] = [[] for _ in patches]
    sharp_edge_pairs: list[tuple[int, int, int, int]] = []  # (u, v, pid1, pid2)
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

    # Build new inset vertices per (patch_id, orig_vert_idx)
    pv_map: dict[tuple[int, int], int] = {}
    new_verts: list[list[float]] = []

    for pid, comp in enumerate(patches):
        pn = patch_normals[pid]
        # Sum inward tangent directions at each boundary vertex of this patch
        inward_acc: dict[int, list[np.ndarray]] = defaultdict(list)
        for u, v in patch_boundary_edges[pid]:
            edge_vec = verts[v] - verts[u]
            e_len = float(np.linalg.norm(edge_vec))
            if e_len < 1e-9:
                continue
            edge_dir = edge_vec / e_len
            # Patch lies on the left of CCW edge u->v, so inward = pn x edge_dir
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
                    # Miter scale clamped safely
                    local_w = min(float(width), 0.38 * float(min_edge[vi]))
                    # Account for miter angle so perpendicular edge inset ~= local_w
                    cos_half = max(0.35, float(np.dot(avg_in, dirs[0])))
                    pos = verts[vi] + avg_in * (local_w / cos_half)
                else:
                    pos = verts[vi]
            pv_map[(pid, vi)] = len(new_verts)
            new_verts.append(pos.tolist())

    new_faces: list[list[int]] = []
    # 1) Original faces mapped to their inset patch vertices
    for fi, (a, b, c) in enumerate(faces):
        pid = patch_of_face[fi]
        new_faces.append([pv_map[(pid, int(a))], pv_map[(pid, int(b))], pv_map[(pid, int(c))]])

    # 2) Bevel quads along sharp edges (u->v in p1 means v->u in p2)
    for u, v, p1, p2 in sharp_edge_pairs:
        u1 = pv_map[(p1, u)]
        v1 = pv_map[(p1, v)]
        u2 = pv_map[(p2, u)]
        v2 = pv_map[(p2, v)]
        # Outward quad across p1 edge (u1->v1) to p2 edge (v2->u2):
        # In p1, face runs u1->v1; bevel must run v1->u1->u2->v2
        new_faces.append([v1, u1, u2])
        new_faces.append([v1, u2, v2])

    # 3) Corner polygons around original vertices touched by >= 3 patches
    vert_patches: dict[int, set[int]] = defaultdict(set)
    for pid, vi in pv_map:
        vert_patches[vi].add(pid)

    # Map (vi, p1, p2) adjacency through sharp edges to order corner fan cleanly
    corner_adj: dict[int, dict[int, list[int]]] = defaultdict(lambda: defaultdict(list))
    for u, v, p1, p2 in sharp_edge_pairs:
        corner_adj[u][p1].append(p2)
        corner_adj[u][p2].append(p1)
        corner_adj[v][p1].append(p2)
        corner_adj[v][p2].append(p1)

    for vi, pids in vert_patches.items():
        if len(pids) < 3:
            continue
        # Order pids around vertex vi onto plane perpendicular to average normal
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
        # Emit CCW triangle fan matching outward avg_n
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

    # Multi-segment bevel: apply gentle smooth subdivision/refinement if requested
    if segments > 1 and not out.is_empty:
        try:
            mf = out.to_manifold()
            mf = mf.smooth_out(min_sharp_angle=15.0, min_smoothness=0.35).refine(min(3, segments))
            out = MeshData.from_manifold(mf)
        except GeometryError:
            pass
    return out


def transform_selection(
    mesh: MeshData,
    query: SelectionQuery,
    translate: Vec3 = (0.0, 0.0, 0.0),
    rotate_deg: Vec3 = (0.0, 0.0, 0.0),
    scale: Vec3 = (1.0, 1.0, 1.0),
) -> MeshData:
    """Transform vertices belonging to semantically selected faces around selection pivot."""
    if mesh.is_empty:
        return mesh.copy()
    sel_mask = resolve_face_selection(mesh, query)
    if not np.any(sel_mask):
        raise GeometryError(f"Selection query ({query.mode}) matched 0 faces.")
    verts = mesh.vertices.copy()
    v_idx = np.unique(mesh.faces[sel_mask].ravel())
    pivot = np.mean(verts[v_idx], axis=0)
    mat = compose_matrix(location=translate, rotation_deg=rotate_deg, scale=scale)
    local_pts = verts[v_idx] - pivot[None, :]
    moved = transform_points(mat, local_pts) + pivot[None, :]
    verts[v_idx] = moved
    return MeshData(
        vertices=verts, faces=mesh.faces.copy(), uvs=mesh.uvs, face_groups=mesh.face_groups
    )


def smooth_laplacian(mesh: MeshData, iterations: int = 2, factor: float = 0.5) -> MeshData:
    """Apply Laplacian mesh smoothing."""
    if mesh.is_empty:
        return mesh.copy()
    verts = mesh.vertices.copy()
    neighbors: dict[int, set[int]] = defaultdict(set)
    for a, b, c in mesh.faces:
        ia, ib, ic = int(a), int(b), int(c)
        neighbors[ia].update((ib, ic))
        neighbors[ib].update((ia, ic))
        neighbors[ic].update((ia, ib))
    for _ in range(max(1, iterations)):
        nxt = verts.copy()
        for vi, nbrs in neighbors.items():
            if nbrs:
                avg = np.mean(verts[list(nbrs)], axis=0)
                nxt[vi] = verts[vi] + factor * (avg - verts[vi])
        verts = nxt
    return MeshData(
        vertices=verts, faces=mesh.faces.copy(), uvs=mesh.uvs, face_groups=mesh.face_groups
    )


def recompute_normals(mesh: MeshData) -> MeshData:
    """Fix inconsistent triangle winding and return clean mesh."""
    if mesh.is_empty:
        return mesh.copy()
    tm = mesh.to_trimesh(process=False)
    trimesh.repair.fix_normals(tm)
    return MeshData(
        vertices=np.asarray(tm.vertices, dtype=np.float64),
        faces=np.asarray(tm.faces, dtype=np.int64),
    )


def repair_manifold(mesh: MeshData) -> MeshData:
    """Merge coincident vertices, remove degenerate faces, fill small holes, fix normals."""
    if mesh.is_empty:
        return mesh.copy()
    tm = trimesh.Trimesh(vertices=mesh.vertices, faces=mesh.faces, process=True)
    tm.merge_vertices(digits_vertex=6)
    trimesh.repair.fill_holes(tm)
    trimesh.repair.fix_normals(tm)
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
