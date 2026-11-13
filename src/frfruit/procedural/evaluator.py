"""Deterministic DAG validation and execution engine for ProceduralGraph."""

from __future__ import annotations

import math
from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Any

import numpy as np
import trimesh

from frfruit.core.math3d import compose_matrix, transform_normals, transform_points
from frfruit.core.types import SelectionQuery
from frfruit.geometry.edit_ops import boolean_meshes, extrude_faces, subdivide_mesh
from frfruit.geometry.mesh import GeometryError, MeshData, concat_meshes
from frfruit.geometry.primitives import (
    make_cone,
    make_cube,
    make_cylinder,
    make_icosphere,
    make_sphere,
    make_torus,
)
from frfruit.procedural.nodes import NODE_CATALOG
from frfruit.procedural.schema import LinkSpec, NodeSpec, ProceduralGraph


class ProceduralError(GeometryError):
    """Raised when a procedural node graph is malformed or fails evaluation."""


@dataclass
class GeometryPayload:
    """Internal geometry carrier passed between procedural nodes."""

    mesh: MeshData | None = None
    points: np.ndarray | None = None  # (K, 3)
    normals: np.ndarray | None = None  # (K, 3)


def validate_and_sort_dag(graph: ProceduralGraph) -> list[NodeSpec]:
    """Validate node types, links, acyclic topology, and return topological order."""
    node_map: dict[str, NodeSpec] = {}
    for node in graph.nodes:
        if node.id in node_map:
            raise ProceduralError(f"Duplicate node id {node.id!r} in ProceduralGraph.")
        if node.type not in NODE_CATALOG:
            raise ProceduralError(
                f"Unknown procedural node type {node.type!r}. "
                f"Available types: {sorted(NODE_CATALOG.keys())}"
            )
        node_map[node.id] = node

    output_nodes = [n for n in graph.nodes if n.type == "GroupOutput"]
    if len(output_nodes) != 1:
        raise ProceduralError(
            f"ProceduralGraph must contain exactly 1 GroupOutput node (found {len(output_nodes)})."
        )

    in_degree: dict[str, int] = {nid: 0 for nid in node_map}
    adj: dict[str, list[str]] = defaultdict(list)
    seen_targets: set[tuple[str, str]] = set()

    for link in graph.links:
        if link.from_node not in node_map:
            raise ProceduralError(f"Link source node {link.from_node!r} not found.")
        if link.to_node not in node_map:
            raise ProceduralError(f"Link target node {link.to_node!r} not found.")
        if link.from_node == link.to_node:
            raise ProceduralError(f"Self-loop detected on node {link.from_node!r}.")

        src_def = NODE_CATALOG[node_map[link.from_node].type]
        dst_def = NODE_CATALOG[node_map[link.to_node].type]
        valid_out = {s.name for s in src_def.outputs}
        valid_in = {s.name for s in dst_def.inputs}
        if link.from_socket not in valid_out:
            raise ProceduralError(
                f"Node {link.from_node!r} ({src_def.type}) has no output socket "
                f"{link.from_socket!r}. Available: {sorted(valid_out)}"
            )
        if link.to_socket not in valid_in:
            raise ProceduralError(
                f"Node {link.to_node!r} ({dst_def.type}) has no input socket "
                f"{link.to_socket!r}. Available: {sorted(valid_in)}"
            )
        target_key = (link.to_node, link.to_socket)
        if target_key in seen_targets:
            raise ProceduralError(
                f"Input socket {link.to_node}.{link.to_socket} has multiple incoming links."
            )
        seen_targets.add(target_key)
        adj[link.from_node].append(link.to_node)
        in_degree[link.to_node] += 1

    # Check required inputs on reachable nodes / all nodes
    for nid, node in node_map.items():
        ndef = NODE_CATALOG[node.type]
        for sdef in ndef.inputs:
            if sdef.required and (nid, sdef.name) not in seen_targets:
                raise ProceduralError(
                    f"Node {nid!r} ({node.type}) is missing required input {sdef.name!r}."
                )

    queue: deque[str] = deque(nid for nid, deg in in_degree.items() if deg == 0)
    ordered: list[NodeSpec] = []
    while queue:
        cur = queue.popleft()
        ordered.append(node_map[cur])
        for nxt in adj[cur]:
            in_degree[nxt] -= 1
            if in_degree[nxt] == 0:
                queue.append(nxt)

    if len(ordered) != len(node_map):
        raise ProceduralError("Cycle detected in ProceduralGraph links.")
    return ordered


def _rotation_aligning_z_to(target_dir: np.ndarray) -> np.ndarray:
    """Return 3x3 rotation matrix mapping +Z to target_dir."""
    z = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    t = np.asarray(target_dir, dtype=np.float64)
    ln = float(np.linalg.norm(t))
    if ln < 1e-12:
        return np.eye(3, dtype=np.float64)
    t /= ln
    dot = float(np.dot(z, t))
    if dot > 0.999999:
        return np.eye(3, dtype=np.float64)
    if dot < -0.999999:
        return np.diag([1.0, -1.0, -1.0])
    v = np.cross(z, t)
    s = float(np.linalg.norm(v))
    vx = np.array([[0.0, -v[2], v[1]], [v[2], 0.0, -v[0]], [-v[1], v[0], 0.0]], dtype=np.float64)
    return np.eye(3, dtype=np.float64) + vx + (vx @ vx) * ((1.0 - dot) / (s * s))


def _param(node: NodeSpec, key: str) -> Any:
    if key in node.params:
        return node.params[key]
    return NODE_CATALOG[node.type].default_params.get(key)


def evaluate_node_graph(graph: ProceduralGraph) -> MeshData:
    """Execute a validated ProceduralGraph deterministically and return MeshData."""
    ordered = validate_and_sort_dag(graph)
    links_by_target: dict[tuple[str, str], LinkSpec] = {
        (lnk.to_node, lnk.to_socket): lnk for lnk in graph.links
    }
    results: dict[tuple[str, str], GeometryPayload] = {}
    base_seed = int(graph.seed)

    def get_in(node_id: str, socket: str) -> GeometryPayload | None:
        lnk = links_by_target.get((node_id, socket))
        if lnk is None:
            return None
        return results.get((lnk.from_node, lnk.from_socket))

    output_payload: GeometryPayload | None = None

    for idx, node in enumerate(ordered):
        nt = node.type
        node_seed = int(
            _param(node, "seed") if _param(node, "seed") is not None else base_seed + idx * 131
        )

        if nt == "PrimitiveCube":
            sz = _param(node, "size")
            size_v = tuple(float(x) for x in sz) if isinstance(sz, (list, tuple)) else float(sz)
            m = make_cube(size=size_v, origin=_param(node, "origin"))
            results[(node.id, "geometry")] = GeometryPayload(mesh=m)

        elif nt == "PrimitiveSphere":
            m = make_sphere(
                radius=float(_param(node, "radius")), segments=int(_param(node, "segments"))
            )
            results[(node.id, "geometry")] = GeometryPayload(mesh=m)

        elif nt == "PrimitiveCylinder":
            rt = _param(node, "radius_top")
            m = make_cylinder(
                radius=float(_param(node, "radius")),
                height=float(_param(node, "height")),
                radius_top=None if rt is None else float(rt),
                segments=int(_param(node, "segments")),
            )
            results[(node.id, "geometry")] = GeometryPayload(mesh=m)

        elif nt == "PrimitiveCone":
            m = make_cone(
                radius=float(_param(node, "radius")),
                height=float(_param(node, "height")),
                segments=int(_param(node, "segments")),
            )
            results[(node.id, "geometry")] = GeometryPayload(mesh=m)

        elif nt == "PrimitiveTorus":
            m = make_torus(
                major_radius=float(_param(node, "major_radius")),
                minor_radius=float(_param(node, "minor_radius")),
                major_segments=int(_param(node, "major_segments")),
                minor_segments=int(_param(node, "minor_segments")),
            )
            results[(node.id, "geometry")] = GeometryPayload(mesh=m)

        elif nt == "GridPoints":
            sx = float(_param(node, "size_x"))
            sy = float(_param(node, "size_y"))
            cx = max(1, int(_param(node, "count_x")))
            cy = max(1, int(_param(node, "count_y")))
            xs = np.linspace(-0.5 * sx, 0.5 * sx, cx) if cx > 1 else np.array([0.0])
            ys = np.linspace(-0.5 * sy, 0.5 * sy, cy) if cy > 1 else np.array([0.0])
            gx, gy = np.meshgrid(xs, ys, indexing="xy")
            pts = np.column_stack([gx.ravel(), gy.ravel(), np.zeros(gx.size, dtype=np.float64)])
            norms = np.tile(np.array([[0.0, 0.0, 1.0]]), (len(pts), 1))
            results[(node.id, "geometry")] = GeometryPayload(points=pts, normals=norms)

        elif nt == "CurveCircle":
            r = float(_param(node, "radius"))
            seg = max(3, int(_param(node, "segments")))
            ang = np.linspace(0.0, 2.0 * math.pi, seg, endpoint=False)
            pts = np.column_stack([r * np.cos(ang), r * np.sin(ang), np.zeros(seg)])
            norms = np.column_stack([np.cos(ang), np.sin(ang), np.zeros(seg)])
            results[(node.id, "geometry")] = GeometryPayload(points=pts, normals=norms)

        elif nt == "CurveLine":
            start = np.asarray(_param(node, "start"), dtype=np.float64)
            end = np.asarray(_param(node, "end"), dtype=np.float64)
            cnt = max(2, int(_param(node, "count")))
            ts = np.linspace(0.0, 1.0, cnt)[:, None]
            pts = start[None, :] * (1.0 - ts) + end[None, :] * ts
            d = end - start
            ln = float(np.linalg.norm(d))
            tangent = d / ln if ln > 1e-12 else np.array([0.0, 0.0, 1.0])
            norms = np.tile(tangent[None, :], (cnt, 1))
            results[(node.id, "geometry")] = GeometryPayload(points=pts, normals=norms)

        elif nt == "TransformGeometry":
            src = get_in(node.id, "geometry")
            assert src is not None
            tr = tuple(float(x) for x in _param(node, "translation"))
            rot = tuple(float(x) for x in _param(node, "rotation_deg"))
            sc = tuple(float(x) for x in _param(node, "scale"))
            mat = compose_matrix(
                location=(tr[0], tr[1], tr[2]),
                rotation_deg=(rot[0], rot[1], rot[2]),
                scale=(sc[0], sc[1], sc[2]),
            )
            out_mesh = None
            if src.mesh is not None and not src.mesh.is_empty:
                out_mesh = MeshData(
                    vertices=transform_points(mat, src.mesh.vertices),
                    faces=src.mesh.faces.copy(),
                    uvs=src.mesh.uvs,
                    face_groups=src.mesh.face_groups,
                )
            out_pts = transform_points(mat, src.points) if src.points is not None else None
            out_nrm = transform_normals(mat, src.normals) if src.normals is not None else None
            results[(node.id, "geometry")] = GeometryPayload(
                mesh=out_mesh, points=out_pts, normals=out_nrm
            )

        elif nt == "JoinGeometry":
            meshes: list[MeshData] = []
            pts_list: list[np.ndarray] = []
            nrm_list: list[np.ndarray] = []
            for sname in ("geometry_a", "geometry_b", "geometry_c", "geometry_d"):
                g = get_in(node.id, sname)
                if g is None:
                    continue
                if g.mesh is not None and not g.mesh.is_empty:
                    meshes.append(g.mesh)
                if g.points is not None and len(g.points) > 0:
                    pts_list.append(g.points)
                    if g.normals is not None and len(g.normals) == len(g.points):
                        nrm_list.append(g.normals)
                    else:
                        nrm_list.append(np.tile(np.array([[0.0, 0.0, 1.0]]), (len(g.points), 1)))
            joined_mesh = concat_meshes(meshes) if meshes else None
            joined_pts = np.vstack(pts_list) if pts_list else None
            joined_nrm = np.vstack(nrm_list) if nrm_list else None
            results[(node.id, "geometry")] = GeometryPayload(
                mesh=joined_mesh, points=joined_pts, normals=joined_nrm
            )

        elif nt == "BooleanGeometry":
            ga = get_in(node.id, "geometry_a")
            gb = get_in(node.id, "geometry_b")
            if ga is None or ga.mesh is None or gb is None or gb.mesh is None:
                raise ProceduralError("BooleanGeometry requires two mesh geometry inputs.")
            m = boolean_meshes(ga.mesh, gb.mesh, str(_param(node, "operation")))
            results[(node.id, "geometry")] = GeometryPayload(mesh=m)

        elif nt == "SubdivideGeometry":
            src = get_in(node.id, "geometry")
            if src is None or src.mesh is None:
                raise ProceduralError("SubdivideGeometry requires a mesh input.")
            m = subdivide_mesh(
                src.mesh, levels=int(_param(node, "levels")), smooth=bool(_param(node, "smooth"))
            )
            results[(node.id, "geometry")] = GeometryPayload(mesh=m)

        elif nt == "ConvexHull":
            src = get_in(node.id, "geometry")
            if src is None:
                raise ProceduralError("ConvexHull requires geometry input.")
            pts = (
                src.mesh.vertices
                if src.mesh is not None and not src.mesh.is_empty
                else (src.points if src.points is not None else np.zeros((0, 3)))
            )
            if len(pts) < 4:
                raise ProceduralError("ConvexHull requires at least 4 non-coplanar points.")
            tm = trimesh.convex.convex_hull(pts)
            results[(node.id, "geometry")] = GeometryPayload(mesh=MeshData.from_trimesh(tm))

        elif nt == "ExtrudeGeometry":
            src = get_in(node.id, "geometry")
            if src is None or src.mesh is None:
                raise ProceduralError("ExtrudeGeometry requires a mesh input.")
            q = SelectionQuery(
                mode=str(_param(node, "mode")),
                angle_threshold_deg=float(_param(node, "angle_threshold_deg") or 45.0),
            )
            m = extrude_faces(
                src.mesh,
                q,
                distance=float(_param(node, "distance")),
                scale=float(_param(node, "scale")),
            )
            results[(node.id, "geometry")] = GeometryPayload(mesh=m)

        elif nt == "SetPositionNoise":
            src = get_in(node.id, "geometry")
            if src is None or src.mesh is None:
                raise ProceduralError("SetPositionNoise requires a mesh input.")
            m = src.mesh.copy()
            strength = float(_param(node, "strength"))
            scale = float(_param(node, "scale"))
            along_n = bool(_param(node, "along_normals"))
            rng = np.random.default_rng(node_seed)
            ph = rng.uniform(0.0, 2.0 * math.pi, size=(3,))
            v = m.vertices
            freq = scale * math.pi
            nval = (
                np.sin(v[:, 0] * freq + ph[0])
                * np.cos(v[:, 1] * freq + ph[1])
                * np.sin(v[:, 2] * freq + ph[2] + 0.5)
            )
            if along_n:
                vn = m.vertex_normals()
                m.vertices = v + vn * (nval[:, None] * strength)
            else:
                m.vertices = v + np.column_stack([nval, nval * 0.7, nval * 1.1]) * strength
            results[(node.id, "geometry")] = GeometryPayload(mesh=m)

        elif nt == "DistributePointsOnFaces":
            src = get_in(node.id, "geometry")
            if src is None or src.mesh is None or src.mesh.is_empty:
                raise ProceduralError("DistributePointsOnFaces requires a non-empty mesh input.")
            count = max(1, int(_param(node, "count")))
            min_dist = float(_param(node, "min_distance") or 0.0)
            tm = src.mesh.to_trimesh(process=False)
            # Area-weighted deterministic sampling
            oversample = count * 4 if min_dist > 0 else count
            pts, face_idx = trimesh.sample.sample_surface(
                tm, oversample, seed=int(node_seed & 0xFFFFFFFF)
            )
            fn = src.mesh.face_normals()[face_idx]
            if min_dist > 0:
                kept_pts: list[np.ndarray] = []
                kept_nrm: list[np.ndarray] = []
                md2 = min_dist * min_dist
                for p, n in zip(pts, fn, strict=True):
                    if not kept_pts or all(float(np.sum((p - kp) ** 2)) >= md2 for kp in kept_pts):
                        kept_pts.append(p)
                        kept_nrm.append(n)
                        if len(kept_pts) >= count:
                            break
                pts = np.asarray(kept_pts, dtype=np.float64)
                fn = np.asarray(kept_nrm, dtype=np.float64)
            results[(node.id, "geometry")] = GeometryPayload(points=pts, normals=fn)

        elif nt == "InstanceOnPoints":
            pts_in = get_in(node.id, "points")
            inst_in = get_in(node.id, "instance")
            if pts_in is None or inst_in is None or inst_in.mesh is None:
                raise ProceduralError(
                    "InstanceOnPoints requires both points and instance mesh inputs."
                )
            if pts_in.points is not None and len(pts_in.points) > 0:
                pts = pts_in.points
                nrms = (
                    pts_in.normals
                    if pts_in.normals is not None and len(pts_in.normals) == len(pts)
                    else np.tile(np.array([[0.0, 0.0, 1.0]]), (len(pts), 1))
                )
            elif pts_in.mesh is not None and not pts_in.mesh.is_empty:
                pts = pts_in.mesh.vertices
                nrms = pts_in.mesh.vertex_normals()
            else:
                raise ProceduralError("InstanceOnPoints points input contains no points.")

            rng = np.random.default_rng(node_seed)
            base_scale = float(_param(node, "scale"))
            s_min = float(_param(node, "scale_min"))
            s_max = float(_param(node, "scale_max"))
            align = bool(_param(node, "align_to_normal"))
            rand_rot = _param(node, "random_rotation_deg") or [0.0, 0.0, 0.0]

            inst_mesh = inst_in.mesh
            copies: list[MeshData] = []
            for p, n in zip(pts, nrms, strict=True):
                sc = base_scale * float(rng.uniform(s_min, s_max))
                rx = float(rng.uniform(-0.5 * rand_rot[0], 0.5 * rand_rot[0]))
                ry = float(rng.uniform(-0.5 * rand_rot[1], 0.5 * rand_rot[1]))
                rz = float(rng.uniform(-0.5 * rand_rot[2], 0.5 * rand_rot[2]))
                local_mat = compose_matrix(rotation_deg=(rx, ry, rz), scale=(sc, sc, sc))
                v = transform_points(local_mat, inst_mesh.vertices)
                if align:
                    r_align = _rotation_aligning_z_to(n)
                    v = v @ r_align.T
                v = v + p[None, :]
                copies.append(MeshData(vertices=v, faces=inst_mesh.faces.copy()))
            results[(node.id, "geometry")] = GeometryPayload(mesh=concat_meshes(copies))

        elif nt == "GroupOutput":
            output_payload = get_in(node.id, "geometry")

    if output_payload is None:
        raise ProceduralError("GroupOutput received no geometry.")
    if output_payload.mesh is not None and not output_payload.mesh.is_empty:
        return output_payload.mesh
    if output_payload.points is not None and len(output_payload.points) > 0:
        # Visualize raw point cloud as tiny icosphere markers if user outputs raw points
        marker = make_icosphere(radius=0.04, subdivisions=1)
        copies = [
            MeshData(vertices=marker.vertices + p[None, :], faces=marker.faces.copy())
            for p in output_payload.points
        ]
        return concat_meshes(copies)
    raise ProceduralError("ProceduralGraph produced empty geometry.")
