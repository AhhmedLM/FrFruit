"""Scene evaluation, spatial queries, collision checks, and AI smart placement."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from scipy.spatial import cKDTree

from frfruit.animation.armature import deform_skinned_mesh
from frfruit.core.math3d import (
    compose_matrix,
    compute_aabb,
    ray_aabb_intersect,
    ray_triangles_intersect,
    transform_points,
)
from frfruit.core.scene import SceneDocument, SceneObject
from frfruit.core.types import BoundingBox, Vec3
from frfruit.geometry.mesh import GeometryError, MeshData
from frfruit.geometry.modifiers import evaluate_modifier_stack
from frfruit.procedural.evaluator import evaluate_node_graph


@dataclass
class EvaluatedObject:
    """Evaluated geometry and world-space bounding info for a visible mesh/procedural object."""

    obj: SceneObject
    world_matrix: np.ndarray
    local_mesh: MeshData
    world_mesh: MeshData
    bounds: BoundingBox


def compute_world_matrix(
    scene: SceneDocument,
    obj_id: str,
    _visited: set[str] | None = None,
) -> np.ndarray:
    """Compute 4x4 world matrix resolving parent hierarchy."""
    obj = scene.objects.get(obj_id)
    if obj is None:
        raise GeometryError(f"Object {obj_id!r} not found.")
    visited = set(_visited or set())
    if obj_id in visited:
        raise GeometryError(f"Cyclic parent relationship involving {obj_id!r}.")
    visited.add(obj_id)

    local_m = compose_matrix(
        location=obj.transform.location,
        rotation_deg=obj.transform.rotation_deg,
        scale=obj.transform.scale,
    )
    if obj.parent_id and obj.parent_id in scene.objects:
        return compute_world_matrix(scene, obj.parent_id, visited) @ local_m
    return local_m


def _get_base_local_mesh(obj: SceneObject) -> MeshData | None:
    if obj.kind == "procedural" and obj.node_graph is not None:
        return evaluate_node_graph(obj.node_graph)
    if obj.mesh is not None:
        return MeshData.from_payload(obj.mesh)
    return None


def evaluate_object_mesh(
    scene: SceneDocument,
    obj_id: str,
    _cache: dict[str, EvaluatedObject] | None = None,
) -> EvaluatedObject:
    """Evaluate a mesh/procedural object with modifiers, vertex cache, skinning, and transform."""
    if _cache is not None and obj_id in _cache:
        return _cache[obj_id]

    obj = scene.objects.get(obj_id)
    if obj is None:
        raise GeometryError(f"Object {obj_id!r} not found.")
    base_mesh = _get_base_local_mesh(obj)
    if base_mesh is None:
        raise GeometryError(f"Object {obj.name!r} ({obj.kind}) has no geometry.")

    def world_lookup(oid: str) -> np.ndarray:
        return compute_world_matrix(scene, oid)

    def raw_lookup(oid: str) -> MeshData | None:
        target = scene.objects.get(oid)
        if target is None:
            return None
        return _get_base_local_mesh(target)

    local_eval = (
        evaluate_modifier_stack(
            obj=obj,
            base_mesh=base_mesh,
            scene=scene,
            world_matrix_lookup=world_lookup,
            raw_mesh_lookup=raw_lookup,
        )
        if obj.modifiers
        else base_mesh
    )

    # Apply baked cloth/soft-body vertex cache frame if present
    if obj.vertex_cache and not local_eval.is_empty:
        cur_f = int(scene.timeline.current_frame)
        cached_pts: np.ndarray | None = None
        if cur_f in obj.vertex_cache and len(obj.vertex_cache[cur_f]) == len(local_eval.vertices):
            cached_pts = np.asarray(obj.vertex_cache[cur_f], dtype=np.float64)
        else:
            valid_frames = sorted(
                f for f, pts in obj.vertex_cache.items() if len(pts) == len(local_eval.vertices)
            )
            if valid_frames:
                if cur_f <= valid_frames[0]:
                    cached_pts = np.asarray(obj.vertex_cache[valid_frames[0]], dtype=np.float64)
                elif cur_f >= valid_frames[-1]:
                    cached_pts = np.asarray(obj.vertex_cache[valid_frames[-1]], dtype=np.float64)
                else:
                    # Linear interpolation between bracketing baked frames
                    pos = int(np.searchsorted(valid_frames, cur_f))
                    f0 = valid_frames[pos - 1]
                    f1 = valid_frames[pos]
                    t = (cur_f - f0) / max(1, f1 - f0)
                    p0 = np.asarray(obj.vertex_cache[f0], dtype=np.float64)
                    p1 = np.asarray(obj.vertex_cache[f1], dtype=np.float64)
                    cached_pts = p0 * (1.0 - t) + p1 * t
        if cached_pts is not None:
            local_eval = local_eval.copy()
            local_eval.vertices = cached_pts

    w_mat = compute_world_matrix(scene, obj_id)

    # Apply armature Linear Blend Skinning if bound to an armature object
    if obj.skin is not None and not local_eval.is_empty:
        arm_obj = scene.resolve_object(obj.skin.armature_id)
        if arm_obj is not None and arm_obj.kind == "armature" and arm_obj.armature is not None:
            arm_mat = compute_world_matrix(scene, arm_obj.id)
            local_eval = deform_skinned_mesh(local_eval, w_mat, obj.skin, arm_obj, arm_mat)

    w_verts = transform_points(w_mat, local_eval.vertices)
    # Flip face winding when negative-scale determinant reflects handedness
    faces = local_eval.faces.copy()
    if np.linalg.det(w_mat[:3, :3]) < 0 and len(faces) > 0:
        faces = faces[:, [0, 2, 1]]
    w_mesh = MeshData(
        vertices=w_verts,
        faces=faces,
        uvs=local_eval.uvs,
        face_groups=local_eval.face_groups,
    )
    ev = EvaluatedObject(
        obj=obj,
        world_matrix=w_mat,
        local_mesh=local_eval,
        world_mesh=w_mesh,
        bounds=compute_aabb(w_verts),
    )
    if _cache is not None:
        _cache[obj_id] = ev
    return ev


def evaluate_scene_meshes(
    scene: SceneDocument,
    include_hidden: bool = False,
) -> dict[str, EvaluatedObject]:
    """Evaluate all geometry objects in the scene."""
    cache: dict[str, EvaluatedObject] = {}
    for oid, obj in scene.objects.items():
        if obj.kind not in ("mesh", "procedural"):
            continue
        if not include_hidden and not scene.is_object_visible(obj):
            continue
        if _get_base_local_mesh(obj) is None:
            continue
        evaluate_object_mesh(scene, oid, _cache=cache)
    return cache


def compute_scene_bounds(evaluated: dict[str, EvaluatedObject]) -> BoundingBox:
    """Compute combined AABB across all evaluated objects."""
    non_empty = [ev for ev in evaluated.values() if not ev.world_mesh.is_empty]
    if not non_empty:
        return BoundingBox(min_corner=(-1.0, -1.0, 0.0), max_corner=(1.0, 1.0, 1.0))
    mins = np.min([ev.bounds.min_corner for ev in non_empty], axis=0)
    maxs = np.max([ev.bounds.max_corner for ev in non_empty], axis=0)
    return BoundingBox(
        min_corner=(float(mins[0]), float(mins[1]), float(mins[2])),
        max_corner=(float(maxs[0]), float(maxs[1]), float(maxs[2])),
    )


def raycast_scene(
    evaluated: dict[str, EvaluatedObject],
    origin: Vec3,
    direction: Vec3,
    ignore_ids: set[str] | None = None,
) -> dict[str, object]:
    """Cast a ray against all evaluated scene meshes and return closest hit info."""
    orig = np.asarray(origin, dtype=np.float64)
    d = np.asarray(direction, dtype=np.float64)
    d_len = float(np.linalg.norm(d))
    if d_len < 1e-12:
        raise GeometryError("Raycast direction cannot be zero.")
    d /= d_len

    best_dist = float("inf")
    best_hit: dict[str, object] | None = None
    ignored = ignore_ids or set()

    for oid, ev in evaluated.items():
        if oid in ignored or ev.world_mesh.is_empty:
            continue
        if ray_aabb_intersect(orig, d, ev.bounds) is None:
            continue
        res = ray_triangles_intersect(orig, d, ev.world_mesh.vertices, ev.world_mesh.faces)
        if res is None:
            continue
        f_idx, dist, normal = res
        if dist < best_dist:
            best_dist = dist
            hit_pt = orig + d * dist
            best_hit = {
                "hit": True,
                "object_id": oid,
                "object_name": ev.obj.name,
                "distance": float(dist),
                "point": [float(hit_pt[0]), float(hit_pt[1]), float(hit_pt[2])],
                "normal": [float(normal[0]), float(normal[1]), float(normal[2])],
                "face_index": int(f_idx),
            }

    if best_hit is None:
        return {"hit": False}
    return best_hit


def _sample_points_for_distance(mesh: MeshData, max_pts: int = 800) -> np.ndarray:
    if mesh.is_empty:
        return np.zeros((0, 3), dtype=np.float64)
    centroids = np.mean(mesh.vertices[mesh.faces], axis=1)
    pts = np.vstack([mesh.vertices, centroids])
    if len(pts) > max_pts:
        step = max(1, len(pts) // max_pts)
        pts = pts[::step]
    return pts


def _bbox_gap(a: BoundingBox, b: BoundingBox) -> float:
    dx = max(0.0, a.min_corner[0] - b.max_corner[0], b.min_corner[0] - a.max_corner[0])
    dy = max(0.0, a.min_corner[1] - b.max_corner[1], b.min_corner[1] - a.max_corner[1])
    dz = max(0.0, a.min_corner[2] - b.max_corner[2], b.min_corner[2] - a.max_corner[2])
    return float(np.sqrt(dx * dx + dy * dy + dz * dz))


def measure_distance(ev_a: EvaluatedObject, ev_b: EvaluatedObject) -> dict[str, object]:
    """Measure center distance, AABB gap, and approximate/exact surface distance."""
    ca = np.asarray(ev_a.bounds.center, dtype=np.float64)
    cb = np.asarray(ev_b.bounds.center, dtype=np.float64)
    center_dist = float(np.linalg.norm(ca - cb))
    box_gap = _bbox_gap(ev_a.bounds, ev_b.bounds)

    # Try exact manifold min_gap and intersection first
    surface_dist: float | None = None
    intersecting = False
    try:
        ma = ev_a.world_mesh.to_manifold()
        mb = ev_b.world_mesh.to_manifold()
        inter = ma ^ mb
        if not inter.is_empty() and inter.volume() > 1e-7:
            intersecting = True
            surface_dist = 0.0
        else:
            search_len = max(1.0, box_gap + center_dist + 1.0)
            surface_dist = float(ma.min_gap(mb, search_len))
    except GeometryError:
        pass

    if surface_dist is None:
        pts_a = _sample_points_for_distance(ev_a.world_mesh)
        pts_b = _sample_points_for_distance(ev_b.world_mesh)
        if len(pts_a) > 0 and len(pts_b) > 0:
            tree = cKDTree(pts_b)
            dists, _ = tree.query(pts_a)
            surface_dist = float(np.min(dists))
        else:
            surface_dist = box_gap

    return {
        "object_a": ev_a.obj.id,
        "object_b": ev_b.obj.id,
        "center_distance": center_dist,
        "bbox_gap": box_gap,
        "surface_distance": float(surface_dist),
        "intersecting": intersecting,
    }


def check_intersections(
    evaluated: dict[str, EvaluatedObject],
    object_ids: list[str] | None = None,
    tolerance: float = 1e-3,
) -> list[dict[str, object]]:
    """Detect solid/mesh intersections across scene objects (ignoring flush touching)."""
    ids = [oid for oid in (object_ids or list(evaluated.keys())) if oid in evaluated]
    collisions: list[dict[str, object]] = []
    # Cache manifolds per object to avoid repeated conversions
    manifold_cache: dict[str, object] = {}

    def get_mf(oid: str):
        if oid not in manifold_cache:
            try:
                manifold_cache[oid] = evaluated[oid].world_mesh.to_manifold()
            except GeometryError:
                manifold_cache[oid] = None
        return manifold_cache[oid]

    for i in range(len(ids)):
        a_id = ids[i]
        ev_a = evaluated[a_id]
        if ev_a.world_mesh.is_empty:
            continue
        for j in range(i + 1, len(ids)):
            b_id = ids[j]
            ev_b = evaluated[b_id]
            if ev_b.world_mesh.is_empty:
                continue
            if not ev_a.bounds.overlaps(ev_b.bounds, tolerance=tolerance):
                continue

            ma = get_mf(a_id)
            mb = get_mf(b_id)
            if ma is not None and mb is not None:
                inter = ma ^ mb
                vol = float(inter.volume()) if not inter.is_empty() else 0.0
                if vol > 1e-6:
                    collisions.append(
                        {
                            "object_a": a_id,
                            "object_a_name": ev_a.obj.name,
                            "object_b": b_id,
                            "object_b_name": ev_b.obj.name,
                            "intersection_volume": vol,
                            "method": "exact_manifold",
                        }
                    )
            else:
                # Non-manifold fallback: strict AABB overlap beyond tolerance
                collisions.append(
                    {
                        "object_a": a_id,
                        "object_a_name": ev_a.obj.name,
                        "object_b": b_id,
                        "object_b_name": ev_b.obj.name,
                        "intersection_volume": None,
                        "method": "aabb_overlap",
                    }
                )
    return collisions


def _apply_world_delta(scene: SceneDocument, obj: SceneObject, delta_world: np.ndarray) -> Vec3:
    """Shift obj.transform.location to achieve a desired world-space translation."""
    if obj.parent_id and obj.parent_id in scene.objects:
        p_mat = compute_world_matrix(scene, obj.parent_id)
        local_delta = np.linalg.inv(p_mat[:3, :3]) @ delta_world
    else:
        local_delta = delta_world
    loc = obj.transform.location
    new_loc = (
        float(loc[0] + local_delta[0]),
        float(loc[1] + local_delta[1]),
        float(loc[2] + local_delta[2]),
    )
    obj.transform.location = new_loc
    return new_loc


def place_on_top(
    scene: SceneDocument,
    target_ev: EvaluatedObject,
    base_ev: EvaluatedObject,
    offset: float = 0.0,
    align_xy: Literal["center", "keep"] = "center",
) -> Vec3:
    """Move target so its bottom sits on base's top (+Z) with optional XY centering."""
    t_b = target_ev.bounds
    b_b = base_ev.bounds
    dz = (b_b.max_corner[2] + offset) - t_b.min_corner[2]
    if align_xy == "center":
        dx = b_b.center[0] - t_b.center[0]
        dy = b_b.center[1] - t_b.center[1]
    else:
        dx, dy = 0.0, 0.0
    return _apply_world_delta(scene, target_ev.obj, np.array([dx, dy, dz], dtype=np.float64))


def attach_to_side(
    scene: SceneDocument,
    target_ev: EvaluatedObject,
    base_ev: EvaluatedObject,
    side: Literal["+x", "-x", "+y", "-y", "+z", "-z"] = "+x",
    gap: float = 0.0,
) -> Vec3:
    """Attach target flush against a specified face of base's bounding box."""
    t_b = target_ev.bounds
    b_b = base_ev.bounds
    delta = np.array(
        [
            b_b.center[0] - t_b.center[0],
            b_b.center[1] - t_b.center[1],
            b_b.center[2] - t_b.center[2],
        ],
        dtype=np.float64,
    )
    axis = {"x": 0, "y": 1, "z": 2}[side[1]]
    if side[0] == "+":
        delta[axis] = (b_b.max_corner[axis] + gap) - t_b.min_corner[axis]
    else:
        delta[axis] = (b_b.min_corner[axis] - gap) - t_b.max_corner[axis]
    return _apply_world_delta(scene, target_ev.obj, delta)


def ground_object(scene: SceneDocument, target_ev: EvaluatedObject, z: float = 0.0) -> Vec3:
    """Move target vertically so its lowest vertex rests on z."""
    dz = z - target_ev.bounds.min_corner[2]
    return _apply_world_delta(scene, target_ev.obj, np.array([0.0, 0.0, dz], dtype=np.float64))


def snap_to_surface(
    scene: SceneDocument,
    target_ev: EvaluatedObject,
    base_ev: EvaluatedObject,
    direction: Vec3 = (0.0, 0.0, -1.0),
    gap: float = 0.0,
) -> Vec3:
    """Snap target along direction until its leading vertex touches base's triangles."""
    d = np.asarray(direction, dtype=np.float64)
    d_len = float(np.linalg.norm(d))
    if d_len < 1e-12:
        raise GeometryError("Snap direction cannot be zero.")
    d /= d_len

    # Start ray far upstream from target's center along -d toward base
    t_center = np.asarray(target_ev.bounds.center, dtype=np.float64)
    b_size = (
        float(np.linalg.norm(base_ev.bounds.size))
        + float(np.linalg.norm(target_ev.bounds.size))
        + 5.0
    )
    ray_orig = t_center - d * b_size
    res = ray_triangles_intersect(
        ray_orig, d, base_ev.world_mesh.vertices, base_ev.world_mesh.faces
    )
    if res is None:
        raise GeometryError(
            f"Snap ray from {target_ev.obj.name!r} along {direction} missed {base_ev.obj.name!r}."
        )
    _, dist_hit, _ = res
    hit_point = ray_orig + d * dist_hit

    # Find extreme vertex of target along +d (the point that touches base first)
    verts = target_ev.world_mesh.vertices
    proj = verts @ d
    lead_pt = verts[int(np.argmax(proj))]
    delta = (hit_point - d * gap) - lead_pt
    return _apply_world_delta(scene, target_ev.obj, delta)


def align_objects(
    scene: SceneDocument,
    evals: list[EvaluatedObject],
    axis: Literal["x", "y", "z"] = "x",
    mode: Literal["min", "center", "max"] = "center",
) -> dict[str, Vec3]:
    """Align multiple objects along X, Y, or Z using min/center/max bounds."""
    if len(evals) < 2:
        return {}
    ax = {"x": 0, "y": 1, "z": 2}[axis]
    anchor_ev = evals[0]
    if mode == "min":
        target_val = anchor_ev.bounds.min_corner[ax]
    elif mode == "max":
        target_val = anchor_ev.bounds.max_corner[ax]
    else:
        target_val = anchor_ev.bounds.center[ax]

    updated: dict[str, Vec3] = {}
    for ev in evals[1:]:
        cur_val = (
            ev.bounds.min_corner[ax]
            if mode == "min"
            else (ev.bounds.max_corner[ax] if mode == "max" else ev.bounds.center[ax])
        )
        delta = np.zeros(3, dtype=np.float64)
        delta[ax] = target_val - cur_val
        updated[ev.obj.id] = _apply_world_delta(scene, ev.obj, delta)
    return updated


def distribute_objects(
    scene: SceneDocument,
    evals: list[EvaluatedObject],
    axis: Literal["x", "y", "z"] = "x",
    spacing: float = 0.5,
) -> dict[str, Vec3]:
    """Distribute objects sequentially along an axis with uniform bounding-box gaps."""
    if len(evals) < 2:
        return {}
    ax = {"x": 0, "y": 1, "z": 2}[axis]
    ordered = sorted(evals, key=lambda e: e.bounds.center[ax])
    cursor = ordered[0].bounds.max_corner[ax] + spacing
    updated: dict[str, Vec3] = {}
    for ev in ordered[1:]:
        delta = np.zeros(3, dtype=np.float64)
        delta[ax] = cursor - ev.bounds.min_corner[ax]
        updated[ev.obj.id] = _apply_world_delta(scene, ev.obj, delta)
        size_ax = ev.bounds.max_corner[ax] - ev.bounds.min_corner[ax]
        cursor += size_ax + spacing
    return updated
