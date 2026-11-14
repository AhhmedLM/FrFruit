"""Rigid-body impulse solver and XPBD Cloth simulator for FrFruit."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.spatial import cKDTree

from frfruit.animation.timeline import FCurveTrack, Keyframe
from frfruit.core.math3d import transform_points
from frfruit.core.scene import ClothSpec, RigidBodySpec, SceneDocument, SceneObject
from frfruit.core.types import BoundingBox, Vec3
from frfruit.geometry.mesh import MeshData
from frfruit.geometry.spatial import compute_world_matrix, evaluate_object_mesh


@dataclass
class _ColliderProxy:
    obj_id: str
    shape: str
    bmin: np.ndarray
    bmax: np.ndarray
    center: np.ndarray
    half: np.ndarray
    radius: float
    surf_pts: np.ndarray
    surf_norms: np.ndarray
    tree: cKDTree | None = None


def _build_collider_proxy(
    obj_id: str,
    world_mesh: MeshData,
    bounds: BoundingBox,
    shape: str = "mesh_aabb",
) -> _ColliderProxy:
    bmin = np.asarray(bounds.min_corner, dtype=np.float64)
    bmax = np.asarray(bounds.max_corner, dtype=np.float64)
    center = 0.5 * (bmin + bmax)
    half = np.maximum(0.01, 0.5 * (bmax - bmin))
    radius = float(np.max(half))

    # Auto-detect near-spherical mesh if requested shape is default mesh_aabb or sphere
    if not world_mesh.is_empty:
        f_centers = np.mean(world_mesh.vertices[world_mesh.faces], axis=1)
        f_norms = world_mesh.face_normals()
        e01 = 0.5 * (
            world_mesh.vertices[world_mesh.faces[:, 0]]
            + world_mesh.vertices[world_mesh.faces[:, 1]]
        )
        e12 = 0.5 * (
            world_mesh.vertices[world_mesh.faces[:, 1]]
            + world_mesh.vertices[world_mesh.faces[:, 2]]
        )
        surf_pts = np.vstack([world_mesh.vertices, f_centers, e01, e12])
        vn = world_mesh.vertex_normals()
        surf_norms = np.vstack([vn, f_norms, f_norms, f_norms])
        tree = cKDTree(surf_pts)
    else:
        surf_pts = np.zeros((0, 3), dtype=np.float64)
        surf_norms = np.zeros((0, 3), dtype=np.float64)
        tree = None

    return _ColliderProxy(
        obj_id=obj_id,
        shape=shape,
        bmin=bmin,
        bmax=bmax,
        center=center,
        half=half,
        radius=radius,
        surf_pts=surf_pts,
        surf_norms=surf_norms,
        tree=tree,
    )


def _push_cloth_out_of_collider(
    pos: np.ndarray,
    vel: np.ndarray,
    col: _ColliderProxy,
    margin: float,
    friction: float = 0.35,
) -> None:
    """Push cloth vertices outside a sphere, box, or surface collider."""
    in_box = np.all((pos >= col.bmin - margin) & (pos <= col.bmax + margin), axis=1)
    if not np.any(in_box):
        return
    idx = np.nonzero(in_box)[0]
    p = pos[idx]

    if col.shape == "sphere":
        delta = p - col.center[None, :]
        dist = np.linalg.norm(delta, axis=1, keepdims=True)
        r_tot = col.radius + margin
        hit = dist[:, 0] < r_tot
        if np.any(hit):
            hi = idx[hit]
            nrm = delta[hit] / np.maximum(dist[hit], 1e-8)
            pos[hi] = col.center[None, :] + nrm * r_tot
            vn = np.sum(vel[hi] * nrm, axis=1, keepdims=True)
            v_tan = vel[hi] - nrm * vn
            vel[hi] = np.where(vn < 0.0, v_tan * (1.0 - friction), vel[hi])
        return

    if col.shape == "box" or col.tree is None:
        rel = p - col.center[None, :]
        pen = (col.half[None, :] + margin) - np.abs(rel)
        inside = np.all(pen > 0.0, axis=1)
        if np.any(inside):
            hi = idx[inside]
            pen_i = pen[inside]
            rel_i = rel[inside]
            axis = np.argmin(pen_i, axis=1)
            nrm = np.zeros_like(rel_i)
            rows = np.arange(len(hi))
            nrm[rows, axis] = np.where(rel_i[rows, axis] >= 0.0, 1.0, -1.0)
            pos[hi] += nrm * pen_i[rows, axis][:, None]
            vn = np.sum(vel[hi] * nrm, axis=1, keepdims=True)
            v_tan = vel[hi] - nrm * vn
            vel[hi] = np.where(vn < 0.0, v_tan * (1.0 - friction), vel[hi])
        return

    # Mesh surface collider via KDTree closest surface sample & signed side check
    dists, nn_idx = col.tree.query(p)
    nearest_p = col.surf_pts[nn_idx]
    nearest_n = col.surf_norms[nn_idx]
    signed = np.sum((p - nearest_p) * nearest_n, axis=1)
    # Push out vertices that lie inside collider (signed < margin) within local proximity
    need_push = (dists < max(margin * 3.0, float(np.max(col.half)) * 0.45)) & (signed < margin)
    if np.any(need_push):
        hi = idx[need_push]
        nrm = nearest_n[need_push]
        push_dist = (margin - signed[need_push])[:, None]
        pos[hi] = pos[hi] + nrm * push_dist
        vn = np.sum(vel[hi] * nrm, axis=1, keepdims=True)
        v_tan = vel[hi] - nrm * vn
        vel[hi] = np.where(vn < 0.0, v_tan * (1.0 - friction), vel[hi])


def _extract_cloth_springs(
    faces: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Extract unique structural edges and 2-hop bend springs from triangle faces."""
    if len(faces) == 0:
        empty = np.zeros((0, 2), dtype=np.int64)
        return empty, empty

    e_raw = np.vstack([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]])
    e_sorted = np.sort(e_raw, axis=1)
    struct_edges = np.unique(e_sorted, axis=0)

    # Opposite vertices sharing an edge form natural quad-diagonal / bend springs
    edge_opp: dict[tuple[int, int], list[int]] = {}
    for a, b, c in faces:
        for u, v, opp in (
            (int(a), int(b), int(c)),
            (int(b), int(c), int(a)),
            (int(c), int(a), int(b)),
        ):
            key = (min(u, v), max(u, v))
            edge_opp.setdefault(key, []).append(opp)

    bend_pairs: list[tuple[int, int]] = []
    for opps in edge_opp.values():
        if len(opps) == 2 and opps[0] != opps[1]:
            o0, o1 = min(opps[0], opps[1]), max(opps[0], opps[1])
            bend_pairs.append((o0, o1))

    bend_edges = (
        np.unique(np.asarray(bend_pairs, dtype=np.int64), axis=0)
        if bend_pairs
        else np.zeros((0, 2), dtype=np.int64)
    )
    return struct_edges, bend_edges


def _detect_pinned_vertices(verts: np.ndarray, faces: np.ndarray, cloth: ClothSpec) -> np.ndarray:
    """Resolve pinned vertex boolean mask from pin_mode (handles vertical & horizontal sheets)."""
    n = len(verts)
    pinned = np.zeros((n,), dtype=bool)
    mode = cloth.pin_mode
    if mode == "none" or n == 0:
        return pinned

    if mode == "indices":
        for idx in cloth.pinned_indices:
            if 0 <= idx < n:
                pinned[idx] = True
        return pinned

    bmin = np.min(verts, axis=0)
    bmax = np.max(verts, axis=0)
    span = bmax - bmin
    tol = max(1e-4, float(np.max(span)) * 0.025)

    # If flat horizontal sheet (Z thin, Y tall), top edge is +Y; else top edge is +Z
    top_ax = 1 if span[2] < max(span[0], span[1]) * 0.12 else 2

    if mode == "top_edge":
        pinned = verts[:, top_ax] >= (bmax[top_ax] - tol)
        return pinned

    if mode == "top_corners":
        top_mask = verts[:, top_ax] >= (bmax[top_ax] - tol)
        top_idx = np.nonzero(top_mask)[0]
        if len(top_idx) > 0:
            x_vals = verts[top_idx, 0]
            min_x = float(np.min(x_vals))
            max_x = float(np.max(x_vals))
            pinned[top_idx[np.abs(x_vals - min_x) <= tol]] = True
            pinned[top_idx[np.abs(x_vals - max_x) <= tol]] = True
        return pinned

    # mode == "border": edges appearing in exactly one triangle
    e_raw = np.vstack([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]])
    e_sorted = np.sort(e_raw, axis=1)
    uniq, counts = np.unique(e_sorted, axis=0, return_counts=True)
    boundary = uniq[counts == 1]
    if len(boundary) > 0:
        pinned[boundary.reshape(-1)] = True
    return pinned


def _solve_distance_constraints_jacobi(
    pos: np.ndarray,
    inv_mass: np.ndarray,
    edges: np.ndarray,
    rest_len: np.ndarray,
    stiffness: float,
    iterations: int = 4,
) -> None:
    """Vectorized weighted-Jacobi XPBD distance constraint projection."""
    if len(edges) == 0 or stiffness <= 1e-6:
        return
    e0 = edges[:, 0]
    e1 = edges[:, 1]
    w0 = inv_mass[e0, 0]
    w1 = inv_mass[e1, 0]
    w_sum = w0 + w1
    valid = w_sum > 1e-9
    if not np.any(valid):
        return

    e0 = e0[valid]
    e1 = e1[valid]
    rest = rest_len[valid]
    f0 = (w0[valid] / w_sum[valid])[:, None]
    f1 = (w1[valid] / w_sum[valid])[:, None]
    active0 = (w0[valid] > 0.0).astype(np.float64)[:, None]
    active1 = (w1[valid] > 0.0).astype(np.float64)[:, None]

    n_verts = len(pos)
    cnt = np.zeros((n_verts, 1), dtype=np.float64)
    np.add.at(cnt, e0, active0)
    np.add.at(cnt, e1, active1)
    cnt = np.maximum(cnt, 1.0)

    corr_sum = np.zeros_like(pos)
    gain = min(1.0, max(0.05, float(stiffness)))
    for _ in range(iterations):
        diff = pos[e1] - pos[e0]
        dist = np.linalg.norm(diff, axis=1, keepdims=True)
        safe_dist = np.where(dist < 1e-8, 1.0, dist)
        c = dist - rest[:, None]
        # Normalized edge correction vector
        grad = diff / safe_dist
        delta = grad * (c * gain)
        corr_sum.fill(0.0)
        np.add.at(corr_sum, e0, delta * f0)
        np.add.at(corr_sum, e1, -delta * f1)
        pos += corr_sum / cnt


def bake_cloth_simulation(
    scene: SceneDocument,
    object_id: str,
    frame_start: int | None = None,
    frame_end: int | None = None,
    gravity: Vec3 = (0.0, 0.0, -9.81),
    ground_z: float = 0.0,
) -> dict[str, Any]:
    """Run XPBD cloth solver on `object_id` and bake deformed vertices into `obj.vertex_cache`."""
    obj = scene.resolve_object(object_id)
    if obj is None:
        raise ValueError(f"Object {object_id!r} not found.")
    if obj.cloth is None:
        obj.cloth = ClothSpec()

    # Temporarily evaluate local mesh without stale vertex_cache override
    saved_cache = obj.vertex_cache
    obj.vertex_cache = {}
    try:
        ev = evaluate_object_mesh(scene, obj.id)
    finally:
        obj.vertex_cache = saved_cache

    local_mesh = ev.local_mesh
    if local_mesh.is_empty:
        raise ValueError(f"Cloth object {obj.name!r} has empty geometry.")

    w_mat = compute_world_matrix(scene, obj.id)
    w_inv = np.linalg.inv(w_mat)
    pos_w = transform_points(w_mat, local_mesh.vertices)
    vel_w = np.zeros_like(pos_w)

    pinned_mask = _detect_pinned_vertices(pos_w, local_mesh.faces, obj.cloth)
    inv_mass = np.where(pinned_mask[:, None], 0.0, 1.0 / max(1e-4, obj.cloth.mass_per_vertex))

    struct_edges, bend_edges = _extract_cloth_springs(local_mesh.faces)
    struct_rest = (
        np.linalg.norm(pos_w[struct_edges[:, 1]] - pos_w[struct_edges[:, 0]], axis=1)
        if len(struct_edges) > 0
        else np.zeros((0,), dtype=np.float64)
    )
    bend_rest = (
        np.linalg.norm(pos_w[bend_edges[:, 1]] - pos_w[bend_edges[:, 0]], axis=1)
        if len(bend_edges) > 0
        else np.zeros((0,), dtype=np.float64)
    )

    # Gather colliders from other visible mesh/procedural objects
    colliders: list[_ColliderProxy] = []
    for oid, other in scene.objects.items():
        if (
            oid == obj.id
            or other.kind not in ("mesh", "procedural")
            or not scene.is_object_visible(other)
        ):
            continue
        if other.cloth is not None and other.cloth.enabled:
            continue
        other_ev = evaluate_object_mesh(scene, oid)
        if not other_ev.world_mesh.is_empty:
            shape = other.rigid_body.collider_shape if other.rigid_body else "mesh_aabb"
            colliders.append(
                _build_collider_proxy(oid, other_ev.world_mesh, other_ev.bounds, shape)
            )

    fs = scene.timeline.frame_start if frame_start is None else int(frame_start)
    fe = scene.timeline.frame_end if frame_end is None else int(frame_end)
    if fe < fs:
        fe = fs
    fps = float(max(1, scene.timeline.fps))
    dt = 1.0 / fps
    substeps = max(1, int(obj.cloth.substeps))
    sub_dt = dt / substeps
    g_vec = np.asarray(gravity, dtype=np.float64)[None, :]
    wind_vec = np.asarray(obj.cloth.wind, dtype=np.float64)[None, :]
    margin = float(obj.cloth.collision_margin)
    damping_factor = max(0.0, 1.0 - float(obj.cloth.damping))

    cache: dict[int, list[list[float]]] = {}
    free_mask = ~pinned_mask

    for frame in range(fs, fe + 1):
        # Frame fs captures initial rest configuration before advancing
        if frame > fs:
            t_sim = (frame - fs) * dt
            for s in range(substeps):
                t_sub = t_sim + s * sub_dt
                prev_pos = pos_w.copy()

                # Wind gust modulation
                if np.any(free_mask):
                    if np.linalg.norm(wind_vec) > 1e-6:
                        gust = 0.78 + 0.32 * np.sin(
                            t_sub * 3.4 + pos_w[free_mask, 0:1] * 1.1 + pos_w[free_mask, 2:3] * 0.7
                        )
                        accel = g_vec + wind_vec * gust
                    else:
                        accel = g_vec

                    vel_w[free_mask] = vel_w[free_mask] * damping_factor + accel * sub_dt
                    pos_w[free_mask] += vel_w[free_mask] * sub_dt

                # Structural + bending constraints
                _solve_distance_constraints_jacobi(
                    pos_w, inv_mass, struct_edges, struct_rest, obj.cloth.stiffness, iterations=5
                )
                if len(bend_edges) > 0 and obj.cloth.bend_stiffness > 1e-4:
                    _solve_distance_constraints_jacobi(
                        pos_w,
                        inv_mass,
                        bend_edges,
                        bend_rest,
                        obj.cloth.bend_stiffness,
                        iterations=2,
                    )

                # Ground plane collision
                below_floor = free_mask & (pos_w[:, 2] < ground_z + margin)
                if np.any(below_floor):
                    pos_w[below_floor, 2] = ground_z + margin
                    vel_w[below_floor, 0] *= 0.85
                    vel_w[below_floor, 1] *= 0.85
                    if np.any(vel_w[below_floor, 2] < 0):
                        vel_w[below_floor, 2] = np.maximum(0.0, vel_w[below_floor, 2])

                # Scene object collider push-out
                for col in colliders:
                    _push_cloth_out_of_collider(pos_w, vel_w, col, margin=margin)

                # XPBD velocity update keeping pinned vertices strictly locked
                vel_w[free_mask] = (pos_w[free_mask] - prev_pos[free_mask]) / sub_dt

        local_pts = transform_points(w_inv, pos_w)
        cache[frame] = np.round(local_pts, 6).tolist()

    obj.vertex_cache = cache
    obj.cloth.enabled = True
    return {
        "object_id": obj.id,
        "object_name": obj.name,
        "frames_baked": len(cache),
        "frame_range": [fs, fe],
        "vertices": len(pos_w),
        "pinned_vertices": int(np.sum(pinned_mask)),
        "structural_springs": len(struct_edges),
        "bend_springs": len(bend_edges),
        "colliders_used": len(colliders),
    }


def _upsert_track(
    scene: SceneDocument,
    target_id: str,
    property_path: str,
    frames: list[int],
    values: list[float],
) -> None:
    scene.timeline.tracks = [
        t
        for t in scene.timeline.tracks
        if not (t.target_id == target_id and t.property_path == property_path)
    ]
    kfs = [
        Keyframe(frame=f, value=round(float(v), 5), interpolation="linear", easing="auto")
        for f, v in zip(frames, values, strict=True)
    ]
    scene.timeline.tracks.append(
        FCurveTrack(target_id=target_id, property_path=property_path, keyframes=kfs)
    )


def bake_rigid_body_simulation(
    scene: SceneDocument,
    frame_start: int | None = None,
    frame_end: int | None = None,
    gravity: Vec3 = (0.0, 0.0, -9.81),
    ground_z: float = 0.0,
) -> dict[str, Any]:
    """Simulate rigid-body dynamics and bake `location`/`rotation_deg` keyframes to Timeline."""
    fs = scene.timeline.frame_start if frame_start is None else int(frame_start)
    fe = scene.timeline.frame_end if frame_end is None else int(frame_end)
    if fe < fs:
        fe = fs
    fps = float(max(1, scene.timeline.fps))
    dt = 1.0 / fps
    substeps = 6
    sub_dt = dt / substeps
    g_vec = np.asarray(gravity, dtype=np.float64)

    @dataclass
    class _BodyState:
        obj: SceneObject
        spec: RigidBodySpec
        loc: np.ndarray
        rot_deg: np.ndarray
        vel: np.ndarray
        ang_vel_deg: np.ndarray
        half_ext: np.ndarray
        center_offset: np.ndarray
        radius: float
        inv_mass: float
        traj_loc: list[np.ndarray]
        traj_rot: list[np.ndarray]

    bodies: list[_BodyState] = []
    for oid, obj in scene.objects.items():
        if obj.rigid_body is None or not obj.rigid_body.enabled:
            continue
        if obj.kind not in ("mesh", "procedural") or not scene.is_object_visible(obj):
            continue
        ev = evaluate_object_mesh(scene, oid)
        bmin = np.asarray(ev.bounds.min_corner, dtype=np.float64)
        bmax = np.asarray(ev.bounds.max_corner, dtype=np.float64)
        half_ext = np.maximum(0.02, 0.5 * (bmax - bmin))
        w_center = 0.5 * (bmin + bmax)
        loc0 = np.asarray(obj.transform.location, dtype=np.float64)
        rot0 = np.asarray(obj.transform.rotation_deg, dtype=np.float64)
        is_dyn = obj.rigid_body.body_type == "dynamic"
        bodies.append(
            _BodyState(
                obj=obj,
                spec=obj.rigid_body,
                loc=loc0.copy(),
                rot_deg=rot0.copy(),
                vel=np.asarray(obj.rigid_body.initial_velocity, dtype=np.float64)
                if is_dyn
                else np.zeros(3),
                ang_vel_deg=np.zeros(3, dtype=np.float64),
                half_ext=half_ext,
                center_offset=w_center - loc0,
                radius=float(np.max(half_ext)),
                inv_mass=(1.0 / max(1e-4, obj.rigid_body.mass)) if is_dyn else 0.0,
                traj_loc=[],
                traj_rot=[],
            )
        )

    if not bodies:
        return {"simulated_bodies": 0, "frames_baked": 0}

    frames = list(range(fs, fe + 1))
    for frame in frames:
        for b in bodies:
            b.traj_loc.append(b.loc.copy())
            b.traj_rot.append(b.rot_deg.copy())
        if frame == fe:
            break

        for _ in range(substeps):
            for b in bodies:
                if b.inv_mass == 0.0:
                    continue
                lin_damp = max(0.0, 1.0 - b.spec.linear_damping * sub_dt * 4.0)
                ang_damp = max(0.0, 1.0 - b.spec.angular_damping * sub_dt * 4.0)
                b.vel = (b.vel + g_vec * sub_dt) * lin_damp
                b.loc = b.loc + b.vel * sub_dt
                b.ang_vel_deg = b.ang_vel_deg * ang_damp
                b.rot_deg = b.rot_deg + b.ang_vel_deg * sub_dt

                # Floor plane contact
                w_c = b.loc + b.center_offset
                bottom_z = w_c[2] - b.half_ext[2]
                if bottom_z < ground_z:
                    b.loc[2] += ground_z - bottom_z
                    if b.vel[2] < 0.0:
                        if abs(b.vel[2]) > 0.25:
                            b.vel[2] = -b.vel[2] * b.spec.restitution
                        else:
                            b.vel[2] = 0.0
                    # Coulomb friction + gentle rolling angular velocity
                    fric = max(0.0, 1.0 - b.spec.friction * sub_dt * 6.0)
                    b.vel[0] *= fric
                    b.vel[1] *= fric
                    roll_r = max(0.05, b.half_ext[2])
                    b.ang_vel_deg[0] = math.degrees(-b.vel[1] / roll_r) * 0.5
                    b.ang_vel_deg[1] = math.degrees(b.vel[0] / roll_r) * 0.5

            # Pairwise body collisions (sphere or AABB)
            for i in range(len(bodies)):
                ba = bodies[i]
                for j in range(i + 1, len(bodies)):
                    bb = bodies[j]
                    w_inv_sum = ba.inv_mass + bb.inv_mass
                    if w_inv_sum <= 1e-12:
                        continue
                    ca = ba.loc + ba.center_offset
                    cb = bb.loc + bb.center_offset

                    use_sphere = (
                        ba.spec.collider_shape == "sphere" and bb.spec.collider_shape == "sphere"
                    )
                    if use_sphere:
                        diff = cb - ca
                        dist = float(np.linalg.norm(diff))
                        min_d = ba.radius + bb.radius
                        if dist >= min_d:
                            continue
                        nrm = diff / max(dist, 1e-8) if dist > 1e-8 else np.array([0.0, 0.0, 1.0])
                        pen = min_d - dist
                    else:
                        diff = cb - ca
                        overlap = (ba.half_ext + bb.half_ext) - np.abs(diff)
                        if np.any(overlap <= 0.0):
                            continue
                        ax = int(np.argmin(overlap))
                        pen = float(overlap[ax])
                        nrm = np.zeros(3, dtype=np.float64)
                        nrm[ax] = 1.0 if diff[ax] >= 0.0 else -1.0

                    # Positional correction
                    corr = nrm * (pen / w_inv_sum)
                    ba.loc -= corr * ba.inv_mass
                    bb.loc += corr * bb.inv_mass

                    # Relative velocity impulse along contact normal
                    rv = bb.vel - ba.vel
                    vn = float(np.dot(rv, nrm))
                    if vn < 0.0:
                        e = min(ba.spec.restitution, bb.spec.restitution)
                        j_imp = -(1.0 + e) * vn / w_inv_sum
                        impulse = j_imp * nrm
                        ba.vel -= impulse * ba.inv_mass
                        bb.vel += impulse * bb.inv_mass

                        # Tangential friction impulse
                        v_tan = rv - vn * nrm
                        vt_len = float(np.linalg.norm(v_tan))
                        if vt_len > 1e-6:
                            tan_dir = v_tan / vt_len
                            mu = math.sqrt(ba.spec.friction * bb.spec.friction)
                            jt = max(-j_imp * mu, min(j_imp * mu, -vt_len / w_inv_sum))
                            f_imp = jt * tan_dir
                            ba.vel -= f_imp * ba.inv_mass
                            bb.vel += f_imp * bb.inv_mass

    # Write keyframes onto timeline tracks for dynamic bodies
    baked_ids: list[str] = []
    for b in bodies:
        if b.inv_mass == 0.0:
            continue
        baked_ids.append(b.obj.id)
        for ax_idx, ax_name in enumerate(("x", "y", "z")):
            _upsert_track(
                scene, b.obj.id, f"location.{ax_name}", frames, [p[ax_idx] for p in b.traj_loc]
            )
            _upsert_track(
                scene, b.obj.id, f"rotation_deg.{ax_name}", frames, [r[ax_idx] for r in b.traj_rot]
            )

    scene.timeline.frame_start = fs
    scene.timeline.frame_end = fe
    return {
        "simulated_bodies": len(bodies),
        "dynamic_bodies_baked": baked_ids,
        "frames_baked": len(frames),
        "frame_range": [fs, fe],
    }


def bake_scene_physics(
    scene: SceneDocument,
    frame_start: int | None = None,
    frame_end: int | None = None,
    gravity: Vec3 = (0.0, 0.0, -9.81),
    ground_z: float = 0.0,
) -> dict[str, Any]:
    """Bake all rigid bodies and cloth objects in the scene."""
    rb_res = bake_rigid_body_simulation(
        scene,
        frame_start=frame_start,
        frame_end=frame_end,
        gravity=gravity,
        ground_z=ground_z,
    )
    cloth_results: list[dict[str, Any]] = []
    for oid, obj in scene.objects.items():
        if obj.cloth is not None and obj.cloth.enabled and scene.is_object_visible(obj):
            cloth_results.append(
                bake_cloth_simulation(
                    scene,
                    oid,
                    frame_start=frame_start,
                    frame_end=frame_end,
                    gravity=gravity,
                    ground_z=ground_z,
                )
            )
    return {"rigid_bodies": rb_res, "cloth_objects": cloth_results}


def clear_physics_caches(scene: SceneDocument, object_id: str | None = None) -> dict[str, Any]:
    """Clear cloth vertex caches and/or physics animation tracks."""
    cleared_cloth: list[str] = []
    target_objs = [scene.resolve_object(object_id)] if object_id else list(scene.objects.values())
    for obj in target_objs:
        if obj is not None and obj.vertex_cache:
            obj.vertex_cache = {}
            cleared_cloth.append(obj.id)
    return {"cleared_vertex_caches": cleared_cloth}
