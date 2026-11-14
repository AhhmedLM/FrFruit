"""Skeletal rigging, forward kinematics bone evaluation, auto-skinning, and LBS deformation."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any

import numpy as np

from frfruit.core.math3d import (
    compose_matrix,
    decompose_matrix,
    euler_deg_to_matrix,
    transform_points,
)

if TYPE_CHECKING:
    from frfruit.core.scene import ArmatureSpec, BoneSpec, SceneObject, SkinWeightSpec
    from frfruit.core.types import Vec3
    from frfruit.geometry.mesh import MeshData


def _point_to_segment_distance(points: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Euclidean distance from (N, 3) points to 3D line segment [a, b]."""
    ab = b - a
    len2 = float(np.dot(ab, ab))
    if len2 < 1e-12:
        return np.linalg.norm(points - a[None, :], axis=1)
    ap = points - a[None, :]
    t = np.clip(np.sum(ap * ab[None, :], axis=1) / len2, 0.0, 1.0)
    proj = a[None, :] + t[:, None] * ab[None, :]
    return np.linalg.norm(points - proj, axis=1)


def compute_armature_matrices(
    armature: ArmatureSpec,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], dict[str, np.ndarray]]:
    """Evaluate local, pose-world (armature-space), and skinning matrices for all bones.

    Returns (local_mats, pose_mats, skin_mats) keyed by bone name.
    """
    bones_by_name: dict[str, BoneSpec] = {b.name: b for b in armature.bones}
    local_mats: dict[str, np.ndarray] = {}
    pose_mats: dict[str, np.ndarray] = {}
    bind_mats: dict[str, np.ndarray] = {}
    skin_mats: dict[str, np.ndarray] = {}

    def resolve_bone(name: str, visiting: set[str]) -> np.ndarray:
        if name in pose_mats:
            return pose_mats[name]
        if name in visiting:
            raise ValueError(f"Cyclic bone hierarchy involving {name!r}.")
        b = bones_by_name[name]
        visiting.add(name)

        head = np.asarray(b.head, dtype=np.float64)
        bind_m = np.eye(4, dtype=np.float64)
        bind_m[:3, 3] = head
        bind_mats[name] = bind_m

        if b.parent and b.parent in bones_by_name:
            parent_pose = resolve_bone(b.parent, visiting)
            p_head = np.asarray(bones_by_name[b.parent].head, dtype=np.float64)
            rest_rel = head - p_head
        else:
            parent_pose = np.eye(4, dtype=np.float64)
            rest_rel = head

        loc_t = (
            float(rest_rel[0] + b.translation[0]),
            float(rest_rel[1] + b.translation[1]),
            float(rest_rel[2] + b.translation[2]),
        )
        loc_m = compose_matrix(location=loc_t, rotation_deg=b.rotation_deg, scale=(1.0, 1.0, 1.0))
        local_mats[name] = loc_m
        pose_m = parent_pose @ loc_m
        pose_mats[name] = pose_m
        skin_mats[name] = pose_m @ np.linalg.inv(bind_m)
        visiting.remove(name)
        return pose_m

    for b in armature.bones:
        resolve_bone(b.name, set())

    return local_mats, pose_mats, skin_mats


def auto_rig_mesh(
    mesh: MeshData,
    mesh_world_mat: np.ndarray,
    armature_obj: SceneObject,
    arm_world_mat: np.ndarray,
    max_influences: int = 4,
    falloff_power: float = 2.5,
) -> SkinWeightSpec:
    """Compute normalized top-K bone skin weights mapping mesh vertices to armature bones."""
    from frfruit.core.scene import SkinWeightSpec

    if armature_obj.armature is None or not armature_obj.armature.bones:
        raise ValueError(f"Armature object {armature_obj.name!r} defines no bones.")
    bones = armature_obj.armature.bones
    if mesh.is_empty:
        return SkinWeightSpec(armature_id=armature_obj.id, weights={b.name: [] for b in bones})

    mesh_to_arm = np.linalg.inv(arm_world_mat) @ mesh_world_mat
    verts_arm = transform_points(mesh_to_arm, mesh.vertices)
    n_verts = len(verts_arm)
    n_bones = len(bones)

    dists = np.empty((n_verts, n_bones), dtype=np.float64)
    for bi, b in enumerate(bones):
        dists[:, bi] = _point_to_segment_distance(
            verts_arm,
            np.asarray(b.head, dtype=np.float64),
            np.asarray(b.tail, dtype=np.float64),
        )

    raw_w = 1.0 / np.power(dists + 0.015, falloff_power)
    k = min(max(1, int(max_influences)), n_bones)
    weights_dict: dict[str, list[tuple[int, float]]] = {b.name: [] for b in bones}

    if n_bones > k:
        top_idx = np.argpartition(-raw_w, k - 1, axis=1)[:, :k]
        mask = np.zeros_like(raw_w)
        rows = np.arange(n_verts)[:, None]
        mask[rows, top_idx] = raw_w[rows, top_idx]
    else:
        mask = raw_w

    row_sums = np.maximum(np.sum(mask, axis=1, keepdims=True), 1e-12)
    norm_w = mask / row_sums

    for bi, b in enumerate(bones):
        col = norm_w[:, bi]
        nz = np.nonzero(col > 1e-4)[0]
        weights_dict[b.name] = [(int(vi), round(float(col[vi]), 6)) for vi in nz]

    return SkinWeightSpec(armature_id=armature_obj.id, weights=weights_dict)


def deform_skinned_mesh(
    local_mesh: MeshData,
    mesh_world_mat: np.ndarray,
    skin: SkinWeightSpec,
    armature_obj: SceneObject,
    arm_world_mat: np.ndarray,
) -> MeshData:
    """Apply Linear Blend Skinning (LBS) in armature space and return deformed local MeshData."""
    if (
        local_mesh.is_empty
        or armature_obj.armature is None
        or not armature_obj.armature.bones
        or not skin.weights
    ):
        return local_mesh

    _, _, skin_mats = compute_armature_matrices(armature_obj.armature)
    mesh_to_arm = np.linalg.inv(arm_world_mat) @ mesh_world_mat
    arm_to_mesh = np.linalg.inv(mesh_to_arm)

    verts_arm = transform_points(mesh_to_arm, local_mesh.vertices)
    n_verts = len(verts_arm)
    blended = np.zeros_like(verts_arm)
    weight_sum = np.zeros((n_verts, 1), dtype=np.float64)

    ones = np.ones((n_verts, 1), dtype=np.float64)
    hom_verts = np.hstack([verts_arm, ones])

    for bone_name, pairs in skin.weights.items():
        if bone_name not in skin_mats or not pairs:
            continue
        mat = skin_mats[bone_name]
        # Skip identity matrices that contribute no deformation beyond weight_sum tracking
        v_indices = np.fromiter((p[0] for p in pairs), dtype=np.int64)
        w_vals = np.fromiter((p[1] for p in pairs), dtype=np.float64)
        valid = (v_indices >= 0) & (v_indices < n_verts) & (w_vals > 0.0)
        if not np.any(valid):
            continue
        vi = v_indices[valid]
        wv = w_vals[valid][:, None]
        transformed = (hom_verts[vi] @ mat.T)[:, :3]
        np.add.at(blended, vi, transformed * wv)
        np.add.at(weight_sum, vi, wv)

    unweighted = weight_sum[:, 0] < 1e-6
    if np.any(unweighted):
        blended[unweighted] = verts_arm[unweighted]
    weighted = ~unweighted
    if np.any(weighted):
        blended[weighted] /= weight_sum[weighted]

    deformed_local = transform_points(arm_to_mesh, blended)
    out = local_mesh.copy()
    out.vertices = deformed_local
    return out


def _rotation_between_vectors(
    src: np.ndarray,
    dst: np.ndarray,
    fallback_axis: np.ndarray | None = None,
) -> np.ndarray:
    """Compute shortest-arc 3x3 rotation matrix mapping unit vector `src` to `dst`."""
    u = src / max(1e-12, float(np.linalg.norm(src)))
    v = dst / max(1e-12, float(np.linalg.norm(dst)))
    dot = max(-1.0, min(1.0, float(np.dot(u, v))))
    if dot > 1.0 - 1e-10:
        return np.eye(3, dtype=np.float64)
    if dot < -1.0 + 1e-10:
        axis = (
            fallback_axis.copy()
            if fallback_axis is not None and float(np.linalg.norm(fallback_axis)) > 1e-8
            else np.cross(u, np.array([1.0, 0.0, 0.0], dtype=np.float64))
        )
        if float(np.linalg.norm(axis)) < 1e-8:
            axis = np.cross(u, np.array([0.0, 1.0, 0.0], dtype=np.float64))
        axis /= max(1e-12, float(np.linalg.norm(axis)))
        kx, ky, kz = float(axis[0]), float(axis[1]), float(axis[2])
        k_mat = np.array(
            [[0.0, -kz, ky], [kz, 0.0, -kx], [-ky, kx, 0.0]],
            dtype=np.float64,
        )
        return np.eye(3, dtype=np.float64) + 2.0 * (k_mat @ k_mat)

    c = np.cross(u, v)
    kx, ky, kz = float(c[0]), float(c[1]), float(c[2])
    k_mat = np.array(
        [[0.0, -kz, ky], [kz, 0.0, -kx], [-ky, kx, 0.0]],
        dtype=np.float64,
    )
    return np.eye(3, dtype=np.float64) + k_mat + (k_mat @ k_mat) * (1.0 / (1.0 + dot))


def solve_armature_ik(
    armature: ArmatureSpec,
    tip_bone: str,
    target_pos: Vec3,
    chain_length: int = 2,
    pole_target: Vec3 | None = None,
    iterations: int = 16,
    save_constraint: bool = True,
) -> dict[str, Any]:
    """Solve Inverse Kinematics for a bone chain using FABRIK + optional pole target.

    Modifies `bone.rotation_deg` in place for bones in the solved chain so the tip bone's
    tail moves toward `target_pos` (in armature space). Optionally stores/updates the
    `IKConstraintSpec` on `armature.ik_targets`.
    """
    from frfruit.core.scene import IKConstraintSpec

    bones_by_name: dict[str, BoneSpec] = {b.name: b for b in armature.bones}
    if tip_bone not in bones_by_name:
        raise KeyError(f"Bone {tip_bone!r} not found in armature.")

    clen = max(1, int(chain_length))
    chain_rev: list[BoneSpec] = []
    cur: BoneSpec | None = bones_by_name[tip_bone]
    seen: set[str] = set()
    while cur is not None and len(chain_rev) < clen and cur.name not in seen:
        chain_rev.append(cur)
        seen.add(cur.name)
        cur = bones_by_name.get(cur.parent) if cur.parent else None
    chain = list(reversed(chain_rev))
    k = len(chain)

    _, pose_mats, _ = compute_armature_matrices(armature)

    # Joint positions P[0..k] in armature space, and bone local rest vectors v_rest[0..k-1]
    pts: list[np.ndarray] = [pose_mats[b.name][:3, 3].copy() for b in chain]
    v_rest: list[np.ndarray] = []
    for i in range(k - 1):
        b_cur, b_nxt = chain[i], chain[i + 1]
        vec = (
            np.asarray(b_nxt.head, dtype=np.float64) - np.asarray(b_cur.head, dtype=np.float64)
        ) + np.asarray(b_nxt.translation, dtype=np.float64)
        v_rest.append(vec)
    tip_b = chain[-1]
    tip_vec = np.asarray(tip_b.tail, dtype=np.float64) - np.asarray(tip_b.head, dtype=np.float64)
    v_rest.append(tip_vec)
    pts.append(pts[-1] + pose_mats[tip_b.name][:3, :3] @ tip_vec)

    lengths = [float(np.linalg.norm(v)) for v in v_rest]
    total_len = sum(lengths)
    root_pos = pts[0].copy()
    target = np.asarray(target_pos, dtype=np.float64)
    to_target = target - root_pos
    dist_target = float(np.linalg.norm(to_target))

    pole_vec_np = np.asarray(pole_target, dtype=np.float64) if pole_target is not None else None

    def _compute_pole_dir(axis_dir: np.ndarray) -> np.ndarray | None:
        if pole_vec_np is None:
            return None
        pv = pole_vec_np - root_pos
        perp = pv - float(np.dot(pv, axis_dir)) * axis_dir
        nrm = float(np.linalg.norm(perp))
        if nrm > 1e-6:
            return perp / nrm
        return None

    if total_len < 1e-8:
        return {
            "chain_bones": [b.name for b in chain],
            "tip_bone": tip_bone,
            "target": [float(target[0]), float(target[1]), float(target[2])],
            "achieved_tip": [float(pts[-1][0]), float(pts[-1][1]), float(pts[-1][2])],
            "error": float(np.linalg.norm(pts[-1] - target)),
            "target_achieved": False,
            "rotations_deg": {b.name: list(b.rotation_deg) for b in chain},
        }

    if dist_target >= total_len - 1e-6:
        # Target is out of reach or exact max extension: align chain straight toward target
        d_dir = (
            to_target / max(1e-12, dist_target)
            if dist_target > 1e-9
            else np.array([0.0, 0.0, 1.0], dtype=np.float64)
        )
        pts[0] = root_pos.copy()
        for i in range(k):
            pts[i + 1] = pts[i] + d_dir * lengths[i]
    else:
        axis_dir = (
            to_target / max(1e-12, dist_target)
            if dist_target > 1e-9
            else np.array([0.0, 0.0, 1.0], dtype=np.float64)
        )
        pole_dir = _compute_pole_dir(axis_dir)

        # Break collinear singularities so 2+ bone chains bend gracefully toward pole / mid normal
        if k >= 2:
            if pole_dir is not None:
                bend_dir = pole_dir
            else:
                ref = (
                    np.array([0.0, 1.0, 0.0], dtype=np.float64)
                    if abs(float(axis_dir[1])) < 0.9
                    else np.array([1.0, 0.0, 0.0], dtype=np.float64)
                )
                bend_dir = np.cross(axis_dir, ref)
                bend_dir /= max(1e-12, float(np.linalg.norm(bend_dir)))

            # Check if intermediate joints are nearly collinear with axis_dir
            max_perp = 0.0
            for i in range(1, k):
                rel = pts[i] - root_pos
                perp = rel - float(np.dot(rel, axis_dir)) * axis_dir
                max_perp = max(max_perp, float(np.linalg.norm(perp)))
            if max_perp < 1e-4 * total_len:
                for i in range(1, k):
                    pts[i] = pts[i] + bend_dir * (0.08 * total_len * math.sin(math.pi * i / k))

        n_iters = max(1, int(iterations))
        for it in range(n_iters):
            # Project interior joints onto the pole half-plane BEFORE final FABRIK snaps
            if pole_dir is not None and k >= 2:
                for i in range(1, k):
                    rel = pts[i] - root_pos
                    t_ax = float(np.dot(rel, axis_dir))
                    rad = float(np.linalg.norm(rel - t_ax * axis_dir))
                    pts[i] = root_pos + t_ax * axis_dir + max(rad, 1e-4 * total_len) * pole_dir

            # Forward reaching pass (tip -> root)
            pts[k] = target.copy()
            for i in range(k - 1, -1, -1):
                seg = pts[i] - pts[i + 1]
                slen = float(np.linalg.norm(seg))
                if slen > 1e-12:
                    pts[i] = pts[i + 1] + (seg / slen) * lengths[i]
                else:
                    pts[i] = pts[i + 1] + np.array([0.0, 0.0, -lengths[i]], dtype=np.float64)

            # Backward reaching pass (root -> tip)
            pts[0] = root_pos.copy()
            for i in range(k):
                seg = pts[i + 1] - pts[i]
                slen = float(np.linalg.norm(seg))
                if slen > 1e-12:
                    pts[i + 1] = pts[i] + (seg / slen) * lengths[i]
                else:
                    pts[i + 1] = pts[i] + np.array([0.0, 0.0, lengths[i]], dtype=np.float64)

            if float(np.linalg.norm(pts[k] - target)) < 1e-5 and it >= 2:
                break

    # Recover local bone rotation_deg via sequential Forward Kinematics alignment
    root_parent = chain[0].parent
    parent_r = (
        pose_mats[root_parent][:3, :3].copy()
        if root_parent and root_parent in pose_mats
        else np.eye(3, dtype=np.float64)
    )
    pole_fallback = _compute_pole_dir(
        to_target / max(1e-12, dist_target)
        if dist_target > 1e-9
        else np.array([0.0, 0.0, 1.0], dtype=np.float64)
    )

    rotations_out: dict[str, list[float]] = {}
    for i, bone in enumerate(chain):
        desired_w = pts[i + 1] - pts[i]
        d_len = float(np.linalg.norm(desired_w))
        v_len = lengths[i]
        if d_len > 1e-10 and v_len > 1e-10:
            v_loc_dir = v_rest[i] / v_len
            d_loc_dir = parent_r.T @ (desired_w / d_len)
            fb_loc = parent_r.T @ pole_fallback if pole_fallback is not None else None
            r_loc = _rotation_between_vectors(v_loc_dir, d_loc_dir, fallback_axis=fb_loc)
            mat4 = np.eye(4, dtype=np.float64)
            mat4[:3, :3] = r_loc
            _, rot_deg, _ = decompose_matrix(mat4)
            bone.rotation_deg = (float(rot_deg[0]), float(rot_deg[1]), float(rot_deg[2]))
            parent_r = parent_r @ euler_deg_to_matrix(*bone.rotation_deg)
        rotations_out[bone.name] = [round(v, 4) for v in bone.rotation_deg]

    # Re-evaluate armature matrices to report exact achieved tip & residual error
    _, final_pose, _ = compute_armature_matrices(armature)
    achieved = (final_pose[tip_b.name] @ np.array([tip_vec[0], tip_vec[1], tip_vec[2], 1.0]))[:3]
    err = float(np.linalg.norm(achieved - target))

    if save_constraint:
        spec = IKConstraintSpec(
            bone_name=tip_bone,
            target=(float(target[0]), float(target[1]), float(target[2])),
            chain_length=clen,
            pole_target=pole_target,
            iterations=max(1, int(iterations)),
        )
        updated = [c for c in armature.ik_targets if c.bone_name != tip_bone]
        updated.append(spec)
        armature.ik_targets = updated

    return {
        "chain_bones": [b.name for b in chain],
        "tip_bone": tip_bone,
        "target": [
            round(float(target[0]), 5),
            round(float(target[1]), 5),
            round(float(target[2]), 5),
        ],
        "achieved_tip": [
            round(float(achieved[0]), 5),
            round(float(achieved[1]), 5),
            round(float(achieved[2]), 5),
        ],
        "error": round(err, 6),
        "target_achieved": err <= 1e-2 or dist_target >= total_len - 1e-6,
        "rotations_deg": rotations_out,
    }
