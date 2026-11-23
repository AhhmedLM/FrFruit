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


def _rotmat_to_quat(r: np.ndarray) -> np.ndarray:
    """Convert 3x3 proper rotation matrix to unit quaternion (w, x, y, z)."""
    tr = float(r[0, 0] + r[1, 1] + r[2, 2])
    if tr > 0.0:
        s = math.sqrt(tr + 1.0) * 2.0
        w = 0.25 * s
        x = (r[2, 1] - r[1, 2]) / s
        y = (r[0, 2] - r[2, 0]) / s
        z = (r[1, 0] - r[0, 1]) / s
    elif r[0, 0] > r[1, 1] and r[0, 0] > r[2, 2]:
        s = math.sqrt(max(1e-12, 1.0 + r[0, 0] - r[1, 1] - r[2, 2])) * 2.0
        w = (r[2, 1] - r[1, 2]) / s
        x = 0.25 * s
        y = (r[0, 1] + r[1, 0]) / s
        z = (r[0, 2] + r[2, 0]) / s
    elif r[1, 1] > r[2, 2]:
        s = math.sqrt(max(1e-12, 1.0 + r[1, 1] - r[0, 0] - r[2, 2])) * 2.0
        w = (r[0, 2] - r[2, 0]) / s
        x = (r[0, 1] + r[1, 0]) / s
        y = 0.25 * s
        z = (r[1, 2] + r[2, 1]) / s
    else:
        s = math.sqrt(max(1e-12, 1.0 + r[2, 2] - r[0, 0] - r[1, 1])) * 2.0
        w = (r[1, 0] - r[0, 1]) / s
        x = (r[0, 2] + r[2, 0]) / s
        y = (r[1, 2] + r[2, 1]) / s
        z = 0.25 * s
    q = np.array([w, x, y, z], dtype=np.float64)
    nrm = float(np.linalg.norm(q))
    return q / nrm if nrm > 1e-12 else np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)


def _matrix_to_dual_quat(mat: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Convert 4x4 rigid transform matrix into unit dual quaternion (q_real, q_dual)."""
    qr = _rotmat_to_quat(mat[:3, :3])
    tx, ty, tz = float(mat[0, 3]), float(mat[1, 3]), float(mat[2, 3])
    qw, qx, qy, qz = float(qr[0]), float(qr[1]), float(qr[2]), float(qr[3])
    qd = np.array(
        [
            -0.5 * (tx * qx + ty * qy + tz * qz),
            0.5 * (tx * qw + ty * qz - tz * qy),
            0.5 * (-tx * qz + ty * qw + tz * qx),
            0.5 * (tx * qy - ty * qx + tz * qw),
        ],
        dtype=np.float64,
    )
    return qr, qd


def deform_skinned_mesh(
    local_mesh: MeshData,
    mesh_world_mat: np.ndarray,
    skin: SkinWeightSpec,
    armature_obj: SceneObject,
    arm_world_mat: np.ndarray,
) -> MeshData:
    """Apply LBS or volume-preserving Dual Quaternion Skinning (DQS) in armature space."""
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

    if getattr(armature_obj.armature, "skinning_mode", "lbs") == "dqs":
        qr_sum = np.zeros((n_verts, 4), dtype=np.float64)
        qd_sum = np.zeros((n_verts, 4), dtype=np.float64)
        q_ref = np.zeros((n_verts, 4), dtype=np.float64)
        has_ref = np.zeros(n_verts, dtype=bool)
        weight_sum = np.zeros(n_verts, dtype=np.float64)

        for bone_name, pairs in skin.weights.items():
            if bone_name not in skin_mats or not pairs:
                continue
            qr, qd = _matrix_to_dual_quat(skin_mats[bone_name])
            v_indices = np.fromiter((p[0] for p in pairs), dtype=np.int64)
            w_vals = np.fromiter((p[1] for p in pairs), dtype=np.float64)
            valid = (v_indices >= 0) & (v_indices < n_verts) & (w_vals > 0.0)
            if not np.any(valid):
                continue
            vi = v_indices[valid]
            wv = w_vals[valid]

            new_ref = ~has_ref[vi]
            if np.any(new_ref):
                vi_new = vi[new_ref]
                q_ref[vi_new] = qr
                has_ref[vi_new] = True

            dots = q_ref[vi] @ qr
            signs = np.where(dots < 0.0, -1.0, 1.0)
            sw = (signs * wv)[:, None]
            np.add.at(qr_sum, vi, sw * qr[None, :])
            np.add.at(qd_sum, vi, sw * qd[None, :])
            np.add.at(weight_sum, vi, wv)

        blended = verts_arm.copy()
        weighted = weight_sum > 1e-6
        if np.any(weighted):
            qr_w = qr_sum[weighted]
            qd_w = qd_sum[weighted]
            norms = np.linalg.norm(qr_w, axis=1, keepdims=True)
            safe_norms = np.where(norms > 1e-12, norms, 1.0)
            qr_n = qr_w / safe_norms
            qd_n = qd_w / safe_norms

            rw = qr_n[:, 0:1]
            rv = qr_n[:, 1:4]
            dw = qd_n[:, 0:1]
            dv = qd_n[:, 1:4]

            t_vec = 2.0 * (rw * dv - dw * rv + np.cross(rv, dv))
            pts_w = verts_arm[weighted]
            u_vec = 2.0 * np.cross(rv, pts_w)
            blended[weighted] = pts_w + rw * u_vec + np.cross(rv, u_vec) + t_vec

        deformed_local = transform_points(arm_to_mesh, blended)
        out = local_mesh.copy()
        out.vertices = deformed_local
        return out

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


def _orthonormal_frame(primary: np.ndarray, secondary_hint: np.ndarray) -> np.ndarray:
    """Build right-handed 3x3 basis [e1 | e2 | e3] with e1 along primary and e2 toward hint."""
    e1 = primary / max(1e-12, float(np.linalg.norm(primary)))
    perp = secondary_hint - float(np.dot(secondary_hint, e1)) * e1
    n_perp = float(np.linalg.norm(perp))
    if n_perp < 1e-8:
        fb = (
            np.array([1.0, 0.0, 0.0], dtype=np.float64)
            if abs(float(e1[0])) < 0.85
            else np.array([0.0, 1.0, 0.0], dtype=np.float64)
        )
        perp = fb - float(np.dot(fb, e1)) * e1
        n_perp = max(1e-12, float(np.linalg.norm(perp)))
    e2 = perp / n_perp
    e3 = np.cross(e1, e2)
    e3 /= max(1e-12, float(np.linalg.norm(e3)))
    return np.column_stack([e1, e2, e3])


def _mat3_to_euler_deg(r_mat: np.ndarray) -> tuple[float, float, float]:
    m4 = np.eye(4, dtype=np.float64)
    m4[:3, :3] = r_mat
    _, rot_deg, _ = decompose_matrix(m4)
    return (float(rot_deg[0]), float(rot_deg[1]), float(rot_deg[2]))


def set_bone_world_rotation(
    armature: ArmatureSpec,
    bone_name: str,
    world_rot: np.ndarray,
) -> tuple[float, float, float]:
    """Set `bone.rotation_deg` so the bone achieves exact armature-space rotation `world_rot`."""
    bones_by_name = {b.name: b for b in armature.bones}
    bone = bones_by_name[bone_name]
    _, pose_mats, _ = compute_armature_matrices(armature)
    if bone.parent and bone.parent in pose_mats:
        parent_r = pose_mats[bone.parent][:3, :3]
    else:
        parent_r = np.eye(3, dtype=np.float64)
    r_loc = parent_r.T @ world_rot
    rot_deg = _mat3_to_euler_deg(r_loc)
    bone.rotation_deg = rot_deg
    return rot_deg


def solve_two_bone_ik(
    armature: ArmatureSpec,
    upper_bone_name: str,
    lower_bone_name: str,
    end_bone_name: str,
    target_pos: Vec3,
    pole_dir: Vec3 = (0.0, -1.0, 0.0),
    rest_bend_dir: Vec3 = (0.0, -1.0, 0.0),
    soft_ratio: float = 0.992,
    twist_deg: float = 0.0,
) -> dict[str, Any]:
    """Analytical 2-bone IK with pole-aligned twist-free frames and soft-extension knee damping.

    Positions `end_bone_name`'s head at `target_pos` (in armature space) while bending
    `lower_bone_name` strictly toward `pole_dir` without unwanted axial limb roll.
    """
    bones_by_name = {b.name: b for b in armature.bones}
    upper = bones_by_name[upper_bone_name]
    lower = bones_by_name[lower_bone_name]
    end = bones_by_name[end_bone_name]

    _, pose_mats, _ = compute_armature_matrices(armature)
    p_root = pose_mats[upper_bone_name][:3, 3].copy()
    r_parent = (
        pose_mats[upper.parent][:3, :3].copy()
        if upper.parent and upper.parent in pose_mats
        else np.eye(3, dtype=np.float64)
    )

    v1_rest = np.asarray(lower.head, dtype=np.float64) - np.asarray(upper.head, dtype=np.float64)
    v2_rest = np.asarray(end.head, dtype=np.float64) - np.asarray(lower.head, dtype=np.float64)
    l1 = float(np.linalg.norm(v1_rest))
    l2 = float(np.linalg.norm(v2_rest))
    max_reach = l1 + l2
    min_reach = abs(l1 - l2) + 0.02

    target = np.asarray(target_pos, dtype=np.float64)
    delta = target - p_root
    raw_dist = float(np.linalg.norm(delta))
    if raw_dist < 1e-8:
        axis = np.array([0.0, 0.0, -1.0], dtype=np.float64)
        dist = min_reach
    else:
        axis = delta / raw_dist
        # Soft-IK damping near full extension to eliminate knee/elbow snap-popping
        soft_start = max_reach * min(0.96, soft_ratio)
        hard_cap = max_reach * soft_ratio
        if raw_dist > soft_start:
            span = max(1e-5, hard_cap - soft_start)
            excess = raw_dist - soft_start
            dist = soft_start + span * (1.0 - math.exp(-excess / span))
        else:
            dist = raw_dist
        dist = max(min_reach, min(hard_cap, dist))

    cos_a = max(-1.0, min(1.0, (l1 * l1 + dist * dist - l2 * l2) / max(1e-12, 2.0 * l1 * dist)))
    sin_a = math.sqrt(max(0.0, 1.0 - cos_a * cos_a))

    pole_v = np.asarray(pole_dir, dtype=np.float64)
    bend = pole_v - float(np.dot(pole_v, axis)) * axis
    bn = float(np.linalg.norm(bend))
    if bn < 1e-6:
        fb = np.array([0.0, -1.0, 0.0], dtype=np.float64)
        if abs(float(np.dot(fb, axis))) > 0.9:
            fb = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        bend = fb - float(np.dot(fb, axis)) * axis
        bn = max(1e-12, float(np.linalg.norm(bend)))
    bend /= bn

    if abs(twist_deg) > 1e-6:
        ang = math.radians(twist_deg)
        ca, sa = math.cos(ang), math.sin(ang)
        bend = bend * ca + np.cross(axis, bend) * sa + axis * float(np.dot(axis, bend)) * (1.0 - ca)
        bend /= max(1e-12, float(np.linalg.norm(bend)))

    p_mid = p_root + (l1 * cos_a) * axis + (l1 * sin_a) * bend
    p_end = p_root + dist * axis

    rest_hint = np.asarray(rest_bend_dir, dtype=np.float64)
    b_rest1 = _orthonormal_frame(v1_rest, rest_hint)
    b_targ1 = _orthonormal_frame(p_mid - p_root, bend)
    r_upper_w = b_targ1 @ b_rest1.T
    upper.rotation_deg = _mat3_to_euler_deg(r_parent.T @ r_upper_w)

    b_rest2 = _orthonormal_frame(v2_rest, rest_hint)
    b_targ2 = _orthonormal_frame(p_end - p_mid, bend)
    r_lower_w = b_targ2 @ b_rest2.T
    lower.rotation_deg = _mat3_to_euler_deg(r_upper_w.T @ r_lower_w)

    return {
        "p_root": p_root.tolist(),
        "p_mid": p_mid.tolist(),
        "p_end": p_end.tolist(),
        "error": float(np.linalg.norm(p_end - target)),
        "upper_rot_deg": list(upper.rotation_deg),
        "lower_rot_deg": list(lower.rotation_deg),
    }
