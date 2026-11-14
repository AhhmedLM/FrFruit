"""Skeletal rigging, forward kinematics bone evaluation, auto-skinning, and LBS deformation."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from frfruit.core.math3d import compose_matrix, transform_points

if TYPE_CHECKING:
    from frfruit.core.scene import ArmatureSpec, BoneSpec, SceneObject, SkinWeightSpec
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
