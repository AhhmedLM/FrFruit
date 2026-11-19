"""High-level Quad-Morph human character builder, anatomical armature generator, and scene sync."""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any

import numpy as np

from frfruit.animation.armature import auto_rig_mesh
from frfruit.character.clothing import build_bottom_mesh, build_shoes_mesh, build_top_mesh
from frfruit.character.hair import build_hair_mesh
from frfruit.character.materials import build_character_material_bundle
from frfruit.character.parts import build_brows_and_lashes_mesh, build_eyeballs_mesh
from frfruit.character.schema import CharacterSpec
from frfruit.character.surface import AnatomicalLandmarks, CharacterSurface
from frfruit.core.scene import ArmatureSpec, BoneSpec, SceneDocument, SceneObject, SkinWeightSpec
from frfruit.core.types import Transform, Vec3
from frfruit.geometry.mesh import MeshData
from frfruit.geometry.spatial import compute_world_matrix


def _v3(arr: np.ndarray) -> Vec3:
    return (round(float(arr[0]), 5), round(float(arr[1]), 5), round(float(arr[2]), 5))


def _rigid_head_skin(mesh: MeshData, armature_obj: SceneObject) -> SkinWeightSpec:
    """Bind all vertices rigidly (1.0) to the 'head' bone so eyes/brows never shear."""
    bones = armature_obj.armature.bones if armature_obj.armature else []
    weights: dict[str, list[tuple[int, float]]] = {b.name: [] for b in bones}
    if "head" in weights and not mesh.is_empty:
        weights["head"] = [(i, 1.0) for i in range(len(mesh.vertices))]
    return SkinWeightSpec(armature_id=armature_obj.id, weights=weights)


def build_human_armature(lm: AnatomicalLandmarks) -> ArmatureSpec:
    """Build a 19-bone game-ready humanoid ArmatureSpec aligned to anatomical landmarks."""
    bones = [
        BoneSpec(name="pelvis", parent=None, head=_v3(lm.pelvis), tail=_v3(lm.waist)),
        BoneSpec(name="spine", parent="pelvis", head=_v3(lm.waist), tail=_v3(lm.chest)),
        BoneSpec(name="chest", parent="spine", head=_v3(lm.chest), tail=_v3(lm.neck_base)),
        BoneSpec(name="neck", parent="chest", head=_v3(lm.neck_base), tail=_v3(lm.neck_top)),
        BoneSpec(name="head", parent="neck", head=_v3(lm.neck_top), tail=_v3(lm.crown)),
        # Left arm
        BoneSpec(
            name="clavicle_L",
            parent="chest",
            head=_v3(lm.neck_base),
            tail=_v3(lm.shoulder_left),
        ),
        BoneSpec(
            name="upper_arm_L",
            parent="clavicle_L",
            head=_v3(lm.shoulder_left),
            tail=_v3(lm.elbow_left),
        ),
        BoneSpec(
            name="forearm_L",
            parent="upper_arm_L",
            head=_v3(lm.elbow_left),
            tail=_v3(lm.wrist_left),
        ),
        BoneSpec(
            name="hand_L",
            parent="forearm_L",
            head=_v3(lm.wrist_left),
            tail=_v3(lm.hand_tip_left),
        ),
        # Right arm
        BoneSpec(
            name="clavicle_R",
            parent="chest",
            head=_v3(lm.neck_base),
            tail=_v3(lm.shoulder_right),
        ),
        BoneSpec(
            name="upper_arm_R",
            parent="clavicle_R",
            head=_v3(lm.shoulder_right),
            tail=_v3(lm.elbow_right),
        ),
        BoneSpec(
            name="forearm_R",
            parent="upper_arm_R",
            head=_v3(lm.elbow_right),
            tail=_v3(lm.wrist_right),
        ),
        BoneSpec(
            name="hand_R",
            parent="forearm_R",
            head=_v3(lm.wrist_right),
            tail=_v3(lm.hand_tip_right),
        ),
        # Left leg
        BoneSpec(
            name="thigh_L",
            parent="pelvis",
            head=_v3(lm.hip_left),
            tail=_v3(lm.knee_left),
        ),
        BoneSpec(
            name="shin_L",
            parent="thigh_L",
            head=_v3(lm.knee_left),
            tail=_v3(lm.ankle_left),
        ),
        BoneSpec(
            name="foot_L",
            parent="shin_L",
            head=_v3(lm.ankle_left),
            tail=_v3(lm.toe_left),
        ),
        # Right leg
        BoneSpec(
            name="thigh_R",
            parent="pelvis",
            head=_v3(lm.hip_right),
            tail=_v3(lm.knee_right),
        ),
        BoneSpec(
            name="shin_R",
            parent="thigh_R",
            head=_v3(lm.knee_right),
            tail=_v3(lm.ankle_right),
        ),
        BoneSpec(
            name="foot_R",
            parent="shin_R",
            head=_v3(lm.ankle_right),
            tail=_v3(lm.toe_right),
        ),
    ]
    return ArmatureSpec(bones=bones)


def landmarks_summary(lm: AnatomicalLandmarks) -> dict[str, Any]:
    """Serialize key anatomical landmarks & anthropometric ratios into JSON-safe dict."""
    return {
        "height_m": round(float(lm.height), 4),
        "head_units": round(float(lm.head_units), 3),
        "head_height_m": round(float(lm.head_height), 4),
        "facial_thirds": [round(float(x), 4) for x in lm.facial_thirds],
        "eye_to_face_width_ratio": round(float(lm.eye_to_face_width_ratio), 4),
        "head_to_body_ratio": lm.head_to_body_ratio,
        "joints": {
            "head_center": _v3(lm.head_center),
            "crown": _v3(lm.crown),
            "chin": _v3(lm.chin),
            "nose_tip": _v3(lm.nose_tip),
            "mouth_center": _v3(lm.mouth_center),
            "eye_left": _v3(lm.eye_left),
            "eye_right": _v3(lm.eye_right),
            "neck_base": _v3(lm.neck_base),
            "shoulder_left": _v3(lm.shoulder_left),
            "shoulder_right": _v3(lm.shoulder_right),
            "pelvis": _v3(lm.pelvis),
            "knee_left": _v3(lm.knee_left),
            "ankle_left": _v3(lm.ankle_left),
        },
    }


def _sync_child_part(
    scene: SceneDocument,
    owner_obj: SceneObject,
    role: str,
    label: str,
    part_mesh: MeshData,
    material_id: str,
    arm_obj: SceneObject | None,
    arm_w: np.ndarray | None,
    *,
    rigid_head: bool = False,
) -> str | None:
    """Create, update, or remove a rigged independent child SceneObject for a character part."""
    part_id = f"{owner_obj.id}_{role}"
    if part_mesh.is_empty:
        scene.objects.pop(part_id, None)
        return None

    if part_id in scene.objects:
        child = scene.objects[part_id]
        child.name = f"{owner_obj.name}_{label}"
        child.parent_id = owner_obj.id
        child.collection_id = owner_obj.collection_id
    else:
        child = SceneObject(
            id=part_id,
            name=scene.unique_name(f"{owner_obj.name}_{label}"),
            kind="mesh",
            parent_id=owner_obj.id,
            transform=Transform(),
            collection_id=owner_obj.collection_id,
            tags=["character_part", f"char_owner:{owner_obj.id}", f"char_part:{role}"],
        )
        scene.objects[part_id] = child

    child.kind = "mesh"
    child.mesh = part_mesh.to_payload()
    child.material_id = material_id
    child.face_material_ids = {}
    for tag in ("character_part", f"char_owner:{owner_obj.id}", f"char_part:{role}"):
        if tag not in child.tags:
            child.tags.append(tag)

    if arm_obj is not None and arm_w is not None:
        if rigid_head:
            child.skin = _rigid_head_skin(part_mesh, arm_obj)
        else:
            part_w = compute_world_matrix(scene, part_id)
            child.skin = auto_rig_mesh(part_mesh, part_w, arm_obj, arm_w, max_influences=4)
    else:
        child.skin = None

    return part_id


def build_or_update_character(
    scene: SceneDocument,
    spec: CharacterSpec,
    *,
    object_id: str | None = None,
    name: str | None = None,
    location: Vec3 = (0.0, 0.0, 0.0),
    rotation_deg: Vec3 = (0.0, 0.0, 0.0),
    collection_id: str = "default",
    cache_root: Path | None = None,
) -> dict[str, Any]:
    """Synthesize or rebuild a Quad-Morph human character and its rigged child parts."""
    surface = CharacterSurface(spec, quality=spec.quality)
    lm = surface.lm
    mesh = surface.skin_mesh

    if object_id and object_id in scene.objects:
        obj = scene.objects[object_id]
        oid = obj.id
        if name:
            obj.name = name
    else:
        oid = scene.next_id("char")
        obj_name = scene.unique_name(name or f"Human_{spec.preset}")
        obj = SceneObject(
            id=oid,
            name=obj_name,
            kind="mesh",
            transform=Transform(location=location, rotation_deg=rotation_deg),
            collection_id=collection_id if collection_id in scene.collections else "default",
            tags=["character", "humanoid", "quad_morph"],
        )
        scene.objects[oid] = obj

    root = cache_root or Path(tempfile.gettempdir()) / "frfruit_chars"
    mat_bundle = build_character_material_bundle(
        scene=scene,
        obj_id=oid,
        spec=spec,
        lm=lm,
        cache_root=root,
        surface=surface,
    )
    skin_mat_id = mat_bundle["skin"]

    obj.kind = "mesh"
    obj.character = spec.model_copy(deep=True)
    obj.mesh = mesh.to_payload()
    obj.material_id = skin_mat_id
    obj.face_material_ids = {}
    if "character" not in obj.tags:
        obj.tags.append("character")

    armature_id: str | None = None
    arm_obj: SceneObject | None = None
    arm_w: np.ndarray | None = None
    if spec.auto_rig and not mesh.is_empty:
        arm_spec = build_human_armature(lm)
        existing_arm_id = obj.skin.armature_id if obj.skin else None
        if existing_arm_id and existing_arm_id in scene.objects:
            arm_obj = scene.objects[existing_arm_id]
            arm_obj.armature = arm_spec
            arm_obj.transform = obj.transform.model_copy(deep=True)
        else:
            arm_id = scene.next_id(f"{oid}_rig")
            arm_obj = SceneObject(
                id=arm_id,
                name=scene.unique_name(f"{obj.name}_Rig"),
                kind="armature",
                transform=obj.transform.model_copy(deep=True),
                collection_id=obj.collection_id,
                armature=arm_spec,
                tags=["character_rig"],
            )
            scene.objects[arm_id] = arm_obj
        armature_id = arm_obj.id
        mesh_w = compute_world_matrix(scene, oid)
        arm_w = compute_world_matrix(scene, armature_id)
        obj.skin = auto_rig_mesh(mesh, mesh_w, arm_obj, arm_w, max_influences=4)
    elif not spec.auto_rig:
        obj.skin = None

    # Build independent clean child meshes (Eyes, Brows/Lashes, Hair Groom, Clothing)
    part_specs: list[tuple[str, str, MeshData, str, bool]] = [
        ("eyes", "Eyes", build_eyeballs_mesh(surface), mat_bundle["eyes"], True),
        ("brows", "BrowsLashes", build_brows_and_lashes_mesh(surface), mat_bundle["brows"], True),
        (
            "hair",
            "Hair",
            build_hair_mesh(surface),
            mat_bundle["hair"],
            spec.materials.hair_style != "long_wavy",
        ),
        ("top", "Top", build_top_mesh(surface), mat_bundle["top"], False),
        ("bottom", "Bottom", build_bottom_mesh(surface), mat_bundle["bottom"], False),
        ("shoes", "Shoes", build_shoes_mesh(surface), mat_bundle["shoes"], False),
    ]

    child_parts: dict[str, str] = {}
    total_verts = len(mesh.vertices)
    total_faces = len(mesh.faces)
    for role, label, p_mesh, p_mat, rigid_head in part_specs:
        cid = _sync_child_part(
            scene,
            obj,
            role,
            label,
            p_mesh,
            p_mat,
            arm_obj,
            arm_w,
            rigid_head=rigid_head,
        )
        if cid is not None:
            child_parts[role] = cid
            total_verts += len(p_mesh.vertices)
            total_faces += len(p_mesh.faces)

    return {
        "status": "ok",
        "object_id": oid,
        "name": obj.name,
        "armature_id": armature_id,
        "child_parts": child_parts,
        "quality": spec.quality,
        "pose": spec.pose,
        "base_quads": len(surface.base.quads),
        "vertex_count": len(mesh.vertices),
        "face_count": len(mesh.faces),
        "total_vertex_count": total_verts,
        "total_face_count": total_faces,
        "material_id": skin_mat_id,
        "materials": mat_bundle,
        "face_material_overrides": 0,
        "rbf_node_count": len(spec.rbf_nodes),
        "stored_deltas": sorted(spec.stored_deltas.keys()),
        "stored_morphs": sorted(spec.stored_morphs.keys()),
        "expressions": spec.expressions.model_dump(),
        "custom_morphs": dict(spec.custom_morphs),
        "anthropometry": landmarks_summary(lm),
    }
