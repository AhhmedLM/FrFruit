"""Tests for skeletal armatures, auto-skinning, LBS deformation, and Game-Ready Binary FBX 7.4."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from frfruit.animation.armature import auto_rig_mesh
from frfruit.animation.evaluator import apply_animation_frame
from frfruit.animation.timeline import FCurveTrack, Keyframe
from frfruit.core.scene import (
    ArmatureSpec,
    BoneSpec,
    ClothSpec,
    MaterialSpec,
    SceneDocument,
    SceneObject,
)
from frfruit.core.types import Transform
from frfruit.geometry.primitives import create_primitive_mesh
from frfruit.geometry.spatial import compute_world_matrix, evaluate_object_mesh
from frfruit.io.fbx_exporter import export_fbx_scene, verify_fbx_binary
from frfruit.physics.solver import bake_scene_physics
from frfruit.textures.fetcher import generate_procedural_pbr_maps


def _build_rigged_arm_scene(tmp_path: Path) -> SceneDocument:
    sc = SceneDocument.new_default("RiggedHero")
    sc.timeline.fps = 24
    sc.timeline.frame_start = 1
    sc.timeline.frame_end = 10

    # Vertical cylinder column along Z from 0 to 2.0
    limb = create_primitive_mesh("cylinder", radius=0.22, radius_top=0.18, height=2.0, segments=12)
    tmap = generate_procedural_pbr_maps("gold", out_dir=tmp_path / "tex", size=64)
    sc.materials["mat_skin"] = MaterialSpec(
        id="mat_skin",
        name="ArmMetal",
        base_color=(0.8, 0.72, 0.45, 1.0),
        metallic=0.75,
        roughness=0.25,
        albedo_map=tmap["maps"]["albedo"],
        normal_map=tmap["maps"]["normal"],
    )

    arm_obj = SceneObject(
        id="arm_main",
        name="ArmRig",
        kind="armature",
        transform=Transform(location=(0.0, 0.0, 0.0)),
        armature=ArmatureSpec(
            bones=[
                BoneSpec(name="lower", parent=None, head=(0.0, 0.0, 0.0), tail=(0.0, 0.0, 1.0)),
                BoneSpec(name="upper", parent="lower", head=(0.0, 0.0, 1.0), tail=(0.0, 0.0, 2.0)),
            ]
        ),
    )
    sc.objects["arm_main"] = arm_obj

    mesh_obj = SceneObject(
        id="mesh_limb",
        name="ArmMesh",
        kind="mesh",
        transform=Transform(location=(0.0, 0.0, 1.0)),
        mesh=limb.to_payload(),
        material_id="mat_skin",
    )
    sc.objects["mesh_limb"] = mesh_obj

    ev_rest = evaluate_object_mesh(sc, "mesh_limb")
    skin = auto_rig_mesh(
        ev_rest.local_mesh,
        ev_rest.world_matrix,
        arm_obj,
        compute_world_matrix(sc, "arm_main"),
        max_influences=4,
    )
    mesh_obj.skin = skin

    # Keyframe upper bone bending 90 deg around X at frame 10
    sc.timeline.tracks.append(
        FCurveTrack(
            target_id="arm_main",
            property_path="bone.upper.rotation_deg.x",
            keyframes=[Keyframe(frame=1, value=0.0), Keyframe(frame=10, value=90.0)],
        )
    )
    return sc


def test_auto_rig_weights_and_lbs_bone_deformation(tmp_path: Path) -> None:
    sc = _build_rigged_arm_scene(tmp_path)
    skin = sc.objects["mesh_limb"].skin
    assert skin is not None

    # Verify normalized vertex weights & max 4 influences per vertex
    ev_rest = evaluate_object_mesh(sc.model_copy(deep=True).model_copy(update={}), "mesh_limb")
    n_verts = len(ev_rest.local_mesh.vertices)
    vert_totals = np.zeros(n_verts)
    vert_counts = np.zeros(n_verts, dtype=int)
    for pairs in skin.weights.values():
        for vi, wt in pairs:
            vert_totals[vi] += wt
            vert_counts[vi] += 1
    assert np.allclose(vert_totals, 1.0, atol=1e-3)
    assert int(vert_counts.max()) <= 4

    # Evaluate at frame 10 (upper bone bent 90 deg) -> top tip bends sideways in Y/Z!
    f1 = apply_animation_frame(sc, 1)
    f10 = apply_animation_frame(sc, 10)
    ev1 = evaluate_object_mesh(f1, "mesh_limb")
    ev10 = evaluate_object_mesh(f10, "mesh_limb")
    assert float(ev10.bounds.max_corner[2]) < float(ev1.bounds.max_corner[2]) - 0.3
    assert float(np.abs(ev10.world_mesh.vertices[:, 1]).max()) > 0.6


def test_binary_fbx_and_ascii_export_rig_cloth_morphs_and_engine_presets(tmp_path: Path) -> None:
    sc = _build_rigged_arm_scene(tmp_path)

    # Also add a tiny cloth banner so BlendShape morph targets & DeformPercent curves get emitted
    banner = create_primitive_mesh("plane", size=1.0, segments=4)
    sc.objects["cape"] = SceneObject(
        id="cape",
        name="CapeCloth",
        kind="mesh",
        transform=Transform(location=(1.5, 0.0, 1.5)),
        mesh=banner.to_payload(),
        cloth=ClothSpec(enabled=True, pin_mode="top_edge", wind=(1.5, 0.0, 0.0), substeps=4),
    )
    bake_scene_physics(sc, frame_start=1, frame_end=6)

    # 1. Unreal Engine Binary FBX 7.4
    fbx_ue = tmp_path / "hero_unreal.fbx"
    res_ue = export_fbx_scene(sc, fbx_ue, ascii_mode=False, target_engine="unreal")
    assert res_ue["format"] == "fbx"
    assert res_ue["target_engine"] == "unreal"
    assert res_ue["counts"]["Deformer"] >= 3  # Skin + 2 Clusters + BlendShape + channels
    assert res_ue["counts"]["Pose"] >= 1
    assert res_ue["counts"]["AnimationStack"] == 1
    assert res_ue["counts"]["AnimationCurve"] >= 4
    assert res_ue["counts"]["Texture"] >= 2

    check = verify_fbx_binary(fbx_ue)
    assert check["valid"] is True
    assert check["version"] == 7400
    assert "Objects" in check["top_level_nodes"]
    assert "Connections" in check["top_level_nodes"]

    # 2. ASCII Unity FBX preset checks Y-up axis + readable node tags
    fbx_unity = tmp_path / "hero_unity_ascii.fbx"
    res_un = export_fbx_scene(sc, fbx_unity, ascii_mode=True, target_engine="unity")
    assert res_un["format"] == "fbx_ascii"
    txt = fbx_unity.read_text(encoding="utf-8")
    for marker in (
        '"UpAxis", "int", "Integer", "", 1',
        "LayerElementNormal",
        "LayerElementTangent",
        "LayerElementBinormal",
        "LayerElementUV",
        "LimbNode",
        "BindPose",
        "BlendShape",
        "AnimationStack",
        "AnimationCurve",
    ):
        assert marker in txt, f"Missing expected FBX node/marker: {marker}"


def test_fbx_per_face_multi_material_bypolygon(tmp_path: Path) -> None:
    sc = SceneDocument.new_default("FbxFaceMat")
    sc.materials["mat_base"] = MaterialSpec(
        id="mat_base", name="BaseSlate", base_color=(0.2, 0.22, 0.26, 1.0)
    )
    sc.materials["mat_gold"] = MaterialSpec(
        id="mat_gold", name="AccentGold", base_color=(0.95, 0.78, 0.25, 1.0), metallic=1.0
    )
    cube = create_primitive_mesh("cube", size=1.0)
    sc.objects["box"] = SceneObject(
        id="box",
        name="MultiMatBox",
        kind="mesh",
        material_id="mat_base",
        face_material_ids={0: "mat_gold", 1: "mat_gold"},
        mesh=cube.to_payload(),
    )

    fbx_ascii = tmp_path / "multimat_ascii.fbx"
    export_fbx_scene(sc, fbx_ascii, ascii_mode=True, target_engine="unreal")
    txt = fbx_ascii.read_text(encoding="utf-8")
    assert '"MappingInformationType", "S", "ByPolygon"' in txt or '"ByPolygon"' in txt
    assert "BaseSlate" in txt
    assert "AccentGold" in txt

    fbx_bin = tmp_path / "multimat_bin.fbx"
    export_fbx_scene(sc, fbx_bin, ascii_mode=False, target_engine="unreal")
    chk = verify_fbx_binary(fbx_bin)
    assert chk["valid"] is True
