"""Tests for Binary FBX import, Dual Quaternion Skinning, 2-bone IK, and AAA locomotion."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from frfruit.animation.armature import (
    auto_rig_mesh,
    compute_armature_matrices,
    deform_skinned_mesh,
    solve_two_bone_ik,
)
from frfruit.animation.locomotion import (
    bake_locomotion_to_timeline,
    build_showcase_schedule,
    detect_humanoid_rig,
    evaluate_locomotion_pose,
)
from frfruit.core.scene import ArmatureSpec, BoneSpec, SceneDocument, SceneObject
from frfruit.core.types import Transform
from frfruit.geometry.primitives import make_cylinder
from frfruit.io.fbx_exporter import export_fbx_scene
from frfruit.io.fbx_importer import import_fbx_file


def _make_humanoid_armature() -> ArmatureSpec:
    bones = [
        BoneSpec(name="mixamorig:Hips", parent=None, head=(0.0, 0.0, 1.0), tail=(0.0, 0.0, 1.1)),
        BoneSpec(
            name="mixamorig:Spine",
            parent="mixamorig:Hips",
            head=(0.0, 0.0, 1.1),
            tail=(0.0, 0.0, 1.22),
        ),
        BoneSpec(
            name="mixamorig:Spine1",
            parent="mixamorig:Spine",
            head=(0.0, 0.0, 1.22),
            tail=(0.0, 0.0, 1.35),
        ),
        BoneSpec(
            name="mixamorig:Spine2",
            parent="mixamorig:Spine1",
            head=(0.0, 0.0, 1.35),
            tail=(0.0, 0.0, 1.48),
        ),
        BoneSpec(
            name="mixamorig:Neck",
            parent="mixamorig:Spine2",
            head=(0.0, 0.0, 1.48),
            tail=(0.0, 0.0, 1.58),
        ),
        BoneSpec(
            name="mixamorig:Head",
            parent="mixamorig:Neck",
            head=(0.0, 0.0, 1.58),
            tail=(0.0, 0.0, 1.78),
        ),
        BoneSpec(
            name="mixamorig:LeftUpLeg",
            parent="mixamorig:Hips",
            head=(0.09, 0.0, 0.93),
            tail=(0.09, 0.0, 0.52),
        ),
        BoneSpec(
            name="mixamorig:LeftLeg",
            parent="mixamorig:LeftUpLeg",
            head=(0.09, 0.0, 0.52),
            tail=(0.09, 0.02, 0.10),
        ),
        BoneSpec(
            name="mixamorig:LeftFoot",
            parent="mixamorig:LeftLeg",
            head=(0.09, 0.02, 0.10),
            tail=(0.09, -0.11, 0.03),
        ),
        BoneSpec(
            name="mixamorig:LeftToeBase",
            parent="mixamorig:LeftFoot",
            head=(0.09, -0.11, 0.03),
            tail=(0.09, -0.21, 0.03),
        ),
        BoneSpec(
            name="mixamorig:RightUpLeg",
            parent="mixamorig:Hips",
            head=(-0.09, 0.0, 0.93),
            tail=(-0.09, 0.0, 0.52),
        ),
        BoneSpec(
            name="mixamorig:RightLeg",
            parent="mixamorig:RightUpLeg",
            head=(-0.09, 0.0, 0.52),
            tail=(-0.09, 0.02, 0.10),
        ),
        BoneSpec(
            name="mixamorig:RightFoot",
            parent="mixamorig:RightLeg",
            head=(-0.09, 0.02, 0.10),
            tail=(-0.09, -0.11, 0.03),
        ),
        BoneSpec(
            name="mixamorig:RightToeBase",
            parent="mixamorig:RightFoot",
            head=(-0.09, -0.11, 0.03),
            tail=(-0.09, -0.21, 0.03),
        ),
        BoneSpec(
            name="mixamorig:LeftShoulder",
            parent="mixamorig:Spine2",
            head=(0.06, 0.03, 1.44),
            tail=(0.18, 0.05, 1.44),
        ),
        BoneSpec(
            name="mixamorig:LeftArm",
            parent="mixamorig:LeftShoulder",
            head=(0.18, 0.05, 1.44),
            tail=(0.45, 0.05, 1.44),
        ),
        BoneSpec(
            name="mixamorig:LeftForeArm",
            parent="mixamorig:LeftArm",
            head=(0.45, 0.05, 1.44),
            tail=(0.72, 0.05, 1.44),
        ),
        BoneSpec(
            name="mixamorig:LeftHand",
            parent="mixamorig:LeftForeArm",
            head=(0.72, 0.05, 1.44),
            tail=(0.82, 0.05, 1.44),
        ),
        BoneSpec(
            name="mixamorig:RightShoulder",
            parent="mixamorig:Spine2",
            head=(-0.06, 0.03, 1.44),
            tail=(-0.18, 0.05, 1.44),
        ),
        BoneSpec(
            name="mixamorig:RightArm",
            parent="mixamorig:RightShoulder",
            head=(-0.18, 0.05, 1.44),
            tail=(-0.45, 0.05, 1.44),
        ),
        BoneSpec(
            name="mixamorig:RightForeArm",
            parent="mixamorig:RightArm",
            head=(-0.45, 0.05, 1.44),
            tail=(-0.72, 0.05, 1.44),
        ),
        BoneSpec(
            name="mixamorig:RightHand",
            parent="mixamorig:RightForeArm",
            head=(-0.72, 0.05, 1.44),
            tail=(-0.82, 0.05, 1.44),
        ),
    ]
    return ArmatureSpec(bones=bones, skinning_mode="dqs")


def test_dual_quaternion_skinning_preserves_rest_and_deforms_cleanly() -> None:
    cyl = make_cylinder(radius=0.12, height=1.0, segments=16, origin="bottom")
    arm = ArmatureSpec(
        bones=[
            BoneSpec(name="lower", parent=None, head=(0.0, 0.0, 0.0), tail=(0.0, 0.0, 0.5)),
            BoneSpec(name="upper", parent="lower", head=(0.0, 0.0, 0.5), tail=(0.0, 0.0, 1.0)),
        ],
        skinning_mode="dqs",
    )
    arm_obj = SceneObject(id="arm", name="Arm", kind="armature", armature=arm)
    skin = auto_rig_mesh(cyl, np.eye(4), arm_obj, np.eye(4))

    # Rest pose should reproduce exact input vertices
    rest_out = deform_skinned_mesh(cyl, np.eye(4), skin, arm_obj, np.eye(4))
    assert np.allclose(rest_out.vertices, cyl.vertices, atol=1e-5)

    # 90-degree twist around Z should preserve cylinder radial volume better than LBS collapse
    arm.bones[1].rotation_deg = (0.0, 0.0, 120.0)
    dqs_out = deform_skinned_mesh(cyl, np.eye(4), skin, arm_obj, np.eye(4))
    mid_mask = np.abs(cyl.vertices[:, 2] - 0.5) < 0.06
    if np.any(mid_mask):
        dqs_radii = np.linalg.norm(dqs_out.vertices[mid_mask, :2], axis=1)
        assert float(dqs_radii.mean()) > 0.10


def test_analytical_two_bone_ik_hits_target_and_respects_pole() -> None:
    arm = _make_humanoid_armature()
    target = (0.10, -0.22, 0.25)
    res = solve_two_bone_ik(
        arm,
        upper_bone_name="mixamorig:LeftUpLeg",
        lower_bone_name="mixamorig:LeftLeg",
        end_bone_name="mixamorig:LeftFoot",
        target_pos=target,
        pole_dir=(0.0, -1.0, 0.0),
        rest_bend_dir=(0.0, -1.0, 0.0),
    )
    assert res["error"] < 1e-3

    _, pose_mats, _ = compute_armature_matrices(arm)
    ankle_pos = pose_mats["mixamorig:LeftFoot"][:3, 3]
    knee_pos = pose_mats["mixamorig:LeftLeg"][:3, 3]
    hip_pos = pose_mats["mixamorig:LeftUpLeg"][:3, 3]
    assert float(np.linalg.norm(ankle_pos - np.asarray(target))) < 1e-3
    # Knee must bend forward (-Y) relative to the hip-ankle line
    mid_y = 0.5 * (hip_pos[1] + ankle_pos[1])
    assert knee_pos[1] < mid_y


def test_locomotion_schedule_and_baking() -> None:
    scene = SceneDocument()
    arm = _make_humanoid_armature()
    arm_obj = SceneObject(id="arm_hero", name="HeroRig", kind="armature", armature=arm)
    scene.objects["arm_hero"] = arm_obj

    rig = detect_humanoid_rig(arm)
    assert rig.hips == "mixamorig:Hips"
    assert rig.left_foot == "mixamorig:LeftFoot"
    assert rig.right_hand == "mixamorig:RightHand"

    sched = build_showcase_schedule(fps=10, duration_s=18.0)
    assert len(sched) == 180
    # Character travels forward along -Y over the showcase
    assert sched[-1].root_y < -30.0

    # Evaluate representative poses across states
    for idx in (5, 45, 75, 110, 136):
        evaluate_locomotion_pose(arm, rig, sched[idx], apply_root_motion_to_hips=True)
        _, pose_mats, _ = compute_armature_matrices(arm)
        # Ankles must never sink below ground clearance
        assert pose_mats[rig.left_foot][2, 3] >= 0.085
        assert pose_mats[rig.right_foot][2, 3] >= 0.085

    meta = bake_locomotion_to_timeline(scene, arm_obj, sched[:20], fps=10)
    assert len(meta) == 20
    assert scene.timeline.frame_end == 20
    assert len(scene.timeline.tracks) > 20


def test_binary_fbx_export_import_roundtrip(tmp_path: Path) -> None:
    scene = SceneDocument()
    cyl = make_cylinder(radius=0.15, height=1.0, segments=12, origin="bottom")
    arm = ArmatureSpec(
        bones=[
            BoneSpec(name="root", parent=None, head=(0.0, 0.0, 0.0), tail=(0.0, 0.0, 0.5)),
            BoneSpec(name="tip", parent="root", head=(0.0, 0.0, 0.5), tail=(0.0, 0.0, 1.0)),
        ]
    )
    arm_obj = SceneObject(id="arm1", name="TestArm", kind="armature", armature=arm)
    skin = auto_rig_mesh(cyl, np.eye(4), arm_obj, np.eye(4))
    mesh_obj = SceneObject(
        id="mesh1",
        name="TestCyl",
        kind="mesh",
        transform=Transform(),
        mesh=cyl.to_payload(),
        skin=skin,
    )
    scene.objects["arm1"] = arm_obj
    scene.objects["mesh1"] = mesh_obj

    fbx_path = tmp_path / "roundtrip.fbx"
    export_fbx_scene(scene, fbx_path, ascii_mode=False, target_engine="generic")

    scene2 = SceneDocument()
    imported = import_fbx_file(scene2, fbx_path, ground_to_z0=True)
    meshes = [o for o in imported if o.kind == "mesh"]
    arms = [o for o in imported if o.kind == "armature"]
    assert len(meshes) == 1
    assert len(arms) == 1
    assert len(arms[0].armature.bones) == 2
    assert meshes[0].skin is not None
    assert len(meshes[0].skin.weights) >= 1
