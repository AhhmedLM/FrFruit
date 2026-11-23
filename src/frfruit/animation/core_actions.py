"""Synthesizers for additional Core Locomotion animations: Climb_Ladder, Fall_Loop, and Get_Up."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any

import numpy as np

from frfruit.animation.armature import (
    compute_armature_matrices,
    set_bone_world_rotation,
    solve_two_bone_ik,
)
from frfruit.animation.locomotion import HumanoidRigMap, detect_humanoid_rig
from frfruit.animation.timeline import FCurveTrack, Keyframe
from frfruit.core.math3d import euler_deg_to_matrix

if TYPE_CHECKING:
    from frfruit.core.scene import ArmatureSpec, SceneDocument, SceneObject


def _lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * t


def _smoothstep(edge0: float, edge1: float, x: float) -> float:
    if edge1 <= edge0:
        return 1.0 if x >= edge1 else 0.0
    t = max(0.0, min(1.0, (x - edge0) / (edge1 - edge0)))
    return t * t * (3.0 - 2.0 * t)


def _reset_armature(armature: ArmatureSpec) -> None:
    for b in armature.bones:
        b.rotation_deg = (0.0, 0.0, 0.0)
        b.translation = (0.0, 0.0, 0.0)


def _set_fingers(
    bones: dict[str, Any],
    fingers: list[tuple[str, int]] | None,
    side: float,
    curl: float,
) -> None:
    if not fingers:
        return
    for fname, j_idx in fingers:
        if fname not in bones:
            continue
        if "thumb" in fname.lower():
            ang_y = side * (16.0 + 22.0 * curl) if j_idx == 1 else side * (10.0 + 24.0 * curl)
            ang_z = -side * (22.0 * curl) if j_idx == 1 else 0.0
            bones[fname].rotation_deg = (0.0, float(ang_y), float(ang_z))
        else:
            base_deg = {1: 42.0, 2: 68.0, 3: 52.0}.get(j_idx, 45.0)
            bones[fname].rotation_deg = (0.0, float(side * base_deg * curl), 0.0)


def evaluate_climb_ladder(
    armature: ArmatureSpec,
    rig: HumanoidRigMap,
    u: float,
) -> None:
    """Pose `armature` for a seamless looping ladder climb at cycle phase u in [0, 1)."""
    _reset_armature(armature)
    bones = {b.name: b for b in armature.bones}

    two_pi = 2.0 * math.pi * u
    # Hips stay slightly back from ladder rungs (ladder at Y = -0.34), bobbing 2x per cycle
    hip_tx = 0.032 * math.sin(two_pi)
    hip_ty = 0.04 + 0.014 * math.cos(2.0 * two_pi)
    hip_tz = -0.055 + 0.022 * math.sin(2.0 * two_pi)
    hip_rx = 7.0 + 2.5 * math.sin(2.0 * two_pi)
    hip_ry = 4.5 * math.sin(two_pi)
    hip_rz = -7.0 * math.cos(two_pi)

    bones[rig.hips].translation = (float(hip_tx), float(hip_ty), float(hip_tz))
    bones[rig.hips].rotation_deg = (float(hip_rx), float(hip_ry), float(hip_rz))

    # Spine leans slightly toward ladder & counter-twists toward reaching arm
    spine_pitch = 5.0
    spine_yaw = 9.0 * math.cos(two_pi)
    spine_roll = -3.5 * math.sin(two_pi)
    for bname, wp, wy, wr in (
        (rig.spine, 0.30, 0.28, 0.30),
        (rig.spine1, 0.35, 0.36, 0.35),
        (rig.spine2, 0.35, 0.36, 0.35),
    ):
        if bname and bname in bones:
            bones[bname].rotation_deg = (
                float(spine_pitch * wp),
                float(spine_roll * wr),
                float(spine_yaw * wy),
            )

    # Head looks up slightly toward the next rung
    if rig.neck and rig.neck in bones:
        set_bone_world_rotation(armature, rig.neck, euler_deg_to_matrix(-10.0, 0.0, 0.0))
    if rig.head and rig.head in bones:
        set_bone_world_rotation(
            armature, rig.head, euler_deg_to_matrix(-18.0, 0.0, 4.0 * math.cos(two_pi))
        )

    # Contralateral limbs: Left Leg + Right Arm push/pull together, Right Leg + Left Arm together
    # Phase u_leg: [0, 0.55) = foot planted on rung pushing down; [0.55, 1.0) = lifting to next rung
    leg_configs = [
        (1.0, 0.0, rig.left_upleg, rig.left_leg, rig.left_foot, rig.left_toe),
        (-1.0, 0.5, rig.right_upleg, rig.right_leg, rig.right_foot, rig.right_toe),
    ]
    rung_y = -0.29
    for side, ph_off, up_b, lo_b, ft_b, toe_b in leg_configs:
        if not (up_b and lo_b and ft_b):
            continue
        ul = (u + ph_off) % 1.0
        if ul < 0.55:
            s = ul / 0.55
            foot_z = _lerp(0.47, 0.14, s)
            foot_y = rung_y
            foot_pitch = _lerp(-8.0, 12.0, s)
        else:
            sw = (ul - 0.55) / 0.45
            prog = sw * sw * (3.0 - 2.0 * sw)
            foot_z = _lerp(0.14, 0.47, prog)
            foot_y = rung_y + 0.12 * math.sin(math.pi * sw)  # Pull back off rung to clear it
            foot_pitch = _lerp(12.0, -8.0, prog)

        foot_x = side * 0.145
        solve_two_bone_ik(
            armature,
            upper_bone_name=up_b,
            lower_bone_name=lo_b,
            end_bone_name=ft_b,
            target_pos=(float(foot_x), float(foot_y), float(foot_z)),
            pole_dir=(side * 0.35, -1.0, 0.15),
            rest_bend_dir=(0.0, -1.0, 0.0),
            soft_ratio=0.985,
        )
        set_bone_world_rotation(armature, ft_b, euler_deg_to_matrix(foot_pitch, 0.0, side * 4.0))
        if toe_b and toe_b in bones:
            set_bone_world_rotation(armature, toe_b, euler_deg_to_matrix(0.0, 0.0, side * 4.0))

    # Arms: opposite phase to same-side leg!
    arm_configs = [
        (
            1.0,
            0.5,
            rig.left_shoulder,
            rig.left_arm,
            rig.left_forearm,
            rig.left_hand,
            rig.left_fingers,
        ),
        (
            -1.0,
            0.0,
            rig.right_shoulder,
            rig.right_arm,
            rig.right_forearm,
            rig.right_hand,
            rig.right_fingers,
        ),
    ]
    hand_rung_y = -0.34
    for side, ph_off, sh_b, arm_b, fa_b, hd_b, fingers in arm_configs:
        if not (arm_b and fa_b and hd_b):
            continue
        ua = (u + ph_off) % 1.0
        if ua < 0.55:
            # Pulling body up while gripping rung
            s = ua / 0.55
            hand_z = _lerp(1.74, 1.38, s)
            hand_y = hand_rung_y
            grip_curl = 0.92
        else:
            # Reaching up to grab next rung
            sw = (ua - 0.55) / 0.45
            prog = sw * sw * (3.0 - 2.0 * sw)
            hand_z = _lerp(1.38, 1.74, prog)
            hand_y = hand_rung_y + 0.11 * math.sin(math.pi * sw)
            grip_curl = 0.92 - 0.55 * math.sin(math.pi * sw)

        if sh_b and sh_b in bones:
            elev = 14.0 if ua >= 0.55 else 6.0
            bones[sh_b].rotation_deg = (0.0, float(-side * elev), float(-side * 8.0))

        hand_x = side * 0.21
        solve_two_bone_ik(
            armature,
            upper_bone_name=arm_b,
            lower_bone_name=fa_b,
            end_bone_name=hd_b,
            target_pos=(float(hand_x), float(hand_y), float(hand_z)),
            pole_dir=(side * 0.55, 0.85, -0.45),
            rest_bend_dir=(0.0, 1.0, 0.0),
            soft_ratio=0.985,
        )
        bones[hd_b].rotation_deg = (0.0, float(-side * 12.0), float(side * 15.0))
        _set_fingers(bones, fingers, side, grip_curl)


def evaluate_fall_loop(
    armature: ArmatureSpec,
    rig: HumanoidRigMap,
    u: float,
) -> None:
    """Pose `armature` for a panicked/athletic mid-air Fall_Loop at phase u in [0, 1)."""
    _reset_armature(armature)
    bones = {b.name: b for b in armature.bones}

    two_pi = 2.0 * math.pi * u
    # Airborne CoM oscillation
    hip_tz = -0.06 + 0.025 * math.sin(two_pi)
    hip_rx = 16.0 + 4.0 * math.sin(two_pi)
    hip_ry = 5.0 * math.cos(two_pi)
    hip_rz = 6.0 * math.sin(two_pi + 0.5)
    bones[rig.hips].translation = (0.0, 0.0, float(hip_tz))
    bones[rig.hips].rotation_deg = (float(hip_rx), float(hip_ry), float(hip_rz))

    # Torso arches slightly backward/forward against wind rush
    sp_pitch = 6.0 + 5.0 * math.cos(two_pi)
    sp_yaw = -8.0 * math.sin(two_pi)
    sp_roll = -5.0 * math.cos(two_pi)
    for bname, wp, wy, wr in (
        (rig.spine, 0.30, 0.30, 0.30),
        (rig.spine1, 0.35, 0.35, 0.35),
        (rig.spine2, 0.35, 0.35, 0.35),
    ):
        if bname and bname in bones:
            bones[bname].rotation_deg = (
                float(sp_pitch * wp),
                float(sp_roll * wr),
                float(sp_yaw * wy),
            )

    # Head looks downward toward impending impact
    if rig.neck and rig.neck in bones:
        set_bone_world_rotation(armature, rig.neck, euler_deg_to_matrix(14.0, 0.0, 0.0))
    if rig.head and rig.head in bones:
        set_bone_world_rotation(
            armature,
            rig.head,
            euler_deg_to_matrix(22.0 + 3.0 * math.sin(two_pi), 0.0, 5.0 * math.cos(two_pi)),
        )

    # Legs flail/bicycle asynchronously in mid-air
    leg_configs = [
        (1.0, 0.0, rig.left_upleg, rig.left_leg, rig.left_foot, rig.left_toe),
        (-1.0, 0.5, rig.right_upleg, rig.right_leg, rig.right_foot, rig.right_toe),
    ]
    for side, ph_off, up_b, lo_b, ft_b, toe_b in leg_configs:
        if not (up_b and lo_b and ft_b):
            continue
        ang = 2.0 * math.pi * ((u + ph_off) % 1.0)
        fx = side * (0.17 + 0.04 * math.sin(ang))
        fy = -0.14 - 0.14 * math.cos(ang)
        fz = 0.28 + 0.12 * math.sin(ang)
        solve_two_bone_ik(
            armature,
            upper_bone_name=up_b,
            lower_bone_name=lo_b,
            end_bone_name=ft_b,
            target_pos=(float(fx), float(fy), float(fz)),
            pole_dir=(side * 0.35, -1.0, 0.15),
            rest_bend_dir=(0.0, -1.0, 0.0),
            soft_ratio=0.98,
        )
        set_bone_world_rotation(
            armature, ft_b, euler_deg_to_matrix(22.0 + 14.0 * math.cos(ang), 0.0, side * 10.0)
        )
        if toe_b and toe_b in bones:
            set_bone_world_rotation(armature, toe_b, euler_deg_to_matrix(15.0, 0.0, side * 10.0))

    # Arms windmill/reach upward & outward against air rush
    arm_configs = [
        (
            1.0,
            0.0,
            rig.left_shoulder,
            rig.left_arm,
            rig.left_forearm,
            rig.left_hand,
            rig.left_fingers,
        ),
        (
            -1.0,
            0.5,
            rig.right_shoulder,
            rig.right_arm,
            rig.right_forearm,
            rig.right_hand,
            rig.right_fingers,
        ),
    ]
    for side, ph_off, sh_b, arm_b, fa_b, hd_b, fingers in arm_configs:
        if not (arm_b and fa_b and hd_b):
            continue
        ang = 2.0 * math.pi * ((u + ph_off) % 1.0)
        if sh_b and sh_b in bones:
            bones[sh_b].rotation_deg = (0.0, float(-side * 16.0), float(-side * 5.0))
        hx = side * (0.36 + 0.07 * math.cos(ang))
        hy = -0.18 + 0.12 * math.sin(ang)
        hz = 1.44 + 0.14 * math.cos(ang)
        solve_two_bone_ik(
            armature,
            upper_bone_name=arm_b,
            lower_bone_name=fa_b,
            end_bone_name=hd_b,
            target_pos=(float(hx), float(hy), float(hz)),
            pole_dir=(side * 0.7, 0.6, -0.5),
            rest_bend_dir=(0.0, 1.0, 0.0),
            soft_ratio=0.98,
        )
        bones[hd_b].rotation_deg = (float(10.0 * math.sin(ang)), 0.0, float(-side * 12.0))
        _set_fingers(bones, fingers, side, 0.14)


def evaluate_get_up(
    armature: ArmatureSpec,
    rig: HumanoidRigMap,
    u: float,
) -> None:
    """Pose `armature` for a realistic 1-shot Get_Up from prone ground recovery to standing Idle."""
    _reset_armature(armature)
    bones = {b.name: b for b in armature.bones}

    l_foot_rest = (
        np.asarray(bones[rig.left_foot].head, dtype=np.float64)
        if rig.left_foot
        else np.array([0.09, 0.02, 0.105])
    )
    ankle_z = float(l_foot_rest[2])

    # Keyframed biomechanical curve across u in [0, 1]:
    # Stage A (0.00 -> 0.25): Prone on ground -> push chest onto hands & knees
    # Stage B (0.25 -> 0.55): Plant Left foot flat forward (half-kneeling push)
    # Stage C (0.55 -> 0.85): Drive upward over Left foot, step Right foot forward
    # Stage D (0.85 -> 1.00): Stabilize into natural standing Idle
    s1 = _smoothstep(0.05, 0.30, u)
    s2 = _smoothstep(0.28, 0.56, u)
    s3 = _smoothstep(0.54, 0.86, u)
    s4 = _smoothstep(0.84, 1.00, u)

    hip_tz = -0.82 * (1.0 - s1) + (-0.54) * (s1 - s2) + (-0.28) * (s2 - s3) + (-0.024) * s3
    # Add realistic weight-shift sway while standing up
    hip_tx = 0.06 * math.sin(math.pi * _smoothstep(0.35, 0.85, u))
    hip_ty = 0.35 * (1.0 - s2) + 0.10 * (s2 - s3)
    hip_rx = 74.0 * (1.0 - s1) + 46.0 * (s1 - s2) + 24.0 * (s2 - s3) + 1.8 * s3
    hip_ry = -6.0 * (s1 - s3)
    hip_rz = -12.0 * (s1 - s3) - 3.5 * s4

    bones[rig.hips].translation = (float(hip_tx), float(hip_ty), float(hip_tz))
    bones[rig.hips].rotation_deg = (float(hip_rx), float(hip_ry), float(hip_rz))

    # Spine arches slightly up on push-up, then straightens cleanly
    sp_pitch = -18.0 * (1.0 - s1) + 10.0 * (s1 - s3) + 2.5 * s3
    sp_yaw = 10.0 * (s1 - s3) + 3.5 * s4
    for bname, wp, wy in (
        (rig.spine, 0.30, 0.30),
        (rig.spine1, 0.35, 0.35),
        (rig.spine2, 0.35, 0.35),
    ):
        if bname and bname in bones:
            bones[bname].rotation_deg = (float(sp_pitch * wp), 0.0, float(sp_yaw * wy))

    # Head looks ahead/slightly down toward ground then levels out
    head_pitch = 16.0 * (1.0 - s2) + 6.0 * (s2 - s3) - 1.5 * s3
    if rig.neck and rig.neck in bones:
        set_bone_world_rotation(armature, rig.neck, euler_deg_to_matrix(head_pitch * 0.5, 0.0, 0.0))
    if rig.head and rig.head in bones:
        set_bone_world_rotation(armature, rig.head, euler_deg_to_matrix(head_pitch, 0.0, 0.0))

    # Left leg: trails on ground -> steps forward & plants flat (-0.24) -> settles to Idle
    l_ank_x = 0.135
    l_ank_y = 0.72 * (1.0 - s2) + (-0.24) * (s2 - s3) + (-0.025) * s3
    l_ank_z = ankle_z + 0.08 * math.sin(math.pi * _smoothstep(0.24, 0.54, u))
    l_foot_pitch = 52.0 * (1.0 - s2)
    solve_two_bone_ik(
        armature,
        upper_bone_name=rig.left_upleg,
        lower_bone_name=rig.left_leg,
        end_bone_name=rig.left_foot,
        target_pos=(float(l_ank_x), float(l_ank_y), float(max(ankle_z, l_ank_z))),
        pole_dir=(0.18, -1.0, 0.1),
        rest_bend_dir=(0.0, -1.0, 0.0),
        soft_ratio=0.992,
    )
    set_bone_world_rotation(armature, rig.left_foot, euler_deg_to_matrix(l_foot_pitch, 0.0, 5.0))
    if rig.left_toe and rig.left_toe in bones:
        set_bone_world_rotation(armature, rig.left_toe, euler_deg_to_matrix(0.0, 0.0, 5.0))

    # Right leg: stays trailing behind pushing off toes -> steps forward after s3
    r_ank_x = -0.135
    r_ank_y = 0.76 * (1.0 - s3) + 0.035 * s3
    r_ank_z = ankle_z + 0.06 * math.sin(math.pi * _smoothstep(0.60, 0.88, u))
    r_foot_pitch = 58.0 * (1.0 - s3)
    solve_two_bone_ik(
        armature,
        upper_bone_name=rig.right_upleg,
        lower_bone_name=rig.right_leg,
        end_bone_name=rig.right_foot,
        target_pos=(float(r_ank_x), float(r_ank_y), float(max(ankle_z, r_ank_z))),
        pole_dir=(-0.18, -1.0, 0.1),
        rest_bend_dir=(0.0, -1.0, 0.0),
        soft_ratio=0.992,
    )
    set_bone_world_rotation(armature, rig.right_foot, euler_deg_to_matrix(r_foot_pitch, 0.0, -8.0))
    if rig.right_toe and rig.right_toe in bones:
        set_bone_world_rotation(armature, rig.right_toe, euler_deg_to_matrix(0.0, 0.0, -8.0))

    # Hands: push firmly against ground (Z=0.05) during Stage A & B, then lift & relax to sides!
    _, pose_arm, _ = compute_armature_matrices(armature)
    hip_pos = pose_arm[rig.hips][:3, 3]
    for side, _sh_b, arm_b, fa_b, hd_b, fingers in (
        (1.0, rig.left_shoulder, rig.left_arm, rig.left_forearm, rig.left_hand, rig.left_fingers),
        (
            -1.0,
            rig.right_shoulder,
            rig.right_arm,
            rig.right_forearm,
            rig.right_hand,
            rig.right_fingers,
        ),
    ):
        if not (arm_b and fa_b and hd_b):
            continue
        sh_pos = pose_arm[arm_b][:3, 3]
        # Ground push target -> idle hanging hand target
        push_x = side * 0.24
        push_y = -0.22 if side > 0 else -0.14
        push_z = 0.055
        idle_x = sh_pos[0] + side * 0.085
        idle_y = sh_pos[1] - 0.055
        idle_z = hip_pos[2] - 0.16

        # Blend from ground push (0..0.52) to lifting off floor (0.52..1.0)
        lift_w = _smoothstep(0.42, 0.85, u)
        hx = _lerp(push_x, idle_x, lift_w)
        hy = _lerp(push_y, idle_y, lift_w)
        hz = max(0.045, _lerp(push_z, idle_z, lift_w))

        solve_two_bone_ik(
            armature,
            upper_bone_name=arm_b,
            lower_bone_name=fa_b,
            end_bone_name=hd_b,
            target_pos=(float(hx), float(hy), float(hz)),
            pole_dir=(side * 0.55, 0.85, 0.25 * (1.0 - lift_w) - 0.35 * lift_w),
            rest_bend_dir=(0.0, 1.0, 0.0),
            soft_ratio=0.985,
        )
        # Flatten palm against ground during push, relax wrist as standing
        bones[hd_b].rotation_deg = (
            float(-28.0 * (1.0 - lift_w)),
            float(side * 6.0),
            0.0,
        )
        _set_fingers(bones, fingers, side, 0.08 * (1.0 - lift_w) + 0.20 * lift_w)


def bake_core_action_to_timeline(
    scene: SceneDocument,
    armature_obj: SceneObject,
    action_name: str,
    *,
    fps: int = 30,
) -> int:
    """Bake Climb_Ladder, Fall_Loop, or Get_Up onto `scene.timeline` and return frame count."""
    assert armature_obj.armature is not None
    rig = detect_humanoid_rig(armature_obj.armature)
    armature_obj.armature.skinning_mode = "dqs"

    durations = {
        "Climb_Ladder": 1.6,  # 48 frames seamless loop
        "Fall_Loop": 1.2,  # 36 frames seamless loop
        "Get_Up": 2.2,  # 66 frames one-shot recovery
    }
    if action_name not in durations:
        raise ValueError(f"Unknown core action {action_name!r}")
    n_frames = max(4, int(round(durations[action_name] * fps)))
    is_loop = action_name != "Get_Up"

    scene.timeline.fps = int(fps)
    scene.timeline.frame_start = 1
    scene.timeline.frame_end = n_frames
    scene.timeline.current_frame = 1
    scene.timeline.tracks = [t for t in scene.timeline.tracks if t.target_id != armature_obj.id]

    bone_tracks: dict[str, list[Keyframe]] = {}
    for fi in range(n_frames):
        fnum = fi + 1
        u = (fi / float(n_frames)) if is_loop else (fi / float(max(1, n_frames - 1)))
        if action_name == "Climb_Ladder":
            evaluate_climb_ladder(armature_obj.armature, rig, u)
        elif action_name == "Fall_Loop":
            evaluate_fall_loop(armature_obj.armature, rig, u)
        else:
            evaluate_get_up(armature_obj.armature, rig, u)

        for b in armature_obj.armature.bones:
            for ai, ax in enumerate(("x", "y", "z")):
                pp_r = f"bone.{b.name}.rotation_deg.{ax}"
                bone_tracks.setdefault(pp_r, []).append(
                    Keyframe(
                        frame=fnum,
                        value=round(float(b.rotation_deg[ai]), 4),
                        interpolation="linear",
                    )
                )
            if b.name == rig.hips:
                for ai, ax in enumerate(("x", "y", "z")):
                    pp_t = f"bone.{b.name}.translation.{ax}"
                    bone_tracks.setdefault(pp_t, []).append(
                        Keyframe(
                            frame=fnum,
                            value=round(float(b.translation[ai]), 5),
                            interpolation="linear",
                        )
                    )

    for pp, kfs in bone_tracks.items():
        if any(abs(k.value) > 1e-4 for k in kfs):
            scene.timeline.tracks.append(
                FCurveTrack(target_id=armature_obj.id, property_path=pp, keyframes=kfs)
            )
    return n_frames
