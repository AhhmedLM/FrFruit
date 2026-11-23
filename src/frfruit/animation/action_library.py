"""Synthesizers for the remaining 27 Unity/Mixamo gameplay animations across 5 categories.

Categories:
  2. Construction & Economy: Carry_Idle, Carry_Walk, Pickup_Item, Drop_Item,
     Hammer_Wall, Hammer_Ground, Saw_Wood, Inspect_Work
  3. Reactions & Civilians: Flee_Run, Cower_Idle, Hit_Flinch, Stunned_Loop, Cheer_Victory
  4. Melee Combat: Melee_Idle, Charge_Run, Slash_Attack_1, Slash_Attack_2, Bash_Gate, Block_Idle
  5. Ranged Combat: Bow_Draw, Bow_Aim_Idle, Bow_Release, Throw_Stone
  6. Siege Operation: Push_Ram, Crank_Wheel, Pull_Lever, Load_Ammo
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any

from frfruit.animation.armature import (
    compute_armature_matrices,
    set_bone_world_rotation,
    solve_two_bone_ik,
)
from frfruit.animation.locomotion import (
    GAIT_PRESETS,
    GaitParams,
    HumanoidRigMap,
    LocomotionFrameState,
    detect_humanoid_rig,
    evaluate_locomotion_pose,
)
from frfruit.animation.timeline import FCurveTrack, Keyframe
from frfruit.core.math3d import euler_deg_to_matrix

if TYPE_CHECKING:
    from frfruit.core.scene import ArmatureSpec, SceneDocument, SceneObject


ACTION_SPECS: dict[str, tuple[float, bool]] = {
    # Category 2: Construction & Economy
    "Carry_Idle": (2.4, True),
    "Carry_Walk": (1.33, True),
    "Pickup_Item": (1.5, False),
    "Drop_Item": (1.3, False),
    "Hammer_Wall": (0.95, True),
    "Hammer_Ground": (1.05, True),
    "Saw_Wood": (0.95, True),
    "Inspect_Work": (2.4, False),
    # Category 3: Reactions & Civilians
    "Flee_Run": (0.72, True),
    "Cower_Idle": (2.0, True),
    "Hit_Flinch": (0.75, False),
    "Stunned_Loop": (2.0, True),
    "Cheer_Victory": (1.6, True),
    # Category 4: Melee Combat
    "Melee_Idle": (2.4, True),
    "Charge_Run": (0.72, True),
    "Slash_Attack_1": (0.90, False),
    "Slash_Attack_2": (1.00, False),
    "Bash_Gate": (1.15, False),
    "Block_Idle": (2.0, True),
    # Category 5: Ranged Combat
    "Bow_Draw": (0.85, False),
    "Bow_Aim_Idle": (2.0, True),
    "Bow_Release": (0.65, False),
    "Throw_Stone": (1.10, False),
    # Category 6: Siege Operation
    "Push_Ram": (1.25, True),
    "Crank_Wheel": (1.35, True),
    "Pull_Lever": (1.20, False),
    "Load_Ammo": (1.80, False),
}


def _lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * t


def _smoothstep(edge0: float, edge1: float, x: float) -> float:
    if edge1 <= edge0:
        return 1.0 if x >= edge1 else 0.0
    t = max(0.0, min(1.0, (x - edge0) / (edge1 - edge0)))
    return t * t * (3.0 - 2.0 * t)


def _reset(armature: ArmatureSpec) -> None:
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


def _ankle_z(bones: dict[str, Any], rig: HumanoidRigMap) -> float:
    if rig.left_foot and rig.left_foot in bones:
        return float(bones[rig.left_foot].head[2])
    return 0.105


def _plant_feet(
    armature: ArmatureSpec,
    rig: HumanoidRigMap,
    left_pos: tuple[float, float, float],
    right_pos: tuple[float, float, float],
    left_yaw: float = 5.0,
    right_yaw: float = -8.0,
    left_pitch: float = 0.0,
    right_pitch: float = 0.0,
    pole_spread: float = 0.16,
) -> None:
    bones = {b.name: b for b in armature.bones}
    for side, pos, yaw, pitch, up_b, lo_b, ft_b, toe_b in (
        (
            1.0,
            left_pos,
            left_yaw,
            left_pitch,
            rig.left_upleg,
            rig.left_leg,
            rig.left_foot,
            rig.left_toe,
        ),
        (
            -1.0,
            right_pos,
            right_yaw,
            right_pitch,
            rig.right_upleg,
            rig.right_leg,
            rig.right_foot,
            rig.right_toe,
        ),
    ):
        if not (up_b and lo_b and ft_b):
            continue
        solve_two_bone_ik(
            armature,
            upper_bone_name=up_b,
            lower_bone_name=lo_b,
            end_bone_name=ft_b,
            target_pos=pos,
            pole_dir=(side * pole_spread, -1.0, 0.08),
            rest_bend_dir=(0.0, -1.0, 0.0),
            soft_ratio=0.994,
        )
        set_bone_world_rotation(armature, ft_b, euler_deg_to_matrix(pitch, 0.0, yaw))
        if toe_b and toe_b in bones:
            set_bone_world_rotation(armature, toe_b, euler_deg_to_matrix(0.0, 0.0, yaw))


def _set_spine(
    bones: dict[str, Any],
    rig: HumanoidRigMap,
    pitch: float,
    roll: float,
    yaw: float,
) -> None:
    for bname, wp, wy, wr in (
        (rig.spine, 0.28, 0.25, 0.30),
        (rig.spine1, 0.34, 0.35, 0.35),
        (rig.spine2, 0.38, 0.40, 0.35),
    ):
        if bname and bname in bones:
            bones[bname].rotation_deg = (float(pitch * wp), float(roll * wr), float(yaw * wy))


def _set_head(
    armature: ArmatureSpec,
    rig: HumanoidRigMap,
    pitch: float,
    roll: float,
    yaw: float,
) -> None:
    bones = {b.name: b for b in armature.bones}
    if rig.neck and rig.neck in bones:
        set_bone_world_rotation(
            armature, rig.neck, euler_deg_to_matrix(pitch * 0.5, roll * 0.5, yaw * 0.45)
        )
    if rig.head and rig.head in bones:
        set_bone_world_rotation(armature, rig.head, euler_deg_to_matrix(pitch, roll, yaw))


def _pose_arm(
    armature: ArmatureSpec,
    rig: HumanoidRigMap,
    side: float,
    target: tuple[float, float, float],
    elbow_pole: tuple[float, float, float] | None = None,
    wrist_deg: tuple[float, float, float] = (0.0, 0.0, 0.0),
    curl: float = 0.65,
    clav_elev: float = 2.0,
    clav_protract: float = 2.0,
) -> None:
    bones = {b.name: b for b in armature.bones}
    sh_b = rig.left_shoulder if side > 0 else rig.right_shoulder
    arm_b = rig.left_arm if side > 0 else rig.right_arm
    fa_b = rig.left_forearm if side > 0 else rig.right_forearm
    hd_b = rig.left_hand if side > 0 else rig.right_hand
    fingers = rig.left_fingers if side > 0 else rig.right_fingers
    if not (arm_b and fa_b and hd_b):
        return
    if sh_b and sh_b in bones:
        bones[sh_b].rotation_deg = (0.0, float(-side * clav_elev), float(-side * clav_protract))
    pole = elbow_pole if elbow_pole is not None else (side * 0.45, 0.9, -0.4)
    solve_two_bone_ik(
        armature,
        upper_bone_name=arm_b,
        lower_bone_name=fa_b,
        end_bone_name=hd_b,
        target_pos=target,
        pole_dir=pole,
        rest_bend_dir=(0.0, 1.0, 0.0),
        soft_ratio=0.985,
    )
    bones[hd_b].rotation_deg = wrist_deg
    _set_fingers(bones, fingers, side, curl)


def _walk_legs_base(
    armature: ArmatureSpec,
    rig: HumanoidRigMap,
    u: float,
    gait: GaitParams,
    time_s: float,
) -> None:
    """Evaluate ground-locked locomotion base for upper-body overrides."""
    st = LocomotionFrameState(
        time_s=time_s,
        label="BASE",
        root_y=0.0,
        speed=gait.speed,
        accel=0.0,
        phase=u,
        gait=gait,
        idle_weight=0.0,
        brake_weight=0.0,
        brake_progress=0.0,
        cadence_hz=gait.cadence,
    )
    evaluate_locomotion_pose(armature, rig, st, apply_root_motion_to_hips=False)


# ============================================================================
# CATEGORY 2: CONSTRUCTION & ECONOMY
# ============================================================================


def _eval_carry_idle(armature: ArmatureSpec, rig: HumanoidRigMap, u: float) -> None:
    _reset(armature)
    bones = {b.name: b for b in armature.bones}
    az = _ankle_z(bones, rig)
    br = math.sin(2.0 * math.pi * u)
    bones[rig.hips].translation = (0.0, 0.02, -0.045 + 0.006 * br)
    bones[rig.hips].rotation_deg = (-2.0, 0.0, 0.0)
    _set_spine(bones, rig, -4.5 - 1.2 * br, 0.0, 0.0)
    _set_head(armature, rig, 2.0, 0.0, 3.0 * math.sin(2.0 * math.pi * u))
    _plant_feet(armature, rig, (0.145, -0.03, az), (-0.145, 0.03, az))
    # Both hands cradling a heavy crate/stone at chest-waist height
    bz = 1.02 + 0.010 * br
    _pose_arm(
        armature,
        rig,
        1.0,
        (0.20, -0.34, bz),
        (0.55, 0.85, -0.45),
        (0.0, 12.0, -25.0),
        0.72,
        4.0,
        8.0,
    )
    _pose_arm(
        armature,
        rig,
        -1.0,
        (-0.20, -0.34, bz),
        (-0.55, 0.85, -0.45),
        (0.0, -12.0, 25.0),
        0.72,
        4.0,
        8.0,
    )


def _eval_carry_walk(armature: ArmatureSpec, rig: HumanoidRigMap, u: float) -> None:
    gait = GaitParams(
        speed=1.28,
        cadence=0.75,
        duty_factor=0.65,
        hip_drop=-0.042,
        bounce_amp=0.024,
        sway_amp=0.040,
        hip_roll_deg=4.8,
        hip_yaw_deg=6.0,
        spine_lean_deg=-3.5,
        spine_twist_deg=4.0,
        foot_lift=0.095,
        stance_width=0.125,
        heel_strike_deg=-15.0,
        toe_off_deg=34.0,
        arm_swing_fwd=0.2,
        arm_swing_back=-0.1,
        arm_height=0.0,
        arm_out=0.05,
        elbow_out=0.4,
        fist_curl=0.7,
    )
    _walk_legs_base(armature, rig, u, gait, u / gait.cadence)
    _, pose_m, _ = compute_armature_matrices(armature)
    hip_p = pose_m[rig.hips][:3, 3]
    bob = 0.012 * math.cos(4.0 * math.pi * u)
    _pose_arm(
        armature,
        rig,
        1.0,
        (0.20, -0.34, float(hip_p[2]) + 0.04 + bob),
        (0.55, 0.85, -0.45),
        (0.0, 12.0, -25.0),
        0.75,
        4.0,
        8.0,
    )
    _pose_arm(
        armature,
        rig,
        -1.0,
        (-0.20, -0.34, float(hip_p[2]) + 0.04 + bob),
        (-0.55, 0.85, -0.45),
        (0.0, -12.0, 25.0),
        0.75,
        4.0,
        8.0,
    )


def _eval_pickup_item(armature: ArmatureSpec, rig: HumanoidRigMap, u: float) -> None:
    _reset(armature)
    bones = {b.name: b for b in armature.bones}
    az = _ankle_z(bones, rig)
    bend = math.sin(math.pi * _smoothstep(0.05, 0.95, u))
    lift = _smoothstep(0.48, 0.95, u)
    bones[rig.hips].translation = (0.0, 0.08 * bend, -0.025 - 0.34 * bend)
    bones[rig.hips].rotation_deg = (34.0 * bend, 0.0, 0.0)
    _set_spine(bones, rig, 42.0 * bend - 4.0 * lift, 0.0, 0.0)
    _set_head(armature, rig, 26.0 * bend, 0.0, 0.0)
    _plant_feet(armature, rig, (0.145, -0.06, az), (-0.145, 0.06, az))
    hz = _lerp(0.84, 0.22, bend) if u < 0.5 else _lerp(0.22, 1.02, lift)
    hy = _lerp(-0.12, -0.42, bend) if u < 0.5 else _lerp(-0.42, -0.34, lift)
    curl = 0.25 if u < 0.45 else 0.75
    _pose_arm(
        armature, rig, 1.0, (0.19, hy, hz), (0.45, 0.85, -0.3), (0.0, 10.0, -20.0), curl, 4.0, 8.0
    )
    _pose_arm(
        armature,
        rig,
        -1.0,
        (-0.19, hy, hz),
        (-0.45, 0.85, -0.3),
        (0.0, -10.0, 20.0),
        curl,
        4.0,
        8.0,
    )


def _eval_drop_item(armature: ArmatureSpec, rig: HumanoidRigMap, u: float) -> None:
    _eval_pickup_item(armature, rig, 1.0 - u)


def _eval_hammer_wall(armature: ArmatureSpec, rig: HumanoidRigMap, u: float) -> None:
    _reset(armature)
    bones = {b.name: b for b in armature.bones}
    az = _ankle_z(bones, rig)
    # Sharp strike at u=0.55, wind-up 0..0.45
    strike = math.sin(2.0 * math.pi * u - 0.5)
    snap = max(0.0, math.sin(math.pi * _smoothstep(0.38, 0.62, u)))
    bones[rig.hips].translation = (0.0, 0.02 * snap, -0.055)
    bones[rig.hips].rotation_deg = (6.0 + 4.0 * snap, 0.0, -14.0)
    _set_spine(bones, rig, 8.0 + 10.0 * snap, 0.0, 10.0 - 16.0 * snap)
    _set_head(armature, rig, 2.0, 0.0, 4.0)
    _plant_feet(armature, rig, (0.15, -0.14, az), (-0.16, 0.16, az), 8.0, -22.0)
    # Right arm swings hammer forward into wall; Left steadies
    rx = -0.16 - 0.08 * snap
    ry = -0.18 - 0.34 * snap if u > 0.35 else _lerp(-0.38, -0.05, u / 0.35)
    rz = 1.62 - 0.42 * snap if u > 0.35 else _lerp(1.22, 1.68, u / 0.35)
    _pose_arm(
        armature,
        rig,
        -1.0,
        (rx, ry, rz),
        (-0.65, 0.7, -0.2),
        (-25.0 * snap, 0.0, 0.0),
        0.90,
        8.0,
        6.0,
    )
    _pose_arm(
        armature,
        rig,
        1.0,
        (0.20, -0.26, 1.06 + 0.03 * strike),
        (0.55, 0.85, -0.4),
        (0.0, 8.0, 0.0),
        0.65,
        2.0,
        4.0,
    )


def _eval_hammer_ground(armature: ArmatureSpec, rig: HumanoidRigMap, u: float) -> None:
    _reset(armature)
    bones = {b.name: b for b in armature.bones}
    az = _ankle_z(bones, rig)
    snap = math.sin(math.pi * _smoothstep(0.35, 0.68, u))
    bones[rig.hips].translation = (0.0, 0.06, -0.26 - 0.04 * snap)
    bones[rig.hips].rotation_deg = (24.0 + 6.0 * snap, 0.0, -10.0)
    _set_spine(bones, rig, 26.0 + 12.0 * snap, 0.0, 8.0 - 12.0 * snap)
    _set_head(armature, rig, 22.0, 0.0, 0.0)
    _plant_feet(armature, rig, (0.16, -0.16, az), (-0.17, 0.18, az), 10.0, -20.0)
    rz = 1.18 - 0.88 * snap if u > 0.32 else _lerp(0.34, 1.24, u / 0.32)
    ry = -0.22 - 0.22 * snap
    _pose_arm(
        armature,
        rig,
        -1.0,
        (-0.12, ry, rz),
        (-0.55, 0.8, 0.1),
        (-30.0 * snap, 0.0, 0.0),
        0.92,
        6.0,
        6.0,
    )
    _pose_arm(
        armature, rig, 1.0, (0.22, -0.24, 0.62), (0.5, 0.85, -0.3), (0.0, 0.0, 0.0), 0.55, 2.0, 4.0
    )


def _eval_saw_wood(armature: ArmatureSpec, rig: HumanoidRigMap, u: float) -> None:
    _reset(armature)
    bones = {b.name: b for b in armature.bones}
    az = _ankle_z(bones, rig)
    saw = math.sin(2.0 * math.pi * u)
    bones[rig.hips].translation = (0.0, 0.04 - 0.03 * saw, -0.14)
    bones[rig.hips].rotation_deg = (18.0 + 4.0 * saw, 0.0, -12.0)
    _set_spine(bones, rig, 20.0 + 6.0 * saw, 0.0, 10.0 - 8.0 * saw)
    _set_head(armature, rig, 18.0, 0.0, 4.0)
    _plant_feet(armature, rig, (0.16, -0.18, az), (-0.16, 0.18, az), 8.0, -22.0)
    _pose_arm(
        armature,
        rig,
        -1.0,
        (-0.14, -0.32 - 0.18 * saw, 0.78 - 0.06 * saw),
        (-0.5, 0.9, -0.2),
        (0.0, 0.0, 0.0),
        0.88,
        3.0,
        6.0,
    )
    _pose_arm(
        armature,
        rig,
        1.0,
        (0.16, -0.34, 0.76),
        (0.5, 0.85, -0.3),
        (-20.0, 0.0, 0.0),
        0.68,
        2.0,
        5.0,
    )


def _eval_inspect_work(armature: ArmatureSpec, rig: HumanoidRigMap, u: float) -> None:
    _reset(armature)
    bones = {b.name: b for b in armature.bones}
    az = _ankle_z(bones, rig)
    wipe = math.sin(math.pi * _smoothstep(0.15, 0.75, u))
    sweep = math.sin(2.0 * math.pi * _smoothstep(0.25, 0.65, u))
    bones[rig.hips].translation = (0.02 * sweep, 0.0, -0.025)
    bones[rig.hips].rotation_deg = (2.0, 0.0, -5.0)
    _set_spine(bones, rig, 3.0, 0.0, 5.0 + 8.0 * sweep)
    _set_head(armature, rig, -4.0 * wipe, 0.0, 18.0 * math.sin(2.0 * math.pi * u))
    _plant_feet(armature, rig, (0.13, -0.04, az), (-0.13, 0.04, az))
    # Right hand wipes brow at peak, left hand rests on hip/side
    hx = -0.18 * (1.0 - wipe) + (-0.06 + 0.08 * sweep) * wipe
    hy = -0.08 * (1.0 - wipe) - 0.16 * wipe
    hz = _lerp(0.84, 1.62, wipe)
    _pose_arm(
        armature, rig, -1.0, (hx, hy, hz), (-0.65, 0.6, -0.2), (0.0, 0.0, 0.0), 0.25, 4.0, 4.0
    )
    _pose_arm(
        armature, rig, 1.0, (0.24, -0.06, 0.86), (0.55, 0.8, -0.4), (0.0, 0.0, 0.0), 0.22, 1.0, 2.0
    )


# ============================================================================
# CATEGORY 3: REACTIONS & CIVILIANS
# ============================================================================


def _eval_flee_run(armature: ArmatureSpec, rig: HumanoidRigMap, u: float) -> None:
    gait = GAIT_PRESETS["explosive_sprint"]
    _walk_legs_base(armature, rig, u, gait, u / gait.cadence)
    _set_head(armature, rig, -8.0, 0.0, 16.0 * math.sin(2.0 * math.pi * u))
    _, pose_m, _ = compute_armature_matrices(armature)
    hip_p = pose_m[rig.hips][:3, 3]
    flail = math.sin(2.0 * math.pi * u)
    _pose_arm(
        armature,
        rig,
        1.0,
        (0.28, -0.24, float(hip_p[2]) + 0.62 + 0.08 * flail),
        (0.65, 0.7, -0.3),
        (0.0, 0.0, 0.0),
        0.15,
        14.0,
        4.0,
    )
    _pose_arm(
        armature,
        rig,
        -1.0,
        (-0.28, -0.24, float(hip_p[2]) + 0.62 - 0.08 * flail),
        (-0.65, 0.7, -0.3),
        (0.0, 0.0, 0.0),
        0.15,
        14.0,
        4.0,
    )


def _eval_cower_idle(armature: ArmatureSpec, rig: HumanoidRigMap, u: float) -> None:
    _reset(armature)
    bones = {b.name: b for b in armature.bones}
    az = _ankle_z(bones, rig)
    shiver = 0.012 * math.sin(4.0 * math.pi * u)
    bones[rig.hips].translation = (shiver, 0.06, -0.34)
    bones[rig.hips].rotation_deg = (26.0, 0.0, 0.0)
    _set_spine(bones, rig, 34.0, 0.0, 0.0)
    _set_head(armature, rig, 28.0, 0.0, 6.0 * math.sin(2.0 * math.pi * u))
    _plant_feet(armature, rig, (0.16, -0.08, az), (-0.16, 0.08, az), 8.0, -8.0)
    # Arms wrapped tightly over head/face to shield from debris
    _pose_arm(
        armature,
        rig,
        1.0,
        (0.08, -0.22, 1.30),
        (0.65, 0.4, -0.4),
        (0.0, 0.0, 0.0),
        0.55,
        12.0,
        10.0,
    )
    _pose_arm(
        armature,
        rig,
        -1.0,
        (-0.08, -0.22, 1.32),
        (-0.65, 0.4, -0.4),
        (0.0, 0.0, 0.0),
        0.55,
        12.0,
        10.0,
    )


def _eval_hit_flinch(armature: ArmatureSpec, rig: HumanoidRigMap, u: float) -> None:
    _reset(armature)
    bones = {b.name: b for b in armature.bones}
    az = _ankle_z(bones, rig)
    impact = (
        math.sin(math.pi * _smoothstep(0.0, 0.35, u))
        if u < 0.35
        else (1.0 - _smoothstep(0.35, 1.0, u))
    )
    bones[rig.hips].translation = (0.0, 0.10 * impact, -0.05 - 0.06 * impact)
    bones[rig.hips].rotation_deg = (-14.0 * impact, 0.0, -10.0 * impact)
    _set_spine(bones, rig, -22.0 * impact, -6.0 * impact, 14.0 * impact)
    _set_head(armature, rig, -26.0 * impact, 8.0 * impact, 18.0 * impact)
    _plant_feet(armature, rig, (0.145, -0.10, az), (-0.155, 0.14, az), 6.0, -16.0)
    _pose_arm(
        armature,
        rig,
        1.0,
        (0.28 + 0.14 * impact, -0.14, 1.04 + 0.16 * impact),
        (0.6, 0.7, -0.3),
        (0.0, 0.0, 0.0),
        0.25,
        6.0,
        0.0,
    )
    _pose_arm(
        armature,
        rig,
        -1.0,
        (-0.28 - 0.14 * impact, -0.10, 1.02 + 0.18 * impact),
        (-0.6, 0.7, -0.3),
        (0.0, 0.0, 0.0),
        0.25,
        6.0,
        0.0,
    )


def _eval_stunned_loop(armature: ArmatureSpec, rig: HumanoidRigMap, u: float) -> None:
    _reset(armature)
    bones = {b.name: b for b in armature.bones}
    az = _ankle_z(bones, rig)
    sw = math.sin(2.0 * math.pi * u)
    cw = math.cos(2.0 * math.pi * u)
    bones[rig.hips].translation = (0.065 * sw, 0.03 * cw, -0.08)
    bones[rig.hips].rotation_deg = (6.0 + 5.0 * cw, 6.0 * sw, -8.0 * cw)
    _set_spine(bones, rig, 10.0 + 6.0 * cw, 8.0 * sw, 10.0 * cw)
    _set_head(armature, rig, 14.0 * cw, 12.0 * sw, 18.0 * sw)
    _plant_feet(armature, rig, (0.17, -0.06, az), (-0.17, 0.08, az), 8.0, -14.0)
    _pose_arm(
        armature,
        rig,
        1.0,
        (0.16, -0.16, 1.50 + 0.04 * cw),
        (0.6, 0.6, -0.3),
        (0.0, 0.0, 0.0),
        0.35,
        6.0,
        6.0,
    )
    _pose_arm(
        armature,
        rig,
        -1.0,
        (-0.30 - 0.05 * sw, -0.14, 0.96),
        (-0.5, 0.8, -0.4),
        (0.0, 0.0, 0.0),
        0.25,
        2.0,
        2.0,
    )


def _eval_cheer_victory(armature: ArmatureSpec, rig: HumanoidRigMap, u: float) -> None:
    _reset(armature)
    bones = {b.name: b for b in armature.bones}
    az = _ankle_z(bones, rig)
    pump = math.sin(4.0 * math.pi * u)
    bones[rig.hips].translation = (0.0, 0.0, -0.03 + 0.03 * abs(pump))
    bones[rig.hips].rotation_deg = (-4.0, 0.0, 0.0)
    _set_spine(bones, rig, -8.0 - 4.0 * pump, 0.0, 4.0 * math.sin(2.0 * math.pi * u))
    _set_head(armature, rig, -14.0, 0.0, 8.0 * math.sin(2.0 * math.pi * u))
    _plant_feet(armature, rig, (0.14, -0.04, az), (-0.14, 0.04, az))
    _pose_arm(
        armature,
        rig,
        1.0,
        (0.34, -0.18, 1.84 + 0.10 * pump),
        (0.6, 0.5, -0.4),
        (0.0, 0.0, 0.0),
        0.90,
        16.0,
        2.0,
    )
    _pose_arm(
        armature,
        rig,
        -1.0,
        (-0.34, -0.18, 1.84 - 0.10 * pump),
        (-0.6, 0.5, -0.4),
        (0.0, 0.0, 0.0),
        0.90,
        16.0,
        2.0,
    )


# ============================================================================
# CATEGORY 4: MELEE COMBAT
# ============================================================================


def _eval_melee_idle(armature: ArmatureSpec, rig: HumanoidRigMap, u: float) -> None:
    _reset(armature)
    bones = {b.name: b for b in armature.bones}
    az = _ankle_z(bones, rig)
    br = math.sin(2.0 * math.pi * u)
    bones[rig.hips].translation = (0.0, 0.0, -0.065 + 0.008 * br)
    bones[rig.hips].rotation_deg = (5.0, 0.0, -16.0)
    _set_spine(bones, rig, 6.0 - 1.5 * br, 0.0, 12.0)
    _set_head(armature, rig, -2.0, 0.0, 4.0 * math.sin(2.0 * math.pi * u))
    _plant_feet(armature, rig, (0.16, -0.15, az), (-0.16, 0.16, az), 8.0, -24.0)
    # Left shield hand forward-chest; Right weapon hand low-ready at side
    _pose_arm(
        armature,
        rig,
        1.0,
        (0.18, -0.32, 1.18 + 0.015 * br),
        (0.55, 0.85, -0.4),
        (0.0, 10.0, -20.0),
        0.85,
        4.0,
        8.0,
    )
    _pose_arm(
        armature,
        rig,
        -1.0,
        (-0.28, -0.22, 0.94 + 0.015 * br),
        (-0.55, 0.85, -0.4),
        (-15.0, 0.0, 0.0),
        0.92,
        2.0,
        2.0,
    )


def _eval_charge_run(armature: ArmatureSpec, rig: HumanoidRigMap, u: float) -> None:
    gait = GAIT_PRESETS["explosive_sprint"]
    _walk_legs_base(armature, rig, u, gait, u / gait.cadence)
    _, pose_m, _ = compute_armature_matrices(armature)
    hip_p = pose_m[rig.hips][:3, 3]
    # Left shield tucked forward, Right weapon held aggressively forward-up
    _pose_arm(
        armature,
        rig,
        1.0,
        (0.20, -0.36, float(hip_p[2]) + 0.24),
        (0.55, 0.85, -0.4),
        (0.0, 10.0, -20.0),
        0.88,
        4.0,
        8.0,
    )
    _pose_arm(
        armature,
        rig,
        -1.0,
        (-0.24, -0.44, float(hip_p[2]) + 0.36),
        (-0.55, 0.85, -0.3),
        (-25.0, 0.0, 0.0),
        0.95,
        8.0,
        8.0,
    )


def _eval_slash_attack_1(armature: ArmatureSpec, rig: HumanoidRigMap, u: float) -> None:
    """Horizontal sword/axe slash: wind-up right -> explosive twist left -> recovery."""
    _reset(armature)
    bones = {b.name: b for b in armature.bones}
    az = _ankle_z(bones, rig)
    if u < 0.35:
        w = _smoothstep(0.0, 0.35, u)
        twist = _lerp(0.0, -32.0, w)
        hx, hy, hz = _lerp(-0.28, -0.48, w), _lerp(-0.22, 0.06, w), _lerp(0.96, 1.24, w)
    elif u < 0.65:
        w = _smoothstep(0.35, 0.65, u)
        twist = _lerp(-32.0, 42.0, w)
        hx, hy, hz = _lerp(-0.48, 0.32, w), _lerp(0.06, -0.44, w), _lerp(1.24, 1.08, w)
    else:
        w = _smoothstep(0.65, 1.0, u)
        twist = _lerp(42.0, 0.0, w)
        hx, hy, hz = _lerp(0.32, -0.28, w), _lerp(-0.44, -0.22, w), _lerp(1.08, 0.96, w)

    bones[rig.hips].translation = (0.0, -0.05 * math.sin(math.pi * u), -0.075)
    bones[rig.hips].rotation_deg = (8.0, 0.0, -14.0 + 0.45 * twist)
    _set_spine(bones, rig, 10.0, 0.0, 12.0 + 0.65 * twist)
    _set_head(armature, rig, 2.0, 0.0, 0.0)
    _plant_feet(armature, rig, (0.16, -0.18, az), (-0.16, 0.16, az), 8.0, -24.0)
    _pose_arm(
        armature, rig, -1.0, (hx, hy, hz), (-0.65, 0.6, -0.2), (-20.0, 0.0, 0.0), 0.95, 6.0, 6.0
    )
    _pose_arm(
        armature,
        rig,
        1.0,
        (0.24, -0.28, 1.14),
        (0.6, 0.85, -0.4),
        (0.0, 10.0, -20.0),
        0.85,
        4.0,
        6.0,
    )


def _eval_slash_attack_2(armature: ArmatureSpec, rig: HumanoidRigMap, u: float) -> None:
    """Vertical overhead chop: high wind-up -> heavy downward cleave -> recovery."""
    _reset(armature)
    bones = {b.name: b for b in armature.bones}
    az = _ankle_z(bones, rig)
    if u < 0.38:
        w = _smoothstep(0.0, 0.38, u)
        pitch = _lerp(6.0, -10.0, w)
        hx, hy, hz = _lerp(-0.26, -0.16, w), _lerp(-0.22, -0.04, w), _lerp(0.98, 1.82, w)
    elif u < 0.64:
        w = _smoothstep(0.38, 0.64, u)
        pitch = _lerp(-10.0, 26.0, w)
        hx, hy, hz = _lerp(-0.16, 0.04, w), _lerp(-0.04, -0.48, w), _lerp(1.82, 0.68, w)
    else:
        w = _smoothstep(0.64, 1.0, u)
        pitch = _lerp(26.0, 6.0, w)
        hx, hy, hz = _lerp(0.04, -0.26, w), _lerp(-0.48, -0.22, w), _lerp(0.68, 0.98, w)

    bones[rig.hips].translation = (
        0.0,
        -0.04 * math.sin(math.pi * u),
        -0.075 - 0.04 * max(0.0, pitch / 26.0),
    )
    bones[rig.hips].rotation_deg = (6.0 + 0.3 * pitch, 0.0, -14.0)
    _set_spine(bones, rig, pitch, 0.0, 10.0)
    _set_head(armature, rig, 2.0, 0.0, 0.0)
    _plant_feet(armature, rig, (0.16, -0.18, az), (-0.16, 0.16, az), 8.0, -24.0)
    _pose_arm(
        armature, rig, -1.0, (hx, hy, hz), (-0.45, 0.85, -0.2), (-25.0, 0.0, 0.0), 0.95, 10.0, 6.0
    )
    _pose_arm(
        armature,
        rig,
        1.0,
        (0.24, -0.26, 1.14),
        (0.6, 0.85, -0.4),
        (0.0, 10.0, -20.0),
        0.85,
        4.0,
        6.0,
    )


def _eval_bash_gate(armature: ArmatureSpec, rig: HumanoidRigMap, u: float) -> None:
    """Two-handed heavy gate/wall slam."""
    _reset(armature)
    bones = {b.name: b for b in armature.bones}
    az = _ankle_z(bones, rig)
    if u < 0.42:
        w = _smoothstep(0.0, 0.42, u)
        pitch = _lerp(4.0, -14.0, w)
        hy, hz = _lerp(-0.24, 0.04, w), _lerp(1.06, 1.84, w)
    elif u < 0.68:
        w = _smoothstep(0.42, 0.68, u)
        pitch = _lerp(-14.0, 28.0, w)
        hy, hz = _lerp(0.04, -0.52, w), _lerp(1.84, 0.88, w)
    else:
        w = _smoothstep(0.68, 1.0, u)
        pitch = _lerp(28.0, 4.0, w)
        hy, hz = _lerp(-0.52, -0.24, w), _lerp(0.88, 1.06, w)

    bones[rig.hips].translation = (0.0, -0.06 * math.sin(math.pi * u), -0.08)
    bones[rig.hips].rotation_deg = (6.0 + 0.35 * pitch, 0.0, 0.0)
    _set_spine(bones, rig, pitch, 0.0, 0.0)
    _set_head(armature, rig, 4.0, 0.0, 0.0)
    _plant_feet(armature, rig, (0.16, -0.18, az), (-0.16, 0.16, az), 8.0, -16.0)
    _pose_arm(
        armature, rig, 1.0, (0.12, hy, hz), (0.45, 0.85, -0.2), (0.0, 10.0, -20.0), 0.96, 12.0, 8.0
    )
    _pose_arm(
        armature,
        rig,
        -1.0,
        (-0.12, hy, hz),
        (-0.45, 0.85, -0.2),
        (0.0, -10.0, 20.0),
        0.96,
        12.0,
        8.0,
    )


def _eval_block_idle(armature: ArmatureSpec, rig: HumanoidRigMap, u: float) -> None:
    _reset(armature)
    bones = {b.name: b for b in armature.bones}
    az = _ankle_z(bones, rig)
    br = math.sin(2.0 * math.pi * u)
    bones[rig.hips].translation = (0.0, 0.02, -0.095 + 0.005 * br)
    bones[rig.hips].rotation_deg = (10.0, 0.0, -22.0)
    _set_spine(bones, rig, 12.0, 0.0, 14.0)
    _set_head(armature, rig, 6.0, 0.0, 6.0)
    _plant_feet(armature, rig, (0.16, -0.16, az), (-0.17, 0.18, az), 8.0, -26.0)
    # Left shield braced directly in front of chest/chin; Right tucked behind shield
    _pose_arm(
        armature,
        rig,
        1.0,
        (0.04, -0.38, 1.24 + 0.01 * br),
        (0.65, 0.7, -0.3),
        (0.0, 14.0, -30.0),
        0.90,
        6.0,
        10.0,
    )
    _pose_arm(
        armature,
        rig,
        -1.0,
        (-0.28, -0.14, 0.96),
        (-0.55, 0.85, -0.4),
        (0.0, 0.0, 0.0),
        0.92,
        2.0,
        2.0,
    )


# ============================================================================
# CATEGORY 5: RANGED COMBAT
# ============================================================================


def _eval_bow_pose(
    armature: ArmatureSpec, rig: HumanoidRigMap, draw_t: float, breath: float = 0.0
) -> None:
    _reset(armature)
    bones = {b.name: b for b in armature.bones}
    az = _ankle_z(bones, rig)
    bones[rig.hips].translation = (0.0, 0.0, -0.045)
    bones[rig.hips].rotation_deg = (2.0, 0.0, -45.0)
    _set_spine(bones, rig, 3.0, 0.0, -18.0)
    _set_head(armature, rig, 0.0, 0.0, 12.0)
    _plant_feet(armature, rig, (0.16, -0.12, az), (-0.16, 0.14, az), 12.0, -35.0)
    # Left bow arm extends straight toward target (-Y); Right draw hand pulls back to cheek!
    _pose_arm(
        armature,
        rig,
        1.0,
        (0.04, -0.66, 1.38 + 0.008 * breath),
        (0.6, 0.4, -0.6),
        (0.0, 0.0, 0.0),
        0.82,
        4.0,
        10.0,
    )
    rx = _lerp(-0.10, -0.18, draw_t)
    ry = _lerp(-0.56, -0.14, draw_t)
    rz = _lerp(1.36, 1.44 + 0.006 * breath, draw_t)
    _pose_arm(
        armature, rig, -1.0, (rx, ry, rz), (-0.85, 0.5, 0.15), (0.0, 0.0, 0.0), 0.78, 6.0, -4.0
    )


def _eval_bow_draw(armature: ArmatureSpec, rig: HumanoidRigMap, u: float) -> None:
    _eval_bow_pose(armature, rig, _smoothstep(0.10, 0.92, u))


def _eval_bow_aim_idle(armature: ArmatureSpec, rig: HumanoidRigMap, u: float) -> None:
    _eval_bow_pose(armature, rig, 1.0, breath=math.sin(2.0 * math.pi * u))


def _eval_bow_release(armature: ArmatureSpec, rig: HumanoidRigMap, u: float) -> None:
    recoil = (
        math.sin(math.pi * _smoothstep(0.0, 0.45, u))
        if u < 0.45
        else (1.0 - _smoothstep(0.45, 1.0, u))
    )
    _eval_bow_pose(armature, rig, 1.0)
    _pose_arm(
        armature,
        rig,
        -1.0,
        (-0.22 - 0.08 * recoil, -0.10 + 0.12 * recoil, 1.44),
        (-0.85, 0.5, 0.15),
        (0.0, 0.0, 0.0),
        0.20,
        6.0,
        -6.0,
    )


def _eval_throw_stone(armature: ArmatureSpec, rig: HumanoidRigMap, u: float) -> None:
    """Overhand throw downward from castle wall."""
    _reset(armature)
    bones = {b.name: b for b in armature.bones}
    az = _ankle_z(bones, rig)
    if u < 0.38:
        w = _smoothstep(0.0, 0.38, u)
        pitch, twist = _lerp(4.0, -14.0, w), _lerp(0.0, -28.0, w)
        hx, hy, hz = _lerp(-0.24, -0.34, w), _lerp(-0.12, 0.14, w), _lerp(1.04, 1.72, w)
        curl = 0.75
    elif u < 0.65:
        w = _smoothstep(0.38, 0.65, u)
        pitch, twist = _lerp(-14.0, 24.0, w), _lerp(-28.0, 26.0, w)
        hx, hy, hz = _lerp(-0.34, 0.08, w), _lerp(0.14, -0.52, w), _lerp(1.72, 1.02, w)
        curl = 0.15
    else:
        w = _smoothstep(0.65, 1.0, u)
        pitch, twist = _lerp(24.0, 4.0, w), _lerp(26.0, 0.0, w)
        hx, hy, hz = _lerp(0.08, -0.24, w), _lerp(-0.52, -0.12, w), _lerp(1.02, 1.04, w)
        curl = 0.25

    bones[rig.hips].translation = (0.0, 0.0, -0.055)
    bones[rig.hips].rotation_deg = (4.0 + 0.25 * pitch, 0.0, -14.0 + 0.4 * twist)
    _set_spine(bones, rig, pitch, 0.0, 10.0 + 0.6 * twist)
    _set_head(armature, rig, 14.0, 0.0, 0.0)
    _plant_feet(armature, rig, (0.16, -0.18, az), (-0.16, 0.16, az), 8.0, -22.0)
    _pose_arm(
        armature, rig, -1.0, (hx, hy, hz), (-0.65, 0.6, -0.2), (0.0, 0.0, 0.0), curl, 8.0, 4.0
    )
    _pose_arm(
        armature, rig, 1.0, (0.24, -0.28, 1.12), (0.6, 0.85, -0.4), (0.0, 0.0, 0.0), 0.35, 2.0, 4.0
    )


# ============================================================================
# CATEGORY 6: SIEGE OPERATION
# ============================================================================


def _eval_push_ram(armature: ArmatureSpec, rig: HumanoidRigMap, u: float) -> None:
    gait = GaitParams(
        speed=1.35,
        cadence=0.78,
        duty_factor=0.65,
        hip_drop=-0.12,
        bounce_amp=0.022,
        sway_amp=0.032,
        hip_roll_deg=4.2,
        hip_yaw_deg=6.0,
        spine_lean_deg=32.0,
        spine_twist_deg=4.0,
        foot_lift=0.09,
        stance_width=0.135,
        heel_strike_deg=-10.0,
        toe_off_deg=42.0,
        arm_swing_fwd=0.2,
        arm_swing_back=-0.1,
        arm_height=0.0,
        arm_out=0.05,
        elbow_out=0.4,
        fist_curl=0.8,
    )
    _walk_legs_base(armature, rig, u, gait, u / gait.cadence)
    _, pose_m, _ = compute_armature_matrices(armature)
    hip_p = pose_m[rig.hips][:3, 3]
    _pose_arm(
        armature,
        rig,
        1.0,
        (0.24, -0.58, float(hip_p[2]) + 0.18),
        (0.55, 0.85, -0.4),
        (-20.0, 0.0, 0.0),
        0.85,
        6.0,
        12.0,
    )
    _pose_arm(
        armature,
        rig,
        -1.0,
        (-0.24, -0.58, float(hip_p[2]) + 0.18),
        (-0.55, 0.85, -0.4),
        (-20.0, 0.0, 0.0),
        0.85,
        6.0,
        12.0,
    )


def _eval_crank_wheel(armature: ArmatureSpec, rig: HumanoidRigMap, u: float) -> None:
    _reset(armature)
    bones = {b.name: b for b in armature.bones}
    az = _ankle_z(bones, rig)
    ang = 2.0 * math.pi * u
    cy, cz = -0.38 + 0.16 * math.cos(ang), 1.06 + 0.16 * math.sin(ang)
    bones[rig.hips].translation = (0.0, 0.04 * math.cos(ang), -0.065)
    bones[rig.hips].rotation_deg = (10.0 + 4.0 * math.cos(ang), 0.0, 0.0)
    _set_spine(bones, rig, 14.0 + 6.0 * math.cos(ang), 0.0, 4.0 * math.sin(ang))
    _set_head(armature, rig, 12.0, 0.0, 0.0)
    _plant_feet(armature, rig, (0.16, -0.10, az), (-0.16, 0.10, az))
    _pose_arm(
        armature, rig, 1.0, (0.18, cy, cz), (0.55, 0.85, -0.4), (0.0, 0.0, 0.0), 0.90, 4.0, 8.0
    )
    _pose_arm(
        armature, rig, -1.0, (-0.18, cy, cz), (-0.55, 0.85, -0.4), (0.0, 0.0, 0.0), 0.90, 4.0, 8.0
    )


def _eval_pull_lever(armature: ArmatureSpec, rig: HumanoidRigMap, u: float) -> None:
    _reset(armature)
    bones = {b.name: b for b in armature.bones}
    az = _ankle_z(bones, rig)
    if u < 0.35:
        w = _smoothstep(0.0, 0.35, u)
        hy, hz, pitch = _lerp(-0.22, -0.52, w), _lerp(1.02, 1.24, w), _lerp(4.0, 22.0, w)
    elif u < 0.72:
        w = _smoothstep(0.35, 0.72, u)
        hy, hz, pitch = _lerp(-0.52, -0.14, w), _lerp(1.24, 0.96, w), _lerp(22.0, -16.0, w)
    else:
        w = _smoothstep(0.72, 1.0, u)
        hy, hz, pitch = _lerp(-0.14, -0.22, w), _lerp(0.96, 1.02, w), _lerp(-16.0, 4.0, w)

    bones[rig.hips].translation = (0.0, 0.08 * math.sin(math.pi * u), -0.075)
    bones[rig.hips].rotation_deg = (4.0 + 0.3 * pitch, 0.0, -12.0)
    _set_spine(bones, rig, pitch, 0.0, 10.0)
    _set_head(armature, rig, 6.0, 0.0, 0.0)
    _plant_feet(armature, rig, (0.16, -0.18, az), (-0.16, 0.18, az), 8.0, -22.0)
    _pose_arm(
        armature, rig, -1.0, (-0.14, hy, hz), (-0.55, 0.85, -0.3), (0.0, 0.0, 0.0), 0.92, 6.0, 6.0
    )
    _pose_arm(
        armature, rig, 1.0, (0.24, -0.18, 0.98), (0.55, 0.85, -0.4), (0.0, 0.0, 0.0), 0.35, 2.0, 2.0
    )


def _eval_load_ammo(armature: ArmatureSpec, rig: HumanoidRigMap, u: float) -> None:
    """Heave heavy stone boulder from ground/rack up into catapult sling."""
    _reset(armature)
    bones = {b.name: b for b in armature.bones}
    az = _ankle_z(bones, rig)
    if u < 0.42:
        w = _smoothstep(0.0, 0.42, u)
        hip_z = _lerp(-0.04, -0.34, w)
        sp_p = _lerp(4.0, 38.0, w)
        hy, hz = _lerp(-0.22, -0.38, w), _lerp(0.92, 0.32, w)
    elif u < 0.78:
        w = _smoothstep(0.42, 0.78, u)
        hip_z = _lerp(-0.34, -0.04, w)
        sp_p = _lerp(38.0, -8.0, w)
        hy, hz = _lerp(-0.38, -0.44, w), _lerp(0.32, 1.32, w)
    else:
        w = _smoothstep(0.78, 1.0, u)
        hip_z = -0.04
        sp_p = _lerp(-8.0, 4.0, w)
        hy, hz = _lerp(-0.44, -0.22, w), _lerp(1.32, 0.92, w)

    bones[rig.hips].translation = (0.0, 0.04, hip_z)
    bones[rig.hips].rotation_deg = (0.6 * sp_p, 0.0, 0.0)
    _set_spine(bones, rig, sp_p, 0.0, 0.0)
    _set_head(armature, rig, 10.0, 0.0, 0.0)
    _plant_feet(armature, rig, (0.16, -0.08, az), (-0.16, 0.08, az))
    _pose_arm(
        armature, rig, 1.0, (0.20, hy, hz), (0.55, 0.85, -0.3), (0.0, 12.0, -24.0), 0.78, 6.0, 8.0
    )
    _pose_arm(
        armature,
        rig,
        -1.0,
        (-0.20, hy, hz),
        (-0.55, 0.85, -0.3),
        (0.0, -12.0, 24.0),
        0.78,
        6.0,
        8.0,
    )


_EVALUATORS = {
    "Carry_Idle": _eval_carry_idle,
    "Carry_Walk": _eval_carry_walk,
    "Pickup_Item": _eval_pickup_item,
    "Drop_Item": _eval_drop_item,
    "Hammer_Wall": _eval_hammer_wall,
    "Hammer_Ground": _eval_hammer_ground,
    "Saw_Wood": _eval_saw_wood,
    "Inspect_Work": _eval_inspect_work,
    "Flee_Run": _eval_flee_run,
    "Cower_Idle": _eval_cower_idle,
    "Hit_Flinch": _eval_hit_flinch,
    "Stunned_Loop": _eval_stunned_loop,
    "Cheer_Victory": _eval_cheer_victory,
    "Melee_Idle": _eval_melee_idle,
    "Charge_Run": _eval_charge_run,
    "Slash_Attack_1": _eval_slash_attack_1,
    "Slash_Attack_2": _eval_slash_attack_2,
    "Bash_Gate": _eval_bash_gate,
    "Block_Idle": _eval_block_idle,
    "Bow_Draw": _eval_bow_draw,
    "Bow_Aim_Idle": _eval_bow_aim_idle,
    "Bow_Release": _eval_bow_release,
    "Throw_Stone": _eval_throw_stone,
    "Push_Ram": _eval_push_ram,
    "Crank_Wheel": _eval_crank_wheel,
    "Pull_Lever": _eval_pull_lever,
    "Load_Ammo": _eval_load_ammo,
}


def evaluate_named_action(armature: ArmatureSpec, rig: HumanoidRigMap, name: str, u: float) -> None:
    """Evaluate any of the 27 named gameplay actions at normalized progress u in [0, 1]."""
    fn = _EVALUATORS.get(name)
    if fn is None:
        raise ValueError(f"Unknown action {name!r}")
    fn(armature, rig, u)


def bake_named_action_to_timeline(
    scene: SceneDocument,
    armature_obj: SceneObject,
    action_name: str,
    *,
    fps: int = 30,
) -> int:
    """Bake any of the 27 named actions onto `scene.timeline` and return frame count."""
    assert armature_obj.armature is not None
    if action_name not in ACTION_SPECS:
        raise ValueError(f"Unknown action {action_name!r}")
    dur_s, is_loop = ACTION_SPECS[action_name]
    rig = detect_humanoid_rig(armature_obj.armature)
    armature_obj.armature.skinning_mode = "dqs"

    n_frames = max(4, int(round(dur_s * fps)))
    scene.timeline.fps = int(fps)
    scene.timeline.frame_start = 1
    scene.timeline.frame_end = n_frames
    scene.timeline.current_frame = 1
    scene.timeline.tracks = [t for t in scene.timeline.tracks if t.target_id != armature_obj.id]

    bone_tracks: dict[str, list[Keyframe]] = {}
    for fi in range(n_frames):
        fnum = fi + 1
        u = (fi / float(n_frames)) if is_loop else (fi / float(max(1, n_frames - 1)))
        evaluate_named_action(armature_obj.armature, rig, action_name, u)
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
