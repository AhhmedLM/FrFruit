"""Physically-grounded AAA gameplay locomotion synthesizer for humanoid armatures.

Supports Mixamo (`mixamorig:*`), Unreal, and standard humanoid rigs in FrFruit Z-up
coordinates (character facing -Y, +X = character left, +Z = up). Uses analytical
2-bone IK for legs and arms, world-space foot-roll & toe-off locking (zero foot
sliding), dual-bounce center-of-mass dynamics, spinal counter-rotation, clavicle
drive, vestibular head stabilization, and realistic athletic hand/finger curling.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

from frfruit.animation.armature import (
    compute_armature_matrices,
    set_bone_world_rotation,
    solve_two_bone_ik,
)
from frfruit.animation.timeline import FCurveTrack, Keyframe
from frfruit.core.math3d import euler_deg_to_matrix

if TYPE_CHECKING:
    from frfruit.core.scene import ArmatureSpec, SceneDocument, SceneObject


@dataclass
class HumanoidRigMap:
    """Canonical bone names detected on a humanoid ArmatureSpec."""

    hips: str
    spine: str | None = None
    spine1: str | None = None
    spine2: str | None = None
    neck: str | None = None
    head: str | None = None

    left_upleg: str | None = None
    left_leg: str | None = None
    left_foot: str | None = None
    left_toe: str | None = None

    right_upleg: str | None = None
    right_leg: str | None = None
    right_foot: str | None = None
    right_toe: str | None = None

    left_shoulder: str | None = None
    left_arm: str | None = None
    left_forearm: str | None = None
    left_hand: str | None = None

    right_shoulder: str | None = None
    right_arm: str | None = None
    right_forearm: str | None = None
    right_hand: str | None = None

    left_fingers: list[tuple[str, int]] | None = None  # (bone_name, joint_idx 1..3)
    right_fingers: list[tuple[str, int]] | None = None


def detect_humanoid_rig(armature: ArmatureSpec) -> HumanoidRigMap:
    """Auto-detect Mixamo / standard humanoid bone names from an ArmatureSpec."""
    names = [b.name for b in armature.bones]
    lower_map = {n.lower(): n for n in names}

    def find(*candidates: str) -> str | None:
        for cand in candidates:
            c = cand.lower()
            if c in lower_map:
                return lower_map[c]
            for ln, orig in lower_map.items():
                short = ln.split(":")[-1]
                if short == c:
                    return orig
        return None

    hips = find("hips", "pelvis", "root")
    if hips is None:
        raise ValueError("Could not detect Hips/Pelvis root bone in armature.")

    left_fingers: list[tuple[str, int]] = []
    right_fingers: list[tuple[str, int]] = []
    finger_keys = ("thumb", "index", "middle", "ring", "pinky")
    for orig in names:
        short = orig.lower().split(":")[-1]
        for fk in finger_keys:
            if fk in short and ("hand" in short or "finger" in short or fk in short):
                j_idx = 0
                for d in ("1", "2", "3"):
                    if short.endswith(d):
                        j_idx = int(d)
                        break
                if j_idx == 0:
                    continue
                if "left" in short:
                    left_fingers.append((orig, j_idx))
                elif "right" in short:
                    right_fingers.append((orig, j_idx))
                break

    return HumanoidRigMap(
        hips=hips,
        spine=find("spine", "spine_01"),
        spine1=find("spine1", "spine_02"),
        spine2=find("spine2", "spine_03", "chest"),
        neck=find("neck", "neck_01"),
        head=find("head"),
        left_upleg=find("leftupleg", "thigh_l", "upperleg_l"),
        left_leg=find("leftleg", "calf_l", "lowerleg_l", "shin_l"),
        left_foot=find("leftfoot", "foot_l"),
        left_toe=find("lefttoebase", "ball_l", "toe_l"),
        right_upleg=find("rightupleg", "thigh_r", "upperleg_r"),
        right_leg=find("rightleg", "calf_r", "lowerleg_r", "shin_r"),
        right_foot=find("rightfoot", "foot_r"),
        right_toe=find("righttoebase", "ball_r", "toe_r"),
        left_shoulder=find("leftshoulder", "clavicle_l"),
        left_arm=find("leftarm", "upperarm_l"),
        left_forearm=find("leftforearm", "lowerarm_l", "forearm_l"),
        left_hand=find("lefthand", "hand_l"),
        right_shoulder=find("rightshoulder", "clavicle_r"),
        right_arm=find("rightarm", "upperarm_r"),
        right_forearm=find("rightforearm", "lowerarm_r", "forearm_r"),
        right_hand=find("righthand", "hand_r"),
        left_fingers=left_fingers,
        right_fingers=right_fingers,
    )


@dataclass
class GaitParams:
    """Biomechanical parameters defining a single locomotion gait style."""

    speed: float  # Forward speed in m/s
    cadence: float  # Full gait cycles (L+R) per second
    duty_factor: float  # Fraction of cycle each foot spends on ground (>0.5 walk, <0.5 run)
    hip_drop: float  # Mean hip vertical drop from rest T-pose (meters, negative = lower)
    bounce_amp: float  # Vertical CoM bounce amplitude (meters)
    sway_amp: float  # Lateral CoM weight shift over stance foot (meters)
    hip_roll_deg: float  # Pelvic Trendelenburg roll (degrees)
    hip_yaw_deg: float  # Pelvic twist yaw (degrees)
    spine_lean_deg: float  # Forward torso pitch (degrees, positive = forward lean)
    spine_twist_deg: float  # Counter-rotation ribcage yaw (degrees)
    foot_lift: float  # Peak swing foot clearance height (meters)
    stance_width: float  # Half-width between feet (meters)
    heel_strike_deg: float  # Dorsiflexion at heel strike (negative = toes up)
    toe_off_deg: float  # Plantarflexion at toe-off (positive = heel lifted)
    arm_swing_fwd: float  # Forward hand reach offset relative to shoulder (meters)
    arm_swing_back: float  # Backward hand reach offset relative to shoulder (meters)
    arm_height: float  # Mean hand Z offset relative to hips (meters)
    arm_out: float  # Lateral hand offset from shoulder X (meters)
    elbow_out: float  # Lateral elbow pole strength
    fist_curl: float  # 0..1 finger clench intensity


GAIT_PRESETS: dict[str, GaitParams] = {
    "combat_idle": GaitParams(
        speed=0.0,
        cadence=0.35,
        duty_factor=1.0,
        hip_drop=-0.058,
        bounce_amp=0.008,
        sway_amp=0.012,
        hip_roll_deg=1.5,
        hip_yaw_deg=15.0,
        spine_lean_deg=8.5,
        spine_twist_deg=-11.0,
        foot_lift=0.0,
        stance_width=0.155,
        heel_strike_deg=0.0,
        toe_off_deg=0.0,
        arm_swing_fwd=0.20,
        arm_swing_back=0.10,
        arm_height=0.08,
        arm_out=0.085,
        elbow_out=0.40,
        fist_curl=0.62,
    ),
    "heavy_walk": GaitParams(
        speed=1.62,
        cadence=0.86,  # 1.72 steps/sec -> heavy grounded 0.94m steps
        duty_factor=0.63,
        hip_drop=-0.030,
        bounce_amp=0.032,
        sway_amp=0.038,
        hip_roll_deg=5.6,
        hip_yaw_deg=9.5,
        spine_lean_deg=6.5,
        spine_twist_deg=12.0,
        foot_lift=0.12,
        stance_width=0.112,
        heel_strike_deg=-20.0,
        toe_off_deg=40.0,
        arm_swing_fwd=0.28,
        arm_swing_back=-0.24,
        arm_height=-0.12,
        arm_out=0.065,
        elbow_out=0.32,
        fist_curl=0.45,
    ),
    "combat_jog": GaitParams(
        speed=3.85,
        cadence=1.38,  # 2.76 steps/sec -> punchy athletic 1.40m steps
        duty_factor=0.44,
        hip_drop=-0.070,
        bounce_amp=0.055,
        sway_amp=0.022,
        hip_roll_deg=4.6,
        hip_yaw_deg=12.0,
        spine_lean_deg=15.5,
        spine_twist_deg=18.5,
        foot_lift=0.28,
        stance_width=0.090,
        heel_strike_deg=-12.0,
        toe_off_deg=50.0,
        arm_swing_fwd=0.34,
        arm_swing_back=-0.12,
        arm_height=0.04,
        arm_out=0.045,
        elbow_out=0.26,
        fist_curl=0.78,
    ),
    "explosive_sprint": GaitParams(
        speed=6.95,
        cadence=1.82,  # 3.64 steps/sec -> explosive 1.91m strides with true flight phase!
        duty_factor=0.33,
        hip_drop=-0.098,
        bounce_amp=0.072,
        sway_amp=0.014,
        hip_roll_deg=4.0,
        hip_yaw_deg=16.0,
        spine_lean_deg=27.5,
        spine_twist_deg=25.0,
        foot_lift=0.44,
        stance_width=0.072,
        heel_strike_deg=-5.0,
        toe_off_deg=58.0,
        arm_swing_fwd=0.42,
        arm_swing_back=-0.16,
        arm_height=0.08,
        arm_out=0.035,
        elbow_out=0.22,
        fist_curl=0.96,
    ),
}


def _lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * t


def _smoothstep(edge0: float, edge1: float, x: float) -> float:
    if edge1 <= edge0:
        return 1.0 if x >= edge1 else 0.0
    t = max(0.0, min(1.0, (x - edge0) / (edge1 - edge0)))
    return t * t * (3.0 - 2.0 * t)


def _blend_gait(a: GaitParams, b: GaitParams, t: float) -> GaitParams:
    u = max(0.0, min(1.0, float(t)))
    return GaitParams(
        speed=_lerp(a.speed, b.speed, u),
        cadence=_lerp(a.cadence, b.cadence, u),
        duty_factor=_lerp(a.duty_factor, b.duty_factor, u),
        hip_drop=_lerp(a.hip_drop, b.hip_drop, u),
        bounce_amp=_lerp(a.bounce_amp, b.bounce_amp, u),
        sway_amp=_lerp(a.sway_amp, b.sway_amp, u),
        hip_roll_deg=_lerp(a.hip_roll_deg, b.hip_roll_deg, u),
        hip_yaw_deg=_lerp(a.hip_yaw_deg, b.hip_yaw_deg, u),
        spine_lean_deg=_lerp(a.spine_lean_deg, b.spine_lean_deg, u),
        spine_twist_deg=_lerp(a.spine_twist_deg, b.spine_twist_deg, u),
        foot_lift=_lerp(a.foot_lift, b.foot_lift, u),
        stance_width=_lerp(a.stance_width, b.stance_width, u),
        heel_strike_deg=_lerp(a.heel_strike_deg, b.heel_strike_deg, u),
        toe_off_deg=_lerp(a.toe_off_deg, b.toe_off_deg, u),
        arm_swing_fwd=_lerp(a.arm_swing_fwd, b.arm_swing_fwd, u),
        arm_swing_back=_lerp(a.arm_swing_back, b.arm_swing_back, u),
        arm_height=_lerp(a.arm_height, b.arm_height, u),
        arm_out=_lerp(a.arm_out, b.arm_out, u),
        elbow_out=_lerp(a.elbow_out, b.elbow_out, u),
        fist_curl=_lerp(a.fist_curl, b.fist_curl, u),
    )


@dataclass
class LocomotionFrameState:
    """Evaluated kinematic state for one frame of a locomotion sequence."""

    time_s: float
    label: str
    root_y: float  # World Y root motion (moves in -Y as character travels forward)
    speed: float  # Instantaneous m/s
    accel: float  # Instantaneous forward acceleration m/s^2
    phase: float  # Continuous gait phase (cycles)
    gait: GaitParams
    idle_weight: float  # 1.0 = pure combat_idle, 0.0 = active locomotion
    brake_weight: float  # 0..1 heavy hard-stop braking intensity
    brake_progress: float  # 0..1 progress through hard stop
    cadence_hz: float


def build_showcase_schedule(fps: int = 30, duration_s: float = 18.0) -> list[LocomotionFrameState]:
    """Build a seamless, thrilling AAA gameplay locomotion showcase timeline.

    Timeline choreography (18.0s @ 30fps = 540 frames):
      0.0s - 2.6s : COMBAT IDLE (athletic breathing, weight shift, vigilant scan)
      2.6s - 6.0s : HEAVY TACTICAL WALK (purposeful start -> heavy grounded walk)
      6.0s - 9.4s : COMBAT JOG / RUN (punchy acceleration -> fast tactical run)
      9.4s - 13.0s: EXPLOSIVE SPRINT (full-throttle lean, airborne flight phase, hard drive)
      13.0s - 15.6s: HEAVY HARD STOP (aggressive skid-plant deceleration -> athletic recoil)
      15.6s - 18.0s: COMBAT IDLE RECOVERY (heavy post-sprint breathing & ready stance)
    """
    n_frames = max(2, int(round(duration_s * fps)))
    dt = 1.0 / float(fps)

    p_idle = GAIT_PRESETS["combat_idle"]
    p_walk = GAIT_PRESETS["heavy_walk"]
    p_jog = GAIT_PRESETS["combat_jog"]
    p_sprint = GAIT_PRESETS["explosive_sprint"]

    states: list[LocomotionFrameState] = []
    root_y = 0.0
    phase = 0.0
    prev_speed = 0.0

    for fi in range(n_frames):
        t = fi * dt
        brake_w = 0.0
        brake_prog = 0.0

        if t < 2.2:
            gait = p_idle
            idle_w = 1.0
            label = "COMBAT IDLE"
        elif t < 3.1:
            u = _smoothstep(2.2, 3.1, t)
            gait = _blend_gait(p_idle, p_walk, u)
            idle_w = 1.0 - u
            label = "START -> HEAVY WALK"
        elif t < 5.7:
            gait = p_walk
            idle_w = 0.0
            label = "HEAVY TACTICAL WALK"
        elif t < 6.5:
            u = _smoothstep(5.7, 6.5, t)
            gait = _blend_gait(p_walk, p_jog, u)
            idle_w = 0.0
            label = "ACCELERATE -> COMBAT JOG"
        elif t < 9.1:
            gait = p_jog
            idle_w = 0.0
            label = "COMBAT JOG / RUN"
        elif t < 9.9:
            u = _smoothstep(9.1, 9.9, t)
            gait = _blend_gait(p_jog, p_sprint, u)
            idle_w = 0.0
            label = "BURST -> EXPLOSIVE SPRINT"
        elif t < 13.0:
            gait = p_sprint
            idle_w = 0.0
            label = "EXPLOSIVE SPRINT"
        elif t < 14.35:
            # Heavy hard stop: aggressive braking skid over 1.35s
            u = (t - 13.0) / 1.35
            brake_prog = u
            # Fast exponential speed kill with dramatic early plant
            spd_mult = (1.0 - _smoothstep(0.0, 0.72, u)) ** 1.4
            gait = _blend_gait(p_idle, p_sprint, spd_mult)
            # Peak brake crouch around u=0.38, smooth recovery toward idle by u=1.0
            brake_w = math.sin(math.pi * min(1.0, u / 0.92)) ** 0.75 if u < 0.92 else 0.0
            idle_w = _smoothstep(0.35, 0.95, u)
            label = "HEAVY HARD STOP"
        else:
            # Post-sprint recovery idle with heavier breathing
            gait = p_idle
            idle_w = 1.0
            label = "COMBAT IDLE (RECOVERY)"

        speed = gait.speed
        accel = (speed - prev_speed) / dt if fi > 0 else 0.0
        # Clamp numerical accel spikes for smooth torso inertia
        accel = max(-10.0, min(10.0, accel))
        prev_speed = speed

        if fi > 0:
            root_y -= speed * dt  # Character faces & travels along -Y
            # Advance gait phase cleanly when moving
            eff_cadence = gait.cadence if idle_w < 0.98 else 0.0
            phase += eff_cadence * dt

        states.append(
            LocomotionFrameState(
                time_s=t,
                label=label,
                root_y=root_y,
                speed=speed,
                accel=accel,
                phase=phase,
                gait=gait,
                idle_weight=idle_w,
                brake_weight=brake_w,
                brake_progress=brake_prog,
                cadence_hz=gait.cadence if idle_w < 0.98 else 0.0,
            )
        )

    return states


def build_cycle_schedule(
    style: str = "heavy_walk",
    fps: int = 30,
    duration_s: float = 4.0,
) -> list[LocomotionFrameState]:
    """Build a steady-state locomotion schedule for a single gait preset."""
    if style not in GAIT_PRESETS:
        raise ValueError(f"Unknown gait style {style!r}. Available: {sorted(GAIT_PRESETS)}")
    gait = GAIT_PRESETS[style]
    n_frames = max(2, int(round(duration_s * fps)))
    dt = 1.0 / float(fps)
    is_idle = style == "combat_idle"
    label = style.replace("_", " ").upper()

    states: list[LocomotionFrameState] = []
    root_y = 0.0
    phase = 0.0
    for fi in range(n_frames):
        t = fi * dt
        if fi > 0:
            root_y -= gait.speed * dt
            if not is_idle:
                phase += gait.cadence * dt
        states.append(
            LocomotionFrameState(
                time_s=t,
                label=label,
                root_y=root_y,
                speed=gait.speed,
                accel=0.0,
                phase=phase,
                gait=gait,
                idle_weight=1.0 if is_idle else 0.0,
                brake_weight=0.0,
                brake_progress=0.0,
                cadence_hz=0.0 if is_idle else gait.cadence,
            )
        )
    return states


def _reset_armature_pose(armature: ArmatureSpec) -> None:
    for b in armature.bones:
        b.rotation_deg = (0.0, 0.0, 0.0)
        b.translation = (0.0, 0.0, 0.0)


def _eval_foot_trajectory(
    u: float,
    side: float,  # +1.0 Left, -1.0 Right
    gait: GaitParams,
    ankle_to_ball_rest: np.ndarray,
    ankle_ground_z: float,
    toe_ground_z: float,
) -> tuple[np.ndarray, float, float]:
    """Compute armature-local (ankle_pos, foot_pitch_deg, toe_pitch_deg) for cycle phase u in [0,1).

    Guarantees zero ground slip during stance [0, D) and realistic heel-strike -> flat ->
    ball-pivoted toe-off -> high-knee swing arc.
    """
    d = max(0.25, min(0.95, gait.duty_factor))
    stride_len = gait.speed / max(0.15, gait.cadence)
    stance_travel = d * stride_len
    y_lead = -0.47 * stance_travel  # Touchdown ahead of hip (-Y)
    y_trail = y_lead + stance_travel  # Toe-off behind hip (+Y)

    foot_x = side * gait.stance_width
    heel_strike_end = 0.16
    heel_lift_start = 0.62

    if u < d:
        # STANCE PHASE: ball/sole contact point moves backward at exact +speed (zero world slide!)
        s = u / d
        y_ball = y_lead + s * stance_travel
        z_ball = toe_ground_z

        if s < heel_strike_end:
            # Heel strike: toes start raised (dorsiflexed), roll smoothly flat
            hs_t = s / heel_strike_end
            foot_pitch = gait.heel_strike_deg * (1.0 - _smoothstep(0.0, 1.0, hs_t))
            # Pivot around heel so heel touches Z=0 and ball lowers onto Z=toe_ground_z
            r_foot = euler_deg_to_matrix(foot_pitch, 0.0, 0.0)
            heel_offset = np.array([0.0, 0.045, -ankle_ground_z], dtype=np.float64)
            heel_world = np.array([foot_x, y_ball - ankle_to_ball_rest[1] + 0.045, 0.0])
            ankle_pos = heel_world - r_foot @ heel_offset
            ankle_pos[2] = max(ankle_ground_z, ankle_pos[2])
            toe_pitch = 0.0
        elif s < heel_lift_start:
            # Full flat-foot loading & mid-stance (sole 100% flush on Z=0)
            foot_pitch = 0.0
            toe_pitch = 0.0
            ankle_pos = np.array(
                [foot_x, y_ball - ankle_to_ball_rest[1], ankle_ground_z], dtype=np.float64
            )
        else:
            # Push-off / Toe-off: ball of foot STAYS LOCKED at (foot_x, y_ball, toe_ground_z)
            # while heel lifts forcefully via +Rx pitch!
            to_t = (s - heel_lift_start) / max(1e-5, 1.0 - heel_lift_start)
            eased = to_t * to_t * (2.0 - to_t)
            foot_pitch = gait.toe_off_deg * eased
            r_foot = euler_deg_to_matrix(foot_pitch, 0.0, 0.0)
            ball_pos = np.array([foot_x, y_ball, z_ball], dtype=np.float64)
            ankle_pos = ball_pos - r_foot @ ankle_to_ball_rest
            # Toes counter-bend to stay flat on the floor during propulsion!
            toe_pitch = 0.0
    else:
        # SWING PHASE: powerful pull-through from y_trail back to y_lead
        sw = (u - d) / max(1e-5, 1.0 - d)  # 0..1
        # Snappy forward progression curve (fast knee drive, slight reach ease-out before strike)
        prog = sw * sw * (3.0 - 2.0 * sw)
        prog = 0.65 * prog + 0.35 * math.sin(0.5 * math.pi * sw)
        y_ball = _lerp(y_trail, y_lead, prog)

        # Asymmetric vertical lift arc (rises steeply after toe-off, descends cleanly to strike)
        lift_curve = math.sin(math.pi * (sw**0.78))
        # Slight outward clearance during mid-swing so legs never clip
        lat_clear = side * 0.018 * math.sin(math.pi * sw)

        # Foot pitch transitions: plantarflexed push-off -> neutral passing -> dorsiflexed strike
        if sw < 0.35:
            t_p = _smoothstep(0.0, 0.35, sw)
            foot_pitch = _lerp(gait.toe_off_deg, 18.0, t_p)
            toe_pitch = _lerp(18.0, 5.0, t_p)
        else:
            t_p = _smoothstep(0.35, 1.0, sw)
            foot_pitch = _lerp(18.0, gait.heel_strike_deg, t_p)
            toe_pitch = _lerp(5.0, -4.0, t_p)

        r_foot = euler_deg_to_matrix(foot_pitch, 0.0, 0.0)
        z_ball = toe_ground_z + gait.foot_lift * lift_curve
        ball_pos = np.array([foot_x + lat_clear, y_ball, z_ball], dtype=np.float64)
        ankle_pos = ball_pos - r_foot @ ankle_to_ball_rest
        ankle_pos[2] = max(ankle_ground_z, ankle_pos[2])

    return ankle_pos, foot_pitch, toe_pitch


def evaluate_locomotion_pose(
    armature: ArmatureSpec,
    rig: HumanoidRigMap,
    state: LocomotionFrameState,
    *,
    apply_root_motion_to_hips: bool = False,
) -> dict[str, Any]:
    """Pose `armature` in place for `state` using 2-bone IK, foot-roll, spine lag & fist curl."""
    _reset_armature_pose(armature)
    bones = {b.name: b for b in armature.bones}

    gait = state.gait
    t = state.time_s
    ph = state.phase
    w_idle = state.idle_weight
    w_move = 1.0 - w_idle
    w_brake = state.brake_weight

    # Reference rest measurements directly from the imported rig
    l_foot_rest = (
        np.asarray(bones[rig.left_foot].head, dtype=np.float64)
        if rig.left_foot
        else np.array([0.09, 0.02, 0.105])
    )
    l_toe_rest = (
        np.asarray(bones[rig.left_toe].head, dtype=np.float64)
        if rig.left_toe
        else np.array([0.09, -0.11, 0.033])
    )
    ankle_to_ball_rest = l_toe_rest - l_foot_rest
    ankle_to_ball_rest[0] = 0.0  # symmetric sagittal offset
    ankle_ground_z = float(l_foot_rest[2])
    toe_ground_z = float(l_toe_rest[2])

    # ---- 1. CENTER OF MASS (HIPS) DYNAMICS ----
    # Double-bounce frequency (2x gait cycle), asymmetric weighty drop on impact
    two_pi_2 = 4.0 * math.pi * ph
    bounce_wave = -math.cos(two_pi_2 - 0.45) - 0.18 * math.sin(2.0 * two_pi_2)
    # Lateral sway (1x gait cycle): shifts OVER left leg (+X) during left stance (u in [0, D])
    sway_wave = math.sin(2.0 * math.pi * ph + 0.15)
    # Pelvic roll & yaw (1x gait cycle)
    roll_wave = math.sin(2.0 * math.pi * ph)
    yaw_wave = math.cos(2.0 * math.pi * ph)

    # Idle athletic breathing & subtle combat weight shift
    breath_rate = 0.55 if state.time_s > 14.0 else 0.36  # heavier post-sprint breathing!
    breath_amp = 1.45 if state.time_s > 14.0 else 1.0
    breath = math.sin(2.0 * math.pi * breath_rate * t)
    idle_sway = math.sin(2.0 * math.pi * 0.22 * t) * 0.016

    hip_tx = w_move * (gait.sway_amp * sway_wave) + w_idle * idle_sway
    hip_ty = (state.root_y if apply_root_motion_to_hips else 0.0) + w_brake * 0.055
    hip_tz = (
        gait.hip_drop
        + w_move * (gait.bounce_amp * bounce_wave)
        + w_idle * (0.008 * breath * breath_amp)
        - w_brake * 0.115  # Deep heavy weight absorption crouch during hard stop!
    )

    # Acceleration-driven pitch inertia (lean forward on burst, lean back on hard stop!)
    accel_pitch = max(-12.0, min(8.0, state.accel * 1.15))

    relaxed_idle = gait.arm_height < 0.0
    idle_hip_rx = 1.8 if relaxed_idle else 4.5
    idle_hip_rz = -gait.hip_yaw_deg if relaxed_idle else -14.0
    idle_spine_pitch = (
        (gait.spine_lean_deg - 1.0 * breath * breath_amp)
        if relaxed_idle
        else (4.5 - 1.4 * breath * breath_amp)
    )
    idle_spine_yaw = -gait.spine_twist_deg if relaxed_idle else 11.0
    idle_spine_roll = 0.8 if relaxed_idle else 2.0

    hip_rx = (
        w_move * (gait.spine_lean_deg * 0.30 + 2.5 * math.sin(two_pi_2))
        + w_idle * idle_hip_rx
        - w_brake * 12.0
    )
    hip_ry = w_move * (gait.hip_roll_deg * roll_wave) + w_idle * (-1.2 + 0.8 * idle_sway * 50.0)
    hip_rz = w_move * (-gait.hip_yaw_deg * yaw_wave) + w_idle * idle_hip_rz - w_brake * 16.0

    bones[rig.hips].translation = (float(hip_tx), float(hip_ty), float(hip_tz))
    bones[rig.hips].rotation_deg = (float(hip_rx), float(hip_ry), float(hip_rz))

    # ---- 2. SPINE COUNTER-ROTATION, FORWARD LEAN & BREATHING (WITH PHASE LAG!) ----
    # Spine counter-twists against pelvis with slight organic follow-through lag (-0.22 rad)
    spine_yaw_wave = math.cos(2.0 * math.pi * ph - 0.22)
    spine_roll_wave = -math.sin(2.0 * math.pi * ph - 0.18)

    total_spine_pitch = (
        w_move * (gait.spine_lean_deg * 0.72 + accel_pitch)
        + w_idle * idle_spine_pitch
        - w_brake * 15.0  # Throw upper torso back against momentum during hard brake!
    )
    total_spine_yaw = (
        w_move * (gait.spine_twist_deg * spine_yaw_wave) + w_idle * idle_spine_yaw + w_brake * 12.0
    )
    total_spine_roll = (
        w_move * (gait.hip_roll_deg * 0.85 * spine_roll_wave)
        + w_idle * idle_spine_roll
        + w_brake * 5.0
    )

    spine_chain = [
        (rig.spine, 0.28, 0.25, 0.30),
        (rig.spine1, 0.34, 0.35, 0.35),
        (rig.spine2, 0.38, 0.40, 0.35),
    ]
    for bname, wp, wy, wr in spine_chain:
        if bname and bname in bones:
            bones[bname].rotation_deg = (
                float(total_spine_pitch * wp),
                float(total_spine_roll * wr),
                float(total_spine_yaw * wy),
            )

    # ---- 3. VESTIBULAR HEAD STABILIZATION & VIGILANT SCAN ----
    # Keep eyes locked forward on travel direction even while torso leans & twists aggressively
    head_scan_yaw = w_idle * ((5.0 if relaxed_idle else 11.0) * math.sin(2.0 * math.pi * 0.18 * t))
    head_target_pitch = w_move * (3.5 + 0.15 * gait.spine_lean_deg) + w_idle * (-1.5 - 0.8 * breath)
    head_target_roll = w_idle * (1.2 * math.sin(2.0 * math.pi * 0.22 * t))
    if rig.neck and rig.neck in bones:
        r_neck_w = euler_deg_to_matrix(
            head_target_pitch * 0.5, head_target_roll * 0.5, head_scan_yaw * 0.45
        )
        set_bone_world_rotation(armature, rig.neck, r_neck_w)
    if rig.head and rig.head in bones:
        r_head_w = euler_deg_to_matrix(head_target_pitch, head_target_roll, head_scan_yaw)
        set_bone_world_rotation(armature, rig.head, r_head_w)

    # ---- 4. EXACT 2-BONE LEG IK + FOOT ROLL & ZERO-SLIP GROUND LOCKING ----
    root_y_off = state.root_y if apply_root_motion_to_hips else 0.0

    if relaxed_idle:
        sw_idle = gait.stance_width
        idle_ankles = {
            1.0: np.array([sw_idle, root_y_off - 0.025, ankle_ground_z], dtype=np.float64),
            -1.0: np.array([-sw_idle, root_y_off + 0.035, ankle_ground_z], dtype=np.float64),
        }
        idle_foot_yaw = {1.0: 5.0, -1.0: -8.0}
    else:
        # Combat idle asymmetric planted foot positions (left foot forward, right foot back-angled)
        idle_ankles = {
            1.0: np.array([0.145, root_y_off - 0.135, ankle_ground_z], dtype=np.float64),
            -1.0: np.array([-0.155, root_y_off + 0.145, ankle_ground_z], dtype=np.float64),
        }
        idle_foot_yaw = {1.0: 7.0, -1.0: -22.0}

    # Hard-stop braking plant foot positions (left foot slammed forward, right trailing)
    brake_ankles = {
        1.0: np.array([0.165, root_y_off - 0.34, ankle_ground_z], dtype=np.float64),
        -1.0: np.array([-0.155, root_y_off + 0.24, ankle_ground_z], dtype=np.float64),
    }

    leg_configs = [
        (1.0, 0.0, rig.left_upleg, rig.left_leg, rig.left_foot, rig.left_toe),
        (-1.0, 0.5, rig.right_upleg, rig.right_leg, rig.right_foot, rig.right_toe),
    ]

    for side, ph_off, up_b, lo_b, ft_b, toe_b in leg_configs:
        if not (up_b and lo_b and ft_b):
            continue
        u = (ph + ph_off) % 1.0
        move_ankle, move_pitch, move_toe_pitch = _eval_foot_trajectory(
            u, side, gait, ankle_to_ball_rest, ankle_ground_z, toe_ground_z
        )
        move_ankle[1] += root_y_off

        # Blend smoothly between active gait, combat idle, and hard-stop braking stance
        ankle_t = move_ankle * w_move + idle_ankles[side] * w_idle
        foot_pitch = move_pitch * w_move
        toe_pitch = move_toe_pitch * w_move
        foot_yaw = side * 4.5 * w_move + idle_foot_yaw[side] * w_idle

        if w_brake > 1e-4:
            ankle_t = ankle_t * (1.0 - w_brake) + brake_ankles[side] * w_brake
            # Lead foot digs heel hard on early brake, slaps flat; trailing foot rolls on ball
            if side > 0:
                brake_pitch = -14.0 * max(0.0, 1.0 - state.brake_progress * 2.2)
            else:
                brake_pitch = 18.0 * w_brake
            foot_pitch = foot_pitch * (1.0 - w_brake) + brake_pitch * w_brake
            foot_yaw = foot_yaw * (1.0 - w_brake) + (side * 12.0) * w_brake
            toe_pitch = 0.0

        # Ensure ankle never sinks below floor
        ankle_t[2] = max(ankle_ground_z, float(ankle_t[2]))

        # Knee pole vector: points forward (-Y) with slight natural athletic knee-out (+side*0.18)
        pole_x = side * (0.14 + 0.14 * w_idle + 0.18 * w_brake)
        knee_pole = (pole_x, -1.0, 0.08)

        solve_two_bone_ik(
            armature,
            upper_bone_name=up_b,
            lower_bone_name=lo_b,
            end_bone_name=ft_b,
            target_pos=(float(ankle_t[0]), float(ankle_t[1]), float(ankle_t[2])),
            pole_dir=knee_pole,
            rest_bend_dir=(0.0, -1.0, 0.0),
            soft_ratio=0.996,
        )

        # World-lock Foot and ToeBase orientation so soles sit flush on Z=0 during stance!
        r_foot_w = euler_deg_to_matrix(foot_pitch, 0.0, foot_yaw)
        set_bone_world_rotation(armature, ft_b, r_foot_w)
        if toe_b and toe_b in bones:
            r_toe_w = euler_deg_to_matrix(toe_pitch, 0.0, foot_yaw)
            set_bone_world_rotation(armature, toe_b, r_toe_w)

    # ---- 5. CLAVICLES (SHOULDERS) + 2-BONE ARM IK + ATHLETIC HANDS ----
    # Recompute pose matrices so shoulders sit at their exact twisted/leaned torso positions!
    _, pose_now, _ = compute_armature_matrices(armature)
    spine2_pose = (
        pose_now[rig.spine2] if (rig.spine2 and rig.spine2 in pose_now) else pose_now[rig.hips]
    )
    chest_center = spine2_pose[:3, 3]

    arm_configs = [
        # Left arm swings OPPOSITE to Left leg (so +0.5 phase offset!)
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

    for side, arm_ph_off, sh_b, arm_b, fa_b, hd_b, fingers in arm_configs:
        if not (arm_b and fa_b and hd_b):
            continue

        # Smooth opposing arm swing wave (-1 = backswing, +1 = forward pump)
        swing = math.cos(2.0 * math.pi * (ph + arm_ph_off) - 0.14)
        swing_01 = 0.5 * (swing + 1.0)

        # Clavicle dynamic protraction/retraction + breathing elevation
        if sh_b and sh_b in bones:
            clav_elev = (
                w_move * (2.5 + 5.5 * max(0.0, swing) * (gait.speed / 6.8))
                + w_idle * (1.8 + 2.0 * breath * breath_amp)
                + w_brake * 3.5
            )
            clav_protract = w_move * (6.5 * swing * min(1.0, gait.speed / 4.2))
            # Left shoulder (+X): -Ry elevates UP, -Rz protracts FORWARD (-Y)
            bones[sh_b].rotation_deg = (
                0.0,
                float(-side * clav_elev),
                float(-side * clav_protract),
            )

        # Compute exact athletic hand target relative to current chest/hip world frame
        _, pose_arm, _ = compute_armature_matrices(armature)
        sh_pos = pose_arm[arm_b][:3, 3]
        hip_pos = pose_arm[rig.hips][:3, 3]

        # Locomotion arm pump target: true "cheek to hip-pocket" athletic sprint mechanic!
        fwd_reach = _lerp(gait.arm_swing_back, gait.arm_swing_fwd, swing_01)
        speed_norm = min(1.0, gait.speed / 5.5)
        arc_z = (
            gait.arm_height
            + 0.28 * max(0.0, swing) * speed_norm  # Forward hand rises high toward chest/cheek
            - 0.09 * max(0.0, -swing) * speed_norm  # Back hand drops cleanly to hip pocket!
        )
        inward = side * (0.045 * max(0.0, swing) * speed_norm)
        move_hand = np.array(
            [
                sh_pos[0] + side * gait.arm_out - inward,
                sh_pos[1] - fwd_reach,
                hip_pos[2] + arc_z,
            ],
            dtype=np.float64,
        )

        if relaxed_idle:
            # Natural relaxed standing: arms hang comfortably by the sides near the hips
            idle_hand = np.array(
                [
                    sh_pos[0] + side * gait.arm_out,
                    sh_pos[1]
                    - gait.arm_swing_fwd
                    + 0.008 * math.sin(2.0 * math.pi * 0.33 * t + side),
                    hip_pos[2] + gait.arm_height + 0.008 * breath * breath_amp,
                ],
                dtype=np.float64,
            )
        else:
            # Combat idle ready stance: athletic, grounded, non-cramped guard
            idle_fwd = 0.27 if side > 0 else 0.17
            idle_z = 0.15 if side > 0 else 0.03
            idle_out = 0.075 if side > 0 else 0.095
            idle_hand = np.array(
                [
                    sh_pos[0] + side * idle_out,
                    sh_pos[1] - idle_fwd + 0.012 * math.sin(2.0 * math.pi * 0.36 * t + side),
                    hip_pos[2] + idle_z + 0.016 * breath * breath_amp,
                ],
                dtype=np.float64,
            )

        # Hard-stop athletic counter-balance: right arm drives forward-low, left pulls down-back
        brake_hand = np.array(
            [
                sh_pos[0] + side * (0.14 if side > 0 else 0.07),
                sh_pos[1] - (0.32 if side < 0 else -0.10),
                hip_pos[2] + (0.02 if side > 0 else 0.12),
            ],
            dtype=np.float64,
        )

        hand_target = move_hand * w_move + idle_hand * w_idle
        if w_brake > 1e-4:
            hand_target = hand_target * (1.0 - w_brake) + brake_hand * w_brake

        # Elbow pole: always points backward (+Y), slightly outward (+side), and downward (-Z)
        elbow_pole = (side * gait.elbow_out, 1.0, -0.35)

        solve_two_bone_ik(
            armature,
            upper_bone_name=arm_b,
            lower_bone_name=fa_b,
            end_bone_name=hd_b,
            target_pos=(float(hand_target[0]), float(hand_target[1]), float(hand_target[2])),
            pole_dir=elbow_pole,
            rest_bend_dir=(0.0, 1.0, 0.0),
            soft_ratio=0.985,
        )

        # Clean neutral wrist alignment (palms naturally inward) + subtle dynamic follow-through
        wrist_lag = w_move * (-6.0 * swing)
        bones[hd_b].rotation_deg = (
            0.0,
            float(side * (2.0 if relaxed_idle else 4.0) * w_idle),
            float(side * wrist_lag),
        )

        # Curl fingers into realistic athletic/tactical semi-fists (or relaxed open hand in Idle)
        idle_curl = gait.fist_curl if relaxed_idle else GAIT_PRESETS["combat_idle"].fist_curl
        curl = gait.fist_curl * w_move + idle_curl * w_idle
        curl = min(1.0, curl + 0.25 * w_brake)
        if fingers:
            for fname, j_idx in fingers:
                if fname not in bones:
                    continue
                is_thumb = "thumb" in fname.lower()
                if is_thumb:
                    ang_y = (
                        side * (18.0 + 22.0 * curl) if j_idx == 1 else side * (12.0 + 24.0 * curl)
                    )
                    ang_z = -side * (24.0 * curl) if j_idx == 1 else 0.0
                    bones[fname].rotation_deg = (0.0, float(ang_y), float(ang_z))
                else:
                    # Fingers curl downward around local Y in T-pose (+Ry for left, -Ry for right!)
                    base_deg = {1: 42.0, 2: 68.0, 3: 52.0}.get(j_idx, 45.0)
                    bones[fname].rotation_deg = (0.0, float(side * base_deg * curl), 0.0)

    return {
        "label": state.label,
        "speed": state.speed,
        "root_y": state.root_y,
        "chest_pos": chest_center.tolist(),
    }


def bake_locomotion_to_timeline(
    scene: SceneDocument,
    armature_obj: SceneObject,
    schedule: list[LocomotionFrameState],
    *,
    fps: int = 30,
    apply_root_motion_to_hips: bool = True,
) -> list[dict[str, Any]]:
    """Bake a full locomotion schedule onto `scene.timeline` F-curves and return frame metadata."""
    assert armature_obj.armature is not None
    rig = detect_humanoid_rig(armature_obj.armature)
    armature_obj.armature.skinning_mode = "dqs"

    n_frames = len(schedule)
    scene.timeline.fps = int(fps)
    scene.timeline.frame_start = 1
    scene.timeline.frame_end = n_frames
    scene.timeline.current_frame = 1

    # Remove existing tracks for this armature
    scene.timeline.tracks = [t for t in scene.timeline.tracks if t.target_id != armature_obj.id]

    bone_tracks: dict[str, list[Keyframe]] = {}
    meta_list: list[dict[str, Any]] = []

    for idx, st in enumerate(schedule):
        fnum = idx + 1
        info = evaluate_locomotion_pose(
            armature_obj.armature,
            rig,
            st,
            apply_root_motion_to_hips=apply_root_motion_to_hips,
        )
        meta_list.append(info)
        for b in armature_obj.armature.bones:
            for ai, ax in enumerate(("x", "y", "z")):
                rv = float(b.rotation_deg[ai])
                pp_r = f"bone.{b.name}.rotation_deg.{ax}"
                bone_tracks.setdefault(pp_r, []).append(
                    Keyframe(frame=fnum, value=round(rv, 4), interpolation="linear")
                )
            if b.name == rig.hips:
                for ai, ax in enumerate(("x", "y", "z")):
                    tv = float(b.translation[ai])
                    pp_t = f"bone.{b.name}.translation.{ax}"
                    bone_tracks.setdefault(pp_t, []).append(
                        Keyframe(frame=fnum, value=round(tv, 5), interpolation="linear")
                    )

    for pp, kfs in bone_tracks.items():
        # Skip tracks that are identically zero across all frames to keep scene JSON lean
        if any(abs(k.value) > 1e-4 for k in kfs):
            scene.timeline.tracks.append(
                FCurveTrack(target_id=armature_obj.id, property_path=pp, keyframes=kfs)
            )

    return meta_list
