"""Animation timeline, F-Curves, and frame evaluator."""

from frfruit.animation.armature import (
    auto_rig_mesh,
    compute_armature_matrices,
    deform_skinned_mesh,
    set_bone_world_rotation,
    solve_two_bone_ik,
)
from frfruit.animation.evaluator import apply_animation_frame
from frfruit.animation.locomotion import (
    GAIT_PRESETS,
    GaitParams,
    HumanoidRigMap,
    LocomotionFrameState,
    bake_locomotion_to_timeline,
    build_showcase_schedule,
    detect_humanoid_rig,
    evaluate_locomotion_pose,
)
from frfruit.animation.timeline import FCurveTrack, Keyframe, TimelineState, evaluate_fcurve

__all__ = [
    "FCurveTrack",
    "GAIT_PRESETS",
    "GaitParams",
    "HumanoidRigMap",
    "Keyframe",
    "LocomotionFrameState",
    "TimelineState",
    "apply_animation_frame",
    "auto_rig_mesh",
    "bake_locomotion_to_timeline",
    "build_showcase_schedule",
    "compute_armature_matrices",
    "deform_skinned_mesh",
    "detect_humanoid_rig",
    "evaluate_fcurve",
    "evaluate_locomotion_pose",
    "set_bone_world_rotation",
    "solve_two_bone_ik",
]
