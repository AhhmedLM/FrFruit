"""Animation timeline, F-Curves, and frame evaluator."""

from frfruit.animation.armature import (
    auto_rig_mesh,
    compute_armature_matrices,
    deform_skinned_mesh,
)
from frfruit.animation.evaluator import apply_animation_frame
from frfruit.animation.timeline import FCurveTrack, Keyframe, TimelineState, evaluate_fcurve

__all__ = [
    "FCurveTrack",
    "Keyframe",
    "TimelineState",
    "apply_animation_frame",
    "auto_rig_mesh",
    "compute_armature_matrices",
    "deform_skinned_mesh",
    "evaluate_fcurve",
]
