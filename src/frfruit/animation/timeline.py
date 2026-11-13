"""Timeline, keyframe, and F-Curve interpolation models."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class Keyframe(BaseModel):
    """A single scalar keyframe along an F-Curve."""

    frame: int
    value: float
    interpolation: Literal["constant", "linear", "bezier"] = "bezier"
    easing: Literal["auto", "ease_in", "ease_out", "ease_in_out"] = "ease_in_out"


class FCurveTrack(BaseModel):
    """Animation curve targeting a scalar property path on a scene entity."""

    target_id: str
    property_path: str = Field(
        description=(
            "Dotted property path, e.g. 'location.x', 'rotation_deg.z', "
            "'scale.y', 'intensity', 'fov_deg', 'base_color.r', 'emission_strength'."
        )
    )
    keyframes: list[Keyframe] = Field(default_factory=list)


class TimelineState(BaseModel):
    """Global animation timeline configuration and tracks."""

    fps: int = Field(default=24, ge=1, le=120)
    frame_start: int = 1
    frame_end: int = 60
    current_frame: int = 1
    tracks: list[FCurveTrack] = Field(default_factory=list)


def _apply_easing(t: float, easing: str) -> float:
    t = max(0.0, min(1.0, t))
    if easing == "ease_in":
        return t * t * t
    if easing == "ease_out":
        inv = 1.0 - t
        return 1.0 - inv * inv * inv
    # auto / ease_in_out smoothstep-cubic Hermite
    return t * t * (3.0 - 2.0 * t)


def evaluate_fcurve(track: FCurveTrack, frame: float) -> float | None:
    """Evaluate an FCurveTrack deterministically at a given frame number."""
    if not track.keyframes:
        return None
    kfs = sorted(track.keyframes, key=lambda k: k.frame)
    if frame <= kfs[0].frame:
        return float(kfs[0].value)
    if frame >= kfs[-1].frame:
        return float(kfs[-1].value)

    for i in range(len(kfs) - 1):
        k0 = kfs[i]
        k1 = kfs[i + 1]
        if k0.frame <= frame <= k1.frame:
            if k1.frame == k0.frame:
                return float(k1.value)
            if k0.interpolation == "constant":
                return float(k0.value)
            t = (frame - k0.frame) / (k1.frame - k0.frame)
            if k0.interpolation == "bezier":
                t = _apply_easing(t, k0.easing)
            return float(k0.value + (k1.value - k0.value) * t)
    return float(kfs[-1].value)
