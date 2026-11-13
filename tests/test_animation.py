"""Tests for animation F-Curve interpolation and TimelineState scene evaluation."""

from __future__ import annotations

import pytest

from frfruit.animation import FCurveTrack, Keyframe, apply_animation_frame, evaluate_fcurve
from frfruit.core.scene import CameraSpec, LightSpec, MaterialSpec, SceneDocument, SceneObject
from frfruit.core.types import Transform
from frfruit.geometry.primitives import make_cube


def test_fcurve_interpolation_modes() -> None:
    empty = FCurveTrack(target_id="c1", property_path="location.x", keyframes=[])
    assert evaluate_fcurve(empty, 5) is None

    # Unsorted input frames; constant then linear then bezier
    track_const = FCurveTrack(
        target_id="c1",
        property_path="location.x",
        keyframes=[
            Keyframe(frame=10, value=10.0, interpolation="constant"),
            Keyframe(frame=0, value=0.0, interpolation="constant"),
        ],
    )
    assert evaluate_fcurve(track_const, -5) == 0.0
    assert evaluate_fcurve(track_const, 5) == 0.0
    assert evaluate_fcurve(track_const, 15) == 10.0

    track_lin = FCurveTrack(
        target_id="c1",
        property_path="location.x",
        keyframes=[
            Keyframe(frame=0, value=0.0, interpolation="linear"),
            Keyframe(frame=10, value=20.0, interpolation="linear"),
        ],
    )
    assert evaluate_fcurve(track_lin, 2.5) == pytest.approx(5.0, abs=1e-6)
    assert evaluate_fcurve(track_lin, 5.0) == pytest.approx(10.0, abs=1e-6)

    track_bez = FCurveTrack(
        target_id="c1",
        property_path="location.x",
        keyframes=[
            Keyframe(frame=0, value=0.0, interpolation="bezier", easing="ease_in_out"),
            Keyframe(frame=10, value=10.0, interpolation="bezier", easing="ease_in_out"),
        ],
    )
    assert evaluate_fcurve(track_bez, 5.0) == pytest.approx(5.0, abs=1e-6)
    # Slow-in means at 10% time, value < linear 1.0
    assert evaluate_fcurve(track_bez, 1.0) < 1.0


def test_apply_animation_frame_transforms_lights_cameras_and_materials() -> None:
    doc = SceneDocument.new_default("AnimTest")
    doc.materials["mat_test"] = MaterialSpec(
        id="mat_test", name="AnimMat", base_color=(0.0, 0.2, 0.4, 1.0)
    )
    doc.objects["box"] = SceneObject(
        id="box",
        name="Box",
        kind="mesh",
        material_id="mat_test",
        transform=Transform(location=(0.0, 0.0, 0.0)),
        mesh=make_cube().to_payload(),
    )
    doc.objects["cam"] = SceneObject(
        id="cam",
        name="Cam",
        kind="camera",
        transform=Transform(location=(0.0, -5.0, 2.0)),
        camera=CameraSpec(fov_deg=45.0, target=(0.0, 0.0, 0.0)),
    )
    doc.objects["sun"] = SceneObject(
        id="sun",
        name="Sun",
        kind="light",
        light=LightSpec(kind="sun", intensity=2.0),
    )

    doc.timeline.tracks = [
        FCurveTrack(
            target_id="box",
            property_path="location.x",
            keyframes=[
                Keyframe(frame=1, value=0.0, interpolation="linear"),
                Keyframe(frame=11, value=10.0, interpolation="linear"),
            ],
        ),
        FCurveTrack(
            target_id="box",
            property_path="rotation_deg.z",
            keyframes=[
                Keyframe(frame=1, value=0.0, interpolation="linear"),
                Keyframe(frame=11, value=180.0, interpolation="linear"),
            ],
        ),
        FCurveTrack(
            target_id="box",
            property_path="scale.y",
            keyframes=[
                Keyframe(frame=1, value=1.0, interpolation="linear"),
                Keyframe(frame=11, value=3.0, interpolation="linear"),
            ],
        ),
        FCurveTrack(
            target_id="cam",
            property_path="fov_deg",
            keyframes=[
                Keyframe(frame=1, value=40.0, interpolation="linear"),
                Keyframe(frame=11, value=60.0, interpolation="linear"),
            ],
        ),
        FCurveTrack(
            target_id="sun",
            property_path="intensity",
            keyframes=[
                Keyframe(frame=1, value=1.0, interpolation="linear"),
                Keyframe(frame=11, value=5.0, interpolation="linear"),
            ],
        ),
        FCurveTrack(
            target_id="mat_test",
            property_path="metallic",
            keyframes=[
                Keyframe(frame=1, value=0.0, interpolation="linear"),
                Keyframe(frame=11, value=1.0, interpolation="linear"),
            ],
        ),
    ]

    mid = apply_animation_frame(doc, 6)
    assert mid.timeline.current_frame == 6
    assert mid.objects["box"].transform.location[0] == pytest.approx(5.0, abs=1e-6)
    assert mid.objects["box"].transform.rotation_deg[2] == pytest.approx(90.0, abs=1e-6)
    assert mid.objects["box"].transform.scale[1] == pytest.approx(2.0, abs=1e-6)
    assert mid.objects["cam"].camera is not None
    assert mid.objects["cam"].camera.fov_deg == pytest.approx(50.0, abs=1e-6)
    assert mid.objects["sun"].light is not None
    assert mid.objects["sun"].light.intensity == pytest.approx(3.0, abs=1e-6)
    assert mid.materials["mat_test"].metallic == pytest.approx(0.5, abs=1e-6)

    # Original scene must remain untouched
    assert doc.timeline.current_frame == 1
    assert doc.objects["box"].transform.location[0] == 0.0
