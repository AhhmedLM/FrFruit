"""Tests for the pure-CPU G-buffer renderer, shading modes, contact sheets, and GIF animations."""

from __future__ import annotations

import io

import numpy as np
from PIL import Image

from frfruit.animation import FCurveTrack, Keyframe
from frfruit.core.scene import MaterialSpec, SceneDocument, SceneObject
from frfruit.core.types import Transform
from frfruit.geometry.primitives import make_cube, make_plane, make_sphere
from frfruit.render import render_animation_sequence, render_multiview_sheet, render_single_view
from frfruit.render.materials import PRESET_MATERIALS


def _sample_render_scene(colliding: bool = False) -> SceneDocument:
    doc = SceneDocument.new_default("RenderScene")
    doc.materials["mat_apple"] = MaterialSpec(id="mat_apple", **PRESET_MATERIALS["apple_skin"])
    doc.materials["mat_gold"] = MaterialSpec(id="mat_gold", **PRESET_MATERIALS["gold"])
    doc.materials["mat_wood"] = MaterialSpec(id="mat_wood", **PRESET_MATERIALS["wood"])

    doc.objects["floor"] = SceneObject(
        id="floor",
        name="Floor",
        kind="mesh",
        material_id="mat_wood",
        transform=Transform(location=(0.0, 0.0, 0.0)),
        mesh=make_plane((4.0, 4.0), subdivisions=2).to_payload(),
    )
    doc.objects["cube"] = SceneObject(
        id="cube",
        name="GoldCube",
        kind="mesh",
        material_id="mat_gold",
        transform=Transform(location=(-0.7, 0.0, 0.5)),
        mesh=make_cube(1.0, origin="center").to_payload(),
    )
    sphere_x = -0.3 if colliding else 0.8
    doc.objects["sphere"] = SceneObject(
        id="sphere",
        name="AppleSphere",
        kind="mesh",
        material_id="mat_apple",
        transform=Transform(location=(sphere_x, 0.0, 0.55)),
        mesh=make_sphere(0.55, segments=16, origin="center").to_payload(),
    )
    return doc


def test_all_six_viewport_modes_produce_valid_images() -> None:
    scene = _sample_render_scene(colliding=True)
    for mode in ("pbr", "clay", "wireframe", "normals", "depth", "collision"):
        png_bytes, meta = render_single_view(
            scene,
            width=320,
            height=240,
            mode=mode,  # type: ignore[arg-type]
            show_bounds=True,
            show_labels=True,
        )
        assert png_bytes.startswith(b"\x89PNG\r\n\x1a\n")
        img = Image.open(io.BytesIO(png_bytes))
        assert img.size == (320, 240)
        arr = np.asarray(img, dtype=np.float32)
        assert arr.std() > 8.0, f"Mode {mode} rendered near-uniform image"
        assert meta["mode"] == mode
        if mode == "collision":
            assert len(meta["collisions"]) >= 1  # type: ignore[arg-type]


def test_multiview_sheet_and_animation_renders() -> None:
    scene = _sample_render_scene(colliding=False)
    sheet_png, meta = render_multiview_sheet(
        scene,
        tile_width=200,
        tile_height=150,
        mode="pbr",
        views=("front", "right", "top", "iso"),
    )
    sheet_img = Image.open(io.BytesIO(sheet_png))
    assert sheet_img.size == (400, 300)
    assert meta["views"] == ["front", "right", "top", "iso"]
    assert np.asarray(sheet_img, dtype=np.float32).std() > 10.0

    # Keyframe animation + timeline & turntable renders
    scene.timeline.frame_start = 1
    scene.timeline.frame_end = 16
    scene.timeline.tracks = [
        FCurveTrack(
            target_id="cube",
            property_path="rotation_deg.z",
            keyframes=[
                Keyframe(frame=1, value=0.0, interpolation="linear"),
                Keyframe(frame=16, value=90.0, interpolation="linear"),
            ],
        )
    ]

    for kind in ("turntable", "timeline"):
        strip_png, gif_bytes, anim_meta = render_animation_sequence(
            scene,
            kind=kind,  # type: ignore[arg-type]
            steps=6,
            width=160,
            height=120,
            mode="clay",
        )
        assert gif_bytes.startswith(b"GIF8")
        strip_img = Image.open(io.BytesIO(strip_png))
        # 4 cols x 2 rows for 6 frames
        assert strip_img.size == (640, 240)
        gif_img = Image.open(io.BytesIO(gif_bytes))
        assert getattr(gif_img, "n_frames", 1) == 6
        assert anim_meta["frames_rendered"] == 6


def test_rasterizer_solid_coverage_and_material_colors() -> None:
    """Regression check: triangles must solidly fill screen pixels (catch barycentric sign bugs)."""
    from frfruit.geometry.spatial import evaluate_scene_meshes
    from frfruit.render.camera import auto_frame_camera_object
    from frfruit.render.software import RenderConfig, SoftwareRenderer

    def render_colored_sphere(color: tuple[float, float, float, float]) -> np.ndarray:
        doc = SceneDocument.new_default("SolidCheck")
        doc.materials["mat_test"] = MaterialSpec(
            id="mat_test",
            name="SolidMat",
            base_color=color,
            metallic=0.0,
            roughness=0.45,
        )
        doc.objects["sphere"] = SceneObject(
            id="sphere",
            name="CenterSphere",
            kind="mesh",
            material_id="mat_test",
            transform=Transform(location=(0.0, 0.0, 0.0)),
            mesh=make_sphere(1.0, segments=20, origin="center").to_payload(),
        )
        ev = evaluate_scene_meshes(doc)
        cam = auto_frame_camera_object(
            ev["sphere"].bounds, direction="front", projection="orthographic", margin=1.15
        )
        cfg = RenderConfig(width=160, height=160, mode="pbr", show_grid=False, show_axes=False)
        img = SoftwareRenderer().render_image(doc, ev, cam, config=cfg)
        return np.asarray(img, dtype=np.float32)

    red_arr = render_colored_sphere((0.92, 0.12, 0.12, 1.0))
    blue_arr = render_colored_sphere((0.12, 0.22, 0.92, 1.0))

    # Background pixels are uniform slate sky; solid sphere should cover > 40% of 160x160 frame
    bg_corner = red_arr[0, 0]
    diff_pixels = np.linalg.norm(red_arr - bg_corner[None, None, :], axis=2) > 15.0
    coverage = float(np.mean(diff_pixels))
    assert coverage > 0.40, f"Expected >40% solid rasterization coverage, got {coverage:.3f}"

    # Center pixel must reflect red vs blue albedo cleanly
    red_center = red_arr[80, 80]
    blue_center = blue_arr[80, 80]
    assert red_center[0] > red_center[2] + 60.0, f"Expected red center pixel, got {red_center}"
    assert blue_center[2] > blue_center[0] + 60.0, f"Expected blue center pixel, got {blue_center}"
