"""Tests for HDR post-processing pipeline and CPU BVH PathTracerRenderer."""

from __future__ import annotations

import io

import numpy as np
from PIL import Image

from frfruit.core.scene import (
    LightSpec,
    MaterialSpec,
    PostProcessConfig,
    SceneDocument,
    SceneObject,
)
from frfruit.core.types import Transform
from frfruit.geometry.primitives import create_primitive_mesh
from frfruit.render.backend import get_renderer
from frfruit.render.postprocess import apply_post_processing
from frfruit.render.viewport import render_single_view


def test_postprocess_pipeline_stages_modify_image_predictably() -> None:
    h, w = 96, 128
    # Dark linear scene with a bright HDR highlight box in the center + step edge depth
    hdr = np.full((h, w, 3), 0.12, dtype=np.float32)
    hdr[36:60, 48:80] = 4.5  # Bright HDR emissive core

    depth = np.full((h, w), 10.0, dtype=np.float32)
    depth[30:70, 40:88] = 2.0  # Foreground block creates strong depth step for SSAO + DOF

    # 1. Raw off / none tone mapper preserves basic sRGB mapping as float32 [0, 1]
    raw_f32 = apply_post_processing(hdr, depth, PostProcessConfig(enabled=False))
    assert raw_f32.shape == (h, w, 3) and raw_f32.dtype == np.float32
    assert float(raw_f32.min()) >= 0.0 and float(raw_f32.max()) <= 1.0

    # 2. HDR Bloom leaks light outside the central emissive box
    no_bloom = apply_post_processing(
        hdr,
        depth,
        PostProcessConfig(
            bloom_enabled=False, ssao_enabled=False, vignette_intensity=0.0, fxaa_enabled=False
        ),
    )
    with_bloom = apply_post_processing(
        hdr,
        depth,
        PostProcessConfig(
            bloom_enabled=True,
            bloom_threshold=0.8,
            bloom_intensity=1.2,
            bloom_radius=6.0,
            ssao_enabled=False,
            vignette_intensity=0.0,
            fxaa_enabled=False,
        ),
    )
    # Halo pixel just outside box (y=32, x=48) must be brighter with Bloom
    assert float(with_bloom[32, 48].mean()) > float(no_bloom[32, 48].mean()) + 0.04

    # 3. SSAO darkens pixels recessed inside a concave groove/crevice
    crevice_depth = np.full((h, w), 5.0, dtype=np.float32)
    crevice_depth[40:56, 56:72] = 6.2  # Recessed crevice within bilateral horizon range
    no_ssao = apply_post_processing(
        np.full((h, w, 3), 0.6, dtype=np.float32),
        crevice_depth,
        PostProcessConfig(
            bloom_enabled=False,
            ssao_enabled=False,
            vignette_intensity=0.0,
            fxaa_enabled=False,
            sharpen=0.0,
        ),
    )
    with_ssao = apply_post_processing(
        np.full((h, w, 3), 0.6, dtype=np.float32),
        crevice_depth,
        PostProcessConfig(
            bloom_enabled=False,
            ssao_enabled=True,
            ssao_strength=1.0,
            ssao_radius=5.0,
            vignette_intensity=0.0,
            fxaa_enabled=False,
            sharpen=0.0,
        ),
    )
    assert float(with_ssao[48, 64].mean()) < float(no_ssao[48, 64].mean()) - 0.03

    # 4. Vignette darkens corners relative to center
    vig = apply_post_processing(
        np.full((h, w, 3), 0.7, dtype=np.float32),
        None,
        PostProcessConfig(
            bloom_enabled=False, ssao_enabled=False, vignette_intensity=0.8, fxaa_enabled=False
        ),
    )
    assert float(vig[0, 0].mean()) < float(vig[h // 2, w // 2].mean()) - 0.10

    # 5. DOF blurs out-of-focus regions when focus=2.0 and background is at 10.0
    bg_step = np.full((h, w, 3), 0.15, dtype=np.float32)
    bg_step[:, 10:15] = 1.5  # crisp vertical stripe at depth=10.0
    raw_bg_step = apply_post_processing(bg_step, depth, PostProcessConfig(enabled=False))
    dof_img = apply_post_processing(
        bg_step,
        depth,
        PostProcessConfig(
            bloom_enabled=False,
            ssao_enabled=False,
            vignette_intensity=0.0,
            dof_enabled=True,
            dof_focus_distance=2.0,
            dof_aperture=1.5,
            fxaa_enabled=False,
        ),
    )
    # Out-of-focus stripe spreads into neighboring column 8
    assert float(dof_img[10, 8].mean()) > float(raw_bg_step[10, 8].mean()) + 0.02


def test_pathtracer_renders_shadows_reflections_refraction_and_modes() -> None:
    sc = SceneDocument.new_default("RaytraceTest")
    sc.render_engine = "raytrace"
    sc.post_process = PostProcessConfig(tone_mapper="aces", bloom_enabled=True, ssao_enabled=False)

    floor = create_primitive_mesh("plane", size=8.0, segments=4)
    sc.materials["mat_floor"] = MaterialSpec(
        id="mat_floor", name="Floor", base_color=(0.82, 0.82, 0.84, 1.0), roughness=0.7
    )
    sc.objects["floor"] = SceneObject(
        id="floor",
        name="Floor",
        kind="mesh",
        transform=Transform(location=(0.0, 0.0, 0.0)),
        mesh=floor.to_payload(),
        material_id="mat_floor",
    )

    # Mirror chrome sphere + red matte cube + refractive glass orb
    orb = create_primitive_mesh("uv_sphere", radius=0.65, segments=16)
    sc.materials["mat_mirror"] = MaterialSpec(
        id="mat_mirror",
        name="Chrome",
        base_color=(0.95, 0.96, 0.98, 1.0),
        metallic=0.98,
        roughness=0.05,
    )
    sc.objects["mirror_orb"] = SceneObject(
        id="mirror_orb",
        name="MirrorOrb",
        kind="mesh",
        transform=Transform(location=(0.0, 0.0, 0.85)),
        mesh=orb.to_payload(),
        material_id="mat_mirror",
    )

    cube = create_primitive_mesh("cube", size=0.9)
    sc.materials["mat_red"] = MaterialSpec(
        id="mat_red", name="RedBlock", base_color=(0.9, 0.15, 0.12, 1.0), roughness=0.45
    )
    sc.objects["red_box"] = SceneObject(
        id="red_box",
        name="RedBox",
        kind="mesh",
        transform=Transform(location=(-1.4, -0.4, 0.45)),
        mesh=cube.to_payload(),
        material_id="mat_red",
    )

    glass_orb = create_primitive_mesh("uv_sphere", radius=0.45, segments=12)
    sc.materials["mat_glass"] = MaterialSpec(
        id="mat_glass",
        name="Glass",
        base_color=(0.85, 0.96, 1.0, 1.0),
        metallic=0.0,
        roughness=0.04,
        alpha=0.25,
    )
    sc.objects["glass_orb"] = SceneObject(
        id="glass_orb",
        name="GlassOrb",
        kind="mesh",
        transform=Transform(location=(1.2, -0.5, 0.45)),
        mesh=glass_orb.to_payload(),
        material_id="mat_glass",
    )

    sc.objects["sun"] = SceneObject(
        id="sun",
        name="Sun",
        kind="light",
        transform=Transform(location=(4.0, -4.0, 6.0)),
        light=LightSpec(
            kind="sun", color=(1.0, 0.98, 0.92), intensity=2.8, direction=(-0.5, 0.4, -0.7)
        ),
    )

    rt = get_renderer("raytrace")
    assert rt.__class__.__name__ == "PathTracerRenderer"

    png_pbr, meta = render_single_view(
        sc, width=200, height=150, mode="pbr", engine="raytrace", samples=2
    )
    assert meta["engine"] == "raytrace"
    im = Image.open(io.BytesIO(png_pbr))
    arr = np.asarray(im)
    assert arr.shape == (150, 200, 3)
    assert arr.std() > 15.0  # Rich shaded scene with sky, geometry, reflections, shadows

    # Diagnostic modes work on PathTracerRenderer too
    for diag in ("clay", "normals", "depth"):
        png_d, _ = render_single_view(
            sc, width=160, height=120, mode=diag, engine="raytrace", samples=1
        )
        assert Image.open(io.BytesIO(png_d)).size == (160, 120)
