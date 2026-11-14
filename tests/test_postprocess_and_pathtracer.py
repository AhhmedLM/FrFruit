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


def test_physical_sky_hdri_ggx_coat_sheen_sss_and_bilateral_denoiser(tmp_path) -> None:
    from frfruit.render.postprocess import denoise_image
    from frfruit.render.sky import (
        evaluate_physical_sky,
        evaluate_world_sky,
        sun_direction_from_angles,
    )

    # 1. Edge-aware denoiser cuts flat noise while preserving crisp step edges
    h, w = 64, 64
    clean = np.full((h, w, 3), 0.15, dtype=np.float32)
    clean[:, 32:] = 0.85
    rng = np.random.default_rng(7)
    noisy = np.clip(clean + rng.normal(0.0, 0.08, size=clean.shape), 0.0, 2.0).astype(np.float32)
    albedo = clean.copy()
    normal = np.zeros((h, w, 3), dtype=np.float32)
    normal[:, :32, 2] = 1.0
    normal[:, 32:, 0] = 1.0
    depth = np.full((h, w), 5.0, dtype=np.float32)
    depth[:, 32:] = 2.0

    den = denoise_image(noisy, albedo_buf=albedo, normal_buf=normal, depth_buf=depth, strength=0.85)
    assert float(den[:, :28].std()) < float(noisy[:, :28].std()) * 0.55
    # Edge contrast across x=32 boundary remains sharp
    edge_jump = float(den[:, 35:].mean() - den[:, :29].mean())
    assert edge_jump > 0.55

    # 2. Physical sky sunset vs noon changes horizon color warmth dramatically
    dirs = np.array([[1.0, 0.0, 0.05], [0.0, 0.0, 0.95]], dtype=np.float64)
    sunset = evaluate_physical_sky(dirs, sun_direction_from_angles(4.0, 0.0), turbidity=3.2)
    noon = evaluate_physical_sky(dirs, sun_direction_from_angles(65.0, 0.0), turbidity=2.0)
    # Sunset horizon has higher red/blue ratio than high-noon blue zenith
    assert (sunset[0, 0] / max(1e-5, sunset[0, 2])) > (noon[1, 0] / max(1e-5, noon[1, 2]))

    # 3. Equirectangular HDRI loading & rotation
    hdri_img = Image.new("RGB", (64, 32), (30, 80, 220))
    # Bright warm spot on right side (+U)
    for x in range(44, 60):
        for y in range(8, 24):
            hdri_img.putpixel((x, y), (255, 220, 140))
    hdri_path = tmp_path / "env.png"
    hdri_img.save(hdri_path)

    sc = SceneDocument.new_default("GGXSky")
    sc.world.sky_mode = "hdri"
    sc.world.hdri_path = str(hdri_path)
    sc.world.hdri_intensity = 1.5
    test_dirs = np.array([[0.0, 1.0, 0.0], [0.0, -1.0, 0.0]], dtype=np.float64)
    s0 = evaluate_world_sky(test_dirs, sc, [])
    sc.world.hdri_rotation_deg = 180.0
    s180 = evaluate_world_sky(test_dirs, sc, [])
    assert not np.allclose(s0, s180)
    assert s0[0, 0] > s0[1, 0] and s180[1, 0] > s180[0, 0]

    # 4. Physical sky + clearcoat/sheen/subsurface materials render cleanly in software & raytrace
    sc.world.sky_mode = "physical_sky"
    sc.world.sun_elevation_deg = 22.0
    sc.materials["mat_coat"] = MaterialSpec(
        id="mat_coat",
        name="CarPaint",
        base_color=(0.75, 0.05, 0.08, 1.0),
        metallic=0.2,
        roughness=0.35,
        clearcoat=0.95,
        clearcoat_roughness=0.04,
    )
    sc.materials["mat_velvet"] = MaterialSpec(
        id="mat_velvet",
        name="Velvet",
        base_color=(0.15, 0.25, 0.65, 1.0),
        roughness=0.82,
        sheen=0.9,
    )
    sc.materials["mat_skin"] = MaterialSpec(
        id="mat_skin",
        name="JadeSSS",
        base_color=(0.25, 0.72, 0.45, 1.0),
        roughness=0.30,
        subsurface=0.75,
        subsurface_color=(0.45, 0.95, 0.55),
    )
    sphere = create_primitive_mesh("uv_sphere", radius=0.55, segments=16)
    sc.objects["s1"] = SceneObject(
        id="s1",
        name="CoatSphere",
        kind="mesh",
        transform=Transform(location=(-1.1, 0.0, 0.6)),
        mesh=sphere.to_payload(),
        material_id="mat_coat",
    )
    sc.objects["s2"] = SceneObject(
        id="s2",
        name="VelvetSphere",
        kind="mesh",
        transform=Transform(location=(0.0, 0.0, 0.6)),
        mesh=sphere.to_payload(),
        material_id="mat_velvet",
    )
    sc.objects["s3"] = SceneObject(
        id="s3",
        name="SSSSphere",
        kind="mesh",
        transform=Transform(location=(1.1, 0.0, 0.6)),
        mesh=sphere.to_payload(),
        material_id="mat_skin",
    )
    for eng in ("software", "raytrace"):
        png_b, _ = render_single_view(sc, width=180, height=135, mode="pbr", engine=eng, samples=2)
        arr = np.asarray(Image.open(io.BytesIO(png_b)))
        assert arr.shape == (135, 180, 3) and arr.std() > 10.0
