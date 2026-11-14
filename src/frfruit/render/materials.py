"""Built-in PBR material presets and vectorized procedural pattern evaluator."""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from frfruit.core.scene import MaterialSpec
from frfruit.textures.loader import compute_triplanar_uvs, perturb_normals, sample_texture

PRESET_MATERIALS: dict[str, dict[str, Any]] = {
    "clay": {
        "name": "Studio Clay",
        "base_color": (0.76, 0.75, 0.73, 1.0),
        "metallic": 0.0,
        "roughness": 0.65,
    },
    "matte_plastic": {
        "name": "Matte Plastic",
        "base_color": (0.22, 0.55, 0.88, 1.0),
        "metallic": 0.0,
        "roughness": 0.55,
    },
    "glossy_ceramic": {
        "name": "Glossy Ceramic",
        "base_color": (0.94, 0.94, 0.92, 1.0),
        "metallic": 0.0,
        "roughness": 0.14,
    },
    "chrome": {
        "name": "Chrome",
        "base_color": (0.88, 0.90, 0.92, 1.0),
        "metallic": 1.0,
        "roughness": 0.12,
    },
    "gold": {
        "name": "Gold",
        "base_color": (0.95, 0.76, 0.28, 1.0),
        "metallic": 1.0,
        "roughness": 0.20,
    },
    "copper": {
        "name": "Copper",
        "base_color": (0.90, 0.48, 0.34, 1.0),
        "metallic": 1.0,
        "roughness": 0.26,
    },
    "wood": {
        "name": "Natural Wood",
        "base_color": (0.56, 0.34, 0.18, 1.0),
        "secondary_color": (0.36, 0.20, 0.09, 1.0),
        "pattern": "wood",
        "pattern_scale": 6.0,
        "metallic": 0.0,
        "roughness": 0.62,
    },
    "apple_skin": {
        "name": "Apple Red",
        "base_color": (0.78, 0.08, 0.10, 1.0),
        "secondary_color": (0.90, 0.24, 0.11, 1.0),
        "pattern": "gradient",
        "pattern_scale": 1.2,
        "metallic": 0.0,
        "roughness": 0.26,
    },
    "leaf_green": {
        "name": "Leaf Green",
        "base_color": (0.18, 0.58, 0.20, 1.0),
        "metallic": 0.0,
        "roughness": 0.48,
    },
    "glass": {
        "name": "Clear Glass",
        "base_color": (0.82, 0.93, 0.98, 0.45),
        "metallic": 0.0,
        "roughness": 0.06,
        "alpha": 0.45,
        "ior": 1.52,
    },
    "emissive_cyan": {
        "name": "Emissive Cyan",
        "base_color": (0.10, 0.85, 0.95, 1.0),
        "emission_color": (0.15, 0.90, 1.0, 1.0),
        "emission_strength": 1.6,
        "metallic": 0.0,
        "roughness": 0.30,
    },
    "emissive_amber": {
        "name": "Emissive Amber",
        "base_color": (0.98, 0.65, 0.12, 1.0),
        "emission_color": (1.0, 0.68, 0.15, 1.0),
        "emission_strength": 1.6,
        "metallic": 0.0,
        "roughness": 0.30,
    },
}


def list_material_presets() -> dict[str, dict[str, Any]]:
    return {k: dict(v) for k, v in PRESET_MATERIALS.items()}


def evaluate_surface_albedo(mat: MaterialSpec, world_pos: np.ndarray) -> np.ndarray:
    """Evaluate RGB base albedo (K, 3) across world positions for procedural patterns."""
    c1 = np.asarray(mat.base_color[:3], dtype=np.float64)[None, :]
    if mat.pattern == "solid" or len(world_pos) == 0:
        return np.repeat(c1, len(world_pos), axis=0)

    c2 = np.asarray(mat.secondary_color[:3], dtype=np.float64)[None, :]
    s = float(mat.pattern_scale)

    if mat.pattern == "checker":
        cell = np.floor(world_pos * s).astype(np.int64)
        parity = ((cell[:, 0] + cell[:, 1] + cell[:, 2]) & 1)[:, None].astype(np.float64)
        return c1 * (1.0 - parity) + c2 * parity

    if mat.pattern == "gradient":
        t = 0.5 + 0.5 * np.sin(world_pos[:, 2] * s * math.pi * 0.5)
        t = np.clip(t, 0.0, 1.0)[:, None]
        return c1 * (1.0 - t) + c2 * t

    if mat.pattern == "wood":
        r = np.sqrt(world_pos[:, 0] ** 2 + world_pos[:, 1] ** 2) * s
        ring = 0.5 + 0.5 * np.sin(r * 4.0 + 0.6 * np.sin(world_pos[:, 2] * s * 2.0))
        t = np.clip(ring, 0.0, 1.0)[:, None]
        return c1 * (1.0 - t) + c2 * t

    # procedural noise
    freq = s * math.pi
    n = (
        np.sin(world_pos[:, 0] * freq)
        * np.cos(world_pos[:, 1] * freq)
        * np.sin(world_pos[:, 2] * freq + 0.8)
    )
    t = np.clip(0.5 + 0.5 * n, 0.0, 1.0)[:, None]
    return c1 * (1.0 - t) + c2 * t


def evaluate_material_maps(
    mat: MaterialSpec,
    world_pos: np.ndarray,
    normals_w: np.ndarray,
    uvs: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Evaluate full PBR material channels (albedo, normal, roughness, metallic, ao) for K samples.

    Combines procedural patterns with any attached bitmap texture maps (`albedo_map`,
    `normal_map`, `roughness_map`, `metallic_map`, `ao_map`), falling back automatically to
    world-space triplanar UVs for untextured meshes.
    """
    k = len(world_pos)
    albedo = evaluate_surface_albedo(mat, world_pos)
    nrm_out = normals_w.copy()
    roughness = np.full((k, 1), float(mat.roughness), dtype=np.float64)
    metallic = np.full((k, 1), float(mat.metallic), dtype=np.float64)
    ao = np.ones((k, 1), dtype=np.float64)

    if k == 0:
        return albedo, nrm_out, roughness, metallic, ao

    has_maps = any(
        (
            mat.albedo_map,
            mat.normal_map,
            mat.roughness_map,
            mat.metallic_map,
            mat.ao_map,
        )
    )
    if not has_maps:
        return albedo, nrm_out, roughness, metallic, ao

    if uvs is not None and len(uvs) == k and np.all(np.isfinite(uvs)):
        samp_uvs = uvs
    else:
        samp_uvs = compute_triplanar_uvs(world_pos, normals_w)
    scale = (float(mat.uv_scale[0]), float(mat.uv_scale[1]))

    if mat.albedo_map:
        tex_alb = sample_texture(mat.albedo_map, samp_uvs, uv_scale=scale)
        if tex_alb is not None:
            tint = np.asarray(mat.base_color[:3], dtype=np.float64)[None, :]
            albedo = np.clip(tex_alb * tint, 0.0, 1.0)

    if mat.normal_map:
        tex_nrm = sample_texture(mat.normal_map, samp_uvs, uv_scale=scale)
        if tex_nrm is not None:
            nrm_out = perturb_normals(nrm_out, tex_nrm, strength=0.85)

    if mat.roughness_map:
        tex_rgh = sample_texture(mat.roughness_map, samp_uvs, uv_scale=scale)
        if tex_rgh is not None:
            roughness = np.clip(tex_rgh[:, 0:1], 0.04, 1.0)

    if mat.metallic_map:
        tex_mtl = sample_texture(mat.metallic_map, samp_uvs, uv_scale=scale)
        if tex_mtl is not None:
            metallic = np.clip(tex_mtl[:, 0:1], 0.0, 1.0)

    if mat.ao_map:
        tex_ao = sample_texture(mat.ao_map, samp_uvs, uv_scale=scale)
        if tex_ao is not None:
            ao = np.clip(tex_ao[:, 0:1], 0.1, 1.0)

    return albedo, nrm_out, roughness, metallic, ao


def evaluate_ggx_brdf(
    nrm: np.ndarray,
    view_dir: np.ndarray,
    light_dir: np.ndarray,
    albedo_lin: np.ndarray,
    roughness: np.ndarray,
    metallic: np.ndarray,
    f0: np.ndarray,
    *,
    clearcoat: np.ndarray | float = 0.0,
    clearcoat_roughness: np.ndarray | float = 0.08,
    sheen: np.ndarray | float = 0.0,
    subsurface: np.ndarray | float = 0.0,
    subsurface_color: np.ndarray | None = None,
) -> np.ndarray:
    """Evaluate physically plausible GGX Cook-Torrance BRDF + Clearcoat + Sheen + Subsurface."""
    ndl_raw = np.sum(nrm * light_dir, axis=1, keepdims=True)
    ndl = np.clip(ndl_raw, 0.0, 1.0)
    ndv = np.clip(np.sum(nrm * view_dir, axis=1, keepdims=True), 1e-4, 1.0)

    half_v = light_dir + view_dir
    hv_len = np.linalg.norm(half_v, axis=1, keepdims=True)
    half_v = half_v / np.where(hv_len < 1e-12, 1.0, hv_len)
    ndh = np.clip(np.sum(nrm * half_v, axis=1, keepdims=True), 0.0, 1.0)
    vdh = np.clip(np.sum(view_dir * half_v, axis=1, keepdims=True), 0.0, 1.0)

    # Trowbridge-Reitz GGX NDF
    r_clamped = np.clip(roughness, 0.04, 1.0)
    alpha = r_clamped * r_clamped
    a2 = alpha * alpha
    d_denom = ndh * ndh * (a2 - 1.0) + 1.0
    d_ggx = a2 / np.maximum(math.pi * d_denom * d_denom, 1e-6)

    # Smith-Schlick GGX visibility/geometry
    k_g = ((r_clamped + 1.0) ** 2) * 0.125
    g_v = ndv / (ndv * (1.0 - k_g) + k_g)
    g_l = ndl / np.maximum(ndl * (1.0 - k_g) + k_g, 1e-5)
    g_smith = g_v * g_l

    # Schlick Fresnel
    f_schlick = f0 + (1.0 - f0) * np.power(1.0 - vdh, 5.0)

    # Cook-Torrance specular lobe (bounded cleanly to suppress extreme grazing fireflies)
    spec = (d_ggx * g_smith * f_schlick) / np.maximum(4.0 * ndv * ndl, 1e-3)
    spec = np.minimum(spec, 8.0)

    # Energy-conserving Lambertian diffuse
    kd = (1.0 - f_schlick) * (1.0 - metallic)
    diff = kd * albedo_lin * (1.0 / math.pi) * 2.2

    # Optional Subsurface wrap scattering for skin/wax/foliage/marble
    if np.any(subsurface > 1e-4):
        wrap_ndl = np.clip((ndl_raw + 0.55) / 1.55, 0.0, 1.0)
        sss_tint = (
            subsurface_color
            if subsurface_color is not None
            else np.clip(albedo_lin * np.array([[1.15, 0.62, 0.48]]), 0.0, 1.0)
        )
        diff = diff * (1.0 - 0.65 * subsurface) + kd * sss_tint * wrap_ndl * subsurface * 0.85

    # Optional Grazing Fabric Sheen (retro-reflective velvet rim)
    sheen_term = 0.0
    if np.any(sheen > 1e-4):
        sheen_tint = 0.65 + 0.35 * albedo_lin
        grazing = np.power(1.0 - ndv, 4.0) + 0.35 * np.power(1.0 - ndh, 3.0)
        sheen_term = sheen * sheen_tint * grazing * (0.25 + 0.75 * ndl) * 0.6

    # Optional Dielectric Clearcoat second specular lobe (IOR ~ 1.5 -> F0 = 0.04)
    coat_spec = 0.0
    coat_atten = 1.0
    if np.any(clearcoat > 1e-4):
        rc = np.clip(clearcoat_roughness, 0.02, 1.0)
        ac = rc * rc
        ac2 = ac * ac
        dc_den = ndh * ndh * (ac2 - 1.0) + 1.0
        dc = ac2 / np.maximum(math.pi * dc_den * dc_den, 1e-6)
        kc = ((rc + 1.0) ** 2) * 0.125
        gc = (ndv / (ndv * (1.0 - kc) + kc)) * (ndl / np.maximum(ndl * (1.0 - kc) + kc, 1e-5))
        fc = 0.04 + 0.96 * np.power(1.0 - vdh, 5.0)
        coat_spec = clearcoat * np.minimum((dc * gc * fc) / np.maximum(4.0 * ndv * ndl, 1e-3), 8.0)
        coat_atten = 1.0 - clearcoat * fc * 0.45

    return ((diff + spec) * coat_atten + coat_spec + sheen_term) * ndl + (
        kd * albedo_lin * subsurface * np.clip((ndl_raw + 0.55) / 1.55, 0.0, 1.0) * 0.25
        if np.any(subsurface > 1e-4)
        else 0.0
    )
