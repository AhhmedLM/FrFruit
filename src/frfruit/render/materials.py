"""Built-in PBR material presets and vectorized procedural pattern evaluator."""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from frfruit.core.scene import MaterialSpec

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
