"""Cached bitmap texture loader, bilinear UV sampler, triplanar fallback, and TBN bump."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np
from PIL import Image


@lru_cache(maxsize=64)
def load_texture_array(path: str) -> np.ndarray | None:
    """Load an image file into an (H, W, 3) float32 array in 0..1 (flipped V=bottom-up)."""
    p = Path(path)
    if not path or not p.is_file():
        return None
    try:
        with Image.open(p) as im:
            rgb = im.convert("RGB")
            arr = np.asarray(rgb, dtype=np.float32) / 255.0
        # Image row 0 is top (V=1); flip vertically so row 0 corresponds to V=0
        return np.ascontiguousarray(arr[::-1, :, :])
    except Exception:
        return None


def clear_texture_cache() -> None:
    load_texture_array.cache_clear()


def compute_triplanar_uvs(world_pos: np.ndarray, normals: np.ndarray) -> np.ndarray:
    """Compute world-space box/triplanar UVs (N, 2) for untextured meshes."""
    if len(world_pos) == 0:
        return np.zeros((0, 2), dtype=np.float64)
    abs_n = np.abs(normals)
    dom = np.argmax(abs_n, axis=1)
    uvs = np.empty((len(world_pos), 2), dtype=np.float64)
    mx = dom == 0
    my = dom == 1
    mz = dom == 2
    uvs[mx, 0] = world_pos[mx, 1] * 0.5
    uvs[mx, 1] = world_pos[mx, 2] * 0.5
    uvs[my, 0] = world_pos[my, 0] * 0.5
    uvs[my, 1] = world_pos[my, 2] * 0.5
    uvs[mz, 0] = world_pos[mz, 0] * 0.5
    uvs[mz, 1] = world_pos[mz, 1] * 0.5
    return uvs


def sample_texture(
    path: str,
    uvs: np.ndarray,
    uv_scale: tuple[float, float] = (1.0, 1.0),
) -> np.ndarray | None:
    """Bilinearly sample an RGB texture (H, W, 3) at (N, 2) UVs with wrapping."""
    tex = load_texture_array(path)
    if tex is None or len(uvs) == 0:
        return None
    h, w, _ = tex.shape
    u = np.mod(uvs[:, 0] * float(uv_scale[0]), 1.0) * (w - 1)
    v = np.mod(uvs[:, 1] * float(uv_scale[1]), 1.0) * (h - 1)

    x0 = np.floor(u).astype(np.int32)
    y0 = np.floor(v).astype(np.int32)
    x1 = (x0 + 1) % w
    y1 = (y0 + 1) % h
    fx = (u - x0)[:, None]
    fy = (v - y0)[:, None]

    c00 = tex[y0, x0]
    c10 = tex[y0, x1]
    c01 = tex[y1, x0]
    c11 = tex[y1, x1]
    top = c00 * (1.0 - fx) + c10 * fx
    bot = c01 * (1.0 - fx) + c11 * fx
    return (top * (1.0 - fy) + bot * fy).astype(np.float64)


def perturb_normals(
    nrm_world: np.ndarray,
    normal_rgb: np.ndarray,
    strength: float = 0.85,
) -> np.ndarray:
    """Perturb surface normals using a tangent-space normal map via Gram-Schmidt TBN."""
    if len(nrm_world) == 0 or len(normal_rgb) != len(nrm_world):
        return nrm_world
    ts = normal_rgb * 2.0 - 1.0
    ts[:, 0] *= strength
    ts[:, 1] *= strength

    up_ref = np.where(
        np.abs(nrm_world[:, 2:3]) > 0.92,
        np.array([[1.0, 0.0, 0.0]]),
        np.array([[0.0, 0.0, 1.0]]),
    )
    tangent = np.cross(up_ref, nrm_world)
    t_len = np.linalg.norm(tangent, axis=1, keepdims=True)
    tangent = tangent / np.where(t_len < 1e-8, 1.0, t_len)
    bitangent = np.cross(nrm_world, tangent)

    perturbed = (
        tangent * ts[:, 0:1] + bitangent * ts[:, 1:2] + nrm_world * np.maximum(0.15, ts[:, 2:3])
    )
    p_len = np.linalg.norm(perturbed, axis=1, keepdims=True)
    return perturbed / np.where(p_len < 1e-8, 1.0, p_len)
