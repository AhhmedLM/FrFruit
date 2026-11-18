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


def compute_triangle_tangents(
    v0: np.ndarray,
    v1: np.ndarray,
    v2: np.ndarray,
    uv0: np.ndarray,
    uv1: np.ndarray,
    uv2: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute per-triangle world-space tangent and bitangent vectors from UV gradients."""
    n = len(v0)
    t_out = np.zeros((n, 3), dtype=np.float64)
    b_out = np.zeros((n, 3), dtype=np.float64)
    if n == 0:
        return t_out, b_out

    dp1 = v1 - v0
    dp2 = v2 - v0
    duv1 = uv1 - uv0
    duv2 = uv2 - uv0

    finite = np.all(np.isfinite(duv1), axis=1) & np.all(np.isfinite(duv2), axis=1)
    det = duv1[:, 0] * duv2[:, 1] - duv1[:, 1] * duv2[:, 0]
    valid = finite & (np.abs(det) > 1e-10)
    if np.any(valid):
        inv_det = np.where(valid, 1.0 / np.where(valid, det, 1.0), 0.0)[:, None]
        t_raw = (dp1 * duv2[:, 1:2] - dp2 * duv1[:, 1:2]) * inv_det
        b_raw = (dp2 * duv1[:, 0:1] - dp1 * duv2[:, 0:1]) * inv_det
        t_len = np.linalg.norm(t_raw, axis=1, keepdims=True)
        b_len = np.linalg.norm(b_raw, axis=1, keepdims=True)
        good_len = valid & (t_len[:, 0] > 1e-8) & (b_len[:, 0] > 1e-8)
        t_out[good_len] = t_raw[good_len] / t_len[good_len]
        b_out[good_len] = b_raw[good_len] / b_len[good_len]
    return t_out, b_out


def perturb_normals(
    nrm_world: np.ndarray,
    normal_rgb: np.ndarray,
    strength: float = 0.85,
    tangent_w: np.ndarray | None = None,
    bitangent_w: np.ndarray | None = None,
) -> np.ndarray:
    """Perturb surface normals using a tangent-space normal map via Gram-Schmidt TBN."""
    if len(nrm_world) == 0 or len(normal_rgb) != len(nrm_world) or strength <= 1e-5:
        return nrm_world
    ts = normal_rgb * 2.0 - 1.0
    ts[:, 0] *= strength
    ts[:, 1] *= strength

    up_ref = np.where(
        np.abs(nrm_world[:, 2:3]) > 0.92,
        np.array([[1.0, 0.0, 0.0]]),
        np.array([[0.0, 0.0, 1.0]]),
    )
    fallback_t = np.cross(up_ref, nrm_world)
    ft_len = np.linalg.norm(fallback_t, axis=1, keepdims=True)
    fallback_t = fallback_t / np.where(ft_len < 1e-8, 1.0, ft_len)
    fallback_b = np.cross(nrm_world, fallback_t)

    if tangent_w is not None and bitangent_w is not None and len(tangent_w) == len(nrm_world):
        t_proj = tangent_w - np.sum(tangent_w * nrm_world, axis=1, keepdims=True) * nrm_world
        tp_len = np.linalg.norm(t_proj, axis=1, keepdims=True)
        has_uv_t = (tp_len[:, 0] > 1e-6) & (np.linalg.norm(bitangent_w, axis=1) > 1e-6)
        tangent = np.where(
            has_uv_t[:, None],
            t_proj / np.where(tp_len < 1e-8, 1.0, tp_len),
            fallback_t,
        )
        # Preserve UV handedness against triangle bitangent
        auto_b = np.cross(nrm_world, tangent)
        handed = np.where(np.sum(auto_b * bitangent_w, axis=1, keepdims=True) < 0.0, -1.0, 1.0)
        bitangent = np.where(has_uv_t[:, None], auto_b * handed, fallback_b)
    else:
        tangent = fallback_t
        bitangent = fallback_b

    perturbed = (
        tangent * ts[:, 0:1] + bitangent * ts[:, 1:2] + nrm_world * np.maximum(0.15, ts[:, 2:3])
    )
    p_len = np.linalg.norm(perturbed, axis=1, keepdims=True)
    return perturbed / np.where(p_len < 1e-8, 1.0, p_len)
