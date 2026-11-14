"""CPU post-processing stack: SSAO, DOF, Bloom, Tone Mapping, Chromatic Aberration, Grade, FXAA."""

from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter, map_coordinates

from frfruit.core.scene import PostProcessConfig


def _luminance(rgb: np.ndarray) -> np.ndarray:
    return 0.2126 * rgb[..., 0] + 0.7152 * rgb[..., 1] + 0.0722 * rgb[..., 2]


def _tonemap_aces(x: np.ndarray) -> np.ndarray:
    a, b, c, d, e = 2.51, 0.03, 2.43, 0.59, 0.14
    return np.clip((x * (a * x + b)) / np.maximum(x * (c * x + d) + e, 1e-6), 0.0, 1.0)


def _tonemap_filmic(x: np.ndarray) -> np.ndarray:
    # Uncharted 2 / Hable filmic operator normalized with white point W=4.0
    def _hable(v: np.ndarray) -> np.ndarray:
        A, B, C, D, E, F = 0.15, 0.50, 0.10, 0.20, 0.02, 0.30
        return ((v * (A * v + C * B) + D * E) / (v * (A * v + B) + D * F)) - E / F

    white_scale = 1.0 / float(_hable(np.array(4.0, dtype=np.float64)))
    return np.clip(_hable(np.maximum(x * 1.4, 0.0)) * white_scale, 0.0, 1.0)


def _tonemap_reinhard(x: np.ndarray) -> np.ndarray:
    return np.clip(x / (1.0 + np.maximum(x, 0.0) * 0.45), 0.0, 1.0)


def _apply_ssao(
    img: np.ndarray,
    depth_buf: np.ndarray,
    radius_px: float,
    strength: float,
) -> np.ndarray:
    """Compute depth-aware bilateral Screen-Space Ambient Occlusion in linear space."""
    hit_mask = np.isfinite(depth_buf)
    if not np.any(hit_mask) or strength <= 1e-4:
        return img

    hit_f = hit_mask.astype(np.float64)
    depth_zero = np.where(hit_mask, depth_buf.astype(np.float64), 0.0)

    num = gaussian_filter(depth_zero, sigma=max(0.8, radius_px))
    den = gaussian_filter(hit_f, sigma=max(0.8, radius_px))
    avg_depth = np.where(den > 1e-4, num / np.maximum(den, 1e-4), depth_zero)

    # Concavities/crevices sit deeper (larger depth) than surrounding surfaces.
    # Ignore large background silhouette jumps using a bounded horizon threshold.
    delta = depth_zero - avg_depth
    rel_range = np.maximum(0.25, depth_zero * 0.18)
    valid_delta = np.where(hit_mask & (delta > 0.0) & (delta < rel_range * 2.5), delta, 0.0)
    ao_factor = 1.0 - min(1.5, strength) * np.tanh(valid_delta / rel_range * 1.6)
    ao_factor = np.clip(np.where(hit_mask, ao_factor, 1.0), 0.2, 1.0)
    return img * ao_factor[..., None]


def _apply_dof(
    img: np.ndarray,
    depth_buf: np.ndarray,
    focus_distance: float,
    aperture: float,
) -> np.ndarray:
    """Apply thin-lens style two-tap Circle-of-Confusion blur."""
    if aperture <= 1e-4:
        return img
    far_val = max(focus_distance * 4.0, 30.0)
    d = np.where(np.isfinite(depth_buf), depth_buf.astype(np.float64), far_val)
    coc = np.clip(
        np.abs(d - focus_distance) / max(0.5, focus_distance) * aperture * 1.4,
        0.0,
        1.0,
    )
    if float(np.max(coc)) < 0.02:
        return img

    blur_near = gaussian_filter(img, sigma=(1.6, 1.6, 0.0))
    blur_far = gaussian_filter(img, sigma=(4.2, 4.2, 0.0))
    w1 = np.clip(coc * 2.0, 0.0, 1.0)[..., None]
    w2 = np.clip((coc - 0.5) * 2.0, 0.0, 1.0)[..., None]
    mid = img * (1.0 - w1) + blur_near * w1
    return mid * (1.0 - w2) + blur_far * w2


def _apply_chromatic_aberration(img: np.ndarray, amount: float) -> np.ndarray:
    """Radially displace red and blue channels outward/inward from the optical center."""
    if amount <= 1e-4:
        return img
    h, w, _ = img.shape
    cy = (h - 1) * 0.5
    cx = (w - 1) * 0.5
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float64)
    dy = yy - cy
    dx = xx - cx
    r2 = (dx / max(1.0, cx)) ** 2 + (dy / max(1.0, cy)) ** 2
    scale_shift = r2 * (amount * 0.006)

    out = img.copy()
    ry = np.clip(yy - dy * scale_shift, 0.0, h - 1.0)
    rx = np.clip(xx - dx * scale_shift, 0.0, w - 1.0)
    by = np.clip(yy + dy * scale_shift, 0.0, h - 1.0)
    bx = np.clip(xx + dx * scale_shift, 0.0, w - 1.0)
    out[..., 0] = map_coordinates(img[..., 0], [ry, rx], order=1, mode="nearest")
    out[..., 2] = map_coordinates(img[..., 2], [by, bx], order=1, mode="nearest")
    return out


def _apply_fxaa(img: np.ndarray) -> np.ndarray:
    """Fast edge-aware FXAA smoothing along high-contrast diagonal/cardinal luma edges."""
    h, w, _ = img.shape
    if h < 4 or w < 4:
        return img
    luma = _luminance(img)
    n = np.roll(luma, 1, axis=0)
    s = np.roll(luma, -1, axis=0)
    w_px = np.roll(luma, 1, axis=1)
    e_px = np.roll(luma, -1, axis=1)

    l_min = np.minimum(luma, np.minimum(np.minimum(n, s), np.minimum(w_px, e_px)))
    l_max = np.maximum(luma, np.maximum(np.maximum(n, s), np.maximum(w_px, e_px)))
    contrast = l_max - l_min
    edge_mask = contrast > np.maximum(0.06, l_max * 0.14)
    if not np.any(edge_mask):
        return img

    avg4 = 0.25 * (
        np.roll(img, 1, axis=0)
        + np.roll(img, -1, axis=0)
        + np.roll(img, 1, axis=1)
        + np.roll(img, -1, axis=1)
    )
    blend = np.clip((contrast - 0.06) * 2.5, 0.0, 0.45) * edge_mask
    blend = blend[..., None]
    out = img * (1.0 - blend) + avg4 * blend
    # Keep 1px borders untouched
    out[0, :, :] = img[0, :, :]
    out[-1, :, :] = img[-1, :, :]
    out[:, 0, :] = img[:, 0, :]
    out[:, -1, :] = img[:, -1, :]
    return out


def apply_post_processing(
    linear_rgb: np.ndarray,
    depth_buf: np.ndarray | None = None,
    config: PostProcessConfig | None = None,
) -> np.ndarray:
    """Run FrFruit's HDR -> sRGB post-processing stack returning float32 RGB in [0, 1]."""
    img = np.asarray(linear_rgb, dtype=np.float64)
    cfg = config or PostProcessConfig()

    if not cfg.enabled:
        srgb = np.power(np.clip(img, 0.0, 1.0), 1.0 / 2.2)
        return np.clip(srgb, 0.0, 1.0).astype(np.float32)

    # 1. SSAO (linear space)
    if cfg.ssao_enabled and depth_buf is not None and cfg.ssao_strength > 0.0:
        img = _apply_ssao(img, depth_buf, radius_px=cfg.ssao_radius, strength=cfg.ssao_strength)

    # 2. Depth of Field (linear space)
    if cfg.dof_enabled and depth_buf is not None and cfg.dof_aperture > 0.0:
        img = _apply_dof(
            img,
            depth_buf,
            focus_distance=cfg.dof_focus_distance,
            aperture=cfg.dof_aperture,
        )

    # 3. Exposure gain
    if abs(cfg.exposure) > 1e-5:
        img = img * (2.0 ** float(cfg.exposure))

    # 4. Multi-scale HDR Bloom
    if cfg.bloom_enabled and cfg.bloom_intensity > 0.0:
        lum = _luminance(img)
        excess = np.maximum(0.0, lum - cfg.bloom_threshold) / np.maximum(lum, 1e-5)
        bright = img * excess[..., None]
        if float(np.max(bright)) > 1e-5:
            r = max(0.5, float(cfg.bloom_radius))
            glow = 0.65 * gaussian_filter(bright, sigma=(r, r, 0.0)) + 0.35 * gaussian_filter(
                bright, sigma=(r * 2.2, r * 2.2, 0.0)
            )
            img = img + glow * float(cfg.bloom_intensity)

    # 5. Optical Chromatic Aberration
    if cfg.chromatic_aberration > 0.0:
        img = _apply_chromatic_aberration(img, amount=float(cfg.chromatic_aberration))

    # 6. Tone Mapping
    img = np.maximum(img, 0.0)
    if cfg.tone_mapper == "aces":
        img = _tonemap_aces(img)
    elif cfg.tone_mapper == "filmic":
        img = _tonemap_filmic(img)
    elif cfg.tone_mapper == "reinhard":
        img = _tonemap_reinhard(img)
    else:
        img = np.clip(img, 0.0, 1.0)

    # 7. Linear -> sRGB encoding
    img = np.power(np.clip(img, 0.0, 1.0), 1.0 / 2.2)

    # 8. Color Temperature, Contrast, and Saturation
    if abs(cfg.temperature) > 1e-4:
        t = float(cfg.temperature)
        img[..., 0] = np.clip(img[..., 0] * (1.0 + 0.14 * t), 0.0, 1.0)
        img[..., 2] = np.clip(img[..., 2] * (1.0 - 0.14 * t), 0.0, 1.0)

    if abs(cfg.contrast - 1.0) > 1e-4:
        img = np.clip((img - 0.5) * float(cfg.contrast) + 0.5, 0.0, 1.0)

    if abs(cfg.saturation - 1.0) > 1e-4:
        luma = _luminance(img)[..., None]
        img = np.clip(luma + (img - luma) * float(cfg.saturation), 0.0, 1.0)

    # 9. Vignette
    if cfg.vignette_intensity > 1e-4:
        h, w, _ = img.shape
        yy = np.linspace(-1.0, 1.0, h, dtype=np.float64)[:, None]
        xx = np.linspace(-1.0, 1.0, w, dtype=np.float64)[None, :]
        r = np.sqrt(xx * xx + yy * yy) / math_sqrt2()
        soft = max(0.1, float(cfg.vignette_softness))
        vig = 1.0 - float(cfg.vignette_intensity) * np.power(np.clip(r, 0.0, 1.0), 1.0 + soft)
        img = img * vig[..., None]

    # 10. Edge Sharpening
    if cfg.sharpen > 1e-4:
        blurred = gaussian_filter(img, sigma=(0.8, 0.8, 0.0))
        img = np.clip(img + float(cfg.sharpen) * (img - blurred), 0.0, 1.0)

    # 11. FXAA anti-aliasing
    if cfg.fxaa_enabled:
        img = _apply_fxaa(img)

    return np.clip(img, 0.0, 1.0).astype(np.float32)


def math_sqrt2() -> float:
    return 1.4142135623730951
