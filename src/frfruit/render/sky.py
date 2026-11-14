"""Physical Nishita/Rayleigh-Mie atmospheric sky & equirectangular HDRI environment sampler."""

from __future__ import annotations

import math
from functools import lru_cache
from pathlib import Path

import numpy as np
from PIL import Image

from frfruit.core.scene import SceneDocument


def _parse_radiance_hdr(path: Path) -> np.ndarray | None:
    """Parse a Radiance RGBE (.hdr) file into linear float32 `(H, W, 3)` top-to-bottom."""
    try:
        raw = path.read_bytes()
    except OSError:
        return None
    if not raw.startswith(b"#?RADIANCE") and not raw.startswith(b"#?RGBE"):
        return None

    idx = raw.find(b"\n\n")
    if idx < 0:
        idx = raw.find(b"\r\n\r\n")
        if idx < 0:
            return None
        header_end = idx + 4
    else:
        header_end = idx + 2

    res_end = raw.find(b"\n", header_end)
    if res_end < 0:
        return None
    res_line = raw[header_end:res_end].decode("ascii", errors="ignore").strip().split()
    if len(res_line) != 4:
        return None

    y_sign, y_str, x_sign, x_str = res_line
    try:
        h = int(y_str)
        w = int(x_str)
    except ValueError:
        return None
    if h <= 0 or w <= 0 or w > 16384 or h > 8192:
        return None

    pos = res_end + 1
    rgbe = np.empty((h, w, 4), dtype=np.uint8)
    mv = memoryview(raw)

    for row in range(h):
        if pos + 4 > len(raw):
            return None
        b0, b1, b2, b3 = raw[pos], raw[pos + 1], raw[pos + 2], raw[pos + 3]
        if b0 == 2 and b1 == 2 and ((b2 << 8) | b3) == w and w >= 8 and w <= 32767:
            # New RLE scanline
            pos += 4
            for ch in range(4):
                col = 0
                while col < w:
                    if pos >= len(raw):
                        return None
                    cnt = raw[pos]
                    pos += 1
                    if cnt > 128:
                        run_len = cnt - 128
                        if pos >= len(raw) or col + run_len > w:
                            return None
                        val = raw[pos]
                        pos += 1
                        rgbe[row, col : col + run_len, ch] = val
                        col += run_len
                    else:
                        if cnt == 0 or pos + cnt > len(raw) or col + cnt > w:
                            return None
                        rgbe[row, col : col + cnt, ch] = np.frombuffer(
                            mv[pos : pos + cnt], dtype=np.uint8
                        )
                        pos += cnt
                        col += cnt
        else:
            # Uncompressed scanline
            n_bytes = w * 4
            if pos + n_bytes > len(raw):
                return None
            rgbe[row] = np.frombuffer(mv[pos : pos + n_bytes], dtype=np.uint8).reshape((w, 4))
            pos += n_bytes

    if y_sign == "+Y":
        rgbe = rgbe[::-1]
    if x_sign == "-X":
        rgbe = rgbe[:, ::-1]

    mant = rgbe[..., :3].astype(np.float32) + 0.5
    exp = rgbe[..., 3].astype(np.int32) - 136
    scale = np.where(rgbe[..., 3] > 0, np.ldexp(np.ones_like(exp, dtype=np.float32), exp), 0.0)
    rgb = mant * scale[..., None]
    return np.ascontiguousarray(rgb)


@lru_cache(maxsize=16)
def load_hdri_array(path: str) -> np.ndarray | None:
    """Load `.hdr` (Radiance RGBE) or standard image into linear `(H, W, 3)` float32 array."""
    if not path:
        return None
    p = Path(path)
    if not p.is_file():
        return None
    if p.suffix.lower() in (".hdr", ".rgbe"):
        parsed = _parse_radiance_hdr(p)
        if parsed is not None:
            return parsed
    try:
        with Image.open(p) as im:
            arr = np.asarray(im.convert("RGB"), dtype=np.float32) / 255.0
        # Convert display sRGB equirectangular texture to linear radiance
        return np.ascontiguousarray(np.power(np.clip(arr, 0.0, 1.0), 2.2))
    except Exception:
        return None


def sample_equirectangular_hdri(
    dirs: np.ndarray,
    hdri: np.ndarray,
    rotation_deg: float = 0.0,
    intensity: float = 1.0,
) -> np.ndarray:
    """Bilinearly sample an equirectangular environment map along `(N, 3)` unit rays."""
    if len(dirs) == 0:
        return np.zeros((0, 3), dtype=np.float64)

    rot = math.radians(rotation_deg)
    dx = dirs[:, 0]
    dy = dirs[:, 1]
    dz = dirs[:, 2]
    if abs(rot) > 1e-6:
        cr = math.cos(rot)
        sr = math.sin(rot)
        rx = dx * cr - dy * sr
        ry = dx * sr + dy * cr
    else:
        rx, ry = dx, dy

    phi = np.arctan2(ry, rx)
    u = np.mod(0.5 + phi / (2.0 * math.pi), 1.0)
    v = np.arccos(np.clip(dz, -1.0, 1.0)) / math.pi

    h, w, _ = hdri.shape
    fx = u * (w - 1)
    fy = v * (h - 1)
    x0 = np.floor(fx).astype(np.int32)
    y0 = np.clip(np.floor(fy).astype(np.int32), 0, h - 1)
    x1 = (x0 + 1) % w
    y1 = np.minimum(y0 + 1, h - 1)
    tx = (fx - x0)[:, None]
    ty = (fy - y0)[:, None]

    c00 = hdri[y0, x0]
    c10 = hdri[y0, x1]
    c01 = hdri[y1, x0]
    c11 = hdri[y1, x1]
    top = c00 * (1.0 - tx) + c10 * tx
    bot = c01 * (1.0 - tx) + c11 * tx
    return (top * (1.0 - ty) + bot * ty).astype(np.float64) * float(intensity)


def sun_direction_from_angles(elevation_deg: float, azimuth_deg: float) -> np.ndarray:
    """Return unit vector pointing toward the sun given elevation and azimuth degrees."""
    el = math.radians(elevation_deg)
    az = math.radians(azimuth_deg)
    cel = math.cos(el)
    v = np.array([cel * math.cos(az), cel * math.sin(az), math.sin(el)], dtype=np.float64)
    return v / max(1e-9, float(np.linalg.norm(v)))


def evaluate_physical_sky(
    dirs: np.ndarray,
    sun_dir: np.ndarray,
    turbidity: float = 2.2,
) -> np.ndarray:
    """Analytical Rayleigh + Mie Nishita-style atmospheric sky radiance for `(N, 3)` unit rays."""
    if len(dirs) == 0:
        return np.zeros((0, 3), dtype=np.float64)

    z = dirs[:, 2:3]
    cos_theta = np.clip(z, 0.0, 1.0)
    cos_gamma = np.sum(dirs * sun_dir[None, :], axis=1, keepdims=True)
    sun_el = float(sun_dir[2])

    # Golden hour warmth factor as sun nears horizon
    low_sun = float(np.clip(1.0 - sun_el * 2.2, 0.0, 1.0))
    turb = float(np.clip(turbidity, 1.0, 10.0))

    # Rayleigh zenith-to-horizon atmospheric gradient
    rayleigh_zenith = np.array([[0.12, 0.30, 0.78]]) * (1.0 - 0.38 * low_sun) + np.array(
        [[0.24, 0.22, 0.42]]
    ) * (0.38 * low_sun)
    rayleigh_horizon = np.array([[0.48, 0.66, 0.90]]) * (1.0 - 0.55 * low_sun) + np.array(
        [[0.96, 0.58, 0.32]]
    ) * (0.55 * low_sun)

    horizon_falloff = np.power(1.0 - cos_theta, 2.8)
    sky = rayleigh_zenith * (1.0 - horizon_falloff) + rayleigh_horizon * horizon_falloff

    # Mie forward-scattering halo around solar aureole & turbidity haze
    g = 0.76
    mie_phase = (1.0 - g * g) / np.power(np.maximum(0.08, 1.0 + g * g - 2.0 * g * cos_gamma), 1.5)
    mie_tint = np.array([[1.0, 0.88, 0.70]]) * (1.0 - 0.45 * low_sun) + np.array(
        [[1.0, 0.62, 0.34]]
    ) * (0.45 * low_sun)
    sky = sky + mie_phase * (0.045 * turb) * mie_tint * np.clip(
        0.35 + 0.65 * (1.0 - cos_theta), 0.15, 1.0
    )

    # Solar disc + corona when looking toward the sun above horizon
    disc = (cos_gamma > 0.9993).astype(np.float64) * 14.0 * max(0.0, sun_el + 0.1)
    corona = np.power(np.clip(cos_gamma, 0.0, 1.0), 120.0) * 1.4
    sky = sky + (disc + corona) * np.array([[1.0, 0.92, 0.80]])

    # Ground hemisphere below horizon (soft bounced earth/haze)
    below = (z < 0.0)[:, 0]
    if np.any(below):
        down_t = np.clip(-z[below] * 3.0, 0.0, 1.0)
        earth = np.array([[0.12, 0.11, 0.10]])
        sky[below] = sky[below] * (1.0 - down_t) + earth * down_t

    return np.maximum(sky, 0.0)


def evaluate_world_sky(
    dirs: np.ndarray,
    scene: SceneDocument,
    sun_dirs: list[tuple[np.ndarray, np.ndarray]],
    *,
    studio_ibl: bool = False,
) -> np.ndarray:
    """Evaluate sky radiance from `scene.world.sky_mode` ('gradient', 'physical_sky', 'hdri')."""
    world = scene.world
    if world.sky_mode == "hdri" and world.hdri_path:
        hdri = load_hdri_array(world.hdri_path)
        if hdri is not None:
            sky = sample_equirectangular_hdri(
                dirs,
                hdri,
                rotation_deg=world.hdri_rotation_deg,
                intensity=world.hdri_intensity,
            )
            for s_dir, s_col in sun_dirs:
                cos_ang = np.sum(dirs * s_dir[None, :], axis=1, keepdims=True)
                disc = (cos_ang > 0.997).astype(np.float64) * 3.0
                sky = sky + disc * s_col[None, :]
            return sky

    if world.sky_mode == "physical_sky":
        if sun_dirs:
            primary_sun = sun_dirs[0][0]
        else:
            primary_sun = sun_direction_from_angles(world.sun_elevation_deg, world.sun_azimuth_deg)
        sky = evaluate_physical_sky(dirs, primary_sun, turbidity=world.air_turbidity)
        amb_boost = float(np.clip(world.ambient_strength / 0.32, 0.35, 2.5))
        return sky * amb_boost

    # Default 'gradient' mode (studio backdrop + optional bounce IBL)
    z_t = np.clip(0.5 + dirs[:, 2:3] * 0.85, 0.0, 1.0)
    horizon_lin = np.power(np.clip(np.asarray(world.horizon_color), 0.0, 1.0), 2.2)[None, :]
    zenith_lin = np.power(np.clip(np.asarray(world.zenith_color), 0.0, 1.0), 2.2)[None, :]
    sky = horizon_lin * (1.0 - z_t) + zenith_lin * z_t

    if studio_ibl and world.ambient_strength > 1e-4:
        sky_avg = float(np.mean(0.5 * horizon_lin + 0.5 * zenith_lin))
        studio_blend = float(np.clip(1.0 - sky_avg * 2.8, 0.12, 1.0))
        hemi = np.clip(0.5 + 0.5 * dirs[:, 2:3], 0.0, 1.0)
        horizon_band = np.exp(-((dirs[:, 2:3] - 0.10) ** 2) / 0.16)
        dome = (
            np.array([[0.10, 0.11, 0.14]]) * (1.0 - hemi)
            + np.array([[0.40, 0.44, 0.52]]) * hemi
            + np.array([[0.26, 0.25, 0.23]]) * horizon_band
        )
        c_warm = np.array([0.48, -0.52, 0.70])
        c_warm /= np.linalg.norm(c_warm)
        c_cool = np.array([-0.68, 0.38, 0.62])
        c_cool /= np.linalg.norm(c_cool)
        card_w = np.power(
            np.clip(np.sum(dirs * c_warm[None, :], axis=1, keepdims=True), 0.0, 1.0), 8.0
        )
        card_c = np.power(
            np.clip(np.sum(dirs * c_cool[None, :], axis=1, keepdims=True), 0.0, 1.0), 10.0
        )
        studio_rad = (
            dome + card_w * np.array([[1.20, 1.08, 0.88]]) + card_c * np.array([[0.52, 0.70, 1.02]])
        )
        amb_scale = float(np.clip(world.ambient_strength / 0.35, 0.25, 1.6))
        sky = np.maximum(sky, studio_rad * amb_scale * studio_blend)

    for s_dir, s_col in sun_dirs:
        cos_ang = np.sum(dirs * s_dir[None, :], axis=1, keepdims=True)
        halo = np.power(np.clip(cos_ang, 0.0, 1.0), 64.0) * 0.35
        disc = (cos_ang > 0.996).astype(np.float64) * 3.5
        sky = sky + (halo + disc) * s_col[None, :]
    return sky
