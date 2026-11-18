"""Build a photorealistic 3D reconstruction of the Gulf Residential Alleyway scene in FrFruit.

Uses real CC0 PBR texture packs (Poly Haven & ambientCG) composited with weathered stucco,
graffiti/water-streak details, 45-degree herringbone interlocking pavers, rusted twin utility
cabinets with 3D perforated vents, corrugated metal compound fence, three sculpted vehicles,
and overcast HDRI path-traced rendering.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

from frfruit.core.math3d import compose_matrix, transform_points
from frfruit.core.scene import (
    CameraSpec,
    CollectionSpec,
    LightSpec,
    MaterialSpec,
    PostProcessConfig,
    SceneDocument,
    SceneObject,
    WorldEnvironment,
)
from frfruit.core.types import Transform, Vec2, Vec3
from frfruit.geometry.curves import sweep_curve
from frfruit.geometry.mesh import MeshData, concat_meshes
from frfruit.geometry.primitives import make_cylinder, make_icosphere
from frfruit.geometry.spatial import evaluate_scene_meshes
from frfruit.io.exporters import export_scene_file
from frfruit.io.project import save_project
from frfruit.render.pathtracer import PathTracerRenderer
from frfruit.render.software import RenderConfig, SoftwareRenderer
from frfruit.render.viewport import render_multiview_sheet
from frfruit.textures.fetcher import download_pbr_texture_pack

REPO_ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = REPO_ROOT / "exports" / "alley_scene"
RAW_TEX_DIR = OUT_DIR / "textures_raw"
TEX_DIR = OUT_DIR / "textures"


# ---------------------------------------------------------------------------
# 1. Texture Preparation (Real CC0 Poly Haven / ambientCG + Custom Weathering)
# ---------------------------------------------------------------------------


def _load_rgb(path: Path, size: tuple[int, int] | None = None) -> Image.Image:
    im = Image.open(path).convert("RGB")
    if size is not None and im.size != size:
        im = im.resize(size, Image.Resampling.BICUBIC)
    return im


def _tile_image(im: Image.Image, nx: int, ny: int, out_size: tuple[int, int]) -> Image.Image:
    w, h = im.size
    canvas = Image.new("RGB", (w * nx, h * ny))
    for j in range(ny):
        for i in range(nx):
            canvas.paste(im, (i * w, j * h))
    return canvas.resize(out_size, Image.Resampling.BICUBIC)


def _add_noise_streaks(
    arr: np.ndarray,
    rng: np.random.Generator,
    n_streaks: int,
    *,
    y_top_frac: float = 0.0,
    y_bot_frac: float = 0.65,
    x_range: tuple[float, float] = (0.0, 1.0),
    darkness: float = 0.16,
    warm_shift: tuple[float, float, float] = (0.94, 0.88, 0.78),
) -> np.ndarray:
    h, w, _ = arr.shape
    out = arr.copy()
    x0_lim = int(x_range[0] * w)
    x1_lim = max(x0_lim + 2, int(x_range[1] * w))
    yt = int(y_top_frac * h)
    yb = int(y_bot_frac * h)
    for _ in range(n_streaks):
        cx = int(rng.integers(x0_lim, x1_lim))
        width = int(rng.integers(max(2, w // 220), max(5, w // 70)))
        len_px = int(rng.integers(max(20, (yb - yt) // 4), max(25, yb - yt + 1)))
        y_start = int(rng.integers(yt, max(yt + 1, yt + max(1, (yb - yt) // 3))))
        y_end = min(h, y_start + len_px)
        if y_end <= y_start:
            continue
        x_l = max(0, cx - width)
        x_r = min(w, cx + width + 1)
        vy = np.linspace(1.0, 0.0, y_end - y_start)[:, None, None]
        vx = np.exp(-((np.arange(x_l, x_r) - cx) ** 2) / max(1.0, (width * 0.6) ** 2))[
            None, :, None
        ]
        mask = vy * vx * darkness * float(rng.uniform(0.5, 1.1))
        tint = np.array(warm_shift, dtype=np.float32)[None, None, :]
        out[y_start:y_end, x_l:x_r] = (
            out[y_start:y_end, x_l:x_r] * (1.0 - mask)
            + out[y_start:y_end, x_l:x_r] * tint * 0.55 * mask
        )
    return np.clip(out, 0.0, 1.0)


def _draw_spray_scribble(
    img: Image.Image,
    strokes: list[list[tuple[float, float]]],
    color: tuple[int, int, int],
    width_px: int,
    opacity: float = 0.65,
    blur_px: float = 2.0,
) -> Image.Image:
    w, h = img.size
    overlay = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    alpha = int(255 * opacity)
    col = (*color, alpha)
    for poly in strokes:
        pts = [(px * w, py * h) for px, py in poly]
        if len(pts) >= 2:
            draw.line(pts, fill=col, width=width_px, joint="curve")
    if blur_px > 0:
        overlay = overlay.filter(ImageFilter.GaussianBlur(radius=blur_px))
    base = img.convert("RGBA")
    return Image.alpha_composite(base, overlay).convert("RGB")


def prepare_alley_textures() -> dict[str, dict[str, str]]:
    """Prepare all downloaded + composited PBR maps inside `TEX_DIR`."""
    RAW_TEX_DIR.mkdir(parents=True, exist_ok=True)
    TEX_DIR.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(2026)

    packs_needed = [
        ("herringbone_brick_03", "polyhaven", "1k"),
        ("white_stucco_02", "polyhaven", "1k"),
        ("Plaster007", "ambientcg", "1k"),
        ("box_profile_metal_sheet", "polyhaven", "1k"),
        ("Concrete034", "ambientcg", "1k"),
        ("Metal064", "ambientcg", "1k"),
    ]
    for aid, src, res in packs_needed:
        pack_dir = RAW_TEX_DIR / f"{src}_{aid}_{res}"
        if not (pack_dir / "manifest.json").exists():
            download_pbr_texture_pack(aid, source=src, resolution=res, target_dir=RAW_TEX_DIR)

    hdr_path = RAW_TEX_DIR / "kloofendal_overcast_puresky_2k.hdr"
    if not hdr_path.exists():
        from frfruit.textures.fetcher import _http_get_bytes, _http_get_json

        hdri_meta = _http_get_json("https://api.polyhaven.com/files/kloofendal_overcast_puresky")
        hdr_url = hdri_meta["hdri"]["2k"]["hdr"]["url"]
        hdr_path.write_bytes(_http_get_bytes(hdr_url, timeout=60.0))

    paths: dict[str, dict[str, str]] = {"hdri": {"path": str(hdr_path)}}

    # 1) Road grey herringbone pavers & sidewalk dusty terracotta pavers
    hb_dir = RAW_TEX_DIR / "polyhaven_herringbone_brick_03_1k"
    hb_alb = (
        np.asarray(_load_rgb(hb_dir / "herringbone_brick_03_albedo.png"), dtype=np.float32) / 255.0
    )
    hb_ao = np.asarray(_load_rgb(hb_dir / "herringbone_brick_03_ao.png"), dtype=np.float32) / 255.0
    hb_luma = 0.299 * hb_alb[..., 0:1] + 0.587 * hb_alb[..., 1:2] + 0.114 * hb_alb[..., 2:3]

    # Weathered mid-dark concrete grey pavers with crisp dark mortar joints
    seam_darken = 0.35 + 0.65 * np.clip((hb_ao[..., 0:1] - 0.35) / 0.65, 0.0, 1.0)
    grey_paver = np.clip(
        (hb_luma * 0.76 + (hb_alb - hb_luma) * 0.05 + np.array([[[0.04, 0.042, 0.046]]]))
        * seam_darken,
        0.05,
        0.54,
    )
    grey_alb_p = TEX_DIR / "pavers_grey_albedo.png"
    Image.fromarray((grey_paver * 255).astype(np.uint8)).save(grey_alb_p)

    hb_rgh = (
        np.asarray(_load_rgb(hb_dir / "herringbone_brick_03_roughness.png"), dtype=np.float32)
        / 255.0
    )
    grey_rgh_p = TEX_DIR / "pavers_grey_roughness.png"
    Image.fromarray((np.clip(0.70 + hb_rgh * 0.26, 0.52, 0.96) * 255).astype(np.uint8)).save(
        grey_rgh_p
    )

    paths["pavers_grey"] = {
        "albedo": str(grey_alb_p),
        "normal": str(hb_dir / "herringbone_brick_03_normal.png"),
        "roughness": str(grey_rgh_p),
        "ao": str(hb_dir / "herringbone_brick_03_ao.png"),
    }

    # Dusty muted taupe-brown sidewalk pavers (desaturated to match Gulf sidewalk photo)
    red_paver = np.clip(
        (hb_luma * np.array([[[0.62, 0.54, 0.51]]]) + (hb_alb - hb_luma) * 0.12) * seam_darken,
        0.06,
        0.54,
    )
    red_alb_p = TEX_DIR / "pavers_red_albedo.png"
    Image.fromarray((red_paver * 255).astype(np.uint8)).save(red_alb_p)
    paths["pavers_red"] = {
        "albedo": str(red_alb_p),
        "normal": str(hb_dir / "herringbone_brick_03_normal.png"),
        "roughness": str(grey_rgh_p),
        "ao": str(hb_dir / "herringbone_brick_03_ao.png"),
    }

    # 2) Plaster/Stucco (smooth white_stucco_02 normal so walls aren't overly speckled)
    pl_dir = RAW_TEX_DIR / "ambientcg_Plaster007_1k"
    st_dir = RAW_TEX_DIR / "polyhaven_white_stucco_02_1k"
    pl_alb_raw = _load_rgb(pl_dir / "Plaster007_albedo.jpg", (1024, 1024))
    st_alb_raw = _load_rgb(st_dir / "white_stucco_02_albedo.png", (1024, 1024))
    st_nrm_raw = _load_rgb(st_dir / "white_stucco_02_normal.png", (1024, 1024))
    st_rgh_raw = _load_rgb(st_dir / "white_stucco_02_roughness.png", (1024, 1024))

    # Left FG wall: 2048x1024 (U: Y=-2.0..10.5 [12.5m], V: Z=0..4.7m)
    pl_tile_l = np.asarray(_tile_image(pl_alb_raw, 3, 1, (2048, 1024)), dtype=np.float32) / 255.0
    st_tile_l = np.asarray(_tile_image(st_alb_raw, 3, 1, (2048, 1024)), dtype=np.float32) / 255.0
    wall_l = 0.45 * pl_tile_l + 0.55 * st_tile_l
    wall_l = np.clip(wall_l * np.array([[[0.96, 0.93, 0.87]]]), 0.14, 0.88)

    h_l, w_l, _ = wall_l.shape
    dado_rows = int(0.115 * h_l)
    dado_y0 = h_l - dado_rows
    grad = np.linspace(0.0, 1.0, 18)[:, None, None]
    wall_l[dado_y0:, :] *= np.array([[[0.55, 0.55, 0.54]]])
    wall_l[dado_y0 - 18 : dado_y0, :] *= 1.0 - grad * np.array([[[0.42, 0.42, 0.43]]])
    spl = np.linspace(0.0, 1.0, 32)[:, None, None]
    wall_l[h_l - 32 :, :] *= 1.0 - spl * 0.30
    wall_l = _add_noise_streaks(wall_l, rng, 65, y_top_frac=0.0, y_bot_frac=0.78, darkness=0.16)
    wall_l = _add_noise_streaks(
        wall_l, rng, 34, y_top_frac=0.30, y_bot_frac=0.92, x_range=(0.05, 0.35), darkness=0.20
    )

    wall_l_img = Image.fromarray((wall_l * 255).astype(np.uint8))
    # Soft feathered charcoal brush marks high on foreground left wall (organic curve)
    wall_l_img = _draw_spray_scribble(
        wall_l_img,
        [
            [(0.11, 0.03), (0.12, 0.09), (0.15, 0.14), (0.19, 0.18)],
            [(0.16, 0.30), (0.19, 0.34), (0.23, 0.36), (0.27, 0.34)],
            [(0.18, 0.35), (0.22, 0.39), (0.26, 0.41), (0.28, 0.39)],
        ],
        color=(64, 66, 68),
        width_px=24,
        opacity=0.48,
        blur_px=4.5,
    )
    # Reddish-orange faded Arabic spray graffiti mid-wall
    wall_l_img = _draw_spray_scribble(
        wall_l_img,
        [
            [(0.49, 0.67), (0.51, 0.64), (0.53, 0.68), (0.56, 0.65), (0.58, 0.63)],
            [(0.59, 0.66), (0.62, 0.68), (0.65, 0.64), (0.68, 0.67), (0.71, 0.65)],
            [(0.55, 0.72), (0.58, 0.70), (0.61, 0.72), (0.65, 0.71)],
        ],
        color=(165, 78, 70),
        width_px=6,
        opacity=0.45,
        blur_px=1.6,
    )
    # Dark grey brush swoosh low at Y ~ 6.5m
    wall_l_img = _draw_spray_scribble(
        wall_l_img,
        [[(0.65, 0.76), (0.69, 0.78), (0.71, 0.84), (0.68, 0.89)]],
        color=(58, 60, 62),
        width_px=15,
        opacity=0.52,
        blur_px=2.5,
    )

    wl_alb_p = TEX_DIR / "wall_left_fg_albedo.png"
    wl_nrm_p = TEX_DIR / "wall_left_fg_normal.png"
    wl_rgh_p = TEX_DIR / "wall_left_fg_roughness.png"
    wall_l_img.save(wl_alb_p)
    _tile_image(st_nrm_raw, 3, 1, (2048, 1024)).save(wl_nrm_p)
    _tile_image(st_rgh_raw, 3, 1, (2048, 1024)).save(wl_rgh_p)
    paths["wall_left_fg"] = {
        "albedo": str(wl_alb_p),
        "normal": str(wl_nrm_p),
        "roughness": str(wl_rgh_p),
    }

    # Right FG wall: 1536x1024 (U: Y=9.0..1.4m, V: Z=0..4.8m)
    pl_tile_r = np.asarray(_tile_image(pl_alb_raw, 2, 1, (1536, 1024)), dtype=np.float32) / 255.0
    st_tile_r = np.asarray(_tile_image(st_alb_raw, 2, 1, (1536, 1024)), dtype=np.float32) / 255.0
    wall_r = 0.44 * pl_tile_r + 0.56 * st_tile_r
    wall_r = np.clip(wall_r * np.array([[[0.92, 0.91, 0.88]]]), 0.14, 0.86)
    h_r, w_r, _ = wall_r.shape
    dado_r = int(0.11 * h_r)
    wall_r[h_r - dado_r :, :] *= 0.52
    wall_r = _add_noise_streaks(wall_r, rng, 48, y_top_frac=0.0, y_bot_frac=0.88, darkness=0.16)
    wall_r = _add_noise_streaks(
        wall_r, rng, 26, y_top_frac=0.45, y_bot_frac=0.98, x_range=(0.15, 0.85), darkness=0.18
    )
    wall_r_img = Image.fromarray((wall_r * 255).astype(np.uint8))
    # Subtle organic looped red/orange graffiti at bottom-right
    wall_r_img = _draw_spray_scribble(
        wall_r_img,
        [
            [(0.74, 0.77), (0.77, 0.74), (0.80, 0.78), (0.83, 0.75), (0.87, 0.79), (0.90, 0.76)],
            [(0.78, 0.82), (0.81, 0.79), (0.84, 0.83), (0.88, 0.80)],
        ],
        color=(156, 70, 58),
        width_px=5,
        opacity=0.46,
        blur_px=1.6,
    )
    wr_alb_p = TEX_DIR / "wall_right_fg_albedo.png"
    wr_nrm_p = TEX_DIR / "wall_right_fg_normal.png"
    wr_rgh_p = TEX_DIR / "wall_right_fg_roughness.png"
    wall_r_img.save(wr_alb_p)
    _tile_image(st_nrm_raw, 2, 1, (1536, 1024)).save(wr_nrm_p)
    _tile_image(st_rgh_raw, 2, 1, (1536, 1024)).save(wr_rgh_p)
    paths["wall_right_fg"] = {
        "albedo": str(wr_alb_p),
        "normal": str(wr_nrm_p),
        "roughness": str(wr_rgh_p),
    }

    # Olive-beige midground wall
    olive = np.clip(
        (0.4 * pl_tile_r + 0.6 * st_tile_r) * np.array([[[0.82, 0.77, 0.60]]]), 0.14, 0.84
    )
    olive = _add_noise_streaks(olive, rng, 30, y_top_frac=0.0, y_bot_frac=0.88, darkness=0.14)
    ol_alb_p = TEX_DIR / "wall_olive_albedo.png"
    Image.fromarray((olive * 255).astype(np.uint8)).save(ol_alb_p)
    paths["wall_olive"] = {
        "albedo": str(ol_alb_p),
        "normal": str(wr_nrm_p),
        "roughness": str(wr_rgh_p),
    }

    # 3) Utility Box Weathered Painted Metal + Edge Rust + Blue/Red Stains (1024x1024)
    m64_dir = RAW_TEX_DIR / "ambientcg_Metal064_1k"
    rust_alb = (
        np.asarray(_load_rgb(m64_dir / "Metal064_albedo.jpg", (1024, 1024)), dtype=np.float32)
        / 255.0
    )
    rust_rgh = (
        np.asarray(_load_rgb(m64_dir / "Metal064_roughness.jpg", (1024, 1024)), dtype=np.float32)
        / 255.0
    )
    rust_nrm = _load_rgb(m64_dir / "Metal064_normal.jpg", (1024, 1024))

    st_arr = np.asarray(st_alb_raw, dtype=np.float32) / 255.0
    paint_base = 0.16 * st_arr + 0.84 * np.array([[[0.71, 0.70, 0.66]]], dtype=np.float32)
    paint_base = _add_noise_streaks(
        paint_base, rng, 34, y_top_frac=0.0, y_bot_frac=0.95, darkness=0.12
    )

    yy, xx = np.mgrid[0:1024, 0:1024].astype(np.float32) / 1023.0
    edge_d = np.minimum(np.minimum(xx, 1.0 - xx), np.minimum(yy, 1.0 - yy))
    rust_noise = np.clip((rust_alb[..., 0] - rust_alb[..., 2]) * 2.0, 0.0, 1.0)
    edge_mask = np.clip(1.0 - edge_d / 0.06, 0.0, 1.0) ** 1.25
    bottom_rust = np.clip((yy - 0.82) / 0.18, 0.0, 1.0) ** 1.3
    rust_mix = np.clip(
        np.maximum(edge_mask * (0.52 + 0.48 * rust_noise), bottom_rust * rust_noise * 0.78)
        + np.maximum(0.0, rust_noise - 0.74) * 0.24,
        0.0,
        0.88,
    )[..., None]
    rust_tint = np.clip(rust_alb * np.array([[[0.66, 0.45, 0.32]]]), 0.06, 0.42)
    box_alb = paint_base * (1.0 - rust_mix) + rust_tint * rust_mix
    box_img = Image.fromarray((np.clip(box_alb, 0.0, 1.0) * 255).astype(np.uint8))

    # Soft blue faded paint scuffs on Box 1
    box_img = _draw_spray_scribble(
        box_img,
        [
            [(0.22, 0.50), (0.28, 0.60), (0.31, 0.72), (0.24, 0.79)],
            [(0.34, 0.66), (0.40, 0.76), (0.45, 0.68)],
        ],
        color=(86, 114, 146),
        width_px=22,
        opacity=0.38,
        blur_px=8.0,
    )
    # Subtle red faded spray writing on Box 2
    box2_img = _draw_spray_scribble(
        box_img,
        [
            [(0.20, 0.28), (0.35, 0.24), (0.50, 0.30), (0.65, 0.25), (0.78, 0.31)],
            [(0.26, 0.40), (0.44, 0.36), (0.62, 0.42), (0.76, 0.38)],
        ],
        color=(165, 66, 58),
        width_px=9,
        opacity=0.45,
        blur_px=2.5,
    )

    box1_alb_p = TEX_DIR / "box1_panel_albedo.png"
    box2_alb_p = TEX_DIR / "box2_panel_albedo.png"
    box_rgh_p = TEX_DIR / "box_panel_roughness.png"
    box_nrm_p = TEX_DIR / "box_panel_normal.png"
    box_img.save(box1_alb_p)
    box2_img.save(box2_alb_p)
    box_rgh_arr = np.clip(0.64 + rust_mix[..., 0] * 0.28 + rust_rgh[..., 0] * 0.08, 0.52, 0.95)
    Image.fromarray((box_rgh_arr * 255).astype(np.uint8)).save(box_rgh_p)
    rust_nrm.save(box_nrm_p)

    # Dark oxidized rust brown-grey metal for legs, latch, & meter box
    rust_frame_p = TEX_DIR / "box_rust_albedo.png"
    Image.fromarray(
        (np.clip(rust_alb * np.array([[[0.58, 0.40, 0.29]]]), 0.05, 0.38) * 255).astype(np.uint8)
    ).save(rust_frame_p)

    # Cream metal with rusted edges for top vent lattice & frame rails
    box_grille_p = TEX_DIR / "box_grille_albedo.png"
    grille_arr = 0.72 * (np.asarray(box_img, dtype=np.float32) / 255.0) + 0.28 * rust_tint
    Image.fromarray((np.clip(grille_arr, 0.08, 0.76) * 255).astype(np.uint8)).save(box_grille_p)

    paths["box1_panel"] = {
        "albedo": str(box1_alb_p),
        "normal": str(box_nrm_p),
        "roughness": str(box_rgh_p),
    }
    paths["box2_panel"] = {
        "albedo": str(box2_alb_p),
        "normal": str(box_nrm_p),
        "roughness": str(box_rgh_p),
    }
    paths["box_grille"] = {
        "albedo": str(box_grille_p),
        "normal": str(box_nrm_p),
        "roughness": str(box_rgh_p),
    }
    paths["box_rust"] = {
        "albedo": str(rust_frame_p),
        "normal": str(m64_dir / "Metal064_normal.jpg"),
        "roughness": str(m64_dir / "Metal064_roughness.jpg"),
    }

    # 4) Right Compound Wall (Concrete034 + vertical ribs + graffiti) & Corrugated Upper Fence
    c34_dir = RAW_TEX_DIR / "ambientcg_Concrete034_1k"
    c34_alb = _load_rgb(c34_dir / "Concrete034_albedo.jpg", (1024, 512))
    c34_nrm = _load_rgb(c34_dir / "Concrete034_normal.jpg", (1024, 512))
    c34_rgh = _load_rgb(c34_dir / "Concrete034_roughness.jpg", (1024, 512))

    fence_arr = np.asarray(_tile_image(c34_alb, 3, 1, (2048, 512)), dtype=np.float32) / 255.0
    fence_arr *= np.array([[[0.68, 0.67, 0.65]]], dtype=np.float32)
    for px in range(48, 2048, 96):
        fence_arr[:, max(0, px - 2) : px + 2] *= 0.80
    fence_arr = _add_noise_streaks(
        fence_arr, rng, 55, y_top_frac=0.0, y_bot_frac=0.92, darkness=0.14
    )
    fence_img = Image.fromarray((np.clip(fence_arr, 0.08, 0.78) * 255).astype(np.uint8))
    fence_img = _draw_spray_scribble(
        fence_img,
        [[(0.24, 0.25), (0.26, 0.45), (0.24, 0.62)]],
        color=(158, 72, 118),
        width_px=14,
        opacity=0.52,
        blur_px=3.0,
    )
    fence_img = _draw_spray_scribble(
        fence_img,
        [[(0.16, 0.42), (0.19, 0.36), (0.21, 0.46), (0.18, 0.52)]],
        color=(36, 40, 48),
        width_px=12,
        opacity=0.65,
        blur_px=2.0,
    )
    fl_alb_p = TEX_DIR / "fence_lower_albedo.png"
    fl_nrm_p = TEX_DIR / "fence_lower_normal.png"
    fl_rgh_p = TEX_DIR / "fence_lower_roughness.png"
    fence_img.save(fl_alb_p)
    _tile_image(c34_nrm, 3, 1, (2048, 512)).save(fl_nrm_p)
    _tile_image(c34_rgh, 3, 1, (2048, 512)).save(fl_rgh_p)
    paths["fence_lower"] = {
        "albedo": str(fl_alb_p),
        "normal": str(fl_nrm_p),
        "roughness": str(fl_rgh_p),
    }

    # Corrugated upper metal sheet (weathered dusty warm beige-tan, soft contrast)
    bm_dir = RAW_TEX_DIR / "polyhaven_box_profile_metal_sheet_1k"
    bm_alb = (
        np.asarray(_load_rgb(bm_dir / "box_profile_metal_sheet_albedo.png"), dtype=np.float32)
        / 255.0
    )
    bm_luma = 0.299 * bm_alb[..., 0:1] + 0.587 * bm_alb[..., 1:2] + 0.114 * bm_alb[..., 2:3]
    cor_tan = np.clip(bm_luma * np.array([[[1.65, 1.50, 1.24]]]) + 0.25, 0.24, 0.62)
    cor_alb_p = TEX_DIR / "corrugated_tan_albedo.png"
    Image.fromarray((cor_tan * 255).astype(np.uint8)).save(cor_alb_p)
    paths["corrugated_tan"] = {
        "albedo": str(cor_alb_p),
        "normal": str(bm_dir / "box_profile_metal_sheet_normal.png"),
        "roughness": str(bm_dir / "box_profile_metal_sheet_roughness.png"),
    }

    # 5) Car 1 & Car 2 Rear Fascia / Taillights / License Plate (512x256)
    def _make_car_rear(filename: str, body_rgb: tuple[int, int, int]) -> str:
        im = Image.new("RGB", (512, 256), body_rgb)
        dr = ImageDraw.Draw(im)
        # Trunk horizontal crease & lower bumper gap
        dr.rectangle([8, 162, 504, 168], fill=(22, 24, 26))
        dr.rectangle([8, 54, 504, 60], fill=(28, 30, 32))
        # Wide horizontal red/amber/clear wraparound taillights
        for lx0, lx1 in ((14, 160), (352, 498)):
            dr.rounded_rectangle([lx0, 74, lx1, 146], radius=8, fill=(24, 18, 18))
            dr.rounded_rectangle([lx0 + 4, 78, lx1 - 4, 118], radius=5, fill=(205, 28, 22))
            dr.rounded_rectangle([lx0 + 4, 120, lx1 - 4, 142], radius=3, fill=(218, 140, 36))
        # Inner white reverse light lenses
        dr.rectangle([126, 84, 156, 114], fill=(228, 226, 218))
        dr.rectangle([356, 84, 386, 114], fill=(228, 226, 218))
        # Center dark trunk trim & white Gulf license plate
        dr.rectangle([162, 82, 350, 142], fill=(26, 28, 30))
        dr.rounded_rectangle(
            [192, 92, 320, 134], radius=4, fill=(238, 238, 234), outline=(45, 48, 52), width=2
        )
        for i, ch_x in enumerate(range(204, 310, 15)):
            bh = 16 if i != 3 else 7
            dr.rectangle([ch_x, 104, ch_x + 8, 104 + bh], fill=(32, 34, 38))
        p = TEX_DIR / filename
        im.save(p)
        return str(p)

    paths["car1_rear"] = {"albedo": _make_car_rear("car1_rear_albedo.png", (38, 56, 48))}
    paths["car2_rear"] = {"albedo": _make_car_rear("car2_rear_albedo.png", (202, 206, 210))}

    return paths


# ---------------------------------------------------------------------------
# 2. Geometry Helpers with Explicit Clean UVs
# ---------------------------------------------------------------------------


def _xform_mesh(
    mesh: MeshData,
    location: Vec3 = (0.0, 0.0, 0.0),
    rotation_deg: Vec3 = (0.0, 0.0, 0.0),
    scale: Vec3 = (1.0, 1.0, 1.0),
) -> MeshData:
    mat = compose_matrix(location=location, rotation_deg=rotation_deg, scale=scale)
    return MeshData(
        vertices=transform_points(mat, mesh.vertices),
        faces=mesh.faces.copy(),
        uvs=None if mesh.uvs is None else mesh.uvs.copy(),
        face_groups=None if mesh.face_groups is None else mesh.face_groups.copy(),
    )


def _uv_quad_wall(
    p_bl: Vec3,
    p_br: Vec3,
    p_tr: Vec3,
    p_tl: Vec3,
    uv_rect: tuple[float, float, float, float] = (0.002, 0.002, 0.998, 0.998),
) -> MeshData:
    u0, v0, u1, v1 = uv_rect
    verts = np.array([p_bl, p_br, p_tr, p_tl], dtype=np.float64)
    faces = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int64)
    uvs = np.array([[u0, v0], [u1, v0], [u1, v1], [u0, v1]], dtype=np.float64)
    return MeshData(vertices=verts, faces=faces, uvs=uvs)


def _uv_box(
    size: Vec3,
    center: Vec3,
    *,
    uv_mode: str = "unit",
    uv_scale: float = 1.0,
    rotation_deg: Vec3 = (0.0, 0.0, 0.0),
) -> MeshData:
    sx, sy, sz = size[0] * 0.5, size[1] * 0.5, size[2] * 0.5
    face_defs = [
        ([(-sx, -sy, -sz), (-sx, sy, -sz), (-sx, sy, sz), (-sx, -sy, sz)], size[1], size[2]),
        ([(sx, sy, -sz), (sx, -sy, -sz), (sx, -sy, sz), (sx, sy, sz)], size[1], size[2]),
        ([(sx, -sy, -sz), (-sx, -sy, -sz), (-sx, -sy, sz), (sx, -sy, sz)], size[0], size[2]),
        ([(-sx, sy, -sz), (sx, sy, -sz), (sx, sy, sz), (-sx, sy, sz)], size[0], size[2]),
        ([(-sx, sy, -sz), (-sx, -sy, -sz), (sx, -sy, -sz), (sx, sy, -sz)], size[0], size[1]),
        ([(-sx, -sy, sz), (sx, -sy, sz), (sx, sy, sz), (-sx, sy, sz)], size[0], size[1]),
    ]
    verts_l: list[Vec3] = []
    faces_l: list[list[int]] = []
    uvs_l: list[Vec2] = []
    for pts, w_dim, h_dim in face_defs:
        base = len(verts_l)
        verts_l.extend(pts)
        faces_l.extend([[base, base + 1, base + 2], [base, base + 2, base + 3]])
        if uv_mode == "world":
            uw, vh = w_dim * uv_scale, h_dim * uv_scale
            uvs_l.extend([(0.0, 0.0), (uw, 0.0), (uw, vh), (0.0, vh)])
        else:
            uvs_l.extend([(0.005, 0.005), (0.995, 0.005), (0.995, 0.995), (0.005, 0.995)])
    m = MeshData(
        vertices=np.asarray(verts_l, dtype=np.float64),
        faces=np.asarray(faces_l, dtype=np.int64),
        uvs=np.asarray(uvs_l, dtype=np.float64),
    )
    return _xform_mesh(m, location=center, rotation_deg=rotation_deg)


def _uv_ground_rotated(
    x0: float,
    x1: float,
    y0: float,
    y1: float,
    z: float,
    *,
    repeat_per_m: float = 0.74,
    angle_deg: float = 45.0,
    nx: int = 12,
    ny: int = 44,
) -> MeshData:
    xs = np.linspace(x0, x1, nx + 1)
    ys = np.linspace(y0, y1, ny + 1)
    gx, gy = np.meshgrid(xs, ys, indexing="xy")
    verts = np.column_stack([gx.ravel(), gy.ravel(), np.full(gx.size, z, dtype=np.float64)])
    rad = math.radians(angle_deg)
    ca, sa = math.cos(rad), math.sin(rad)
    u = (verts[:, 0] * ca - verts[:, 1] * sa) * repeat_per_m
    v = (verts[:, 0] * sa + verts[:, 1] * ca) * repeat_per_m
    uvs = np.column_stack([u, v])
    faces: list[list[int]] = []
    stride = nx + 1
    for j in range(ny):
        for i in range(nx):
            a = j * stride + i
            b = a + 1
            c = a + stride + 1
            d = a + stride
            faces.append([a, b, c])
            faces.append([a, c, d])
    return MeshData(vertices=verts, faces=np.asarray(faces, dtype=np.int64), uvs=uvs)


def _corrugated_sheet(
    x_base: float,
    y0: float,
    y1: float,
    z0: float,
    z1: float,
    *,
    rib_pitch: float = 0.11,
    rib_depth: float = 0.014,
) -> MeshData:
    n_ribs = max(4, int(round((z1 - z0) / rib_pitch)))
    zs = np.linspace(z0, z1, n_ribs * 4 + 1)
    xs = x_base + rib_depth * np.sin(np.arange(len(zs)) * (math.pi * 0.5))
    verts: list[Vec3] = []
    uvs: list[Vec2] = []
    faces: list[list[int]] = []
    n_y = 20
    ys = np.linspace(y0, y1, n_y + 1)
    for yv in ys:
        u_val = (yv - y0) * 0.55
        for xv, zv in zip(xs, zs, strict=True):
            v_val = (zv - z0) / max(0.1, (z1 - z0))
            verts.append((float(xv), float(yv), float(zv)))
            uvs.append((u_val, v_val))
    stride = len(zs)
    for j in range(n_y):
        for i in range(stride - 1):
            a = j * stride + i
            b = a + 1
            c = a + stride + 1
            d = a + stride
            faces.append([a, b, c])
            faces.append([a, c, d])
    return MeshData(
        vertices=np.asarray(verts, dtype=np.float64),
        faces=np.asarray(faces, dtype=np.int64),
        uvs=np.asarray(uvs, dtype=np.float64),
    )


def _add_obj(
    scene: SceneDocument,
    oid: str,
    name: str,
    mesh: MeshData,
    material_id: str,
    collection_id: str = "default",
    location: Vec3 = (0.0, 0.0, 0.0),
    rotation_deg: Vec3 = (0.0, 0.0, 0.0),
) -> None:
    scene.objects[oid] = SceneObject(
        id=oid,
        name=name,
        kind="mesh",
        collection_id=collection_id,
        material_id=material_id,
        transform=Transform(location=location, rotation_deg=rotation_deg),
        mesh=mesh.to_payload(),
    )


# ---------------------------------------------------------------------------
# 3. Hero Prop: Twin Weathered Metal Utility Boxes
# ---------------------------------------------------------------------------


def _build_utility_box_pair(scene: SceneDocument) -> None:
    """Construct the two weathered sheet-metal utility boxes on the left foreground curb."""
    _add_obj(
        scene,
        "box_curb_plinth",
        "UtilityBox_ConcretePlinth",
        _uv_box((0.72, 1.92, 0.09), (-1.75, 1.14, 0.045), uv_mode="world", uv_scale=0.6),
        "mat_concrete",
        collection_id="props_fg",
    )
    # Small crushed cinder block under front corner leg of Box 1
    _add_obj(
        scene,
        "box_under_brick",
        "UtilityBox_SupportBlock",
        _uv_box(
            (0.14, 0.17, 0.065),
            (-1.54, 0.43, 0.112),
            uv_mode="world",
            rotation_deg=(0.0, 0.0, 12.0),
        ),
        "mat_concrete_dark",
        collection_id="props_fg",
    )

    def _build_single_box(
        prefix: str,
        y_start: float,
        y_end: float,
        x_back: float,
        x_front: float,
        z_leg_bot: float,
        z_top_front: float,
        z_top_back: float,
        panel_mat: str,
    ) -> None:
        cx = 0.5 * (x_back + x_front)
        cy = 0.5 * (y_start + y_end)
        w_x = x_front - x_back
        l_y = y_end - y_start
        z_box_bot = z_leg_bot + 0.072
        z_grille_bot = z_top_front - 0.185
        leg_t = 0.022

        # Thin rusty corner legs + center padlock latch only (avoid dark wireframe cage look)
        rust_parts: list[MeshData] = []
        for lx in (x_back + leg_t * 0.5, x_front - leg_t * 0.5):
            for ly in (y_start + leg_t * 0.5, y_end - leg_t * 0.5):
                zt = z_top_back if lx < cx else z_top_front
                rust_parts.append(
                    _uv_box(
                        (leg_t, leg_t, zt - z_leg_bot),
                        (lx, ly, 0.5 * (z_leg_bot + zt)),
                        uv_mode="world",
                    )
                )
        # Center door latch & padlock clasp
        rust_parts.append(
            _uv_box((0.030, 0.044, 0.065), (x_front + 0.005, cy, 0.5 * (z_box_bot + z_grille_bot)))
        )
        _add_obj(
            scene,
            f"{prefix}_rust_frame",
            f"{prefix}_RustFrame",
            concat_meshes(rust_parts),
            "mat_box_rust",
            "props_fg",
        )

        # Cream-with-rust-edge perforated upper vent grille (2 rows of punched rectangular holes)
        # plus cream horizontal door seam rails
        grille_parts: list[MeshData] = []
        for z_rail in (z_box_bot, 0.5 * (z_box_bot + z_grille_bot), z_grille_bot):
            grille_parts.append(_uv_box((w_x + 0.006, 0.018, 0.022), (cx, y_start + 0.009, z_rail)))
            grille_parts.append(_uv_box((0.018, l_y + 0.006, 0.022), (x_front - 0.009, cy, z_rail)))
            grille_parts.append(_uv_box((w_x + 0.006, 0.018, 0.022), (cx, y_end - 0.009, z_rail)))
        # Horizontal middle crossbar splitting the vent grille into 2 rows of openings
        z_mid_grille = 0.5 * (z_grille_bot + z_top_front)
        grille_parts.append(
            _uv_box((0.022, l_y - 0.02, 0.020), (x_front - 0.010, cy, z_mid_grille))
        )
        grille_parts.append(
            _uv_box((w_x - 0.02, 0.022, 0.020), (cx, y_start + 0.010, z_mid_grille + 0.025))
        )

        n_bars_front = 8
        for bi in range(n_bars_front):
            by = y_start + 0.042 + (bi + 0.5) * (l_y - 0.084) / n_bars_front
            grille_parts.append(
                _uv_box(
                    (0.022, 0.032, z_top_front - z_grille_bot),
                    (x_front - 0.010, by, 0.5 * (z_top_front + z_grille_bot)),
                )
            )
        n_bars_side = 4
        for bi in range(n_bars_side):
            bx = x_back + 0.045 + (bi + 0.5) * (w_x - 0.09) / n_bars_side
            frac = (x_front - bx) / max(1e-4, w_x)
            zt_here = z_top_front + frac * (z_top_back - z_top_front)
            grille_parts.append(
                _uv_box(
                    (0.032, 0.022, zt_here - z_grille_bot),
                    (bx, y_start + 0.010, 0.5 * (zt_here + z_grille_bot)),
                )
            )
        _add_obj(
            scene,
            f"{prefix}_grille",
            f"{prefix}_PerforatedGrille",
            concat_meshes(grille_parts),
            "mat_box_grille",
            "props_fg",
        )

        # Solid lower/mid painted sheet-metal cabinet body + sloped overhang lid
        slope_deg = math.degrees(math.atan2(z_top_back - z_top_front, w_x))
        lid_len = math.hypot(w_x + 0.04, z_top_back - z_top_front)
        panel_parts: list[MeshData] = [
            _uv_box(
                (w_x - 0.016, l_y - 0.016, z_grille_bot - z_box_bot),
                (cx, cy, 0.5 * (z_box_bot + z_grille_bot)),
            ),
            _uv_box(
                (lid_len, l_y + 0.032, 0.026),
                (cx + 0.005, cy, 0.5 * (z_top_front + z_top_back) + 0.014),
                rotation_deg=(0.0, slope_deg, 0.0),
            ),
        ]
        _add_obj(
            scene,
            f"{prefix}_panels",
            f"{prefix}_Panels",
            concat_meshes(panel_parts),
            panel_mat,
            "props_fg",
        )

        # Dark recessed interior ONLY inside the upper grille cavity
        _add_obj(
            scene,
            f"{prefix}_interior",
            f"{prefix}_UpperCavity",
            _uv_box(
                (w_x - 0.11, l_y - 0.11, z_top_front - z_grille_bot - 0.018),
                (cx, cy, 0.5 * (z_grille_bot + z_top_front)),
            ),
            "mat_dark_cavity",
            "props_fg",
        )

    _build_single_box("util_box_1", 0.30, 1.14, -2.08, -1.46, 0.09, 1.14, 1.24, "mat_box1_panel")
    _build_single_box("util_box_2", 1.16, 1.98, -2.08, -1.47, 0.09, 1.13, 1.23, "mat_box2_panel")

    # Thin black cable hugging the left wall from rear edge of Box 1
    cable_pts: list[Vec3] = [
        (-2.01, 0.56, 1.22),
        (-2.04, 0.50, 1.65),
        (-2.06, 0.42, 2.25),
        (-2.07, 0.34, 2.95),
        (-2.075, 0.25, 3.65),
    ]
    cable = sweep_curve(cable_pts, radius=0.0065, radial_segments=8, closed=False)
    _add_obj(scene, "util_box_cable", "UtilityBox_WallCable", cable, "mat_dark_cable", "props_fg")


# ---------------------------------------------------------------------------
# 4. Sculpted Parked Vehicles
# ---------------------------------------------------------------------------


def _loft_closed_rings(rings: list[list[Vec3]]) -> MeshData:
    """Loft a sequence of equal-vertex-count closed 3D rings with end caps."""
    n_s = len(rings)
    n_p = len(rings[0])
    verts: list[Vec3] = [pt for ring in rings for pt in ring]
    faces: list[list[int]] = []
    uvs: list[Vec2] = []
    for si in range(n_s):
        u_s = si / max(1, n_s - 1)
        for pi in range(n_p):
            uvs.append((u_s, pi / n_p))
    for si in range(n_s - 1):
        b0 = si * n_p
        b1 = (si + 1) * n_p
        for pi in range(n_p):
            pj = (pi + 1) % n_p
            a, b, c, d = b0 + pi, b0 + pj, b1 + pj, b1 + pi
            faces.append([a, b, c])
            faces.append([a, c, d])
    # End caps (back cap at ring 0, front cap at ring -1)
    c_back = len(verts)
    pts0 = np.asarray(rings[0], dtype=np.float64)
    verts.append((float(pts0[:, 0].mean()), float(pts0[:, 1].mean()), float(pts0[:, 2].mean())))
    uvs.append((0.0, 0.5))
    for pi in range(n_p):
        pj = (pi + 1) % n_p
        faces.append([c_back, pj, pi])

    c_front = len(verts)
    pts1 = np.asarray(rings[-1], dtype=np.float64)
    verts.append((float(pts1[:, 0].mean()), float(pts1[:, 1].mean()), float(pts1[:, 2].mean())))
    uvs.append((1.0, 0.5))
    base_f = (n_s - 1) * n_p
    for pi in range(n_p):
        pj = (pi + 1) % n_p
        faces.append([c_front, base_f + pi, base_f + pj])

    return MeshData(
        vertices=np.asarray(verts, dtype=np.float64),
        faces=np.asarray(faces, dtype=np.int64),
        uvs=np.asarray(uvs, dtype=np.float64),
    )


def _sedan_ring(
    y: float, w_bot: float, w_belt: float, w_top: float, z_bot: float, z_belt: float, z_top: float
) -> list[Vec3]:
    """8-point symmetric automotive cross-section ring (CW looking from -Y)."""
    return [
        (-w_bot * 0.5, y, z_bot),
        (-w_belt * 0.5, y, z_belt),
        (-w_top * 0.5, y, z_top),
        (w_top * 0.5, y, z_top),
        (w_belt * 0.5, y, z_belt),
        (w_bot * 0.5, y, z_bot),
    ]


def _build_sedan(
    scene: SceneDocument,
    prefix: str,
    center_xy: Vec2,
    yaw_deg: float,
    paint_mat: str,
    rear_mat: str,
    *,
    scale_xyz: Vec3 = (1.0, 1.0, 1.0),
) -> None:
    """Build a sculpted multi-section 1998-2001 sedan facing +Y (rear visible to camera)."""
    parts_body: list[MeshData] = []
    parts_glass: list[MeshData] = []
    parts_tire: list[MeshData] = []
    parts_rim: list[MeshData] = []

    # Sculpted tapered lower body (narrower at bumpers, wider at wheel fenders & beltline)
    lower_rings = [
        _sedan_ring(-2.22, 1.48, 1.56, 1.44, 0.28, 0.54, 0.87),  # Rear trunk edge
        _sedan_ring(-1.90, 1.62, 1.70, 1.56, 0.27, 0.55, 0.90),  # Rear quarter panel
        _sedan_ring(-1.10, 1.68, 1.74, 1.58, 0.26, 0.56, 0.92),  # Rear door / wheel arch
        _sedan_ring(0.50, 1.68, 1.74, 1.58, 0.26, 0.56, 0.91),  # Front door / fender
        _sedan_ring(1.65, 1.64, 1.68, 1.52, 0.27, 0.55, 0.82),  # Hood mid
        _sedan_ring(2.18, 1.46, 1.52, 1.38, 0.28, 0.52, 0.64),  # Front bumper
    ]
    parts_body.append(_loft_closed_rings(lower_rings))

    # Tapered greenhouse / cabin with sloped C-pillar and tumblehome side windows
    cabin_rings = [
        _sedan_ring(-1.62, 1.50, 1.44, 1.24, 0.88, 0.96, 1.02),  # Base of rear glass
        _sedan_ring(-1.05, 1.46, 1.34, 1.20, 0.90, 1.18, 1.38),  # Rear C-pillar top
        _sedan_ring(-0.20, 1.46, 1.34, 1.20, 0.90, 1.20, 1.43),  # Roof crown
        _sedan_ring(0.52, 1.44, 1.32, 1.18, 0.90, 1.18, 1.38),  # Front A-pillar top
        _sedan_ring(1.12, 1.48, 1.42, 1.28, 0.88, 0.94, 0.96),  # Windshield base
    ]
    parts_body.append(_loft_closed_rings(cabin_rings))

    # Side mirrors
    for sx_sign in (-1.0, 1.0):
        parts_body.append(_uv_box((0.14, 0.10, 0.08), (sx_sign * 0.88, 0.62, 0.97)))

    # Sculpted rear & front wraparound dark lower bumper trim
    parts_tire.append(_uv_box((1.56, 0.14, 0.18), (0.0, -2.20, 0.34)))
    parts_tire.append(_uv_box((1.52, 0.14, 0.20), (0.0, 2.16, 0.35)))

    # Rear windshield, front windshield, tapered side windows
    parts_glass.append(
        _uv_box((1.16, 0.60, 0.03), (0.0, -1.33, 1.18), rotation_deg=(38.0, 0.0, 0.0))
    )
    parts_glass.append(
        _uv_box((1.16, 0.62, 0.03), (0.0, 0.82, 1.16), rotation_deg=(-36.0, 0.0, 0.0))
    )
    for sx_sign in (-1.0, 1.0):
        parts_glass.append(
            _uv_box(
                (0.03, 1.40, 0.35),
                (sx_sign * 0.66, -0.25, 1.16),
                rotation_deg=(0.0, sx_sign * 16.0, 0.0),
            )
        )

    for wx in (-0.77, 0.77):
        for wy in (-1.38, 1.36):
            tire = make_cylinder(radius=0.295, height=0.20, segments=22, origin="center")
            tire = _xform_mesh(tire, location=(wx, wy, 0.295), rotation_deg=(0.0, 90.0, 0.0))
            parts_tire.append(tire)
            rim = make_cylinder(radius=0.185, height=0.215, segments=14, origin="center")
            rim = _xform_mesh(rim, location=(wx, wy, 0.295), rotation_deg=(0.0, 90.0, 0.0))
            parts_rim.append(rim)

    # Rear fascia UV-mapped quad placed PROUD of the trunk face (y = -2.235)
    # UVs are preserved because concat_meshes now keeps UVs!
    rear_panel = _uv_quad_wall(
        (-0.74, -2.235, 0.46),
        (0.74, -2.235, 0.46),
        (0.74, -2.235, 0.85),
        (-0.74, -2.235, 0.85),
    )

    x0, y0 = center_xy
    for suffix, mesh_list, mat_id in (
        ("body", parts_body, paint_mat),
        ("glass", parts_glass, "mat_car_glass"),
        ("trim", parts_tire, "mat_dark_rubber"),
        ("rims", parts_rim, "mat_alloy_rim"),
        ("rear", [rear_panel], rear_mat),
    ):
        merged = concat_meshes(mesh_list)
        merged = _xform_mesh(
            merged, location=(x0, y0, 0.0), rotation_deg=(0.0, 0.0, yaw_deg), scale=scale_xyz
        )
        _add_obj(scene, f"{prefix}_{suffix}", f"{prefix}_{suffix}", merged, mat_id, "vehicles")


def _build_far_suv(scene: SceneDocument, center_xy: Vec2, yaw_deg: float) -> None:
    """Build the distant silver minivan/SUV parked facing the camera near the alley bend."""
    body = concat_meshes(
        [
            _uv_box((1.68, 4.00, 0.68), (0.0, 0.0, 0.58)),
            _uv_box((1.46, 2.75, 0.64), (0.0, -0.35, 1.24)),
        ]
    )
    glass = _uv_box((1.34, 0.08, 0.48), (0.0, 1.08, 1.21), rotation_deg=(18.0, 0.0, 0.0))
    # Front grille + headlights + dark front bumper facing -Y? Wait: +Y is forward, -Y faces camera!
    front_details = concat_meshes(
        [
            _uv_box((1.68, 0.14, 0.24), (0.0, -2.02, 0.34)),
            _uv_box((0.68, 0.08, 0.22), (0.0, -2.02, 0.64)),
        ]
    )
    headlamps = concat_meshes(
        [
            _uv_box((0.30, 0.08, 0.18), (-0.60, -2.02, 0.66)),
            _uv_box((0.30, 0.08, 0.18), (0.60, -2.02, 0.66)),
        ]
    )
    wheels: list[MeshData] = []
    for wx in (-0.78, 0.78):
        for wy in (-1.22, 1.22):
            wheels.append(
                _xform_mesh(
                    make_cylinder(radius=0.30, height=0.22, segments=16, origin="center"),
                    location=(wx, wy, 0.30),
                    rotation_deg=(0.0, 90.0, 0.0),
                )
            )
    x0, y0 = center_xy
    for suffix, m, mat_id in (
        ("body", body, "mat_car2_silver"),
        ("glass", glass, "mat_car_glass"),
        ("grille", front_details, "mat_dark_rubber"),
        ("lamps", headlamps, "mat_cream_wall"),
        ("wheels", concat_meshes(wheels), "mat_dark_rubber"),
    ):
        _add_obj(
            scene,
            f"car3_{suffix}",
            f"Car3_{suffix}",
            _xform_mesh(m, location=(x0, y0, 0.0), rotation_deg=(0.0, 0.0, yaw_deg)),
            mat_id,
            "vehicles",
        )


# ---------------------------------------------------------------------------
# 5. Scene Assembly
# ---------------------------------------------------------------------------


def build_gulf_alley_scene() -> tuple[SceneDocument, dict[str, dict[str, str]]]:
    tex_paths = prepare_alley_textures()
    scene = SceneDocument(
        name="Gulf_Alley_3D",
        collections={
            "default": CollectionSpec(id="default", name="Default"),
            "ground": CollectionSpec(id="ground", name="01_Ground_Pavers"),
            "buildings": CollectionSpec(id="buildings", name="02_Buildings_Walls"),
            "props_fg": CollectionSpec(id="props_fg", name="03_Utility_Boxes_Pipes"),
            "vehicles": CollectionSpec(id="vehicles", name="04_Parked_Cars"),
            "distant": CollectionSpec(id="distant", name="05_Distant_Skyline"),
        },
        world=WorldEnvironment(
            sky_mode="hdri",
            hdri_path=tex_paths["hdri"]["path"],
            hdri_rotation_deg=110.0,
            hdri_intensity=1.18,
            ambient_strength=0.26,
            background_color=(0.76, 0.78, 0.81),
            horizon_color=(0.82, 0.83, 0.85),
            zenith_color=(0.72, 0.75, 0.79),
        ),
        post_process=PostProcessConfig(
            enabled=True,
            denoise_enabled=True,
            denoise_strength=0.92,
            ssao_enabled=True,
            ssao_radius=4.5,
            ssao_strength=0.36,
            bloom_enabled=False,
            dof_enabled=False,
            tone_mapper="aces",
            exposure=0.14,
            contrast=1.05,
            saturation=0.95,
            vignette_intensity=0.06,
            sharpen=0.05,
            fxaa_enabled=True,
        ),
    )

    mats = [
        MaterialSpec(
            id="mat_pavers_grey",
            name="Grey Herringbone Pavers",
            base_color=(1.0, 1.0, 1.0, 1.0),
            roughness=0.86,
            albedo_map=tex_paths["pavers_grey"]["albedo"],
            normal_map=tex_paths["pavers_grey"]["normal"],
            roughness_map=tex_paths["pavers_grey"]["roughness"],
            ao_map=tex_paths["pavers_grey"]["ao"],
            normal_strength=0.80,
        ),
        MaterialSpec(
            id="mat_pavers_red",
            name="Terracotta Sidewalk Pavers",
            base_color=(1.0, 1.0, 1.0, 1.0),
            roughness=0.86,
            albedo_map=tex_paths["pavers_red"]["albedo"],
            normal_map=tex_paths["pavers_red"]["normal"],
            roughness_map=tex_paths["pavers_red"]["roughness"],
            ao_map=tex_paths["pavers_red"]["ao"],
            normal_strength=0.75,
        ),
        MaterialSpec(
            id="mat_wall_left_fg",
            name="Left Stucco Wall Weathered",
            base_color=(1.0, 1.0, 1.0, 1.0),
            roughness=0.90,
            albedo_map=tex_paths["wall_left_fg"]["albedo"],
            normal_map=tex_paths["wall_left_fg"]["normal"],
            roughness_map=tex_paths["wall_left_fg"]["roughness"],
            normal_strength=0.06,
        ),
        MaterialSpec(
            id="mat_wall_right_fg",
            name="Right Stucco Wall Weathered",
            base_color=(1.0, 1.0, 1.0, 1.0),
            roughness=0.90,
            albedo_map=tex_paths["wall_right_fg"]["albedo"],
            normal_map=tex_paths["wall_right_fg"]["normal"],
            roughness_map=tex_paths["wall_right_fg"]["roughness"],
            normal_strength=0.06,
        ),
        MaterialSpec(
            id="mat_wall_olive",
            name="Olive Beige Plaster",
            base_color=(1.0, 1.0, 1.0, 1.0),
            roughness=0.90,
            albedo_map=tex_paths["wall_olive"]["albedo"],
            normal_map=tex_paths["wall_olive"]["normal"],
            roughness_map=tex_paths["wall_olive"]["roughness"],
            normal_strength=0.06,
        ),
        MaterialSpec(
            id="mat_concrete",
            name="Weathered Concrete",
            base_color=(0.64, 0.63, 0.61, 1.0),
            roughness=0.86,
            albedo_map=str(RAW_TEX_DIR / "ambientcg_Concrete034_1k" / "Concrete034_albedo.jpg"),
            normal_map=str(RAW_TEX_DIR / "ambientcg_Concrete034_1k" / "Concrete034_normal.jpg"),
            roughness_map=str(
                RAW_TEX_DIR / "ambientcg_Concrete034_1k" / "Concrete034_roughness.jpg"
            ),
            normal_strength=0.30,
        ),
        MaterialSpec(
            id="mat_concrete_dark",
            name="Dark Concrete / Manhole",
            base_color=(0.36, 0.36, 0.35, 1.0),
            roughness=0.84,
            albedo_map=str(RAW_TEX_DIR / "ambientcg_Concrete034_1k" / "Concrete034_albedo.jpg"),
            normal_map=str(RAW_TEX_DIR / "ambientcg_Concrete034_1k" / "Concrete034_normal.jpg"),
            normal_strength=0.30,
        ),
        MaterialSpec(
            id="mat_box1_panel",
            name="Utility Box 1 Painted Metal",
            base_color=(1.0, 1.0, 1.0, 1.0),
            metallic=0.04,
            roughness=0.72,
            albedo_map=tex_paths["box1_panel"]["albedo"],
            normal_map=tex_paths["box1_panel"]["normal"],
            roughness_map=tex_paths["box1_panel"]["roughness"],
            normal_strength=0.12,
        ),
        MaterialSpec(
            id="mat_box2_panel",
            name="Utility Box 2 Painted Metal",
            base_color=(1.0, 1.0, 1.0, 1.0),
            metallic=0.04,
            roughness=0.72,
            albedo_map=tex_paths["box2_panel"]["albedo"],
            normal_map=tex_paths["box2_panel"]["normal"],
            roughness_map=tex_paths["box2_panel"]["roughness"],
            normal_strength=0.12,
        ),
        MaterialSpec(
            id="mat_box_grille",
            name="Utility Box Vent Grille",
            base_color=(1.0, 1.0, 1.0, 1.0),
            metallic=0.06,
            roughness=0.74,
            albedo_map=tex_paths["box_grille"]["albedo"],
            normal_map=tex_paths["box_grille"]["normal"],
            roughness_map=tex_paths["box_grille"]["roughness"],
            normal_strength=0.14,
        ),
        MaterialSpec(
            id="mat_box_rust",
            name="Oxidized Rusted Steel",
            base_color=(1.0, 1.0, 1.0, 1.0),
            metallic=0.18,
            roughness=0.86,
            albedo_map=tex_paths["box_rust"]["albedo"],
            normal_map=tex_paths["box_rust"]["normal"],
            roughness_map=tex_paths["box_rust"]["roughness"],
            normal_strength=0.25,
        ),
        MaterialSpec(
            id="mat_dark_cavity",
            name="Box Dark Interior",
            base_color=(0.05, 0.04, 0.03, 1.0),
            roughness=0.95,
        ),
        MaterialSpec(
            id="mat_dark_cable",
            name="Black Rubber Cable",
            base_color=(0.09, 0.09, 0.10, 1.0),
            roughness=0.68,
        ),
        MaterialSpec(
            id="mat_pvc_pipe",
            name="Weathered Light Grey PVC",
            base_color=(0.72, 0.72, 0.70, 1.0),
            roughness=0.52,
        ),
        MaterialSpec(
            id="mat_alu_frame",
            name="Anodized Window Frame",
            base_color=(0.36, 0.38, 0.40, 1.0),
            metallic=0.60,
            roughness=0.45,
        ),
        MaterialSpec(
            id="mat_fence_lower",
            name="Compound Wall Ribbed Concrete",
            base_color=(1.0, 1.0, 1.0, 1.0),
            roughness=0.88,
            albedo_map=tex_paths["fence_lower"]["albedo"],
            normal_map=tex_paths["fence_lower"]["normal"],
            roughness_map=tex_paths["fence_lower"]["roughness"],
            normal_strength=0.24,
        ),
        MaterialSpec(
            id="mat_corrugated_tan",
            name="Corrugated Metal Sheet Tan",
            base_color=(1.0, 1.0, 1.0, 1.0),
            metallic=0.08,
            roughness=0.78,
            albedo_map=tex_paths["corrugated_tan"]["albedo"],
            normal_map=tex_paths["corrugated_tan"]["normal"],
            roughness_map=tex_paths["corrugated_tan"]["roughness"],
            normal_strength=0.16,
        ),
        MaterialSpec(
            id="mat_window_dark",
            name="Recessed Dark Window",
            base_color=(0.10, 0.12, 0.14, 1.0),
            roughness=0.12,
            clearcoat=0.80,
            clearcoat_roughness=0.06,
        ),
        MaterialSpec(
            id="mat_window_bars",
            name="Window Iron Grille",
            base_color=(0.15, 0.15, 0.16, 1.0),
            metallic=0.60,
            roughness=0.56,
        ),
        MaterialSpec(
            id="mat_car1_green",
            name="Car1 Forest Green Metallic",
            base_color=(0.13, 0.21, 0.17, 1.0),
            metallic=0.32,
            roughness=0.24,
            clearcoat=0.92,
            clearcoat_roughness=0.05,
        ),
        MaterialSpec(
            id="mat_car2_silver",
            name="Car2 Silver Metallic",
            base_color=(0.78, 0.80, 0.82, 1.0),
            metallic=0.34,
            roughness=0.24,
            clearcoat=0.88,
            clearcoat_roughness=0.06,
        ),
        MaterialSpec(
            id="mat_car1_rear",
            name="Car1 Rear Fascia",
            base_color=(1.0, 1.0, 1.0, 1.0),
            roughness=0.26,
            clearcoat=0.70,
            albedo_map=tex_paths["car1_rear"]["albedo"],
        ),
        MaterialSpec(
            id="mat_car2_rear",
            name="Car2 Rear Fascia",
            base_color=(1.0, 1.0, 1.0, 1.0),
            roughness=0.26,
            clearcoat=0.70,
            albedo_map=tex_paths["car2_rear"]["albedo"],
        ),
        MaterialSpec(
            id="mat_car_glass",
            name="Tinted Auto Glass",
            base_color=(0.08, 0.10, 0.13, 1.0),
            roughness=0.05,
            clearcoat=0.95,
            clearcoat_roughness=0.03,
        ),
        MaterialSpec(
            id="mat_dark_rubber",
            name="Tire & Bumper Trim",
            base_color=(0.10, 0.10, 0.11, 1.0),
            roughness=0.78,
        ),
        MaterialSpec(
            id="mat_alloy_rim",
            name="Wheel Alloy Rim",
            base_color=(0.68, 0.70, 0.72, 1.0),
            metallic=0.80,
            roughness=0.30,
        ),
        MaterialSpec(
            id="mat_ochre_wall",
            name="Distant Ochre Building",
            base_color=(0.79, 0.59, 0.28, 1.0),
            roughness=0.84,
        ),
        MaterialSpec(
            id="mat_cream_wall",
            name="Distant Cream Stucco",
            base_color=(0.76, 0.74, 0.69, 1.0),
            roughness=0.86,
            albedo_map=str(
                RAW_TEX_DIR / "polyhaven_white_stucco_02_1k" / "white_stucco_02_albedo.png"
            ),
            normal_map=str(
                RAW_TEX_DIR / "polyhaven_white_stucco_02_1k" / "white_stucco_02_normal.png"
            ),
            normal_strength=0.05,
        ),
        MaterialSpec(
            id="mat_hood_cream",
            name="Window Concrete Hood Cream",
            base_color=(0.76, 0.70, 0.61, 1.0),
            roughness=0.86,
        ),
        MaterialSpec(
            id="mat_bg_white",
            name="Distant White Concrete",
            base_color=(0.80, 0.81, 0.82, 1.0),
            roughness=0.88,
        ),
        MaterialSpec(
            id="mat_wood_pergola",
            name="Weathered Roof Pergola Wood",
            base_color=(0.40, 0.34, 0.27, 1.0),
            roughness=0.86,
        ),
        MaterialSpec(
            id="mat_foliage",
            name="Alley Tree Foliage",
            base_color=(0.11, 0.18, 0.08, 1.0),
            roughness=0.88,
        ),
        MaterialSpec(
            id="mat_foliage_light",
            name="Alley Tree Foliage Olive",
            base_color=(0.17, 0.24, 0.11, 1.0),
            roughness=0.88,
        ),
        MaterialSpec(
            id="mat_flag_red",
            name="Flag Red",
            base_color=(0.76, 0.16, 0.15, 1.0),
            roughness=0.75,
        ),
        MaterialSpec(
            id="mat_orange_box",
            name="Orange Utility Sign",
            base_color=(0.76, 0.38, 0.16, 1.0),
            roughness=0.65,
        ),
    ]
    for m in mats:
        scene.materials[m.id] = m

    # Ground: Grey Herringbone Road + Manhole + Right Curb + Muted Sidewalk
    _add_obj(
        scene,
        "road_pavers",
        "Road_HerringboneGrey",
        _uv_ground_rotated(-2.05, 1.52, -3.5, 36.0, 0.0, repeat_per_m=0.74, angle_deg=45.0),
        "mat_pavers_grey",
        "ground",
    )
    _add_obj(
        scene,
        "road_manhole_frame",
        "Road_ManholeFrame",
        _uv_box((0.92, 0.92, 0.024), (-0.24, 11.2, 0.008), uv_mode="world"),
        "mat_concrete",
        "ground",
    )
    _add_obj(
        scene,
        "road_manhole_lid",
        "Road_ManholeLid",
        _uv_box((0.68, 0.68, 0.027), (-0.24, 11.2, 0.011), uv_mode="world"),
        "mat_concrete_dark",
        "ground",
    )
    _add_obj(
        scene,
        "right_curb",
        "RightSidewalk_Curb",
        _uv_box((0.18, 35.0, 0.11), (1.60, 15.5, 0.045), uv_mode="world", uv_scale=0.5),
        "mat_concrete",
        "ground",
    )
    _add_obj(
        scene,
        "right_sidewalk",
        "RightSidewalk_RedPavers",
        _uv_ground_rotated(
            1.68, 2.75, 0.0, 32.0, 0.075, repeat_per_m=0.78, angle_deg=45.0, nx=4, ny=28
        ),
        "mat_pavers_red",
        "ground",
    )

    # Left FG Building & Windows
    _add_obj(
        scene,
        "left_fg_building",
        "LeftFG_BuildingMass",
        _uv_box((2.5, 12.6, 5.2), (-3.35, 4.3, 2.6), uv_mode="world", uv_scale=0.25),
        "mat_cream_wall",
        "buildings",
    )
    _add_obj(
        scene,
        "left_fg_facade",
        "LeftFG_FacadeWeathered",
        _uv_quad_wall(
            (-2.09, -2.0, 0.0), (-2.09, 10.5, 0.0), (-2.09, 10.5, 4.7), (-2.09, -2.0, 4.7)
        ),
        "mat_wall_left_fg",
        "buildings",
    )

    # Left Window 1 + clean chamfered concrete hood + recessed dark bars
    _add_obj(
        scene,
        "left_win1_recess",
        "LeftWin1_Glass",
        _uv_box((0.08, 1.32, 1.52), (-2.095, 3.90, 2.25)),
        "mat_window_dark",
        "buildings",
    )
    hood_parts = [
        _uv_box((0.28, 1.54, 0.16), (-1.96, 3.90, 3.12), uv_mode="world"),
        _uv_box((0.26, 0.16, 0.30), (-1.97, 3.21, 2.94), uv_mode="world"),
        _uv_box((0.26, 0.16, 0.30), (-1.97, 4.59, 2.94), uv_mode="world"),
    ]
    _add_obj(
        scene,
        "left_win1_hood",
        "LeftWin1_ConcreteHood",
        concat_meshes(hood_parts),
        "mat_hood_cream",
        "buildings",
    )
    _add_obj(
        scene,
        "left_win1_sill",
        "LeftWin1_Sill",
        _uv_box((0.18, 1.48, 0.07), (-2.01, 3.90, 1.44), uv_mode="world"),
        "mat_hood_cream",
        "buildings",
    )
    bars1 = [_uv_box((0.014, 0.014, 1.50), (-2.05, 3.35 + bi * 0.18, 2.25)) for bi in range(7)]
    _add_obj(
        scene,
        "left_win1_bars",
        "LeftWin1_SecurityBars",
        concat_meshes(bars1),
        "mat_window_bars",
        "buildings",
    )

    # Left Window 2
    _add_obj(
        scene,
        "left_win2_recess",
        "LeftWin2_Glass",
        _uv_box((0.08, 1.05, 1.22), (-2.095, 6.75, 2.12)),
        "mat_window_dark",
        "buildings",
    )
    _add_obj(
        scene,
        "left_win2_hood",
        "LeftWin2_Brow",
        _uv_box((0.30, 1.24, 0.09), (-1.95, 6.75, 2.78), uv_mode="world"),
        "mat_hood_cream",
        "buildings",
    )
    bars2 = [_uv_box((0.014, 0.014, 1.20), (-2.05, 6.32 + bi * 0.17, 2.12)) for bi in range(6)]
    _add_obj(
        scene,
        "left_win2_bars",
        "LeftWin2_SecurityBars",
        concat_meshes(bars2),
        "mat_window_bars",
        "buildings",
    )

    # Left pipes
    _add_obj(
        scene,
        "left_pipe_1",
        "LeftWall_Conduit",
        _xform_mesh(
            make_cylinder(radius=0.016, height=4.5, segments=10, origin="bottom"),
            location=(-2.07, 2.55, 0.0),
        ),
        "mat_pvc_pipe",
        "props_fg",
    )
    _add_obj(
        scene,
        "left_pipe_2",
        "LeftWall_Downspout",
        _xform_mesh(
            make_cylinder(radius=0.042, height=4.6, segments=12, origin="bottom"),
            location=(-2.03, 9.65, 0.0),
        ),
        "mat_pvc_pipe",
        "props_fg",
    )

    # Midground Left Olive Building
    _add_obj(
        scene,
        "left_mid_building",
        "LeftMid_OliveBuilding",
        _uv_box((2.4, 14.2, 5.4), (-3.16, 17.6, 2.7), uv_mode="world", uv_scale=0.2),
        "mat_wall_olive",
        "buildings",
    )
    _add_obj(
        scene,
        "left_mid_door",
        "LeftMid_Door",
        _uv_box((0.10, 1.05, 2.25), (-1.93, 12.6, 1.12)),
        "mat_window_dark",
        "buildings",
    )
    _add_obj(
        scene,
        "left_mid_ac",
        "LeftMid_ACUnit",
        _uv_box((0.40, 0.76, 0.54), (-1.78, 17.2, 3.65)),
        "mat_cream_wall",
        "props_fg",
    )
    _add_obj(
        scene,
        "left_far_orange_box",
        "LeftFar_OrangeBox",
        _uv_box((0.26, 0.40, 0.62), (-1.85, 23.6, 0.85)),
        "mat_orange_box",
        "props_fg",
    )

    # Right FG Building + Revealed Window + Pipe + Wire + Flush Meter Box
    _add_obj(
        scene,
        "right_fg_building",
        "RightFG_BuildingMass",
        _uv_box((2.8, 7.6, 5.4), (4.05, 5.2, 2.7), uv_mode="world", uv_scale=0.25),
        "mat_cream_wall",
        "buildings",
    )
    _add_obj(
        scene,
        "right_fg_facade",
        "RightFG_FacadeWeathered",
        _uv_quad_wall((2.64, 9.0, 0.0), (2.64, 1.4, 0.0), (2.64, 1.4, 4.8), (2.64, 9.0, 4.8)),
        "mat_wall_right_fg",
        "buildings",
    )
    # Square vent reveal + dark louver inset
    _add_obj(
        scene,
        "right_fg_vent_trim",
        "RightFG_VentReveal",
        _uv_box((0.05, 0.56, 0.56), (2.63, 6.45, 2.05), uv_mode="world"),
        "mat_concrete",
        "buildings",
    )
    _add_obj(
        scene,
        "right_fg_vent",
        "RightFG_VentOpening",
        _uv_box((0.05, 0.44, 0.44), (2.61, 6.45, 2.05)),
        "mat_dark_cavity",
        "buildings",
    )
    # Right window concrete surround + glass + silver-grey aluminum sliding frame
    _add_obj(
        scene,
        "right_fg_win_trim",
        "RightFG_WindowReveal",
        _uv_box((0.06, 1.24, 1.32), (2.63, 4.55, 2.08), uv_mode="world"),
        "mat_concrete",
        "buildings",
    )
    _add_obj(
        scene,
        "right_fg_window",
        "RightFG_Window",
        _uv_box((0.05, 1.12, 1.20), (2.605, 4.55, 2.08)),
        "mat_window_dark",
        "buildings",
    )
    win_r_frame = concat_meshes(
        [
            _uv_box((0.06, 1.16, 0.045), (2.59, 4.55, 1.48)),
            _uv_box((0.06, 1.16, 0.045), (2.59, 4.55, 2.68)),
            _uv_box((0.06, 0.045, 1.22), (2.59, 3.98, 2.08)),
            _uv_box((0.06, 0.045, 1.22), (2.59, 5.12, 2.08)),
            _uv_box((0.05, 0.036, 1.18), (2.58, 4.55, 2.08)),
        ]
    )
    _add_obj(
        scene,
        "right_fg_win_frame",
        "RightFG_WindowFrame",
        win_r_frame,
        "mat_alu_frame",
        "buildings",
    )

    right_pipe_pts: list[Vec3] = [
        (2.61, 6.95, 1.62),
        (2.60, 6.08, 1.18),
        (2.60, 4.80, 1.09),
        (2.61, 3.60, 1.02),
    ]
    _add_obj(
        scene,
        "right_fg_pipe",
        "RightFG_CurvedPipe",
        sweep_curve(right_pipe_pts, radius=0.024, radial_segments=10, closed=False),
        "mat_pvc_pipe",
        "props_fg",
    )
    right_wire_pts: list[Vec3] = [
        (2.61, 6.15, 4.25),
        (2.60, 6.02, 3.65),
        (2.61, 5.75, 3.12),
    ]
    _add_obj(
        scene,
        "right_fg_wire",
        "RightFG_HangingWire",
        sweep_curve(right_wire_pts, radius=0.007, radial_segments=8, closed=False),
        "mat_dark_cable",
        "props_fg",
    )
    _add_obj(
        scene,
        "right_fg_meter",
        "RightFG_RustMeterBox",
        _uv_box((0.07, 0.56, 0.86), (2.61, 2.15, 1.50)),
        "mat_box_grille",
        "props_fg",
    )

    # Right Compound Wall + Gentle 3D Corrugated Metal Fence
    _add_obj(
        scene,
        "right_fence_wall_mass",
        "RightFence_LowerMass",
        _uv_box((0.25, 14.5, 2.15), (2.62, 16.25, 1.075), uv_mode="world"),
        "mat_concrete",
        "buildings",
    )
    _add_obj(
        scene,
        "right_fence_lower",
        "RightFence_LowerGraffiti",
        _uv_quad_wall((2.49, 23.5, 0.0), (2.49, 9.0, 0.0), (2.49, 9.0, 2.15), (2.49, 23.5, 2.15)),
        "mat_fence_lower",
        "buildings",
    )
    _add_obj(
        scene,
        "right_fence_upper",
        "RightFence_CorrugatedMetal",
        _corrugated_sheet(2.52, 9.0, 23.5, 2.15, 3.22, rib_pitch=0.10, rib_depth=0.008),
        "mat_corrugated_tan",
        "buildings",
    )
    _add_obj(
        scene,
        "right_fence_cap",
        "RightFence_TopCap",
        _uv_box((0.10, 14.6, 0.07), (2.53, 16.25, 3.25)),
        "mat_concrete_dark",
        "buildings",
    )

    # Twin Hero Utility Boxes + 3 Parked Cars (rear-left 3/4 alignment matching photo!)
    _build_utility_box_pair(scene)
    _build_sedan(
        scene,
        "car1_green",
        (0.68, 10.6),
        yaw_deg=4.5,
        paint_mat="mat_car1_green",
        rear_mat="mat_car1_rear",
    )
    _build_sedan(
        scene,
        "car2_silver",
        (0.70, 16.0),
        yaw_deg=3.5,
        paint_mat="mat_car2_silver",
        rear_mat="mat_car2_rear",
        scale_xyz=(0.97, 0.96, 1.01),
    )
    _build_far_suv(scene, (-0.52, 28.0), yaw_deg=5.0)

    # Background Buildings, Rooftop Pergola, Dense Organic Foliage, Ochre Landmark & Wires
    _add_obj(
        scene,
        "bg_right_2story",
        "BG_RightTwoStory",
        _uv_box((5.5, 11.5, 6.2), (5.8, 18.8, 3.1), uv_mode="world", uv_scale=0.18),
        "mat_bg_white",
        "distant",
    )
    pergola_parts = [_uv_box((3.2, 0.14, 0.14), (4.3, 14.5 + i * 0.72, 6.75)) for i in range(9)] + [
        _uv_box((0.14, 6.4, 0.16), (3.1, 17.4, 6.65)),
        _uv_box((0.14, 6.4, 0.16), (5.2, 17.4, 6.65)),
        _uv_box((0.16, 0.16, 0.65), (3.1, 14.6, 6.42)),
        _uv_box((0.16, 0.16, 0.65), (3.1, 20.2, 6.42)),
    ]
    _add_obj(
        scene,
        "bg_roof_pergola",
        "BG_RoofPergola",
        concat_meshes(pergola_parts),
        "mat_wood_pergola",
        "distant",
    )
    _add_obj(
        scene,
        "bg_right_yellow",
        "BG_RightPaleYellow",
        _uv_box((4.5, 9.0, 8.2), (3.6, 28.5, 4.1)),
        "mat_hood_cream",
        "distant",
    )
    _add_obj(
        scene,
        "bg_bend_awning",
        "BG_BendAwning",
        _uv_box((1.9, 3.8, 0.10), (1.75, 25.0, 2.75), rotation_deg=(0.0, 8.0, 0.0)),
        "mat_concrete_dark",
        "distant",
    )

    # Organic foliage cluster (42 small varied scaled leaf-clump icospheres)
    rng_fol = np.random.default_rng(77)
    fol_dark: list[MeshData] = []
    fol_light: list[MeshData] = []
    for fi in range(42):
        fx = float(rng_fol.uniform(1.30, 2.38))
        fy = float(rng_fol.uniform(21.6, 24.5))
        fz = float(rng_fol.uniform(1.55, 2.98))
        fr = float(rng_fol.uniform(0.14, 0.28))
        rot_f = (
            float(rng_fol.uniform(0, 180)),
            float(rng_fol.uniform(0, 180)),
            float(rng_fol.uniform(0, 180)),
        )
        m_leaf = _xform_mesh(
            make_icosphere(radius=fr, subdivisions=1),
            location=(fx, fy, fz),
            rotation_deg=rot_f,
            scale=(1.25, 0.90, 0.72),
        )
        if fi % 2 == 0:
            fol_dark.append(m_leaf)
        else:
            fol_light.append(m_leaf)
    _add_obj(
        scene,
        "bg_foliage_dark",
        "BG_BendFoliageDark",
        concat_meshes(fol_dark),
        "mat_foliage",
        "distant",
    )
    _add_obj(
        scene,
        "bg_foliage_light",
        "BG_BendFoliageLight",
        concat_meshes(fol_light),
        "mat_foliage_light",
        "distant",
    )

    # Far end buildings & ochre landmark + flagpole
    _add_obj(
        scene,
        "bg_end_white",
        "BG_EndWhiteBuilding",
        _uv_box((7.5, 4.0, 5.8), (0.1, 35.5, 2.9)),
        "mat_bg_white",
        "distant",
    )
    _add_obj(
        scene,
        "bg_end_garage",
        "BG_EndBrownGarage",
        _uv_box((2.4, 0.12, 2.3), (0.35, 33.48, 1.15)),
        "mat_wood_pergola",
        "distant",
    )
    for wx in (-1.35, 0.95):
        _add_obj(
            scene,
            f"bg_end_win_{wx:.1f}",
            "BG_EndWindow",
            _uv_box((0.85, 0.10, 0.95), (wx, 33.48, 4.15)),
            "mat_window_dark",
            "distant",
        )

    _add_obj(
        scene,
        "bg_ochre_tower",
        "BG_OchreLandmark",
        _uv_box((5.2, 4.5, 9.8), (-2.1, 39.2, 4.9)),
        "mat_ochre_wall",
        "distant",
    )
    _add_obj(
        scene,
        "bg_ochre_trim",
        "BG_OchreWhiteTrim",
        _uv_box((0.45, 0.25, 9.85), (0.35, 36.9, 4.92)),
        "mat_bg_white",
        "distant",
    )
    _add_obj(
        scene,
        "bg_ochre_parapet",
        "BG_OchreParapet",
        _uv_box((5.3, 0.30, 0.28), (-2.1, 36.9, 9.82)),
        "mat_bg_white",
        "distant",
    )
    _add_obj(
        scene,
        "bg_flagpole",
        "BG_Flagpole",
        _xform_mesh(
            make_cylinder(radius=0.04, height=3.8, segments=8, origin="bottom"),
            location=(-2.4, 37.5, 9.8),
        ),
        "mat_pvc_pipe",
        "distant",
    )
    _add_obj(
        scene,
        "bg_flag",
        "BG_Flag",
        _uv_box((0.95, 0.04, 0.55), (-1.92, 37.5, 13.15), rotation_deg=(4.0, 8.0, -12.0)),
        "mat_flag_red",
        "distant",
    )

    # Thin overhead utility cables
    for wi, (wy, wz) in enumerate(((12.0, 6.2), (16.5, 6.6), (22.0, 7.1))):
        w_pts = [(-2.6, wy, wz), (0.0, wy + 0.3, wz - 0.55), (3.2, wy + 0.6, wz + 0.1)]
        _add_obj(
            scene,
            f"overhead_wire_{wi}",
            f"OverheadWire_{wi}",
            sweep_curve(w_pts, radius=0.006, radial_segments=6, closed=False),
            "mat_dark_cable",
            "distant",
        )

    # Camera: shifted slightly left (-0.50) to match photo perspective
    # and show Car1 rear-left 3/4 view!
    scene.objects["Camera"] = SceneObject(
        id="Camera",
        name="AlleyMainCamera",
        kind="camera",
        transform=Transform(location=(-0.50, -3.20, 1.54)),
        camera=CameraSpec(
            projection="perspective",
            fov_deg=47.0,
            near=0.05,
            far=160.0,
            target=(-0.18, 20.0, 0.36),
            up=(0.0, 0.0, 1.0),
        ),
    )
    scene.active_camera_id = "Camera"

    # Soft diffused overcast sky fill
    scene.objects["OvercastKey"] = SceneObject(
        id="OvercastKey",
        name="OvercastSkyFill",
        kind="light",
        light=LightSpec(
            kind="sun",
            color=(0.94, 0.96, 1.0),
            intensity=0.52,
            direction=(0.14, 0.22, -0.96),
            size=3.5,
            cast_shadows=True,
        ),
    )
    scene.objects["AlleyBounce"] = SceneObject(
        id="AlleyBounce",
        name="AlleySkyBounce",
        kind="light",
        transform=Transform(location=(-0.2, 4.0, 5.5)),
        light=LightSpec(
            kind="area",
            color=(0.92, 0.93, 0.96),
            intensity=2.2,
            size=5.0,
            cast_shadows=False,
        ),
    )

    return scene, tex_paths


def main() -> None:
    parser = argparse.ArgumentParser(description="Build & render the Gulf Alley 3D scene.")
    parser.add_argument("--draft", action="store_true", help="Render fast draft previews.")
    parser.add_argument("--samples", type=int, default=6, help="Path tracer samples per pixel.")
    parser.add_argument("--max-bounces", type=int, default=2, help="Path tracer max bounces.")
    parser.add_argument("--width", type=int, default=960)
    parser.add_argument("--height", type=int, default=720)
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    scene, _ = build_gulf_alley_scene()

    proj_path = OUT_DIR / "gulf_alley.frf.json"
    save_project(scene, proj_path)
    print(
        f"Saved scene project: {proj_path} "
        f"({len(scene.objects)} objects, {len(scene.materials)} materials)",
        flush=True,
    )

    evaluated = evaluate_scene_meshes(scene)
    cam_obj = scene.objects[scene.active_camera_id]

    if args.draft:
        sw = SoftwareRenderer()
        sw_img = sw.render_image(
            scene,
            evaluated,
            cam_obj,
            RenderConfig(width=640, height=480, mode="pbr", show_grid=False, show_axes=False),
        )
        sw_path = OUT_DIR / "render_software_draft.png"
        sw_img.save(sw_path)
        print(f"Saved software draft: {sw_path}", flush=True)

        pt = PathTracerRenderer()
        rt_img = pt.render_image(
            scene,
            evaluated,
            cam_obj,
            RenderConfig(
                width=520,
                height=390,
                mode="pbr",
                show_grid=False,
                show_axes=False,
                samples=4,
                max_bounces=2,
            ),
        )
        rt_path = OUT_DIR / "render_raytrace_draft.png"
        rt_img.save(rt_path)
        print(f"Saved raytrace draft: {rt_path}", flush=True)
        return

    mv_bytes, _ = render_multiview_sheet(
        scene,
        tile_width=420,
        tile_height=320,
        mode="pbr",
        views=("front", "right", "top", "iso"),
        show_grid=False,
        show_labels=False,
        engine="software",
    )
    mv_path = OUT_DIR / "render_multiview.png"
    mv_path.write_bytes(mv_bytes)
    print(f"Saved multiview sheet: {mv_path}", flush=True)

    glb_path = OUT_DIR / "gulf_alley.glb"
    export_scene_file(scene, glb_path)
    print(f"Exported GLB: {glb_path}", flush=True)

    pt = PathTracerRenderer()
    rt_img = pt.render_image(
        scene,
        evaluated,
        cam_obj,
        RenderConfig(
            width=args.width,
            height=args.height,
            mode="pbr",
            show_grid=False,
            show_axes=False,
            samples=args.samples,
            max_bounces=args.max_bounces,
        ),
    )
    rt_path = OUT_DIR / "render_final_raytrace.png"
    rt_img.save(rt_path)
    print(f"Saved final raytrace: {rt_path}", flush=True)


if __name__ == "__main__":
    main()
