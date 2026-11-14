"""Automatic PBR texture search and multi-channel download (Poly Haven, ambientCG, & local bake)."""

from __future__ import annotations

import io
import json
import math
import urllib.parse
import urllib.request
import zipfile
from pathlib import Path
from typing import Any, Literal

import numpy as np
from PIL import Image

from frfruit.core.scene import MaterialSpec

TextureSource = Literal["all", "polyhaven", "ambientcg", "procedural"]
SUPPORTED_SOURCES = ("polyhaven", "ambientcg", "procedural")
DEFAULT_TEXTURE_CACHE = Path.home() / ".cache" / "frfruit" / "textures"

_USER_AGENT = "FrFruit-3D-Engine/1.0 (+https://github.com/AhhmedLM/FrFruit)"

# Maps Poly Haven file keys & ambientCG filename suffixes into unified FrFruit PBR slots
_CHANNEL_ALIASES: dict[str, tuple[str, ...]] = {
    "albedo": ("diffuse", "col", "albedo", "color", "basecolor", "diff"),
    "normal": ("nor_gl", "normalgl", "normal", "nor_dx", "normaldx", "bump"),
    "roughness": ("rough", "roughness", "rgh"),
    "metallic": ("metal", "metallic", "metalness"),
    "ao": ("ao", "ambientocclusion", "occ"),
    "height": ("disp", "displacement", "height"),
}


def _http_get_json(url: str, timeout: float = 20.0) -> Any:
    req = urllib.request.Request(
        url, headers={"User-Agent": _USER_AGENT, "Accept": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        raw = resp.read().decode("utf-8", errors="replace")
    return json.loads(raw)


def _http_get_bytes(url: str, timeout: float = 45.0) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": _USER_AGENT})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read()


def _classify_map_key(name: str) -> str | None:
    low = name.lower()
    for channel, tokens in _CHANNEL_ALIASES.items():
        if any(tok in low for tok in tokens):
            return channel
    return None


def search_online_textures(
    query: str = "",
    source: TextureSource = "all",
    limit: int = 12,
    timeout: float = 15.0,
) -> list[dict[str, Any]]:
    """Search online PBR repositories (Poly Haven & ambientCG) and return normalized entries."""
    q = query.strip().lower()
    results: list[dict[str, Any]] = []

    if source in ("all", "polyhaven"):
        try:
            assets = _http_get_json("https://api.polyhaven.com/assets?t=textures", timeout=timeout)
            if isinstance(assets, dict):
                matches: list[dict[str, Any]] = []
                for aid, meta in assets.items():
                    if not isinstance(meta, dict):
                        continue
                    title = str(meta.get("name", aid))
                    cats = [str(c) for c in meta.get("categories", [])]
                    tags = [str(t) for t in meta.get("tags", [])]
                    haystack = " ".join([aid, title, *cats, *tags]).lower()
                    if not q or all(tok in haystack for tok in q.split()):
                        matches.append(
                            {
                                "asset_id": aid,
                                "name": title,
                                "source": "polyhaven",
                                "categories": cats,
                                "tags": tags[:8],
                                "resolutions": ["1k", "2k", "4k"],
                                "maps_available": [
                                    "albedo",
                                    "normal",
                                    "roughness",
                                    "ao",
                                    "height",
                                ],
                                "preview_url": f"https://cdn.polyhaven.com/asset_img/thumbs/{aid}.png?width=256",
                            }
                        )
                results.extend(matches[:limit])
        except Exception as exc:
            if source == "polyhaven":
                raise RuntimeError(f"Poly Haven search failed: {exc}") from exc

    if source in ("all", "ambientcg"):
        try:
            encoded = urllib.parse.quote(query.strip())
            url = (
                f"https://ambientcg.com/api/v2/full_json?"
                f"q={encoded}&type=Material&limit={limit}&include=tagData,previewData"
            )
            data = _http_get_json(url, timeout=timeout)
            for item in data.get("foundAssets", [])[:limit]:
                aid = str(item.get("assetId", ""))
                if not aid:
                    continue
                cats = [str(item.get("displayCategory", "material"))]
                tags = [str(t) for t in item.get("tags", [])]
                previews = item.get("previewImage", {})
                preview_url = (
                    previews.get("256-PNG")
                    or previews.get("256-JPG")
                    or f"https://ambientcg.com/get?file={aid}_PREVIEW.jpg"
                )
                results.append(
                    {
                        "asset_id": aid,
                        "name": aid,
                        "source": "ambientcg",
                        "categories": cats,
                        "tags": tags[:8],
                        "resolutions": ["1k", "2k", "4k"],
                        "maps_available": [
                            "albedo",
                            "normal",
                            "roughness",
                            "metallic",
                            "ao",
                            "height",
                        ],
                        "preview_url": preview_url,
                    }
                )
        except Exception as exc:
            if source == "ambientcg":
                raise RuntimeError(f"ambientCG search failed: {exc}") from exc

    if source == "procedural" or (source == "all" and not results):
        presets = [
            "stone",
            "brick",
            "marble",
            "wood",
            "metal",
            "fabric",
            "tiles",
            "concrete",
            "gold_leaf",
        ]
        for style in presets:
            if not q or q in style:
                results.append(
                    {
                        "asset_id": style,
                        "name": f"Procedural {style.replace('_', ' ').title()}",
                        "source": "procedural",
                        "categories": ["procedural", style],
                        "tags": ["pbr", "tileable", "offline"],
                        "resolutions": ["256", "512", "1024"],
                        "maps_available": [
                            "albedo",
                            "normal",
                            "roughness",
                            "metallic",
                            "ao",
                            "height",
                        ],
                        "preview_url": None,
                    }
                )

    return results[:limit]


def _pick_polyhaven_file(
    map_dict: dict[str, Any], resolution: str = "1k"
) -> tuple[str, str] | None:
    res_order = [resolution, "1k", "2k", "4k"]
    for res in res_order:
        bucket = map_dict.get(res)
        if not isinstance(bucket, dict):
            continue
        for fmt in ("png", "jpg"):
            entry = bucket.get(fmt)
            if isinstance(entry, dict) and entry.get("url"):
                return str(entry["url"]), fmt
    return None


def download_pbr_texture_pack(
    asset_id: str,
    source: Literal["polyhaven", "ambientcg", "procedural"] = "polyhaven",
    resolution: str = "1k",
    target_dir: str | Path | None = None,
    map_types: list[str] | None = None,
    fallback_to_procedural: bool = False,
    timeout: float = 45.0,
) -> dict[str, Any]:
    """Download PBR map pack (albedo, normal, roughness, metallic, ao, height) cleanly to disk."""
    out_root = Path(target_dir) if target_dir else DEFAULT_TEXTURE_CACHE
    pack_dir = (out_root / f"{source}_{asset_id}_{resolution}").resolve()
    pack_dir.mkdir(parents=True, exist_ok=True)

    wanted = set(map_types or ("albedo", "normal", "roughness", "metallic", "ao", "height"))
    maps_out: dict[str, str] = {}

    try:
        if source == "procedural":
            return generate_procedural_pbr_maps(
                style=asset_id,
                out_dir=pack_dir,
                size=512 if resolution == "1k" else 1024,
            )

        if source == "polyhaven":
            files_meta = _http_get_json(
                f"https://api.polyhaven.com/files/{urllib.parse.quote(asset_id)}",
                timeout=timeout,
            )
            if not isinstance(files_meta, dict):
                raise RuntimeError(f"Asset {asset_id!r} not found on Poly Haven.")

            for raw_key, map_data in files_meta.items():
                if not isinstance(map_data, dict):
                    continue
                slot = _classify_map_key(raw_key)
                if slot is None or slot not in wanted:
                    continue
                # Prefer OpenGL normal map over DirectX if both exist
                if slot == "normal" and "nor_dx" in raw_key.lower() and "nor_gl" in files_meta:
                    continue
                picked = _pick_polyhaven_file(map_data, resolution=resolution)
                if picked is None:
                    continue
                url, ext = picked
                dest = pack_dir / f"{asset_id}_{slot}.{ext}"
                if not dest.exists():
                    dest.write_bytes(_http_get_bytes(url, timeout=timeout))
                maps_out[slot] = str(dest)

            # Extract ARM packed channels (AO/Roughness/Metallic) if individual maps omitted
            if "arm" in files_meta and not ({"ao", "roughness", "metallic"} <= set(maps_out)):
                picked = _pick_polyhaven_file(files_meta["arm"], resolution=resolution)
                if picked is not None:
                    arm_bytes = _http_get_bytes(picked[0], timeout=timeout)
                    arm_img = np.asarray(Image.open(io.BytesIO(arm_bytes)).convert("RGB"))
                    for ch_idx, slot in ((0, "ao"), (1, "roughness"), (2, "metallic")):
                        if slot in wanted and slot not in maps_out:
                            ch_path = pack_dir / f"{asset_id}_{slot}.png"
                            Image.fromarray(arm_img[:, :, ch_idx], mode="L").save(ch_path)
                            maps_out[slot] = str(ch_path)

        elif source == "ambientcg":
            res_tag = resolution.upper()
            url = (
                f"https://ambientcg.com/api/v2/full_json?"
                f"id={urllib.parse.quote(asset_id)}&include=downloadData"
            )
            data = _http_get_json(url, timeout=timeout)
            found = data.get("foundAssets", [])
            if not found:
                raise RuntimeError(f"Asset {asset_id!r} not found on ambientCG.")
            dl_cats = found[0].get("downloadFolders", {}).get("default", {})
            zip_list = (
                dl_cats.get("downloadFiletypeCategories", {}).get("zip", {}).get("downloads", [])
            )
            chosen = None
            for cand in zip_list:
                attr = str(cand.get("attribute", "")).upper()
                if attr.startswith(res_tag) and ("JPG" in attr or "PNG" in attr):
                    chosen = cand
                    break
            if chosen is None and zip_list:
                chosen = zip_list[0]
            if chosen is None or not chosen.get("downloadLink"):
                raise RuntimeError(f"No downloadable ZIP archive for ambientCG {asset_id!r}.")

            zip_bytes = _http_get_bytes(str(chosen["downloadLink"]), timeout=timeout)
            with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
                for member in zf.namelist():
                    low = member.lower()
                    if not (low.endswith(".png") or low.endswith(".jpg") or low.endswith(".jpeg")):
                        continue
                    slot = _classify_map_key(Path(member).stem)
                    if slot is None or slot not in wanted:
                        continue
                    if slot == "normal" and "normaldx" in low:
                        continue
                    ext = Path(member).suffix.lower() or ".png"
                    dest = pack_dir / f"{asset_id}_{slot}{ext}"
                    dest.write_bytes(zf.read(member))
                    maps_out[slot] = str(dest)

        if not maps_out:
            raise RuntimeError(f"No matching PBR texture maps retrieved for {source}:{asset_id}.")

    except Exception as exc:
        if not fallback_to_procedural:
            raise RuntimeError(f"Failed downloading texture {source}:{asset_id}: {exc}") from exc
        gen = generate_procedural_pbr_maps(
            style=asset_id,
            out_dir=pack_dir,
            size=512,
        )
        gen["fallback_reason"] = str(exc)
        gen["requested_source"] = source
        return gen

    manifest = {
        "asset_id": asset_id,
        "source": source,
        "resolution": resolution,
        "pack_dir": str(pack_dir),
        "maps": maps_out,
    }
    (pack_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def _height_to_normal(height: np.ndarray, strength: float = 2.2) -> np.ndarray:
    """Convert scalar heightfield (H, W) in [0, 1] into tangent-space RGB normal map."""
    gx = np.roll(height, -1, axis=1) - np.roll(height, 1, axis=1)
    gy = np.roll(height, -1, axis=0) - np.roll(height, 1, axis=0)
    nx = -gx * strength
    ny = gy * strength
    nz = np.ones_like(height)
    norm = np.sqrt(nx * nx + ny * ny + nz * nz)
    n_rgb = np.stack([nx / norm, ny / norm, nz / norm], axis=-1) * 0.5 + 0.5
    return np.clip(n_rgb * 255.0, 0, 255).astype(np.uint8)


def generate_procedural_pbr_maps(
    style: str = "stone",
    out_dir: str | Path | None = None,
    size: int = 512,
    seed: int = 42,
) -> dict[str, Any]:
    """Synthesize seamless multi-channel PBR maps locally (albedo/normal/rough/metal/ao/height)."""
    out_p = Path(out_dir) if out_dir else (DEFAULT_TEXTURE_CACHE / f"procedural_{style}_{size}")
    out_p.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)

    u = np.linspace(0.0, 1.0, size, endpoint=False)[None, :]
    v = np.linspace(0.0, 1.0, size, endpoint=False)[:, None]
    tau = 2.0 * math.pi

    # Periodic multi-octave smooth noise (seamless on all boundaries)
    base_noise = np.zeros((size, size), dtype=np.float64)
    for freq, amp in ((2, 0.45), (4, 0.25), (8, 0.16), (16, 0.09), (32, 0.05)):
        px = float(rng.uniform(0, tau))
        py = float(rng.uniform(0, tau))
        base_noise += amp * (0.5 + 0.5 * np.sin(u * freq * tau + px) * np.cos(v * freq * tau + py))

    low = style.lower()
    metallic_arr = np.zeros((size, size), dtype=np.float64)

    if "brick" in low or "tile" in low:
        rows, cols = 8, 4
        gv = np.floor(v * rows)
        su = (u * cols + 0.5 * (gv % 2.0)) % 1.0
        sv = (v * rows) % 1.0
        mortar = (su < 0.06) | (su > 0.94) | (sv < 0.10) | (sv > 0.90)
        height = np.where(mortar, 0.15 + 0.1 * base_noise, 0.65 + 0.35 * base_noise)
        if "tile" in low:
            c1 = np.array([0.07, 0.34, 0.58])
            c2 = np.array([0.12, 0.56, 0.62])
            albedo = c1 * (1.0 - base_noise[..., None]) + c2 * base_noise[..., None]
            albedo[mortar] = np.array([0.78, 0.74, 0.66])
            roughness = np.where(mortar, 0.85, 0.16 + 0.12 * base_noise)
        else:
            c1 = np.array([0.62, 0.26, 0.18])
            c2 = np.array([0.46, 0.18, 0.13])
            albedo = c1 * (1.0 - base_noise[..., None]) + c2 * base_noise[..., None]
            albedo[mortar] = np.array([0.72, 0.69, 0.63])
            roughness = np.where(mortar, 0.90, 0.68 + 0.15 * base_noise)
    elif "wood" in low:
        grain = 0.5 + 0.5 * np.sin((u * 10.0 + 0.35 * base_noise) * tau)
        height = 0.45 * grain + 0.55 * base_noise
        c1 = np.array([0.52, 0.31, 0.16])
        c2 = np.array([0.30, 0.17, 0.08])
        albedo = c1 * grain[..., None] + c2 * (1.0 - grain[..., None])
        roughness = 0.42 + 0.28 * (1.0 - grain)
    elif "marble" in low:
        vein = np.abs(np.sin((u * 3.0 + v * 2.0 + 1.8 * base_noise) * tau)) ** 0.35
        height = 0.75 + 0.25 * vein
        c1 = np.array([0.84, 0.80, 0.72])
        c2 = np.array([0.54, 0.47, 0.40])
        albedo = c2 * (1.0 - vein[..., None]) + c1 * vein[..., None]
        roughness = 0.16 + 0.18 * (1.0 - vein)
    elif "metal" in low or "gold" in low:
        brush = np.tile(np.mean(base_noise, axis=0, keepdims=True), (size, 1))
        height = 0.5 * brush + 0.5 * base_noise
        tint = np.array([0.94, 0.75, 0.26]) if "gold" in low else np.array([0.76, 0.78, 0.82])
        albedo = tint[None, None, :] * (0.88 + 0.12 * height[..., None])
        metallic_arr[:] = 1.0
        roughness = 0.18 + 0.16 * (1.0 - brush)
    elif "fabric" in low or "cloth" in low:
        weave_u = 0.5 + 0.5 * np.sin(u * 48.0 * tau)
        weave_v = 0.5 + 0.5 * np.sin(v * 48.0 * tau)
        weave = 0.5 * (weave_u + weave_v)
        height = 0.65 * weave + 0.35 * base_noise
        c1 = np.array([0.56, 0.16, 0.16])
        c2 = np.array([0.38, 0.11, 0.11])
        albedo = c1 * weave[..., None] + c2 * (1.0 - weave[..., None])
        roughness = 0.78 + 0.18 * (1.0 - weave)
    else:  # stone / concrete / default
        height = base_noise
        c1 = np.array([0.74, 0.66, 0.54])
        c2 = np.array([0.52, 0.46, 0.37])
        albedo = c1 * height[..., None] + c2 * (1.0 - height[..., None])
        roughness = 0.58 + 0.30 * (1.0 - height)

    ao = np.clip(0.35 + 0.65 * height, 0.0, 1.0)
    normal_u8 = _height_to_normal(height, strength=2.4)

    paths: dict[str, str] = {}
    Image.fromarray(np.clip(albedo * 255.0, 0, 255).astype(np.uint8), mode="RGB").save(
        p_alb := out_p / f"{style}_albedo.png"
    )
    Image.fromarray(normal_u8, mode="RGB").save(p_nrm := out_p / f"{style}_normal.png")
    Image.fromarray(np.clip(roughness * 255.0, 0, 255).astype(np.uint8), mode="L").save(
        p_rgh := out_p / f"{style}_roughness.png"
    )
    Image.fromarray(np.clip(metallic_arr * 255.0, 0, 255).astype(np.uint8), mode="L").save(
        p_mtl := out_p / f"{style}_metallic.png"
    )
    Image.fromarray(np.clip(ao * 255.0, 0, 255).astype(np.uint8), mode="L").save(
        p_ao := out_p / f"{style}_ao.png"
    )
    Image.fromarray(np.clip(height * 255.0, 0, 255).astype(np.uint8), mode="L").save(
        p_hgt := out_p / f"{style}_height.png"
    )

    paths["albedo"] = str(p_alb)
    paths["normal"] = str(p_nrm)
    paths["roughness"] = str(p_rgh)
    paths["metallic"] = str(p_mtl)
    paths["ao"] = str(p_ao)
    paths["height"] = str(p_hgt)

    manifest = {
        "asset_id": style,
        "source": "procedural",
        "resolution": f"{size}px",
        "pack_dir": str(out_p),
        "maps": paths,
    }
    (out_p / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def build_material_from_maps(
    material_id: str,
    name: str,
    maps: dict[str, str],
    texture_source: str | None = None,
    uv_scale: tuple[float, float] = (1.0, 1.0),
    base_color: tuple[float, float, float, float] = (1.0, 1.0, 1.0, 1.0),
    metallic: float = 0.0,
    roughness: float = 0.5,
    alpha: float = 1.0,
) -> MaterialSpec:
    """Create a PBR MaterialSpec pre-bound to downloaded albedo/normal/roughness/metal/AO maps."""
    return MaterialSpec(
        id=material_id,
        name=name,
        base_color=base_color,
        metallic=1.0 if "metal" in name.lower() or "gold" in name.lower() else metallic,
        roughness=roughness,
        alpha=alpha,
        albedo_map=maps.get("albedo"),
        normal_map=maps.get("normal"),
        roughness_map=maps.get("roughness"),
        metallic_map=maps.get("metallic"),
        ao_map=maps.get("ao"),
        height_map=maps.get("height"),
        uv_scale=uv_scale,
        texture_source=texture_source,
    )
