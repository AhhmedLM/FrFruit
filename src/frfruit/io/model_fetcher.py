"""Online CC0 3D model repository search, download, & one-call PBR scene import (Poly Haven)."""

from __future__ import annotations

import urllib.parse
from pathlib import Path
from typing import Any, Literal

import numpy as np

from frfruit.core.math3d import compute_aabb
from frfruit.core.scene import MaterialSpec, SceneDocument, SceneObject
from frfruit.core.types import Transform, Vec3
from frfruit.geometry.mesh import MeshData, concat_meshes
from frfruit.geometry.primitives import make_cube, make_cylinder, make_icosphere
from frfruit.geometry.spatial import compute_world_matrix
from frfruit.io.importers import import_mesh_file
from frfruit.textures.fetcher import _http_get_bytes, _http_get_json

ModelSource = Literal["all", "polyhaven", "procedural"]
DEFAULT_MODEL_CACHE = Path.home() / ".cache" / "frfruit" / "models"


def _build_procedural_preset_mesh(preset: str) -> tuple[MeshData, tuple[float, float, float]]:
    """Build a compact offline fallback mesh + base RGB tint for common prop presets."""
    key = preset.strip().lower()
    if key in ("chair", "armchair"):
        parts = [
            _shift_mesh(make_cube(0.50, origin="center"), (0.0, 0.0, 0.45), (1.0, 1.0, 0.12)),
            _shift_mesh(make_cube(0.50, origin="center"), (0.0, 0.22, 0.72), (1.0, 0.12, 1.0)),
        ]
        for lx in (-0.21, 0.21):
            for ly in (-0.21, 0.21):
                parts.append(
                    _shift_mesh(
                        make_cylinder(radius=0.025, height=0.44, segments=10),
                        (lx, ly, 0.22),
                    )
                )
        return concat_meshes(parts), (0.48, 0.34, 0.22)

    if key in ("table", "desk"):
        parts = [
            _shift_mesh(make_cube(1.0, origin="center"), (0.0, 0.0, 0.74), (1.2, 0.75, 0.06)),
        ]
        for lx in (-0.52, 0.52):
            for ly in (-0.30, 0.30):
                parts.append(
                    _shift_mesh(
                        make_cylinder(radius=0.03, height=0.72, segments=10),
                        (lx, ly, 0.36),
                    )
                )
        return concat_meshes(parts), (0.52, 0.38, 0.24)

    if key in ("street_lamp", "lamp"):
        parts = [
            _shift_mesh(make_cylinder(radius=0.07, height=3.8, segments=12), (0.0, 0.0, 1.9)),
            _shift_mesh(make_cube(1.0, origin="center"), (0.35, 0.0, 3.76), (0.75, 0.10, 0.08)),
            _shift_mesh(make_icosphere(radius=0.14, subdivisions=1), (0.68, 0.0, 3.68)),
        ]
        return concat_meshes(parts), (0.18, 0.20, 0.22)

    if key in ("tree", "bush", "foliage"):
        parts = [
            _shift_mesh(make_cylinder(radius=0.11, height=1.6, segments=10), (0.0, 0.0, 0.8)),
            _shift_mesh(make_icosphere(radius=0.75, subdivisions=2), (0.0, 0.0, 2.0)),
            _shift_mesh(make_icosphere(radius=0.52, subdivisions=2), (0.42, 0.20, 1.85)),
            _shift_mesh(make_icosphere(radius=0.48, subdivisions=2), (-0.35, -0.25, 1.75)),
        ]
        return concat_meshes(parts), (0.20, 0.38, 0.16)

    if key in ("barrel", "drum"):
        m = make_cylinder(radius=0.30, height=0.88, segments=18)
        return _shift_mesh(m, (0.0, 0.0, 0.44)), (0.28, 0.42, 0.58)

    # Default crate / box
    m = make_cube(0.8, origin="center")
    return _shift_mesh(m, (0.0, 0.0, 0.4)), (0.55, 0.42, 0.28)


def _shift_mesh(
    mesh: MeshData,
    offset: tuple[float, float, float],
    scale: tuple[float, float, float] = (1.0, 1.0, 1.0),
) -> MeshData:
    out = mesh.copy()
    if len(out.vertices) > 0:
        out.vertices = (
            out.vertices * np.asarray(scale, dtype=np.float64)[None, :]
            + np.asarray(offset, dtype=np.float64)[None, :]
        )
    return out


def search_online_models(
    query: str = "",
    source: ModelSource = "all",
    limit: int = 12,
    timeout: float = 15.0,
) -> list[dict[str, Any]]:
    """Search online CC0 3D model repositories (Poly Haven 3D Models + procedural presets)."""
    q = query.strip().lower()
    tokens = [tok for tok in q.split() if tok]
    results: list[dict[str, Any]] = []

    if source in ("all", "polyhaven"):
        try:
            assets = _http_get_json("https://api.polyhaven.com/assets?t=models", timeout=timeout)
            if isinstance(assets, dict):
                scored: list[tuple[int, int, dict[str, Any]]] = []
                for aid, meta in assets.items():
                    if not isinstance(meta, dict):
                        continue
                    title = str(meta.get("name", aid))
                    cats = [str(c) for c in meta.get("categories", [])]
                    tags = [str(t) for t in meta.get("tags", [])]
                    desc = str(meta.get("description", ""))
                    haystack = " ".join([aid, title, *cats, *tags, desc]).lower()
                    if not tokens or all(tok in haystack for tok in tokens):
                        name_low = f"{aid} {title}".lower()
                        exact = 1 if (q and q in name_low) else 0
                        dl_cnt = int(meta.get("download_count", 0) or 0)
                        dims = meta.get("dimensions")
                        scored.append(
                            (
                                exact,
                                dl_cnt,
                                {
                                    "asset_id": aid,
                                    "name": title,
                                    "source": "polyhaven",
                                    "categories": cats,
                                    "tags": tags[:8],
                                    "polycount": int(meta.get("polycount", 0) or 0),
                                    "dimensions_mm": dims if isinstance(dims, list) else None,
                                    "resolutions": ["1k", "2k", "4k"],
                                    "preview_url": meta.get("thumbnail_url")
                                    or f"https://cdn.polyhaven.com/asset_img/thumbs/{aid}.png?width=256",
                                },
                            )
                        )
                scored.sort(key=lambda item: (item[0], item[1]), reverse=True)
                results.extend(item[2] for item in scored[:limit])
        except Exception as exc:
            if source == "polyhaven":
                raise RuntimeError(f"Poly Haven model search failed: {exc}") from exc

    if source == "procedural" or (source == "all" and not results):
        for preset in ("chair", "table", "street_lamp", "tree", "barrel", "wooden_crate"):
            if not tokens or any(tok in preset for tok in tokens):
                results.append(
                    {
                        "asset_id": preset,
                        "name": f"Procedural {preset.replace('_', ' ').title()}",
                        "source": "procedural",
                        "categories": ["procedural", preset],
                        "tags": ["offline", "parametric"],
                        "polycount": 0,
                        "dimensions_mm": None,
                        "resolutions": ["1k"],
                        "preview_url": None,
                    }
                )

    return results[:limit]


def download_online_model(
    asset_id: str,
    source: Literal["polyhaven", "procedural"] = "polyhaven",
    resolution: str = "1k",
    target_dir: str | Path | None = None,
    fallback_to_procedural: bool = False,
    timeout: float = 60.0,
) -> dict[str, Any]:
    """Download a CC0 3D model (.gltf/.bin/textures) from Poly Haven into a local directory."""
    out_root = Path(target_dir) if target_dir else DEFAULT_MODEL_CACHE
    pack_dir = (out_root / f"{source}_{asset_id}_{resolution}").resolve()
    pack_dir.mkdir(parents=True, exist_ok=True)

    try:
        if source == "procedural":
            raise RuntimeError("Using procedural fallback generator")

        files_meta = _http_get_json(
            f"https://api.polyhaven.com/files/{urllib.parse.quote(asset_id)}",
            timeout=timeout,
        )
        if not isinstance(files_meta, dict) or "gltf" not in files_meta:
            raise RuntimeError(f"Model {asset_id!r} has no glTF package on Poly Haven.")

        gltf_bucket = files_meta["gltf"]
        picked_entry: dict[str, Any] | None = None
        picked_res = resolution
        picked_ext = "gltf"
        for res in (resolution, "1k", "2k", "4k"):
            res_dict = gltf_bucket.get(res)
            if not isinstance(res_dict, dict):
                continue
            for fmt in ("gltf", "glb"):
                candidate = res_dict.get(fmt)
                if isinstance(candidate, dict) and candidate.get("url"):
                    picked_entry = candidate
                    picked_res = res
                    picked_ext = fmt
                    break
            if picked_entry is not None:
                break

        if picked_entry is None:
            raise RuntimeError(f"No downloadable glTF/GLB entry found for {asset_id!r}.")

        main_url = str(picked_entry["url"])
        main_dest = pack_dir / f"{asset_id}_{picked_res}.{picked_ext}"
        if not main_dest.exists():
            main_dest.write_bytes(_http_get_bytes(main_url, timeout=timeout))

        includes = picked_entry.get("include") or {}
        downloaded_files = [str(main_dest)]
        if isinstance(includes, dict):
            for rel_path, meta in includes.items():
                if not isinstance(meta, dict) or not meta.get("url"):
                    continue
                # Normalize relative include path safely inside pack_dir
                safe_rel = Path(*[p for p in Path(rel_path).parts if p not in ("..", "", "/")])
                dest = (pack_dir / safe_rel).resolve()
                if not str(dest).startswith(str(pack_dir)):
                    continue
                dest.parent.mkdir(parents=True, exist_ok=True)
                if not dest.exists():
                    dest.write_bytes(_http_get_bytes(str(meta["url"]), timeout=timeout))
                downloaded_files.append(str(dest))

        return {
            "status": "ok",
            "source": "polyhaven",
            "asset_id": asset_id,
            "resolution": picked_res,
            "model_path": str(main_dest),
            "pack_dir": str(pack_dir),
            "files_count": len(downloaded_files),
        }
    except Exception as exc:
        if not fallback_to_procedural and source != "procedural":
            raise RuntimeError(f"Failed to download model {asset_id!r}: {exc}") from exc

        mesh, color = _build_procedural_preset_mesh(asset_id)
        from frfruit.core.math3d import compose_matrix

        zup_to_yup = compose_matrix(rotation_deg=(-90.0, 0.0, 0.0))[:3, :3]
        mesh_yup = mesh.copy()
        if len(mesh_yup.vertices) > 0:
            mesh_yup.vertices = mesh_yup.vertices @ zup_to_yup.T
        tm = mesh_yup.to_trimesh(process=False)
        dest_glb = pack_dir / f"{asset_id}.glb"
        dest_glb.write_bytes(tm.export(file_type="glb"))
        return {
            "status": "ok",
            "source": "procedural",
            "asset_id": asset_id,
            "resolution": resolution,
            "model_path": str(dest_glb),
            "pack_dir": str(pack_dir),
            "files_count": 1,
            "fallback_color": color,
        }


def import_online_model(
    scene: SceneDocument,
    asset_id: str,
    *,
    source: Literal["polyhaven", "procedural"] = "polyhaven",
    resolution: str = "1k",
    target_dir: str | Path | None = None,
    name_prefix: str | None = None,
    collection_id: str = "default",
    location: Vec3 = (0.0, 0.0, 0.0),
    rotation_deg: Vec3 = (0.0, 0.0, 0.0),
    scale: float = 1.0,
    target_height: float | None = None,
    place_on_ground: bool = True,
    fallback_to_procedural: bool = False,
    timeout: float = 60.0,
) -> dict[str, Any]:
    """Download an online 3D model, extract PBR maps, and place cleanly under a parent group."""
    dl = download_online_model(
        asset_id=asset_id,
        source=source,
        resolution=resolution,
        target_dir=target_dir,
        fallback_to_procedural=fallback_to_procedural,
        timeout=timeout,
    )
    prefix = name_prefix or asset_id
    imported = import_mesh_file(
        scene,
        dl["model_path"],
        name_prefix=prefix,
        collection_id=collection_id,
        convert_yup=True,
    )
    if not imported:
        raise RuntimeError(f"No mesh geometry found in downloaded model {asset_id!r}.")

    # Apply fallback material tint if procedural preset had no embedded material
    if dl.get("source") == "procedural" and dl.get("fallback_color"):
        c = dl["fallback_color"]
        for obj in imported:
            if obj.material_id is None:
                mid = scene.next_id(f"mat_{prefix}")
                scene.materials[mid] = MaterialSpec(
                    id=mid,
                    name=f"{prefix} Material",
                    base_color=(float(c[0]), float(c[1]), float(c[2]), 1.0),
                    roughness=0.68,
                )
                obj.material_id = mid

    # Compute combined bounds in Z-up world space to center XY & optionally ground Z
    all_pts: list[np.ndarray] = []
    for obj in imported:
        if obj.mesh is None:
            continue
        md = MeshData.from_payload(obj.mesh)
        if md.is_empty:
            continue
        m_w = compute_world_matrix(scene, obj.id)
        pts = md.vertices @ m_w[:3, :3].T + m_w[:3, 3][None, :]
        all_pts.append(pts)

    combined_pts = np.vstack(all_pts) if all_pts else np.zeros((1, 3), dtype=np.float64)
    bbox = compute_aabb(combined_pts)
    cx, cy, cz = bbox.center
    anchor_z = bbox.min_corner[2] if place_on_ground else cz

    eff_scale = float(scale)
    if target_height is not None and bbox.size[2] > 1e-5:
        eff_scale = float(target_height) / float(bbox.size[2])

    # Create a single parent Empty so the imported asset transforms as one cohesive unit
    root_name = scene.unique_name(prefix)
    root_id = scene.next_id(root_name)
    root_obj = SceneObject(
        id=root_id,
        name=root_name,
        kind="empty",
        collection_id=collection_id if collection_id in scene.collections else "default",
        transform=Transform(
            location=location,
            rotation_deg=rotation_deg,
            scale=(eff_scale, eff_scale, eff_scale),
        ),
    )
    scene.objects[root_id] = root_obj

    mat_ids: list[str] = []
    for obj in imported:
        lx, ly, lz = obj.transform.location
        obj.transform.location = (lx - cx, ly - cy, lz - anchor_z)
        obj.parent_id = root_id
        if obj.material_id and obj.material_id not in mat_ids:
            mat_ids.append(obj.material_id)

    return {
        "status": "ok",
        "source": dl["source"],
        "asset_id": asset_id,
        "root_id": root_id,
        "object_ids": [o.id for o in imported],
        "material_ids": mat_ids,
        "model_path": dl["model_path"],
        "original_size": list(bbox.size),
        "applied_scale": eff_scale,
    }
