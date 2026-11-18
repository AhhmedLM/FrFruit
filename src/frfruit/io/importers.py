"""Mesh/scene importers for GLB, GLTF, OBJ, STL, and PLY with full PBR texture extraction."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import numpy as np
import trimesh
from PIL import Image

from frfruit.core.math3d import compose_matrix, decompose_matrix
from frfruit.core.scene import MaterialSpec, SceneDocument, SceneObject
from frfruit.core.types import Transform
from frfruit.geometry.mesh import MeshData

DEFAULT_IMPORT_TEX_DIR = Path.home() / ".cache" / "frfruit" / "imported_textures"


def _save_pil_texture(img: Any, dest: Path) -> str | None:
    if img is None:
        return None
    try:
        if isinstance(img, np.ndarray):
            pil = Image.fromarray(img)
        elif isinstance(img, Image.Image):
            pil = img
        else:
            return None
        dest.parent.mkdir(parents=True, exist_ok=True)
        if pil.mode not in ("RGB", "RGBA", "L"):
            pil = pil.convert("RGB")
        pil.save(dest, format="PNG")
        return str(dest)
    except Exception:
        return None


def _extract_pbr_material(
    scene: SceneDocument,
    visual: Any,
    obj_name: str,
    tex_dir: Path,
    mat_cache: dict[int, str],
) -> str | None:
    pbr = getattr(visual, "material", None)
    if pbr is None:
        return None

    cache_key = id(pbr)
    if cache_key in mat_cache:
        return mat_cache[cache_key]

    mat_name_raw = str(getattr(pbr, "name", "") or f"{obj_name}_mat")
    mat_id = scene.next_id(f"mat_{mat_name_raw}")
    safe_stem = "".join(ch if ch.isalnum() or ch in ("_", "-") else "_" for ch in mat_id)[:48]

    base_color = (0.78, 0.78, 0.80, 1.0)
    bcf = getattr(pbr, "baseColorFactor", None)
    if bcf is not None and len(bcf) >= 3:
        vals = [float(x) / 255.0 if max(bcf) > 1.0 else float(x) for x in bcf[:4]]
        base_color = (
            max(0.0, min(1.0, vals[0])),
            max(0.0, min(1.0, vals[1])),
            max(0.0, min(1.0, vals[2])),
            max(0.0, min(1.0, vals[3] if len(vals) == 4 else 1.0)),
        )
    elif getattr(pbr, "diffuse", None) is not None:
        diff = pbr.diffuse
        vals = [float(x) / 255.0 if max(diff) > 1.0 else float(x) for x in diff[:4]]
        base_color = (
            max(0.0, min(1.0, vals[0])),
            max(0.0, min(1.0, vals[1])),
            max(0.0, min(1.0, vals[2])),
            max(0.0, min(1.0, vals[3] if len(vals) == 4 else 1.0)),
        )

    mf = getattr(pbr, "metallicFactor", None)
    rf = getattr(pbr, "roughnessFactor", None)
    has_mr_tex = isinstance(getattr(pbr, "metallicRoughnessTexture", None), Image.Image)
    metallic = float(mf) if mf is not None else 0.0
    roughness = float(rf) if rf is not None else (1.0 if has_mr_tex else 0.5)
    alpha_mode = str(getattr(pbr, "alphaMode", "OPAQUE") or "OPAQUE").upper()
    alpha = base_color[3] if alpha_mode in ("BLEND", "MASK") else 1.0

    emi_col = (0.0, 0.0, 0.0, 1.0)
    emi_strength = 0.0
    ef = getattr(pbr, "emissiveFactor", None)
    if ef is not None and len(ef) >= 3:
        ev = [max(0.0, float(x)) for x in ef[:3]]
        peak = max(ev)
        if peak > 1e-4:
            emi_strength = peak
            emi_col = (min(1.0, ev[0] / peak), min(1.0, ev[1] / peak), min(1.0, ev[2] / peak), 1.0)

    albedo_tex = getattr(pbr, "baseColorTexture", None) or getattr(pbr, "image", None)
    normal_tex = getattr(pbr, "normalTexture", None)
    mr_tex = getattr(pbr, "metallicRoughnessTexture", None)
    occ_tex = getattr(pbr, "occlusionTexture", None)

    albedo_map = _save_pil_texture(albedo_tex, tex_dir / f"{safe_stem}_albedo.png")
    normal_map = _save_pil_texture(normal_tex, tex_dir / f"{safe_stem}_normal.png")
    ao_map = _save_pil_texture(occ_tex, tex_dir / f"{safe_stem}_ao.png")

    roughness_map: str | None = None
    metallic_map: str | None = None
    if isinstance(mr_tex, Image.Image):
        try:
            mr_rgb = np.asarray(mr_tex.convert("RGB"))
            g_ch = mr_rgb[:, :, 1]
            b_ch = mr_rgb[:, :, 2]
            r_dest = tex_dir / f"{safe_stem}_roughness.png"
            r_dest.parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(g_ch, mode="L").save(r_dest, format="PNG")
            roughness_map = str(r_dest)
            # In glTF metallicRoughnessTexture, G=roughness and B=metallic
            if rf is None or roughness == 0.0:
                roughness = 1.0
            if int(b_ch.max()) > 4:
                m_dest = tex_dir / f"{safe_stem}_metallic.png"
                Image.fromarray(b_ch, mode="L").save(m_dest, format="PNG")
                metallic_map = str(m_dest)
                if mf is None or metallic == 0.0:
                    metallic = 1.0
        except Exception:
            pass

    scene.materials[mat_id] = MaterialSpec(
        id=mat_id,
        name=mat_name_raw,
        base_color=base_color,
        metallic=max(0.0, min(1.0, metallic)),
        roughness=max(0.0, min(1.0, roughness)),
        alpha=max(0.0, min(1.0, alpha)),
        emission_color=emi_col,
        emission_strength=emi_strength,
        albedo_map=albedo_map,
        normal_map=normal_map,
        roughness_map=roughness_map,
        metallic_map=metallic_map,
        ao_map=ao_map,
        normal_strength=0.85 if normal_map else 0.0,
    )
    mat_cache[cache_key] = mat_id
    return mat_id


def import_mesh_file(
    scene: SceneDocument,
    file_path: str | Path,
    name_prefix: str | None = None,
    collection_id: str = "default",
    *,
    convert_yup: bool | None = None,
    texture_dir: str | Path | None = None,
) -> list[SceneObject]:
    """Import a .glb/.gltf/.obj/.stl/.ply file into `scene` as SceneObjects with PBR maps."""
    p = Path(file_path)
    if not p.exists():
        raise FileNotFoundError(f"Mesh file not found: {p}")

    is_gltf = p.suffix.lower() in (".glb", ".gltf")
    if convert_yup is None:
        convert_yup = is_gltf

    path_hash = hashlib.sha1(str(p.resolve()).encode("utf-8")).hexdigest()[:8]
    tex_dir = (
        Path(texture_dir)
        if texture_dir is not None
        else DEFAULT_IMPORT_TEX_DIR / f"{p.stem}_{path_hash}"
    )

    loaded = trimesh.load(str(p), force=None)
    created: list[SceneObject] = []
    mat_cache: dict[int, str] = {}
    yup_to_zup = compose_matrix(rotation_deg=(90.0, 0.0, 0.0)) if convert_yup else np.eye(4)

    if isinstance(loaded, trimesh.Trimesh):
        md = MeshData.from_trimesh(loaded)
        if convert_yup and len(md.vertices) > 0:
            md.vertices = md.vertices @ yup_to_zup[:3, :3].T
        base_name = scene.unique_name(name_prefix or p.stem)
        oid = scene.next_id(base_name)
        mat_id = _extract_pbr_material(scene, loaded.visual, base_name, tex_dir, mat_cache)
        obj = SceneObject(
            id=oid,
            name=base_name,
            kind="mesh",
            collection_id=collection_id if collection_id in scene.collections else "default",
            material_id=mat_id,
            mesh=md.to_payload(),
        )
        scene.objects[oid] = obj
        created.append(obj)
        return created

    if isinstance(loaded, trimesh.Scene):
        for node_name in loaded.graph.nodes_geometry:
            mat4, geom_name = loaded.graph[node_name]
            geom = loaded.geometry.get(geom_name)
            if not isinstance(geom, trimesh.Trimesh) or len(geom.faces) == 0:
                continue
            md = MeshData.from_trimesh(geom)
            mat_world = yup_to_zup @ np.asarray(mat4, dtype=np.float64)
            loc, rot_deg, scale = decompose_matrix(mat_world)
            scale = tuple(s if abs(s) > 1e-6 else 1.0 for s in scale)  # type: ignore[assignment]
            base = f"{name_prefix}_{node_name}" if name_prefix else node_name
            obj_name = scene.unique_name(base)
            oid = scene.next_id(obj_name)
            mat_id = _extract_pbr_material(scene, geom.visual, obj_name, tex_dir, mat_cache)

            obj = SceneObject(
                id=oid,
                name=obj_name,
                kind="mesh",
                transform=Transform(location=loc, rotation_deg=rot_deg, scale=scale),
                collection_id=collection_id if collection_id in scene.collections else "default",
                material_id=mat_id,
                mesh=md.to_payload(),
            )
            scene.objects[oid] = obj
            created.append(obj)
        return created

    raise ValueError(f"Unsupported loaded object type from {p}: {type(loaded).__name__}")
