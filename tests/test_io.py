"""Tests for .frf.json project I/O, GLB/GLTF/OBJ/STL/PLY/Blender exports, and importers."""

from __future__ import annotations

import ast
import json
import struct
from pathlib import Path

import pytest
import trimesh

from frfruit.animation import FCurveTrack, Keyframe
from frfruit.core.scene import MaterialSpec, SceneDocument, SceneObject
from frfruit.core.types import Transform
from frfruit.geometry.primitives import make_cube, make_sphere
from frfruit.io import (
    export_blender_python,
    export_scene_file,
    import_mesh_file,
    load_project,
    save_project,
)
from frfruit.render.materials import PRESET_MATERIALS


def _build_export_scene() -> SceneDocument:
    doc = SceneDocument.new_default("ExportScene")
    doc.materials["mat_gold"] = MaterialSpec(id="mat_gold", **PRESET_MATERIALS["gold"])
    doc.objects["box"] = SceneObject(
        id="box",
        name="GoldBox",
        kind="mesh",
        material_id="mat_gold",
        transform=Transform(location=(1.0, 0.5, 0.5), rotation_deg=(0.0, 0.0, 15.0)),
        mesh=make_cube(1.0, origin="center").to_payload(),
    )
    doc.objects["orb"] = SceneObject(
        id="orb",
        name="Orb",
        kind="mesh",
        transform=Transform(location=(-1.0, 0.0, 0.5)),
        mesh=make_sphere(0.45, segments=16, origin="center").to_payload(),
    )
    doc.timeline.frame_start = 1
    doc.timeline.frame_end = 24
    doc.timeline.tracks = [
        FCurveTrack(
            target_id="box",
            property_path="location.x",
            keyframes=[
                Keyframe(frame=1, value=1.0, interpolation="linear"),
                Keyframe(frame=24, value=3.0, interpolation="linear"),
            ],
        ),
        FCurveTrack(
            target_id="box",
            property_path="rotation_deg.z",
            keyframes=[
                Keyframe(frame=1, value=0.0, interpolation="linear"),
                Keyframe(frame=24, value=90.0, interpolation="linear"),
            ],
        ),
        FCurveTrack(
            target_id="box",
            property_path="scale.y",
            keyframes=[
                Keyframe(frame=1, value=1.0, interpolation="linear"),
                Keyframe(frame=24, value=1.5, interpolation="linear"),
            ],
        ),
    ]
    return doc


def test_project_save_load_roundtrip(tmp_path: Path) -> None:
    doc = _build_export_scene()
    p = save_project(doc, tmp_path / "project.frf.json")
    loaded = load_project(p)
    assert loaded.name == doc.name
    assert set(loaded.objects.keys()) == set(doc.objects.keys())
    assert loaded.materials["mat_gold"].metallic == 1.0
    assert len(loaded.timeline.tracks) == 3

    with pytest.raises(FileNotFoundError):
        load_project(tmp_path / "missing.frf.json")


def test_glb_animation_and_all_export_formats(tmp_path: Path) -> None:
    doc = _build_export_scene()

    glb_path = tmp_path / "out" / "asset.glb"
    res_glb = export_scene_file(doc, glb_path)
    assert res_glb["format"] == "glb"
    raw = glb_path.read_bytes()
    assert raw[:4] == b"glTF"

    # Inspect glTF JSON chunk for valid animations & TRS node compliance
    json_len, _ = struct.unpack("<II", raw[12:20])
    gltf = json.loads(raw[20 : 20 + json_len].decode("utf-8"))
    assert "animations" in gltf and len(gltf["animations"]) == 1
    anim_channels = gltf["animations"][0]["channels"]
    assert len(anim_channels) == 3  # translation, rotation, scale
    animated_node_indices = {ch["target"]["node"] for ch in anim_channels}
    for nidx in animated_node_indices:
        assert "matrix" not in gltf["nodes"][nidx]
        assert "translation" in gltf["nodes"][nidx]

    loaded_glb = trimesh.load(str(glb_path))
    assert isinstance(loaded_glb, trimesh.Scene)
    assert len(loaded_glb.geometry) >= 2

    # Static formats
    for ext in ("gltf", "obj", "stl", "ply"):
        out_f = tmp_path / "out" / f"asset.{ext}"
        res = export_scene_file(doc, out_f)
        assert out_f.exists() and res["bytes"] > 50
        reloaded = trimesh.load(str(out_f))
        assert reloaded is not None

    # Blender bpy script
    py_path = tmp_path / "out" / "scene_bpy.py"
    res_py = export_scene_file(doc, py_path)
    assert res_py["format"] == "blender_py"
    script = export_blender_python(doc)
    ast.parse(script)
    assert "GoldBox" in script
    assert "bpy.data.materials.new" in script

    with pytest.raises(ValueError, match="Unsupported export format"):
        export_scene_file(doc, tmp_path / "bad.xyz")


def test_import_mesh_into_new_scene(tmp_path: Path) -> None:
    doc = _build_export_scene()
    glb_path = tmp_path / "roundtrip.glb"
    export_scene_file(doc, glb_path)

    fresh = SceneDocument.new_default("Fresh")
    imported_glb = import_mesh_file(fresh, glb_path, name_prefix="imp")
    assert len(imported_glb) >= 2
    assert any("box" in o.id or "box" in o.name.lower() for o in imported_glb)

    obj_path = tmp_path / "roundtrip.obj"
    export_scene_file(doc, obj_path)
    imported_obj = import_mesh_file(fresh, obj_path, name_prefix="mesh_obj")
    assert len(imported_obj) >= 1

    with pytest.raises(FileNotFoundError):
        import_mesh_file(fresh, tmp_path / "nope.glb")


def test_single_face_material_exports_glb_obj_blender_py(tmp_path: Path) -> None:
    doc = SceneDocument.new_default("MultiMat")
    doc.materials["mat_clay"] = MaterialSpec(id="mat_clay", **PRESET_MATERIALS["clay"])
    doc.materials["mat_gold"] = MaterialSpec(id="mat_gold", **PRESET_MATERIALS["gold"])
    doc.objects["cube"] = SceneObject(
        id="cube",
        name="SingleFaceCube",
        kind="mesh",
        material_id="mat_clay",
        face_material_ids={0: "mat_gold"},
        mesh=make_cube(1.0, origin="center").to_payload(),
    )

    glb_path = tmp_path / "multimat.glb"
    export_scene_file(doc, glb_path)
    loaded = trimesh.load(str(glb_path))
    assert isinstance(loaded, trimesh.Scene)
    assert len(loaded.geometry) == 2

    obj_path = tmp_path / "multimat.obj"
    export_scene_file(doc, obj_path)
    obj_txt = obj_path.read_text(encoding="utf-8")
    assert "mtllib multimat.mtl" in obj_txt
    assert obj_txt.count("usemtl ") >= 2
    assert (tmp_path / "multimat.mtl").is_file()

    py_txt = export_blender_python(doc)
    ast.parse(py_txt)
    assert "obj.data.materials.append(materials['mat_clay'])" in py_txt
    assert "obj.data.materials.append(materials['mat_gold'])" in py_txt
    assert "mesh_data.polygons[_fi].material_index = _si" in py_txt


def test_glb_pbr_texture_extraction_and_yup_roundtrip(tmp_path: Path) -> None:
    import numpy as np
    from PIL import Image

    from frfruit.geometry.primitives import make_cylinder
    from frfruit.geometry.spatial import compute_scene_bounds, evaluate_scene_meshes

    tex_dir = tmp_path / "src_tex"
    tex_dir.mkdir()
    albedo_p = tex_dir / "alb.png"
    normal_p = tex_dir / "nrm.png"
    rough_p = tex_dir / "rgh.png"
    metal_p = tex_dir / "mtl.png"

    Image.new("RGB", (32, 32), (180, 90, 45)).save(albedo_p)
    Image.new("RGB", (32, 32), (128, 128, 255)).save(normal_p)
    Image.new("L", (32, 32), 160).save(rough_p)
    Image.new("L", (32, 32), 200).save(metal_p)

    doc = SceneDocument.new_default("TexturedGLB")
    doc.materials["mat_pbr"] = MaterialSpec(
        id="mat_pbr",
        name="BrushedBronze",
        base_color=(1.0, 1.0, 1.0, 1.0),
        metallic=1.0,
        roughness=1.0,
        albedo_map=str(albedo_p),
        normal_map=str(normal_p),
        roughness_map=str(rough_p),
        metallic_map=str(metal_p),
    )
    # Tall cylinder (Z height = 2.4, diameter = 0.6) to verify Y-up -> Z-up roundtrip
    cyl = make_cylinder(radius=0.3, height=2.4, segments=12)
    doc.objects["pillar"] = SceneObject(
        id="pillar",
        name="TallPillar",
        kind="mesh",
        material_id="mat_pbr",
        transform=Transform(location=(0.0, 0.0, 1.2)),
        mesh=cyl.to_payload(),
    )

    glb_path = tmp_path / "pillar.glb"
    export_scene_file(doc, glb_path)

    fresh = SceneDocument.new_default("Imported")
    out_tex = tmp_path / "imp_tex"
    objs = import_mesh_file(fresh, glb_path, name_prefix="imp", texture_dir=out_tex)
    assert len(objs) == 1
    imp_obj = objs[0]
    assert imp_obj.material_id is not None
    mat = fresh.materials[imp_obj.material_id]
    assert mat.albedo_map is not None and Path(mat.albedo_map).is_file()
    assert mat.normal_map is not None and Path(mat.normal_map).is_file()
    assert mat.roughness_map is not None and Path(mat.roughness_map).is_file()
    assert mat.metallic_map is not None and Path(mat.metallic_map).is_file()

    bounds = compute_scene_bounds(evaluate_scene_meshes(fresh))
    size = bounds.size
    assert size[2] > 2.0 and np.isclose(size[2], 2.4, atol=0.05)


def test_mesh_payload_null_uv_roundtrip(tmp_path: Path) -> None:
    import math

    import numpy as np

    from frfruit.geometry.mesh import concat_meshes
    from frfruit.geometry.primitives import make_sphere

    cube = make_cube(1.0)
    cube.uvs = np.tile([0.25, 0.75], (len(cube.vertices), 1))
    sphere_no_uv = make_sphere(0.5, segments=8)
    sphere_no_uv.uvs = None
    merged = concat_meshes([cube, sphere_no_uv])

    doc = SceneDocument.new_default("NullUVScene")
    doc.objects["merged"] = SceneObject(
        id="merged",
        name="Merged",
        kind="mesh",
        mesh=merged.to_payload(),
    )
    proj_path = save_project(doc, tmp_path / "null_uv.frf.json")
    raw_txt = proj_path.read_text(encoding="utf-8")
    assert "null" in raw_txt  # NaN UVs serialize to JSON null

    reloaded = load_project(proj_path)
    uvs = reloaded.objects["merged"].mesh.uvs
    assert uvs is not None and len(uvs) == len(merged.vertices)
    assert any(math.isnan(c) for pair in uvs for c in pair)


def test_online_model_fetcher_and_cli_models(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import numpy as np

    from frfruit.cli import main as cli_main
    from frfruit.core.math3d import compose_matrix
    from frfruit.geometry.spatial import compute_scene_bounds, evaluate_scene_meshes
    from frfruit.io import download_online_model, import_online_model, search_online_models

    # 1. Procedural catalog search + upright import + height scaling + ground placement
    hits = search_online_models("lamp", source="procedural", limit=5)
    assert len(hits) == 1 and hits[0]["asset_id"] == "street_lamp"

    dl_dir = tmp_path / "dl"
    dl_info = download_online_model("chair", source="procedural", target_dir=dl_dir)
    assert dl_info["status"] == "ok" and Path(dl_info["model_path"]).is_file()

    sc = SceneDocument.new_default("ModelsTest")
    res = import_online_model(
        sc,
        "street_lamp",
        source="procedural",
        target_dir=dl_dir,
        location=(2.0, -1.5, 0.0),
        target_height=3.0,
        place_on_ground=True,
    )
    assert res["status"] == "ok"
    assert res["root_id"] in sc.objects
    assert sc.objects[res["root_id"]].kind == "empty"
    bounds = compute_scene_bounds(evaluate_scene_meshes(sc))
    assert np.isclose(bounds.min_corner[2], 0.0, atol=1e-4)
    assert np.isclose(bounds.max_corner[2], 3.0, atol=1e-3)
    assert np.isclose(bounds.center[0], 2.0, atol=0.5)
    assert np.isclose(bounds.center[1], -1.5, atol=0.5)

    # 2. Mocked Poly Haven API workflow (offline test of live glTF JSON + include download)
    cube_mesh = make_cube(0.8, origin="center")
    cube_yup = cube_mesh.copy()
    cube_yup.vertices = cube_yup.vertices @ compose_matrix(rotation_deg=(-90.0, 0.0, 0.0))[:3, :3].T
    tm_glb = cube_yup.to_trimesh(process=False).export(file_type="glb")

    def fake_get_json(url: str, timeout: float = 15.0) -> dict:
        if "assets?t=models" in url:
            return {
                "wooden_crate_01": {
                    "name": "Wooden Crate 01",
                    "categories": ["props", "wooden"],
                    "tags": ["crate", "wood"],
                    "download_count": 1200,
                    "polycount": 420,
                    "dimensions": [800, 800, 800],
                }
            }
        if "files/wooden_crate_01" in url:
            return {
                "gltf": {
                    "1k": {
                        "glb": {
                            "url": "https://dl.polyhaven.org/file/ph-assets/Models/gltf/1k/wooden_crate_01.glb",
                            "include": {},
                        }
                    }
                }
            }
        raise AssertionError(f"Unexpected URL: {url}")

    monkeypatch.setattr("frfruit.io.model_fetcher._http_get_json", fake_get_json)
    monkeypatch.setattr(
        "frfruit.io.model_fetcher._http_get_bytes", lambda url, timeout=60.0: tm_glb
    )

    ph_hits = search_online_models("crate", source="polyhaven", limit=5)
    assert len(ph_hits) == 1 and ph_hits[0]["asset_id"] == "wooden_crate_01"

    sc_ph = SceneDocument.new_default("PolyHavenMock")
    ph_res = import_online_model(
        sc_ph,
        "wooden_crate_01",
        source="polyhaven",
        target_dir=dl_dir,
        target_height=1.2,
    )
    assert ph_res["source"] == "polyhaven"
    ph_bounds = compute_scene_bounds(evaluate_scene_meshes(sc_ph))
    assert np.isclose(ph_bounds.min_corner[2], 0.0, atol=1e-4)
    assert np.isclose(ph_bounds.max_corner[2], 1.2, atol=1e-3)

    # 3. CLI `frfruit models` end-to-end
    proj_file = tmp_path / "cli_proj.frf.json"
    save_project(SceneDocument.new_default("CLIModels"), proj_file)
    assert cli_main(["models", "chair", "--source", "procedural"]) == 0
    assert (
        cli_main(
            [
                "models",
                "--import-into",
                str(proj_file),
                "--asset",
                "barrel",
                "--source",
                "procedural",
                "-o",
                str(dl_dir),
                "--pos",
                "1.0",
                "2.0",
                "0.0",
                "--height",
                "0.9",
            ]
        )
        == 0
    )
    loaded_cli = load_project(proj_file)
    cli_bounds = compute_scene_bounds(evaluate_scene_meshes(loaded_cli))
    assert np.isclose(cli_bounds.max_corner[2], 0.9, atol=1e-3)
