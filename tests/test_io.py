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
