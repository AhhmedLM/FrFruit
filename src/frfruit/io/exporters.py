"""Scene exporters: GLB/GLTF (with PBR & animations), OBJ+MTL, STL, PLY, and Blender .py."""

from __future__ import annotations

import json
import math
import struct
from pathlib import Path
from typing import Any

import numpy as np
import trimesh
from trimesh.visual import TextureVisuals
from trimesh.visual.material import PBRMaterial

from frfruit.animation.timeline import evaluate_fcurve
from frfruit.core.math3d import compose_matrix, euler_deg_to_matrix
from frfruit.core.scene import MaterialSpec, SceneDocument
from frfruit.geometry.mesh import MeshData, concat_meshes
from frfruit.geometry.spatial import evaluate_object_mesh, evaluate_scene_meshes


def _euler_deg_to_quat_xyzw(rx_deg: float, ry_deg: float, rz_deg: float) -> list[float]:
    m = euler_deg_to_matrix(rx_deg, ry_deg, rz_deg)
    tr = m[0, 0] + m[1, 1] + m[2, 2]
    if tr > 0:
        s = math.sqrt(tr + 1.0) * 2.0
        w = 0.25 * s
        x = (m[2, 1] - m[1, 2]) / s
        y = (m[0, 2] - m[2, 0]) / s
        z = (m[1, 0] - m[0, 1]) / s
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = math.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2.0
        w = (m[2, 1] - m[1, 2]) / s
        x = 0.25 * s
        y = (m[0, 1] + m[1, 0]) / s
        z = (m[0, 2] + m[2, 0]) / s
    elif m[1, 1] > m[2, 2]:
        s = math.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2.0
        w = (m[0, 2] - m[2, 0]) / s
        x = (m[0, 1] + m[1, 0]) / s
        y = 0.25 * s
        z = (m[1, 2] + m[2, 1]) / s
    else:
        s = math.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2.0
        w = (m[1, 0] - m[0, 1]) / s
        x = (m[0, 2] + m[2, 0]) / s
        y = (m[1, 2] + m[2, 1]) / s
        z = 0.25 * s
    return [float(x), float(y), float(z), float(w)]


def _material_to_pbr(mat: MaterialSpec | None) -> PBRMaterial:
    if mat is None:
        return PBRMaterial(
            name="Default",
            baseColorFactor=[0.78, 0.78, 0.80, 1.0],
            metallicFactor=0.0,
            roughnessFactor=0.5,
        )
    alpha_mode = "BLEND" if mat.alpha < 0.999 else "OPAQUE"
    emissive = (
        [
            min(1.0, mat.emission_color[0] * mat.emission_strength),
            min(1.0, mat.emission_color[1] * mat.emission_strength),
            min(1.0, mat.emission_color[2] * mat.emission_strength),
        ]
        if mat.emission_strength > 0
        else [0.0, 0.0, 0.0]
    )
    return PBRMaterial(
        name=mat.name,
        baseColorFactor=[
            float(mat.base_color[0]),
            float(mat.base_color[1]),
            float(mat.base_color[2]),
            float(mat.alpha * mat.base_color[3]),
        ],
        metallicFactor=float(mat.metallic),
        roughnessFactor=float(mat.roughness),
        emissiveFactor=emissive,
        alphaMode=alpha_mode,
    )


def _build_trimesh_scene(scene: SceneDocument, yup: bool = True) -> trimesh.Scene:
    tm_scene = trimesh.Scene()
    root_name = "FrFruitRoot"
    if yup:
        # Rotate -90 deg around X so Z-up maps to glTF Y-up
        yup_mat = compose_matrix(rotation_deg=(-90.0, 0.0, 0.0))
        tm_scene.graph.update(frame_to=root_name, matrix=yup_mat)

    for oid, obj in scene.objects.items():
        if obj.kind not in ("mesh", "procedural") or not scene.is_object_visible(obj):
            continue
        ev = evaluate_object_mesh(scene, oid)
        if ev.local_mesh.is_empty:
            continue
        tm = ev.local_mesh.to_trimesh(process=False)
        mat = scene.materials.get(obj.material_id) if obj.material_id else None
        pbr = _material_to_pbr(mat)
        uvs = (
            ev.local_mesh.uvs
            if ev.local_mesh.uvs is not None and len(ev.local_mesh.uvs) == len(tm.vertices)
            else None
        )
        tm.visual = TextureVisuals(uv=uvs, material=pbr)
        local_mat = compose_matrix(
            location=obj.transform.location,
            rotation_deg=obj.transform.rotation_deg,
            scale=obj.transform.scale,
        )
        parent_frame = (
            obj.parent_id
            if (obj.parent_id and obj.parent_id in scene.objects)
            else (root_name if yup else "world")
        )
        tm_scene.add_geometry(
            tm,
            node_name=oid,
            geom_name=f"{oid}_geom",
            parent_node_name=parent_frame,
            transform=local_mat,
        )
    return tm_scene


def _eval_vec3(
    by_prop: dict[str, Any],
    prefix: str,
    default_val: tuple[float, float, float],
    frame: int,
) -> list[float]:
    res = []
    for idx, ax in enumerate(("x", "y", "z")):
        tr = by_prop.get(f"{prefix}.{ax}")
        v = evaluate_fcurve(tr, float(frame)) if tr else None
        res.append(float(default_val[idx] if v is None else v))
    return res


def _inject_glb_animations(glb_bytes: bytes, scene: SceneDocument) -> bytes:
    """Inject glTF 2.0 TRS keyframe animation tracks into a GLB binary."""
    if not scene.timeline.tracks:
        return glb_bytes

    magic, version, total_len = struct.unpack("<III", glb_bytes[:12])
    if magic != 0x46546C67:
        return glb_bytes

    offset = 12
    json_len, json_type = struct.unpack("<II", glb_bytes[offset : offset + 8])
    offset += 8
    gltf = json.loads(glb_bytes[offset : offset + json_len].decode("utf-8"))
    offset += json_len

    bin_data = bytearray()
    if offset + 8 <= len(glb_bytes):
        bin_len, bin_type = struct.unpack("<II", glb_bytes[offset : offset + 8])
        if bin_type == 0x004E4942:
            bin_data = bytearray(glb_bytes[offset + 8 : offset + 8 + bin_len])

    nodes = gltf.get("nodes", [])
    node_idx_by_name = {n.get("name"): i for i, n in enumerate(nodes) if n.get("name")}

    # Group tracks by object id
    tracks_by_obj: dict[str, list] = {}
    for tr in scene.timeline.tracks:
        obj = scene.resolve_object(tr.target_id)
        if obj is not None and obj.id in node_idx_by_name:
            tracks_by_obj.setdefault(obj.id, []).append(tr)

    if not tracks_by_obj:
        return glb_bytes

    buffer_views = gltf.setdefault("bufferViews", [])
    accessors = gltf.setdefault("accessors", [])
    samplers: list[dict] = []
    channels: list[dict] = []
    fps = float(max(1, scene.timeline.fps))

    def append_bytes(raw: bytes) -> tuple[int, int]:
        while len(bin_data) % 4 != 0:
            bin_data.append(0)
        byte_off = len(bin_data)
        bin_data.extend(raw)
        return byte_off, len(raw)

    for oid, obj_tracks in tracks_by_obj.items():
        obj = scene.objects[oid]
        node_idx = node_idx_by_name[oid]
        frames_set = sorted({kf.frame for tr in obj_tracks for kf in tr.keyframes})
        if len(frames_set) < 2:
            continue
        # glTF 2.0 forbids `matrix` on nodes targeted by TRS animation channels
        target_node = nodes[node_idx]
        if "matrix" in target_node:
            target_node.pop("matrix")
            target_node["translation"] = [float(x) for x in obj.transform.location]
            target_node["rotation"] = _euler_deg_to_quat_xyzw(*obj.transform.rotation_deg)
            target_node["scale"] = [float(x) for x in obj.transform.scale]
        times = np.array(
            [(f - scene.timeline.frame_start) / fps for f in frames_set], dtype=np.float32
        )
        t_off, t_len = append_bytes(times.tobytes())
        t_bv_idx = len(buffer_views)
        buffer_views.append({"buffer": 0, "byteOffset": t_off, "byteLength": t_len})
        t_acc_idx = len(accessors)
        accessors.append(
            {
                "bufferView": t_bv_idx,
                "componentType": 5126,
                "count": len(times),
                "type": "SCALAR",
                "min": [float(times[0])],
                "max": [float(times[-1])],
            }
        )

        # Sample translation, rotation, scale across frames
        by_prop = {tr.property_path: tr for tr in obj_tracks}
        has_loc = any(p.startswith("location.") for p in by_prop)
        has_rot = any(p.startswith("rotation_deg.") for p in by_prop)
        has_scl = any(p.startswith("scale.") for p in by_prop)

        if has_loc:
            arr = np.array(
                [_eval_vec3(by_prop, "location", obj.transform.location, f) for f in frames_set],
                dtype=np.float32,
            )
            off, ln = append_bytes(arr.tobytes())
            bv = len(buffer_views)
            buffer_views.append({"buffer": 0, "byteOffset": off, "byteLength": ln})
            acc = len(accessors)
            accessors.append(
                {"bufferView": bv, "componentType": 5126, "count": len(arr), "type": "VEC3"}
            )
            samp = len(samplers)
            samplers.append({"input": t_acc_idx, "interpolation": "LINEAR", "output": acc})
            channels.append({"sampler": samp, "target": {"node": node_idx, "path": "translation"}})

        if has_rot:
            quats = np.array(
                [
                    _euler_deg_to_quat_xyzw(
                        *_eval_vec3(by_prop, "rotation_deg", obj.transform.rotation_deg, f)
                    )
                    for f in frames_set
                ],
                dtype=np.float32,
            )
            off, ln = append_bytes(quats.tobytes())
            bv = len(buffer_views)
            buffer_views.append({"buffer": 0, "byteOffset": off, "byteLength": ln})
            acc = len(accessors)
            accessors.append(
                {"bufferView": bv, "componentType": 5126, "count": len(quats), "type": "VEC4"}
            )
            samp = len(samplers)
            samplers.append({"input": t_acc_idx, "interpolation": "LINEAR", "output": acc})
            channels.append({"sampler": samp, "target": {"node": node_idx, "path": "rotation"}})

        if has_scl:
            arr = np.array(
                [_eval_vec3(by_prop, "scale", obj.transform.scale, f) for f in frames_set],
                dtype=np.float32,
            )
            off, ln = append_bytes(arr.tobytes())
            bv = len(buffer_views)
            buffer_views.append({"buffer": 0, "byteOffset": off, "byteLength": ln})
            acc = len(accessors)
            accessors.append(
                {"bufferView": bv, "componentType": 5126, "count": len(arr), "type": "VEC3"}
            )
            samp = len(samplers)
            samplers.append({"input": t_acc_idx, "interpolation": "LINEAR", "output": acc})
            channels.append({"sampler": samp, "target": {"node": node_idx, "path": "scale"}})

    if not channels:
        return glb_bytes

    gltf.setdefault("animations", []).append(
        {"name": f"{scene.name}_Anim", "samplers": samplers, "channels": channels}
    )
    while len(bin_data) % 4 != 0:
        bin_data.append(0)
    if gltf.get("buffers"):
        gltf["buffers"][0]["byteLength"] = len(bin_data)

    json_bytes = bytearray(json.dumps(gltf, separators=(",", ":")).encode("utf-8"))
    while len(json_bytes) % 4 != 0:
        json_bytes.append(0x20)

    new_total = 12 + 8 + len(json_bytes) + 8 + len(bin_data)
    out = bytearray()
    out.extend(struct.pack("<III", 0x46546C67, 2, new_total))
    out.extend(struct.pack("<II", len(json_bytes), 0x4E4F534A))
    out.extend(json_bytes)
    out.extend(struct.pack("<II", len(bin_data), 0x004E4942))
    out.extend(bin_data)
    return bytes(out)


def export_blender_python(scene: SceneDocument) -> str:
    """Generate a standalone Blender bpy script recreating the evaluated scene."""
    evaluated = evaluate_scene_meshes(scene)
    lines: list[str] = [
        '"""Auto-generated Blender bpy script exported from FrFruit."""',
        "import bpy",
        "import math",
        "",
        "bpy.ops.wm.read_factory_settings(use_empty=True)",
        "scene = bpy.context.scene",
        f"scene.name = {scene.name!r}",
        f"scene.render.fps = {scene.timeline.fps}",
        f"scene.frame_start = {scene.timeline.frame_start}",
        f"scene.frame_end = {scene.timeline.frame_end}",
        "",
        "materials = {}",
    ]
    for mid, mat in scene.materials.items():
        lines.extend(
            [
                f"m = bpy.data.materials.new(name={mat.name!r})",
                "m.use_nodes = True",
                "bsdf = m.node_tree.nodes.get('Principled BSDF')",
                "if bsdf:",
                f"    bsdf.inputs['Base Color'].default_value = {list(mat.base_color)!r}",
                f"    bsdf.inputs['Metallic'].default_value = {mat.metallic}",
                f"    bsdf.inputs['Roughness'].default_value = {mat.roughness}",
                f"materials[{mid!r}] = m",
            ]
        )

    lines.append("")
    for _oid, ev in evaluated.items():
        if ev.local_mesh.is_empty:
            continue
        verts = np.round(ev.local_mesh.vertices, 6).tolist()
        faces = ev.local_mesh.faces.tolist()
        loc = list(ev.obj.transform.location)
        rot_rad = [math.radians(a) for a in ev.obj.transform.rotation_deg]
        scale = list(ev.obj.transform.scale)
        lines.extend(
            [
                f"mesh_data = bpy.data.meshes.new({ev.obj.name + '_Mesh'!r})",
                f"mesh_data.from_pydata({verts!r}, [], {faces!r})",
                "mesh_data.update()",
                f"obj = bpy.data.objects.new({ev.obj.name!r}, mesh_data)",
                "scene.collection.objects.link(obj)",
                f"obj.location = {loc!r}",
                f"obj.rotation_euler = {rot_rad!r}",
                f"obj.scale = {scale!r}",
            ]
        )
        if ev.obj.material_id and ev.obj.material_id in scene.materials:
            lines.append(f"obj.data.materials.append(materials[{ev.obj.material_id!r}])")
        lines.append("")

    for oid, obj in scene.objects.items():
        if obj.kind == "camera" and obj.camera is not None:
            lines.extend(
                [
                    f"cam_data = bpy.data.cameras.new({obj.name!r})",
                    f"cam_obj = bpy.data.objects.new({obj.name!r}, cam_data)",
                    "scene.collection.objects.link(cam_obj)",
                    f"cam_obj.location = {list(obj.transform.location)!r}",
                ]
            )
            if oid == scene.active_camera_id:
                lines.append("scene.camera = cam_obj")
        elif obj.kind == "light" and obj.light is not None:
            lmap = {"sun": "SUN", "point": "POINT", "spot": "SPOT", "area": "AREA"}
            lines.extend(
                [
                    f"l_data = bpy.data.lights.new({obj.name!r}, type={lmap[obj.light.kind]!r})",
                    f"l_data.energy = {obj.light.intensity * 100.0}",
                    f"l_data.color = {list(obj.light.color)!r}",
                    f"l_obj = bpy.data.objects.new({obj.name!r}, l_data)",
                    "scene.collection.objects.link(l_obj)",
                    f"l_obj.location = {list(obj.transform.location)!r}",
                ]
            )

    return "\n".join(lines) + "\n"


def export_scene_file(
    scene: SceneDocument,
    out_path: str | Path,
    fmt: str | None = None,
) -> dict[str, object]:
    """Export SceneDocument to glb, gltf, obj, stl, ply, or Blender .py."""
    p = Path(out_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    ext = (fmt or p.suffix.lstrip(".")).lower()

    if ext in ("glb", "gltf"):
        tm_scene = _build_trimesh_scene(scene, yup=True)
        if ext == "glb":
            raw = tm_scene.export(file_type="glb")
            raw = _inject_glb_animations(raw, scene)
            p.write_bytes(raw)
        else:
            exported = tm_scene.export(file_type="gltf")
            if isinstance(exported, dict):
                for fname, data in exported.items():
                    target = p if fname == "model.gltf" else p.parent / fname
                    if isinstance(data, bytes):
                        target.write_bytes(data)
                    else:
                        target.write_text(str(data), encoding="utf-8")
            elif isinstance(exported, bytes):
                p.write_bytes(exported)
            else:
                p.write_text(str(exported), encoding="utf-8")
        return {"path": str(p), "format": ext, "bytes": p.stat().st_size}

    if ext in ("obj", "stl", "ply"):
        evaluated = evaluate_scene_meshes(scene)
        merged = concat_meshes(
            [ev.world_mesh for ev in evaluated.values() if not ev.world_mesh.is_empty]
        )
        tm = (
            merged.to_trimesh(process=False)
            if not merged.is_empty
            else MeshData(
                vertices=np.zeros((0, 3)), faces=np.zeros((0, 3), dtype=np.int64)
            ).to_trimesh()
        )
        tm.export(str(p), file_type=ext)
        return {"path": str(p), "format": ext, "bytes": p.stat().st_size}

    if ext == "py":
        script = export_blender_python(scene)
        p.write_text(script, encoding="utf-8")
        return {"path": str(p), "format": "blender_py", "bytes": p.stat().st_size}

    raise ValueError(f"Unsupported export format {ext!r}. Use glb, gltf, obj, stl, ply, or py.")
