"""Scene exporters: GLB/GLTF (with PBR & animations), OBJ+MTL, STL, PLY, and Blender .py."""

from __future__ import annotations

import json
import math
import struct
from pathlib import Path
from typing import Any

import numpy as np
import trimesh
from PIL import Image
from trimesh.visual import TextureVisuals
from trimesh.visual.material import PBRMaterial

from frfruit.animation.timeline import evaluate_fcurve
from frfruit.core.math3d import compose_matrix, euler_deg_to_matrix
from frfruit.core.scene import MaterialSpec, SceneDocument
from frfruit.geometry.mesh import MeshData, concat_meshes
from frfruit.geometry.spatial import evaluate_object_mesh, evaluate_scene_meshes
from frfruit.io.fbx_exporter import TargetEngine, export_fbx_scene
from frfruit.textures.loader import compute_triplanar_uvs


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


def _open_pil(path_str: str | None) -> Image.Image | None:
    if not path_str:
        return None
    p = Path(path_str)
    if not p.is_file():
        return None
    try:
        return Image.open(p).convert("RGB")
    except Exception:
        return None


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
    base_tex = _open_pil(mat.albedo_map)
    nrm_tex = _open_pil(mat.normal_map)
    rgh_tex = _open_pil(mat.roughness_map)
    mtl_tex = _open_pil(mat.metallic_map)
    mr_tex: Image.Image | None = None
    if rgh_tex is not None or mtl_tex is not None:
        ref = rgh_tex or mtl_tex
        assert ref is not None
        size = ref.size
        r_arr = np.zeros((size[1], size[0]), dtype=np.uint8)
        g_arr = (
            np.asarray(rgh_tex.resize(size).convert("L"), dtype=np.uint8)
            if rgh_tex is not None
            else np.full((size[1], size[0]), int(mat.roughness * 255), dtype=np.uint8)
        )
        b_arr = (
            np.asarray(mtl_tex.resize(size).convert("L"), dtype=np.uint8)
            if mtl_tex is not None
            else np.full((size[1], size[0]), int(mat.metallic * 255), dtype=np.uint8)
        )
        mr_tex = Image.fromarray(np.stack([r_arr, g_arr, b_arr], axis=-1), mode="RGB")

    kwargs: dict[str, Any] = {
        "name": mat.name,
        "baseColorFactor": [
            float(mat.base_color[0]),
            float(mat.base_color[1]),
            float(mat.base_color[2]),
            float(mat.alpha * mat.base_color[3]),
        ],
        "metallicFactor": 1.0 if mtl_tex is not None else float(mat.metallic),
        "roughnessFactor": 1.0 if rgh_tex is not None else float(mat.roughness),
        "emissiveFactor": emissive,
        "alphaMode": alpha_mode,
    }
    if base_tex is not None:
        kwargs["baseColorTexture"] = base_tex
    if nrm_tex is not None:
        kwargs["normalTexture"] = nrm_tex
    if mr_tex is not None:
        kwargs["metallicRoughnessTexture"] = mr_tex
    return PBRMaterial(**kwargs)


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
        uvs = (
            ev.local_mesh.uvs
            if ev.local_mesh.uvs is not None
            and len(ev.local_mesh.uvs) == len(ev.local_mesh.vertices)
            else None
        )
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

        # Group faces by effective material ID
        n_faces = len(ev.local_mesh.faces)
        face_groups: dict[str | None, list[int]] = {}
        if obj.face_material_ids:
            for fi in range(n_faces):
                mid = obj.face_material_ids.get(fi, obj.material_id)
                if mid not in scene.materials:
                    mid = obj.material_id if (obj.material_id in scene.materials) else None
                face_groups.setdefault(mid, []).append(fi)
        else:
            eff = obj.material_id if (obj.material_id in scene.materials) else None
            face_groups[eff] = list(range(n_faces))

        if len(face_groups) <= 1:
            mid = next(iter(face_groups.keys()))
            mat = scene.materials.get(mid) if mid else None
            pbr = _material_to_pbr(mat)
            if (
                uvs is None
                and mat
                and any((mat.albedo_map, mat.normal_map, mat.roughness_map, mat.metallic_map))
            ):
                uvs = compute_triplanar_uvs(ev.local_mesh.vertices, ev.local_mesh.vertex_normals())
            tm = ev.local_mesh.to_trimesh(process=False)
            tm.visual = TextureVisuals(uv=uvs, material=pbr)
            tm_scene.add_geometry(
                tm,
                node_name=oid,
                geom_name=f"{oid}_geom",
                parent_node_name=parent_frame,
                transform=local_mat,
            )
        else:
            tm_scene.graph.update(frame_from=parent_frame, frame_to=oid, matrix=local_mat)
            base_norms = ev.local_mesh.vertex_normals()
            for slot_idx, (mid, f_idxs) in enumerate(face_groups.items()):
                mat = scene.materials.get(mid) if mid else None
                pbr = _material_to_pbr(mat)
                eff_uvs = uvs
                if (
                    eff_uvs is None
                    and mat
                    and any((mat.albedo_map, mat.normal_map, mat.roughness_map, mat.metallic_map))
                ):
                    eff_uvs = compute_triplanar_uvs(ev.local_mesh.vertices, base_norms)
                sub_faces_orig = ev.local_mesh.faces[f_idxs]
                uniq_v, inv = np.unique(sub_faces_orig.reshape(-1), return_inverse=True)
                sub_tm = trimesh.Trimesh(
                    vertices=ev.local_mesh.vertices[uniq_v],
                    faces=inv.reshape(-1, 3),
                    process=False,
                )
                sub_uvs = eff_uvs[uniq_v] if eff_uvs is not None else None
                sub_tm.visual = TextureVisuals(uv=sub_uvs, material=pbr)
                tm_scene.add_geometry(
                    sub_tm,
                    node_name=f"{oid}_mat{slot_idx}",
                    geom_name=f"{oid}_geom_{slot_idx}",
                    parent_node_name=oid,
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
                "nt = m.node_tree",
                "bsdf = nt.nodes.get('Principled BSDF')",
                "if bsdf:",
                f"    bsdf.inputs['Base Color'].default_value = {list(mat.base_color)!r}",
                f"    bsdf.inputs['Metallic'].default_value = {mat.metallic}",
                f"    bsdf.inputs['Roughness'].default_value = {mat.roughness}",
            ]
        )
        for slot_path, socket_name, is_noncolor in (
            (mat.albedo_map, "Base Color", False),
            (mat.roughness_map, "Roughness", True),
            (mat.metallic_map, "Metallic", True),
        ):
            if slot_path and Path(slot_path).is_file():
                s_rep = repr(str(slot_path))
                lines.extend(
                    [
                        "    tex = nt.nodes.new('ShaderNodeTexImage')",
                        f"    tex.image = bpy.data.images.load({s_rep}, check_existing=True)",
                        f"    if {is_noncolor!r}: tex.image.colorspace_settings.name = 'Non-Color'",
                        f"    nt.links.new(tex.outputs['Color'], bsdf.inputs[{socket_name!r}])",
                    ]
                )
        if mat.normal_map and Path(mat.normal_map).is_file():
            n_rep = repr(str(mat.normal_map))
            lines.extend(
                [
                    "    tex_n = nt.nodes.new('ShaderNodeTexImage')",
                    f"    tex_n.image = bpy.data.images.load({n_rep}, check_existing=True)",
                    "    tex_n.image.colorspace_settings.name = 'Non-Color'",
                    "    nmap = nt.nodes.new('ShaderNodeNormalMap')",
                    "    nt.links.new(tex_n.outputs['Color'], nmap.inputs['Color'])",
                    "    nt.links.new(nmap.outputs['Normal'], bsdf.inputs['Normal'])",
                ]
            )
        lines.append(f"materials[{mid!r}] = m")

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
        slot_mids: list[str] = []
        slot_of_mid: dict[str, int] = {}
        if ev.obj.material_id and ev.obj.material_id in scene.materials:
            slot_of_mid[ev.obj.material_id] = 0
            slot_mids.append(ev.obj.material_id)
        for _fi, mid in sorted(ev.obj.face_material_ids.items()):
            if mid in scene.materials and mid not in slot_of_mid:
                slot_of_mid[mid] = len(slot_mids)
                slot_mids.append(mid)
        for mid in slot_mids:
            lines.append(f"obj.data.materials.append(materials[{mid!r}])")
        face_slot_map = {
            int(fi): slot_of_mid[mid]
            for fi, mid in sorted(ev.obj.face_material_ids.items())
            if mid in slot_of_mid and 0 <= int(fi) < len(faces)
        }
        if face_slot_map:
            lines.append(f"for _fi, _si in {face_slot_map!r}.items():")
            lines.append("    mesh_data.polygons[_fi].material_index = _si")
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


def _export_obj_with_mtl(scene: SceneDocument, out_obj: Path) -> None:
    evaluated = evaluate_scene_meshes(scene)
    mtl_path = out_obj.with_suffix(".mtl")
    obj_lines: list[str] = [f"# FrFruit OBJ Export ({scene.name})", f"mtllib {mtl_path.name}"]
    mtl_lines: list[str] = [f"# FrFruit MTL ({scene.name})"]

    used_mats: dict[str, MaterialSpec] = {}
    for _oid, ev in evaluated.items():
        if ev.obj.material_id and ev.obj.material_id in scene.materials:
            used_mats[ev.obj.material_id] = scene.materials[ev.obj.material_id]
        for mid in ev.obj.face_material_ids.values():
            if mid in scene.materials:
                used_mats[mid] = scene.materials[mid]

    safe_mtl_names: dict[str, str] = {}
    for mid, m in used_mats.items():
        sname = f"{m.name}_{mid}".replace(" ", "_")
        safe_mtl_names[mid] = sname
        r, g, b, _ = m.base_color
        er, eg, eb, _ = m.emission_color
        ks = max(0.04, m.metallic)
        ke_r = er * m.emission_strength
        ke_g = eg * m.emission_strength
        ke_b = eb * m.emission_strength
        mtl_lines.extend(
            [
                f"newmtl {sname}",
                f"Kd {r:.5f} {g:.5f} {b:.5f}",
                f"Ks {ks:.5f} {ks:.5f} {ks:.5f}",
                f"Ns {max(2.0, (1.0 - m.roughness) ** 2 * 250.0):.2f}",
                f"Ke {ke_r:.5f} {ke_g:.5f} {ke_b:.5f}",
                f"d {m.alpha:.4f}",
                "",
            ]
        )
    if used_mats:
        mtl_path.write_text("\n".join(mtl_lines), encoding="utf-8")

    v_offset = 1
    vt_offset = 1
    vn_offset = 1
    for _oid, ev in evaluated.items():
        wm = ev.world_mesh
        if wm.is_empty:
            continue
        obj_lines.append(f"o {ev.obj.name.replace(' ', '_')}")
        verts = wm.vertices
        norms = wm.vertex_normals()
        uvs = (
            wm.uvs
            if wm.uvs is not None and len(wm.uvs) == len(verts)
            else compute_triplanar_uvs(verts, norms)
        )
        for v in verts:
            obj_lines.append(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}")
        for uv in uvs:
            obj_lines.append(f"vt {uv[0]:.5f} {uv[1]:.5f}")
        for vn in norms:
            obj_lines.append(f"vn {vn[0]:.5f} {vn[1]:.5f} {vn[2]:.5f}")

        cur_mat: str | None = "__unset__"
        for fi, (a, b, c) in enumerate(wm.faces):
            f_mid = ev.obj.face_material_ids.get(fi, ev.obj.material_id)
            sname = safe_mtl_names.get(f_mid) if f_mid else None
            if sname != cur_mat:
                cur_mat = sname
                if sname:
                    obj_lines.append(f"usemtl {sname}")
            ia, ib, ic = int(a), int(b), int(c)
            obj_lines.append(
                f"f {ia + v_offset}/{ia + vt_offset}/{ia + vn_offset} "
                f"{ib + v_offset}/{ib + vt_offset}/{ib + vn_offset} "
                f"{ic + v_offset}/{ic + vt_offset}/{ic + vn_offset}"
            )
        v_offset += len(verts)
        vt_offset += len(verts)
        vn_offset += len(verts)

    out_obj.write_text("\n".join(obj_lines) + "\n", encoding="utf-8")


def export_scene_file(
    scene: SceneDocument,
    out_path: str | Path,
    fmt: str | None = None,
    target_engine: TargetEngine = "generic",
) -> dict[str, object]:
    """Export SceneDocument to fbx, fbx_ascii, glb, gltf, obj, stl, ply, or Blender .py."""
    p = Path(out_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    ext = (fmt or p.suffix.lstrip(".")).lower()

    if ext in ("fbx", "fbx_ascii"):
        return export_fbx_scene(
            scene,
            p,
            ascii_mode=(ext == "fbx_ascii"),
            target_engine=target_engine,
        )

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

    if ext == "obj":
        _export_obj_with_mtl(scene, p)
        return {"path": str(p), "format": ext, "bytes": p.stat().st_size}

    if ext in ("stl", "ply"):
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

    raise ValueError(
        f"Unsupported export format {ext!r}. Use fbx, fbx_ascii, glb, gltf, obj, stl, ply, or py."
    )
