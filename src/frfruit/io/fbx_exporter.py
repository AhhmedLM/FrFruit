"""Game-ready Autodesk Binary FBX 7.4 & ASCII FBX exporter with rigging, animation, & morphs."""

from __future__ import annotations

import struct
import zlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import numpy as np

from frfruit.animation.timeline import evaluate_fcurve
from frfruit.core.scene import BoneSpec, SceneDocument
from frfruit.geometry.mesh import MeshData
from frfruit.geometry.spatial import compute_world_matrix, evaluate_object_mesh
from frfruit.textures.loader import compute_triplanar_uvs

FBX_HEADER_MAGIC = b"Kaydara FBX Binary  \x00\x1a\x00"
FBX_VERSION = 7400
KTIME_ONE_SECOND = 46186158000

TargetEngine = Literal["generic", "unreal", "unity", "godot"]

# Standard 16-byte binary footer tail expected by Autodesk FBX SDK / Blender / Assimp readers
_FBX_FOOTER_UNKNOWN = bytes.fromhex("fa bc ab 09 d0 c8 d4 66 b1 76 fb 83 1c f7 26 7e")


@dataclass
class _FBXNode:
    name: str
    props: list[tuple[str, Any]] = field(default_factory=list)
    children: list[_FBXNode] | None = None

    def add(self, name: str, *props: tuple[str, Any], block: bool = False) -> _FBXNode:
        if self.children is None:
            self.children = []
        child = _FBXNode(name=name, props=list(props), children=[] if block else None)
        self.children.append(child)
        return child

    def p_prop(self, name: str, ptype: str, psub: str, flags: str, *vals: Any) -> _FBXNode:
        return self.add(
            "P",
            ("S", name),
            ("S", ptype),
            ("S", psub),
            ("S", flags),
            *vals,
        )


def _fbx_id_name(name: str, cls: str) -> str:
    return f"{name}\x00\x01{cls}"


def _encode_property(ptype: str, val: Any) -> bytes:
    code = ptype.encode("ascii")
    if ptype == "Y":
        return code + struct.pack("<h", int(val))
    if ptype == "C":
        return code + struct.pack("<B", 1 if val else 0)
    if ptype == "I":
        return code + struct.pack("<i", int(val))
    if ptype == "F":
        return code + struct.pack("<f", float(val))
    if ptype == "D":
        return code + struct.pack("<d", float(val))
    if ptype == "L":
        return code + struct.pack("<q", int(val))
    if ptype == "S":
        raw = str(val).encode("utf-8")
        return code + struct.pack("<I", len(raw)) + raw
    if ptype == "R":
        raw = bytes(val)
        return code + struct.pack("<I", len(raw)) + raw
    if ptype in ("f", "d", "l", "i", "b"):
        np_map = {
            "f": np.float32,
            "d": np.float64,
            "l": np.int64,
            "i": np.int32,
            "b": np.uint8,
        }
        arr = np.asarray(val, dtype=np_map[ptype]).reshape(-1)
        raw = arr.tobytes()
        n = len(arr)
        if len(raw) > 256:
            comp = zlib.compress(raw)
            return code + struct.pack("<III", n, 1, len(comp)) + comp
        return code + struct.pack("<III", n, 0, len(raw)) + raw
    raise ValueError(f"Unsupported FBX property type: {ptype!r}")


def _write_node(buf: bytearray, node: _FBXNode) -> None:
    start_offset = len(buf)
    name_bytes = node.name.encode("utf-8")
    props_bytes = bytearray()
    for pt, pv in node.props:
        props_bytes.extend(_encode_property(pt, pv))

    # Reserve 13-byte record header: EndOffset (4), NumProps (4), PropListLen (4), NameLen (1)
    buf.extend(b"\x00" * 13)
    buf.extend(name_bytes)
    buf.extend(props_bytes)

    if node.children is not None:
        for ch in node.children:
            _write_node(buf, ch)
        buf.extend(b"\x00" * 13)

    end_offset = len(buf)
    header = struct.pack("<IIIB", end_offset, len(node.props), len(props_bytes), len(name_bytes))
    buf[start_offset : start_offset + 13] = header


def _serialize_fbx_binary(root_nodes: list[_FBXNode]) -> bytes:
    buf = bytearray()
    buf.extend(FBX_HEADER_MAGIC)
    buf.extend(struct.pack("<I", FBX_VERSION))
    for n in root_nodes:
        _write_node(buf, n)
    buf.extend(b"\x00" * 13)

    # Standard FBX footer alignment & signature block
    buf.extend(_FBX_FOOTER_UNKNOWN)
    pad_len = (-len(buf)) % 16
    if pad_len == 0:
        pad_len = 16
    buf.extend(b"\x00" * pad_len)
    buf.extend(struct.pack("<I", FBX_VERSION))
    buf.extend(b"\x00" * 120)
    buf.extend(bytes.fromhex("f8 5a 8c 6a de f5 d9 7e ec e9 0c e3 75 8f 29 0b"))
    return bytes(buf)


def verify_fbx_binary(path_or_bytes: str | Path | bytes) -> dict[str, Any]:
    """Parse and validate a Kaydara Binary FBX 7.4 file/buffer structurally."""
    raw = (
        path_or_bytes
        if isinstance(path_or_bytes, (bytes, bytearray))
        else Path(path_or_bytes).read_bytes()
    )
    if len(raw) < 27 or raw[:23] != FBX_HEADER_MAGIC:
        raise ValueError("Invalid FBX binary header magic.")
    (version,) = struct.unpack("<I", raw[23:27])

    def _read_node(offset: int) -> tuple[dict[str, Any] | None, int]:
        if offset + 13 > len(raw):
            raise ValueError(f"Truncated FBX node header at offset {offset}.")
        end_off, num_props, prop_len, name_len = struct.unpack("<IIIB", raw[offset : offset + 13])
        if end_off == 0 and num_props == 0 and prop_len == 0 and name_len == 0:
            return None, offset + 13
        pos = offset + 13
        name = raw[pos : pos + name_len].decode("utf-8", errors="replace")
        pos += name_len + prop_len
        children = []
        if pos < end_off:
            # Must end with 13-byte null record inside child list
            while pos < end_off - 13:
                ch, pos = _read_node(pos)
                if ch is None:
                    break
                children.append(ch)
            # Consume trailing 13-byte null record if loop ended cleanly before end_off
            if pos + 13 == end_off and raw[pos:end_off] == b"\x00" * 13:
                pos += 13
        if pos != end_off:
            raise ValueError(f"FBX node {name!r} end_off mismatch: pos={pos}, end={end_off}")
        return {"name": name, "num_props": num_props, "children": children}, end_off

    top_nodes: list[dict[str, Any]] = []
    cur = 27
    while cur + 13 <= len(raw):
        node, cur = _read_node(cur)
        if node is None:
            break
        top_nodes.append(node)

    def _count_nodes(n: dict[str, Any]) -> int:
        return 1 + sum(_count_nodes(c) for c in n["children"])

    top_names = [n["name"] for n in top_nodes]
    return {
        "valid": True,
        "version": version,
        "top_level_nodes": top_names,
        "total_nodes": sum(_count_nodes(n) for n in top_nodes),
        "bytes": len(raw),
    }


def _compute_polygon_vertex_normals_and_tangents(
    verts: np.ndarray,
    faces: np.ndarray,
    uvs: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute per-polygon-vertex smooth normals plus MikkTSpace-style tangents and binormals."""
    m = len(faces)
    if m == 0:
        empty = np.zeros((0,), dtype=np.float64)
        return empty, empty, empty

    v0 = verts[faces[:, 0]]
    v1 = verts[faces[:, 1]]
    v2 = verts[faces[:, 2]]
    face_cross = np.cross(v1 - v0, v2 - v0)

    # Accumulate area-weighted smooth normals per vertex
    v_norms = np.zeros_like(verts)
    np.add.at(v_norms, faces[:, 0], face_cross)
    np.add.at(v_norms, faces[:, 1], face_cross)
    np.add.at(v_norms, faces[:, 2], face_cross)
    v_lens = np.linalg.norm(v_norms, axis=1, keepdims=True)
    v_norms = v_norms / np.where(v_lens < 1e-12, 1.0, v_lens)

    pv_normals = v_norms[faces.reshape(-1)]

    # Triangle tangent/binormal from UV edge gradients
    uv0 = uvs[faces[:, 0]]
    uv1 = uvs[faces[:, 1]]
    uv2 = uvs[faces[:, 2]]
    e1 = v1 - v0
    e2 = v2 - v0
    du1 = uv1[:, 0] - uv0[:, 0]
    dv1 = uv1[:, 1] - uv0[:, 1]
    du2 = uv2[:, 0] - uv0[:, 0]
    dv2 = uv2[:, 1] - uv0[:, 1]
    det = du1 * dv2 - du2 * dv1
    safe_det = np.where(np.abs(det) > 1e-9, det, 1.0)
    inv_det = np.where(np.abs(det) > 1e-9, 1.0 / safe_det, 0.0)[:, None]

    tri_t = (e1 * dv2[:, None] - e2 * dv1[:, None]) * inv_det
    tri_b = (e2 * du1[:, None] - e1 * du2[:, None]) * inv_det

    # Fallback orthonormal tangent if UV is degenerate
    degenerate = np.linalg.norm(tri_t, axis=1) < 1e-6
    if np.any(degenerate):
        fn = face_cross[degenerate]
        fn_len = np.linalg.norm(fn, axis=1, keepdims=True)
        fn = fn / np.where(fn_len < 1e-12, 1.0, fn_len)
        up = np.where(np.abs(fn[:, 2:3]) > 0.9, [[1.0, 0.0, 0.0]], [[0.0, 0.0, 1.0]])
        t_fb = np.cross(up, fn)
        t_fb /= np.maximum(np.linalg.norm(t_fb, axis=1, keepdims=True), 1e-8)
        tri_t[degenerate] = t_fb
        tri_b[degenerate] = np.cross(fn, t_fb)

    pv_t = np.repeat(tri_t, 3, axis=0)
    pv_b = np.repeat(tri_b, 3, axis=0)

    # Gram-Schmidt orthogonalize tangent to each polygon vertex normal
    dot_nt = np.sum(pv_normals * pv_t, axis=1, keepdims=True)
    pv_t = pv_t - pv_normals * dot_nt
    pv_t /= np.maximum(np.linalg.norm(pv_t, axis=1, keepdims=True), 1e-8)
    pv_b = np.cross(pv_normals, pv_t)
    pv_b /= np.maximum(np.linalg.norm(pv_b, axis=1, keepdims=True), 1e-8)

    return pv_normals.reshape(-1), pv_t.reshape(-1), pv_b.reshape(-1)


def _node_to_ascii(node: _FBXNode, indent: int = 0) -> list[str]:
    pad = "\t" * indent
    prop_strs: list[str] = []
    for pt, pv in node.props:
        if pt == "S":
            clean = str(pv).replace("\x00\x01", "::").replace('"', '\\"')
            prop_strs.append(f'"{clean}"')
        elif pt == "C":
            prop_strs.append("1" if pv else "0")
        elif pt in ("I", "L", "Y"):
            prop_strs.append(str(int(pv)))
        elif pt in ("F", "D"):
            prop_strs.append(f"{float(pv):.6f}")
        elif pt in ("f", "d", "l", "i", "b"):
            arr = np.asarray(pv).reshape(-1)
            body = ",".join(f"{float(x):.6f}" if pt in ("f", "d") else str(int(x)) for x in arr)
            prop_strs.append(f"*{len(arr)} {{\n{pad}\ta: {body}\n{pad}}}")
    line = f"{pad}{node.name}: " + ", ".join(prop_strs)
    if node.children is not None:
        lines = [line + " {"]
        for ch in node.children:
            lines.extend(_node_to_ascii(ch, indent + 1))
        lines.append(f"{pad}}}")
        return lines
    return [line]


def export_fbx_scene(
    scene: SceneDocument,
    out_path: str | Path,
    ascii_mode: bool = False,
    target_engine: TargetEngine = "generic",
) -> dict[str, Any]:
    """Export SceneDocument to Binary FBX 7.4 or ASCII FBX with bones, animations, and morphs."""
    p = Path(out_path)
    p.parent.mkdir(parents=True, exist_ok=True)

    yup_convert = target_engine in ("unity", "godot")
    next_uid = 100000

    def uid() -> int:
        nonlocal next_uid
        next_uid += 1
        return next_uid

    # Evaluate rest bind meshes at frame_start (without baked vertex_cache overwrite)
    fps = max(1, int(scene.timeline.fps))
    fs = int(scene.timeline.frame_start)
    fe = int(scene.timeline.frame_end)
    if fe < fs:
        fe = fs

    # --- Build HeaderExtension & GlobalSettings ---
    header_ext = _FBXNode("FBXHeaderExtension", children=[])
    header_ext.add("FBXHeaderVersion", ("I", 1003))
    header_ext.add("FBXVersion", ("I", FBX_VERSION))
    header_ext.add("EncryptionType", ("I", 0))
    ts = header_ext.add("CreationTimeStamp", block=True)
    ts.add("Version", ("I", 1000))
    ts.add("Year", ("I", 2026))
    ts.add("Month", ("I", 1))
    ts.add("Day", ("I", 1))
    header_ext.add("Creator", ("S", "FrFruit 3D Engine FBX 7.4 Exporter"))

    global_settings = _FBXNode("GlobalSettings", children=[])
    global_settings.add("Version", ("I", 1000))
    gs_p70 = global_settings.add("Properties70", block=True)
    if yup_convert:
        gs_p70.p_prop("UpAxis", "int", "Integer", "", ("I", 1))
        gs_p70.p_prop("UpAxisSign", "int", "Integer", "", ("I", 1))
        gs_p70.p_prop("FrontAxis", "int", "Integer", "", ("I", 2))
        gs_p70.p_prop("FrontAxisSign", "int", "Integer", "", ("I", 1))
        gs_p70.p_prop("CoordAxis", "int", "Integer", "", ("I", 0))
        gs_p70.p_prop("CoordAxisSign", "int", "Integer", "", ("I", 1))
    else:
        gs_p70.p_prop("UpAxis", "int", "Integer", "", ("I", 2))
        gs_p70.p_prop("UpAxisSign", "int", "Integer", "", ("I", 1))
        gs_p70.p_prop("FrontAxis", "int", "Integer", "", ("I", 1))
        gs_p70.p_prop("FrontAxisSign", "int", "Integer", "", ("I", -1))
        gs_p70.p_prop("CoordAxis", "int", "Integer", "", ("I", 0))
        gs_p70.p_prop("CoordAxisSign", "int", "Integer", "", ("I", 1))

    unit_scale = 100.0 if scene.unit_system == "meters" else 1.0
    gs_p70.p_prop("UnitScaleFactor", "double", "Number", "", ("D", unit_scale))
    gs_p70.p_prop("CustomFrameRate", "double", "Number", "", ("D", float(fps)))

    documents = _FBXNode("Documents", children=[])
    documents.add("Count", ("I", 1))
    doc_node = documents.add(
        "Document", ("L", uid()), ("S", scene.name), ("S", "Scene"), block=True
    )
    doc_node.add("Properties70", block=True).p_prop("SourceObject", "object", "", "")
    doc_node.add("RootNode", ("L", 0))

    references = _FBXNode("References", children=[])
    objects_node = _FBXNode("Objects", children=[])
    connections_node = _FBXNode("Connections", children=[])

    counts: dict[str, int] = {
        "Geometry": 0,
        "Model": 0,
        "Material": 0,
        "Texture": 0,
        "Video": 0,
        "NodeAttribute": 0,
        "Deformer": 0,
        "Pose": 0,
        "AnimationStack": 0,
        "AnimationLayer": 0,
        "AnimationCurveNode": 0,
        "AnimationCurve": 0,
    }

    def connect_oo(child_id: int, parent_id: int) -> None:
        connections_node.add("C", ("S", "OO"), ("L", child_id), ("L", parent_id))

    def connect_op(child_id: int, parent_id: int, prop_name: str) -> None:
        connections_node.add("C", ("S", "OP"), ("L", child_id), ("L", parent_id), ("S", prop_name))

    def _conv_pos(v: tuple[float, float, float] | list[float]) -> tuple[float, float, float]:
        if yup_convert:
            return (float(v[0]), float(v[2]), float(-v[1]))
        return (float(v[0]), float(v[1]), float(v[2]))

    def _conv_rot(r: tuple[float, float, float] | list[float]) -> tuple[float, float, float]:
        if yup_convert:
            return (float(r[0]), float(r[2]), float(-r[1]))
        return (float(r[0]), float(r[1]), float(r[2]))

    def _conv_scl(s: tuple[float, float, float] | list[float]) -> tuple[float, float, float]:
        if yup_convert:
            return (float(s[0]), float(s[2]), float(s[1]))
        return (float(s[0]), float(s[1]), float(s[2]))

    def _conv_pts_arr(pts: np.ndarray) -> np.ndarray:
        if yup_convert and len(pts) > 0:
            return np.column_stack([pts[:, 0], pts[:, 2], -pts[:, 1]])
        return pts

    def _conv_mat4(mat: np.ndarray) -> np.ndarray:
        if not yup_convert:
            return mat
        # Basis change C * mat * C^T rotating -90 deg about X: (x, y, z) -> (x, z, -y)
        c_mat = np.array(
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, -1.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )
        return c_mat @ mat @ c_mat.T

    # 1. Materials & PBR Texture Maps
    mat_fbx_ids: dict[str, int] = {}

    def add_texture_slot(mat_id_fbx: int, slot_prop: str, file_path: str, name_tag: str) -> None:
        tex_path = Path(file_path)
        vid_uid = uid()
        tex_uid = uid()
        counts["Video"] += 1
        counts["Texture"] += 1

        vid_node = objects_node.add(
            "Video",
            ("L", vid_uid),
            ("S", _fbx_id_name(name_tag, "Video")),
            ("S", "Clip"),
            block=True,
        )
        vid_node.add("Type", ("S", "Clip"))
        vid_p = vid_node.add("Properties70", block=True)
        vid_p.p_prop("Path", "KString", "XRefUrl", "", ("S", str(tex_path)))
        vid_node.add("UseMipMap", ("I", 0))
        vid_node.add("Filename", ("S", str(tex_path)))
        vid_node.add("RelativeFilename", ("S", tex_path.name))

        tex_node = objects_node.add(
            "Texture",
            ("L", tex_uid),
            ("S", _fbx_id_name(name_tag, "Texture")),
            ("S", ""),
            block=True,
        )
        tex_node.add("Type", ("S", "TextureVideoClip"))
        tex_node.add("Version", ("I", 202))
        tex_node.add("TextureName", ("S", _fbx_id_name(name_tag, "Texture")))
        tex_node.add("Media", ("S", _fbx_id_name(name_tag, "Video")))
        tex_node.add("FileName", ("S", str(tex_path)))
        tex_node.add("RelativeFilename", ("S", tex_path.name))
        tex_node.add("ModelUVTranslation", ("D", 0.0), ("D", 0.0))
        tex_node.add("ModelUVScaling", ("D", 1.0), ("D", 1.0))
        tex_node.add("Texture_Alpha_Source", ("S", "None"))
        connect_oo(vid_uid, tex_uid)
        connect_op(tex_uid, mat_id_fbx, slot_prop)

    for mid, mat in scene.materials.items():
        m_uid = uid()
        counts["Material"] += 1
        mat_fbx_ids[mid] = m_uid
        m_node = objects_node.add(
            "Material",
            ("L", m_uid),
            ("S", _fbx_id_name(mat.name, "Material")),
            ("S", ""),
            block=True,
        )
        m_node.add("Version", ("I", 102))
        m_node.add("ShadingModel", ("S", "Phong"))
        m_node.add("MultiLayer", ("I", 0))
        mp = m_node.add("Properties70", block=True)
        r, g, b, _a = mat.base_color
        mp.p_prop("DiffuseColor", "Color", "", "A", ("D", r), ("D", g), ("D", b))
        mp.p_prop("SpecularColor", "Color", "", "A", ("D", 0.5), ("D", 0.5), ("D", 0.5))
        shin = max(2.0, (1.0 - mat.roughness) ** 2 * 100.0)
        mp.p_prop("ShininessExponent", "Number", "", "A", ("D", shin))
        mp.p_prop("Reflectivity", "Number", "", "A", ("D", mat.metallic))
        mp.p_prop("Opacity", "Number", "", "A", ("D", mat.alpha))
        if mat.emission_strength > 0:
            er, eg, eb, _ = mat.emission_color
            mp.p_prop("EmissiveColor", "Color", "", "A", ("D", er), ("D", eg), ("D", eb))
            mp.p_prop("EmissiveFactor", "Number", "", "A", ("D", mat.emission_strength))

        if mat.albedo_map:
            add_texture_slot(m_uid, "DiffuseColor", mat.albedo_map, f"{mat.name}_Albedo")
        if mat.normal_map:
            add_texture_slot(m_uid, "NormalMap", mat.normal_map, f"{mat.name}_Normal")
        if mat.roughness_map:
            add_texture_slot(m_uid, "ShininessExponent", mat.roughness_map, f"{mat.name}_Rough")
        if mat.metallic_map:
            add_texture_slot(m_uid, "ReflectionFactor", mat.metallic_map, f"{mat.name}_Metal")

    # 2. Scene Objects: Armatures (LimbNode trees) + Meshes (Geometry + Skins + Cloth Morphs)
    obj_model_uids: dict[str, int] = {}
    bone_model_uids: dict[tuple[str, str], int] = {}  # (arm_obj_id, bone_name) -> uid
    bone_bind_world: dict[tuple[str, str], np.ndarray] = {}
    morph_channel_uids: dict[
        str, list[tuple[int, int, int]]
    ] = {}  # obj_id -> [(chan_uid, frame, total)]

    # First pass: create Model nodes & Armature LimbNodes
    for oid, obj in scene.objects.items():
        if obj.kind == "armature" and obj.armature is not None:
            arm_uid = uid()
            counts["Model"] += 1
            obj_model_uids[oid] = arm_uid
            arm_node = objects_node.add(
                "Model",
                ("L", arm_uid),
                ("S", _fbx_id_name(obj.name, "Model")),
                ("S", "Null"),
                block=True,
            )
            arm_node.add("Version", ("I", 232))
            ap = arm_node.add("Properties70", block=True)
            lx, ly, lz = _conv_pos(obj.transform.location)
            rx, ry, rz = _conv_rot(obj.transform.rotation_deg)
            sx, sy, sz = _conv_scl(obj.transform.scale)
            ap.p_prop(
                "Lcl Translation", "Lcl Translation", "", "A", ("D", lx), ("D", ly), ("D", lz)
            )
            ap.p_prop("Lcl Rotation", "Lcl Rotation", "", "A", ("D", rx), ("D", ry), ("D", rz))
            ap.p_prop("Lcl Scaling", "Lcl Scaling", "", "A", ("D", sx), ("D", sy), ("D", sz))

            arm_w_mat = compute_world_matrix(scene, oid)
            bones_map: dict[str, BoneSpec] = {b.name: b for b in obj.armature.bones}
            for bone in obj.armature.bones:
                b_uid = uid()
                na_uid = uid()
                counts["Model"] += 1
                counts["NodeAttribute"] += 1
                bone_model_uids[(oid, bone.name)] = b_uid

                # World bind matrix at bone rest head
                bind_local = np.eye(4, dtype=np.float64)
                bind_local[:3, 3] = np.asarray(bone.head, dtype=np.float64)
                bone_bind_world[(oid, bone.name)] = arm_w_mat @ bind_local

                na = objects_node.add(
                    "NodeAttribute",
                    ("L", na_uid),
                    ("S", _fbx_id_name(bone.name, "NodeAttribute")),
                    ("S", "LimbNode"),
                    block=True,
                )
                na.add("TypeFlags", ("S", "Skeleton"))
                na.add("Properties70", block=True).p_prop(
                    "Size", "double", "Number", "", ("D", 0.25)
                )

                bm = objects_node.add(
                    "Model",
                    ("L", b_uid),
                    ("S", _fbx_id_name(bone.name, "Model")),
                    ("S", "LimbNode"),
                    block=True,
                )
                bm.add("Version", ("I", 232))
                bp = bm.add("Properties70", block=True)
                p_head = (
                    np.asarray(bones_map[bone.parent].head, dtype=np.float64)
                    if (bone.parent and bone.parent in bones_map)
                    else np.zeros(3, dtype=np.float64)
                )
                rel_h = (
                    np.asarray(bone.head, dtype=np.float64) - p_head + np.asarray(bone.translation)
                )
                bx, by, bz = _conv_pos(rel_h.tolist())
                brx, bry, brz = _conv_rot(bone.rotation_deg)
                bp.p_prop(
                    "Lcl Translation", "Lcl Translation", "", "A", ("D", bx), ("D", by), ("D", bz)
                )
                bp.p_prop(
                    "Lcl Rotation", "Lcl Rotation", "", "A", ("D", brx), ("D", bry), ("D", brz)
                )
                bp.p_prop("Lcl Scaling", "Lcl Scaling", "", "A", ("D", 1.0), ("D", 1.0), ("D", 1.0))

                connect_oo(na_uid, b_uid)

            # Connect bone parent-child relationships
            for bone in obj.armature.bones:
                b_uid = bone_model_uids[(oid, bone.name)]
                if bone.parent and (oid, bone.parent) in bone_model_uids:
                    connect_oo(b_uid, bone_model_uids[(oid, bone.parent)])
                else:
                    connect_oo(b_uid, arm_uid)

        elif obj.kind in ("mesh", "procedural") and scene.is_object_visible(obj):
            # Evaluate rest pose without vertex_cache & runtime skinning for clean bind geometry
            saved_cache = obj.vertex_cache
            saved_skin = obj.skin
            obj.vertex_cache = {}
            obj.skin = None
            try:
                ev_bind = evaluate_object_mesh(scene, oid)
            finally:
                obj.vertex_cache = saved_cache
                obj.skin = saved_skin

            if ev_bind.local_mesh.is_empty:
                continue

            m_uid = uid()
            g_uid = uid()
            counts["Model"] += 1
            counts["Geometry"] += 1
            obj_model_uids[oid] = m_uid

            mdl = objects_node.add(
                "Model",
                ("L", m_uid),
                ("S", _fbx_id_name(obj.name, "Model")),
                ("S", "Mesh"),
                block=True,
            )
            mdl.add("Version", ("I", 232))
            mp = mdl.add("Properties70", block=True)
            lx, ly, lz = _conv_pos(obj.transform.location)
            rx, ry, rz = _conv_rot(obj.transform.rotation_deg)
            sx, sy, sz = _conv_scl(obj.transform.scale)
            mp.p_prop(
                "Lcl Translation", "Lcl Translation", "", "A", ("D", lx), ("D", ly), ("D", lz)
            )
            mp.p_prop("Lcl Rotation", "Lcl Rotation", "", "A", ("D", rx), ("D", ry), ("D", rz))
            mp.p_prop("Lcl Scaling", "Lcl Scaling", "", "A", ("D", sx), ("D", sy), ("D", sz))
            mdl.add("Shading", ("C", True))
            mdl.add("Culling", ("S", "CullingOff"))

            # Mesh Geometry
            lm: MeshData = ev_bind.local_mesh
            verts = _conv_pts_arr(lm.vertices)
            faces = lm.faces
            if lm.uvs is not None and len(lm.uvs) == len(verts):
                uvs = lm.uvs
            else:
                uvs = compute_triplanar_uvs(lm.vertices, lm.vertex_normals())

            pv_norms, pv_tans, pv_bins = _compute_polygon_vertex_normals_and_tangents(
                verts, faces, uvs
            )
            poly_indices = faces.copy().astype(np.int32)
            poly_indices[:, 2] = ~poly_indices[:, 2]

            geom = objects_node.add(
                "Geometry",
                ("L", g_uid),
                ("S", _fbx_id_name(f"{obj.name}_Mesh", "Geometry")),
                ("S", "Mesh"),
                block=True,
            )
            geom.add("Vertices", ("d", verts.reshape(-1)))
            geom.add("PolygonVertexIndex", ("i", poly_indices.reshape(-1)))
            geom.add("GeometryVersion", ("I", 124))

            # Normals
            le_n = geom.add("LayerElementNormal", ("I", 0), block=True)
            le_n.add("Version", ("I", 101))
            le_n.add("Name", ("S", ""))
            le_n.add("MappingInformationType", ("S", "ByPolygonVertex"))
            le_n.add("ReferenceInformationType", ("S", "Direct"))
            le_n.add("Normals", ("d", pv_norms))

            # Tangents & Binormals (Unreal/Unity MikkTSpace ready)
            le_t = geom.add("LayerElementTangent", ("I", 0), block=True)
            le_t.add("Version", ("I", 101))
            le_t.add("Name", ("S", ""))
            le_t.add("MappingInformationType", ("S", "ByPolygonVertex"))
            le_t.add("ReferenceInformationType", ("S", "Direct"))
            le_t.add("Tangents", ("d", pv_tans))

            le_b = geom.add("LayerElementBinormal", ("I", 0), block=True)
            le_b.add("Version", ("I", 101))
            le_b.add("Name", ("S", ""))
            le_b.add("MappingInformationType", ("S", "ByPolygonVertex"))
            le_b.add("ReferenceInformationType", ("S", "Direct"))
            le_b.add("Binormals", ("d", pv_bins))

            # UVs
            le_uv = geom.add("LayerElementUV", ("I", 0), block=True)
            le_uv.add("Version", ("I", 101))
            le_uv.add("Name", ("S", "UVMap"))
            le_uv.add("MappingInformationType", ("S", "ByPolygonVertex"))
            le_uv.add("ReferenceInformationType", ("S", "IndexToDirect"))
            le_uv.add("UV", ("d", uvs.reshape(-1)))
            le_uv.add("UVIndex", ("i", faces.reshape(-1)))

            # Materials (AllSame or ByPolygon when face_material_ids is populated)
            le_m = geom.add("LayerElementMaterial", ("I", 0), block=True)
            le_m.add("Version", ("I", 101))
            le_m.add("Name", ("S", ""))
            valid_face_mats = {
                int(fi): mid
                for fi, mid in obj.face_material_ids.items()
                if mid in mat_fbx_ids and 0 <= int(fi) < len(faces)
            }
            if valid_face_mats:
                slot_mids: list[str] = []
                slot_of_mid: dict[str, int] = {}
                if obj.material_id and obj.material_id in mat_fbx_ids:
                    slot_of_mid[obj.material_id] = 0
                    slot_mids.append(obj.material_id)
                fallback_mid = (
                    obj.material_id
                    if (obj.material_id and obj.material_id in mat_fbx_ids)
                    else next(iter(valid_face_mats.values()))
                )
                per_face_slots: list[int] = []
                for fi in range(len(faces)):
                    f_mid = valid_face_mats.get(fi, fallback_mid)
                    if f_mid not in slot_of_mid:
                        slot_of_mid[f_mid] = len(slot_mids)
                        slot_mids.append(f_mid)
                    per_face_slots.append(slot_of_mid[f_mid])
                le_m.add("MappingInformationType", ("S", "ByPolygon"))
                le_m.add("ReferenceInformationType", ("S", "IndexToDirect"))
                le_m.add("Materials", ("i", per_face_slots))
                connect_mids = slot_mids
            else:
                le_m.add("MappingInformationType", ("S", "AllSame"))
                le_m.add("ReferenceInformationType", ("S", "IndexToDirect"))
                le_m.add("Materials", ("i", [0]))
                connect_mids = (
                    [obj.material_id]
                    if (obj.material_id and obj.material_id in mat_fbx_ids)
                    else []
                )

            layer0 = geom.add("Layer", ("I", 0), block=True)
            layer0.add("Version", ("I", 100))
            for le_type in (
                "LayerElementNormal",
                "LayerElementTangent",
                "LayerElementBinormal",
                "LayerElementMaterial",
                "LayerElementUV",
            ):
                le_ref = layer0.add("LayerElement", block=True)
                le_ref.add("Type", ("S", le_type))
                le_ref.add("TypedIndex", ("I", 0))

            connect_oo(g_uid, m_uid)
            for mid_c in connect_mids:
                connect_oo(mat_fbx_ids[mid_c], m_uid)

            # Skin Deformer + Clusters + BindPose if rigged
            if obj.skin is not None and obj.skin.armature_id in scene.objects:
                arm_obj = scene.resolve_object(obj.skin.armature_id)
                if arm_obj is not None and arm_obj.armature is not None:
                    skin_uid = uid()
                    counts["Deformer"] += 1
                    skin_node = objects_node.add(
                        "Deformer",
                        ("L", skin_uid),
                        ("S", _fbx_id_name(f"{obj.name}_Skin", "Deformer")),
                        ("S", "Skin"),
                        block=True,
                    )
                    skin_node.add("Version", ("I", 101))
                    skin_node.add("Link_DeformAcuracy", ("D", 50.0))
                    connect_oo(skin_uid, g_uid)

                    mesh_w_mat = _conv_mat4(compute_world_matrix(scene, oid))

                    pose_uid = uid()
                    counts["Pose"] += 1
                    pose_node = objects_node.add(
                        "Pose",
                        ("L", pose_uid),
                        ("S", _fbx_id_name(f"{obj.name}_BindPose", "Pose")),
                        ("S", "BindPose"),
                        block=True,
                    )
                    pose_node.add("Type", ("S", "BindPose"))
                    pose_node.add("Version", ("I", 100))

                    pose_entries: list[tuple[int, np.ndarray]] = [(m_uid, mesh_w_mat)]

                    for bname, pairs in obj.skin.weights.items():
                        b_uid = bone_model_uids.get((arm_obj.id, bname))
                        if b_uid is None or not pairs:
                            continue
                        cl_uid = uid()
                        counts["Deformer"] += 1
                        cl = objects_node.add(
                            "Deformer",
                            ("L", cl_uid),
                            ("S", _fbx_id_name(f"Cluster_{bname}", "Deformer")),
                            ("S", "Cluster"),
                            block=True,
                        )
                        cl.add("Version", ("I", 100))
                        cl.add("UserData", ("S", ""), ("S", ""))
                        cl.add("Indexes", ("i", [int(vi) for vi, _ in pairs]))
                        cl.add("Weights", ("d", [float(wt) for _, wt in pairs]))

                        b_bind_w = _conv_mat4(bone_bind_world[(arm_obj.id, bname)])
                        # FBX Transform = inverse mesh matrix relative to bone;
                        # TransformLink = bone world bind matrix
                        xf = (np.linalg.inv(b_bind_w) @ mesh_w_mat).T.reshape(-1)
                        xf_link = b_bind_w.T.reshape(-1)
                        cl.add("Transform", ("d", xf))
                        cl.add("TransformLink", ("d", xf_link))
                        connect_oo(cl_uid, skin_uid)
                        connect_oo(b_uid, cl_uid)
                        pose_entries.append((b_uid, b_bind_w))

                    pose_node.add("NbPoseNodes", ("I", len(pose_entries)))
                    for p_uid, p_mat in pose_entries:
                        pn = pose_node.add("PoseNode", block=True)
                        pn.add("Node", ("L", p_uid))
                        pn.add("Matrix", ("d", p_mat.T.reshape(-1)))

            # BlendShape Deformers for Cloth / Soft-body baked vertex_cache
            if obj.vertex_cache:
                valid_frames = sorted(
                    f for f, pts in obj.vertex_cache.items() if len(pts) == len(lm.vertices)
                )
                if len(valid_frames) > 1:
                    # Sample up to 16 shape targets across the baked cache cleanly
                    if len(valid_frames) > 16:
                        sel = [
                            valid_frames[int(round(i * (len(valid_frames) - 1) / 15))]
                            for i in range(16)
                        ]
                        valid_frames = sorted(set(sel))

                    bs_uid = uid()
                    counts["Deformer"] += 1
                    bs_node = objects_node.add(
                        "Deformer",
                        ("L", bs_uid),
                        ("S", _fbx_id_name(f"{obj.name}_ClothSim", "Deformer")),
                        ("S", "BlendShape"),
                        block=True,
                    )
                    bs_node.add("Version", ("I", 100))
                    connect_oo(bs_uid, g_uid)

                    rest_pts = verts
                    chan_list: list[tuple[int, int, int]] = []
                    for kf_idx, frame_num in enumerate(valid_frames):
                        if frame_num == valid_frames[0]:
                            continue
                        target_pts = _conv_pts_arr(
                            np.asarray(obj.vertex_cache[frame_num], dtype=np.float64)
                        )
                        deltas = target_pts - rest_pts
                        changed = np.nonzero(np.linalg.norm(deltas, axis=1) > 1e-6)[0]
                        if len(changed) == 0:
                            continue

                        bsc_uid = uid()
                        shp_uid = uid()
                        counts["Deformer"] += 1
                        counts["Geometry"] += 1

                        bsc = objects_node.add(
                            "Deformer",
                            ("L", bsc_uid),
                            ("S", _fbx_id_name(f"ClothFrame_{frame_num}", "Deformer")),
                            ("S", "BlendShapeChannel"),
                            block=True,
                        )
                        bsc.add("Version", ("I", 100))
                        bsc.add("DeformPercent", ("D", 0.0))
                        bsc.add("FullWeights", ("d", [100.0]))
                        connect_oo(bsc_uid, bs_uid)

                        shp = objects_node.add(
                            "Geometry",
                            ("L", shp_uid),
                            ("S", _fbx_id_name(f"ClothShape_{frame_num}", "Geometry")),
                            ("S", "Shape"),
                            block=True,
                        )
                        shp.add("Version", ("I", 100))
                        shp.add("Indexes", ("i", changed.astype(np.int32)))
                        shp.add("Vertices", ("d", deltas[changed].reshape(-1)))
                        connect_oo(shp_uid, bsc_uid)
                        chan_list.append((bsc_uid, frame_num, kf_idx))

                    if chan_list:
                        morph_channel_uids[oid] = chan_list

    # Connect top-level and parented scene object models to RootNode (0)
    for oid, m_uid in obj_model_uids.items():
        obj = scene.objects[oid]
        if obj.parent_id and obj.parent_id in obj_model_uids:
            connect_oo(m_uid, obj_model_uids[obj.parent_id])
        else:
            connect_oo(m_uid, 0)

    # 3. Animation Stack, Layer & CurveNodes (Transform TRS, Bone Rot/Trans, Cloth DeformPercent)
    has_anim = bool(scene.timeline.tracks) or bool(morph_channel_uids)
    if has_anim:
        stack_uid = uid()
        layer_uid = uid()
        counts["AnimationStack"] += 1
        counts["AnimationLayer"] += 1

        duration_t = int(round(((fe - fs) / fps) * KTIME_ONE_SECOND))
        stack = objects_node.add(
            "AnimationStack",
            ("L", stack_uid),
            ("S", _fbx_id_name(f"{scene.name}_Take001", "AnimStack")),
            ("S", ""),
            block=True,
        )
        sp = stack.add("Properties70", block=True)
        sp.p_prop("LocalStop", "KTime", "Time", "", ("L", duration_t))
        sp.p_prop("ReferenceStop", "KTime", "Time", "", ("L", duration_t))

        objects_node.add(
            "AnimationLayer",
            ("L", layer_uid),
            ("S", _fbx_id_name("BaseLayer", "AnimLayer")),
            ("S", ""),
            block=True,
        )
        connect_oo(layer_uid, stack_uid)

        def emit_curve_node_vec3(
            target_uid: int,
            fbx_prop: str,
            label: str,
            frames: list[int],
            triples: list[tuple[float, float, float]],
        ) -> None:
            cn_uid = uid()
            counts["AnimationCurveNode"] += 1
            cn = objects_node.add(
                "AnimationCurveNode",
                ("L", cn_uid),
                ("S", _fbx_id_name(label, "AnimCurveNode")),
                ("S", ""),
                block=True,
            )
            cnp = cn.add("Properties70", block=True)
            v0 = triples[0]
            cnp.p_prop("d|X", "Number", "", "A", ("D", v0[0]))
            cnp.p_prop("d|Y", "Number", "", "A", ("D", v0[1]))
            cnp.p_prop("d|Z", "Number", "", "A", ("D", v0[2]))
            connect_oo(cn_uid, layer_uid)
            connect_op(cn_uid, target_uid, fbx_prop)

            times_k = [int(round(((f - fs) / fps) * KTIME_ONE_SECOND)) for f in frames]
            for ax_i, ax_lbl in enumerate(("d|X", "d|Y", "d|Z")):
                c_uid = uid()
                counts["AnimationCurve"] += 1
                curve = objects_node.add(
                    "AnimationCurve",
                    ("L", c_uid),
                    ("S", _fbx_id_name("", "AnimCurve")),
                    ("S", ""),
                    block=True,
                )
                curve.add("Default", ("D", triples[0][ax_i]))
                curve.add("KeyVer", ("I", 4008))
                curve.add("KeyTime", ("l", times_k))
                curve.add("KeyValueFloat", ("f", [float(t[ax_i]) for t in triples]))
                curve.add("KeyAttrFlags", ("i", [24836]))
                curve.add("KeyAttrDataFloat", ("f", [0.0, 0.0, 2.0, 0.0]))
                curve.add("KeyAttrRefCount", ("i", [len(frames)]))
                connect_op(c_uid, cn_uid, ax_lbl)

        # Group FCurves per target object & property group
        obj_tracks: dict[str, list] = {}
        for tr in scene.timeline.tracks:
            resolved = scene.resolve_object(tr.target_id)
            if resolved is not None:
                obj_tracks.setdefault(resolved.id, []).append(tr)

        for oid, tr_list in obj_tracks.items():
            obj = scene.objects[oid]
            by_path = {tr.property_path: tr for tr in tr_list}

            # Object transform animations
            if oid in obj_model_uids:
                m_uid = obj_model_uids[oid]
                obj_frames = sorted(
                    {
                        kf.frame
                        for tr in tr_list
                        if tr.property_path.split(".")[0] in ("location", "rotation_deg", "scale")
                        for kf in tr.keyframes
                    }
                )
                if obj_frames:
                    for prefix, fbx_prop, default_vec, conv_fn in (
                        ("location", "Lcl Translation", obj.transform.location, _conv_pos),
                        ("rotation_deg", "Lcl Rotation", obj.transform.rotation_deg, _conv_rot),
                        ("scale", "Lcl Scaling", obj.transform.scale, _conv_scl),
                    ):
                        if any(p.startswith(prefix + ".") for p in by_path):
                            vals: list[tuple[float, float, float]] = []
                            for fnum in obj_frames:
                                raw_v = [
                                    evaluate_fcurve(by_path[f"{prefix}.{ax}"], float(fnum))
                                    if f"{prefix}.{ax}" in by_path
                                    else default_vec[i]
                                    for i, ax in enumerate(("x", "y", "z"))
                                ]
                                fixed = tuple(
                                    float(default_vec[i] if v is None else v)
                                    for i, v in enumerate(raw_v)
                                )
                                vals.append(conv_fn(fixed))
                            emit_curve_node_vec3(
                                m_uid, fbx_prop, prefix[0].upper(), obj_frames, vals
                            )

            # Armature bone animations
            if obj.kind == "armature" and obj.armature is not None:
                bones_map = {b.name: b for b in obj.armature.bones}
                for bone in obj.armature.bones:
                    b_uid = bone_model_uids.get((oid, bone.name))
                    if b_uid is None:
                        continue
                    b_tracks = [
                        tr for tr in tr_list if tr.property_path.startswith(f"bone.{bone.name}.")
                    ]
                    if not b_tracks:
                        continue
                    b_frames = sorted({kf.frame for tr in b_tracks for kf in tr.keyframes})
                    for prop, fbx_prop in (
                        ("rotation_deg", "Lcl Rotation"),
                        ("translation", "Lcl Translation"),
                    ):
                        if any(
                            f"bone.{bone.name}.{prop}.{ax}" in by_path for ax in ("x", "y", "z")
                        ):
                            vals = []
                            p_head = (
                                np.asarray(bones_map[bone.parent].head)
                                if (bone.parent and bone.parent in bones_map)
                                else np.zeros(3)
                            )
                            rest_rel = np.asarray(bone.head) - p_head
                            for fnum in b_frames:
                                if prop == "rotation_deg":
                                    v_raw = tuple(
                                        float(
                                            evaluate_fcurve(
                                                by_path[f"bone.{bone.name}.rotation_deg.{ax}"],
                                                float(fnum),
                                            )
                                            or bone.rotation_deg[i]
                                        )
                                        if f"bone.{bone.name}.rotation_deg.{ax}" in by_path
                                        else float(bone.rotation_deg[i])
                                        for i, ax in enumerate(("x", "y", "z"))
                                    )
                                    vals.append(_conv_rot(v_raw))
                                else:
                                    v_raw = tuple(
                                        float(
                                            rest_rel[i]
                                            + (
                                                evaluate_fcurve(
                                                    by_path[f"bone.{bone.name}.translation.{ax}"],
                                                    float(fnum),
                                                )
                                                or bone.translation[i]
                                            )
                                        )
                                        if f"bone.{bone.name}.translation.{ax}" in by_path
                                        else float(rest_rel[i] + bone.translation[i])
                                        for i, ax in enumerate(("x", "y", "z"))
                                    )
                                    vals.append(_conv_pos(v_raw))
                            emit_curve_node_vec3(
                                b_uid, fbx_prop, bone.name[:1].upper(), b_frames, vals
                            )

        # Cloth BlendShape DeformPercent animation curves
        for _oid, chan_list in morph_channel_uids.items():
            key_frames = [fs] + [fr for _, fr, _ in chan_list]
            times_k = [int(round(((f - fs) / fps) * KTIME_ONE_SECOND)) for f in key_frames]
            for bsc_uid, target_fr, _ in chan_list:
                cn_uid = uid()
                c_uid = uid()
                counts["AnimationCurveNode"] += 1
                counts["AnimationCurve"] += 1
                cn = objects_node.add(
                    "AnimationCurveNode",
                    ("L", cn_uid),
                    ("S", _fbx_id_name("DeformPercent", "AnimCurveNode")),
                    ("S", ""),
                    block=True,
                )
                cn.add("Properties70", block=True).p_prop(
                    "d|DeformPercent", "Number", "", "A", ("D", 0.0)
                )
                connect_oo(cn_uid, layer_uid)
                connect_op(cn_uid, bsc_uid, "DeformPercent")

                weights = [100.0 if kf == target_fr else 0.0 for kf in key_frames]
                curve = objects_node.add(
                    "AnimationCurve",
                    ("L", c_uid),
                    ("S", _fbx_id_name("", "AnimCurve")),
                    ("S", ""),
                    block=True,
                )
                curve.add("Default", ("D", 0.0))
                curve.add("KeyVer", ("I", 4008))
                curve.add("KeyTime", ("l", times_k))
                curve.add("KeyValueFloat", ("f", weights))
                curve.add("KeyAttrFlags", ("i", [24836]))
                curve.add("KeyAttrDataFloat", ("f", [0.0, 0.0, 2.0, 0.0]))
                curve.add("KeyAttrRefCount", ("i", [len(key_frames)]))
                connect_op(c_uid, cn_uid, "d|DeformPercent")

    # 4. Definitions summarizing object types
    definitions = _FBXNode("Definitions", children=[])
    definitions.add("Version", ("I", 100))
    total_defs = 1 + sum(counts.values())
    definitions.add("Count", ("I", total_defs))
    gs_def = definitions.add("ObjectType", ("S", "GlobalSettings"), block=True)
    gs_def.add("Count", ("I", 1))
    for obj_type, cnt in counts.items():
        if cnt > 0:
            ot = definitions.add("ObjectType", ("S", obj_type), block=True)
            ot.add("Count", ("I", cnt))

    root_nodes = [
        header_ext,
        global_settings,
        documents,
        references,
        definitions,
        objects_node,
        connections_node,
    ]

    if ascii_mode:
        lines = [
            f"; FBX {FBX_VERSION / 1000.0:.1f}.0 project file generated by FrFruit",
            "; ----------------------------------------------------",
            "",
        ]
        for rn in root_nodes:
            lines.extend(_node_to_ascii(rn, indent=0))
            lines.append("")
        p.write_text("\n".join(lines), encoding="utf-8")
        fmt_label = "fbx_ascii"
    else:
        raw = _serialize_fbx_binary(root_nodes)
        p.write_bytes(raw)
        fmt_label = "fbx"

    return {
        "path": str(p.resolve()),
        "format": fmt_label,
        "target_engine": target_engine,
        "bytes": p.stat().st_size,
        "counts": {k: v for k, v in counts.items() if v > 0},
    }
