"""Pure-Python Binary FBX (7.1-7.7) importer with meshes, materials, skeletons, and skin weights."""

from __future__ import annotations

import math
import struct
import zlib
from pathlib import Path
from typing import Any

import numpy as np

from frfruit.core.math3d import compose_matrix, euler_deg_to_matrix
from frfruit.core.scene import (
    ArmatureSpec,
    BoneSpec,
    MaterialSpec,
    SceneDocument,
    SceneObject,
    SkinWeightSpec,
)
from frfruit.core.types import Transform
from frfruit.geometry.mesh import MeshData

FBX_BINARY_MAGIC = b"Kaydara FBX Binary  \x00"


def _parse_binary_fbx(data: bytes) -> tuple[int, list[tuple[str, list[Any], list[Any]]]]:
    """Parse Kaydara Binary FBX (v7100..v7700) into nested (name, props, children) nodes."""
    if not data.startswith(FBX_BINARY_MAGIC):
        raise ValueError("Not a valid Kaydara Binary FBX file.")
    version = struct.unpack_from("<I", data, 23)[0]
    is64 = version >= 7500
    sentinel = 25 if is64 else 13
    n_len = len(data)

    def read_prop(p: int) -> tuple[Any, int]:
        t = chr(data[p])
        p += 1
        if t == "Y":
            return struct.unpack_from("<h", data, p)[0], p + 2
        if t == "C":
            return bool(data[p]), p + 1
        if t == "I":
            return struct.unpack_from("<i", data, p)[0], p + 4
        if t == "F":
            return struct.unpack_from("<f", data, p)[0], p + 4
        if t == "D":
            return struct.unpack_from("<d", data, p)[0], p + 8
        if t == "L":
            return struct.unpack_from("<q", data, p)[0], p + 8
        if t in ("S", "R"):
            length = struct.unpack_from("<I", data, p)[0]
            p += 4
            raw = data[p : p + length]
            p += length
            return (raw.decode("utf-8", "replace") if t == "S" else raw), p
        if t in ("f", "d", "l", "i", "b"):
            _arr_len, enc, comp_len = struct.unpack_from("<III", data, p)
            p += 12
            raw = data[p : p + comp_len]
            p += comp_len
            if enc == 1:
                raw = zlib.decompress(raw)
            dtype = {"f": "<f4", "d": "<f8", "l": "<i8", "i": "<i4", "b": "u1"}[t]
            return np.frombuffer(raw, dtype=dtype).copy(), p
        raise ValueError(f"Unsupported FBX property type {t!r} at byte {p - 1}.")

    def read_node(p: int) -> tuple[tuple[str, list[Any], list[Any]] | None, int]:
        if is64:
            end_off, num_props, _prop_len = struct.unpack_from("<QQQ", data, p)
            p += 24
        else:
            end_off, num_props, _prop_len = struct.unpack_from("<III", data, p)
            p += 12
        name_len = data[p]
        p += 1
        if end_off == 0:
            return None, p
        name = data[p : p + name_len].decode("utf-8", "replace")
        p += name_len
        props: list[Any] = []
        for _ in range(num_props):
            val, p = read_prop(p)
            props.append(val)
        children: list[Any] = []
        while p < end_off - sentinel:
            ch, p = read_node(p)
            if ch is None:
                break
            children.append(ch)
        return (name, props, children), int(end_off)

    nodes: list[tuple[str, list[Any], list[Any]]] = []
    pos = 27
    while pos < n_len - sentinel:
        nd, pos = read_node(pos)
        if nd is None:
            break
        nodes.append(nd)
    return version, nodes


def _clean_fbx_name(raw: Any) -> str:
    s = str(raw)
    if "\x00\x01" in s:
        s = s.split("\x00\x01", 1)[0]
    elif "::" in s:
        parts = s.split("::", 1)
        if parts[0] in ("Model", "Geometry", "Material", "Deformer", "SubDeformer", "Pose"):
            s = parts[1]
    return s.strip()


def _extract_p70(children: list[Any]) -> dict[str, list[Any]]:
    out: dict[str, list[Any]] = {}
    for cname, _cprops, cch in children:
        if cname == "Properties70":
            for pname, pprops, _pch in cch:
                if pname == "P" and len(pprops) >= 5:
                    out[str(pprops[0])] = list(pprops[4:])
    return out


def _triangulate_pvi(pvi: np.ndarray) -> tuple[np.ndarray, list[tuple[int, int, int]]]:
    """Convert FBX PolygonVertexIndex into (M, 3) triangles and polygon-vertex index triples."""
    tris: list[tuple[int, int, int]] = []
    pv_tris: list[tuple[int, int, int]] = []
    poly: list[int] = []
    poly_pv: list[int] = []
    for pv_i, raw_idx in enumerate(pvi.tolist()):
        if raw_idx < 0:
            v_idx = (-raw_idx) - 1
            poly.append(v_idx)
            poly_pv.append(pv_i)
            n = len(poly)
            if n >= 3:
                v0, pv0 = poly[0], poly_pv[0]
                for k in range(1, n - 1):
                    tris.append((v0, poly[k], poly[k + 1]))
                    pv_tris.append((pv0, poly_pv[k], poly_pv[k + 1]))
            poly.clear()
            poly_pv.clear()
        else:
            poly.append(raw_idx)
            poly_pv.append(pv_i)
    if not tris:
        return np.zeros((0, 3), dtype=np.int64), []
    return np.asarray(tris, dtype=np.int64), pv_tris


def _extract_uvs(
    geom_children: list[Any],
    n_verts: int,
    pvi: np.ndarray,
) -> np.ndarray | None:
    """Extract per-vertex UVs (best-effort collapse of ByPolygonVertex onto control points)."""
    for cname, _cprops, cch in geom_children:
        if cname != "LayerElementUV":
            continue
        uv_arr: np.ndarray | None = None
        uv_idx: np.ndarray | None = None
        mapping = ""
        ref_mode = ""
        for sub_name, sub_props, _ in cch:
            if sub_name == "UV" and sub_props:
                uv_arr = np.asarray(sub_props[0], dtype=np.float64).reshape(-1, 2)
            elif sub_name == "UVIndex" and sub_props:
                uv_idx = np.asarray(sub_props[0], dtype=np.int64)
            elif sub_name == "MappingInformationType" and sub_props:
                mapping = str(sub_props[0])
            elif sub_name == "ReferenceInformationType" and sub_props:
                ref_mode = str(sub_props[0])
        if uv_arr is None or len(uv_arr) == 0:
            continue
        if mapping == "ByVertice" and len(uv_arr) == n_verts:
            return uv_arr
        if mapping == "ByPolygonVertex":
            out_uv = np.zeros((n_verts, 2), dtype=np.float64)
            seen = np.zeros(n_verts, dtype=bool)
            for pv_i, raw_v in enumerate(pvi.tolist()):
                vi = (-raw_v - 1) if raw_v < 0 else raw_v
                if 0 <= vi < n_verts and not seen[vi]:
                    if ref_mode == "IndexToDirect" and uv_idx is not None and pv_i < len(uv_idx):
                        ui = int(uv_idx[pv_i])
                        if 0 <= ui < len(uv_arr):
                            out_uv[vi] = uv_arr[ui]
                            seen[vi] = True
                    elif pv_i < len(uv_arr):
                        out_uv[vi] = uv_arr[pv_i]
                        seen[vi] = True
            return out_uv
    return None


def _fbx_local_matrix(p70: dict[str, list[Any]]) -> np.ndarray:
    """Evaluate an FBX node local matrix T * R_pre * R_lcl * S (column-vector convention)."""
    t = p70.get("Lcl Translation", [0.0, 0.0, 0.0])
    r_pre = p70.get("PreRotation", [0.0, 0.0, 0.0])
    r_lcl = p70.get("Lcl Rotation", [0.0, 0.0, 0.0])
    s = p70.get("Lcl Scaling", [1.0, 1.0, 1.0])

    m_t = np.eye(4, dtype=np.float64)
    m_t[:3, 3] = [float(t[0]), float(t[1]), float(t[2])]

    m_pre = np.eye(4, dtype=np.float64)
    m_pre[:3, :3] = euler_deg_to_matrix(float(r_pre[0]), float(r_pre[1]), float(r_pre[2]))

    m_rot = np.eye(4, dtype=np.float64)
    m_rot[:3, :3] = euler_deg_to_matrix(float(r_lcl[0]), float(r_lcl[1]), float(r_lcl[2]))

    m_s = np.eye(4, dtype=np.float64)
    m_s[0, 0], m_s[1, 1], m_s[2, 2] = float(s[0]), float(s[1]), float(s[2])

    return m_t @ m_pre @ m_rot @ m_s


def import_fbx_file(
    scene: SceneDocument,
    file_path: str | Path,
    name_prefix: str | None = None,
    collection_id: str = "default",
    *,
    convert_yup: bool | None = None,
    scale_override: float | None = None,
    ground_to_z0: bool = True,
) -> list[SceneObject]:
    """Import Binary FBX (7.1-7.7) meshes, materials, armature bones, and skin weights."""
    p = Path(file_path)
    if not p.exists():
        raise FileNotFoundError(f"FBX file not found: {p}")

    raw = p.read_bytes()
    _version, top_nodes = _parse_binary_fbx(raw)
    top_map = {n: (props, ch) for n, props, ch in top_nodes}

    # 1. GlobalSettings: detect UpAxis and unit scale
    gs_p70 = _extract_p70(top_map["GlobalSettings"][1]) if "GlobalSettings" in top_map else {}
    up_axis = int(gs_p70.get("UpAxis", [1])[0])  # 1 = Y-up, 2 = Z-up
    unit_scale_cm = float(gs_p70.get("UnitScaleFactor", [1.0])[0])
    if convert_yup is None:
        convert_yup = up_axis == 1

    # 2. Index Objects and Connections
    objects_raw = top_map.get("Objects", ([], []))[1]
    conns_raw = top_map.get("Connections", ([], []))[1]

    geoms: dict[int, tuple[str, list[Any]]] = {}
    mesh_models: dict[int, tuple[str, dict[str, list[Any]]]] = {}
    limb_models: dict[int, tuple[str, dict[str, list[Any]]]] = {}
    materials_fbx: dict[int, tuple[str, dict[str, list[Any]]]] = {}
    skins: dict[int, str] = {}
    clusters: dict[int, dict[str, Any]] = {}
    bind_pose_mats: dict[int, np.ndarray] = {}

    for oname, oprops, och in objects_raw:
        if not oprops:
            continue
        oid = int(oprops[0])
        clean = _clean_fbx_name(oprops[1]) if len(oprops) > 1 else f"{oname}_{oid}"
        subtype = str(oprops[2]) if len(oprops) > 2 else ""

        if oname == "Geometry" and subtype == "Mesh":
            geoms[oid] = (clean, och)
        elif oname == "Model" and subtype == "Mesh":
            mesh_models[oid] = (clean, _extract_p70(och))
        elif oname == "Model" and subtype in ("LimbNode", "Limb", "Root", "Null"):
            # Capture LimbNodes (and any Null/Root parents in skeleton hierarchy)
            limb_models[oid] = (clean, _extract_p70(och))
        elif oname == "Material":
            materials_fbx[oid] = (clean, _extract_p70(och))
        elif oname == "Deformer" and subtype == "Skin":
            skins[oid] = clean
        elif oname == "Deformer" and subtype == "Cluster":
            cl_info: dict[str, Any] = {"name": clean}
            for cn, cp, _ in och:
                if cn == "Indexes" and cp:
                    cl_info["indexes"] = np.asarray(cp[0], dtype=np.int64)
                elif cn == "Weights" and cp:
                    cl_info["weights"] = np.asarray(cp[0], dtype=np.float64)
                elif cn == "Transform" and cp:
                    cl_info["transform"] = np.asarray(cp[0], dtype=np.float64).reshape(4, 4).T
                elif cn == "TransformLink" and cp:
                    cl_info["transform_link"] = np.asarray(cp[0], dtype=np.float64).reshape(4, 4).T
            clusters[oid] = cl_info
        elif oname == "Pose" and subtype == "BindPose":
            for cn, _cp, cch in och:
                if cn == "PoseNode":
                    nid = None
                    mat = None
                    for sn, sp, _ in cch:
                        if sn == "Node" and sp:
                            nid = int(sp[0])
                        elif sn == "Matrix" and sp:
                            mat = np.asarray(sp[0], dtype=np.float64).reshape(4, 4).T
                    if nid is not None and mat is not None:
                        bind_pose_mats[nid] = mat

    # Keep only actual LimbNodes (and ancestors referenced by LimbNodes)
    limb_only_ids: set[int] = set()
    for oname, oprops, _ in objects_raw:
        if oname == "Model" and len(oprops) > 2 and str(oprops[2]) in ("LimbNode", "Limb"):
            limb_only_ids.add(int(oprops[0]))

    # Parse OO connections: keep bone-to-bone hierarchy separate from deformer/cluster links
    parent_of: dict[int, int] = {}
    children_of: dict[int, list[int]] = {}
    for _cname, cprops, _ in conns_raw:
        if len(cprops) >= 3 and cprops[0] == "OO":
            child_id, parent_id = int(cprops[1]), int(cprops[2])
            if child_id in limb_models and (parent_id in limb_models or parent_id == 0):
                parent_of[child_id] = parent_id
                children_of.setdefault(parent_id, []).append(child_id)

    # Map relationships
    geom_to_model: dict[int, int] = {}
    model_to_mat: dict[int, int] = {}
    skin_to_geom: dict[int, int] = {}
    cluster_to_skin: dict[int, int] = {}
    cluster_to_bone: dict[int, int] = {}

    for _cname, cprops, _ in conns_raw:
        if len(cprops) < 3 or cprops[0] != "OO":
            continue
        cid, pid = int(cprops[1]), int(cprops[2])
        if cid in geoms and pid in mesh_models:
            geom_to_model[cid] = pid
        elif cid in materials_fbx and pid in mesh_models:
            model_to_mat[pid] = cid
        elif cid in skins and pid in geoms:
            skin_to_geom[cid] = pid
        elif cid in clusters and pid in skins:
            cluster_to_skin[cid] = pid
        elif cid in limb_models and pid in clusters:
            cluster_to_bone[pid] = cid

    # 3. Auto-determine scale (cm -> m if UnitScaleFactor == 1.0 and geometry is > 10 units tall)
    all_raw_verts: list[np.ndarray] = []
    parsed_geoms: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray | None]] = {}
    for gid, (_gname, gch) in geoms.items():
        v_raw = next((c[1][0] for c in gch if c[0] == "Vertices" and c[1]), None)
        pvi_raw = next((c[1][0] for c in gch if c[0] == "PolygonVertexIndex" and c[1]), None)
        if v_raw is None or pvi_raw is None:
            continue
        verts = np.asarray(v_raw, dtype=np.float64).reshape(-1, 3)
        pvi = np.asarray(pvi_raw, dtype=np.int64)
        faces, _ = _triangulate_pvi(pvi)
        uvs = _extract_uvs(gch, len(verts), pvi)
        parsed_geoms[gid] = (verts, faces, uvs)
        if len(verts) > 0:
            all_raw_verts.append(verts)

    if scale_override is not None:
        unit_scale = float(scale_override)
    else:
        if all_raw_verts:
            stacked = np.vstack(all_raw_verts)
            span = float(np.max(stacked.max(axis=0) - stacked.min(axis=0)))
            if span > 15.0:
                unit_scale = 0.01 * unit_scale_cm
            else:
                unit_scale = 1.0
        else:
            unit_scale = 0.01 * unit_scale_cm

    rot_conv = compose_matrix(rotation_deg=(90.0, 0.0, 0.0))[:3, :3] if convert_yup else np.eye(3)

    def conv_pts(pts: np.ndarray) -> np.ndarray:
        return (pts * unit_scale) @ rot_conv.T

    def conv_pt(pt: np.ndarray) -> np.ndarray:
        return (rot_conv @ (pt * unit_scale)).astype(np.float64)

    # Compute ground shift so lowest mesh vertex rests at Z = 0.0
    z_shift = 0.0
    if ground_to_z0 and all_raw_verts:
        min_z = min(float(conv_pts(v)[:, 2].min()) for v in all_raw_verts)
        z_shift = -min_z

    def conv_pt_grounded(pt: np.ndarray) -> tuple[float, float, float]:
        p3 = conv_pt(pt)
        return (float(p3[0]), float(p3[1]), float(p3[2] + z_shift))

    # 4. Create Materials
    fbx_mat_to_frf: dict[int, str] = {}
    for mid, (mname, p70) in materials_fbx.items():
        mat_id = scene.next_id(f"mat_{mname}")
        diff = p70.get("DiffuseColor", [0.72, 0.72, 0.75])
        spec = p70.get("SpecularColor", [0.25, 0.25, 0.25])
        shin = float(p70.get("Shininess", [16.0])[0]) if "Shininess" in p70 else 16.0
        spec_mag = float(np.mean([float(x) for x in spec[:3]]))
        roughness = max(0.18, min(0.85, 0.72 - 0.10 * math.log2(max(1.0, shin))))
        metallic = 0.28 if spec_mag > 0.35 else 0.12
        base_color = (
            max(0.0, min(1.0, float(diff[0]))),
            max(0.0, min(1.0, float(diff[1]))),
            max(0.0, min(1.0, float(diff[2]))),
            1.0,
        )
        scene.materials[mat_id] = MaterialSpec(
            id=mat_id,
            name=mname,
            base_color=base_color,
            metallic=metallic,
            roughness=roughness,
        )
        fbx_mat_to_frf[mid] = mat_id

    created: list[SceneObject] = []
    coll = collection_id if collection_id in scene.collections else "default"

    # 5. Build Armature from LimbNodes
    armature_obj: SceneObject | None = None
    bone_id_to_name: dict[int, str] = {}
    if limb_only_ids:
        # Compute world matrices of all limb nodes (prefer TransformLink -> BindPose -> FK)
        bone_world_mats: dict[int, np.ndarray] = {}
        for cl_id, cl_info in clusters.items():
            bid = cluster_to_bone.get(cl_id)
            if bid is not None and "transform_link" in cl_info:
                bone_world_mats[bid] = cl_info["transform_link"]
        for bid, bmat in bind_pose_mats.items():
            if bid in limb_only_ids and bid not in bone_world_mats:
                bone_world_mats[bid] = bmat

        def resolve_world_mat(nid: int, visiting: set[int]) -> np.ndarray:
            if nid in bone_world_mats:
                return bone_world_mats[nid]
            if nid in visiting or nid not in limb_models:
                return np.eye(4, dtype=np.float64)
            visiting.add(nid)
            pid = parent_of.get(nid, 0)
            p_mat = resolve_world_mat(pid, visiting) if pid in limb_models else np.eye(4)
            loc_m = _fbx_local_matrix(limb_models[nid][1])
            w_mat = p_mat @ loc_m
            bone_world_mats[nid] = w_mat
            visiting.remove(nid)
            return w_mat

        for bid in limb_only_ids:
            resolve_world_mat(bid, set())

        # Topological order (parents before children)
        ordered_bids: list[int] = []
        visited_b: set[int] = set()

        def visit_bone(bid: int) -> None:
            if bid in visited_b or bid not in limb_only_ids:
                return
            pid = parent_of.get(bid, 0)
            if pid in limb_only_ids:
                visit_bone(pid)
            visited_b.add(bid)
            ordered_bids.append(bid)

        for bid in sorted(limb_only_ids):
            visit_bone(bid)

        used_bnames: set[str] = set()
        for bid in ordered_bids:
            raw_bname = limb_models[bid][0]
            bname = raw_bname
            suffix = 2
            while bname in used_bnames:
                bname = f"{raw_bname}_{suffix}"
                suffix += 1
            used_bnames.add(bname)
            bone_id_to_name[bid] = bname

        head_by_bid: dict[int, tuple[float, float, float]] = {
            bid: conv_pt_grounded(bone_world_mats[bid][:3, 3]) for bid in ordered_bids
        }

        bone_specs: list[BoneSpec] = []
        for bid in ordered_bids:
            bname = bone_id_to_name[bid]
            pid = parent_of.get(bid, 0)
            parent_name = bone_id_to_name.get(pid) if pid in bone_id_to_name else None
            head_pos = np.asarray(head_by_bid[bid], dtype=np.float64)

            child_bids = [c for c in children_of.get(bid, []) if c in head_by_bid]
            if child_bids:
                child_pts = np.asarray([head_by_bid[c] for c in child_bids], dtype=np.float64)
                tail_pos = child_pts.mean(axis=0)
            elif pid in head_by_bid:
                p_head = np.asarray(head_by_bid[pid], dtype=np.float64)
                delta = head_pos - p_head
                dlen = float(np.linalg.norm(delta))
                ext = (
                    (delta / dlen) * min(0.08, max(0.03, dlen * 0.5))
                    if dlen > 1e-5
                    else np.array([0.0, 0.0, 0.05])
                )
                tail_pos = head_pos + ext
            else:
                tail_pos = head_pos + np.array([0.0, 0.0, 0.12], dtype=np.float64)

            if float(np.linalg.norm(tail_pos - head_pos)) < 1e-4:
                tail_pos = head_pos + np.array([0.0, 0.0, 0.04], dtype=np.float64)

            bone_specs.append(
                BoneSpec(
                    name=bname,
                    parent=parent_name,
                    head=(float(head_pos[0]), float(head_pos[1]), float(head_pos[2])),
                    tail=(float(tail_pos[0]), float(tail_pos[1]), float(tail_pos[2])),
                )
            )

        arm_name = scene.unique_name(
            f"{name_prefix}_Armature" if name_prefix else f"{p.stem}_Armature"
        )
        arm_id = scene.next_id(arm_name.lower())
        armature_obj = SceneObject(
            id=arm_id,
            name=arm_name,
            kind="armature",
            collection_id=coll,
            transform=Transform(),
            armature=ArmatureSpec(bones=bone_specs),
        )
        scene.objects[arm_id] = armature_obj
        created.append(armature_obj)

    # 6. Build Skinned Mesh Objects
    for gid, (verts_raw, faces, uvs) in parsed_geoms.items():
        if len(verts_raw) == 0 or len(faces) == 0:
            continue
        mid_model = geom_to_model.get(gid)
        mname = mesh_models[mid_model][0] if mid_model in mesh_models else geoms[gid][0]
        obj_name = scene.unique_name(f"{name_prefix}_{mname}" if name_prefix else mname)
        oid = scene.next_id(obj_name.lower())

        verts_conv = conv_pts(verts_raw)
        verts_conv[:, 2] += z_shift
        md = MeshData(vertices=verts_conv, faces=faces, uvs=uvs)

        mat_id = (
            fbx_mat_to_frf.get(model_to_mat.get(mid_model, -1)) if mid_model is not None else None
        )

        # Assemble exact FBX skin weights from clusters connected to this geometry's skin
        skin_spec: SkinWeightSpec | None = None
        if armature_obj is not None:
            geom_skins = [sid for sid, sgid in skin_to_geom.items() if sgid == gid]
            if geom_skins:
                n_v = len(verts_conv)
                bone_weights_accum: dict[str, dict[int, float]] = {}
                v_totals = np.zeros(n_v, dtype=np.float64)
                for cl_id, sid in cluster_to_skin.items():
                    if sid not in geom_skins:
                        continue
                    bid = cluster_to_bone.get(cl_id)
                    if bid is None or bid not in bone_id_to_name:
                        continue
                    bname = bone_id_to_name[bid]
                    cl_info = clusters[cl_id]
                    idxs = cl_info.get("indexes")
                    wgts = cl_info.get("weights")
                    if idxs is None or wgts is None:
                        continue
                    bmap = bone_weights_accum.setdefault(bname, {})
                    for vi_raw, w_raw in zip(idxs.tolist(), wgts.tolist(), strict=False):
                        vi = int(vi_raw)
                        w = float(w_raw)
                        if 0 <= vi < n_v and w > 1e-6:
                            bmap[vi] = bmap.get(vi, 0.0) + w
                            v_totals[vi] += w

                if bone_weights_accum:
                    norm_weights: dict[str, list[tuple[int, float]]] = {}
                    for bname, bmap in bone_weights_accum.items():
                        pairs = [
                            (vi, round(w / max(1e-12, float(v_totals[vi])), 6))
                            for vi, w in sorted(bmap.items())
                            if w / max(1e-12, float(v_totals[vi])) > 1e-5
                        ]
                        if pairs:
                            norm_weights[bname] = pairs
                    skin_spec = SkinWeightSpec(armature_id=armature_obj.id, weights=norm_weights)

        obj = SceneObject(
            id=oid,
            name=obj_name,
            kind="mesh",
            collection_id=coll,
            material_id=mat_id,
            mesh=md.to_payload(),
            skin=skin_spec,
        )
        scene.objects[oid] = obj
        created.append(obj)

    return created
