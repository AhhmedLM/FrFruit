"""Co-hosted Starlette backend bridging FrFruit Studio UI, WebSockets, REST API, and Live MCP."""

from __future__ import annotations

import asyncio
import contextlib
import json
import re
import sys
import threading
import time
import urllib.parse
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
from mcp.server import MCPServer
from mcp.server.mcpserver.exceptions import ToolError
from mcp.server.transport_security import TransportSecuritySettings
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import FileResponse, JSONResponse, Response
from starlette.routing import Mount, Route, WebSocketRoute
from starlette.staticfiles import StaticFiles
from starlette.types import ASGIApp, Message, Receive, Scope, Send
from starlette.websockets import WebSocket, WebSocketDisconnect

from frfruit.animation.evaluator import apply_animation_frame
from frfruit.character import (
    CHARACTER_PRESETS,
    SEMANTIC_RBF_ANCHORS,
    SKIN_TONE_PRESETS,
    BodySliders,
    CharacterMaterialParams,
    CharacterSurface,
    ExpressionSliders,
    FaceSliders,
    add_rbf_sculpt_node,
    apply_character_delta,
    blend_characters,
    build_or_update_character,
    compute_character_delta,
    extract_custom_morph_from_mesh_edit,
    extract_rbf_delta_from_mesh_edit,
    get_character_preset,
)
from frfruit.character.schema import FACE_REGION_FIELDS
from frfruit.core.math3d import transform_points
from frfruit.core.scene import (
    ClothSpec,
    LightSpec,
    MaterialSpec,
    PostProcessConfig,
    RigidBodySpec,
    SceneObject,
)
from frfruit.core.types import SelectionQuery, Transform
from frfruit.geometry.edit_ops import (
    inspect_mesh_elements,
    resolve_edge_selection,
    resolve_face_selection,
    resolve_vertex_selection,
    run_edit_operation,
)
from frfruit.geometry.mesh import GeometryError, MeshData
from frfruit.geometry.primitives import create_primitive_mesh
from frfruit.geometry.spatial import (
    compute_scene_bounds,
    compute_world_matrix,
    evaluate_object_mesh,
    evaluate_scene_meshes,
)
from frfruit.io.exporters import export_scene_file
from frfruit.io.project import load_project, save_project
from frfruit.mcp_server.context import FrFruitSession
from frfruit.mcp_server.server import create_server
from frfruit.mcp_server.tools_render import _apply_face_material_selection
from frfruit.mcp_server.tools_sim_fx import _POSTPROCESS_PRESETS
from frfruit.physics.solver import bake_scene_physics, clear_physics_caches
from frfruit.render.video import render_video
from frfruit.render.viewport import render_single_view
from frfruit.textures.fetcher import (
    build_material_from_maps,
    download_pbr_texture_pack,
    generate_procedural_pbr_maps,
    search_online_textures,
)
from frfruit.textures.loader import compute_triplanar_uvs


def _resolve_static_dir() -> Path:
    meipass = getattr(sys, "_MEIPASS", None)
    if meipass:
        cand = Path(meipass) / "frfruit" / "desktop" / "static"
        if cand.is_dir():
            return cand
    return Path(__file__).resolve().parent / "static"


def _error_json(msg: str, status_code: int = 400) -> JSONResponse:
    return JSONResponse({"status": "error", "error": str(msg)}, status_code=status_code)


def _serialize_state(
    sess: FrFruitSession,
    mcp_srv: MCPServer,
    mcp_clients_fn: Callable[[], list[dict[str, Any]]] | None = None,
) -> dict[str, Any]:
    with sess.lock:
        sc = sess.scene
        evaluated = evaluate_scene_meshes(sc)
        bounds = compute_scene_bounds(evaluated)

        objects_out: list[dict[str, Any]] = []
        total_verts = 0
        total_faces = 0
        for oid, obj in sc.objects.items():
            ev = evaluated.get(oid)
            v_count = len(ev.world_mesh.vertices) if ev else 0
            f_count = len(ev.world_mesh.faces) if ev else 0
            total_verts += v_count
            total_faces += f_count

            entry: dict[str, Any] = {
                "id": obj.id,
                "name": obj.name,
                "kind": obj.kind,
                "visible": sc.is_object_visible(obj),
                "location": [round(float(x), 4) for x in obj.transform.location],
                "rotation_deg": [round(float(x), 3) for x in obj.transform.rotation_deg],
                "scale": [round(float(x), 4) for x in obj.transform.scale],
                "material_id": obj.material_id,
                "face_material_ids": {str(k): v for k, v in obj.face_material_ids.items()},
                "face_material_count": len(obj.face_material_ids),
                "parent_id": obj.parent_id,
                "collection_id": obj.collection_id,
                "vertex_count": v_count,
                "face_count": f_count,
                "bounds": (
                    {
                        "min": [round(float(x), 4) for x in ev.bounds.min_corner],
                        "max": [round(float(x), 4) for x in ev.bounds.max_corner],
                        "center": [round(float(x), 4) for x in ev.bounds.center],
                        "size": [round(float(x), 4) for x in ev.bounds.size],
                    }
                    if ev is not None
                    else None
                ),
                "rigid_body": obj.rigid_body.model_dump() if obj.rigid_body else None,
                "cloth": obj.cloth.model_dump() if obj.cloth else None,
                "cloth_cached_frames": len(obj.vertex_cache),
                "skin_armature_id": obj.skin.armature_id if obj.skin else None,
                "character": obj.character.model_dump() if obj.character else None,
                "bones": (
                    [b.model_dump() for b in obj.armature.bones]
                    if obj.armature is not None
                    else None
                ),
            }
            if obj.light is not None:
                entry["light"] = obj.light.model_dump()
            if obj.camera is not None:
                entry["camera"] = obj.camera.model_dump()
            objects_out.append(entry)

        materials_out = {mid: mat.model_dump() for mid, mat in sc.materials.items()}
        tools_count = len(mcp_srv._tool_manager.list_tools())
        clients = mcp_clients_fn() if mcp_clients_fn is not None else []
        return {
            "scene_name": sc.name,
            "workspace_root": str(sess.workspace_root),
            "render_engine": sc.render_engine,
            "post_process": sc.post_process.model_dump(),
            "active_camera_id": sc.active_camera_id,
            "world": sc.world.model_dump(),
            "timeline": {
                "fps": sc.timeline.fps,
                "frame_start": sc.timeline.frame_start,
                "frame_end": sc.timeline.frame_end,
                "current_frame": sc.timeline.current_frame,
                "track_count": len(sc.timeline.tracks),
            },
            "history": sess.history.status(),
            "stats": {
                "object_count": len(sc.objects),
                "mesh_count": len(evaluated),
                "total_vertices": total_verts,
                "total_faces": total_faces,
                "material_count": len(sc.materials),
            },
            "scene_bounds": {
                "min": [float(x) for x in bounds.min_corner],
                "max": [float(x) for x in bounds.max_corner],
                "center": [float(x) for x in bounds.center],
                "size": [float(x) for x in bounds.size],
            },
            "objects": objects_out,
            "materials": materials_out,
            "mcp_tools_count": tools_count,
            "mcp_clients": clients,
            "mcp_clients_count": len(clients),
            "audit_log": list(sess.audit_log[-40:]),
        }


def _material_to_gl_dict(mat: MaterialSpec | None) -> dict[str, Any]:
    if mat is None:
        return {
            "id": "default",
            "name": "Default",
            "base_color": [0.76, 0.78, 0.82, 1.0],
            "metallic": 0.05,
            "roughness": 0.45,
            "emission": [0.0, 0.0, 0.0],
            "alpha": 1.0,
            "uv_scale": [1.0, 1.0],
            "has_albedo_map": False,
            "albedo_sig": "",
        }
    return {
        "id": mat.id,
        "name": mat.name,
        "base_color": [float(x) for x in mat.base_color],
        "metallic": float(mat.metallic),
        "roughness": float(mat.roughness),
        "emission": [float(mat.emission_color[i] * mat.emission_strength) for i in range(3)],
        "alpha": float(mat.alpha),
        "uv_scale": [float(mat.uv_scale[0]), float(mat.uv_scale[1])],
        "has_albedo_map": bool(mat.albedo_map and Path(mat.albedo_map).is_file()),
        "albedo_sig": Path(mat.albedo_map).name if mat.albedo_map else "",
    }


def _serialize_geometry(sess: FrFruitSession) -> dict[str, Any]:
    with sess.lock:
        sc = sess.scene
        evaluated = evaluate_scene_meshes(sc)
        bounds = compute_scene_bounds(evaluated)

        meshes_payload: list[dict[str, Any]] = []
        for oid, ev in evaluated.items():
            wm = ev.world_mesh
            if wm.is_empty:
                continue
            norms = wm.vertex_normals()
            if wm.uvs is not None and len(wm.uvs) == len(wm.vertices):
                uvs = wm.uvs
            else:
                uvs = compute_triplanar_uvs(wm.vertices, norms)

            bounds_dict = {
                "min": [float(x) for x in ev.bounds.min_corner],
                "max": [float(x) for x in ev.bounds.max_corner],
                "center": [float(x) for x in ev.bounds.center],
            }

            n_faces = len(wm.faces)
            face_groups: dict[str | None, list[int]] = {}
            if ev.obj.face_material_ids:
                for fi in range(n_faces):
                    mid = ev.obj.face_material_ids.get(fi, ev.obj.material_id)
                    if mid not in sc.materials:
                        mid = ev.obj.material_id if (ev.obj.material_id in sc.materials) else None
                    face_groups.setdefault(mid, []).append(fi)
            else:
                eff = ev.obj.material_id if (ev.obj.material_id in sc.materials) else None
                face_groups[eff] = list(range(n_faces))

            if len(face_groups) <= 1:
                mid = next(iter(face_groups.keys()))
                mat = sc.materials.get(mid) if mid else None
                meshes_payload.append(
                    {
                        "id": oid,
                        "name": ev.obj.name,
                        "positions": [round(float(v), 5) for v in wm.vertices.reshape(-1)],
                        "normals": [round(float(n), 4) for n in norms.reshape(-1)],
                        "uvs": [round(float(u), 4) for u in uvs.reshape(-1)],
                        "indices": [int(i) for i in wm.faces.reshape(-1)],
                        "bounds": bounds_dict,
                        "material": _material_to_gl_dict(mat),
                    }
                )
            else:
                for slot_idx, (mid, f_idxs) in enumerate(face_groups.items()):
                    mat = sc.materials.get(mid) if mid else None
                    sub_faces = wm.faces[f_idxs]
                    uniq_v, inv = np.unique(sub_faces.reshape(-1), return_inverse=True)
                    meshes_payload.append(
                        {
                            "id": oid,
                            "submesh_slot": slot_idx,
                            "name": ev.obj.name,
                            "positions": [
                                round(float(v), 5) for v in wm.vertices[uniq_v].reshape(-1)
                            ],
                            "normals": [round(float(n), 4) for n in norms[uniq_v].reshape(-1)],
                            "uvs": [round(float(u), 4) for u in uvs[uniq_v].reshape(-1)],
                            "indices": [int(i) for i in inv.tolist()],
                            "bounds": bounds_dict,
                            "material": _material_to_gl_dict(mat),
                        }
                    )

        armatures_payload: list[dict[str, Any]] = []
        for oid, obj in sc.objects.items():
            if obj.kind == "armature" and obj.armature is not None and sc.is_object_visible(obj):
                from frfruit.animation.armature import compute_armature_matrices
                from frfruit.geometry.spatial import compute_world_matrix

                arm_w = compute_world_matrix(sc, oid)
                _, pose_mats, _ = compute_armature_matrices(obj.armature)
                bones_seg: list[dict[str, Any]] = []
                for b in obj.armature.bones:
                    pm = arm_w @ pose_mats[b.name]
                    head_w = pm[:3, 3]
                    tail_rel = [float(b.tail[i] - b.head[i]) for i in range(3)]
                    tail_w = (pm @ [tail_rel[0], tail_rel[1], tail_rel[2], 1.0])[:3]
                    bones_seg.append(
                        {
                            "name": b.name,
                            "head": [round(float(x), 4) for x in head_w],
                            "tail": [round(float(x), 4) for x in tail_w],
                        }
                    )
                armatures_payload.append({"id": oid, "name": obj.name, "bones": bones_seg})

        lights_payload = [
            {
                "id": o.id,
                "name": o.name,
                "kind": o.light.kind,
                "location": [float(x) for x in o.transform.location],
                "direction": (
                    [float(x) for x in o.light.direction]
                    if o.light.direction is not None
                    else (
                        [-float(x) for x in o.transform.location]
                        if any(abs(float(x)) > 1e-4 for x in o.transform.location)
                        else [-0.55, 0.45, -0.7]
                    )
                ),
                "color": [float(x) for x in o.light.color],
                "intensity": float(o.light.intensity),
            }
            for o in sc.objects.values()
            if o.kind == "light" and o.light is not None and sc.is_object_visible(o)
        ]

        return {
            "frame": sc.timeline.current_frame,
            "scene_bounds": {
                "min": [float(x) for x in bounds.min_corner],
                "max": [float(x) for x in bounds.max_corner],
                "center": [float(x) for x in bounds.center],
                "size": [float(x) for x in bounds.size],
            },
            "meshes": meshes_payload,
            "armatures": armatures_payload,
            "lights": lights_payload,
            "world": sc.world.model_dump(),
        }


def _populate_demo_showcase(sess: FrFruitSession, preset: str) -> str:
    with sess.lock:
        if preset == "physics_cloth":
            sess.reset("PhysicsClothShowcase")
            sc = sess.scene
            sc.timeline.fps = 24
            sc.timeline.frame_start = 1
            sc.timeline.frame_end = 36
            sc.render_engine = "software"

            sphere = create_primitive_mesh("uv_sphere", radius=0.85, segments=28)
            mat_pedestal = build_material_from_maps(
                "mat_bronze",
                "BrushedBronze",
                {},
                base_color=(0.76, 0.56, 0.28, 1.0),
                metallic=0.88,
                roughness=0.25,
            )
            sc.materials[mat_pedestal.id] = mat_pedestal
            sc.objects["collider_orb"] = SceneObject(
                id="collider_orb",
                name="BronzeSphereCollider",
                kind="mesh",
                transform=Transform(location=(0.0, 0.0, 0.95)),
                mesh=sphere.to_payload(),
                material_id=mat_pedestal.id,
                rigid_body=RigidBodySpec(enabled=True, body_type="static", collider_shape="sphere"),
            )

            cloth_mesh = create_primitive_mesh("plane", size=2.4, segments=14)
            tex_dir = sess.workspace_root / "textures" / "procedural_fabric"
            manifest = generate_procedural_pbr_maps("fabric", out_dir=tex_dir, size=256)
            mat_cloth = build_material_from_maps(
                "mat_cloth",
                "VelvetBanner",
                manifest["maps"],
                base_color=(0.98, 0.96, 0.96, 1.0),
                metallic=0.04,
                roughness=0.72,
                texture_source="procedural:fabric",
            )
            sc.materials[mat_cloth.id] = mat_cloth
            sc.objects["cloth_drape"] = SceneObject(
                id="cloth_drape",
                name="DrapedClothBanner",
                kind="mesh",
                transform=Transform(location=(0.0, 0.0, 2.15)),
                mesh=cloth_mesh.to_payload(),
                material_id=mat_cloth.id,
                cloth=ClothSpec(
                    enabled=True,
                    pin_mode="none",
                    wind=(1.4, 0.3, 0.0),
                    stiffness=0.90,
                    bend_stiffness=0.08,
                    substeps=8,
                ),
            )
            rb_ball = create_primitive_mesh("icosphere", radius=0.28, segments=16)
            mat_red = build_material_from_maps(
                "mat_ruby",
                "AnodizedRuby",
                {},
                base_color=(0.82, 0.20, 0.16, 1.0),
                metallic=0.55,
                roughness=0.28,
            )
            sc.materials[mat_red.id] = mat_red
            sc.objects["rb_drop"] = SceneObject(
                id="rb_drop",
                name="DynamicDropBall",
                kind="mesh",
                transform=Transform(location=(-1.45, -0.5, 2.6)),
                mesh=rb_ball.to_payload(),
                material_id=mat_red.id,
                rigid_body=RigidBodySpec(
                    enabled=True,
                    body_type="dynamic",
                    restitution=0.65,
                    collider_shape="sphere",
                ),
            )
        else:
            sess.reset("PBRMaterialsShowcase")
            sc = sess.scene
            sc.post_process = PostProcessConfig(**_POSTPROCESS_PRESETS["cinematic"])

            ped_mesh = create_primitive_mesh(
                "cylinder", radius=1.9, radius_top=2.0, height=0.22, segments=36
            )
            stone_dir = sess.workspace_root / "textures" / "procedural_marble"
            stone_m = generate_procedural_pbr_maps("marble", out_dir=stone_dir, size=256)
            mat_stone = build_material_from_maps(
                "mat_plinth",
                "TravertinePlinth",
                stone_m["maps"],
                base_color=(0.94, 0.92, 0.88, 1.0),
                roughness=0.35,
                uv_scale=(2.0, 2.0),
                texture_source="procedural:marble",
            )
            sc.materials[mat_stone.id] = mat_stone
            sc.objects["plinth"] = SceneObject(
                id="plinth",
                name="MarblePlinth",
                kind="mesh",
                transform=Transform(location=(0.0, 0.0, 0.11)),
                mesh=ped_mesh.to_payload(),
                material_id=mat_stone.id,
            )

            torus = create_primitive_mesh("torus", radius=0.68, minor_radius=0.22, segments=32)
            gold_m = generate_procedural_pbr_maps(
                "gold", out_dir=sess.workspace_root / "textures" / "procedural_gold", size=256
            )
            mat_gold = build_material_from_maps(
                "mat_gold",
                "BurnishedGold",
                gold_m["maps"],
                base_color=(1.0, 0.98, 0.92, 1.0),
                metallic=0.95,
                roughness=0.18,
                texture_source="procedural:gold",
            )
            sc.materials[mat_gold.id] = mat_gold
            sc.objects["hero_torus"] = SceneObject(
                id="hero_torus",
                name="MachinedGoldRing",
                kind="mesh",
                transform=Transform(location=(0.0, 0.0, 0.92), rotation_deg=(38.0, 12.0, 24.0)),
                mesh=torus.to_payload(),
                material_id=mat_gold.id,
            )

            orb = create_primitive_mesh("uv_sphere", radius=0.45, segments=28)
            mat_glass = build_material_from_maps(
                "mat_glass",
                "OpticalCrystal",
                {},
                base_color=(0.76, 0.89, 0.98, 1.0),
                metallic=0.05,
                roughness=0.08,
                alpha=0.25,
            )
            sc.materials[mat_glass.id] = mat_glass
            sc.objects["crystal_orb"] = SceneObject(
                id="crystal_orb",
                name="OpticalSphere",
                kind="mesh",
                transform=Transform(location=(-0.95, 0.55, 0.67)),
                mesh=orb.to_payload(),
                material_id=mat_glass.id,
            )

        # Balance studio key + fill lights for ACES tone mapping without plinth bloom clipping
        if "light_key" in sc.objects and sc.objects["light_key"].light is not None:
            sc.objects["light_key"].light.intensity = 1.65
        if "light_fill" in sc.objects and sc.objects["light_fill"].light is not None:
            sc.objects["light_fill"].light.intensity = 0.85
        elif not any(o.kind == "light" for o in sc.objects.values()):
            sc.objects["key_light"] = SceneObject(
                id="key_light",
                name="KeySun",
                kind="light",
                transform=Transform(location=(5.0, -5.0, 7.0)),
                light=LightSpec(
                    kind="sun",
                    color=(1.0, 0.96, 0.90),
                    intensity=1.65,
                    direction=(-0.55, 0.45, -0.7),
                ),
            )
            sc.objects["fill_light"] = SceneObject(
                id="fill_light",
                name="SkyFill",
                kind="light",
                transform=Transform(location=(-4.0, 3.0, 4.0)),
                light=LightSpec(
                    kind="point", color=(0.76, 0.86, 1.0), intensity=0.85, cast_shadows=False
                ),
            )
        sess.history.clear()
    sess.notify("demo_loaded", {"preset": preset})
    return preset


async def _json_body(req: Request) -> dict[str, Any]:
    raw = await req.body()
    if not raw or not raw.strip():
        return {}
    data = json.loads(raw)
    return data if isinstance(data, dict) else {}


def create_studio_app(
    workspace_root: str | Path | None = None,
    host: str = "127.0.0.1",
    port: int = 8765,
    preload_project: str | Path | None = None,
) -> tuple[Starlette, FrFruitSession, MCPServer]:
    """Build unified Starlette app serving Studio UI, REST/WebSocket API, and live MCP endpoints."""
    mcp_srv, sess = create_server(workspace_root=workspace_root)
    if preload_project:
        loaded = load_project(preload_project)
        with sess.lock:
            sess.scene = loaded
            sess.history.clear()

    # Transport security allowing local workstation & TestClient
    tsec = TransportSecuritySettings(
        enable_dns_rebinding_protection=True,
        allowed_hosts=[
            "127.0.0.1",
            "127.0.0.1:*",
            "localhost",
            "localhost:*",
            "[::1]",
            "[::1]:*",
            "testserver",
            "testserver:*",
        ],
        allowed_origins=[
            "http://127.0.0.1",
            "http://127.0.0.1:*",
            "http://localhost",
            "http://localhost:*",
            "http://[::1]",
            "http://[::1]:*",
            "http://testserver",
            "http://testserver:*",
        ],
    )
    mcp_stream_app = mcp_srv.streamable_http_app(
        streamable_http_path="/mcp", transport_security=tsec
    )
    mcp_sse_app = mcp_srv.sse_app(
        sse_path="/sse", message_path="/messages/", transport_security=tsec
    )

    ws_queues: set[asyncio.Queue[dict[str, Any]]] = set()
    ws_lock = asyncio.Lock()

    mcp_clients_lock = threading.RLock()
    mcp_clients: dict[str, dict[str, Any]] = {}
    _CLIENT_IDLE_TTL_SEC = 300.0

    def _snapshot_mcp_clients() -> list[dict[str, Any]]:
        now = time.time()
        with mcp_clients_lock:
            stale_ids = [
                sid
                for sid, info in mcp_clients.items()
                if info.get("active_streams", 0) <= 0
                and (now - float(info.get("last_seen", now))) > _CLIENT_IDLE_TTL_SEC
            ]
            for sid in stale_ids:
                mcp_clients.pop(sid, None)
            return [
                {
                    "session_id": info["session_id"],
                    "client_name": info["client_name"],
                    "client_version": info["client_version"],
                    "transport": info["transport"],
                    "connected_at": info["connected_at"],
                    "last_seen": info["last_seen"],
                    "tool_calls": info["tool_calls"],
                    "active_streams": info["active_streams"],
                }
                for info in mcp_clients.values()
            ]

    def _register_mcp_client(
        session_id: str,
        transport: str,
        client_name: str | None = None,
        client_version: str | None = None,
        tool_called: str | None = None,
        stream_delta: int = 0,
    ) -> None:
        now = time.time()
        is_new = False
        became_named = False
        with mcp_clients_lock:
            entry = mcp_clients.get(session_id)
            if entry is None:
                is_new = True
                entry = {
                    "session_id": session_id,
                    "client_name": client_name or "MCP Client",
                    "client_version": client_version or "",
                    "transport": transport,
                    "connected_at": now,
                    "last_seen": now,
                    "tool_calls": 0,
                    "active_streams": 0,
                }
                mcp_clients[session_id] = entry
            else:
                entry["last_seen"] = now
                if client_name and entry["client_name"] == "MCP Client":
                    became_named = True
                if client_name:
                    entry["client_name"] = client_name
                if client_version:
                    entry["client_version"] = client_version
                if transport:
                    entry["transport"] = transport
            if tool_called:
                entry["tool_calls"] = int(entry.get("tool_calls", 0)) + 1
            if stream_delta:
                entry["active_streams"] = max(0, int(entry.get("active_streams", 0)) + stream_delta)
            snap_entry = dict(entry)
        clients_snap = _snapshot_mcp_clients()
        if is_new or became_named:
            sess.notify(
                "mcp_client_connected",
                {
                    "session_id": session_id,
                    "client_name": snap_entry["client_name"],
                    "transport": snap_entry["transport"],
                    "mcp_clients": clients_snap,
                    "mcp_clients_count": len(clients_snap),
                },
            )
        if tool_called:
            sess.notify(
                "mcp_tool_called",
                {
                    "session_id": session_id,
                    "client_name": snap_entry["client_name"],
                    "tool": tool_called,
                    "mcp_clients": clients_snap,
                    "mcp_clients_count": len(clients_snap),
                },
            )

    def _unregister_mcp_client(session_id: str, stream_closed_only: bool = False) -> None:
        removed: dict[str, Any] | None = None
        with mcp_clients_lock:
            entry = mcp_clients.get(session_id)
            if entry is None:
                return
            if stream_closed_only:
                entry["active_streams"] = max(0, int(entry.get("active_streams", 0)) - 1)
                if entry["active_streams"] > 0:
                    return
                removed = mcp_clients.pop(session_id, None)
            else:
                removed = mcp_clients.pop(session_id, None)
        if removed is not None:
            clients_snap = _snapshot_mcp_clients()
            sess.notify(
                "mcp_client_disconnected",
                {
                    "session_id": session_id,
                    "client_name": removed.get("client_name", "MCP Client"),
                    "transport": removed.get("transport", ""),
                    "mcp_clients": clients_snap,
                    "mcp_clients_count": len(clients_snap),
                },
            )

    async def api_state(_req: Request) -> JSONResponse:
        return JSONResponse(_serialize_state(sess, mcp_srv, _snapshot_mcp_clients))

    async def api_geometry(_req: Request) -> JSONResponse:
        return JSONResponse(_serialize_geometry(sess))

    async def api_texture_preview(req: Request) -> Response:
        mid = req.query_params.get("material_id", "")
        channel = req.query_params.get("channel", "albedo")
        with sess.lock:
            mat = sess.scene.materials.get(mid)
            if mat is None:
                return Response(status_code=404)
            path_map = {
                "albedo": mat.albedo_map,
                "normal": mat.normal_map,
                "roughness": mat.roughness_map,
                "metallic": mat.metallic_map,
            }
            fpath = path_map.get(channel)
            if not fpath:
                return Response(status_code=404)
            p = Path(fpath)
            if not p.is_file():
                return Response(status_code=404)
            try:
                from io import BytesIO

                from PIL import Image

                im = Image.open(p).convert("RGBA")
                im.thumbnail((512, 512))
                buf = BytesIO()
                im.save(buf, format="PNG")
                return Response(content=buf.getvalue(), media_type="image/png")
            except Exception:
                return Response(status_code=404)

    async def api_render(req: Request) -> Response:
        try:
            w = max(128, min(1920, int(req.query_params.get("width", "720"))))
            h = max(128, min(1080, int(req.query_params.get("height", "540"))))
            mode = req.query_params.get("mode", "pbr")
            if mode not in ("pbr", "clay", "wireframe", "normals", "depth", "collision"):
                mode = "pbr"
            eng = req.query_params.get("engine") or None
            samples = max(1, min(16, int(req.query_params.get("samples", "2"))))
            vdir = req.query_params.get("view") or None
            with sess.lock:
                png_bytes, _meta = render_single_view(
                    sess.scene,
                    width=w,
                    height=h,
                    mode=mode,  # type: ignore[arg-type]
                    view_direction=vdir,  # type: ignore[arg-type]
                    engine=eng,
                    samples=samples,
                )
            return Response(content=png_bytes, media_type="image/png")
        except Exception as exc:
            return _error_json(str(exc), 400)

    async def api_scene_new(req: Request) -> JSONResponse:
        body = await _json_body(req)
        name = str(body.get("name") or "FrFruitScene")
        sess.reset(name)
        return JSONResponse({"status": "ok", "scene_name": sess.scene.name})

    async def api_scene_demo(req: Request) -> JSONResponse:
        body = await req.json()
        preset = str(body.get("preset") or "pbr_showcase")
        used = _populate_demo_showcase(sess, preset)
        return JSONResponse({"status": "ok", "preset": used})

    async def api_scene_open(req: Request) -> JSONResponse:
        body = await req.json()
        path = str(body.get("path") or "")
        if not path:
            return _error_json("path is required")
        try:
            with sess.lock:
                p = sess.resolve_path(path, must_exist=True)
                doc = load_project(p)
                sess.scene = doc
                sess.history.clear()
            sess.notify("scene_opened", {"path": str(p)})
            return JSONResponse({"status": "ok", "scene_name": doc.name, "path": str(p)})
        except (ToolError, Exception) as exc:
            return _error_json(str(exc))

    async def api_scene_save(req: Request) -> JSONResponse:
        body = await req.json()
        path = str(body.get("path") or f"{sess.scene.name}.frf.json")
        try:
            with sess.lock:
                p = sess.resolve_path(path)
                save_project(sess.scene, p)
            sess.notify("scene_saved", {"path": str(p)})
            return JSONResponse({"status": "ok", "path": str(p)})
        except (ToolError, Exception) as exc:
            return _error_json(str(exc))

    async def api_undo(_req: Request) -> JSONResponse:
        with sess.lock:
            restored, label = sess.history.undo(sess.scene)
            if restored is None:
                return _error_json("Nothing to undo.")
            sess.scene = restored
        sess.notify("undo", {"action": label})
        return JSONResponse({"status": "ok", "restored_action": label})

    async def api_redo(_req: Request) -> JSONResponse:
        with sess.lock:
            restored, label = sess.history.redo(sess.scene)
            if restored is None:
                return _error_json("Nothing to redo.")
            sess.scene = restored
        sess.notify("redo", {"action": label})
        return JSONResponse({"status": "ok", "restored_action": label})

    async def api_add_primitive(req: Request) -> JSONResponse:
        body = await req.json()
        prim = str(body.get("primitive") or "cube")
        name = body.get("name")
        loc = tuple(float(x) for x in body.get("location", [0.0, 0.0, 0.5]))
        size = float(body.get("size", 1.0))
        radius = float(body.get("radius", 0.5))
        segments = int(body.get("segments", 24))
        try:
            mesh = create_primitive_mesh(prim, size=size, radius=radius, segments=segments)  # type: ignore[arg-type]
            with sess.lock:
                oname = sess.scene.unique_name(str(name or prim.capitalize()))
                oid = sess.scene.next_id(oname.lower())
                sess.push_history(f"add_primitive:{oname}")
                obj = SceneObject(
                    id=oid,
                    name=oname,
                    kind="mesh",
                    transform=Transform(location=(loc[0], loc[1], loc[2])),
                    mesh=mesh.to_payload(),
                )
                sess.scene.objects[oid] = obj
            sess.notify("object_added", {"object_id": oid, "name": oname})
            return JSONResponse({"status": "ok", "object_id": oid, "name": oname})
        except Exception as exc:
            return _error_json(str(exc))

    async def api_transform_object(req: Request) -> JSONResponse:
        oid = req.path_params["object_id"]
        body = await req.json()
        try:
            with sess.lock:
                obj = sess.require_object(oid)
                sess.push_history(f"transform:{obj.name}")
                if "name" in body and body["name"]:
                    obj.name = str(body["name"])
                if "visible" in body:
                    obj.visible = bool(body["visible"])
                if "location" in body:
                    v = body["location"]
                    obj.transform.location = (float(v[0]), float(v[1]), float(v[2]))
                if "rotation_deg" in body:
                    v = body["rotation_deg"]
                    obj.transform.rotation_deg = (float(v[0]), float(v[1]), float(v[2]))
                if "scale" in body:
                    v = body["scale"]
                    obj.transform.scale = (float(v[0]), float(v[1]), float(v[2]))
                if "material_id" in body:
                    obj.material_id = body["material_id"] or None
            sess.notify("object_updated", {"object_id": oid})
            return JSONResponse({"status": "ok", "object_id": oid})
        except (ToolError, Exception) as exc:
            return _error_json(str(exc))

    async def api_delete_object(req: Request) -> JSONResponse:
        oid = req.path_params["object_id"]
        try:
            with sess.lock:
                obj = sess.require_object(oid)
                sess.push_history(f"delete:{obj.name}")
                del sess.scene.objects[obj.id]
            sess.notify("object_deleted", {"object_id": obj.id})
            return JSONResponse({"status": "ok", "deleted_id": obj.id})
        except (ToolError, Exception) as exc:
            return _error_json(str(exc))

    async def api_object_elements(req: Request) -> JSONResponse:
        oid = req.path_params["object_id"]
        sel_mode = req.query_params.get("selection_mode")
        seed_raw = req.query_params.get("seed_face")
        invert = req.query_params.get("invert", "false").lower() in ("1", "true", "yes")
        limit = max(10, min(6000, int(req.query_params.get("limit", "2500"))))
        try:
            with sess.lock:
                obj = sess.require_mesh_like(oid)
                base_mesh = (
                    MeshData.from_payload(obj.mesh)
                    if obj.mesh is not None
                    else evaluate_object_mesh(sess.scene, obj.id).local_mesh
                )
                wmat = compute_world_matrix(sess.scene, obj.id)
                wverts = (
                    transform_points(wmat, base_mesh.vertices)
                    if not base_mesh.is_empty
                    else base_mesh.vertices
                )
                unique_edges_set: set[tuple[int, int]] = set()
                for a, b, c in base_mesh.faces:
                    for u, v in ((int(a), int(b)), (int(b), int(c)), (int(c), int(a))):
                        unique_edges_set.add((min(u, v), max(u, v)))
                edges_list = sorted(unique_edges_set)

                sel_faces: list[int] = []
                sel_edges: list[list[int]] = []
                sel_verts: list[int] = []
                if sel_mode:
                    query = SelectionQuery(
                        mode=sel_mode,  # type: ignore[arg-type]
                        seed_face=int(seed_raw) if seed_raw is not None else None,
                        invert=invert,
                    )
                    fmask = resolve_face_selection(
                        base_mesh,
                        query,
                        face_material_ids=obj.face_material_ids,
                        base_material_id=obj.material_id,
                    )
                    sel_faces = [int(i) for i in np.nonzero(fmask)[0].tolist()]
                    sel_edges = [[u, v] for u, v in resolve_edge_selection(base_mesh, query)]
                    sel_verts = [
                        int(i) for i in np.nonzero(resolve_vertex_selection(base_mesh, query))[0]
                    ]

                summary = inspect_mesh_elements(
                    base_mesh,
                    detail="summary",
                    face_material_ids=obj.face_material_ids,
                    base_material_id=obj.material_id,
                )
                return JSONResponse(
                    {
                        "status": "ok",
                        "object_id": obj.id,
                        "name": obj.name,
                        "material_id": obj.material_id,
                        "face_material_ids": {str(k): v for k, v in obj.face_material_ids.items()},
                        "vertex_count": len(base_mesh.vertices),
                        "edge_count": len(edges_list),
                        "face_count": len(base_mesh.faces),
                        "summary": summary,
                        "world_vertices": (
                            [[round(float(c), 5) for c in p] for p in wverts[:limit]]
                            if len(wverts) <= limit
                            else []
                        ),
                        "faces": (
                            [[int(x) for x in f] for f in base_mesh.faces[:limit]]
                            if len(base_mesh.faces) <= limit
                            else []
                        ),
                        "edges": (
                            [[u, v] for u, v in edges_list[:limit]]
                            if len(edges_list) <= limit
                            else []
                        ),
                        "selected_faces": sel_faces,
                        "selected_edges": sel_edges,
                        "selected_vertices": sel_verts,
                    }
                )
        except (ToolError, ValueError, GeometryError, Exception) as exc:
            return _error_json(str(exc))

    async def api_raycast_face(req: Request) -> JSONResponse:
        oid = req.path_params["object_id"]
        body = await _json_body(req)
        try:
            origin = np.asarray(body.get("origin", [0.0, 0.0, 0.0]), dtype=np.float64)
            d = np.asarray(body.get("direction", [0.0, 0.0, -1.0]), dtype=np.float64)
            d_len = float(np.linalg.norm(d))
            if d_len < 1e-12:
                return _error_json("direction vector must be non-zero")
            d /= d_len

            with sess.lock:
                obj = sess.require_mesh_like(oid)
                base_mesh = (
                    MeshData.from_payload(obj.mesh)
                    if obj.mesh is not None
                    else evaluate_object_mesh(sess.scene, obj.id).local_mesh
                )
                if base_mesh.is_empty:
                    return JSONResponse({"status": "ok", "hit": False, "object_id": obj.id})
                wmat = compute_world_matrix(sess.scene, obj.id)
                wverts = transform_points(wmat, base_mesh.vertices)
                faces = base_mesh.faces

                v0 = wverts[faces[:, 0]]
                v1 = wverts[faces[:, 1]]
                v2 = wverts[faces[:, 2]]
                e1 = v1 - v0
                e2 = v2 - v0
                h = np.cross(d[None, :], e2)
                a = np.einsum("ij,ij->i", e1, h)
                valid = np.abs(a) > 1e-8
                f = np.where(valid, 1.0 / np.where(valid, a, 1.0), 0.0)
                s = origin[None, :] - v0
                u = f * np.einsum("ij,ij->i", s, h)
                q = np.cross(s, e1)
                v = f * np.einsum("ij,ij->i", np.broadcast_to(d, q.shape), q)
                t = f * np.einsum("ij,ij->i", e2, q)
                mask = (
                    valid
                    & (u >= -1e-5)
                    & (u <= 1.00001)
                    & (v >= -1e-5)
                    & ((u + v) <= 1.00001)
                    & (t > 1e-5)
                )
                hit_indices = np.nonzero(mask)[0]
                if len(hit_indices) == 0:
                    return JSONResponse({"status": "ok", "hit": False, "object_id": obj.id})

                best_pos = int(np.argmin(t[hit_indices]))
                fi = int(hit_indices[best_pos])
                t_val = float(t[fi])
                hit_pt = origin + d * t_val

                tri = [int(x) for x in faces[fi]]
                closest_v = min(tri, key=lambda vi: float(np.linalg.norm(wverts[vi] - hit_pt)))

                def _pt_seg_dist(pt: np.ndarray, p1: np.ndarray, p2: np.ndarray) -> float:
                    seg = p2 - p1
                    sln2 = float(np.dot(seg, seg))
                    if sln2 < 1e-12:
                        return float(np.linalg.norm(pt - p1))
                    proj = max(0.0, min(1.0, float(np.dot(pt - p1, seg)) / sln2))
                    return float(np.linalg.norm(pt - (p1 + seg * proj)))

                tri_edges = ((tri[0], tri[1]), (tri[1], tri[2]), (tri[2], tri[0]))
                best_e = min(
                    tri_edges, key=lambda e: _pt_seg_dist(hit_pt, wverts[e[0]], wverts[e[1]])
                )
                return JSONResponse(
                    {
                        "status": "ok",
                        "hit": True,
                        "object_id": obj.id,
                        "face_index": fi,
                        "distance": round(t_val, 5),
                        "point": [round(float(c), 5) for c in hit_pt],
                        "closest_vertex": closest_v,
                        "closest_edge": [min(best_e), max(best_e)],
                        "face_material_id": obj.face_material_ids.get(fi, obj.material_id),
                    }
                )
        except (ToolError, Exception) as exc:
            return _error_json(str(exc))

    async def api_edit_object(req: Request) -> JSONResponse:
        oid = req.path_params["object_id"]
        body = await _json_body(req)
        op = str(body.get("operation") or "")
        if not op:
            return _error_json("operation is required")
        try:
            with sess.lock:
                obj = sess.require_object(oid)
                if obj.kind == "procedural" and obj.mesh is None:
                    obj.mesh = evaluate_object_mesh(sess.scene, obj.id).local_mesh.to_payload()
                    obj.kind = "mesh"
                if obj.mesh is None:
                    return _error_json(f"Object {obj.name!r} has no editable mesh.")

                base_mesh = MeshData.from_payload(obj.mesh)
                raw_edges = body.get("edge_indices")
                parsed_edges = (
                    [(int(e[0]), int(e[1])) for e in raw_edges if len(e) == 2]
                    if raw_edges
                    else None
                )
                sel_mat_raw = body.get("selection_material_id")
                if sel_mat_raw:
                    rm = sess.scene.resolve_material(str(sel_mat_raw))
                    sel_mat_raw = rm.id if rm else str(sel_mat_raw)

                query = SelectionQuery(
                    mode=body.get("selection_mode", "top"),
                    direction=(
                        tuple(float(x) for x in body["direction"])
                        if body.get("direction") is not None
                        else None
                    ),
                    angle_threshold_deg=float(body.get("angle_threshold_deg", 45.0)),
                    bbox_min=(
                        tuple(float(x) for x in body["bbox_min"])
                        if body.get("bbox_min") is not None
                        else None
                    ),
                    bbox_max=(
                        tuple(float(x) for x in body["bbox_max"])
                        if body.get("bbox_max") is not None
                        else None
                    ),
                    indices=(
                        [int(i) for i in body["indices"]]
                        if body.get("indices") is not None
                        else None
                    ),
                    vertex_indices=(
                        [int(i) for i in body["vertex_indices"]]
                        if body.get("vertex_indices") is not None
                        else None
                    ),
                    edge_indices=parsed_edges,
                    seed_face=(
                        int(body["seed_face"]) if body.get("seed_face") is not None else None
                    ),
                    material_id=sel_mat_raw,
                    invert=bool(body.get("invert", False)),
                )

                mat_id = body.get("material_id")
                resolved_mat_id: str | None = None
                if mat_id:
                    rm = sess.scene.resolve_material(str(mat_id))
                    resolved_mat_id = rm.id if rm else str(mat_id)

                res = run_edit_operation(
                    base_mesh,
                    op,  # type: ignore[arg-type]
                    query,
                    face_material_ids=obj.face_material_ids,
                    base_material_id=obj.material_id,
                    distance=float(body.get("distance", 0.3)),
                    thickness=float(body.get("thickness", 0.1)),
                    depth=float(body.get("depth", 0.0)),
                    width=float(body.get("width", 0.04)),
                    segments=int(body.get("segments", 1)),
                    levels=int(body.get("levels", 1)),
                    cuts=int(body.get("cuts", 1)),
                    slide=float(body.get("slide", 0.0)),
                    smooth_subdiv=bool(body.get("smooth_subdiv", True)),
                    individual=bool(body.get("individual", False)),
                    along_normals=bool(body.get("along_normals", False)),
                    scale=float(body.get("scale", 1.0)),
                    translate=tuple(float(x) for x in body.get("translate", [0.0, 0.0, 0.0])),
                    rotate_deg=tuple(float(x) for x in body.get("rotate_deg", [0.0, 0.0, 0.0])),
                    scale_vec=(
                        tuple(float(x) for x in body["scale_vec"])
                        if body.get("scale_vec") is not None
                        else None
                    ),
                    direction=query.direction,
                    iterations=int(body.get("iterations", 2)),
                    factor=float(body.get("factor", 0.5)),
                    axis=body.get("axis", "z"),
                    shear_axis=body.get("shear_axis", "x"),
                    plane_point=tuple(float(x) for x in body.get("plane_point", [0.0, 0.0, 0.0])),
                    plane_normal=tuple(float(x) for x in body.get("plane_normal", [0.0, 0.0, 1.0])),
                    clear_inner=bool(body.get("clear_inner", False)),
                    clear_outer=bool(body.get("clear_outer", False)),
                    fill_caps=bool(body.get("fill_caps", True)),
                    merge_mode=body.get("merge_mode", "by_distance"),
                    merge_distance=float(body.get("merge_distance", 1e-4)),
                    delete_mode=body.get("delete_mode", "faces"),
                    proportional_radius=float(body.get("proportional_radius", 0.0)),
                    proportional_falloff=body.get("proportional_falloff", "smooth"),
                    symmetrize_direction=body.get("symmetrize_direction", "-x_to_+x"),
                    projection=body.get("projection", "box"),
                    uv_scale=tuple(float(x) for x in body.get("uv_scale", [1.0, 1.0])),
                    flip_normals_flag=bool(body.get("flip_normals_flag", False)),
                    inside=bool(body.get("inside", False)),
                    offset=float(body.get("offset", 0.0)),
                    seed=int(body.get("seed", 42)),
                    normal_only=bool(body.get("normal_only", False)),
                    twist=int(body.get("twist", 0)),
                    material_id=resolved_mat_id,
                )

                sess.push_history(f"edit_ui:{obj.name}:{op}")
                obj.mesh = res.mesh.to_payload()
                obj.face_material_ids = res.face_material_ids

                separated_id: str | None = None
                if res.separated_mesh is not None:
                    sep_name = sess.scene.unique_name(
                        str(body.get("separated_name") or f"{obj.name}_Part")
                    )
                    separated_id = sess.scene.next_id(sep_name.lower())
                    sep_obj = SceneObject(
                        id=separated_id,
                        name=sep_name,
                        kind="mesh",
                        transform=obj.transform.model_copy(deep=True),
                        material_id=obj.material_id,
                        face_material_ids=res.separated_face_material_ids or {},
                        parent_id=obj.parent_id,
                        collection_id=obj.collection_id,
                        mesh=res.separated_mesh.to_payload(),
                    )
                    sess.scene.objects[separated_id] = sep_obj

                out_stats = res.mesh.stats()
            sess.notify("mesh_edited", {"object_id": obj.id, "operation": op})
            return JSONResponse(
                {
                    "status": "ok",
                    "object_id": obj.id,
                    "operation": op,
                    "stats": out_stats,
                    "face_material_count": len(obj.face_material_ids),
                    "separated_object_id": separated_id,
                    **res.extra,
                }
            )
        except (ToolError, ValueError, GeometryError, Exception) as exc:
            return _error_json(str(exc))

    async def api_assign_face_material(req: Request) -> JSONResponse:
        body = await _json_body(req)
        oid = str(body.get("object_id") or "")
        mid_raw = body.get("material_id")
        clear = bool(body.get("clear", False))
        if not oid:
            return _error_json("object_id is required")
        try:
            with sess.lock:
                obj = sess.require_mesh_like(oid)
                mat_id: str | None = None
                if not clear:
                    if not mid_raw:
                        return _error_json("material_id is required unless clear=true")
                    mat = sess.scene.resolve_material(str(mid_raw))
                    if mat is None:
                        return _error_json(f"Material {mid_raw!r} not found.")
                    mat_id = mat.id
                sess.push_history(f"assign_faces_ui:{obj.name}")
                matched = _apply_face_material_selection(
                    sess.scene,
                    obj,
                    mat_id,
                    face_index=body.get("face_index"),
                    face_indices=body.get("face_indices"),
                    selection_mode=body.get("selection_mode"),
                    direction=(
                        tuple(float(x) for x in body["direction"])
                        if body.get("direction") is not None
                        else None
                    ),
                    angle_threshold_deg=float(body.get("angle_threshold_deg", 45.0)),
                    bbox_min=None,
                    bbox_max=None,
                    seed_face=body.get("seed_face"),
                    invert=bool(body.get("invert", False)),
                    clear=clear,
                )
            sess.notify(
                "face_material_assigned",
                {"object_id": obj.id, "material_id": mat_id, "count": len(matched)},
            )
            return JSONResponse(
                {
                    "status": "ok",
                    "object_id": obj.id,
                    "material_id": mat_id,
                    "faces": matched,
                    "count": len(matched),
                }
            )
        except (ToolError, Exception) as exc:
            return _error_json(str(exc))

    async def api_postprocess(req: Request) -> JSONResponse:
        body = await req.json()
        try:
            with sess.lock:
                sess.push_history("postprocess_ui")
                if "render_engine" in body and body["render_engine"] in ("software", "raytrace"):
                    sess.scene.render_engine = body["render_engine"]
                if "preset" in body and body["preset"] in _POSTPROCESS_PRESETS:
                    merged = sess.scene.post_process.model_dump()
                    merged.update(_POSTPROCESS_PRESETS[body["preset"]])
                    sess.scene.post_process = PostProcessConfig(**merged)
                if "post_process" in body and isinstance(body["post_process"], dict):
                    merged = sess.scene.post_process.model_dump()
                    merged.update(body["post_process"])
                    sess.scene.post_process = PostProcessConfig(**merged)
                pp = sess.scene.post_process.model_dump()
                eng = sess.scene.render_engine
            sess.notify("postprocess_updated", {"render_engine": eng})
            return JSONResponse({"status": "ok", "render_engine": eng, "post_process": pp})
        except Exception as exc:
            return _error_json(str(exc))

    async def api_textures_search(req: Request) -> JSONResponse:
        query = req.query_params.get("query", "")
        source = req.query_params.get("source", "all")
        if source not in ("all", "polyhaven", "ambientcg", "procedural"):
            source = "all"
        limit = max(1, min(40, int(req.query_params.get("limit", "12"))))
        try:
            hits = search_online_textures(query=query, source=source, limit=limit)  # type: ignore[arg-type]
        except Exception:
            hits = search_online_textures(query=query, source="procedural", limit=limit)
        return JSONResponse({"count": len(hits), "source": source, "results": hits})

    async def api_textures_apply(req: Request) -> JSONResponse:
        body = await req.json()
        aid = str(body.get("asset_id") or "stone")
        src = str(body.get("source") or "polyhaven")
        res = str(body.get("resolution") or "1k")
        object_id = body.get("object_id")
        uv_raw = body.get("uv_scale", [1.0, 1.0])
        uv_scale = (float(uv_raw[0]), float(uv_raw[1]))
        procedural_only = bool(body.get("procedural_only", src == "procedural"))
        try:
            with sess.lock:
                tex_dir = sess.workspace_root / "textures"
                if procedural_only or src == "procedural":
                    manifest = generate_procedural_pbr_maps(
                        style=aid,
                        out_dir=tex_dir / f"procedural_{aid}",
                        size=512 if res == "1k" else 1024,
                    )
                else:
                    dl_src = "ambientcg" if src == "ambientcg" else "polyhaven"
                    manifest = download_pbr_texture_pack(
                        asset_id=aid,
                        source=dl_src,  # type: ignore[arg-type]
                        resolution=res,  # type: ignore[arg-type]
                        target_dir=tex_dir,
                        fallback_to_procedural=True,
                    )
                mid = sess.scene.next_id(f"mat_{aid.lower()}")
                mat_name = str(body.get("material_name") or f"PBR_{aid}")
                mat = build_material_from_maps(
                    material_id=mid,
                    name=mat_name,
                    maps=manifest.get("maps", {}),
                    texture_source=f"{manifest.get('source', src)}:{aid}",
                    uv_scale=uv_scale,
                )
                sess.push_history(f"texture_apply:{mat_name}")
                sess.scene.materials[mid] = mat
                assigned_to = None
                if object_id:
                    obj = sess.require_mesh_like(str(object_id))
                    obj.material_id = mid
                    assigned_to = obj.id
            sess.notify("texture_applied", {"material_id": mid, "object_id": assigned_to})
            return JSONResponse(
                {
                    "status": "ok",
                    "material_id": mid,
                    "assigned_to": assigned_to,
                    "manifest": manifest,
                }
            )
        except (ToolError, Exception) as exc:
            return _error_json(str(exc))

    async def api_physics_setup(req: Request) -> JSONResponse:
        body = await req.json()
        oid = str(body.get("object_id") or "")
        kind = str(body.get("kind") or "cloth")
        try:
            with sess.lock:
                obj = sess.require_mesh_like(oid)
                sess.push_history(f"physics_setup:{obj.name}")
                if kind == "rigid_body":
                    obj.rigid_body = RigidBodySpec(
                        enabled=bool(body.get("enabled", True)),
                        body_type=body.get("body_type", "dynamic"),
                        mass=float(body.get("mass", 1.0)),
                        restitution=float(body.get("restitution", 0.45)),
                        friction=float(body.get("friction", 0.55)),
                        collider_shape=body.get("collider_shape", "box"),
                    )
                else:
                    w = body.get("wind", [0.0, 0.0, 0.0])
                    obj.cloth = ClothSpec(
                        enabled=bool(body.get("enabled", True)),
                        stiffness=float(body.get("stiffness", 0.92)),
                        bend_stiffness=float(body.get("bend_stiffness", 0.25)),
                        damping=float(body.get("damping", 0.03)),
                        wind=(float(w[0]), float(w[1]), float(w[2])),
                        pin_mode=body.get("pin_mode", "top_corners"),
                        substeps=int(body.get("substeps", 8)),
                    )
            sess.notify("physics_configured", {"object_id": oid, "kind": kind})
            return JSONResponse({"status": "ok", "object_id": oid})
        except (ToolError, Exception) as exc:
            return _error_json(str(exc))

    async def api_physics_bake(req: Request) -> JSONResponse:
        body = await _json_body(req)
        fs = body.get("frame_start")
        fe = body.get("frame_end")
        gz = float(body.get("ground_z", 0.0))
        try:
            with sess.lock:
                sess.push_history("physics_bake_ui")
                res = bake_scene_physics(
                    sess.scene,
                    frame_start=int(fs) if fs is not None else None,
                    frame_end=int(fe) if fe is not None else None,
                    ground_z=gz,
                )
            sess.notify("physics_baked", res)
            return JSONResponse({"status": "ok", **res})
        except Exception as exc:
            return _error_json(str(exc))

    async def api_physics_clear(_req: Request) -> JSONResponse:
        with sess.lock:
            sess.push_history("physics_clear_ui")
            res = clear_physics_caches(sess.scene)
        sess.notify("physics_cleared", res)
        return JSONResponse({"status": "ok", **res})

    async def api_timeline_frame(req: Request) -> JSONResponse:
        body = await req.json()
        frame = int(body.get("frame", 1))
        with sess.lock:
            sess.scene = apply_animation_frame(sess.scene, frame)
            cur = sess.scene.timeline.current_frame
        return JSONResponse({"status": "ok", "current_frame": cur})

    async def api_export(req: Request) -> JSONResponse:
        body = await req.json()
        path = str(body.get("path") or "exports/model.fbx")
        fmt = body.get("format")
        target_engine = str(body.get("target_engine") or "generic")
        try:
            with sess.lock:
                out_p = sess.resolve_path(path)
                res = export_scene_file(
                    sess.scene,
                    out_p,
                    fmt=fmt,
                    target_engine=target_engine,  # type: ignore[arg-type]
                )
            sess.notify("scene_exported", {"path": res["path"], "format": res["format"]})
            return JSONResponse({"status": "ok", **res})
        except (ToolError, Exception) as exc:
            return _error_json(str(exc))

    async def api_video_render(req: Request) -> JSONResponse:
        body = await req.json()
        path = str(body.get("path") or "renders/turntable.mp4")
        kind = str(body.get("kind") or "turntable")
        fps = body.get("fps")
        fs = body.get("frame_start")
        fe = body.get("frame_end")
        w = int(body.get("width", 640))
        h = int(body.get("height", 480))
        mode = str(body.get("mode") or "pbr")
        eng = body.get("engine") or None
        samples = int(body.get("samples", 2))
        hud = bool(body.get("overlay_hud", True))
        try:
            with sess.lock:
                out_p = sess.resolve_path(path)
                res = render_video(
                    sess.scene,
                    out_path=out_p,
                    kind=kind,  # type: ignore[arg-type]
                    frame_start=int(fs) if fs is not None else None,
                    frame_end=int(fe) if fe is not None else None,
                    fps=int(fps) if fps is not None else None,
                    width=w,
                    height=h,
                    mode=mode,  # type: ignore[arg-type]
                    engine=eng,
                    samples=samples,
                    overlay_hud=hud,
                )
            sess.notify("video_rendered", {"path": res["path"], "format": res["format"]})
            return JSONResponse({"status": "ok", **res})
        except (ToolError, Exception) as exc:
            return _error_json(str(exc))

    async def api_file_download(req: Request) -> Response:
        raw_path = req.query_params.get("path", "")
        if not raw_path:
            return _error_json("path query param required", 400)
        try:
            p = sess.resolve_path(raw_path, must_exist=True)
            return FileResponse(path=str(p), filename=p.name)
        except (ToolError, Exception) as exc:
            return _error_json(str(exc), 404)

    async def api_character_presets(_req: Request) -> JSONResponse:
        return JSONResponse(
            {
                "status": "ok",
                "presets": {k: v.model_dump() for k, v in CHARACTER_PRESETS.items()},
                "expressions": list(ExpressionSliders.model_fields.keys()),
                "semantic_rbf_regions": sorted(SEMANTIC_RBF_ANCHORS.keys()),
                "skin_tone_presets": SKIN_TONE_PRESETS,
                "delta_regions": list(FACE_REGION_FIELDS.keys()),
            }
        )

    async def api_character_create(req: Request) -> JSONResponse:
        body = await _json_body(req)
        try:
            preset = str(body.get("preset") or "realistic_neutral")
            spec = get_character_preset(preset)
            if body.get("quality") is not None:
                spec.quality = str(body["quality"])  # type: ignore[assignment]
            if body.get("pose") is not None:
                spec.pose = str(body["pose"])  # type: ignore[assignment]
            if body.get("auto_rig") is not None:
                spec.auto_rig = bool(body["auto_rig"])
            if isinstance(body.get("body"), dict):
                b_data = spec.body.model_dump()
                b_data.update(body["body"])
                spec.body = BodySliders(**b_data)
            if isinstance(body.get("face"), dict):
                f_data = spec.face.model_dump()
                f_data.update(body["face"])
                spec.face = FaceSliders(**f_data)
            if isinstance(body.get("expressions"), dict):
                e_data = spec.expressions.model_dump()
                e_data.update(body["expressions"])
                spec.expressions = ExpressionSliders(**e_data)
            if isinstance(body.get("custom_morphs"), dict):
                for mk, mv in body["custom_morphs"].items():
                    spec.custom_morphs[str(mk)] = float(mv)
            if isinstance(body.get("materials"), dict):
                m_data = spec.materials.model_dump()
                m_data.update(body["materials"])
                spec.materials = CharacterMaterialParams(**m_data)

            name = str(body["name"]) if body.get("name") else None
            loc_raw = body.get("location", [0.0, 0.0, 0.0])
            rot_raw = body.get("rotation_deg", [0.0, 0.0, 0.0])
            loc = (float(loc_raw[0]), float(loc_raw[1]), float(loc_raw[2]))
            rot = (float(rot_raw[0]), float(rot_raw[1]), float(rot_raw[2]))

            with sess.lock:
                sess.push_history(f"character_create:{spec.preset}")
                res = build_or_update_character(
                    sess.scene,
                    spec,
                    name=name,
                    location=loc,
                    rotation_deg=rot,
                    cache_root=sess.workspace_root,
                )
            sess.notify(
                "character_created",
                {"object_id": res["object_id"], "name": res["name"]},
            )
            return JSONResponse(res)
        except Exception as exc:
            return _error_json(str(exc))

    async def api_character_update(req: Request) -> JSONResponse:
        oid = req.path_params["object_id"]
        body = await _json_body(req)
        try:
            with sess.lock:
                obj = sess.require_object(oid)
                if obj.character is None:
                    return _error_json(
                        f"Object {obj.name!r} is not a procedural Quad-Morph character."
                    )
                spec = obj.character.model_copy(deep=True)
                if body.get("quality") is not None:
                    spec.quality = str(body["quality"])  # type: ignore[assignment]
                if body.get("pose") is not None:
                    spec.pose = str(body["pose"])  # type: ignore[assignment]
                if body.get("auto_rig") is not None:
                    spec.auto_rig = bool(body["auto_rig"])
                if isinstance(body.get("body"), dict):
                    b_data = spec.body.model_dump()
                    b_data.update(body["body"])
                    spec.body = BodySliders(**b_data)
                if isinstance(body.get("face"), dict):
                    f_data = spec.face.model_dump()
                    f_data.update(body["face"])
                    spec.face = FaceSliders(**f_data)
                if isinstance(body.get("expressions"), dict):
                    e_data = spec.expressions.model_dump()
                    e_data.update(body["expressions"])
                    spec.expressions = ExpressionSliders(**e_data)
                if isinstance(body.get("custom_morphs"), dict):
                    for mk, mv in body["custom_morphs"].items():
                        spec.custom_morphs[str(mk)] = float(mv)
                if isinstance(body.get("materials"), dict):
                    m_data = spec.materials.model_dump()
                    m_data.update(body["materials"])
                    spec.materials = CharacterMaterialParams(**m_data)

                sess.push_history(f"character_update:{obj.id}")
                res = build_or_update_character(
                    sess.scene,
                    spec,
                    object_id=obj.id,
                    cache_root=sess.workspace_root,
                )
            sess.notify("character_updated", {"object_id": obj.id})
            return JSONResponse(res)
        except (ToolError, Exception) as exc:
            return _error_json(str(exc))

    async def api_character_sculpt_rbf(req: Request) -> JSONResponse:
        oid = req.path_params["object_id"]
        body = await _json_body(req)
        try:
            with sess.lock:
                obj = sess.require_object(oid)
                if obj.character is None:
                    return _error_json(
                        f"Object {obj.name!r} is not a procedural Quad-Morph character."
                    )
                spec = obj.character.model_copy(deep=True)
                if body.get("clear_rbf"):
                    spec.rbf_nodes = []
                node_info: dict[str, Any] | None = None
                region = body.get("region")
                center_raw = body.get("center")
                if region is not None or center_raw is not None:
                    center = (
                        (float(center_raw[0]), float(center_raw[1]), float(center_raw[2]))
                        if center_raw is not None
                        else None
                    )
                    vec_raw = body.get("vector_delta", [0.0, 0.0, 0.0])
                    vec_delta = (float(vec_raw[0]), float(vec_raw[1]), float(vec_raw[2]))
                    spec, node = add_rbf_sculpt_node(
                        spec,
                        region=str(region or "custom"),
                        center=center,
                        radius=float(body["radius"]) if body.get("radius") is not None else None,
                        weight=float(body.get("weight", 0.0)),
                        vector_delta=vec_delta,
                        mirror_x=body.get("mirror_x"),
                    )
                    node_info = node.model_dump()
                elif not body.get("clear_rbf"):
                    return _error_json("Specify region/center or clear_rbf=true.")

                sess.push_history(f"character_sculpt_rbf:{obj.id}")
                res = build_or_update_character(
                    sess.scene,
                    spec,
                    object_id=obj.id,
                    cache_root=sess.workspace_root,
                )
                res["added_node"] = node_info
            sess.notify("character_updated", {"object_id": obj.id})
            return JSONResponse(res)
        except (ToolError, Exception) as exc:
            return _error_json(str(exc))

    async def api_character_blend_delta(req: Request) -> JSONResponse:
        oid = req.path_params["object_id"]
        body = await _json_body(req)
        try:
            source_id = body.get("source_id")
            preset = body.get("preset")
            region = str(body.get("region") or "all")
            alpha = float(body.get("alpha", 0.5))
            mode = str(body.get("mode") or "blend")
            delta_name = str(body.get("delta_name") or "delta_1")

            with sess.lock:
                obj = sess.require_object(oid)
                if obj.character is None:
                    return _error_json(
                        f"Object {obj.name!r} is not a procedural Quad-Morph character."
                    )

                src_spec = None
                src_label = ""
                if source_id:
                    sobj = sess.scene.resolve_object(str(source_id))
                    if sobj is not None and sobj.character is not None:
                        src_spec = sobj.character
                        src_label = sobj.name
                    elif str(source_id).strip().lower() in CHARACTER_PRESETS:
                        src_spec = get_character_preset(str(source_id))
                        src_label = str(source_id)
                if src_spec is None and preset:
                    src_spec = get_character_preset(str(preset))
                    src_label = str(preset)

                if mode == "apply_delta":
                    if delta_name in obj.character.stored_deltas and src_spec is None:
                        delta = obj.character.stored_deltas[delta_name]
                    elif src_spec is not None:
                        base_spec = get_character_preset("realistic_neutral")
                        delta = compute_character_delta(
                            base_spec,
                            src_spec,
                            region=region,  # type: ignore[arg-type]
                            name=delta_name,
                            source_a_name="realistic_neutral",
                            source_b_name=src_label,
                        )
                    else:
                        return _error_json("Provide source_id, preset, or a stored delta_name.")
                    updated = apply_character_delta(
                        obj.character,
                        delta,
                        weight=alpha,
                        region=region,  # type: ignore[arg-type]
                    )
                else:
                    if src_spec is None:
                        return _error_json("Provide source_id or preset for blending.")
                    updated = blend_characters(
                        obj.character,
                        src_spec,
                        alpha=alpha,
                        region=region,  # type: ignore[arg-type]
                    )

                sess.push_history(f"character_blend_delta:{obj.id}")
                res = build_or_update_character(
                    sess.scene,
                    updated,
                    object_id=obj.id,
                    cache_root=sess.workspace_root,
                )
                res["mode"] = mode
                res["region"] = region
                res["alpha"] = alpha
            sess.notify("character_updated", {"object_id": obj.id})
            return JSONResponse(res)
        except (ToolError, Exception) as exc:
            return _error_json(str(exc))

    async def api_character_extract_edit_delta(req: Request) -> JSONResponse:
        oid = req.path_params["object_id"]
        body = await _json_body(req)
        try:
            region_name = str(body.get("region") or "mesh_edit")
            with sess.lock:
                obj = sess.require_object(oid)
                if obj.character is None or obj.mesh is None:
                    return _error_json(
                        f"Object {obj.name!r} is not an active Quad-Morph character mesh."
                    )
                edited_mesh = MeshData.from_payload(obj.mesh)
                surface = CharacterSurface(obj.character, quality=obj.character.quality)
                orig_mesh = surface.skin_mesh
                if len(orig_mesh.vertices) != len(edited_mesh.vertices):
                    return _error_json(
                        f"Mesh topology changed ({len(orig_mesh.vertices)} -> "
                        f"{len(edited_mesh.vertices)} vertices). Only non-topological edits "
                        "can be extracted to RBF deltas."
                    )
                nodes = extract_rbf_delta_from_mesh_edit(
                    orig_mesh,
                    edited_mesh,
                    surface.lm,
                    region_name=region_name,
                )
                if not nodes:
                    return JSONResponse(
                        {
                            "status": "ok",
                            "object_id": obj.id,
                            "extracted_nodes": 0,
                            "message": "No significant vertex displacements detected.",
                        }
                    )
                spec = obj.character.model_copy(deep=True)
                spec.rbf_nodes.extend(nodes)
                sess.push_history(f"character_extract_edit_delta:{obj.id}")
                res = build_or_update_character(
                    sess.scene,
                    spec,
                    object_id=obj.id,
                    cache_root=sess.workspace_root,
                )
                res["extracted_nodes"] = len(nodes)
                res["nodes"] = [n.model_dump() for n in nodes]
            sess.notify("character_updated", {"object_id": obj.id})
            return JSONResponse(res)
        except (ToolError, Exception) as exc:
            return _error_json(str(exc))

    async def api_character_save_morph(req: Request) -> JSONResponse:
        oid = req.path_params["object_id"]
        body = await _json_body(req)
        try:
            morph_name = str(body.get("morph_name") or "custom_morph_1")
            region = str(body.get("region") or "all")
            with sess.lock:
                obj = sess.require_object(oid)
                if obj.character is None or obj.mesh is None:
                    return _error_json(
                        f"Object {obj.name!r} is not an active Quad-Morph character mesh."
                    )
                edited_mesh = MeshData.from_payload(obj.mesh)
                morph = extract_custom_morph_from_mesh_edit(
                    obj.character,
                    edited_mesh,
                    morph_name=morph_name,
                    region=region,
                )
                if not morph.deltas:
                    return JSONResponse(
                        {
                            "status": "ok",
                            "object_id": obj.id,
                            "morph_name": morph.name,
                            "active_vertices": 0,
                            "message": "No significant base-cage displacements detected.",
                        }
                    )
                spec = obj.character.model_copy(deep=True)
                spec.stored_morphs[morph.name] = morph
                spec.custom_morphs[morph.name] = 1.0
                sess.push_history(f"character_save_morph:{obj.id}:{morph.name}")
                res = build_or_update_character(
                    sess.scene,
                    spec,
                    object_id=obj.id,
                    cache_root=sess.workspace_root,
                )
                res["saved_morph"] = morph.name
                res["active_vertices"] = len(morph.deltas)
            sess.notify("character_updated", {"object_id": obj.id})
            return JSONResponse(res)
        except (ToolError, Exception) as exc:
            return _error_json(str(exc))

    async def api_mcp_info(_req: Request) -> JSONResponse:
        base_url = f"http://{host}:{port}"
        ws_url = f"ws://{host}:{port}/ws/events"
        stream_url = f"{base_url}/mcp"
        sse_url = f"{base_url}/sse"
        tools_list = sorted(t.name for t in mcp_srv._tool_manager.list_tools())
        ws_path_str = str(sess.workspace_root)
        clients_snap = _snapshot_mcp_clients()
        configs = {
            "streamable_http_client": {
                "mcpServers": {
                    "frfruit": {
                        "url": stream_url,
                    }
                }
            },
            "sse_client": {
                "mcpServers": {
                    "frfruit": {
                        "url": sse_url,
                    }
                }
            },
            "stdio_client": {
                "mcpServers": {
                    "frfruit": {
                        "command": "frfruit-mcp",
                        "args": ["--studio-url", stream_url],
                        "env": {"FRFRUIT_WORKSPACE": ws_path_str},
                    }
                }
            },
        }
        return JSONResponse(
            {
                "status": "online",
                "base_url": base_url,
                "streamable_http_url": stream_url,
                "sse_url": sse_url,
                "ws_url": ws_url,
                "workspace_root": ws_path_str,
                "tools_count": len(tools_list),
                "tools": tools_list,
                "mcp_clients": clients_snap,
                "mcp_clients_count": len(clients_snap),
                "configs": configs,
            }
        )

    async def ws_events(ws: WebSocket) -> None:
        await ws.accept()
        q: asyncio.Queue[dict[str, Any]] = asyncio.Queue(maxsize=128)
        async with ws_lock:
            ws_queues.add(q)
        try:
            clients_snap = _snapshot_mcp_clients()
            await ws.send_json(
                {
                    "event": "connected",
                    "scene_name": sess.scene.name,
                    "tools_count": len(mcp_srv._tool_manager.list_tools()),
                    "mcp_clients": clients_snap,
                    "mcp_clients_count": len(clients_snap),
                }
            )
            while True:
                msg = await q.get()
                await ws.send_json(msg)
        except WebSocketDisconnect:
            pass
        except Exception:
            pass
        finally:
            async with ws_lock:
                ws_queues.discard(q)

    @contextlib.asynccontextmanager
    async def lifespan(_app: Starlette):
        loop = asyncio.get_running_loop()

        def on_session_event(payload: dict[str, Any]) -> None:
            def _dispatch() -> None:
                for q in list(ws_queues):
                    try:
                        q.put_nowait(payload)
                    except asyncio.QueueFull:
                        pass

            with contextlib.suppress(RuntimeError):
                loop.call_soon_threadsafe(_dispatch)

        sess.add_listener(on_session_event)
        try:
            async with mcp_stream_app.router.lifespan_context(mcp_stream_app):
                yield
        finally:
            sess.remove_listener(on_session_event)

    static_dir = _resolve_static_dir()
    routes: list[Any] = [
        Route("/api/state", api_state, methods=["GET"]),
        Route("/api/geometry", api_geometry, methods=["GET"]),
        Route("/api/texture_preview", api_texture_preview, methods=["GET"]),
        Route("/api/render", api_render, methods=["GET"]),
        Route("/api/scene/new", api_scene_new, methods=["POST"]),
        Route("/api/scene/demo", api_scene_demo, methods=["POST"]),
        Route("/api/scene/open", api_scene_open, methods=["POST"]),
        Route("/api/scene/save", api_scene_save, methods=["POST"]),
        Route("/api/undo", api_undo, methods=["POST"]),
        Route("/api/redo", api_redo, methods=["POST"]),
        Route("/api/objects/primitive", api_add_primitive, methods=["POST"]),
        Route("/api/objects/{object_id}/transform", api_transform_object, methods=["POST"]),
        Route("/api/objects/{object_id}/elements", api_object_elements, methods=["GET"]),
        Route("/api/objects/{object_id}/raycast_face", api_raycast_face, methods=["POST"]),
        Route("/api/objects/{object_id}/edit", api_edit_object, methods=["POST"]),
        Route("/api/objects/{object_id}", api_delete_object, methods=["DELETE"]),
        Route("/api/materials/assign_faces", api_assign_face_material, methods=["POST"]),
        Route("/api/postprocess", api_postprocess, methods=["POST"]),
        Route("/api/textures/search", api_textures_search, methods=["GET"]),
        Route("/api/textures/apply", api_textures_apply, methods=["POST"]),
        Route("/api/physics/setup", api_physics_setup, methods=["POST"]),
        Route("/api/physics/bake", api_physics_bake, methods=["POST"]),
        Route("/api/physics/clear", api_physics_clear, methods=["POST"]),
        Route("/api/timeline/frame", api_timeline_frame, methods=["POST"]),
        Route("/api/export", api_export, methods=["POST"]),
        Route("/api/video/render", api_video_render, methods=["POST"]),
        Route("/api/files/download", api_file_download, methods=["GET"]),
        Route("/api/character/presets", api_character_presets, methods=["GET"]),
        Route("/api/character/create", api_character_create, methods=["POST"]),
        Route("/api/character/{object_id}/update", api_character_update, methods=["POST"]),
        Route("/api/character/{object_id}/sculpt_rbf", api_character_sculpt_rbf, methods=["POST"]),
        Route(
            "/api/character/{object_id}/blend_delta", api_character_blend_delta, methods=["POST"]
        ),
        Route(
            "/api/character/{object_id}/extract_edit_delta",
            api_character_extract_edit_delta,
            methods=["POST"],
        ),
        Route(
            "/api/character/{object_id}/save_morph",
            api_character_save_morph,
            methods=["POST"],
        ),
        Route("/api/mcp/info", api_mcp_info, methods=["GET"]),
        WebSocketRoute("/ws/events", ws_events),
    ]
    # Co-host both Streamable HTTP (/mcp) and SSE (/sse, /messages/) from the shared MCPServer
    routes.extend(mcp_stream_app.routes)
    routes.extend(mcp_sse_app.routes)
    if static_dir.is_dir():
        routes.append(
            Mount("/", app=StaticFiles(directory=str(static_dir), html=True), name="static")
        )

    class _McpTelemetryMiddleware:
        """Observe MCP Streamable HTTP & SSE connections and tool calls without altering streams."""

        def __init__(self, asgi_app: ASGIApp) -> None:
            self.asgi_app = asgi_app

        @staticmethod
        def _parse_rpc_metadata(
            body_bytes: bytes,
        ) -> tuple[str | None, str | None, str | None, bool]:
            if not body_bytes:
                return None, None, None, False
            try:
                parsed = json.loads(body_bytes)
            except Exception:
                return None, None, None, False
            msgs = parsed if isinstance(parsed, list) else [parsed]
            client_name: str | None = None
            client_version: str | None = None
            tool_name: str | None = None
            has_rpc = False
            for m in msgs:
                if not isinstance(m, dict):
                    continue
                method = m.get("method")
                if not isinstance(method, str):
                    continue
                has_rpc = True
                params = m.get("params")
                if not isinstance(params, dict):
                    params = {}
                if method == "initialize":
                    cinfo = params.get("clientInfo")
                    if isinstance(cinfo, dict):
                        client_name = str(cinfo.get("name") or "MCP Client")
                        client_version = str(cinfo.get("version") or "")
                    else:
                        client_name = "MCP Client"
                elif method == "tools/call":
                    tname = params.get("name")
                    if isinstance(tname, str) and tname:
                        tool_name = tname
            return client_name, client_version, tool_name, has_rpc

        @staticmethod
        def _headers_dict(scope: Scope) -> dict[str, str]:
            out: dict[str, str] = {}
            for k, v in scope.get("headers") or []:
                out[k.decode("latin-1").lower()] = v.decode("latin-1")
            return out

        async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
            if scope.get("type") != "http":
                await self.asgi_app(scope, receive, send)
                return

            path = scope.get("path", "")
            method = scope.get("method", "GET").upper()

            if path == "/sse" and method == "GET":
                sse_sid: str | None = None
                fallback_sid = f"sse:{id(scope)}"

                async def sse_send(message: Message) -> None:
                    nonlocal sse_sid
                    if message["type"] == "http.response.body" and sse_sid is None:
                        chunk = message.get("body", b"").decode("utf-8", errors="ignore")
                        m = re.search(r"session_id=([A-Za-z0-9_-]+)", chunk)
                        if m:
                            sse_sid = m.group(1)
                            _register_mcp_client(sse_sid, "sse", stream_delta=1)
                        else:
                            sse_sid = fallback_sid
                            _register_mcp_client(fallback_sid, "sse", stream_delta=1)
                    await send(message)

                try:
                    await self.asgi_app(scope, receive, sse_send)
                finally:
                    if sse_sid is not None:
                        _unregister_mcp_client(sse_sid, stream_closed_only=True)
                return

            if path == "/mcp":
                req_headers = self._headers_dict(scope)
                req_sid = req_headers.get("mcp-session-id")

                if method == "GET":
                    if req_sid:
                        _register_mcp_client(req_sid, "streamable-http", stream_delta=1)
                    try:
                        await self.asgi_app(scope, receive, send)
                    finally:
                        if req_sid:
                            _unregister_mcp_client(req_sid, stream_closed_only=True)
                    return

                if method == "DELETE":
                    await self.asgi_app(scope, receive, send)
                    if req_sid:
                        _unregister_mcp_client(req_sid)
                    return

                if method == "POST":
                    chunks: list[bytes] = []
                    more = True
                    while more:
                        msg = await receive()
                        if msg["type"] != "http.request":
                            break
                        chunks.append(msg.get("body", b""))
                        more = bool(msg.get("more_body", False))
                    body_bytes = b"".join(chunks)
                    sent_body = False

                    async def replay_receive() -> Message:
                        nonlocal sent_body
                        if not sent_body:
                            sent_body = True
                            return {"type": "http.request", "body": body_bytes, "more_body": False}
                        return await receive()

                    resp_status = 200
                    resp_sid: str | None = None

                    async def capture_send(message: Message) -> None:
                        nonlocal resp_status, resp_sid
                        if message["type"] == "http.response.start":
                            resp_status = int(message.get("status", 200))
                            for k, v in message.get("headers") or []:
                                if k.decode("latin-1").lower() == "mcp-session-id":
                                    resp_sid = v.decode("latin-1")
                        await send(message)

                    await self.asgi_app(scope, replay_receive, capture_send)
                    cname, cver, tool_name, has_rpc = self._parse_rpc_metadata(body_bytes)
                    sid = resp_sid or req_sid
                    if resp_status < 400 and (sid or has_rpc):
                        _register_mcp_client(
                            sid or "http-stateless",
                            "streamable-http",
                            client_name=cname,
                            client_version=cver,
                            tool_called=tool_name,
                        )
                    return

            if path.startswith("/messages") and method == "POST":
                qs = urllib.parse.parse_qs((scope.get("query_string") or b"").decode("latin-1"))
                msg_sid = (qs.get("session_id") or [None])[0]
                chunks = []
                more = True
                while more:
                    msg = await receive()
                    if msg["type"] != "http.request":
                        break
                    chunks.append(msg.get("body", b""))
                    more = bool(msg.get("more_body", False))
                body_bytes = b"".join(chunks)
                sent_body = False

                async def replay_msg_receive() -> Message:
                    nonlocal sent_body
                    if not sent_body:
                        sent_body = True
                        return {"type": "http.request", "body": body_bytes, "more_body": False}
                    return await receive()

                resp_status = 200

                async def capture_msg_send(message: Message) -> None:
                    nonlocal resp_status
                    if message["type"] == "http.response.start":
                        resp_status = int(message.get("status", 200))
                    await send(message)

                await self.asgi_app(scope, replay_msg_receive, capture_msg_send)
                cname, cver, tool_name, has_rpc = self._parse_rpc_metadata(body_bytes)
                if resp_status < 400 and (msg_sid or has_rpc):
                    _register_mcp_client(
                        msg_sid or "sse-default",
                        "sse",
                        client_name=cname,
                        client_version=cver,
                        tool_called=tool_name,
                    )
                return

            await self.asgi_app(scope, receive, send)

    app = Starlette(routes=routes, lifespan=lifespan)
    app.add_middleware(_McpTelemetryMiddleware)
    app.state.frfruit_session = sess
    app.state.frfruit_mcp = mcp_srv
    app.state.get_mcp_clients = _snapshot_mcp_clients
    return app, sess, mcp_srv
