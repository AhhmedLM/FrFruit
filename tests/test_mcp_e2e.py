"""End-to-end MCP workflow tests exercising all FrFruit tools, resources, prompts, and CLI."""

from __future__ import annotations

import ast
import base64
import io
import json
import struct
from pathlib import Path

import pytest
from mcp.types import ImageContent, TextContent
from PIL import Image

from frfruit.cli import main as cli_main

pytestmark = pytest.mark.anyio


def _parse_text_json(result) -> dict:
    assert not result.is_error, f"Unexpected tool error: {result.content}"
    texts = [c.text for c in result.content if isinstance(c, TextContent)]
    assert texts, f"Expected TextContent in result: {result.content}"
    return json.loads(texts[0])


async def test_mcp_discovery_surfaces_29_tools_resources_and_prompts(
    mcp_session_and_client,
) -> None:
    _, client, _ = mcp_session_and_client

    tools = await client.list_tools()
    tool_names = {t.name for t in tools.tools}
    assert len(tool_names) == 29
    expected_tools = {
        "scene_new",
        "scene_open",
        "scene_save",
        "scene_inspect",
        "scene_undo_redo",
        "collection_manage",
        "create_primitive",
        "create_from_curve",
        "object_transform",
        "import_mesh",
        "mesh_edit",
        "mesh_boolean",
        "modifier_add",
        "modifier_manage",
        "mesh_inspect_topology",
        "node_catalog_list",
        "node_graph_set",
        "node_graph_patch",
        "spatial_measure",
        "spatial_place",
        "material_create_or_assign",
        "light_manage",
        "camera_manage",
        "render_viewport",
        "render_views",
        "export_scene",
        "timeline_manage",
        "keyframe_manage",
        "render_animation",
    }
    assert expected_tools == tool_names

    resources = await client.list_resources()
    uris = {str(r.uri) for r in resources.resources}
    assert {
        "frfruit://scene/summary",
        "frfruit://scene/document",
        "frfruit://catalog/nodes",
        "frfruit://catalog/materials",
    }.issubset(uris)

    templates = await client.list_resource_templates()
    assert any(
        "frfruit://object/{object_id}" in str(rt.uri_template)
        for rt in templates.resource_templates
    )

    prompts = await client.list_prompts()
    assert {p.name for p in prompts.prompts} == {
        "build_3d_asset",
        "fix_geometry_issues",
        "animate_turntable",
    }

    prompt_res = await client.get_prompt(
        "build_3d_asset", {"concept": "stylized red apple", "style": "stylized"}
    )
    assert prompt_res.messages

    mat_res = await client.read_resource("frfruit://catalog/materials")
    assert "apple_skin" in str(mat_res.contents[0])


async def test_mcp_full_ai_workflow_construct_place_render_animate_export(
    mcp_session_and_client,
) -> None:
    _, client, workspace = mcp_session_and_client

    # 1. Reset scene & verify node catalog tool
    r_new = _parse_text_json(await client.call_tool("scene_new", {"name": "FrFruitHero"}))
    assert r_new["scene_name"] == "FrFruitHero"

    cat = _parse_text_json(await client.call_tool("node_catalog_list", {"category": "scatter"}))
    assert {n["type"] for n in cat["nodes"]} == {"DistributePointsOnFaces", "InstanceOnPoints"}

    # 2. Build plate (cylinder + bevel modifier), lathe apple body, swept curved stem
    plate = _parse_text_json(
        await client.call_tool(
            "create_primitive",
            {
                "primitive": "cylinder",
                "name": "CeramicPlate",
                "radius": 1.15,
                "radius_top": 1.35,
                "height": 0.16,
                "segments": 28,
            },
        )
    )
    _parse_text_json(
        await client.call_tool(
            "modifier_add",
            {
                "object_id": plate["object_id"],
                "modifier_type": "bevel",
                "params": {"width": 0.025, "segments": 1},
            },
        )
    )

    apple = _parse_text_json(
        await client.call_tool(
            "create_from_curve",
            {
                "mode": "revolve_lathe",
                "name": "AppleBody",
                "points": [
                    [0.0, 0.08],
                    [0.32, 0.02],
                    [0.56, 0.30],
                    [0.58, 0.65],
                    [0.42, 0.92],
                    [0.14, 0.96],
                    [0.0, 0.85],
                ],
                "segments": 28,
                "location": [0.0, 0.0, 4.0],
            },
        )
    )
    assert apple["stats"]["is_watertight"] is True

    _parse_text_json(
        await client.call_tool(
            "create_from_curve",
            {
                "mode": "sweep_tube",
                "name": "AppleStem",
                "points": [[0.0, 0.0, 0.0], [0.02, 0.01, 0.16], [0.10, 0.04, 0.32]],
                "radius": 0.038,
                "taper_end": 0.6,
                "segments": 12,
                "location": [0.0, 0.0, 8.0],
            },
        )
    )

    # 3. Smart spatial placement: ground plate -> place apple on plate -> snap stem to apple top
    _parse_text_json(
        await client.call_tool(
            "spatial_place", {"action": "ground", "object_id": "CeramicPlate", "z": 0.0}
        )
    )
    _parse_text_json(
        await client.call_tool(
            "spatial_place",
            {
                "action": "place_on_top",
                "object_id": "AppleBody",
                "base_object_id": "CeramicPlate",
                "offset": 0.002,
            },
        )
    )
    _parse_text_json(
        await client.call_tool(
            "spatial_place",
            {
                "action": "snap_to_surface",
                "object_id": "AppleStem",
                "base_object_id": "AppleBody",
                "direction": [0.0, 0.0, -1.0],
                "gap": -0.02,
            },
        )
    )

    # Check procedural scatter + patch DAG
    dots = _parse_text_json(
        await client.call_tool(
            "node_graph_set",
            {
                "object_id_or_name": "FruitDrops",
                "nodes": [
                    {"id": "ring", "type": "CurveCircle", "params": {"radius": 0.9, "segments": 8}},
                    {
                        "id": "orb",
                        "type": "PrimitiveSphere",
                        "params": {"radius": 0.06, "segments": 10},
                    },
                    {"id": "inst", "type": "InstanceOnPoints", "params": {"scale": 1.0}},
                    {"id": "out", "type": "GroupOutput"},
                ],
                "links": [
                    {
                        "from_node": "ring",
                        "from_socket": "geometry",
                        "to_node": "inst",
                        "to_socket": "points",
                    },
                    {
                        "from_node": "orb",
                        "from_socket": "geometry",
                        "to_node": "inst",
                        "to_socket": "instance",
                    },
                    {
                        "from_node": "inst",
                        "from_socket": "geometry",
                        "to_node": "out",
                        "to_socket": "geometry",
                    },
                ],
                "location": [0.0, 0.0, 0.22],
            },
        )
    )
    assert dots["node_count"] == 4
    patched = _parse_text_json(
        await client.call_tool(
            "node_graph_patch",
            {
                "object_id": dots["object_id"],
                "upsert_nodes": [
                    {"id": "ring", "type": "CurveCircle", "params": {"radius": 1.0, "segments": 10}}
                ],
            },
        )
    )
    assert patched["stats"]["face_count"] > 0

    # Check intersections on non-touching items (plate vs drops should have zero collision)
    col_res = _parse_text_json(
        await client.call_tool(
            "spatial_measure",
            {
                "action": "check_intersections",
                "object_ids": ["CeramicPlate", "AppleBody", "FruitDrops"],
            },
        )
    )
    assert col_res["intersection_count"] == 0

    # 4. Materials, lighting, auto-frame camera
    _parse_text_json(
        await client.call_tool(
            "material_create_or_assign",
            {
                "action": "create_and_assign",
                "preset": "glossy_ceramic",
                "object_id": "CeramicPlate",
            },
        )
    )
    _parse_text_json(
        await client.call_tool(
            "material_create_or_assign",
            {"action": "create_and_assign", "preset": "apple_skin", "object_id": "AppleBody"},
        )
    )
    _parse_text_json(
        await client.call_tool(
            "material_create_or_assign",
            {"action": "create_and_assign", "preset": "wood", "object_id": "AppleStem"},
        )
    )
    _parse_text_json(
        await client.call_tool(
            "material_create_or_assign",
            {"action": "create_and_assign", "preset": "emissive_amber", "object_id": "FruitDrops"},
        )
    )
    _parse_text_json(await client.call_tool("light_manage", {"action": "setup_studio_3point"}))
    _parse_text_json(
        await client.call_tool("camera_manage", {"action": "auto_frame", "direction": "iso"})
    )

    # Read object detail resource
    obj_res = await client.read_resource(f"frfruit://object/{apple['object_id']}")
    assert "AppleBody" in str(obj_res.contents[0])

    # 5. AI Visual Feedback via render_views and render_viewport
    views_res = await client.call_tool(
        "render_views",
        {"mode": "pbr", "tile_width": 200, "tile_height": 150, "save_path": "renders/views.png"},
    )
    assert not views_res.is_error
    imgs = [c for c in views_res.content if isinstance(c, ImageContent)]
    assert len(imgs) == 1
    assert imgs[0].mime_type == "image/png"
    decoded = Image.open(io.BytesIO(base64.b64decode(imgs[0].data)))
    assert decoded.size == (400, 300)
    assert (workspace / "renders" / "views.png").exists()

    vp_res = await client.call_tool(
        "render_viewport",
        {
            "mode": "wireframe",
            "width": 240,
            "height": 180,
            "show_bounds": True,
            "show_labels": True,
        },
    )
    assert any(isinstance(c, ImageContent) for c in vp_res.content)

    # 6. Undo / Redo / Snapshot
    _parse_text_json(
        await client.call_tool(
            "scene_undo_redo", {"action": "create_snapshot", "snapshot_name": "before_move"}
        )
    )
    _parse_text_json(
        await client.call_tool(
            "object_transform", {"object_id": "AppleBody", "translate_delta": [0.0, 0.0, 2.0]}
        )
    )
    _parse_text_json(await client.call_tool("scene_undo_redo", {"action": "undo"}))
    _parse_text_json(await client.call_tool("scene_undo_redo", {"action": "redo"}))
    _parse_text_json(
        await client.call_tool(
            "scene_undo_redo", {"action": "restore_snapshot", "snapshot_name": "before_move"}
        )
    )

    # 7. Animation keyframes & render_animation (timeline GIF)
    _parse_text_json(
        await client.call_tool(
            "timeline_manage",
            {"action": "set_range", "fps": 24, "frame_start": 1, "frame_end": 16},
        )
    )
    _parse_text_json(
        await client.call_tool(
            "keyframe_manage",
            {
                "action": "set",
                "target_id": "AppleBody",
                "property_path": "rotation_deg.z",
                "frame": 1,
                "value": 0.0,
            },
        )
    )
    _parse_text_json(
        await client.call_tool(
            "keyframe_manage",
            {
                "action": "set",
                "target_id": "AppleBody",
                "property_path": "rotation_deg.z",
                "frame": 16,
                "value": 180.0,
            },
        )
    )
    anim_res = await client.call_tool(
        "render_animation",
        {
            "kind": "timeline",
            "steps": 4,
            "width": 160,
            "height": 128,
            "save_gif_path": "renders/apple.gif",
            "save_filmstrip_path": "renders/strip.png",
        },
    )
    assert not anim_res.is_error
    assert (workspace / "renders" / "apple.gif").read_bytes().startswith(b"GIF8")

    # 8. Save project, Export GLB & Blender .py, Re-open
    proj_path = workspace / "apple.frf.json"
    _parse_text_json(await client.call_tool("scene_save", {"path": "apple.frf.json"}))
    assert proj_path.exists()

    glb_res = _parse_text_json(await client.call_tool("export_scene", {"path": "out/apple.glb"}))
    glb_bytes = Path(glb_res["path"]).read_bytes()
    assert glb_bytes[:4] == b"glTF"
    jlen, _ = struct.unpack("<II", glb_bytes[12:20])
    assert "animations" in json.loads(glb_bytes[20 : 20 + jlen])

    py_res = _parse_text_json(await client.call_tool("export_scene", {"path": "out/apple_bpy.py"}))
    ast.parse(Path(py_res["path"]).read_text(encoding="utf-8"))

    _parse_text_json(await client.call_tool("scene_open", {"path": "apple.frf.json"}))

    # 9. CLI smoke check on the saved project file
    cli_info_rc = cli_main(["info", str(proj_path)])
    assert cli_info_rc == 0
    cli_render_rc = cli_main(
        [
            "render",
            str(proj_path),
            "-o",
            str(workspace / "cli.png"),
            "--multiview",
            "--width",
            "160",
            "--height",
            "120",
        ]
    )
    assert cli_render_rc == 0
    assert (workspace / "cli.png").read_bytes().startswith(b"\x89PNG")
    cli_exp_rc = cli_main(["export", str(proj_path), "-o", str(workspace / "cli.stl")])
    assert cli_exp_rc == 0
    assert (workspace / "cli.stl").exists()


async def test_mcp_error_handling_and_workspace_sandbox(
    mcp_session_and_client,
) -> None:
    _, client, _ = mcp_session_and_client

    # Path traversal outside workspace must fail with ToolError
    bad_save = await client.call_tool("scene_save", {"path": "../../outside.frf.json"})
    assert bad_save.is_error is True

    # Unknown object must fail gracefully with ToolError
    missing_obj = await client.call_tool("mesh_inspect_topology", {"object_id": "does_not_exist"})
    assert missing_obj.is_error is True

    # Invalid boolean on open plane must fail gracefully with ToolError
    _parse_text_json(
        await client.call_tool("create_primitive", {"primitive": "cube", "name": "C1"})
    )
    _parse_text_json(
        await client.call_tool("create_primitive", {"primitive": "plane", "name": "P1"})
    )
    bad_csg = await client.call_tool(
        "mesh_boolean",
        {"object_id": "C1", "target_object_id": "P1", "operation": "difference"},
    )
    assert bad_csg.is_error is True
