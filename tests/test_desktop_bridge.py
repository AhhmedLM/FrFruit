"""Integration tests for Desktop Workstation Starlette API, WebSockets, and Live Co-Hosted MCP."""

from __future__ import annotations

import json
from pathlib import Path

from starlette.testclient import TestClient

from frfruit.desktop.server import create_studio_app


def test_desktop_api_websocket_and_cohosted_mcp_bridge(tmp_path: Path) -> None:
    app, sess, _mcp = create_studio_app(workspace_root=tmp_path, host="127.0.0.1", port=8765)

    with TestClient(app) as client:
        # 1. Static workstation UI HTML served at /
        home = client.get("/")
        assert home.status_code == 200
        assert "FRFRUIT" in home.text
        assert "/viewport_gl.js" in home.text

        # 2. Load showcase preset & query state/geometry
        demo = client.post("/api/scene/demo", json={"preset": "pbr_showcase"})
        assert demo.status_code == 200
        st = client.get("/api/state").json()
        assert st["scene_name"] == "PBRMaterialsShowcase"
        assert st["mcp_tools_count"] == 34

        geo = client.get("/api/geometry").json()
        assert len(geo["meshes"]) >= 3
        assert len(geo["meshes"][0]["positions"]) > 0

        # 3. CPU Render endpoint produces PNG
        r_png = client.get("/api/render?width=160&height=120&mode=pbr&engine=software")
        assert r_png.status_code == 200
        assert r_png.content.startswith(b"\x89PNG")

        # 4. WebSocket live telemetry push when MCP session mutates!
        with client.websocket_connect("/ws/events") as ws:
            hello = ws.receive_json()
            assert hello["event"] == "connected"

            add_res = client.post(
                "/api/objects/primitive",
                json={"primitive": "cube", "name": "LiveCube", "location": [1.0, 0.0, 0.5]},
            )
            assert add_res.status_code == 200
            oid = add_res.json()["object_id"]

            # Expect live mutation broadcast over WebSocket
            evt = ws.receive_json()
            assert evt["event"] in ("scene_mutated", "object_added")

        # 5. Textures apply (procedural) + Post-FX + Physics + FBX Export + Safe Download
        tex_r = client.post(
            "/api/textures/apply",
            json={"asset_id": "stone", "source": "procedural", "object_id": oid},
        )
        assert tex_r.status_code == 200
        assert Path(tex_r.json()["manifest"]["maps"]["albedo"]).is_file()

        pp_r = client.post(
            "/api/postprocess",
            json={"render_engine": "raytrace", "preset": "architectural"},
        )
        assert pp_r.status_code == 200
        assert pp_r.json()["render_engine"] == "raytrace"

        exp_r = client.post(
            "/api/export",
            json={"path": "out/studio_model.fbx", "format": "fbx", "target_engine": "unreal"},
        )
        assert exp_r.status_code == 200
        exp_data = exp_r.json()
        assert exp_data["format"] == "fbx"
        assert Path(exp_data["path"]).read_bytes().startswith(b"Kaydara FBX Binary")

        dl_r = client.get("/api/files/download?path=out/studio_model.fbx")
        assert dl_r.status_code == 200
        assert dl_r.content.startswith(b"Kaydara FBX Binary")

        # Path traversal outside workspace rejected
        bad_dl = client.get("/api/files/download?path=../../etc/passwd")
        assert bad_dl.status_code == 404

        # 6. Live MCP Info & Co-Hosted Streamable HTTP /mcp initialize handshake
        mcp_info = client.get("/api/mcp/info").json()
        assert mcp_info["tools_count"] == 34
        assert mcp_info["streamable_http_url"] == "http://127.0.0.1:8765/mcp"

        init_req = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {
                "protocolVersion": "2025-03-26",
                "capabilities": {},
                "clientInfo": {"name": "test-client", "version": "1.0"},
            },
        }
        mcp_resp = client.post(
            "/mcp",
            json=init_req,
            headers={"Accept": "application/json, text/event-stream"},
        )
        assert mcp_resp.status_code == 200
        body_text = mcp_resp.text
        # Streamable HTTP responds either directly with JSON or SSE data frame containing serverInfo
        if mcp_resp.headers.get("content-type", "").startswith("application/json"):
            payload = json.loads(body_text)
            assert payload["result"]["serverInfo"]["name"] == "FrFruit"
        else:
            assert "FrFruit" in body_text


def test_desktop_studio_edit_mode_raycast_and_face_material_api(tmp_path: Path) -> None:
    app, sess, _mcp = create_studio_app(workspace_root=tmp_path, host="127.0.0.1", port=8766)

    with TestClient(app) as client:
        # 1. Verify UI includes Edit Mode controls and 7-tab workspace
        home = client.get("/")
        assert home.status_code == 200
        assert "btn-mode-edit" in home.text
        assert "edit-elem-switch" in home.text
        assert 'data-tab="edit"' in home.text

        # 2. Load showcase or create a cube + two materials
        client.post("/api/scene/new", json={"name": "StudioEditTest"})
        add_res = client.post(
            "/api/objects/primitive",
            json={"primitive": "cube", "name": "StudioCube", "location": [0.0, 0.0, 0.5]},
        )
        assert add_res.status_code == 200
        oid = add_res.json()["object_id"]

        # Create base and face-accent materials directly on the session
        with sess.lock:
            from frfruit.core.scene import MaterialSpec

            sess.scene.materials["mat_base"] = MaterialSpec(
                id="mat_base", name="BaseSlate", base_color=(0.25, 0.28, 0.35, 1.0)
            )
            sess.scene.materials["mat_face"] = MaterialSpec(
                id="mat_face",
                name="FaceGold",
                base_color=(0.95, 0.72, 0.18, 1.0),
                metallic=0.9,
                roughness=0.2,
            )
            sess.scene.objects[oid].material_id = "mat_base"

        # 3. Query /api/objects/{id}/elements with semantic selection
        elem_r = client.get(f"/api/objects/{oid}/elements?selection_mode=top")
        assert elem_r.status_code == 200
        elem_data = elem_r.json()
        assert elem_data["vertex_count"] == 8
        assert elem_data["face_count"] == 12
        assert len(elem_data["selected_faces"]) == 2
        assert len(elem_data["edges"]) == 18

        # 4. Raycast top face from straight above [0, 0, 5] looking down [0, 0, -1]
        ray_r = client.post(
            f"/api/objects/{oid}/raycast_face",
            json={"origin": [0.0, 0.0, 5.0], "direction": [0.0, 0.0, -1.0]},
        )
        assert ray_r.status_code == 200
        ray_data = ray_r.json()
        assert ray_data["hit"] is True
        picked_face = ray_data["face_index"]
        assert isinstance(picked_face, int)
        assert picked_face in elem_data["selected_faces"]

        # 5. Assign gold material to ONLY that single picked face
        assign_r = client.post(
            "/api/materials/assign_faces",
            json={
                "object_id": oid,
                "material_id": "mat_face",
                "face_index": picked_face,
            },
        )
        assert assign_r.status_code == 200
        assign_data = assign_r.json()
        assert assign_data["faces"] == [picked_face]
        assert assign_data["count"] == 1

        # Verify /api/state reflects face_material_count=1 and /api/geometry splits sub-meshes
        st = client.get("/api/state").json()
        obj_state = next(o for o in st["objects"] if o["id"] == oid)
        assert obj_state["face_material_count"] == 1
        assert obj_state["face_material_ids"][str(picked_face)] == "mat_face"

        geo = client.get("/api/geometry").json()
        cube_submeshes = [m for m in geo["meshes"] if m["id"] == oid]
        assert len(cube_submeshes) == 2
        sub_mat_ids = {m["material"]["id"] for m in cube_submeshes}
        assert sub_mat_ids == {"mat_base", "mat_face"}

        # 6. Execute Edit Mode operation via /api/objects/{id}/edit
        edit_r = client.post(
            f"/api/objects/{oid}/edit",
            json={
                "operation": "extrude_faces",
                "selection_mode": "top",
                "distance": 0.4,
                "scale": 0.7,
            },
        )
        assert edit_r.status_code == 200
        edit_data = edit_r.json()
        assert edit_data["stats"]["face_count"] > 12
        assert edit_data["face_material_count"] >= 1

        # Clear per-face override via /api/materials/assign_faces with clear=true
        clr_r = client.post(
            "/api/materials/assign_faces",
            json={"object_id": oid, "clear": True},
        )
        assert clr_r.status_code == 200
        geo_after_clr = client.get("/api/geometry").json()
        assert len([m for m in geo_after_clr["meshes"] if m["id"] == oid]) == 1
