"""Read-only MCP Resources exposing scene state, catalogs, and individual object details."""

from __future__ import annotations

import json
from collections.abc import Callable

from mcp.server import MCPServer
from mcp.server.mcpserver.exceptions import ResourceNotFoundError

from frfruit.geometry.spatial import evaluate_object_mesh
from frfruit.mcp_server.context import FrFruitSession
from frfruit.procedural.nodes import list_node_catalog
from frfruit.render.materials import list_material_presets
from frfruit.textures.fetcher import search_online_textures


def register_resources(mcp: MCPServer, get_session: Callable[[], FrFruitSession]) -> None:
    @mcp.resource("frfruit://scene/summary", mime_type="application/json")
    def scene_summary_resource() -> str:
        """Compact overview of objects, materials, timeline, and undo status."""
        sess = get_session()
        return json.dumps(sess.scene_summary(), indent=2)

    @mcp.resource("frfruit://scene/document", mime_type="application/json")
    def scene_document_resource() -> str:
        """Complete serialized .frf.json SceneDocument."""
        sess = get_session()
        with sess.lock:
            return sess.scene.model_dump_json(indent=2)

    @mcp.resource("frfruit://scene/postprocess", mime_type="application/json")
    def scene_postprocess_resource() -> str:
        """Active render engine ('software' or 'raytrace') and PostProcessConfig."""
        sess = get_session()
        with sess.lock:
            return json.dumps(
                {
                    "render_engine": sess.scene.render_engine,
                    "post_process": sess.scene.post_process.model_dump(),
                },
                indent=2,
            )

    @mcp.resource("frfruit://catalog/nodes", mime_type="application/json")
    def node_catalog_resource() -> str:
        """Procedural node catalog schema for Geometry Nodes construction."""
        return json.dumps({"nodes": list_node_catalog()}, indent=2)

    @mcp.resource("frfruit://catalog/materials", mime_type="application/json")
    def material_presets_resource() -> str:
        """Built-in PBR material presets."""
        return json.dumps({"presets": list_material_presets()}, indent=2)

    @mcp.resource("frfruit://catalog/textures", mime_type="application/json")
    def texture_catalog_resource() -> str:
        """Curated offline & procedural PBR texture catalog alongside remote API sources."""
        hits = search_online_textures(query="", source="procedural", limit=32)
        return json.dumps(
            {
                "remote_providers": ["polyhaven", "ambientcg"],
                "procedural_styles": hits,
            },
            indent=2,
        )

    @mcp.resource("frfruit://object/{object_id}", mime_type="application/json")
    def object_detail_resource(object_id: str) -> str:
        """Detailed metadata and evaluated world topology/bounds for a single scene object."""
        sess = get_session()
        with sess.lock:
            obj = sess.scene.resolve_object(object_id)
            if obj is None:
                raise ResourceNotFoundError(f"frfruit://object/{object_id}")
            payload = obj.model_dump(exclude={"mesh", "vertex_cache"})
            if obj.kind in ("mesh", "procedural"):
                ev = evaluate_object_mesh(sess.scene, obj.id)
                payload["evaluated_stats"] = ev.world_mesh.stats()
            if obj.vertex_cache:
                payload["vertex_cache_frames"] = sorted(obj.vertex_cache.keys())
            return json.dumps(payload, indent=2)
