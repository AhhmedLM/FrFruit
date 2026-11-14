"""FrFruit MCPServer assembly and runner."""

from __future__ import annotations

from pathlib import Path

from mcp.server import MCPServer

from frfruit.mcp_server.context import FrFruitSession
from frfruit.mcp_server.prompts import register_prompts
from frfruit.mcp_server.resources import register_resources
from frfruit.mcp_server.tools_anim import register_anim_tools
from frfruit.mcp_server.tools_modeling import register_modeling_tools
from frfruit.mcp_server.tools_nodes import register_node_tools
from frfruit.mcp_server.tools_render import register_render_tools
from frfruit.mcp_server.tools_scene import register_scene_tools
from frfruit.mcp_server.tools_sim_fx import register_sim_fx_tools
from frfruit.mcp_server.tools_spatial import register_spatial_tools

SERVER_INSTRUCTIONS = (
    "FrFruit is an AI-first headless 3D creation engine (Blender for AI) over MCP. "
    "Build scenes with primitives, curves/lathe/sweep, semantic Edit Mesh, CSG booleans, "
    "non-destructive modifiers, JSON procedural node graphs, rigid-body/XPBD cloth physics, "
    "skeletal armatures, and online PBR texture libraries (Poly Haven / ambientCG / procedural). "
    "Render with either fast Software rasterizer or BVH PathTracer ('raytrace') plus HDR "
    "post-processing (SSAO, Bloom, DOF, ACES), inspect renders via render_views/render_viewport, "
    "record MP4/WebM/AVI/GIF videos via render_video_file, and export game-ready animated FBX."
)


def create_server(workspace_root: str | Path | None = None) -> tuple[MCPServer, FrFruitSession]:
    """Create a configured FrFruit MCPServer bound to a dedicated FrFruitSession."""
    session = FrFruitSession(workspace_root=workspace_root)
    server = MCPServer(
        "FrFruit",
        title="FrFruit AI 3D Engine",
        description=(
            "AI-first headless 3D modeling, physics/cloth, rigging, path-traced rendering, "
            "video/FBX export, and MCP engine."
        ),
        instructions=SERVER_INSTRUCTIONS,
        version="0.2.0",
    )

    def get_session() -> FrFruitSession:
        return session

    register_scene_tools(server, get_session)
    register_modeling_tools(server, get_session)
    register_node_tools(server, get_session)
    register_spatial_tools(server, get_session)
    register_render_tools(server, get_session)
    register_anim_tools(server, get_session)
    register_sim_fx_tools(server, get_session)
    register_resources(server, get_session)
    register_prompts(server)
    return server, session


mcp, default_session = create_server()


def main() -> None:
    """Entry point for `frfruit-mcp` over stdio."""
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
