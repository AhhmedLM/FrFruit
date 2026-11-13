"""Reusable AI workflow MCP Prompts for 3D modeling, inspection, and animation."""

from __future__ import annotations

from typing import Annotated

from mcp.server import MCPServer
from pydantic import Field


def register_prompts(mcp: MCPServer) -> None:
    @mcp.prompt(title="Build 3D Asset (AI Workflow)")
    def build_3d_asset(
        concept: Annotated[str, Field(description="What 3D object or scene to construct.")],
        style: Annotated[
            str, Field(description="Visual style (stylized, low-poly, CAD, realistic).")
        ] = "stylized",
    ) -> str:
        """Step-by-step workflow guiding AI agents to build, inspect, and export 3D assets."""
        return (
            f"Build a {style} 3D asset in FrFruit for: {concept}\n\n"
            "Follow this AI-first 3D workflow:\n"
            "1. Start clean with `scene_new` (or inspect existing objects via `scene_inspect`).\n"
            "2. Block out volumes via `create_primitive` or `create_from_curve` "
            "(`revolve_lathe`, `sweep_tube`, `extrude_2d`).\n"
            "3. Refine geometry via `mesh_edit`, `mesh_boolean`, "
            "`modifier_add`, or `node_graph_set`.\n"
            "4. Position parts with `spatial_place` (`place_on_top`, "
            "`attach_to_side`, `snap_to_surface`, `ground`) instead of guessing.\n"
            "5. Run `spatial_measure(action='check_intersections')` and call "
            "`render_views(mode='pbr')` plus `render_views(mode='wireframe')`.\n"
            "6. Fix floating/clipping parts, assign PBR materials (`material_create_or_assign`), "
            "and configure `light_manage(action='setup_studio_3point')`.\n"
            "7. Take a final `render_viewport` shot and export with "
            "`export_scene` (.glb / .frf.json)."
        )

    @mcp.prompt(title="Diagnose & Fix Geometry Issues")
    def fix_geometry_issues(
        focus: Annotated[
            str, Field(description="Specific object or scene problem to investigate.")
        ] = "all",
    ) -> str:
        """Diagnostic workflow to detect intersections, inverted normals, or floating meshes."""
        return (
            f"Inspect and repair geometry issues (focus: {focus}):\n"
            "1. Call `spatial_measure(action='check_intersections')` and "
            "`render_views(mode='collision')`.\n"
            "2. Call `render_views(mode='normals')` and `render_views(mode='wireframe')` "
            "to inspect topology and face orientation.\n"
            "3. Fix inverted normals or non-manifold edges with "
            "`mesh_edit(operation='repair_manifold')` or `recompute_normals`.\n"
            "4. Fix floating or overlapping parts using `spatial_place`."
        )

    @mcp.prompt(title="Animate & Render Turntable")
    def animate_turntable(
        subject: Annotated[
            str, Field(description="Object ID/name to animate or showcase.")
        ] = "all",
    ) -> str:
        """Workflow to keyframe an animation and render a filmstrip/GIF preview."""
        return (
            f"Create an animation showcase for {subject}:\n"
            "1. Configure timeline via `timeline_manage(action='set_range', "
            "fps=24, frame_start=1, frame_end=48)`.\n"
            "2. Add keyframes via `keyframe_manage(action='set', ...)` on "
            "`rotation_deg.z` or `location.z`.\n"
            "3. Inspect motion with `render_animation(kind='timeline', steps=8)` "
            "or `kind='turntable'`.\n"
            "4. Export animated `.glb` via `export_scene`."
        )
