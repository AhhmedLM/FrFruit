"""Command-line interface for FrFruit (`serve`, `render`, `export`, `info`)."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path

from frfruit.geometry.spatial import compute_scene_bounds, evaluate_scene_meshes
from frfruit.io.exporters import export_scene_file
from frfruit.io.project import load_project
from frfruit.mcp_server.server import default_session, mcp
from frfruit.render.viewport import render_multiview_sheet, render_single_view


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="frfruit",
        description="FrFruit: AI-first headless 3D creation engine & MCP server.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_serve = sub.add_parser("serve", help="Run the FrFruit MCP server.")
    p_serve.add_argument(
        "--transport",
        choices=["stdio", "streamable-http", "sse"],
        default="stdio",
        help="MCP transport protocol.",
    )
    p_serve.add_argument("--host", default="127.0.0.1", help="Bind host for HTTP transport.")
    p_serve.add_argument("--port", type=int, default=8000, help="Bind port for HTTP transport.")
    p_serve.add_argument(
        "--workspace",
        default=None,
        help="Workspace root directory for safe file reads/writes.",
    )

    p_render = sub.add_parser("render", help="Render a .frf.json project file to PNG.")
    p_render.add_argument("project", help="Input .frf.json project file.")
    p_render.add_argument("-o", "--output", required=True, help="Output PNG file path.")
    p_render.add_argument(
        "--mode",
        choices=["pbr", "clay", "wireframe", "normals", "depth", "collision"],
        default="pbr",
        help="Viewport shading mode.",
    )
    p_render.add_argument("--multiview", action="store_true", help="Render 2x2 multi-view sheet.")
    p_render.add_argument("--width", type=int, default=640, help="Render width (or tile width).")
    p_render.add_argument("--height", type=int, default=480, help="Render height (or tile height).")

    p_export = sub.add_parser("export", help="Export a .frf.json project to GLB/OBJ/STL/PLY/PY.")
    p_export.add_argument("project", help="Input .frf.json project file.")
    p_export.add_argument("-o", "--output", required=True, help="Output file path.")
    p_export.add_argument("--format", default=None, help="Explicit output format override.")

    p_info = sub.add_parser("info", help="Print JSON summary of a .frf.json project.")
    p_info.add_argument("project", help="Input .frf.json project file.")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "serve":
        if args.workspace:
            default_session.set_workspace_root(args.workspace)
        if args.transport == "stdio":
            mcp.run(transport="stdio")
        else:
            mcp.run(transport=args.transport, host=args.host, port=args.port)
        return 0

    if args.command == "render":
        scene = load_project(args.project)
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        if args.multiview:
            png, meta = render_multiview_sheet(
                scene, tile_width=args.width, tile_height=args.height, mode=args.mode
            )
        else:
            png, meta = render_single_view(
                scene, width=args.width, height=args.height, mode=args.mode
            )
        out.write_bytes(png)
        print(json.dumps({"status": "ok", "output": str(out), **meta}, indent=2))
        return 0

    if args.command == "export":
        scene = load_project(args.project)
        res = export_scene_file(scene, args.output, fmt=args.format)
        print(json.dumps({"status": "ok", **res}, indent=2))
        return 0

    if args.command == "info":
        scene = load_project(args.project)
        evaluated = evaluate_scene_meshes(scene)
        bounds = compute_scene_bounds(evaluated)
        info = {
            "name": scene.name,
            "objects": len(scene.objects),
            "evaluated_meshes": len(evaluated),
            "materials": len(scene.materials),
            "bounds": {"min": list(bounds.min_corner), "max": list(bounds.max_corner)},
        }
        print(json.dumps(info, indent=2))
        return 0

    return 1


if __name__ == "__main__":
    sys.exit(main())
