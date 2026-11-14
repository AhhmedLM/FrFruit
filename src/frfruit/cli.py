"""CLI for FrFruit: serve, app, render, video, export, textures, physics, info."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path

from frfruit.geometry.spatial import compute_scene_bounds, evaluate_scene_meshes
from frfruit.io.exporters import export_scene_file
from frfruit.io.project import load_project, save_project
from frfruit.mcp_server.server import default_session, mcp
from frfruit.physics.solver import bake_scene_physics
from frfruit.render.video import render_video
from frfruit.render.viewport import render_multiview_sheet, render_single_view
from frfruit.textures.fetcher import download_pbr_texture_pack, search_online_textures


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="frfruit",
        description="FrFruit: AI-first headless 3D creation engine, workstation UI, & MCP server.",
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

    p_app = sub.add_parser("app", help="Launch FrFruit Studio Desktop App + Co-Hosted MCP Bridge.")
    p_app.add_argument("--host", default="127.0.0.1", help="Bind host for local Studio server.")
    p_app.add_argument("--port", type=int, default=8765, help="Bind port for local Studio server.")
    p_app.add_argument(
        "--workspace", default=None, help="Workspace directory for scenes & exports."
    )
    p_app.add_argument(
        "--project", default=None, help="Optional .frf.json project file to preload."
    )
    p_app.add_argument(
        "--no-window",
        action="store_true",
        help="Run headless server/MCP bridge without opening a native window/browser.",
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
    p_render.add_argument(
        "--engine",
        choices=["software", "raytrace"],
        default=None,
        help="Render backend override (software or raytrace).",
    )
    p_render.add_argument("--samples", type=int, default=2, help="Raytracer samples per pixel.")
    p_render.add_argument("--multiview", action="store_true", help="Render 2x2 multi-view sheet.")
    p_render.add_argument("--width", type=int, default=640, help="Render width (or tile width).")
    p_render.add_argument("--height", type=int, default=480, help="Render height (or tile height).")

    p_video = sub.add_parser("video", help="Render a .frf.json project to MP4/WebM/AVI/GIF video.")
    p_video.add_argument("project", help="Input .frf.json project file.")
    p_video.add_argument(
        "-o", "--output", required=True, help="Output video (.mp4/.webm/.avi/.gif)."
    )
    p_video.add_argument(
        "--kind",
        choices=["timeline", "turntable", "camera_path"],
        default="turntable",
        help="Video sequence type.",
    )
    p_video.add_argument("--fps", type=int, default=None, help="Frames per second.")
    p_video.add_argument("--frame-start", type=int, default=None, help="Start frame.")
    p_video.add_argument("--frame-end", type=int, default=None, help="End frame.")
    p_video.add_argument("--width", type=int, default=640, help="Frame width.")
    p_video.add_argument("--height", type=int, default=480, help="Frame height.")
    p_video.add_argument(
        "--mode",
        choices=["pbr", "clay", "wireframe", "normals", "depth", "collision"],
        default="pbr",
    )
    p_video.add_argument("--engine", choices=["software", "raytrace"], default=None)
    p_video.add_argument("--samples", type=int, default=2)

    p_export = sub.add_parser(
        "export", help="Export a .frf.json project to FBX/GLB/GLTF/OBJ/STL/PLY/PY."
    )
    p_export.add_argument("project", help="Input .frf.json project file.")
    p_export.add_argument("-o", "--output", required=True, help="Output file path.")
    p_export.add_argument("--format", default=None, help="Explicit output format override.")
    p_export.add_argument(
        "--target-engine",
        choices=["generic", "unreal", "unity", "godot"],
        default="generic",
        help="Game engine axis/rig convention for FBX export.",
    )

    p_tex = sub.add_parser("textures", help="Search or download PBR texture packs.")
    p_tex.add_argument("query", nargs="?", default="", help="Search keywords or asset ID.")
    p_tex.add_argument(
        "--source",
        choices=["all", "polyhaven", "ambientcg", "procedural"],
        default="all",
    )
    p_tex.add_argument("--download", default=None, help="Download specific asset_id to --out dir.")
    p_tex.add_argument("--resolution", choices=["1k", "2k", "4k"], default="1k")
    p_tex.add_argument("-o", "--out", default="./textures", help="Target directory for downloads.")

    p_phys = sub.add_parser(
        "physics", help="Bake rigid-body & XPBD cloth physics into a .frf.json."
    )
    p_phys.add_argument("project", help="Input .frf.json project file.")
    p_phys.add_argument(
        "-o", "--output", default=None, help="Output project file (default in-place)."
    )
    p_phys.add_argument("--frame-start", type=int, default=None)
    p_phys.add_argument("--frame-end", type=int, default=None)

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

    if args.command == "app":
        from frfruit.desktop.win_shell import launch_studio

        launch_studio(
            host=args.host,
            port=args.port,
            workspace=args.workspace,
            project_path=args.project,
            open_window=not args.no_window,
        )
        return 0

    if args.command == "render":
        scene = load_project(args.project)
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        if args.multiview:
            png, meta = render_multiview_sheet(
                scene,
                tile_width=args.width,
                tile_height=args.height,
                mode=args.mode,
                engine=args.engine,
                samples=args.samples,
            )
        else:
            png, meta = render_single_view(
                scene,
                width=args.width,
                height=args.height,
                mode=args.mode,
                engine=args.engine,
                samples=args.samples,
            )
        out.write_bytes(png)
        print(json.dumps({"status": "ok", "output": str(out), **meta}, indent=2))
        return 0

    if args.command == "video":
        scene = load_project(args.project)
        res = render_video(
            scene,
            out_path=args.output,
            kind=args.kind,
            frame_start=args.frame_start,
            frame_end=args.frame_end,
            fps=args.fps,
            width=args.width,
            height=args.height,
            mode=args.mode,
            engine=args.engine,
            samples=args.samples,
        )
        print(json.dumps({"status": "ok", **res}, indent=2))
        return 0

    if args.command == "export":
        scene = load_project(args.project)
        res = export_scene_file(
            scene,
            args.output,
            fmt=args.format,
            target_engine=args.target_engine,
        )
        print(json.dumps({"status": "ok", **res}, indent=2))
        return 0

    if args.command == "textures":
        if args.download:
            src = "polyhaven" if args.source == "all" else args.source
            res = download_pbr_texture_pack(
                asset_id=args.download,
                source=src,
                resolution=args.resolution,
                target_dir=args.out,
                fallback_to_procedural=True,
            )
            print(json.dumps(res, indent=2))
        else:
            hits = search_online_textures(query=args.query, source=args.source, limit=15)
            print(json.dumps({"count": len(hits), "results": hits}, indent=2))
        return 0

    if args.command == "physics":
        scene = load_project(args.project)
        res = bake_scene_physics(scene, frame_start=args.frame_start, frame_end=args.frame_end)
        out_p = args.output or args.project
        save_project(scene, out_p)
        print(json.dumps({"status": "ok", "saved": out_p, **res}, indent=2))
        return 0

    if args.command == "info":
        scene = load_project(args.project)
        evaluated = evaluate_scene_meshes(scene)
        bounds = compute_scene_bounds(evaluated)
        info = {
            "name": scene.name,
            "render_engine": scene.render_engine,
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
