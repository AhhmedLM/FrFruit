"""MCP tools for timeline control, keyframing, and animation/turntable rendering."""

from __future__ import annotations

import base64
import json
from collections.abc import Callable
from typing import Annotated, Any, Literal

from mcp.server import MCPServer
from mcp.server.mcpserver.exceptions import ToolError
from mcp.types import ImageContent, TextContent, ToolAnnotations
from pydantic import Field

from frfruit.animation.evaluator import apply_animation_frame
from frfruit.animation.timeline import FCurveTrack, Keyframe
from frfruit.mcp_server.context import FrFruitSession
from frfruit.render.software import RenderMode
from frfruit.render.viewport import render_animation_sequence


def register_anim_tools(mcp: MCPServer, get_session: Callable[[], FrFruitSession]) -> None:
    @mcp.tool(
        title="Manage Animation Timeline",
        annotations=ToolAnnotations(open_world_hint=False),
    )
    def timeline_manage(
        action: Annotated[
            Literal["get", "set_range", "set_frame"],
            Field(description="Timeline action."),
        ] = "get",
        fps: Annotated[int | None, Field(ge=1, le=120)] = None,
        frame_start: Annotated[int | None, Field(description="Start frame.")] = None,
        frame_end: Annotated[int | None, Field(description="End frame.")] = None,
        current_frame: Annotated[
            int | None, Field(description="Frame to evaluate onto scene.")
        ] = None,
    ) -> dict[str, Any]:
        """Inspect or update timeline FPS/range, or evaluate scene transforms at a target frame."""
        sess = get_session()
        with sess.lock:
            tl = sess.scene.timeline
            if action == "get":
                return tl.model_dump()

            sess.push_history(f"timeline:{action}")
            if action == "set_range":
                if fps is not None:
                    tl.fps = fps
                if frame_start is not None:
                    tl.frame_start = frame_start
                if frame_end is not None:
                    tl.frame_end = frame_end
                if tl.frame_end < tl.frame_start:
                    tl.frame_end = tl.frame_start
                return {"status": "ok", "timeline": tl.model_dump(exclude={"tracks"})}

            if current_frame is None:
                raise ToolError("current_frame is required for action='set_frame'.")
            sess.scene = apply_animation_frame(sess.scene, current_frame)
            return {
                "status": "ok",
                "current_frame": sess.scene.timeline.current_frame,
                "track_count": len(sess.scene.timeline.tracks),
            }

    @mcp.tool(
        title="Manage Keyframes",
        annotations=ToolAnnotations(open_world_hint=False),
    )
    def keyframe_manage(
        action: Annotated[
            Literal["set", "delete", "list"], Field(description="Keyframe operation.")
        ] = "set",
        target_id: Annotated[
            str | None, Field(description="Object, light, camera, or material ID/name.")
        ] = None,
        property_path: Annotated[
            str | None,
            Field(
                description=(
                    "Property path such as 'location.x', 'rotation_deg.z', "
                    "'scale.x', 'intensity', 'fov_deg', 'emission_strength'."
                )
            ),
        ] = None,
        frame: Annotated[int | None, Field(description="Frame number.")] = None,
        value: Annotated[float | None, Field(description="Scalar keyframe value.")] = None,
        interpolation: Annotated[
            Literal["constant", "linear", "bezier"], Field(description="Interpolation mode.")
        ] = "bezier",
        easing: Annotated[
            Literal["auto", "ease_in", "ease_out", "ease_in_out"],
            Field(description="Bezier easing curve."),
        ] = "ease_in_out",
    ) -> dict[str, Any]:
        """Insert, update, delete, or list keyframes on objects, lights, cams, or materials."""
        sess = get_session()
        with sess.lock:
            tl = sess.scene.timeline
            if action == "list":
                tracks = tl.tracks
                if target_id:
                    obj = sess.scene.resolve_object(target_id)
                    mat = sess.scene.resolve_material(target_id)
                    tid = obj.id if obj else (mat.id if mat else target_id)
                    tracks = [t for t in tracks if t.target_id == tid]
                return {"tracks": [t.model_dump() for t in tracks]}

            if not target_id or not property_path or frame is None:
                raise ToolError("target_id, property_path, and frame are required for set/delete.")

            obj = sess.scene.resolve_object(target_id)
            mat = sess.scene.resolve_material(target_id)
            if obj is None and mat is None:
                raise ToolError(f"Target {target_id!r} not found in objects or materials.")
            canonical_id = obj.id if obj else mat.id  # type: ignore[union-attr]

            sess.push_history(f"keyframe_{action}:{canonical_id}.{property_path}@{frame}")
            track = next(
                (
                    t
                    for t in tl.tracks
                    if t.target_id == canonical_id and t.property_path == property_path
                ),
                None,
            )

            if action == "delete":
                if track is None:
                    raise ToolError(f"No track found for {canonical_id}.{property_path}.")
                before = len(track.keyframes)
                track.keyframes = [k for k in track.keyframes if k.frame != frame]
                if len(track.keyframes) == before:
                    raise ToolError(
                        f"No keyframe at frame {frame} on {canonical_id}.{property_path}."
                    )
                if not track.keyframes:
                    tl.tracks = [t for t in tl.tracks if t is not track]
                return {"status": "ok", "deleted_frame": frame}

            if value is None:
                raise ToolError("value is required when action='set'.")
            if track is None:
                track = FCurveTrack(target_id=canonical_id, property_path=property_path)
                tl.tracks.append(track)

            track.keyframes = [k for k in track.keyframes if k.frame != frame]
            track.keyframes.append(
                Keyframe(
                    frame=frame, value=float(value), interpolation=interpolation, easing=easing
                )
            )
            track.keyframes.sort(key=lambda k: k.frame)
            return {"status": "ok", "track": track.model_dump()}

    @mcp.tool(
        title="Render Animation or Turntable",
        annotations=ToolAnnotations(read_only_hint=True, open_world_hint=False),
    )
    def render_animation(
        kind: Annotated[
            Literal["timeline", "turntable"],
            Field(
                description=(
                    "'timeline' renders animated keyframes; "
                    "'turntable' orbits 360 deg around scene."
                )
            ),
        ] = "turntable",
        frame_start: Annotated[
            int | None, Field(description="Start frame for timeline render.")
        ] = None,
        frame_end: Annotated[
            int | None, Field(description="End frame for timeline render.")
        ] = None,
        steps: Annotated[
            int, Field(ge=2, le=24, description="Number of sampled frames to render.")
        ] = 8,
        width: Annotated[int, Field(ge=128, le=800)] = 360,
        height: Annotated[int, Field(ge=128, le=600)] = 270,
        mode: Annotated[RenderMode, Field(description="Visual render mode.")] = "pbr",
        camera_id: Annotated[
            str | None, Field(description="Optional camera ID for timeline renders.")
        ] = None,
        save_gif_path: Annotated[
            str | None, Field(description="Optional animated GIF output path.")
        ] = None,
        save_filmstrip_path: Annotated[
            str | None, Field(description="Optional filmstrip PNG output path.")
        ] = None,
    ) -> list[TextContent | ImageContent]:
        """Render a turntable orbit or keyframed animation into filmstrip PNG and GIF."""
        sess = get_session()
        with sess.lock:
            gif_p = sess.resolve_path(save_gif_path) if save_gif_path else None
            strip_p = sess.resolve_path(save_filmstrip_path) if save_filmstrip_path else None
            strip_png, gif_bytes, meta = render_animation_sequence(
                sess.scene,
                kind=kind,
                frame_start=frame_start,
                frame_end=frame_end,
                steps=steps,
                width=width,
                height=height,
                mode=mode,
                camera_id=camera_id,
            )
            if gif_p is not None:
                gif_p.parent.mkdir(parents=True, exist_ok=True)
                gif_p.write_bytes(gif_bytes)
                meta["saved_gif_path"] = str(gif_p)
                meta["gif_bytes"] = len(gif_bytes)
            if strip_p is not None:
                strip_p.parent.mkdir(parents=True, exist_ok=True)
                strip_p.write_bytes(strip_png)
                meta["saved_filmstrip_path"] = str(strip_p)

            b64 = base64.b64encode(strip_png).decode("ascii")
            return [
                TextContent(type="text", text=json.dumps(meta)),
                ImageContent(type="image", data=b64, mime_type="image/png"),
            ]
