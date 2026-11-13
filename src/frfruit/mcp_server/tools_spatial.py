"""MCP tools for spatial inspection, collision checks, raycasting, and smart placement."""

from __future__ import annotations

from collections.abc import Callable
from typing import Annotated, Any, Literal

from mcp.server import MCPServer
from mcp.server.mcpserver.exceptions import ToolError
from mcp.types import ToolAnnotations
from pydantic import Field

from frfruit.core.types import Vec3
from frfruit.geometry.mesh import GeometryError
from frfruit.geometry.spatial import (
    align_objects,
    attach_to_side,
    check_intersections,
    distribute_objects,
    evaluate_object_mesh,
    evaluate_scene_meshes,
    ground_object,
    measure_distance,
    place_on_top,
    raycast_scene,
    snap_to_surface,
)
from frfruit.mcp_server.context import FrFruitSession


def register_spatial_tools(mcp: MCPServer, get_session: Callable[[], FrFruitSession]) -> None:
    @mcp.tool(
        title="Spatial Measure & Collision Check",
        annotations=ToolAnnotations(read_only_hint=True, open_world_hint=False),
    )
    def spatial_measure(
        action: Annotated[
            Literal["bounding_box", "distance", "raycast", "check_intersections"],
            Field(description="Spatial inspection query to perform."),
        ],
        object_id: Annotated[str | None, Field(description="Primary object ID or name.")] = None,
        target_object_id: Annotated[
            str | None, Field(description="Second object ID or name for action='distance'.")
        ] = None,
        object_ids: Annotated[
            list[str] | None,
            Field(description="Subset of objects for action='check_intersections'."),
        ] = None,
        origin: Annotated[
            Vec3 | None, Field(description="Ray origin for action='raycast'.")
        ] = None,
        direction: Annotated[
            Vec3 | None, Field(description="Ray direction for action='raycast'.")
        ] = None,
        tolerance: Annotated[
            float, Field(ge=0.0, description="Overlap tolerance to ignore flush touching.")
        ] = 1e-3,
    ) -> dict[str, Any]:
        """Inspect bounds, measure surface distances, cast rays, or detect intersections."""
        sess = get_session()
        with sess.lock:
            evaluated = evaluate_scene_meshes(sess.scene)

            if action == "bounding_box":
                if not object_id:
                    raise ToolError("object_id is required for action='bounding_box'.")
                obj = sess.require_mesh_like(object_id)
                ev = evaluated.get(obj.id) or evaluate_object_mesh(sess.scene, obj.id)
                return {
                    "object_id": obj.id,
                    "name": obj.name,
                    "min": list(ev.bounds.min_corner),
                    "max": list(ev.bounds.max_corner),
                    "center": list(ev.bounds.center),
                    "size": list(ev.bounds.size),
                }

            if action == "distance":
                if not object_id or not target_object_id:
                    raise ToolError(
                        "Both object_id and target_object_id are required for distance."
                    )
                oa = sess.require_mesh_like(object_id)
                ob = sess.require_mesh_like(target_object_id)
                eva = evaluated.get(oa.id) or evaluate_object_mesh(sess.scene, oa.id)
                evb = evaluated.get(ob.id) or evaluate_object_mesh(sess.scene, ob.id)
                return measure_distance(eva, evb)

            if action == "raycast":
                if origin is None or direction is None:
                    raise ToolError("Both origin and direction are required for raycast.")
                ignore = {sess.require_object(object_id).id} if object_id else None
                try:
                    return raycast_scene(
                        evaluated, origin=origin, direction=direction, ignore_ids=ignore
                    )
                except GeometryError as exc:
                    raise ToolError(str(exc)) from exc

            resolved_ids = (
                [sess.require_mesh_like(oid).id for oid in object_ids] if object_ids else None
            )
            cols = check_intersections(evaluated, object_ids=resolved_ids, tolerance=tolerance)
            return {"intersection_count": len(cols), "collisions": cols}

    @mcp.tool(
        title="Smart Spatial Placement",
        annotations=ToolAnnotations(open_world_hint=False),
    )
    def spatial_place(
        action: Annotated[
            Literal[
                "place_on_top", "attach_to_side", "ground", "align", "distribute", "snap_to_surface"
            ],
            Field(description="Semantic placement helper."),
        ],
        object_id: Annotated[str | None, Field(description="Object to move (ID or name).")] = None,
        base_object_id: Annotated[
            str | None,
            Field(
                description=(
                    "Reference anchor object for place_on_top/attach_to_side/snap_to_surface."
                )
            ),
        ] = None,
        object_ids: Annotated[
            list[str] | None, Field(description="List of object IDs/names for align or distribute.")
        ] = None,
        offset: Annotated[float, Field(description="Vertical/gap offset in meters.")] = 0.0,
        align_xy: Annotated[
            Literal["center", "keep"], Field(description="XY centering mode for place_on_top.")
        ] = "center",
        side: Annotated[
            Literal["+x", "-x", "+y", "-y", "+z", "-z"],
            Field(description="Face side for attach_to_side."),
        ] = "+x",
        axis: Annotated[
            Literal["x", "y", "z"], Field(description="Axis for align or distribute.")
        ] = "x",
        align_mode: Annotated[
            Literal["min", "center", "max"], Field(description="Alignment anchor.")
        ] = "center",
        spacing: Annotated[float, Field(description="Gap spacing for distribute.")] = 0.5,
        z: Annotated[float, Field(description="Ground Z elevation for action='ground'.")] = 0.0,
        direction: Annotated[Vec3, Field(description="Snap direction for snap_to_surface.")] = (
            0.0,
            0.0,
            -1.0,
        ),
    ) -> dict[str, Any]:
        """Precisely place, stack, attach, ground, align, or distribute objects cleanly."""
        sess = get_session()
        with sess.lock:
            if action in ("align", "distribute"):
                if not object_ids or len(object_ids) < 2:
                    raise ToolError(f"{action} requires at least 2 object_ids.")
                evals = [
                    evaluate_object_mesh(sess.scene, sess.require_mesh_like(oid).id)
                    for oid in object_ids
                ]
                sess.push_history(f"spatial:{action}")
                updated = (
                    align_objects(sess.scene, evals, axis=axis, mode=align_mode)
                    if action == "align"
                    else distribute_objects(sess.scene, evals, axis=axis, spacing=spacing)
                )
                return {
                    "status": "ok",
                    "action": action,
                    "updated_locations": {k: list(v) for k, v in updated.items()},
                }

            if not object_id:
                raise ToolError(f"object_id is required for action={action!r}.")
            obj = sess.require_mesh_like(object_id)
            target_ev = evaluate_object_mesh(sess.scene, obj.id)

            if action == "ground":
                sess.push_history(f"spatial:ground:{obj.name}")
                new_loc = ground_object(sess.scene, target_ev, z=z)
                return {"status": "ok", "object_id": obj.id, "location": list(new_loc)}

            if not base_object_id:
                raise ToolError(f"base_object_id is required for action={action!r}.")
            base = sess.require_mesh_like(base_object_id)
            base_ev = evaluate_object_mesh(sess.scene, base.id)

            sess.push_history(f"spatial:{action}:{obj.name}")
            try:
                if action == "place_on_top":
                    new_loc = place_on_top(
                        sess.scene, target_ev, base_ev, offset=offset, align_xy=align_xy
                    )
                elif action == "attach_to_side":
                    new_loc = attach_to_side(sess.scene, target_ev, base_ev, side=side, gap=offset)
                else:
                    new_loc = snap_to_surface(
                        sess.scene, target_ev, base_ev, direction=direction, gap=offset
                    )
            except GeometryError as exc:
                raise ToolError(str(exc)) from exc

            after_ev = evaluate_object_mesh(sess.scene, obj.id)
            return {
                "status": "ok",
                "object_id": obj.id,
                "location": list(new_loc),
                "world_bounds": {
                    "min": list(after_ev.bounds.min_corner),
                    "max": list(after_ev.bounds.max_corner),
                },
            }
