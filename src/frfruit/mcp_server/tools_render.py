"""MCP tools for PBR materials, lights, cameras, AI viewport rendering, and scene export."""

from __future__ import annotations

import base64
import json
from collections.abc import Callable
from typing import Annotated, Any, Literal

from mcp.server import MCPServer
from mcp.server.mcpserver.exceptions import ToolError
from mcp.types import ImageContent, TextContent, ToolAnnotations
from pydantic import Field

from frfruit.core.scene import CameraSpec, LightSpec, MaterialSpec, SceneObject
from frfruit.core.types import ColorRGB, SelectionQuery, Transform, Vec3, parse_color_rgba
from frfruit.geometry.edit_ops import resolve_face_selection
from frfruit.geometry.mesh import GeometryError, MeshData
from frfruit.geometry.spatial import (
    compute_scene_bounds,
    evaluate_object_mesh,
    evaluate_scene_meshes,
)
from frfruit.io.exporters import export_scene_file
from frfruit.io.fbx_exporter import TargetEngine
from frfruit.mcp_server.context import FrFruitSession
from frfruit.render.camera import ViewDirection, auto_frame_camera_object
from frfruit.render.materials import PRESET_MATERIALS, list_material_presets
from frfruit.render.software import RenderMode
from frfruit.render.viewport import render_multiview_sheet, render_single_view


def _apply_face_material_selection(
    sess_scene: Any,
    obj: SceneObject,
    mat_id: str | None,
    *,
    face_index: int | None,
    face_indices: list[int] | None,
    selection_mode: str | None,
    direction: Vec3 | None,
    angle_threshold_deg: float,
    bbox_min: Vec3 | None,
    bbox_max: Vec3 | None,
    seed_face: int | None,
    invert: bool,
    clear: bool = False,
) -> list[int]:
    if obj.mesh is not None:
        base_mesh = MeshData.from_payload(obj.mesh)
    else:
        ev = evaluate_object_mesh(sess_scene, obj.id)
        base_mesh = ev.local_mesh
    if base_mesh.is_empty:
        raise ToolError(f"Object {obj.name!r} has an empty mesh.")

    eff_indices: list[int] | None = None
    if face_indices is not None:
        eff_indices = [int(i) for i in face_indices]
    elif face_index is not None:
        eff_indices = [int(face_index)]

    mode = selection_mode or ("indices" if eff_indices is not None else "all")
    if clear and selection_mode is None and eff_indices is None and not invert:
        cleared = sorted(obj.face_material_ids.keys())
        obj.face_material_ids.clear()
        return cleared

    try:
        query = SelectionQuery(
            mode=mode,  # type: ignore[arg-type]
            direction=direction,
            angle_threshold_deg=angle_threshold_deg,
            bbox_min=bbox_min,
            bbox_max=bbox_max,
            indices=eff_indices,
            seed_face=seed_face,
            invert=invert,
        )
        mask = resolve_face_selection(
            base_mesh,
            query,
            face_material_ids=obj.face_material_ids,
            base_material_id=obj.material_id,
        )
    except (ValueError, GeometryError) as exc:
        raise ToolError(str(exc)) from exc

    import numpy as np

    matched = [int(i) for i in np.nonzero(mask)[0].tolist()]
    if not matched:
        raise ToolError(f"Face selection ({mode}) matched 0 faces on {obj.name!r}.")
    if clear:
        for fi in matched:
            obj.face_material_ids.pop(fi, None)
    else:
        assert mat_id is not None
        for fi in matched:
            obj.face_material_ids[fi] = mat_id
    return matched


def register_render_tools(mcp: MCPServer, get_session: Callable[[], FrFruitSession]) -> None:
    @mcp.tool(
        title="Create or Assign Material",
        annotations=ToolAnnotations(open_world_hint=False),
    )
    def material_create_or_assign(
        action: Annotated[
            Literal[
                "create",
                "assign",
                "create_and_assign",
                "assign_faces",
                "clear_faces",
                "list_presets",
            ],
            Field(description="Material operation."),
        ] = "create_and_assign",
        material_id: Annotated[
            str | None, Field(description="Existing material ID or name.")
        ] = None,
        name: Annotated[str | None, Field(description="Name for new material.")] = None,
        preset: Annotated[
            str | None,
            Field(
                description=(
                    "Preset name (clay, gold, chrome, apple_skin, leaf_green, wood, glass, ...)."
                )
            ),
        ] = None,
        object_id: Annotated[
            str | None, Field(description="Target object ID or name to assign material to.")
        ] = None,
        base_color: Annotated[
            str | list[float] | None,
            Field(description="Hex '#rrggbb' or [r, g, b] / [r, g, b, a] in 0..1."),
        ] = None,
        secondary_color: Annotated[
            str | list[float] | None,
            Field(description="Secondary color for checker/gradient/wood/noise patterns."),
        ] = None,
        metallic: Annotated[float | None, Field(ge=0.0, le=1.0)] = None,
        roughness: Annotated[float | None, Field(ge=0.0, le=1.0)] = None,
        emission_color: Annotated[
            str | list[float] | None, Field(description="Emission color.")
        ] = None,
        emission_strength: Annotated[float | None, Field(ge=0.0)] = None,
        alpha: Annotated[float | None, Field(ge=0.0, le=1.0)] = None,
        clearcoat: Annotated[
            float | None, Field(ge=0.0, le=1.0, description="Dielectric clearcoat layer weight.")
        ] = None,
        clearcoat_roughness: Annotated[
            float | None, Field(ge=0.01, le=1.0, description="Clearcoat specular roughness.")
        ] = None,
        sheen: Annotated[
            float | None, Field(ge=0.0, le=1.0, description="Grazing fabric velvet sheen weight.")
        ] = None,
        subsurface: Annotated[
            float | None,
            Field(ge=0.0, le=1.0, description="Subsurface wrap scattering weight."),
        ] = None,
        subsurface_color: Annotated[
            str | list[float] | None,
            Field(description="Subsurface scattering tint color (hex or [r,g,b])."),
        ] = None,
        pattern: Annotated[
            Literal["solid", "checker", "gradient", "wood", "noise"] | None,
            Field(description="Procedural surface pattern."),
        ] = None,
        pattern_scale: Annotated[float | None, Field(gt=0.0)] = None,
        face_index: Annotated[
            int | None,
            Field(ge=0, description="Single face index to assign material to (Edit Mode slot)."),
        ] = None,
        face_indices: Annotated[
            list[int] | None,
            Field(description="List of face indices to assign material to."),
        ] = None,
        selection_mode: Annotated[
            Literal[
                "all",
                "top",
                "bottom",
                "front",
                "back",
                "left",
                "right",
                "normal_dir",
                "bbox",
                "indices",
                "coplanar",
                "boundary",
                "material",
            ]
            | None,
            Field(description="Optional face selection mode for per-face material assignment."),
        ] = None,
        direction: Annotated[
            Vec3 | None, Field(description="Direction vector when selection_mode='normal_dir'.")
        ] = None,
        angle_threshold_deg: Annotated[float, Field(ge=0.5, le=180.0)] = 45.0,
        bbox_min: Annotated[
            Vec3 | None, Field(description="Bounding box min when selection_mode='bbox'.")
        ] = None,
        bbox_max: Annotated[
            Vec3 | None, Field(description="Bounding box max when selection_mode='bbox'.")
        ] = None,
        seed_face: Annotated[
            int | None, Field(ge=0, description="Seed face index for coplanar flood-fill.")
        ] = None,
        invert: Annotated[bool, Field(description="Invert resolved face selection.")] = False,
    ) -> dict[str, Any]:
        """List presets, create PBR/procedural materials, or assign to objects or single faces."""
        sess = get_session()
        with sess.lock:
            if action == "list_presets":
                return {"presets": list_material_presets()}

            has_face_target = (
                face_index is not None or face_indices is not None or selection_mode is not None
            )

            if action == "clear_faces":
                if not object_id:
                    raise ToolError("object_id is required for action='clear_faces'.")
                obj = sess.require_mesh_like(object_id)
                sess.push_history(f"clear_face_materials:{obj.name}")
                cleared = _apply_face_material_selection(
                    sess.scene,
                    obj,
                    None,
                    face_index=face_index,
                    face_indices=face_indices,
                    selection_mode=selection_mode,
                    direction=direction,
                    angle_threshold_deg=angle_threshold_deg,
                    bbox_min=bbox_min,
                    bbox_max=bbox_max,
                    seed_face=seed_face,
                    invert=invert,
                    clear=True,
                )
                return {
                    "status": "ok",
                    "object_id": obj.id,
                    "cleared_faces": cleared,
                    "remaining_face_materials": len(obj.face_material_ids),
                }

            if action in ("assign", "assign_faces"):
                if not material_id or not object_id:
                    raise ToolError(
                        f"Both material_id and object_id are required for action={action!r}."
                    )
                mat = sess.scene.resolve_material(material_id)
                if mat is None:
                    avail = list(sess.scene.materials.keys())
                    raise ToolError(f"Material {material_id!r} not found. Available: {avail}")
                obj = sess.require_mesh_like(object_id)
                if action == "assign_faces" or has_face_target:
                    sess.push_history(f"assign_face_material:{obj.name}")
                    assigned = _apply_face_material_selection(
                        sess.scene,
                        obj,
                        mat.id,
                        face_index=face_index,
                        face_indices=face_indices,
                        selection_mode=selection_mode,
                        direction=direction,
                        angle_threshold_deg=angle_threshold_deg,
                        bbox_min=bbox_min,
                        bbox_max=bbox_max,
                        seed_face=seed_face,
                        invert=invert,
                    )
                    return {
                        "status": "ok",
                        "object_id": obj.id,
                        "material_id": mat.id,
                        "assigned_faces": assigned,
                        "assigned_face_count": len(assigned),
                    }
                sess.push_history(f"assign_material:{obj.name}")
                obj.material_id = mat.id
                return {"status": "ok", "object_id": obj.id, "material_id": mat.id}

            # Create (and optionally assign)
            base_kwargs: dict[str, Any] = {}
            if preset:
                if preset not in PRESET_MATERIALS:
                    raise ToolError(
                        f"Unknown preset {preset!r}. Available: {sorted(PRESET_MATERIALS.keys())}"
                    )
                base_kwargs.update(PRESET_MATERIALS[preset])

            try:
                if base_color is not None:
                    base_kwargs["base_color"] = parse_color_rgba(base_color)
                if secondary_color is not None:
                    base_kwargs["secondary_color"] = parse_color_rgba(secondary_color)
                if emission_color is not None:
                    base_kwargs["emission_color"] = parse_color_rgba(emission_color)
                if subsurface_color is not None:
                    base_kwargs["subsurface_color"] = parse_color_rgba(subsurface_color)[:3]
            except ValueError as exc:
                raise ToolError(str(exc)) from exc

            for k, val in (
                ("metallic", metallic),
                ("roughness", roughness),
                ("emission_strength", emission_strength),
                ("alpha", alpha),
                ("clearcoat", clearcoat),
                ("clearcoat_roughness", clearcoat_roughness),
                ("sheen", sheen),
                ("subsurface", subsurface),
                ("pattern", pattern),
                ("pattern_scale", pattern_scale),
            ):
                if val is not None:
                    base_kwargs[k] = val

            mat_name = name or base_kwargs.get("name") or preset or "Material"
            mid = material_id or sess.scene.next_id(f"mat_{mat_name.lower()}")
            base_kwargs["name"] = mat_name
            mat = MaterialSpec(id=mid, **base_kwargs)

            sess.push_history(f"material:{mat.name}")
            sess.scene.materials[mid] = mat

            assigned_to: str | None = None
            assigned_faces: list[int] | None = None
            if action == "create_and_assign":
                if not object_id:
                    raise ToolError("object_id is required when action='create_and_assign'.")
                obj = sess.require_mesh_like(object_id)
                if has_face_target:
                    assigned_faces = _apply_face_material_selection(
                        sess.scene,
                        obj,
                        mid,
                        face_index=face_index,
                        face_indices=face_indices,
                        selection_mode=selection_mode,
                        direction=direction,
                        angle_threshold_deg=angle_threshold_deg,
                        bbox_min=bbox_min,
                        bbox_max=bbox_max,
                        seed_face=seed_face,
                        invert=invert,
                    )
                else:
                    obj.material_id = mid
                assigned_to = obj.id

            out: dict[str, Any] = {
                "status": "ok",
                "material": mat.model_dump(),
                "assigned_to": assigned_to,
            }
            if assigned_faces is not None:
                out["assigned_faces"] = assigned_faces
                out["assigned_face_count"] = len(assigned_faces)
            return out

    @mcp.tool(
        title="Manage Scene Lights & World",
        annotations=ToolAnnotations(open_world_hint=False),
    )
    def light_manage(
        action: Annotated[
            Literal["add", "update", "delete", "setup_studio_3point", "set_world", "list"],
            Field(description="Light or world environment operation."),
        ],
        light_id: Annotated[str | None, Field(description="Light ID or name.")] = None,
        kind: Annotated[
            Literal["sun", "point", "spot", "area"], Field(description="Light type.")
        ] = "sun",
        name: Annotated[str | None, Field(description="Light object name.")] = None,
        location: Annotated[Vec3 | None, Field(description="Light XYZ position.")] = None,
        direction: Annotated[
            Vec3 | None, Field(description="Direction vector for sun/spot lights.")
        ] = None,
        color: Annotated[ColorRGB | None, Field(description="RGB color in 0..1.")] = None,
        intensity: Annotated[float | None, Field(ge=0.0, description="Light intensity.")] = None,
        spot_angle_deg: Annotated[float | None, Field(ge=1.0, le=175.0)] = None,
        cast_shadows: Annotated[bool | None, Field(description="Enable shadow casting.")] = None,
        background_color: Annotated[
            ColorRGB | None, Field(description="World background/horizon RGB.")
        ] = None,
        horizon_color: Annotated[ColorRGB | None, Field(description="World horizon RGB.")] = None,
        zenith_color: Annotated[ColorRGB | None, Field(description="World zenith RGB.")] = None,
        ambient_strength: Annotated[float | None, Field(ge=0.0, le=2.0)] = None,
        sky_mode: Annotated[
            Literal["gradient", "physical_sky", "hdri"] | None,
            Field(description="World sky and IBL lighting model."),
        ] = None,
        hdri_path: Annotated[
            str | None,
            Field(description="Workspace path to .hdr/.png/.jpg equirectangular HDRI map."),
        ] = None,
        hdri_rotation_deg: Annotated[
            float | None, Field(description="HDRI azimuth rotation in degrees.")
        ] = None,
        hdri_intensity: Annotated[
            float | None, Field(ge=0.0, le=20.0, description="HDRI brightness multiplier.")
        ] = None,
        sun_elevation_deg: Annotated[
            float | None,
            Field(ge=-15.0, le=90.0, description="Physical sky sun elevation in degrees."),
        ] = None,
        sun_azimuth_deg: Annotated[
            float | None,
            Field(ge=-360.0, le=360.0, description="Physical sky sun azimuth in degrees."),
        ] = None,
        air_turbidity: Annotated[
            float | None, Field(ge=1.0, le=10.0, description="Physical sky atmospheric turbidity.")
        ] = None,
    ) -> dict[str, Any]:
        """Add, update, delete lights, configure 3-point studio lighting, or adjust world sky."""
        sess = get_session()
        with sess.lock:
            if action == "list":
                return {
                    "world": sess.scene.world.model_dump(),
                    "lights": [
                        {
                            "id": o.id,
                            "name": o.name,
                            "location": list(o.transform.location),
                            **o.light.model_dump(),
                        }
                        for o in sess.scene.objects.values()
                        if o.kind == "light" and o.light is not None
                    ],
                }

            if action == "set_world":
                sess.push_history("set_world")
                w = sess.scene.world
                if background_color is not None:
                    w.background_color = background_color
                    if horizon_color is None:
                        w.horizon_color = background_color
                if horizon_color is not None:
                    w.horizon_color = horizon_color
                if zenith_color is not None:
                    w.zenith_color = zenith_color
                if ambient_strength is not None:
                    w.ambient_strength = ambient_strength
                if sky_mode is not None:
                    w.sky_mode = sky_mode
                if hdri_path is not None:
                    w.hdri_path = str(sess.resolve_path(hdri_path, must_exist=True))
                if hdri_rotation_deg is not None:
                    w.hdri_rotation_deg = hdri_rotation_deg
                if hdri_intensity is not None:
                    w.hdri_intensity = hdri_intensity
                if sun_elevation_deg is not None:
                    w.sun_elevation_deg = sun_elevation_deg
                if sun_azimuth_deg is not None:
                    w.sun_azimuth_deg = sun_azimuth_deg
                if air_turbidity is not None:
                    w.air_turbidity = air_turbidity
                return {"status": "ok", "world": w.model_dump()}

            if action == "setup_studio_3point":
                sess.push_history("setup_studio_3point")
                evaluated = evaluate_scene_meshes(sess.scene)
                bounds = compute_scene_bounds(evaluated)
                cx, cy, cz = bounds.center
                r = max(2.5, float(max(bounds.size)) * 2.5)
                # Remove old lights cleanly
                old_ids = [oid for oid, o in sess.scene.objects.items() if o.kind == "light"]
                for oid in old_ids:
                    del sess.scene.objects[oid]

                sess.scene.objects["light_key"] = SceneObject(
                    id="light_key",
                    name="StudioKey",
                    kind="light",
                    transform=Transform(location=(cx + r, cy - r, cz + r * 1.3)),
                    light=LightSpec(
                        kind="sun",
                        color=(1.0, 0.96, 0.90),
                        intensity=2.5,
                        direction=(-0.55, 0.45, -0.70),
                        cast_shadows=True,
                    ),
                )
                sess.scene.objects["light_fill"] = SceneObject(
                    id="light_fill",
                    name="StudioFill",
                    kind="light",
                    transform=Transform(location=(cx - r, cy - r * 0.6, cz + r * 0.5)),
                    light=LightSpec(
                        kind="point",
                        color=(0.75, 0.86, 1.0),
                        intensity=1.4,
                        cast_shadows=False,
                    ),
                )
                sess.scene.objects["light_rim"] = SceneObject(
                    id="light_rim",
                    name="StudioRim",
                    kind="light",
                    transform=Transform(location=(cx - r * 0.4, cy + r * 1.1, cz + r * 0.9)),
                    light=LightSpec(
                        kind="point",
                        color=(0.95, 0.95, 1.0),
                        intensity=1.8,
                        cast_shadows=False,
                    ),
                )
                return {"status": "ok", "lights": ["light_key", "light_fill", "light_rim"]}

            if action == "add":
                lname = sess.scene.unique_name(name or f"{kind.capitalize()}Light")
                lid = sess.scene.next_id(lname.lower())
                sess.push_history(f"light_add:{lname}")
                lobj = SceneObject(
                    id=lid,
                    name=lname,
                    kind="light",
                    transform=Transform(location=location or (4.0, -4.0, 6.0)),
                    light=LightSpec(
                        kind=kind,
                        color=color or (1.0, 0.98, 0.94),
                        intensity=2.2 if intensity is None else intensity,
                        direction=direction or (-0.5, 0.4, -0.7),
                        spot_angle_deg=45.0 if spot_angle_deg is None else spot_angle_deg,
                        cast_shadows=True if cast_shadows is None else cast_shadows,
                    ),
                )
                sess.scene.objects[lid] = lobj
                return {"status": "ok", "light_id": lid, "name": lobj.name}

            if not light_id:
                raise ToolError(f"light_id is required for action={action!r}.")
            lobj = sess.require_object(light_id, expected_kind="light")
            if action == "delete":
                sess.push_history(f"light_delete:{lobj.name}")
                del sess.scene.objects[lobj.id]
                return {"status": "ok", "deleted_id": lobj.id}

            # update
            assert lobj.light is not None
            sess.push_history(f"light_update:{lobj.name}")
            if location is not None:
                lobj.transform.location = location
            if direction is not None:
                lobj.light.direction = direction
            if color is not None:
                lobj.light.color = color
            if intensity is not None:
                lobj.light.intensity = intensity
            if spot_angle_deg is not None:
                lobj.light.spot_angle_deg = spot_angle_deg
            if cast_shadows is not None:
                lobj.light.cast_shadows = cast_shadows
            return {"status": "ok", "light_id": lobj.id, "light": lobj.light.model_dump()}

    @mcp.tool(
        title="Manage Cameras",
        annotations=ToolAnnotations(open_world_hint=False),
    )
    def camera_manage(
        action: Annotated[
            Literal["add", "update", "set_active", "auto_frame", "delete", "list"],
            Field(description="Camera management operation."),
        ],
        camera_id: Annotated[str | None, Field(description="Camera ID or name.")] = None,
        name: Annotated[str | None, Field(description="Name for new camera.")] = None,
        projection: Annotated[
            Literal["perspective", "orthographic"], Field(description="Projection type.")
        ] = "perspective",
        location: Annotated[Vec3 | None, Field(description="Camera eye position.")] = None,
        target: Annotated[Vec3 | None, Field(description="Look-at target point.")] = None,
        fov_deg: Annotated[float | None, Field(ge=5.0, le=150.0)] = None,
        ortho_scale: Annotated[float | None, Field(gt=0.0)] = None,
        direction: Annotated[
            ViewDirection, Field(description="View direction for auto_frame.")
        ] = "iso",
        target_object_id: Annotated[
            str | None,
            Field(description="Optional object to frame in auto_frame (default: whole scene)."),
        ] = None,
        margin: Annotated[
            float, Field(gt=0.5, le=5.0, description="Framing margin multiplier.")
        ] = 1.25,
    ) -> dict[str, Any]:
        """Add, update, activate, delete, or auto-frame cameras around objects or scenes."""
        sess = get_session()
        with sess.lock:
            if action == "list":
                return {
                    "active_camera_id": sess.scene.active_camera_id,
                    "cameras": [
                        {
                            "id": o.id,
                            "name": o.name,
                            "location": list(o.transform.location),
                            **o.camera.model_dump(),
                        }
                        for o in sess.scene.objects.values()
                        if o.kind == "camera" and o.camera is not None
                    ],
                }

            if action == "add":
                cname = sess.scene.unique_name(name or "Camera")
                cid = sess.scene.next_id(cname.lower())
                sess.push_history(f"camera_add:{cname}")
                cobj = SceneObject(
                    id=cid,
                    name=cname,
                    kind="camera",
                    transform=Transform(location=location or (4.0, -4.0, 3.0)),
                    camera=CameraSpec(
                        projection=projection,
                        fov_deg=45.0 if fov_deg is None else fov_deg,
                        ortho_scale=5.0 if ortho_scale is None else ortho_scale,
                        target=target or (0.0, 0.0, 0.5),
                    ),
                )
                sess.scene.objects[cid] = cobj
                sess.scene.active_camera_id = cid
                return {"status": "ok", "camera_id": cid, "active_camera_id": cid}

            if action == "auto_frame":
                if target_object_id:
                    tobj = sess.require_mesh_like(target_object_id)
                    ev = evaluate_object_mesh(sess.scene, tobj.id)
                    bounds = ev.bounds
                else:
                    bounds = compute_scene_bounds(evaluate_scene_meshes(sess.scene))
                framed = auto_frame_camera_object(
                    bounds=bounds,
                    direction=direction,
                    margin=margin,
                    projection=projection,
                    fov_deg=45.0 if fov_deg is None else fov_deg,
                )
                cid = camera_id or sess.scene.active_camera_id
                sess.push_history("camera_auto_frame")
                if cid and sess.scene.resolve_object(cid) is not None:
                    cam_obj = sess.require_object(cid, expected_kind="camera")
                    cam_obj.transform.location = framed.transform.location
                    cam_obj.camera = framed.camera
                else:
                    cid = sess.scene.next_id("cam_main")
                    framed.id = cid
                    framed.name = name or "MainCamera"
                    sess.scene.objects[cid] = framed
                    sess.scene.active_camera_id = cid
                    cam_obj = framed
                return {
                    "status": "ok",
                    "camera_id": cam_obj.id,
                    "location": list(cam_obj.transform.location),
                    "target": list(cam_obj.camera.target) if cam_obj.camera else [0, 0, 0],
                }

            cid = camera_id or sess.scene.active_camera_id
            if not cid:
                raise ToolError("No camera_id specified and no active camera in scene.")
            cam_obj = sess.require_object(cid, expected_kind="camera")

            if action == "set_active":
                sess.scene.active_camera_id = cam_obj.id
                return {"status": "ok", "active_camera_id": cam_obj.id}

            if action == "delete":
                sess.push_history(f"camera_delete:{cam_obj.name}")
                del sess.scene.objects[cam_obj.id]
                if sess.scene.active_camera_id == cam_obj.id:
                    remaining = [o.id for o in sess.scene.objects.values() if o.kind == "camera"]
                    sess.scene.active_camera_id = remaining[0] if remaining else None
                return {"status": "ok", "deleted_id": cam_obj.id}

            assert cam_obj.camera is not None
            sess.push_history(f"camera_update:{cam_obj.name}")
            if location is not None:
                cam_obj.transform.location = location
            if target is not None:
                cam_obj.camera.target = target
            if fov_deg is not None:
                cam_obj.camera.fov_deg = fov_deg
            if ortho_scale is not None:
                cam_obj.camera.ortho_scale = ortho_scale
            cam_obj.camera.projection = projection
            return {
                "status": "ok",
                "camera_id": cam_obj.id,
                "location": list(cam_obj.transform.location),
                "camera": cam_obj.camera.model_dump(),
            }

    @mcp.tool(
        title="Render AI Viewport",
        annotations=ToolAnnotations(read_only_hint=True, open_world_hint=False),
    )
    def render_viewport(
        mode: Annotated[
            RenderMode,
            Field(description="Shader mode: pbr, clay, wireframe, normals, depth, or collision."),
        ] = "pbr",
        engine: Annotated[
            Literal["software", "raytrace"] | None,
            Field(description="Optional render engine override ('software' or 'raytrace')."),
        ] = None,
        samples: Annotated[int, Field(ge=1, le=16, description="Raytracer samples per pixel.")] = 2,
        width: Annotated[int, Field(ge=128, le=1920)] = 640,
        height: Annotated[int, Field(ge=128, le=1080)] = 480,
        camera_id: Annotated[str | None, Field(description="Optional camera ID or name.")] = None,
        view_direction: Annotated[
            ViewDirection | None,
            Field(
                description=(
                    "Optional auto-framed view direction "
                    "(iso, front, back, left, right, top, bottom)."
                )
            ),
        ] = None,
        show_grid: Annotated[bool, Field(description="Render 1m ground grid.")] = True,
        show_axes: Annotated[bool, Field(description="Render XYZ orientation triad.")] = True,
        show_bounds: Annotated[bool, Field(description="Overlay object bounding boxes.")] = False,
        show_labels: Annotated[bool, Field(description="Overlay object name labels.")] = False,
        save_path: Annotated[
            str | None, Field(description="Optional PNG output path in workspace.")
        ] = None,
    ) -> list[TextContent | ImageContent]:
        """Render the 3D scene headlessly and return JSON metadata plus a PNG ImageContent."""
        sess = get_session()
        with sess.lock:
            out_path = sess.resolve_path(save_path) if save_path else None
            png_bytes, meta = render_single_view(
                sess.scene,
                width=width,
                height=height,
                mode=mode,
                camera_id=camera_id,
                view_direction=view_direction,
                show_grid=show_grid,
                show_axes=show_axes,
                show_bounds=show_bounds,
                show_labels=show_labels,
                engine=engine,
                samples=samples,
            )
            if out_path is not None:
                out_path.parent.mkdir(parents=True, exist_ok=True)
                out_path.write_bytes(png_bytes)
                meta["saved_path"] = str(out_path)
            b64 = base64.b64encode(png_bytes).decode("ascii")
            return [
                TextContent(type="text", text=json.dumps(meta)),
                ImageContent(type="image", data=b64, mime_type="image/png"),
            ]

    @mcp.tool(
        title="Render Multi-View Sheet",
        annotations=ToolAnnotations(read_only_hint=True, open_world_hint=False),
    )
    def render_views(
        mode: Annotated[
            RenderMode,
            Field(description="Shader mode: pbr, clay, wireframe, normals, depth, or collision."),
        ] = "pbr",
        engine: Annotated[
            Literal["software", "raytrace"] | None,
            Field(description="Optional render engine override ('software' or 'raytrace')."),
        ] = None,
        samples: Annotated[int, Field(ge=1, le=16, description="Raytracer samples per pixel.")] = 2,
        tile_width: Annotated[int, Field(ge=160, le=960)] = 420,
        tile_height: Annotated[int, Field(ge=120, le=720)] = 320,
        views: Annotated[
            list[ViewDirection] | None,
            Field(description="List of views for contact sheet (default: front, right, top, iso)."),
        ] = None,
        show_grid: Annotated[bool, Field(description="Show ground grid.")] = True,
        show_axes: Annotated[bool, Field(description="Show XYZ triad.")] = True,
        show_bounds: Annotated[bool, Field(description="Overlay bounding boxes.")] = False,
        show_labels: Annotated[
            bool, Field(description="Overlay object labels on perspective view.")
        ] = True,
        save_path: Annotated[
            str | None, Field(description="Optional PNG output path in workspace.")
        ] = None,
    ) -> list[TextContent | ImageContent]:
        """Render a labeled multi-angle contact sheet (Front, Right, Top, ISO) in one image."""
        view_list = list(views) if views else ["front", "right", "top", "iso"]
        sess = get_session()
        with sess.lock:
            out_path = sess.resolve_path(save_path) if save_path else None
            png_bytes, meta = render_multiview_sheet(
                sess.scene,
                tile_width=tile_width,
                tile_height=tile_height,
                mode=mode,
                views=tuple(view_list),
                show_grid=show_grid,
                show_axes=show_axes,
                show_bounds=show_bounds,
                show_labels=show_labels,
                engine=engine,
                samples=samples,
            )
            if out_path is not None:
                out_path.parent.mkdir(parents=True, exist_ok=True)
                out_path.write_bytes(png_bytes)
                meta["saved_path"] = str(out_path)
            b64 = base64.b64encode(png_bytes).decode("ascii")
            return [
                TextContent(type="text", text=json.dumps(meta)),
                ImageContent(type="image", data=b64, mime_type="image/png"),
            ]

    @mcp.tool(
        title="Export 3D Scene",
        annotations=ToolAnnotations(idempotent_hint=True, open_world_hint=False),
    )
    def export_scene(
        path: Annotated[
            str,
            Field(
                description=(
                    "Output path relative to workspace (e.g. 'out/model.fbx', 'out/model.glb')."
                )
            ),
        ],
        format: Annotated[
            Literal["fbx", "fbx_ascii", "glb", "gltf", "obj", "stl", "ply", "py"] | None,
            Field(
                description=(
                    "Optional explicit format override "
                    "(fbx, fbx_ascii, glb, gltf, obj, stl, ply, or py)."
                )
            ),
        ] = None,
        target_engine: Annotated[
            TargetEngine,
            Field(
                description="Game-engine FBX axis/rig convention: generic, unreal, unity, or godot."
            ),
        ] = "generic",
    ) -> dict[str, Any]:
        """Export scene to FBX (bones/anim/morphs), GLB/GLTF, OBJ, STL, PLY, or Blender .py."""
        sess = get_session()
        with sess.lock:
            out_p = sess.resolve_path(path)
            try:
                res = export_scene_file(
                    sess.scene,
                    out_p,
                    fmt=format,
                    target_engine=target_engine,
                )
            except ValueError as exc:
                raise ToolError(str(exc)) from exc
            return {"status": "ok", **res}
