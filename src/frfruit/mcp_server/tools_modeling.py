"""MCP tools for primitives, curves, transforms, Edit Mesh, CSG booleans, and modifiers."""

from __future__ import annotations

from collections.abc import Callable
from typing import Annotated, Any, Literal

from mcp.server import MCPServer
from mcp.server.mcpserver.exceptions import ToolError
from mcp.types import ToolAnnotations
from pydantic import Field, ValidationError

from frfruit.core.scene import (
    ArrayModifier,
    BendModifier,
    BevelModifier,
    BooleanModifier,
    DisplaceModifier,
    MirrorModifier,
    ModifierSpec,
    SceneObject,
    SmoothModifier,
    SolidifyModifier,
    SubdivisionModifier,
    TaperModifier,
    TwistModifier,
)
from frfruit.core.types import SelectionQuery, Transform, Vec2, Vec3
from frfruit.geometry.curves import extrude_2d_profile, revolve_profile, sweep_curve
from frfruit.geometry.edit_ops import (
    bevel_edges,
    boolean_meshes,
    extrude_faces,
    inset_faces,
    recompute_normals,
    repair_manifold,
    smooth_laplacian,
    subdivide_mesh,
    transform_selection,
)
from frfruit.geometry.mesh import GeometryError, MeshData
from frfruit.geometry.primitives import (
    make_capsule,
    make_cone,
    make_cube,
    make_cylinder,
    make_icosphere,
    make_plane,
    make_sphere,
    make_torus,
)
from frfruit.geometry.spatial import evaluate_object_mesh
from frfruit.io.importers import import_mesh_file
from frfruit.mcp_server.context import FrFruitSession

_MODIFIER_CLASSES: dict[str, type[Any]] = {
    "mirror": MirrorModifier,
    "array": ArrayModifier,
    "solidify": SolidifyModifier,
    "subdivision": SubdivisionModifier,
    "bevel": BevelModifier,
    "boolean": BooleanModifier,
    "taper": TaperModifier,
    "twist": TwistModifier,
    "bend": BendModifier,
    "displace": DisplaceModifier,
    "smooth": SmoothModifier,
}


def register_modeling_tools(mcp: MCPServer, get_session: Callable[[], FrFruitSession]) -> None:
    @mcp.tool(
        title="Create 3D Primitive",
        annotations=ToolAnnotations(open_world_hint=False),
    )
    def create_primitive(
        primitive: Annotated[
            Literal["cube", "sphere", "icosphere", "cylinder", "cone", "torus", "plane", "capsule"],
            Field(description="Primitive shape to create."),
        ],
        name: Annotated[str | None, Field(description="Optional object name.")] = None,
        size: Annotated[
            Vec3, Field(description="Dimensions (sx, sy, sz) for cube or (sx, sy, _) for plane.")
        ] = (1.0, 1.0, 1.0),
        radius: Annotated[
            float, Field(gt=0.0, description="Primary radius for sphere/cylinder/cone/capsule.")
        ] = 0.5,
        height: Annotated[
            float, Field(gt=0.0, description="Height for cylinder/cone/capsule.")
        ] = 1.0,
        radius_top: Annotated[
            float | None, Field(ge=0.0, description="Optional top radius for cylinder frustum.")
        ] = None,
        major_radius: Annotated[float, Field(gt=0.0, description="Major radius for torus.")] = 0.5,
        minor_radius: Annotated[
            float, Field(gt=0.0, description="Minor tube radius for torus.")
        ] = 0.15,
        segments: Annotated[int, Field(ge=3, le=128, description="Radial/UV segments.")] = 24,
        subdivisions: Annotated[
            int, Field(ge=0, le=32, description="Subdivisions for icosphere or plane.")
        ] = 2,
        origin: Annotated[
            Literal["bottom", "center"],
            Field(description="'bottom' places local min-Z at 0; 'center' centers around origin."),
        ] = "bottom",
        location: Annotated[Vec3, Field(description="World/local XYZ position in meters.")] = (
            0.0,
            0.0,
            0.0,
        ),
        rotation_deg: Annotated[Vec3, Field(description="XYZ Euler rotation in degrees.")] = (
            0.0,
            0.0,
            0.0,
        ),
        scale: Annotated[Vec3, Field(description="XYZ scale factors.")] = (1.0, 1.0, 1.0),
        material_id: Annotated[
            str | None, Field(description="Optional material ID or name.")
        ] = None,
        collection_id: Annotated[str, Field(description="Target collection ID.")] = "default",
    ) -> dict[str, Any]:
        """Create a primitive mesh object and add it to the scene."""
        sess = get_session()
        with sess.lock:
            if primitive == "cube":
                mesh = make_cube(size=size, origin=origin)
            elif primitive == "sphere":
                mesh = make_sphere(radius=radius, segments=segments, origin=origin)
            elif primitive == "icosphere":
                mesh = make_icosphere(
                    radius=radius, subdivisions=min(4, subdivisions), origin=origin
                )
            elif primitive == "cylinder":
                mesh = make_cylinder(
                    radius=radius,
                    height=height,
                    radius_top=radius_top,
                    segments=segments,
                    origin=origin,
                )
            elif primitive == "cone":
                mesh = make_cone(radius=radius, height=height, segments=segments, origin=origin)
            elif primitive == "torus":
                mesh = make_torus(
                    major_radius=major_radius,
                    minor_radius=minor_radius,
                    major_segments=segments,
                    minor_segments=max(8, segments // 2),
                    origin=origin,
                )
            elif primitive == "capsule":
                mesh = make_capsule(radius=radius, height=height, segments=segments, origin=origin)
            else:
                mesh = make_plane(size=(size[0], size[1]), subdivisions=max(1, subdivisions))

            obj_name = sess.scene.unique_name(name or primitive.capitalize())
            oid = sess.scene.next_id(obj_name.lower())
            mat_resolved = sess.scene.resolve_material(material_id) if material_id else None
            sess.push_history(f"create_primitive:{obj_name}")
            obj = SceneObject(
                id=oid,
                name=obj_name,
                kind="mesh",
                transform=Transform(location=location, rotation_deg=rotation_deg, scale=scale),
                collection_id=collection_id
                if collection_id in sess.scene.collections
                else "default",
                material_id=mat_resolved.id if mat_resolved else None,
                mesh=mesh.to_payload(),
            )
            sess.scene.objects[oid] = obj
            ev = evaluate_object_mesh(sess.scene, oid)
            return {
                "status": "ok",
                "object_id": oid,
                "name": obj.name,
                "stats": ev.world_mesh.stats(),
            }

    @mcp.tool(
        title="Create Mesh From Curve or Profile",
        annotations=ToolAnnotations(open_world_hint=False),
    )
    def create_from_curve(
        mode: Annotated[
            Literal["extrude_2d", "revolve_lathe", "sweep_tube"],
            Field(description="Curve-to-mesh construction mode."),
        ],
        points: Annotated[
            list[list[float]],
            Field(
                description=(
                    "2D [x, y] (or [r, z] for revolve_lathe), or 3D [x, y, z] path for sweep_tube."
                )
            ),
        ],
        name: Annotated[str | None, Field(description="Optional object name.")] = None,
        height: Annotated[
            float, Field(gt=0.0, description="Extrusion height for extrude_2d.")
        ] = 1.0,
        twist_deg: Annotated[float, Field(description="Twist along height for extrude_2d.")] = 0.0,
        scale_top: Annotated[Vec2, Field(description="Top X/Y scale for extrude_2d.")] = (1.0, 1.0),
        angle_deg: Annotated[
            float, Field(gt=0.0, le=360.0, description="Revolve sweep angle.")
        ] = 360.0,
        radius: Annotated[float, Field(gt=0.0, description="Tube radius for sweep_tube.")] = 0.08,
        taper_end: Annotated[
            float, Field(gt=0.0, description="End radius multiplier for sweep_tube.")
        ] = 1.0,
        segments: Annotated[int, Field(ge=6, le=128, description="Circular/radial segments.")] = 24,
        closed: Annotated[bool, Field(description="Closed spline loop for sweep_tube.")] = False,
        location: Annotated[Vec3, Field(description="World/local XYZ position.")] = (0.0, 0.0, 0.0),
        material_id: Annotated[
            str | None, Field(description="Optional material ID or name.")
        ] = None,
    ) -> dict[str, Any]:
        """Build watertight 3D shapes from 2D profiles (Extrude/Lathe) or 3D paths (Sweep)."""
        sess = get_session()
        with sess.lock:
            try:
                if mode == "extrude_2d":
                    pts2 = [(float(p[0]), float(p[1])) for p in points]
                    mesh = extrude_2d_profile(
                        pts2,
                        height=height,
                        twist_deg=twist_deg,
                        scale_top=scale_top,
                        divisions=max(2, segments // 3),
                    )
                elif mode == "revolve_lathe":
                    pts2 = [(float(p[0]), float(p[1])) for p in points]
                    mesh = revolve_profile(pts2, angle_deg=angle_deg, segments=segments)
                else:
                    pts3 = [(float(p[0]), float(p[1]), float(p[2])) for p in points]
                    mesh = sweep_curve(
                        pts3,
                        radius=radius,
                        radial_segments=segments,
                        closed=closed,
                        taper_end=taper_end,
                    )
            except (IndexError, ValueError, GeometryError) as exc:
                raise ToolError(f"Curve generation failed: {exc}") from exc

            obj_name = sess.scene.unique_name(name or f"Curve_{mode}")
            oid = sess.scene.next_id(obj_name.lower())
            mat_resolved = sess.scene.resolve_material(material_id) if material_id else None
            sess.push_history(f"create_from_curve:{obj_name}")
            obj = SceneObject(
                id=oid,
                name=obj_name,
                kind="mesh",
                transform=Transform(location=location),
                material_id=mat_resolved.id if mat_resolved else None,
                mesh=mesh.to_payload(),
            )
            sess.scene.objects[oid] = obj
            ev = evaluate_object_mesh(sess.scene, oid)
            return {
                "status": "ok",
                "object_id": oid,
                "name": obj.name,
                "stats": ev.world_mesh.stats(),
            }

    @mcp.tool(
        title="Transform or Manage Object",
        annotations=ToolAnnotations(open_world_hint=False),
    )
    def object_transform(
        object_id: Annotated[str, Field(description="Object ID or name.")],
        location: Annotated[Vec3 | None, Field(description="Absolute XYZ location.")] = None,
        rotation_deg: Annotated[
            Vec3 | None, Field(description="Absolute XYZ Euler rotation in degrees.")
        ] = None,
        scale: Annotated[Vec3 | None, Field(description="Absolute XYZ scale.")] = None,
        translate_delta: Annotated[
            Vec3 | None, Field(description="Relative translation offset.")
        ] = None,
        rotate_delta_deg: Annotated[
            Vec3 | None, Field(description="Relative rotation delta in degrees.")
        ] = None,
        parent_id: Annotated[str | None, Field(description="Set parent object ID or name.")] = None,
        unparent: Annotated[bool, Field(description="Clear parent relationship.")] = False,
        new_name: Annotated[str | None, Field(description="Rename object.")] = None,
        visible: Annotated[bool | None, Field(description="Toggle object visibility.")] = None,
        duplicate_as: Annotated[
            str | None, Field(description="Clone this object with a new name.")
        ] = None,
        delete: Annotated[bool, Field(description="Delete this object from the scene.")] = False,
    ) -> dict[str, Any]:
        """Set transforms, parent/unparent, duplicate, rename, hide, or delete an object."""
        sess = get_session()
        with sess.lock:
            obj = sess.require_object(object_id)

            if delete:
                sess.push_history(f"delete:{obj.name}")
                del sess.scene.objects[obj.id]
                for child in sess.scene.objects.values():
                    if child.parent_id == obj.id:
                        child.parent_id = None
                if sess.scene.active_camera_id == obj.id:
                    sess.scene.active_camera_id = None
                return {"status": "ok", "deleted_id": obj.id}

            if duplicate_as is not None:
                sess.push_history(f"duplicate:{obj.name}")
                clone_name = sess.scene.unique_name(duplicate_as or f"{obj.name}_copy")
                clone_id = sess.scene.next_id(clone_name.lower())
                clone = obj.model_copy(deep=True)
                clone.id = clone_id
                clone.name = clone_name
                sess.scene.objects[clone_id] = clone
                target_obj = clone
            else:
                sess.push_history(f"transform:{obj.name}")
                target_obj = obj

            if location is not None:
                target_obj.transform.location = location
            if translate_delta is not None:
                lx, ly, lz = target_obj.transform.location
                target_obj.transform.location = (
                    lx + translate_delta[0],
                    ly + translate_delta[1],
                    lz + translate_delta[2],
                )
            if rotation_deg is not None:
                target_obj.transform.rotation_deg = rotation_deg
            if rotate_delta_deg is not None:
                rx, ry, rz = target_obj.transform.rotation_deg
                target_obj.transform.rotation_deg = (
                    rx + rotate_delta_deg[0],
                    ry + rotate_delta_deg[1],
                    rz + rotate_delta_deg[2],
                )
            if scale is not None:
                try:
                    target_obj.transform = Transform(
                        location=target_obj.transform.location,
                        rotation_deg=target_obj.transform.rotation_deg,
                        scale=scale,
                    )
                except ValidationError as exc:
                    raise ToolError(f"Invalid scale: {exc}") from exc

            if unparent:
                target_obj.parent_id = None
            elif parent_id is not None:
                p_obj = sess.require_object(parent_id)
                if p_obj.id == target_obj.id:
                    raise ToolError("An object cannot be its own parent.")
                target_obj.parent_id = p_obj.id

            if new_name is not None:
                target_obj.name = sess.scene.unique_name(new_name)
            if visible is not None:
                target_obj.visible = visible

            return {
                "status": "ok",
                "object_id": target_obj.id,
                "name": target_obj.name,
                "transform": target_obj.transform.model_dump(),
                "parent_id": target_obj.parent_id,
            }

    @mcp.tool(
        title="Import External 3D Mesh",
        annotations=ToolAnnotations(open_world_hint=False),
    )
    def import_mesh(
        path: Annotated[
            str, Field(description="Path to .glb, .gltf, .obj, .stl, or .ply inside workspace.")
        ],
        name_prefix: Annotated[
            str | None, Field(description="Optional name prefix for imported objects.")
        ] = None,
        collection_id: Annotated[str, Field(description="Destination collection ID.")] = "default",
    ) -> dict[str, Any]:
        """Import external mesh files (.glb, .gltf, .obj, .stl, .ply) into the scene."""
        sess = get_session()
        with sess.lock:
            p = sess.resolve_path(path, must_exist=True)
            sess.push_history(f"import:{p.name}")
            created = import_mesh_file(
                sess.scene, p, name_prefix=name_prefix, collection_id=collection_id
            )
            return {
                "status": "ok",
                "imported": [{"id": o.id, "name": o.name} for o in created],
            }

    @mcp.tool(
        title="Edit Mesh Geometry",
        annotations=ToolAnnotations(open_world_hint=False),
    )
    def mesh_edit(
        object_id: Annotated[str, Field(description="Mesh object ID or name.")],
        operation: Annotated[
            Literal[
                "extrude_faces",
                "inset_faces",
                "bevel_edges",
                "subdivide",
                "transform_selection",
                "smooth",
                "recompute_normals",
                "repair_manifold",
            ],
            Field(description="Edit Mesh operation to apply destructively to base mesh."),
        ],
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
            ],
            Field(description="Semantic selection rule for face-based edit operations."),
        ] = "top",
        direction: Annotated[
            Vec3 | None, Field(description="Custom direction when selection_mode='normal_dir'.")
        ] = None,
        angle_threshold_deg: Annotated[float, Field(ge=1.0, le=180.0)] = 45.0,
        bbox_min: Annotated[
            Vec3 | None, Field(description="Selection box min when selection_mode='bbox'.")
        ] = None,
        bbox_max: Annotated[
            Vec3 | None, Field(description="Selection box max when selection_mode='bbox'.")
        ] = None,
        indices: Annotated[
            list[int] | None,
            Field(description="Explicit face indices when selection_mode='indices'."),
        ] = None,
        distance: Annotated[float, Field(description="Extrude distance.")] = 0.3,
        thickness: Annotated[float, Field(gt=0.0, description="Inset thickness.")] = 0.1,
        depth: Annotated[float, Field(description="Inset depth offset.")] = 0.0,
        width: Annotated[float, Field(gt=0.0, description="Bevel width.")] = 0.04,
        segments: Annotated[int, Field(ge=1, le=4, description="Bevel segments.")] = 1,
        levels: Annotated[int, Field(ge=1, le=3, description="Subdivision levels.")] = 1,
        smooth_subdiv: Annotated[
            bool, Field(description="Use Loop smoothing during subdivision.")
        ] = True,
        scale: Annotated[float, Field(gt=0.0, description="Uniform face scale for extrude.")] = 1.0,
        translate: Annotated[Vec3, Field(description="Translation for transform_selection.")] = (
            0.0,
            0.0,
            0.0,
        ),
        rotate_deg: Annotated[
            Vec3, Field(description="Rotation in degrees for transform_selection.")
        ] = (0.0, 0.0, 0.0),
        iterations: Annotated[int, Field(ge=1, le=25, description="Smoothing iterations.")] = 2,
        factor: Annotated[
            float, Field(ge=0.0, le=1.0, description="Laplacian smooth factor.")
        ] = 0.5,
    ) -> dict[str, Any]:
        """Perform Blender-style Edit Mesh operations using semantic face/edge queries."""
        sess = get_session()
        with sess.lock:
            obj = sess.require_object(object_id, expected_kind="mesh")
            if obj.mesh is None:
                raise ToolError(f"Object {obj.name!r} has no editable base mesh.")
            cur = MeshData.from_payload(obj.mesh)
            query = SelectionQuery(
                mode=selection_mode,
                direction=direction,
                angle_threshold_deg=angle_threshold_deg,
                bbox_min=bbox_min,
                bbox_max=bbox_max,
                indices=indices,
            )
            try:
                if operation == "extrude_faces":
                    res = extrude_faces(cur, query, distance=distance, scale=scale)
                elif operation == "inset_faces":
                    res = inset_faces(cur, query, thickness=thickness, depth=depth)
                elif operation == "bevel_edges":
                    res = bevel_edges(
                        cur, angle_threshold_deg=angle_threshold_deg, width=width, segments=segments
                    )
                elif operation == "subdivide":
                    res = subdivide_mesh(cur, levels=levels, smooth=smooth_subdiv)
                elif operation == "transform_selection":
                    res = transform_selection(
                        cur, query, translate=translate, rotate_deg=rotate_deg
                    )
                elif operation == "smooth":
                    res = smooth_laplacian(cur, iterations=iterations, factor=factor)
                elif operation == "recompute_normals":
                    res = recompute_normals(cur)
                else:
                    res = repair_manifold(cur)
            except GeometryError as exc:
                raise ToolError(str(exc)) from exc

            sess.push_history(f"mesh_edit:{obj.name}:{operation}")
            obj.mesh = res.to_payload()
            ev = evaluate_object_mesh(sess.scene, obj.id)
            return {
                "status": "ok",
                "object_id": obj.id,
                "operation": operation,
                "stats": ev.world_mesh.stats(),
            }

    @mcp.tool(
        title="CSG Boolean Operation",
        annotations=ToolAnnotations(open_world_hint=False),
    )
    def mesh_boolean(
        object_id: Annotated[str, Field(description="Primary mesh object ID or name.")],
        target_object_id: Annotated[str, Field(description="Operand mesh object ID or name.")],
        operation: Annotated[
            Literal["union", "difference", "intersection"],
            Field(description="CSG Boolean operation."),
        ] = "difference",
        keep_target: Annotated[
            bool, Field(description="Keep operand object in scene after CSG.")
        ] = False,
        result_name: Annotated[
            str | None,
            Field(description="If set, creates a new object instead of overwriting object_id."),
        ] = None,
    ) -> dict[str, Any]:
        """Execute watertight manifold3d CSG Boolean in world space."""
        sess = get_session()
        with sess.lock:
            obj_a = sess.require_mesh_like(object_id)
            obj_b = sess.require_mesh_like(target_object_id)
            if obj_a.id == obj_b.id:
                raise ToolError("Boolean operands must be two distinct objects.")
            ev_a = evaluate_object_mesh(sess.scene, obj_a.id)
            ev_b = evaluate_object_mesh(sess.scene, obj_b.id)
            try:
                world_res = boolean_meshes(ev_a.world_mesh, ev_b.world_mesh, operation)
            except GeometryError as exc:
                raise ToolError(f"Boolean CSG failed: {exc}") from exc

            sess.push_history(f"boolean:{operation}:{obj_a.name}")
            if result_name:
                r_name = sess.scene.unique_name(result_name)
                r_id = sess.scene.next_id(r_name.lower())
                new_obj = SceneObject(
                    id=r_id,
                    name=r_name,
                    kind="mesh",
                    transform=Transform(),
                    material_id=obj_a.material_id,
                    collection_id=obj_a.collection_id,
                    mesh=world_res.to_payload(),
                )
                sess.scene.objects[r_id] = new_obj
                out_obj = new_obj
            else:
                # Result is in world space, reset transform & modifiers on primary object
                obj_a.kind = "mesh"
                obj_a.node_graph = None
                obj_a.modifiers = []
                obj_a.transform = Transform()
                obj_a.mesh = world_res.to_payload()
                out_obj = obj_a

            if not keep_target and obj_b.id in sess.scene.objects:
                del sess.scene.objects[obj_b.id]

            return {
                "status": "ok",
                "object_id": out_obj.id,
                "name": out_obj.name,
                "operation": operation,
                "stats": world_res.stats(),
            }

    @mcp.tool(
        title="Add Non-Destructive Modifier",
        annotations=ToolAnnotations(open_world_hint=False),
    )
    def modifier_add(
        object_id: Annotated[
            str, Field(description="Target mesh or procedural object ID or name.")
        ],
        modifier_type: Annotated[
            Literal[
                "mirror",
                "array",
                "solidify",
                "subdivision",
                "bevel",
                "boolean",
                "taper",
                "twist",
                "bend",
                "displace",
                "smooth",
            ],
            Field(description="Non-destructive modifier type."),
        ],
        params: Annotated[
            dict[str, Any] | None, Field(description="Modifier parameters dictionary.")
        ] = None,
        name: Annotated[str | None, Field(description="Optional custom modifier name.")] = None,
    ) -> dict[str, Any]:
        """Append a non-destructive modifier to an object's modifier stack."""
        sess = get_session()
        with sess.lock:
            obj = sess.require_mesh_like(object_id)
            cls = _MODIFIER_CLASSES[modifier_type]
            payload = dict(params or {})
            payload["type"] = modifier_type
            if name:
                payload["name"] = name
            try:
                mod: ModifierSpec = cls.model_validate(payload)
            except ValidationError as exc:
                raise ToolError(
                    f"Invalid modifier parameters for {modifier_type!r}: {exc}"
                ) from exc

            sess.push_history(f"modifier_add:{obj.name}:{modifier_type}")
            obj.modifiers.append(mod)
            try:
                ev = evaluate_object_mesh(sess.scene, obj.id)
            except GeometryError as exc:
                obj.modifiers.pop()
                raise ToolError(f"Modifier evaluation failed: {exc}") from exc
            return {
                "status": "ok",
                "object_id": obj.id,
                "modifier_index": len(obj.modifiers) - 1,
                "modifier": mod.model_dump(),
                "stats": ev.world_mesh.stats(),
            }

    @mcp.tool(
        title="Manage Modifier Stack",
        annotations=ToolAnnotations(open_world_hint=False),
    )
    def modifier_manage(
        object_id: Annotated[str, Field(description="Object ID or name.")],
        action: Annotated[
            Literal["list", "toggle", "remove", "reorder", "apply_all"],
            Field(description="Modifier stack operation."),
        ],
        modifier_index: Annotated[
            int | None, Field(ge=0, description="0-based index in modifier stack.")
        ] = None,
        new_index: Annotated[
            int | None, Field(ge=0, description="Target index for action='reorder'.")
        ] = None,
        enabled: Annotated[
            bool | None, Field(description="Enable/disable state for action='toggle'.")
        ] = None,
    ) -> dict[str, Any]:
        """List, toggle, remove, reorder, or bake/apply all modifiers onto base mesh."""
        sess = get_session()
        with sess.lock:
            obj = sess.require_mesh_like(object_id)
            if action == "list":
                return {
                    "object_id": obj.id,
                    "modifiers": [
                        {"index": i, **m.model_dump()} for i, m in enumerate(obj.modifiers)
                    ],
                }

            if action == "apply_all":
                ev = evaluate_object_mesh(sess.scene, obj.id)
                sess.push_history(f"modifier_apply_all:{obj.name}")
                obj.kind = "mesh"
                obj.node_graph = None
                obj.mesh = ev.local_mesh.to_payload()
                obj.modifiers = []
                return {"status": "ok", "object_id": obj.id, "stats": ev.world_mesh.stats()}

            if modifier_index is None or modifier_index >= len(obj.modifiers):
                raise ToolError(
                    f"Invalid modifier_index={modifier_index}; "
                    f"object has {len(obj.modifiers)} modifiers."
                )

            sess.push_history(f"modifier_{action}:{obj.name}")
            if action == "toggle":
                mod = obj.modifiers[modifier_index]
                mod.enabled = (not mod.enabled) if enabled is None else enabled
            elif action == "remove":
                obj.modifiers.pop(modifier_index)
            elif action == "reorder":
                if new_index is None or new_index >= len(obj.modifiers):
                    raise ToolError("Valid new_index is required for action='reorder'.")
                mod = obj.modifiers.pop(modifier_index)
                obj.modifiers.insert(new_index, mod)

            ev = evaluate_object_mesh(sess.scene, obj.id)
            return {
                "status": "ok",
                "object_id": obj.id,
                "modifiers": [{"index": i, **m.model_dump()} for i, m in enumerate(obj.modifiers)],
                "stats": ev.world_mesh.stats(),
            }

    @mcp.tool(
        title="Inspect Mesh Topology",
        annotations=ToolAnnotations(read_only_hint=True, open_world_hint=False),
    )
    def mesh_inspect_topology(
        object_id: Annotated[str, Field(description="Mesh or procedural object ID or name.")],
    ) -> dict[str, Any]:
        """Return vertex/face/edge counts, watertight flags, volume, area, and bounds."""
        sess = get_session()
        with sess.lock:
            obj = sess.require_mesh_like(object_id)
            ev = evaluate_object_mesh(sess.scene, obj.id)
            return {
                "object_id": obj.id,
                "name": obj.name,
                "local_stats": ev.local_mesh.stats(),
                "world_stats": ev.world_mesh.stats(),
            }
