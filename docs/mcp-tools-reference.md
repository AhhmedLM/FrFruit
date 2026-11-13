# FrFruit MCP Tools, Resources & Prompts Reference

FrFruit exposes **29 MCP tools**, **4 static resources + 1 resource template**, and **3 guided workflow prompts** using the official `mcp` Python SDK (`MCPServer`). All modeling, spatial, procedural, animation, rendering, and I/O operations run deterministically and headlessly.

---

## Table of Contents

1. [Scene Lifecycle, Undo/Redo & Collections (6 Tools)](#1-scene-lifecycle-undoredo--collections-6-tools)
2. [3D Modeling, Curves, Edit Mesh, CSG & Modifiers (9 Tools)](#2-3d-modeling-curves-edit-mesh-csg--modifiers-9-tools)
3. [Procedural Node Graphs — Geometry Nodes for AI (3 Tools + 18 Nodes)](#3-procedural-node-graphs--geometry-nodes-for-ai-3-tools--18-nodes)
4. [Spatial Reasoning, Collision & Smart Placement (2 Tools)](#4-spatial-reasoning-collision--smart-placement-2-tools)
5. [Materials, Lights, Cameras, Rendering & Export (6 Tools)](#5-materials-lights-cameras-rendering--export-6-tools)
6. [Keyframe Animation & Turntable Rendering (3 Tools)](#6-keyframe-animation--turntable-rendering-3-tools)
7. [MCP Resources & Prompts](#7-mcp-resources--prompts)

---

## 1. Scene Lifecycle, Undo/Redo & Collections (6 Tools)

| Tool | Purpose | Key Arguments |
| --- | --- | --- |
| `scene_new` | Reset workspace to a clean default scene with camera and 3-point/key-fill studio lights. | `name="FrFruitScene"`, `clear_history=True` |
| `scene_open` | Load a `.frf.json` project file inside the workspace sandbox. | `path` |
| `scene_save` | Save the complete `SceneDocument` (meshes, modifiers, node graphs, timeline) as `.frf.json`. | `path="scene.frf.json"` |
| `scene_inspect` | Inspect scene hierarchy, evaluated world bounds, triangle counts, or a single object's details. | `detail="summary"\|"full"`, `object_id=None` |
| `scene_undo_redo` | Step backward/forward through mutation history or manage named snapshots. | `action` (`undo` \| `redo` \| `create_snapshot` \| `restore_snapshot` \| `list_snapshots`), `snapshot_name=None` |
| `collection_manage` | Organize objects into Blender-style collections and toggle visibility. | `action` (`create` \| `assign` \| `set_visibility` \| `delete` \| `list`), `collection_id`, `object_ids`, `visible` |

---

## 2. 3D Modeling, Curves, Edit Mesh, CSG & Modifiers (9 Tools)

### `create_primitive`
Create watertight primitive geometry: `cube`, `sphere`, `icosphere`, `cylinder`, `cone`, `torus`, `plane`, `capsule`.
- **Origin alignment**: `origin="bottom"` (local min-Z = 0, ready to rest on `Z=0`) or `"center"`.
- **Common arguments**: `primitive`, `name`, `size`, `radius`, `height`, `radius_top`, `major_radius`, `minor_radius`, `segments`, `subdivisions`, `location`, `rotation_deg`, `scale`, `material_id`, `collection_id`.

### `create_from_curve`
Construct watertight meshes from 2D profiles or 3D spline paths:
- `mode="extrude_2d"`: 2D polygon `[x, y]` points extruded along `+Z` with optional `height`, `twist_deg`, `scale_top`.
- `mode="revolve_lathe"`: 2D meridian profile `[radius, z]` revolved around `+Z` (`angle_deg`, `segments`).
- `mode="sweep_tube"`: 3D `[x, y, z]` path swept with circular profile (`radius`, `taper_end`, `closed`, `segments`).

### `object_transform`
Set absolute `location`/`rotation_deg`/`scale` or apply `translate_delta`/`rotate_delta_deg`, change `parent_id`/`unparent`, `new_name`, toggle `visible`, `duplicate_as`, or `delete=True`.

### `import_mesh`
Import `.glb`, `.gltf`, `.obj`, `.stl`, or `.ply` files from inside the workspace (`path`, `name_prefix`, `collection_id`).

### `mesh_edit`
Semantic Blender-style Edit Mesh operations without requiring raw vertex index guessing:
- **Operations**: `extrude_faces`, `inset_faces`, `bevel_edges`, `subdivide`, `transform_selection`, `smooth`, `recompute_normals`, `repair_manifold`.
- **Semantic selection (`selection_mode`)**: `all`, `top`, `bottom`, `front`, `back`, `left`, `right`, `normal_dir` (with `direction`), `bbox` (with `bbox_min`/`bbox_max`), or `indices`.

### `mesh_boolean`
Execute exact watertight CSG (`operation="union" | "difference" | "intersection"`) in world space via `manifold3d` (`object_id`, `target_object_id`, `keep_target=False`, `result_name=None`).

### `modifier_add` & `modifier_manage`
Manage non-destructive modifier stacks evaluated dynamically on top of base meshes or procedural graphs:
- **11 Modifier Types**: `mirror`, `array` (`linear` / `radial`), `solidify`, `subdivision`, `bevel`, `boolean`, `taper`, `twist`, `bend`, `displace`, `smooth`.
- `modifier_manage` supports `list`, `toggle`, `remove`, `reorder`, and `apply_all`.

### `mesh_inspect_topology`
Return local and world-space vertex/face/edge counts, `is_watertight`, `is_winding_consistent`, signed `volume`, `surface_area`, and bounding box extents.

---

## 3. Procedural Node Graphs — Geometry Nodes for AI (3 Tools + 18 Nodes)

### Tools
- `node_catalog_list(category=None)`: Query schemas, sockets, and defaults dynamically (`generator`, `points`, `transform`, `combine`, `csg`, `mesh`, `deform`, `scatter`, `output`).
- `node_graph_set(object_id_or_name, nodes, links, seed=42, ...)`: Define a full DAG (`ProceduralGraph`) terminating in `GroupOutput`.
- `node_graph_patch(object_id, upsert_nodes=..., remove_node_ids=..., add_links=..., remove_links=..., seed=...)`: Incrementally tweak parameters or topology without resending the whole graph.

### Example `node_graph_set` Payload
```json
{
  "object_id_or_name": "GoldDrops",
  "seed": 7,
  "nodes": [
    {"id": "ring", "type": "CurveCircle", "params": {"radius": 1.12, "segments": 10}},
    {"id": "drop", "type": "PrimitiveSphere", "params": {"radius": 0.085, "segments": 16}},
    {"id": "inst", "type": "InstanceOnPoints", "params": {"scale": 1.0}},
    {"id": "out", "type": "GroupOutput", "params": {}}
  ],
  "links": [
    {"from_node": "ring", "from_socket": "geometry", "to_node": "inst", "to_socket": "points"},
    {"from_node": "drop", "from_socket": "geometry", "to_node": "inst", "to_socket": "instance"},
    {"from_node": "inst", "from_socket": "geometry", "to_node": "out", "to_socket": "geometry"}
  ]
}
```

### Supported Node Types (18)
| Category | Node Types |
| --- | --- |
| `generator` | `PrimitiveCube`, `PrimitiveSphere`, `PrimitiveCylinder`, `PrimitiveCone`, `PrimitiveTorus` |
| `points` | `GridPoints`, `CurveCircle`, `CurveLine` |
| `transform` | `TransformGeometry` |
| `combine` | `JoinGeometry` |
| `csg` | `BooleanGeometry` |
| `mesh` | `SubdivideGeometry`, `ConvexHull`, `ExtrudeGeometry` |
| `deform` | `SetPositionNoise` |
| `scatter` | `DistributePointsOnFaces`, `InstanceOnPoints` |
| `output` | `GroupOutput` |

---

## 4. Spatial Reasoning, Collision & Smart Placement (2 Tools)

### `spatial_measure`
Read-only geometric queries (`action`):
- `bounding_box`: World-space min, max, center, size of `object_id`.
- `distance`: Exact closest-point surface distance and gap vector between `object_id` and `target_object_id`.
- `raycast`: Moller-Trumbore ray intersection returning hit point, normal, distance, and hit object ID.
- `check_intersections`: Pairwise overlap detection (`manifold3d` / SAT) checking accidental clipping above `tolerance`.

### `spatial_place`
Deterministic placement (`action`):
- `ground`: Rest object cleanly at target `z` plane.
- `place_on_top`: Rest `object_id` atop `base_object_id` (`align_xy="center" | "keep"`, optional `offset`).
- `attach_to_side`: Flush-mount `object_id` onto `+x`, `-x`, `+y`, `-y`, `+z`, or `-z` face of `base_object_id`.
- `snap_to_surface`: Ray-project `object_id` onto curved surface of `base_object_id` along `direction` with optional `gap`.
- `align` & `distribute`: Align or evenly space multiple `object_ids` along `x`, `y`, or `z`.

---

## 5. Materials, Lights, Cameras, Rendering & Export (6 Tools)

### `material_create_or_assign`
Actions: `create`, `assign`, `create_and_assign`, `list_presets`. Supports linear PBR properties (`base_color`, `secondary_color`, `metallic`, `roughness`, `emission_color`, `emission_strength`, `alpha`) and procedural surface patterns (`solid`, `checker`, `gradient`, `wood`, `noise`).
- **Built-in Presets (12)**: `clay`, `matte_plastic`, `glossy_ceramic`, `chrome`, `gold`, `copper`, `wood`, `apple_skin`, `leaf_green`, `glass`, `emissive_cyan`, `emissive_amber`.

### `light_manage` & `camera_manage`
- `light_manage`: `add`, `update`, `delete`, `setup_studio_3point`, `set_world`, `list` (`sun`, `point`, `spot`, `area`).
- `camera_manage`: `add`, `update`, `set_active`, `auto_frame`, `delete`, `list` (perspective & orthographic).

### `render_viewport` & `render_views`
Headless pure-CPU G-buffer rasterizer returning `[TextContent(metadata_json), ImageContent(png_base64)]`:
- **6 Viewport Shading Modes (`mode`)**: `pbr`, `clay`, `wireframe`, `normals`, `depth`, `collision`.
- **Viewport Overlays**: `show_grid`, `show_axes` (XYZ orientation triad), `show_bounds` (yellow wireframe AABBs), `show_labels` (overlap-free object tags).
- `render_views`: Renders a multi-angle 2x2 contact sheet (`front`, `right`, `top`, `iso` by default) so AI agents verify proportions and contacts in one turn.

### `export_scene`
Export scene to `.glb`, `.gltf` (including PBR materials and glTF 2.0 TRS animations), `.obj`, `.stl`, `.ply`, or standalone Blender `.py` build script.

---

## 6. Keyframe Animation & Turntable Rendering (3 Tools)

- `timeline_manage`: `get`, `set_range` (`fps`, `frame_start`, `frame_end`), `set_frame` (`current_frame`).
- `keyframe_manage`: `set`, `delete`, `list` scalar F-curves (`constant`, `linear`, `bezier` with `auto`/`ease_in`/`ease_out`/`ease_in_out`) targeting transforms (`location.x/y/z`, `rotation_deg.x/y/z`, `scale.x/y/z`), lights (`intensity`), cameras (`fov_deg`), or materials (`emission_strength`).
- `render_animation`: Render `kind="turntable"` or `kind="timeline"` sequences (`steps=2..24`) returning filmstrip PNG `ImageContent` and optionally writing `.gif` + `.png` files.

---

## 7. MCP Resources & Prompts

### Resources
- `frfruit://scene/summary` — Live hierarchy, materials, timeline summary, undo/redo counts.
- `frfruit://scene/document` — Full serialized `.frf.json` `SceneDocument`.
- `frfruit://catalog/nodes` — Full procedural node definitions & sockets.
- `frfruit://catalog/materials` — All 12 built-in PBR presets.
- `frfruit://object/{object_id}` — Detailed metadata & evaluated world stats for a specific object.

### Prompts
- `build_3d_asset(concept, style="stylized")` — Construct -> place -> inspect -> shade -> render -> export loop.
- `fix_geometry_issues(focus="all")` — Collision/normal/wireframe audit and repair loop.
- `animate_turntable(subject="all")` — Timeline setup -> keyframes -> filmstrip/GIF preview -> GLB export.
