# FrFruit MCP Tools, Resources & Prompts Reference

FrFruit exposes **34 MCP tools**, **6 static resources + 1 resource template**, and **3 guided workflow prompts** using the official `mcp` Python SDK (`MCPServer`). All modeling, spatial, procedural, physics, rigging, path-traced rendering, video, and FBX/GLB export operations run deterministically and headlessly — or co-hosted live with **FrFruit Studio**.

---

## Table of Contents

1. [Scene Lifecycle, Undo/Redo & Collections (6 Tools)](#1-scene-lifecycle-undoredo--collections-6-tools)
2. [3D Modeling, Curves, Edit Mesh, CSG & Modifiers (9 Tools)](#2-3d-modeling-curves-edit-mesh-csg--modifiers-9-tools)
3. [Procedural Node Graphs — Geometry Nodes for AI (3 Tools + 18 Nodes)](#3-procedural-node-graphs--geometry-nodes-for-ai-3-tools--18-nodes)
4. [Spatial Reasoning, Collision & Smart Placement (2 Tools)](#4-spatial-reasoning-collision--smart-placement-2-tools)
5. [Materials, Textures, Lighting, Rendering, Post-FX & Export (8 Tools)](#5-materials-textures-lighting-rendering-post-fx--export-8-tools)
6. [Keyframe Animation, Physics/Cloth, Skeletal Rigging & Video (6 Tools)](#6-keyframe-animation-physicscloth-skeletal-rigging--video-6-tools)
7. [MCP Resources & Prompts](#7-mcp-resources--prompts)

---

## 1. Scene Lifecycle, Undo/Redo & Collections (6 Tools)

| Tool | Purpose | Key Arguments |
| --- | --- | --- |
| `scene_new` | Reset workspace to a clean default scene with camera and studio lights. | `name="FrFruitScene"`, `clear_history=True` |
| `scene_open` | Load a `.frf.json` project file inside the workspace sandbox. | `path` |
| `scene_save` | Save the complete `SceneDocument` (meshes, rigs, physics caches, post-FX, timeline) as `.frf.json`. | `path="scene.frf.json"` |
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
Full Blender-style Edit Mode modeling suite (`36` operations) with semantic or explicit element selection and automatic per-face material slot preservation:
- **Extrude**: `extrude_faces`, `extrude_individual`, `extrude_along_normals`, `extrude_manifold`, `extrude_edges`, `extrude_vertices`.
- **Inset & Bevel**: `inset_faces` (`individual=True/False`), `bevel_edges` (angle threshold or targeted edges), `bevel_vertices`.
- **Cut, Subdivide & Shell**: `subdivide` (Quad B-Mesh **Catmull-Clark** with optional `crease_angle_deg` sharp-edge preservation, `loop`, `simple`, or conformal selected-faces split via `subdiv_mode`), `loop_cut` (`axis`, `cuts`, `slide`, or custom `direction`), `bisect_plane` (`clear_inner`, `clear_outer`, `fill_caps`), `poke_faces`, `bridge_edge_loops`, `fill_holes`, `solidify_faces`, `wireframe_edges`.
- **Topology & Components**: `merge_vertices` (`by_distance`, `at_center`), `delete_elements` (`faces`, `vertices`, `edges`, `only_faces`), `dissolve_faces`, `duplicate_faces`, `split_selection`, `separate_selection` (creates new `SceneObject`).
- **Deform, Normals, UVs & Face Materials**: `transform_selection` (with Proportional Editing `proportional_radius` & `proportional_falloff`), `shrink_fatten`, `to_sphere`, `shear`, `symmetrize`, `randomize_vertices`, `smooth`, `flip_normals`, `recompute_normals`, `repair_manifold`, `project_uvs` (`box`, `smart_uv` angle-split island packing with `uv_angle_limit_deg` & `island_margin`, `planar_x/y/z`, `cylindrical`, `spherical`), `assign_face_material`, `clear_face_materials`.
- **Selection (`selection_mode`)**: `all`, `top`, `bottom`, `front`, `back`, `left`, `right`, `normal_dir`, `bbox`, `indices`, `vertex_indices`, `edge_indices`, `coplanar` (with `seed_face`), `boundary`, `material` (with `selection_material_id`), plus `invert=True`.

### `mesh_boolean`
Execute exact watertight CSG (`operation="union" | "difference" | "intersection"`) in world space via `manifold3d` (`object_id`, `target_object_id`, `keep_target=False`, `result_name=None`).

### `modifier_add` & `modifier_manage`
Manage non-destructive modifier stacks evaluated dynamically on top of base meshes or procedural graphs:
- **11 Modifier Types**: `mirror`, `array` (`linear` / `radial`), `solidify`, `subdivision` (`algorithm="catmull_clark"|"loop"|"simple"`, `crease_angle_deg`), `bevel`, `boolean`, `taper`, `twist`, `bend`, `displace`, `smooth`.
- `modifier_manage` supports `list`, `toggle`, `remove`, `reorder`, and `apply_all`.

### `mesh_inspect_topology`
Inspect local/world topology (`vertex_count`, `face_count`, `edge_count`, `is_watertight`, `is_winding_consistent`, signed `volume`, `surface_area`, bounding box extents), plus paginated Edit Mode details via `detail="summary" | "faces" | "edges" | "vertices" | "selection" | "materials"` (`offset`, `limit`, and optional selection preview params).

---

## 3. Procedural Node Graphs — Geometry Nodes for AI (3 Tools + 18 Nodes)

### Tools
- `node_catalog_list(category=None)`: Query schemas, sockets, and defaults dynamically (`generator`, `points`, `transform`, `combine`, `csg`, `mesh`, `deform`, `scatter`, `output`).
- `node_graph_set(object_id_or_name, nodes, links, seed=42, ...)`: Define a full DAG (`ProceduralGraph`) terminating in `GroupOutput`.
- `node_graph_patch(object_id, upsert_nodes=..., remove_node_ids=..., add_links=..., remove_links=..., seed=...)`: Incrementally tweak parameters or topology without resending the whole graph.

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

## 5. Materials, Textures, Lighting, Rendering, Post-FX & Export (8 Tools)

### `material_create_or_assign`
Actions: `create`, `assign`, `create_and_assign`, `assign_faces`, `clear_faces`, `list_presets`. Supports whole-object OR single-face / selected-face material slot assignment (`face_index`, `face_indices`, or `selection_mode` + `seed_face`/`invert`). Uses Cook-Torrance **GGX BRDF** supporting `base_color`, `secondary_color`, `metallic`, `roughness`, `clearcoat`, `clearcoat_roughness`, `sheen`, `subsurface`, `subsurface_color`, `emission_color`, `emission_strength`, `alpha`, and procedural patterns (`solid`, `checker`, `gradient`, `wood`, `noise`).
- **Built-in Presets (12)**: `clay`, `matte_plastic`, `glossy_ceramic`, `chrome`, `gold`, `copper`, `wood`, `apple_skin`, `leaf_green`, `glass`, `emissive_cyan`, `emissive_amber`.

### `texture_library`
Search & fetch online multi-channel PBR texture packs (**Poly Haven**, **ambientCG**) or generate seamless local procedural PBR maps (`albedo`, `normal`, `roughness`, `metallic`, `ao`, `height`):
- `action`: `"search"` | `"download_and_apply"` | `"generate_procedural"`.
- `source`: `"all"` | `"polyhaven"` | `"ambientcg"` | `"procedural"`.
- Automatically unpacks ARM channel-packed maps, builds `MaterialSpec`, assigns to `object_id`, and supports world-space triplanar UV fallback + Gram-Schmidt TBN bump mapping.

### `light_manage` & `camera_manage`
- `light_manage`: `add`, `update`, `delete`, `setup_studio_3point`, `set_world` (`sky_mode="gradient"|"physical_sky"|"hdri"`, `hdri_path`, `hdri_rotation_deg`, `hdri_intensity`, `sun_elevation_deg`, `sun_azimuth_deg`, `air_turbidity`, `horizon_color`, `zenith_color`, `ambient_strength`), `list` (`sun`, `point`, `spot`, `area`).
- `camera_manage`: `add`, `update`, `set_active`, `auto_frame`, `delete`, `list` (perspective & orthographic).

### `post_process_manage`
Configure HDR post-processing and default scene render engine:
- `action`: `"get"` | `"configure"` | `"set_preset"` | `"set_engine"`.
- `render_engine`: `"software"` | `"raytrace"` (CPU BVH Path Tracer).
- `preset`: `"cinematic"`, `"architectural"`, `"studio_clean"`, `"dramatic_bloom"`, `"raw_off"`.
- Fine controls: edge-aware A-Trous bilateral `denoise_enabled` & `denoise_strength`, `tone_mapper` (`aces`, `filmic`, `reinhard`, `none`), `exposure`, `bloom_*`, `ssao_*`, `dof_*`, `vignette_intensity`, `chromatic_aberration`, `saturation`, `contrast`, `temperature`, `sharpen`, `fxaa_enabled`.

### `render_viewport` & `render_views`
Headless rendering returning `[TextContent(metadata_json), ImageContent(png_base64)]`:
- `engine`: `None` (uses scene default), `"software"`, or `"raytrace"` (with `samples=1..16`).
- **6 Viewport Shading Modes (`mode`)**: `pbr`, `clay`, `wireframe`, `normals`, `depth`, `collision`.
- **Viewport Overlays**: `show_grid`, `show_axes`, `show_bounds`, `show_labels`.
- `render_views`: 2x2 multi-angle contact sheet (`front`, `right`, `top`, `iso` by default).

### `export_scene`
Export scene to **`.fbx`** (Kaydara Binary FBX 7.400), **`.fbx` (`format="fbx_ascii"`)**, **`.glb`**, **`.gltf`**, **`.obj`**, **`.stl`**, **`.ply`**, or standalone Blender **`.py`**.
- `target_engine` (`"generic" | "unreal" | "unity" | "godot"`): Automatically handles Z-up vs Y-up axes, `UnitScaleFactor`, skeletal `LimbNode`s + `BindPose`, animation `AnimationCurve`s, baked cloth `BlendShape` morph targets, tangents/binormals, and textures.

---

## 6. Keyframe Animation, Physics/Cloth, Skeletal Rigging & Video (6 Tools)

- `timeline_manage`: `get`, `set_range` (`fps`, `frame_start`, `frame_end`), `set_frame` (`current_frame`).
- `keyframe_manage`: `set`, `delete`, `list` scalar F-curves (`constant`, `linear`, `bezier` with `auto`/`ease_in`/`ease_out`/`ease_in_out`) targeting transforms (`location.x/y/z`, `rotation_deg.x/y/z`, `scale.x/y/z`), lights (`intensity`), cameras (`fov_deg`), or materials (`emission_strength`).
- `render_animation`: Render `kind="turntable"` or `kind="timeline"` sequences (`steps=2..24`, optional `engine` / `samples`) returning filmstrip PNG `ImageContent` and optional `.gif` / `.png` outputs.
- `render_video_file`: Render high-frame-rate **`.mp4` (H.264)**, **`.webm` (VP9)**, **`.avi`**, or **`.gif`** videos for `kind="timeline" | "turntable" | "camera_path"` (with Catmull-Rom `camera_waypoints`), plus optional PNG frame sequences (`frames_dir`).
- `physics_manage`:
  - `set_rigid_body`: Configure `dynamic`/`static`/`kinematic` rigid bodies (`mass`, `restitution`, `friction`, `initial_velocity`, `collider_shape="box"|"sphere"|"mesh_aabb"`).
  - `set_cloth`: Configure vectorized XPBD cloth (`stiffness`, `bend_stiffness`, `damping`, `wind`, `pin_mode="none"|"top_corners"|"top_edge"|"border"|"indices"`, `collision_margin`, `substeps`).
  - `bake` / `clear_cache` / `list`: Simulate dynamics into F-curves + per-frame `vertex_cache`.
- `armature_rig`:
  - `create` / `add_bone`: Create FK hierarchical skeletons (`bones=[{"name", "parent", "head", "tail"}, ...]`).
  - `auto_rig`: Compute normalized top-K inverse-distance bone skin weights (`SkinWeightSpec`) binding any mesh to an armature.
  - `solve_ik`: FABRIK Inverse Kinematics solver (`bone_name`, `target_pos`, `chain_length`, optional `pole_target`, `ik_iterations`, and optional `frame` keyframing) computing local bone rotations across multi-bone chains.
  - `set_pose` / `keyframe_pose` / `list`: Pose or animate bone rotations/translations on `TimelineState`.

---

## 7. MCP Resources & Prompts

### Resources
- `frfruit://scene/summary` — Live hierarchy, materials, timeline summary, undo/redo status.
- `frfruit://scene/document` — Full serialized `.frf.json` `SceneDocument`.
- `frfruit://scene/postprocess` — Active `render_engine` and `PostProcessConfig`.
- `frfruit://catalog/nodes` — Full procedural node definitions & sockets.
- `frfruit://catalog/materials` — All 12 built-in PBR presets.
- `frfruit://catalog/textures` — Offline procedural styles and supported remote texture providers (`polyhaven`, `ambientcg`).
- `frfruit://object/{object_id}` — Detailed metadata, `vertex_cache_frames`, and evaluated world stats for a specific object.

### Prompts
- `build_3d_asset(concept, style="stylized")` — Construct -> place -> inspect -> shade -> render -> export loop.
- `fix_geometry_issues(focus="all")` — Collision/normal/wireframe audit and repair loop.
- `animate_turntable(subject="all")` — Timeline setup -> keyframes -> filmstrip/GIF preview -> export.
