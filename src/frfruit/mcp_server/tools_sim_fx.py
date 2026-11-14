"""MCP tools for online PBR textures, post-processing, video export, physics/cloth, and rigging."""

from __future__ import annotations

from collections.abc import Callable
from typing import Annotated, Any, Literal

import numpy as np
from mcp.server import MCPServer
from mcp.server.mcpserver.exceptions import ToolError
from mcp.types import ToolAnnotations
from pydantic import Field

from frfruit.animation.armature import (
    auto_rig_mesh,
    compute_armature_matrices,
    solve_armature_ik,
)
from frfruit.animation.timeline import FCurveTrack, Keyframe
from frfruit.core.math3d import transform_points
from frfruit.core.scene import (
    ArmatureSpec,
    BoneSpec,
    ClothSpec,
    PostProcessConfig,
    RigidBodySpec,
    SceneObject,
)
from frfruit.core.types import Transform, Vec2, Vec3
from frfruit.geometry.spatial import compute_world_matrix, evaluate_object_mesh
from frfruit.mcp_server.context import FrFruitSession
from frfruit.physics.solver import bake_scene_physics, clear_physics_caches
from frfruit.render.software import RenderMode
from frfruit.render.video import VideoKind, render_video
from frfruit.textures.fetcher import (
    build_material_from_maps,
    download_pbr_texture_pack,
    generate_procedural_pbr_maps,
    search_online_textures,
)

_POSTPROCESS_PRESETS: dict[str, dict[str, Any]] = {
    "cinematic": {
        "enabled": True,
        "tone_mapper": "aces",
        "exposure": 0.12,
        "bloom_enabled": True,
        "bloom_threshold": 0.85,
        "bloom_intensity": 0.34,
        "bloom_radius": 6.5,
        "ssao_enabled": True,
        "ssao_radius": 4.0,
        "ssao_strength": 0.42,
        "vignette_intensity": 0.24,
        "chromatic_aberration": 0.4,
        "saturation": 1.08,
        "contrast": 1.08,
        "temperature": 0.06,
        "fxaa_enabled": True,
        "sharpen": 0.12,
    },
    "architectural": {
        "enabled": True,
        "tone_mapper": "filmic",
        "exposure": 0.05,
        "bloom_enabled": True,
        "bloom_threshold": 0.96,
        "bloom_intensity": 0.18,
        "ssao_enabled": True,
        "ssao_radius": 4.5,
        "ssao_strength": 0.36,
        "vignette_intensity": 0.10,
        "chromatic_aberration": 0.0,
        "saturation": 1.03,
        "contrast": 1.04,
        "temperature": 0.04,
        "fxaa_enabled": True,
        "sharpen": 0.14,
    },
    "studio_clean": {
        "enabled": True,
        "tone_mapper": "aces",
        "exposure": 0.0,
        "bloom_enabled": True,
        "bloom_threshold": 0.95,
        "bloom_intensity": 0.16,
        "ssao_enabled": True,
        "ssao_strength": 0.25,
        "vignette_intensity": 0.08,
        "chromatic_aberration": 0.0,
        "saturation": 1.02,
        "contrast": 1.02,
        "temperature": 0.0,
        "fxaa_enabled": True,
        "sharpen": 0.08,
    },
    "dramatic_bloom": {
        "enabled": True,
        "tone_mapper": "aces",
        "exposure": 0.18,
        "bloom_enabled": True,
        "bloom_threshold": 0.72,
        "bloom_intensity": 0.65,
        "bloom_radius": 9.0,
        "ssao_enabled": True,
        "ssao_strength": 0.45,
        "vignette_intensity": 0.32,
        "chromatic_aberration": 0.8,
        "contrast": 1.14,
        "saturation": 1.12,
    },
    "raw_off": {
        "enabled": False,
    },
}


def register_sim_fx_tools(mcp: MCPServer, get_session: Callable[[], FrFruitSession]) -> None:
    @mcp.tool(
        title="Online & Procedural PBR Texture Library",
        annotations=ToolAnnotations(open_world_hint=True),
    )
    def texture_library(
        action: Annotated[
            Literal["search", "download_and_apply", "generate_procedural"],
            Field(
                description="Search Poly Haven / ambientCG, download maps, or synthesize PBR maps."
            ),
        ] = "search",
        query: Annotated[
            str, Field(description="Search query e.g. 'marble', 'gold', 'brick', 'wood', 'fabric'.")
        ] = "",
        source: Annotated[
            Literal["all", "polyhaven", "ambientcg", "procedural"],
            Field(description="Texture catalog source."),
        ] = "all",
        asset_id: Annotated[
            str | None, Field(description="Texture asset ID from search or procedural style name.")
        ] = None,
        resolution: Annotated[
            Literal["1k", "2k", "4k"], Field(description="Download resolution.")
        ] = "1k",
        map_types: Annotated[
            list[str] | None,
            Field(description="Subset of albedo, normal, roughness, metallic, ao, height."),
        ] = None,
        material_name: Annotated[
            str | None, Field(description="Name for created PBR MaterialSpec.")
        ] = None,
        object_id: Annotated[
            str | None,
            Field(description="Optional object ID or name to assign the texture material."),
        ] = None,
        uv_scale: Annotated[Vec2, Field(description="UV tiling multiplier [u, v].")] = (1.0, 1.0),
        save_dir: Annotated[
            str | None,
            Field(description="Optional workspace directory for storing downloaded textures."),
        ] = None,
        limit: Annotated[int, Field(ge=1, le=50)] = 10,
        fallback_to_procedural: Annotated[
            bool, Field(description="Fallback to local seamless synthesis if remote API fails.")
        ] = True,
    ) -> dict[str, Any]:
        """Search Poly Haven & ambientCG, download PBR maps, or bake procedural maps."""
        sess = get_session()
        if action == "search":
            try:
                hits = search_online_textures(query=query, source=source, limit=limit)
            except Exception as exc:
                if not fallback_to_procedural:
                    raise ToolError(str(exc)) from exc
                hits = search_online_textures(query=query, source="procedural", limit=limit)
            return {"count": len(hits), "source": source, "results": hits}

        with sess.lock:
            target_dir = (
                sess.resolve_path(save_dir) if save_dir else (sess.workspace_root / "textures")
            )
            aid = asset_id or (query.strip() or "stone")

            if action == "generate_procedural":
                manifest = generate_procedural_pbr_maps(
                    style=aid,
                    out_dir=target_dir / f"procedural_{aid}",
                    size=512 if resolution == "1k" else 1024,
                )
            else:
                dl_src: Literal["polyhaven", "ambientcg", "procedural"] = (
                    "polyhaven" if source == "all" else source  # type: ignore[assignment]
                )
                try:
                    manifest = download_pbr_texture_pack(
                        asset_id=aid,
                        source=dl_src,
                        resolution=resolution,
                        target_dir=target_dir,
                        map_types=map_types,
                        fallback_to_procedural=fallback_to_procedural,
                    )
                except Exception as exc:
                    raise ToolError(str(exc)) from exc

            maps: dict[str, str] = manifest.get("maps", {})
            mat_label = material_name or f"PBR_{aid}"
            mid = sess.scene.next_id(f"mat_{aid.lower()}")
            mat = build_material_from_maps(
                material_id=mid,
                name=mat_label,
                maps=maps,
                texture_source=f"{manifest.get('source', source)}:{aid}",
                uv_scale=(float(uv_scale[0]), float(uv_scale[1])),
            )
            sess.push_history(f"texture_apply:{mat_label}")
            sess.scene.materials[mid] = mat

            assigned_to: str | None = None
            if object_id:
                obj = sess.require_mesh_like(object_id)
                obj.material_id = mid
                assigned_to = obj.id

            sess.notify("texture_applied", {"material_id": mid, "object_id": assigned_to})
            return {
                "status": "ok",
                "manifest": manifest,
                "material_id": mid,
                "assigned_to": assigned_to,
            }

    @mcp.tool(
        title="Configure Post-Processing & Render Engine",
        annotations=ToolAnnotations(open_world_hint=False),
    )
    def post_process_manage(
        action: Annotated[
            Literal["get", "configure", "set_preset", "set_engine"],
            Field(description="Inspect, configure, apply preset, or switch render engine."),
        ] = "get",
        preset: Annotated[
            Literal["cinematic", "architectural", "studio_clean", "dramatic_bloom", "raw_off"]
            | None,
            Field(description="Post-processing preset name."),
        ] = None,
        render_engine: Annotated[
            Literal["software", "raytrace"] | None,
            Field(
                description=(
                    "Default render backend: 'software' (fast) or 'raytrace' (BVH path tracer)."
                )
            ),
        ] = None,
        enabled: Annotated[
            bool | None, Field(description="Master toggle for post-processing.")
        ] = None,
        denoise_enabled: Annotated[
            bool | None,
            Field(description="Enable edge-aware bilateral denoiser on raytraced renders."),
        ] = None,
        denoise_strength: Annotated[
            float | None, Field(ge=0.0, le=1.0, description="Denoiser smoothing strength.")
        ] = None,
        tone_mapper: Annotated[
            Literal["aces", "filmic", "reinhard", "none"] | None,
            Field(description="HDR tone-mapping operator."),
        ] = None,
        exposure: Annotated[float | None, Field(ge=-5.0, le=5.0)] = None,
        bloom_enabled: Annotated[bool | None, Field(description="Enable HDR bloom glow.")] = None,
        bloom_threshold: Annotated[float | None, Field(ge=0.1, le=10.0)] = None,
        bloom_intensity: Annotated[float | None, Field(ge=0.0, le=3.0)] = None,
        bloom_radius: Annotated[float | None, Field(ge=0.5, le=24.0)] = None,
        ssao_enabled: Annotated[bool | None, Field(description="Enable Screen-Space AO.")] = None,
        ssao_radius: Annotated[float | None, Field(ge=0.5, le=20.0)] = None,
        ssao_strength: Annotated[float | None, Field(ge=0.0, le=1.5)] = None,
        dof_enabled: Annotated[bool | None, Field(description="Enable Depth of Field.")] = None,
        dof_focus_distance: Annotated[float | None, Field(gt=0.05)] = None,
        dof_aperture: Annotated[float | None, Field(ge=0.0, le=2.0)] = None,
        vignette_intensity: Annotated[float | None, Field(ge=0.0, le=1.0)] = None,
        chromatic_aberration: Annotated[float | None, Field(ge=0.0, le=5.0)] = None,
        saturation: Annotated[float | None, Field(ge=0.0, le=2.5)] = None,
        contrast: Annotated[float | None, Field(ge=0.2, le=2.5)] = None,
        temperature: Annotated[float | None, Field(ge=-1.0, le=1.0)] = None,
        fxaa_enabled: Annotated[
            bool | None, Field(description="Enable FXAA anti-aliasing.")
        ] = None,
        sharpen: Annotated[float | None, Field(ge=0.0, le=1.0)] = None,
    ) -> dict[str, Any]:
        """Inspect/update HDR post-processing (Denoiser, Bloom, SSAO, DOF, ACES) & engine."""
        sess = get_session()
        with sess.lock:
            if action == "get":
                return {
                    "render_engine": sess.scene.render_engine,
                    "post_process": sess.scene.post_process.model_dump(),
                    "available_presets": sorted(_POSTPROCESS_PRESETS.keys()),
                }

            sess.push_history(f"postprocess:{action}")
            if action == "set_engine":
                if render_engine is None:
                    raise ToolError("render_engine ('software' or 'raytrace') is required.")
                sess.scene.render_engine = render_engine
            elif action == "set_preset":
                if preset is None or preset not in _POSTPROCESS_PRESETS:
                    raise ToolError(f"Unknown post-processing preset {preset!r}.")
                merged = sess.scene.post_process.model_dump()
                merged.update(_POSTPROCESS_PRESETS[preset])
                sess.scene.post_process = PostProcessConfig(**merged)
            else:
                if render_engine is not None:
                    sess.scene.render_engine = render_engine
                updates: dict[str, Any] = {
                    k: v
                    for k, v in {
                        "enabled": enabled,
                        "denoise_enabled": denoise_enabled,
                        "denoise_strength": denoise_strength,
                        "tone_mapper": tone_mapper,
                        "exposure": exposure,
                        "bloom_enabled": bloom_enabled,
                        "bloom_threshold": bloom_threshold,
                        "bloom_intensity": bloom_intensity,
                        "bloom_radius": bloom_radius,
                        "ssao_enabled": ssao_enabled,
                        "ssao_radius": ssao_radius,
                        "ssao_strength": ssao_strength,
                        "dof_enabled": dof_enabled,
                        "dof_focus_distance": dof_focus_distance,
                        "dof_aperture": dof_aperture,
                        "vignette_intensity": vignette_intensity,
                        "chromatic_aberration": chromatic_aberration,
                        "saturation": saturation,
                        "contrast": contrast,
                        "temperature": temperature,
                        "fxaa_enabled": fxaa_enabled,
                        "sharpen": sharpen,
                    }.items()
                    if v is not None
                }
                if updates:
                    sess.scene.post_process = sess.scene.post_process.model_copy(update=updates)

            sess.notify("postprocess_updated", {"render_engine": sess.scene.render_engine})
            return {
                "status": "ok",
                "render_engine": sess.scene.render_engine,
                "post_process": sess.scene.post_process.model_dump(),
            }

    @mcp.tool(
        title="Render Video (MP4 / WebM / AVI / GIF)",
        annotations=ToolAnnotations(idempotent_hint=True, open_world_hint=False),
    )
    def render_video_file(
        path: Annotated[
            str,
            Field(description="Output video path inside workspace (e.g. 'renders/showcase.mp4')."),
        ],
        kind: Annotated[
            VideoKind,
            Field(
                description=(
                    "'timeline' (F-curves/physics), 'turntable' (360 orbit), "
                    "or 'camera_path' (Catmull-Rom)."
                )
            ),
        ] = "turntable",
        frame_start: Annotated[int | None, Field(description="Start frame.")] = None,
        frame_end: Annotated[int | None, Field(description="End frame.")] = None,
        fps: Annotated[int | None, Field(ge=1, le=60)] = None,
        width: Annotated[int, Field(ge=128, le=1920)] = 640,
        height: Annotated[int, Field(ge=128, le=1080)] = 480,
        mode: Annotated[RenderMode, Field(description="Visual shader mode.")] = "pbr",
        engine: Annotated[
            Literal["software", "raytrace"] | None,
            Field(description="Optional engine override ('software' or 'raytrace')."),
        ] = None,
        samples: Annotated[int, Field(ge=1, le=16, description="Raytracer samples per pixel.")] = 2,
        camera_id: Annotated[str | None, Field(description="Camera ID for timeline mode.")] = None,
        camera_waypoints: Annotated[
            list[dict[str, Any]] | None,
            Field(
                description=(
                    "Waypoints for camera_path: "
                    "[{'location':[x,y,z],'target':[x,y,z],'fov_deg':45}, ...]."
                )
            ),
        ] = None,
        overlay_hud: Annotated[bool, Field(description="Overlay frame/axes HUD badge.")] = True,
        bitrate: Annotated[str, Field(description="FFmpeg target video bitrate e.g. '4M'.")] = "4M",
        frames_dir: Annotated[
            str | None, Field(description="Optional workspace directory to save PNG frames.")
        ] = None,
    ) -> dict[str, Any]:
        """Render timeline, 360 turntable, or camera-path video to MP4/WebM/AVI/GIF."""
        sess = get_session()
        with sess.lock:
            out_p = sess.resolve_path(path)
            f_dir = sess.resolve_path(frames_dir) if frames_dir else None
            try:
                res = render_video(
                    sess.scene,
                    out_path=out_p,
                    kind=kind,
                    frame_start=frame_start,
                    frame_end=frame_end,
                    fps=fps,
                    width=width,
                    height=height,
                    mode=mode,
                    engine=engine,
                    samples=samples,
                    camera_id=camera_id,
                    camera_waypoints=camera_waypoints,
                    overlay_hud=overlay_hud,
                    bitrate=bitrate,
                    frames_dir=f_dir,
                )
            except Exception as exc:
                raise ToolError(str(exc)) from exc
            return {"status": "ok", **res}

    @mcp.tool(
        title="Manage Rigid-Body & Cloth Physics",
        annotations=ToolAnnotations(open_world_hint=False),
    )
    def physics_manage(
        action: Annotated[
            Literal["set_rigid_body", "set_cloth", "bake", "clear_cache", "list"],
            Field(description="Physics configuration, baking, or cache cleanup."),
        ],
        object_id: Annotated[
            str | None, Field(description="Target object ID or name for physics setup.")
        ] = None,
        # Rigid Body options
        body_type: Annotated[
            Literal["dynamic", "static", "kinematic"], Field(description="Rigid body mode.")
        ] = "dynamic",
        mass: Annotated[float, Field(gt=0.0)] = 1.0,
        restitution: Annotated[float, Field(ge=0.0, le=1.0)] = 0.45,
        friction: Annotated[float, Field(ge=0.0, le=1.5)] = 0.55,
        initial_velocity: Annotated[
            Vec3, Field(description="Initial linear velocity [vx, vy, vz].")
        ] = (
            0.0,
            0.0,
            0.0,
        ),
        collider_shape: Annotated[
            Literal["box", "sphere", "mesh_aabb"], Field(description="Collider geometry proxy.")
        ] = "box",
        # Cloth options
        stiffness: Annotated[float, Field(ge=0.05, le=1.0)] = 0.92,
        bend_stiffness: Annotated[float, Field(ge=0.0, le=1.0)] = 0.25,
        damping: Annotated[float, Field(ge=0.0, le=0.5)] = 0.03,
        wind: Annotated[Vec3, Field(description="Wind force vector [wx, wy, wz].")] = (
            0.0,
            0.0,
            0.0,
        ),
        pin_mode: Annotated[
            Literal["none", "top_corners", "top_edge", "border", "indices"],
            Field(description="Cloth vertex pinning preset."),
        ] = "top_corners",
        pinned_indices: Annotated[
            list[int] | None, Field(description="Vertex indices when pin_mode='indices'.")
        ] = None,
        collision_margin: Annotated[float, Field(ge=0.001, le=0.5)] = 0.03,
        substeps: Annotated[int, Field(ge=1, le=32)] = 8,
        # Bake options
        frame_start: Annotated[int | None, Field(description="Simulation start frame.")] = None,
        frame_end: Annotated[int | None, Field(description="Simulation end frame.")] = None,
        gravity: Annotated[Vec3, Field(description="World gravity vector [gx, gy, gz].")] = (
            0.0,
            0.0,
            -9.81,
        ),
        ground_z: Annotated[float, Field(description="Floor plane elevation Z.")] = 0.0,
    ) -> dict[str, Any]:
        """Configure Rigid-Body dynamics or XPBD Cloth + wind, bake physics, or clear cache."""
        sess = get_session()
        with sess.lock:
            if action == "list":
                rb_list = [
                    {"id": o.id, "name": o.name, **o.rigid_body.model_dump()}
                    for o in sess.scene.objects.values()
                    if o.rigid_body is not None
                ]
                cloth_list = [
                    {
                        "id": o.id,
                        "name": o.name,
                        "cached_frames": len(o.vertex_cache),
                        **o.cloth.model_dump(),
                    }
                    for o in sess.scene.objects.values()
                    if o.cloth is not None
                ]
                return {"rigid_bodies": rb_list, "cloth_objects": cloth_list}

            if action == "bake":
                sess.push_history("physics_bake")
                res = bake_scene_physics(
                    sess.scene,
                    frame_start=frame_start,
                    frame_end=frame_end,
                    gravity=gravity,
                    ground_z=ground_z,
                )
                sess.notify("physics_baked", res)
                return {"status": "ok", **res}

            if action == "clear_cache":
                sess.push_history("physics_clear_cache")
                res = clear_physics_caches(sess.scene, object_id=object_id)
                sess.notify("physics_cleared", res)
                return {"status": "ok", **res}

            if not object_id:
                raise ToolError(f"object_id is required for action={action!r}.")
            obj = sess.require_mesh_like(object_id)
            sess.push_history(f"physics_{action}:{obj.name}")

            if action == "set_rigid_body":
                obj.rigid_body = RigidBodySpec(
                    enabled=True,
                    body_type=body_type,
                    mass=mass,
                    restitution=restitution,
                    friction=friction,
                    initial_velocity=initial_velocity,
                    collider_shape=collider_shape,
                )
                sess.notify("rigid_body_set", {"object_id": obj.id})
                return {
                    "status": "ok",
                    "object_id": obj.id,
                    "rigid_body": obj.rigid_body.model_dump(),
                }

            # set_cloth
            obj.cloth = ClothSpec(
                enabled=True,
                stiffness=stiffness,
                bend_stiffness=bend_stiffness,
                damping=damping,
                wind=wind,
                pin_mode=pin_mode,
                pinned_indices=list(pinned_indices or []),
                collision_margin=collision_margin,
                substeps=substeps,
            )
            sess.notify("cloth_set", {"object_id": obj.id})
            return {"status": "ok", "object_id": obj.id, "cloth": obj.cloth.model_dump()}

    @mcp.tool(
        title="Skeletal Armature Rigging & Pose",
        annotations=ToolAnnotations(open_world_hint=False),
    )
    def armature_rig(
        action: Annotated[
            Literal[
                "create",
                "add_bone",
                "auto_rig",
                "set_pose",
                "solve_ik",
                "keyframe_pose",
                "list",
            ],
            Field(description="Skeletal rig creation, auto-skinning, FK/IK posing, or keyframing."),
        ],
        armature_id: Annotated[str | None, Field(description="Armature object ID or name.")] = None,
        name: Annotated[str | None, Field(description="Name for new armature object.")] = None,
        location: Annotated[Vec3 | None, Field(description="World/local XYZ position.")] = None,
        bones: Annotated[
            list[dict[str, Any]] | None,
            Field(
                description=(
                    "Bone dicts: [{'name':'root','parent':None,'head':[0,0,0],'tail':[0,0,1]}]."
                )
            ),
        ] = None,
        bone_name: Annotated[
            str | None, Field(description="Bone name for add/pose/solve_ik/keyframe.")
        ] = None,
        parent_bone: Annotated[str | None, Field(description="Optional parent bone name.")] = None,
        head: Annotated[Vec3 | None, Field(description="Bone rest head XYZ.")] = None,
        tail: Annotated[Vec3 | None, Field(description="Bone rest tail XYZ.")] = None,
        mesh_object_id: Annotated[
            str | None, Field(description="Target mesh ID or name to auto-skin.")
        ] = None,
        max_influences: Annotated[int, Field(ge=1, le=8)] = 4,
        rotation_deg: Annotated[
            Vec3 | None, Field(description="Bone pose Euler XYZ degrees.")
        ] = None,
        translation: Annotated[Vec3 | None, Field(description="Bone pose translation XYZ.")] = None,
        target_pos: Annotated[
            Vec3 | None, Field(description="World-space IK target XYZ for solve_ik.")
        ] = None,
        chain_length: Annotated[
            int, Field(ge=1, le=16, description="Bone chain depth for solve_ik.")
        ] = 2,
        pole_target: Annotated[
            Vec3 | None, Field(description="Optional world-space pole vector target for solve_ik.")
        ] = None,
        ik_iterations: Annotated[
            int, Field(ge=1, le=64, description="FABRIK solver iterations for solve_ik.")
        ] = 16,
        frame: Annotated[
            int | None, Field(description="Keyframe number for keyframe_pose or solve_ik.")
        ] = None,
    ) -> dict[str, Any]:
        """Create armatures, auto-skin meshes, pose FK/IK chains, and keyframe bone poses."""
        sess = get_session()
        with sess.lock:
            if action == "list":
                arms = [
                    {
                        "id": o.id,
                        "name": o.name,
                        "location": list(o.transform.location),
                        "bones": [b.model_dump() for b in o.armature.bones],
                        "ik_targets": [c.model_dump() for c in o.armature.ik_targets],
                    }
                    for o in sess.scene.objects.values()
                    if o.kind == "armature" and o.armature is not None
                ]
                return {"armatures": arms}

            if action == "create":
                aname = sess.scene.unique_name(name or "Armature")
                aid = sess.scene.next_id(aname.lower())
                bone_specs = [
                    BoneSpec(**b)
                    for b in (bones or [{"name": "root", "head": (0, 0, 0), "tail": (0, 0, 0.5)}])
                ]
                sess.push_history(f"armature_create:{aname}")
                arm_obj = SceneObject(
                    id=aid,
                    name=aname,
                    kind="armature",
                    transform=Transform(location=location or (0.0, 0.0, 0.0)),
                    armature=ArmatureSpec(bones=bone_specs),
                )
                sess.scene.objects[aid] = arm_obj
                sess.notify("armature_created", {"armature_id": aid})
                return {
                    "status": "ok",
                    "armature_id": aid,
                    "name": aname,
                    "bone_count": len(bone_specs),
                }

            if not armature_id:
                raise ToolError(f"armature_id is required for action={action!r}.")
            arm_obj = sess.require_object(armature_id, expected_kind="armature")
            assert arm_obj.armature is not None

            if action == "add_bone":
                if not bone_name:
                    raise ToolError("bone_name is required for action='add_bone'.")
                if any(b.name == bone_name for b in arm_obj.armature.bones):
                    raise ToolError(f"Bone {bone_name!r} already exists on {arm_obj.name!r}.")
                sess.push_history(f"add_bone:{bone_name}")
                arm_obj.armature.bones.append(
                    BoneSpec(
                        name=bone_name,
                        parent=parent_bone,
                        head=head or (0.0, 0.0, 0.0),
                        tail=tail or (0.0, 0.0, 0.5),
                    )
                )
                return {
                    "status": "ok",
                    "armature_id": arm_obj.id,
                    "bone_count": len(arm_obj.armature.bones),
                }

            if action == "auto_rig":
                if not mesh_object_id:
                    raise ToolError("mesh_object_id is required for action='auto_rig'.")
                mobj = sess.require_mesh_like(mesh_object_id)
                saved_skin = mobj.skin
                mobj.skin = None
                try:
                    ev = evaluate_object_mesh(sess.scene, mobj.id)
                finally:
                    mobj.skin = saved_skin
                arm_mat = compute_world_matrix(sess.scene, arm_obj.id)
                sess.push_history(f"auto_rig:{mobj.name}")
                skin = auto_rig_mesh(
                    ev.local_mesh,
                    ev.world_matrix,
                    arm_obj,
                    arm_mat,
                    max_influences=max_influences,
                )
                mobj.skin = skin
                sess.notify("mesh_rigged", {"object_id": mobj.id, "armature_id": arm_obj.id})
                return {
                    "status": "ok",
                    "object_id": mobj.id,
                    "armature_id": arm_obj.id,
                    "bones_weighted": {b: len(w) for b, w in skin.weights.items() if w},
                }

            # set_pose / solve_ik / keyframe_pose
            if not bone_name:
                raise ToolError(f"bone_name is required for action={action!r}.")
            target_bone = next((b for b in arm_obj.armature.bones if b.name == bone_name), None)
            if target_bone is None:
                raise ToolError(f"Bone {bone_name!r} not found on {arm_obj.name!r}.")

            if action == "solve_ik":
                if target_pos is None:
                    raise ToolError("target_pos is required for action='solve_ik'.")
                sess.push_history(f"solve_ik:{bone_name}")
                arm_inv = np.linalg.inv(compute_world_matrix(sess.scene, arm_obj.id))
                t_arm_arr = transform_points(arm_inv, np.asarray([target_pos], dtype=np.float64))[0]
                t_arm: Vec3 = (float(t_arm_arr[0]), float(t_arm_arr[1]), float(t_arm_arr[2]))
                p_arm: Vec3 | None = None
                if pole_target is not None:
                    p_arr = transform_points(arm_inv, np.asarray([pole_target], dtype=np.float64))[
                        0
                    ]
                    p_arm = (float(p_arr[0]), float(p_arr[1]), float(p_arr[2]))

                ik_res = solve_armature_ik(
                    arm_obj.armature,
                    tip_bone=bone_name,
                    target_pos=t_arm,
                    chain_length=chain_length,
                    pole_target=p_arm,
                    iterations=ik_iterations,
                )

                tracks_added: list[str] = []
                if frame is not None:
                    tl = sess.scene.timeline
                    bones_map = {b.name: b for b in arm_obj.armature.bones}
                    for cb_name in ik_res["chain_bones"]:
                        cb_bone = bones_map[cb_name]
                        for idx, ax in enumerate(("x", "y", "z")):
                            pp = f"bone.{cb_name}.rotation_deg.{ax}"
                            tr = next(
                                (
                                    t
                                    for t in tl.tracks
                                    if t.target_id == arm_obj.id and t.property_path == pp
                                ),
                                None,
                            )
                            if tr is None:
                                tr = FCurveTrack(target_id=arm_obj.id, property_path=pp)
                                tl.tracks.append(tr)
                            tr.keyframes = [k for k in tr.keyframes if k.frame != frame]
                            tr.keyframes.append(
                                Keyframe(
                                    frame=frame,
                                    value=float(cb_bone.rotation_deg[idx]),
                                    interpolation="bezier",
                                )
                            )
                            tr.keyframes.sort(key=lambda k: k.frame)
                            tracks_added.append(pp)

                sess.notify(
                    "armature_ik_solved",
                    {"armature_id": arm_obj.id, "bone": bone_name, "error": ik_res["error"]},
                )
                out_ik: dict[str, Any] = {
                    "status": "ok",
                    "armature_id": arm_obj.id,
                    "ik_result": ik_res,
                }
                if tracks_added:
                    out_ik["frame"] = frame
                    out_ik["tracks"] = tracks_added
                return out_ik

            if action == "set_pose":
                sess.push_history(f"pose:{bone_name}")
                if rotation_deg is not None:
                    target_bone.rotation_deg = rotation_deg
                if translation is not None:
                    target_bone.translation = translation
                compute_armature_matrices(arm_obj.armature)
                sess.notify("armature_posed", {"armature_id": arm_obj.id, "bone": bone_name})
                return {
                    "status": "ok",
                    "armature_id": arm_obj.id,
                    "bone": target_bone.model_dump(),
                }

            # keyframe_pose
            if frame is None:
                raise ToolError("frame is required for action='keyframe_pose'.")
            sess.push_history(f"keyframe_pose:{bone_name}@{frame}")
            tl = sess.scene.timeline
            tracks_added = []
            for prop, vec in (("rotation_deg", rotation_deg), ("translation", translation)):
                if vec is None:
                    continue
                for idx, ax in enumerate(("x", "y", "z")):
                    pp = f"bone.{bone_name}.{prop}.{ax}"
                    tr = next(
                        (
                            t
                            for t in tl.tracks
                            if t.target_id == arm_obj.id and t.property_path == pp
                        ),
                        None,
                    )
                    if tr is None:
                        tr = FCurveTrack(target_id=arm_obj.id, property_path=pp)
                        tl.tracks.append(tr)
                    tr.keyframes = [k for k in tr.keyframes if k.frame != frame]
                    tr.keyframes.append(
                        Keyframe(frame=frame, value=float(vec[idx]), interpolation="bezier")
                    )
                    tr.keyframes.sort(key=lambda k: k.frame)
                    tracks_added.append(pp)
            sess.notify(
                "bone_keyframed", {"armature_id": arm_obj.id, "bone": bone_name, "frame": frame}
            )
            return {
                "status": "ok",
                "armature_id": arm_obj.id,
                "bone": bone_name,
                "frame": frame,
                "tracks": tracks_added,
            }
