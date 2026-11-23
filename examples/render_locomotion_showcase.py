"""Render an HD 1280x720 30fps AAA Locomotion Showcase video for a rigged FBX humanoid character.

Imports `T-Pose.fbx` via FrFruit's Binary FBX importer, enables Dual Quaternion Skinning (DQS),
builds a 64m AAA Animation Test Gym with 1m grid floor and distance speed pylons, synthesizes
realistic heavy/fast gameplay locomotion (Combat Idle -> Heavy Walk -> Combat Jog -> Explosive
Sprint -> Heavy Hard Stop -> Recovery Idle) with 2-bone IK and zero-slip foot locking, renders
all frames in parallel, encodes H.264 MP4 via FFmpeg, and exports an animated Binary FBX 7.4.
"""

from __future__ import annotations

import argparse
import math
import subprocess
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from frfruit.animation.locomotion import (
    LocomotionFrameState,
    bake_locomotion_to_timeline,
    build_showcase_schedule,
    detect_humanoid_rig,
    evaluate_locomotion_pose,
)
from frfruit.core.scene import (
    CameraSpec,
    LightSpec,
    MaterialSpec,
    SceneDocument,
    SceneObject,
)
from frfruit.core.types import Transform
from frfruit.geometry.mesh import MeshData
from frfruit.geometry.primitives import make_cube
from frfruit.geometry.spatial import evaluate_scene_meshes
from frfruit.io.fbx_exporter import export_fbx_scene
from frfruit.io.importers import import_mesh_file
from frfruit.render.software import RenderConfig, SoftwareRenderer


def _make_subdivided_runway(
    width: float = 14.0,
    y_start: float = 6.0,
    y_end: float = -58.0,
    nx: int = 14,
    ny: int = 64,
) -> MeshData:
    """Create a finely subdivided Z=0 runway floor mesh with world-metric UVs (1 UV unit = 1m)."""
    xs = np.linspace(-width * 0.5, width * 0.5, nx + 1)
    ys = np.linspace(y_start, y_end, ny + 1)
    gx, gy = np.meshgrid(xs, ys)
    verts = np.column_stack([gx.ravel(), gy.ravel(), np.zeros(gx.size, dtype=np.float64)])
    uvs = np.column_stack([(gx.ravel() + width * 0.5), (y_start - gy.ravel())])

    faces: list[tuple[int, int, int]] = []
    stride = nx + 1
    for j in range(ny):
        for i in range(nx):
            v00 = j * stride + i
            v10 = v00 + 1
            v01 = v00 + stride
            v11 = v01 + 1
            faces.append((v00, v01, v11))
            faces.append((v00, v11, v10))
    return MeshData(vertices=verts, faces=np.asarray(faces, dtype=np.int64), uvs=uvs)


def _build_showcase_scene(fbx_path: Path) -> tuple[SceneDocument, SceneObject]:
    """Assemble the character + AAA Animation Test Gym environment."""
    scene = SceneDocument()
    scene.world.background_color = (0.055, 0.068, 0.095)
    scene.world.horizon_color = (0.12, 0.155, 0.21)
    scene.world.zenith_color = (0.03, 0.04, 0.068)
    scene.world.ambient_strength = 0.34

    # Crisp cinematic post-processing (skip Monte-Carlo denoiser for noise-free rasterizer)
    scene.post_process.enabled = True
    scene.post_process.denoise_enabled = False
    scene.post_process.tone_mapper = "aces"
    scene.post_process.exposure = 0.08
    scene.post_process.ssao_enabled = False
    scene.post_process.bloom_enabled = True
    scene.post_process.bloom_threshold = 0.88
    scene.post_process.bloom_intensity = 0.28
    scene.post_process.vignette_intensity = 0.16
    scene.post_process.fxaa_enabled = True
    scene.post_process.sharpen = 0.10

    # 3-point AAA studio lighting
    scene.objects["key_sun"] = SceneObject(
        id="key_sun",
        name="KeySun",
        kind="light",
        transform=Transform(location=(5.0, -6.0, 8.0)),
        light=LightSpec(
            kind="sun",
            color=(1.0, 0.96, 0.90),
            intensity=2.65,
            direction=(-0.52, 0.58, -0.78),
            cast_shadows=True,
        ),
    )
    scene.objects["rim_sun"] = SceneObject(
        id="rim_sun",
        name="RimLight",
        kind="light",
        transform=Transform(location=(-5.0, 5.0, 4.5)),
        light=LightSpec(
            kind="sun",
            color=(0.45, 0.78, 1.0),
            intensity=1.35,
            direction=(0.62, -0.55, -0.42),
            cast_shadows=False,
        ),
    )
    scene.objects["fill_sun"] = SceneObject(
        id="fill_sun",
        name="FillLight",
        kind="light",
        transform=Transform(location=(-4.0, -4.0, 2.5)),
        light=LightSpec(
            kind="sun",
            color=(0.78, 0.85, 0.98),
            intensity=0.68,
            direction=(0.55, 0.50, -0.35),
            cast_shadows=False,
        ),
    )

    # Materials for the Test Gym
    scene.materials["mat_runway"] = MaterialSpec(
        id="mat_runway",
        name="RunwayFloor",
        base_color=(0.115, 0.128, 0.155, 1.0),
        secondary_color=(0.155, 0.172, 0.205, 1.0),
        pattern="checker",
        pattern_scale=1.0,  # Exact 1m x 1m metric tiles!
        roughness=0.48,
        metallic=0.10,
    )
    scene.materials["mat_rail"] = MaterialSpec(
        id="mat_rail",
        name="LaneRail",
        base_color=(0.08, 0.09, 0.11, 1.0),
        roughness=0.35,
        metallic=0.65,
    )
    scene.materials["mat_pylon_body"] = MaterialSpec(
        id="mat_pylon_body",
        name="PylonBody",
        base_color=(0.09, 0.105, 0.13, 1.0),
        roughness=0.38,
        metallic=0.72,
    )
    scene.materials["mat_glow_cyan"] = MaterialSpec(
        id="mat_glow_cyan",
        name="GlowCyan",
        base_color=(0.10, 0.85, 1.0, 1.0),
        emission_color=(0.15, 0.88, 1.0, 1.0),
        emission_strength=2.2,
        roughness=0.2,
    )
    scene.materials["mat_glow_amber"] = MaterialSpec(
        id="mat_glow_amber",
        name="GlowAmber",
        base_color=(1.0, 0.58, 0.12, 1.0),
        emission_color=(1.0, 0.55, 0.10, 1.0),
        emission_strength=2.4,
        roughness=0.2,
    )
    scene.materials["mat_glow_crimson"] = MaterialSpec(
        id="mat_glow_crimson",
        name="GlowCrimson",
        base_color=(1.0, 0.22, 0.24, 1.0),
        emission_color=(1.0, 0.20, 0.22, 1.0),
        emission_strength=2.5,
        roughness=0.2,
    )

    # 1. Subdivided metric floor
    runway = _make_subdivided_runway(width=16.0, y_start=6.0, y_end=-58.0, nx=16, ny=64)
    scene.objects["runway_floor"] = SceneObject(
        id="runway_floor",
        name="RunwayFloor",
        kind="mesh",
        material_id="mat_runway",
        mesh=runway.to_payload(),
    )

    # 2. Low-profile lane border strips along X = +/- 1.55m
    rail_mesh = make_cube(size=(0.08, 62.0, 0.012)).to_payload()
    for side_tag, sx in (("L", 1.55), ("R", -1.55)):
        scene.objects[f"lane_rail_{side_tag}"] = SceneObject(
            id=f"lane_rail_{side_tag}",
            name=f"LaneRail_{side_tag}",
            kind="mesh",
            material_id="mat_rail",
            transform=Transform(location=(sx, -26.0, 0.006)),
            mesh=rail_mesh,
        )

    # 3. Far-side speed pylons every 4m (X=-2.35) + near-side low-profile glow markers
    pylon_mesh = make_cube(size=(0.15, 0.15, 1.45)).to_payload()
    strip_mesh = make_cube(size=(0.17, 0.045, 0.38)).to_payload()
    near_marker_mesh = make_cube(size=(0.12, 0.55, 0.022)).to_payload()
    bar_mesh = make_cube(size=(3.4, 0.045, 0.006)).to_payload()
    for idx, py in enumerate(range(0, -56, -4)):
        # Zone color coding along track: Idle/Walk=Cyan, Jog=Amber, Sprint=Crimson
        if py > -16:
            glow_mat = "mat_glow_cyan"
        elif py > -34:
            glow_mat = "mat_glow_amber"
        else:
            glow_mat = "mat_glow_crimson"

        # Crisp transverse 4m metric floor bar
        scene.objects[f"meter_bar_{idx}"] = SceneObject(
            id=f"meter_bar_{idx}",
            name=f"MeterBar_{idx}",
            kind="mesh",
            material_id=glow_mat if (idx % 2 == 0) else "mat_rail",
            transform=Transform(location=(0.0, float(py), 0.003)),
            mesh=bar_mesh,
        )

        # Far-side tall architectural speed pylon (always behind character from +X camera!)
        scene.objects[f"pylon_{idx}_R"] = SceneObject(
            id=f"pylon_{idx}_R",
            name=f"Pylon_{idx}_R",
            kind="mesh",
            material_id="mat_pylon_body",
            transform=Transform(location=(-2.35, float(py), 0.725)),
            mesh=pylon_mesh,
        )
        scene.objects[f"pylon_glow_{idx}_R"] = SceneObject(
            id=f"pylon_glow_{idx}_R",
            name=f"PylonGlow_{idx}_R",
            kind="mesh",
            material_id=glow_mat,
            transform=Transform(location=(-2.35, float(py), 1.12)),
            mesh=strip_mesh,
        )
        # Near-side low-profile runway edge light strip (never occludes character!)
        scene.objects[f"near_glow_{idx}_L"] = SceneObject(
            id=f"near_glow_{idx}_L",
            name=f"NearGlow_{idx}_L",
            kind="mesh",
            material_id=glow_mat,
            transform=Transform(location=(1.82, float(py), 0.011)),
            mesh=near_marker_mesh,
        )

    # 4. Import the user's rigged FBX character with Dual Quaternion Skinning
    objs = import_mesh_file(scene, fbx_path)
    arm_obj = next(o for o in objs if o.kind == "armature")
    assert arm_obj.armature is not None
    arm_obj.armature.skinning_mode = "dqs"

    # Slightly enhance character material finish for crisp AAA metallic/satin highlights
    for mat in scene.materials.values():
        if "Body" in mat.name:
            mat.roughness = 0.34
            mat.metallic = 0.22
            mat.clearcoat = 0.25
        elif "Joints" in mat.name:
            mat.roughness = 0.30
            mat.metallic = 0.68

    return scene, arm_obj


def _smoothstep(a: float, b: float, x: float) -> float:
    if b <= a:
        return 1.0 if x >= b else 0.0
    t = max(0.0, min(1.0, (x - a) / (b - a)))
    return t * t * (3.0 - 2.0 * t)


def _lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * t


def _camera_for_state(st: LocomotionFrameState) -> SceneObject:
    """Compute smooth cinematic tracking camera keyed to each locomotion showcase phase."""
    t = st.time_s
    ry = st.root_y

    # Keyframed azimuth (deg: 0=direct front -Y, 90=right side +X), distance, height, fov
    if t < 2.2:
        u = t / 2.2
        az_deg = _lerp(24.0, 44.0, u)
        dist = _lerp(2.70, 2.85, u)
        cam_z = _lerp(1.12, 1.08, u)
        fov = 42.0
        lead_y = 0.0
    elif t < 5.7:
        u = _smoothstep(2.2, 3.6, t)
        az_deg = _lerp(44.0, 62.0, u)
        dist = _lerp(2.85, 3.05, u)
        cam_z = _lerp(1.08, 1.02, u)
        fov = _lerp(42.0, 44.0, u)
        lead_y = -0.15 * u
    elif t < 9.1:
        u = _smoothstep(5.7, 7.2, t)
        az_deg = _lerp(62.0, 82.0, u)
        dist = _lerp(3.05, 3.20, u)
        cam_z = _lerp(1.02, 0.98, u)
        fov = _lerp(44.0, 46.0, u)
        lead_y = -0.25
    elif t < 13.0:
        u = _smoothstep(9.1, 10.6, t)
        sweep = _smoothstep(10.6, 13.0, t)
        az_deg = _lerp(82.0, 88.0, u) - 22.0 * sweep  # Dramatic sweep from side to action 3/4!
        dist = _lerp(3.20, 3.12, u)
        cam_z = _lerp(0.98, 0.86, u)  # Low heroic action angle during sprint!
        fov = _lerp(46.0, 50.0, u)
        lead_y = -0.35
    elif t < 15.2:
        u = _smoothstep(13.0, 14.2, t)
        az_deg = _lerp(66.0, 26.0, u)  # Whip around to front-3/4 to catch the heavy skid stop!
        dist = _lerp(3.12, 2.68, u)
        cam_z = _lerp(0.86, 0.98, u)
        fov = _lerp(50.0, 43.0, u)
        lead_y = _lerp(-0.35, 0.0, u)
    else:
        u = _smoothstep(15.2, 18.0, t)
        az_deg = _lerp(26.0, 14.0, u)
        dist = _lerp(2.68, 2.42, u)
        cam_z = _lerp(0.98, 1.10, u)
        fov = _lerp(43.0, 41.0, u)
        lead_y = 0.0

    az = math.radians(az_deg)
    # Character faces -Y: az=0 puts camera at (0, ry - dist), az=90 puts camera at (+dist, ry)
    cx = math.sin(az) * dist
    cy = ry + lead_y - math.cos(az) * dist
    target_z = 0.92 - 0.06 * st.brake_weight

    return SceneObject(
        id="showcase_cam",
        name="ShowcaseCam",
        kind="camera",
        transform=Transform(location=(cx, cy, cam_z)),
        camera=CameraSpec(target=(0.0, ry + lead_y * 0.4, target_z), fov_deg=fov),
    )


def _load_font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = (
        [
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
            "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
        ]
        if bold
        else [
            "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        ]
    )
    for path in candidates:
        if Path(path).exists():
            try:
                return ImageFont.truetype(path, size=size)
            except Exception:
                pass
    return ImageFont.load_default()


def _draw_hud_overlay(
    img: Image.Image,
    st: LocomotionFrameState,
    frame_idx: int,
    total_frames: int,
) -> Image.Image:
    """Composite a sleek AAA game-engine telemetry HUD onto the rendered frame."""
    base = img.convert("RGBA")
    w, h = base.size
    overlay = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    # Accent color per locomotion state
    lbl = st.label
    if "SPRINT" in lbl:
        accent = (255, 92, 46, 255)
    elif "STOP" in lbl:
        accent = (255, 52, 78, 255)
    elif "JOG" in lbl:
        accent = (255, 178, 42, 255)
    elif "WALK" in lbl:
        accent = (42, 214, 198, 255)
    else:
        accent = (72, 186, 255, 255)

    # Top-left HUD telemetry card
    pad = 20
    card_w, card_h = 370, 108
    draw.rounded_rectangle(
        [pad, pad, pad + card_w, pad + card_h],
        radius=8,
        fill=(10, 13, 19, 195),
        outline=(55, 68, 88, 180),
        width=1,
    )
    draw.rectangle([pad, pad + 8, pad + 5, pad + card_h - 8], fill=accent)

    f_title = _load_font(20, bold=True)
    f_sub = _load_font(13, bold=False)
    f_small = _load_font(12, bold=False)

    draw.text((pad + 18, pad + 12), lbl, font=f_title, fill=(242, 247, 255, 255))
    draw.text(
        (pad + 18, pad + 40),
        f"SPEED {st.speed:4.2f} m/s   |   CADENCE {st.cadence_hz * 2.0:3.1f} step/s",
        font=f_sub,
        fill=(178, 196, 220, 255),
    )

    # Speedometer bar
    bx0, by0, bx1, by1 = pad + 18, pad + 64, pad + card_w - 18, pad + 74
    draw.rounded_rectangle([bx0, by0, bx1, by1], radius=3, fill=(28, 34, 46, 230))
    fill_frac = max(0.0, min(1.0, st.speed / 7.2))
    if fill_frac > 0.005:
        draw.rounded_rectangle(
            [bx0, by0, bx0 + int((bx1 - bx0) * fill_frac), by1],
            radius=3,
            fill=accent,
        )

    draw.text(
        (pad + 18, pad + 82),
        f"RIG: DQS + 2-BONE IK FOOT LOCK   |   F{frame_idx + 1:03d}/{total_frames:03d}",
        font=f_small,
        fill=(128, 146, 172, 240),
    )

    # Top-right subtle branding badge
    badge_w = 216
    draw.rounded_rectangle(
        [w - pad - badge_w, pad, w - pad, pad + 34],
        radius=6,
        fill=(10, 13, 19, 175),
        outline=(55, 68, 88, 150),
        width=1,
    )
    draw.text(
        (w - pad - badge_w + 14, pad + 9),
        "FRFRUIT LOCOMOTION RIG",
        font=f_small,
        fill=(196, 215, 238, 240),
    )

    # Bottom phase timeline scrubber
    bar_x0, bar_x1 = pad, w - pad
    bar_y0, bar_y1 = h - 26, h - 18
    draw.rounded_rectangle([bar_x0, bar_y0, bar_x1, bar_y1], radius=3, fill=(14, 18, 26, 195))
    prog = frame_idx / max(1, total_frames - 1)
    px = bar_x0 + int((bar_x1 - bar_x0) * prog)
    draw.rounded_rectangle([bar_x0, bar_y0, max(bar_x0 + 4, px), bar_y1], radius=3, fill=accent)

    # Phase markers along timeline (Idle, Walk, Jog, Sprint, Stop, Idle)
    dur = 18.0
    markers = [
        (0.0, "IDLE"),
        (2.2, "WALK"),
        (5.7, "JOG"),
        (9.1, "SPRINT"),
        (13.0, "HARD STOP"),
        (15.2, "RECOVERY"),
    ]
    for mt, mtxt in markers:
        mx = bar_x0 + int((bar_x1 - bar_x0) * (mt / dur))
        draw.line([(mx, bar_y0 - 4), (mx, bar_y1 + 2)], fill=(190, 205, 225, 160), width=1)
        draw.text((mx + 4, bar_y0 - 18), mtxt, font=f_small, fill=(195, 210, 230, 210))

    return Image.alpha_composite(base, overlay).convert("RGB")


# Worker globals for ProcessPoolExecutor (avoid re-importing FBX per frame!)
_W_SCENE: SceneDocument | None = None
_W_ARM: SceneObject | None = None
_W_RIG: Any = None
_W_RENDERER: SoftwareRenderer | None = None
_W_CFG: RenderConfig | None = None
_W_TOTAL: int = 1


def _init_worker(fbx_path_str: str, width: int, height: int, total_frames: int) -> None:
    global _W_SCENE, _W_ARM, _W_RIG, _W_RENDERER, _W_CFG, _W_TOTAL
    _W_SCENE, _W_ARM = _build_showcase_scene(Path(fbx_path_str))
    assert _W_ARM.armature is not None
    _W_RIG = detect_humanoid_rig(_W_ARM.armature)
    _W_RENDERER = SoftwareRenderer()
    _W_CFG = RenderConfig(
        width=width,
        height=height,
        mode="pbr",
        show_grid=False,  # Using physical metric runway floor + glow bars!
        show_axes=False,
        enable_shadows=True,
    )
    _W_TOTAL = total_frames


def _render_frame_worker(args: tuple[int, LocomotionFrameState, str]) -> int:
    fi, st, out_file = args
    assert _W_SCENE is not None and _W_ARM is not None and _W_ARM.armature is not None
    assert _W_RENDERER is not None and _W_CFG is not None

    evaluate_locomotion_pose(_W_ARM.armature, _W_RIG, st, apply_root_motion_to_hips=True)
    evaluated = evaluate_scene_meshes(_W_SCENE)
    cam = _camera_for_state(st)
    img = _W_RENDERER.render_image(_W_SCENE, evaluated, cam, _W_CFG)
    final = _draw_hud_overlay(img, st, fi, _W_TOTAL)
    final.save(out_file, format="PNG", compress_level=1)
    return fi


def main() -> None:
    parser = argparse.ArgumentParser(description="Render AAA Locomotion Showcase video.")
    parser.add_argument("--fbx", default="workspace/T-Pose.fbx")
    parser.add_argument("--out-video", default="workspace/locomotion_showcase.mp4")
    parser.add_argument("--out-fbx", default="workspace/T-Pose_Locomotion_Baked.fbx")
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--duration", type=float, default=18.0)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--resume", action="store_true", help="Skip already-rendered PNG frames.")
    parser.add_argument(
        "--preview-frame", type=int, default=-1, help="If >=0, render 1 frame only."
    )
    args = parser.parse_args()

    fbx_path = Path(args.fbx).resolve()
    out_video = Path(args.out_video).resolve()
    out_video.parent.mkdir(parents=True, exist_ok=True)
    sched = build_showcase_schedule(fps=args.fps, duration_s=args.duration)
    n_frames = len(sched)

    if args.preview_frame >= 0:
        _init_worker(str(fbx_path), args.width, args.height, n_frames)
        t0 = time.perf_counter()
        prev_path = out_video.parent / f"preview_frame_{args.preview_frame:04d}.png"
        _render_frame_worker((args.preview_frame, sched[args.preview_frame], str(prev_path)))
        elapsed_p = time.perf_counter() - t0
        print(f"Preview frame {args.preview_frame} rendered in {elapsed_p:.2f}s -> {prev_path}")
        return

    frames_dir = out_video.parent / "_locomotion_frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    if not args.resume:
        for old in frames_dir.glob("frame_*.png"):
            old.unlink()

    t_start = time.perf_counter()
    tasks = []
    for fi in range(n_frames):
        fp = frames_dir / f"frame_{fi:05d}.png"
        if args.resume and fp.exists() and fp.stat().st_size > 10_000:
            continue
        tasks.append((fi, sched[fi], str(fp)))

    print(
        f"Rendering {len(tasks)}/{n_frames} frames ({args.width}x{args.height} @ {args.fps}fps) "
        f"with {args.workers} workers..."
    )

    done = n_frames - len(tasks)
    if tasks:
        with ProcessPoolExecutor(
            max_workers=max(1, args.workers),
            initializer=_init_worker,
            initargs=(str(fbx_path), args.width, args.height, n_frames),
        ) as pool:
            for _fi in pool.map(_render_frame_worker, tasks, chunksize=4):
                done += 1
                if done % 30 == 0 or done == n_frames:
                    elapsed = time.perf_counter() - t_start
                    rendered_now = done - (n_frames - len(tasks))
                    fps_eff = rendered_now / max(1e-3, elapsed)
                    eta = (n_frames - done) / max(1e-3, fps_eff)
                    print(
                        f"  [{done:3d}/{n_frames}] {elapsed:5.1f}s | "
                        f"{fps_eff:.2f} fps | ETA {eta:.0f}s",
                        flush=True,
                    )

    # Encode crisp H.264 MP4 via FFmpeg
    print(f"Encoding MP4 -> {out_video} ...")
    cmd = [
        "ffmpeg",
        "-y",
        "-framerate",
        str(args.fps),
        "-i",
        str(frames_dir / "frame_%05d.png"),
        "-c:v",
        "libx264",
        "-preset",
        "medium",
        "-crf",
        "17",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(out_video),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if proc.returncode != 0:
        raise RuntimeError(f"FFmpeg failed:\n{proc.stderr}")

    # Export game-ready baked Binary FBX 7.4 (character + armature + baked locomotion curves)
    if args.out_fbx:
        print(f"Baking locomotion timeline & exporting game-ready FBX -> {args.out_fbx} ...")
        fbx_scene = SceneDocument()
        imported = import_mesh_file(fbx_scene, fbx_path)
        arm = next(o for o in imported if o.kind == "armature")
        bake_locomotion_to_timeline(
            fbx_scene, arm, sched, fps=args.fps, apply_root_motion_to_hips=True
        )
        export_fbx_scene(
            fbx_scene, Path(args.out_fbx).resolve(), ascii_mode=False, target_engine="unreal"
        )

    total_s = time.perf_counter() - t_start
    size_mb = out_video.stat().st_size / (1024 * 1024)
    print(f"SUCCESS! Rendered {out_video} ({size_mb:.2f} MB) in {total_s:.1f}s.")


if __name__ == "__main__":
    main()
