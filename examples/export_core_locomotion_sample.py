"""Export the 4 Core Locomotion sample clips (Idle, Walk, Run, Sprint) for Unity + 2x2 video.

Produces:
  - workspace/animations/Idle.fbx    (Unity Y-up Binary FBX 7.4, seamless loop)
  - workspace/animations/Walk.fbx    (Unity Y-up Binary FBX 7.4, seamless 1-cycle loop)
  - workspace/animations/Run.fbx     (Unity Y-up Binary FBX 7.4, seamless 1-cycle loop)
  - workspace/animations/Sprint.fbx  (Unity Y-up Binary FBX 7.4, seamless 1-cycle loop)
  - workspace/sample_4_locomotion.mp4 (2x2 side-by-side comparison video @ 30fps)
  - workspace/sample_4_filmstrip.png  (4-panel preview image)
"""

from __future__ import annotations

import argparse
import subprocess
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont

from frfruit.animation.locomotion import (
    GAIT_PRESETS,
    GaitParams,
    LocomotionFrameState,
    bake_locomotion_to_timeline,
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
from frfruit.geometry.primitives import make_plane
from frfruit.geometry.spatial import evaluate_scene_meshes
from frfruit.io.fbx_exporter import export_fbx_scene, verify_fbx_binary
from frfruit.io.importers import import_mesh_file
from frfruit.render.software import RenderConfig, SoftwareRenderer

# Tuned Core Locomotion presets matching the Unity Production To-Do list
CORE_SAMPLE_GAITS: dict[str, GaitParams] = {
    "Idle": GaitParams(
        speed=0.0,
        cadence=1.0 / 3.0,  # Exact 3.0s seamless breathing cycle
        duty_factor=1.0,
        hip_drop=-0.024,
        bounce_amp=0.005,
        sway_amp=0.008,
        hip_roll_deg=0.8,
        hip_yaw_deg=3.5,
        spine_lean_deg=2.5,
        spine_twist_deg=-3.5,
        foot_lift=0.0,
        stance_width=0.125,
        heel_strike_deg=0.0,
        toe_off_deg=0.0,
        arm_swing_fwd=0.055,
        arm_swing_back=0.02,
        arm_height=-0.16,
        arm_out=0.085,
        elbow_out=0.35,
        fist_curl=0.20,
    ),
    "Walk": GaitParams(
        speed=1.50,
        cadence=0.85,  # ~1.18s per full L+R cycle
        duty_factor=0.62,
        hip_drop=-0.025,
        bounce_amp=0.028,
        sway_amp=0.034,
        hip_roll_deg=4.8,
        hip_yaw_deg=8.5,
        spine_lean_deg=5.0,
        spine_twist_deg=10.5,
        foot_lift=0.11,
        stance_width=0.108,
        heel_strike_deg=-18.0,
        toe_off_deg=38.0,
        arm_swing_fwd=0.25,
        arm_swing_back=-0.22,
        arm_height=-0.13,
        arm_out=0.068,
        elbow_out=0.30,
        fist_curl=0.35,
    ),
    "Run": GAIT_PRESETS["combat_jog"],
    "Sprint": GAIT_PRESETS["explosive_sprint"],
}


def _build_loop_schedule(
    name: str,
    gait: GaitParams,
    fps: int = 30,
    cycles: int = 1,
    duration_s: float | None = None,
) -> list[LocomotionFrameState]:
    """Build an in-place seamless locomotion schedule."""
    is_idle = name == "Idle"
    if duration_s is None:
        duration_s = (3.0 * cycles) if is_idle else (cycles / max(0.1, gait.cadence))
    n_frames = max(4, int(round(duration_s * fps)))
    states: list[LocomotionFrameState] = []
    for fi in range(n_frames):
        frac = fi / float(n_frames)
        t = fi / float(fps)
        phase = (frac * cycles) if not is_idle else 0.0
        states.append(
            LocomotionFrameState(
                time_s=t,
                label=name,
                root_y=0.0,
                speed=gait.speed,
                accel=0.0,
                phase=phase,
                gait=gait,
                idle_weight=1.0 if is_idle else 0.0,
                brake_weight=0.0,
                brake_progress=0.0,
                cadence_hz=0.0 if is_idle else gait.cadence,
            )
        )
    return states


def _build_studio_scene(fbx_path: Path) -> tuple[SceneDocument, SceneObject]:
    scene = SceneDocument()
    scene.world.background_color = (0.06, 0.075, 0.105)
    scene.world.horizon_color = (0.13, 0.16, 0.21)
    scene.world.zenith_color = (0.035, 0.045, 0.075)
    scene.world.ambient_strength = 0.36

    scene.post_process.enabled = True
    scene.post_process.denoise_enabled = False
    scene.post_process.tone_mapper = "aces"
    scene.post_process.exposure = 0.08
    scene.post_process.ssao_enabled = False
    scene.post_process.bloom_enabled = True
    scene.post_process.bloom_threshold = 0.90
    scene.post_process.bloom_intensity = 0.20
    scene.post_process.vignette_intensity = 0.14
    scene.post_process.fxaa_enabled = True

    scene.objects["key_sun"] = SceneObject(
        id="key_sun",
        name="KeySun",
        kind="light",
        transform=Transform(location=(4.0, -5.0, 6.0)),
        light=LightSpec(
            kind="sun",
            color=(1.0, 0.96, 0.90),
            intensity=2.6,
            direction=(-0.55, 0.62, -0.78),
            cast_shadows=True,
        ),
    )
    scene.objects["rim_sun"] = SceneObject(
        id="rim_sun",
        name="RimLight",
        kind="light",
        transform=Transform(location=(-4.0, 4.0, 4.0)),
        light=LightSpec(
            kind="sun",
            color=(0.48, 0.80, 1.0),
            intensity=1.25,
            direction=(0.60, -0.60, -0.45),
            cast_shadows=False,
        ),
    )

    scene.materials["mat_floor"] = MaterialSpec(
        id="mat_floor",
        name="Floor",
        base_color=(0.12, 0.135, 0.165, 1.0),
        secondary_color=(0.165, 0.185, 0.22, 1.0),
        pattern="checker",
        pattern_scale=2.0,
        roughness=0.52,
        metallic=0.10,
    )
    scene.objects["floor"] = SceneObject(
        id="floor",
        name="Floor",
        kind="mesh",
        material_id="mat_floor",
        mesh=make_plane(size=10.0).to_payload(),
    )

    objs = import_mesh_file(scene, fbx_path)
    arm_obj = next(o for o in objs if o.kind == "armature")
    assert arm_obj.armature is not None
    arm_obj.armature.skinning_mode = "dqs"
    for mat in scene.materials.values():
        if "Body" in mat.name:
            mat.roughness = 0.34
            mat.metallic = 0.22
        elif "Joints" in mat.name:
            mat.roughness = 0.30
            mat.metallic = 0.68
    return scene, arm_obj


def _load_font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    path = (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
        if bold
        else "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf"
    )
    if Path(path).exists():
        return ImageFont.truetype(path, size=size)
    return ImageFont.load_default()


# Worker globals
_W_SCENE: SceneDocument | None = None
_W_ARM: SceneObject | None = None
_W_RIG: Any = None
_W_REND: SoftwareRenderer | None = None
_W_CFG: RenderConfig | None = None
_W_CAMS: dict[str, SceneObject] = {}


def _init_worker(fbx_str: str, tile_w: int, tile_h: int) -> None:
    global _W_SCENE, _W_ARM, _W_RIG, _W_REND, _W_CFG, _W_CAMS
    _W_SCENE, _W_ARM = _build_studio_scene(Path(fbx_str))
    assert _W_ARM.armature is not None
    _W_RIG = detect_humanoid_rig(_W_ARM.armature)
    _W_REND = SoftwareRenderer()
    _W_CFG = RenderConfig(
        width=tile_w,
        height=tile_h,
        mode="pbr",
        show_grid=True,
        show_axes=False,
        enable_shadows=True,
    )
    _W_CAMS = {
        "Idle": SceneObject(
            id="c_idle",
            name="CamIdle",
            kind="camera",
            transform=Transform(location=(1.35, -2.55, 1.08)),
            camera=CameraSpec(target=(0.0, 0.0, 0.90), fov_deg=41.0),
        ),
        "Walk": SceneObject(
            id="c_walk",
            name="CamWalk",
            kind="camera",
            transform=Transform(location=(2.20, -1.75, 1.02)),
            camera=CameraSpec(target=(0.0, 0.0, 0.90), fov_deg=42.0),
        ),
        "Run": SceneObject(
            id="c_run",
            name="CamRun",
            kind="camera",
            transform=Transform(location=(2.55, -1.25, 0.98)),
            camera=CameraSpec(target=(0.0, 0.0, 0.88), fov_deg=43.0),
        ),
        "Sprint": SceneObject(
            id="c_sprint",
            name="CamSprint",
            kind="camera",
            transform=Transform(location=(2.75, -0.55, 0.94)),
            camera=CameraSpec(target=(0.0, 0.0, 0.86), fov_deg=45.0),
        ),
    }


def _render_grid_frame(args: tuple[int, dict[str, LocomotionFrameState], str]) -> int:
    fi, states_by_name, out_path = args
    assert _W_SCENE is not None and _W_ARM is not None and _W_ARM.armature is not None
    assert _W_REND is not None and _W_CFG is not None

    tw, th = _W_CFG.width, _W_CFG.height
    grid = Image.new("RGB", (tw * 2, th * 2))
    f_title = _load_font(16, bold=True)
    f_sub = _load_font(12, bold=False)

    order = [
        ("Idle", (72, 186, 255)),
        ("Walk", (42, 214, 198)),
        ("Run", (255, 178, 42)),
        ("Sprint", (255, 92, 46)),
    ]
    for idx, (name, accent) in enumerate(order):
        st = states_by_name[name]
        evaluate_locomotion_pose(_W_ARM.armature, _W_RIG, st, apply_root_motion_to_hips=False)
        ev = evaluate_scene_meshes(_W_SCENE)
        tile = _W_REND.render_image(_W_SCENE, ev, _W_CAMS[name], _W_CFG).convert("RGB")
        draw = ImageDraw.Draw(tile)
        draw.rectangle([0, 0, tw, 30], fill=(10, 13, 19))
        draw.rectangle([0, 0, 5, 30], fill=accent)
        draw.text((12, 6), f"{name}", font=f_title, fill=(242, 247, 255))
        draw.text(
            (110, 9),
            f"v={st.speed:.2f} m/s | {st.cadence_hz * 2.0:.1f} step/s | F{fi + 1:02d}",
            font=f_sub,
            fill=(180, 198, 222),
        )
        grid.paste(tile, ((idx % 2) * tw, (idx // 2) * th))

    grid.save(out_path, format="PNG", compress_level=1)
    return fi


def main() -> None:
    parser = argparse.ArgumentParser(description="Export 4 Core Locomotion FBXs + 2x2 video.")
    parser.add_argument("--fbx", default="workspace/T-Pose.fbx")
    parser.add_argument("--anim-dir", default="workspace/animations")
    parser.add_argument("--out-video", default="workspace/sample_4_locomotion.mp4")
    parser.add_argument("--out-filmstrip", default="workspace/sample_4_filmstrip.png")
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--duration", type=float, default=3.0)
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()

    t0 = time.perf_counter()
    fbx_path = Path(args.fbx).resolve()
    anim_dir = Path(args.anim_dir).resolve()
    anim_dir.mkdir(parents=True, exist_ok=True)

    # 1. Export 4 separate Unity-ready Binary FBX 7.4 clips (exact 1-cycle seamless loops)
    print("Exporting 4 separate Unity Binary FBX 7.4 animations...")
    for name, gait in CORE_SAMPLE_GAITS.items():
        scene_fbx = SceneDocument()
        objs = import_mesh_file(scene_fbx, fbx_path)
        arm = next(o for o in objs if o.kind == "armature")
        sched = _build_loop_schedule(name, gait, fps=args.fps, cycles=1)
        bake_locomotion_to_timeline(
            scene_fbx, arm, sched, fps=args.fps, apply_root_motion_to_hips=False
        )
        out_fbx = anim_dir / f"{name}.fbx"
        export_fbx_scene(scene_fbx, out_fbx, ascii_mode=False, target_engine="unity")
        info = verify_fbx_binary(out_fbx)
        print(
            f"  {name:8s} -> {out_fbx.name} ({len(sched)} frames, "
            f"{out_fbx.stat().st_size / 1024:.0f} KB, valid={info['valid']})"
        )

    # 2. Build 3.0s continuous schedules for synchronized 2x2 video comparison
    n_frames = max(2, int(round(args.duration * args.fps)))
    vid_scheds: dict[str, list[LocomotionFrameState]] = {}
    for name, gait in CORE_SAMPLE_GAITS.items():
        is_idle = name == "Idle"
        lst: list[LocomotionFrameState] = []
        for fi in range(n_frames):
            t = fi / float(args.fps)
            phase = (t * gait.cadence) % 1.0 if not is_idle else 0.0
            lst.append(
                LocomotionFrameState(
                    time_s=t,
                    label=name,
                    root_y=0.0,
                    speed=gait.speed,
                    accel=0.0,
                    phase=phase,
                    gait=gait,
                    idle_weight=1.0 if is_idle else 0.0,
                    brake_weight=0.0,
                    brake_progress=0.0,
                    cadence_hz=0.0 if is_idle else gait.cadence,
                )
            )
        vid_scheds[name] = lst

    out_video = Path(args.out_video).resolve()
    frames_dir = out_video.parent / "_sample4_frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    for old in frames_dir.glob("frame_*.png"):
        old.unlink()

    tasks = [
        (
            fi,
            {k: vid_scheds[k][fi] for k in CORE_SAMPLE_GAITS},
            str(frames_dir / f"frame_{fi:05d}.png"),
        )
        for fi in range(n_frames)
    ]

    tile_w, tile_h = 480, 270  # Total 960x540 HD 2x2 grid
    print(f"Rendering {n_frames} 2x2 grid frames ({tile_w * 2}x{tile_h * 2} @ {args.fps}fps)...")
    done = 0
    with ProcessPoolExecutor(
        max_workers=max(1, args.workers),
        initializer=_init_worker,
        initargs=(str(fbx_path), tile_w, tile_h),
    ) as pool:
        for _ in pool.map(_render_grid_frame, tasks, chunksize=3):
            done += 1
            if done % 15 == 0 or done == n_frames:
                el = time.perf_counter() - t0
                print(f"  [{done:2d}/{n_frames}] {el:.1f}s elapsed", flush=True)

    # Save representative mid-stride filmstrip frame
    mid_png = frames_dir / f"frame_{min(n_frames - 1, 22):05d}.png"
    out_strip = Path(args.out_filmstrip).resolve()
    Image.open(mid_png).save(out_strip)

    # Encode H.264 MP4
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
    subprocess.run(cmd, capture_output=True, text=True, check=True)
    for old in frames_dir.glob("frame_*.png"):
        old.unlink()
    frames_dir.rmdir()

    print(
        f"DONE in {time.perf_counter() - t0:.1f}s -> {out_video} "
        f"({out_video.stat().st_size / (1024 * 1024):.2f} MB)"
    )


if __name__ == "__main__":
    main()
