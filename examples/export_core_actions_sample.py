"""Export Climb_Ladder, Fall_Loop, and Get_Up as separate Unity FBXs + 3-panel video."""

from __future__ import annotations

import argparse
import subprocess
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont

from frfruit.animation.core_actions import (
    bake_core_action_to_timeline,
    evaluate_climb_ladder,
    evaluate_fall_loop,
    evaluate_get_up,
)
from frfruit.animation.locomotion import detect_humanoid_rig
from frfruit.core.scene import (
    CameraSpec,
    LightSpec,
    MaterialSpec,
    SceneDocument,
    SceneObject,
)
from frfruit.core.types import Transform
from frfruit.geometry.primitives import make_cube, make_cylinder, make_plane
from frfruit.geometry.spatial import evaluate_scene_meshes
from frfruit.io.fbx_exporter import export_fbx_scene, verify_fbx_binary
from frfruit.io.importers import import_mesh_file
from frfruit.render.software import RenderConfig, SoftwareRenderer


def _build_panel_scene(fbx_path: Path, mode: str) -> tuple[SceneDocument, SceneObject]:
    """Build studio scene tailored for Climb_Ladder (with wooden ladder), Fall_Loop, or Get_Up."""
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
    scene.post_process.bloom_intensity = 0.18
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
            cast_shadows=(mode != "Fall_Loop"),
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

    if mode != "Fall_Loop":
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

    if mode == "Climb_Ladder":
        scene.materials["mat_wood"] = MaterialSpec(
            id="mat_wood",
            name="LadderWood",
            base_color=(0.45, 0.28, 0.14, 1.0),
            roughness=0.72,
            metallic=0.02,
        )
        rail_m = make_cube(size=(0.055, 0.055, 2.6)).to_payload()
        for tag, rx in (("L", 0.24), ("R", -0.24)):
            scene.objects[f"ladder_rail_{tag}"] = SceneObject(
                id=f"ladder_rail_{tag}",
                name=f"LadderRail_{tag}",
                kind="mesh",
                material_id="mat_wood",
                transform=Transform(location=(rx, -0.35, 1.3)),
                mesh=rail_m,
            )
        rung_m = make_cylinder(radius=0.020, height=0.48, segments=12, origin="center").to_payload()
        for ri, rz in enumerate((0.14, 0.48, 0.82, 1.16, 1.48, 1.80, 2.12)):
            scene.objects[f"ladder_rung_{ri}"] = SceneObject(
                id=f"ladder_rung_{ri}",
                name=f"LadderRung_{ri}",
                kind="mesh",
                material_id="mat_wood",
                transform=Transform(location=(0.0, -0.35, rz), rotation_deg=(0.0, 90.0, 0.0)),
                mesh=rung_m,
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


# Worker globals (3 scenes per worker)
_W_SCENES: dict[str, tuple[SceneDocument, SceneObject, Any, SceneObject]] = {}
_W_REND: SoftwareRenderer | None = None
_W_CFG: RenderConfig | None = None


def _init_worker(fbx_str: str, tile_w: int, tile_h: int) -> None:
    global _W_SCENES, _W_REND, _W_CFG
    fbx_p = Path(fbx_str)
    cams = {
        "Climb_Ladder": SceneObject(
            id="c_climb",
            name="CamClimb",
            kind="camera",
            transform=Transform(location=(2.05, 1.15, 1.28)),
            camera=CameraSpec(target=(0.0, -0.18, 0.98), fov_deg=44.0),
        ),
        "Fall_Loop": SceneObject(
            id="c_fall",
            name="CamFall",
            kind="camera",
            transform=Transform(location=(2.15, -1.65, 0.95)),
            camera=CameraSpec(target=(0.0, 0.0, 0.88), fov_deg=44.0),
        ),
        "Get_Up": SceneObject(
            id="c_getup",
            name="CamGetUp",
            kind="camera",
            transform=Transform(location=(2.45, -1.80, 1.05)),
            camera=CameraSpec(target=(0.0, 0.15, 0.85), fov_deg=47.0),
        ),
    }
    _W_SCENES = {}
    for mode in ("Climb_Ladder", "Fall_Loop", "Get_Up"):
        sc, arm = _build_panel_scene(fbx_p, mode)
        assert arm.armature is not None
        rig = detect_humanoid_rig(arm.armature)
        _W_SCENES[mode] = (sc, arm, rig, cams[mode])
    _W_REND = SoftwareRenderer()
    _W_CFG = RenderConfig(
        width=tile_w,
        height=tile_h,
        mode="pbr",
        show_grid=True,
        show_axes=False,
        enable_shadows=True,
    )


def _render_trio_frame(args: tuple[int, int, str]) -> int:
    fi, total_frames, out_path = args
    assert _W_REND is not None and _W_CFG is not None
    tw, th = _W_CFG.width, _W_CFG.height
    strip = Image.new("RGB", (tw * 3, th))
    f_title = _load_font(15, bold=True)
    f_sub = _load_font(11, bold=False)

    t_sec = fi / 30.0
    specs = [
        ("Climb_Ladder", (255, 178, 42), "LOOP | 1.6s", (t_sec / 1.6) % 1.0, evaluate_climb_ladder),
        ("Fall_Loop", (72, 186, 255), "LOOP | 1.2s", (t_sec / 1.2) % 1.0, evaluate_fall_loop),
        (
            "Get_Up",
            (42, 214, 198),
            "1-SHOT | 2.2s",
            fi / float(max(1, total_frames - 1)),
            evaluate_get_up,
        ),
    ]
    for idx, (name, accent, tag, u, eval_fn) in enumerate(specs):
        sc, arm, rig, cam = _W_SCENES[name]
        assert arm.armature is not None
        eval_fn(arm.armature, rig, u)
        ev = evaluate_scene_meshes(sc)
        tile = _W_REND.render_image(sc, ev, cam, _W_CFG).convert("RGB")
        draw = ImageDraw.Draw(tile)
        draw.rectangle([0, 0, tw, 28], fill=(10, 13, 19))
        draw.rectangle([0, 0, 5, 28], fill=accent)
        draw.text((12, 5), name, font=f_title, fill=(242, 247, 255))
        draw.text((tw - 130, 8), f"{tag} F{fi + 1:02d}", font=f_sub, fill=(180, 198, 222))
        strip.paste(tile, (idx * tw, 0))

    strip.save(out_path, format="PNG", compress_level=1)
    return fi


def main() -> None:
    parser = argparse.ArgumentParser(description="Export Climb_Ladder, Fall_Loop, Get_Up + video.")
    parser.add_argument("--fbx", default="workspace/T-Pose.fbx")
    parser.add_argument("--anim-dir", default="workspace/animations")
    parser.add_argument("--out-video", default="workspace/sample_3_core_actions.mp4")
    parser.add_argument("--out-filmstrip", default="workspace/sample_3_filmstrip.png")
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--preview-only", action="store_true")
    args = parser.parse_args()

    t0 = time.perf_counter()
    fbx_path = Path(args.fbx).resolve()
    anim_dir = Path(args.anim_dir).resolve()
    anim_dir.mkdir(parents=True, exist_ok=True)

    # 1. Export 3 separate Unity Binary FBX 7.4 files
    print("Exporting 3 separate Unity Binary FBX 7.4 animations...")
    for name in ("Climb_Ladder", "Fall_Loop", "Get_Up"):
        sc = SceneDocument()
        objs = import_mesh_file(sc, fbx_path)
        arm = next(o for o in objs if o.kind == "armature")
        nf = bake_core_action_to_timeline(sc, arm, name, fps=args.fps)
        out_fbx = anim_dir / f"{name}.fbx"
        export_fbx_scene(sc, out_fbx, ascii_mode=False, target_engine="unity")
        info = verify_fbx_binary(out_fbx)
        print(
            f"  {name:14s} -> {out_fbx.name} ({nf} frames, "
            f"{out_fbx.stat().st_size / 1024:.0f} KB, valid={info['valid']})"
        )

    tile_w, tile_h = 360, 360  # Total 1080x360 crisp 3-panel strip
    n_frames = int(round(2.2 * args.fps))  # 66 frames @ 30fps

    if args.preview_only:
        _init_worker(str(fbx_path), tile_w, tile_h)
        picks = [4, 24, 48]
        rows = []
        for fi in picks:
            tmp = Path(args.out_filmstrip).parent / f"_tmp_prev_{fi}.png"
            _render_trio_frame((fi, n_frames, str(tmp)))
            rows.append(Image.open(tmp))
            tmp.unlink()
        sheet = Image.new("RGB", (tile_w * 3, tile_h * len(rows)))
        for ri, im in enumerate(rows):
            sheet.paste(im, (0, ri * tile_h))
        sheet.save(args.out_filmstrip)
        print(f"Preview sheet saved -> {args.out_filmstrip}")
        return

    out_video = Path(args.out_video).resolve()
    frames_dir = out_video.parent / "_sample3_frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    for old in frames_dir.glob("frame_*.png"):
        old.unlink()

    tasks = [(fi, n_frames, str(frames_dir / f"frame_{fi:05d}.png")) for fi in range(n_frames)]
    print(f"Rendering {n_frames} 3-panel frames ({tile_w * 3}x{tile_h} @ {args.fps}fps)...")
    done = 0
    with ProcessPoolExecutor(
        max_workers=max(1, args.workers),
        initializer=_init_worker,
        initargs=(str(fbx_path), tile_w, tile_h),
    ) as pool:
        for _ in pool.map(_render_trio_frame, tasks, chunksize=3):
            done += 1
            if done % 15 == 0 or done == n_frames:
                print(f"  [{done:2d}/{n_frames}] {time.perf_counter() - t0:.1f}s", flush=True)

    # Build 2-row contact sheet (early & mid/late phases)
    r1 = Image.open(frames_dir / "frame_00010.png")
    r2 = Image.open(frames_dir / "frame_00034.png")
    sheet = Image.new("RGB", (tile_w * 3, tile_h * 2))
    sheet.paste(r1, (0, 0))
    sheet.paste(r2, (0, tile_h))
    sheet.save(args.out_filmstrip)

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
