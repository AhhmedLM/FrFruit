"""Export all 27 remaining Unity FBX animations (Categories 2-6) + 5-category showcase MP4."""

from __future__ import annotations

import argparse
import subprocess
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont

from frfruit.animation.action_library import (
    ACTION_SPECS,
    bake_named_action_to_timeline,
    evaluate_named_action,
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
from frfruit.geometry.primitives import make_plane
from frfruit.geometry.spatial import evaluate_scene_meshes
from frfruit.io.fbx_exporter import export_fbx_scene, verify_fbx_binary
from frfruit.io.importers import import_mesh_file
from frfruit.render.software import RenderConfig, SoftwareRenderer

CATEGORIES: list[tuple[str, tuple[int, int, int], list[str], tuple[int, int]]] = [
    (
        "2. CONSTRUCTION & ECONOMY",
        (42, 214, 198),
        [
            "Carry_Idle",
            "Carry_Walk",
            "Pickup_Item",
            "Drop_Item",
            "Hammer_Wall",
            "Hammer_Ground",
            "Saw_Wood",
            "Inspect_Work",
        ],
        (2, 4),
    ),
    (
        "3. REACTIONS & CIVILIANS",
        (255, 178, 42),
        ["Flee_Run", "Cower_Idle", "Hit_Flinch", "Stunned_Loop", "Cheer_Victory"],
        (2, 3),
    ),
    (
        "4. MELEE COMBAT",
        (255, 92, 46),
        [
            "Melee_Idle",
            "Charge_Run",
            "Slash_Attack_1",
            "Slash_Attack_2",
            "Bash_Gate",
            "Block_Idle",
        ],
        (2, 3),
    ),
    (
        "5. RANGED COMBAT",
        (168, 130, 255),
        ["Bow_Draw", "Bow_Aim_Idle", "Bow_Release", "Throw_Stone"],
        (2, 2),
    ),
    (
        "6. SIEGE OPERATION",
        (72, 186, 255),
        ["Push_Ram", "Crank_Wheel", "Pull_Lever", "Load_Ammo"],
        (2, 2),
    ),
]


def _build_studio(fbx_path: Path) -> tuple[SceneDocument, SceneObject]:
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
    scene.post_process.bloom_enabled = False
    scene.post_process.vignette_intensity = 0.12
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


_W_SCENE: SceneDocument | None = None
_W_ARM: SceneObject | None = None
_W_RIG: Any = None
_W_REND: SoftwareRenderer | None = None
_W_CAM: SceneObject | None = None


def _init_worker(fbx_str: str) -> None:
    global _W_SCENE, _W_ARM, _W_RIG, _W_REND, _W_CAM
    _W_SCENE, _W_ARM = _build_studio(Path(fbx_str))
    assert _W_ARM.armature is not None
    _W_RIG = detect_humanoid_rig(_W_ARM.armature)
    _W_REND = SoftwareRenderer()
    _W_CAM = SceneObject(
        id="cam",
        name="Cam",
        kind="camera",
        transform=Transform(location=(1.95, -2.15, 1.08)),
        camera=CameraSpec(target=(0.0, -0.08, 0.88), fov_deg=44.0),
    )


def _render_category_frame(
    args: tuple[int, str, tuple[int, int, int], list[str], tuple[int, int], float, str],
) -> int:
    seq_idx, cat_title, accent, names, (rows, cols), u, out_path = args
    assert _W_SCENE is not None and _W_ARM is not None and _W_ARM.armature is not None
    assert _W_REND is not None and _W_CAM is not None

    total_w, total_h = 960, 540
    tw, th = total_w // cols, total_h // rows
    cfg = RenderConfig(
        width=tw, height=th, mode="pbr", show_grid=True, show_axes=False, enable_shadows=True
    )
    canvas = Image.new("RGB", (total_w, total_h), (12, 15, 22))
    f_title = _load_font(13, bold=True)
    f_cat = _load_font(13, bold=True)

    for idx, name in enumerate(names):
        evaluate_named_action(_W_ARM.armature, _W_RIG, name, u)
        ev = evaluate_scene_meshes(_W_SCENE)
        tile = _W_REND.render_image(_W_SCENE, ev, _W_CAM, cfg).convert("RGB")
        draw = ImageDraw.Draw(tile)
        draw.rectangle([0, 0, tw, 24], fill=(10, 13, 19))
        draw.rectangle([0, 0, 4, 24], fill=accent)
        draw.text((10, 4), name, font=f_title, fill=(242, 247, 255))
        r, c = divmod(idx, cols)
        canvas.paste(tile, (c * tw, r * th))

    # If grid has an empty slot (e.g. 5 clips in 2x3), draw category summary card there
    if len(names) < rows * cols:
        r, c = divmod(len(names), cols)
        x0, y0 = c * tw, r * th
        d2 = ImageDraw.Draw(canvas)
        d2.rectangle([x0 + 8, y0 + 8, x0 + tw - 8, y0 + th - 8], fill=(15, 19, 28), outline=accent)
        d2.text((x0 + 22, y0 + th // 2 - 12), cat_title, font=f_cat, fill=(235, 245, 255))

    canvas.save(out_path, format="PNG", compress_level=1)
    return seq_idx


def main() -> None:
    parser = argparse.ArgumentParser(description="Export all 27 remaining Unity FBXs + video.")
    parser.add_argument("--fbx", default="workspace/T-Pose.fbx")
    parser.add_argument("--anim-dir", default="workspace/animations")
    parser.add_argument("--out-video", default="workspace/all_categories_showcase.mp4")
    parser.add_argument("--out-filmstrip", default="workspace/all_27_filmstrip.png")
    parser.add_argument("--fps", type=int, default=24)
    parser.add_argument("--frames-per-cat", type=int, default=30)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--preview-only", action="store_true")
    args = parser.parse_args()

    t0 = time.perf_counter()
    fbx_path = Path(args.fbx).resolve()
    anim_dir = Path(args.anim_dir).resolve()
    anim_dir.mkdir(parents=True, exist_ok=True)

    # 1. Export all 27 separate Unity Binary FBX 7.4 files
    print(f"Exporting {len(ACTION_SPECS)} separate Unity Binary FBX 7.4 animations...")
    for name in ACTION_SPECS:
        sc = SceneDocument()
        objs = import_mesh_file(sc, fbx_path)
        arm = next(o for o in objs if o.kind == "armature")
        nf = bake_named_action_to_timeline(sc, arm, name, fps=30)
        out_fbx = anim_dir / f"{name}.fbx"
        export_fbx_scene(sc, out_fbx, ascii_mode=False, target_engine="unity")
        info = verify_fbx_binary(out_fbx)
        assert info["valid"], f"Invalid FBX for {name}"
        print(f"  {name:16s} -> {out_fbx.name} ({nf}f, {out_fbx.stat().st_size // 1024} KB)")

    if args.preview_only:
        _init_worker(str(fbx_path))
        # Render 1 peak-action frame (u=0.52) for all 27 clips into a 5-row overview sheet
        out_strip = Path(args.out_filmstrip).resolve()
        sheet = Image.new("RGB", (960, 540 * len(CATEGORIES)))
        for ci, (cat_title, accent, names, grid_rc) in enumerate(CATEGORIES):
            tmp = out_strip.parent / f"_tmp_cat_{ci}.png"
            _render_category_frame((ci, cat_title, accent, names, grid_rc, 0.52, str(tmp)))
            sheet.paste(Image.open(tmp), (0, ci * 540))
            tmp.unlink()
        sheet.save(out_strip)
        print(f"Saved 27-animation contact sheet -> {out_strip} in {time.perf_counter() - t0:.1f}s")
        return

    out_video = Path(args.out_video).resolve()
    frames_dir = out_video.parent / "_all27_frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    for old in frames_dir.glob("frame_*.png"):
        old.unlink()

    tasks = []
    seq = 0
    n_cat = args.frames_per_cat
    for cat_title, accent, names, grid_rc in CATEGORIES:
        for fi in range(n_cat):
            u = fi / float(max(1, n_cat - 1))
            tasks.append(
                (
                    seq,
                    cat_title,
                    accent,
                    names,
                    grid_rc,
                    u,
                    str(frames_dir / f"frame_{seq:05d}.png"),
                )
            )
            seq += 1

    print(
        f"Rendering {len(tasks)} category-grid frames @ {args.fps}fps "
        f"with {args.workers} workers..."
    )
    done = 0
    with ProcessPoolExecutor(
        max_workers=max(1, args.workers),
        initializer=_init_worker,
        initargs=(str(fbx_path),),
    ) as pool:
        for _ in pool.map(_render_category_frame, tasks, chunksize=3):
            done += 1
            if done % 25 == 0 or done == len(tasks):
                print(f"  [{done:3d}/{len(tasks)}] {time.perf_counter() - t0:.1f}s", flush=True)

    # Build 27-clip contact sheet from mid-action frames
    out_strip = Path(args.out_filmstrip).resolve()
    sheet = Image.new("RGB", (960, 540 * len(CATEGORIES)))
    for ci in range(len(CATEGORIES)):
        mid_idx = ci * n_cat + (n_cat // 2)
        sheet.paste(Image.open(frames_dir / f"frame_{mid_idx:05d}.png"), (0, ci * 540))
    sheet.save(out_strip)

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
