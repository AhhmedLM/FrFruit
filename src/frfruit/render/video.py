"""Video rendering pipeline exporting MP4 (H.264), WebM (VP9), AVI, and GIF via FFmpeg."""

from __future__ import annotations

import io
import math
import shutil
import subprocess
from pathlib import Path
from typing import Any, Literal

import numpy as np
from PIL import Image

from frfruit.animation.evaluator import apply_animation_frame
from frfruit.core.scene import CameraSpec, SceneDocument, SceneObject
from frfruit.core.types import Transform
from frfruit.geometry.spatial import compute_scene_bounds, evaluate_scene_meshes
from frfruit.render.backend import RendererBackend, get_renderer
from frfruit.render.camera import auto_frame_camera_object
from frfruit.render.software import RenderConfig, RenderMode
from frfruit.render.viewport import _resolve_camera, annotate_viewport

VideoKind = Literal["timeline", "turntable", "camera_path"]
VideoFormat = Literal["mp4", "webm", "avi", "gif"]


def _catmull_rom_point(points: np.ndarray, t_global: float) -> np.ndarray:
    """Evaluate a Catmull-Rom spline smoothly through (K, D) waypoints for t in [0, 1]."""
    k = len(points)
    if k == 0:
        raise ValueError("camera_waypoints cannot be empty.")
    if k == 1:
        return points[0].copy()

    scaled = max(0.0, min(1.0, float(t_global))) * (k - 1)
    idx = min(k - 2, int(math.floor(scaled)))
    u = scaled - idx

    p0 = points[max(0, idx - 1)]
    p1 = points[idx]
    p2 = points[min(k - 1, idx + 1)]
    p3 = points[min(k - 1, idx + 2)]

    u2 = u * u
    u3 = u2 * u
    return 0.5 * (
        (2.0 * p1)
        + (-p0 + p2) * u
        + (2.0 * p0 - 5.0 * p1 + 4.0 * p2 - p3) * u2
        + (-p0 + 3.0 * p1 - 3.0 * p2 + p3) * u3
    )


def render_video(
    scene: SceneDocument,
    out_path: str | Path,
    kind: VideoKind = "timeline",
    frame_start: int | None = None,
    frame_end: int | None = None,
    fps: int | None = None,
    width: int = 640,
    height: int = 480,
    mode: RenderMode = "pbr",
    engine: str | None = None,
    samples: int = 2,
    camera_id: str | None = None,
    camera_waypoints: list[dict[str, Any]] | None = None,
    show_grid: bool = True,
    show_axes: bool = True,
    overlay_hud: bool = True,
    bitrate: str = "4M",
    frames_dir: str | Path | None = None,
    renderer: RendererBackend | None = None,
) -> dict[str, Any]:
    """Render an animation, turntable, or smooth camera-path video to MP4/WebM/AVI/GIF."""
    p = Path(out_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    fmt = p.suffix.lstrip(".").lower() or "mp4"
    if fmt not in ("mp4", "webm", "avi", "gif"):
        raise ValueError(f"Unsupported video format {fmt!r}. Use mp4, webm, avi, or gif.")

    # Even dimensions required by yuv420p video codecs
    w = int(width) + (int(width) % 2)
    h = int(height) + (int(height) % 2)

    fps_val = max(1, int(fps if fps is not None else scene.timeline.fps))
    fs = scene.timeline.frame_start if frame_start is None else int(frame_start)
    fe = scene.timeline.frame_end if frame_end is None else int(frame_end)
    if fe < fs:
        fe = fs
    total_frames = max(2, fe - fs + 1)

    engine_name = engine or scene.render_engine
    rend = renderer or get_renderer(engine_name)
    cfg = RenderConfig(
        width=w,
        height=h,
        mode=mode,
        show_grid=show_grid,
        show_axes=show_axes,
        enable_shadows=(mode == "pbr"),
        samples=samples,
    )

    frames_out_dir: Path | None = Path(frames_dir) if frames_dir else None
    if frames_out_dir is not None:
        frames_out_dir.mkdir(parents=True, exist_ok=True)

    # Precompute turntable / camera-path geometry if scene topology is static
    wp_locs: np.ndarray | None = None
    wp_targets: np.ndarray | None = None
    wp_fovs: np.ndarray | None = None
    if kind == "camera_path":
        if not camera_waypoints or len(camera_waypoints) < 2:
            raise ValueError("kind='camera_path' requires at least 2 camera_waypoints.")
        wp_locs = np.array([w["location"] for w in camera_waypoints], dtype=np.float64)
        wp_targets = np.array(
            [w.get("target", (0.0, 0.0, 0.5)) for w in camera_waypoints], dtype=np.float64
        )
        wp_fovs = np.array(
            [float(w.get("fov_deg", 45.0)) for w in camera_waypoints], dtype=np.float64
        )

    rendered_images: list[Image.Image] = []
    static_eval = (
        evaluate_scene_meshes(scene) if kind == "turntable" and not scene.timeline.tracks else None
    )
    static_bounds = compute_scene_bounds(static_eval) if static_eval is not None else None

    for idx in range(total_frames):
        fnum = fs + idx
        t_norm = idx / max(1, total_frames - 1)

        if kind == "turntable" and static_eval is not None and static_bounds is not None:
            evaluated = static_eval
            bounds = static_bounds
            eval_doc = scene
        else:
            eval_doc = apply_animation_frame(scene, fnum)
            evaluated = evaluate_scene_meshes(eval_doc)
            bounds = compute_scene_bounds(evaluated)

        if kind == "turntable":
            base_cam = auto_frame_camera_object(
                bounds=bounds, direction="iso", projection="perspective"
            )
            center = np.asarray(bounds.center, dtype=np.float64)
            offset = np.asarray(base_cam.transform.location, dtype=np.float64) - center
            ang = 2.0 * math.pi * (idx / max(1, total_frames))
            ca, sa = math.cos(ang), math.sin(ang)
            rx = offset[0] * ca - offset[1] * sa
            ry = offset[0] * sa + offset[1] * ca
            eye = (float(center[0] + rx), float(center[1] + ry), float(center[2] + offset[2]))
            cam = SceneObject(
                id=f"vid_turn_{idx}",
                name=f"Orbit {int(math.degrees(ang))}deg",
                kind="camera",
                transform=Transform(location=eye),
                camera=CameraSpec(
                    projection="perspective",
                    fov_deg=base_cam.camera.fov_deg if base_cam.camera else 45.0,
                    target=bounds.center,
                ),
            )
            title = f"Orbit {int(math.degrees(ang))}deg | {mode.upper()}"
        elif kind == "camera_path":
            assert wp_locs is not None and wp_targets is not None and wp_fovs is not None
            loc = _catmull_rom_point(wp_locs, t_norm)
            tgt = _catmull_rom_point(wp_targets, t_norm)
            fov = float(_catmull_rom_point(wp_fovs[:, None], t_norm)[0])
            cam = SceneObject(
                id=f"vid_path_{idx}",
                name="PathCam",
                kind="camera",
                transform=Transform(location=(float(loc[0]), float(loc[1]), float(loc[2]))),
                camera=CameraSpec(
                    projection="perspective",
                    fov_deg=max(10.0, min(140.0, fov)),
                    target=(float(tgt[0]), float(tgt[1]), float(tgt[2])),
                ),
            )
            title = f"Path Frame {fnum}/{fe} | {mode.upper()}"
        else:
            cam = _resolve_camera(eval_doc, evaluated, camera_id=camera_id)
            title = f"{cam.name} | Frame {fnum} | {mode.upper()}"

        img = rend.render_image(eval_doc, evaluated, cam, config=cfg)
        if overlay_hud:
            img = annotate_viewport(img, evaluated, cam, show_axes=show_axes, title=title)
        if img.mode != "RGB":
            img = img.convert("RGB")
        if frames_out_dir is not None:
            img.save(frames_out_dir / f"frame_{fnum:04d}.png")
        rendered_images.append(img)

    if fmt == "gif":
        duration_ms = max(20, int(1000 / fps_val))
        gif_buf = io.BytesIO()
        rendered_images[0].save(
            gif_buf,
            format="GIF",
            save_all=True,
            append_images=rendered_images[1:],
            duration=duration_ms,
            loop=0,
        )
        p.write_bytes(gif_buf.getvalue())
    else:
        ffmpeg_bin = shutil.which("ffmpeg")
        if not ffmpeg_bin:
            raise RuntimeError(
                "FFmpeg binary not found on PATH; cannot encode MP4/WebM/AVI video. "
                "Use '.gif' format or install ffmpeg."
            )
        if fmt == "mp4":
            codec_args = [
                "-c:v",
                "libx264",
                "-preset",
                "medium",
                "-pix_fmt",
                "yuv420p",
                "-b:v",
                bitrate,
                "-movflags",
                "+faststart",
            ]
        elif fmt == "webm":
            codec_args = ["-c:v", "libvpx-vp9", "-pix_fmt", "yuv420p", "-b:v", bitrate]
        else:  # avi
            codec_args = ["-c:v", "mpeg4", "-q:v", "3"]

        cmd = [
            ffmpeg_bin,
            "-y",
            "-f",
            "rawvideo",
            "-vcodec",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "-s",
            f"{w}x{h}",
            "-r",
            str(fps_val),
            "-i",
            "-",
            "-an",
            *codec_args,
            str(p),
        ]
        raw_stream = b"".join(img.tobytes() for img in rendered_images)
        proc = subprocess.run(cmd, input=raw_stream, capture_output=True, check=False)
        if proc.returncode != 0:
            err_msg = proc.stderr.decode("utf-8", errors="replace").strip()
            raise RuntimeError(f"FFmpeg exited with code {proc.returncode}: {err_msg}")

    return {
        "path": str(p.resolve()),
        "format": fmt,
        "kind": kind,
        "engine": engine_name,
        "mode": mode,
        "width": w,
        "height": h,
        "fps": fps_val,
        "frames_rendered": len(rendered_images),
        "duration_sec": round(len(rendered_images) / fps_val, 3),
        "bytes": p.stat().st_size,
        "frames_dir": str(frames_out_dir.resolve()) if frames_out_dir else None,
    }
