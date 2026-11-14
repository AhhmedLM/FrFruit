"""AI Viewport overlays, multi-view contact sheets, and turntable/animation renders."""

from __future__ import annotations

import io
import math
from typing import Literal

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from frfruit.animation.evaluator import apply_animation_frame
from frfruit.core.scene import CameraSpec, SceneDocument, SceneObject
from frfruit.core.types import Transform
from frfruit.geometry.spatial import (
    EvaluatedObject,
    check_intersections,
    compute_scene_bounds,
    evaluate_scene_meshes,
)
from frfruit.render.backend import RendererBackend, get_renderer
from frfruit.render.camera import ViewDirection, auto_frame_camera_object, build_camera_matrices
from frfruit.render.software import RenderConfig, RenderMode, image_to_png_bytes


def _load_font(size: int = 13) -> ImageFont.ImageFont:
    try:
        return ImageFont.load_default(size=size)
    except TypeError:
        return ImageFont.load_default()


def _project_point(
    pt: np.ndarray, vp: np.ndarray, w: int, h: int
) -> tuple[float, float, float] | None:
    hom = np.array([pt[0], pt[1], pt[2], 1.0], dtype=np.float64)
    clip = vp @ hom
    if clip[3] <= 1e-4:
        return None
    ndc = clip[:3] / clip[3]
    sx = (ndc[0] * 0.5 + 0.5) * w
    sy = (0.5 - ndc[1] * 0.5) * h
    return float(sx), float(sy), float(ndc[2])


def annotate_viewport(
    img: Image.Image,
    evaluated: dict[str, EvaluatedObject],
    cam_obj: SceneObject,
    show_axes: bool = True,
    show_bounds: bool = False,
    show_labels: bool = False,
    title: str = "",
) -> Image.Image:
    """Draw XYZ orientation triad, optional object labels/bounds, and view badge."""
    out = img.copy()
    draw = ImageDraw.Draw(out)
    w, h = out.size
    font = _load_font(12)
    aspect = w / max(1, h)
    view, proj, _ = build_camera_matrices(cam_obj, aspect)
    vp = proj @ view

    if show_bounds:
        edges = [
            (0, 1),
            (1, 3),
            (3, 2),
            (2, 0),
            (4, 5),
            (5, 7),
            (7, 6),
            (6, 4),
            (0, 4),
            (1, 5),
            (2, 6),
            (3, 7),
        ]
        for ev in evaluated.values():
            if ev.world_mesh.is_empty:
                continue
            mn = ev.bounds.min_corner
            mx = ev.bounds.max_corner
            corners = [
                np.array([x, y, z], dtype=np.float64)
                for x in (mn[0], mx[0])
                for y in (mn[1], mx[1])
                for z in (mn[2], mx[2])
            ]
            proj_c = [_project_point(c, vp, w, h) for c in corners]
            for a, b in edges:
                pa, pb = proj_c[a], proj_c[b]
                if pa is not None and pb is not None:
                    draw.line([(pa[0], pa[1]), (pb[0], pb[1])], fill=(255, 210, 80), width=1)

    if show_labels:
        candidates: list[tuple[float, float, str, int, int]] = []
        for ev in evaluated.values():
            if ev.world_mesh.is_empty:
                continue
            center = np.asarray(ev.bounds.center, dtype=np.float64)
            pt2 = _project_point(center, vp, w, h)
            if pt2 is None:
                continue
            sx, sy, _ = pt2
            if 0 <= sx < w and 0 <= sy < h:
                txt = ev.obj.name
                bbox = draw.textbbox((0, 0), txt, font=font)
                tw = bbox[2] - bbox[0]
                th = bbox[3] - bbox[1]
                candidates.append((sy, sx, txt, tw, th))

        candidates.sort(key=lambda item: (item[0], item[1]))
        placed: list[tuple[int, int, int, int]] = []
        for sy, sx, txt, tw, th in candidates:
            bw, bh = tw + 6, th + 4
            bx0 = max(4, min(w - bw - 4, int(sx - bw / 2)))
            base_y0 = max(28, min(h - bh - 4, int(sy - bh / 2)))
            step = bh + 3
            best_y0 = base_y0
            for k in range(0, 10):
                offsets = (0,) if k == 0 else (-k * step, k * step)
                found = False
                for dy in offsets:
                    cand_y0 = base_y0 + dy
                    if cand_y0 < 28 or cand_y0 + bh > h - 4:
                        continue
                    bx1, by1 = bx0 + bw, cand_y0 + bh
                    if not any(
                        bx0 < px1 and bx1 > px0 and cand_y0 < py1 and by1 > py0
                        for px0, py0, px1, py1 in placed
                    ):
                        best_y0 = cand_y0
                        found = True
                        break
                if found:
                    break
            placed.append((bx0, best_y0, bx0 + bw, best_y0 + bh))
            draw.rectangle([bx0, best_y0, bx0 + bw, best_y0 + bh], fill=(20, 22, 28))
            draw.text((bx0 + 3, best_y0 + 2), txt, fill=(240, 245, 255), font=font)

    if show_axes:
        cx, cy = 36, h - 36
        arm = 22.0
        rot = view[:3, :3]
        axes = [
            ("X", np.array([1.0, 0.0, 0.0]), (235, 75, 75)),
            ("Y", np.array([0.0, 1.0, 0.0]), (80, 220, 100)),
            ("Z", np.array([0.0, 0.0, 1.0]), (85, 155, 255)),
        ]
        draw.ellipse([cx - 26, cy - 26, cx + 26, cy + 26], fill=(18, 20, 26), outline=(70, 75, 88))
        # Sort back-to-front by camera-space Z
        transformed = [(name, rot @ vec, col) for name, vec, col in axes]
        transformed.sort(key=lambda item: float(item[1][2]))
        for name, v_cam, col in transformed:
            ex = cx + float(v_cam[0]) * arm
            ey = cy - float(v_cam[1]) * arm
            draw.line([(cx, cy), (ex, ey)], fill=col, width=2)
            draw.text((ex - 3, ey - 6), name, fill=col, font=font)

    if title:
        bbox = draw.textbbox((0, 0), title, font=font)
        tw = bbox[2] - bbox[0]
        th = bbox[3] - bbox[1]
        draw.rectangle([8, 8, 8 + tw + 12, 8 + th + 8], fill=(16, 18, 24))
        draw.text((14, 12), title, fill=(235, 240, 250), font=font)

    return out


def _resolve_camera(
    scene: SceneDocument,
    evaluated: dict[str, EvaluatedObject],
    camera_id: str | None = None,
    view_direction: ViewDirection | None = None,
) -> SceneObject:
    if view_direction is not None:
        bounds = compute_scene_bounds(evaluated)
        proj: Literal["perspective", "orthographic"] = (
            "perspective" if view_direction == "iso" else "orthographic"
        )
        return auto_frame_camera_object(
            bounds=bounds,
            direction=view_direction,
            projection=proj,
            name=f"View:{view_direction}",
        )
    cid = camera_id or scene.active_camera_id
    if cid:
        cam = scene.resolve_object(cid)
        if cam is not None and cam.kind == "camera" and cam.camera is not None:
            return cam
    bounds = compute_scene_bounds(evaluated)
    return auto_frame_camera_object(bounds=bounds, direction="iso", projection="perspective")


def render_single_view(
    scene: SceneDocument,
    width: int = 640,
    height: int = 480,
    mode: RenderMode = "pbr",
    camera_id: str | None = None,
    view_direction: ViewDirection | None = None,
    show_grid: bool = True,
    show_axes: bool = True,
    show_bounds: bool = False,
    show_labels: bool = False,
    engine: str | None = None,
    samples: int = 2,
    renderer: RendererBackend | None = None,
) -> tuple[bytes, dict[str, object]]:
    """Render one view of the scene to PNG bytes plus metadata."""
    engine_name = engine or scene.render_engine
    rend = renderer or get_renderer(engine_name)
    evaluated = evaluate_scene_meshes(scene)
    cam = _resolve_camera(scene, evaluated, camera_id=camera_id, view_direction=view_direction)

    colliding_ids: set[str] = set()
    collisions: list[dict[str, object]] = []
    if mode == "collision":
        collisions = check_intersections(evaluated)
        for c in collisions:
            colliding_ids.add(str(c["object_a"]))
            colliding_ids.add(str(c["object_b"]))

    cfg = RenderConfig(
        width=width,
        height=height,
        mode=mode,
        show_grid=show_grid,
        show_axes=show_axes,
        enable_shadows=(mode == "pbr"),
        samples=samples,
    )
    img = rend.render_image(scene, evaluated, cam, config=cfg, colliding_ids=colliding_ids)
    title = f"{cam.name} | {mode.upper()} ({engine_name}) | frame {scene.timeline.current_frame}"
    img = annotate_viewport(
        img,
        evaluated,
        cam,
        show_axes=show_axes,
        show_bounds=show_bounds,
        show_labels=show_labels,
        title=title,
    )
    png = image_to_png_bytes(img)
    meta: dict[str, object] = {
        "width": width,
        "height": height,
        "mode": mode,
        "engine": engine_name,
        "camera": cam.name,
        "object_count": len(evaluated),
        "frame": scene.timeline.current_frame,
    }
    if mode == "collision":
        meta["collisions"] = collisions
    return png, meta


def render_multiview_sheet(
    scene: SceneDocument,
    tile_width: int = 420,
    tile_height: int = 320,
    mode: RenderMode = "pbr",
    views: tuple[ViewDirection, ...] = ("front", "right", "top", "iso"),
    show_grid: bool = True,
    show_axes: bool = True,
    show_bounds: bool = False,
    show_labels: bool = True,
    engine: str | None = None,
    samples: int = 2,
    renderer: RendererBackend | None = None,
) -> tuple[bytes, dict[str, object]]:
    """Render a labeled 2x2 (or 2x3) multi-angle contact sheet into one PNG."""
    engine_name = engine or scene.render_engine
    rend = renderer or get_renderer(engine_name)
    evaluated = evaluate_scene_meshes(scene)
    bounds = compute_scene_bounds(evaluated)

    colliding_ids: set[str] = set()
    collisions: list[dict[str, object]] = []
    if mode == "collision":
        collisions = check_intersections(evaluated)
        for c in collisions:
            colliding_ids.add(str(c["object_a"]))
            colliding_ids.add(str(c["object_b"]))

    cols = 2 if len(views) <= 4 else 3
    rows = math.ceil(len(views) / cols)
    sheet = Image.new("RGB", (cols * tile_width, rows * tile_height), color=(14, 16, 22))

    cfg = RenderConfig(
        width=tile_width,
        height=tile_height,
        mode=mode,
        show_grid=show_grid,
        show_axes=show_axes,
        enable_shadows=(mode == "pbr"),
        samples=samples,
    )

    for idx, vdir in enumerate(views):
        proj: Literal["perspective", "orthographic"] = (
            "perspective" if vdir == "iso" else "orthographic"
        )
        cam = auto_frame_camera_object(
            bounds=bounds, direction=vdir, projection=proj, name=vdir.upper()
        )
        tile = rend.render_image(scene, evaluated, cam, config=cfg, colliding_ids=colliding_ids)
        tile = annotate_viewport(
            tile,
            evaluated,
            cam,
            show_axes=show_axes,
            show_bounds=show_bounds,
            show_labels=(show_labels and vdir == "iso"),
            title=f"{vdir.upper()} ({proj[:5]}) | {mode.upper()}",
        )
        r = idx // cols
        c = idx % cols
        sheet.paste(tile, (c * tile_width, r * tile_height))

    draw = ImageDraw.Draw(sheet)
    for c in range(1, cols):
        draw.line(
            [(c * tile_width, 0), (c * tile_width, rows * tile_height)], fill=(55, 62, 78), width=2
        )
    for r in range(1, rows):
        draw.line(
            [(0, r * tile_height), (cols * tile_width, r * tile_height)], fill=(55, 62, 78), width=2
        )

    meta: dict[str, object] = {
        "sheet_width": cols * tile_width,
        "sheet_height": rows * tile_height,
        "views": list(views),
        "mode": mode,
        "engine": engine_name,
        "object_count": len(evaluated),
        "bounds_min": list(bounds.min_corner),
        "bounds_max": list(bounds.max_corner),
    }
    if mode == "collision":
        meta["collisions"] = collisions
    return image_to_png_bytes(sheet), meta


def render_animation_sequence(
    scene: SceneDocument,
    kind: Literal["timeline", "turntable"] = "timeline",
    frame_start: int | None = None,
    frame_end: int | None = None,
    steps: int = 8,
    width: int = 360,
    height: int = 270,
    mode: RenderMode = "pbr",
    camera_id: str | None = None,
    engine: str | None = None,
    samples: int = 2,
    renderer: RendererBackend | None = None,
) -> tuple[bytes, bytes, dict[str, object]]:
    """Render an animation or turntable into (filmstrip_png_bytes, animated_gif_bytes, meta)."""
    engine_name = engine or scene.render_engine
    rend = renderer or get_renderer(engine_name)
    fs = scene.timeline.frame_start if frame_start is None else frame_start
    fe = scene.timeline.frame_end if frame_end is None else frame_end
    count = max(2, min(24, int(steps)))

    frames_rgb: list[Image.Image] = []
    frame_labels: list[str] = []

    if kind == "turntable":
        evaluated = evaluate_scene_meshes(scene)
        bounds = compute_scene_bounds(evaluated)
        base_cam = auto_frame_camera_object(
            bounds=bounds, direction="iso", projection="perspective"
        )
        center = np.asarray(bounds.center, dtype=np.float64)
        offset = np.asarray(base_cam.transform.location, dtype=np.float64) - center
        cfg = RenderConfig(
            width=width,
            height=height,
            mode=mode,
            show_grid=True,
            show_axes=True,
            samples=samples,
        )
        for i in range(count):
            ang = 2.0 * math.pi * (i / count)
            ca, sa = math.cos(ang), math.sin(ang)
            rx = offset[0] * ca - offset[1] * sa
            ry = offset[0] * sa + offset[1] * ca
            eye = (float(center[0] + rx), float(center[1] + ry), float(center[2] + offset[2]))
            cam_i = SceneObject(
                id=f"turn_{i}",
                name=f"Turn {int(math.degrees(ang))}deg",
                kind="camera",
                transform=Transform(location=eye),
                camera=CameraSpec(
                    projection="perspective",
                    fov_deg=base_cam.camera.fov_deg if base_cam.camera else 45.0,
                    target=bounds.center,
                ),
            )
            img = rend.render_image(scene, evaluated, cam_i, config=cfg)
            img = annotate_viewport(
                img, evaluated, cam_i, show_axes=True, title=f"{int(math.degrees(ang))}deg"
            )
            frames_rgb.append(img)
            frame_labels.append(f"{int(math.degrees(ang))}deg")
    else:
        frame_nums = [int(round(x)) for x in np.linspace(fs, fe, count)]
        cfg = RenderConfig(
            width=width,
            height=height,
            mode=mode,
            show_grid=True,
            show_axes=True,
            samples=samples,
        )
        for fnum in frame_nums:
            eval_doc = apply_animation_frame(scene, fnum)
            evaluated = evaluate_scene_meshes(eval_doc)
            cam = _resolve_camera(eval_doc, evaluated, camera_id=camera_id)
            img = rend.render_image(eval_doc, evaluated, cam, config=cfg)
            img = annotate_viewport(img, evaluated, cam, show_axes=True, title=f"Frame {fnum}")
            frames_rgb.append(img)
            frame_labels.append(f"f{fnum}")

    # Build filmstrip contact sheet
    cols = min(4, len(frames_rgb))
    rows = math.ceil(len(frames_rgb) / cols)
    strip = Image.new("RGB", (cols * width, rows * height), color=(14, 16, 22))
    for i, fr in enumerate(frames_rgb):
        strip.paste(fr, ((i % cols) * width, (i // cols) * height))

    gif_buf = io.BytesIO()
    duration_ms = max(40, int(1000 / max(1, scene.timeline.fps)))
    frames_rgb[0].save(
        gif_buf,
        format="GIF",
        save_all=True,
        append_images=frames_rgb[1:],
        duration=duration_ms,
        loop=0,
    )

    meta: dict[str, object] = {
        "kind": kind,
        "engine": engine_name,
        "frames_rendered": len(frames_rgb),
        "labels": frame_labels,
        "filmstrip_size": [cols * width, rows * height],
        "fps": scene.timeline.fps,
    }
    return image_to_png_bytes(strip), gif_buf.getvalue(), meta
