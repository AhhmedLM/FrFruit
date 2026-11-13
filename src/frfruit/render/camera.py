"""Camera framing and view/projection helpers."""

from __future__ import annotations

import math
from typing import Literal

import numpy as np

from frfruit.core.math3d import look_at_matrix, orthographic_matrix, perspective_matrix
from frfruit.core.scene import CameraSpec, SceneObject
from frfruit.core.types import BoundingBox, Transform, Vec3

ViewDirection = Literal["iso", "front", "back", "left", "right", "top", "bottom"]

_DIR_VECTORS: dict[str, tuple[Vec3, Vec3]] = {
    "iso": ((1.0, -1.0, 0.72), (0.0, 0.0, 1.0)),
    "front": ((0.0, -1.0, 0.0), (0.0, 0.0, 1.0)),
    "back": ((0.0, 1.0, 0.0), (0.0, 0.0, 1.0)),
    "right": ((1.0, 0.0, 0.0), (0.0, 0.0, 1.0)),
    "left": ((-1.0, 0.0, 0.0), (0.0, 0.0, 1.0)),
    "top": ((0.0, 0.0, 1.0), (0.0, 1.0, 0.0)),
    "bottom": ((0.0, 0.0, -1.0), (0.0, 1.0, 0.0)),
}


def auto_frame_camera_object(
    bounds: BoundingBox,
    direction: ViewDirection = "iso",
    margin: float = 1.25,
    projection: Literal["perspective", "orthographic"] = "perspective",
    fov_deg: float = 45.0,
    name: str = "AutoCamera",
) -> SceneObject:
    """Create a SceneObject camera automatically framed around `bounds`."""
    raw_dir, up = _DIR_VECTORS[direction]
    d = np.asarray(raw_dir, dtype=np.float64)
    d /= np.linalg.norm(d)

    center = np.asarray(bounds.center, dtype=np.float64)
    size = np.asarray(bounds.size, dtype=np.float64)
    radius = max(0.25, 0.5 * float(np.linalg.norm(size)))
    max_dim = max(0.5, float(np.max(size)))

    half_fov = math.radians(fov_deg * 0.5)
    dist = (radius * margin) / max(1e-4, math.sin(half_fov))
    eye = center + d * dist

    return SceneObject(
        id=f"cam_{direction}",
        name=name,
        kind="camera",
        transform=Transform(location=(float(eye[0]), float(eye[1]), float(eye[2]))),
        camera=CameraSpec(
            projection=projection,
            fov_deg=fov_deg,
            ortho_scale=max_dim * margin * 1.15,
            near=max(0.01, dist * 0.02),
            far=max(100.0, dist * 20.0),
            target=(float(center[0]), float(center[1]), float(center[2])),
            up=up,
        ),
    )


def build_camera_matrices(
    cam_obj: SceneObject,
    aspect: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (view_matrix, projection_matrix, eye_world) for a camera SceneObject."""
    if cam_obj.camera is None:
        raise ValueError(f"Object {cam_obj.name!r} has no camera specification.")
    cam = cam_obj.camera
    eye = cam_obj.transform.location
    view = look_at_matrix(eye=eye, target=cam.target, up=cam.up)
    if cam.projection == "orthographic":
        proj = orthographic_matrix(cam.ortho_scale, aspect, cam.near, cam.far)
    else:
        proj = perspective_matrix(cam.fov_deg, aspect, cam.near, cam.far)
    return view, proj, np.asarray(eye, dtype=np.float64)
