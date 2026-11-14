"""Headless 3D rendering, materials, cameras, post-processing, video, and AI viewport."""

from frfruit.render.backend import RendererBackend, get_renderer
from frfruit.render.camera import ViewDirection, auto_frame_camera_object, build_camera_matrices
from frfruit.render.materials import (
    PRESET_MATERIALS,
    evaluate_material_maps,
    evaluate_surface_albedo,
    list_material_presets,
)
from frfruit.render.pathtracer import PathTracerRenderer
from frfruit.render.postprocess import apply_post_processing
from frfruit.render.software import RenderConfig, RenderMode, SoftwareRenderer, image_to_png_bytes
from frfruit.render.video import render_video
from frfruit.render.viewport import (
    annotate_viewport,
    render_animation_sequence,
    render_multiview_sheet,
    render_single_view,
)

__all__ = [
    "PRESET_MATERIALS",
    "PathTracerRenderer",
    "RenderConfig",
    "RenderMode",
    "RendererBackend",
    "SoftwareRenderer",
    "ViewDirection",
    "annotate_viewport",
    "apply_post_processing",
    "auto_frame_camera_object",
    "build_camera_matrices",
    "evaluate_material_maps",
    "evaluate_surface_albedo",
    "get_renderer",
    "image_to_png_bytes",
    "list_material_presets",
    "render_animation_sequence",
    "render_multiview_sheet",
    "render_single_view",
    "render_video",
]
