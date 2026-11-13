"""Headless 3D rendering, materials, cameras, and AI viewport visualizations."""

from frfruit.render.backend import RendererBackend, get_renderer
from frfruit.render.camera import ViewDirection, auto_frame_camera_object, build_camera_matrices
from frfruit.render.materials import (
    PRESET_MATERIALS,
    evaluate_surface_albedo,
    list_material_presets,
)
from frfruit.render.software import RenderConfig, RenderMode, SoftwareRenderer, image_to_png_bytes
from frfruit.render.viewport import (
    annotate_viewport,
    render_animation_sequence,
    render_multiview_sheet,
    render_single_view,
)

__all__ = [
    "PRESET_MATERIALS",
    "RenderConfig",
    "RenderMode",
    "RendererBackend",
    "SoftwareRenderer",
    "ViewDirection",
    "annotate_viewport",
    "auto_frame_camera_object",
    "build_camera_matrices",
    "evaluate_surface_albedo",
    "get_renderer",
    "image_to_png_bytes",
    "list_material_presets",
    "render_animation_sequence",
    "render_multiview_sheet",
    "render_single_view",
]
