"""Abstract renderer backend protocol and factory."""

from __future__ import annotations

from typing import Protocol

from PIL import Image

from frfruit.core.scene import SceneDocument, SceneObject
from frfruit.geometry.spatial import EvaluatedObject
from frfruit.render.pathtracer import PathTracerRenderer
from frfruit.render.software import RenderConfig, SoftwareRenderer


class RendererBackend(Protocol):
    def render_image(
        self,
        scene: SceneDocument,
        evaluated: dict[str, EvaluatedObject],
        cam_obj: SceneObject,
        config: RenderConfig | None = None,
        colliding_ids: set[str] | None = None,
    ) -> Image.Image: ...


def get_renderer(backend: str = "auto") -> RendererBackend:
    key = backend.lower().strip()
    if key in ("auto", "software", "cpu", "raster"):
        return SoftwareRenderer()
    if key in ("raytrace", "pathtracer", "rt", "cycles"):
        return PathTracerRenderer()
    raise ValueError(
        f"Unknown renderer backend: {backend!r}. "
        "Available: 'auto', 'software', 'raytrace', 'pathtracer'."
    )
