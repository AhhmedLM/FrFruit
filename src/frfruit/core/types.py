"""Core type definitions for FrFruit."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, field_validator

Vec2 = tuple[float, float]
Vec3 = tuple[float, float, float]
ColorRGB = tuple[float, float, float]
ColorRGBA = tuple[float, float, float, float]


def parse_color_rgba(value: str | list[float] | tuple[float, ...]) -> ColorRGBA:
    """Parse hex string or 3/4-element float sequence into clamped RGBA (0..1)."""
    if isinstance(value, str):
        s = value.strip().lstrip("#")
        if len(s) == 6:
            r, g, b = int(s[0:2], 16) / 255.0, int(s[2:4], 16) / 255.0, int(s[4:6], 16) / 255.0
            return (r, g, b, 1.0)
        if len(s) == 8:
            r = int(s[0:2], 16) / 255.0
            g = int(s[2:4], 16) / 255.0
            b = int(s[4:6], 16) / 255.0
            a = int(s[6:8], 16) / 255.0
            return (r, g, b, a)
        raise ValueError(f"Invalid hex color: {value!r}")
    if len(value) == 3:
        r, g, b = float(value[0]), float(value[1]), float(value[2])
        return (
            max(0.0, min(1.0, r)),
            max(0.0, min(1.0, g)),
            max(0.0, min(1.0, b)),
            1.0,
        )
    if len(value) == 4:
        r, g, b, a = float(value[0]), float(value[1]), float(value[2]), float(value[3])
        return (
            max(0.0, min(1.0, r)),
            max(0.0, min(1.0, g)),
            max(0.0, min(1.0, b)),
            max(0.0, min(1.0, a)),
        )
    raise ValueError(f"Expected 3 or 4 color components, got {len(value)}")


class Transform(BaseModel):
    """3D transform using meters, XYZ Euler degrees, and axis scale."""

    location: Vec3 = (0.0, 0.0, 0.0)
    rotation_deg: Vec3 = (0.0, 0.0, 0.0)
    scale: Vec3 = (1.0, 1.0, 1.0)

    @field_validator("scale")
    @classmethod
    def _validate_nonzero_scale(cls, v: Vec3) -> Vec3:
        if any(abs(s) < 1e-8 for s in v):
            raise ValueError(f"Scale components cannot be zero: {v}")
        return v


class BoundingBox(BaseModel):
    """Axis-aligned bounding box in world or local space."""

    min_corner: Vec3
    max_corner: Vec3

    @property
    def center(self) -> Vec3:
        return (
            0.5 * (self.min_corner[0] + self.max_corner[0]),
            0.5 * (self.min_corner[1] + self.max_corner[1]),
            0.5 * (self.min_corner[2] + self.max_corner[2]),
        )

    @property
    def size(self) -> Vec3:
        return (
            max(0.0, self.max_corner[0] - self.min_corner[0]),
            max(0.0, self.max_corner[1] - self.min_corner[1]),
            max(0.0, self.max_corner[2] - self.min_corner[2]),
        )

    def overlaps(self, other: BoundingBox, tolerance: float = 0.0) -> bool:
        """Check whether two AABBs overlap beyond optional tolerance."""
        return (
            self.min_corner[0] <= other.max_corner[0] - tolerance
            and self.max_corner[0] >= other.min_corner[0] + tolerance
            and self.min_corner[1] <= other.max_corner[1] - tolerance
            and self.max_corner[1] >= other.min_corner[1] + tolerance
            and self.min_corner[2] <= other.max_corner[2] - tolerance
            and self.max_corner[2] >= other.min_corner[2] + tolerance
        )


class SelectionQuery(BaseModel):
    """Semantic geometry selection descriptor for AI-friendly mesh editing."""

    mode: Literal[
        "all",
        "top",
        "bottom",
        "front",
        "back",
        "left",
        "right",
        "normal_dir",
        "bbox",
        "indices",
        "vertex_indices",
        "edge_indices",
        "coplanar",
        "boundary",
        "material",
    ] = "all"
    direction: Vec3 | None = Field(
        default=None,
        description="Target unit normal direction when mode='normal_dir' or 'coplanar'.",
    )
    angle_threshold_deg: float = Field(
        default=45.0,
        ge=0.5,
        le=180.0,
        description="Maximum angle deviation in degrees for directional/coplanar selection.",
    )
    bbox_min: Vec3 | None = None
    bbox_max: Vec3 | None = None
    indices: list[int] | None = None
    vertex_indices: list[int] | None = None
    edge_indices: list[tuple[int, int]] | None = None
    seed_face: int | None = None
    material_id: str | None = None
    invert: bool = False


class MeshPayload(BaseModel):
    """JSON-serializable triangle mesh data stored on a SceneObject."""

    vertices: list[list[float]] = Field(default_factory=list)
    faces: list[list[int]] = Field(default_factory=list)
    uvs: list[list[float]] | None = None
    face_groups: list[int] | None = None
    vertex_colors: list[list[float]] | None = None
