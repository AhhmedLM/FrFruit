"""Complete JSON-serializable SceneDocument IR for FrFruit."""

from __future__ import annotations

import uuid
from typing import Annotated, Literal

from pydantic import BaseModel, Field

from frfruit.animation.timeline import TimelineState
from frfruit.core.types import ColorRGB, ColorRGBA, MeshPayload, Transform, Vec3
from frfruit.procedural.schema import ProceduralGraph

# --- Modifiers (Discriminated Union) ---


class MirrorModifier(BaseModel):
    type: Literal["mirror"] = "mirror"
    name: str = "Mirror"
    enabled: bool = True
    axis: Literal["x", "y", "z"] = "x"
    bisect: bool = False
    offset: float = 0.0


class ArrayModifier(BaseModel):
    type: Literal["array"] = "array"
    name: str = "Array"
    enabled: bool = True
    mode: Literal["linear", "radial"] = "linear"
    count: int = Field(default=3, ge=1, le=200)
    offset: Vec3 = (1.5, 0.0, 0.0)
    radial_axis: Literal["x", "y", "z"] = "z"
    radial_angle_deg: float = 360.0
    radial_center: Vec3 = (0.0, 0.0, 0.0)


class SolidifyModifier(BaseModel):
    type: Literal["solidify"] = "solidify"
    name: str = "Solidify"
    enabled: bool = True
    thickness: float = Field(default=0.05, gt=0.0)
    offset: float = 0.0


class SubdivisionModifier(BaseModel):
    type: Literal["subdivision"] = "subdivision"
    name: str = "Subdivision"
    enabled: bool = True
    levels: int = Field(default=1, ge=1, le=3)
    smooth: bool = True


class BevelModifier(BaseModel):
    type: Literal["bevel"] = "bevel"
    name: str = "Bevel"
    enabled: bool = True
    width: float = Field(default=0.04, gt=0.0)
    segments: int = Field(default=1, ge=1, le=4)
    angle_threshold_deg: float = Field(default=35.0, ge=1.0, le=179.0)


class BooleanModifier(BaseModel):
    type: Literal["boolean"] = "boolean"
    name: str = "Boolean"
    enabled: bool = True
    operation: Literal["union", "difference", "intersection"] = "difference"
    target_object_id: str


class TaperModifier(BaseModel):
    type: Literal["taper"] = "taper"
    name: str = "Taper"
    enabled: bool = True
    axis: Literal["x", "y", "z"] = "z"
    factor: float = 0.5


class TwistModifier(BaseModel):
    type: Literal["twist"] = "twist"
    name: str = "Twist"
    enabled: bool = True
    axis: Literal["x", "y", "z"] = "z"
    angle_deg: float = 90.0


class BendModifier(BaseModel):
    type: Literal["bend"] = "bend"
    name: str = "Bend"
    enabled: bool = True
    axis: Literal["x", "y", "z"] = "x"
    bend_axis: Literal["x", "y", "z"] = "z"
    angle_deg: float = 45.0


class DisplaceModifier(BaseModel):
    type: Literal["displace"] = "displace"
    name: str = "Displace"
    enabled: bool = True
    strength: float = 0.1
    scale: float = 2.0
    seed: int = 42


class SmoothModifier(BaseModel):
    type: Literal["smooth"] = "smooth"
    name: str = "Smooth"
    enabled: bool = True
    iterations: int = Field(default=2, ge=1, le=25)
    factor: float = Field(default=0.5, ge=0.0, le=1.0)


ModifierSpec = Annotated[
    MirrorModifier
    | ArrayModifier
    | SolidifyModifier
    | SubdivisionModifier
    | BevelModifier
    | BooleanModifier
    | TaperModifier
    | TwistModifier
    | BendModifier
    | DisplaceModifier
    | SmoothModifier,
    Field(discriminator="type"),
]


# --- Scene Entities ---


class MaterialSpec(BaseModel):
    """PBR + optional procedural material definition."""

    id: str
    name: str
    base_color: ColorRGBA = (0.75, 0.75, 0.78, 1.0)
    metallic: float = Field(default=0.0, ge=0.0, le=1.0)
    roughness: float = Field(default=0.45, ge=0.0, le=1.0)
    emission_color: ColorRGBA = (0.0, 0.0, 0.0, 1.0)
    emission_strength: float = Field(default=0.0, ge=0.0)
    alpha: float = Field(default=1.0, ge=0.0, le=1.0)
    ior: float = Field(default=1.45, ge=1.0, le=3.0)
    pattern: Literal["solid", "checker", "gradient", "wood", "noise"] = "solid"
    secondary_color: ColorRGBA = (0.2, 0.2, 0.25, 1.0)
    pattern_scale: float = Field(default=4.0, gt=0.0)


class LightSpec(BaseModel):
    """Scene illumination source."""

    kind: Literal["sun", "point", "spot", "area"] = "sun"
    color: ColorRGB = (1.0, 0.98, 0.94)
    intensity: float = Field(default=2.2, ge=0.0)
    direction: Vec3 | None = None
    spot_angle_deg: float = Field(default=45.0, ge=1.0, le=175.0)
    size: float = Field(default=1.0, gt=0.0)
    cast_shadows: bool = True


class CameraSpec(BaseModel):
    """Scene camera using explicit target look-at."""

    projection: Literal["perspective", "orthographic"] = "perspective"
    fov_deg: float = Field(default=45.0, ge=5.0, le=150.0)
    ortho_scale: float = Field(default=5.0, gt=0.0)
    near: float = Field(default=0.05, gt=0.0)
    far: float = Field(default=250.0, gt=0.1)
    target: Vec3 = (0.0, 0.0, 0.0)
    up: Vec3 = (0.0, 0.0, 1.0)


class WorldEnvironment(BaseModel):
    """World sky/background and global ambient illumination."""

    background_color: ColorRGB = (0.14, 0.16, 0.20)
    horizon_color: ColorRGB = (0.22, 0.25, 0.30)
    zenith_color: ColorRGB = (0.09, 0.12, 0.18)
    ambient_strength: float = Field(default=0.32, ge=0.0, le=2.0)


class CollectionSpec(BaseModel):
    """Logical grouping of SceneObjects (Blender Collection equivalent)."""

    id: str
    name: str
    parent_id: str | None = None
    visible: bool = True


class SceneObject(BaseModel):
    """Any object in the 3D scene tree (mesh, procedural, light, camera, empty)."""

    id: str
    name: str
    kind: Literal["mesh", "procedural", "light", "camera", "empty"] = "mesh"
    transform: Transform = Field(default_factory=Transform)
    parent_id: str | None = None
    collection_id: str = "default"
    visible: bool = True
    tags: list[str] = Field(default_factory=list)
    material_id: str | None = None
    face_material_ids: dict[int, str] = Field(default_factory=dict)
    mesh: MeshPayload | None = None
    modifiers: list[ModifierSpec] = Field(default_factory=list)
    node_graph: ProceduralGraph | None = None
    light: LightSpec | None = None
    camera: CameraSpec | None = None


class SceneDocument(BaseModel):
    """Top-level serializable FrFruit 3D project document (.frf.json)."""

    schema_version: str = "1.0"
    name: str = "UntitledScene"
    up_axis: Literal["Z", "Y"] = "Z"
    unit_system: Literal["meters", "centimeters", "millimeters"] = "meters"
    active_camera_id: str | None = None
    world: WorldEnvironment = Field(default_factory=WorldEnvironment)
    collections: dict[str, CollectionSpec] = Field(default_factory=dict)
    materials: dict[str, MaterialSpec] = Field(default_factory=dict)
    objects: dict[str, SceneObject] = Field(default_factory=dict)
    timeline: TimelineState = Field(default_factory=TimelineState)

    @classmethod
    def new_default(cls, name: str = "UntitledScene") -> SceneDocument:
        """Create a clean scene pre-populated with default collection, lighting, and camera."""
        doc = cls(name=name)
        doc.collections["default"] = CollectionSpec(id="default", name="Scene Collection")
        doc.materials["mat_default"] = MaterialSpec(
            id="mat_default",
            name="Default Clay",
            base_color=(0.78, 0.78, 0.80, 1.0),
            metallic=0.0,
            roughness=0.5,
        )
        doc.objects["light_key"] = SceneObject(
            id="light_key",
            name="KeySun",
            kind="light",
            transform=Transform(location=(5.0, -5.0, 8.0)),
            light=LightSpec(
                kind="sun",
                color=(1.0, 0.97, 0.92),
                intensity=2.4,
                direction=(-0.55, 0.45, -0.70),
                cast_shadows=True,
            ),
        )
        doc.objects["light_fill"] = SceneObject(
            id="light_fill",
            name="FillLight",
            kind="light",
            transform=Transform(location=(-4.0, -2.0, 3.5)),
            light=LightSpec(
                kind="point",
                color=(0.75, 0.86, 1.0),
                intensity=1.2,
                cast_shadows=False,
            ),
        )
        doc.objects["cam_main"] = SceneObject(
            id="cam_main",
            name="MainCamera",
            kind="camera",
            transform=Transform(location=(4.0, -4.0, 3.0)),
            camera=CameraSpec(
                projection="perspective",
                fov_deg=45.0,
                target=(0.0, 0.0, 0.5),
            ),
        )
        doc.active_camera_id = "cam_main"
        return doc

    def resolve_object(self, id_or_name: str) -> SceneObject | None:
        """Resolve an object by exact id or unique/first matching name."""
        if id_or_name in self.objects:
            return self.objects[id_or_name]
        for obj in self.objects.values():
            if obj.name == id_or_name:
                return obj
        return None

    def resolve_material(self, id_or_name: str) -> MaterialSpec | None:
        """Resolve a material by exact id or name."""
        if id_or_name in self.materials:
            return self.materials[id_or_name]
        for mat in self.materials.values():
            if mat.name == id_or_name:
                return mat
        return None

    def next_id(self, prefix: str = "obj") -> str:
        """Generate a short deterministic-friendly unique ID."""
        clean = "".join(c if c.isalnum() or c == "_" else "_" for c in prefix).strip("_") or "item"
        if (
            clean not in self.objects
            and clean not in self.materials
            and clean not in self.collections
        ):
            return clean
        for i in range(1, 10000):
            cand = f"{clean}_{i}"
            if (
                cand not in self.objects
                and cand not in self.materials
                and cand not in self.collections
            ):
                return cand
        return f"{clean}_{uuid.uuid4().hex[:6]}"

    def unique_name(self, base_name: str) -> str:
        """Ensure an object name is unique inside the scene."""
        existing = {o.name for o in self.objects.values()}
        if base_name not in existing:
            return base_name
        idx = 1
        while f"{base_name}.{idx:03d}" in existing:
            idx += 1
        return f"{base_name}.{idx:03d}"

    def is_object_visible(self, obj: SceneObject) -> bool:
        """Check visibility taking parent collection hierarchy into account."""
        if not obj.visible:
            return False
        cid: str | None = obj.collection_id
        visited: set[str] = set()
        while cid and cid in self.collections and cid not in visited:
            visited.add(cid)
            col = self.collections[cid]
            if not col.visible:
                return False
            cid = col.parent_id
        return True
