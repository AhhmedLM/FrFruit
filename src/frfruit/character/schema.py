"""Pydantic schemas and presets for realistic procedural Quad + Morph human characters."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

ColorRGB = tuple[float, float, float]
Vec3 = tuple[float, float, float]

CharacterQuality = Literal["preview", "medium", "high"]
CharacterPose = Literal["a_pose", "t_pose", "relaxed"]
HairStyle = Literal["none", "short", "swept", "bob", "long_wavy"]
ClothingStyle = Literal["none", "casual", "athletic", "tunic"]
DeltaRegion = Literal[
    "all",
    "body",
    "face",
    "expressions",
    "upper_face",
    "eyes",
    "nose",
    "mouth",
    "jaw",
    "ears",
]


class BodySliders(BaseModel):
    """Continuous realistic body anthropometry sliders driving Quad Base Morph Targets."""

    gender: float = Field(
        default=0.0,
        ge=-1.0,
        le=1.0,
        description="Sexual dimorphism spectrum: -1.0 = feminine, 0.0 = neutral, +1.0 = masculine.",
    )
    height: float = Field(default=1.74, ge=1.45, le=2.10, description="Total stature in meters.")
    muscularity: float = Field(default=0.38, ge=0.0, le=1.0)
    body_fat: float = Field(default=0.22, ge=0.0, le=1.0)
    age: float = Field(default=0.28, ge=0.0, le=1.0)
    head_scale: float = Field(default=1.0, ge=0.85, le=1.18)
    shoulder_width: float = Field(default=1.0, ge=0.78, le=1.30)
    hip_width: float = Field(default=1.0, ge=0.78, le=1.30)
    limb_thickness: float = Field(default=1.0, ge=0.75, le=1.35)


class FaceSliders(BaseModel):
    """Bidirectional realistic facial anthropometry sliders (0.0 = balanced canonical)."""

    # Cranium & overall silhouette
    face_width: float = Field(default=0.0, ge=-1.0, le=1.0)
    face_length: float = Field(default=0.0, ge=-1.0, le=1.0)
    forehead_height: float = Field(default=0.0, ge=-1.0, le=1.0)
    brow_ridge: float = Field(default=0.0, ge=-1.0, le=1.0)
    cheekbone_prominence: float = Field(default=0.0, ge=-1.0, le=1.0)
    cheek_fullness: float = Field(default=0.0, ge=-1.0, le=1.0)
    jaw_width: float = Field(default=0.0, ge=-1.0, le=1.0)
    jaw_angle: float = Field(default=0.0, ge=-1.0, le=1.0)
    chin_prominence: float = Field(default=0.0, ge=-1.0, le=1.0)
    chin_width: float = Field(default=0.0, ge=-1.0, le=1.0)

    # Eyes & brows
    eye_size: float = Field(default=0.0, ge=-1.0, le=1.0)
    eye_spacing: float = Field(default=0.0, ge=-1.0, le=1.0)
    eye_height: float = Field(default=0.0, ge=-1.0, le=1.0)
    eye_depth: float = Field(default=0.0, ge=-1.0, le=1.0)
    eye_tilt: float = Field(default=0.0, ge=-1.0, le=1.0)
    brow_arch: float = Field(default=0.0, ge=-1.0, le=1.0)

    # Nose
    nose_length: float = Field(default=0.0, ge=-1.0, le=1.0)
    nose_width: float = Field(default=0.0, ge=-1.0, le=1.0)
    nose_bridge: float = Field(default=0.0, ge=-1.0, le=1.0)
    nose_tip_projection: float = Field(default=0.0, ge=-1.0, le=1.0)
    nose_tip_tilt: float = Field(default=0.0, ge=-1.0, le=1.0)

    # Mouth & lips
    mouth_width: float = Field(default=0.0, ge=-1.0, le=1.0)
    upper_lip_fullness: float = Field(default=0.0, ge=-1.0, le=1.0)
    lower_lip_fullness: float = Field(default=0.0, ge=-1.0, le=1.0)
    mouth_height: float = Field(default=0.0, ge=-1.0, le=1.0)
    smile: float = Field(default=0.0, ge=-1.0, le=1.0)

    # Ears & micro-realism
    ear_size: float = Field(default=0.0, ge=-1.0, le=1.0)
    ear_protrusion: float = Field(default=0.0, ge=-1.0, le=1.0)
    asymmetry: float = Field(default=0.14, ge=0.0, le=1.0)
    pore_detail: float = Field(default=0.65, ge=0.0, le=1.0)


class ExpressionSliders(BaseModel):
    """Facial expression shape keys evaluated cleanly on the facial edge-loops."""

    smile: float = Field(
        default=0.0,
        ge=-1.0,
        le=1.0,
        description="Mouth corner expression (-1.0 = frown, +1.0 = warm smile).",
    )
    blink_left: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Left upper eyelid closure (0.0 = open, 1.0 = closed).",
    )
    blink_right: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Right upper eyelid closure (0.0 = open, 1.0 = closed).",
    )
    brow_raise: float = Field(
        default=0.0,
        ge=-1.0,
        le=1.0,
        description="Eyebrow expression (-1.0 = furrowed, +1.0 = raised).",
    )
    jaw_open: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Mandible & lower lip opening (0.0 = closed, 1.0 = open).",
    )


class RBFDeltaNode(BaseModel):
    """Compactly-supported Wendland C2 RBF sculpting / delta deformation node.

    `center` is specified in normalized head-relative coordinates when `space='head_norm'`
    (origin at head center, unit = head half-height ~0.116m) or in character-local meters
    when `space='local'`.
    """

    id: str
    region: str = "custom"
    space: Literal["head_norm", "local"] = "head_norm"
    center: Vec3 = (0.0, -0.7, 0.0)
    radius: float = Field(default=0.35, gt=0.01, le=2.0)
    weight: float = Field(
        default=0.0,
        ge=-0.5,
        le=0.5,
        description=(
            "Normal-axis sculpt offset (negative inflates/protrudes outward, "
            "positive pushes inward)."
        ),
    )
    vector_delta: Vec3 = Field(
        default=(0.0, 0.0, 0.0),
        description="Directional displacement vector in target space.",
    )
    mirror_x: bool = Field(
        default=False,
        description="If true, symmetrically mirror this node across X=0.",
    )


class CustomMorphTarget(BaseModel):
    """Named user-defined custom morph target stored as sparse base-cage vertex deltas."""

    name: str
    region: DeltaRegion = "all"
    deltas: list[tuple[int, Vec3]] = Field(
        default_factory=list,
        description="Sparse list of (base_vertex_index, (dx, dy, dz)) in meters.",
    )


class CharacterMaterialParams(BaseModel):
    """Realistic skin, lip, iris, hair, and procedural clothing parameters."""

    skin_tone: ColorRGB = (0.78, 0.58, 0.46)
    lip_tint: ColorRGB = (0.64, 0.34, 0.31)
    eye_color: ColorRGB = (0.24, 0.42, 0.56)
    hair_color: ColorRGB = (0.13, 0.09, 0.06)
    hair_style: HairStyle = "short"
    hair_length: float = Field(default=1.0, ge=0.35, le=2.0)
    hair_volume: float = Field(default=1.0, ge=0.40, le=1.8)
    hair_curl: float = Field(default=0.15, ge=0.0, le=1.0)
    hair_part: float = Field(default=0.25, ge=-1.0, le=1.0)
    clothing_style: ClothingStyle = "casual"
    top_color: ColorRGB = (0.20, 0.32, 0.46)
    bottom_color: ColorRGB = (0.14, 0.17, 0.23)
    subsurface_strength: float = Field(default=0.55, ge=0.0, le=1.0)
    roughness_base: float = Field(default=0.64, ge=0.10, le=0.95)


class CharacterDelta(BaseModel):
    """Stored mathematical difference (Delta / الفارق) between two character configurations."""

    name: str
    region: DeltaRegion = "all"
    source_a: str = "base"
    source_b: str = "target"
    body_delta: dict[str, float] = Field(default_factory=dict)
    face_delta: dict[str, float] = Field(default_factory=dict)
    expression_delta: dict[str, float] = Field(default_factory=dict)
    custom_morph_delta: dict[str, float] = Field(default_factory=dict)
    material_delta: dict[str, float | list[float]] = Field(default_factory=dict)
    rbf_nodes: list[RBFDeltaNode] = Field(default_factory=list)


class CharacterSpec(BaseModel):
    """Complete realistic Quad-Base + Morph-Target human character specification."""

    version: str = "2.0"
    preset: str = "realistic_neutral"
    quality: CharacterQuality = "medium"
    pose: CharacterPose = "a_pose"
    auto_rig: bool = True
    body: BodySliders = Field(default_factory=BodySliders)
    face: FaceSliders = Field(default_factory=FaceSliders)
    expressions: ExpressionSliders = Field(default_factory=ExpressionSliders)
    materials: CharacterMaterialParams = Field(default_factory=CharacterMaterialParams)
    custom_morphs: dict[str, float] = Field(
        default_factory=dict,
        description="Active weight map {morph_name: weight} for stored custom morph targets.",
    )
    stored_morphs: dict[str, CustomMorphTarget] = Field(
        default_factory=dict,
        description="Library of named custom morph targets attached to this character.",
    )
    rbf_nodes: list[RBFDeltaNode] = Field(default_factory=list)
    stored_deltas: dict[str, CharacterDelta] = Field(default_factory=dict)


# Semantic region mapping used for both regional delta transfer & face field queries
FACE_REGION_FIELDS: dict[DeltaRegion, set[str]] = {
    "all": (
        set(BodySliders.model_fields)
        | set(FaceSliders.model_fields)
        | set(ExpressionSliders.model_fields)
    ),
    "body": set(BodySliders.model_fields),
    "face": set(FaceSliders.model_fields),
    "expressions": set(ExpressionSliders.model_fields),
    "upper_face": {
        "face_width",
        "forehead_height",
        "brow_ridge",
        "cheekbone_prominence",
        "eye_size",
        "eye_spacing",
        "eye_height",
        "eye_depth",
        "eye_tilt",
        "brow_arch",
    },
    "eyes": {
        "eye_size",
        "eye_spacing",
        "eye_height",
        "eye_depth",
        "eye_tilt",
        "brow_arch",
        "brow_ridge",
        "blink_left",
        "blink_right",
        "brow_raise",
    },
    "nose": {
        "nose_length",
        "nose_width",
        "nose_bridge",
        "nose_tip_projection",
        "nose_tip_tilt",
    },
    "mouth": {
        "mouth_width",
        "upper_lip_fullness",
        "lower_lip_fullness",
        "mouth_height",
        "smile",
        "jaw_open",
    },
    "jaw": {
        "jaw_width",
        "jaw_angle",
        "chin_prominence",
        "chin_width",
        "cheek_fullness",
        "cheekbone_prominence",
        "jaw_open",
    },
    "ears": {
        "ear_size",
        "ear_protrusion",
    },
}

# Canonical normalized head centers (unit = head half-height) for quick semantic RBF sculpting
SEMANTIC_RBF_ANCHORS: dict[str, tuple[Vec3, float, bool]] = {
    "nose_tip": ((0.0, -0.96, -0.10), 0.24, False),
    "nose_bridge": ((0.0, -0.84, 0.08), 0.24, False),
    "cheekbone": ((0.44, -0.64, -0.06), 0.34, True),
    "cheek": ((0.38, -0.68, -0.26), 0.34, True),
    "jaw_corner": ((0.46, -0.24, -0.56), 0.36, True),
    "chin": ((0.0, -0.68, -0.80), 0.30, False),
    "brow": ((0.24, -0.78, 0.28), 0.30, True),
    "forehead": ((0.0, -0.70, 0.56), 0.48, False),
    "upper_lip": ((0.0, -0.78, -0.36), 0.22, False),
    "lower_lip": ((0.0, -0.76, -0.47), 0.22, False),
    "ear": ((0.64, 0.05, -0.05), 0.28, True),
}

SKIN_TONE_PRESETS: dict[str, ColorRGB] = {
    "fair": (0.87, 0.69, 0.58),
    "medium_warm": (0.78, 0.57, 0.45),
    "olive": (0.72, 0.54, 0.39),
    "tan": (0.61, 0.42, 0.30),
    "brown": (0.43, 0.27, 0.19),
    "deep": (0.27, 0.17, 0.12),
}

CHARACTER_PRESETS: dict[str, CharacterSpec] = {
    "realistic_neutral": CharacterSpec(
        preset="realistic_neutral",
        body=BodySliders(gender=0.0, height=1.74, muscularity=0.38, body_fat=0.22),
        face=FaceSliders(asymmetry=0.14, pore_detail=0.70),
        materials=CharacterMaterialParams(
            skin_tone=(0.78, 0.57, 0.45),
            eye_color=(0.25, 0.44, 0.56),
            hair_color=(0.14, 0.09, 0.06),
            hair_style="short",
        ),
    ),
    "realistic_male": CharacterSpec(
        preset="realistic_male",
        body=BodySliders(
            gender=0.80,
            height=1.79,
            muscularity=0.48,
            body_fat=0.18,
            shoulder_width=1.03,
        ),
        face=FaceSliders(
            jaw_width=0.22,
            jaw_angle=0.30,
            chin_prominence=0.18,
            chin_width=0.16,
            brow_ridge=0.28,
            cheekbone_prominence=0.24,
            nose_bridge=0.14,
            nose_width=0.06,
            upper_lip_fullness=-0.06,
            asymmetry=0.12,
            pore_detail=0.68,
        ),
        materials=CharacterMaterialParams(
            skin_tone=(0.76, 0.56, 0.44),
            lip_tint=(0.58, 0.34, 0.31),
            eye_color=(0.24, 0.40, 0.31),
            hair_color=(0.11, 0.08, 0.06),
            hair_style="short",
            hair_length=1.0,
            hair_volume=1.05,
            hair_curl=0.12,
            hair_part=0.30,
            clothing_style="casual",
            top_color=(0.22, 0.29, 0.37),
            bottom_color=(0.14, 0.16, 0.22),
        ),
    ),
    "realistic_female": CharacterSpec(
        preset="realistic_female",
        body=BodySliders(
            gender=-0.80,
            height=1.67,
            muscularity=0.20,
            body_fat=0.22,
            shoulder_width=0.95,
            hip_width=1.02,
        ),
        face=FaceSliders(
            face_width=-0.06,
            jaw_width=-0.24,
            jaw_angle=-0.24,
            chin_width=-0.18,
            cheekbone_prominence=0.26,
            brow_ridge=-0.22,
            brow_arch=0.28,
            eye_size=0.14,
            eye_tilt=0.16,
            nose_width=-0.16,
            nose_tip_tilt=0.12,
            upper_lip_fullness=0.18,
            lower_lip_fullness=0.22,
            asymmetry=0.10,
            pore_detail=0.52,
        ),
        expressions=ExpressionSliders(smile=0.12),
        materials=CharacterMaterialParams(
            skin_tone=(0.82, 0.63, 0.52),
            lip_tint=(0.68, 0.34, 0.35),
            eye_color=(0.22, 0.44, 0.62),
            hair_color=(0.18, 0.10, 0.06),
            hair_style="long_wavy",
            hair_length=1.05,
            hair_volume=1.10,
            hair_curl=0.28,
            hair_part=0.22,
            clothing_style="casual",
            top_color=(0.50, 0.23, 0.28),
            bottom_color=(0.14, 0.16, 0.21),
        ),
    ),
    "realistic_athletic": CharacterSpec(
        preset="realistic_athletic",
        body=BodySliders(
            gender=0.55,
            height=1.83,
            muscularity=0.82,
            body_fat=0.10,
            shoulder_width=1.12,
            hip_width=0.95,
            limb_thickness=1.08,
        ),
        face=FaceSliders(
            cheekbone_prominence=0.38,
            cheek_fullness=-0.22,
            jaw_width=0.26,
            jaw_angle=0.38,
            chin_prominence=0.22,
            brow_ridge=0.24,
            nose_bridge=0.18,
            asymmetry=0.10,
            pore_detail=0.62,
        ),
        expressions=ExpressionSliders(smile=0.06),
        materials=CharacterMaterialParams(
            skin_tone=(0.68, 0.48, 0.36),
            lip_tint=(0.56, 0.33, 0.30),
            eye_color=(0.28, 0.22, 0.16),
            hair_color=(0.09, 0.07, 0.05),
            hair_style="swept",
            hair_length=0.95,
            hair_volume=1.05,
            hair_curl=0.15,
            hair_part=0.35,
            clothing_style="athletic",
            top_color=(0.17, 0.37, 0.54),
            bottom_color=(0.13, 0.15, 0.20),
        ),
    ),
    "elder_portrait": CharacterSpec(
        preset="elder_portrait",
        body=BodySliders(
            gender=0.60,
            height=1.73,
            muscularity=0.25,
            body_fat=0.20,
            age=0.88,
        ),
        face=FaceSliders(
            face_length=0.18,
            cheekbone_prominence=0.30,
            cheek_fullness=-0.35,
            brow_ridge=0.45,
            eye_depth=0.35,
            nose_length=0.22,
            nose_bridge=0.30,
            ear_size=0.25,
            asymmetry=0.22,
            pore_detail=0.92,
        ),
        materials=CharacterMaterialParams(
            skin_tone=(0.77, 0.58, 0.48),
            eye_color=(0.35, 0.45, 0.48),
            hair_color=(0.68, 0.68, 0.70),
            hair_style="short",
            clothing_style="tunic",
            top_color=(0.32, 0.25, 0.20),
            bottom_color=(0.15, 0.14, 0.13),
        ),
    ),
}


def get_character_preset(name: str) -> CharacterSpec:
    """Return a deep copy of a named built-in realistic character preset."""
    key = name.strip().lower()
    if key not in CHARACTER_PRESETS:
        raise ValueError(
            f"Unknown character preset {name!r}. Available: {sorted(CHARACTER_PRESETS.keys())}"
        )
    return CHARACTER_PRESETS[key].model_copy(deep=True)
