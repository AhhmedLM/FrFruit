"""Procedural Realistic Edge-Looped Quad Base Mesh, Hierarchical Morph Targets & Delta Engine."""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING, Any

from frfruit.character.schema import (
    CHARACTER_PRESETS,
    SEMANTIC_RBF_ANCHORS,
    SKIN_TONE_PRESETS,
    BodySliders,
    CharacterDelta,
    CharacterMaterialParams,
    CharacterQuality,
    CharacterSpec,
    ClothingStyle,
    CustomMorphTarget,
    DeltaRegion,
    ExpressionSliders,
    FaceSliders,
    HairStyle,
    RBFDeltaNode,
    get_character_preset,
)

if TYPE_CHECKING:
    from frfruit.character.builder import (
        build_human_armature,
        build_or_update_character,
        landmarks_summary,
    )
    from frfruit.character.clothing import build_bottom_mesh, build_shoes_mesh, build_top_mesh
    from frfruit.character.delta import (
        add_rbf_sculpt_node,
        apply_character_delta,
        blend_characters,
        compute_character_delta,
        extract_custom_morph_from_mesh_edit,
        extract_rbf_delta_from_mesh_edit,
    )
    from frfruit.character.hair import build_hair_mesh
    from frfruit.character.morphs import (
        MorphTargetBank,
        build_character_skin_mesh,
        evaluate_base_cage,
        get_morph_target_bank,
    )
    from frfruit.character.parts import build_brows_and_lashes_mesh, build_eyeballs_mesh
    from frfruit.character.quad_base import (
        CanonicalQuadBase,
        build_canonical_quad_base,
        subdivide_quad_mesh,
    )
    from frfruit.character.surface import (
        AnatomicalLandmarks,
        CharacterSurface,
        compute_landmarks,
        compute_landmarks_from_verts,
    )

_LAZY_IMPORTS: dict[str, tuple[str, str]] = {
    "AnatomicalLandmarks": ("frfruit.character.surface", "AnatomicalLandmarks"),
    "CharacterSurface": ("frfruit.character.surface", "CharacterSurface"),
    "compute_landmarks": ("frfruit.character.surface", "compute_landmarks"),
    "compute_landmarks_from_verts": ("frfruit.character.surface", "compute_landmarks_from_verts"),
    "CanonicalQuadBase": ("frfruit.character.quad_base", "CanonicalQuadBase"),
    "build_canonical_quad_base": ("frfruit.character.quad_base", "build_canonical_quad_base"),
    "subdivide_quad_mesh": ("frfruit.character.quad_base", "subdivide_quad_mesh"),
    "MorphTargetBank": ("frfruit.character.morphs", "MorphTargetBank"),
    "get_morph_target_bank": ("frfruit.character.morphs", "get_morph_target_bank"),
    "evaluate_base_cage": ("frfruit.character.morphs", "evaluate_base_cage"),
    "build_character_skin_mesh": ("frfruit.character.morphs", "build_character_skin_mesh"),
    "build_human_armature": ("frfruit.character.builder", "build_human_armature"),
    "build_or_update_character": ("frfruit.character.builder", "build_or_update_character"),
    "landmarks_summary": ("frfruit.character.builder", "landmarks_summary"),
    "build_top_mesh": ("frfruit.character.clothing", "build_top_mesh"),
    "build_bottom_mesh": ("frfruit.character.clothing", "build_bottom_mesh"),
    "build_shoes_mesh": ("frfruit.character.clothing", "build_shoes_mesh"),
    "build_hair_mesh": ("frfruit.character.hair", "build_hair_mesh"),
    "build_eyeballs_mesh": ("frfruit.character.parts", "build_eyeballs_mesh"),
    "build_brows_and_lashes_mesh": ("frfruit.character.parts", "build_brows_and_lashes_mesh"),
    "add_rbf_sculpt_node": ("frfruit.character.delta", "add_rbf_sculpt_node"),
    "apply_character_delta": ("frfruit.character.delta", "apply_character_delta"),
    "blend_characters": ("frfruit.character.delta", "blend_characters"),
    "compute_character_delta": ("frfruit.character.delta", "compute_character_delta"),
    "extract_custom_morph_from_mesh_edit": (
        "frfruit.character.delta",
        "extract_custom_morph_from_mesh_edit",
    ),
    "extract_rbf_delta_from_mesh_edit": (
        "frfruit.character.delta",
        "extract_rbf_delta_from_mesh_edit",
    ),
}


def __getattr__(name: str) -> Any:
    target = _LAZY_IMPORTS.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    mod_name, attr_name = target
    val = getattr(import_module(mod_name), attr_name)
    globals()[name] = val
    return val


def __dir__() -> list[str]:
    return sorted(__all__)


__all__ = [
    "AnatomicalLandmarks",
    "BodySliders",
    "CHARACTER_PRESETS",
    "CanonicalQuadBase",
    "CharacterDelta",
    "CharacterMaterialParams",
    "CharacterQuality",
    "CharacterSpec",
    "CharacterSurface",
    "ClothingStyle",
    "CustomMorphTarget",
    "DeltaRegion",
    "ExpressionSliders",
    "FaceSliders",
    "HairStyle",
    "MorphTargetBank",
    "RBFDeltaNode",
    "SEMANTIC_RBF_ANCHORS",
    "SKIN_TONE_PRESETS",
    "add_rbf_sculpt_node",
    "apply_character_delta",
    "blend_characters",
    "build_bottom_mesh",
    "build_brows_and_lashes_mesh",
    "build_canonical_quad_base",
    "build_character_skin_mesh",
    "build_eyeballs_mesh",
    "build_hair_mesh",
    "build_human_armature",
    "build_or_update_character",
    "build_shoes_mesh",
    "build_top_mesh",
    "compute_character_delta",
    "compute_landmarks",
    "compute_landmarks_from_verts",
    "evaluate_base_cage",
    "extract_custom_morph_from_mesh_edit",
    "extract_rbf_delta_from_mesh_edit",
    "get_character_preset",
    "get_morph_target_bank",
    "landmarks_summary",
    "subdivide_quad_mesh",
]
