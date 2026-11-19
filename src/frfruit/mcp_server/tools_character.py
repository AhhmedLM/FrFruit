"""MCP tools for realistic Quad-Morph human characters, expressions, custom morphs, and blending."""

from __future__ import annotations

from collections.abc import Callable
from typing import Annotated, Any, Literal

from mcp.server import MCPServer
from mcp.server.mcpserver.exceptions import ToolError
from mcp.types import ToolAnnotations
from pydantic import Field, ValidationError

from frfruit.character import (
    CHARACTER_PRESETS,
    SEMANTIC_RBF_ANCHORS,
    SKIN_TONE_PRESETS,
    BodySliders,
    CharacterMaterialParams,
    CharacterQuality,
    CharacterSpec,
    CharacterSurface,
    DeltaRegion,
    ExpressionSliders,
    FaceSliders,
    add_rbf_sculpt_node,
    apply_character_delta,
    blend_characters,
    build_or_update_character,
    compute_character_delta,
    compute_landmarks,
    extract_custom_morph_from_mesh_edit,
    extract_rbf_delta_from_mesh_edit,
    get_character_preset,
    landmarks_summary,
)
from frfruit.core.scene import SceneObject
from frfruit.core.types import Vec3
from frfruit.geometry.mesh import MeshData
from frfruit.mcp_server.context import FrFruitSession


def _require_character_object(sess: FrFruitSession, object_id: str) -> SceneObject:
    obj = sess.require_object(object_id)
    if obj.character is None:
        raise ToolError(
            f"Object {obj.name!r} ({obj.id}) is not a procedural Quad-Morph character. "
            "Create one first with character_manage(action='create')."
        )
    return obj


def _resolve_source_spec(
    sess: FrFruitSession,
    source_id: str | None,
    preset: str | None,
) -> tuple[CharacterSpec, str]:
    if source_id:
        obj = sess.scene.resolve_object(source_id)
        if obj is not None and obj.character is not None:
            return obj.character, obj.name
        if source_id.strip().lower() in CHARACTER_PRESETS:
            return get_character_preset(source_id), source_id
        raise ToolError(f"Source character {source_id!r} not found.")
    if preset:
        return get_character_preset(preset), preset
    raise ToolError("Provide either source_character_id or preset.")


def register_character_tools(
    mcp: MCPServer,
    get_session: Callable[[], FrFruitSession],
) -> None:
    @mcp.tool(
        title="Design Realistic Quad-Morph Human Character",
        annotations=ToolAnnotations(open_world_hint=False),
    )
    def character_manage(
        action: Annotated[
            Literal[
                "create",
                "update",
                "sculpt_rbf",
                "compute_delta",
                "apply_delta",
                "blend",
                "extract_edit_delta",
                "save_morph",
                "inspect",
                "list_presets",
            ],
            Field(description="Character modeling, morphing, sculpting, or delta blending action."),
        ] = "create",
        object_id: Annotated[
            str | None,
            Field(
                description=(
                    "Target character object ID or name "
                    "(required for update/sculpt/blend/save_morph/inspect)."
                )
            ),
        ] = None,
        name: Annotated[
            str | None, Field(description="Optional object name for created character.")
        ] = None,
        preset: Annotated[
            str | None,
            Field(
                description=(
                    "Built-in realistic character preset ('realistic_neutral', 'realistic_male', "
                    "'realistic_female', 'realistic_athletic', 'elder_portrait')."
                )
            ),
        ] = None,
        gender: Annotated[
            float | None,
            Field(
                ge=-1.0,
                le=1.0,
                description="Sexual dimorphism: -1.0 feminine .. +1.0 masculine.",
            ),
        ] = None,
        height: Annotated[
            float | None, Field(ge=1.45, le=2.10, description="Total character height in meters.")
        ] = None,
        muscularity: Annotated[float | None, Field(ge=0.0, le=1.0)] = None,
        body_fat: Annotated[float | None, Field(ge=0.0, le=1.0)] = None,
        age: Annotated[float | None, Field(ge=0.0, le=1.0)] = None,
        quality: Annotated[
            CharacterQuality | None,
            Field(description="Catmull-Clark subdivision tier: 'preview', 'medium', or 'high'."),
        ] = None,
        pose: Annotated[
            Literal["a_pose", "t_pose", "relaxed"] | None,
            Field(description="Rest pose for arms and stance."),
        ] = None,
        auto_rig: Annotated[
            bool | None, Field(description="Auto-generate & bind 19-bone humanoid armature.")
        ] = None,
        body_params: Annotated[
            dict[str, float] | None,
            Field(description="Partial dictionary of BodySliders fields to set."),
        ] = None,
        face_params: Annotated[
            dict[str, float] | None,
            Field(
                description=(
                    "Partial dictionary of FaceSliders fields to set "
                    "(e.g. jaw_width, nose_bridge, upper_lip_fullness)."
                )
            ),
        ] = None,
        expressions: Annotated[
            dict[str, float] | None,
            Field(
                description=(
                    "Partial dictionary of ExpressionSliders fields "
                    "(smile, blink_left, blink_right, brow_raise, jaw_open)."
                )
            ),
        ] = None,
        custom_morphs: Annotated[
            dict[str, float] | None,
            Field(description=("Weights for stored custom morph targets, e.g. {'hero_jaw': 0.8}.")),
        ] = None,
        morph_name: Annotated[
            str,
            Field(description="Name for saving a custom morph target via action='save_morph'."),
        ] = "custom_morph_1",
        material_params: Annotated[
            dict[str, Any] | None,
            Field(
                description=(
                    "Partial CharacterMaterialParams (skin_tone, eye_color, lip_tint, "
                    "hair_color, hair_style ['none','short','swept','bob','long_wavy'], "
                    "hair_length, hair_volume, hair_curl, hair_part, clothing_style "
                    "['casual','tunic','athletic','none'], top_color, bottom_color, "
                    "subsurface_strength)."
                )
            ),
        ] = None,
        location: Annotated[Vec3, Field(description="World spawn location in meters.")] = (
            0.0,
            0.0,
            0.0,
        ),
        rotation_deg: Annotated[Vec3, Field(description="World Euler XYZ rotation in degrees.")] = (
            0.0,
            0.0,
            0.0,
        ),
        collection_id: Annotated[str, Field(description="Target collection ID.")] = "default",
        # RBF sculpting parameters
        sculpt_region: Annotated[
            str | None,
            Field(
                description=(
                    "Semantic RBF region ('nose_tip', 'nose_bridge', 'cheekbone', 'cheek', "
                    "'jaw_corner', 'chin', 'brow', 'forehead', 'upper_lip', 'lower_lip', "
                    "'ear') or 'custom'."
                )
            ),
        ] = None,
        sculpt_center: Annotated[
            Vec3 | None, Field(description="Optional explicit 3D RBF center (head_norm or local).")
        ] = None,
        sculpt_radius: Annotated[
            float | None, Field(gt=0.01, le=2.0, description="RBF influence radius.")
        ] = None,
        sculpt_weight: Annotated[
            float,
            Field(
                ge=-0.5,
                le=0.5,
                description=(
                    "Normal-axis sculpt offset (negative protrudes, positive pushes inward)."
                ),
            ),
        ] = 0.0,
        sculpt_vector: Annotated[Vec3, Field(description="Directional displacement vector.")] = (
            0.0,
            0.0,
            0.0,
        ),
        mirror_x: Annotated[bool | None, Field(description="Mirror RBF node across X=0.")] = None,
        clear_rbf: Annotated[
            bool, Field(description="Clear existing RBF sculpt nodes before applying.")
        ] = False,
        # Delta ("الفارق") & blending parameters
        source_character_id: Annotated[
            str | None,
            Field(
                description=(
                    "Second character ID/name (or preset name) used for "
                    "compute_delta / blend / apply_delta."
                )
            ),
        ] = None,
        delta_name: Annotated[
            str, Field(description="Name of stored CharacterDelta to save or apply.")
        ] = "delta_1",
        delta_region: Annotated[
            DeltaRegion,
            Field(description="Anatomical scope for delta computation/application."),
        ] = "all",
        delta_weight: Annotated[
            float,
            Field(
                ge=-2.0,
                le=2.5,
                description="Blending / delta transfer multiplier alpha (e.g. 0.65).",
            ),
        ] = 0.5,
        region_weights: Annotated[
            dict[DeltaRegion, float] | None,
            Field(
                description=(
                    "Optional per-region blend weights for action='blend', "
                    "e.g. {'nose': 0.8, 'jaw': 0.4}."
                )
            ),
        ] = None,
    ) -> dict[str, Any]:
        """Create, morph, sculpt, and delta-blend realistic Quad-Morph human characters."""
        sess = get_session()
        with sess.lock:
            if action == "list_presets":
                return {
                    "presets": {
                        k: {
                            "gender": v.body.gender,
                            "height": v.body.height,
                            "muscularity": v.body.muscularity,
                            "hair_style": v.materials.hair_style,
                            "clothing_style": v.materials.clothing_style,
                        }
                        for k, v in CHARACTER_PRESETS.items()
                    },
                    "expressions": list(ExpressionSliders.model_fields.keys()),
                    "semantic_rbf_regions": sorted(SEMANTIC_RBF_ANCHORS.keys()),
                    "skin_tone_presets": SKIN_TONE_PRESETS,
                    "delta_regions": [
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
                    ],
                }

            if action == "inspect":
                if not object_id:
                    raise ToolError("object_id is required for action='inspect'.")
                obj = _require_character_object(sess, object_id)
                assert obj.character is not None
                lm = compute_landmarks(obj.character)
                return {
                    "object_id": obj.id,
                    "name": obj.name,
                    "character": obj.character.model_dump(),
                    "anthropometry": landmarks_summary(lm),
                }

            if action == "create":
                try:
                    spec = get_character_preset(preset or "realistic_neutral")
                    if quality is not None:
                        spec.quality = quality
                    if pose is not None:
                        spec.pose = pose
                    if auto_rig is not None:
                        spec.auto_rig = auto_rig

                    b_data = spec.body.model_dump()
                    if body_params:
                        b_data.update(body_params)
                    if gender is not None:
                        b_data["gender"] = gender
                    if height is not None:
                        b_data["height"] = height
                    if muscularity is not None:
                        b_data["muscularity"] = muscularity
                    if body_fat is not None:
                        b_data["body_fat"] = body_fat
                    if age is not None:
                        b_data["age"] = age
                    spec.body = BodySliders(**b_data)

                    if face_params:
                        f_data = spec.face.model_dump()
                        f_data.update(face_params)
                        spec.face = FaceSliders(**f_data)

                    if expressions:
                        e_data = spec.expressions.model_dump()
                        e_data.update(expressions)
                        spec.expressions = ExpressionSliders(**e_data)

                    if custom_morphs:
                        for mk, mv in custom_morphs.items():
                            spec.custom_morphs[str(mk)] = float(mv)

                    if material_params:
                        m_data = spec.materials.model_dump()
                        m_data.update(material_params)
                        spec.materials = CharacterMaterialParams(**m_data)
                except (ValueError, ValidationError) as exc:
                    raise ToolError(f"Invalid character parameters: {exc}") from exc

                sess.push_history(f"character_create:{spec.preset}", notify=False)
                res = build_or_update_character(
                    sess.scene,
                    spec,
                    name=name,
                    location=location,
                    rotation_deg=rotation_deg,
                    collection_id=collection_id,
                    cache_root=sess.workspace_root,
                )
                sess.notify("character_created", {"object_id": res["object_id"]})
                return res

            # Remaining actions operate on an existing character object
            if not object_id:
                raise ToolError(f"object_id is required for action={action!r}.")
            obj = _require_character_object(sess, object_id)
            assert obj.character is not None

            if action == "update":
                try:
                    spec = obj.character.model_copy(deep=True)
                    if quality is not None:
                        spec.quality = quality
                    if pose is not None:
                        spec.pose = pose
                    if auto_rig is not None:
                        spec.auto_rig = auto_rig

                    b_data = spec.body.model_dump()
                    if body_params:
                        b_data.update(body_params)
                    if gender is not None:
                        b_data["gender"] = gender
                    if height is not None:
                        b_data["height"] = height
                    if muscularity is not None:
                        b_data["muscularity"] = muscularity
                    if body_fat is not None:
                        b_data["body_fat"] = body_fat
                    if age is not None:
                        b_data["age"] = age
                    spec.body = BodySliders(**b_data)

                    if face_params:
                        f_data = spec.face.model_dump()
                        f_data.update(face_params)
                        spec.face = FaceSliders(**f_data)

                    if expressions:
                        e_data = spec.expressions.model_dump()
                        e_data.update(expressions)
                        spec.expressions = ExpressionSliders(**e_data)

                    if custom_morphs:
                        for mk, mv in custom_morphs.items():
                            spec.custom_morphs[str(mk)] = float(mv)

                    if material_params:
                        m_data = spec.materials.model_dump()
                        m_data.update(material_params)
                        spec.materials = CharacterMaterialParams(**m_data)
                except (ValueError, ValidationError) as exc:
                    raise ToolError(f"Invalid character update: {exc}") from exc

                sess.push_history(f"character_update:{obj.id}", notify=False)
                res = build_or_update_character(
                    sess.scene,
                    spec,
                    object_id=obj.id,
                    cache_root=sess.workspace_root,
                )
                sess.notify("character_updated", {"object_id": obj.id})
                return res

            if action == "sculpt_rbf":
                spec = obj.character.model_copy(deep=True)
                if clear_rbf:
                    spec.rbf_nodes = []
                node_info: dict[str, Any] | None = None
                if sculpt_region is not None or sculpt_center is not None:
                    try:
                        spec, node = add_rbf_sculpt_node(
                            spec,
                            region=sculpt_region or "custom",
                            center=sculpt_center,
                            radius=sculpt_radius,
                            weight=sculpt_weight,
                            vector_delta=sculpt_vector,
                            mirror_x=mirror_x,
                        )
                        node_info = node.model_dump()
                    except ValueError as exc:
                        raise ToolError(str(exc)) from exc
                elif not clear_rbf:
                    raise ToolError(
                        "Specify sculpt_region or sculpt_center (or clear_rbf=True) for sculpt_rbf."
                    )

                sess.push_history(f"character_sculpt_rbf:{obj.id}", notify=False)
                res = build_or_update_character(
                    sess.scene,
                    spec,
                    object_id=obj.id,
                    cache_root=sess.workspace_root,
                )
                res["added_node"] = node_info
                sess.notify("character_updated", {"object_id": obj.id})
                return res

            if action == "compute_delta":
                src_spec, src_label = _resolve_source_spec(sess, source_character_id, preset)
                delta = compute_character_delta(
                    obj.character,
                    src_spec,
                    region=delta_region,
                    name=delta_name,
                    source_a_name=obj.name,
                    source_b_name=src_label,
                )
                sess.push_history(f"character_compute_delta:{obj.id}:{delta_name}")
                obj.character.stored_deltas[delta_name] = delta
                return {
                    "status": "ok",
                    "object_id": obj.id,
                    "delta_name": delta_name,
                    "region": delta_region,
                    "body_delta_keys": sorted(delta.body_delta.keys()),
                    "face_delta_keys": sorted(delta.face_delta.keys()),
                    "expression_delta_keys": sorted(delta.expression_delta.keys()),
                    "rbf_delta_count": len(delta.rbf_nodes),
                    "delta": delta.model_dump(),
                }

            if action == "apply_delta":
                spec = obj.character.model_copy(deep=True)
                if delta_name in spec.stored_deltas and not source_character_id and not preset:
                    delta = spec.stored_deltas[delta_name]
                else:
                    src_spec, src_label = _resolve_source_spec(sess, source_character_id, preset)
                    base_spec = get_character_preset("realistic_neutral")
                    delta = compute_character_delta(
                        base_spec,
                        src_spec,
                        region=delta_region,
                        name=delta_name,
                        source_a_name="realistic_neutral",
                        source_b_name=src_label,
                    )
                updated = apply_character_delta(
                    spec, delta, weight=delta_weight, region=delta_region
                )
                sess.push_history(f"character_apply_delta:{obj.id}:{delta_name}")
                res = build_or_update_character(
                    sess.scene,
                    updated,
                    object_id=obj.id,
                    cache_root=sess.workspace_root,
                )
                res["applied_delta"] = delta.name
                res["delta_weight"] = delta_weight
                sess.notify("character_updated", {"object_id": obj.id})
                return res

            if action == "blend":
                src_spec, src_label = _resolve_source_spec(sess, source_character_id, preset)
                updated = blend_characters(
                    obj.character,
                    src_spec,
                    alpha=delta_weight,
                    region=delta_region,
                    region_weights=region_weights,
                )
                sess.push_history(f"character_blend:{obj.id}:{src_label}")
                res = build_or_update_character(
                    sess.scene,
                    updated,
                    object_id=obj.id,
                    cache_root=sess.workspace_root,
                )
                res["blended_with"] = src_label
                res["alpha"] = delta_weight
                sess.notify("character_updated", {"object_id": obj.id})
                return res

            if action == "extract_edit_delta":
                if obj.mesh is None:
                    raise ToolError(f"Object {obj.name!r} has no mesh payload.")
                edited_mesh = MeshData.from_payload(obj.mesh)
                surface = CharacterSurface(obj.character, quality=obj.character.quality)
                orig_mesh = surface.skin_mesh
                if len(orig_mesh.vertices) != len(edited_mesh.vertices):
                    raise ToolError(
                        f"Mesh topology changed ({len(orig_mesh.vertices)} -> "
                        f"{len(edited_mesh.vertices)} vertices). Only non-topological "
                        "face/vertex sculpting edits (move/scale/rotate) can be extracted "
                        "to RBF deltas."
                    )
                nodes = extract_rbf_delta_from_mesh_edit(
                    orig_mesh,
                    edited_mesh,
                    surface.lm,
                    region_name=sculpt_region or "mesh_edit",
                )
                if not nodes:
                    return {
                        "status": "ok",
                        "object_id": obj.id,
                        "extracted_nodes": 0,
                        "message": "No significant vertex displacements detected.",
                    }
                spec = obj.character.model_copy(deep=True)
                spec.rbf_nodes.extend(nodes)
                sess.push_history(f"character_extract_edit_delta:{obj.id}")
                res = build_or_update_character(
                    sess.scene,
                    spec,
                    object_id=obj.id,
                    cache_root=sess.workspace_root,
                )
                res["extracted_nodes"] = len(nodes)
                res["nodes"] = [n.model_dump() for n in nodes]
                sess.notify("character_updated", {"object_id": obj.id})
                return res

            if action == "save_morph":
                if obj.mesh is None:
                    raise ToolError(f"Object {obj.name!r} has no mesh payload.")
                edited_mesh = MeshData.from_payload(obj.mesh)
                try:
                    morph = extract_custom_morph_from_mesh_edit(
                        obj.character,
                        edited_mesh,
                        morph_name=morph_name,
                        region=delta_region,
                    )
                except ValueError as exc:
                    raise ToolError(str(exc)) from exc
                if not morph.deltas:
                    return {
                        "status": "ok",
                        "object_id": obj.id,
                        "morph_name": morph.name,
                        "active_vertices": 0,
                        "message": "No significant base-cage displacements detected.",
                    }
                spec = obj.character.model_copy(deep=True)
                spec.stored_morphs[morph.name] = morph
                spec.custom_morphs[morph.name] = 1.0
                sess.push_history(f"character_save_morph:{obj.id}:{morph.name}")
                res = build_or_update_character(
                    sess.scene,
                    spec,
                    object_id=obj.id,
                    cache_root=sess.workspace_root,
                )
                res["saved_morph"] = morph.name
                res["active_vertices"] = len(morph.deltas)
                sess.notify("character_updated", {"object_id": obj.id})
                return res

            raise ToolError(f"Unsupported character_manage action: {action!r}")
