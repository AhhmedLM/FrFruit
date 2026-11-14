"""Evaluates keyframed FCurveTracks onto a SceneDocument at a given frame."""

from __future__ import annotations

from typing import TYPE_CHECKING

from frfruit.animation.timeline import evaluate_fcurve

if TYPE_CHECKING:
    from frfruit.core.scene import SceneDocument

_AXIS_IDX = {"x": 0, "y": 1, "z": 2}
_RGBA_IDX = {"r": 0, "g": 1, "b": 2, "a": 3}


def _set_tuple_axis(tup: tuple[float, ...], idx: int, val: float) -> tuple[float, ...]:
    lst = list(tup)
    lst[idx] = val
    return tuple(lst)


def apply_animation_frame(scene: SceneDocument, frame: int) -> SceneDocument:
    """Return a copy of `scene` evaluated at `frame` without mutating the source tracks."""
    eval_doc = scene.model_copy(deep=True)
    eval_doc.timeline.current_frame = int(frame)

    for track in eval_doc.timeline.tracks:
        val = evaluate_fcurve(track, float(frame))
        if val is None:
            continue

        obj = eval_doc.resolve_object(track.target_id)
        mat = eval_doc.resolve_material(track.target_id)
        parts = track.property_path.split(".")

        if obj is not None:
            if (
                len(parts) == 2
                and parts[0] in ("location", "rotation_deg", "scale")
                and parts[1] in _AXIS_IDX
            ):
                idx = _AXIS_IDX[parts[1]]
                if parts[0] == "location":
                    obj.transform.location = _set_tuple_axis(obj.transform.location, idx, val)  # type: ignore[assignment]
                elif parts[0] == "rotation_deg":
                    obj.transform.rotation_deg = _set_tuple_axis(
                        obj.transform.rotation_deg, idx, val
                    )  # type: ignore[assignment]
                elif parts[0] == "scale":
                    v_safe = val if abs(val) > 1e-6 else 1e-6
                    obj.transform.scale = _set_tuple_axis(obj.transform.scale, idx, v_safe)  # type: ignore[assignment]
            elif (
                obj.armature is not None
                and len(parts) == 4
                and parts[0] == "bone"
                and parts[2] in ("rotation_deg", "translation")
                and parts[3] in _AXIS_IDX
            ):
                bname, prop, ax = parts[1], parts[2], _AXIS_IDX[parts[3]]
                for bone in obj.armature.bones:
                    if bone.name == bname:
                        if prop == "rotation_deg":
                            bone.rotation_deg = _set_tuple_axis(bone.rotation_deg, ax, val)  # type: ignore[assignment]
                        else:
                            bone.translation = _set_tuple_axis(bone.translation, ax, val)  # type: ignore[assignment]
                        break
            elif obj.camera is not None:
                if len(parts) == 2 and parts[0] == "target" and parts[1] in _AXIS_IDX:
                    obj.camera.target = _set_tuple_axis(obj.camera.target, _AXIS_IDX[parts[1]], val)  # type: ignore[assignment]
                elif track.property_path == "fov_deg":
                    obj.camera.fov_deg = max(5.0, min(150.0, val))
                elif track.property_path == "ortho_scale":
                    obj.camera.ortho_scale = max(0.05, val)
            elif obj.light is not None:
                if track.property_path == "intensity":
                    obj.light.intensity = max(0.0, val)
                elif len(parts) == 2 and parts[0] == "color" and parts[1] in ("r", "g", "b"):
                    obj.light.color = _set_tuple_axis(
                        obj.light.color, _RGBA_IDX[parts[1]], max(0.0, min(1.0, val))
                    )  # type: ignore[assignment]

        elif mat is not None:
            if len(parts) == 2 and parts[0] == "base_color" and parts[1] in _RGBA_IDX:
                mat.base_color = _set_tuple_axis(
                    mat.base_color, _RGBA_IDX[parts[1]], max(0.0, min(1.0, val))
                )  # type: ignore[assignment]
            elif track.property_path == "metallic":
                mat.metallic = max(0.0, min(1.0, val))
            elif track.property_path == "roughness":
                mat.roughness = max(0.0, min(1.0, val))
            elif track.property_path == "emission_strength":
                mat.emission_strength = max(0.0, val)

    return eval_doc
