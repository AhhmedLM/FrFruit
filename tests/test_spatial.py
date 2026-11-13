"""Tests for spatial reasoning, collision detection, raycast, and smart placement."""

from __future__ import annotations

import pytest

from frfruit.core.scene import SceneDocument, SceneObject
from frfruit.core.types import Transform
from frfruit.geometry.primitives import make_cube, make_sphere
from frfruit.geometry.spatial import (
    align_objects,
    attach_to_side,
    check_intersections,
    distribute_objects,
    evaluate_object_mesh,
    evaluate_scene_meshes,
    ground_object,
    measure_distance,
    place_on_top,
    raycast_scene,
    snap_to_surface,
)


def _add_box(
    doc: SceneDocument, oid: str, loc: tuple[float, float, float], size: float = 1.0
) -> None:
    doc.objects[oid] = SceneObject(
        id=oid,
        name=oid,
        kind="mesh",
        transform=Transform(location=loc),
        mesh=make_cube(size, origin="center").to_payload(),
    )


def test_place_on_top_ground_attach_and_no_false_collision() -> None:
    doc = SceneDocument.new_default("Spatial")
    _add_box(doc, "base", (1.0, 2.0, 2.0), size=2.0)
    _add_box(doc, "top", (-3.0, 0.0, 10.0), size=1.0)

    ground_object(doc, evaluate_object_mesh(doc, "base"), z=0.0)
    ev_base = evaluate_object_mesh(doc, "base")
    assert ev_base.bounds.min_corner[2] == pytest.approx(0.0, abs=1e-6)
    assert ev_base.bounds.max_corner[2] == pytest.approx(2.0, abs=1e-6)

    place_on_top(doc, evaluate_object_mesh(doc, "top"), ev_base, offset=0.0, align_xy="center")
    ev_top = evaluate_object_mesh(doc, "top")
    assert ev_top.bounds.min_corner[2] == pytest.approx(2.0, abs=1e-6)
    assert ev_top.bounds.center[:2] == pytest.approx((1.0, 2.0), abs=1e-6)

    # Touching flush should NOT count as an intersection
    cols = check_intersections(evaluate_scene_meshes(doc), tolerance=1e-3)
    assert cols == []

    # Attach to +X side of base
    attach_to_side(doc, ev_top, ev_base, side="+x", gap=0.25)
    ev_top2 = evaluate_object_mesh(doc, "top")
    assert ev_top2.bounds.min_corner[0] == pytest.approx(
        ev_base.bounds.max_corner[0] + 0.25, abs=1e-6
    )


def test_collision_detection_and_distance_and_raycast() -> None:
    doc = SceneDocument.new_default("Col")
    _add_box(doc, "a", (0.0, 0.0, 0.5), size=1.0)
    _add_box(doc, "b", (0.4, 0.0, 0.5), size=1.0)

    evaluated = evaluate_scene_meshes(doc)
    cols = check_intersections(evaluated)
    assert len(cols) == 1
    assert cols[0]["intersection_volume"] > 0.5

    dist_info = measure_distance(evaluated["a"], evaluated["b"])
    assert dist_info["intersecting"] is True
    assert dist_info["surface_distance"] == 0.0

    hit = raycast_scene(evaluated, origin=(0.0, 0.0, 5.0), direction=(0.0, 0.0, -1.0))
    assert hit["hit"] is True
    assert hit["point"][2] == pytest.approx(1.0, abs=1e-5)


def test_snap_to_curved_surface_align_and_distribute() -> None:
    doc = SceneDocument.new_default("Snap")
    doc.objects["ball"] = SceneObject(
        id="ball",
        name="Ball",
        kind="mesh",
        transform=Transform(location=(0.0, 0.0, 1.0)),
        mesh=make_sphere(1.0, segments=24, origin="center").to_payload(),
    )
    _add_box(doc, "pin", (0.0, 0.0, 5.0), size=0.2)
    snap_to_surface(
        doc,
        evaluate_object_mesh(doc, "pin"),
        evaluate_object_mesh(doc, "ball"),
        direction=(0.0, 0.0, -1.0),
    )
    ev_pin = evaluate_object_mesh(doc, "pin")
    assert ev_pin.bounds.min_corner[2] == pytest.approx(2.0, abs=0.02)

    _add_box(doc, "p1", (0.0, 0.0, 0.0), size=1.0)
    _add_box(doc, "p2", (0.0, 4.0, 2.0), size=1.0)
    align_objects(
        doc,
        [evaluate_object_mesh(doc, "p1"), evaluate_object_mesh(doc, "p2")],
        axis="y",
        mode="center",
    )
    assert evaluate_object_mesh(doc, "p2").bounds.center[1] == pytest.approx(0.0, abs=1e-6)

    distribute_objects(
        doc,
        [evaluate_object_mesh(doc, "p1"), evaluate_object_mesh(doc, "p2")],
        axis="x",
        spacing=1.0,
    )
    assert evaluate_object_mesh(doc, "p2").bounds.min_corner[0] == pytest.approx(1.5, abs=1e-6)
