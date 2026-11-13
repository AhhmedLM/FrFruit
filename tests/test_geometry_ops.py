"""Tests for primitives, curves, semantic Edit Mesh, CSG booleans, and modifiers."""

from __future__ import annotations

import pytest

from frfruit.core.scene import (
    ArrayModifier,
    BendModifier,
    BevelModifier,
    BooleanModifier,
    DisplaceModifier,
    MirrorModifier,
    SceneDocument,
    SceneObject,
    SmoothModifier,
    SolidifyModifier,
    SubdivisionModifier,
    TaperModifier,
    TwistModifier,
)
from frfruit.core.types import SelectionQuery, Transform
from frfruit.geometry.curves import extrude_2d_profile, revolve_profile, sweep_curve
from frfruit.geometry.edit_ops import (
    bevel_edges,
    boolean_meshes,
    extrude_faces,
    inset_faces,
    subdivide_mesh,
    transform_selection,
)
from frfruit.geometry.mesh import GeometryError
from frfruit.geometry.primitives import (
    make_capsule,
    make_cone,
    make_cube,
    make_cylinder,
    make_icosphere,
    make_plane,
    make_sphere,
    make_torus,
)
from frfruit.geometry.spatial import evaluate_object_mesh


def test_primitives_are_valid_and_watertight() -> None:
    solids = [
        make_cube(1.0, origin="bottom"),
        make_sphere(0.5, segments=16, origin="bottom"),
        make_icosphere(0.5, subdivisions=2, origin="bottom"),
        make_cylinder(0.4, 1.2, segments=16, origin="bottom"),
        make_cone(0.4, 1.0, segments=16, origin="bottom"),
        make_torus(0.6, 0.15, origin="bottom"),
        make_capsule(0.3, 1.2, origin="bottom"),
    ]
    for m in solids:
        st = m.stats()
        assert st["vertex_count"] > 0
        assert st["face_count"] > 0
        assert st["is_watertight"] is True
        assert m.bounds().min_corner[2] == pytest.approx(0.0, abs=1e-6)

    plane = make_plane((4.0, 2.0), subdivisions=3)
    assert plane.stats()["face_count"] == 18
    assert plane.stats()["is_watertight"] is False


def test_curve_extrude_revolve_and_sweep() -> None:
    tri = [(0.0, 0.0), (1.0, 0.0), (0.5, 1.0)]
    ext = extrude_2d_profile(tri, height=2.0, twist_deg=30.0)
    assert ext.stats()["is_watertight"] is True

    # Lathe apple-like profile
    profile = [(0.0, 0.0), (0.5, 0.05), (0.65, 0.4), (0.45, 0.8), (0.0, 0.72)]
    lathe = revolve_profile(profile, segments=24)
    assert lathe.stats()["is_watertight"] is True
    assert lathe.stats()["volume"] > 0.3

    # Curved stem via parallel-transport sweep
    stem = sweep_curve(
        [(0.0, 0.0, 0.0), (0.05, 0.0, 0.2), (0.15, 0.05, 0.45)],
        radius=0.05,
        taper_end=0.6,
    )
    assert stem.stats()["is_watertight"] is True


def test_edit_ops_keep_watertight_and_grow_dimensions() -> None:
    cube = make_cube(1.0, origin="bottom")
    extruded = extrude_faces(cube, SelectionQuery(mode="top"), distance=0.5, scale=0.6)
    assert extruded.stats()["is_watertight"] is True
    assert extruded.bounds().max_corner[2] == pytest.approx(1.5, abs=1e-6)

    inset = inset_faces(cube, SelectionQuery(mode="top"), thickness=0.15, depth=-0.05)
    assert inset.stats()["is_watertight"] is True

    beveled = bevel_edges(cube, angle_threshold_deg=35.0, width=0.08, segments=1)
    assert beveled.stats()["is_watertight"] is True
    assert beveled.bounds().size == pytest.approx((1.0, 1.0, 1.0), abs=1e-5)
    assert beveled.stats()["face_count"] > cube.stats()["face_count"]

    sub = subdivide_mesh(cube, levels=1, smooth=True)
    assert sub.stats()["is_watertight"] is True
    assert sub.stats()["face_count"] == cube.stats()["face_count"] * 4

    moved_top = transform_selection(cube, SelectionQuery(mode="top"), translate=(0.0, 0.0, 0.5))
    assert moved_top.bounds().max_corner[2] == pytest.approx(1.5, abs=1e-6)


def test_csg_boolean_and_error_on_open_mesh() -> None:
    a = make_cube(1.0, origin="center")
    b = make_sphere(0.65, segments=16, origin="center")
    diff = boolean_meshes(a, b, "difference")
    assert diff.stats()["is_watertight"] is True
    assert diff.stats()["volume"] < 1.0

    plane = make_plane(2.0)
    with pytest.raises(GeometryError):
        boolean_meshes(a, plane, "union")


def test_modifier_stack_evaluation() -> None:
    doc = SceneDocument.new_default("Mods")
    cube = make_cube(1.0, origin="center")
    plane = make_plane(1.0, subdivisions=1)

    doc.objects["solid_box"] = SceneObject(
        id="solid_box",
        name="SolidBox",
        kind="mesh",
        transform=Transform(location=(0.0, 0.0, 0.5)),
        mesh=cube.to_payload(),
        modifiers=[
            BevelModifier(width=0.05, segments=1),
            TaperModifier(axis="z", factor=0.2),
            TwistModifier(axis="z", angle_deg=30.0),
            BendModifier(axis="z", bend_axis="x", angle_deg=15.0),
            DisplaceModifier(strength=0.02, scale=2.0),
            SmoothModifier(iterations=1, factor=0.2),
            SubdivisionModifier(levels=1, smooth=False),
            ArrayModifier(mode="linear", count=2, offset=(2.0, 0.0, 0.0)),
            MirrorModifier(axis="y", offset=-1.0),
        ],
    )
    ev = evaluate_object_mesh(doc, "solid_box")
    assert ev.world_mesh.stats()["face_count"] > 100

    # Plane + Solidify becomes closed solid & can participate in Boolean modifier
    doc.objects["cutter"] = SceneObject(
        id="cutter",
        name="Cutter",
        kind="mesh",
        transform=Transform(location=(0.0, 0.0, 0.0)),
        mesh=make_cylinder(0.25, 2.0, origin="center").to_payload(),
    )
    doc.objects["plate"] = SceneObject(
        id="plate",
        name="Plate",
        kind="mesh",
        mesh=plane.to_payload(),
        modifiers=[
            SolidifyModifier(thickness=0.2),
            BooleanModifier(operation="difference", target_object_id="cutter"),
        ],
    )
    ev_plate = evaluate_object_mesh(doc, "plate")
    assert ev_plate.world_mesh.stats()["is_watertight"] is True
    assert ev_plate.world_mesh.stats()["volume"] < 0.2

    # Radial array test
    doc.objects["petal"] = SceneObject(
        id="petal",
        name="Petal",
        kind="mesh",
        mesh=make_cube((0.3, 0.1, 0.05), origin="center").to_payload(),
        modifiers=[
            ArrayModifier(mode="radial", count=6, radial_axis="z", radial_center=(-0.6, 0.0, 0.0))
        ],
    )
    ev_petal = evaluate_object_mesh(doc, "petal")
    assert ev_petal.world_mesh.stats()["face_count"] == 12 * 6
