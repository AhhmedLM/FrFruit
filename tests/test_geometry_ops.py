"""Tests for primitives, curves, semantic Edit Mesh, CSG booleans, and modifiers."""

from __future__ import annotations

import numpy as np
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


def test_full_edit_mode_modeling_suite_and_face_material_preservation() -> None:
    from frfruit.geometry.edit_ops import inspect_mesh_elements, run_edit_operation

    cube = make_cube(1.0, origin="bottom")
    # Assign material slot to single face 0 and test preservation across operations
    res_assign = run_edit_operation(
        cube,
        "assign_face_material",
        SelectionQuery(mode="indices", indices=[0]),
        material_id="mat_gold",
    )
    assert res_assign.face_material_ids == {0: "mat_gold"}

    # Extrude individual faces + along normals + material override
    res_ind = run_edit_operation(
        res_assign.mesh,
        "extrude_individual",
        SelectionQuery(mode="top"),
        face_material_ids=res_assign.face_material_ids,
        distance=0.35,
        scale=0.8,
        material_id="mat_cap",
    )
    assert res_ind.mesh.stats()["is_watertight"] is True
    assert res_ind.mesh.bounds().max_corner[2] == pytest.approx(1.35, abs=1e-6)
    assert "mat_cap" in res_ind.face_material_ids.values()

    res_along = run_edit_operation(
        cube,
        "extrude_along_normals",
        SelectionQuery(mode="top"),
        distance=0.25,
    )
    assert res_along.mesh.stats()["is_watertight"] is True
    assert res_along.mesh.bounds().max_corner[2] == pytest.approx(1.25, abs=1e-6)

    # Extrude manifold
    res_mf = run_edit_operation(
        cube, "extrude_manifold", SelectionQuery(mode="top"), distance=-0.25
    )
    assert res_mf.mesh.stats()["is_watertight"] is True

    # Extrude vertices (struts)
    res_ev = run_edit_operation(
        cube,
        "extrude_vertices",
        SelectionQuery(mode="vertex_indices", vertex_indices=[0, 1]),
        distance=0.3,
        direction=(0.0, 0.0, 1.0),
        width=0.05,
    )
    assert res_ev.mesh.stats()["face_count"] > cube.stats()["face_count"]

    # Extrude edges on open plane
    plane = make_plane(1.0, subdivisions=1)
    res_ee = run_edit_operation(
        plane,
        "extrude_edges",
        SelectionQuery(mode="edge_indices", edge_indices=[[0, 1]]),
        distance=0.4,
        direction=(0.0, 0.0, 1.0),
    )
    assert res_ee.mesh.stats()["face_count"] == plane.stats()["face_count"] + 2

    # Loop cut + material preservation
    res_lc = run_edit_operation(
        res_assign.mesh,
        "loop_cut",
        SelectionQuery(mode="all"),
        face_material_ids=res_assign.face_material_ids,
        axis="z",
        cuts=2,
        slide=0.2,
    )
    assert res_lc.mesh.stats()["is_watertight"] is True
    assert res_lc.mesh.stats()["face_count"] > cube.stats()["face_count"]
    assert "mat_gold" in res_lc.face_material_ids.values()

    # Partial face selection subdivide (conformal)
    res_sub = run_edit_operation(
        cube,
        "subdivide",
        SelectionQuery(mode="top"),
        levels=1,
    )
    assert res_sub.mesh.stats()["is_watertight"] is True
    assert res_sub.mesh.stats()["face_count"] > cube.stats()["face_count"]

    # Bisect plane with fill_caps=True
    res_bis = run_edit_operation(
        cube,
        "bisect_plane",
        SelectionQuery(mode="all"),
        plane_point=(0.0, 0.0, 0.5),
        plane_normal=(0.0, 0.0, 1.0),
        clear_outer=True,
        fill_caps=True,
    )
    assert res_bis.mesh.stats()["is_watertight"] is True
    assert res_bis.mesh.bounds().max_corner[2] == pytest.approx(0.5, abs=1e-5)

    # Poke faces, inset individual, bevel vertices, solidify, wireframe
    res_poke = run_edit_operation(cube, "poke_faces", SelectionQuery(mode="top"), offset=0.15)
    assert res_poke.mesh.stats()["is_watertight"] is True
    assert res_poke.mesh.bounds().max_corner[2] == pytest.approx(1.15, abs=1e-5)

    res_in = run_edit_operation(
        cube, "inset_faces", SelectionQuery(mode="top"), thickness=0.1, individual=True
    )
    assert res_in.mesh.stats()["is_watertight"] is True

    res_bv = run_edit_operation(cube, "bevel_vertices", SelectionQuery(mode="top"), width=0.12)
    assert res_bv.mesh.stats()["is_watertight"] is True
    assert res_bv.mesh.stats()["face_count"] > cube.stats()["face_count"]

    res_sol = run_edit_operation(plane, "solidify_faces", SelectionQuery(mode="all"), thickness=0.1)
    assert res_sol.mesh.stats()["is_watertight"] is True

    res_wire = run_edit_operation(cube, "wireframe_edges", SelectionQuery(mode="all"), width=0.04)
    assert res_wire.mesh.stats()["face_count"] > cube.stats()["face_count"]

    # Delete + fill_holes + bridge_edge_loops
    res_del = run_edit_operation(
        cube, "delete_elements", SelectionQuery(mode="top"), delete_mode="only_faces"
    )
    assert res_del.mesh.stats()["is_watertight"] is False
    res_fill = run_edit_operation(res_del.mesh, "fill_holes", SelectionQuery(mode="all"))
    assert res_fill.mesh.stats()["is_watertight"] is True

    # Two open rings in one mesh -> bridge_edge_loops
    ring_b = make_cube(1.0, origin="bottom")
    open_bot = run_edit_operation(
        ring_b, "delete_elements", SelectionQuery(mode="top"), delete_mode="only_faces"
    ).mesh
    open_top = run_edit_operation(
        make_cube(1.0, origin="bottom"),
        "delete_elements",
        SelectionQuery(mode="bottom"),
        delete_mode="only_faces",
    ).mesh
    import numpy as np

    from frfruit.geometry.mesh import MeshData, concat_meshes

    open_top = MeshData(
        vertices=open_top.vertices + np.array([0.0, 0.0, 2.0]),
        faces=open_top.faces,
        uvs=open_top.uvs,
    )

    two_rings = concat_meshes([open_bot, open_top])
    res_bridge = run_edit_operation(two_rings, "bridge_edge_loops", SelectionQuery(mode="all"))
    assert res_bridge.mesh.stats()["face_count"] > two_rings.stats()["face_count"]

    # Duplicate, split, separate_selection
    res_dup = run_edit_operation(cube, "duplicate_faces", SelectionQuery(mode="top"), distance=0.25)
    assert res_dup.mesh.stats()["face_count"] == cube.stats()["face_count"] + 2

    res_sep = run_edit_operation(
        res_assign.mesh,
        "separate_selection",
        SelectionQuery(mode="indices", indices=[0, 1]),
        face_material_ids=res_assign.face_material_ids,
    )
    assert res_sep.separated_mesh is not None
    assert res_sep.separated_mesh.stats()["face_count"] == 2
    assert res_sep.mesh.stats()["face_count"] == cube.stats()["face_count"] - 2
    assert res_sep.separated_face_material_ids == {0: "mat_gold"}

    # Proportional transform, shrink_fatten, to_sphere, shear, symmetrize, normals, UV project
    res_prop = run_edit_operation(
        res_lc.mesh,
        "transform_selection",
        SelectionQuery(mode="top"),
        translate=(0.0, 0.0, 0.4),
        proportional_radius=0.8,
        proportional_falloff="smooth",
    )
    assert res_prop.mesh.bounds().max_corner[2] == pytest.approx(1.4, abs=1e-5)

    sf_size = (
        run_edit_operation(cube, "shrink_fatten", SelectionQuery(mode="all"), distance=0.1)
        .mesh.bounds()
        .size[0]
    )
    assert sf_size > 1.15 and sf_size <= 1.2 + 1e-5

    assert (
        run_edit_operation(
            res_lc.mesh, "to_sphere", SelectionQuery(mode="all"), factor=1.0
        ).mesh.stats()["vertex_count"]
        == res_lc.mesh.stats()["vertex_count"]
    )
    assert run_edit_operation(
        cube, "shear", SelectionQuery(mode="all"), axis="x", shear_axis="z", factor=0.5
    ).mesh.bounds().max_corner[0] == pytest.approx(0.75, abs=1e-5)
    assert (
        run_edit_operation(
            cube, "symmetrize", SelectionQuery(mode="all"), symmetrize_direction="-x_to_+x"
        ).mesh.stats()["is_watertight"]
        is True
    )

    flipped = run_edit_operation(cube, "flip_normals", SelectionQuery(mode="all")).mesh
    assert flipped.to_trimesh(process=False).volume < 0
    recalc = run_edit_operation(flipped, "recompute_normals", SelectionQuery(mode="all")).mesh
    assert recalc.to_trimesh(process=False).volume > 0

    uv_res = run_edit_operation(
        cube,
        "project_uvs",
        SelectionQuery(mode="all"),
        projection="cylindrical",
        uv_scale=(2.0, 2.0),
    )
    assert uv_res.mesh.uvs is not None and uv_res.mesh.uvs.shape == (len(cube.vertices), 2)

    # Coplanar flood-fill + inspect_mesh_elements
    summary = inspect_mesh_elements(
        res_assign.mesh,
        detail="selection",
        query=SelectionQuery(mode="coplanar", seed_face=0),
        face_material_ids=res_assign.face_material_ids,
    )
    assert len(summary["selection"]["selected_face_indices"]) == 2
    assert np.isfinite(uv_res.mesh.uvs).all()


def test_quad_catmull_clark_subdivision_creases_and_smart_uv() -> None:
    from frfruit.geometry.bmesh import catmull_clark_subdivide, reconstruct_quads
    from frfruit.geometry.edit_ops import run_edit_operation
    from frfruit.textures.uv_unwrap import smart_uv_unwrap

    cube = make_cube(1.0, origin="center")
    polygons, poly_sources = reconstruct_quads(cube, angle_threshold_deg=15.0)
    assert len(polygons) == 6
    assert len(poly_sources) == 6
    assert all(len(f) == 4 for f in polygons)

    # Smooth Catmull-Clark rounds a cube into a smooth sphere-like watertight cage
    cc_smooth, _, cc_quads = catmull_clark_subdivide(
        cube, levels=2, smooth=True, crease_angle_deg=180.0
    )
    assert cc_smooth.stats()["is_watertight"] is True
    assert cc_quads == 6 * 16
    assert cc_smooth.stats()["face_count"] == 6 * 16 * 2  # 6 quads -> 96 quads -> 192 tris
    # Rounded cube shrinks slightly inward from corners (radius < 0.5*sqrt(3))
    corner_r = float(np.linalg.norm(cc_smooth.vertices, axis=1).max())
    assert corner_r < 0.80

    # Crease-preserving Catmull-Clark keeps sharp 90-degree edges crisp
    cc_creased, _, _ = catmull_clark_subdivide(cube, levels=1, smooth=True, crease_angle_deg=45.0)
    assert cc_creased.stats()["is_watertight"] is True
    assert cc_creased.bounds().size == pytest.approx((1.0, 1.0, 1.0), abs=1e-5)

    # SubdivisionModifier defaults to Catmull-Clark
    doc = SceneDocument.new_default("CCMod")
    doc.objects["c"] = SceneObject(
        id="c",
        name="CCube",
        kind="mesh",
        mesh=cube.to_payload(),
        modifiers=[SubdivisionModifier(levels=1, smooth=True, algorithm="catmull_clark")],
    )
    ev = evaluate_object_mesh(doc, "c")
    assert ev.world_mesh.stats()["is_watertight"] is True
    assert ev.world_mesh.stats()["face_count"] == 48

    # Smart UV unwrap + island packing keeps per-face material assignments & bounds in [0, 1]^2
    uv_mesh, uv_info = smart_uv_unwrap(cube, angle_limit_deg=60.0, island_margin=0.03)
    assert uv_mesh.uvs is not None
    assert uv_info["uv_island_count"] == 6  # 6 faces of a cube split at 90 deg > 60 deg
    assert float(uv_mesh.uvs.min()) >= -1e-6 and float(uv_mesh.uvs.max()) <= 1.0 + 1e-6
    assert len(uv_mesh.faces) == len(cube.faces)

    # Also test via run_edit_operation
    res_uv = run_edit_operation(
        cube,
        "project_uvs",
        SelectionQuery(mode="all"),
        face_material_ids={0: "mat_gold", 1: "mat_gold"},
        projection="smart_uv",
        uv_angle_limit_deg=60.0,
        island_margin=0.03,
    )
    assert res_uv.face_material_ids == {0: "mat_gold", 1: "mat_gold"}
    assert res_uv.extra.get("uv_island_count") == 6
