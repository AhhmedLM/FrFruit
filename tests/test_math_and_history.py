"""Tests for 3D math, color parsing, SceneDocument resolution, and Undo/Redo history."""

from __future__ import annotations

import numpy as np
import pytest

from frfruit.core.history import HistoryManager
from frfruit.core.math3d import (
    compose_matrix,
    decompose_matrix,
    look_at_matrix,
    ray_triangles_intersect,
    transform_normals,
    transform_points,
)
from frfruit.core.scene import SceneDocument, SceneObject
from frfruit.core.types import BoundingBox, Transform, parse_color_rgba
from frfruit.geometry.primitives import make_cube


def test_color_parsing() -> None:
    assert parse_color_rgba("#ff8000") == pytest.approx((1.0, 128 / 255.0, 0.0, 1.0))
    assert parse_color_rgba([0.2, 0.4, 0.6]) == (0.2, 0.4, 0.6, 1.0)
    assert parse_color_rgba([2.0, -1.0, 0.5, 0.8]) == (1.0, 0.0, 0.5, 0.8)
    with pytest.raises(ValueError):
        parse_color_rgba("#123")


def test_compose_and_decompose_matrix_roundtrip() -> None:
    loc = (1.5, -2.25, 3.75)
    rot = (20.0, -35.0, 60.0)
    scale = (1.2, 0.8, 2.5)
    mat = compose_matrix(loc, rot, scale)
    d_loc, d_rot, d_scale = decompose_matrix(mat)
    assert d_loc == pytest.approx(loc, abs=1e-6)
    assert d_rot == pytest.approx(rot, abs=1e-5)
    assert d_scale == pytest.approx(scale, abs=1e-6)


def test_transform_points_and_normals() -> None:
    mat = compose_matrix((1.0, 0.0, 0.0), (0.0, 0.0, 90.0), (2.0, 1.0, 1.0))
    pt = transform_points(mat, np.array([[1.0, 0.0, 0.0]]))
    assert pt[0] == pytest.approx([1.0, 2.0, 0.0], abs=1e-6)

    nrm = transform_normals(mat, np.array([[1.0, 0.0, 0.0]]))
    assert nrm[0] == pytest.approx([0.0, 1.0, 0.0], abs=1e-6)


def test_look_at_and_ray_triangle() -> None:
    view = look_at_matrix((0.0, -5.0, 0.0), (0.0, 0.0, 0.0))
    cam_pt = transform_points(view, np.array([[0.0, 0.0, 0.0]]))
    assert cam_pt[0] == pytest.approx([0.0, 0.0, -5.0], abs=1e-6)

    verts = np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [0.0, 2.0, 0.0]])
    faces = np.array([[0, 1, 2]])
    hit = ray_triangles_intersect(
        np.array([0.5, 0.5, 3.0]),
        np.array([0.0, 0.0, -1.0]),
        verts,
        faces,
    )
    assert hit is not None
    f_idx, dist, normal = hit
    assert f_idx == 0
    assert dist == pytest.approx(3.0, abs=1e-6)
    assert normal == pytest.approx([0.0, 0.0, 1.0], abs=1e-6)


def test_bounding_box_overlap() -> None:
    a = BoundingBox(min_corner=(0.0, 0.0, 0.0), max_corner=(1.0, 1.0, 1.0))
    b = BoundingBox(min_corner=(1.0, 0.0, 0.0), max_corner=(2.0, 1.0, 1.0))
    assert a.overlaps(b, tolerance=0.0)
    assert not a.overlaps(b, tolerance=1e-3)


def test_history_undo_redo_and_snapshots() -> None:
    doc = SceneDocument.new_default("Hist")
    hm = HistoryManager()

    hm.push(doc, "add_cube")
    doc.objects["c1"] = SceneObject(
        id="c1",
        name="Cube",
        kind="mesh",
        transform=Transform(location=(1.0, 0.0, 0.0)),
        mesh=make_cube().to_payload(),
    )
    hm.create_snapshot(doc, "with_cube")

    hm.push(doc, "move_cube")
    doc.objects["c1"].transform.location = (5.0, 0.0, 0.0)

    res = hm.undo(doc)
    assert res is not None
    doc, label = res
    assert label == "move_cube"
    assert doc.objects["c1"].transform.location == (1.0, 0.0, 0.0)

    res2 = hm.redo(doc)
    assert res2 is not None
    doc, _ = res2
    assert doc.objects["c1"].transform.location == (5.0, 0.0, 0.0)

    restored = hm.restore_snapshot(doc, "with_cube")
    assert restored is not None
    assert restored.objects["c1"].transform.location == (1.0, 0.0, 0.0)
