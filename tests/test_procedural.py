"""Tests for JSON procedural node graphs (Geometry Nodes for AI)."""

from __future__ import annotations

import pytest

from frfruit.procedural import (
    NODE_CATALOG,
    LinkSpec,
    NodeSpec,
    ProceduralError,
    ProceduralGraph,
    evaluate_node_graph,
    list_node_catalog,
)


def test_node_catalog_contains_expected_nodes() -> None:
    catalog = list_node_catalog()
    types = {item["type"] for item in catalog}
    assert len(NODE_CATALOG) >= 18
    for required in (
        "PrimitiveCube",
        "PrimitiveSphere",
        "PrimitiveCone",
        "GridPoints",
        "CurveCircle",
        "CurveLine",
        "TransformGeometry",
        "JoinGeometry",
        "BooleanGeometry",
        "SubdivideGeometry",
        "ConvexHull",
        "ExtrudeGeometry",
        "SetPositionNoise",
        "DistributePointsOnFaces",
        "InstanceOnPoints",
        "GroupOutput",
    ):
        assert required in types


def test_scatter_and_instance_dag_is_deterministic() -> None:
    graph = ProceduralGraph(
        seed=123,
        nodes=[
            NodeSpec(id="sphere", type="PrimitiveSphere", params={"radius": 1.0, "segments": 16}),
            NodeSpec(
                id="dist",
                type="DistributePointsOnFaces",
                params={"count": 12, "min_distance": 0.15},
            ),
            NodeSpec(
                id="cone",
                type="PrimitiveCone",
                params={"radius": 0.08, "height": 0.25, "segments": 8},
            ),
            NodeSpec(
                id="inst",
                type="InstanceOnPoints",
                params={"scale": 1.0, "align_to_normal": True},
            ),
            NodeSpec(id="join", type="JoinGeometry"),
            NodeSpec(id="out", type="GroupOutput"),
        ],
        links=[
            LinkSpec(
                from_node="sphere", from_socket="geometry", to_node="dist", to_socket="geometry"
            ),
            LinkSpec(from_node="dist", from_socket="geometry", to_node="inst", to_socket="points"),
            LinkSpec(
                from_node="cone", from_socket="geometry", to_node="inst", to_socket="instance"
            ),
            LinkSpec(
                from_node="sphere", from_socket="geometry", to_node="join", to_socket="geometry_a"
            ),
            LinkSpec(
                from_node="inst", from_socket="geometry", to_node="join", to_socket="geometry_b"
            ),
            LinkSpec(from_node="join", from_socket="geometry", to_node="out", to_socket="geometry"),
        ],
    )

    m1 = evaluate_node_graph(graph)
    m2 = evaluate_node_graph(graph)
    assert m1.stats()["face_count"] > 200
    assert (m1.vertices == m2.vertices).all()
    assert (m1.faces == m2.faces).all()

    graph_alt = graph.model_copy(deep=True)
    graph_alt.seed = 999
    m3 = evaluate_node_graph(graph_alt)
    assert not (m1.vertices == m3.vertices).all()


def test_procedural_transforms_csg_noise_hull_and_extrude() -> None:
    graph = ProceduralGraph(
        seed=7,
        nodes=[
            NodeSpec(id="c1", type="PrimitiveCube", params={"size": [1.0, 1.0, 1.0]}),
            NodeSpec(id="c2", type="PrimitiveCube", params={"size": [0.6, 0.6, 1.6]}),
            NodeSpec(
                id="tr",
                type="TransformGeometry",
                params={"translation": [0.1, 0.0, 0.0]},
            ),
            NodeSpec(id="csg", type="BooleanGeometry", params={"operation": "difference"}),
            NodeSpec(id="sub", type="SubdivideGeometry", params={"levels": 1, "smooth": True}),
            NodeSpec(id="noise", type="SetPositionNoise", params={"strength": 0.05, "scale": 2.0}),
            NodeSpec(id="out", type="GroupOutput"),
        ],
        links=[
            LinkSpec(from_node="c2", from_socket="geometry", to_node="tr", to_socket="geometry"),
            LinkSpec(from_node="c1", from_socket="geometry", to_node="csg", to_socket="geometry_a"),
            LinkSpec(from_node="tr", from_socket="geometry", to_node="csg", to_socket="geometry_b"),
            LinkSpec(from_node="csg", from_socket="geometry", to_node="sub", to_socket="geometry"),
            LinkSpec(
                from_node="sub", from_socket="geometry", to_node="noise", to_socket="geometry"
            ),
            LinkSpec(
                from_node="noise", from_socket="geometry", to_node="out", to_socket="geometry"
            ),
        ],
    )
    mesh = evaluate_node_graph(graph)
    assert mesh.stats()["is_watertight"] is True
    assert mesh.stats()["face_count"] > 40

    # GridPoints + CurveCircle + ConvexHull
    hull_graph = ProceduralGraph(
        nodes=[
            NodeSpec(
                id="grid",
                type="GridPoints",
                params={"size_x": 2.0, "size_y": 2.0, "count_x": 3, "count_y": 3},
            ),
            NodeSpec(
                id="line",
                type="CurveLine",
                params={"start": [0, 0, 0], "end": [0, 0, 1.5], "count": 4},
            ),
            NodeSpec(id="join", type="JoinGeometry"),
            NodeSpec(id="hull", type="ConvexHull"),
            NodeSpec(
                id="ext",
                type="ExtrudeGeometry",
                params={"mode": "top", "angle_threshold_deg": 65.0, "distance": 0.2},
            ),
            NodeSpec(id="out", type="GroupOutput"),
        ],
        links=[
            LinkSpec(
                from_node="grid", from_socket="geometry", to_node="join", to_socket="geometry_a"
            ),
            LinkSpec(
                from_node="line", from_socket="geometry", to_node="join", to_socket="geometry_b"
            ),
            LinkSpec(
                from_node="join", from_socket="geometry", to_node="hull", to_socket="geometry"
            ),
            LinkSpec(from_node="hull", from_socket="geometry", to_node="ext", to_socket="geometry"),
            LinkSpec(from_node="ext", from_socket="geometry", to_node="out", to_socket="geometry"),
        ],
    )
    hull_mesh = evaluate_node_graph(hull_graph)
    assert hull_mesh.bounds().max_corner[2] > 1.5


def test_procedural_validation_errors() -> None:
    # Missing GroupOutput
    with pytest.raises(ProceduralError, match="GroupOutput"):
        evaluate_node_graph(
            ProceduralGraph(nodes=[NodeSpec(id="a", type="PrimitiveCube")], links=[])
        )

    # Unknown node type
    with pytest.raises(ProceduralError, match="Unknown procedural node type"):
        evaluate_node_graph(
            ProceduralGraph(
                nodes=[
                    NodeSpec(id="a", type="DoesNotExist"),
                    NodeSpec(id="out", type="GroupOutput"),
                ],
                links=[
                    LinkSpec(
                        from_node="a", from_socket="geometry", to_node="out", to_socket="geometry"
                    )
                ],
            )
        )

    # Cycle detection
    with pytest.raises(ProceduralError, match="Cycle"):
        evaluate_node_graph(
            ProceduralGraph(
                nodes=[
                    NodeSpec(id="t1", type="TransformGeometry"),
                    NodeSpec(id="t2", type="TransformGeometry"),
                    NodeSpec(id="out", type="GroupOutput"),
                ],
                links=[
                    LinkSpec(
                        from_node="t1", from_socket="geometry", to_node="t2", to_socket="geometry"
                    ),
                    LinkSpec(
                        from_node="t2", from_socket="geometry", to_node="t1", to_socket="geometry"
                    ),
                    LinkSpec(
                        from_node="t2", from_socket="geometry", to_node="out", to_socket="geometry"
                    ),
                ],
            )
        )

    # Missing required input
    with pytest.raises(ProceduralError, match="missing required input"):
        evaluate_node_graph(
            ProceduralGraph(
                nodes=[NodeSpec(id="out", type="GroupOutput")],
                links=[],
            )
        )
