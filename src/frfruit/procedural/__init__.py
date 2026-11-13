"""Procedural JSON node graph (Geometry Nodes for AI)."""

from frfruit.procedural.nodes import NODE_CATALOG, list_node_catalog
from frfruit.procedural.schema import LinkSpec, NodeSpec, ProceduralGraph, SocketType

__all__ = [
    "LinkSpec",
    "NODE_CATALOG",
    "NodeSpec",
    "ProceduralError",
    "ProceduralGraph",
    "SocketType",
    "evaluate_node_graph",
    "list_node_catalog",
    "validate_and_sort_dag",
]


def __getattr__(name: str):
    if name in ("ProceduralError", "evaluate_node_graph", "validate_and_sort_dag"):
        from frfruit.procedural import evaluator

        return getattr(evaluator, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
