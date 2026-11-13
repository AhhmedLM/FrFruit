"""JSON-serializable procedural node graph schema (Geometry Nodes for AI)."""

from __future__ import annotations

from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field


class SocketType(StrEnum):
    GEOMETRY = "geometry"
    FLOAT = "float"
    INT = "int"
    VECTOR = "vector"
    BOOL = "bool"


class NodeSpec(BaseModel):
    """A single node inside a ProceduralGraph DAG."""

    id: str = Field(min_length=1)
    type: str = Field(min_length=1)
    params: dict[str, Any] = Field(default_factory=dict)


class LinkSpec(BaseModel):
    """Directed connection between two node sockets."""

    from_node: str
    from_socket: str
    to_node: str
    to_socket: str


class ProceduralGraph(BaseModel):
    """Serializable DAG describing procedural geometry generation."""

    nodes: list[NodeSpec] = Field(default_factory=list)
    links: list[LinkSpec] = Field(default_factory=list)
    seed: int = 42
