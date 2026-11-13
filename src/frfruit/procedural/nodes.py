"""Built-in procedural node catalog metadata and definitions."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from frfruit.procedural.schema import SocketType


@dataclass(frozen=True)
class SocketDef:
    name: str
    socket_type: SocketType
    required: bool = True


@dataclass(frozen=True)
class NodeDef:
    type: str
    category: str
    description: str
    inputs: tuple[SocketDef, ...] = ()
    outputs: tuple[SocketDef, ...] = ()
    default_params: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": self.type,
            "category": self.category,
            "description": self.description,
            "inputs": [
                {"name": s.name, "type": s.socket_type.value, "required": s.required}
                for s in self.inputs
            ],
            "outputs": [{"name": s.name, "type": s.socket_type.value} for s in self.outputs],
            "default_params": self.default_params,
        }


NODE_CATALOG: dict[str, NodeDef] = {
    "PrimitiveCube": NodeDef(
        type="PrimitiveCube",
        category="generator",
        description="Generate a box/cube mesh.",
        outputs=(SocketDef("geometry", SocketType.GEOMETRY),),
        default_params={"size": [1.0, 1.0, 1.0], "origin": "center"},
    ),
    "PrimitiveSphere": NodeDef(
        type="PrimitiveSphere",
        category="generator",
        description="Generate a UV sphere mesh.",
        outputs=(SocketDef("geometry", SocketType.GEOMETRY),),
        default_params={"radius": 0.5, "segments": 24},
    ),
    "PrimitiveCylinder": NodeDef(
        type="PrimitiveCylinder",
        category="generator",
        description="Generate a cylinder or frustum mesh.",
        outputs=(SocketDef("geometry", SocketType.GEOMETRY),),
        default_params={"radius": 0.5, "height": 1.0, "radius_top": None, "segments": 24},
    ),
    "PrimitiveCone": NodeDef(
        type="PrimitiveCone",
        category="generator",
        description="Generate a cone mesh.",
        outputs=(SocketDef("geometry", SocketType.GEOMETRY),),
        default_params={"radius": 0.5, "height": 1.0, "segments": 24},
    ),
    "PrimitiveTorus": NodeDef(
        type="PrimitiveTorus",
        category="generator",
        description="Generate a torus mesh.",
        outputs=(SocketDef("geometry", SocketType.GEOMETRY),),
        default_params={
            "major_radius": 0.5,
            "minor_radius": 0.15,
            "major_segments": 32,
            "minor_segments": 16,
        },
    ),
    "GridPoints": NodeDef(
        type="GridPoints",
        category="points",
        description="Generate an ordered 2D grid of points with +Z normals.",
        outputs=(SocketDef("geometry", SocketType.GEOMETRY),),
        default_params={"size_x": 2.0, "size_y": 2.0, "count_x": 5, "count_y": 5},
    ),
    "CurveCircle": NodeDef(
        type="CurveCircle",
        category="points",
        description="Generate points arranged in a circle with outward radial normals.",
        outputs=(SocketDef("geometry", SocketType.GEOMETRY),),
        default_params={"radius": 1.0, "segments": 12},
    ),
    "CurveLine": NodeDef(
        type="CurveLine",
        category="points",
        description="Generate evenly spaced points along a 3D line segment.",
        outputs=(SocketDef("geometry", SocketType.GEOMETRY),),
        default_params={"start": [0.0, 0.0, 0.0], "end": [0.0, 0.0, 2.0], "count": 8},
    ),
    "TransformGeometry": NodeDef(
        type="TransformGeometry",
        category="transform",
        description="Translate, rotate (degrees), and scale geometry or points.",
        inputs=(SocketDef("geometry", SocketType.GEOMETRY),),
        outputs=(SocketDef("geometry", SocketType.GEOMETRY),),
        default_params={
            "translation": [0.0, 0.0, 0.0],
            "rotation_deg": [0.0, 0.0, 0.0],
            "scale": [1.0, 1.0, 1.0],
        },
    ),
    "JoinGeometry": NodeDef(
        type="JoinGeometry",
        category="combine",
        description="Combine up to 4 geometry inputs into one mesh/point set.",
        inputs=(
            SocketDef("geometry_a", SocketType.GEOMETRY, required=True),
            SocketDef("geometry_b", SocketType.GEOMETRY, required=False),
            SocketDef("geometry_c", SocketType.GEOMETRY, required=False),
            SocketDef("geometry_d", SocketType.GEOMETRY, required=False),
        ),
        outputs=(SocketDef("geometry", SocketType.GEOMETRY),),
    ),
    "BooleanGeometry": NodeDef(
        type="BooleanGeometry",
        category="csg",
        description="Perform watertight CSG union, difference, or intersection.",
        inputs=(
            SocketDef("geometry_a", SocketType.GEOMETRY),
            SocketDef("geometry_b", SocketType.GEOMETRY),
        ),
        outputs=(SocketDef("geometry", SocketType.GEOMETRY),),
        default_params={"operation": "union"},
    ),
    "SubdivideGeometry": NodeDef(
        type="SubdivideGeometry",
        category="mesh",
        description="Subdivide input mesh (1..3 levels) with optional Loop smoothing.",
        inputs=(SocketDef("geometry", SocketType.GEOMETRY),),
        outputs=(SocketDef("geometry", SocketType.GEOMETRY),),
        default_params={"levels": 1, "smooth": True},
    ),
    "ConvexHull": NodeDef(
        type="ConvexHull",
        category="mesh",
        description="Compute the convex hull of input mesh vertices or points.",
        inputs=(SocketDef("geometry", SocketType.GEOMETRY),),
        outputs=(SocketDef("geometry", SocketType.GEOMETRY),),
    ),
    "ExtrudeGeometry": NodeDef(
        type="ExtrudeGeometry",
        category="mesh",
        description="Semantically select faces and extrude them along average normal.",
        inputs=(SocketDef("geometry", SocketType.GEOMETRY),),
        outputs=(SocketDef("geometry", SocketType.GEOMETRY),),
        default_params={"mode": "top", "angle_threshold_deg": 45.0, "distance": 0.4, "scale": 1.0},
    ),
    "SetPositionNoise": NodeDef(
        type="SetPositionNoise",
        category="deform",
        description="Displace mesh vertices with deterministic 3D procedural wave noise.",
        inputs=(SocketDef("geometry", SocketType.GEOMETRY),),
        outputs=(SocketDef("geometry", SocketType.GEOMETRY),),
        default_params={
            "strength": 0.08,
            "scale": 3.0,
            "along_normals": True,
            "seed": None,
        },
    ),
    "DistributePointsOnFaces": NodeDef(
        type="DistributePointsOnFaces",
        category="scatter",
        description="Area-weighted deterministic point scattering on input mesh faces.",
        inputs=(SocketDef("geometry", SocketType.GEOMETRY),),
        outputs=(SocketDef("geometry", SocketType.GEOMETRY),),
        default_params={"count": 32, "min_distance": 0.0, "seed": None},
    ),
    "InstanceOnPoints": NodeDef(
        type="InstanceOnPoints",
        category="scatter",
        description="Copy instance geometry onto every point in the points input.",
        inputs=(
            SocketDef("points", SocketType.GEOMETRY),
            SocketDef("instance", SocketType.GEOMETRY),
        ),
        outputs=(SocketDef("geometry", SocketType.GEOMETRY),),
        default_params={
            "scale": 1.0,
            "scale_min": 0.85,
            "scale_max": 1.15,
            "align_to_normal": True,
            "random_rotation_deg": [0.0, 0.0, 360.0],
            "seed": None,
        },
    ),
    "GroupOutput": NodeDef(
        type="GroupOutput",
        category="output",
        description="Final output geometry sink for the ProceduralGraph.",
        inputs=(SocketDef("geometry", SocketType.GEOMETRY),),
        outputs=(),
    ),
}


def list_node_catalog() -> list[dict[str, Any]]:
    """Return structured definitions of every supported procedural node."""
    return [NODE_CATALOG[k].to_dict() for k in sorted(NODE_CATALOG.keys())]
