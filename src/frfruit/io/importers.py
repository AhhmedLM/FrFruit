"""Mesh/scene importers for GLB, GLTF, OBJ, STL, and PLY."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import trimesh

from frfruit.core.math3d import decompose_matrix
from frfruit.core.scene import MaterialSpec, SceneDocument, SceneObject
from frfruit.core.types import Transform
from frfruit.geometry.mesh import MeshData


def import_mesh_file(
    scene: SceneDocument,
    file_path: str | Path,
    name_prefix: str | None = None,
    collection_id: str = "default",
) -> list[SceneObject]:
    """Import a .glb/.gltf/.obj/.stl/.ply file into `scene` as SceneObjects."""
    p = Path(file_path)
    if not p.exists():
        raise FileNotFoundError(f"Mesh file not found: {p}")

    loaded = trimesh.load(str(p), force=None)
    created: list[SceneObject] = []

    if isinstance(loaded, trimesh.Trimesh):
        md = MeshData.from_trimesh(loaded)
        base_name = scene.unique_name(name_prefix or p.stem)
        oid = scene.next_id(base_name)
        obj = SceneObject(
            id=oid,
            name=base_name,
            kind="mesh",
            collection_id=collection_id if collection_id in scene.collections else "default",
            mesh=md.to_payload(),
        )
        scene.objects[oid] = obj
        created.append(obj)
        return created

    if isinstance(loaded, trimesh.Scene):
        for node_name in loaded.graph.nodes_geometry:
            mat4, geom_name = loaded.graph[node_name]
            geom = loaded.geometry.get(geom_name)
            if not isinstance(geom, trimesh.Trimesh) or len(geom.faces) == 0:
                continue
            md = MeshData.from_trimesh(geom)
            loc, rot_deg, scale = decompose_matrix(np.asarray(mat4, dtype=np.float64))
            # Guard zero scales
            scale = tuple(s if abs(s) > 1e-6 else 1.0 for s in scale)  # type: ignore[assignment]
            base = f"{name_prefix}_{node_name}" if name_prefix else node_name
            obj_name = scene.unique_name(base)
            oid = scene.next_id(obj_name)

            mat_id: str | None = None
            pbr = getattr(geom.visual, "material", None)
            if (
                pbr is not None
                and hasattr(pbr, "baseColorFactor")
                and pbr.baseColorFactor is not None
            ):
                cf = [
                    float(x) / 255.0 if max(pbr.baseColorFactor) > 1.0 else float(x)
                    for x in pbr.baseColorFactor[:4]
                ]
                mat_id = scene.next_id(f"mat_{obj_name}")
                scene.materials[mat_id] = MaterialSpec(
                    id=mat_id,
                    name=f"{obj_name} Material",
                    base_color=(cf[0], cf[1], cf[2], cf[3] if len(cf) == 4 else 1.0),
                    metallic=float(getattr(pbr, "metallicFactor", 0.0) or 0.0),
                    roughness=float(getattr(pbr, "roughnessFactor", 0.5) or 0.5),
                )

            obj = SceneObject(
                id=oid,
                name=obj_name,
                kind="mesh",
                transform=Transform(location=loc, rotation_deg=rot_deg, scale=scale),
                collection_id=collection_id if collection_id in scene.collections else "default",
                material_id=mat_id,
                mesh=md.to_payload(),
            )
            scene.objects[oid] = obj
            created.append(obj)
        return created

    raise ValueError(f"Unsupported loaded object type from {p}: {type(loaded).__name__}")
