"""MCP tools for JSON Procedural Node Graphs (Geometry Nodes for AI)."""

from __future__ import annotations

from collections.abc import Callable
from typing import Annotated, Any

from mcp.server import MCPServer
from mcp.server.mcpserver.exceptions import ToolError
from mcp.types import ToolAnnotations
from pydantic import Field, ValidationError

from frfruit.core.scene import SceneObject
from frfruit.core.types import Transform, Vec3
from frfruit.geometry.spatial import evaluate_object_mesh
from frfruit.mcp_server.context import FrFruitSession
from frfruit.procedural.evaluator import ProceduralError, evaluate_node_graph
from frfruit.procedural.nodes import list_node_catalog
from frfruit.procedural.schema import LinkSpec, NodeSpec, ProceduralGraph


def register_node_tools(mcp: MCPServer, get_session: Callable[[], FrFruitSession]) -> None:
    @mcp.tool(
        title="List Procedural Node Catalog",
        annotations=ToolAnnotations(read_only_hint=True, open_world_hint=False),
    )
    def node_catalog_list(
        category: Annotated[
            str | None,
            Field(
                description=(
                    "Optional category filter (generator, points, transform, combine, "
                    "csg, mesh, deform, scatter, output)."
                )
            ),
        ] = None,
    ) -> dict[str, Any]:
        """Return schemas, sockets, and default parameters for all procedural nodes."""
        items = list_node_catalog()
        if category:
            items = [n for n in items if n["category"] == category]
        return {"nodes": items}

    @mcp.tool(
        title="Set Procedural Node Graph",
        annotations=ToolAnnotations(open_world_hint=False),
    )
    def node_graph_set(
        object_id_or_name: Annotated[
            str,
            Field(
                description=(
                    "Existing object ID/name to convert, or name for a new procedural object."
                )
            ),
        ],
        nodes: Annotated[
            list[dict[str, Any]],
            Field(description="List of nodes: {id, type, params} (must include one GroupOutput)."),
        ],
        links: Annotated[
            list[dict[str, Any]],
            Field(description="List of links: {from_node, from_socket, to_node, to_socket}."),
        ],
        seed: Annotated[
            int, Field(description="Deterministic random seed for noise and scattering.")
        ] = 42,
        location: Annotated[
            Vec3, Field(description="World/local XYZ position when creating new object.")
        ] = (0.0, 0.0, 0.0),
        material_id: Annotated[
            str | None, Field(description="Optional material ID or name.")
        ] = None,
        create_if_missing: Annotated[
            bool, Field(description="Create procedural object if not found.")
        ] = True,
    ) -> dict[str, Any]:
        """Define or replace a JSON ProceduralGraph DAG on an object and evaluate it."""
        sess = get_session()
        with sess.lock:
            try:
                graph = ProceduralGraph(
                    nodes=[NodeSpec.model_validate(n) for n in nodes],
                    links=[LinkSpec.model_validate(lnk) for lnk in links],
                    seed=seed,
                )
                mesh = evaluate_node_graph(graph)
            except (ValidationError, ProceduralError) as exc:
                raise ToolError(f"ProceduralGraph validation/evaluation failed: {exc}") from exc

            existing = sess.scene.resolve_object(object_id_or_name)
            mat_resolved = sess.scene.resolve_material(material_id) if material_id else None

            if existing is not None:
                sess.push_history(f"node_graph_set:{existing.name}")
                existing.kind = "procedural"
                existing.node_graph = graph
                existing.mesh = None
                if mat_resolved:
                    existing.material_id = mat_resolved.id
                obj = existing
            else:
                if not create_if_missing:
                    raise ToolError(f"Object {object_id_or_name!r} not found.")
                obj_name = sess.scene.unique_name(object_id_or_name)
                oid = sess.scene.next_id(obj_name.lower())
                sess.push_history(f"node_graph_create:{obj_name}")
                obj = SceneObject(
                    id=oid,
                    name=obj_name,
                    kind="procedural",
                    transform=Transform(location=location),
                    material_id=mat_resolved.id if mat_resolved else None,
                    node_graph=graph,
                )
                sess.scene.objects[oid] = obj

            ev = evaluate_object_mesh(sess.scene, obj.id)
            return {
                "status": "ok",
                "object_id": obj.id,
                "name": obj.name,
                "node_count": len(graph.nodes),
                "link_count": len(graph.links),
                "stats": ev.world_mesh.stats() if not mesh.is_empty else {},
            }

    @mcp.tool(
        title="Patch Procedural Node Graph",
        annotations=ToolAnnotations(open_world_hint=False),
    )
    def node_graph_patch(
        object_id: Annotated[str, Field(description="Procedural object ID or name.")],
        upsert_nodes: Annotated[
            list[dict[str, Any]] | None,
            Field(description="Nodes to add or update parameters on (existing params are merged)."),
        ] = None,
        remove_node_ids: Annotated[
            list[str] | None, Field(description="Node IDs to delete.")
        ] = None,
        add_links: Annotated[
            list[dict[str, Any]] | None, Field(description="Links to add.")
        ] = None,
        remove_links: Annotated[
            list[dict[str, Any]] | None, Field(description="Links to remove.")
        ] = None,
        seed: Annotated[int | None, Field(description="Updated graph seed.")] = None,
    ) -> dict[str, Any]:
        """Incrementally modify nodes, parameters, or links in an object's ProceduralGraph."""
        sess = get_session()
        with sess.lock:
            obj = sess.require_object(object_id, expected_kind="procedural")
            if obj.node_graph is None:
                raise ToolError(f"Object {obj.name!r} does not have a ProceduralGraph.")

            candidate = obj.node_graph.model_copy(deep=True)
            node_map = {n.id: n for n in candidate.nodes}

            for nid in remove_node_ids or []:
                node_map.pop(nid, None)
                candidate.links = [
                    lnk for lnk in candidate.links if lnk.from_node != nid and lnk.to_node != nid
                ]

            for raw_n in upsert_nodes or []:
                try:
                    spec = NodeSpec.model_validate(raw_n)
                except ValidationError as exc:
                    raise ToolError(f"Invalid node spec: {exc}") from exc
                if spec.id in node_map and node_map[spec.id].type == spec.type:
                    merged_params = {**node_map[spec.id].params, **spec.params}
                    node_map[spec.id] = NodeSpec(id=spec.id, type=spec.type, params=merged_params)
                else:
                    node_map[spec.id] = spec

            candidate.nodes = list(node_map.values())

            if remove_links:
                rem_tuples = {
                    (lnk["from_node"], lnk["from_socket"], lnk["to_node"], lnk["to_socket"])
                    for lnk in remove_links
                }
                candidate.links = [
                    lnk
                    for lnk in candidate.links
                    if (lnk.from_node, lnk.from_socket, lnk.to_node, lnk.to_socket)
                    not in rem_tuples
                ]

            for raw_l in add_links or []:
                try:
                    candidate.links.append(LinkSpec.model_validate(raw_l))
                except ValidationError as exc:
                    raise ToolError(f"Invalid link spec: {exc}") from exc

            if seed is not None:
                candidate.seed = seed

            try:
                evaluate_node_graph(candidate)
            except ProceduralError as exc:
                raise ToolError(f"Patched ProceduralGraph failed evaluation: {exc}") from exc

            sess.push_history(f"node_graph_patch:{obj.name}")
            obj.node_graph = candidate
            ev = evaluate_object_mesh(sess.scene, obj.id)
            return {
                "status": "ok",
                "object_id": obj.id,
                "node_count": len(candidate.nodes),
                "stats": ev.world_mesh.stats(),
            }
