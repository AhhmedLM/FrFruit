"""MCP tools for scene lifecycle, inspection, undo/redo snapshots, and collections."""

from __future__ import annotations

from collections.abc import Callable
from typing import Annotated, Any, Literal

from mcp.server import MCPServer
from mcp.server.mcpserver.exceptions import ToolError
from mcp.types import ToolAnnotations
from pydantic import Field

from frfruit.core.scene import CollectionSpec
from frfruit.geometry.spatial import (
    compute_scene_bounds,
    evaluate_object_mesh,
    evaluate_scene_meshes,
)
from frfruit.io.project import load_project, save_project
from frfruit.mcp_server.context import FrFruitSession


def register_scene_tools(mcp: MCPServer, get_session: Callable[[], FrFruitSession]) -> None:
    @mcp.tool(
        title="New Scene",
        annotations=ToolAnnotations(destructive_hint=True, open_world_hint=False),
    )
    def scene_new(
        name: Annotated[str, Field(description="Name for the new 3D scene.")] = "FrFruitScene",
        clear_history: Annotated[
            bool, Field(description="Whether to wipe undo history and snapshots.")
        ] = True,
    ) -> dict[str, Any]:
        """Reset the workspace to a clean default scene with camera and studio lighting."""
        sess = get_session()
        doc = sess.reset(name=name, clear_history=clear_history)
        return {
            "status": "ok",
            "scene_name": doc.name,
            "active_camera_id": doc.active_camera_id,
            "objects": list(doc.objects.keys()),
        }

    @mcp.tool(
        title="Open Scene Project",
        annotations=ToolAnnotations(destructive_hint=True, open_world_hint=False),
    )
    def scene_open(
        path: Annotated[str, Field(description="Path to .frf.json project file inside workspace.")],
    ) -> dict[str, Any]:
        """Load an existing .frf.json project file into the active session."""
        sess = get_session()
        with sess.lock:
            p = sess.resolve_path(path, must_exist=True)
            doc = load_project(p)
            sess.push_history(f"open:{p.name}")
            sess.scene = doc
            return {"status": "ok", "loaded_from": str(p), "summary": sess.scene_summary()}

    @mcp.tool(
        title="Save Scene Project",
        annotations=ToolAnnotations(idempotent_hint=True, open_world_hint=False),
    )
    def scene_save(
        path: Annotated[
            str, Field(description="Output .frf.json path relative to workspace.")
        ] = "scene.frf.json",
    ) -> dict[str, Any]:
        """Save the full SceneDocument (meshes, modifiers, node graphs, animation) as .frf.json."""
        sess = get_session()
        with sess.lock:
            p = sess.resolve_path(path)
            saved = save_project(sess.scene, p)
            return {"status": "ok", "path": str(saved), "bytes": saved.stat().st_size}

    @mcp.tool(
        title="Inspect Scene",
        annotations=ToolAnnotations(read_only_hint=True, open_world_hint=False),
    )
    def scene_inspect(
        detail: Annotated[
            Literal["summary", "full"],
            Field(
                description=(
                    "'summary' returns hierarchy + bounds; 'full' includes full document JSON."
                )
            ),
        ] = "summary",
        object_id: Annotated[
            str | None,
            Field(description="Optional specific object ID or name to inspect in detail."),
        ] = None,
    ) -> dict[str, Any]:
        """Inspect scene hierarchy, world bounding boxes, lights, cameras, and materials."""
        sess = get_session()
        with sess.lock:
            if object_id is not None:
                obj = sess.require_object(object_id)
                info: dict[str, Any] = obj.model_dump(exclude={"mesh"})
                if obj.kind in ("mesh", "procedural"):
                    ev = evaluate_object_mesh(sess.scene, obj.id)
                    info["evaluated_stats"] = ev.world_mesh.stats()
                    info["world_bounds"] = {
                        "min": list(ev.bounds.min_corner),
                        "max": list(ev.bounds.max_corner),
                        "center": list(ev.bounds.center),
                        "size": list(ev.bounds.size),
                    }
                return info

            if detail == "full":
                return sess.scene.model_dump()

            evaluated = evaluate_scene_meshes(sess.scene)
            bounds = compute_scene_bounds(evaluated)
            summary = sess.scene_summary()
            summary["scene_bounds"] = {
                "min": list(bounds.min_corner),
                "max": list(bounds.max_corner),
                "center": list(bounds.center),
                "size": list(bounds.size),
            }
            for o_dict in summary["objects"]:
                oid = o_dict["id"]
                if oid in evaluated:
                    ev = evaluated[oid]
                    o_dict["world_bounds"] = {
                        "min": list(ev.bounds.min_corner),
                        "max": list(ev.bounds.max_corner),
                        "center": list(ev.bounds.center),
                        "size": list(ev.bounds.size),
                    }
                    o_dict["face_count"] = int(len(ev.world_mesh.faces))
            summary["materials"] = [
                {"id": m.id, "name": m.name, "base_color": list(m.base_color), "pattern": m.pattern}
                for m in sess.scene.materials.values()
            ]
            return summary

    @mcp.tool(
        title="Undo, Redo & Snapshots",
        annotations=ToolAnnotations(open_world_hint=False),
    )
    def scene_undo_redo(
        action: Annotated[
            Literal["undo", "redo", "create_snapshot", "restore_snapshot", "list_snapshots"],
            Field(description="History action to execute."),
        ],
        snapshot_name: Annotated[
            str | None,
            Field(description="Snapshot checkpoint name for create_snapshot / restore_snapshot."),
        ] = None,
    ) -> dict[str, Any]:
        """Undo or redo scene mutations, or save/restore named checkpoints."""
        sess = get_session()
        with sess.lock:
            if action == "undo":
                res = sess.history.undo(sess.scene)
                if res is None:
                    raise ToolError("Nothing to undo.")
                sess.scene, label = res
                return {"status": "ok", "action": "undo", "reverted_label": label}

            if action == "redo":
                res = sess.history.redo(sess.scene)
                if res is None:
                    raise ToolError("Nothing to redo.")
                sess.scene, label = res
                return {"status": "ok", "action": "redo", "reapplied_label": label}

            if action == "list_snapshots":
                return {"snapshots": sess.history.list_snapshots()}

            if not snapshot_name:
                raise ToolError(f"snapshot_name is required for action={action!r}.")

            if action == "create_snapshot":
                sess.history.create_snapshot(sess.scene, snapshot_name)
                return {"status": "ok", "snapshot": snapshot_name}

            restored = sess.history.restore_snapshot(sess.scene, snapshot_name)
            if restored is None:
                avail = sess.history.list_snapshots()
                raise ToolError(f"Snapshot {snapshot_name!r} not found. Available: {avail}")
            sess.scene = restored
            return {"status": "ok", "restored_snapshot": snapshot_name}

    @mcp.tool(
        title="Manage Collections",
        annotations=ToolAnnotations(open_world_hint=False),
    )
    def collection_manage(
        action: Annotated[
            Literal["create", "assign", "set_visibility", "delete", "list"],
            Field(description="Collection management operation."),
        ],
        collection_id: Annotated[str, Field(description="Collection ID.")] = "default",
        name: Annotated[str | None, Field(description="Collection display name.")] = None,
        object_ids: Annotated[
            list[str] | None, Field(description="Objects to assign when action='assign'.")
        ] = None,
        visible: Annotated[bool, Field(description="Visibility flag.")] = True,
        parent_id: Annotated[
            str | None, Field(description="Optional parent collection ID.")
        ] = None,
    ) -> dict[str, Any]:
        """Create, assign, toggle visibility, list, or delete scene collections."""
        sess = get_session()
        with sess.lock:
            if action == "list":
                return {"collections": [c.model_dump() for c in sess.scene.collections.values()]}

            if action == "create":
                if collection_id in sess.scene.collections:
                    raise ToolError(f"Collection {collection_id!r} already exists.")
                sess.push_history(f"collection_create:{collection_id}")
                sess.scene.collections[collection_id] = CollectionSpec(
                    id=collection_id,
                    name=name or collection_id,
                    parent_id=parent_id,
                    visible=visible,
                )
                return {
                    "status": "ok",
                    "collection": sess.scene.collections[collection_id].model_dump(),
                }

            if collection_id not in sess.scene.collections:
                raise ToolError(f"Collection {collection_id!r} not found.")

            if action == "set_visibility":
                sess.push_history(f"collection_vis:{collection_id}")
                sess.scene.collections[collection_id].visible = visible
                return {"status": "ok", "collection_id": collection_id, "visible": visible}

            if action == "assign":
                if not object_ids:
                    raise ToolError("object_ids is required for action='assign'.")
                sess.push_history(f"collection_assign:{collection_id}")
                moved: list[str] = []
                for oid in object_ids:
                    obj = sess.require_object(oid)
                    obj.collection_id = collection_id
                    moved.append(obj.id)
                return {"status": "ok", "collection_id": collection_id, "assigned": moved}

            if collection_id == "default":
                raise ToolError("Cannot delete the 'default' collection.")
            sess.push_history(f"collection_delete:{collection_id}")
            del sess.scene.collections[collection_id]
            for obj in sess.scene.objects.values():
                if obj.collection_id == collection_id:
                    obj.collection_id = "default"
            return {"status": "ok", "deleted": collection_id}
