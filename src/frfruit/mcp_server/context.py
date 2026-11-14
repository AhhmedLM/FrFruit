"""Stateful FrFruit MCP workspace session, thread-safe mutations, and path safety."""

from __future__ import annotations

import os
import threading
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from mcp.server.mcpserver.exceptions import ToolError

from frfruit.core.history import HistoryManager
from frfruit.core.scene import SceneDocument, SceneObject


class FrFruitSession:
    """Thread-safe active scene session with Undo/Redo, workspace sandboxing, and live listeners."""

    def __init__(self, workspace_root: str | Path | None = None) -> None:
        self._lock = threading.RLock()
        root_env = os.environ.get("FRFRUIT_WORKSPACE")
        base = workspace_root or root_env or Path.cwd()
        self.workspace_root: Path = Path(base).resolve()
        self.scene: SceneDocument = SceneDocument.new_default("FrFruitScene")
        self.history: HistoryManager = HistoryManager(max_entries=50)
        self._listeners: list[Callable[[dict[str, Any]], None]] = []
        self.audit_log: list[dict[str, Any]] = []

    def add_listener(self, cb: Callable[[dict[str, Any]], None]) -> None:
        with self._lock:
            if cb not in self._listeners:
                self._listeners.append(cb)

    def remove_listener(self, cb: Callable[[dict[str, Any]], None]) -> None:
        with self._lock:
            self._listeners = [x for x in self._listeners if x is not cb]

    def notify(self, event: str, data: dict[str, Any] | None = None) -> None:
        payload: dict[str, Any] = {
            "event": event,
            "timestamp": time.time(),
            "scene_name": self.scene.name,
            "object_count": len(self.scene.objects),
            "data": data or {},
        }
        with self._lock:
            self.audit_log.append(payload)
            if len(self.audit_log) > 200:
                self.audit_log = self.audit_log[-200:]
            listeners = list(self._listeners)
        for cb in listeners:
            try:
                cb(payload)
            except Exception:
                pass

    def reset(self, name: str = "FrFruitScene", clear_history: bool = True) -> SceneDocument:
        with self._lock:
            if not clear_history:
                self.history.push(self.scene, "scene_new")
            else:
                self.history.clear()
            self.scene = SceneDocument.new_default(name)
            scene_out = self.scene
        self.notify("scene_reset", {"name": name})
        return scene_out

    def set_workspace_root(self, root: str | Path) -> Path:
        with self._lock:
            self.workspace_root = Path(root).resolve()
            return self.workspace_root

    def resolve_path(self, user_path: str | Path, must_exist: bool = False) -> Path:
        """Resolve path inside workspace_root, rejecting traversal outside it."""
        raw = Path(user_path)
        candidate = (raw if raw.is_absolute() else (self.workspace_root / raw)).resolve()
        try:
            candidate.relative_to(self.workspace_root)
        except ValueError as exc:
            raise ToolError(
                f"Path {str(user_path)!r} resolves outside workspace root "
                f"{str(self.workspace_root)!r}."
            ) from exc
        if must_exist and not candidate.exists():
            raise ToolError(f"File not found: {candidate}")
        return candidate

    def push_history(self, label: str) -> None:
        self.history.push(self.scene, label)
        self.notify("scene_mutated", {"action": label})

    def require_object(self, id_or_name: str, expected_kind: str | None = None) -> SceneObject:
        obj = self.scene.resolve_object(id_or_name)
        if obj is None:
            available = [f"{o.id} ({o.name}, {o.kind})" for o in self.scene.objects.values()]
            raise ToolError(f"Object {id_or_name!r} not found. Available objects: {available[:25]}")
        if expected_kind is not None and obj.kind != expected_kind:
            raise ToolError(
                f"Object {obj.name!r} has kind={obj.kind!r}, expected {expected_kind!r}."
            )
        return obj

    def require_mesh_like(self, id_or_name: str) -> SceneObject:
        obj = self.require_object(id_or_name)
        if obj.kind not in ("mesh", "procedural"):
            raise ToolError(
                f"Object {obj.name!r} ({obj.kind}) does not carry mesh/procedural geometry."
            )
        return obj

    def scene_summary(self) -> dict[str, Any]:
        with self._lock:
            objs = [
                {
                    "id": o.id,
                    "name": o.name,
                    "kind": o.kind,
                    "visible": self.scene.is_object_visible(o),
                    "location": list(o.transform.location),
                    "rotation_deg": list(o.transform.rotation_deg),
                    "scale": list(o.transform.scale),
                    "material_id": o.material_id,
                    "modifiers": [m.type for m in o.modifiers],
                    "parent_id": o.parent_id,
                    "collection_id": o.collection_id,
                }
                for o in self.scene.objects.values()
            ]
            return {
                "name": self.scene.name,
                "active_camera_id": self.scene.active_camera_id,
                "object_count": len(self.scene.objects),
                "material_count": len(self.scene.materials),
                "collection_count": len(self.scene.collections),
                "timeline": {
                    "fps": self.scene.timeline.fps,
                    "frame_start": self.scene.timeline.frame_start,
                    "frame_end": self.scene.timeline.frame_end,
                    "current_frame": self.scene.timeline.current_frame,
                    "track_count": len(self.scene.timeline.tracks),
                },
                "history": self.history.status(),
                "objects": objs,
            }

    @property
    def lock(self) -> threading.RLock:
        return self._lock
