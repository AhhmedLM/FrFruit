"""Undo/Redo stack and named checkpoint snapshots for SceneDocument."""

from __future__ import annotations

from dataclasses import dataclass

from frfruit.core.scene import SceneDocument


@dataclass
class HistoryEntry:
    label: str
    snapshot_json: str


class HistoryManager:
    """Manages bounded undo/redo history and named checkpoints."""

    def __init__(self, max_entries: int = 50) -> None:
        self.max_entries = max_entries
        self._undo_stack: list[HistoryEntry] = []
        self._redo_stack: list[HistoryEntry] = []
        self._snapshots: dict[str, str] = {}

    @property
    def can_undo(self) -> bool:
        return bool(self._undo_stack)

    @property
    def can_redo(self) -> bool:
        return bool(self._redo_stack)

    def clear(self) -> None:
        self._undo_stack.clear()
        self._redo_stack.clear()
        self._snapshots.clear()

    def push(self, doc: SceneDocument, label: str) -> None:
        """Record state before applying a mutation."""
        self._undo_stack.append(HistoryEntry(label=label, snapshot_json=doc.model_dump_json()))
        if len(self._undo_stack) > self.max_entries:
            self._undo_stack.pop(0)
        self._redo_stack.clear()

    def undo(self, current_doc: SceneDocument) -> tuple[SceneDocument, str] | None:
        """Revert to the most recent recorded state."""
        if not self._undo_stack:
            return None
        entry = self._undo_stack.pop()
        self._redo_stack.append(
            HistoryEntry(label=entry.label, snapshot_json=current_doc.model_dump_json())
        )
        restored = SceneDocument.model_validate_json(entry.snapshot_json)
        return restored, entry.label

    def redo(self, current_doc: SceneDocument) -> tuple[SceneDocument, str] | None:
        """Re-apply a previously undone state."""
        if not self._redo_stack:
            return None
        entry = self._redo_stack.pop()
        self._undo_stack.append(
            HistoryEntry(label=entry.label, snapshot_json=current_doc.model_dump_json())
        )
        restored = SceneDocument.model_validate_json(entry.snapshot_json)
        return restored, entry.label

    def create_snapshot(self, doc: SceneDocument, name: str) -> None:
        """Save a named checkpoint that survives undo/redo."""
        self._snapshots[name] = doc.model_dump_json()

    def restore_snapshot(self, current_doc: SceneDocument, name: str) -> SceneDocument | None:
        """Restore a named checkpoint and push current state onto undo stack."""
        raw = self._snapshots.get(name)
        if raw is None:
            return None
        self.push(current_doc, f"restore_snapshot:{name}")
        return SceneDocument.model_validate_json(raw)

    def list_snapshots(self) -> list[str]:
        return sorted(self._snapshots.keys())

    def status(self) -> dict[str, object]:
        return {
            "undo_depth": len(self._undo_stack),
            "redo_depth": len(self._redo_stack),
            "undo_labels": [e.label for e in self._undo_stack[-10:]],
            "snapshots": self.list_snapshots(),
        }
