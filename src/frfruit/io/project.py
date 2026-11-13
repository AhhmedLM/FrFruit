"""Native .frf.json project persistence."""

from __future__ import annotations

from pathlib import Path

from frfruit.core.scene import SceneDocument


def save_project(scene: SceneDocument, path: str | Path) -> Path:
    """Save a SceneDocument atomically to a .frf.json file."""
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = out_path.with_suffix(out_path.suffix + ".tmp")
    tmp_path.write_text(scene.model_dump_json(indent=2), encoding="utf-8")
    tmp_path.replace(out_path)
    return out_path


def load_project(path: str | Path) -> SceneDocument:
    """Load and validate a .frf.json SceneDocument."""
    in_path = Path(path)
    if not in_path.exists():
        raise FileNotFoundError(f"Project file not found: {in_path}")
    return SceneDocument.model_validate_json(in_path.read_text(encoding="utf-8"))
