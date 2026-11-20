"""Optional cloud workspace synchronization (Hugging Face Dataset) for stateless containers."""

from __future__ import annotations

import logging
import os
import threading
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class CloudWorkspaceSync:
    """Bidirectional workspace sync with a Hugging Face Dataset repository.

    Designed for free-tier cloud containers (such as Hugging Face Spaces) whose local
    filesystem is ephemeral across restarts or sleep cycles.
    """

    def __init__(
        self,
        workspace_root: Path,
        repo_id: str,
        token: str,
        interval_sec: float = 15.0,
    ) -> None:
        self.workspace_root = Path(workspace_root).resolve()
        self.repo_id = repo_id.strip()
        self.token = token.strip()
        self.interval_sec = max(5.0, float(interval_sec))

        self._lock = threading.Lock()
        self._dirty = False
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None

    def pull_on_startup(self) -> bool:
        """Download existing workspace files from the remote Dataset repo if available."""
        try:
            from huggingface_hub import HfApi, snapshot_download  # type: ignore[import-not-found]
        except ImportError:
            logger.warning("huggingface-hub is not installed; skipping cloud workspace pull.")
            return False

        self.workspace_root.mkdir(parents=True, exist_ok=True)
        api = HfApi(token=self.token)
        try:
            api.create_repo(
                repo_id=self.repo_id,
                repo_type="dataset",
                private=True,
                exist_ok=True,
            )
        except Exception as exc:
            logger.warning("Could not verify/create HF dataset %s: %s", self.repo_id, exc)

        try:
            snapshot_download(
                repo_id=self.repo_id,
                repo_type="dataset",
                local_dir=str(self.workspace_root),
                token=self.token,
                ignore_patterns=[".git*", "*.tmp"],
            )
            logger.info("Pulled workspace from HF dataset %s", self.repo_id)
            return True
        except Exception as exc:
            logger.warning("Cloud workspace pull skipped/failed (%s): %s", self.repo_id, exc)
            return False

    def mark_dirty(self) -> None:
        """Mark the local workspace as having unpushed changes."""
        with self._lock:
            self._dirty = True

    def sync_now(self) -> bool:
        """Upload local workspace contents to the remote Dataset repo."""
        try:
            from huggingface_hub import HfApi  # type: ignore[import-not-found]
        except ImportError:
            return False

        if not self.workspace_root.is_dir():
            return False

        with self._lock:
            self._dirty = False

        try:
            api = HfApi(token=self.token)
            api.upload_folder(
                folder_path=str(self.workspace_root),
                repo_id=self.repo_id,
                repo_type="dataset",
                commit_message="Sync FrFruit workspace",
                ignore_patterns=[".git*", "__pycache__/*", "*.tmp", "*.pyc"],
            )
            logger.info("Synced workspace to HF dataset %s", self.repo_id)
            return True
        except Exception as exc:
            logger.warning("Cloud workspace sync failed (%s): %s", self.repo_id, exc)
            with self._lock:
                self._dirty = True
            return False

    def _run_loop(self) -> None:
        while not self._stop_event.wait(self.interval_sec):
            with self._lock:
                should_sync = self._dirty
            if should_sync:
                self.sync_now()

    def start(self) -> None:
        """Start background debounced sync thread."""
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._run_loop,
            name="frfruit-cloud-sync",
            daemon=True,
        )
        self._thread.start()

    def stop(self) -> None:
        """Flush any pending changes and stop the background sync thread."""
        self._stop_event.set()
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=5.0)
        with self._lock:
            should_sync = self._dirty
        if should_sync:
            self.sync_now()


def create_cloud_sync_from_env(
    workspace_root: Path,
    env: dict[str, str] | Any = os.environ,
) -> CloudWorkspaceSync | None:
    """Create a CloudWorkspaceSync instance if HF_DATASET_REPO and HF_TOKEN are configured."""
    repo_id = str(env.get("HF_DATASET_REPO") or env.get("FRFRUIT_HF_DATASET") or "").strip()
    token = str(env.get("HF_TOKEN") or "").strip()
    if not repo_id or not token:
        return None
    interval_raw = str(env.get("FRFRUIT_SYNC_INTERVAL") or "15").strip()
    try:
        interval = float(interval_raw)
    except ValueError:
        interval = 15.0
    return CloudWorkspaceSync(
        workspace_root=workspace_root,
        repo_id=repo_id,
        token=token,
        interval_sec=interval,
    )
