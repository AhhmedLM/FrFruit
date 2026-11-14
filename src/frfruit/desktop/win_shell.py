"""Native Windows & Cross-Platform Workstation Launcher for FrFruit Studio."""

from __future__ import annotations

import argparse
import os
import socket
import subprocess
import sys
import tempfile
import threading
import time
import urllib.request
import webbrowser
from collections.abc import Sequence
from pathlib import Path

import uvicorn

from frfruit.desktop.server import create_studio_app


def _find_free_port(preferred: int = 8765, host: str = "127.0.0.1") -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        try:
            s.bind((host, preferred))
            return preferred
        except OSError:
            s.bind((host, 0))
            return int(s.getsockname()[1])


def _wait_ready(url: str, timeout: float = 10.0) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=1.0) as resp:
                if resp.status == 200:
                    return True
        except Exception:
            time.sleep(0.1)
    return False


def _launch_windows_edge_app(url: str) -> subprocess.Popen | None:
    """Launch Edge or Chrome in clean frameless `--app=` kiosk mode on Windows."""
    if sys.platform != "win32":
        return None
    candidates = [
        Path(os.environ.get("ProgramFiles(x86)", "C:/Program Files (x86)"))
        / "Microsoft/Edge/Application/msedge.exe",
        Path(os.environ.get("ProgramFiles", "C:/Program Files"))
        / "Microsoft/Edge/Application/msedge.exe",
        Path(os.environ.get("ProgramFiles", "C:/Program Files"))
        / "Google/Chrome/Application/chrome.exe",
        Path(os.environ.get("ProgramFiles(x86)", "C:/Program Files (x86)"))
        / "Google/Chrome/Application/chrome.exe",
    ]
    for exe in candidates:
        if exe.is_file():
            profile_dir = Path(tempfile.gettempdir()) / "frfruit_studio_webview"
            return subprocess.Popen(
                [
                    str(exe),
                    f"--app={url}",
                    "--window-size=1500,920",
                    f"--user-data-dir={profile_dir}",
                ]
            )
    return None


def launch_studio(
    host: str = "127.0.0.1",
    port: int = 8765,
    workspace: str | Path | None = None,
    project_path: str | Path | None = None,
    open_window: bool = True,
) -> None:
    """Start the FrFruit Studio + Co-Hosted MCP server and attach a native window."""
    chosen_port = (
        _find_free_port(preferred=port, host=host) if port != 0 else _find_free_port(0, host)
    )
    ws_root = Path(workspace).resolve() if workspace else Path.cwd().resolve()
    app, _sess, _mcp = create_studio_app(
        workspace_root=ws_root,
        host=host,
        port=chosen_port,
        preload_project=project_path,
    )

    config = uvicorn.Config(app, host=host, port=chosen_port, log_level="warning", access_log=False)
    server = uvicorn.Server(config)

    thread = threading.Thread(target=server.run, name="frfruit-studio-uvicorn", daemon=True)
    thread.start()

    base_url = f"http://{host}:{chosen_port}"
    _wait_ready(f"{base_url}/api/mcp/info", timeout=12.0)

    print(f"FrFruit Studio Workstation online: {base_url}")
    print(f"MCP Streamable HTTP Endpoint   : {base_url}/mcp")
    print(f"MCP SSE Endpoint               : {base_url}/sse")
    print(f"Workspace Directory            : {ws_root}")

    if not open_window:
        try:
            while thread.is_alive():
                time.sleep(0.4)
        except KeyboardInterrupt:
            server.should_exit = True
        return

    # 1. Try native Edge WebView2 / pywebview desktop window
    try:
        import webview  # type: ignore[import-not-found]

        webview.create_window(
            "FrFruit Studio Workstation",
            url=base_url,
            width=1500,
            height=920,
            min_size=(1080, 680),
            background_color="#0e1013",
        )
        webview.start()
        server.should_exit = True
        return
    except ImportError:
        pass
    except Exception:
        pass

    # 2. Try native Edge/Chrome --app= desktop mode on Windows
    proc = _launch_windows_edge_app(base_url)
    if proc is not None:
        try:
            proc.wait()
        except KeyboardInterrupt:
            pass
        finally:
            server.should_exit = True
        return

    # 3. Fallback: open browser tab and serve until Ctrl+C
    webbrowser.open(base_url)
    try:
        while thread.is_alive():
            time.sleep(0.4)
    except KeyboardInterrupt:
        server.should_exit = True


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="frfruit-app",
        description="Launch FrFruit Studio Desktop Workstation + Co-Hosted Live MCP Server.",
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--workspace", default=None)
    parser.add_argument("--project", default=None, help="Optional .frf.json project file.")
    parser.add_argument("--no-window", action="store_true", help="Serve headlessly without window.")
    args = parser.parse_args(argv)
    launch_studio(
        host=args.host,
        port=args.port,
        workspace=args.workspace,
        project_path=args.project,
        open_window=not args.no_window,
    )


if __name__ == "__main__":
    main()
