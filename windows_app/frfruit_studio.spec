# -*- mode: python ; coding: utf-8 -*-
from pathlib import Path

ROOT = Path(SPECPATH).resolve().parent
STATIC_SRC = ROOT / "src" / "frfruit" / "desktop" / "static"

a = Analysis(
    [str(ROOT / "windows_app" / "launch_windows.py")],
    pathex=[str(ROOT / "src")],
    binaries=[],
    datas=[(str(STATIC_SRC), "frfruit/desktop/static")],
    hiddenimports=[
        "frfruit.mcp_server.server",
        "frfruit.desktop.server",
        "uvicorn.logging",
        "uvicorn.loops.auto",
        "uvicorn.protocols.http.auto",
        "uvicorn.protocols.websockets.auto",
        "uvicorn.lifespan.on",
        "webview",
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
)
pyz = PYZ(a.pure)
exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name="FrFruitStudio",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,
    disable_windowed_traceback=False,
)
