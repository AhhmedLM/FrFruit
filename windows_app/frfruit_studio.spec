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
        "frfruit.mcp_server.tools_character",
        "frfruit.desktop.server",
        "frfruit.character.builder",
        "frfruit.character.clothing",
        "frfruit.character.delta",
        "frfruit.character.hair",
        "frfruit.character.materials",
        "frfruit.character.morphs",
        "frfruit.character.parts",
        "frfruit.character.quad_base",
        "frfruit.character.schema",
        "frfruit.character.sdf_ops",
        "frfruit.character.surface",
        "uvicorn.logging",
        "uvicorn.loops.auto",
        "uvicorn.protocols.http.auto",
        "uvicorn.protocols.websockets.auto",
        "uvicorn.protocols.websockets.websockets_impl",
        "uvicorn.lifespan.on",
        "websockets",
        "websockets.legacy",
        "websockets.legacy.server",
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
