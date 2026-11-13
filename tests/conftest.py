"""Shared pytest fixtures for FrFruit tests."""

from __future__ import annotations

from pathlib import Path

import pytest
from mcp import Client

from frfruit.core.scene import SceneDocument
from frfruit.mcp_server.server import create_server


@pytest.fixture
def empty_scene() -> SceneDocument:
    return SceneDocument.new_default("TestScene")


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


@pytest.fixture
async def mcp_session_and_client(tmp_path: Path):
    server, session = create_server(workspace_root=tmp_path)
    async with Client(server, raise_exceptions=True) as client:
        yield session, client, tmp_path
