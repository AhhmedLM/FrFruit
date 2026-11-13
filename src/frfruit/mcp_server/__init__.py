"""MCP server interface for FrFruit."""

from frfruit.mcp_server.context import FrFruitSession
from frfruit.mcp_server.server import create_server, default_session, mcp

__all__ = [
    "FrFruitSession",
    "create_server",
    "default_session",
    "mcp",
]
