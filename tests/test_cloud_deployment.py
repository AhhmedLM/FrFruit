"""Tests for cloud MCP deployment features (healthcheck, Bearer auth, host security, sync)."""

from __future__ import annotations

from pathlib import Path

from starlette.testclient import TestClient

from frfruit.desktop.server import create_studio_app
from frfruit.io.cloud_sync import create_cloud_sync_from_env


def test_health_endpoint_and_bearer_auth(tmp_path: Path) -> None:
    app, _sess, _mcp = create_studio_app(
        workspace_root=tmp_path,
        host="0.0.0.0",
        port=7860,
        api_key="super-secret-token",
    )
    with TestClient(app) as client:
        # /health must remain accessible without auth for container healthchecks
        health = client.get("/health")
        assert health.status_code == 200
        assert health.json() == {"status": "ok"}

        # Protected routes reject missing/invalid tokens
        assert client.get("/api/mcp/info").status_code == 401
        assert (
            client.get("/api/mcp/info", headers={"Authorization": "Bearer wrong-token"}).status_code
            == 401
        )
        mcp_unauth = client.post("/mcp", json={"jsonrpc": "2.0", "id": 1, "method": "initialize"})
        assert mcp_unauth.status_code == 401

        # Valid Bearer, X-API-Key, or ?token= succeeds
        ok_bearer = client.get(
            "/api/mcp/info", headers={"Authorization": "Bearer super-secret-token"}
        )
        assert ok_bearer.status_code == 200
        assert ok_bearer.json()["status"] == "online"

        ok_header = client.get("/api/mcp/info", headers={"X-API-Key": "super-secret-token"})
        assert ok_header.status_code == 200

        ok_query = client.get("/api/mcp/info?token=super-secret-token")
        assert ok_query.status_code == 200


def test_cloud_host_header_allowed_for_hf_spaces(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("SPACE_HOST", "user-frfruit.hf.space")
    app, _sess, _mcp = create_studio_app(
        workspace_root=tmp_path,
        host="0.0.0.0",
        port=7860,
    )
    with TestClient(app, base_url="https://user-frfruit.hf.space") as client:
        resp = client.post(
            "/mcp",
            headers={
                "Accept": "application/json, text/event-stream",
                "Content-Type": "application/json",
            },
            json={
                "jsonrpc": "2.0",
                "id": 1,
                "method": "initialize",
                "params": {
                    "protocolVersion": "2025-03-26",
                    "capabilities": {},
                    "clientInfo": {"name": "cloud-test", "version": "1.0"},
                },
            },
        )
        assert resp.status_code == 200


def test_cloud_sync_env_config(tmp_path: Path) -> None:
    assert create_cloud_sync_from_env(tmp_path, env={}) is None
    assert create_cloud_sync_from_env(tmp_path, env={"HF_DATASET_REPO": "user/ds"}) is None
    sync = create_cloud_sync_from_env(
        tmp_path,
        env={"HF_DATASET_REPO": "user/ds", "HF_TOKEN": "hf_123", "FRFRUIT_SYNC_INTERVAL": "20"},
    )
    assert sync is not None
    assert sync.repo_id == "user/ds"
    assert sync.interval_sec == 20.0
