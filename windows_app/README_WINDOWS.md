# FrFruit Studio — Windows Desktop App & Live MCP Connector

## Quick Launch on Windows 10 / 11

### Option 1: Double-Click Batch Launcher
Run:
```bat
windows_app\FrFruitStudio.bat
```
Or from PowerShell / Command Prompt:
```bat
uv run frfruit app --workspace C:\path\to\your\3d_projects
```
For the cleanest native window experience (Edge WebView2 embedded inside a desktop frame), optionally install `pywebview`:
```bat
uv sync --extra desktop
```
Even without `pywebview`, `FrFruitStudio` automatically launches Microsoft Edge or Google Chrome in frameless `--app=` workstation window mode backed by the co-hosted local server.

### Option 2: Compile Standalone `FrFruitStudio.exe`
```powershell
powershell -ExecutionPolicy Bypass -File windows_app\build_windows_exe.ps1
```
Output binary: `dist\FrFruitStudio.exe`.

## Connecting AI Agents via Live MCP
While FrFruit Studio is running, click the **MCP** tab in the right inspector dock or **MCP READY / MCP CONNECTED** in the top bar:
- **Streamable HTTP**: `http://127.0.0.1:8765/mcp`
- **SSE Transport**: `http://127.0.0.1:8765/sse`
- **WebSocket Telemetry**: `ws://127.0.0.1:8765/ws/events`

> **Note for `stdio` clients (Antigravity / Cursor / Claude Desktop)**: Running plain `frfruit-mcp` or `frfruit serve` starts a separate headless session. To connect a `stdio` client into the running FrFruit Studio window, either use the HTTP/SSE URL above or pass `--studio-url http://127.0.0.1:8765/mcp` to `frfruit-mcp` / `frfruit serve`. Once connected, the top-bar badge switches from **MCP READY** to green **MCP CONNECTED**.

Every primitive, cloth simulation, texture download, post-processing edit, or render called by your connected AI client appears in real time inside the FrFruit Studio viewport and audit log.
