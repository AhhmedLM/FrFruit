# Cloud MCP Deployment (Hugging Face Spaces & Docker)

FrFruit can run 24/7 as a remote cloud MCP server (with optional headless FrFruit Studio Web UI) on **Hugging Face Spaces** (free 2 vCPU / 16 GB RAM Docker Space) or any Docker host.

## 1. Deploy on Hugging Face Spaces (Free)

1. Create a new **Space** on [huggingface.co/new-space](https://huggingface.co/new-space):
   - **SDK:** `Docker` (Blank)
   - **Hardware:** `CPU basic · 2 vCPU · 16 GB` (Free)
   - **Visibility:** Public or Private
2. Push this repository (or upload `Dockerfile`, `pyproject.toml`, `README.md`, and `src/`) to the Space repo.
3. In your Space **Settings -> Variables and secrets**, add:
   - **`FRFRUIT_API_KEY`** *(Secret, strongly recommended)*: Any strong random password/token to protect `/mcp`, `/sse`, `/api/*`, and `/ws/events`.
   - **Optional persistent workspace sync** (so `.frf.json` scenes and renders survive Space restarts/sleep):
     - **`HF_TOKEN`** *(Secret)*: A Hugging Face access token with **Write** permission.
     - **`HF_DATASET_REPO`** *(Variable)*: A private HF Dataset repo, e.g. `your-username/frfruit-workspace` (created automatically if needed).

## 2. Connect Your MCP Client

Once the Space is running, your MCP endpoints are:
- **Streamable HTTP (Recommended):** `https://<your-user>-<your-space>.hf.space/mcp`
- **SSE:** `https://<your-user>-<your-space>.hf.space/sse`
- **Health Check:** `https://<your-user>-<your-space>.hf.space/health`

Example MCP client configuration:

```json
{
  "mcpServers": {
    "frfruit-cloud": {
      "url": "https://your-user-your-space.hf.space/mcp",
      "headers": {
        "Authorization": "Bearer YOUR_FRFRUIT_API_KEY"
      }
    }
  }
}
```

## 3. Environment Variables Reference

| Variable | Default | Description |
|---|---|---|
| `PORT` | `7860` | HTTP port to bind (`7860` is required by HF Spaces). |
| `FRFRUIT_WORKSPACE` | `/home/user/workspace` | Directory for `.frf.json` projects and exports. |
| `FRFRUIT_API_KEY` | *(unset)* | Optional Bearer/API-key token protecting MCP & API routes. |
| `HF_DATASET_REPO` | *(unset)* | Optional HF Dataset (`owner/repo`) for automatic workspace backup/restore. |
| `HF_TOKEN` | *(unset)* | Hugging Face token used with `HF_DATASET_REPO`. |
| `FRFRUIT_SYNC_INTERVAL` | `15` | Debounced workspace upload interval in seconds. |
| `FRFRUIT_ALLOWED_HOSTS` | *(auto)* | Comma-separated allowed `Host` headers, or `*` to disable DNS rebinding check. |
