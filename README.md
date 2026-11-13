# FrFruit

**FrFruit** is an AI-first headless 3D modeling, procedural generation, animation, and rendering engine built from the ground up for AI agents via the **Model Context Protocol (MCP)** — *"like Blender, designed natively for AI rather than mouse-and-keyboard GUIs."*

Instead of fragile screen coordinates or Blender GUI state hacks, FrFruit gives AI models semantic geometry operations, deterministic JSON procedural node graphs, collision-aware spatial placement tools, and an ultra-fast pure-CPU G-buffer viewport returning PNG `ImageContent` directly inside MCP responses so agents can **build, inspect, critique, fix, animate, and export** 3D assets cleanly.

---

## Key Capabilities

- **6 Viewport Shading Modes (`render_viewport` & `render_views`)**: Instant pure-CPU G-buffer rasterizer with **PBR + 4-tap PCF shadows**, **Studio Clay**, **Cyan Wireframe**, **World Normals**, **Depth Map**, and **Collision Highlighting** — plus 2x2 multi-view contact sheets (`FRONT`, `RIGHT`, `TOP`, `ISO`), XYZ triad, bounding boxes, and overlap-free labels.
- **Constructive & Semantic Modeling**: Watertight primitives (`cube`, `sphere`, `icosphere`, `cylinder`, `cone`, `torus`, `plane`, `capsule`), curves (`extrude_2d`, `revolve_lathe`, `sweep_tube`), semantic **Edit Mesh** (`extrude_faces`, `inset_faces`, `bevel_edges`, `subdivide`, `smooth`, `repair_manifold`), exact CSG booleans via `manifold3d`, and **11 non-destructive modifiers**.
- **JSON Procedural Node Graphs (18 Nodes)**: Built-in Geometry-Nodes-style DAG (`node_catalog_list`, `node_graph_set`, `node_graph_patch`) with generators, points (`CurveCircle`, `GridPoints`, `CurveLine`), transforms, CSG booleans, wave noise, deterministic face scattering (`DistributePointsOnFaces`), and instancing (`InstanceOnPoints`).
- **Spatial Reasoning & Placement**: `spatial_measure` (`bounding_box`, `distance`, `raycast`, `check_intersections`) and `spatial_place` (`ground`, `place_on_top`, `attach_to_side`, `snap_to_surface`, `align`, `distribute`) so objects never float or accidentally clip.
- **Animation & Multi-Format I/O**: F-curve keyframe tracks (`constant`, `linear`, `bezier`), turntable & timeline filmstrip/GIF rendering, native `.frf.json` project files with undo/redo checkpoints, and exports to **GLB/glTF 2.0 (with PBR & TRS animations)**, **OBJ**, **STL**, **PLY**, and **Blender `.py`**.
- **29 MCP Tools, 5 Resources/Templates, 3 Prompts**: Exposed cleanly via the official `mcp` Python SDK (`stdio`, `streamable-http`, and `sse` transports).

---

## Architecture

```
src/frfruit/
├── core/         # Pydantic SceneDocument IR (.frf.json), 3D transforms, history snapshots
├── geometry/     # Primitives, curves/lathe/sweep, Edit Mesh, CSG, 11 modifiers, spatial tools
├── procedural/   # 18-node JSON ProceduralGraph DAG schema & deterministic evaluator
├── animation/    # Timeline F-curve tracks & keyframe evaluation
├── render/       # Pure-CPU G-buffer rasterizer, PBR/IBL shaders, contact sheets, turntable GIFs
├── io/           # .frf.json projects, mesh importers, GLB/GLTF/OBJ/STL/PLY/Blender exporters
├── mcp_server/   # FastMCP/MCPServer tool registrations (29 tools, 5 resources, 3 prompts)
└── cli.py        # `frfruit` CLI (`serve`, `render`, `export`, `info`)
```

---

## Quickstart (`uv`)

Requires **Python >= 3.11** and [uv](https://docs.astral.sh/uv/).

```bash
git clone https://github.com/AhhmedLM/FrFruit.git
cd FrFruit
uv sync --extra dev
```

### Run the Test Suite & Linter

```bash
uv run ruff check .
uv run ruff format --check .
uv run pytest -v
```

---

## Using FrFruit with MCP Clients

Add FrFruit to your MCP client configuration (Claude Desktop, Cursor, Antigravity, etc.):

```json
{
  "mcpServers": {
    "frfruit": {
      "command": "uv",
      "args": [
        "run",
        "--directory",
        "/path/to/FrFruit",
        "frfruit",
        "serve",
        "--workspace",
        "/path/to/your-3d-workspace"
      ]
    }
  }
}
```

Or launch via Streamable HTTP transport:

```bash
uv run frfruit serve --transport streamable-http --host 127.0.0.1 --port 8000 --workspace ./workspace
```

---

## CLI Usage

```bash
# Inspect project summary & bounds
uv run frfruit info scene.frf.json

# Render a 2x2 multi-view PBR contact sheet to PNG
uv run frfruit render scene.frf.json --multiview --mode pbr -o preview_views.png

# Export scene to GLB (including animated TRS tracks) or Blender .py script
uv run frfruit export scene.frf.json -o asset.glb
uv run frfruit export scene.frf.json -o asset_build.py
```

---

## Documentation

- [MCP Tools, Resources & Prompts Reference](docs/mcp-tools-reference.md)

---

## License

Licensed under the **GNU General Public License v3.0 or later (GPL-3.0-or-later)**. See [pyproject.toml](pyproject.toml).