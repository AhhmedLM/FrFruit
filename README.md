# FrFruit

**FrFruit** is an AI-first 3D modeling, procedural generation, physics/cloth simulation, animation, path-traced rendering, and workstation visualization engine built natively for AI agents via the **Model Context Protocol (MCP)** — *"like Blender, engineered from the ground up for AI MCP rather than GUI mouse clicks."*

FrFruit gives AI models semantic geometry editing, JSON procedural node graphs, collision-aware spatial placement, XPBD cloth & rigid-body physics, skeletal rigging, online PBR texture libraries, HDR post-processing, video export, a CPU BVH path tracer, game-ready Binary FBX 7.4 export, and **FrFruit Studio** — a sleek CAD-grade desktop workstation UI co-hosted alongside a live MCP server so humans and AI watch the exact same 3D scene update in real time.

---

## Key Capabilities

- **Dual Render Engines (`software` & `raytrace`)**:
  - **`software`**: Ultra-fast pure-CPU G-buffer rasterizer (`render_viewport`, `render_views`) with **PBR + 4-tap PCF shadows**, **Studio Clay**, **Cyan Wireframe**, **World Normals**, **Depth Map**, and **Collision Highlighting**, plus 2x2 contact sheets (`FRONT`, `RIGHT`, `TOP`, `ISO`).
  - **`raytrace` (Path Tracer)**: Vectorized `_TriangleBVH` Monte Carlo ray/path tracer (`engine="raytrace"`, `samples=1..16`) with stratified AA, thin-lens optical DOF, soft shadows, metallic/dielectric Fresnel reflections, IOR glass refraction, and 1-bounce diffuse GI.
- **HDR Post-Processing Stack (`post_process_manage`)**: Bilateral Screen-Space Ambient Occlusion (**SSAO**), optical **Depth of Field**, multi-scale **HDR Bloom**, **ACES / Filmic / Reinhard** tone mapping, exposure, chromatic aberration, vignette, saturation/contrast/temperature grading, edge sharpen, and **FXAA** (presets: `cinematic`, `architectural`, `studio_clean`, `dramatic_bloom`, `raw_off`).
- **Video Rendering (`render_video_file`, `frfruit video`)**: Renders `timeline`, `turntable` (360° orbit), and Catmull-Rom spline `camera_path` sequences to **`.mp4` (H.264)**, **`.webm` (VP9)**, **`.avi` (MPEG-4)**, and **`.gif`**, plus optional PNG frame sequences.
- **FrFruit Studio Desktop App + Live MCP Bridge (`frfruit app`, `windows_app\FrFruitStudio.bat`)**:
  - Precision dark CAD/DCC workstation UI (`#0e1013` / `#15181d` / `#1b1f26`, hairline `1px` borders, monospace telemetry) with offline **WebGL2 Z-up viewport**, Outliner, 6-tab Inspector (`Prop | Tex | Phys | Post | Out | MCP`), and Timeline scrubber.
  - Co-hosted Starlette backend serving REST API (`/api/*`), live WebSocket (`/ws/events`), and unified **Streamable HTTP (`/mcp`)** + **SSE (`/sse`)** MCP transports sharing a single live `FrFruitSession`.
  - Runs inside native **WebView2** (`pywebview`), Windows Edge/Chrome `--app=` frameless mode, or packages into a standalone Windows `.exe` (`windows_app\build_windows_exe.ps1`).
- **Rigid-Body & XPBD Cloth Physics (`physics_manage`, `frfruit physics`)**: Impulse rigid-body dynamics with restitution/friction and pairwise sphere/AABB collisions baked to `FCurveTrack`s, plus vectorized weighted-Jacobi **XPBD Cloth** with pinning presets (`top_corners`, `top_edge`, `border`, `indices`, `none`), wind gusts, floor bounds, and scene collider push-out baked to `vertex_cache`.
- **Skeletal Rigging & Game-Ready Animated FBX 7.4 (`armature_rig`, `export_scene`)**: FK armature bone hierarchies, automatic top-K vertex weight skinning (`auto_rig`), Linear Blend Skinning (LBS), and a zero-dependency Kaydara **Binary FBX 7.400** & **ASCII FBX** exporter supporting **`unreal`** (Z-up), **`unity`** (Y-up), **`godot`**, and **`generic`** presets — complete with normals, MikkTSpace tangents/binormals, UVs, textures, `Skin`/`Cluster`/`BindPose`, MorphTarget `Shape` blendshapes (baked cloth), and `AnimationCurve`s.
- **Online & Procedural PBR Texture Libraries (`texture_library`, `frfruit textures`)**: Search & download 6-channel PBR maps (`albedo`, `normal`, `roughness`, `metallic`, `ao`, `height`) from **Poly Haven** and **ambientCG** (with ARM unpacking and local seamless procedural fallback), bilinear UV sampling, world-space triplanar UV fallback, and Gram-Schmidt TBN normal perturbation.
- **Constructive & Semantic Modeling + Procedural Nodes + Spatial AI Tools**: Primitives, 2D/3D curves, semantic Edit Mesh, exact `manifold3d` CSG booleans, 11 non-destructive modifiers, 18-node JSON Procedural DAGs, collision detection, smart placement, GLB/GLTF/OBJ/STL/PLY/Blender `.py` export.
- **34 MCP Tools, 6 Resources + 1 Resource Template, 3 Prompts**: Exposed via official `mcp` Python SDK (`stdio`, `streamable-http`, `sse`).

---

## Architecture

```
src/frfruit/
├── core/         # Pydantic SceneDocument IR (.frf.json), transforms, undo/redo snapshots
├── geometry/     # Primitives, curves, Edit Mesh, CSG booleans, 11 modifiers, spatial tools
├── procedural/   # 18-node JSON ProceduralGraph DAG schema & deterministic evaluator
├── animation/    # Timeline F-curves, armature FK bones, auto-skinning, LBS evaluation
├── physics/      # Impulse rigid-body dynamics & vectorized XPBD cloth solver
├── textures/     # Poly Haven / ambientCG fetcher, procedural PBR maps, bilinear/TBN loader
├── render/       # Software rasterizer, BVH PathTracer, Post-Processing, MP4/WebM/GIF video
├── io/           # .frf.json projects, Binary/ASCII FBX 7.4, GLB/GLTF/OBJ/STL/PLY/Blender
├── desktop/      # Co-hosted Starlette UI/WS/MCP bridge, WebGL2 CAD UI, Windows shell launcher
├── mcp_server/   # 34 MCP tools, 6 static resources + template, 3 guided prompts
└── cli.py        # `frfruit` CLI (`serve`, `app`, `render`, `video`, `export`, `textures`, `physics`, `info`)

windows_app/      # Windows `.bat` launcher, PyInstaller `.spec`, PowerShell `.exe` builder
```

---

## Quickstart (`uv`)

Requires **Python >= 3.11** and [uv](https://docs.astral.sh/uv/) (plus `ffmpeg` on PATH for MP4/WebM/AVI video encoding).

```bash
git clone https://github.com/AhhmedLM/FrFruit.git
cd FrFruit
uv sync --extra dev
# Optional native WebView2 window support on Windows:
# uv sync --extra desktop
```

### Run the Test Suite & Linter

```bash
uv run ruff check src tests windows_app
uv run ruff format --check src tests windows_app
uv run pytest -v
```

---

## Launching FrFruit Studio (Windows Desktop App + Live MCP Bridge)

Start the workstation UI + co-hosted live MCP server together:

```bash
# Cross-platform / developer launch
uv run frfruit app --workspace ./workspace --port 8765

# Headless Studio server + MCP endpoint without opening a local window
uv run frfruit app --no-window --workspace ./workspace --port 8765
```

On Windows, double-click [`windows_app\FrFruitStudio.bat`](windows_app/FrFruitStudio.bat) or build a standalone binary (`dist\FrFruitStudio.exe`) using [`windows_app\build_windows_exe.ps1`](windows_app/build_windows_exe.ps1). See [`windows_app/README_WINDOWS.md`](windows_app/README_WINDOWS.md).

While FrFruit Studio is open, point any MCP client at the co-hosted endpoint so every AI change streams live over WebSockets into your 3D viewport:
- **Streamable HTTP MCP**: `http://127.0.0.1:8765/mcp`
- **SSE MCP**: `http://127.0.0.1:8765/sse`

---

## Using FrFruit Headlessly with MCP Clients

Add FrFruit via `stdio` to your MCP client configuration (Claude Desktop, Cursor, Antigravity, etc.):

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

Or launch standalone via Streamable HTTP transport:

```bash
uv run frfruit serve --transport streamable-http --host 127.0.0.1 --port 8000 --workspace ./workspace
```

---

## CLI Usage

```bash
# Inspect project summary & bounds
uv run frfruit info scene.frf.json

# Render via CPU BVH path tracer with post-processing or 2x2 multi-view sheet
uv run frfruit render scene.frf.json --engine raytrace --samples 4 -o render_rt.png
uv run frfruit render scene.frf.json --multiview --mode pbr -o preview_views.png

# Search or download PBR texture packs (Poly Haven / ambientCG / Procedural)
uv run frfruit textures "marble" --source all
uv run frfruit textures --download Marble012 --source ambientcg --resolution 2k -o ./textures

# Bake rigid-body & XPBD cloth simulations
uv run frfruit physics scene.frf.json --frame-start 1 --frame-end 48 -o scene_baked.frf.json

# Render MP4 turntable or timeline video
uv run frfruit video scene_baked.frf.json --kind turntable --fps 24 -o turntable.mp4

# Export game-ready Binary FBX 7.4 (Unreal/Unity/Godot) or GLB / Blender .py
uv run frfruit export scene_baked.frf.json --target-engine unreal -o hero.fbx
uv run frfruit export scene_baked.frf.json -o asset.glb
```

---

## Documentation

- [MCP Tools, Resources & Prompts Reference](docs/mcp-tools-reference.md)
- [FrFruit Studio Windows Desktop Guide](windows_app/README_WINDOWS.md)

---

## License

Licensed under the **GNU General Public License v3.0 or later (GPL-3.0-or-later)**. See [pyproject.toml](pyproject.toml).