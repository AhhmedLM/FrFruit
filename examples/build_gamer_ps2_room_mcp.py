"""Build a 2000s American Gamer's Cutaway Bedroom with PlayStation 2 via real FrFruit MCP calls.

Launches an actual FrFruit MCP server subprocess over stdio (`uv run frfruit serve`), constructs
the complete multi-zone bedroom (architecture cutout + Boolean window + mini-blinds, Sony
Trinitron CRT TV + standing Fat PS2 + 2x DualShock 2 with swept cables + PS2 game cases, 2000s
beige CRT desktop PC setup, XPBD cloth-simulated bed comforter, beanbag + pizza + soda cans +
lava lamp + skateboard + boombox + wall posters), runs spatial collision verification, bakes
physics, renders software & BVH raytraced views, exports turntable MP4/GIF, Unreal FBX, GLB,
and `.frf.json`.
"""

from __future__ import annotations

import asyncio
import json
import math
import sys
from pathlib import Path
from typing import Any

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from mcp.types import TextContent

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_WORKSPACE = REPO_ROOT / "exports" / "gamer_ps2_room"


def _extract_json(result: Any, tool_name: str) -> dict[str, Any]:
    if result.is_error:
        raise RuntimeError(f"MCP tool '{tool_name}' failed: {result.content}")
    texts = [c.text for c in result.content if isinstance(c, TextContent)]
    if not texts:
        return {}
    return json.loads(texts[0])


class McpRoomBuilder:
    """Thin ergonomic wrapper driving real `ClientSession.call_tool` RPCs."""

    def __init__(self, session: ClientSession) -> None:
        self.session = session
        self.call_count = 0

    async def call(self, _tool_name: str, **kwargs: Any) -> dict[str, Any]:
        args = {k: v for k, v in kwargs.items() if v is not None}
        res = await self.session.call_tool(_tool_name, args)
        self.call_count += 1
        return _extract_json(res, _tool_name)

    async def prim(
        self,
        primitive: str,
        name: str,
        *,
        size: list[float] | None = None,
        radius: float | None = None,
        height: float | None = None,
        radius_top: float | None = None,
        major_radius: float | None = None,
        minor_radius: float | None = None,
        segments: int = 24,
        subdivisions: int = 2,
        origin: str = "bottom",
        location: list[float] | None = None,
        rotation_deg: list[float] | None = None,
        scale: list[float] | None = None,
        material_id: str | None = None,
        collection_id: str = "default",
        parent_id: str | None = None,
    ) -> str:
        res = await self.call(
            "create_primitive",
            primitive=primitive,
            name=name,
            size=size,
            radius=radius,
            height=height,
            radius_top=radius_top,
            major_radius=major_radius,
            minor_radius=minor_radius,
            segments=segments,
            subdivisions=subdivisions,
            origin=origin,
            location=location,
            rotation_deg=rotation_deg,
            scale=scale,
            material_id=material_id,
            collection_id=collection_id,
        )
        oid = str(res["object_id"])
        if parent_id:
            await self.call("object_transform", object_id=oid, parent_id=parent_id)
        return oid

    async def mat(
        self,
        name: str,
        base_color: list[float] | str,
        *,
        metallic: float = 0.0,
        roughness: float = 0.5,
        emission_color: list[float] | str | None = None,
        emission_strength: float = 0.0,
        alpha: float = 1.0,
        pattern: str = "solid",
        secondary_color: list[float] | str | None = None,
        pattern_scale: float = 1.0,
    ) -> str:
        res = await self.call(
            "material_create_or_assign",
            action="create",
            name=name,
            base_color=base_color,
            metallic=metallic,
            roughness=roughness,
            emission_color=emission_color,
            emission_strength=emission_strength,
            alpha=alpha,
            pattern=pattern,
            secondary_color=secondary_color,
            pattern_scale=pattern_scale,
        )
        return str(res["material"]["id"])


async def build_gamer_room_via_mcp(
    workspace_dir: Path,
    *,
    artifact_dir: Path | None = None,
    render_raytrace: bool = True,
    render_video: bool = True,
) -> dict[str, Any]:
    workspace_dir.mkdir(parents=True, exist_ok=True)
    server_params = StdioServerParameters(
        command=sys.executable,
        args=["-m", "frfruit", "serve", "--transport", "stdio", "--workspace", str(workspace_dir)],
        cwd=str(REPO_ROOT),
    )

    async with stdio_client(server_params) as (read_stream, write_stream):
        async with ClientSession(read_stream, write_stream) as session:
            await session.initialize()
            b = McpRoomBuilder(session)

            # 0. Initialize scene & organize Blender-style collections
            await b.call("scene_new", name="Gamer2000s_PS2_Room", clear_history=True)
            for cid, cname in (
                ("architecture", "01_Architecture"),
                ("ps2_station", "02_PS2_CRT_Station"),
                ("pc_desk", "03_Beige_2000s_PC"),
                ("bedroom", "04_Bed_And_XPBD_Cloth"),
                ("gamer_props", "05_2000s_Gamer_Props"),
            ):
                await b.call("collection_manage", action="create", collection_id=cid, name=cname)

            # 1. Procedural PBR Texture Packs & Curated 2000s Material Palette
            #    Oak plank floor, matte burgundy rug, cherry wood furniture, brushed steel
            await b.prim(
                "cube",
                "FloorSlab",
                size=[4.62, 4.02, 0.14],
                origin="bottom",
                location=[0.0, 0.0, -0.14],
                collection_id="architecture",
            )
            tex_floor = await b.call(
                "texture_library",
                action="generate_procedural",
                asset_id="wood",
                material_name="OakHardwood",
                object_id="FloorSlab",
                uv_scale=[1.4, 0.55],
                resolution="1k",
            )
            _ = tex_floor["material_id"]

            tex_wood_furn = await b.call(
                "texture_library",
                action="generate_procedural",
                asset_id="wood",
                material_name="CherryFurnitureWood",
                uv_scale=[0.65, 0.42],
                resolution="1k",
            )
            mat_furniture_wood = tex_wood_furn["material_id"]

            await b.prim(
                "cube",
                "AreaRug",
                size=[2.45, 2.10, 0.016],
                origin="bottom",
                location=[0.22, -0.18, 0.0],
                collection_id="architecture",
            )
            tex_rug = await b.call(
                "texture_library",
                action="generate_procedural",
                asset_id="fabric",
                material_name="BurgundyRug",
                object_id="AreaRug",
                uv_scale=[3.2, 2.8],
                resolution="1k",
            )
            _ = tex_rug["material_id"]

            tex_metal = await b.call(
                "texture_library",
                action="generate_procedural",
                asset_id="metal",
                material_name="BrushedSteel",
                uv_scale=[1.5, 1.5],
                resolution="1k",
            )
            mat_metal = tex_metal["material_id"]

            mat_wall = await b.mat("DrywallSageSlate", [0.36, 0.42, 0.48], roughness=0.84)
            mat_trim = await b.mat("OffWhiteTrim", [0.86, 0.84, 0.79], roughness=0.45)
            mat_blinds = await b.mat("VenetianBlinds", [0.85, 0.83, 0.76], roughness=0.42)
            mat_glass_win = await b.mat(
                "WindowDuskGlass",
                [0.24, 0.36, 0.62],
                roughness=0.06,
                alpha=0.28,
                emission_color=[0.18, 0.32, 0.65],
                emission_strength=0.22,
            )
            mat_crt_silver = await b.mat(
                "TrinitronSilver", [0.76, 0.78, 0.81], metallic=0.48, roughness=0.34
            )
            mat_dark_plastic = await b.mat("CharcoalABS", [0.11, 0.12, 0.14], roughness=0.46)
            mat_ps2_body = await b.mat("PS2_Obsidian", [0.075, 0.082, 0.098], roughness=0.44)
            mat_ps2_gloss = await b.mat("PS2_TrayGloss", [0.05, 0.055, 0.068], roughness=0.18)
            mat_ps2_stand = await b.mat(
                "PS2_CobaltStand",
                [0.08, 0.28, 0.85],
                metallic=0.15,
                roughness=0.16,
                alpha=0.80,
                emission_color=[0.10, 0.38, 1.0],
                emission_strength=0.20,
            )
            mat_ps2_stripe = await b.mat(
                "PS2_CyanStripe",
                [0.10, 0.56, 1.0],
                emission_color=[0.12, 0.58, 1.0],
                emission_strength=0.85,
            )
            mat_led_green = await b.mat(
                "LED_Green",
                [0.18, 0.95, 0.35],
                emission_color=[0.20, 1.0, 0.38],
                emission_strength=1.2,
            )
            mat_led_blue = await b.mat(
                "LED_Blue",
                [0.15, 0.55, 1.0],
                emission_color=[0.18, 0.62, 1.0],
                emission_strength=1.2,
            )
            mat_led_red = await b.mat(
                "LED_Red",
                [0.98, 0.18, 0.15],
                emission_color=[1.0, 0.18, 0.12],
                emission_strength=1.1,
            )
            mat_led_amber = await b.mat(
                "LED_Amber",
                [1.0, 0.55, 0.12],
                emission_color=[1.0, 0.45, 0.12],
                emission_strength=1.25,
            )
            mat_mc_red = await b.mat(
                "MemCardTranslucentRed", [0.78, 0.08, 0.12], roughness=0.18, alpha=0.78
            )
            mat_crt_screen = await b.mat(
                "CRT_GameScreen",
                [0.10, 0.24, 0.48],
                roughness=0.08,
                emission_color=[0.18, 0.52, 0.95],
                emission_strength=0.82,
            )
            mat_pc_screen = await b.mat(
                "XP_MonitorScreen",
                [0.18, 0.48, 0.78],
                roughness=0.10,
                emission_color=[0.28, 0.68, 0.95],
                emission_strength=0.78,
            )
            mat_hud_cyan = await b.mat(
                "ScreenHUD_Cyan",
                [0.30, 0.85, 1.0],
                emission_color=[0.35, 0.88, 1.0],
                emission_strength=1.1,
            )
            mat_hud_amber = await b.mat(
                "ScreenHUD_Gold",
                [1.0, 0.72, 0.20],
                emission_color=[1.0, 0.70, 0.22],
                emission_strength=1.0,
            )
            mat_beige_pc = await b.mat("RetroPC_Beige", [0.80, 0.76, 0.66], roughness=0.52)
            mat_mattress = await b.mat("MattressCream", [0.88, 0.86, 0.82], roughness=0.78)
            mat_pillow = await b.mat("PillowSoftBlue", [0.76, 0.82, 0.88], roughness=0.76)
            mat_blanket = await b.mat(
                "ComforterNavyPlaid",
                [0.15, 0.28, 0.52],
                secondary_color=[0.10, 0.19, 0.36],
                pattern="checker",
                pattern_scale=8.0,
                roughness=0.82,
            )
            mat_beanbag = await b.mat("GamerBeanbagCorduroy", [0.17, 0.20, 0.28], roughness=0.85)
            mat_ps2_case_blue = await b.mat("PS2_GameCaseBlue", [0.06, 0.22, 0.62], roughness=0.25)
            mat_ps2_case_blk = await b.mat("PS2_GameCaseBlack", [0.14, 0.15, 0.18], roughness=0.28)
            mat_ps2_label = await b.mat("CaseWhiteLabel", [0.88, 0.88, 0.85], roughness=0.45)
            mat_cardboard = await b.mat("PizzaCardboard", [0.74, 0.58, 0.40], roughness=0.85)
            mat_cheese = await b.mat("PizzaCheese", [0.92, 0.72, 0.24], roughness=0.55)
            mat_crust = await b.mat("PizzaCrust", [0.78, 0.48, 0.22], roughness=0.72)
            mat_pepperoni = await b.mat("Pepperoni", [0.68, 0.12, 0.09], roughness=0.45)
            mat_soda_green = await b.mat(
                "NeonSodaCan", [0.22, 0.78, 0.18], metallic=0.65, roughness=0.24
            )
            mat_gold = await b.mat("TrophyGold", [0.94, 0.74, 0.24], metallic=0.92, roughness=0.20)
            mat_pink = await b.mat("BtnPink", [0.90, 0.36, 0.68], roughness=0.35)
            mat_lava_glass = await b.mat(
                "LavaLampGlass",
                [0.68, 0.16, 0.54],
                alpha=0.58,
                roughness=0.08,
                emission_color=[0.55, 0.10, 0.45],
                emission_strength=0.45,
            )
            mat_maple_deck = await b.mat("SkateMaple", [0.76, 0.56, 0.32], roughness=0.45)
            mat_wheel_red = await b.mat("UrethaneRed", [0.85, 0.18, 0.14], roughness=0.30)

            # 2. Architectural L-Corner Cutaway Room & CSG Window + Venetian Mini-Blinds
            await b.prim(
                "cube",
                "SubfloorTrim",
                size=[4.64, 4.04, 0.04],
                origin="bottom",
                location=[0.0, 0.0, -0.18],
                material_id=mat_dark_plastic,
                collection_id="architecture",
            )
            await b.prim(
                "cube",
                "WallBack",
                size=[4.76, 0.16, 2.70],
                origin="bottom",
                location=[-0.08, 2.08, 0.0],
                material_id=mat_wall,
                collection_id="architecture",
            )
            await b.prim(
                "cube",
                "WindowCutout",
                size=[1.36, 0.36, 1.16],
                origin="center",
                location=[1.44, 2.08, 1.56],
                collection_id="architecture",
            )
            await b.call(
                "mesh_boolean",
                operation="difference",
                object_id="WallBack",
                target_object_id="WindowCutout",
                keep_target=False,
            )
            await b.prim(
                "cube",
                "WallLeft",
                size=[0.16, 4.00, 2.70],
                origin="bottom",
                location=[-2.38, 0.00, 0.0],
                material_id=mat_wall,
                collection_id="architecture",
            )

            # Baseboards & Crown Molding
            await b.prim(
                "cube",
                "BaseboardBack",
                size=[4.60, 0.022, 0.10],
                origin="bottom",
                location=[0.0, 1.989, 0.0],
                material_id=mat_trim,
                collection_id="architecture",
            )
            await b.prim(
                "cube",
                "BaseboardLeft",
                size=[0.022, 3.98, 0.10],
                origin="bottom",
                location=[-2.289, -0.01, 0.0],
                material_id=mat_trim,
                collection_id="architecture",
            )
            await b.prim(
                "cube",
                "CrownBack",
                size=[4.62, 0.035, 0.06],
                origin="bottom",
                location=[0.0, 1.982, 2.64],
                material_id=mat_trim,
                collection_id="architecture",
            )
            await b.prim(
                "cube",
                "CrownLeft",
                size=[0.035, 4.00, 0.06],
                origin="bottom",
                location=[-2.282, 0.0, 2.64],
                material_id=mat_trim,
                collection_id="architecture",
            )

            # Window Casing + Glass + Venetian Mini-Blinds (X=0.76..2.12, Z=0.98..2.14)
            await b.prim(
                "cube",
                "WinSill",
                size=[1.46, 0.22, 0.035],
                origin="bottom",
                location=[1.44, 2.04, 0.96],
                material_id=mat_trim,
                collection_id="architecture",
            )
            await b.prim(
                "cube",
                "WinHeader",
                size=[1.46, 0.20, 0.045],
                origin="bottom",
                location=[1.44, 2.06, 2.13],
                material_id=mat_trim,
                collection_id="architecture",
            )
            for sname, wx in (("WinJambL", 0.745), ("WinJambR", 2.135)):
                await b.prim(
                    "cube",
                    sname,
                    size=[0.045, 0.20, 1.17],
                    origin="bottom",
                    location=[wx, 2.06, 0.98],
                    material_id=mat_trim,
                    collection_id="architecture",
                )
            await b.prim(
                "cube",
                "WindowGlass",
                size=[1.34, 0.012, 1.13],
                origin="bottom",
                location=[1.44, 2.10, 0.99],
                material_id=mat_glass_win,
                collection_id="architecture",
            )
            for idx in range(7):
                await b.prim(
                    "cube",
                    f"BlindSlat_{idx}",
                    size=[1.32, 0.028, 0.006],
                    origin="center",
                    location=[1.44, 2.04, 2.05 - idx * 0.085],
                    rotation_deg=[26.0, 0.0, 0.0],
                    material_id=mat_blinds,
                    collection_id="architecture",
                )

            # Left Wall Closet Door + Doorknob
            await b.prim(
                "cube",
                "ClosetDoorFrame",
                size=[0.025, 0.92, 2.10],
                origin="bottom",
                location=[-2.29, -1.30, 0.0],
                material_id=mat_trim,
                collection_id="architecture",
            )
            await b.prim(
                "cube",
                "ClosetDoorLeaf",
                size=[0.018, 0.80, 2.02],
                origin="bottom",
                location=[-2.275, -1.30, 0.0],
                material_id=mat_trim,
                collection_id="architecture",
            )
            await b.prim(
                "sphere",
                "ClosetKnob",
                radius=0.028,
                segments=14,
                origin="center",
                location=[-2.235, -0.98, 0.98],
                material_id=mat_gold,
                collection_id="architecture",
            )

            # 3. 2000s Entertainment Stand + 27" Sony Trinitron CRT TV + HERO FAT PS2 SETUP
            # Stand X=[-0.72..+0.64] (center -0.04), Y=[1.44..1.96] (center 1.70), Top Z=0.54
            stand_legs = [(-0.68, 1.48), (0.60, 1.48), (-0.68, 1.92), (0.60, 1.92)]
            for i, (lx, ly) in enumerate(stand_legs):
                await b.prim(
                    "cylinder",
                    f"StandLeg_{i}",
                    radius=0.024,
                    height=0.03,
                    segments=12,
                    origin="bottom",
                    location=[lx, ly, 0.0],
                    material_id=mat_dark_plastic,
                    collection_id="ps2_station",
                )
            await b.prim(
                "cube",
                "TvStandBottom",
                size=[1.36, 0.52, 0.03],
                origin="bottom",
                location=[-0.04, 1.70, 0.03],
                material_id=mat_furniture_wood,
                collection_id="ps2_station",
            )
            tv_panels = (
                ("TvStandSideL", -0.695),
                ("TvStandSideR", 0.615),
                ("TvStandDiv", -0.04),
            )
            for pname, px in tv_panels:
                await b.prim(
                    "cube",
                    pname,
                    size=[0.03, 0.50, 0.45],
                    origin="bottom",
                    location=[px, 1.70, 0.06],
                    material_id=mat_furniture_wood,
                    collection_id="ps2_station",
                )
            await b.prim(
                "cube",
                "TvStandBackPanel",
                size=[1.34, 0.02, 0.45],
                origin="bottom",
                location=[-0.04, 1.94, 0.06],
                material_id=mat_dark_plastic,
                collection_id="ps2_station",
            )
            await b.prim(
                "cube",
                "TvStandMidShelf",
                size=[1.28, 0.48, 0.026],
                origin="bottom",
                location=[-0.04, 1.70, 0.27],
                material_id=mat_furniture_wood,
                collection_id="ps2_station",
            )
            await b.prim(
                "cube",
                "TvStandTopDeck",
                size=[1.38, 0.54, 0.03],
                origin="bottom",
                location=[-0.04, 1.70, 0.51],
                material_id=mat_furniture_wood,
                collection_id="ps2_station",
            )

            # Lower Left Shelf: Silver VHS / DVD Combo Player
            await b.prim(
                "cube",
                "VhsDvdPlayer",
                size=[0.44, 0.28, 0.085],
                origin="bottom",
                location=[-0.37, 1.66, 0.296],
                material_id=mat_crt_silver,
                collection_id="ps2_station",
            )
            await b.prim(
                "cube",
                "VhsSlotFront",
                size=[0.18, 0.01, 0.026],
                origin="bottom",
                location=[-0.44, 1.518, 0.33],
                material_id=mat_dark_plastic,
                collection_id="ps2_station",
            )
            await b.prim(
                "cube",
                "VhsClockDisplay",
                size=[0.08, 0.01, 0.022],
                origin="bottom",
                location=[-0.24, 1.518, 0.33],
                material_id=mat_hud_cyan,
                collection_id="ps2_station",
            )

            # Lower Right Bay: Row of Standing 2000s PS2 Game Cases
            case_colors = [
                mat_ps2_case_blue,
                mat_ps2_case_blk,
                mat_ps2_case_blue,
                mat_ps2_case_blue,
                mat_ps2_case_blk,
                mat_ps2_case_blue,
                mat_ps2_case_blue,
            ]
            for idx, cm in enumerate(case_colors):
                cx = 0.06 + idx * 0.024
                await b.prim(
                    "cube",
                    f"ShelfGameCase_{idx}",
                    size=[0.018, 0.145, 0.19],
                    origin="bottom",
                    location=[cx, 1.62, 0.296],
                    material_id=cm,
                    collection_id="ps2_station",
                )
                await b.prim(
                    "cube",
                    f"ShelfGameSpine_{idx}",
                    size=[0.016, 0.006, 0.032],
                    origin="bottom",
                    location=[cx, 1.546, 0.445],
                    material_id=mat_ps2_label,
                    collection_id="ps2_station",
                )

            # 27" Sony Trinitron Silver CRT TV (resting Z=0.54 on TvStandTopDeck)
            tv_x, tv_y, tv_z = -0.27, 1.70, 0.54
            tv_root = await b.prim(
                "cube",
                "CRT_TV_Base",
                size=[0.54, 0.34, 0.03],
                origin="bottom",
                location=[tv_x, tv_y, tv_z],
                material_id=mat_dark_plastic,
                collection_id="ps2_station",
            )
            await b.prim(
                "cube",
                "CRT_TV_Bezel",
                size=[0.70, 0.15, 0.52],
                origin="bottom",
                location=[0.0, -0.09, 0.03],
                material_id=mat_crt_silver,
                collection_id="ps2_station",
                parent_id=tv_root,
            )
            await b.prim(
                "cube",
                "CRT_TV_RearTubeA",
                size=[0.56, 0.14, 0.42],
                origin="bottom",
                location=[0.0, 0.055, 0.07],
                material_id=mat_dark_plastic,
                collection_id="ps2_station",
                parent_id=tv_root,
            )
            await b.prim(
                "cube",
                "CRT_TV_RearTubeB",
                size=[0.34, 0.12, 0.28],
                origin="bottom",
                location=[0.0, 0.18, 0.13],
                material_id=mat_dark_plastic,
                collection_id="ps2_station",
                parent_id=tv_root,
            )
            await b.prim(
                "cube",
                "CRT_TV_ScreenMask",
                size=[0.56, 0.025, 0.41],
                origin="bottom",
                location=[0.0, -0.162, 0.09],
                material_id=mat_dark_plastic,
                collection_id="ps2_station",
                parent_id=tv_root,
            )
            await b.prim(
                "cube",
                "CRT_TV_ScreenGlow",
                size=[0.52, 0.015, 0.37],
                origin="bottom",
                location=[0.0, -0.172, 0.11],
                material_id=mat_crt_screen,
                collection_id="ps2_station",
                parent_id=tv_root,
            )
            # Stylized 2000s PS2 floating boot towers + glowing crystal orbs + HUD bar
            for ti, (tx, tz, th) in enumerate(
                [(-0.15, 0.19, 0.15), (-0.05, 0.20, 0.21), (0.05, 0.18, 0.18), (0.14, 0.21, 0.12)]
            ):
                await b.prim(
                    "cube",
                    f"CRT_HudTower_{ti}",
                    size=[0.042, 0.008, th],
                    origin="bottom",
                    location=[tx, -0.181, tz],
                    material_id=mat_hud_cyan if ti % 2 == 0 else mat_hud_amber,
                    collection_id="ps2_station",
                    parent_id=tv_root,
                )
            await b.prim(
                "sphere",
                "CRT_HudOrbCore",
                radius=0.032,
                segments=12,
                origin="center",
                location=[0.0, -0.182, 0.415],
                scale=[1.0, 0.25, 1.0],
                material_id=mat_hud_cyan,
                collection_id="ps2_station",
                parent_id=tv_root,
            )
            await b.prim(
                "cube",
                "CRT_HudMenuBar",
                size=[0.36, 0.008, 0.024],
                origin="bottom",
                location=[0.0, -0.181, 0.132],
                material_id=mat_ps2_label,
                collection_id="ps2_station",
                parent_id=tv_root,
            )
            await b.prim(
                "cube",
                "CRT_SonyBadge",
                size=[0.075, 0.006, 0.014],
                origin="bottom",
                location=[0.0, -0.168, 0.052],
                material_id=mat_dark_plastic,
                collection_id="ps2_station",
                parent_id=tv_root,
            )
            for sp_name, sx in (("CRT_SpkL", -0.31), ("CRT_SpkR", 0.31)):
                await b.prim(
                    "cube",
                    sp_name,
                    size=[0.045, 0.015, 0.38],
                    origin="bottom",
                    location=[sx, -0.166, 0.10],
                    material_id=mat_dark_plastic,
                    collection_id="ps2_station",
                    parent_id=tv_root,
                )
            await b.prim(
                "cylinder",
                "CRT_PowerLED",
                radius=0.008,
                height=0.01,
                segments=10,
                origin="center",
                location=[-0.23, -0.168, 0.055],
                rotation_deg=[90.0, 0.0, 0.0],
                material_id=mat_led_green,
                collection_id="ps2_station",
                parent_id=tv_root,
            )

            # HERO PLAYSTATION 2 (Fat SCPH-30001 Standing on Blue Vertical Stand)
            # Positioned right of CRT TV on stand top (Z=0.54) angled -16 deg toward camera!
            ps2_root = await b.prim(
                "cube",
                "PS2_VerticalStand",
                size=[0.145, 0.225, 0.020],
                origin="bottom",
                location=[0.35, 1.66, 0.54],
                rotation_deg=[0.0, 0.0, -16.0],
                material_id=mat_ps2_stand,
                collection_id="ps2_station",
            )
            await b.prim(
                "cube",
                "PS2_StandLip",
                size=[0.110, 0.025, 0.070],
                origin="bottom",
                location=[0.0, 0.095, 0.020],
                material_id=mat_ps2_stand,
                collection_id="ps2_station",
                parent_id=ps2_root,
            )
            await b.prim(
                "cube",
                "PS2_ChassisMain",
                size=[0.054, 0.185, 0.306],
                origin="bottom",
                location=[0.014, 0.0, 0.020],
                material_id=mat_ps2_body,
                collection_id="ps2_station",
                parent_id=ps2_root,
            )
            await b.prim(
                "cube",
                "PS2_ChassisStepTier",
                size=[0.026, 0.172, 0.272],
                origin="bottom",
                location=[-0.025, 0.004, 0.020],
                material_id=mat_ps2_body,
                collection_id="ps2_station",
                parent_id=ps2_root,
            )
            await b.prim(
                "cube",
                "PS2_BlueAccentStripe",
                size=[0.007, 0.007, 0.245],
                origin="bottom",
                location=[-0.010, -0.091, 0.045],
                material_id=mat_ps2_stripe,
                collection_id="ps2_station",
                parent_id=ps2_root,
            )
            await b.prim(
                "cube",
                "PS2_DiscTray",
                size=[0.018, 0.010, 0.132],
                origin="bottom",
                location=[-0.024, -0.084, 0.162],
                material_id=mat_ps2_gloss,
                collection_id="ps2_station",
                parent_id=ps2_root,
            )
            for vi in range(5):
                await b.prim(
                    "cube",
                    f"PS2_VentRib_{vi}",
                    size=[0.020, 0.008, 0.006],
                    origin="bottom",
                    location=[0.018, -0.094, 0.042 + vi * 0.016],
                    material_id=mat_ps2_gloss,
                    collection_id="ps2_station",
                    parent_id=ps2_root,
                )
            # Dual glowing Power (Green) & Eject (Blue) LEDs
            await b.prim(
                "cube",
                "PS2_PowerButtonLED",
                size=[0.009, 0.010, 0.014],
                origin="bottom",
                location=[0.026, -0.095, 0.276],
                material_id=mat_led_green,
                collection_id="ps2_station",
                parent_id=ps2_root,
            )
            await b.prim(
                "cube",
                "PS2_EjectButtonLED",
                size=[0.009, 0.010, 0.014],
                origin="bottom",
                location=[0.026, -0.095, 0.254],
                material_id=mat_led_blue,
                collection_id="ps2_station",
                parent_id=ps2_root,
            )
            # 2x Plugged-In 8MB Memory Cards (Obsidian + Translucent Cherry Red!)
            await b.prim(
                "cube",
                "PS2_MemoryCard1",
                size=[0.010, 0.026, 0.040],
                origin="bottom",
                location=[0.018, -0.100, 0.195],
                material_id=mat_ps2_gloss,
                collection_id="ps2_station",
                parent_id=ps2_root,
            )
            await b.prim(
                "cube",
                "PS2_MemoryCard2_Red",
                size=[0.010, 0.026, 0.040],
                origin="bottom",
                location=[0.018, -0.100, 0.145],
                material_id=mat_mc_red,
                collection_id="ps2_station",
                parent_id=ps2_root,
            )
            # 2x Controller Port Plugs in Slots 1 & 2
            await b.prim(
                "cube",
                "PS2_CtrlPlug1",
                size=[0.012, 0.022, 0.032],
                origin="bottom",
                location=[-0.004, -0.098, 0.118],
                material_id=mat_dark_plastic,
                collection_id="ps2_station",
                parent_id=ps2_root,
            )
            await b.prim(
                "cube",
                "PS2_CtrlPlug2",
                size=[0.012, 0.022, 0.032],
                origin="bottom",
                location=[-0.004, -0.098, 0.072],
                material_id=mat_dark_plastic,
                collection_id="ps2_station",
                parent_id=ps2_root,
            )
            # USB Ports x2 + i.LINK S400 port + Front Cyan "PS2" tray badge
            for ui, uz in enumerate((0.056, 0.074)):
                await b.prim(
                    "cube",
                    f"PS2_UsbPort_{ui}",
                    size=[0.008, 0.006, 0.012],
                    origin="bottom",
                    location=[-0.028, -0.088, uz],
                    material_id=mat_metal,
                    collection_id="ps2_station",
                    parent_id=ps2_root,
                )
            await b.prim(
                "cube",
                "PS2_ILinkPort",
                size=[0.006, 0.006, 0.008],
                origin="bottom",
                location=[-0.028, -0.088, 0.094],
                material_id=mat_metal,
                collection_id="ps2_station",
                parent_id=ps2_root,
            )
            await b.prim(
                "cube",
                "PS2_TrayBadgeCyan",
                size=[0.010, 0.005, 0.036],
                origin="bottom",
                location=[-0.024, -0.090, 0.245],
                material_id=mat_ps2_stripe,
                collection_id="ps2_station",
                parent_id=ps2_root,
            )

            # Horizontal Hero Game Stack right beside PS2 on top deck (X=+0.54)
            for hi, cm in enumerate((mat_ps2_case_blue, mat_ps2_case_blk, mat_ps2_case_blue)):
                await b.prim(
                    "cube",
                    f"DeckGameCase_{hi}",
                    size=[0.135, 0.190, 0.016],
                    origin="bottom",
                    location=[0.54, 1.72, 0.54 + hi * 0.017],
                    rotation_deg=[0.0, 0.0, -6.0 + hi * 8.0],
                    material_id=cm,
                    collection_id="ps2_station",
                )

            # 2x DualShock 2 Controllers + Swept Black Cables from PS2 Ports to Controllers
            async def build_dualshock2(
                tag: str, loc: list[float], rot_z: float, cable_pts: list[list[float]]
            ) -> None:
                c_root = await b.prim(
                    "cube",
                    f"{tag}_Center",
                    size=[0.092, 0.068, 0.026],
                    origin="bottom",
                    location=loc,
                    rotation_deg=[0.0, 0.0, rot_z],
                    material_id=mat_ps2_body,
                    collection_id="ps2_station",
                )
                for hside, hx, hrot in (("GripL", -0.050, 22.0), ("GripR", 0.050, -22.0)):
                    await b.prim(
                        "capsule",
                        f"{tag}_{hside}",
                        radius=0.017,
                        height=0.088,
                        segments=12,
                        origin="center",
                        location=[hx, -0.032, 0.014],
                        rotation_deg=[76.0, 0.0, hrot],
                        material_id=mat_ps2_body,
                        collection_id="ps2_station",
                        parent_id=c_root,
                    )
                for st_name, sx in (("StickL", -0.022), ("StickR", 0.022)):
                    await b.prim(
                        "cylinder",
                        f"{tag}_{st_name}Base",
                        radius=0.011,
                        height=0.012,
                        segments=12,
                        origin="bottom",
                        location=[sx, -0.014, 0.026],
                        material_id=mat_dark_plastic,
                        collection_id="ps2_station",
                        parent_id=c_root,
                    )
                    await b.prim(
                        "sphere",
                        f"{tag}_{st_name}Cap",
                        radius=0.012,
                        segments=12,
                        origin="center",
                        location=[sx, -0.014, 0.038],
                        scale=[1.0, 1.0, 0.45],
                        material_id=mat_dark_plastic,
                        collection_id="ps2_station",
                        parent_id=c_root,
                    )
                # D-pad cross on left
                await b.prim(
                    "cube",
                    f"{tag}_DpadH",
                    size=[0.024, 0.008, 0.005],
                    origin="bottom",
                    location=[-0.042, 0.012, 0.026],
                    material_id=mat_dark_plastic,
                    collection_id="ps2_station",
                    parent_id=c_root,
                )
                await b.prim(
                    "cube",
                    f"{tag}_DpadV",
                    size=[0.008, 0.024, 0.005],
                    origin="bottom",
                    location=[-0.042, 0.012, 0.026],
                    material_id=mat_dark_plastic,
                    collection_id="ps2_station",
                    parent_id=c_root,
                )
                # 4 PS action buttons (Tri/Cir/X/Sqr) on right
                for bn, bx, by, bm in (
                    ("Tri", 0.042, 0.022, mat_led_green),
                    ("Cir", 0.052, 0.012, mat_led_red),
                    ("X", 0.042, 0.002, mat_ps2_stripe),
                    ("Sqr", 0.032, 0.012, mat_pink),
                ):
                    await b.prim(
                        "cylinder",
                        f"{tag}_Btn{bn}",
                        radius=0.0045,
                        height=0.005,
                        segments=8,
                        origin="bottom",
                        location=[bx, by, 0.026],
                        material_id=bm,
                        collection_id="ps2_station",
                        parent_id=c_root,
                    )
                # Red Analog status LED
                await b.prim(
                    "cube",
                    f"{tag}_AnalogLED",
                    size=[0.006, 0.004, 0.003],
                    origin="bottom",
                    location=[0.0, -0.004, 0.027],
                    material_id=mat_led_red,
                    collection_id="ps2_station",
                    parent_id=c_root,
                )
                await b.call(
                    "create_from_curve",
                    mode="sweep_tube",
                    name=f"{tag}_Cable",
                    points=cable_pts,
                    radius=0.0055,
                    segments=8,
                    material_id=mat_dark_plastic,
                )

            await build_dualshock2(
                "DS2_Player1",
                [-0.18, 0.24, 0.016],
                18.0,
                [
                    [0.32, 1.56, 0.66],
                    [0.27, 1.46, 0.59],
                    [0.21, 1.40, 0.562],
                    [0.16, 1.32, 0.42],
                    [0.10, 1.15, 0.03],
                    [-0.02, 0.74, 0.022],
                    [-0.14, 0.44, 0.022],
                    [-0.19, 0.28, 0.024],
                ],
            )
            await build_dualshock2(
                "DS2_Player2",
                [0.24, 0.56, 0.016],
                -26.0,
                [
                    [0.32, 1.56, 0.61],
                    [0.34, 1.46, 0.575],
                    [0.36, 1.40, 0.560],
                    [0.39, 1.31, 0.40],
                    [0.42, 1.12, 0.03],
                    [0.40, 0.84, 0.022],
                    [0.31, 0.69, 0.022],
                    [0.25, 0.60, 0.024],
                ],
            )

            # 4. Chunky Beige 2000s Desktop PC Corner (X=[0.84..2.22], Y=[1.30..1.96])
            await b.prim(
                "cube",
                "PC_DeskTop",
                size=[1.38, 0.66, 0.035],
                origin="bottom",
                location=[1.53, 1.63, 0.72],
                material_id=mat_furniture_wood,
                collection_id="pc_desk",
            )
            await b.prim(
                "cube",
                "PC_DeskDrawersL",
                size=[0.36, 0.62, 0.72],
                origin="bottom",
                location=[1.02, 1.64, 0.0],
                material_id=mat_furniture_wood,
                collection_id="pc_desk",
            )
            await b.prim(
                "cube",
                "PC_DeskLegR",
                size=[0.04, 0.62, 0.72],
                origin="bottom",
                location=[2.19, 1.64, 0.0],
                material_id=mat_furniture_wood,
                collection_id="pc_desk",
            )
            await b.prim(
                "cube",
                "PC_DeskBackModesty",
                size=[1.20, 0.025, 0.38],
                origin="bottom",
                location=[1.60, 1.92, 0.34],
                material_id=mat_furniture_wood,
                collection_id="pc_desk",
            )
            # 17" Beige CRT Monitor resting on top of Desk (Z=0.755)
            mon_root = await b.prim(
                "cylinder",
                "PC_MonitorStand",
                radius=0.12,
                height=0.04,
                segments=16,
                origin="bottom",
                location=[1.42, 1.72, 0.755],
                rotation_deg=[0.0, 0.0, 12.0],
                material_id=mat_beige_pc,
                collection_id="pc_desk",
            )
            await b.prim(
                "cube",
                "PC_MonitorBezel",
                size=[0.42, 0.16, 0.36],
                origin="bottom",
                location=[0.0, -0.06, 0.04],
                material_id=mat_beige_pc,
                collection_id="pc_desk",
                parent_id=mon_root,
            )
            await b.prim(
                "cube",
                "PC_MonitorTube",
                size=[0.30, 0.20, 0.24],
                origin="bottom",
                location=[0.0, 0.11, 0.09],
                material_id=mat_beige_pc,
                collection_id="pc_desk",
                parent_id=mon_root,
            )
            await b.prim(
                "cube",
                "PC_MonitorScreen",
                size=[0.34, 0.015, 0.27],
                origin="bottom",
                location=[0.0, -0.142, 0.085],
                material_id=mat_pc_screen,
                collection_id="pc_desk",
                parent_id=mon_root,
            )
            # Windows XP Bliss green hill, classic taskbar, Start button & open IM window
            await b.prim(
                "cube",
                "PC_MonitorXpHill",
                size=[0.33, 0.004, 0.085],
                origin="bottom",
                location=[0.0, -0.151, 0.11],
                material_id=mat_soda_green,
                collection_id="pc_desk",
                parent_id=mon_root,
            )
            await b.prim(
                "cube",
                "PC_MonitorTaskbar",
                size=[0.34, 0.005, 0.024],
                origin="bottom",
                location=[0.0, -0.152, 0.085],
                material_id=mat_ps2_case_blue,
                collection_id="pc_desk",
                parent_id=mon_root,
            )
            await b.prim(
                "cube",
                "PC_MonitorStartBtn",
                size=[0.055, 0.006, 0.024],
                origin="bottom",
                location=[-0.142, -0.153, 0.085],
                material_id=mat_led_green,
                collection_id="pc_desk",
                parent_id=mon_root,
            )
            await b.prim(
                "cube",
                "PC_MonitorWindow",
                size=[0.17, 0.005, 0.12],
                origin="bottom",
                location=[0.02, -0.152, 0.185],
                material_id=mat_ps2_label,
                collection_id="pc_desk",
                parent_id=mon_root,
            )
            await b.prim(
                "cube",
                "PC_MonitorWinTitle",
                size=[0.17, 0.006, 0.022],
                origin="bottom",
                location=[0.02, -0.153, 0.285],
                material_id=mat_ps2_stripe,
                collection_id="pc_desk",
                parent_id=mon_root,
            )
            await b.prim(
                "cube",
                "PC_MonitorLED",
                size=[0.012, 0.01, 0.008],
                origin="bottom",
                location=[0.16, -0.142, 0.055],
                material_id=mat_led_green,
                collection_id="pc_desk",
                parent_id=mon_root,
            )

            # Mid-Tower Beige ATX PC Case
            atx_root = await b.prim(
                "cube",
                "PC_TowerATX",
                size=[0.20, 0.44, 0.44],
                origin="bottom",
                location=[2.02, 1.68, 0.755],
                material_id=mat_beige_pc,
                collection_id="pc_desk",
            )
            for bi, bz in enumerate((0.36, 0.31)):
                await b.prim(
                    "cube",
                    f"PC_CDDrive_{bi}",
                    size=[0.15, 0.012, 0.036],
                    origin="bottom",
                    location=[0.0, -0.222, bz],
                    material_id=mat_trim,
                    collection_id="pc_desk",
                    parent_id=atx_root,
                )
            await b.prim(
                "cube",
                "PC_FloppyDrive",
                size=[0.11, 0.012, 0.022],
                origin="bottom",
                location=[0.0, -0.222, 0.25],
                material_id=mat_dark_plastic,
                collection_id="pc_desk",
                parent_id=atx_root,
            )
            await b.prim(
                "cylinder",
                "PC_PowerLED",
                radius=0.008,
                height=0.012,
                segments=10,
                origin="center",
                location=[-0.04, -0.224, 0.16],
                rotation_deg=[90.0, 0.0, 0.0],
                material_id=mat_led_green,
                collection_id="pc_desk",
                parent_id=atx_root,
            )
            await b.prim(
                "cylinder",
                "PC_HddLED",
                radius=0.006,
                height=0.012,
                segments=10,
                origin="center",
                location=[0.02, -0.224, 0.16],
                rotation_deg=[90.0, 0.0, 0.0],
                material_id=mat_led_amber,
                collection_id="pc_desk",
                parent_id=atx_root,
            )

            # Keyboard, Mousepad, Mouse, Speakers
            await b.prim(
                "cube",
                "PC_Keyboard",
                size=[0.44, 0.16, 0.022],
                origin="bottom",
                location=[1.42, 1.44, 0.755],
                material_id=mat_beige_pc,
                collection_id="pc_desk",
            )
            await b.prim(
                "cube",
                "PC_Mousepad",
                size=[0.22, 0.20, 0.004],
                origin="bottom",
                location=[1.78, 1.44, 0.755],
                material_id=mat_ps2_case_blue,
                collection_id="pc_desk",
            )
            await b.prim(
                "capsule",
                "PC_Mouse",
                radius=0.026,
                height=0.085,
                segments=12,
                origin="bottom",
                location=[1.78, 1.44, 0.759],
                rotation_deg=[90.0, 0.0, 0.0],
                scale=[1.0, 0.65, 1.0],
                material_id=mat_beige_pc,
                collection_id="pc_desk",
            )
            for sp_id, spx in (("PC_SpkL", 1.12), ("PC_SpkR", 1.72)):
                await b.prim(
                    "cube",
                    sp_id,
                    size=[0.085, 0.095, 0.175],
                    origin="bottom",
                    location=[spx, 1.66, 0.755],
                    material_id=mat_dark_plastic,
                    collection_id="pc_desk",
                )

            # Rolling Office Task Chair at Desk
            ch_root = await b.prim(
                "cylinder",
                "PC_ChairStem",
                radius=0.028,
                height=0.38,
                segments=12,
                origin="bottom",
                location=[1.48, 0.86, 0.045],
                rotation_deg=[0.0, 0.0, 14.0],
                material_id=mat_metal,
                collection_id="pc_desk",
            )
            for li in range(5):
                ang = li * 72.0
                ang_rad = ang * 3.14159265 / 180.0
                await b.prim(
                    "cube",
                    f"PC_ChairLeg_{li}",
                    size=[0.036, 0.28, 0.028],
                    origin="bottom",
                    location=[0.0, 0.0, 0.0],
                    rotation_deg=[0.0, 0.0, ang],
                    material_id=mat_dark_plastic,
                    collection_id="pc_desk",
                    parent_id=ch_root,
                )
                wx = -0.135 * math.sin(ang_rad)
                wy = 0.135 * math.cos(ang_rad)
                await b.prim(
                    "cylinder",
                    f"PC_ChairWheel_{li}",
                    radius=0.022,
                    height=0.024,
                    segments=10,
                    origin="center",
                    location=[wx, wy, -0.022],
                    rotation_deg=[0.0, 90.0, ang],
                    material_id=mat_dark_plastic,
                    collection_id="pc_desk",
                    parent_id=ch_root,
                )
            await b.prim(
                "cube",
                "PC_ChairSeat",
                size=[0.46, 0.44, 0.075],
                origin="bottom",
                location=[0.0, 0.0, 0.38],
                material_id=mat_beanbag,
                collection_id="pc_desk",
                parent_id=ch_root,
            )
            await b.prim(
                "cube",
                "PC_ChairSpine",
                size=[0.065, 0.028, 0.34],
                origin="bottom",
                location=[0.0, -0.23, 0.38],
                material_id=mat_metal,
                collection_id="pc_desk",
                parent_id=ch_root,
            )
            await b.prim(
                "cube",
                "PC_ChairBack",
                size=[0.42, 0.055, 0.44],
                origin="bottom",
                location=[0.0, -0.20, 0.45],
                rotation_deg=[-8.0, 0.0, 0.0],
                material_id=mat_beanbag,
                collection_id="pc_desk",
                parent_id=ch_root,
            )

            # 5. Twin Bed + XPBD Cloth Blanket + Nightstand + Lava Lamp (X=[-2.22..-0.78])
            await b.prim(
                "cube",
                "Bed_BaseFrame",
                size=[0.90, 1.90, 0.24],
                origin="bottom",
                location=[-1.75, 1.00, 0.0],
                material_id=mat_furniture_wood,
                collection_id="bedroom",
            )
            await b.call(
                "physics_manage",
                action="set_rigid_body",
                object_id="Bed_BaseFrame",
                body_type="static",
                collider_shape="box",
            )
            await b.prim(
                "cube",
                "Bed_Headboard",
                size=[0.94, 0.07, 0.90],
                origin="bottom",
                location=[-1.75, 1.93, 0.0],
                material_id=mat_furniture_wood,
                collection_id="bedroom",
            )
            await b.call(
                "physics_manage",
                action="set_rigid_body",
                object_id="Bed_Headboard",
                body_type="static",
                collider_shape="box",
            )
            await b.prim(
                "cube",
                "Bed_Footboard",
                size=[0.94, 0.06, 0.42],
                origin="bottom",
                location=[-1.75, 0.05, 0.0],
                material_id=mat_furniture_wood,
                collection_id="bedroom",
            )
            await b.call(
                "physics_manage",
                action="set_rigid_body",
                object_id="Bed_Footboard",
                body_type="static",
                collider_shape="box",
            )
            await b.prim(
                "cube",
                "Bed_Mattress",
                size=[0.82, 1.80, 0.20],
                origin="bottom",
                location=[-1.75, 0.99, 0.24],
                material_id=mat_mattress,
                collection_id="bedroom",
            )
            await b.call(
                "physics_manage",
                action="set_rigid_body",
                object_id="Bed_Mattress",
                body_type="static",
                collider_shape="box",
            )
            for p_idx, px in enumerate((-1.93, -1.57)):
                await b.prim(
                    "cube",
                    f"Bed_Pillow_{p_idx}",
                    size=[0.33, 0.28, 0.11],
                    origin="bottom",
                    location=[px, 1.72, 0.44],
                    material_id=mat_pillow,
                    collection_id="bedroom",
                )
                await b.call(
                    "modifier_add",
                    object_id=f"Bed_Pillow_{p_idx}",
                    modifier_type="subdivision",
                    params={"levels": 1, "smooth": True},
                )

            # XPBD Cloth Comforter Blanket with generous overhang for billowing side folds
            await b.prim(
                "plane",
                "Bed_ComforterCloth",
                size=[1.06, 1.44, 1.0],
                subdivisions=20,
                origin="bottom",
                location=[-1.75, 0.85, 0.456],
                material_id=mat_blanket,
                collection_id="bedroom",
            )
            await b.call(
                "physics_manage",
                action="set_cloth",
                object_id="Bed_ComforterCloth",
                pin_mode="top_edge",
                stiffness=0.82,
                bend_stiffness=0.04,
                damping=0.03,
                wind=[0.04, -0.02, 0.0],
                collision_margin=0.018,
                substeps=10,
            )

            # Oak Nightstand beside Bed (X=-0.94, Y=1.72, top Z=0.52)
            await b.prim(
                "cube",
                "Nightstand",
                size=[0.32, 0.44, 0.52],
                origin="bottom",
                location=[-0.94, 1.72, 0.0],
                material_id=mat_furniture_wood,
                collection_id="bedroom",
            )
            await b.call(
                "physics_manage",
                action="set_rigid_body",
                object_id="Nightstand",
                body_type="static",
                collider_shape="box",
            )
            # 2000s Lava Lamp (Revolved Lathe) placed cleanly atop Nightstand
            await b.call(
                "create_from_curve",
                mode="revolve_lathe",
                name="LavaLampBase",
                points=[[0.0, 0.0], [0.068, 0.0], [0.045, 0.07], [0.038, 0.08], [0.0, 0.08]],
                segments=18,
                location=[-1.00, 1.76, 0.52],
                material_id=mat_metal,
            )
            await b.call(
                "create_from_curve",
                mode="revolve_lathe",
                name="LavaLampGlass",
                points=[[0.0, 0.08], [0.038, 0.08], [0.056, 0.16], [0.028, 0.30], [0.0, 0.30]],
                segments=18,
                location=[-1.00, 1.76, 0.52],
                material_id=mat_lava_glass,
            )
            await b.prim(
                "sphere",
                "LavaLampWax",
                radius=0.032,
                segments=14,
                origin="center",
                location=[-1.00, 1.76, 0.69],
                scale=[0.9, 0.9, 1.3],
                material_id=mat_led_amber,
                collection_id="bedroom",
            )
            await b.prim(
                "cube",
                "AlarmClock",
                size=[0.11, 0.065, 0.052],
                origin="bottom",
                location=[-0.85, 1.66, 0.52],
                rotation_deg=[0.0, 0.0, 14.0],
                material_id=mat_dark_plastic,
                collection_id="bedroom",
            )
            await b.prim(
                "cube",
                "AlarmClockLED",
                size=[0.075, 0.008, 0.022],
                origin="bottom",
                location=[-0.85, 1.627, 0.538],
                rotation_deg=[0.0, 0.0, 14.0],
                material_id=mat_led_red,
                collection_id="bedroom",
            )

            # 6. Foreground Lounge: Gamer Beanbag, Low Snack Table, Pizza Box, Soda Cans,
            #    Skateboard, Wall Posters & Retro Boombox Shelf
            await b.prim(
                "icosphere",
                "GamerBeanbag",
                radius=0.48,
                subdivisions=2,
                origin="bottom",
                location=[-0.32, -0.44, 0.0],
                scale=[1.0, 1.05, 0.68],
                material_id=mat_beanbag,
                collection_id="gamer_props",
            )
            await b.prim(
                "icosphere",
                "GamerBeanbagBack",
                radius=0.34,
                subdivisions=2,
                origin="bottom",
                location=[-0.35, -0.58, 0.14],
                rotation_deg=[-28.0, 0.0, 6.0],
                scale=[1.05, 0.78, 0.96],
                material_id=mat_beanbag,
                collection_id="gamer_props",
            )

            # Low Retro Snack Coffee Table (X=+0.56, Y=-0.34, top Z=0.30)
            for li, (lx, ly) in enumerate(
                [(0.34, -0.52), (0.78, -0.52), (0.34, -0.16), (0.78, -0.16)]
            ):
                await b.prim(
                    "cylinder",
                    f"SnackTableLeg_{li}",
                    radius=0.018,
                    height=0.275,
                    segments=10,
                    origin="bottom",
                    location=[lx, ly, 0.0],
                    material_id=mat_dark_plastic,
                    collection_id="gamer_props",
                )
            await b.prim(
                "cube",
                "SnackTableTop",
                size=[0.52, 0.44, 0.025],
                origin="bottom",
                location=[0.56, -0.34, 0.275],
                material_id=mat_furniture_wood,
                collection_id="gamer_props",
            )

            # Open Cardboard Pizza Box (rear-hinged -104 deg) + 6 Pepperoni Slices
            pbox = await b.prim(
                "cube",
                "PizzaBoxBase",
                size=[0.32, 0.32, 0.028],
                origin="bottom",
                location=[0.49, -0.34, 0.30],
                rotation_deg=[0.0, 0.0, 12.0],
                material_id=mat_cardboard,
                collection_id="gamer_props",
            )
            await b.prim(
                "cube",
                "PizzaBoxLiner",
                size=[0.285, 0.285, 0.004],
                origin="bottom",
                location=[0.0, 0.0, 0.0285],
                material_id=mat_ps2_label,
                collection_id="gamer_props",
                parent_id=pbox,
            )
            await b.prim(
                "cube",
                "PizzaBoxLid",
                size=[0.32, 0.32, 0.010],
                origin="center",
                location=[0.0, 0.199, 0.183],
                rotation_deg=[-104.0, 0.0, 0.0],
                material_id=mat_cardboard,
                collection_id="gamer_props",
                parent_id=pbox,
            )
            for s_idx, ang_deg in enumerate((0.0, 48.0, 96.0, 144.0, 192.0, 240.0)):
                sl = await b.call(
                    "create_from_curve",
                    mode="extrude_2d",
                    name=f"PizzaSlice_{s_idx}",
                    points=[[0.0, 0.0], [0.115, -0.044], [0.115, 0.044]],
                    height=0.010,
                    location=[0.0, 0.0, 0.034],
                    material_id=mat_cheese,
                )
                sl_id = sl["object_id"]
                await b.call(
                    "object_transform",
                    object_id=sl_id,
                    parent_id=pbox,
                    rotation_deg=[0.0, 0.0, ang_deg],
                )
                await b.prim(
                    "cube",
                    f"Crust_{s_idx}",
                    size=[0.018, 0.088, 0.014],
                    origin="bottom",
                    location=[0.114, 0.0, 0.0],
                    material_id=mat_crust,
                    collection_id="gamer_props",
                    parent_id=sl_id,
                )
                await b.prim(
                    "cylinder",
                    f"Pepperoni_{s_idx}",
                    radius=0.014,
                    height=0.004,
                    segments=10,
                    origin="bottom",
                    location=[0.068, 0.0, 0.010],
                    material_id=mat_pepperoni,
                    collection_id="gamer_props",
                    parent_id=sl_id,
                )

            # 2x Neon Green Soda Cans next to Pizza Box (Z=0.30)
            for c_idx, (cx, cy) in enumerate([(0.72, -0.44), (0.76, -0.26)]):
                await b.prim(
                    "cylinder",
                    f"SodaCanBody_{c_idx}",
                    radius=0.030,
                    height=0.112,
                    segments=16,
                    origin="bottom",
                    location=[cx, cy, 0.30],
                    material_id=mat_soda_green,
                    collection_id="gamer_props",
                )
                await b.prim(
                    "cylinder",
                    f"SodaCanRim_{c_idx}",
                    radius=0.027,
                    height=0.008,
                    segments=16,
                    origin="bottom",
                    location=[cx, cy, 0.412],
                    material_id=mat_metal,
                    collection_id="gamer_props",
                )

            # Leaning 2000s Popsicle Skateboard against Left Wall (X=-2.15, Y=-0.45)
            # Deck length runs along local X; rotating Y=-75 deg lifts +X against wall
            # and leaves maple underside, silver axle trucks, and 4 red wheels facing room!
            skate_res = await b.call(
                "create_from_curve",
                mode="extrude_2d",
                name="SkateboardDeck",
                points=[
                    [-0.36, -0.095],
                    [0.0, -0.105],
                    [0.36, -0.095],
                    [0.36, 0.095],
                    [0.0, 0.105],
                    [-0.36, 0.095],
                ],
                height=0.014,
                location=[-2.15, -0.45, 0.37],
                material_id=mat_maple_deck,
            )
            skate_oid = skate_res["object_id"]
            await b.prim(
                "cube",
                "SkateGripTape",
                size=[0.66, 0.18, 0.004],
                origin="bottom",
                location=[0.0, 0.0, 0.014],
                material_id=mat_dark_plastic,
                collection_id="gamer_props",
                parent_id=skate_oid,
            )
            for t_idx, tx in enumerate((-0.20, 0.20)):
                await b.prim(
                    "cylinder",
                    f"SkateTruck_{t_idx}",
                    radius=0.012,
                    height=0.16,
                    segments=10,
                    origin="center",
                    location=[tx, 0.0, -0.022],
                    rotation_deg=[90.0, 0.0, 0.0],
                    material_id=mat_metal,
                    collection_id="gamer_props",
                    parent_id=skate_oid,
                )
                for w_side, wy in (("L", -0.085), ("R", 0.085)):
                    await b.prim(
                        "cylinder",
                        f"SkateWheel_{t_idx}_{w_side}",
                        radius=0.028,
                        height=0.026,
                        segments=14,
                        origin="center",
                        location=[tx, wy, -0.022],
                        rotation_deg=[90.0, 0.0, 0.0],
                        material_id=mat_wheel_red,
                        collection_id="gamer_props",
                        parent_id=skate_oid,
                    )
            await b.call(
                "object_transform",
                object_id=skate_oid,
                rotation_deg=[0.0, -75.0, 0.0],
            )

            # Floating Wall Shelf + Retro Boombox + Gold Gaming Trophy above TV (X=-0.12, Z=1.52)
            await b.prim(
                "cube",
                "WallShelf",
                size=[1.12, 0.24, 0.032],
                origin="bottom",
                location=[-0.10, 1.88, 1.50],
                material_id=mat_furniture_wood,
                collection_id="gamer_props",
            )
            await b.prim(
                "cube",
                "BoomboxBody",
                size=[0.42, 0.14, 0.17],
                origin="bottom",
                location=[-0.28, 1.87, 1.532],
                material_id=mat_crt_silver,
                collection_id="gamer_props",
            )
            for spx in (-0.41, -0.15):
                await b.prim(
                    "cylinder",
                    f"BoomboxSpeaker_{spx:.2f}",
                    radius=0.058,
                    height=0.015,
                    segments=16,
                    origin="center",
                    location=[spx, 1.795, 1.61],
                    rotation_deg=[90.0, 0.0, 0.0],
                    material_id=mat_dark_plastic,
                    collection_id="gamer_props",
                )
            await b.prim(
                "cube",
                "BoomboxHandle",
                size=[0.24, 0.025, 0.025],
                origin="bottom",
                location=[-0.28, 1.87, 1.72],
                material_id=mat_dark_plastic,
                collection_id="gamer_props",
            )
            await b.prim(
                "cylinder",
                "GameTrophy",
                radius=0.045,
                radius_top=0.065,
                height=0.17,
                segments=16,
                origin="bottom",
                location=[0.26, 1.87, 1.532],
                material_id=mat_gold,
                collection_id="gamer_props",
            )

            # 3 Framed Multi-Layer 2000s Game & Music Wall Posters attached flush on walls
            async def build_poster(
                pname: str,
                wall_side: str,
                loc: list[float],
                w: float,
                h: float,
                bg_mat: str,
                accent_mat: str,
            ) -> str:
                is_left = wall_side == "left"
                sz = [0.024, w, h] if is_left else [w, 0.024, h]
                frm = await b.prim(
                    "cube",
                    f"{pname}_Frame",
                    size=sz,
                    origin="bottom",
                    location=loc,
                    material_id=mat_dark_plastic,
                    collection_id="gamer_props",
                )
                art_sz = [0.010, w - 0.06, h - 0.06] if is_left else [w - 0.06, 0.010, h - 0.06]
                art_loc = [0.012, 0.0, 0.03] if is_left else [0.0, -0.012, 0.03]
                await b.prim(
                    "cube",
                    f"{pname}_Art",
                    size=art_sz,
                    origin="bottom",
                    location=art_loc,
                    material_id=bg_mat,
                    collection_id="gamer_props",
                    parent_id=frm,
                )
                stripe_sz = [0.012, w * 0.62, h * 0.16] if is_left else [w * 0.62, 0.012, h * 0.16]
                stripe_loc = [0.014, 0.0, h * 0.18] if is_left else [0.0, -0.014, h * 0.18]
                await b.prim(
                    "cube",
                    f"{pname}_Banner",
                    size=stripe_sz,
                    origin="bottom",
                    location=stripe_loc,
                    material_id=accent_mat,
                    collection_id="gamer_props",
                    parent_id=frm,
                )
                return frm

            p_vice = await build_poster(
                "Poster_ViceSunset",
                "back",
                [-1.62, 1.988, 1.24],
                0.62,
                0.84,
                mat_pink,
                mat_ps2_body,
            )
            await b.prim(
                "cylinder",
                "Poster_ViceSunOrb",
                radius=0.14,
                height=0.012,
                segments=18,
                origin="center",
                location=[0.0, -0.016, 0.54],
                rotation_deg=[90.0, 0.0, 0.0],
                material_id=mat_hud_amber,
                collection_id="gamer_props",
                parent_id=p_vice,
            )
            await b.prim(
                "cube",
                "Poster_ViceHorizon",
                size=[0.52, 0.014, 0.04],
                origin="bottom",
                location=[0.0, -0.017, 0.44],
                material_id=mat_hud_cyan,
                collection_id="gamer_props",
                parent_id=p_vice,
            )

            p_stealth = await build_poster(
                "Poster_StealthOps",
                "left",
                [-2.288, 0.85, 1.24],
                0.68,
                0.88,
                mat_ps2_case_blk,
                mat_dark_plastic,
            )
            for gi, (gy, gz) in enumerate(((-0.08, 0.50), (0.08, 0.50), (0.0, 0.63))):
                await b.prim(
                    "cylinder",
                    f"Poster_StealthNvgLens_{gi}",
                    radius=0.048,
                    height=0.014,
                    segments=14,
                    origin="center",
                    location=[0.016, gy, gz],
                    rotation_deg=[0.0, 90.0, 0.0],
                    material_id=mat_led_green,
                    collection_id="gamer_props",
                    parent_id=p_stealth,
                )

            p_turbo = await build_poster(
                "Poster_TurboGT",
                "left",
                [-2.288, -0.15, 1.26],
                0.56,
                0.76,
                mat_ps2_case_blue,
                mat_led_red,
            )
            await b.prim(
                "cube",
                "Poster_TurboCarBody",
                size=[0.014, 0.34, 0.10],
                origin="bottom",
                location=[0.016, 0.0, 0.34],
                material_id=mat_crt_silver,
                collection_id="gamer_props",
                parent_id=p_turbo,
            )
            await b.prim(
                "cube",
                "Poster_TurboCarRoof",
                size=[0.014, 0.18, 0.07],
                origin="bottom",
                location=[0.016, -0.02, 0.44],
                material_id=mat_hud_cyan,
                collection_id="gamer_props",
                parent_id=p_turbo,
            )

            # 7. Bake XPBD Cloth Physics on Bed Comforter & Evaluate Frame 24
            await b.call("timeline_manage", action="set_range", fps=24, frame_start=1, frame_end=24)
            phys_bake = await b.call(
                "physics_manage", action="bake", frame_start=1, frame_end=24, ground_z=0.0
            )
            await b.call("timeline_manage", action="set_frame", current_frame=24)

            # 8. Spatial Collision Audit across Core Furniture
            collision_res = await b.call(
                "spatial_measure",
                action="check_intersections",
                object_ids=[
                    "TvStandBottom",
                    "Bed_BaseFrame",
                    "Nightstand",
                    "PC_DeskDrawersL",
                    "GamerBeanbag",
                    "SnackTableTop",
                    "PC_ChairSeat",
                ],
                tolerance=0.005,
            )

            # 9. Balanced 2000s Evening Room Lighting & HDR Post-Processing
            await b.call(
                "light_manage",
                action="set_world",
                background_color=[0.07, 0.09, 0.15],
                ambient_strength=0.45,
            )
            # Clear default lights cleanly before setting custom room lighting rig
            lights_cur = await b.call("light_manage", action="list")
            for litem in lights_cur.get("lights", []):
                await b.call("light_manage", action="delete", light_id=litem["id"])

            await b.call(
                "light_manage",
                action="add",
                name="DuskWindowMoon",
                kind="sun",
                location=[3.2, 4.0, 4.8],
                direction=[-0.42, -0.58, -0.68],
                color=[0.74, 0.84, 1.0],
                intensity=0.95,
                cast_shadows=True,
            )
            await b.call(
                "light_manage",
                action="add",
                name="BedroomCeilingWarm",
                kind="point",
                location=[0.05, 0.15, 2.45],
                color=[1.0, 0.90, 0.76],
                intensity=1.20,
                cast_shadows=False,
            )
            await b.call(
                "light_manage",
                action="add",
                name="CRT_BlueSpill",
                kind="point",
                location=[-0.15, 1.20, 0.86],
                color=[0.30, 0.72, 1.0],
                intensity=0.72,
                cast_shadows=False,
            )
            await b.call(
                "light_manage",
                action="add",
                name="LavaLampWarmGlow",
                kind="point",
                location=[-0.98, 1.62, 0.80],
                color=[1.0, 0.50, 0.25],
                intensity=0.52,
                cast_shadows=False,
            )

            await b.call(
                "post_process_manage",
                action="configure",
                render_engine="software",
                tone_mapper="aces",
                exposure=0.05,
                bloom_enabled=True,
                bloom_threshold=0.88,
                bloom_intensity=0.28,
                bloom_radius=6.0,
                ssao_enabled=True,
                ssao_radius=4.5,
                ssao_strength=0.38,
                vignette_intensity=0.14,
                saturation=1.08,
                contrast=1.05,
                temperature=0.02,
                fxaa_enabled=True,
                sharpen=0.08,
            )

            # 10. Curated Cameras + Headless Inspection Renders via MCP
            await b.call(
                "camera_manage",
                action="add",
                name="HeroRoomCam",
                location=[3.15, -2.95, 2.24],
                target=[-0.16, 0.72, 0.70],
                fov_deg=45.5,
            )
            await b.call(
                "camera_manage",
                action="add",
                name="CloseUp_PS2_CRT",
                location=[0.42, 0.42, 0.94],
                target=[0.06, 1.68, 0.68],
                fov_deg=39.0,
            )
            await b.call(
                "camera_manage",
                action="add",
                name="Corner_Bed_PC",
                location=[1.65, -1.85, 1.65],
                target=[-0.65, 1.05, 0.68],
                fov_deg=46.0,
            )
            await b.call("camera_manage", action="set_active", camera_id="HeroRoomCam")

            # Renders saved inside workspace + optional copies in artifact_dir for inspection
            await b.call(
                "render_views",
                mode="pbr",
                engine="software",
                tile_width=420,
                tile_height=320,
                show_labels=False,
                save_path="renders/gamer_room_multiview.png",
            )
            await b.call(
                "render_viewport",
                camera_id="HeroRoomCam",
                mode="pbr",
                engine="software",
                width=720,
                height=520,
                show_grid=False,
                save_path="renders/gamer_room_hero_sw.png",
            )
            await b.call(
                "render_viewport",
                camera_id="CloseUp_PS2_CRT",
                mode="pbr",
                engine="software",
                width=640,
                height=460,
                show_grid=False,
                save_path="renders/gamer_room_ps2_closeup_sw.png",
            )
            await b.call(
                "render_viewport",
                camera_id="Corner_Bed_PC",
                mode="pbr",
                engine="software",
                width=640,
                height=460,
                show_grid=False,
                save_path="renders/gamer_room_bed_corner_sw.png",
            )
            await b.call(
                "render_viewport",
                camera_id="HeroRoomCam",
                mode="wireframe",
                engine="software",
                width=640,
                height=460,
                save_path="renders/gamer_room_wireframe.png",
            )

            if render_raytrace:
                await b.call(
                    "render_viewport",
                    camera_id="HeroRoomCam",
                    mode="pbr",
                    engine="raytrace",
                    samples=2,
                    width=560,
                    height=400,
                    show_grid=False,
                    save_path="renders/gamer_room_hero_raytrace.png",
                )
                await b.call(
                    "render_viewport",
                    camera_id="CloseUp_PS2_CRT",
                    mode="pbr",
                    engine="raytrace",
                    samples=3,
                    width=520,
                    height=380,
                    show_grid=False,
                    save_path="renders/gamer_room_ps2_closeup_raytrace.png",
                )

            video_res: dict[str, Any] = {}
            if render_video:
                video_res = await b.call(
                    "render_video_file",
                    path="renders/gamer_room_turntable.mp4",
                    kind="turntable",
                    frame_start=1,
                    frame_end=18,
                    fps=12,
                    width=360,
                    height=260,
                    engine="software",
                )
                await b.call(
                    "render_video_file",
                    path="renders/gamer_room_turntable.gif",
                    kind="turntable",
                    frame_start=1,
                    frame_end=14,
                    fps=10,
                    width=320,
                    height=240,
                    engine="software",
                )

            # 11. Export Game-Ready Binary FBX 7.4 (Unreal), GLB, Blender .py, & Project .frf.json
            fbx_res = await b.call(
                "export_scene",
                path="gamer_ps2_room_unreal.fbx",
                format="fbx",
                target_engine="unreal",
            )
            glb_res = await b.call("export_scene", path="gamer_ps2_room.glb", format="glb")
            py_res = await b.call("export_scene", path="gamer_ps2_room_blender.py", format="py")
            proj_res = await b.call("scene_save", path="gamer_ps2_room.frf.json")
            insp = await b.call("scene_inspect", detail="summary")

            if artifact_dir is not None:
                artifact_dir.mkdir(parents=True, exist_ok=True)
                for src in (workspace_dir / "renders").glob("*.*"):
                    if src.suffix.lower() in (".png", ".gif"):
                        (artifact_dir / src.name).write_bytes(src.read_bytes())

            return {
                "mcp_tool_calls_executed": b.call_count,
                "scene_name": insp["name"],
                "total_objects": len(insp["objects"]),
                "total_materials": len(insp["materials"]),
                "scene_bounds": insp["scene_bounds"],
                "furniture_intersections": collision_res["intersection_count"],
                "physics_cloth": phys_bake.get("cloth_objects", []),
                "exports": {
                    "project": proj_res["path"],
                    "fbx_unreal": fbx_res,
                    "glb": glb_res,
                    "blender_py": py_res,
                    "video": video_res,
                },
            }


def main() -> None:
    summary = asyncio.run(
        build_gamer_room_via_mcp(
            DEFAULT_WORKSPACE,
            artifact_dir=Path(
                "/home/antigravity/.gemini/jetski/brain/v1/default/jetski_remote_agents_3p/d25b9703-d6d3-4e45-a6c1-d3fe042f743a"
            ),
            render_raytrace=True,
            render_video=True,
        )
    )
    print(json.dumps(summary, indent=2, default=str))


if __name__ == "__main__":
    main()
