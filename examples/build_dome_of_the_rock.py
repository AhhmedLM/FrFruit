"""Build an interior + exterior + environment 3D model of the Dome of the Rock in FrFruit.

Generates:
- Complete SceneDocument (.frf.json)
- glTF 2.0 binary (.glb) with PBR materials
- Standalone Blender build script (.py)
- Curated multi-angle exterior, cutaway, interior, wireframe, and turntable renders
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np

from frfruit.core.math3d import compose_matrix, transform_points
from frfruit.core.scene import (
    CameraSpec,
    CollectionSpec,
    LightSpec,
    MaterialSpec,
    SceneDocument,
    SceneObject,
    WorldEnvironment,
)
from frfruit.core.types import Transform, Vec2, Vec3
from frfruit.geometry.curves import extrude_2d_profile, revolve_profile, sweep_curve
from frfruit.geometry.edit_ops import subdivide_mesh
from frfruit.geometry.mesh import MeshData, concat_meshes
from frfruit.geometry.primitives import make_cone, make_cube, make_cylinder, make_plane, make_sphere
from frfruit.geometry.spatial import check_intersections, evaluate_scene_meshes
from frfruit.io.exporters import export_scene_file
from frfruit.io.project import save_project
from frfruit.render.software import RenderConfig, SoftwareRenderer, image_to_png_bytes
from frfruit.render.viewport import render_animation_sequence, render_multiview_sheet

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_EXPORT_DIR = REPO_ROOT / "exports" / "dome_of_the_rock"

# Architectural constants (meters, Z-up)
Z_COURTYARD = 0.0
Z_PLAZA = 2.40
Z_MOSQUE_FLOOR = 2.68
OCT_APOTHEM = 14.50
OCT_FACE_WIDTH = 2.0 * OCT_APOTHEM * math.tan(math.pi / 8.0)  # ~12.012 m
OCT_WALL_THICK = 1.05
Z_WALL_TOP = Z_MOSQUE_FLOOR + 9.20  # 11.88 m
R_OUTER_ARCADE = 10.60
R_INNER_ARCADE = 7.60
Z_DRUM_BASE = 12.10
Z_DRUM_TOP = 20.30
Z_DOME_TOP = 30.80


def _flat_shaded(mesh: MeshData) -> MeshData:
    """Split triangle vertices so planar architectural masonry renders crisp flat normals."""
    if mesh.is_empty:
        return mesh
    verts = mesh.vertices[mesh.faces.reshape(-1)]
    faces = np.arange(len(verts), dtype=np.int64).reshape(-1, 3)
    return MeshData(vertices=verts, faces=faces)


def _xform_mesh(
    mesh: MeshData,
    location: Vec3 = (0.0, 0.0, 0.0),
    rotation_deg: Vec3 = (0.0, 0.0, 0.0),
    scale: Vec3 = (1.0, 1.0, 1.0),
) -> MeshData:
    """Return a copy of MeshData transformed into a target TRS pose."""
    mat = compose_matrix(location=location, rotation_deg=rotation_deg, scale=scale)
    return MeshData(
        vertices=transform_points(mat, mesh.vertices),
        faces=mesh.faces.copy(),
        uvs=None if mesh.uvs is None else mesh.uvs.copy(),
        face_groups=None if mesh.face_groups is None else mesh.face_groups.copy(),
    )


def _box_at(
    size: Vec3,
    center: Vec3,
    rotation_deg: Vec3 = (0.0, 0.0, 0.0),
    flat: bool = True,
) -> MeshData:
    """Create a box centered at `center`."""
    m = make_cube(size=size, origin="center")
    if flat:
        m = _flat_shaded(m)
    return _xform_mesh(m, location=center, rotation_deg=rotation_deg)


def _cyl_at(
    radius: float,
    height: float,
    base_center: Vec3,
    radius_top: float | None = None,
    segments: int = 16,
    rotation_deg: Vec3 = (0.0, 0.0, 0.0),
) -> MeshData:
    m = make_cylinder(
        radius=radius,
        height=height,
        radius_top=radius_top,
        segments=segments,
        origin="bottom",
    )
    return _xform_mesh(m, location=base_center, rotation_deg=rotation_deg)


def _upright_extrude(
    points_uv: list[Vec2],
    thickness: float,
    location: Vec3 = (0.0, 0.0, 0.0),
    rotation_z_deg: float = 0.0,
    flat: bool = True,
) -> MeshData:
    """Extrude a 2D (u, z_up) silhouette symmetrically in depth and stand it upright."""
    m = extrude_2d_profile(points_uv, height=thickness, divisions=1, center=True)
    if flat:
        m = _flat_shaded(m)
    return _xform_mesh(m, location=location, rotation_deg=(90.0, 0.0, rotation_z_deg))


def _pointed_arch_half(
    half_span: float, spring_h: float, apex_h: float, n_arc: int = 7
) -> list[Vec2]:
    """Return left-side arch curve points from (-half_span, spring_h) to (0, apex_h)."""
    rise = max(0.05, apex_h - spring_h)
    c = (rise * rise - half_span * half_span) / max(1e-6, 2.0 * half_span)
    r_arc = half_span + c
    ang_end = math.atan2(rise, -c)
    # Sweep angle around (+c, spring_h) from PI (springing at -half_span) up to ang_end
    pts: list[Vec2] = []
    for i in range(n_arc + 1):
        t = i / n_arc
        ang = math.pi + (ang_end - math.pi) * t
        u = c + r_arc * math.cos(ang)
        z = spring_h + r_arc * math.sin(ang)
        pts.append((float(u), float(z)))
    return pts


def _make_arch_spandrel_and_ablaq(
    outer_w: float,
    inner_span: float,
    spring_h: float,
    apex_h: float,
    top_h: float,
    depth: float,
    voussoir_thick: float = 0.32,
    n_voussoirs: int = 9,
) -> tuple[MeshData, MeshData, MeshData]:
    """Build (spandrel_wall, ablaq_light_blocks, ablaq_dark_blocks) in local centered u-z plane."""
    hs = inner_span * 0.5
    hw = outer_w * 0.5
    left_inner = _pointed_arch_half(hs, spring_h, apex_h, n_arc=n_voussoirs)
    right_inner = [(-u, z) for u, z in reversed(left_inner)]

    # Outer frame spandrel halves (with slight notch for the arch ring)
    arch_outer_left = _pointed_arch_half(
        hs + voussoir_thick, spring_h, apex_h + voussoir_thick * 0.92, n_arc=n_voussoirs
    )
    spandrel_left = [
        (-hw, spring_h),
        (-hs - voussoir_thick, spring_h),
        *arch_outer_left,
        (0.0, top_h),
        (-hw, top_h),
    ]
    spandrel_right = [(-u, z) for u, z in reversed(spandrel_left)]
    spandrel_mesh = concat_meshes(
        [
            _upright_extrude(spandrel_left, depth),
            _upright_extrude(spandrel_right, depth),
        ]
    )

    # Radial Ablaq alternating wedge stones across left & right halves
    full_in = [*left_inner[:-1], *right_inner]
    arch_outer_right = [(-u, z) for u, z in reversed(arch_outer_left)]
    full_out = [*arch_outer_left[:-1], *arch_outer_right]

    # Sample n_voussoirs evenly along the full arch (odd count centers keystone on apex)
    light_parts: list[MeshData] = []
    dark_parts: list[MeshData] = []
    total_pts = len(full_in) - 1
    for vi in range(n_voussoirs):
        i0 = int(round(vi * total_pts / n_voussoirs))
        i1 = int(round((vi + 1) * total_pts / n_voussoirs))
        if i1 <= i0:
            i1 = min(total_pts, i0 + 1)
        poly = [*full_in[i0 : i1 + 1], *reversed(full_out[i0 : i1 + 1])]
        wedge = _upright_extrude(poly, depth + 0.03)
        if vi % 2 == 0:
            dark_parts.append(wedge)
        else:
            light_parts.append(wedge)

    return spandrel_mesh, concat_meshes(light_parts), concat_meshes(dark_parts)


def _add_obj(
    scene: SceneDocument,
    oid: str,
    name: str,
    mesh: MeshData,
    material_id: str,
    collection_id: str = "default",
    location: Vec3 = (0.0, 0.0, 0.0),
    rotation_deg: Vec3 = (0.0, 0.0, 0.0),
    face_material_ids: dict[int, str] | None = None,
) -> None:
    scene.objects[oid] = SceneObject(
        id=oid,
        name=name,
        kind="mesh",
        collection_id=collection_id,
        material_id=material_id,
        face_material_ids=face_material_ids or {},
        transform=Transform(location=location, rotation_deg=rotation_deg),
        mesh=mesh.to_payload(),
    )


def _register_materials(scene: SceneDocument) -> None:
    specs = [
        MaterialSpec(
            id="mat_stone_jerusalem",
            name="Jerusalem Limestone",
            base_color=(0.76, 0.67, 0.52, 1.0),
            secondary_color=(0.64, 0.55, 0.41, 1.0),
            pattern="noise",
            pattern_scale=0.38,
            roughness=0.72,
        ),
        MaterialSpec(
            id="mat_plaza_stone",
            name="Haram Plaza Paving",
            base_color=(0.80, 0.74, 0.61, 1.0),
            secondary_color=(0.69, 0.62, 0.50, 1.0),
            pattern="checker",
            pattern_scale=0.25,
            roughness=0.62,
        ),
        MaterialSpec(
            id="mat_courtyard_stone",
            name="Lower Courtyard Stone",
            base_color=(0.68, 0.61, 0.49, 1.0),
            secondary_color=(0.58, 0.52, 0.41, 1.0),
            pattern="noise",
            pattern_scale=0.24,
            roughness=0.76,
        ),
        MaterialSpec(
            id="mat_marble_cream",
            name="Jerusalem Cream Marble",
            base_color=(0.82, 0.77, 0.67, 1.0),
            secondary_color=(0.66, 0.60, 0.50, 1.0),
            pattern="wood",
            pattern_scale=0.45,
            roughness=0.28,
        ),
        MaterialSpec(
            id="mat_marble_rose",
            name="Rose Column Marble",
            base_color=(0.78, 0.58, 0.50, 1.0),
            secondary_color=(0.65, 0.46, 0.39, 1.0),
            pattern="wood",
            pattern_scale=0.55,
            roughness=0.25,
        ),
        MaterialSpec(
            id="mat_floor_mosaic",
            name="Sanctuary Marble Mosaic",
            base_color=(0.84, 0.79, 0.70, 1.0),
            secondary_color=(0.48, 0.38, 0.32, 1.0),
            pattern="checker",
            pattern_scale=0.72,
            roughness=0.18,
        ),
        MaterialSpec(
            id="mat_stone_dark_ablaq",
            name="Ablaq Umber Stone",
            base_color=(0.35, 0.19, 0.14, 1.0),
            secondary_color=(0.24, 0.13, 0.10, 1.0),
            pattern="noise",
            pattern_scale=0.65,
            roughness=0.54,
        ),
        MaterialSpec(
            id="mat_tile_cobalt",
            name="Qashani Cobalt Tiles",
            base_color=(0.05, 0.18, 0.54, 1.0),
            secondary_color=(0.07, 0.32, 0.68, 1.0),
            pattern="noise",
            pattern_scale=1.20,
            roughness=0.18,
        ),
        MaterialSpec(
            id="mat_tile_turquoise",
            name="Iznik Turquoise Tiles",
            base_color=(0.04, 0.48, 0.54, 1.0),
            secondary_color=(0.07, 0.62, 0.60, 1.0),
            pattern="noise",
            pattern_scale=1.35,
            roughness=0.18,
        ),
        MaterialSpec(
            id="mat_tile_gold_band",
            name="Gold Calligraphic Frieze",
            base_color=(0.90, 0.68, 0.18, 1.0),
            secondary_color=(0.05, 0.14, 0.42, 1.0),
            pattern="checker",
            pattern_scale=8.50,
            metallic=0.55,
            roughness=0.22,
        ),
        MaterialSpec(
            id="mat_stained_glass",
            name="Mashrabiya Stained Glass",
            base_color=(0.05, 0.32, 0.42, 0.94),
            secondary_color=(0.84, 0.48, 0.10, 0.94),
            emission_color=(0.90, 0.58, 0.20, 1.0),
            emission_strength=0.24,
            pattern="checker",
            pattern_scale=9.00,
            roughness=0.14,
        ),
        MaterialSpec(
            id="mat_gold_dome",
            name="24k Golden Dome Leaf",
            base_color=(0.96, 0.77, 0.25, 1.0),
            metallic=1.0,
            roughness=0.14,
        ),
        MaterialSpec(
            id="mat_bronze_gold",
            name="Gilded Sanctuary Bronze",
            base_color=(0.76, 0.56, 0.18, 1.0),
            metallic=0.88,
            roughness=0.28,
        ),
        MaterialSpec(
            id="mat_dome_interior",
            name="Crimson Gold Dome Ceiling",
            base_color=(0.46, 0.06, 0.06, 1.0),
            secondary_color=(0.80, 0.55, 0.14, 1.0),
            pattern="gradient",
            pattern_scale=0.42,
            metallic=0.20,
            roughness=0.38,
        ),
        MaterialSpec(
            id="mat_wood_cedar",
            name="Carved Cedar Wood",
            base_color=(0.38, 0.20, 0.10, 1.0),
            secondary_color=(0.24, 0.12, 0.06, 1.0),
            pattern="wood",
            pattern_scale=3.0,
            roughness=0.52,
        ),
        MaterialSpec(
            id="mat_sacred_rock",
            name="Al-Sakhra Sacred Limestone",
            base_color=(0.56, 0.47, 0.34, 1.0),
            secondary_color=(0.35, 0.27, 0.19, 1.0),
            pattern="noise",
            pattern_scale=0.95,
            roughness=0.85,
        ),
        MaterialSpec(
            id="mat_lead_roof",
            name="Weathered Lead Roof",
            base_color=(0.40, 0.43, 0.46, 1.0),
            metallic=0.50,
            roughness=0.48,
        ),
        MaterialSpec(
            id="mat_water",
            name="Al-Kas Fountain Water",
            base_color=(0.06, 0.38, 0.48, 0.92),
            roughness=0.06,
        ),
        MaterialSpec(
            id="mat_bark",
            name="Tree Trunk Bark",
            base_color=(0.32, 0.21, 0.13, 1.0),
            roughness=0.88,
        ),
        MaterialSpec(
            id="mat_foliage_cypress",
            name="Cypress Deep Green",
            base_color=(0.12, 0.30, 0.14, 1.0),
            secondary_color=(0.08, 0.21, 0.09, 1.0),
            pattern="noise",
            pattern_scale=1.2,
            roughness=0.78,
        ),
        MaterialSpec(
            id="mat_foliage_olive",
            name="Olive Silver-Green",
            base_color=(0.28, 0.39, 0.22, 1.0),
            secondary_color=(0.20, 0.30, 0.17, 1.0),
            pattern="noise",
            pattern_scale=1.4,
            roughness=0.75,
        ),
        MaterialSpec(
            id="mat_lantern_glow",
            name="Warm Sanctuary Lantern",
            base_color=(1.0, 0.85, 0.45, 1.0),
            emission_color=(1.0, 0.80, 0.38, 1.0),
            emission_strength=1.9,
            roughness=0.20,
        ),
        MaterialSpec(
            id="mat_hill_limestone",
            name="Jerusalem Hillside",
            base_color=(0.62, 0.56, 0.44, 1.0),
            secondary_color=(0.44, 0.44, 0.32, 1.0),
            pattern="noise",
            pattern_scale=0.14,
            roughness=0.90,
        ),
    ]
    for m in specs:
        scene.materials[m.id] = m


def _create_collections(scene: SceneDocument) -> None:
    colls = [
        ("Env_Terrain", "Surrounding Terrain & Old City"),
        ("Env_Plaza", "Upper Platform & Stairways"),
        ("Env_Mawazin", "Mawazin Arcades"),
        ("Env_ChainDome", "Cupola of the Chain & Fountain"),
        ("Env_Greenery", "Cypress & Olive Trees"),
        ("Mosque_Ext_Static", "Mosque Exterior (North/East/West)"),
        ("Mosque_Ext_CutawayFront", "Mosque Front Cutaway Section (South/SE)"),
        ("Mosque_Porticoes", "Four Cardinal Porticoes"),
        ("Mosque_Drum_Dome", "Drum & Golden Dome"),
        ("Mosque_Interior_Arcade", "Interior Ablaq Colonnades & Floor"),
        ("Mosque_Interior_Rock", "Sacred Rock & Cave Enclosure"),
        ("Mosque_Interior_Ceiling", "Painted Inner Dome & Chandelier"),
    ]
    for cid, cname in colls:
        scene.collections[cid] = CollectionSpec(id=cid, name=cname)


def _build_environment_and_courtyard(scene: SceneDocument) -> None:
    """Build lower Haram courtyard, Jerusalem hills horizon, minarets, plaza, stairs, & Mawazin."""
    # 1. Subdivided lower Haram courtyard (108 x 108m, 24x24 grid -> triangles 4.5m max edge)
    court = _flat_shaded(make_plane(size=(108.0, 108.0), subdivisions=24))
    _add_obj(
        scene, "env_courtyard", "LowerHaramCourtyard", court, "mat_courtyard_stone", "Env_Terrain"
    )

    # 2. Surrounding Haram perimeter wall + arched west/north colonnades + 2 stone minarets
    perim_parts: list[MeshData] = [
        _box_at((106.0, 1.8, 4.2), (0.0, 53.0, 2.1)),
        _box_at((1.8, 106.0, 4.2), (-53.0, 0.0, 2.1)),
        _box_at((106.0, 1.8, 3.6), (0.0, -53.0, 1.8)),
        _box_at((1.8, 106.0, 3.6), (53.0, 0.0, 1.8)),
    ]
    # North & West perimeter riwaq piers/arches
    for i in range(-5, 6):
        perim_parts.append(_box_at((1.4, 1.4, 4.4), (i * 8.0, 48.0, 2.2)))
        perim_parts.append(_box_at((8.2, 1.4, 1.2), (i * 8.0, 48.0, 4.9)))
        perim_parts.append(_box_at((1.4, 1.4, 4.4), (-48.0, i * 8.0, 2.2)))
        perim_parts.append(_box_at((1.4, 8.2, 1.2), (-48.0, i * 8.0, 4.9)))
    _add_obj(
        scene,
        "env_perim_walls",
        "HaramPerimeterRiwaq",
        concat_meshes(perim_parts),
        "mat_stone_jerusalem",
        "Env_Terrain",
    )

    # Two slender Mamluk Stone Minarets framing North & West skyline behind hero & aerial views
    minaret_stone: list[MeshData] = []
    minaret_gold: list[MeshData] = []
    for mx, my, mh in ((-26.0, 46.0, 28.5), (-46.0, -6.0, 26.0)):
        minaret_stone.append(_box_at((4.2, 4.2, mh * 0.68), (mx, my, mh * 0.34)))
        minaret_stone.append(_box_at((4.9, 4.9, 0.7), (mx, my, mh * 0.68 + 0.35)))
        minaret_stone.append(_cyl_at(1.45, mh * 0.24, (mx, my, mh * 0.68 + 0.7), segments=12))
        cap_z = mh * 0.92 + 0.7
        minaret_gold.append(_xform_mesh(make_sphere(1.45, 16, origin="bottom"), (mx, my, cap_z)))
        minaret_gold.append(_cyl_at(0.10, 2.1, (mx, my, cap_z + 1.3), segments=8))
    _add_obj(
        scene,
        "env_minarets_stone",
        "HaramMinarets",
        concat_meshes(minaret_stone),
        "mat_stone_jerusalem",
        "Env_Terrain",
    )
    _add_obj(
        scene,
        "env_minarets_domes",
        "MinaretFinials",
        concat_meshes(minaret_gold),
        "mat_gold_dome",
        "Env_Terrain",
    )

    # 3. Continuous Surrounding Earth + Sculpted Jerusalem Hills Amphitheater (horizon r=58..165 m)
    outer_earth = _flat_shaded(make_plane(size=(340.0, 340.0), subdivisions=16))
    _add_obj(
        scene,
        "env_surrounding_earth",
        "SurroundingEarthApron",
        outer_earth,
        "mat_hill_limestone",
        "Env_Terrain",
        location=(0.0, 0.0, -0.18),
    )

    n_theta = 48
    ring_radii = [56.0, 70.0, 86.0, 106.0, 128.0, 156.0]
    hill_verts: list[list[float]] = []
    hill_faces: list[list[int]] = []
    for rad in ring_radii:
        for ti in range(n_theta):
            ang = math.radians(ti * (360.0 / n_theta))
            hx = rad * math.cos(ang)
            hy = rad * math.sin(ang)
            d_box = max(abs(hx), abs(hy))
            rise = max(0.0, (d_box - 58.0) / 82.0) ** 1.25
            ridge = 1.0 + 0.34 * math.sin(2.0 * ang + 0.5) + 0.22 * math.cos(3.0 * ang - 0.8)
            hz = -0.14 + 10.5 * rise * ridge
            hill_verts.append([hx, hy, hz])
    for ri in range(len(ring_radii) - 1):
        for ti in range(n_theta):
            t_next = (ti + 1) % n_theta
            p00 = ri * n_theta + ti
            p10 = (ri + 1) * n_theta + ti
            p01 = ri * n_theta + t_next
            p11 = (ri + 1) * n_theta + t_next
            hill_faces.append([p00, p10, p11])
            hill_faces.append([p00, p11, p01])
    hills_mesh = _flat_shaded(
        MeshData(
            vertices=np.array(hill_verts, dtype=np.float64),
            faces=np.array(hill_faces, dtype=np.int64),
        )
    )
    _add_obj(
        scene,
        "env_jerusalem_hills",
        "JerusalemHillsBackdrop",
        hills_mesh,
        "mat_hill_limestone",
        "Env_Terrain",
    )

    # 4. Elevated Sahn Platform (62 x 58 m, top at Z_PLAZA = 2.40 m), subdivided top slabs
    # Build platform retaining core + subdivided surface slabs + parapet with 4 stair openings
    plaza_w, plaza_d = 62.0, 58.0
    plaza_core = _box_at((plaza_w, plaza_d, Z_PLAZA - 0.12), (0.0, 0.0, (Z_PLAZA - 0.12) * 0.5))
    # Subdivided paving surface tiles cleanly at Z = Z_PLAZA
    tile_nx, tile_ny = 10, 10
    tw, td = plaza_w / tile_nx, plaza_d / tile_ny
    deck_parts: list[MeshData] = []
    for ix in range(tile_nx):
        for iy in range(tile_ny):
            cx = -plaza_w * 0.5 + (ix + 0.5) * tw
            cy = -plaza_d * 0.5 + (iy + 0.5) * td
            deck_parts.append(_box_at((tw - 0.04, td - 0.04, 0.12), (cx, cy, Z_PLAZA - 0.06)))
    _add_obj(
        scene,
        "plaza_deck",
        "UpperPlatformDeck",
        concat_meshes(deck_parts),
        "mat_plaza_stone",
        "Env_Plaza",
    )
    _add_obj(
        scene,
        "plaza_retaining",
        "PlatformRetainingWall",
        plaza_core,
        "mat_stone_jerusalem",
        "Env_Plaza",
    )

    # Perimeter balustrade (gap = 9.4 m centered at N/S/E/W for grand stairways!)
    stair_w = 9.4
    parapet_parts: list[MeshData] = []
    par_h, par_t = 0.85, 0.55
    pz = Z_PLAZA + par_h * 0.5
    seg_x = (plaza_w * 0.5) - (stair_w * 0.5)
    seg_y = (plaza_d * 0.5) - (stair_w * 0.5)
    for sy_sign in (-1.0, 1.0):
        py = sy_sign * (plaza_d * 0.5 - par_t * 0.5)
        for sx_sign in (-1.0, 1.0):
            parapet_parts.append(
                _box_at((seg_x, par_t, par_h), (sx_sign * (stair_w * 0.5 + seg_x * 0.5), py, pz))
            )
    for sx_sign in (-1.0, 1.0):
        px = sx_sign * (plaza_w * 0.5 - par_t * 0.5)
        for sy_sign in (-1.0, 1.0):
            parapet_parts.append(
                _box_at((par_t, seg_y, par_h), (px, sy_sign * (stair_w * 0.5 + seg_y * 0.5), pz))
            )

    # 4 Grand Stone Stairways (10 steps rising from Z=0 to Z=2.40) + Mawazin Arcades
    n_steps = 10
    run_total = 4.6
    step_d = run_total / n_steps
    step_h = Z_PLAZA / n_steps
    mawazin_stone: list[MeshData] = []
    mawazin_marble: list[MeshData] = []
    mawazin_dark: list[MeshData] = []

    spandrel_unit, ablaq_l_unit, ablaq_d_unit = _make_arch_spandrel_and_ablaq(
        outer_w=2.85,
        inner_span=2.30,
        spring_h=0.0,
        apex_h=1.45,
        top_h=2.15,
        depth=0.62,
        voussoir_thick=0.28,
        n_voussoirs=7,
    )

    stair_configs = [
        ("S", 0.0, -plaza_d * 0.5, 0.0),
        ("N", 0.0, plaza_d * 0.5, 180.0),
        ("E", plaza_w * 0.5, 0.0, 90.0),
        ("W", -plaza_w * 0.5, 0.0, -90.0),
    ]
    for _, sx, sy, rz in stair_configs:
        local_stairs: list[MeshData] = []
        for s_i in range(n_steps):
            h_cur = (n_steps - s_i) * step_h
            y_off = -(s_i + 0.5) * step_d
            local_stairs.append(_box_at((stair_w, step_d + 0.04, h_cur), (0.0, y_off, h_cur * 0.5)))
        # Flanking cheek walls
        for side_x in (-stair_w * 0.5 - 0.3, stair_w * 0.5 + 0.3):
            local_stairs.append(_box_at((0.55, run_total, 1.35), (side_x, -run_total * 0.5, 0.675)))
        parapet_parts.append(
            _xform_mesh(concat_meshes(local_stairs), (sx, sy, 0.0), (0.0, 0.0, rz))
        )

        # Mawazin Arcade (4 marble columns + 3 pointed Ablaq arches topping the stair!)
        bay_pitch = 2.85
        col_xs = [-1.5 * bay_pitch, -0.5 * bay_pitch, 0.5 * bay_pitch, 1.5 * bay_pitch]
        arc_loc_stone: list[MeshData] = []
        arc_loc_marb: list[MeshData] = []
        arc_loc_dark: list[MeshData] = []
        for cx in col_xs:
            arc_loc_marb.append(_box_at((0.72, 0.72, 0.32), (cx, 0.85, Z_PLAZA + 0.16)))
            arc_loc_marb.append(_cyl_at(0.24, 3.65, (cx, 0.85, Z_PLAZA + 0.32), 0.21, segments=14))
            arc_loc_stone.append(_box_at((0.68, 0.68, 0.38), (cx, 0.85, Z_PLAZA + 4.16)))
        arch_base_z = Z_PLAZA + 4.35
        for b_i in range(3):
            bx = (b_i - 1) * bay_pitch
            arc_loc_stone.append(_xform_mesh(spandrel_unit, (bx, 0.85, arch_base_z)))
            arc_loc_marb.append(_xform_mesh(ablaq_l_unit, (bx, 0.85, arch_base_z)))
            arc_loc_dark.append(_xform_mesh(ablaq_d_unit, (bx, 0.85, arch_base_z)))
        # Top entablature cornice
        arc_loc_stone.append(
            _box_at((bay_pitch * 3 + 0.85, 0.88, 0.42), (0.0, 0.85, arch_base_z + 2.36))
        )
        mawazin_stone.append(
            _xform_mesh(concat_meshes(arc_loc_stone), (sx, sy, 0.0), (0.0, 0.0, rz))
        )
        mawazin_marble.append(
            _xform_mesh(concat_meshes(arc_loc_marb), (sx, sy, 0.0), (0.0, 0.0, rz))
        )
        mawazin_dark.append(_xform_mesh(concat_meshes(arc_loc_dark), (sx, sy, 0.0), (0.0, 0.0, rz)))

    _add_obj(
        scene,
        "plaza_parapet_stairs",
        "PlazaBalustradeAndStairs",
        concat_meshes(parapet_parts),
        "mat_stone_jerusalem",
        "Env_Plaza",
    )
    _add_obj(
        scene,
        "mawazin_stone",
        "MawazinArcadesStone",
        concat_meshes(mawazin_stone),
        "mat_stone_jerusalem",
        "Env_Mawazin",
    )
    _add_obj(
        scene,
        "mawazin_marble",
        "MawazinMarbleColumns",
        concat_meshes(mawazin_marble),
        "mat_marble_cream",
        "Env_Mawazin",
    )
    _add_obj(
        scene,
        "mawazin_ablaq_dark",
        "MawazinAblaqVoussoirs",
        concat_meshes(mawazin_dark),
        "mat_stone_dark_ablaq",
        "Env_Mawazin",
    )

    # 5. Cupola of the Chain (Qubbat al-Silsila) just east of East Portico at (21.2, 0.0, Z_PLAZA)
    cx_ch, cy_ch = 21.2, 0.0
    chain_stone: list[MeshData] = [_cyl_at(4.8, 0.25, (cx_ch, cy_ch, Z_PLAZA), segments=11)]
    chain_marble: list[MeshData] = []
    # Outer 11 columns + inner 6 columns
    for r_ring, n_col in ((4.25, 11), (2.25, 6)):
        for ci in range(n_col):
            ang = math.radians(ci * (360.0 / n_col))
            px = cx_ch + r_ring * math.cos(ang)
            py = cy_ch + r_ring * math.sin(ang)
            chain_marble.append(_box_at((0.50, 0.50, 0.22), (px, py, Z_PLAZA + 0.36)))
            chain_marble.append(_cyl_at(0.16, 2.85, (px, py, Z_PLAZA + 0.47), 0.14, segments=12))
            chain_stone.append(_box_at((0.48, 0.48, 0.28), (px, py, Z_PLAZA + 3.46)))
    # 11 Pointed arches bridging the outer ring of Cupola of the Chain
    ch_side = 2.0 * 4.25 * math.sin(math.pi / 11.0)
    ch_sp, ch_al, _ = _make_arch_spandrel_and_ablaq(
        outer_w=ch_side + 0.12,
        inner_span=ch_side - 0.38,
        spring_h=0.0,
        apex_h=0.78,
        top_h=1.18,
        depth=0.40,
        voussoir_thick=0.18,
        n_voussoirs=5,
    )
    for ci in range(11):
        a0 = math.radians(ci * (360.0 / 11.0))
        a1 = math.radians((ci + 1) * (360.0 / 11.0))
        p0 = np.array([cx_ch + 4.25 * math.cos(a0), cy_ch + 4.25 * math.sin(a0), 0.0])
        p1 = np.array([cx_ch + 4.25 * math.cos(a1), cy_ch + 4.25 * math.sin(a1), 0.0])
        mid = 0.5 * (p0 + p1)
        f_deg = math.degrees(math.atan2(p1[1] - p0[1], p1[0] - p0[0]))
        chain_stone.append(
            _xform_mesh(ch_sp, (float(mid[0]), float(mid[1]), Z_PLAZA + 3.60), (0, 0, f_deg))
        )
        chain_marble.append(
            _xform_mesh(ch_al, (float(mid[0]), float(mid[1]), Z_PLAZA + 3.60), (0, 0, f_deg))
        )
    # Conical outer roof ring below miniature drum
    chain_stone.append(_cyl_at(4.55, 0.35, (cx_ch, cy_ch, Z_PLAZA + 4.78), 2.65, segments=11))
    # Turquoise tiled mini-drum
    drum_ch = _cyl_at(2.45, 1.85, (cx_ch, cy_ch, Z_PLAZA + 5.10), segments=14)
    # Golden mini-dome + finial
    dome_ch_prof = [
        (0.0, 0.0),
        (2.48, 0.0),
        (2.55, 0.85),
        (2.10, 1.90),
        (1.15, 2.75),
        (0.0, 3.15),
    ]
    dome_ch = _xform_mesh(
        revolve_profile(dome_ch_prof, segments=28), (cx_ch, cy_ch, Z_PLAZA + 6.95)
    )
    finial_ch = _cyl_at(0.07, 1.45, (cx_ch, cy_ch, Z_PLAZA + 10.05), segments=8)
    crescent_ch = sweep_curve(
        [
            (cx_ch - 0.25, cy_ch, Z_PLAZA + 11.75),
            (cx_ch, cy_ch, Z_PLAZA + 11.45),
            (cx_ch + 0.25, cy_ch, Z_PLAZA + 11.75),
        ],
        radius=0.045,
        radial_segments=6,
    )
    _add_obj(
        scene,
        "chain_stone",
        "CupolaChainMasonry",
        concat_meshes(chain_stone),
        "mat_stone_jerusalem",
        "Env_ChainDome",
    )
    _add_obj(
        scene,
        "chain_columns",
        "CupolaChainColumns",
        concat_meshes(chain_marble),
        "mat_marble_rose",
        "Env_ChainDome",
    )
    _add_obj(
        scene, "chain_drum", "CupolaChainTiledDrum", drum_ch, "mat_tile_turquoise", "Env_ChainDome"
    )
    _add_obj(
        scene,
        "chain_dome",
        "CupolaChainGoldDome",
        concat_meshes([dome_ch, finial_ch, crescent_ch]),
        "mat_gold_dome",
        "Env_ChainDome",
    )

    # 6. Al-Kas Ablution Fountain in southern lower courtyard (0.0, -40.5, 0.0)
    fx, fy = 0.0, -40.5
    kas_parts = [
        _cyl_at(4.4, 0.25, (fx, fy, Z_COURTYARD), segments=12),
        _cyl_at(3.6, 0.90, (fx, fy, 0.25), segments=12),
        _cyl_at(0.55, 1.65, (fx, fy, 0.40), 0.38, segments=12),
        _cyl_at(1.50, 0.35, (fx, fy, 1.95), 0.95, segments=16),
    ]
    kas_brass = [_cyl_at(0.18, 0.62, (fx, fy, 2.30), 0.06, segments=10)]
    for ki in range(12):
        ka = math.radians(ki * 30.0)
        kas_brass.append(
            _box_at(
                (0.28, 0.08, 0.08),
                (fx + 3.55 * math.cos(ka), fy + 3.55 * math.sin(ka), 0.85),
                (0.0, 0.0, math.degrees(ka)),
            )
        )
    kas_water = _cyl_at(3.22, 0.07, (fx, fy, 1.15), segments=16)
    _add_obj(
        scene,
        "alkas_basin",
        "AlKasAblutionFountain",
        concat_meshes(kas_parts),
        "mat_marble_cream",
        "Env_ChainDome",
    )
    _add_obj(
        scene,
        "alkas_brass",
        "AlKasSpouts",
        concat_meshes(kas_brass),
        "mat_bronze_gold",
        "Env_ChainDome",
    )
    _add_obj(scene, "alkas_water", "AlKasWater", kas_water, "mat_water", "Env_ChainDome")

    # 7. Cypress & Olive Trees around lower & upper perimeter (carefully outside walkways)
    trunks: list[MeshData] = []
    cypress_crowns: list[MeshData] = []
    olive_crowns: list[MeshData] = []

    cypress_spots = [
        (-36.0, -34.0, 9.5),
        (-38.0, -18.0, 10.5),
        (-38.0, 18.0, 10.0),
        (-36.0, 34.0, 9.8),
        (36.0, -34.0, 9.2),
        (38.0, -18.0, 10.2),
        (38.0, 18.0, 10.4),
        (36.0, 34.0, 9.5),
        (-22.0, 35.0, 8.8),
        (22.0, 35.0, 9.0),
        (-22.0, -35.0, 8.6),
        (22.0, -35.0, 8.9),
    ]
    for tx, ty, th in cypress_spots:
        trunks.append(_cyl_at(0.28, th * 0.28, (tx, ty, Z_COURTYARD), 0.20, segments=10))
        cypress_crowns.append(
            _xform_mesh(
                make_cone(radius=1.45, height=th * 0.80, segments=14, origin="bottom"),
                (tx, ty, th * 0.22),
            )
        )
        cypress_crowns.append(
            _xform_mesh(
                make_cone(radius=1.15, height=th * 0.65, segments=14, origin="bottom"),
                (tx, ty, th * 0.42),
            )
        )

    olive_spots = [
        (-26.5, 22.5, Z_PLAZA, 1.0),
        (26.5, 22.5, Z_PLAZA, 1.05),
        (-26.5, -22.5, Z_PLAZA, 0.95),
        (26.5, -22.5, Z_PLAZA, 1.0),
        (-16.0, -40.0, Z_COURTYARD, 1.15),
        (16.0, -40.0, Z_COURTYARD, 1.10),
        (-28.0, 39.0, Z_COURTYARD, 1.15),
        (28.0, 39.0, Z_COURTYARD, 1.15),
    ]
    for ox, oy, oz, sc in olive_spots:
        trunks.append(_cyl_at(0.32 * sc, 2.4 * sc, (ox, oy, oz), 0.22 * sc, segments=10))
        for dx, dy, dz, r_c in (
            (0.0, 0.0, 3.2, 2.1),
            (-1.1, 0.6, 2.8, 1.55),
            (1.1, -0.5, 2.9, 1.65),
        ):
            olive_crowns.append(
                _xform_mesh(
                    make_sphere(radius=r_c * sc, segments=12, origin="center"),
                    (ox + dx * sc, oy + dy * sc, oz + dz * sc),
                    scale=(1.15, 1.15, 0.78),
                )
            )

    _add_obj(scene, "env_trunks", "TreeTrunks", concat_meshes(trunks), "mat_bark", "Env_Greenery")
    _add_obj(
        scene,
        "env_cypress",
        "CypressTrees",
        concat_meshes(cypress_crowns),
        "mat_foliage_cypress",
        "Env_Greenery",
    )
    _add_obj(
        scene,
        "env_olives",
        "OliveTrees",
        concat_meshes(olive_crowns),
        "mat_foliage_olive",
        "Env_Greenery",
    )


def _build_octagon_wall_bay(has_door: bool) -> dict[str, list[MeshData]]:
    """Build one octagonal exterior face module centered at local x=0, y=0, z in [0, 9.2].

    Returns grouped MeshData lists keyed by material ID.
    """
    out: dict[str, list[MeshData]] = {
        "mat_marble_cream": [],
        "mat_marble_rose": [],
        "mat_tile_cobalt": [],
        "mat_tile_turquoise": [],
        "mat_tile_gold_band": [],
        "mat_stained_glass": [],
        "mat_wood_cedar": [],
        "mat_bronze_gold": [],
    }
    w = OCT_FACE_WIDTH  # ~12.012 m
    bay_w = w / 7.0  # ~1.716 m
    pier_w = 0.42
    t = OCT_WALL_THICK

    # 8 full-height vertical pilasters / corner piers cleanly spanning lower & upper registers
    for pi in range(8):
        px = -w * 0.5 + pi * bay_w
        p_cur_w = pier_w * 1.45 if pi in (0, 7) else pier_w
        out["mat_marble_cream"].append(_box_at((p_cur_w, t + 0.16, 4.60), (px, 0.0, 2.30)))
        out["mat_tile_cobalt"].append(_box_at((p_cur_w, t + 0.12, 3.65), (px, 0.0, 4.60 + 1.825)))

    # Arch spandrel unit for the 7 bays (bay_w inner width ~ 1.30 m)
    win_span = bay_w - pier_w + 0.04
    lower_spandrel, _, _ = _make_arch_spandrel_and_ablaq(
        outer_w=bay_w,
        inner_span=win_span * 0.84,
        spring_h=0.0,
        apex_h=0.72,
        top_h=1.05,
        depth=t - 0.14,
        voussoir_thick=0.14,
        n_voussoirs=5,
    )
    upper_spandrel, _, _ = _make_arch_spandrel_and_ablaq(
        outer_w=bay_w,
        inner_span=win_span * 0.84,
        spring_h=0.0,
        apex_h=0.82,
        top_h=1.18,
        depth=t - 0.12,
        voussoir_thick=0.14,
        n_voussoirs=5,
    )

    for bi in range(7):
        bx = -w * 0.5 + (bi + 0.5) * bay_w
        is_door_bay = has_door and bi == 3
        is_outer_blind = bi in (0, 6)

        if is_door_bay:
            # Tall open doorway (z=0..3.65) with recessed bronze door wings slightly ajar
            out["mat_marble_cream"].append(_xform_mesh(lower_spandrel, (bx, 0.0, 3.55)))
            # Bronze double door wings angled inward (open welcoming interior sightline!)
            for side, ang_z in ((-1.0, 52.0), (1.0, -52.0)):
                hinge_x = bx + side * (win_span * 0.44)
                door_w = win_span * 0.42
                dx = -side * (door_w * 0.5) * math.cos(math.radians(52.0))
                dy = (door_w * 0.5) * math.sin(math.radians(52.0)) - 0.15
                out["mat_bronze_gold"].append(
                    _box_at((door_w, 0.10, 3.45), (hinge_x + dx, dy, 1.725), (0.0, 0.0, ang_z))
                )
        else:
            # Lower cream marble dado wall (z=0..2.25) with recessed rose marble inlay panel
            out["mat_marble_cream"].append(_box_at((bay_w, t - 0.18, 2.25), (bx, 0.0, 1.125)))
            out["mat_marble_rose"].append(
                _box_at((win_span * 0.72, t + 0.02, 1.65), (bx, 0.0, 1.125))
            )
            if is_outer_blind:
                # Blind recessed marble bay
                out["mat_marble_cream"].append(_box_at((bay_w, t - 0.24, 2.35), (bx, 0.0, 3.425)))
            else:
                # Open arched window frame + stained glass + Mashrabiya wood lattice mullions
                out["mat_marble_cream"].append(_box_at((bay_w, t - 0.22, 0.22), (bx, 0.0, 2.36)))
                out["mat_stained_glass"].append(
                    _box_at((win_span * 0.82, 0.16, 1.95), (bx, 0.0, 3.40))
                )
                for mx_off in (-win_span * 0.20, 0.0, win_span * 0.20):
                    out["mat_wood_cedar"].append(
                        _box_at((0.05, 0.22, 1.85), (bx + mx_off, 0.0, 3.38))
                    )
                for mz_bar in (2.88, 3.38, 3.88):
                    out["mat_wood_cedar"].append(
                        _box_at((win_span * 0.82, 0.22, 0.05), (bx, 0.0, mz_bar))
                    )
                out["mat_marble_cream"].append(_xform_mesh(lower_spandrel, (bx, 0.0, 3.55)))

        # Upper Qashani Tile Zone (z = 4.60 .. 8.25)
        out["mat_tile_turquoise"].append(_box_at((bay_w, t - 0.12, 0.85), (bx, 0.0, 5.025)))
        if is_outer_blind or is_door_bay:
            # Recessed solid Qashani cobalt tile niche
            out["mat_tile_cobalt"].append(_box_at((bay_w, t - 0.20, 2.80), (bx, 0.0, 6.85)))
            out["mat_tile_turquoise"].append(
                _box_at((win_span * 0.72, t - 0.04, 1.95), (bx, 0.0, 6.50))
            )
        else:
            # Upper arched Mashrabiya stained-glass window with fine grille mullions
            out["mat_stained_glass"].append(_box_at((win_span * 0.82, 0.16, 2.15), (bx, 0.0, 6.48)))
            for mx_off in (-win_span * 0.20, 0.0, win_span * 0.20):
                out["mat_wood_cedar"].append(_box_at((0.05, 0.22, 2.05), (bx + mx_off, 0.0, 6.45)))
            for mz_bar in (5.92, 6.42, 6.92):
                out["mat_wood_cedar"].append(
                    _box_at((win_span * 0.82, 0.22, 0.05), (bx, 0.0, mz_bar))
                )
            out["mat_tile_cobalt"].append(_xform_mesh(upper_spandrel, (bx, 0.0, 7.07)))

    # Horizontal Marble Belt Course, Golden Calligraphy Frieze, & Crenellated Turquoise Parapet
    out["mat_marble_cream"].append(_box_at((w + 0.35, t + 0.26, 0.26), (0.0, 0.0, 4.60)))
    out["mat_bronze_gold"].append(_box_at((w + 0.34, t + 0.28, 0.08), (0.0, 0.0, 8.08)))
    out["mat_tile_gold_band"].append(_box_at((w + 0.30, t + 0.24, 0.56), (0.0, 0.0, 8.38)))
    out["mat_bronze_gold"].append(_box_at((w + 0.34, t + 0.28, 0.08), (0.0, 0.0, 8.68)))
    out["mat_tile_turquoise"].append(_box_at((w + 0.32, t + 0.22, 0.42), (0.0, 0.0, 8.91)))
    # Merlons / Crenellations along parapet top
    for mi in range(11):
        mx = -w * 0.5 + (mi + 0.5) * (w / 11.0)
        out["mat_tile_cobalt"].append(_box_at((0.52, t + 0.10, 0.34), (mx, 0.0, 9.28)))
    return out


def _build_portico() -> dict[str, list[MeshData]]:
    """Build one projecting cardinal entrance portico centered at local x=0, y in [0, 3.0]."""
    out: dict[str, list[MeshData]] = {
        "mat_marble_cream": [],
        "mat_marble_rose": [],
        "mat_tile_cobalt": [],
        "mat_tile_turquoise": [],
        "mat_tile_gold_band": [],
        "mat_bronze_gold": [],
    }
    # 3 stepped marble threshold stairs leading inward (-y is outward!)
    for si in range(3):
        out["mat_marble_cream"].append(
            _box_at((4.4 - si * 0.25, 0.50, 0.12), (0.0, -2.55 + si * 0.45, -0.22 + si * 0.10))
        )
    # 4 paired slender columns (2 outer at y=-2.1, 2 inner at y=-0.8)
    for col_x in (-1.65, 1.65):
        for col_y in (-2.05, -0.75):
            out["mat_marble_cream"].append(_box_at((0.56, 0.56, 0.24), (col_x, col_y, 0.12)))
            out["mat_marble_rose"].append(
                _cyl_at(0.19, 3.25, (col_x, col_y, 0.24), 0.17, segments=14)
            )
            out["mat_bronze_gold"].append(_box_at((0.54, 0.54, 0.30), (col_x, col_y, 3.64)))

    # Front arched portico pediment & side arches
    front_sp, _, _ = _make_arch_spandrel_and_ablaq(
        outer_w=4.10,
        inner_span=2.80,
        spring_h=0.0,
        apex_h=1.25,
        top_h=1.75,
        depth=0.45,
        voussoir_thick=0.22,
        n_voussoirs=7,
    )
    out["mat_tile_cobalt"].append(_xform_mesh(front_sp, (0.0, -2.05, 3.79)))
    out["mat_tile_gold_band"].append(_box_at((4.25, 0.54, 0.34), (0.0, -2.05, 5.65)))
    # Side beam roofs connecting portico back to octagon wall
    for sx in (-1.65, 1.65):
        out["mat_tile_cobalt"].append(_box_at((0.46, 2.25, 1.75), (sx, -1.05, 4.66)))
        out["mat_tile_gold_band"].append(_box_at((0.52, 2.35, 0.34), (sx, -1.05, 5.65)))
    # Roof slab capping porch
    out["mat_tile_turquoise"].append(_box_at((4.15, 2.50, 0.25), (0.0, -1.10, 5.88)))
    return out


def _build_exterior_and_dome(scene: SceneDocument) -> None:
    """Build the 8 octagonal walls, 4 porticoes, roof wedges, drum clerestory, and golden dome."""
    wall_normal_mats = _build_octagon_wall_bay(has_door=False)
    wall_door_mats = _build_octagon_wall_bay(has_door=True)
    portico_mats = _build_portico()

    # Rotate and place 8 octagon walls around center at radius = OCT_APOTHEM.
    # Rotation rz = ang_deg + 90 maps local (-Y) radially outward along ang_deg.
    # Indices: 0=E, 1=NE, 2=N, 3=NW, 4=W, 5=SW, 6=S, 7=SE. Put S & SE into CutawayFront.
    static_buckets: dict[str, list[MeshData]] = {}
    front_buckets: dict[str, list[MeshData]] = {}
    portico_buckets: dict[str, list[MeshData]] = {}

    for k in range(8):
        ang_deg = k * 45.0
        ang_rad = math.radians(ang_deg)
        rz = ang_deg + 90.0
        wx = OCT_APOTHEM * math.cos(ang_rad)
        wy = OCT_APOTHEM * math.sin(ang_rad)

        is_cardinal = (k % 2) == 0
        is_front_cut = k in (6, 7)
        src_dict = wall_door_mats if is_cardinal else wall_normal_mats
        dst_dict = front_buckets if is_front_cut else static_buckets

        for mat_id, mesh_list in src_dict.items():
            if not mesh_list:
                continue
            merged = concat_meshes(mesh_list)
            placed = _xform_mesh(merged, (wx, wy, Z_MOSQUE_FLOOR), (0.0, 0.0, rz))
            dst_dict.setdefault(mat_id, []).append(placed)

        if is_cardinal:
            dst_portico = front_buckets if is_front_cut else portico_buckets
            for mat_id, mesh_list in portico_mats.items():
                if not mesh_list:
                    continue
                merged = concat_meshes(mesh_list)
                placed = _xform_mesh(merged, (wx, wy, Z_MOSQUE_FLOOR), (0.0, 0.0, rz))
                dst_portico.setdefault(mat_id, []).append(placed)

        # Sloped lead roof wedge from outer octagon up to drum collar + interior coffered wood
        a0 = math.radians(ang_deg - 22.5)
        a1 = math.radians(ang_deg + 22.5)
        c_ang = math.cos(math.pi / 8.0)
        r_out = (OCT_APOTHEM + 0.15) / c_ang
        r_in = 7.70 / c_ang
        z_out, z_in = Z_WALL_TOP - 0.55, Z_DRUM_BASE + 0.35
        slab_thick = 0.28
        verts_w = np.array(
            [
                [r_out * math.cos(a0), r_out * math.sin(a0), z_out],
                [r_out * math.cos(a1), r_out * math.sin(a1), z_out],
                [r_in * math.cos(a1), r_in * math.sin(a1), z_in],
                [r_in * math.cos(a0), r_in * math.sin(a0), z_in],
                [r_out * math.cos(a0), r_out * math.sin(a0), z_out + slab_thick],
                [r_out * math.cos(a1), r_out * math.sin(a1), z_out + slab_thick],
                [r_in * math.cos(a1), r_in * math.sin(a1), z_in + slab_thick],
                [r_in * math.cos(a0), r_in * math.sin(a0), z_in + slab_thick],
            ],
            dtype=np.float64,
        )
        faces_w = np.array(
            [
                [4, 5, 6],
                [4, 6, 7],
                [0, 2, 1],
                [0, 3, 2],
                [0, 1, 5],
                [0, 5, 4],
                [2, 3, 7],
                [2, 7, 6],
                [0, 4, 7],
                [0, 7, 3],
                [1, 2, 6],
                [1, 6, 5],
            ],
            dtype=np.int64,
        )
        roof_wedge = _flat_shaded(MeshData(vertices=verts_w, faces=faces_w))
        dst_dict.setdefault("mat_lead_roof", []).append(roof_wedge)

        # Warm carved cedarwood coffered ceiling soffit panel + radial beams under lead roof
        r_c_out = (OCT_APOTHEM - 0.32) / c_ang
        r_c_in = 7.35 / c_ang
        zc_out, zc_in = z_out - 0.38, z_in - 0.42
        c_thick = 0.22
        verts_c = np.array(
            [
                [r_c_out * math.cos(a0), r_c_out * math.sin(a0), zc_out],
                [r_c_out * math.cos(a1), r_c_out * math.sin(a1), zc_out],
                [r_c_in * math.cos(a1), r_c_in * math.sin(a1), zc_in],
                [r_c_in * math.cos(a0), r_c_in * math.sin(a0), zc_in],
                [r_c_out * math.cos(a0), r_c_out * math.sin(a0), zc_out + c_thick],
                [r_c_out * math.cos(a1), r_c_out * math.sin(a1), zc_out + c_thick],
                [r_c_in * math.cos(a1), r_c_in * math.sin(a1), zc_in + c_thick],
                [r_c_in * math.cos(a0), r_c_in * math.sin(a0), zc_in + c_thick],
            ],
            dtype=np.float64,
        )
        ceil_panel = _flat_shaded(MeshData(vertices=verts_c, faces=faces_w))
        dst_dict.setdefault("mat_wood_cedar", []).append(ceil_panel)
        for rb in (-15.0, 0.0, 15.0):
            ba = math.radians(ang_deg + rb)
            rm = 10.95
            dst_dict.setdefault("mat_bronze_gold", []).append(
                _box_at(
                    (6.3, 0.22, 0.26),
                    (rm * math.cos(ba), rm * math.sin(ba), 0.5 * (zc_out + zc_in) - 0.14),
                    (0.0, 9.5, ang_deg + rb),
                )
            )

    for mat_id, mlist in static_buckets.items():
        _add_obj(
            scene,
            f"ext_static_{mat_id}",
            f"Exterior_{mat_id}",
            concat_meshes(mlist),
            mat_id,
            "Mosque_Ext_Static",
        )
    for mat_id, mlist in front_buckets.items():
        _add_obj(
            scene,
            f"ext_front_{mat_id}",
            f"CutawayFront_{mat_id}",
            concat_meshes(mlist),
            mat_id,
            "Mosque_Ext_CutawayFront",
        )
    for mat_id, mlist in portico_buckets.items():
        _add_obj(
            scene,
            f"ext_portico_{mat_id}",
            f"Porticoes_{mat_id}",
            concat_meshes(mlist),
            mat_id,
            "Mosque_Porticoes",
        )

    # Central Cylindrical Drum (R = 7.75..8.45, Z = 12.10..20.30) with 16 Arched Clerestory Windows
    n_drum = 16
    r_mid = 8.10
    drum_seg_w = 2.0 * r_mid * math.tan(math.pi / n_drum) + 0.04
    drum_spandrel, _, _ = _make_arch_spandrel_and_ablaq(
        outer_w=drum_seg_w,
        inner_span=drum_seg_w * 0.56,
        spring_h=0.0,
        apex_h=0.88,
        top_h=1.25,
        depth=0.72,
        voussoir_thick=0.14,
        n_voussoirs=5,
    )
    drum_cobalt: list[MeshData] = []
    drum_turq: list[MeshData] = []
    drum_gold: list[MeshData] = []
    drum_glass: list[MeshData] = []
    drum_lattice: list[MeshData] = []
    drum_trim: list[MeshData] = []

    for di in range(n_drum):
        ang_deg = di * (360.0 / n_drum) + 11.25
        ang_rad = math.radians(ang_deg)
        rz = ang_deg + 90.0
        dx, dy = r_mid * math.cos(ang_rad), r_mid * math.sin(ang_rad)
        # Lower drum wall zone (Z = 12.10 .. 14.90)
        drum_cobalt.append(
            _xform_mesh(
                _box_at((drum_seg_w, 0.72, 2.80), (0.0, 0.0, 1.40)),
                (dx, dy, Z_DRUM_BASE),
                (0, 0, rz),
            )
        )
        # Piers flanking each clerestory window (windows 2.80..4.15 + arch 4.15..5.40)
        pier_local = concat_meshes(
            [
                _box_at((drum_seg_w * 0.24, 0.72, 1.35), (-drum_seg_w * 0.38, 0.0, 3.475)),
                _box_at((drum_seg_w * 0.24, 0.72, 1.35), (drum_seg_w * 0.38, 0.0, 3.475)),
                _xform_mesh(drum_spandrel, (0.0, 0.0, 4.15)),
            ]
        )
        drum_cobalt.append(_xform_mesh(pier_local, (dx, dy, Z_DRUM_BASE), (0.0, 0.0, rz)))
        # Recessed stained-glass clerestory window + Mashrabiya lattice bars
        win_local = _box_at((drum_seg_w * 0.56, 0.14, 2.15), (0.0, 0.0, 3.92))
        drum_glass.append(_xform_mesh(win_local, (dx, dy, Z_DRUM_BASE), (0.0, 0.0, rz)))
        for mx_off in (-drum_seg_w * 0.14, 0.0, drum_seg_w * 0.14):
            drum_lattice.append(
                _xform_mesh(
                    _box_at((0.05, 0.20, 2.00), (mx_off, 0.0, 3.88)),
                    (dx, dy, Z_DRUM_BASE),
                    (0.0, 0.0, rz),
                )
            )
        for mz_bar in (3.35, 3.90, 4.45):
            drum_lattice.append(
                _xform_mesh(
                    _box_at((drum_seg_w * 0.56, 0.20, 0.05), (0.0, 0.0, mz_bar)),
                    (dx, dy, Z_DRUM_BASE),
                    (0.0, 0.0, rz),
                )
            )
        # Upper Turquoise & Gold Calligraphy Drum Collar (Z = 17.50 .. 20.30)
        drum_turq.append(
            _xform_mesh(
                _box_at((drum_seg_w + 0.08, 0.78, 1.15), (0.0, 0.0, 5.975)),
                (dx, dy, Z_DRUM_BASE),
                (0.0, 0.0, rz),
            )
        )
        drum_trim.append(
            _xform_mesh(
                _box_at((drum_seg_w + 0.14, 0.88, 0.09), (0.0, 0.0, 6.53)),
                (dx, dy, Z_DRUM_BASE),
                (0.0, 0.0, rz),
            )
        )
        drum_gold.append(
            _xform_mesh(
                _box_at((drum_seg_w + 0.12, 0.84, 0.85), (0.0, 0.0, 6.975)),
                (dx, dy, Z_DRUM_BASE),
                (0.0, 0.0, rz),
            )
        )
        drum_trim.append(
            _xform_mesh(
                _box_at((drum_seg_w + 0.14, 0.88, 0.09), (0.0, 0.0, 7.42)),
                (dx, dy, Z_DRUM_BASE),
                (0.0, 0.0, rz),
            )
        )
        drum_turq.append(
            _xform_mesh(
                _box_at((drum_seg_w + 0.16, 0.90, 0.80), (0.0, 0.0, 7.80)),
                (dx, dy, Z_DRUM_BASE),
                (0.0, 0.0, rz),
            )
        )

    _add_obj(
        scene,
        "drum_cobalt",
        "DrumQashaniMasonry",
        concat_meshes(drum_cobalt),
        "mat_tile_cobalt",
        "Mosque_Drum_Dome",
    )
    _add_obj(
        scene,
        "drum_turq",
        "DrumTurquoiseFrieze",
        concat_meshes(drum_turq),
        "mat_tile_turquoise",
        "Mosque_Drum_Dome",
    )
    _add_obj(
        scene,
        "drum_gold",
        "DrumCalligraphyBand",
        concat_meshes(drum_gold),
        "mat_tile_gold_band",
        "Mosque_Drum_Dome",
    )
    _add_obj(
        scene,
        "drum_trim",
        "DrumGoldFillets",
        concat_meshes(drum_trim),
        "mat_bronze_gold",
        "Mosque_Drum_Dome",
    )
    _add_obj(
        scene,
        "drum_windows",
        "DrumClerestoryGlass",
        concat_meshes(drum_glass),
        "mat_stained_glass",
        "Mosque_Drum_Dome",
    )
    _add_obj(
        scene,
        "drum_lattice",
        "DrumClerestoryLattice",
        concat_meshes(drum_lattice),
        "mat_wood_cedar",
        "Mosque_Drum_Dome",
    )

    # Iconic Outer 24k Golden Dome (Syrian pointed bulbous dome profile, 48 segments)
    # Profile hollow thick outer dome so it never protrudes inside the inner decorative shell!
    dome_outer_pts: list[Vec2] = []
    n_prof = 18
    for i in range(n_prof + 1):
        t = i / n_prof
        ang = t * math.pi * 0.52
        # Bulbous lower haunch (~8.45m widening briefly to 8.68m, curving gracefully to apex)
        r = 8.45 * math.cos(ang) * (1.0 + 0.06 * math.sin(t * math.pi))
        if i == n_prof:
            r = 0.0
        z = Z_DRUM_TOP + 10.55 * math.sin(ang) + 0.40 * (t**2.2)
        dome_outer_pts.append((max(0.0, r), z))
    # Close inner hollow profile 0.35m inset so outer dome shell is clean
    for i in range(n_prof, -1, -1):
        r_out, z_out = dome_outer_pts[i]
        if i == n_prof:
            dome_outer_pts.append((0.0, z_out - 0.30))
        else:
            dome_outer_pts.append((max(0.0, r_out - 0.32), z_out - 0.16))
    outer_dome_mesh = revolve_profile(dome_outer_pts, segments=48)

    # 32 Subtle Golden Meridian Ribs hugging outer dome curve
    rib_pts: list[Vec3] = []
    for i in range(n_prof + 1):
        t = i / n_prof
        ang = t * math.pi * 0.52
        r = (8.45 * math.cos(ang) * (1.0 + 0.06 * math.sin(t * math.pi))) + 0.04
        if i == n_prof:
            r = 0.08
        z = Z_DRUM_TOP + 10.55 * math.sin(ang) + 0.40 * (t**2.2)
        rib_pts.append((r, 0.0, z))
    single_rib = sweep_curve(rib_pts, radius=0.055, radial_segments=6, taper_end=0.65)
    ribs_all = [
        _xform_mesh(single_rib, rotation_deg=(0.0, 0.0, i * (360.0 / 32.0))) for i in range(32)
    ]

    # Golden Spire Mast + 3 Tiered Orbs + Upward-Facing Islamic Crescent Finial (Z=30.8..35.0)
    z_apex = Z_DOME_TOP + 0.35
    finial_parts: list[MeshData] = [
        _cyl_at(0.18, 0.55, (0.0, 0.0, z_apex - 0.20), 0.12, segments=14),
        _cyl_at(0.075, 3.45, (0.0, 0.0, z_apex + 0.20), 0.045, segments=10),
        _xform_mesh(make_sphere(0.32, 14, origin="center"), (0.0, 0.0, z_apex + 0.95)),
        _xform_mesh(make_sphere(0.24, 14, origin="center"), (0.0, 0.0, z_apex + 1.85)),
        _xform_mesh(make_sphere(0.17, 12, origin="center"), (0.0, 0.0, z_apex + 2.60)),
    ]
    c_pts = [
        (-0.52, 0.0, z_apex + 3.85),
        (-0.34, 0.0, z_apex + 3.42),
        (0.0, 0.0, z_apex + 3.30),
        (0.34, 0.0, z_apex + 3.42),
        (0.52, 0.0, z_apex + 3.85),
    ]
    finial_parts.append(sweep_curve(c_pts, radius=0.085, radial_segments=10, taper_end=0.55))

    _add_obj(
        scene,
        "dome_outer_gold",
        "GoldenDome",
        concat_meshes([outer_dome_mesh, *ribs_all, *finial_parts]),
        "mat_gold_dome",
        "Mosque_Drum_Dome",
    )


def _build_interior_sanctuary(scene: SceneDocument) -> None:
    """Build interior mosaic floor, Ablaq arcades, Sacred Rock, railing, inner dome & chandelier."""
    # 1. Subdivided Octagonal Interior Marble Mosaic Floor flush to all 8 octagon perimeter edges
    # Radial octagon sector quads avoid dropped triangles and keep the cutaway edge clean.
    floor_parts: list[MeshData] = []
    r_oct_v = (OCT_APOTHEM - 0.08) / math.cos(math.pi / 8.0)
    n_rad = 6
    n_ang = 3
    z_top_f = Z_MOSQUE_FLOOR
    z_bot_f = Z_PLAZA - 0.02
    for oct_i in range(8):
        a0 = math.radians(oct_i * 45.0 - 22.5)
        a1 = math.radians(oct_i * 45.0 + 22.5)
        u0 = np.array([math.cos(a0), math.sin(a0), 0.0])
        u1 = np.array([math.cos(a1), math.sin(a1), 0.0])
        verts_sec: list[list[float]] = []
        faces_sec: list[list[int]] = []
        # Grid vertices along radial ring (ri=0..n_rad) and angular chord (ai=0..n_ang)
        for ri in range(n_rad + 1):
            frac_r = ri / n_rad
            for ai in range(n_ang + 1):
                frac_a = ai / n_ang
                pt_xy = r_oct_v * frac_r * ((1.0 - frac_a) * u0 + frac_a * u1)
                verts_sec.append([float(pt_xy[0]), float(pt_xy[1]), z_top_f])
        stride = n_ang + 1
        for ri in range(n_rad):
            for ai in range(n_ang):
                p00 = ri * stride + ai
                p10 = (ri + 1) * stride + ai
                p01 = p00 + 1
                p11 = p10 + 1
                if ri == 0:
                    faces_sec.append([p00, p10, p11])
                else:
                    faces_sec.append([p00, p10, p11])
                    faces_sec.append([p00, p11, p01])
        floor_parts.append(_flat_shaded(MeshData(np.array(verts_sec), np.array(faces_sec))))
        # Octagonal marble threshold plinth apron around outer octagon face
        v_o0 = r_oct_v * u0
        v_o1 = r_oct_v * u1
        edge_v = np.array(
            [
                [v_o0[0], v_o0[1], z_bot_f],
                [v_o1[0], v_o1[1], z_bot_f],
                [v_o1[0], v_o1[1], z_top_f],
                [v_o0[0], v_o0[1], z_top_f],
            ],
            dtype=np.float64,
        )
        floor_parts.append(
            _flat_shaded(MeshData(edge_v, np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int64)))
        )

    _add_obj(
        scene,
        "int_marble_floor",
        "SanctuaryMarbleFloor",
        concat_meshes(floor_parts),
        "mat_floor_mosaic",
        "Mosque_Interior_Arcade",
    )

    # 2. Outer Octagonal Arcade (r = 10.60 m): 8 piers + 16 columns + 24 Ablaq arches + beams
    outer_cream: list[MeshData] = []
    outer_rose: list[MeshData] = []
    outer_dark: list[MeshData] = []
    outer_cedar: list[MeshData] = []

    r_out_arc = R_OUTER_ARCADE
    side_len_out = 2.0 * r_out_arc * math.sin(math.pi / 8.0)  # ~8.11 m between corners
    bay_out_w = side_len_out / 3.0  # ~2.70 m center-to-center across 3 bays per side
    o_sp, o_al, o_ad = _make_arch_spandrel_and_ablaq(
        outer_w=bay_out_w,
        inner_span=bay_out_w - 0.56,
        spring_h=0.0,
        apex_h=1.45,
        top_h=2.05,
        depth=0.58,
        voussoir_thick=0.28,
        n_voussoirs=7,
    )
    for si in range(8):
        a0 = math.radians(si * 45.0 + 22.5)
        a1 = math.radians((si + 1) * 45.0 + 22.5)
        p0 = np.array([r_out_arc * math.cos(a0), r_out_arc * math.sin(a0), 0.0])
        p1 = np.array([r_out_arc * math.cos(a1), r_out_arc * math.sin(a1), 0.0])
        # Corner pier at p0
        outer_cream.append(
            _box_at((1.05, 1.05, 5.20), (float(p0[0]), float(p0[1]), Z_MOSQUE_FLOOR + 2.60))
        )
        outer_dark.append(
            _box_at((1.10, 1.10, 0.34), (float(p0[0]), float(p0[1]), Z_MOSQUE_FLOOR + 5.37))
        )

        # Side normal facing outward = mid angle (si*45 + 45) deg
        side_dir = p1 - p0
        face_ang_deg = math.degrees(math.atan2(side_dir[1], side_dir[0]))
        # 2 interior columns along edge at 1/3 and 2/3
        for frac in (1.0 / 3.0, 2.0 / 3.0):
            cp = p0 + side_dir * frac
            outer_cream.append(
                _box_at((0.72, 0.72, 0.35), (float(cp[0]), float(cp[1]), Z_MOSQUE_FLOOR + 0.175))
            )
            outer_rose.append(
                _cyl_at(
                    0.26,
                    4.45,
                    (float(cp[0]), float(cp[1]), Z_MOSQUE_FLOOR + 0.35),
                    0.23,
                    segments=14,
                )
            )
            outer_cream.append(
                _box_at((0.74, 0.74, 0.42), (float(cp[0]), float(cp[1]), Z_MOSQUE_FLOOR + 5.01))
            )

        # 3 Pointed Ablaq Arches + Cedar Tie-Beam spanning p0 -> p1
        mid_pt = 0.5 * (p0 + p1)
        outer_cedar.append(
            _box_at(
                (side_len_out - 0.4, 0.24, 0.24),
                (float(mid_pt[0]), float(mid_pt[1]), Z_MOSQUE_FLOOR + 5.20),
                (0.0, 0.0, face_ang_deg),
            )
        )
        for b_i in range(3):
            bp = p0 + side_dir * ((b_i + 0.5) / 3.0)
            loc_b = (float(bp[0]), float(bp[1]), Z_MOSQUE_FLOOR + 5.35)
            outer_cream.append(_xform_mesh(o_sp, loc_b, (0.0, 0.0, face_ang_deg)))
            outer_cream.append(_xform_mesh(o_al, loc_b, (0.0, 0.0, face_ang_deg)))
            outer_dark.append(_xform_mesh(o_ad, loc_b, (0.0, 0.0, face_ang_deg)))
        # Upper spandrel beam reaching sloped ceiling soffit
        outer_cream.append(
            _box_at(
                (side_len_out, 0.62, 1.55),
                (float(mid_pt[0]), float(mid_pt[1]), Z_MOSQUE_FLOOR + 8.18),
                (0.0, 0.0, face_ang_deg),
            )
        )

    # 3. Inner Circular Arcade (r = 7.60 m): 4 stout corner piers + 12 columns + 16 Ablaq arches
    r_in_arc = R_INNER_ARCADE
    n_in = 16
    bay_in_w = 2.0 * r_in_arc * math.sin(math.pi / n_in)  # ~2.96 m
    i_sp, i_al, i_ad = _make_arch_spandrel_and_ablaq(
        outer_w=bay_in_w,
        inner_span=bay_in_w - 0.55,
        spring_h=0.0,
        apex_h=1.65,
        top_h=2.40,
        depth=0.68,
        voussoir_thick=0.32,
        n_voussoirs=7,
    )
    for ii in range(n_in):
        ang = math.radians(ii * (360.0 / n_in))
        px, py = r_in_arc * math.cos(ang), r_in_arc * math.sin(ang)
        if ii % 4 == 0:
            # 4 stout structural corner piers carrying the great drum
            outer_cream.append(
                _box_at(
                    (1.18, 1.18, 6.20), (px, py, Z_MOSQUE_FLOOR + 3.10), (0, 0, math.degrees(ang))
                )
            )
            outer_dark.append(
                _box_at(
                    (1.24, 1.24, 0.36), (px, py, Z_MOSQUE_FLOOR + 6.38), (0, 0, math.degrees(ang))
                )
            )
        else:
            outer_cream.append(_box_at((0.82, 0.82, 0.42), (px, py, Z_MOSQUE_FLOOR + 0.21)))
            outer_rose.append(
                _cyl_at(0.30, 5.45, (px, py, Z_MOSQUE_FLOOR + 0.42), 0.27, segments=16)
            )
            outer_cream.append(_box_at((0.86, 0.86, 0.50), (px, py, Z_MOSQUE_FLOOR + 6.12)))

        # Arch connecting support ii -> ii+1
        a_next = math.radians((ii + 1) * (360.0 / n_in))
        p_cur = np.array([px, py, 0.0])
        p_nxt = np.array([r_in_arc * math.cos(a_next), r_in_arc * math.sin(a_next), 0.0])
        seg = p_nxt - p_cur
        mid = 0.5 * (p_cur + p_nxt)
        f_deg = math.degrees(math.atan2(seg[1], seg[0]))
        loc_arc = (float(mid[0]), float(mid[1]), Z_MOSQUE_FLOOR + 6.38)
        outer_cream.append(_xform_mesh(i_sp, loc_arc, (0.0, 0.0, f_deg)))
        outer_cream.append(_xform_mesh(i_al, loc_arc, (0.0, 0.0, f_deg)))
        outer_dark.append(_xform_mesh(i_ad, loc_arc, (0.0, 0.0, f_deg)))
        outer_cedar.append(
            _box_at(
                (bay_in_w - 0.4, 0.26, 0.26),
                (float(mid[0]), float(mid[1]), Z_MOSQUE_FLOOR + 6.20),
                (0, 0, f_deg),
            )
        )
        # Clean aligned drum collar spandrel (Z = 11.46 .. 12.14) without jutting box edges
        outer_cream.append(
            _box_at(
                (bay_in_w + 0.08, 0.56, 0.68),
                (float(mid[0]), float(mid[1]), Z_MOSQUE_FLOOR + 9.12),
                (0, 0, f_deg),
            )
        )

    _add_obj(
        scene,
        "int_arcade_marble",
        "InteriorArcadeMarble",
        concat_meshes(outer_cream),
        "mat_marble_cream",
        "Mosque_Interior_Arcade",
    )
    _add_obj(
        scene,
        "int_arcade_cols",
        "InteriorRoseColumns",
        concat_meshes(outer_rose),
        "mat_marble_rose",
        "Mosque_Interior_Arcade",
    )
    _add_obj(
        scene,
        "int_arcade_ablaq",
        "InteriorAblaqDark",
        concat_meshes(outer_dark),
        "mat_stone_dark_ablaq",
        "Mosque_Interior_Arcade",
    )
    _add_obj(
        scene,
        "int_tie_beams",
        "InteriorCedarTieBeams",
        concat_meshes(outer_cedar),
        "mat_wood_cedar",
        "Mosque_Interior_Arcade",
    )

    # 4. Sacred Rock (Al-Sakhra al-Musharrafa) + Cave of Souls Stairs + Carved Enclosure Screen
    # Sculpted organic stratified rock mass centered at (0, 0, Z_MOSQUE_FLOOR)
    rock_base = subdivide_mesh(
        make_sphere(radius=1.0, segments=22, origin="center"), levels=1, smooth=True
    )
    rv = rock_base.vertices.copy()
    # Stretch horizontally to oval rock shape (~9.4m x 8.0m x 2.35m height) with limestone strata
    rv[:, 0] *= 4.65
    rv[:, 1] *= 4.05
    rv[:, 2] = rv[:, 2] * 1.45 + 0.95
    strata = (
        0.22 * np.sin(rv[:, 0] * 1.3 + rv[:, 1] * 0.9)
        + 0.14 * np.cos(rv[:, 1] * 2.1 - rv[:, 2] * 2.4)
        + 0.10 * np.sin(rv[:, 0] * 3.1 + rv[:, 2] * 3.6)
    )
    rv += np.column_stack([strata * 0.45, strata * 0.35, strata * 0.85])
    # Flatten bottom cleanly at z=0 relative to floor and elevate slightly above floor
    rv[:, 2] = np.maximum(0.0, rv[:, 2])
    rock_mesh = MeshData(vertices=rv, faces=rock_base.faces.copy())
    rock_mesh = _xform_mesh(rock_mesh, (0.0, 0.0, Z_MOSQUE_FLOOR - 0.05))
    # Cave of Souls Portal Stone Arch & Steps tucked under South-East Rock side (+2.6, -2.6)
    cave_portal = concat_meshes(
        [
            _box_at((1.4, 1.2, 1.4), (2.7, -2.5, Z_MOSQUE_FLOOR + 0.70), (0, 0, -45.0)),
            _box_at((1.8, 0.6, 0.25), (3.4, -3.2, Z_MOSQUE_FLOOR + 0.12), (0, 0, -45.0)),
            _box_at((1.6, 0.5, 0.25), (3.1, -2.9, Z_MOSQUE_FLOOR + 0.36), (0, 0, -45.0)),
        ]
    )
    _add_obj(
        scene,
        "int_sacred_rock",
        "SacredRock_AlSakhra",
        rock_mesh,
        "mat_sacred_rock",
        "Mosque_Interior_Rock",
    )
    _add_obj(
        scene,
        "int_cave_portal",
        "WellOfSoulsPortal",
        cave_portal,
        "mat_marble_cream",
        "Mosque_Interior_Rock",
    )

    # Carved Wooden Maqsurah / Protective Balustrade encircling the Sacred Rock (r = 5.55 m)
    rail_wood: list[MeshData] = []
    n_rail = 24
    r_rail = 5.55
    for ri in range(n_rail):
        # Leave entrance gap facing SE cave steps around 315 deg
        ang_d = ri * (360.0 / n_rail)
        if 305.0 <= ang_d <= 330.0:
            continue
        ang_r = math.radians(ang_d)
        a_nx = math.radians((ri + 1) * (360.0 / n_rail))
        p0 = np.array([r_rail * math.cos(ang_r), r_rail * math.sin(ang_r), 0.0])
        p1 = np.array([r_rail * math.cos(a_nx), r_rail * math.sin(a_nx), 0.0])
        rail_wood.append(
            _box_at((0.20, 0.20, 1.48), (float(p0[0]), float(p0[1]), Z_MOSQUE_FLOOR + 0.74))
        )
        seg = p1 - p0
        seg_l = float(np.linalg.norm(seg))
        mid = 0.5 * (p0 + p1)
        f_deg = math.degrees(math.atan2(seg[1], seg[0]))
        rail_wood.append(
            _box_at(
                (seg_l, 0.12, 0.14),
                (float(mid[0]), float(mid[1]), Z_MOSQUE_FLOOR + 1.38),
                (0, 0, f_deg),
            )
        )
        rail_wood.append(
            _box_at(
                (seg_l, 0.10, 0.72),
                (float(mid[0]), float(mid[1]), Z_MOSQUE_FLOOR + 0.56),
                (0, 0, f_deg),
            )
        )
    _add_obj(
        scene,
        "int_rock_railing",
        "SacredRockWoodenScreen",
        concat_meshes(rail_wood),
        "mat_wood_cedar",
        "Mosque_Interior_Rock",
    )

    # 5. Painted Crimson & Gold Inner Dome Ceiling + Golden Meridian Ribs + Chandelier
    dome_in_pts: list[Vec2] = []
    n_in_p = 16
    for i in range(n_in_p + 1):
        t = i / n_in_p
        ang = t * math.pi * 0.50
        r = 7.62 * math.cos(ang)
        if i == n_in_p:
            r = 0.0
        z = Z_DRUM_TOP - 0.10 + 9.65 * math.sin(ang)
        dome_in_pts.append((max(0.0, r), z))
    for i in range(n_in_p, -1, -1):
        r_i, z_i = dome_in_pts[i]
        if i == n_in_p:
            dome_in_pts.append((0.0, z_i + 0.22))
        else:
            dome_in_pts.append((r_i + 0.20, z_i + 0.12))
    inner_dome = revolve_profile(dome_in_pts, segments=40)

    # 16 Golden Inner Dome Meridian Ribs + Circular Cornice & Zenith Medallion
    in_rib_pts: list[Vec3] = []
    for i in range(n_in_p + 1):
        t = i / n_in_p
        ang = t * math.pi * 0.50
        r = 7.56 * math.cos(ang)
        if i == n_in_p:
            r = 0.15
        z = Z_DRUM_TOP - 0.10 + 9.65 * math.sin(ang) - 0.04
        in_rib_pts.append((r, 0.0, z))
    in_rib_one = sweep_curve(in_rib_pts, radius=0.075, radial_segments=6, taper_end=0.75)
    in_ribs = [
        _xform_mesh(in_rib_one, rotation_deg=(0.0, 0.0, i * (360.0 / 16.0))) for i in range(16)
    ]
    cornice_pts = [
        (7.55 * math.cos(math.radians(a)), 7.55 * math.sin(math.radians(a)), Z_DRUM_TOP - 0.08)
        for a in range(0, 360, 15)
    ]
    in_ribs.append(sweep_curve(cornice_pts, radius=0.14, radial_segments=8, closed=True))
    in_ribs.append(_cyl_at(1.15, 0.14, (0.0, 0.0, Z_DRUM_TOP + 9.48), segments=24))

    # Grand Hanging Tiered Brass Chandelier over the Sacred Rock (chain from Z=29.8 down to Z=10.6)
    chand_brass: list[MeshData] = [_cyl_at(0.06, 19.2, (0.0, 0.0, 10.60), segments=8)]
    chand_bulbs: list[MeshData] = []
    for ring_r, ring_z, n_bulbs in ((3.25, 10.65, 16), (2.05, 11.45, 10), (1.05, 12.15, 6)):
        hoop_pts = [
            (ring_r * math.cos(math.radians(a)), ring_r * math.sin(math.radians(a)), ring_z)
            for a in range(0, 360, 20)
        ]
        chand_brass.append(sweep_curve(hoop_pts, radius=0.07, radial_segments=6, closed=True))
        for bi in range(n_bulbs):
            ang = math.radians(bi * (360.0 / n_bulbs))
            bx, by = ring_r * math.cos(ang), ring_r * math.sin(ang)
            chand_bulbs.append(
                _xform_mesh(make_sphere(0.14, 10, origin="center"), (bx, by, ring_z - 0.16))
            )
            if bi % 2 == 0:
                chand_brass.append(
                    sweep_curve(
                        [(bx, by, ring_z), (0.0, 0.0, ring_z + 1.25)],
                        radius=0.025,
                        radial_segments=5,
                    )
                )

    _add_obj(
        scene,
        "int_dome_painted",
        "InnerPaintedDomeCeiling",
        inner_dome,
        "mat_dome_interior",
        "Mosque_Interior_Ceiling",
    )
    _add_obj(
        scene,
        "int_dome_ribs_chand",
        "InnerGoldenRibsAndChandelier",
        concat_meshes([*in_ribs, *chand_brass]),
        "mat_bronze_gold",
        "Mosque_Interior_Ceiling",
    )
    _add_obj(
        scene,
        "int_chandelier_bulbs",
        "ChandelierLanternGlow",
        concat_meshes(chand_bulbs),
        "mat_lantern_glow",
        "Mosque_Interior_Ceiling",
    )


def _setup_lighting_and_cameras(scene: SceneDocument) -> None:
    """Configure balanced golden-hour exterior + warm interior illumination and 6 cameras."""
    scene.world = WorldEnvironment(
        background_color=(0.56, 0.66, 0.80),
        zenith_color=(0.14, 0.36, 0.72),
        horizon_color=(0.88, 0.76, 0.58),
        ambient_strength=0.24,
    )

    # Clear default starter lights & cameras cleanly
    for oid in [k for k, v in scene.objects.items() if v.kind in ("light", "camera")]:
        del scene.objects[oid]

    lights = [
        (
            "light_sun_key",
            "GoldenSunKey",
            (45.0, -45.0, 60.0),
            LightSpec(
                kind="sun",
                color=(1.0, 0.92, 0.78),
                intensity=1.60,
                direction=(-0.60, 0.42, -0.68),
                cast_shadows=True,
            ),
        ),
        (
            "light_sky_fill",
            "MediterraneanSkyFill",
            (-40.0, 35.0, 50.0),
            LightSpec(
                kind="sun",
                color=(0.56, 0.74, 1.0),
                intensity=0.36,
                direction=(0.50, -0.38, -0.78),
                cast_shadows=False,
            ),
        ),
        (
            "light_stone_bounce",
            "LimestoneBounce",
            (20.0, -25.0, -10.0),
            LightSpec(
                kind="sun",
                color=(1.0, 0.85, 0.65),
                intensity=0.20,
                direction=(0.28, 0.55, 0.78),
                cast_shadows=False,
            ),
        ),
        (
            "light_golden_rim",
            "DomeRimLight",
            (-50.0, 50.0, 30.0),
            LightSpec(
                kind="sun",
                color=(1.0, 0.88, 0.66),
                intensity=0.30,
                direction=(0.68, -0.60, -0.42),
                cast_shadows=False,
            ),
        ),
        # Interior Sanctuary & Sacred Rock Warm Chandelier Glow
        (
            "light_int_chand",
            "ChandelierWarmGlow",
            (0.0, 0.0, 11.0),
            LightSpec(kind="point", color=(1.0, 0.80, 0.45), intensity=3.4, cast_shadows=False),
        ),
        (
            "light_int_dome",
            "InnerDomeUpGlow",
            (0.0, 0.0, 21.5),
            LightSpec(kind="point", color=(1.0, 0.84, 0.52), intensity=4.6, cast_shadows=False),
        ),
        (
            "light_int_rock",
            "SacredRockAccent",
            (2.2, -2.5, 6.5),
            LightSpec(kind="point", color=(1.0, 0.89, 0.68), intensity=3.0, cast_shadows=False),
        ),
    ]
    for gx, gy, lid in (
        (-11.2, -11.2, "light_gal_sw"),
        (11.2, -11.2, "light_gal_se"),
        (-11.2, 11.2, "light_gal_nw"),
        (11.2, 11.2, "light_gal_ne"),
    ):
        lights.append(
            (
                lid,
                lid,
                (gx, gy, 7.2),
                LightSpec(kind="point", color=(1.0, 0.83, 0.54), intensity=1.5, cast_shadows=False),
            )
        )

    for lid, lname, lloc, lspec in lights:
        scene.objects[lid] = SceneObject(
            id=lid, name=lname, kind="light", transform=Transform(location=lloc), light=lspec
        )

    cams = [
        ("cam_ext_hero", "Cam_Exterior_Hero", (40.0, -47.0, 15.8), (0.5, 0.0, 13.5), 42.0),
        ("cam_aerial_site", "Cam_Aerial_Site", (52.0, -56.0, 42.0), (0.0, 0.0, 9.5), 43.0),
        ("cam_ext_portico", "Cam_Exterior_Portico", (9.2, -25.2, 6.0), (-0.3, -14.2, 10.6), 48.0),
        ("cam_cutaway", "Cam_Cutaway_Section", (36.0, -38.0, 17.5), (-0.5, 0.5, 14.5), 45.0),
        ("cam_int_rock", "Cam_Interior_Rock", (4.15, -5.95, 5.25), (-0.9, 1.8, 5.80), 64.0),
        ("cam_int_dome_up", "Cam_Interior_DomeUp", (-0.3, -4.2, 4.45), (0.0, 0.8, 25.0), 66.0),
    ]
    for cid, cname, cloc, ctgt, fov in cams:
        scene.objects[cid] = SceneObject(
            id=cid,
            name=cname,
            kind="camera",
            transform=Transform(location=cloc),
            camera=CameraSpec(
                projection="perspective",
                fov_deg=fov,
                near=0.12,
                far=320.0,
                target=ctgt,
            ),
        )
    scene.active_camera_id = "cam_ext_hero"


def build_scene() -> SceneDocument:
    """Assemble the complete Dome of the Rock SceneDocument."""
    scene = SceneDocument.new_default("DomeOfTheRock_AlQuds")
    scene.collections.clear()
    scene.collections["default"] = CollectionSpec(id="default", name="Scene Root")
    _create_collections(scene)
    _register_materials(scene)
    _build_environment_and_courtyard(scene)
    _build_exterior_and_dome(scene)
    _build_interior_sanctuary(scene)
    _setup_lighting_and_cameras(scene)
    return scene


def render_showcase(scene: SceneDocument, output_dir: Path, fast: bool = False) -> dict[str, str]:
    """Render exterior, interior, cutaway, wireframe, and turntable showcase images."""
    output_dir.mkdir(parents=True, exist_ok=True)
    rend = SoftwareRenderer()
    evaluated = evaluate_scene_meshes(scene)

    total_tris = sum(len(ev.world_mesh.faces) for ev in evaluated.values())
    print(f"Evaluated objects: {len(evaluated)}, total triangles: {total_tris:,}", flush=True)

    w_hd, h_hd = (640, 420) if fast else (960, 640)
    saved: dict[str, str] = {}

    def _save_cam(cam_id: str, fname: str, mode: str = "pbr") -> None:
        cam = scene.objects[cam_id]
        cfg = RenderConfig(
            width=w_hd,
            height=h_hd,
            mode=mode,  # type: ignore[arg-type]
            show_grid=False,
            show_axes=False,
            enable_shadows=(mode == "pbr"),
        )
        img = rend.render_image(scene, evaluated, cam, config=cfg)
        out_p = output_dir / fname
        out_p.write_bytes(image_to_png_bytes(img))
        saved[fname] = str(out_p)
        print(f"Rendered {fname} ({w_hd}x{h_hd}, {mode})", flush=True)

    # 1. Exterior Hero Shot & Aerial Site View
    _save_cam("cam_ext_hero", "dome_exterior_hero.png", mode="pbr")
    _save_cam("cam_aerial_site", "dome_aerial_site.png", mode="pbr")
    _save_cam("cam_ext_portico", "dome_portico_detail.png", mode="pbr")

    # 2. Interior Sanctuary Views (Eye-level Rock + Upward Gilded Dome Ceiling)
    _save_cam("cam_int_rock", "dome_interior_rock.png", mode="pbr")
    _save_cam("cam_int_dome_up", "dome_interior_dome.png", mode="pbr")

    # 3. Architectural Cutaway View (temporarily hide South/SE facade + front roof sectors)
    scene.collections["Mosque_Ext_CutawayFront"].visible = False
    ev_cut = evaluate_scene_meshes(scene)
    cut_cfg = RenderConfig(width=w_hd, height=h_hd, mode="pbr", show_grid=False, show_axes=False)
    cut_img = rend.render_image(scene, ev_cut, scene.objects["cam_cutaway"], config=cut_cfg)
    cut_p = output_dir / "dome_cutaway_section.png"
    cut_p.write_bytes(image_to_png_bytes(cut_img))
    saved["dome_cutaway_section.png"] = str(cut_p)
    print(f"Rendered dome_cutaway_section.png ({w_hd}x{h_hd})", flush=True)
    scene.collections["Mosque_Ext_CutawayFront"].visible = True

    # 4. Multi-View PBR Sheet on Core Mosque + Cupola of the Chain (tight 75% fill framing)
    scene.collections["Env_Terrain"].visible = False
    scene.collections["Env_Plaza"].visible = False
    scene.collections["Env_Mawazin"].visible = False
    scene.collections["Env_Greenery"].visible = False
    for oid in ("alkas_basin", "alkas_brass", "alkas_water"):
        scene.objects[oid].visible = False
    mv_bytes, _ = render_multiview_sheet(
        scene,
        tile_width=360 if fast else 460,
        tile_height=260 if fast else 340,
        mode="pbr",
        show_grid=False,
        show_axes=True,
        show_labels=False,
    )
    mv_p = output_dir / "dome_multiview_pbr.png"
    mv_p.write_bytes(mv_bytes)
    saved["dome_multiview_pbr.png"] = str(mv_p)
    print("Rendered dome_multiview_pbr.png", flush=True)

    # Restore Sahn Plaza, Mawazin & Greenery for Wireframe
    scene.collections["Env_Plaza"].visible = True
    scene.collections["Env_Mawazin"].visible = True
    scene.collections["Env_Greenery"].visible = True
    for oid in ("alkas_basin", "alkas_brass", "alkas_water"):
        scene.objects[oid].visible = True

    ev_core = evaluate_scene_meshes(scene)
    cam_wf = SceneObject(
        id="cam_wireframe",
        name="Cam_Wireframe",
        kind="camera",
        transform=Transform(location=(40.0, -42.0, 31.0)),
        camera=CameraSpec(
            projection="perspective",
            fov_deg=46.0,
            near=0.2,
            far=320.0,
            target=(0.0, 0.0, 13.5),
        ),
    )
    wf_img = rend.render_image(
        scene,
        ev_core,
        cam_wf,
        config=RenderConfig(
            width=w_hd, height=h_hd, mode="wireframe", show_grid=False, show_axes=False
        ),
    )
    wf_p = output_dir / "dome_wireframe.png"
    wf_p.write_bytes(image_to_png_bytes(wf_img))
    saved["dome_wireframe.png"] = str(wf_p)
    print("Rendered dome_wireframe.png", flush=True)

    # 5. 8-Step Turntable Animation focused tightly on Upper Sahn + Mawazin + Mosques
    scene.collections["Env_Greenery"].visible = False
    for oid in ("alkas_basin", "alkas_brass", "alkas_water"):
        scene.objects[oid].visible = False
    strip_png, gif_bytes, _ = render_animation_sequence(
        scene, kind="turntable", steps=8, width=320, height=220, mode="pbr"
    )
    (output_dir / "dome_turntable.png").write_bytes(strip_png)
    (output_dir / "dome_turntable.gif").write_bytes(gif_bytes)
    saved["dome_turntable.png"] = str(output_dir / "dome_turntable.png")
    saved["dome_turntable.gif"] = str(output_dir / "dome_turntable.gif")
    print("Rendered dome_turntable.png + dome_turntable.gif", flush=True)

    scene.collections["Env_Terrain"].visible = True
    scene.collections["Env_Greenery"].visible = True
    for oid in ("alkas_basin", "alkas_brass", "alkas_water"):
        scene.objects[oid].visible = True
    return saved


def main() -> None:
    parser = argparse.ArgumentParser(description="Build & render Dome of the Rock in FrFruit.")
    parser.add_argument("--export-dir", type=Path, default=DEFAULT_EXPORT_DIR)
    parser.add_argument("--render-dir", type=Path, default=DEFAULT_EXPORT_DIR / "renders")
    parser.add_argument("--fast", action="store_true", help="Render at 640x420 preview resolution.")
    parser.add_argument(
        "--check-collisions", action="store_true", help="Run manifold intersection check."
    )
    args = parser.parse_args()

    scene = build_scene()
    args.export_dir.mkdir(parents=True, exist_ok=True)
    save_project(scene, args.export_dir / "dome_of_the_rock.frf.json")
    export_scene_file(scene, args.export_dir / "dome_of_the_rock.glb", fmt="glb")
    export_scene_file(scene, args.export_dir / "dome_of_the_rock_blender.py", fmt="py")
    print(f"Exported project & GLB & Blender .py to {args.export_dir}", flush=True)

    if args.check_collisions:
        ev = evaluate_scene_meshes(scene)
        cols = check_intersections(ev, tolerance=1e-2)
        print(f"Intersection check count: {len(cols)}", flush=True)

    render_showcase(scene, args.render_dir, fast=args.fast)


if __name__ == "__main__":
    main()
