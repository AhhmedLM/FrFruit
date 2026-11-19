"""Unit, integration, MCP E2E, and Studio REST API tests for realistic Quad-Morph characters."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import numpy as np
import pytest
from mcp.types import TextContent
from starlette.testclient import TestClient

from frfruit.character import (
    CHARACTER_PRESETS,
    BodySliders,
    CharacterSurface,
    ExpressionSliders,
    add_rbf_sculpt_node,
    apply_character_delta,
    blend_characters,
    build_canonical_quad_base,
    build_or_update_character,
    compute_character_delta,
    compute_landmarks,
    extract_custom_morph_from_mesh_edit,
    extract_rbf_delta_from_mesh_edit,
    get_character_preset,
    subdivide_quad_mesh,
)
from frfruit.character.sdf_ops import eval_sdf_gradient_tetra, sd_round_cone, wendland_c2
from frfruit.core.scene import SceneDocument
from frfruit.desktop.server import create_studio_app
from frfruit.geometry.mesh import MeshData
from frfruit.io.project import load_project, save_project
from frfruit.render.viewport import render_single_view


def _parse_text_json(result) -> dict:
    assert not result.is_error, f"Unexpected tool error: {result.content}"
    texts = [c.text for c in result.content if isinstance(c, TextContent)]
    assert texts
    return json.loads(texts[0])


def test_sdf_ops_and_wendland_kernel() -> None:
    q = np.array([0.0, 0.5, 1.0, 1.5], dtype=np.float64)
    w = wendland_c2(q)
    assert w[0] == pytest.approx(1.0)
    assert 0.0 < w[1] < 1.0
    assert w[2] == pytest.approx(0.0)
    assert w[3] == pytest.approx(0.0)

    p = np.array([[1.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, -3.0]], dtype=np.float64)
    a = np.array([0.0, 0.0, 0.0], dtype=np.float64)
    b = np.array([0.0, 0.0, 1.0], dtype=np.float64)

    def _sphere_fn(pts: np.ndarray) -> np.ndarray:
        return np.linalg.norm(pts, axis=1) - 1.0

    grad = eval_sdf_gradient_tetra(_sphere_fn, p, eps=1e-4)
    norms = np.linalg.norm(grad, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-3)
    assert grad[0, 0] == pytest.approx(1.0, abs=1e-3)
    assert grad[2, 2] == pytest.approx(-1.0, abs=1e-3)

    d_cone = sd_round_cone(p, a, b, 0.5, 0.25)
    assert d_cone.shape == (3,)
    assert np.all(d_cone > 0.0)


def test_quad_base_manifold_and_catmull_clark() -> None:
    qb = build_canonical_quad_base()
    assert len(qb.verts) > 1000
    assert len(qb.quads) > 1000
    assert qb.uvs.shape == (len(qb.verts), 2)
    assert float(np.min(qb.uvs)) >= 0.0
    assert float(np.max(qb.uvs)) <= 1.0

    # Verify closed 2-manifold pure-quad topology (every undirected edge shared by exactly 2 quads)
    edge_counts: Counter[tuple[int, int]] = Counter()
    for q in qb.quads:
        a, b, c, d = (int(q[0]), int(q[1]), int(q[2]), int(q[3]))
        for u, v in ((a, b), (b, c), (c, d), (d, a)):
            edge_counts[(min(u, v), max(u, v))] += 1
    assert all(cnt == 2 for cnt in edge_counts.values()), "Quad base must be watertight 2-manifold"

    # Required anatomical edge-loop tags exist
    for required_tag in ("eye_margin_L", "eye_margin_R", "upper_lip", "lower_lip", "head"):
        assert required_tag in qb.tags and len(qb.tags[required_tag]) > 0

    # Sparse Catmull-Clark level 1 and level 2
    sub1 = subdivide_quad_mesh(qb.verts, qb.quads, qb.uvs, levels=1)
    sub2 = subdivide_quad_mesh(qb.verts, qb.quads, qb.uvs, levels=2)
    assert len(sub1.faces) == len(qb.quads) * 4 * 2
    assert len(sub2.faces) == len(qb.quads) * 16 * 2
    assert np.all(np.isfinite(sub2.vertices))
    assert np.all(np.isfinite(sub2.vertex_normals()))


def test_realistic_only_presets_and_landmarks() -> None:
    assert "stylization" not in BodySliders.model_fields
    assert "cartoon_stylized" not in CHARACTER_PRESETS
    assert "semi_stylized_hero" not in CHARACTER_PRESETS
    assert set(CHARACTER_PRESETS.keys()) == {
        "realistic_neutral",
        "realistic_male",
        "realistic_female",
        "realistic_athletic",
        "elder_portrait",
    }

    for pname in CHARACTER_PRESETS:
        spec = get_character_preset(pname)
        lm = compute_landmarks(spec)
        assert 6.5 <= lm.head_units <= 8.3
        assert sum(lm.facial_thirds) == pytest.approx(1.0, abs=0.02)
        assert lm.eye_to_face_width_ratio > 0.10


def test_surface_expressions_materials_and_armature(tmp_path: Path) -> None:
    spec = get_character_preset("realistic_male")
    spec.quality = "preview"
    surface = CharacterSurface(spec, quality="preview")
    mesh = surface.skin_mesh

    assert len(mesh.vertices) > 5000
    assert len(mesh.faces) > 10000
    assert mesh.uvs is not None and len(mesh.uvs) == len(mesh.vertices)
    assert float(np.min(mesh.uvs)) >= 0.0
    assert float(np.max(mesh.uvs)) <= 1.0
    assert np.all(np.isfinite(mesh.vertices))
    assert np.all(np.isfinite(mesh.vertex_normals()))

    # Verify facial expressions deform the eyelid margin & jaw measurably
    open_lid = surface.eyelid_margin_points(+1.0, upper=True)
    spec_expr = spec.model_copy(deep=True)
    spec_expr.expressions = ExpressionSliders(
        smile=0.8, blink_left=1.0, blink_right=0.0, brow_raise=0.6, jaw_open=0.8
    )
    surf_expr = CharacterSurface(spec_expr, quality="preview")
    closed_lid = surf_expr.eyelid_margin_points(+1.0, upper=True)
    assert float(np.mean(closed_lid[:, 2])) < float(np.mean(open_lid[:, 2])) - 0.002
    assert float(surf_expr.lm.chin[2]) < float(surface.lm.chin[2]) - 0.005

    scene = SceneDocument.new_default("CharTest")
    res = build_or_update_character(scene, spec, name="HeroMale", cache_root=tmp_path)
    obj = scene.objects[res["object_id"]]
    assert obj.character is not None
    assert obj.skin is not None
    assert res["armature_id"] in scene.objects
    arm_obj = scene.objects[res["armature_id"]]
    assert arm_obj.armature is not None
    assert len(arm_obj.armature.bones) == 19

    # Verify independent rigged child objects exist for eyes, brows, hair, top, bottom, shoes
    child_parts = res["child_parts"]
    assert set(child_parts.keys()) == {"eyes", "brows", "hair", "top", "bottom", "shoes"}
    for role, child_id in child_parts.items():
        cobj = scene.objects[child_id]
        assert cobj.parent_id == obj.id
        assert cobj.skin is not None
        assert cobj.skin.armature_id == res["armature_id"]
        assert cobj.mesh is not None and len(cobj.mesh.vertices) > 20, f"Empty {role}"

    # Verify generated procedural textures exist and scene renders cleanly
    skin_mat = scene.materials[obj.material_id]
    assert skin_mat.albedo_map and Path(skin_mat.albedo_map).is_file()
    assert skin_mat.normal_map and Path(skin_mat.normal_map).is_file()
    assert skin_mat.roughness_map and Path(skin_mat.roughness_map).is_file()
    assert skin_mat.ao_map and Path(skin_mat.ao_map).is_file()
    assert f"mat_{obj.id}_eyes" in scene.materials
    assert f"mat_{obj.id}_hair" in scene.materials
    assert f"mat_{obj.id}_top" in scene.materials

    png_bytes, _ = render_single_view(scene, width=240, height=240, mode="pbr")
    assert png_bytes.startswith(b"\x89PNG")

    # Verify switching to long_wavy + clothing_style='none' cleanly updates/removes child parts
    spec_upd = spec.model_copy(deep=True)
    spec_upd.materials.hair_style = "long_wavy"
    spec_upd.materials.clothing_style = "none"
    res2 = build_or_update_character(scene, spec_upd, object_id=obj.id, cache_root=tmp_path)
    assert set(res2["child_parts"].keys()) == {"eyes", "brows", "hair"}
    assert f"{obj.id}_top" not in scene.objects
    assert f"{obj.id}_bottom" not in scene.objects
    assert f"{obj.id}_shoes" not in scene.objects

    # Verify project save/load round-trip preserves CharacterSpec + expressions
    save_p = tmp_path / "char_scene.frf.json"
    save_project(scene, save_p)
    reloaded = load_project(save_p)
    assert reloaded.objects[obj.id].character is not None
    assert reloaded.objects[obj.id].character.preset == "realistic_male"


def test_rbf_sculpt_custom_morphs_and_delta_blending() -> None:
    male = get_character_preset("realistic_male")
    male.quality = "preview"
    female = get_character_preset("realistic_female")
    female.quality = "preview"
    athletic = get_character_preset("realistic_athletic")
    athletic.quality = "preview"

    # 1. Compute and apply facial-only delta (Male -> Female jaw, keep Male height)
    delta_face = compute_character_delta(male, female, region="face", name="fem_face")
    assert "height" not in delta_face.body_delta
    assert "jaw_width" in delta_face.face_delta
    applied = apply_character_delta(male, delta_face, weight=0.6)
    assert applied.body.height == pytest.approx(male.body.height)
    assert applied.face.jaw_width < male.face.jaw_width

    # 2. Blend toward realistic athletic preset
    blended = blend_characters(male, athletic, alpha=0.5, region="all")
    expected_m = 0.5 * male.body.muscularity + 0.5 * athletic.body.muscularity
    assert blended.body.muscularity == pytest.approx(expected_m, abs=1e-4)

    # 3. Wendland C2 RBF sculpting modifies nose tip geometry measurably
    surf_orig = CharacterSurface(male, quality="preview")
    sculpted, node = add_rbf_sculpt_node(
        male, region="nose_tip", weight=-0.06, radius=0.28, mirror_x=False
    )
    assert node.region == "nose_tip"
    surf_scul = CharacterSurface(sculpted, quality="preview")
    # Most anterior (-Y) head point (nose tip) should protrude forward further
    m_orig = surf_orig.skin_mesh
    m_scul = surf_scul.skin_mesh
    orig_head_y = m_orig.vertices[m_orig.vertices[:, 2] >= float(surf_orig.lm.chin[2]), 1]
    scul_head_y = m_scul.vertices[m_scul.vertices[:, 2] >= float(surf_orig.lm.chin[2]), 1]
    assert float(np.min(scul_head_y)) < float(np.min(orig_head_y))

    # 4. Extract custom morph target & RBF delta from direct vertex edit
    edited_verts = m_orig.vertices.copy()
    dist_tip = np.linalg.norm(edited_verts - surf_orig.lm.nose_tip[None, :], axis=1)
    mask = dist_tip < 0.045
    assert np.any(mask)
    edited_verts[mask, 1] -= 0.015 * (1.0 - dist_tip[mask] / 0.045)
    edited_mesh = MeshData(vertices=edited_verts, faces=m_orig.faces, uvs=m_orig.uvs)

    morph = extract_custom_morph_from_mesh_edit(male, edited_mesh, morph_name="aquiline_nose")
    assert morph.name == "aquiline_nose"
    assert len(morph.deltas) >= 1
    spec_with_morph = male.model_copy(deep=True)
    spec_with_morph.stored_morphs[morph.name] = morph
    spec_with_morph.custom_morphs[morph.name] = 1.0
    surf_morphed = CharacterSurface(spec_with_morph, quality="preview")
    morphed_y = surf_morphed.skin_mesh.vertices[
        surf_morphed.skin_mesh.vertices[:, 2] >= float(surf_orig.lm.chin[2]), 1
    ]
    assert float(np.min(morphed_y)) < float(np.min(orig_head_y)) - 0.003

    nodes = extract_rbf_delta_from_mesh_edit(
        m_orig, edited_mesh, surf_orig.lm, max_nodes=8, region_name="nose_edit"
    )
    assert len(nodes) >= 1
    assert any(n.space == "head_norm" for n in nodes)


@pytest.mark.anyio
async def test_character_manage_mcp_e2e(mcp_session_and_client) -> None:
    _, client, _ = mcp_session_and_client
    await client.call_tool("scene_new", {"name": "CharMCPE2E", "clear_history": True})

    presets_res = _parse_text_json(
        await client.call_tool("character_manage", {"action": "list_presets"})
    )
    assert set(CHARACTER_PRESETS.keys()).issubset(presets_res["presets"].keys())
    assert "smile" in presets_res["expressions"]
    assert "expressions" in presets_res["delta_regions"]

    c1 = _parse_text_json(
        await client.call_tool(
            "character_manage",
            {
                "action": "create",
                "preset": "realistic_neutral",
                "name": "HeroBase",
                "quality": "preview",
                "muscularity": 0.55,
            },
        )
    )
    cid = c1["object_id"]
    assert c1["vertex_count"] > 5000
    assert c1["armature_id"] is not None

    upd = _parse_text_json(
        await client.call_tool(
            "character_manage",
            {
                "action": "update",
                "object_id": cid,
                "face_params": {"jaw_width": 0.25},
                "expressions": {"smile": 0.65, "brow_raise": 0.3},
                "material_params": {"hair_style": "swept"},
            },
        )
    )
    assert upd["object_id"] == cid
    assert upd["expressions"]["smile"] == pytest.approx(0.65)

    sc_res = _parse_text_json(
        await client.call_tool(
            "character_manage",
            {
                "action": "sculpt_rbf",
                "object_id": cid,
                "sculpt_region": "cheekbone",
                "sculpt_weight": -0.03,
            },
        )
    )
    assert sc_res["rbf_node_count"] == 1

    d_res = _parse_text_json(
        await client.call_tool(
            "character_manage",
            {
                "action": "compute_delta",
                "object_id": cid,
                "preset": "realistic_athletic",
                "delta_name": "to_athletic",
                "delta_region": "body",
            },
        )
    )
    assert "muscularity" in d_res["body_delta_keys"]

    b_res = _parse_text_json(
        await client.call_tool(
            "character_manage",
            {
                "action": "blend",
                "object_id": cid,
                "preset": "realistic_female",
                "delta_region": "nose",
                "delta_weight": 0.5,
            },
        )
    )
    assert b_res["blended_with"] == "realistic_female"

    # Non-topological mesh edit -> save_morph (CustomMorphTarget)
    await client.call_tool(
        "mesh_edit",
        {
            "object_id": cid,
            "operation": "transform_selection",
            "selection_mode": "top",
            "translate": [0.0, -0.015, 0.0],
        },
    )
    sm_res = _parse_text_json(
        await client.call_tool(
            "character_manage",
            {
                "action": "save_morph",
                "object_id": cid,
                "morph_name": "crown_tilt",
            },
        )
    )
    assert sm_res["saved_morph"] == "crown_tilt"
    assert sm_res["active_vertices"] >= 1
    assert "crown_tilt" in sm_res["stored_morphs"]

    # Another non-topological mesh edit -> extract_edit_delta (RBF nodes)
    await client.call_tool(
        "mesh_edit",
        {
            "object_id": cid,
            "operation": "transform_selection",
            "selection_mode": "top",
            "translate": [0.0, -0.012, 0.0],
        },
    )
    ex_res = _parse_text_json(
        await client.call_tool(
            "character_manage",
            {
                "action": "extract_edit_delta",
                "object_id": cid,
                "sculpt_region": "crown_pull",
            },
        )
    )
    assert ex_res["extracted_nodes"] >= 1

    insp = _parse_text_json(
        await client.call_tool(
            "character_manage",
            {"action": "inspect", "object_id": cid},
        )
    )
    assert insp["object_id"] == cid
    assert 6.0 < insp["anthropometry"]["head_units"] < 9.0


def test_desktop_character_rest_endpoints(tmp_path: Path) -> None:
    app, _sess, _mcp = create_studio_app(workspace_root=tmp_path, host="127.0.0.1", port=8777)
    with TestClient(app) as client:
        pr = client.get("/api/character/presets")
        assert pr.status_code == 200
        assert "realistic_athletic" in pr.json()["presets"]
        assert "smile" in pr.json()["expressions"]

        cr = client.post(
            "/api/character/create",
            json={"preset": "realistic_athletic", "quality": "preview", "name": "StudioAthlete"},
        )
        assert cr.status_code == 200
        cid = cr.json()["object_id"]
        assert cr.json()["vertex_count"] > 5000

        up = client.post(
            f"/api/character/{cid}/update",
            json={
                "body": {"muscularity": 0.72},
                "face": {"nose_bridge": 0.3},
                "expressions": {"smile": 0.55, "blink_left": 0.4},
            },
        )
        assert up.status_code == 200
        assert up.json()["expressions"]["smile"] == pytest.approx(0.55)

        sc = client.post(
            f"/api/character/{cid}/sculpt_rbf",
            json={"region": "chin", "weight": -0.025, "radius": 0.25},
        )
        assert sc.status_code == 200
        assert sc.json()["rbf_node_count"] == 1

        bl = client.post(
            f"/api/character/{cid}/blend_delta",
            json={"preset": "realistic_female", "region": "jaw", "alpha": 0.5, "mode": "blend"},
        )
        assert bl.status_code == 200
        assert bl.json()["mode"] == "blend"

        # State should serialize character payload & material albedo_sig
        st = client.get("/api/state").json()
        char_obj = next(o for o in st["objects"] if o["id"] == cid)
        assert char_obj["character"] is not None
        geo = client.get("/api/geometry").json()
        assert any("albedo_sig" in m["material"] for m in geo["meshes"])
