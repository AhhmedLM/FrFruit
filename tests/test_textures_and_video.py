"""Tests for online/procedural PBR textures and FFmpeg/GIF video rendering."""

from __future__ import annotations

import io
import json
import zipfile
from pathlib import Path
from unittest.mock import patch

import numpy as np
from PIL import Image

from frfruit.core.scene import MaterialSpec, SceneDocument, SceneObject
from frfruit.core.types import Transform
from frfruit.geometry.primitives import create_primitive_mesh
from frfruit.render.materials import evaluate_material_maps
from frfruit.render.video import render_video
from frfruit.textures.fetcher import (
    download_pbr_texture_pack,
    generate_procedural_pbr_maps,
    search_online_textures,
)


class _FakeHTTPResponse:
    def __init__(self, data: bytes) -> None:
        self._data = data

    def read(self) -> bytes:
        return self._data

    def __enter__(self) -> _FakeHTTPResponse:
        return self

    def __exit__(self, *args: object) -> None:
        return None


def _make_png_bytes(color: tuple[int, int, int], size: int = 32) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (size, size), color=color).save(buf, format="PNG")
    return buf.getvalue()


def test_polyhaven_and_ambientcg_mocked_search_and_download(tmp_path: Path) -> None:
    ph_assets = {
        "marble_01": {
            "name": "Marble 01",
            "categories": ["stone", "marble"],
            "tags": ["white", "polished"],
        },
        "rusty_metal": {
            "name": "Rusty Metal",
            "categories": ["metal"],
            "tags": ["iron"],
        },
    }
    ph_files = {
        "Diffuse": {"1k": {"png": {"url": "https://cdn.example/marble_diff.png"}}},
        "nor_gl": {"1k": {"png": {"url": "https://cdn.example/marble_nrm.png"}}},
        "arm": {"1k": {"png": {"url": "https://cdn.example/marble_arm.png"}}},
    }
    acg_assets = {
        "foundAssets": [
            {
                "assetId": "Marble012",
                "displayCategory": "Marble",
                "tags": ["marble", "tile"],
                "previewImage": {"256-PNG": "https://cdn.example/acg_prev.png"},
                "downloadFolders": {
                    "default": {
                        "downloadFiletypeCategories": {
                            "zip": {
                                "downloads": [
                                    {
                                        "attribute": "1K-PNG",
                                        "downloadLink": "https://cdn.example/Marble012_1K-PNG.zip",
                                    }
                                ]
                            }
                        }
                    }
                },
            }
        ]
    }
    acg_detail = {"foundAssets": acg_assets["foundAssets"]}

    zbuf = io.BytesIO()
    with zipfile.ZipFile(zbuf, "w") as zf:
        zf.writestr("Marble012_1K-PNG_Color.png", _make_png_bytes((220, 210, 200)))
        zf.writestr("Marble012_1K-PNG_NormalGL.png", _make_png_bytes((128, 128, 255)))
        zf.writestr("Marble012_1K-PNG_Roughness.png", _make_png_bytes((90, 90, 90)))
    zip_bytes = zbuf.getvalue()

    def fake_urlopen(req, timeout=15):
        url = req.full_url if hasattr(req, "full_url") else str(req)
        if "api.polyhaven.com/assets" in url:
            return _FakeHTTPResponse(json.dumps(ph_assets).encode())
        if "api.polyhaven.com/files/marble_01" in url:
            return _FakeHTTPResponse(json.dumps(ph_files).encode())
        if "ambientcg.com/api/v2/full_json" in url:
            if "id=Marble012" in url:
                return _FakeHTTPResponse(json.dumps(acg_detail).encode())
            return _FakeHTTPResponse(json.dumps(acg_assets).encode())
        if "marble_diff.png" in url:
            return _FakeHTTPResponse(_make_png_bytes((210, 195, 180)))
        if "marble_nrm.png" in url:
            return _FakeHTTPResponse(_make_png_bytes((128, 128, 255)))
        if "marble_arm.png" in url:
            return _FakeHTTPResponse(_make_png_bytes((200, 100, 20)))
        if "Marble012_1K-PNG.zip" in url:
            return _FakeHTTPResponse(zip_bytes)
        raise AssertionError(f"Unexpected URL: {url}")

    with patch("frfruit.textures.fetcher.urllib.request.urlopen", side_effect=fake_urlopen):
        hits = search_online_textures("marble", source="all", limit=10)
        assert any(h["source"] == "polyhaven" and h["asset_id"] == "marble_01" for h in hits)
        assert any(h["source"] == "ambientcg" and h["asset_id"] == "Marble012" for h in hits)

        ph_pack = download_pbr_texture_pack(
            "marble_01", source="polyhaven", resolution="1k", target_dir=tmp_path / "ph"
        )
        assert Path(ph_pack["maps"]["albedo"]).is_file()
        assert Path(ph_pack["maps"]["normal"]).is_file()
        # ARM channel unpacker produces individual AO, roughness, and metallic PNG files
        assert Path(ph_pack["maps"]["ao"]).is_file()
        assert Path(ph_pack["maps"]["roughness"]).is_file()
        assert Path(ph_pack["maps"]["metallic"]).is_file()

        acg_pack = download_pbr_texture_pack(
            "Marble012", source="ambientcg", resolution="1k", target_dir=tmp_path / "acg"
        )
        assert Path(acg_pack["maps"]["albedo"]).is_file()
        assert Path(acg_pack["maps"]["normal"]).is_file()
        assert Path(acg_pack["maps"]["roughness"]).is_file()


def test_procedural_pbr_maps_and_material_evaluation(tmp_path: Path) -> None:
    manifest = generate_procedural_pbr_maps("marble", out_dir=tmp_path / "marble", size=128)
    for k in ("albedo", "normal", "roughness", "metallic", "ao", "height"):
        assert Path(manifest["maps"][k]).is_file()

    mat = MaterialSpec(
        id="m1",
        name="MarbleMat",
        albedo_map=manifest["maps"]["albedo"],
        normal_map=manifest["maps"]["normal"],
        roughness_map=manifest["maps"]["roughness"],
        metallic_map=manifest["maps"]["metallic"],
        ao_map=manifest["maps"]["ao"],
        height_map=manifest["maps"]["height"],
        uv_scale=(2.0, 2.0),
    )
    pts = np.array([[0.1, 0.2, 0.0], [0.5, 0.6, 0.0], [0.9, 0.3, 0.0]], dtype=np.float64)
    nrm = np.array([[0.0, 0.0, 1.0]] * 3, dtype=np.float64)
    uvs = np.array([[0.1, 0.1], [0.4, 0.7], [0.85, 0.25]], dtype=np.float64)
    col, perturbed_n, rgh, met, ao = evaluate_material_maps(mat, pts, nrm, uvs=uvs)
    assert col.shape == (3, 3) and np.std(col) > 0.005
    assert rgh.shape == (3, 1) and met.shape == (3, 1) and ao.shape == (3, 1)
    assert np.allclose(np.linalg.norm(perturbed_n, axis=1), 1.0, atol=1e-4)


def test_render_video_mp4_webm_gif_and_camera_path(tmp_path: Path) -> None:
    sc = SceneDocument.new_default("VideoTest")
    sc.timeline.fps = 12
    sc.timeline.frame_start = 1
    sc.timeline.frame_end = 6
    cube = create_primitive_mesh("cube", size=1.0)
    sc.objects["box"] = SceneObject(
        id="box",
        name="Box",
        kind="mesh",
        transform=Transform(location=(0.0, 0.0, 0.5)),
        mesh=cube.to_payload(),
    )

    mp4_res = render_video(
        sc,
        out_path=tmp_path / "turn.mp4",
        kind="turntable",
        frame_start=1,
        frame_end=5,
        fps=12,
        width=161,  # odd pixel test rounds up to even pixels for H.264 yuv420p
        height=121,
        frames_dir=tmp_path / "frames",
    )
    assert mp4_res["format"] == "mp4"
    assert mp4_res["width"] == 162 and mp4_res["height"] == 122
    assert (tmp_path / "turn.mp4").stat().st_size > 1000
    assert (tmp_path / "frames" / "frame_0001.png").is_file()

    gif_res = render_video(
        sc,
        out_path=tmp_path / "spline.gif",
        kind="camera_path",
        frame_start=1,
        frame_end=4,
        fps=10,
        width=160,
        height=120,
        camera_waypoints=[
            {"location": [3.0, -3.0, 2.0], "target": [0.0, 0.0, 0.5], "fov_deg": 45.0},
            {"location": [-3.0, -3.0, 2.5], "target": [0.0, 0.0, 0.5], "fov_deg": 40.0},
            {"location": [0.0, 4.0, 3.0], "target": [0.0, 0.0, 0.5], "fov_deg": 50.0},
        ],
    )
    assert gif_res["format"] == "gif"
    assert (tmp_path / "spline.gif").read_bytes().startswith(b"GIF8")
