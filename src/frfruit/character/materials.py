"""Realistic procedural PBR skin, eye, hair strand, and woven fabric material generator."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image

from frfruit.character.quad_base import (
    CANON_HEIGHT,
    HEAD_CENTER,
    HEAD_HALF_H,
    NECK_SEAM_Z,
    build_canonical_quad_base,
)
from frfruit.character.schema import CharacterSpec
from frfruit.character.surface import AnatomicalLandmarks, CharacterSurface
from frfruit.core.scene import MaterialSpec, SceneDocument
from frfruit.textures.loader import clear_texture_cache


def _lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * t


def _smoothstep(edge0: float, edge1: float, x: np.ndarray) -> np.ndarray:
    denom = edge1 - edge0
    if abs(denom) < 1e-6:
        denom = 1e-6
    t = np.clip((x - edge0) / denom, 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def _smooth_noise_2d(res: int, cells: int, seed: int) -> np.ndarray:
    """Fast isotropic 2D Hermite value noise on [0, 1]^2 without periodic striping."""
    rng = np.random.default_rng(seed)
    g = rng.uniform(-1.0, 1.0, size=(cells + 1, cells + 1))
    t = np.linspace(0.0, float(cells), res, endpoint=False)
    i0 = np.floor(t).astype(np.int32)
    f = t - i0
    w = f * f * (3.0 - 2.0 * f)
    row_interp = g[i0, :] * (1.0 - w[:, None]) + g[i0 + 1, :] * w[:, None]
    return row_interp[:, i0] * (1.0 - w[None, :]) + row_interp[:, i0 + 1] * w[None, :]


def _canon_pos_to_head_uv(pos: np.ndarray) -> tuple[float, float]:
    """Exact match to quad_base._head_uv for canonical 3D head coordinates."""
    rel = pos - HEAD_CENTER
    phi = float(np.arctan2(rel[0], -rel[1]))
    u = 0.50 + (phi / np.pi) * 0.46
    t_v = float(np.clip((float(pos[2]) - NECK_SEAM_Z) / (CANON_HEIGHT - NECK_SEAM_Z), 0.0, 1.0))
    v = 0.44 + t_v * 0.54
    return float(np.clip(u, 0.04, 0.96)), float(np.clip(v, 0.44, 0.98))


def _dist_to_polyline_uv(
    uu: np.ndarray,
    vv: np.ndarray,
    pts_uv: np.ndarray,
) -> np.ndarray:
    """Vectorized minimum distance in UV space from grid (uu, vv) to a 2D polyline."""
    d_min = np.full(uu.shape, 1e6, dtype=np.float64)
    for i in range(len(pts_uv) - 1):
        ax, ay = float(pts_uv[i, 0]), float(pts_uv[i, 1])
        bx, by = float(pts_uv[i + 1, 0]), float(pts_uv[i + 1, 1])
        abx, aby = bx - ax, by - ay
        l2 = abx * abx + aby * aby
        if l2 < 1e-12:
            d = np.sqrt((uu - ax) ** 2 + (vv - ay) ** 2)
        else:
            t = np.clip(((uu - ax) * abx + (vv - ay) * aby) / l2, 0.0, 1.0)
            px = ax + t * abx
            py = ay + t * aby
            d = np.sqrt((uu - px) ** 2 + (vv - py) ** 2)
        d_min = np.minimum(d_min, d)
    return d_min


def _compute_material_signature(spec: CharacterSpec, lm: AnatomicalLandmarks) -> str:
    payload = {
        "v": "3.2",
        "g": round(float(spec.body.gender), 3),
        "age": round(float(spec.body.age), 3),
        "pore": round(float(spec.face.pore_detail), 3),
        "lip_full": [
            round(float(spec.face.upper_lip_fullness), 3),
            round(float(spec.face.lower_lip_fullness), 3),
        ],
        "mat": spec.materials.model_dump(),
        "q": spec.quality,
    }
    raw = json.dumps(payload, sort_keys=True).encode("utf-8")
    return hashlib.sha1(raw).hexdigest()[:10]


def _generate_skin_maps(
    spec: CharacterSpec,
    lm: AnatomicalLandmarks,
    out_dir: Path,
    sig: str,
    surface: CharacterSurface | None = None,
) -> tuple[Path, Path, Path, Path]:
    """Synthesize realistic matte skin Albedo (with loop-locked lips), Normal, Roughness, and AO."""
    out_dir.mkdir(parents=True, exist_ok=True)
    albedo_path = out_dir / f"skin_albedo_{sig}.png"
    normal_path = out_dir / f"skin_normal_{sig}.png"
    rough_path = out_dir / f"skin_rough_{sig}.png"
    ao_path = out_dir / f"skin_ao_{sig}.png"

    if (
        albedo_path.is_file()
        and normal_path.is_file()
        and rough_path.is_file()
        and ao_path.is_file()
    ):
        return albedo_path, normal_path, rough_path, ao_path

    res = 512 if spec.quality == "preview" else 1024
    g = float(spec.body.gender)
    f = spec.face
    detail_w = float(f.pore_detail)
    age_w = float(spec.body.age)

    # Row 0 = V=1.0, Row res-1 = V=0.0
    u = np.linspace(0.0, 1.0, res, dtype=np.float64)[None, :]
    v = np.linspace(1.0, 0.0, res, dtype=np.float64)[:, None]
    uu = np.broadcast_to(u, (res, res))
    vv = np.broadcast_to(v, (res, res))

    skin = np.asarray(spec.materials.skin_tone, dtype=np.float64)
    lip_col = np.asarray(spec.materials.lip_tint, dtype=np.float64)

    alb = np.empty((res, res, 3), dtype=np.float64)
    alb[:] = skin[None, None, :]

    # Realistic natural matte human skin roughness (~0.62)
    base_rgh = max(0.52, float(spec.materials.roughness_base))
    rgh = np.full((res, res), base_rgh, dtype=np.float64)
    ao = np.ones((res, res), dtype=np.float64)
    height = np.zeros((res, res), dtype=np.float64)

    # --- 1. Organic Melanin Variation & Fine Facial Pores ---
    # Seamless continuous UV: head v >= 0.44, body v < 0.44, shared azimuth u in [0.04, 0.96]
    in_face = vv >= 0.435
    melanin_var = 0.65 * _smooth_noise_2d(res, 14, 101) + 0.35 * _smooth_noise_2d(res, 32, 202)
    pore_noise = 0.60 * _smooth_noise_2d(
        res, 128 if res >= 1024 else 64, 303
    ) + 0.40 * _smooth_noise_2d(res, 224 if res >= 1024 else 112, 404)

    face_f = in_face.astype(np.float64)
    body_f = 1.0 - face_f
    alb += detail_w * (
        (0.015 * face_f + 0.008 * body_f)[:, :, None] * melanin_var[:, :, None]
        - (0.008 * face_f)[:, :, None] * pore_noise[:, :, None]
    )
    height += detail_w * (0.22 * face_f + 0.018 * body_f) * pore_noise

    # --- 2. Cylindrical-Frontal Head UV Exact Coordinates ---
    # Inverse of quad_base._head_uv:
    # u = 0.50 + (phi / pi) * 0.46 -> phi = (u - 0.50) / 0.46 * pi
    # v = 0.44 + (z - NECK_SEAM_Z) / (CANON_HEIGHT - NECK_SEAM_Z) * 0.54
    phi = ((uu - 0.50) / 0.46) * np.pi
    z_uv = NECK_SEAM_Z + ((vv - 0.44) / 0.54) * (CANON_HEIGHT - NECK_SEAM_Z)
    zn = (z_uv - float(HEAD_CENTER[2])) / HEAD_HALF_H
    front_face = in_face & (np.abs(phi) < 0.56 * np.pi) & (zn > -1.05) & (zn < 1.02)

    # Query canonical Quad Base UV landmarks directly so every zone is 100% locked to topology!
    qb = surface.base if surface is not None else build_canonical_quad_base()
    tags = qb.tags
    uv_base = qb.uvs

    # Physiological skin color zones (warm forehead/cheeks/nose/ears, cooler beard zone on males)
    warm_sss = np.clip(skin * np.array([1.12, 0.76, 0.68]), 0.0, 1.0)
    sss_amt = float(spec.materials.subsurface_strength)

    # Cheek flush (centered around canonical cheekbone UV)
    u_ch, v_ch = _canon_pos_to_head_uv(
        HEAD_CENTER + np.array([0.044, -0.072, -0.026], dtype=np.float64)
    )
    d_cheek = np.sqrt(((np.abs(uu - 0.50) - abs(u_ch - 0.50)) * 0.80) ** 2 + (vv - v_ch) ** 2)
    cheek_m = (1.0 - _smoothstep(0.025, 0.088, d_cheek)) * front_face

    # Nose tip & alar warmth
    nt_uv = np.mean(uv_base[tags["nose_tip_v"]], axis=0)
    d_nose = np.sqrt(((uu - float(nt_uv[0])) * 0.78) ** 2 + (vv - float(nt_uv[1])) ** 2)
    nose_m = (1.0 - _smoothstep(0.014, 0.048, d_nose)) * front_face

    # Ear warmth
    ear_l_uv = np.mean(uv_base[tags["ear_L"]], axis=0)
    d_ear = np.sqrt(
        ((np.abs(uu - 0.50) - abs(float(ear_l_uv[0]) - 0.50)) * 0.75) ** 2
        + ((vv - float(ear_l_uv[1])) * 0.85) ** 2
    )
    ear_m = (1.0 - _smoothstep(0.025, 0.075, d_ear)) * in_face

    warm_mask = np.clip(0.28 * cheek_m + 0.34 * nose_m + 0.28 * ear_m, 0.0, 0.55) * sss_amt
    alb = alb * (1.0 - warm_mask[:, :, None]) + warm_sss[None, None, :] * warm_mask[:, :, None]

    # Subtle male beard/jaw coolness (physiological facial color zone)
    if g > 0.15:
        beard_zone = (zn < -0.44) & (zn > -1.04) & (np.abs(phi) < 0.36 * np.pi) & front_face
        b_mask = (1.0 - _smoothstep(0.24 * np.pi, 0.38 * np.pi, np.abs(phi))) * beard_zone
        cool_jaw = skin * np.array([0.88, 0.92, 0.96])
        b_w = 0.22 * min(1.0, g) * b_mask
        alb = alb * (1.0 - b_w[:, :, None]) + cool_jaw[None, None, :] * b_w[:, :, None]

    # Subtle T-zone slightly smoother sheen (still natural, NEVER plastic: min ~0.52)
    tzone = (
        (1.0 - _smoothstep(0.032, 0.105, np.abs(uu - 0.50)))
        * (zn > -0.40)
        * (zn < 0.55)
        * front_face
    )
    rgh -= 0.085 * tzone

    # Orbital socket warmth/depth & soft eyelid-rim AO shadow locked to eye_margin_L UV
    eye_m_uv = uv_base[tags["eye_margin_L"]]
    u_el = float(np.mean(eye_m_uv[:, 0]))
    v_el = float(np.mean(eye_m_uv[:, 1]))
    ew_uv = max(0.022, 0.5 * float(np.max(eye_m_uv[:, 0]) - np.min(eye_m_uv[:, 0])))
    eh_uv = max(0.008, 0.5 * float(np.max(eye_m_uv[:, 1]) - np.min(eye_m_uv[:, 1])))
    du_e = np.abs(uu - 0.50) - abs(u_el - 0.50)
    dv_e = vv - v_el
    d_eye_uv = np.sqrt((du_e / (ew_uv * 1.35)) ** 2 + (dv_e / (eh_uv * 1.65)) ** 2)
    socket_ao = (1.0 - _smoothstep(0.65, 1.85, d_eye_uv)) * front_face
    ao *= 1.0 - 0.24 * socket_ao
    # Subtle warm eyelid crease shading
    lid_shadow = (1.0 - _smoothstep(0.45, 1.25, d_eye_uv)) * front_face
    lid_tint = skin * np.array([0.74, 0.60, 0.58])
    alb = alb * (1.0 - 0.28 * lid_shadow[:, :, None]) + lid_tint[None, None, :] * (
        0.28 * lid_shadow[:, :, None]
    )

    # Nostril underside AO darkening locked below nose_tip_v UV
    u_nos = 0.50 + 0.018
    v_nos = float(nt_uv[1]) - 0.016
    d_nostril_uv = np.sqrt(
        ((np.abs(uu - 0.50) - abs(u_nos - 0.50)) * 0.82) ** 2 + ((vv - v_nos) * 1.40) ** 2
    )
    nos_ao = (1.0 - _smoothstep(0.005, 0.016, d_nostril_uv)) * front_face
    ao *= 1.0 - 0.48 * nos_ao
    alb *= 1.0 - 0.30 * nos_ao[:, :, None]

    # --- 3. Smooth Anti-Aliased Sculpted Lips Locked to Exact Quad-Base Lip Loops! ---
    if surface is not None:
        up_pts_uv, lo_pts_uv, mid_pts_uv = surface.lip_uv_contours()
    else:

        def _sorted_canon(tag_name: str, upper: bool | None) -> np.ndarray:
            idx = tags[tag_name]
            pts = qb.verts[idx]
            uv_a = uv_base[idx]
            if upper is not None:
                mz = float(np.mean(qb.verts[tags["mouth_lip_seam"], 2]))
                keep = pts[:, 2] >= mz - 0.001 if upper else pts[:, 2] <= mz + 0.001
                if np.sum(keep) >= 3:
                    pts = pts[keep]
                    uv_a = uv_a[keep]
            return uv_a[np.argsort(pts[:, 0])]

        up_pts_uv = _sorted_canon("mouth_lip_crest", upper=True)
        lo_pts_uv = _sorted_canon("mouth_lip_crest", upper=False)
        mid_pts_uv = _sorted_canon("mouth_lip_seam", upper=None)

    # Compute exact UV band half-thickness from canonical vermilion vs seam loop
    verm_uv = uv_base[tags["mouth_vermilion"]]
    seam_v_mid = float(np.mean(mid_pts_uv[:, 1]))
    up_band = max(0.006, float(np.max(verm_uv[:, 1]) - seam_v_mid) * 0.62)
    lo_band = max(0.007, float(seam_v_mid - np.min(verm_uv[:, 1])) * 0.66)
    r_up_uv = up_band * (1.0 + 0.14 * float(f.upper_lip_fullness))
    r_lo_uv = lo_band * (1.0 + 0.14 * float(f.lower_lip_fullness))

    corner_span = max(0.018, abs(float(mid_pts_uv[-1, 0] - mid_pts_uv[0, 0])) * 0.5)
    t_corner = np.clip(np.abs(uu - 0.50) / corner_span, 0.0, 1.0)
    corner_taper = 1.0 - 0.45 * (t_corner**2)

    d_up = _dist_to_polyline_uv(uu, vv, up_pts_uv) / np.maximum(0.002, r_up_uv * corner_taper)
    d_lo = _dist_to_polyline_uv(uu, vv, lo_pts_uv) / np.maximum(0.002, r_lo_uv * corner_taper)
    d_mid = _dist_to_polyline_uv(uu, vv, mid_pts_uv)

    up_mask = (1.0 - _smoothstep(0.62, 1.25, d_up)) * front_face
    lo_mask = (1.0 - _smoothstep(0.62, 1.25, d_lo)) * front_face
    lip_mask = np.maximum(up_mask, lo_mask)

    # Natural lip shading: subtle on males (~0.46), richer rose tone on females (~0.74)
    lip_strength = _lerp(0.46, 0.74, max(0.0, -g))
    upper_col = lip_col * 0.88
    lower_col = np.clip(lip_col * np.array([1.06, 1.02, 1.02]), 0.0, 1.0)
    lip_rgb = np.where(
        (up_mask >= lo_mask)[:, :, None],
        upper_col[None, None, :],
        lower_col[None, None, :],
    )
    eff_lip = lip_mask * lip_strength
    alb = alb * (1.0 - eff_lip[:, :, None]) + lip_rgb * eff_lip[:, :, None]

    # Soft satin sheen on lips (~0.40) + crisp mouth fissure AO line
    rgh = rgh * (1.0 - 0.34 * lip_mask) + 0.40 * (0.34 * lip_mask)
    fissure_m = (1.0 - _smoothstep(0.0014, 0.0046, d_mid)) * front_face
    ao *= 1.0 - 0.52 * fissure_m
    alb *= 1.0 - 0.28 * fissure_m[:, :, None]

    # Realistic age wrinkles on forehead & crow's feet
    if age_w > 0.05:
        forehead_mask = (zn > 0.28) * (zn < 0.68) * (np.abs(phi) < 0.26 * np.pi) * front_face
        wrinkles = (np.sin(zn * 48.0 + melanin_var * 2.5) * 0.5 + 0.5) * (
            1.0 - _smoothstep(0.14 * np.pi, 0.26 * np.pi, np.abs(phi))
        )
        height -= age_w * 0.30 * wrinkles * forehead_mask
        alb -= age_w * 0.018 * wrinkles[:, :, None] * forehead_mask[:, :, None]

    # Convert height field -> subtle tangent-space Normal Map
    gy, gx = np.gradient(height)
    bump_scale = 1.65 * detail_w + 0.15
    nx_m = -gx * bump_scale
    ny_m = -gy * bump_scale
    nz_m = np.ones_like(nx_m)
    n_len = np.sqrt(nx_m * nx_m + ny_m * ny_m + nz_m * nz_m)
    n_rgb = np.stack([nx_m / n_len, ny_m / n_len, nz_m / n_len], axis=-1) * 0.5 + 0.5

    alb_u8 = np.clip(alb * 255.0, 0, 255).astype(np.uint8)
    nrm_u8 = np.clip(n_rgb * 255.0, 0, 255).astype(np.uint8)
    rgh_u8 = np.clip(np.repeat(rgh[:, :, None], 3, axis=2) * 255.0, 0, 255).astype(np.uint8)
    ao_u8 = np.clip(np.repeat(ao[:, :, None], 3, axis=2) * 255.0, 0, 255).astype(np.uint8)

    Image.fromarray(alb_u8, mode="RGB").save(albedo_path)
    Image.fromarray(nrm_u8, mode="RGB").save(normal_path)
    Image.fromarray(rgh_u8, mode="RGB").save(rough_path)
    Image.fromarray(ao_u8, mode="RGB").save(ao_path)
    clear_texture_cache()
    return albedo_path, normal_path, rough_path, ao_path


def _generate_eye_map(spec: CharacterSpec, out_dir: Path, sig: str) -> Path:
    """Synthesize a crisp 512x512 realistic Eyeball Albedo map (Sclera, Limbal, Iris, Pupil)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    eye_path = out_dir / f"eye_albedo_{sig}.png"
    if eye_path.is_file():
        return eye_path

    res = 256 if spec.quality == "preview" else 512
    eye_col = np.asarray(spec.materials.eye_color, dtype=np.float64)

    u = np.linspace(0.0, 1.0, res, dtype=np.float64)[None, :]
    v = np.linspace(1.0, 0.0, res, dtype=np.float64)[:, None]
    du = u - 0.5
    dv = v - 0.5
    r = np.sqrt(du * du + dv * dv)
    ang = np.arctan2(dv, du)

    # Realistic human iris spans ~0.50*eye_r -> theta=0.524 rad -> r_uv=0.082
    iris_r = 0.083
    pupil_r = iris_r * 0.41

    # Warm natural sclera with subtle rosy vascular warmth toward outer corners
    sclera = np.empty((res, res, 3), dtype=np.float64)
    sclera[:] = np.array([0.95, 0.935, 0.915])[None, None, :]
    corner_warm = _smoothstep(iris_r * 1.45, iris_r * 3.0, r)
    sclera = sclera * (1.0 - 0.12 * corner_warm[:, :, None]) + np.array([0.90, 0.74, 0.72])[
        None, None, :
    ] * (0.12 * corner_warm[:, :, None])

    # Detailed radial iris stroma fibers + warmer inner collarette ring + dark limbal ring
    r_norm = np.clip(r / max(1e-6, iris_r), 0.0, 2.0)
    fiber_fine = np.sin(ang * 38.0 + 1.8 * np.sin(ang * 7.0)) * 0.5 + 0.5
    fiber_med = np.cos(ang * 19.0 - r_norm * 10.0) * 0.5 + 0.5
    fibers = 0.76 + 0.24 * (0.55 * fiber_fine + 0.45 * fiber_med)

    # Subtle golden-amber collarette ring around pupil
    collarette = np.exp(-(((r_norm - 0.52) / 0.16) ** 2))
    iris_base = eye_col[None, None, :] * fibers[:, :, None]
    warm_ring = np.clip(eye_col * np.array([1.35, 1.18, 0.78]), 0.0, 1.0)
    iris_rgb = iris_base * (1.0 - 0.28 * collarette[:, :, None]) + warm_ring[None, None, :] * (
        0.28 * collarette[:, :, None]
    )

    # Crisp dark limbal ring at outer edge of iris
    limbal = _smoothstep(0.76, 0.96, r_norm)
    iris_rgb *= 1.0 - 0.62 * limbal[:, :, None]

    # Subtle upper-eyelid ambient shadow cast onto top of eyeball (dv > 0 is +Z top!)
    lid_cast = _smoothstep(0.015, 0.085, dv) * 0.22
    sclera *= 1.0 - lid_cast[:, :, None]
    iris_rgb *= 1.0 - lid_cast[:, :, None]

    # Anti-aliased compositing: sclera -> iris -> jet-black pupil
    in_iris = 1.0 - _smoothstep(iris_r * 0.95, iris_r * 1.04, r)
    in_pupil = 1.0 - _smoothstep(pupil_r * 0.88, pupil_r * 1.06, r)

    out = sclera * (1.0 - in_iris[:, :, None]) + iris_rgb * in_iris[:, :, None]
    pupil_col = np.array([0.02, 0.02, 0.028])[None, None, :]
    out = out * (1.0 - in_pupil[:, :, None]) + pupil_col * in_pupil[:, :, None]

    Image.fromarray(np.clip(out * 255.0, 0, 255).astype(np.uint8), mode="RGB").save(eye_path)
    clear_texture_cache()
    return eye_path


def _generate_hair_maps(
    spec: CharacterSpec,
    out_dir: Path,
    sig: str,
) -> tuple[Path, Path, Path]:
    """Synthesize silky longitudinal hair-strand Albedo, Normal, and Roughness (zero moire)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    alb_p = out_dir / f"hair_albedo_{sig}.png"
    nrm_p = out_dir / f"hair_normal_{sig}.png"
    rgh_p = out_dir / f"hair_rough_{sig}.png"
    if alb_p.is_file() and nrm_p.is_file() and rgh_p.is_file():
        return alb_p, nrm_p, rgh_p

    res = 256 if spec.quality == "preview" else 512
    hair_col = np.asarray(spec.materials.hair_color, dtype=np.float64)

    # Row 0 = V=1 (tip), Row res-1 = V=0 (root)
    u = np.linspace(0.0, 1.0, res, dtype=np.float64)[None, :]
    v = np.linspace(1.0, 0.0, res, dtype=np.float64)[:, None]

    # Smooth anti-aliased longitudinal hair strands (avoids high-frequency moire striping!)
    rng = np.random.default_rng(771)
    n_strands = 44 if res >= 512 else 28
    strand_1d = rng.uniform(-1.0, 1.0, size=(n_strands + 1,))
    tu = u[0] * n_strands
    i0 = np.floor(tu).astype(np.int32) % n_strands
    f_u = tu - np.floor(tu)
    w_u = f_u * f_u * (3.0 - 2.0 * f_u)
    fine_u = strand_1d[i0] * (1.0 - w_u) + strand_1d[i0 + 1] * w_u

    strands_hi = np.sin(u * (44.0 if res >= 512 else 30.0) * np.pi + 0.8 * np.sin(v * 6.0))
    strands_lo = np.broadcast_to(fine_u[None, :], (res, res))
    strand_h = 0.62 * strands_lo + 0.38 * strands_hi

    # Root-to-tip melanin gradient: darker shadowed roots (v=0), richer mid, slightly warmer tips
    root_dark = 0.64 + 0.36 * _smoothstep(0.0, 0.32, v)
    tip_lift = 1.0 + 0.14 * _smoothstep(0.65, 1.0, v)
    strand_tone = 1.0 + 0.14 * strand_h

    alb = np.clip(
        hair_col[None, None, :] * (root_dark * tip_lift * strand_tone)[:, :, None],
        0.015,
        1.0,
    )

    # Subtle silky longitudinal normal map
    gy, gx = np.gradient(strand_h)
    bump = 0.65
    nx = -gx * bump
    ny = -gy * bump * 0.20
    nz = np.ones_like(nx)
    nlen = np.sqrt(nx * nx + ny * ny + nz * nz)
    n_rgb = np.stack([nx / nlen, ny / nlen, nz / nlen], axis=-1) * 0.5 + 0.5

    # Silky satin roughness modulated per strand (~0.44 - 0.54, never plastic!)
    rgh = np.clip(0.49 - 0.05 * strand_h, 0.36, 0.72)

    Image.fromarray(np.clip(alb * 255.0, 0, 255).astype(np.uint8), mode="RGB").save(alb_p)
    Image.fromarray(np.clip(n_rgb * 255.0, 0, 255).astype(np.uint8), mode="RGB").save(nrm_p)
    Image.fromarray(
        np.clip(np.repeat(rgh[:, :, None], 3, axis=2) * 255.0, 0, 255).astype(np.uint8),
        mode="RGB",
    ).save(rgh_p)
    clear_texture_cache()
    return alb_p, nrm_p, rgh_p


def _generate_fabric_maps(
    base_rgb: tuple[float, float, float],
    tag: str,
    res: int,
    out_dir: Path,
    sig: str,
    *,
    is_athletic: bool = False,
) -> tuple[Path, Path]:
    """Synthesize subtle woven cotton/twill fabric Albedo & Normal maps."""
    out_dir.mkdir(parents=True, exist_ok=True)
    alb_p = out_dir / f"cloth_{tag}_albedo_{sig}.png"
    nrm_p = out_dir / f"cloth_{tag}_normal_{sig}.png"
    if alb_p.is_file() and nrm_p.is_file():
        return alb_p, nrm_p

    u = np.linspace(0.0, 1.0, res, dtype=np.float64)[None, :]
    v = np.linspace(1.0, 0.0, res, dtype=np.float64)[:, None]
    freq = 96.0 if res >= 256 else 48.0
    warp = np.sin(u * freq * np.pi)
    weft = np.sin(v * freq * np.pi)
    weave = 0.5 * (warp * weft) + 0.25 * np.sin((u + v) * freq * np.pi)
    mottle = _smooth_noise_2d(res, 18, 515 if tag == "top" else 616)

    col = np.asarray(base_rgb, dtype=np.float64)
    var = (0.025 if is_athletic else 0.045) * weave + 0.025 * mottle
    alb = np.clip(col[None, None, :] * (1.0 + var[:, :, None]), 0.01, 1.0)

    gy, gx = np.gradient(weave)
    bump = 0.35 if is_athletic else 0.60
    nx = -gx * bump
    ny = -gy * bump
    nz = np.ones_like(nx)
    nlen = np.sqrt(nx * nx + ny * ny + nz * nz)
    n_rgb = np.stack([nx / nlen, ny / nlen, nz / nlen], axis=-1) * 0.5 + 0.5

    Image.fromarray(np.clip(alb * 255.0, 0, 255).astype(np.uint8), mode="RGB").save(alb_p)
    Image.fromarray(np.clip(n_rgb * 255.0, 0, 255).astype(np.uint8), mode="RGB").save(nrm_p)
    clear_texture_cache()
    return alb_p, nrm_p


def build_character_material_bundle(
    scene: SceneDocument,
    obj_id: str,
    spec: CharacterSpec,
    lm: AnatomicalLandmarks,
    cache_root: Path,
    surface: CharacterSurface | None = None,
) -> dict[str, str]:
    """Create/update all realistic character PBR materials and return role -> material_id."""
    sig = _compute_material_signature(spec, lm)
    tex_dir = cache_root / ".frfruit_cache" / "characters" / obj_id

    skin_alb, skin_nrm, skin_rgh, skin_ao = _generate_skin_maps(
        spec, lm, tex_dir, sig, surface=surface
    )
    eye_alb = _generate_eye_map(spec, tex_dir, sig)
    hair_alb, hair_nrm, hair_rgh = _generate_hair_maps(spec, tex_dir, sig)

    sss = float(spec.materials.subsurface_strength)
    skin_rgb = spec.materials.skin_tone
    sss_col = (
        min(1.0, skin_rgb[0] * 1.12),
        max(0.0, skin_rgb[1] * 0.58),
        max(0.0, skin_rgb[2] * 0.48),
    )

    m_skin_id = f"mat_{obj_id}_skin"
    m_eyes_id = f"mat_{obj_id}_eyes"
    m_brows_id = f"mat_{obj_id}_brows"
    m_hair_id = f"mat_{obj_id}_hair"
    m_top_id = f"mat_{obj_id}_top"
    m_bot_id = f"mat_{obj_id}_bottom"
    m_shoe_id = f"mat_{obj_id}_shoes"

    # 1. Realistic Matte Skin (includes loop-locked lips & physiological AO/subsurface zones)
    scene.materials[m_skin_id] = MaterialSpec(
        id=m_skin_id,
        name=f"{obj_id}_Skin",
        base_color=(1.0, 1.0, 1.0, 1.0),
        metallic=0.0,
        roughness=max(0.52, min(0.88, float(spec.materials.roughness_base))),
        subsurface=sss,
        subsurface_color=sss_col,
        albedo_map=str(skin_alb),
        normal_map=str(skin_nrm),
        roughness_map=str(skin_rgh),
        ao_map=str(skin_ao),
        normal_strength=0.34,
        uv_scale=(1.0, 1.0),
        texture_source="procedural_character",
    )

    # 2. Glossy Wet Cornea Eyeballs (procedural 512x512 Iris/Pupil/Limbal/Sclera map)
    scene.materials[m_eyes_id] = MaterialSpec(
        id=m_eyes_id,
        name=f"{obj_id}_Eyes",
        base_color=(1.0, 1.0, 1.0, 1.0),
        metallic=0.0,
        roughness=0.05,
        clearcoat=1.0,
        clearcoat_roughness=0.03,
        subsurface=0.10,
        albedo_map=str(eye_alb),
        uv_scale=(1.0, 1.0),
        texture_source="procedural_character_eye",
    )

    # 3. Eyebrows & Eyelashes (grounded hair tone matching character palette)
    hc = spec.materials.hair_color
    brow_c = (max(0.025, hc[0] * 0.80), max(0.025, hc[1] * 0.80), max(0.025, hc[2] * 0.80))
    scene.materials[m_brows_id] = MaterialSpec(
        id=m_brows_id,
        name=f"{obj_id}_BrowsLashes",
        base_color=(brow_c[0], brow_c[1], brow_c[2], 1.0),
        metallic=0.0,
        roughness=0.62,
        sheen=0.18,
    )

    # 4. Multi-Strand Layered Hair Groom
    scene.materials[m_hair_id] = MaterialSpec(
        id=m_hair_id,
        name=f"{obj_id}_Hair",
        base_color=(1.0, 1.0, 1.0, 1.0),
        metallic=0.0,
        roughness=0.50,
        sheen=0.38,
        albedo_map=str(hair_alb),
        normal_map=str(hair_nrm),
        roughness_map=str(hair_rgh),
        normal_strength=0.26,
        uv_scale=(1.0, 1.0),
        texture_source="procedural_character_hair",
    )

    # 5. Matte Woven Clothing (Top, Bottom, Shoes)
    is_ath = spec.materials.clothing_style == "athletic"
    cloth_res = 256 if spec.quality == "preview" else 512
    top_alb, top_nrm = _generate_fabric_maps(
        spec.materials.top_color, "top", cloth_res, tex_dir, sig, is_athletic=is_ath
    )
    bot_alb, bot_nrm = _generate_fabric_maps(
        spec.materials.bottom_color, "bottom", cloth_res, tex_dir, sig, is_athletic=is_ath
    )

    cloth_rough = 0.72 if is_ath else 0.86
    scene.materials[m_top_id] = MaterialSpec(
        id=m_top_id,
        name=f"{obj_id}_Top",
        base_color=(1.0, 1.0, 1.0, 1.0),
        metallic=0.0,
        roughness=cloth_rough,
        sheen=0.38,
        albedo_map=str(top_alb),
        normal_map=str(top_nrm),
        normal_strength=0.35,
        uv_scale=(3.0, 3.0),
        texture_source="procedural_character_cloth",
    )

    scene.materials[m_bot_id] = MaterialSpec(
        id=m_bot_id,
        name=f"{obj_id}_Bottom",
        base_color=(1.0, 1.0, 1.0, 1.0),
        metallic=0.0,
        roughness=min(0.92, cloth_rough + 0.04),
        sheen=0.25,
        albedo_map=str(bot_alb),
        normal_map=str(bot_nrm),
        normal_strength=0.35,
        uv_scale=(3.0, 3.0),
        texture_source="procedural_character_cloth",
    )

    scene.materials[m_shoe_id] = MaterialSpec(
        id=m_shoe_id,
        name=f"{obj_id}_Shoes",
        base_color=(0.16, 0.16, 0.18, 1.0),
        metallic=0.0,
        roughness=0.58,
    )

    return {
        "skin": m_skin_id,
        "eyes": m_eyes_id,
        "brows": m_brows_id,
        "hair": m_hair_id,
        "top": m_top_id,
        "bottom": m_bot_id,
        "shoes": m_shoe_id,
    }
