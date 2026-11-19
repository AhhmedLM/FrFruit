"""Hierarchical Morph Target (Shape Key) engine for FrFruit's procedural Quad Base Mesh."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import numpy as np

from frfruit.character.quad_base import (
    EYE_CENTER_L,
    EYE_RADIUS,
    HEAD_CENTER,
    HEAD_HALF_H,
    MOUTH_W,
    MOUTH_Z,
    CanonicalQuadBase,
    build_canonical_quad_base,
)
from frfruit.character.schema import CharacterSpec, RBFDeltaNode
from frfruit.character.sdf_ops import wendland_c2
from frfruit.geometry.mesh import MeshData


def _smoothstep(edge0: float, edge1: float, x: np.ndarray) -> np.ndarray:
    denom = edge1 - edge0
    if abs(denom) < 1e-9:
        denom = 1e-9
    t = np.clip((x - edge0) / denom, 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def _gauss3(
    pts: np.ndarray,
    center: tuple[float, float, float] | np.ndarray,
    radii: tuple[float, float, float] | np.ndarray,
) -> np.ndarray:
    c = np.asarray(center, dtype=np.float64)
    r = np.maximum(np.asarray(radii, dtype=np.float64), 1e-6)
    d2 = np.sum(((pts - c[None, :]) / r[None, :]) ** 2, axis=1)
    return np.exp(-d2)


def _compute_quad_vertex_normals(verts: np.ndarray, quads: np.ndarray) -> np.ndarray:
    """Compute smooth unit outward vertex normals directly on the base quad cage."""
    v0 = verts[quads[:, 0]]
    v1 = verts[quads[:, 1]]
    v2 = verts[quads[:, 2]]
    v3 = verts[quads[:, 3]]
    fn = np.cross(v2 - v0, v3 - v1)
    nrms = np.zeros_like(verts)
    for k in range(4):
        np.add.at(nrms, quads[:, k], fn)
    ln = np.linalg.norm(nrms, axis=1, keepdims=True)
    return nrms / np.where(ln < 1e-12, 1.0, ln)


def _tag_mask(n: int, tags: dict[str, np.ndarray], names: tuple[str, ...]) -> np.ndarray:
    m = np.zeros(n, dtype=np.float64)
    for name in names:
        idx = tags.get(name)
        if idx is not None and len(idx) > 0:
            m[idx] = 1.0
    return m


def _rotate_around_axis_x(pts: np.ndarray, pivot: np.ndarray, angle_rad: float) -> np.ndarray:
    rel = pts - pivot[None, :]
    ca = float(np.cos(angle_rad))
    sa = float(np.sin(angle_rad))
    y_new = rel[:, 1] * ca - rel[:, 2] * sa
    z_new = rel[:, 1] * sa + rel[:, 2] * ca
    out = rel.copy()
    out[:, 1] = y_new
    out[:, 2] = z_new
    return out + pivot[None, :]


@dataclass(frozen=True)
class MorphTargetBank:
    """Precomputed library of region-masked `(N_base, 3)` Shape Keys on `CanonicalQuadBase`."""

    base_normals: np.ndarray  # (N_base, 3)
    morphs: dict[str, np.ndarray]  # name -> (N_base, 3) displacement vectors in meters


@lru_cache(maxsize=1)
def get_morph_target_bank() -> MorphTargetBank:
    """Build and cache all anatomical Body, Face, and Expression Morph Targets."""
    base = build_canonical_quad_base()
    v = base.verts
    n = len(v)
    nrms = _compute_quad_vertex_normals(v, base.quads)
    tags = base.tags

    x = v[:, 0]
    y = v[:, 1]
    z = v[:, 2]
    ax = np.abs(x)
    sx = np.sign(x)
    # Normalized head coordinates for clean facial sculpting masks
    q = (v - HEAD_CENTER[None, :]) / HEAD_HALF_H
    qx = np.abs(q[:, 0])
    qy = q[:, 1]
    qz = q[:, 2]

    head_m = _smoothstep(1.475, 1.525, z)
    face_front = head_m * _smoothstep(0.15, -0.35, qy)
    torso_m = _smoothstep(0.84, 0.90, z) * (1.0 - _smoothstep(1.44, 1.49, z))
    arm_m = _tag_mask(n, tags, ("arms",))
    leg_m = _tag_mask(n, tags, ("legs",))
    hand_m = _tag_mask(n, tags, ("hand",))
    ear_m = _tag_mask(n, tags, ("ears",))

    morphs: dict[str, np.ndarray] = {}

    # ------------------------------------------------------------------
    # 1. BODY MORPH TARGETS
    # ------------------------------------------------------------------

    # Masculine (+gender)
    dm = np.zeros((n, 3), dtype=np.float64)
    sh_band = _smoothstep(1.22, 1.38, z) * (1.0 - _smoothstep(1.43, 1.48, z))
    dm[:, 0] += sx * (0.020 * sh_band * (1.0 - hand_m) + 0.014 * arm_m)
    dm[:, 0] -= sx * 0.012 * np.exp(-(((z - 0.91) / 0.085) ** 2)) * torso_m
    dm[:, 0] += sx * 0.008 * np.exp(-(((z - 1.05) / 0.08) ** 2)) * torso_m
    # Stronger masculine chest/ribcage, neck, brow & square jaw
    pec_m = (
        np.exp(-(((z - 1.29) / 0.065) ** 2))
        * np.exp(-(((ax - 0.075) / 0.065) ** 2))
        * _smoothstep(0.02, -0.06, y)
    )
    dm[:, 1] -= 0.009 * pec_m
    neck_m = np.exp(-(((z - 1.48) / 0.045) ** 2)) * _smoothstep(0.10, 0.04, ax)
    dm[:, 0] += sx * 0.006 * neck_m
    dm[:, 1] += np.sign(y) * 0.005 * neck_m
    # Masculine brow ridge & mandibular angle
    brow_m = _gauss3(q, (0.0, -0.82, 0.22), (0.62, 0.30, 0.18)) * head_m
    jaw_m = (
        _gauss3(np.column_stack([qx, qy, qz]), (0.45, -0.25, -0.68), (0.34, 0.45, 0.28)) * head_m
    )
    dm[:, 1] -= 0.0045 * brow_m
    dm[:, 0] += sx * 0.0055 * jaw_m
    dm[:, 2] -= 0.0025 * jaw_m
    morphs["gender_masc"] = dm

    # Feminine (-gender)
    df = np.zeros((n, 3), dtype=np.float64)
    df[:, 0] -= sx * (0.020 * sh_band * (1.0 - hand_m) + 0.015 * arm_m)
    # Slender hourglass waist & fuller hips/glutes
    waist_f = np.exp(-(((z - 1.045) / 0.075) ** 2)) * torso_m
    hip_f = np.exp(-(((z - 0.905) / 0.085) ** 2)) * (torso_m + 0.65 * leg_m)
    df[:, 0] -= sx * 0.019 * waist_f
    df[:, 1] -= np.sign(y) * 0.008 * waist_f
    df[:, 0] += sx * 0.022 * hip_f
    df[:, 1] += 0.014 * hip_f * _smoothstep(0.0, 0.08, y)
    # Soft natural feminine teardrop bust
    bust_f = (
        np.exp(-(((z - 1.268) / 0.060) ** 2))
        * np.exp(-(((ax - 0.068) / 0.056) ** 2))
        * _smoothstep(0.0, -0.08, y)
    )
    df[:, 1] -= 0.026 * bust_f
    df[:, 2] -= 0.007 * bust_f
    # Slender neck, arms, softer jaw & lifted cheekbones
    df[:, 0] -= sx * 0.006 * neck_m
    df[:, 1] -= np.sign(y) * 0.004 * neck_m
    df -= nrms * (0.0045 * arm_m[:, None] * (1.0 - hand_m[:, None]))
    df[:, 0] -= sx * 0.0055 * jaw_m
    df[:, 1] += 0.0032 * brow_m
    zyg_f = (
        _gauss3(np.column_stack([qx, qy, qz]), (0.46, -0.58, -0.18), (0.28, 0.30, 0.22)) * head_m
    )
    df[:, 1] -= 0.0030 * zyg_f
    df[:, 0] += sx * 0.0022 * zyg_f
    morphs["gender_fem"] = df

    # Muscularity (0..1, delta from neutral 0.38)
    dmus = np.zeros((n, 3), dtype=np.float64)
    delt_m = np.exp(-(((z - 1.375) / 0.065) ** 2)) * _smoothstep(0.13, 0.20, ax)
    bicep_m = _tag_mask(n, tags, ("upper_arm", "forearm"))
    calf_quad = _tag_mask(n, tags, ("thigh", "calf"))
    lat_m = np.exp(-(((z - 1.24) / 0.095) ** 2)) * _smoothstep(0.11, 0.16, ax) * torso_m
    dmus += (
        nrms
        * (0.015 * delt_m + 0.014 * pec_m + 0.011 * bicep_m + 0.012 * calf_quad + 0.012 * lat_m)[
            :, None
        ]
    )
    dmus[:, 0] -= sx * 0.006 * waist_f  # athletic V-taper
    morphs["muscularity"] = dmus

    # Body Fat (0..1, delta from neutral 0.22)
    dfat = np.zeros((n, 3), dtype=np.float64)
    belly_m = np.exp(-(((z - 1.02) / 0.12) ** 2)) * torso_m
    cheek_fat = (
        _gauss3(np.column_stack([qx, qy, qz]), (0.42, -0.55, -0.45), (0.35, 0.38, 0.30)) * head_m
    )
    subm_fat = _gauss3(q, (0.0, -0.32, -1.04), (0.45, 0.42, 0.22)) * head_m
    dfat += (
        nrms
        * (
            0.036 * belly_m
            + 0.024 * torso_m
            + 0.018 * (leg_m * _smoothstep(0.15, 0.65, z))
            + 0.013 * (arm_m * (1.0 - hand_m))
            + 0.007 * cheek_fat
            + 0.008 * subm_fat
        )[:, None]
    )
    dfat[:, 1] -= 0.022 * belly_m * _smoothstep(0.0, -0.08, y)
    morphs["body_fat"] = dfat

    # Age (0..1, delta from neutral 0.28)
    dage = np.zeros((n, 3), dtype=np.float64)
    # Slight cheek hollowing, subtle jowl settle, nose tip & ear slight lengthening, thinner lips
    dage -= nrms * (0.0045 * cheek_fat)[:, None]
    jowl_m = (
        _gauss3(np.column_stack([qx, qy, qz]), (0.38, -0.50, -0.76), (0.28, 0.30, 0.20)) * head_m
    )
    dage[:, 2] -= 0.0040 * jowl_m
    dage[:, 0] += sx * 0.0025 * jowl_m
    nose_tip_m = _gauss3(q, (0.0, -0.95, -0.32), (0.22, 0.25, 0.18)) * face_front
    dage[:, 2] -= 0.0045 * nose_tip_m
    dage[:, 1] -= 0.0020 * nose_tip_m
    dage[:, 2] -= 0.0035 * ear_m
    lip_all_m = _tag_mask(n, tags, ("upper_lip", "lower_lip"))
    dage[:, 1] += 0.0018 * lip_all_m
    morphs["age"] = dage

    # Shoulder width (multiplier delta around 1.0)
    dsh = np.zeros((n, 3), dtype=np.float64)
    sh_ext = _smoothstep(1.16, 1.38, z) * (1.0 - _smoothstep(1.44, 1.49, z))
    dsh[:, 0] += sx * (0.14 * sh_ext * (1.0 - hand_m) * np.clip(ax / 0.18, 0.0, 1.0) + 0.14 * arm_m)
    morphs["shoulder_width"] = dsh

    # Hip width (multiplier delta around 1.0)
    dhip = np.zeros((n, 3), dtype=np.float64)
    hip_band = np.exp(-(((z - 0.91) / 0.14) ** 2)) * (torso_m + leg_m)
    dhip[:, 0] += sx * 0.12 * hip_band * np.clip(ax / 0.16, 0.0, 1.0)
    morphs["hip_width"] = dhip

    # Limb thickness (multiplier delta around 1.0)
    dlimb = np.zeros((n, 3), dtype=np.float64)
    dlimb += nrms * (0.022 * arm_m * (1.0 - 0.7 * hand_m) + 0.026 * leg_m)[:, None]
    morphs["limb_thickness"] = dlimb

    # Head scale (centered on neck top so head never detaches!)
    dhs = np.zeros((n, 3), dtype=np.float64)
    neck_piv = np.array([0.0, 0.012, 1.495], dtype=np.float64)
    dhs = (v - neck_piv[None, :]) * head_m[:, None]
    morphs["head_scale"] = dhs

    # ------------------------------------------------------------------
    # 2. FACIAL ANTHROPOMETRY MORPH TARGETS (All 28 Spatial Sliders)
    # ------------------------------------------------------------------

    # Cranium & Silhouette
    d = np.zeros((n, 3), dtype=np.float64)
    d[:, 0] = x * 0.11 * head_m
    morphs["face_width"] = d

    d = np.zeros((n, 3), dtype=np.float64)
    lower_face = head_m * _smoothstep(0.20, -0.35, qz)
    d[:, 2] = -0.014 * lower_face * np.clip(-qz, 0.0, 1.1)
    morphs["face_length"] = d

    d = np.zeros((n, 3), dtype=np.float64)
    fh_m = head_m * _smoothstep(0.15, 0.65, qz)
    d[:, 2] = 0.012 * fh_m
    morphs["forehead_height"] = d

    d = np.zeros((n, 3), dtype=np.float64)
    d[:, 1] = -0.008 * brow_m
    d[:, 2] = 0.0015 * brow_m
    morphs["brow_ridge"] = d

    d = np.zeros((n, 3), dtype=np.float64)
    d[:, 0] = sx * 0.0075 * zyg_f
    d[:, 1] = -0.0065 * zyg_f
    morphs["cheekbone_prominence"] = d

    d = np.zeros((n, 3), dtype=np.float64)
    d += nrms * (0.0085 * cheek_fat)[:, None]
    morphs["cheek_fullness"] = d

    d = np.zeros((n, 3), dtype=np.float64)
    jaw_all = head_m * _smoothstep(-0.20, -0.65, qz) * _smoothstep(0.12, 0.50, qx)
    d[:, 0] = sx * 0.011 * jaw_all
    morphs["jaw_width"] = d

    d = np.zeros((n, 3), dtype=np.float64)
    d[:, 0] = sx * 0.008 * jaw_m
    d[:, 2] = -0.007 * jaw_m
    morphs["jaw_angle"] = d

    d = np.zeros((n, 3), dtype=np.float64)
    chin_m = _gauss3(q, (0.0, -0.78, -0.92), (0.30, 0.34, 0.22)) * head_m
    d[:, 1] = -0.010 * chin_m
    d[:, 2] = -0.004 * chin_m
    morphs["chin_prominence"] = d

    d = np.zeros((n, 3), dtype=np.float64)
    d[:, 0] = sx * 0.008 * chin_m * np.clip(qx / 0.22, 0.0, 1.0)
    morphs["chin_width"] = d

    # Eyes & Brows
    eye_l_c = EYE_CENTER_L
    eye_r_c = np.array([-EYE_CENTER_L[0], EYE_CENTER_L[1], EYE_CENTER_L[2]])
    eye_l_m = np.maximum(
        _tag_mask(n, tags, ("eye_L",)),
        0.55 * _gauss3(v, eye_l_c, (0.028, 0.030, 0.022)) * face_front,
    )
    eye_r_m = np.maximum(
        _tag_mask(n, tags, ("eye_R",)),
        0.55 * _gauss3(v, eye_r_c, (0.028, 0.030, 0.022)) * face_front,
    )
    eye_both_m = np.maximum(eye_l_m, eye_r_m)
    eye_nearest_c = np.where(x[:, None] >= 0.0, eye_l_c[None, :], eye_r_c[None, :])
    rel_eye = v - eye_nearest_c

    d = np.zeros((n, 3), dtype=np.float64)
    d[:, 0] = rel_eye[:, 0] * 0.16 * eye_both_m
    d[:, 2] = rel_eye[:, 2] * 0.22 * eye_both_m
    morphs["eye_size"] = d

    d = np.zeros((n, 3), dtype=np.float64)
    d[:, 0] = sx * 0.0055 * eye_both_m
    morphs["eye_spacing"] = d

    d = np.zeros((n, 3), dtype=np.float64)
    d[:, 2] = 0.0065 * eye_both_m
    morphs["eye_height"] = d

    d = np.zeros((n, 3), dtype=np.float64)
    d[:, 1] = 0.0060 * eye_both_m
    morphs["eye_depth"] = d

    d = np.zeros((n, 3), dtype=np.float64)
    # Rotate eyelid corners gently around eye center (positive = upward outer canthal tilt)
    d[:, 2] = (sx * rel_eye[:, 0]) * 0.18 * eye_both_m
    morphs["eye_tilt"] = d

    d = np.zeros((n, 3), dtype=np.float64)
    brow_lat = (
        _gauss3(np.column_stack([qx, qy, qz]), (0.34, -0.78, 0.22), (0.24, 0.26, 0.14)) * face_front
    )
    brow_tag = _tag_mask(n, tags, ("brow_L", "brow_R"))
    d[:, 2] = 0.0065 * np.maximum(brow_lat, brow_tag * 0.85)
    morphs["brow_arch"] = d

    # Nose
    nose_full = _gauss3(q, (0.0, -0.88, -0.18), (0.24, 0.34, 0.28)) * face_front
    nose_bridge_m = _gauss3(q, (0.0, -0.84, -0.06), (0.16, 0.24, 0.18)) * face_front
    nose_ala_m = (
        _gauss3(np.column_stack([qx, qy, qz]), (0.16, -0.84, -0.32), (0.14, 0.22, 0.13))
        * face_front
    )

    d = np.zeros((n, 3), dtype=np.float64)
    d[:, 2] = -0.0085 * np.maximum(nose_tip_m, 0.65 * nose_ala_m)
    morphs["nose_length"] = d

    d = np.zeros((n, 3), dtype=np.float64)
    d[:, 0] = sx * 0.0055 * np.maximum(nose_ala_m, 0.45 * nose_full * np.clip(qx / 0.18, 0.0, 1.0))
    morphs["nose_width"] = d

    d = np.zeros((n, 3), dtype=np.float64)
    d[:, 1] = -0.0065 * nose_bridge_m
    morphs["nose_bridge"] = d

    d = np.zeros((n, 3), dtype=np.float64)
    d[:, 1] = -0.0085 * nose_tip_m
    morphs["nose_tip_projection"] = d

    d = np.zeros((n, 3), dtype=np.float64)
    d[:, 2] = 0.0065 * nose_tip_m
    d[:, 1] = -0.0018 * nose_tip_m
    morphs["nose_tip_tilt"] = d

    # Mouth & Lips
    mouth_zone = np.maximum(
        _tag_mask(n, tags, ("mouth",)),
        0.55 * _gauss3(v, (0.0, -0.095, MOUTH_Z), (0.042, 0.034, 0.022)) * face_front,
    )
    up_lip_m = _tag_mask(n, tags, ("upper_lip",))
    lo_lip_m = _tag_mask(n, tags, ("lower_lip",))
    corner_m = np.maximum(
        _tag_mask(n, tags, ("mouth_corners",)),
        0.60
        * _gauss3(np.column_stack([ax, y, z]), (MOUTH_W, -0.088, MOUTH_Z), (0.015, 0.020, 0.014))
        * face_front,
    )

    d = np.zeros((n, 3), dtype=np.float64)
    d[:, 0] = x * 0.22 * mouth_zone
    morphs["mouth_width"] = d

    d = np.zeros((n, 3), dtype=np.float64)
    d[:, 1] = -0.0042 * up_lip_m
    d[:, 2] = 0.0032 * up_lip_m
    morphs["upper_lip_fullness"] = d

    d = np.zeros((n, 3), dtype=np.float64)
    d[:, 1] = -0.0048 * lo_lip_m
    d[:, 2] = -0.0038 * lo_lip_m
    morphs["lower_lip_fullness"] = d

    d = np.zeros((n, 3), dtype=np.float64)
    d[:, 2] = 0.0065 * mouth_zone
    morphs["mouth_height"] = d

    # Ears
    ear_c_near = np.where(
        x[:, None] >= 0.0,
        np.array([[0.075, 0.008, 1.595]]),
        np.array([[-0.075, 0.008, 1.595]]),
    )
    d = np.zeros((n, 3), dtype=np.float64)
    d = (v - ear_c_near) * 0.25 * ear_m[:, None]
    morphs["ear_size"] = d

    d = np.zeros((n, 3), dtype=np.float64)
    d[:, 0] = sx * 0.010 * ear_m
    d[:, 1] = -0.003 * ear_m
    morphs["ear_protrusion"] = d

    # ------------------------------------------------------------------
    # 3. FACIAL EXPRESSION SHAPE KEYS
    # ------------------------------------------------------------------

    # Smile (+1 smile / -1 frown): lifts corners along dental arch + subtle cheek lift
    d_smile = np.zeros((n, 3), dtype=np.float64)
    cheek_smile = (
        _gauss3(np.column_stack([ax, y, z]), (0.042, -0.078, 1.572), (0.026, 0.028, 0.022))
        * face_front
    )
    d_smile[:, 0] = sx * (0.0045 * corner_m + 0.0020 * cheek_smile)
    d_smile[:, 1] = 0.0055 * corner_m - 0.0025 * cheek_smile
    d_smile[:, 2] = 0.0075 * corner_m + 0.0030 * cheek_smile
    morphs["smile"] = d_smile
    morphs["expr_smile"] = d_smile

    # Blink Left & Blink Right: lower upper eyelid smoothly over curved eyeball surface!
    for side_tag, side_s in (("L", 1.0), ("R", -1.0)):
        d_bl = np.zeros((n, 3), dtype=np.float64)
        up_m = _tag_mask(n, tags, (f"upper_lid_{side_tag}",))
        ec = np.array([side_s * EYE_CENTER_L[0], EYE_CENTER_L[1], EYE_CENTER_L[2]])
        # Amount upper lid drops down to meet lower lid (strongest at lid margin, gentler at crease)
        margin_m = _tag_mask(n, tags, (f"eye_margin_{side_tag}", f"eye_wetline_{side_tag}"))
        crease_m = _tag_mask(n, tags, (f"eye_crease_{side_tag}",))
        drop_w = up_m * (1.0 * margin_m + 0.55 * crease_m)
        # Drop down across palpebral fissure (~11mm) and arc slightly forward (-Y) over eyeball!
        d_bl[:, 2] = -0.0112 * drop_w
        # Extra -Y clearance when passing the eyeball equator so lid never intersects iris!
        horiz_c = np.clip(1.0 - ((x - ec[0]) / (EYE_RADIUS * 1.25)) ** 2, 0.0, 1.0)
        d_bl[:, 1] = -0.0032 * drop_w * horiz_c
        morphs[f"expr_blink_{'left' if side_tag == 'L' else 'right'}"] = d_bl

    # Brow Raise (+1 surprised / -1 furrowed)
    d_brow = np.zeros((n, 3), dtype=np.float64)
    brow_zone = (
        _gauss3(np.column_stack([qx, qy, qz]), (0.28, -0.80, 0.23), (0.34, 0.28, 0.16)) * face_front
    )
    d_brow[:, 2] = 0.0085 * np.maximum(brow_zone, brow_tag)
    d_brow[:, 0] = sx * 0.0020 * brow_zone
    morphs["expr_brow_raise"] = d_brow

    # Jaw Open (0..1): anatomical TMJ hinge rotation of mandible + lower lip + lower cheeks!
    tmj_pivot = np.array([0.0, 0.014, 1.592], dtype=np.float64)
    rotated_jaw = _rotate_around_axis_x(v, tmj_pivot, np.radians(15.5))
    # Smooth lower-jaw mask: 1.0 on chin/lower lip/lower mandible, 0.0 on upper lip & nose!
    jaw_open_m = head_m * (1.0 - _smoothstep(1.546, 1.564, z)) * _smoothstep(0.03, -0.04, y)
    # Exclude upper lip strictly, include lower lip fully
    jaw_open_m = np.where(up_lip_m > 0.5, 0.0, jaw_open_m)
    jaw_open_m = np.maximum(jaw_open_m, lo_lip_m)
    # Mouth corners move halfway so lips stretch naturally without tearing
    jaw_open_m = np.where(_tag_mask(n, tags, ("mouth_corners",)) > 0.5, 0.45, jaw_open_m)
    morphs["expr_jaw_open"] = (rotated_jaw - v) * jaw_open_m[:, None]

    return MorphTargetBank(base_normals=nrms, morphs=morphs)


def _apply_rbf_nodes_to_verts(
    verts: np.ndarray,
    normals: np.ndarray,
    nodes: list[RBFDeltaNode],
) -> np.ndarray:
    """Apply Wendland C2 RBF sculpting nodes directly to base-cage vertices."""
    if not nodes:
        return verts
    out = verts.copy()
    q_pts = (verts - HEAD_CENTER[None, :]) / HEAD_HALF_H

    for node in nodes:
        is_head = node.space == "head_norm"
        pts_sp = q_pts if is_head else verts
        scale_m = HEAD_HALF_H if is_head else 1.0
        centers = [np.asarray(node.center, dtype=np.float64)]
        vd_list = [np.asarray(node.vector_delta, dtype=np.float64)]
        if node.mirror_x and abs(centers[0][0]) > 1e-6:
            c_m = centers[0].copy()
            c_m[0] = -c_m[0]
            v_m = vd_list[0].copy()
            v_m[0] = -v_m[0]
            centers.append(c_m)
            vd_list.append(v_m)

        rad = max(1e-4, float(node.radius))
        for c_vec, vd in zip(centers, vd_list, strict=True):
            dist = np.linalg.norm(pts_sp - c_vec[None, :], axis=1)
            w = wendland_c2(dist / rad)
            if not np.any(w > 1e-6):
                continue
            # Convention: negative weight inflates outward along normal, positive carves inward
            if abs(node.weight) > 1e-7:
                out += normals * (-float(node.weight) * scale_m * w)[:, None]
            if float(np.linalg.norm(vd)) > 1e-7:
                out += (vd * scale_m)[None, :] * w[:, None]
    return out


def _apply_pose_to_base_verts(
    verts: np.ndarray,
    base: CanonicalQuadBase,
    pose: str,
) -> np.ndarray:
    """Smoothly rotate arms around shoulder joints for `t_pose` or `relaxed` poses."""
    if pose == "a_pose":
        return verts
    angle_deg = 34.0 if pose == "t_pose" else -16.0
    out = verts.copy()
    arm_l = base.tags.get("arm_L", np.empty(0, dtype=np.int64))
    arm_r = base.tags.get("arm_R", np.empty(0, dtype=np.int64))

    for idxs, side_s in ((arm_l, 1.0), (arm_r, -1.0)):
        if len(idxs) == 0:
            continue
        pts = out[idxs]
        sh_piv = np.array([side_s * 0.194, 0.012, 1.385], dtype=np.float64)
        # Smooth blend near deltoid root so shoulder never creases
        blend = _smoothstep(0.185, 0.255, np.abs(pts[:, 0]))
        ang = np.radians(angle_deg * side_s)
        ca = np.cos(ang * blend)
        sa = np.sin(ang * blend)
        rel = pts - sh_piv[None, :]
        # Rotate in XZ plane around Y axis
        rx = rel[:, 0] * ca - rel[:, 2] * sa
        rz = rel[:, 0] * sa + rel[:, 2] * ca
        out[idxs, 0] = sh_piv[0] + rx
        out[idxs, 2] = sh_piv[2] + rz
    return out


def evaluate_base_cage(spec: CharacterSpec) -> np.ndarray:
    """Evaluate all Body, Face, Expression, RBF, and Custom Morphs on the Quad Base Cage."""
    base = build_canonical_quad_base()
    bank = get_morph_target_bank()
    m = bank.morphs
    v = base.verts.copy()

    b = spec.body
    f = spec.face
    ex = spec.expressions

    # 1) Body Morphs
    g = float(b.gender)
    if g > 0.0:
        v += g * m["gender_masc"]
    elif g < 0.0:
        v += (-g) * m["gender_fem"]

    v += (float(b.muscularity) - 0.38) * m["muscularity"]
    v += (float(b.body_fat) - 0.22) * m["body_fat"]
    v += (float(b.age) - 0.28) * m["age"]
    v += (float(b.shoulder_width) - 1.0) * m["shoulder_width"]
    v += (float(b.hip_width) - 1.0) * m["hip_width"]
    v += (float(b.limb_thickness) - 1.0) * m["limb_thickness"]
    v += (float(b.head_scale) - 1.0) * m["head_scale"]

    # 2) Face Sliders
    for field_name in (
        "face_width",
        "face_length",
        "forehead_height",
        "brow_ridge",
        "cheekbone_prominence",
        "cheek_fullness",
        "jaw_width",
        "jaw_angle",
        "chin_prominence",
        "chin_width",
        "eye_size",
        "eye_spacing",
        "eye_height",
        "eye_depth",
        "eye_tilt",
        "brow_arch",
        "nose_length",
        "nose_width",
        "nose_bridge",
        "nose_tip_projection",
        "nose_tip_tilt",
        "mouth_width",
        "upper_lip_fullness",
        "lower_lip_fullness",
        "mouth_height",
        "ear_size",
        "ear_protrusion",
    ):
        val = float(getattr(f, field_name))
        if abs(val) > 1e-6:
            v += val * m[field_name]

    # 3) Facial Expressions (combine FaceSliders.smile + ExpressionSliders.smile cleanly)
    net_smile = float(np.clip(float(f.smile) + float(ex.smile), -1.0, 1.0))
    if abs(net_smile) > 1e-6:
        v += net_smile * m["expr_smile"]
    if float(ex.blink_left) > 1e-6:
        v += float(ex.blink_left) * m["expr_blink_left"]
    if float(ex.blink_right) > 1e-6:
        v += float(ex.blink_right) * m["expr_blink_right"]
    if abs(float(ex.brow_raise)) > 1e-6:
        v += float(ex.brow_raise) * m["expr_brow_raise"]
    if float(ex.jaw_open) > 1e-6:
        v += float(ex.jaw_open) * m["expr_jaw_open"]

    # 4) Subtle organic facial asymmetry (realistic micro-variation)
    asym = float(f.asymmetry)
    if asym > 1e-4:
        head_m = _smoothstep(1.48, 1.54, base.verts[:, 2])
        v[:, 0] += asym * 0.0014 * np.sin(base.verts[:, 2] * 22.0) * head_m
        v[:, 2] += asym * 0.0010 * np.sin(base.verts[:, 0] * 35.0 + 0.5) * head_m

    # 5) Wendland C2 RBF Sculpting Nodes
    if spec.rbf_nodes:
        cur_nrms = _compute_quad_vertex_normals(v, base.quads)
        v = _apply_rbf_nodes_to_verts(v, cur_nrms, spec.rbf_nodes)

    # 6) Named Custom Morph Targets
    if spec.custom_morphs and spec.stored_morphs:
        n_v = len(v)
        for mname, weight in spec.custom_morphs.items():
            w_val = float(weight)
            if abs(w_val) <= 1e-6:
                continue
            target = spec.stored_morphs.get(mname)
            if target is None or not target.deltas:
                continue
            for vi, dxyz in target.deltas:
                if 0 <= vi < n_v:
                    v[vi, 0] += w_val * float(dxyz[0])
                    v[vi, 1] += w_val * float(dxyz[1])
                    v[vi, 2] += w_val * float(dxyz[2])

    # 7) Pose adjustment
    v = _apply_pose_to_base_verts(v, base, spec.pose)

    # 8) Global stature scaling & exact floor grounding (feet strictly on Z = 0.0)
    z_min = float(np.min(v[:, 2]))
    v[:, 2] -= z_min
    cur_h = max(0.5, float(np.max(v[:, 2])))
    scale = float(b.height) / cur_h
    v *= scale
    return v


def build_character_skin_mesh(
    spec: CharacterSpec,
    quality: str | None = None,
) -> tuple[MeshData, np.ndarray]:
    """Evaluate Morph Targets + cached sparse Catmull-Clark, returning `(skin_mesh, base_verts)`."""
    base = build_canonical_quad_base()
    base_verts = evaluate_base_cage(spec)
    q = quality or spec.quality
    levels = 1 if q == "preview" else 2
    skin_mesh = base.subdivide(base_verts, levels=levels)
    return skin_mesh, base_verts
