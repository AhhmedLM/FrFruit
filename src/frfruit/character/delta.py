"""Character Delta ('الفارق'), regional blending, RBF sculpting, and Custom Morph extraction."""

from __future__ import annotations

import uuid

import numpy as np

from frfruit.character.quad_base import build_canonical_quad_base
from frfruit.character.schema import (
    FACE_REGION_FIELDS,
    SEMANTIC_RBF_ANCHORS,
    BodySliders,
    CharacterDelta,
    CharacterMaterialParams,
    CharacterSpec,
    CustomMorphTarget,
    DeltaRegion,
    ExpressionSliders,
    FaceSliders,
    RBFDeltaNode,
)
from frfruit.character.sdf_ops import wendland_c2
from frfruit.character.surface import AnatomicalLandmarks, CharacterSurface, compute_landmarks
from frfruit.core.types import Vec3
from frfruit.geometry.mesh import MeshData


def _clamp_field(
    model_cls: type[BodySliders] | type[FaceSliders] | type[ExpressionSliders],
    key: str,
    val: float,
) -> float:
    field_info = model_cls.model_fields[key]
    lo = -1e9
    hi = 1e9
    for meta in field_info.metadata:
        if hasattr(meta, "ge") and meta.ge is not None:
            lo = float(meta.ge)
        if hasattr(meta, "le") and meta.le is not None:
            hi = float(meta.le)
    return max(lo, min(hi, float(val)))


def add_rbf_sculpt_node(
    spec: CharacterSpec,
    *,
    region: str = "custom",
    center: Vec3 | None = None,
    radius: float | None = None,
    weight: float = 0.0,
    vector_delta: Vec3 = (0.0, 0.0, 0.0),
    mirror_x: bool | None = None,
    space: str = "head_norm",
    node_id: str | None = None,
) -> tuple[CharacterSpec, RBFDeltaNode]:
    """Return a copy of `spec` with a new Wendland C2 RBF sculpting node attached."""
    out = spec.model_copy(deep=True)
    reg_key = region.strip().lower()
    if center is None:
        if reg_key not in SEMANTIC_RBF_ANCHORS:
            raise ValueError(
                f"Unknown semantic sculpt region {region!r} and no explicit center provided. "
                f"Available regions: {sorted(SEMANTIC_RBF_ANCHORS.keys())}"
            )
        anc_center, anc_rad, anc_mirror = SEMANTIC_RBF_ANCHORS[reg_key]
        use_center = anc_center
        use_radius = radius if radius is not None else anc_rad
        use_mirror = mirror_x if mirror_x is not None else anc_mirror
        use_space = "head_norm"
    else:
        use_center = center
        use_radius = radius if radius is not None else 0.30
        use_mirror = bool(mirror_x) if mirror_x is not None else False
        use_space = "local" if space == "local" else "head_norm"

    nid = node_id or f"rbf_{reg_key}_{uuid.uuid4().hex[:6]}"
    node = RBFDeltaNode(
        id=nid,
        region=reg_key,
        space=use_space,  # type: ignore[arg-type]
        center=(float(use_center[0]), float(use_center[1]), float(use_center[2])),
        radius=float(max(0.02, min(2.0, use_radius))),
        weight=float(max(-0.5, min(0.5, weight))),
        vector_delta=(
            float(vector_delta[0]),
            float(vector_delta[1]),
            float(vector_delta[2]),
        ),
        mirror_x=use_mirror,
    )
    out.rbf_nodes.append(node)
    return out, node


def _node_in_region(node: RBFDeltaNode, region: DeltaRegion) -> bool:
    if region in ("all", "face"):
        return node.space == "head_norm" or region == "all"
    if region == "body":
        return node.space == "local"
    if region == "expressions":
        return False
    r = node.region.lower()
    if region == "nose":
        return "nose" in r
    if region == "mouth":
        return "lip" in r or "mouth" in r
    if region == "jaw":
        return "jaw" in r or "chin" in r or "cheek" in r
    if region == "eyes":
        return "eye" in r or "brow" in r
    if region == "upper_face":
        return any(k in r for k in ("eye", "brow", "forehead", "cheekbone"))
    if region == "ears":
        return "ear" in r
    return True


def compute_character_delta(
    char_a: CharacterSpec,
    char_b: CharacterSpec,
    region: DeltaRegion = "all",
    name: str = "delta",
    source_a_name: str = "A",
    source_b_name: str = "B",
) -> CharacterDelta:
    """Compute the mathematical difference `Delta = B - A` scoped to `region`."""
    allowed = FACE_REGION_FIELDS[region]
    body_d: dict[str, float] = {}
    for k in BodySliders.model_fields:
        if k in allowed:
            diff = float(getattr(char_b.body, k)) - float(getattr(char_a.body, k))
            if abs(diff) > 1e-6:
                body_d[k] = round(diff, 6)

    face_d: dict[str, float] = {}
    for k in FaceSliders.model_fields:
        if k in allowed:
            diff = float(getattr(char_b.face, k)) - float(getattr(char_a.face, k))
            if abs(diff) > 1e-6:
                face_d[k] = round(diff, 6)

    expr_d: dict[str, float] = {}
    for k in ExpressionSliders.model_fields:
        if k in allowed or region in ("all", "expressions"):
            diff = float(getattr(char_b.expressions, k)) - float(getattr(char_a.expressions, k))
            if abs(diff) > 1e-6:
                expr_d[k] = round(diff, 6)

    cm_d: dict[str, float] = {}
    if region == "all":
        all_cm_keys = set(char_a.custom_morphs.keys()) | set(char_b.custom_morphs.keys())
        for ck in sorted(all_cm_keys):
            diff = float(char_b.custom_morphs.get(ck, 0.0)) - float(
                char_a.custom_morphs.get(ck, 0.0)
            )
            if abs(diff) > 1e-6:
                cm_d[ck] = round(diff, 6)

    mat_d: dict[str, float | list[float]] = {}
    if region in ("all", "face"):
        for ck in ("skin_tone", "lip_tint", "eye_color", "hair_color"):
            ca = getattr(char_a.materials, ck)
            cb = getattr(char_b.materials, ck)
            cd = [round(float(cb[i]) - float(ca[i]), 5) for i in range(3)]
            if any(abs(v) > 1e-5 for v in cd):
                mat_d[ck] = cd
        for sk in ("subsurface_strength", "roughness_base"):
            sd = float(getattr(char_b.materials, sk)) - float(getattr(char_a.materials, sk))
            if abs(sd) > 1e-5:
                mat_d[sk] = round(sd, 5)

    # RBF node difference: nodes present in B (filtered by region) minus matching IDs in A
    a_by_id = {n.id: n for n in char_a.rbf_nodes}
    rbf_d: list[RBFDeltaNode] = []
    for nb in char_b.rbf_nodes:
        if not _node_in_region(nb, region):
            continue
        na = a_by_id.get(nb.id)
        if na is None:
            rbf_d.append(nb.model_copy(deep=True))
        else:
            dw = float(nb.weight) - float(na.weight)
            dv = tuple(float(nb.vector_delta[i]) - float(na.vector_delta[i]) for i in range(3))
            if abs(dw) > 1e-6 or any(abs(x) > 1e-6 for x in dv):
                rbf_d.append(
                    RBFDeltaNode(
                        id=nb.id,
                        region=nb.region,
                        space=nb.space,
                        center=nb.center,
                        radius=nb.radius,
                        weight=dw,
                        vector_delta=dv,  # type: ignore[arg-type]
                        mirror_x=nb.mirror_x,
                    )
                )

    return CharacterDelta(
        name=name,
        region=region,
        source_a=source_a_name,
        source_b=source_b_name,
        body_delta=body_d,
        face_delta=face_d,
        expression_delta=expr_d,
        custom_morph_delta=cm_d,
        material_delta=mat_d,
        rbf_nodes=rbf_d,
    )


def apply_character_delta(
    target: CharacterSpec,
    delta: CharacterDelta,
    weight: float = 1.0,
    region: DeltaRegion | None = None,
) -> CharacterSpec:
    """Apply `weight * delta` onto `target`, optionally filtering to a sub-region."""
    out = target.model_copy(deep=True)
    eff_region = region or delta.region
    allowed = FACE_REGION_FIELDS[eff_region]
    alpha = float(weight)

    body_data = out.body.model_dump()
    for k, dv in delta.body_delta.items():
        if k in allowed and k in body_data:
            body_data[k] = _clamp_field(BodySliders, k, body_data[k] + alpha * float(dv))
    out.body = BodySliders(**body_data)

    face_data = out.face.model_dump()
    for k, dv in delta.face_delta.items():
        if k in allowed and k in face_data:
            face_data[k] = _clamp_field(FaceSliders, k, face_data[k] + alpha * float(dv))
    out.face = FaceSliders(**face_data)

    if delta.expression_delta:
        expr_data = out.expressions.model_dump()
        for k, dv in delta.expression_delta.items():
            if (k in allowed or eff_region in ("all", "expressions")) and k in expr_data:
                expr_data[k] = _clamp_field(ExpressionSliders, k, expr_data[k] + alpha * float(dv))
        out.expressions = ExpressionSliders(**expr_data)

    if eff_region == "all" and delta.custom_morph_delta:
        for ck, dv in delta.custom_morph_delta.items():
            cur_v = float(out.custom_morphs.get(ck, 0.0))
            out.custom_morphs[ck] = max(-2.0, min(2.0, cur_v + alpha * float(dv)))

    if eff_region in ("all", "face") and delta.material_delta:
        mat_data = out.materials.model_dump()
        for mk, mv in delta.material_delta.items():
            if isinstance(mv, list) and mk in mat_data:
                base_c = mat_data[mk]
                mat_data[mk] = tuple(
                    max(0.0, min(1.0, float(base_c[i]) + alpha * float(mv[i]))) for i in range(3)
                )
            elif isinstance(mv, int | float) and mk in mat_data:
                hi = 1.0 if mk == "subsurface_strength" else 0.95
                lo = 0.0 if mk == "subsurface_strength" else 0.05
                mat_data[mk] = max(lo, min(hi, float(mat_data[mk]) + alpha * float(mv)))
        out.materials = CharacterMaterialParams(**mat_data)

    # Merge scaled RBF nodes
    existing = {n.id: n for n in out.rbf_nodes}
    for dn in delta.rbf_nodes:
        if not _node_in_region(dn, eff_region):
            continue
        scaled_w = max(-0.5, min(0.5, dn.weight * alpha))
        scaled_v = tuple(dn.vector_delta[i] * alpha for i in range(3))
        if dn.id in existing:
            cur = existing[dn.id]
            cur.weight = max(-0.5, min(0.5, cur.weight + scaled_w))
            cur.vector_delta = tuple(cur.vector_delta[i] + scaled_v[i] for i in range(3))  # type: ignore[assignment]
        else:
            out.rbf_nodes.append(
                RBFDeltaNode(
                    id=f"{dn.id}_{uuid.uuid4().hex[:4]}",
                    region=dn.region,
                    space=dn.space,
                    center=dn.center,
                    radius=dn.radius,
                    weight=scaled_w,
                    vector_delta=scaled_v,  # type: ignore[arg-type]
                    mirror_x=dn.mirror_x,
                )
            )
    return out


def blend_characters(
    char_a: CharacterSpec,
    char_b: CharacterSpec,
    alpha: float = 0.5,
    region: DeltaRegion = "all",
    region_weights: dict[DeltaRegion, float] | None = None,
) -> CharacterSpec:
    """Blend `char_a` toward `char_b` globally or via per-region delta weights."""
    if region_weights:
        out = char_a.model_copy(deep=True)
        for reg, w in region_weights.items():
            d = compute_character_delta(char_a, char_b, region=reg, name=f"blend_{reg}")
            out = apply_character_delta(out, d, weight=float(w), region=reg)
        return out

    d = compute_character_delta(char_a, char_b, region=region, name=f"blend_{region}")
    return apply_character_delta(char_a, d, weight=float(alpha), region=region)


def extract_custom_morph_from_mesh_edit(
    spec: CharacterSpec,
    edited_mesh: MeshData,
    morph_name: str,
    *,
    region: str = "all",
    threshold: float = 1e-4,
) -> CustomMorphTarget:
    """Project a user mesh edit back onto base-cage vertex deltas as a reusable `CustomMorphTarget`.

    Works on either base-cage edits OR Catmull-Clark subdivided skin edits (since base vertices
    correspond to the first `N_base` rows of the subdivided mesh in Catmull-Clark ordering).
    """
    surface = CharacterSurface(spec, quality=spec.quality)
    orig_mesh = surface.skin_mesh
    qb = build_canonical_quad_base()
    n_base = len(qb.verts)

    if len(edited_mesh.vertices) != len(orig_mesh.vertices) or orig_mesh.is_empty:
        raise ValueError(
            "extract_custom_morph_from_mesh_edit requires identical vertex counts between "
            "the character's current skin mesh and the edited mesh."
        )

    # First N_base vertices in Catmull-Clark subdivision correspond 1-to-1 with base control verts!
    disp_base = edited_mesh.vertices[:n_base] - orig_mesh.vertices[:n_base]

    # Undo height stature scaling so the saved morph target transfers cleanly across any stature
    h_scale = max(0.5, float(surface.lm.height) / 1.740)
    disp_canon = disp_base / h_scale

    deltas: list[tuple[int, Vec3]] = []
    mags = np.linalg.norm(disp_canon, axis=1)
    for vi in np.nonzero(mags > float(threshold))[0]:
        dv = disp_canon[int(vi)]
        deltas.append(
            (int(vi), (round(float(dv[0]), 6), round(float(dv[1]), 6), round(float(dv[2]), 6)))
        )

    return CustomMorphTarget(
        name=morph_name.strip() or f"morph_{uuid.uuid4().hex[:6]}",
        region=region.strip().lower() or "all",
        deltas=deltas,
    )


def extract_rbf_delta_from_mesh_edit(
    orig_mesh: MeshData,
    edited_mesh: MeshData,
    spec_or_lm: CharacterSpec | AnatomicalLandmarks,
    *,
    max_nodes: int = 16,
    radius_norm: float = 0.28,
    region_name: str = "mesh_edit",
) -> list[RBFDeltaNode]:
    """Fit sparse Wendland C2 RBF displacement nodes from a vertex/face mesh edit.

    Requires `orig_mesh` and `edited_mesh` to share vertex topology (e.g. after Edit Mode
    translating/moving/scaling faces or vertices).
    """
    if len(orig_mesh.vertices) != len(edited_mesh.vertices) or orig_mesh.is_empty:
        raise ValueError(
            "extract_rbf_delta_from_mesh_edit requires identical vertex counts between "
            "original and edited meshes."
        )
    lm = (
        spec_or_lm if isinstance(spec_or_lm, AnatomicalLandmarks) else compute_landmarks(spec_or_lm)
    )

    v0 = orig_mesh.vertices
    disp = edited_mesh.vertices - v0
    mags = np.linalg.norm(disp, axis=1)
    moved = np.nonzero(mags > 1e-4)[0]
    if len(moved) == 0:
        return []

    # Greedy weighted Farthest-Point Sampling to choose well-spaced RBF control centers
    pts_m = v0[moved]
    disp_m = disp[moved]
    mags_m = mags[moved]

    k = min(max(1, int(max_nodes)), len(moved))
    selected_idx: list[int] = [int(np.argmax(mags_m))]
    min_d2 = np.sum((pts_m - pts_m[selected_idx[0]]) ** 2, axis=1)

    for _ in range(1, k):
        # Balance spatial spread with displacement magnitude
        score = min_d2 * (mags_m + 1e-4)
        nxt = int(np.argmax(score))
        if min_d2[nxt] < 1e-6:
            break
        selected_idx.append(nxt)
        d2_new = np.sum((pts_m - pts_m[nxt]) ** 2, axis=1)
        min_d2 = np.minimum(min_d2, d2_new)

    centers_w = pts_m[selected_idx]
    m = len(centers_w)

    # Determine head vs body space per center and solve damped RBF linear system
    hh = lm.head_half_h
    head_radius_m = hh * 1.45
    rad_m = radius_norm * hh

    # Build Wendland C2 interpolation matrix Phi (m x m) in world meters
    diff_cc = centers_w[:, None, :] - centers_w[None, :, :]
    q_cc = np.linalg.norm(diff_cc, axis=2) / max(1e-4, rad_m)
    phi_mat = wendland_c2(q_cc)
    reg = np.eye(m, dtype=np.float64) * 0.08
    target_disp = disp_m[selected_idx]
    weights_w = np.linalg.solve(phi_mat + reg, target_disp)

    nodes: list[RBFDeltaNode] = []
    for idx_i in range(m):
        cw = centers_w[idx_i]
        vw = weights_w[idx_i]
        if float(np.linalg.norm(vw)) < 1e-5:
            continue
        in_head = float(np.linalg.norm(cw - lm.head_center)) <= head_radius_m
        if in_head:
            c_norm = (cw - lm.head_center) / hh
            v_norm = vw / hh
            nodes.append(
                RBFDeltaNode(
                    id=f"edit_{uuid.uuid4().hex[:6]}",
                    region=region_name,
                    space="head_norm",
                    center=(float(c_norm[0]), float(c_norm[1]), float(c_norm[2])),
                    radius=float(radius_norm),
                    weight=0.0,
                    vector_delta=(float(v_norm[0]), float(v_norm[1]), float(v_norm[2])),
                    mirror_x=False,
                )
            )
        else:
            nodes.append(
                RBFDeltaNode(
                    id=f"edit_{uuid.uuid4().hex[:6]}",
                    region=region_name,
                    space="local",
                    center=(float(cw[0]), float(cw[1]), float(cw[2])),
                    radius=float(rad_m),
                    weight=0.0,
                    vector_delta=(float(vw[0]), float(vw[1]), float(vw[2])),
                    mirror_x=False,
                )
            )
    return nodes
