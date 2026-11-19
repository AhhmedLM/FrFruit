"""CharacterSurface: KDTree surface queries & vertex-anchored AnatomicalLandmarks."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.spatial import cKDTree

from frfruit.character.morphs import build_character_skin_mesh, evaluate_base_cage
from frfruit.character.quad_base import EYE_RADIUS, CanonicalQuadBase, build_canonical_quad_base
from frfruit.character.schema import CharacterSpec


@dataclass(frozen=True)
class AnatomicalLandmarks:
    """Exact 3D anatomical landmarks derived directly from the morphed Quad Base Cage."""

    height: float
    head_units: float
    head_height: float
    head_half_h: float
    head_center: np.ndarray
    crown: np.ndarray
    chin: np.ndarray
    neck_top: np.ndarray
    neck_base: np.ndarray
    chest: np.ndarray
    waist: np.ndarray
    pelvis: np.ndarray
    shoulder_left: np.ndarray
    shoulder_right: np.ndarray
    elbow_left: np.ndarray
    elbow_right: np.ndarray
    wrist_left: np.ndarray
    wrist_right: np.ndarray
    hand_tip_left: np.ndarray
    hand_tip_right: np.ndarray
    hip_left: np.ndarray
    hip_right: np.ndarray
    knee_left: np.ndarray
    knee_right: np.ndarray
    ankle_left: np.ndarray
    ankle_right: np.ndarray
    toe_left: np.ndarray
    toe_right: np.ndarray
    eye_left: np.ndarray
    eye_right: np.ndarray
    eye_radius: float
    nose_tip: np.ndarray
    mouth_center: np.ndarray
    ear_left: np.ndarray
    ear_right: np.ndarray
    facial_thirds: tuple[float, float, float]
    eye_to_face_width_ratio: float

    @property
    def head_to_body_ratio(self) -> float:
        return round(float(self.head_height / max(1e-6, self.height)), 4)


def _tag_mean(verts: np.ndarray, tags: dict[str, np.ndarray], name: str) -> np.ndarray:
    idx = tags[name]
    return np.mean(verts[idx], axis=0)


def _tag_intersect_mean(
    verts: np.ndarray,
    tags: dict[str, np.ndarray],
    name_a: str,
    name_b: str,
) -> np.ndarray:
    a = tags.get(name_a, np.empty(0, dtype=np.int64))
    b = tags.get(name_b, np.empty(0, dtype=np.int64))
    common = np.intersect1d(a, b)
    if len(common) == 0:
        return np.mean(verts[a if len(a) > 0 else b], axis=0)
    return np.mean(verts[common], axis=0)


def compute_landmarks_from_verts(
    spec: CharacterSpec,
    base_verts: np.ndarray,
    base: CanonicalQuadBase | None = None,
) -> AnatomicalLandmarks:
    """Compute exact 3D `AnatomicalLandmarks` directly from morphed base-cage vertices."""
    qb = base if base is not None else build_canonical_quad_base()
    tags = qb.tags
    v = base_verts

    height = float(np.max(v[:, 2]) - np.min(v[:, 2]))
    crown_z = float(np.max(v[:, 2]))
    crown = np.array([0.0, float(_tag_mean(v, tags, "crown_ring")[1]), crown_z], dtype=np.float64)
    chin = _tag_mean(v, tags, "chin_tip")
    head_height = max(0.16, float(crown[2] - chin[2]))
    head_half_h = 0.5 * head_height
    head_units = height / head_height

    head_pts = v[tags["head"]]
    head_center = np.array(
        [0.0, float(np.mean(head_pts[:, 1])), float(chin[2] + head_half_h)],
        dtype=np.float64,
    )

    neck_base = _tag_mean(v, tags, "neck_base_ring")
    neck_top = _tag_mean(v, tags, "neck_top_ring")
    chest = _tag_mean(v, tags, "chest_ring")
    waist = _tag_mean(v, tags, "waist_ring")
    pelvis = _tag_mean(v, tags, "pelvis_ring")

    # Arms
    sh_l = _tag_intersect_mean(v, tags, "shoulder", "arm_L")
    sh_r = _tag_intersect_mean(v, tags, "shoulder", "arm_R")
    el_l = _tag_intersect_mean(v, tags, "elbow", "arm_L")
    el_r = _tag_intersect_mean(v, tags, "elbow", "arm_R")
    wr_l = _tag_intersect_mean(v, tags, "wrist", "arm_L")
    wr_r = _tag_intersect_mean(v, tags, "wrist", "arm_R")
    hand_l_pts = v[np.intersect1d(tags["hand"], tags["arm_L"])]
    hand_r_pts = v[np.intersect1d(tags["hand"], tags["arm_R"])]
    ht_l = hand_l_pts[int(np.argmax(hand_l_pts[:, 0]))]
    ht_r = hand_r_pts[int(np.argmin(hand_r_pts[:, 0]))]

    # Legs
    thigh_l_pts = v[np.intersect1d(tags["thigh"], tags["leg_L"])]
    thigh_r_pts = v[np.intersect1d(tags["thigh"], tags["leg_R"])]
    hip_l = np.array(
        [float(np.mean(thigh_l_pts[:, 0])), float(np.mean(thigh_l_pts[:, 1])), float(pelvis[2])],
        dtype=np.float64,
    )
    hip_r = np.array(
        [float(np.mean(thigh_r_pts[:, 0])), float(np.mean(thigh_r_pts[:, 1])), float(pelvis[2])],
        dtype=np.float64,
    )
    kn_l = _tag_intersect_mean(v, tags, "knee", "leg_L")
    kn_r = _tag_intersect_mean(v, tags, "knee", "leg_R")
    an_l = _tag_intersect_mean(v, tags, "ankle", "leg_L")
    an_r = _tag_intersect_mean(v, tags, "ankle", "leg_R")
    foot_l_pts = v[np.intersect1d(tags["foot"], tags["leg_L"])]
    foot_r_pts = v[np.intersect1d(tags["foot"], tags["leg_R"])]
    toe_l = foot_l_pts[int(np.argmin(foot_l_pts[:, 1]))]
    toe_r = foot_r_pts[int(np.argmin(foot_r_pts[:, 1]))]

    # Eyes: seat eyeball sphere precisely behind the morphed eyelid margin ring!
    margin_l = v[tags["eye_margin_L"]]
    margin_r = v[tags["eye_margin_R"]]
    scale_h = height / 1.740
    eye_scale = (1.0 + 0.14 * float(spec.face.eye_size)) * float(spec.body.head_scale) * scale_h
    eye_r = EYE_RADIUS * eye_scale

    # Margin center is nestled right around the cornea front pole along -Y
    el_c_xz = np.mean(margin_l, axis=0)
    er_c_xz = np.mean(margin_r, axis=0)
    eye_l = np.array(
        [float(el_c_xz[0]), float(np.mean(margin_l[:, 1]) + eye_r * 1.12), float(el_c_xz[2])],
        dtype=np.float64,
    )
    eye_r_pos = np.array(
        [float(er_c_xz[0]), float(np.mean(margin_r[:, 1]) + eye_r * 1.12), float(er_c_xz[2])],
        dtype=np.float64,
    )

    nose_tip = _tag_mean(v, tags, "nose_tip_v")
    mouth_center = _tag_mean(v, tags, "mouth_lip_seam")
    ear_l = _tag_mean(v, tags, "ear_L")
    ear_r = _tag_mean(v, tags, "ear_R")

    # Anthropometric metrics
    face_pts = v[tags["face"]]
    face_w = max(0.08, float(np.max(face_pts[:, 0]) - np.min(face_pts[:, 0])))
    eye_w = float(np.max(margin_l[:, 0]) - np.min(margin_l[:, 0]))
    eye_ratio = float(eye_w / face_w)

    brow_z = float(_tag_mean(v, tags, "brow_L")[2])
    hairline_z = float(crown[2] - 0.14 * head_height)
    t_upper = max(0.01, hairline_z - brow_z)
    t_mid = max(0.01, brow_z - float(nose_tip[2]))
    t_low = max(0.01, float(nose_tip[2]) - float(chin[2]))
    t_sum = t_upper + t_mid + t_low
    facial_thirds = (t_upper / t_sum, t_mid / t_sum, t_low / t_sum)

    return AnatomicalLandmarks(
        height=height,
        head_units=head_units,
        head_height=head_height,
        head_half_h=head_half_h,
        head_center=head_center,
        crown=crown,
        chin=chin,
        neck_top=neck_top,
        neck_base=neck_base,
        chest=chest,
        waist=waist,
        pelvis=pelvis,
        shoulder_left=sh_l,
        shoulder_right=sh_r,
        elbow_left=el_l,
        elbow_right=el_r,
        wrist_left=wr_l,
        wrist_right=wr_r,
        hand_tip_left=ht_l,
        hand_tip_right=ht_r,
        hip_left=hip_l,
        hip_right=hip_r,
        knee_left=kn_l,
        knee_right=kn_r,
        ankle_left=an_l,
        ankle_right=an_r,
        toe_left=toe_l,
        toe_right=toe_r,
        eye_left=eye_l,
        eye_right=eye_r_pos,
        eye_radius=eye_r,
        nose_tip=nose_tip,
        mouth_center=mouth_center,
        ear_left=ear_l,
        ear_right=ear_r,
        facial_thirds=facial_thirds,
        eye_to_face_width_ratio=eye_ratio,
    )


def compute_landmarks(spec: CharacterSpec) -> AnatomicalLandmarks:
    """Compute exact 3D landmarks for `spec` by evaluating the fast Quad Base Cage."""
    base_verts = evaluate_base_cage(spec)
    return compute_landmarks_from_verts(spec, base_verts)


class CharacterSurface:
    """Fast KDTree-backed skin surface wrapper replacing the old SDF evaluator.

    Provides exact signed distance, surface projection, skin-clearance pushing, and
    morphed landmark & UV queries for Eyes, Brows, Hair, Clothing, and PBR Materials.
    """

    def __init__(
        self,
        spec: CharacterSpec,
        quality: str | None = None,
    ) -> None:
        self.spec = spec
        self.base = build_canonical_quad_base()
        self.skin_mesh, self.base_verts = build_character_skin_mesh(spec, quality=quality)
        self.lm = compute_landmarks_from_verts(spec, self.base_verts, self.base)

        self._verts = self.skin_mesh.vertices
        self._normals = self.skin_mesh.vertex_normals()
        self._tree = cKDTree(self._verts)

    def query_nearest(self, pts: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return `(nearest_surface_pts, nearest_normals, unsigned_dists)` for `(N, 3)` points."""
        p = np.asarray(pts, dtype=np.float64)
        dists, idxs = self._tree.query(p, workers=1)
        return self._verts[idxs], self._normals[idxs], dists

    def signed_distance(self, pts: np.ndarray) -> np.ndarray:
        """Approximate signed distance to skin surface (< 0 inside, > 0 outside) via KDTree."""
        p = np.asarray(pts, dtype=np.float64)
        if len(p) == 0:
            return np.zeros((0,), dtype=np.float64)
        surf_p, surf_n, dists = self.query_nearest(p)
        dot = np.sum((p - surf_p) * surf_n, axis=1)
        # Signed distance: negative inside skin, positive outside
        sign = np.where(dot >= 0.0, 1.0, -1.0)
        return sign * dists

    def evaluate(self, pts: np.ndarray) -> np.ndarray:
        """Drop-in alias for `signed_distance(pts)`."""
        return self.signed_distance(pts)

    def project_to_surface(
        self,
        pts: np.ndarray,
        offset: float = 0.0,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Project `(N, 3)` points cleanly onto skin surface + `offset * normal`."""
        surf_p, surf_n, _ = self.query_nearest(pts)
        return surf_p + surf_n * float(offset), surf_n

    def push_clear_of_skin(
        self,
        pts: np.ndarray,
        min_clear: float = 0.008,
    ) -> np.ndarray:
        """Guarantee every point in `(N, 3)` lies at least `min_clear` meters outside skin."""
        p = np.asarray(pts, dtype=np.float64).copy()
        if len(p) == 0:
            return p
        surf_p, surf_n, dists = self.query_nearest(p)
        diff = p - surf_p
        signed_n = np.sum(diff * surf_n, axis=1)
        signed_d = np.where(signed_n >= 0.0, dists, -dists)
        pen = float(min_clear) - signed_d
        hit = pen > 0.0
        if np.any(hit):
            # Move penetrating or too-close points directly to surf_p + min_clear * surf_n
            p[hit] = surf_p[hit] + surf_n[hit] * float(min_clear)
        return p

    def lip_uv_contours(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return exact UV polylines `(upper_uv, lower_uv, seam_uv)` from Quad Base lip loops."""
        uvs = self.base.uvs
        v = self.base_verts
        tags = self.base.tags

        def _sorted_curve(idx_arr: np.ndarray, upper: bool | None) -> np.ndarray:
            pts = v[idx_arr]
            uv_arr = uvs[idx_arr]
            if upper is not None:
                mz = float(self.lm.mouth_center[2])
                if upper:
                    keep = pts[:, 2] >= mz - 0.001
                else:
                    keep = pts[:, 2] <= mz + 0.001
                if np.sum(keep) >= 3:
                    pts = pts[keep]
                    uv_arr = uv_arr[keep]
            order = np.argsort(pts[:, 0])
            return uv_arr[order]

        up_uv = _sorted_curve(tags["mouth_lip_crest"], upper=True)
        lo_uv = _sorted_curve(tags["mouth_lip_crest"], upper=False)
        mid_uv = _sorted_curve(tags["mouth_lip_seam"], upper=None)
        return up_uv, lo_uv, mid_uv

    def brow_spine_points(self, side_sign: float) -> np.ndarray:
        """Return ordered inner-to-outer 3D brow points directly from the morphed brow vertices."""
        tag = "brow_L" if side_sign > 0 else "brow_R"
        pts = self.base_verts[self.base.tags[tag]]
        order = np.argsort(np.abs(pts[:, 0]))
        return pts[order]

    def eyelid_margin_points(self, side_sign: float, upper: bool) -> np.ndarray:
        """Return ordered inner-to-outer 3D eyelid margin points (tracks blinking automatically)."""
        side_tag = "L" if side_sign > 0 else "R"
        margin_idx = self.base.tags[f"eye_margin_{side_tag}"]
        sub_idx = self.base.tags[f"{'upper' if upper else 'lower'}_lid_{side_tag}"]
        common = np.intersect1d(margin_idx, sub_idx)
        pts = self.base_verts[common if len(common) >= 3 else margin_idx]
        order = np.argsort(np.abs(pts[:, 0]))
        return pts[order]
