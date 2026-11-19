"""Procedural 100% manifold Edge-Looped Quad Base Mesh and cached Catmull-Clark operator.

Constructs a canonical realistic neutral human (1.74m, ~7.45 Head Units, A-pose, facing -Y,
+X = character's left, Z = up with feet at Z = 0) entirely in NumPy/Python using constructive
pure-quad topology with concentric anatomical edge-loops around eyes, mouth/lips, ears, arms,
5-finger hands, and legs.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from functools import lru_cache

import numpy as np
from scipy import sparse

from frfruit.geometry.mesh import MeshData


def _lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * t


def _smoothstep(edge0: float, edge1: float, x: np.ndarray | float) -> np.ndarray | float:
    denom = edge1 - edge0
    if abs(denom) < 1e-9:
        denom = 1e-9
    t = np.clip((x - edge0) / denom, 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def _normalize(v: np.ndarray) -> np.ndarray:
    ln = float(np.linalg.norm(v))
    return v / ln if ln > 1e-12 else np.array([0.0, 0.0, 1.0], dtype=np.float64)


class _QuadBuilder:
    """Constructive pure-quad manifold mesh builder with topological inset & extrude ops."""

    def __init__(self) -> None:
        self.verts: list[np.ndarray] = []
        self.uvs: list[np.ndarray] = []
        self.quads: list[list[int]] = []
        self.tags: dict[str, set[int]] = defaultdict(set)

    def add_vert(
        self,
        pos: tuple[float, float, float] | np.ndarray,
        uv: tuple[float, float] | np.ndarray = (0.5, 0.5),
        tags: tuple[str, ...] = (),
    ) -> int:
        idx = len(self.verts)
        self.verts.append(np.asarray(pos, dtype=np.float64).copy())
        self.uvs.append(np.asarray(uv, dtype=np.float64).copy())
        for t in tags:
            self.tags[t].add(idx)
        return idx

    def add_quad(self, a: int, b: int, c: int, d: int) -> int:
        qi = len(self.quads)
        self.quads.append([int(a), int(b), int(c), int(d)])
        return qi

    def tag_verts(self, name: str, indices: list[int] | set[int]) -> None:
        self.tags[name].update(int(i) for i in indices)

    def extract_patch_boundary(self, patch_quad_ids: list[int]) -> list[int]:
        """Extract the single ordered directed CCW boundary loop of a quad patch."""
        edge_count: dict[tuple[int, int], int] = defaultdict(int)
        directed_edges: set[tuple[int, int]] = set()
        for qi in patch_quad_ids:
            q = self.quads[qi]
            for j in range(4):
                u, v = q[j], q[(j + 1) % 4]
                edge_count[(min(u, v), max(u, v))] += 1
                directed_edges.add((u, v))

        nxt: dict[int, int] = {}
        for u, v in directed_edges:
            if edge_count[(min(u, v), max(u, v))] == 1:
                nxt[u] = v

        if not nxt:
            raise ValueError("Patch has no boundary edges")
        start = next(iter(nxt))
        loop = [start]
        cur = nxt[start]
        while cur != start:
            loop.append(cur)
            cur = nxt[cur]
            if len(loop) > len(nxt) + 1:
                raise ValueError("Patch boundary is not a single simple cycle")
        if len(loop) != len(nxt):
            raise ValueError("Patch boundary has multiple disconnected cycles")
        return loop

    def cap_even_loop(self, loop: list[int], reverse: bool = False) -> list[int]:
        """Close any even-length vertex loop with 100% pure quads (no triangles or poles)."""
        k = len(loop)
        if k < 4 or k % 2 != 0:
            raise ValueError(f"Loop must have even length >= 4, got {k}")
        ordered = list(reversed(loop)) if reverse else list(loop)
        if k == 4:
            return [self.add_quad(ordered[0], ordered[1], ordered[2], ordered[3])]
        m = k // 2
        created: list[int] = []
        for i in range(m - 1):
            a = ordered[i]
            b = ordered[i + 1]
            c = ordered[k - 2 - i]
            d = ordered[k - 1 - i]
            created.append(self.add_quad(a, b, c, d))
        return created

    def bridge_loops(self, loop_a: list[int], loop_b: list[int]) -> list[int]:
        """Connect two equal-length CCW loops with a ring of pure quads."""
        k = len(loop_a)
        if len(loop_b) != k:
            raise ValueError(f"Loop length mismatch: {k} vs {len(loop_b)}")
        created: list[int] = []
        for i in range(k):
            ni = (i + 1) % k
            created.append(self.add_quad(loop_a[i], loop_a[ni], loop_b[ni], loop_b[i]))
        return created

    def inset_patch(
        self,
        patch_quad_ids: list[int],
        ring_positions: list[list[np.ndarray]],
        ring_uvs: list[list[np.ndarray]] | None = None,
        ring_tags: list[tuple[str, ...]] | None = None,
        cap_inner: bool = True,
    ) -> tuple[list[int], list[list[int]]]:
        """Replace `patch_quad_ids` with concentric quad rings and optional pure-quad inner cap.

        Returns `(outer_loop, created_rings)` preserving exact manifold winding.
        """
        outer_loop = self.extract_patch_boundary(patch_quad_ids)
        k = len(outer_loop)
        for ri, rpos in enumerate(ring_positions):
            if len(rpos) != k:
                raise ValueError(f"Ring {ri} has {len(rpos)} positions, expected {k}")

        # Mark replaced quads as deleted (compacted in finalize())
        for qi in patch_quad_ids:
            self.quads[qi] = []

        created_rings: list[list[int]] = []
        prev_loop = outer_loop
        for ri, rpos in enumerate(ring_positions):
            ruvs = ring_uvs[ri] if ring_uvs is not None else [self.uvs[v].copy() for v in prev_loop]
            rtags = ring_tags[ri] if ring_tags is not None else ()
            cur_ring: list[int] = []
            for j in range(k):
                vi = self.add_vert(rpos[j], ruvs[j], tags=rtags)
                cur_ring.append(vi)
            self.bridge_loops(prev_loop, cur_ring)
            created_rings.append(cur_ring)
            prev_loop = cur_ring

        if cap_inner and created_rings:
            self.cap_even_loop(created_rings[-1], reverse=False)

        return outer_loop, created_rings

    def finalize(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, np.ndarray]]:
        """Compact deleted quads & orphaned vertices, returning `(verts, uvs, quads, tags)`."""
        active_quads = [q for q in self.quads if len(q) == 4]
        used = sorted({v for q in active_quads for v in q})
        remap = {old: new for new, old in enumerate(used)}
        verts = np.asarray([self.verts[i] for i in used], dtype=np.float64)
        uvs = np.asarray([self.uvs[i] for i in used], dtype=np.float64)
        quads = np.asarray([[remap[v] for v in q] for q in active_quads], dtype=np.int64)
        out_tags: dict[str, np.ndarray] = {}
        for name, idxs in self.tags.items():
            mapped = sorted(remap[i] for i in idxs if i in remap)
            if mapped:
                out_tags[name] = np.asarray(mapped, dtype=np.int64)
        return verts, uvs, quads, out_tags


# ---------------------------------------------------------------------------
# Canonical Anthropometric Constants (Neutral Realistic Adult, 1.74m, 7.45 HU)
# ---------------------------------------------------------------------------

CANON_HEIGHT = 1.740
NECK_SEAM_Z = 1.462
HEAD_CENTER = np.array([0.0, 0.004, 1.622], dtype=np.float64)
HEAD_HALF_H = 0.118  # Crown at ~1.740m, chin at ~1.504m (head height = 0.236m -> 7.37 HU)
EYE_CENTER_L = np.array([0.0325, -0.078, 1.617], dtype=np.float64)
EYE_RADIUS = 0.0116
MOUTH_Z = 1.553
MOUTH_W = 0.0235
NOSE_TIP_Z = 1.588
NOSE_TIP_Y = -0.113


def _head_uv(pos: np.ndarray) -> np.ndarray:
    """Seam-free continuous head UV inside u in [0.04, 0.96], v in [0.44, 0.98]."""
    rel = pos - HEAD_CENTER
    phi = np.arctan2(rel[0], -rel[1])  # 0 = front nose, +/-pi = back head
    u = 0.50 + (phi / np.pi) * 0.46
    t_v = float(np.clip((float(pos[2]) - NECK_SEAM_Z) / (CANON_HEIGHT - NECK_SEAM_Z), 0.0, 1.0))
    v = 0.44 + t_v * 0.54
    return np.array(
        [float(np.clip(u, 0.04, 0.96)), float(np.clip(v, 0.44, 0.98))], dtype=np.float64
    )


def _body_uv(pos: np.ndarray) -> np.ndarray:
    """Seam-free continuous body UV sharing azimuth with _head_uv, v in [0.02, 0.44]."""
    phi = np.arctan2(pos[0], -pos[1])
    u = 0.50 + (phi / np.pi) * 0.46
    v = 0.02 + float(np.clip(float(pos[2]) / NECK_SEAM_Z, 0.0, 1.0)) * 0.42
    return np.array(
        [float(np.clip(u, 0.04, 0.96)), float(np.clip(v, 0.02, 0.44))], dtype=np.float64
    )


def _eval_head_ring_point(theta: float, z: float) -> np.ndarray:
    """Evaluate canonical sculpted anatomical head & neck surface at `(theta, z)`.

    Convention: `theta = 0` is front midline (-Y), `+pi/2` is left (+X), `pi` is back (+Y).
    """
    ct = float(np.cos(theta))
    st = float(np.sin(theta))
    abs_st = abs(st)
    front = max(0.0, ct)

    # Normalized head Z: -1.0 = chin (1.504), 0.0 = eye/mid-head (1.622), +1.0 = crown (1.740)
    zn = (z - HEAD_CENTER[2]) / HEAD_HALF_H

    if zn < -0.96:
        # Anatomical neck cylinder tapering gently into trapezius & sternocleidomastoid
        t_neck = float(np.clip((-0.96 - zn) / 0.56, 0.0, 1.0))
        rx = _lerp(0.050, 0.068, t_neck**1.35)
        ry_f = _lerp(0.052, 0.064, t_neck**1.25)
        ry_b = _lerp(0.060, 0.072, t_neck**1.25)
        y_off = _lerp(0.020, 0.015, t_neck)
        x = st * rx
        y = y_off - ct * (ry_f if ct >= 0.0 else ry_b)
        return np.array([x, y, z], dtype=np.float64)

    # 1) Full rounded Loomis cranium oval (wider parietal dome, gentle oval cheek/jaw taper)
    if zn >= 0.14:
        u_dome = float(np.clip((zn - 0.14) / 0.87, 0.0, 0.995))
        dome_scale = float(np.sqrt(max(0.02, 1.0 - u_dome * u_dome)))
        rx = 0.0840 * dome_scale * (1.0 + 0.030 * float(np.exp(-(((zn - 0.34) / 0.30) ** 2))))
        ry_f = 0.0955 * dome_scale
        ry_b = 0.1095 * dome_scale * (1.0 + 0.045 * float(np.exp(-(((zn - 0.28) / 0.34) ** 2))))
    else:
        low_t = float(np.clip((0.14 - zn) / 1.10, 0.0, 1.0))
        # Natural anatomical oval taper from zygoma through jawline to chin
        rx = 0.0840 * (1.0 - 0.50 * (low_t**1.45))
        # Front face holds balanced forward plane from brow through maxilla to chin
        ry_f = 0.0955 * (1.0 - 0.09 * (low_t**1.30))
        # Full occiput curving cleanly under back skull into upper nape
        nape_t = float(np.clip((-0.12 - zn) / 0.84, 0.0, 1.0))
        ry_b = _lerp(0.1095 * (1.0 - 0.08 * low_t), 0.058, nape_t**1.15)

    # Shift upper cranium slightly back (+Y) for authentic anatomical skull balance
    y_cranium_shift = 0.006 * float(_smoothstep(-0.10, 0.45, zn))
    x = st * rx
    y = y_cranium_shift - ct * (ry_f if ct >= 0.0 else ry_b)

    # --- Anatomical Facial Planes & Landmarks ---
    # 2) Subtle temporal planes on upper sides
    temp_w = float(_smoothstep(0.62, 0.94, abs_st)) * float(np.exp(-(((zn - 0.28) / 0.28) ** 2)))
    x *= 1.0 - 0.032 * temp_w

    # 3) Gentle supraorbital brow ridge & glabella overhang (zn ~ +0.15)
    brow_w = float(np.exp(-(((zn - 0.15) / 0.11) ** 2))) * float(_smoothstep(0.32, 0.90, front))
    y -= 0.0065 * brow_w * (0.68 + 0.32 * np.cos(2.2 * theta))

    # 4) Soft recessed orbital hollows (zn ~ -0.04, abs_st ~ 0.43) — avoids harsh crease!
    orb_w = (
        float(np.exp(-(((zn + 0.04) / 0.13) ** 2)))
        * float(np.exp(-(((abs_st - 0.44) / 0.22) ** 2)))
        * float(_smoothstep(0.30, 0.85, front))
    )
    y += 0.0082 * orb_w

    # 5) Soft malar cheekbone prominence (zn ~ -0.20, abs_st ~ 0.62)
    zyg_w = (
        float(np.exp(-(((zn + 0.20) / 0.17) ** 2)))
        * float(np.exp(-(((abs_st - 0.62) / 0.25) ** 2)))
        * float(_smoothstep(0.14, 0.74, front))
    )
    x += np.sign(st) * 0.0040 * zyg_w
    y -= 0.0050 * zyg_w

    # 6) Very subtle cheek contour below zygoma (prevents gaunt pinched cheeks!)
    bucc_w = (
        float(np.exp(-(((zn + 0.46) / 0.18) ** 2)))
        * float(np.exp(-(((abs_st - 0.56) / 0.24) ** 2)))
        * float(_smoothstep(0.24, 0.78, front))
    )
    y += 0.0018 * bucc_w
    x *= 1.0 - 0.012 * bucc_w

    # 7) Smooth mandibular jaw angle (zn ~ -0.62, abs_st ~ 0.74)
    jaw_w = float(np.exp(-(((zn + 0.62) / 0.18) ** 2))) * float(
        np.exp(-(((abs_st - 0.74) / 0.24) ** 2))
    )
    x += np.sign(st) * 0.0045 * jaw_w

    # 8) Smooth dental arch / muzzle forward curve around mouth (zn ~ -0.56)
    muz_w = float(np.exp(-(((zn + 0.56) / 0.22) ** 2))) * float(_smoothstep(0.48, 0.98, front))
    y -= 0.0048 * muz_w

    # 9) Mentolabial fold (dip under lower lip) & sculpted Chin prominence (zn ~ -0.84)
    mento_dip = float(np.exp(-(((zn + 0.73) / 0.068) ** 2))) * float(_smoothstep(0.70, 0.98, front))
    y += 0.0042 * mento_dip
    chin_w = float(np.exp(-(((zn + 0.85) / 0.105) ** 2))) * float(_smoothstep(0.66, 0.99, front))
    y -= 0.0072 * chin_w

    # 10) Crisp under-chin / submental curve into upper neck (zn < -0.86)
    if zn < -0.86:
        sub_t = float(np.clip((-0.86 - zn) / 0.12, 0.0, 1.0))
        y = _lerp(y, 0.020 - ct * (0.052 if ct >= 0.0 else 0.060), sub_t**1.15)
        x = _lerp(x, st * 0.050, sub_t**1.15)

    # 11) Smooth Proportional Nasal Pyramid (wider cheek transition -> zero under-eye V-crease!)
    if -0.40 < zn < 0.12 and front > 0.52:
        # Soft nasion indent between eyes (zn ~ -0.01)
        nasion = float(np.exp(-(((zn + 0.01) / 0.065) ** 2))) * float(
            np.exp(-((abs_st / 0.14) ** 2))
        )
        y += 0.0038 * nasion

        # Proportional straight nasal bridge peaking at tip (zn = -0.27, z ~ 1.590)
        nose_v = float(np.clip((-0.01 - zn) / 0.28, 0.0, 1.0))
        nose_env = float(_smoothstep(-0.37, -0.27, zn)) * float(1.0 - _smoothstep(-0.01, 0.09, zn))
        # Wider smooth Gaussian slope blends nose walls cleanly into cheeks without vertical creases
        ridge_w = 0.155 + 0.060 * (nose_v**1.1)
        ridge_profile = float(np.exp(-((abs_st / ridge_w) ** 2.0)))
        tip_bump = 0.0055 * float(np.exp(-(((zn + 0.275) / 0.060) ** 2)))
        y -= (0.0055 + 0.0185 * (nose_v**0.95) + tip_bump) * ridge_profile * nose_env

        # Soft alar wings (nostril sides at zn ~ -0.29, abs_st ~ 0.165)
        ala_w = float(np.exp(-(((zn + 0.29) / 0.055) ** 2))) * float(
            np.exp(-(((abs_st - 0.165) / 0.072) ** 2))
        )
        y -= 0.0105 * ala_w
        x += np.sign(st) * 0.0028 * ala_w

        # Subtle philtrum dimple beneath columella (zn ~ -0.40)
        phil_w = float(np.exp(-(((zn + 0.40) / 0.052) ** 2))) * float(
            np.exp(-((abs_st / 0.080) ** 2))
        )
        y += 0.0024 * phil_w

    return np.array([x, y, z], dtype=np.float64)


def _eval_torso_ring_point(theta: float, z: float) -> np.ndarray:
    """Evaluate canonical neutral realistic adult torso cross-section at `(theta, z)`.

    `z` spans pelvis (`0.865m`) through waist (`1.045m`), chest (`1.295m`), shoulders (`1.415m`),
    up to neck base (`1.462m`).
    """
    ct = float(np.cos(theta))
    st = float(np.sin(theta))
    abs_st = abs(st)
    front = max(0.0, ct)
    back = max(0.0, -ct)

    # Piecewise-smooth anatomical torso key stations: (z, rx, ry_front, ry_back, y_spine_offset)
    stations = np.array(
        [
            [0.865, 0.160, 0.106, 0.122, 0.005],  # Lower pelvis / hip crest
            [0.915, 0.170, 0.108, 0.132, 0.007],  # Hip widest / gluteal fullness
            [0.975, 0.148, 0.100, 0.110, 0.003],  # Iliac / lower abdomen
            [1.045, 0.130, 0.094, 0.094, -0.005],  # Natural tapered waist (lumbar lordosis)
            [1.135, 0.144, 0.106, 0.108, 0.002],  # Lower ribcage
            [1.225, 0.164, 0.122, 0.120, 0.009],  # Mid ribcage / lower chest
            [1.295, 0.180, 0.128, 0.126, 0.012],  # Chest / pectoral apex
            [1.365, 0.198, 0.112, 0.118, 0.014],  # Upper chest / deltoid root
            [1.415, 0.184, 0.090, 0.100, 0.014],  # Shoulder crest / clavicle
            [1.444, 0.114, 0.074, 0.082, 0.015],  # Trapezius slope
            [1.462, 0.068, 0.064, 0.072, 0.015],  # Neck base (matches head bottom!)
        ],
        dtype=np.float64,
    )
    zs = stations[:, 0]
    rx = float(np.interp(z, zs, stations[:, 1]))
    ry_f = float(np.interp(z, zs, stations[:, 2]))
    ry_b = float(np.interp(z, zs, stations[:, 3]))
    y_c = float(np.interp(z, zs, stations[:, 4]))

    # Squircle factor gives realistic anatomical flat-front chest & broader shoulders/hips
    sq = 0.08 * (abs_st**2) * (ct**2) * 4.0
    x = st * rx * (1.0 + sq)
    y = y_c - ct * (ry_f if ct >= 0.0 else ry_b) * (1.0 + 0.5 * sq)

    # Sternum & dorsal spine central vertical furrows for anatomical shading
    sternum_w = float(np.exp(-(((z - 1.28) / 0.11) ** 2))) * float(np.exp(-((abs_st / 0.15) ** 2)))
    y += 0.0070 * sternum_w * front
    spine_w = float(_smoothstep(0.92, 1.05, z)) * float(1.0 - _smoothstep(1.36, 1.44, z))
    y -= 0.0085 * spine_w * float(np.exp(-((abs_st / 0.15) ** 2))) * back

    # Clavicle contour at z ~ 1.41m
    clav_w = float(np.exp(-(((z - 1.412) / 0.022) ** 2))) * float(
        np.exp(-(((abs_st - 0.45) / 0.28) ** 2))
    )
    y -= 0.0055 * clav_w * front

    # Natural pectoral contour (neutral adult)
    pec_w = float(np.exp(-(((z - 1.288) / 0.055) ** 2))) * float(
        np.exp(-(((abs_st - 0.36) / 0.24) ** 2))
    )
    y -= 0.0085 * pec_w * front

    # Gluteal fullness at back hips (z ~ 0.905m)
    glut_w = float(np.exp(-(((z - 0.905) / 0.065) ** 2))) * float(
        np.exp(-(((abs_st - 0.36) / 0.28) ** 2))
    )
    y += 0.0135 * glut_w * back

    return np.array([x, y, z], dtype=np.float64)


def _extrude_eye_socket(
    qb: _QuadBuilder,
    patch_quads: list[int],
    side_sign: float,
) -> dict[str, list[int]]:
    """Inset a 4-ring concentric anatomical eyelid & socket loop around the recessed eyeball."""
    outer_loop = qb.extract_patch_boundary(patch_quads)
    k = len(outer_loop)
    eye_c = np.array([side_sign * EYE_CENTER_L[0], EYE_CENTER_L[1], EYE_CENTER_L[2]])
    eye_r = EYE_RADIUS

    outer_pts = np.asarray([qb.verts[v] for v in outer_loop], dtype=np.float64)
    dx_loc = (outer_pts[:, 0] - eye_c[0]) * side_sign
    dz_loc = outer_pts[:, 2] - eye_c[2]
    angs = np.arctan2(dz_loc, dx_loc)  # 0 = outer canthus, pi = inner canthus, +pi/2 = upper lid

    # 4 concentric rings wrapping cleanly around deeply-seated eyeball:
    # Ring 0: Eyelid crease / supratarsal fold
    # Ring 1: Eyelid anterior margin / lash line (almond aperture hugging cornea front!)
    # Ring 2: Eyelid inner wetline (thickness bevel behind cornea!)
    # Ring 3: Deep orbital back-cup behind eyeball center (watertight closure)
    ring_specs = [
        ("crease", 1.44, 0.80, 0.0028),
        ("margin", 1.28, 0.46, 0.0014),
        ("wetline", 1.16, 0.38, -0.0025),
        ("socket", 0.76, 0.30, -0.0120),
    ]

    ring_positions: list[list[np.ndarray]] = []
    ring_uvs: list[list[np.ndarray]] = []
    side_tag = "L" if side_sign > 0 else "R"
    ring_tags: list[tuple[str, ...]] = []

    for rname, rx_m, rz_m, y_pad in ring_specs:
        r_pts: list[np.ndarray] = []
        r_uv: list[np.ndarray] = []
        for j in range(k):
            a = float(angs[j])
            ca = float(np.cos(a))
            sa = float(np.sin(a))
            tilt_z = 0.0014 * ca
            med_peak = 0.12 * max(0.0, sa) * max(0.0, -ca)
            x_off = eye_r * rx_m * ca
            almond_pinch = 0.78 + 0.22 * (1.0 - ca * ca)
            z_off = eye_r * rz_m * sa * (1.0 + med_peak) * almond_pinch + tilt_z
            y_extra = 0.0026 * (sa**1.2) if (rname == "crease" and sa > 0.15) else 0.0

            rad_xz2 = x_off * x_off + z_off * z_off
            sphere_r = eye_r + max(0.0, y_pad)
            if rname in ("margin", "crease"):
                y_sphere = -float(np.sqrt(max(0.00001, (sphere_r * 1.12) ** 2 - 0.65 * rad_xz2)))
                if rname == "crease":
                    y_w = _lerp(outer_pts[j, 1], eye_c[1] + y_sphere, 0.56) + y_extra
                else:
                    y_w = eye_c[1] + y_sphere - y_pad * 0.25
            elif rname == "wetline":
                y_sphere = -float(np.sqrt(max(0.00001, (eye_r * 1.00) ** 2 - 0.65 * rad_xz2)))
                y_w = eye_c[1] + y_sphere + 0.0030
            else:
                y_w = eye_c[1] + 0.005

            pt = np.array([eye_c[0] + side_sign * x_off, y_w, eye_c[2] + z_off], dtype=np.float64)
            r_pts.append(pt)
            r_uv.append(_head_uv(pt))
        ring_positions.append(r_pts)
        ring_uvs.append(r_uv)
        ring_tags.append((f"eye_{rname}_{side_tag}", f"eye_{side_tag}", "head", "face"))

    _, created = qb.inset_patch(
        patch_quads,
        ring_positions,
        ring_uvs=ring_uvs,
        ring_tags=ring_tags,
        cap_inner=True,
    )

    upper_lid: list[int] = []
    lower_lid: list[int] = []
    for ring_idx in (0, 1, 2):
        for j, vi in enumerate(created[ring_idx]):
            if np.sin(angs[j]) > 0.12:
                upper_lid.append(vi)
            elif np.sin(angs[j]) < -0.12:
                lower_lid.append(vi)
    qb.tag_verts(f"upper_lid_{side_tag}", upper_lid)
    qb.tag_verts(f"lower_lid_{side_tag}", lower_lid)
    return {
        "crease": created[0],
        "margin": created[1],
        "wetline": created[2],
        "socket": created[3],
        "upper": upper_lid,
        "lower": lower_lid,
    }


def _extrude_mouth_lips(qb: _QuadBuilder, patch_quads: list[int]) -> dict[str, list[int]]:
    """Inset a 4-ring concentric anatomical lip & oral cavity loop curving smoothly into cheeks."""
    outer_loop = qb.extract_patch_boundary(patch_quads)
    k = len(outer_loop)
    outer_pts = np.asarray([qb.verts[v] for v in outer_loop], dtype=np.float64)

    dx = outer_pts[:, 0]
    dz = outer_pts[:, 2] - MOUTH_Z
    angs = np.arctan2(dz, dx)

    # Smooth horseshoe dental barrel curving corners back into the cheeks (zero vertical seam!)
    arch_center_y = -0.036
    arch_r = 0.056

    ring_names = ["vermilion", "lip_crest", "lip_seam", "mouth_inner"]
    ring_positions: list[list[np.ndarray]] = []
    ring_uvs: list[list[np.ndarray]] = []
    ring_tags: list[tuple[str, ...]] = []

    for rname in ring_names:
        r_pts: list[np.ndarray] = []
        r_uv: list[np.ndarray] = []
        for j in range(k):
            a = float(angs[j])
            ca = float(np.cos(a))
            sa = float(np.sin(a))
            abs_ca = abs(ca)
            is_upper = sa >= 0.0

            if rname == "vermilion":
                wx = MOUTH_W * 1.06
                cupid = 0.0022 * np.exp(-(((abs_ca - 0.34) / 0.22) ** 2)) - 0.0016 * np.exp(
                    -((abs_ca / 0.14) ** 2)
                )
                hz = (0.0076 + cupid) if is_upper else 0.0094
                y_proj = 0.0022
            elif rname == "lip_crest":
                wx = MOUTH_W * 0.98
                tubercle = 0.0007 * np.exp(-((abs_ca / 0.18) ** 2))
                hz = (0.0040 - tubercle) if is_upper else 0.0052
                y_proj = 0.0056 if is_upper else 0.0052
            elif rname == "lip_seam":
                wx = MOUTH_W * 0.92
                hz = 0.0005
                y_proj = 0.0014
            else:  # mouth_inner
                wx = MOUTH_W * 0.78
                hz = 0.0016
                y_proj = -0.0110

            corner_fade = float(max(0.0, 1.0 - ca * ca)) ** 0.75
            x_w = wx * ca
            if abs(outer_pts[j, 0]) < 1e-6:
                x_w = 0.0
            z_w = MOUTH_Z + hz * sa * (0.55 + 0.45 * corner_fade)
            y_arch = arch_center_y - float(np.sqrt(max(0.0004, arch_r * arch_r - x_w * x_w)))
            if rname == "vermilion":
                # Blend vermilion outer ring smoothly with surrounding facial plane at corners!
                y_w = (
                    _lerp(outer_pts[j, 1], y_arch, 0.45 + 0.55 * corner_fade) - y_proj * corner_fade
                )
            else:
                y_w = y_arch - y_proj * (0.15 + 0.85 * corner_fade)

            pt = np.array([x_w, y_w, z_w], dtype=np.float64)
            r_pts.append(pt)
            r_uv.append(_head_uv(pt))

        ring_positions.append(r_pts)
        ring_uvs.append(r_uv)
        ring_tags.append((f"mouth_{rname}", "mouth", "head", "face"))

    _, created = qb.inset_patch(
        patch_quads,
        ring_positions,
        ring_uvs=ring_uvs,
        ring_tags=ring_tags,
        cap_inner=True,
    )

    upper_lip: list[int] = []
    lower_lip: list[int] = []
    corners: list[int] = []
    for ring_idx in (0, 1, 2):
        for j, vi in enumerate(created[ring_idx]):
            sa = float(np.sin(angs[j]))
            ca = abs(float(np.cos(angs[j])))
            if ca > 0.88:
                corners.append(vi)
            if sa > 0.10:
                upper_lip.append(vi)
            elif sa < -0.10:
                lower_lip.append(vi)

    qb.tag_verts("upper_lip", upper_lip)
    qb.tag_verts("lower_lip", lower_lip)
    qb.tag_verts("mouth_corners", corners)
    return {
        "vermilion": created[0],
        "crest": created[1],
        "seam": created[2],
        "inner": created[3],
    }


def _extrude_ear(qb: _QuadBuilder, patch_quads: list[int], side_sign: float) -> list[int]:
    """Extrude a sleek swept-back oval anatomical ear (helix, antihelix, concha, soft lobe)."""
    outer_loop = qb.extract_patch_boundary(patch_quads)
    k = len(outer_loop)
    outer_pts = np.asarray([qb.verts[v] for v in outer_loop], dtype=np.float64)
    ear_c = np.mean(outer_pts, axis=0)

    # Local ear axes: swept cleanly backward (~54 deg behind +X) so ear tucks neatly against skull!
    out_dir = _normalize(np.array([side_sign * 0.58, 0.81, 0.04], dtype=np.float64))
    back_dir = _normalize(np.array([side_sign * 0.16, 0.98, 0.0], dtype=np.float64))
    up_dir = np.array([0.0, 0.0, 1.0], dtype=np.float64)

    rel = outer_pts - ear_c[None, :]
    dy_e = np.sum(rel * back_dir[None, :], axis=1)
    dz_e = np.sum(rel * up_dir[None, :], axis=1)
    angs = np.arctan2(dz_e, dy_e)

    # 4 rings shaped into a true rounded anatomical ear oval with recessed concha bowl
    specs = [
        (0.014, 0.025, 0.0035),
        (0.017, 0.029, 0.0125),
        (0.012, 0.021, 0.0065),
        (0.007, 0.012, -0.0025),
    ]
    ring_positions: list[list[np.ndarray]] = []
    ring_uvs: list[list[np.ndarray]] = []
    side_tag = "L" if side_sign > 0 else "R"
    ring_tags: list[tuple[str, ...]] = []

    for ry_e, rz_e, push in specs:
        r_pts: list[np.ndarray] = []
        r_uv: list[np.ndarray] = []
        for j in range(k):
            a = float(angs[j])
            ca = float(np.cos(a))
            sa = float(np.sin(a))
            # Wider rounded upper helix, narrower soft lobe at bottom
            lobe_taper = 0.72 + 0.34 * (0.5 * (sa + 1.0))
            # Front tragus stays flush against cheek; back helix flares smoothly outward
            back_flare = 0.14 + 0.86 * max(0.0, ca) ** 1.15
            # Slight anatomical backward tilt (~12 deg)
            y_loc = ry_e * ca * lobe_taper + 0.0045 * sa + 0.005 * back_flare
            z_loc = rz_e * sa
            pt = ear_c + back_dir * y_loc + up_dir * z_loc + out_dir * (push * back_flare)
            r_pts.append(pt)
            r_uv.append(_head_uv(pt))
        ring_positions.append(r_pts)
        ring_uvs.append(r_uv)
        ring_tags.append((f"ear_{side_tag}", "ears", "head"))

    _, created = qb.inset_patch(
        patch_quads,
        ring_positions,
        ring_uvs=ring_uvs,
        ring_tags=ring_tags,
        cap_inner=True,
    )
    return [vi for ring in created for vi in ring]


def _extrude_arm_and_hand(
    qb: _QuadBuilder,
    patch_quads: list[int],
    side_sign: float,
) -> dict[str, list[int]]:
    """Extrude a clean 10-edge A-pose arm and a 100% pure-quad 5-finger anatomical hand."""
    outer_loop = qb.extract_patch_boundary(patch_quads)
    k = len(outer_loop)
    assert k == 10, f"Expected 10-edge shoulder socket, got {k}"
    outer_pts = np.asarray([qb.verts[v] for v in outer_loop], dtype=np.float64)

    # Canonical A-pose arm skeleton landmarks (matching realistic proportions)
    sh = np.array([side_sign * 0.194, 0.012, 1.385], dtype=np.float64)
    el = np.array([side_sign * 0.365, 0.022, 1.168], dtype=np.float64)
    wr = np.array([side_sign * 0.515, 0.002, 0.968], dtype=np.float64)
    arm_dir = _normalize(wr - sh)

    # Orthonormal basis along arm:
    # u_arm = forward (-Y/palm-thumb direction), v_arm = upward-dorsal perpendicular
    v_arm = _normalize(np.cross(arm_dir, np.array([0.0, -1.0, 0.0]) * side_sign))
    if v_arm[2] < 0.0:
        v_arm = -v_arm
    u_arm = _normalize(np.cross(v_arm, arm_dir))
    if u_arm[1] > 0.0:
        u_arm = -u_arm

    # Align outer_loop start index so vertex 0 is at the anterior/thumb edge (-Y) and cycles CCW
    # around arm_dir! This keeps the 4 knuckles cleanly oriented along the palm!
    c0 = np.mean(outer_pts, axis=0)
    ang0 = np.arctan2(
        np.sum((outer_pts - c0) * v_arm[None, :], axis=1),
        np.sum((outer_pts - c0) * u_arm[None, :], axis=1),
    )

    # Arm stations: (position, rx_front_back, rz_top_bot, tag)
    arm_stations: list[tuple[np.ndarray, float, float, str]] = [
        (sh + arm_dir * 0.025, 0.056, 0.058, "shoulder"),
        (_lerp(sh, el, 0.25), 0.049, 0.050, "upper_arm"),
        (_lerp(sh, el, 0.55), 0.045, 0.046, "upper_arm"),
        (_lerp(sh, el, 0.85), 0.042, 0.043, "upper_arm"),
        (el, 0.040, 0.041, "elbow"),
        (_lerp(el, wr, 0.25), 0.042, 0.041, "forearm"),
        (_lerp(el, wr, 0.55), 0.036, 0.035, "forearm"),
        (_lerp(el, wr, 0.82), 0.029, 0.026, "forearm"),
        (wr, 0.028, 0.021, "wrist"),
        (wr + arm_dir * 0.038, 0.039, 0.015, "hand"),
        (wr + arm_dir * 0.078, 0.041, 0.013, "hand"),
    ]

    side_tag = "L" if side_sign > 0 else "R"
    ring_positions: list[list[np.ndarray]] = []
    ring_uvs: list[list[np.ndarray]] = []
    ring_tags: list[tuple[str, ...]] = []

    # Target canonical angles for the 10-vertex cross-section preserving outer_loop cyclic order
    # Check loop direction around arm_dir
    d_ang = float(np.mean(np.sin(np.roll(ang0, -1) - ang0)))
    step_sign = 1.0 if d_ang >= 0.0 else -1.0
    start_ang = float(ang0[0])
    target_angs = [start_ang + step_sign * (2.0 * np.pi * j / k) for j in range(k)]

    for si, (pos, ru, rv, st_tag) in enumerate(arm_stations):
        r_pts: list[np.ndarray] = []
        r_uv: list[np.ndarray] = []
        # Blend smoothly from exact socket angles into uniform palm angles
        blend = min(1.0, (si + 1) / 3.0)
        for j in range(k):
            d_a = (target_angs[j] - float(ang0[j]) + np.pi) % (2.0 * np.pi) - np.pi
            a = float(ang0[j]) + d_a * blend
            ca = float(np.cos(a))
            sa = float(np.sin(a))
            if st_tag == "hand":
                # Rectangular-rounded palm cross-section (flat dorsal/palmar, rounded sides)
                ca_s = float(np.sign(ca) * (abs(ca) ** 0.78))
                sa_s = float(np.sign(sa) * (abs(sa) ** 0.82))
                pt = pos + u_arm * (ru * ca_s) + v_arm * (rv * sa_s)
            else:
                pt = pos + u_arm * (ru * ca) + v_arm * (rv * sa)
            r_pts.append(pt)
            r_uv.append(_body_uv(pt))
        ring_positions.append(r_pts)
        ring_uvs.append(r_uv)
        ring_tags.append((st_tag, f"arm_{side_tag}", "arms", "body"))

    _, created = qb.inset_patch(
        patch_quads,
        ring_positions,
        ring_uvs=ring_uvs,
        ring_tags=ring_tags,
        cap_inner=False,
    )

    # Cap distal palm ring with 4 pure quads (aligned from thumb-side -> pinky-side!)
    # Find index on palm_end ring with maximum +u_arm (thumb-side anterior edge)
    palm_end = created[-1]
    palm_pts = np.asarray([qb.verts[v] for v in palm_end], dtype=np.float64)
    palm_c = np.mean(palm_pts, axis=0)
    u_proj = np.sum((palm_pts - palm_c) * u_arm[None, :], axis=1)
    thumb_j = int(np.argmax(u_proj))
    # Rotate palm_end cyclically so index 0 is thumb-side edge!
    ordered_palm = [palm_end[(thumb_j + j) % k] for j in range(k)]

    # Build the 4 distal knuckle quads bridging top/bottom of the 10-edge palm loop
    # Ordered indices: [0] = thumb corner, [1,2,3,4] = dorsal knuckles, [5] = pinky corner,
    # [6,7,8,9] = palmar knuckles!
    # Wait: verify cyclic orientation matches cap_even_loop(ordered_palm, reverse=False):
    # i=0: [0, 1, 8, 9], i=1: [1, 2, 7, 8], i=2: [2, 3, 6, 7], i=3: [3, 4, 5, 6]
    knuckle_quads = qb.cap_even_loop(ordered_palm, reverse=False)

    # Extrude 4 separate anatomical fingers (index, middle, ring, pinky) from the 4 knuckle quads!
    finger_specs = [
        ("index", 0.068, 0.0085, 0.09),
        ("middle", 0.076, 0.0088, 0.02),
        ("ring", 0.070, 0.0082, -0.05),
        ("pinky", 0.055, 0.0072, -0.13),
    ]
    # Check ordering: whether knuckle_quads[0] is thumb-side or pinky-side
    q0_c = np.mean([qb.verts[v] for v in qb.quads[knuckle_quads[0]]], axis=0)
    q3_c = np.mean([qb.verts[v] for v in qb.quads[knuckle_quads[3]]], axis=0)
    if float(np.dot(q0_c - q3_c, u_arm)) < 0.0:
        knuckle_quads = list(reversed(knuckle_quads))

    # Determine outward palm normal (pointing toward +arm_dir, away from palm center)
    palm_inward = -v_arm  # Slight natural finger curl toward palm interior

    for f_idx, (fname, flen, frad, splay) in enumerate(finger_specs):
        q_id = knuckle_quads[f_idx]
        b_loop = qb.extract_patch_boundary([q_id])
        b_pts = np.asarray([qb.verts[v] for v in b_loop], dtype=np.float64)
        fc = np.mean(b_pts, axis=0)
        f_dir = _normalize(arm_dir + u_arm * splay + palm_inward * 0.06)

        f_rings: list[list[np.ndarray]] = []
        f_uvs: list[list[np.ndarray]] = []
        f_tags: list[tuple[str, ...]] = []
        # Ring 0: web inset on knuckle surface; Rings 1..3: proximal/middle/distal phalanges
        fracs = [(0.0, 0.78, 0.0), (0.36, 0.74, 0.002), (0.72, 0.64, 0.007), (1.0, 0.44, 0.013)]
        for frac, rad_s, curl in fracs:
            r_pts: list[np.ndarray] = []
            r_uv: list[np.ndarray] = []
            center_i = fc + f_dir * (flen * frac) + palm_inward * curl
            for j in range(4):
                rel_j = b_pts[j] - fc
                # Clamp radial spread smoothly to anatomical finger thickness
                rel_len = float(np.linalg.norm(rel_j))
                target_r = (
                    min(rel_len * rad_s, frad * (1.05 - 0.30 * frac))
                    if frac > 0
                    else rel_len * rad_s
                )
                pt = center_i + (rel_j / max(1e-9, rel_len)) * target_r
                r_pts.append(pt)
                r_uv.append(_body_uv(pt))
            f_rings.append(r_pts)
            f_uvs.append(r_uv)
            f_tags.append((f"finger_{fname}_{side_tag}", "hand", f"arm_{side_tag}", "body"))

        qb.inset_patch([q_id], f_rings, ring_uvs=f_uvs, ring_tags=f_tags, cap_inner=True)

    # Extrude opposable Thumb from the lateral thumb-side quad between wrist and mid-palm!
    palm_ring_mid = created[-2]
    palm_ring_w = created[-3]
    # Find the exact quad connecting (palm_ring_w[thumb_j], palm_ring_mid[thumb_j])
    thumb_quad_id: int | None = None
    tv_a = palm_ring_w[thumb_j]
    tv_b = palm_ring_mid[thumb_j]
    for qi, q in enumerate(qb.quads):
        if len(q) == 4 and tv_a in q and tv_b in q:
            thumb_quad_id = qi
            break

    if thumb_quad_id is not None:
        tb_loop = qb.extract_patch_boundary([thumb_quad_id])
        tb_pts = np.asarray([qb.verts[v] for v in tb_loop], dtype=np.float64)
        tc = np.mean(tb_pts, axis=0)
        t_dir = _normalize(arm_dir * 0.62 + u_arm * 0.72 + palm_inward * 0.30)
        t_len = 0.056
        t_rings: list[list[np.ndarray]] = []
        t_uvs: list[list[np.ndarray]] = []
        t_tags: list[tuple[str, ...]] = []
        for frac, rad_s, curl in [(0.0, 0.76, 0.0), (0.50, 0.68, 0.003), (1.0, 0.44, 0.008)]:
            r_pts_t: list[np.ndarray] = []
            r_uv_t: list[np.ndarray] = []
            ci = tc + t_dir * (t_len * frac) + palm_inward * curl
            for j in range(4):
                rj = tb_pts[j] - tc
                rl = float(np.linalg.norm(rj))
                tr = min(rl * rad_s, 0.0098 * (1.05 - 0.28 * frac)) if frac > 0 else rl * rad_s
                pt = ci + (rj / max(1e-9, rl)) * tr
                r_pts_t.append(pt)
                r_uv_t.append(_body_uv(pt))
            t_rings.append(r_pts_t)
            t_uvs.append(r_uv_t)
            t_tags.append((f"thumb_{side_tag}", "hand", f"arm_{side_tag}", "body"))
        qb.inset_patch([thumb_quad_id], t_rings, ring_uvs=t_uvs, ring_tags=t_tags, cap_inner=True)

    return {"arm_rings": [vi for ring in created for vi in ring]}


def _extrude_leg_and_foot(
    qb: _QuadBuilder,
    hip_loop: list[int],
    side_sign: float,
) -> list[int]:
    """Extrude a 14-edge anatomical leg and arched foot resting flat at Z = 0.0."""
    k = len(hip_loop)
    assert k == 14, f"Expected 14-edge leg loop, got {k}"
    hip_pts = np.asarray([qb.verts[v] for v in hip_loop], dtype=np.float64)
    c_hip = np.mean(hip_pts, axis=0)
    ang0 = np.arctan2(hip_pts[:, 0] - c_hip[0], -(hip_pts[:, 1] - c_hip[1]))
    d_ang = float(np.mean(np.sin(np.roll(ang0, -1) - ang0)))
    step_sign = 1.0 if d_ang >= 0.0 else -1.0
    target_angs = [float(ang0[0]) + step_sign * (2.0 * np.pi * j / k) for j in range(k)]

    # Anatomical leg stations: (z, cx, cy, rx, ry_front, ry_back, tag)
    leg_stations: list[tuple[float, float, float, float, float, float, str]] = [
        (0.810, side_sign * 0.095, 0.006, 0.084, 0.086, 0.094, "thigh"),
        (0.715, side_sign * 0.093, 0.004, 0.077, 0.080, 0.084, "thigh"),
        (0.610, side_sign * 0.091, 0.002, 0.068, 0.070, 0.072, "thigh"),
        (0.505, side_sign * 0.089, 0.004, 0.058, 0.060, 0.058, "knee"),
        (0.440, side_sign * 0.089, 0.010, 0.054, 0.054, 0.058, "calf"),
        (0.355, side_sign * 0.089, 0.016, 0.057, 0.054, 0.066, "calf"),
        (0.245, side_sign * 0.089, 0.012, 0.047, 0.046, 0.050, "calf"),
        (0.145, side_sign * 0.089, 0.008, 0.039, 0.040, 0.042, "ankle"),
        (0.082, side_sign * 0.089, 0.008, 0.038, 0.044, 0.045, "ankle"),
    ]

    side_tag = "L" if side_sign > 0 else "R"
    prev_loop = hip_loop
    created_all: list[int] = []

    for si, (z, cx, cy, rx, ryf, ryb, tag) in enumerate(leg_stations):
        blend = min(1.0, (si + 1) / 3.0)
        cur_ring: list[int] = []
        for j in range(k):
            d_a = (target_angs[j] - float(ang0[j]) + np.pi) % (2.0 * np.pi) - np.pi
            a = float(ang0[j]) + d_a * blend
            sa = float(np.sin(a))
            ca = float(np.cos(a))
            pt = np.array(
                [cx + sa * rx, cy - ca * (ryf if ca >= 0.0 else ryb), z],
                dtype=np.float64,
            )
            vi = qb.add_vert(pt, _body_uv(pt), tags=(tag, f"leg_{side_tag}", "legs", "body"))
            cur_ring.append(vi)
        qb.bridge_loops(prev_loop, cur_ring)
        prev_loop = cur_ring
        created_all.extend(cur_ring)

    # Foot: smoothly sweep the 14-edge ankle loop forward into heel -> arch -> forefoot -> toe cap!
    # Keep sole vertices strictly at Z = 0.0 so the character stands planted on the floor!
    foot_rings = [
        # (z_bot, z_top, y_front, y_back, cx_off, rx)
        (0.036, 0.072, -0.075, 0.058, 0.0, 0.041),  # Instep / heel bend
        (0.002, 0.056, -0.125, 0.056, 0.002, 0.044),  # Midfoot / arch
        (0.000, 0.042, -0.172, 0.040, 0.003, 0.046),  # Ball of foot
        (0.000, 0.028, -0.206, -0.020, 0.002, 0.042),  # Forefoot
        (0.004, 0.018, -0.222, -0.145, -0.002, 0.031),  # Rounded toe tip
    ]

    for zb, zt, yf, yb, cx_o, rx_f in foot_rings:
        cur_ring = []
        for j in range(k):
            a = target_angs[j]
            sa = float(np.sin(a))
            ca = float(np.cos(a))
            # Map front half (ca > 0) to dorsum/toes and back half (ca < 0) to sole/heel!
            # Wait: around ankle, ca=+1 is front (-Y), ca=-1 is back (+Y)!
            # Keep same orientation around the ring so quads never twist!
            # Front vertices move forward to yf & arch up; back vertices stay at heel & drop!
            t_fb = 0.5 * (ca + 1.0)  # 0 at back (+Y), 1 at front (-Y)
            y_p = _lerp(yb, yf, t_fb)
            # Rounded cross-section: flat on bottom/back-heel, domed on top/front-instep
            dome = float(np.sqrt(max(0.0, 1.0 - sa * sa)))
            z_p = zb + (zt - zb) * (0.15 + 0.85 * dome) * (0.20 + 0.80 * t_fb)
            if t_fb < 0.35:
                z_p = zb
            x_p = side_sign * (0.089 + cx_o) + sa * rx_f * (0.88 + 0.12 * t_fb)
            pt = np.array([x_p, y_p, z_p], dtype=np.float64)
            vi = qb.add_vert(pt, _body_uv(pt), tags=("foot", f"leg_{side_tag}", "legs", "body"))
            cur_ring.append(vi)
        qb.bridge_loops(prev_loop, cur_ring)
        prev_loop = cur_ring
        created_all.extend(cur_ring)

    # Pure-quad closure across sole/toe end!
    qb.cap_even_loop(prev_loop, reverse=False)
    return created_all


# ---------------------------------------------------------------------------
# Sparse Cached Catmull-Clark Operator (Exact Match to bmesh.py Rules!)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SubdivLevelCache:
    """Precomputed sparse linear Catmull-Clark operator & triangulated topology."""

    levels: int
    matrix: sparse.csr_matrix  # (N_sub, N_base) linear subdivision operator
    uvs: np.ndarray  # (N_sub, 2)
    faces: np.ndarray  # (F_sub, 3) triangulated outward faces
    quad_count: int


def _catmull_clark_sparse_step(
    n_verts: int,
    quads: np.ndarray,
) -> tuple[sparse.csr_matrix, np.ndarray]:
    """Compute one exact Catmull-Clark step as a sparse matrix `(M, new_quads)` for closed quads."""
    n_q = len(quads)
    edge_to_quads: dict[tuple[int, int], list[int]] = defaultdict(list)
    vert_to_quads: list[list[int]] = [[] for _ in range(n_verts)]
    vert_neighbors: list[set[int]] = [set() for _ in range(n_verts)]

    for qi in range(n_q):
        q = quads[qi]
        for j in range(4):
            u = int(q[j])
            v = int(q[(j + 1) % 4])
            vert_to_quads[u].append(qi)
            vert_neighbors[u].add(v)
            vert_neighbors[v].add(u)
            edge_to_quads[(min(u, v), max(u, v))].append(qi)

    edge_keys = list(edge_to_quads.keys())
    n_e = len(edge_keys)
    edge_idx = {ek: n_verts + n_q + ei for ei, ek in enumerate(edge_keys)}

    rows: list[int] = []
    cols: list[int] = []
    vals: list[float] = []

    def _add(r: int, c: int, w: float) -> None:
        rows.append(r)
        cols.append(c)
        vals.append(w)

    # 1) Face points: row = n_verts + qi = 1/4 * sum(quad verts)
    for qi in range(n_q):
        r = n_verts + qi
        for v in quads[qi]:
            _add(r, int(v), 0.25)

    # 2) Edge points: row = n_verts + n_q + ei
    for ei, ek in enumerate(edge_keys):
        r = n_verts + n_q + ei
        u, v = ek
        inc = edge_to_quads[ek]
        if len(inc) == 2:
            # 0.25*(u + v + f0 + f1) = 0.25*u + 0.25*v + 0.0625*sum(q0) + 0.0625*sum(q1)
            acc: dict[int, float] = defaultdict(float)
            acc[u] += 0.25
            acc[v] += 0.25
            for qi in inc:
                for qv in quads[qi]:
                    acc[int(qv)] += 0.0625
            for c, w in acc.items():
                _add(r, c, w)
        else:
            _add(r, u, 0.5)
            _add(r, v, 0.5)

    # 3) Updated original vertices: row = vi
    for vi in range(n_verts):
        inc_q = vert_to_quads[vi]
        nbrs = vert_neighbors[vi]
        n = len(inc_q)
        if n < 3 or len(nbrs) < 3:
            _add(vi, vi, 1.0)
            continue
        # Check boundary edges
        bnd_nbrs = [nb for nb in nbrs if len(edge_to_quads[(min(vi, nb), max(vi, nb))]) != 2]
        if len(bnd_nbrs) == 2:
            _add(vi, vi, 0.75)
            _add(vi, bnd_nbrs[0], 0.125)
            _add(vi, bnd_nbrs[1], 0.125)
        elif len(bnd_nbrs) > 0:
            _add(vi, vi, 1.0)
        else:
            # Smooth interior Catmull-Clark rule: (Q + 2R + (n-3)V) / n
            # Q = (1/n) * sum_{q in inc} (1/4 * sum(q))
            # R = (1/n) * sum_{nb in nbrs} (0.5*V + 0.5*nb)
            acc = defaultdict(float)
            inv_n = 1.0 / float(n)
            acc[vi] += (n - 3.0) * inv_n
            # R term: (2/n) * (1/len(nbrs)) * sum(0.5*V + 0.5*nb)
            r_scale = (2.0 * inv_n) / float(len(nbrs))
            for nb in nbrs:
                acc[vi] += 0.5 * r_scale
                acc[nb] += 0.5 * r_scale
            # Q term: (1/n^2) * sum_{q} (0.25 * sum(q))
            q_scale = (inv_n * inv_n) * 0.25
            for qi in inc_q:
                for qv in quads[qi]:
                    acc[int(qv)] += q_scale
            for c, w in acc.items():
                _add(vi, c, w)

    n_out = n_verts + n_q + n_e
    mat = sparse.coo_matrix((vals, (rows, cols)), shape=(n_out, n_verts)).tocsr()

    # 4) Subdivided pure-quad faces
    new_quads = np.empty((n_q * 4, 4), dtype=np.int64)
    out_i = 0
    for qi in range(n_q):
        q = quads[qi]
        fp = n_verts + qi
        for j in range(4):
            v = int(q[j])
            v_next = int(q[(j + 1) % 4])
            v_prev = int(q[(j - 1) % 4])
            e_next = edge_idx[(min(v, v_next), max(v, v_next))]
            e_prev = edge_idx[(min(v_prev, v), max(v_prev, v))]
            new_quads[out_i] = (v, e_next, fp, e_prev)
            out_i += 1

    return mat, new_quads


def _triangulate_quads_array(verts: np.ndarray, quads: np.ndarray) -> np.ndarray:
    """Vectorized shorter-diagonal triangulation of pure quads."""
    a = quads[:, 0]
    b = quads[:, 1]
    c = quads[:, 2]
    d = quads[:, 3]
    d_ac = np.sum((verts[a] - verts[c]) ** 2, axis=1)
    d_bd = np.sum((verts[b] - verts[d]) ** 2, axis=1)
    use_ac = d_ac <= d_bd
    t1 = np.where(use_ac[:, None], np.stack([a, b, c], axis=1), np.stack([a, b, d], axis=1))
    t2 = np.where(use_ac[:, None], np.stack([a, c, d], axis=1), np.stack([b, c, d], axis=1))
    return np.vstack([t1, t2]).astype(np.int64)


# ---------------------------------------------------------------------------
# Canonical Quad Base Data Container & Builder
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CanonicalQuadBase:
    """Immutable canonical neutral realistic human quad cage + cached subdivision operators."""

    verts: np.ndarray  # (N_base, 3)
    uvs: np.ndarray  # (N_base, 2)
    quads: np.ndarray  # (Q_base, 4)
    tags: dict[str, np.ndarray]  # semantic vertex index arrays
    subdiv_cache: dict[int, SubdivLevelCache]  # level -> SubdivLevelCache

    def subdivide(self, deformed_verts: np.ndarray, levels: int = 2) -> MeshData:
        """Apply exact cached Catmull-Clark subdivision in ~2ms using sparse matrix multiply."""
        lv = max(1, min(2, int(levels)))
        cache = self.subdiv_cache[lv]
        sub_v = cache.matrix @ deformed_verts
        return MeshData(vertices=sub_v, faces=cache.faces.copy(), uvs=cache.uvs.copy())


@lru_cache(maxsize=1)
def build_canonical_quad_base() -> CanonicalQuadBase:
    """Construct and cache the 100% manifold pure-quad realistic human base mesh."""
    qb = _QuadBuilder()
    n_seg = 28

    # Non-uniform angular spacing around head/body circumference concentrates columns on the face!
    # k = 0 is front midline (-Y), k = 14 is back midline (+Y), symmetric left/right!
    half_angles = np.array(
        [
            0.000,  # 0: front midline (nose ridge / philtrum / chin)
            0.092,  # 1: nasal wall / inner eye / Cupid's bow peak
            0.205,  # 2: medial eye / nose ala / mid-lip
            0.340,  # 3: eye center / mouth corner
            0.495,  # 4: outer canthus / cheekbone
            0.690,  # 5: temple / lateral cheek / jaw angle
            0.940,  # 6: pre-ear / front shoulder
            1.230,  # 7: ear center / outer shoulder
            1.530,  # 8: behind ear / rear shoulder
            1.830,  # 9: lateral occiput / scapula
            2.120,  # 10: rear-side head / back
            2.400,  # 11: occiput
            2.660,  # 12: back
            2.905,  # 13: near back midline
            np.pi,  # 14: back midline (+Y)
        ],
        dtype=np.float64,
    )
    thetas = np.concatenate([half_angles, -half_angles[-2:0:-1]])
    assert len(thetas) == n_seg

    # Vertical Z stations from pelvis (0.865m) continuously up through torso, neck, face, crown!
    torso_z = [
        0.865,  # 0: pelvis bottom (crotch split)
        0.915,  # 1: hips
        0.975,  # 2: lower abdomen
        1.045,  # 3: waist
        1.135,  # 4: lower ribs
        1.225,  # 5: mid ribs
        1.295,  # 6: chest apex (arm socket bottom)
        1.345,  # 7: upper chest (arm socket mid)
        1.392,  # 8: deltoid/clavicle (arm socket top)
        1.430,  # 9: shoulder slope
        1.462,  # 10: neck base
    ]
    head_z = [
        1.484,  # 11: mid neck
        1.504,  # 12: upper neck / chin base
        1.522,  # 13: chin / jawline
        1.539,  # 14: mouth bottom (mouth patch bottom)
        1.553,  # 15: mouth midline (mouth patch center)
        1.568,  # 16: upper lip / philtrum (mouth patch top)
        1.585,  # 17: nose base / alar wings
        1.598,  # 18: infraorbital cheek (eye patch bottom)
        1.617,  # 19: eye center (eye patch center)
        1.636,  # 20: brow ridge (eye patch top)
        1.659,  # 21: lower forehead
        1.686,  # 22: upper forehead
        1.711,  # 23: frontal boss / crown slope
        1.730,  # 24: upper crown ring
    ]
    all_z = torso_z + head_z
    n_rows = len(all_z)
    head_start_row = len(torso_z)

    # 1) Create unified vertical ring vertices
    rings: list[list[int]] = []
    for ri, z in enumerate(all_z):
        row_v: list[int] = []
        is_head = ri >= head_start_row
        for ki in range(n_seg):
            th = float(thetas[ki])
            if is_head:
                pt = _eval_head_ring_point(th, z)
                uv = _head_uv(pt)
                tgs: tuple[str, ...] = ("head", "face" if abs(th) < 1.35 else "cranium")
            else:
                pt = _eval_torso_ring_point(th, z)
                uv = _body_uv(pt)
                tgs = ("torso", "body")
            # Enforce exact X = 0 on symmetry columns (k=0 and k=14)
            if ki in (0, 14):
                pt[0] = 0.0
            row_v.append(qb.add_vert(pt, uv, tags=tgs))
        rings.append(row_v)

    # 2) Stitch continuous torso + neck + head rings into pure quads (track grid quad IDs!)
    # grid_quad[ri][ki] is the quad between ring ri..ri+1 and col ki..(ki+1)%n_seg
    grid_quads: list[list[int]] = []
    for ri in range(n_rows - 1):
        row_q: list[int] = []
        for ki in range(n_seg):
            kn = (ki + 1) % n_seg
            # CCW outward viewed from outside: (ri, ki) -> (ri, kn) -> (ri+1, kn) -> (ri+1, ki)
            # Orientation: tangent_k (+X at front -Y) x tangent_z (+Z) = -Y (outward)
            qi = qb.add_quad(rings[ri][ki], rings[ri][kn], rings[ri + 1][kn], rings[ri + 1][ki])
            row_q.append(qi)
        grid_quads.append(row_q)

    # 3) Close crown top ring (row = n_rows - 1) with pure-quad cap lifted gently to Z = 1.740m
    crown_ring = rings[-1]
    crown_cap = qb.cap_even_loop(crown_ring, reverse=False)
    # Add one inner crown ring before cap so dome rounds smoothly to 1.740m
    for qi in crown_cap:
        qb.quads[qi] = []
    crown_inner: list[int] = []
    for ki in range(n_seg):
        p_out = qb.verts[crown_ring[ki]]
        pt = np.array([p_out[0] * 0.48, p_out[1] * 0.48 + 0.002, 1.739], dtype=np.float64)
        crown_inner.append(qb.add_vert(pt, _head_uv(pt), tags=("head", "cranium")))
    qb.bridge_loops(crown_ring, crown_inner)
    qb.cap_even_loop(crown_inner, reverse=False)

    # 4) Inset Left & Right Eye Concentric Edge-Loops (rows 18..19, cols 1..3 & 25..27)
    # Left eye: cols 1, 2, 3 (3 quads wide) x rows 18, 19 (2 quads tall) -> 10-edge loop!
    eye_l_patch = [grid_quads[r][c] for r in (18, 19) for c in (1, 2, 3)]
    eye_r_patch = [grid_quads[r][c] for r in (18, 19) for c in (24, 25, 26)]
    _extrude_eye_socket(qb, eye_l_patch, side_sign=1.0)
    _extrude_eye_socket(qb, eye_r_patch, side_sign=-1.0)

    # 5) Inset Mouth & Lips Concentric Loop (rows 14..15, cols 25..27, 0..2 -> 6x2 = 16-edge)
    mouth_patch = [grid_quads[r][c] for r in (14, 15) for c in (25, 26, 27, 0, 1, 2)]
    _extrude_mouth_lips(qb, mouth_patch)

    # 6) Extrude Left & Right Ears (rows 16..18, cols 6..7 & 20..21 -> 2x3 = 10-edge loop!)
    ear_l_patch = [grid_quads[r][c] for r in (16, 17, 18) for c in (6, 7)]
    ear_r_patch = [grid_quads[r][c] for r in (16, 17, 18) for c in (20, 21)]
    _extrude_ear(qb, ear_l_patch, side_sign=1.0)
    _extrude_ear(qb, ear_r_patch, side_sign=-1.0)

    # 7) Extrude Left & Right Arms + 5-Finger Hands (rows 6..8, cols 6..7 & 20..21 -> 10-edge)
    arm_l_patch = [grid_quads[r][c] for r in (6, 7, 8) for c in (6, 7)]
    arm_r_patch = [grid_quads[r][c] for r in (6, 7, 8) for c in (20, 21)]
    _extrude_arm_and_hand(qb, arm_l_patch, side_sign=1.0)
    _extrude_arm_and_hand(qb, arm_r_patch, side_sign=-1.0)

    # 8) Split Pelvis Bottom Ring (row 0, 28 edges) via 4-quad Perineum Bridge into Two Legs
    pelv = rings[0]
    z_crotch = 0.838
    m_0 = qb.add_vert(
        (0.0, 0.010, z_crotch), _body_uv(np.array([0.0, 0.01, z_crotch])), ("pelvis", "body")
    )
    m_l = qb.add_vert(
        (0.018, 0.010, z_crotch), _body_uv(np.array([0.018, 0.01, z_crotch])), ("pelvis", "body")
    )
    m_r = qb.add_vert(
        (-0.018, 0.010, z_crotch), _body_uv(np.array([-0.018, 0.01, z_crotch])), ("pelvis", "body")
    )

    # Winding check: torso bottom ring edges run (pelv[k] -> pelv[k+1]) at bottom boundary.
    # For outward downward-facing perineum quads, traverse pelv[k+1] -> pelv[k]!
    qb.add_quad(pelv[1], pelv[0], m_0, m_l)
    qb.add_quad(pelv[0], pelv[27], m_r, m_0)
    qb.add_quad(m_l, m_0, pelv[14], pelv[13])
    qb.add_quad(m_0, m_r, pelv[15], pelv[14])

    # Left leg open loop (14 verts, oriented matching downward extrusion!)
    # Remaining boundary edges on Left leg hole: (pelv[2]->pelv[1]), ..., (pelv[13]->pelv[12]),
    # plus (m_l->pelv[1]) & (pelv[13]->m_l) -> ordered CCW around -Z: [pelv[13], ..., pelv[1], m_l]
    left_hip_loop = [pelv[k] for k in range(13, 0, -1)] + [m_l]
    right_hip_loop = [pelv[k] for k in range(27, 14, -1)] + [m_r]
    _extrude_leg_and_foot(qb, left_hip_loop, side_sign=1.0)
    _extrude_leg_and_foot(qb, right_hip_loop, side_sign=-1.0)

    # Tag key landmark vertex groups for automatic landmark & brow/nose/chin tracking!
    qb.tag_verts("chin_tip", [rings[13][0], rings[12][0]])
    qb.tag_verts("nose_bridge_v", [rings[19][0], rings[18][0]])
    qb.tag_verts("nose_tip_v", [rings[17][0]])
    qb.tag_verts("brow_L", [rings[20][1], rings[20][2], rings[20][3], rings[20][4]])
    qb.tag_verts("brow_R", [rings[20][27], rings[20][26], rings[20][25], rings[20][24]])
    qb.tag_verts("neck_base_ring", rings[10])
    qb.tag_verts("neck_top_ring", rings[12])
    qb.tag_verts("waist_ring", rings[3])
    qb.tag_verts("chest_ring", rings[6])
    qb.tag_verts("pelvis_ring", rings[0])
    qb.tag_verts("crown_ring", crown_inner)

    verts, uvs, quads, tags = qb.finalize()

    # Verify global outward winding (exact closed-mesh signed volume > 0) and flip if needed
    c0 = np.mean(verts, axis=0)
    a_v = verts[quads[:, 0]] - c0
    b_v = verts[quads[:, 1]] - c0
    c_v = verts[quads[:, 2]] - c0
    d_v = verts[quads[:, 3]] - c0
    signed_vol = float(
        np.sum(
            np.einsum("ij,ij->i", a_v, np.cross(b_v, c_v))
            + np.einsum("ij,ij->i", a_v, np.cross(c_v, d_v))
        )
    )
    if signed_vol < 0.0:
        quads = quads[:, [0, 3, 2, 1]]

    # Precompute exact sparse Catmull-Clark operators for levels 1 and 2
    subdiv_cache: dict[int, SubdivLevelCache] = {}
    cur_mat = sparse.eye(len(verts), format="csr", dtype=np.float64)
    cur_quads = quads.copy()
    for lv in (1, 2):
        step_mat, cur_quads = _catmull_clark_sparse_step(cur_mat.shape[0], cur_quads)
        cur_mat = (step_mat @ cur_mat).tocsr()
        sub_v = cur_mat @ verts
        sub_uv = cur_mat @ uvs
        sub_f = _triangulate_quads_array(sub_v, cur_quads)
        subdiv_cache[lv] = SubdivLevelCache(
            levels=lv,
            matrix=cur_mat,
            uvs=sub_uv,
            faces=sub_f,
            quad_count=len(cur_quads),
        )

    return CanonicalQuadBase(
        verts=verts,
        uvs=uvs,
        quads=quads,
        tags=tags,
        subdiv_cache=subdiv_cache,
    )


def subdivide_quad_mesh(
    verts: np.ndarray,
    quads: np.ndarray,
    uvs: np.ndarray | None = None,
    levels: int = 2,
) -> MeshData:
    """Apply ``levels`` iterations of sparse Catmull-Clark subdivision to a quad mesh."""
    v = np.asarray(verts, dtype=np.float64)
    q = np.asarray(quads, dtype=np.int32)
    uv = np.asarray(uvs, dtype=np.float64) if uvs is not None else None
    for _ in range(max(0, int(levels))):
        step_mat, q = _catmull_clark_sparse_step(len(v), q)
        v = step_mat @ v
        if uv is not None:
            uv = step_mat @ uv
    faces = _triangulate_quads_array(v, q)
    return MeshData(vertices=v, faces=faces, uvs=uv)
