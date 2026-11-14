"""Pure-CPU BVH-accelerated Monte Carlo Ray / Path Tracer backend (`raytrace`)."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from PIL import Image

from frfruit.core.scene import LightSpec, MaterialSpec, SceneDocument, SceneObject
from frfruit.geometry.spatial import EvaluatedObject
from frfruit.render.camera import build_camera_matrices
from frfruit.render.materials import evaluate_ggx_brdf, evaluate_material_maps
from frfruit.render.postprocess import apply_post_processing
from frfruit.render.sky import evaluate_world_sky
from frfruit.render.software import RenderConfig, _draw_segment_z


@dataclass
class _BVHNode:
    bmin: np.ndarray
    bmax: np.ndarray
    left: _BVHNode | None = None
    right: _BVHNode | None = None
    start: int = 0
    end: int = 0


class _TriangleBVH:
    """Median-split AABB BVH with reordered contiguous triangle arrays for NumPy packets."""

    def __init__(self, verts: np.ndarray, faces: np.ndarray, leaf_size: int = 16) -> None:
        self.count = len(faces)
        if self.count == 0:
            self.root: _BVHNode | None = None
            self.orig_idx = np.zeros((0,), dtype=np.int32)
            self.v0 = np.zeros((0, 3), dtype=np.float64)
            self.v1 = np.zeros((0, 3), dtype=np.float64)
            self.v2 = np.zeros((0, 3), dtype=np.float64)
            self.e1 = np.zeros((0, 3), dtype=np.float64)
            self.e2 = np.zeros((0, 3), dtype=np.float64)
            return

        v0 = verts[faces[:, 0]]
        v1 = verts[faces[:, 1]]
        v2 = verts[faces[:, 2]]
        tri_min = np.minimum(v0, np.minimum(v1, v2)) - 1e-6
        tri_max = np.maximum(v0, np.maximum(v1, v2)) + 1e-6
        tri_cent = (v0 + v1 + v2) * (1.0 / 3.0)

        order = np.arange(self.count, dtype=np.int32)
        self.leaf_size = max(4, int(leaf_size))
        self.root = self._build(order, 0, self.count, tri_min, tri_max, tri_cent, depth=0)

        self.orig_idx = order
        self.v0 = np.ascontiguousarray(v0[order])
        self.v1 = np.ascontiguousarray(v1[order])
        self.v2 = np.ascontiguousarray(v2[order])
        self.e1 = np.ascontiguousarray(self.v1 - self.v0)
        self.e2 = np.ascontiguousarray(self.v2 - self.v0)

    def _build(
        self,
        order: np.ndarray,
        start: int,
        end: int,
        tri_min: np.ndarray,
        tri_max: np.ndarray,
        tri_cent: np.ndarray,
        depth: int,
    ) -> _BVHNode:
        sl = order[start:end]
        bmin = np.min(tri_min[sl], axis=0)
        bmax = np.max(tri_max[sl], axis=0)

        if (end - start) <= self.leaf_size or depth >= 22:
            return _BVHNode(bmin=bmin, bmax=bmax, start=start, end=end)

        cents = tri_cent[sl]
        c_span = np.max(cents, axis=0) - np.min(cents, axis=0)
        axis = int(np.argmax(c_span))
        if c_span[axis] < 1e-8:
            return _BVHNode(bmin=bmin, bmax=bmax, start=start, end=end)

        mid = (start + end) // 2
        part = np.argpartition(cents[:, axis], mid - start)
        order[start:end] = sl[part]

        left = self._build(order, start, mid, tri_min, tri_max, tri_cent, depth + 1)
        right = self._build(order, mid, end, tri_min, tri_max, tri_cent, depth + 1)
        return _BVHNode(bmin=bmin, bmax=bmax, left=left, right=right)

    def intersect_closest(
        self,
        origins: np.ndarray,
        directions: np.ndarray,
        t_max_init: float = 1e9,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Vectorized packet closest-hit query: returns (best_t, tri_idx, u, v)."""
        n_rays = len(origins)
        best_t = np.full((n_rays,), t_max_init, dtype=np.float64)
        best_tri = np.full((n_rays,), -1, dtype=np.int32)
        best_u = np.zeros((n_rays,), dtype=np.float64)
        best_v = np.zeros((n_rays,), dtype=np.float64)

        if self.root is None or n_rays == 0:
            return best_t, best_tri, best_u, best_v

        inv_d = np.where(
            np.abs(directions) > 1e-12,
            1.0 / np.where(np.abs(directions) > 1e-12, directions, 1.0),
            1e12 * np.where(directions >= 0.0, 1.0, -1.0),
        )

        # Traverse in spatially coherent 4096-ray chunks so working memory stays in cache
        chunk = 4096
        for s_idx in range(0, n_rays, chunk):
            e_idx = min(n_rays, s_idx + chunk)
            idx = np.arange(s_idx, e_idx, dtype=np.int32)
            self._traverse_closest(
                self.root,
                idx,
                origins,
                directions,
                inv_d,
                best_t,
                best_tri,
                best_u,
                best_v,
            )

        return best_t, best_tri, best_u, best_v

    def _hit_box_mask(
        self,
        node: _BVHNode,
        idx: np.ndarray,
        origins: np.ndarray,
        inv_d: np.ndarray,
        best_t: np.ndarray,
    ) -> np.ndarray:
        t1 = (node.bmin[None, :] - origins[idx]) * inv_d[idx]
        t2 = (node.bmax[None, :] - origins[idx]) * inv_d[idx]
        tmin = np.max(np.minimum(t1, t2), axis=1)
        tmax = np.min(np.maximum(t1, t2), axis=1)
        return (tmax >= np.maximum(0.0, tmin)) & (tmin <= best_t[idx])

    def _traverse_closest(
        self,
        node: _BVHNode,
        idx: np.ndarray,
        origins: np.ndarray,
        directions: np.ndarray,
        inv_d: np.ndarray,
        best_t: np.ndarray,
        best_tri: np.ndarray,
        best_u: np.ndarray,
        best_v: np.ndarray,
    ) -> None:
        mask = self._hit_box_mask(node, idx, origins, inv_d, best_t)
        act = idx[mask]
        if len(act) == 0:
            return

        if node.left is None and node.right is None:
            s, e = node.start, node.end
            ro = origins[act][:, None, :]  # (A, 1, 3)
            rd = directions[act][:, None, :]  # (A, 1, 3)
            lv0 = self.v0[s:e][None, :, :]  # (1, C, 3)
            le1 = self.e1[s:e][None, :, :]
            le2 = self.e2[s:e][None, :, :]

            h = np.cross(rd, le2)  # (A, C, 3)
            a = np.sum(le1 * h, axis=2)  # (A, C)
            eps = 1e-7
            valid = np.abs(a) > eps
            inv_a = np.where(valid, 1.0 / np.where(valid, a, 1.0), 0.0)

            s_vec = ro - lv0
            u = inv_a * np.sum(s_vec * h, axis=2)
            valid &= (u >= -eps) & (u <= 1.0 + eps)

            q = np.cross(s_vec, le1)
            v = inv_a * np.sum(rd * q, axis=2)
            valid &= (v >= -eps) & (u + v <= 1.0 + eps)

            t = inv_a * np.sum(le2 * q, axis=2)
            valid &= (t > 1e-5) & (t < best_t[act, None])

            if not np.any(valid):
                return

            t_cand = np.where(valid, t, np.inf)
            local_k = np.argmin(t_cand, axis=1)
            min_t = t_cand[np.arange(len(act)), local_k]
            improved = np.isfinite(min_t) & (min_t < best_t[act])
            if np.any(improved):
                g_idx = act[improved]
                k_sel = local_k[improved]
                best_t[g_idx] = min_t[improved]
                best_tri[g_idx] = self.orig_idx[s + k_sel]
                best_u[g_idx] = u[improved, k_sel]
                best_v[g_idx] = v[improved, k_sel]
            return

        if node.left is not None:
            self._traverse_closest(
                node.left, act, origins, directions, inv_d, best_t, best_tri, best_u, best_v
            )
        if node.right is not None:
            self._traverse_closest(
                node.right, act, origins, directions, inv_d, best_t, best_tri, best_u, best_v
            )

    def intersect_occluded(
        self,
        origins: np.ndarray,
        directions: np.ndarray,
        max_dist: np.ndarray,
    ) -> np.ndarray:
        """Return boolean occlusion array (N,) checking whether a hit exists in (1e-4, max_dist)."""
        n_rays = len(origins)
        if self.root is None or n_rays == 0:
            return np.zeros((n_rays,), dtype=bool)
        best_t, best_tri, _, _ = self.intersect_closest(origins, directions, t_max_init=1e9)
        return (best_tri >= 0) & (best_t < (max_dist - 1e-3))


def _cosine_hemisphere(normals: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Sample cosine-weighted hemisphere directions around (N, 3) normals."""
    n = len(normals)
    r1 = rng.random(n)
    r2 = rng.random(n)
    phi = 2.0 * math.pi * r1
    r = np.sqrt(r2)
    z = np.sqrt(np.maximum(0.0, 1.0 - r2))
    local = np.stack([r * np.cos(phi), r * np.sin(phi), z], axis=1)

    up_ref = np.where(
        np.abs(normals[:, 2:3]) > 0.92,
        np.array([[1.0, 0.0, 0.0]]),
        np.array([[0.0, 0.0, 1.0]]),
    )
    t_vec = np.cross(up_ref, normals)
    t_len = np.linalg.norm(t_vec, axis=1, keepdims=True)
    t_vec = t_vec / np.where(t_len < 1e-8, 1.0, t_len)
    b_vec = np.cross(normals, t_vec)

    world_d = t_vec * local[:, 0:1] + b_vec * local[:, 1:2] + normals * local[:, 2:3]
    w_len = np.linalg.norm(world_d, axis=1, keepdims=True)
    return world_d / np.where(w_len < 1e-8, 1.0, w_len)


def _sample_sky(
    dirs: np.ndarray,
    scene: SceneDocument,
    sun_dirs: list[tuple[np.ndarray, np.ndarray]],
    *,
    studio_ibl: bool = False,
) -> np.ndarray:
    """Evaluate world radiance (gradient, physical Rayleigh/Mie sky, or equirectangular HDRI)."""
    return evaluate_world_sky(dirs, scene, sun_dirs, studio_ibl=studio_ibl)


class PathTracerRenderer:
    """CPU BVH-accelerated Monte Carlo Path Tracer supporting reflections, refraction, GI & DOF."""

    def render_image(
        self,
        scene: SceneDocument,
        evaluated: dict[str, EvaluatedObject],
        cam_obj: SceneObject,
        config: RenderConfig | None = None,
        colliding_ids: set[str] | None = None,
    ) -> Image.Image:
        cfg = config or RenderConfig()
        w, h = int(cfg.width), int(cfg.height)
        aspect = w / max(1, h)
        view, proj, eye_world = build_camera_matrices(cam_obj, aspect)
        vp = proj @ view
        c2w = np.linalg.inv(view)
        cam_spec = cam_obj.camera
        assert cam_spec is not None

        # Gather triangles, vertex normals, UVs, and materials
        all_verts: list[np.ndarray] = []
        all_vnormals: list[np.ndarray] = []
        all_uvs: list[np.ndarray] = []
        all_faces: list[np.ndarray] = []
        tri_mat_list: list[MaterialSpec] = []
        tri_mat_idx: list[np.ndarray] = []
        tri_is_colliding: list[np.ndarray] = []

        default_mat = scene.materials.get("mat_default") or MaterialSpec(
            id="mat_default", name="Default"
        )
        mat_to_idx: dict[str, int] = {}

        def get_mat_index(mat_id: str | None) -> int:
            m = scene.materials.get(mat_id) if mat_id else None
            m = m or default_mat
            if m.id not in mat_to_idx:
                mat_to_idx[m.id] = len(tri_mat_list)
                tri_mat_list.append(m)
            return mat_to_idx[m.id]

        v_offset = 0
        colliding = colliding_ids or set()
        for oid, ev in evaluated.items():
            mesh = ev.world_mesh
            if mesh.is_empty:
                continue
            all_verts.append(mesh.vertices)
            all_vnormals.append(mesh.vertex_normals())
            if mesh.uvs is not None and len(mesh.uvs) == len(mesh.vertices):
                all_uvs.append(mesh.uvs)
            else:
                all_uvs.append(np.full((len(mesh.vertices), 2), np.nan, dtype=np.float64))
            all_faces.append(mesh.faces + v_offset)
            v_offset += len(mesh.vertices)

            base_mi = get_mat_index(ev.obj.material_id)
            m_arr = np.full((len(mesh.faces),), base_mi, dtype=np.int32)
            if ev.obj.face_material_ids:
                for fidx, fmat_id in ev.obj.face_material_ids.items():
                    if 0 <= fidx < len(m_arr):
                        m_arr[fidx] = get_mat_index(fmat_id)
            tri_mat_idx.append(m_arr)
            tri_is_colliding.append(np.full((len(mesh.faces),), oid in colliding, dtype=bool))

        if all_verts:
            verts_w = np.vstack(all_verts)
            vnormals_w = np.vstack(all_vnormals)
            uvs_w = np.vstack(all_uvs)
            faces_w = np.vstack(all_faces)
            mat_idx_per_tri = np.concatenate(tri_mat_idx)
            coll_per_tri = np.concatenate(tri_is_colliding)
        else:
            verts_w = np.zeros((0, 3), dtype=np.float64)
            vnormals_w = np.zeros((0, 3), dtype=np.float64)
            uvs_w = np.zeros((0, 2), dtype=np.float64)
            faces_w = np.zeros((0, 3), dtype=np.int64)
            mat_idx_per_tri = np.zeros((0,), dtype=np.int32)
            coll_per_tri = np.zeros((0,), dtype=bool)

        bvh = _TriangleBVH(verts_w, faces_w, leaf_size=16)

        # Collect lights
        lights: list[tuple[SceneObject, LightSpec]] = [
            (o, o.light)
            for o in scene.objects.values()
            if o.kind == "light" and o.light is not None and scene.is_object_visible(o)
        ]
        sun_dirs: list[tuple[np.ndarray, np.ndarray]] = []
        for _, lspec in lights:
            if lspec.kind == "sun":
                ldir = -np.asarray(lspec.direction or (-0.5, 0.4, -0.7), dtype=np.float64)
                ldir /= max(1e-9, float(np.linalg.norm(ldir)))
                sun_dirs.append(
                    (ldir, np.asarray(lspec.color, dtype=np.float64) * float(lspec.intensity))
                )

        # Pixel coordinate grid
        yy, xx = np.mgrid[0:h, 0:w]
        px_flat = xx.reshape(-1).astype(np.float64)
        py_flat = yy.reshape(-1).astype(np.float64)
        n_pix = w * h

        rng = np.random.default_rng(1337)
        samples = max(1, int(cfg.samples))
        max_bounces = max(1, int(cfg.max_bounces))

        def build_camera_rays(
            jx: np.ndarray, jy: np.ndarray, lens_jitter: bool = False
        ) -> tuple[np.ndarray, np.ndarray]:
            ndc_x = ((px_flat + jx) / w) * 2.0 - 1.0
            ndc_y = 1.0 - ((py_flat + jy) / h) * 2.0
            if cam_spec.projection == "orthographic":
                half_h = max(cam_spec.ortho_scale * 0.5, 1e-6)
                half_w = half_h * aspect
                orig_cam = np.stack([ndc_x * half_w, ndc_y * half_h, np.zeros_like(ndc_x)], axis=1)
                dir_cam = np.broadcast_to(np.array([[0.0, 0.0, -1.0]]), (n_pix, 3))
                orig_w = orig_cam @ c2w[:3, :3].T + c2w[:3, 3][None, :]
                dir_w = dir_cam @ c2w[:3, :3].T
                dir_w /= np.linalg.norm(dir_w, axis=1, keepdims=True)
                return orig_w, dir_w

            tan_y = math.tan(math.radians(cam_spec.fov_deg) * 0.5)
            tan_x = tan_y * aspect
            dir_cam = np.stack([ndc_x * tan_x, ndc_y * tan_y, np.full_like(ndc_x, -1.0)], axis=1)
            dir_cam /= np.linalg.norm(dir_cam, axis=1, keepdims=True)
            dir_w = dir_cam @ c2w[:3, :3].T
            dir_w /= np.linalg.norm(dir_w, axis=1, keepdims=True)
            orig_w = np.broadcast_to(eye_world[None, :], (n_pix, 3)).copy()

            if lens_jitter and scene.post_process.enabled and scene.post_process.dof_enabled:
                ap = float(scene.post_process.dof_aperture) * 0.09
                f_dist = max(0.2, float(scene.post_process.dof_focus_distance))
                if ap > 1e-5:
                    r_disk = np.sqrt(rng.random(n_pix)) * ap
                    theta = rng.random(n_pix) * (2.0 * math.pi)
                    right = c2w[:3, 0]
                    up_v = c2w[:3, 1]
                    focus_pt = orig_w + dir_w * f_dist
                    offset = (r_disk * np.cos(theta))[:, None] * right[None, :] + (
                        r_disk * np.sin(theta)
                    )[:, None] * up_v[None, :]
                    orig_w = orig_w + offset
                    dir_w = focus_pt - orig_w
                    dir_w /= np.linalg.norm(dir_w, axis=1, keepdims=True)
            return orig_w, dir_w

        # Fast primary-only trace for diagnostic modes
        if cfg.mode in ("normals", "depth", "collision"):
            orig_w, dir_w = build_camera_rays(np.full(n_pix, 0.5), np.full(n_pix, 0.5))
            best_t, best_tri, best_u, best_v = bvh.intersect_closest(orig_w, dir_w)
            hit = best_tri >= 0
            out = (
                np.zeros((n_pix, 3), dtype=np.float32)
                if cfg.mode == "depth"
                else np.full((n_pix, 3), 0.14, dtype=np.float32)
            )
            if np.any(hit):
                ht = best_tri[hit]
                u_h = best_u[hit][:, None]
                v_h = best_v[hit][:, None]
                w_h = 1.0 - u_h - v_h
                nrm = (
                    w_h * vnormals_w[faces_w[ht, 0]]
                    + u_h * vnormals_w[faces_w[ht, 1]]
                    + v_h * vnormals_w[faces_w[ht, 2]]
                )
                nrm /= np.maximum(np.linalg.norm(nrm, axis=1, keepdims=True), 1e-12)
                if cfg.mode == "normals":
                    out[hit] = np.clip(nrm * 0.5 + 0.5, 0.0, 1.0).astype(np.float32)
                elif cfg.mode == "depth":
                    d_hit = best_t[hit]
                    d_min, d_max = float(np.min(d_hit)), float(np.max(d_hit))
                    span = max(1e-5, d_max - d_min)
                    d_norm = np.clip(1.0 - (d_hit - d_min) / span, 0.08, 1.0)[:, None]
                    out[hit] = np.repeat(d_norm, 3, axis=1).astype(np.float32)
                else:
                    is_col = coll_per_tri[ht][:, None]
                    v_dir = -dir_w[hit]
                    n_ori = nrm * np.where(
                        np.sum(nrm * v_dir, axis=1, keepdims=True) < 0, -1.0, 1.0
                    )
                    ndv = np.clip(np.sum(n_ori * v_dir, axis=1, keepdims=True), 0.15, 1.0)
                    base_c = np.where(is_col, [[0.96, 0.14, 0.14]], [[0.45, 0.48, 0.54]])
                    out[hit] = (base_c * (0.35 + 0.65 * ndv)).astype(np.float32)
            bg = out.reshape((h, w, 3))
            u8 = np.clip(bg * 255.0, 0, 255).astype(np.uint8)
            return Image.fromarray(u8, mode="RGB")

        # Full Monte Carlo Path Tracing accumulation buffer + denoiser feature guides
        radiance_accum = np.zeros((n_pix, 3), dtype=np.float64)
        albedo_accum = np.zeros((n_pix, 3), dtype=np.float64)
        normal_accum = np.zeros((n_pix, 3), dtype=np.float64)
        depth_cam = np.full((n_pix,), np.inf, dtype=np.float32)
        z_ndc_buf = np.full((n_pix,), np.inf, dtype=np.float32)

        def shade_direct(
            pos: np.ndarray,
            nrm: np.ndarray,
            view_dir: np.ndarray,
            albedo_lin: np.ndarray,
            roughness: np.ndarray,
            metallic: np.ndarray,
            f0: np.ndarray,
            ao: np.ndarray,
            emission_lin: np.ndarray,
            extra_mat: dict[str, np.ndarray] | None = None,
            soft_shadows: bool = True,
        ) -> np.ndarray:
            k = len(pos)
            # Hemisphere ambient & ground/sky fill driven by active world sky mode
            if scene.world.sky_mode == "gradient":
                hemi_t = 0.5 + 0.5 * nrm[:, 2:3]
                sky_c = (
                    np.power(np.clip(np.asarray(scene.world.zenith_color), 0.0, 1.0), 2.2)[None, :]
                    * 1.4
                    + 0.08
                )
                gnd_c = (
                    np.power(np.clip(np.asarray(scene.world.horizon_color), 0.0, 1.0), 2.2)[None, :]
                    * 0.8
                    + 0.04
                )
                amb_col = gnd_c * (1.0 - hemi_t) + sky_c * hemi_t
            else:
                amb_col = _sample_sky(nrm, scene, []) * 0.72

            direct = (
                albedo_lin * amb_col * scene.world.ambient_strength * (1.0 - metallic) * ao * 0.65
                + emission_lin
            )

            cc = extra_mat["clearcoat"] if extra_mat else 0.0
            cc_r = extra_mat["clearcoat_roughness"] if extra_mat else 0.08
            shn = extra_mat["sheen"] if extra_mat else 0.0
            sss = extra_mat["subsurface"] if extra_mat else 0.0
            sss_c = extra_mat.get("subsurface_color") if extra_mat else None

            for lobj, lspec in lights:
                lcol = np.asarray(lspec.color, dtype=np.float64)[None, :] * float(lspec.intensity)
                if lspec.kind == "sun":
                    base_ldir = -np.asarray(lspec.direction or (-0.5, 0.4, -0.7), dtype=np.float64)
                    base_ldir /= max(1e-9, float(np.linalg.norm(base_ldir)))
                    l_vec = np.broadcast_to(base_ldir[None, :], (k, 3)).copy()
                    if soft_shadows:
                        jitter = rng.normal(0.0, 0.025, size=(k, 3))
                        s_ray_d = l_vec + jitter
                        s_ray_d /= np.linalg.norm(s_ray_d, axis=1, keepdims=True)
                    else:
                        s_ray_d = l_vec
                    max_d = np.full((k,), 1e6, dtype=np.float64)
                    atten = 1.0
                else:
                    lpos = np.asarray(lobj.transform.location, dtype=np.float64)[None, :]
                    if soft_shadows and lspec.size > 0.05:
                        lpos = lpos + rng.normal(0.0, min(0.35, lspec.size * 0.25), size=(k, 3))
                    delta = lpos - pos
                    dist2 = np.sum(delta * delta, axis=1, keepdims=True)
                    dist = np.sqrt(np.maximum(dist2, 1e-6))
                    l_vec = delta / dist
                    s_ray_d = l_vec
                    max_d = dist[:, 0]
                    atten = 1.0 / (1.0 + 0.08 * dist + 0.02 * dist2)
                    if lspec.kind == "spot":
                        sdir = np.asarray(lspec.direction or (0.0, 0.0, -1.0), dtype=np.float64)
                        sdir /= max(1e-9, float(np.linalg.norm(sdir)))
                        cos_cone = math.cos(math.radians(lspec.spot_angle_deg * 0.5))
                        cos_ang = np.sum((-l_vec) * sdir[None, :], axis=1, keepdims=True)
                        spot_mask = np.clip(
                            (cos_ang - cos_cone) / max(1e-4, 1.0 - cos_cone), 0.0, 1.0
                        )
                        atten = atten * spot_mask

                ndl = np.clip(np.sum(nrm * l_vec, axis=1, keepdims=True), 0.0, 1.0)
                atten_mask = atten[:, 0] > 1e-4 if np.ndim(atten) > 1 else True
                has_sss = np.any(sss > 1e-4)
                lit_mask = ((ndl[:, 0] > 1e-4) | has_sss) & atten_mask
                shadow = np.ones((k, 1), dtype=np.float64)
                if cfg.enable_shadows and lspec.cast_shadows and np.any(lit_mask):
                    lit_idx = np.nonzero(lit_mask)[0]
                    s_orig = pos[lit_idx] + nrm[lit_idx] * 1.5e-3
                    occ = bvh.intersect_occluded(s_orig, s_ray_d[lit_idx], max_d[lit_idx])
                    shadow[lit_idx, 0] = np.where(occ, 0.0, 1.0)

                brdf = evaluate_ggx_brdf(
                    nrm,
                    view_dir,
                    l_vec,
                    albedo_lin,
                    roughness,
                    metallic,
                    f0,
                    clearcoat=cc,
                    clearcoat_roughness=cc_r,
                    sheen=shn,
                    subsurface=sss,
                    subsurface_color=sss_c,
                )
                direct += brdf * lcol * atten * shadow

            return direct

        def evaluate_hit_surface(
            orig: np.ndarray,
            dirs: np.ndarray,
            t_hit: np.ndarray,
            tri_ids: np.ndarray,
            u_b: np.ndarray,
            v_b: np.ndarray,
        ) -> tuple[
            np.ndarray,
            np.ndarray,
            np.ndarray,
            np.ndarray,
            np.ndarray,
            np.ndarray,
            np.ndarray,
            np.ndarray,
            np.ndarray,
            np.ndarray,
            np.ndarray,
            dict[str, np.ndarray],
        ]:
            k = len(tri_ids)
            pos = orig + dirs * t_hit[:, None]
            u = u_b[:, None]
            v = v_b[:, None]
            w_b = 1.0 - u - v

            tv0 = verts_w[faces_w[tri_ids, 0]]
            tv1 = verts_w[faces_w[tri_ids, 1]]
            tv2 = verts_w[faces_w[tri_ids, 2]]
            fn_tri = np.cross(tv1 - tv0, tv2 - tv0)
            fn_tri /= np.maximum(np.linalg.norm(fn_tri, axis=1, keepdims=True), 1e-12)
            tn0 = vnormals_w[faces_w[tri_ids, 0]]
            tn1 = vnormals_w[faces_w[tri_ids, 1]]
            tn2 = vnormals_w[faces_w[tri_ids, 2]]
            tn0 = np.where(np.sum(tn0 * fn_tri, axis=1, keepdims=True) < 0.743, fn_tri, tn0)
            tn1 = np.where(np.sum(tn1 * fn_tri, axis=1, keepdims=True) < 0.743, fn_tri, tn1)
            tn2 = np.where(np.sum(tn2 * fn_tri, axis=1, keepdims=True) < 0.743, fn_tri, tn2)

            nrm_g = w_b * tn0 + u * tn1 + v * tn2
            nrm_g /= np.maximum(np.linalg.norm(nrm_g, axis=1, keepdims=True), 1e-12)
            view_dir = -dirs
            front_face = np.sum(nrm_g * view_dir, axis=1, keepdims=True) >= 0.0
            nrm_oriented = np.where(front_face, nrm_g, -nrm_g)

            uv_hit = (
                w_b * uvs_w[faces_w[tri_ids, 0]]
                + u * uvs_w[faces_w[tri_ids, 1]]
                + v * uvs_w[faces_w[tri_ids, 2]]
            )

            albedo = np.zeros((k, 3), dtype=np.float64)
            roughness = np.full((k, 1), 0.5, dtype=np.float64)
            metallic = np.zeros((k, 1), dtype=np.float64)
            ao = np.ones((k, 1), dtype=np.float64)
            emission = np.zeros((k, 3), dtype=np.float64)
            alpha = np.ones((k, 1), dtype=np.float64)
            ior = np.full((k, 1), 1.45, dtype=np.float64)
            clearcoat = np.zeros((k, 1), dtype=np.float64)
            clearcoat_roughness = np.full((k, 1), 0.08, dtype=np.float64)
            sheen = np.zeros((k, 1), dtype=np.float64)
            subsurface = np.zeros((k, 1), dtype=np.float64)
            subsurface_color = np.full((k, 3), -1.0, dtype=np.float64)
            shaded_nrm = nrm_oriented.copy()

            if cfg.mode == "clay":
                albedo[:] = np.array([0.78, 0.76, 0.73])
                roughness[:] = 0.62
            else:
                hit_mids = mat_idx_per_tri[tri_ids]
                for mi, mspec in enumerate(tri_mat_list):
                    mask_m = hit_mids == mi
                    if not np.any(mask_m):
                        continue
                    alb_m, nrm_m, rgh_m, mtl_m, ao_m = evaluate_material_maps(
                        mspec, pos[mask_m], shaded_nrm[mask_m], uvs=uv_hit[mask_m]
                    )
                    albedo[mask_m] = alb_m
                    shaded_nrm[mask_m] = nrm_m
                    roughness[mask_m] = rgh_m
                    metallic[mask_m] = mtl_m
                    ao[mask_m] = ao_m
                    alpha[mask_m] = float(mspec.alpha * mspec.base_color[3])
                    ior[mask_m] = float(mspec.ior)
                    clearcoat[mask_m] = float(mspec.clearcoat)
                    clearcoat_roughness[mask_m] = float(mspec.clearcoat_roughness)
                    sheen[mask_m] = float(mspec.sheen)
                    subsurface[mask_m] = float(mspec.subsurface)
                    if mspec.subsurface_color is not None:
                        subsurface_color[mask_m] = np.power(
                            np.clip(np.asarray(mspec.subsurface_color), 0.0, 1.0), 2.2
                        )
                    if mspec.emission_strength > 0:
                        emission[mask_m] = (
                            np.asarray(mspec.emission_color[:3])[None, :] * mspec.emission_strength
                        )

            albedo_lin = np.power(np.clip(albedo, 0.0, 1.0), 2.2)
            emission_lin = np.power(np.clip(emission, 0.0, 1.0), 2.2)
            # Fall back subsurface_color to warm albedo tint where unspecified
            sss_col_resolved = np.where(
                subsurface_color >= 0.0,
                subsurface_color,
                np.clip(albedo_lin * np.array([[1.15, 0.62, 0.48]]), 0.0, 1.0),
            )
            extra = {
                "clearcoat": clearcoat,
                "clearcoat_roughness": clearcoat_roughness,
                "sheen": sheen,
                "subsurface": subsurface,
                "subsurface_color": sss_col_resolved,
            }
            return (
                pos,
                shaded_nrm,
                view_dir,
                albedo_lin,
                roughness,
                metallic,
                ao,
                emission_lin,
                alpha,
                ior,
                front_face,
                extra,
            )

        for s_idx in range(samples):
            jx = (
                np.full(n_pix, 0.5)
                if samples == 1
                else np.clip(0.5 + (rng.random(n_pix) - 0.5) * 0.85, 0.0, 1.0)
            )
            jy = (
                np.full(n_pix, 0.5)
                if samples == 1
                else np.clip(0.5 + (rng.random(n_pix) - 0.5) * 0.85, 0.0, 1.0)
            )
            orig, dirs = build_camera_rays(jx, jy, lens_jitter=(samples > 0))

            best_t, best_tri, best_u, best_v = bvh.intersect_closest(orig, dirs)
            hit_mask = best_tri >= 0

            if s_idx == 0 and np.any(hit_mask):
                depth_cam[hit_mask] = best_t[hit_mask].astype(np.float32)
                # Compute NDC depth for optional grid overlay alignment
                hit_pts = orig[hit_mask] + dirs[hit_mask] * best_t[hit_mask, None]
                hom = np.hstack([hit_pts, np.ones((len(hit_pts), 1), dtype=np.float64)])
                clip = hom @ vp.T
                w_safe = np.where(np.abs(clip[:, 3]) < 1e-6, 1e-6, clip[:, 3])
                z_ndc_buf[hit_mask] = (clip[:, 2] / w_safe).astype(np.float32)

            sample_rgb = np.zeros((n_pix, 3), dtype=np.float64)
            miss_mask = ~hit_mask
            if np.any(miss_mask):
                sky_miss = _sample_sky(dirs[miss_mask], scene, sun_dirs)
                sample_rgb[miss_mask] = sky_miss
                albedo_accum[miss_mask] += np.clip(sky_miss, 0.0, 1.0)

            if np.any(hit_mask):
                h_idx = np.nonzero(hit_mask)[0]
                (
                    pos,
                    nrm,
                    view_dir,
                    albedo_lin,
                    roughness,
                    metallic,
                    ao,
                    emission_lin,
                    alpha,
                    ior,
                    front_face,
                    mat_extra,
                ) = evaluate_hit_surface(
                    orig[h_idx],
                    dirs[h_idx],
                    best_t[h_idx],
                    best_tri[h_idx],
                    best_u[h_idx],
                    best_v[h_idx],
                )
                albedo_accum[h_idx] += albedo_lin
                normal_accum[h_idx] += nrm

                ndv = np.clip(np.sum(nrm * view_dir, axis=1, keepdims=True), 0.0, 1.0)
                f0 = 0.04 * (1.0 - metallic) + albedo_lin * metallic
                fresnel = f0 + (1.0 - f0) * ((1.0 - ndv) ** 5)

                direct = shade_direct(
                    pos,
                    nrm,
                    view_dir,
                    albedo_lin,
                    roughness,
                    metallic,
                    f0,
                    ao,
                    emission_lin,
                    extra_mat=mat_extra,
                    soft_shadows=True,
                )
                pixel_rad = direct * alpha

                # 1. Indirect diffuse GI bounce (cosine hemisphere weighted)
                if max_bounces >= 2:
                    gi_weight = albedo_lin * (1.0 - metallic) * alpha * ao * 0.45
                    gi_active = np.max(gi_weight, axis=1) > 0.02
                    if np.any(gi_active):
                        g_i = np.nonzero(gi_active)[0]
                        gi_dirs = _cosine_hemisphere(nrm[g_i], rng)
                        gi_orig = pos[g_i] + nrm[g_i] * 2e-3
                        gt, gtri, gu, gv = bvh.intersect_closest(gi_orig, gi_dirs)
                        ghit = gtri >= 0
                        gi_rad = np.zeros((len(g_i), 3), dtype=np.float64)
                        if np.any(~ghit):
                            gi_rad[~ghit] = _sample_sky(gi_dirs[~ghit], scene, [])
                        if np.any(ghit):
                            gh_sub = np.nonzero(ghit)[0]
                            (
                                g_pos,
                                g_nrm,
                                g_vdir,
                                g_alb,
                                g_rgh,
                                g_mtl,
                                g_ao,
                                g_emi,
                                _,
                                _,
                                _,
                                g_extra,
                            ) = evaluate_hit_surface(
                                gi_orig[gh_sub],
                                gi_dirs[gh_sub],
                                gt[gh_sub],
                                gtri[gh_sub],
                                gu[gh_sub],
                                gv[gh_sub],
                            )
                            g_f0 = 0.04 * (1.0 - g_mtl) + g_alb * g_mtl
                            gi_rad[gh_sub] = shade_direct(
                                g_pos,
                                g_nrm,
                                g_vdir,
                                g_alb,
                                g_rgh,
                                g_mtl,
                                g_f0,
                                g_ao,
                                g_emi,
                                extra_mat=g_extra,
                                soft_shadows=False,
                            )
                        pixel_rad[g_i] += gi_rad * gi_weight[g_i]

                # 2. Specular Fresnel reflection (recursive up to max_bounces, including Clearcoat)
                cc_w = mat_extra["clearcoat"] * (0.04 + 0.96 * ((1.0 - ndv) ** 5))
                base_refl = fresnel * (metallic + (1.0 - metallic) * (1.0 - roughness * 0.7))
                refl_weight = np.maximum(
                    base_refl, cc_w * (1.0 - mat_extra["clearcoat_roughness"] * 0.5)
                )
                eff_rough = np.where(cc_w > base_refl, mat_extra["clearcoat_roughness"], roughness)
                refl_active = (np.max(refl_weight, axis=1) > 0.04) & (max_bounces >= 2)
                if np.any(refl_active):
                    r_i = np.nonzero(refl_active)[0]
                    cur_o = pos[r_i] + nrm[r_i] * 2e-3
                    cur_d = (
                        dirs[h_idx[r_i]]
                        - 2.0
                        * np.sum(dirs[h_idx[r_i]] * nrm[r_i], axis=1, keepdims=True)
                        * nrm[r_i]
                    )
                    # Glossy microfacet roughness perturbation
                    cur_d += rng.normal(0.0, 1.0, size=cur_d.shape) * (eff_rough[r_i] ** 2 * 0.35)
                    cur_d /= np.maximum(np.linalg.norm(cur_d, axis=1, keepdims=True), 1e-12)
                    tp = refl_weight[r_i].copy()
                    active_map = r_i.copy()

                    for _b in range(max_bounces - 1):
                        if len(active_map) == 0:
                            break
                        rt, rtri, ru, rv = bvh.intersect_closest(cur_o, cur_d)
                        rhit = rtri >= 0
                        if np.any(~rhit):
                            pixel_rad[active_map[~rhit]] += (
                                _sample_sky(cur_d[~rhit], scene, sun_dirs, studio_ibl=True)
                                * tp[~rhit]
                            )
                        if not np.any(rhit):
                            break
                        rh_sub = np.nonzero(rhit)[0]
                        (
                            r_pos,
                            r_nrm,
                            r_vdir,
                            r_alb,
                            r_rgh,
                            r_mtl,
                            r_ao,
                            r_emi,
                            _,
                            _,
                            _,
                            r_extra,
                        ) = evaluate_hit_surface(
                            cur_o[rh_sub],
                            cur_d[rh_sub],
                            rt[rh_sub],
                            rtri[rh_sub],
                            ru[rh_sub],
                            rv[rh_sub],
                        )
                        r_ndv = np.clip(np.sum(r_nrm * r_vdir, axis=1, keepdims=True), 0.0, 1.0)
                        r_f0 = 0.04 * (1.0 - r_mtl) + r_alb * r_mtl
                        r_fres = r_f0 + (1.0 - r_f0) * ((1.0 - r_ndv) ** 5)
                        r_direct = shade_direct(
                            r_pos,
                            r_nrm,
                            r_vdir,
                            r_alb,
                            r_rgh,
                            r_mtl,
                            r_f0,
                            r_ao,
                            r_emi,
                            extra_mat=r_extra,
                            soft_shadows=False,
                        )
                        pixel_rad[active_map[rh_sub]] += r_direct * tp[rh_sub]
                        # Next reflection bounce on mirror-like hits
                        next_w = r_fres * (r_mtl + (1.0 - r_mtl) * (1.0 - r_rgh))
                        keep = np.max(next_w, axis=1) > 0.08
                        if not np.any(keep):
                            break
                        k_sub = np.nonzero(keep)[0]
                        d_in = cur_d[rh_sub[k_sub]]
                        n_in = r_nrm[k_sub]
                        cur_d = d_in - 2.0 * np.sum(d_in * n_in, axis=1, keepdims=True) * n_in
                        cur_d /= np.maximum(np.linalg.norm(cur_d, axis=1, keepdims=True), 1e-12)
                        cur_o = r_pos[k_sub] + n_in * 2e-3
                        tp = tp[rh_sub[k_sub]] * next_w[k_sub]
                        active_map = active_map[rh_sub[k_sub]]

                # 3. Dielectric refraction / transmission for transparent objects (alpha < 0.98)
                trans_active = (alpha[:, 0] < 0.98) & (max_bounces >= 2)
                if np.any(trans_active):
                    t_i = np.nonzero(trans_active)[0]
                    eta = np.where(front_face[t_i], 1.0 / np.maximum(1.01, ior[t_i]), ior[t_i])
                    d_inc = dirs[h_idx[t_i]]
                    n_t = nrm[t_i]
                    cos_i = np.clip(np.sum(-d_inc * n_t, axis=1, keepdims=True), 0.0, 1.0)
                    sin2_t = (eta**2) * (1.0 - cos_i**2)
                    tir = sin2_t > 1.0
                    cos_t = np.sqrt(np.maximum(0.0, 1.0 - sin2_t))
                    t_dir = eta * d_inc + (eta * cos_i - cos_t) * n_t
                    r_dir_tir = d_inc - 2.0 * np.sum(d_inc * n_t, axis=1, keepdims=True) * n_t
                    t_dir = np.where(tir, r_dir_tir, t_dir)
                    t_dir /= np.maximum(np.linalg.norm(t_dir, axis=1, keepdims=True), 1e-12)
                    t_orig = np.where(tir, pos[t_i] + n_t * 2e-3, pos[t_i] - n_t * 2.5e-3)
                    t_weight = (
                        (1.0 - alpha[t_i]) * (1.0 - fresnel[t_i]) * (0.35 + 0.65 * albedo_lin[t_i])
                    )

                    tt, ttri, tu, tv = bvh.intersect_closest(t_orig, t_dir)
                    thit = ttri >= 0
                    trans_rad = np.zeros((len(t_i), 3), dtype=np.float64)
                    if np.any(~thit):
                        trans_rad[~thit] = _sample_sky(
                            t_dir[~thit], scene, sun_dirs, studio_ibl=True
                        )
                    if np.any(thit):
                        th_sub = np.nonzero(thit)[0]
                        (
                            t_pos,
                            t_nrm,
                            t_vdir,
                            t_alb,
                            t_rgh,
                            t_mtl,
                            t_ao,
                            t_emi,
                            _,
                            _,
                            _,
                            t_extra,
                        ) = evaluate_hit_surface(
                            t_orig[th_sub],
                            t_dir[th_sub],
                            tt[th_sub],
                            ttri[th_sub],
                            tu[th_sub],
                            tv[th_sub],
                        )
                        t_f0 = 0.04 * (1.0 - t_mtl) + t_alb * t_mtl
                        trans_rad[th_sub] = shade_direct(
                            t_pos,
                            t_nrm,
                            t_vdir,
                            t_alb,
                            t_rgh,
                            t_mtl,
                            t_f0,
                            t_ao,
                            t_emi,
                            extra_mat=t_extra,
                            soft_shadows=False,
                        )
                    pixel_rad[t_i] += trans_rad * t_weight

                if cfg.mode == "wireframe":
                    # Edge barycentric distance overlay for wireframe in raytrace mode
                    edge_dist = np.minimum(
                        1.0 - best_u[h_idx] - best_v[h_idx],
                        np.minimum(best_u[h_idx], best_v[h_idx]),
                    )[:, None]
                    w_mix = np.clip(1.0 - edge_dist / 0.04, 0.0, 1.0)
                    pixel_rad = (
                        pixel_rad * 0.65 * (1.0 - w_mix) + np.array([[0.12, 0.92, 0.98]]) * w_mix
                    )

                sample_rgb[h_idx] = pixel_rad

            radiance_accum += sample_rgb

        hdr_buf = (radiance_accum / samples).reshape((h, w, 3)).astype(np.float32)
        albedo_2d = (albedo_accum / samples).reshape((h, w, 3)).astype(np.float32)
        normal_2d = (normal_accum / samples).reshape((h, w, 3)).astype(np.float32)
        depth_2d = depth_cam.reshape((h, w))
        z_2d = z_ndc_buf.reshape((h, w))

        # Avoid double-blurring DOF since physical lens jitter already ran during ray generation
        pp_cfg = scene.post_process.model_copy(update={"dof_enabled": False})
        bg = apply_post_processing(
            hdr_buf,
            depth_buf=depth_2d,
            config=pp_cfg,
            albedo_buf=albedo_2d,
            normal_buf=normal_2d,
        )

        if cfg.show_grid:
            grid_extent = 5
            for i in range(-grid_extent, grid_extent + 1):
                col_x = (0.72, 0.25, 0.25) if i == 0 else (0.34, 0.36, 0.40)
                col_y = (0.25, 0.72, 0.28) if i == 0 else (0.34, 0.36, 0.40)
                alpha_g = 0.75 if i == 0 else 0.35
                _draw_segment_z(
                    bg,
                    z_2d,
                    np.array([-grid_extent, float(i), 0.0]),
                    np.array([grid_extent, float(i), 0.0]),
                    vp,
                    col_x,
                    alpha=alpha_g,
                )
                _draw_segment_z(
                    bg,
                    z_2d,
                    np.array([float(i), -grid_extent, 0.0]),
                    np.array([float(i), grid_extent, 0.0]),
                    vp,
                    col_y,
                    alpha=alpha_g,
                )

        u8 = np.clip(bg * 255.0, 0, 255).astype(np.uint8)
        return Image.fromarray(u8, mode="RGB")
