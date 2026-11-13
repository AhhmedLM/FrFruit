"""Pure-CPU vectorized G-buffer triangle rasterizer and diagnostic shader suite."""

from __future__ import annotations

import io
import math
from dataclasses import dataclass
from typing import Literal

import numpy as np
from PIL import Image

from frfruit.core.math3d import look_at_matrix, orthographic_matrix, transform_points
from frfruit.core.scene import LightSpec, MaterialSpec, SceneDocument, SceneObject
from frfruit.geometry.spatial import EvaluatedObject, compute_scene_bounds
from frfruit.render.camera import build_camera_matrices
from frfruit.render.materials import evaluate_surface_albedo

RenderMode = Literal["pbr", "clay", "wireframe", "normals", "depth", "collision"]


@dataclass
class RenderConfig:
    width: int = 640
    height: int = 480
    mode: RenderMode = "pbr"
    show_grid: bool = True
    show_axes: bool = True
    enable_shadows: bool = True


def _rasterize_gbuffer(
    verts_world: np.ndarray,
    faces: np.ndarray,
    view: np.ndarray,
    proj: np.ndarray,
    width: int,
    height: int,
    cull_backfaces: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Rasterize triangles into screen G-buffers (z_buf, tri_id, u, v, edge_min_px)."""
    z_buf = np.full((height, width), np.inf, dtype=np.float32)
    tri_id_buf = np.full((height, width), -1, dtype=np.int32)
    u_buf = np.zeros((height, width), dtype=np.float32)
    v_buf = np.zeros((height, width), dtype=np.float32)
    edge_px_buf = np.full((height, width), 999.0, dtype=np.float32)

    if len(faces) == 0 or len(verts_world) == 0:
        return z_buf, tri_id_buf, u_buf, v_buf, edge_px_buf

    vp = proj @ view
    ones = np.ones((len(verts_world), 1), dtype=np.float64)
    v_hom = np.hstack([verts_world, ones])
    clip = v_hom @ vp.T
    w_clip = clip[:, 3]

    # Screen coordinates
    w_safe = np.where(np.abs(w_clip) < 1e-6, 1e-6, w_clip)
    ndc_x = clip[:, 0] / w_safe
    ndc_y = clip[:, 1] / w_safe
    ndc_z = clip[:, 2] / w_safe

    sx = (ndc_x * 0.5 + 0.5) * width
    sy = (0.5 - ndc_y * 0.5) * height

    f0 = faces[:, 0]
    f1 = faces[:, 1]
    f2 = faces[:, 2]

    # Keep triangles where all 3 vertices are in front of near plane (w_clip > 0)
    in_front = (w_clip[f0] > 1e-4) & (w_clip[f1] > 1e-4) & (w_clip[f2] > 1e-4)

    x0_all, y0_all, z0_all = sx[f0], sy[f0], ndc_z[f0]
    x1_all, y1_all, z1_all = sx[f1], sy[f1], ndc_z[f1]
    x2_all, y2_all, z2_all = sx[f2], sy[f2], ndc_z[f2]

    # Signed double area in screen space (remember Y is flipped downward!)
    # CCW in NDC (+Y up) becomes negative signed area in +Y-down screen space.
    area2_all = (x1_all - x0_all) * (y2_all - y0_all) - (y1_all - y0_all) * (x2_all - x0_all)

    if cull_backfaces:
        valid_mask = in_front & (area2_all < -1e-3)
    else:
        valid_mask = in_front & (np.abs(area2_all) > 1e-3)

    # Bounding-box screen frustum culling
    min_x = np.floor(np.minimum(x0_all, np.minimum(x1_all, x2_all))).astype(np.int32)
    max_x = np.ceil(np.maximum(x0_all, np.maximum(x1_all, x2_all))).astype(np.int32)
    min_y = np.floor(np.minimum(y0_all, np.minimum(y1_all, y2_all))).astype(np.int32)
    max_y = np.ceil(np.maximum(y0_all, np.maximum(y1_all, y2_all))).astype(np.int32)

    valid_mask &= (max_x > 0) & (min_x < width) & (max_y > 0) & (min_y < height)
    valid_mask &= (np.minimum(z0_all, np.minimum(z1_all, z2_all)) <= 1.0) & (
        np.maximum(z0_all, np.maximum(z1_all, z2_all)) >= -1.0
    )

    cand_idx = np.nonzero(valid_mask)[0]
    if len(cand_idx) == 0:
        return z_buf, tri_id_buf, u_buf, v_buf, edge_px_buf

    px_grid = np.arange(width, dtype=np.float32)[None, :] + 0.5
    py_grid = np.arange(height, dtype=np.float32)[:, None] + 0.5

    for ti in cand_idx:
        ix0 = max(0, int(min_x[ti]))
        ix1 = min(width, int(max_x[ti]))
        iy0 = max(0, int(min_y[ti]))
        iy1 = min(height, int(max_y[ti]))
        if ix1 <= ix0 or iy1 <= iy0:
            continue

        ax, ay, az = float(x0_all[ti]), float(y0_all[ti]), float(z0_all[ti])
        bx, by, bz = float(x1_all[ti]), float(y1_all[ti]), float(z1_all[ti])
        cx, cy, cz = float(x2_all[ti]), float(y2_all[ti]), float(z2_all[ti])
        inv_a2 = 1.0 / float(area2_all[ti])

        px = px_grid[:, ix0:ix1]
        py = py_grid[iy0:iy1, :]

        # Barycentric weights: e0 corresponds to vertex A (opposite BC), e1 to B (CA), e2 to C (AB)
        e0 = ((cx - bx) * (py - by) - (cy - by) * (px - bx)) * inv_a2
        e1 = ((ax - cx) * (py - cy) - (ay - cy) * (px - cx)) * inv_a2
        e2 = 1.0 - e0 - e1

        inside = (e0 >= -1e-4) & (e1 >= -1e-4) & (e2 >= -1e-4)
        if not np.any(inside):
            continue

        z_interp = e0 * az + e1 * bz + e2 * cz
        inside &= (z_interp >= -1.0) & (z_interp <= 1.0)
        z_slice = z_buf[iy0:iy1, ix0:ix1]
        closer = inside & (z_interp < z_slice)
        if not np.any(closer):
            continue

        z_slice[closer] = z_interp[closer]
        tri_id_buf[iy0:iy1, ix0:ix1][closer] = int(ti)
        u_buf[iy0:iy1, ix0:ix1][closer] = e1[closer]
        v_buf[iy0:iy1, ix0:ix1][closer] = e2[closer]

        # Exact screen-space perpendicular distance to nearest triangle edge (in pixels)
        abs_a2 = abs(float(area2_all[ti]))
        l0 = math.hypot(cx - bx, cy - by) + 1e-6
        l1 = math.hypot(ax - cx, ay - cy) + 1e-6
        l2 = math.hypot(bx - ax, by - ay) + 1e-6
        d_min = np.minimum(
            e0 * abs_a2 / l0,
            np.minimum(e1 * abs_a2 / l1, e2 * abs_a2 / l2),
        )
        edge_px_buf[iy0:iy1, ix0:ix1][closer] = d_min[closer]

    return z_buf, tri_id_buf, u_buf, v_buf, edge_px_buf


def _draw_segment_z(
    img_rgb: np.ndarray,
    z_buf: np.ndarray,
    p0: np.ndarray,
    p1: np.ndarray,
    vp: np.ndarray,
    color: tuple[float, float, float],
    alpha: float = 0.65,
) -> None:
    """Rasterize a 3D line segment depth-tested against z_buf."""
    h, w, _ = img_rgb.shape
    pts = np.vstack([p0, p1])
    hom = np.hstack([pts, np.ones((2, 1))])
    clip = hom @ vp.T
    if clip[0, 3] <= 1e-4 or clip[1, 3] <= 1e-4:
        return
    ndc = clip[:, :3] / clip[:, 3:4]
    sx0 = (ndc[0, 0] * 0.5 + 0.5) * w
    sy0 = (0.5 - ndc[0, 1] * 0.5) * h
    sx1 = (ndc[1, 0] * 0.5 + 0.5) * w
    sy1 = (0.5 - ndc[1, 1] * 0.5) * h

    steps = int(max(abs(sx1 - sx0), abs(sy1 - sy0), 1))
    steps = min(steps, max(w, h) * 2)
    ts = np.linspace(0.0, 1.0, steps + 1)
    xs = np.round(sx0 + (sx1 - sx0) * ts).astype(np.int32)
    ys = np.round(sy0 + (sy1 - sy0) * ts).astype(np.int32)
    zs = ndc[0, 2] + (ndc[1, 2] - ndc[0, 2]) * ts

    valid = (xs >= 0) & (xs < w) & (ys >= 0) & (ys < h) & (zs >= -1.0) & (zs <= 1.0)
    xs, ys, zs = xs[valid], ys[valid], zs[valid]
    if len(xs) == 0:
        return
    vis = zs <= z_buf[ys, xs] + 1e-3
    xs, ys = xs[vis], ys[vis]
    if len(xs) == 0:
        return
    col_arr = np.asarray(color, dtype=np.float32)
    img_rgb[ys, xs] = img_rgb[ys, xs] * (1.0 - alpha) + col_arr * alpha


class SoftwareRenderer:
    """Pure-CPU vectorized renderer producing deterministic RGBA/PNG buffers."""

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

        # Background sky gradient
        y_frac = np.linspace(0.0, 1.0, h, dtype=np.float32)[:, None, None]
        zenith = np.asarray(scene.world.zenith_color, dtype=np.float32)[None, None, :]
        horizon = np.asarray(scene.world.horizon_color, dtype=np.float32)[None, None, :]
        if cfg.mode in ("depth",):
            bg = np.zeros((h, w, 3), dtype=np.float32)
        elif cfg.mode in ("normals", "clay"):
            bg = np.full((h, w, 3), 0.14, dtype=np.float32)
        else:
            bg = zenith * (1.0 - y_frac) + horizon * y_frac
            bg = np.broadcast_to(bg, (h, w, 3)).copy()

        # Gather flattened world triangles across all evaluated objects
        all_verts: list[np.ndarray] = []
        all_vnormals: list[np.ndarray] = []
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
            faces_w = np.vstack(all_faces)
            mat_idx_per_tri = np.concatenate(tri_mat_idx)
            coll_per_tri = np.concatenate(tri_is_colliding)
        else:
            verts_w = np.zeros((0, 3), dtype=np.float64)
            vnormals_w = np.zeros((0, 3), dtype=np.float64)
            faces_w = np.zeros((0, 3), dtype=np.int64)
            mat_idx_per_tri = np.zeros((0,), dtype=np.int32)
            coll_per_tri = np.zeros((0,), dtype=bool)

        z_buf, tri_id_buf, u_buf, v_buf, edge_px_buf = _rasterize_gbuffer(
            verts_w,
            faces_w,
            view,
            proj,
            w,
            h,
            cull_backfaces=False,
        )

        # Shade hit pixels first, then overlay depth-tested ground grid on open pixels
        hit_mask = tri_id_buf >= 0
        if np.any(hit_mask):
            tids = tri_id_buf[hit_mask]
            u = u_buf[hit_mask][:, None].astype(np.float64)
            v = v_buf[hit_mask][:, None].astype(np.float64)
            w_b = 1.0 - u - v

            tri_v0 = verts_w[faces_w[tids, 0]]
            tri_v1 = verts_w[faces_w[tids, 1]]
            tri_v2 = verts_w[faces_w[tids, 2]]
            pos_w = w_b * tri_v0 + u * tri_v1 + v * tri_v2

            tri_n0 = vnormals_w[faces_w[tids, 0]]
            tri_n1 = vnormals_w[faces_w[tids, 1]]
            tri_n2 = vnormals_w[faces_w[tids, 2]]
            nrm = w_b * tri_n0 + u * tri_n1 + v * tri_n2
            n_len = np.linalg.norm(nrm, axis=1, keepdims=True)
            nrm = nrm / np.where(n_len < 1e-12, 1.0, n_len)

            # Orient normal toward camera if double-sided thin surface is seen from back
            view_dir = eye_world[None, :] - pos_w
            vd_len = np.linalg.norm(view_dir, axis=1, keepdims=True)
            view_dir = view_dir / np.where(vd_len < 1e-12, 1.0, vd_len)
            flip_sign = np.where(np.sum(nrm * view_dir, axis=1, keepdims=True) < 0.0, -1.0, 1.0)
            nrm_oriented = nrm * flip_sign

            if cfg.mode == "normals":
                shaded = np.clip(nrm * 0.5 + 0.5, 0.0, 1.0)
            elif cfg.mode == "depth":
                z_hit = z_buf[hit_mask]
                z_min = float(np.min(z_hit))
                z_max = float(np.max(z_hit))
                span = max(1e-5, z_max - z_min)
                # Brighter = closer to camera
                d_norm = np.clip(1.0 - (z_hit - z_min) / span, 0.08, 1.0)[:, None]
                shaded = np.repeat(d_norm, 3, axis=1)
            elif cfg.mode == "collision":
                is_col = coll_per_tri[tids][:, None]
                ndv = np.clip(np.sum(nrm_oriented * view_dir, axis=1, keepdims=True), 0.15, 1.0)
                base_col = np.where(
                    is_col,
                    np.array([[0.96, 0.14, 0.14]]),
                    np.array([[0.45, 0.48, 0.54]]),
                )
                shaded = base_col * (0.35 + 0.65 * ndv)
            else:
                # PBR / Clay / Wireframe lighting pass
                n_px = len(tids)
                albedo = np.zeros((n_px, 3), dtype=np.float64)
                metallic = np.zeros((n_px, 1), dtype=np.float64)
                roughness = np.full((n_px, 1), 0.5, dtype=np.float64)
                emission = np.zeros((n_px, 3), dtype=np.float64)

                if cfg.mode == "clay":
                    albedo[:] = np.array([0.78, 0.76, 0.73])
                    roughness[:] = 0.62
                else:
                    hit_mat_ids = mat_idx_per_tri[tids]
                    for mi, mspec in enumerate(tri_mat_list):
                        mask_m = hit_mat_ids == mi
                        if not np.any(mask_m):
                            continue
                        albedo[mask_m] = evaluate_surface_albedo(mspec, pos_w[mask_m])
                        metallic[mask_m] = mspec.metallic
                        roughness[mask_m] = mspec.roughness
                        if mspec.emission_strength > 0:
                            emission[mask_m] = (
                                np.asarray(mspec.emission_color[:3])[None, :]
                                * mspec.emission_strength
                            )

                # Convert UI sRGB albedo & emission into linear space for accurate PBR lighting
                albedo_lin = np.power(np.clip(albedo, 0.0, 1.0), 2.2)
                emission_lin = np.power(np.clip(emission, 0.0, 1.0), 2.2)

                # Collect visible lights
                lights: list[tuple[SceneObject, LightSpec]] = []
                for lobj in scene.objects.values():
                    if (
                        lobj.kind == "light"
                        and lobj.light is not None
                        and scene.is_object_visible(lobj)
                    ):
                        lights.append((lobj, lobj.light))

                # Optional directional sun shadow map with tight bounds & 4-tap PCF
                sun_shadow_factor = np.ones((n_px, 1), dtype=np.float64)
                if cfg.enable_shadows and cfg.mode == "pbr" and len(verts_w) > 0:
                    sun_lights = [lp for lp in lights if lp[1].kind == "sun" and lp[1].cast_shadows]
                    if sun_lights:
                        _, sun_spec = sun_lights[0]
                        s_dir = np.asarray(
                            sun_spec.direction or (-0.5, 0.4, -0.7), dtype=np.float64
                        )
                        s_dir /= max(1e-9, float(np.linalg.norm(s_dir)))
                        s_bounds = compute_scene_bounds(evaluated)
                        s_center = np.asarray(s_bounds.center, dtype=np.float64)
                        s_half = max(0.8, 0.65 * float(np.linalg.norm(s_bounds.size)))
                        s_eye = s_center - s_dir * (s_half * 3.0)
                        s_view = look_at_matrix(
                            (float(s_eye[0]), float(s_eye[1]), float(s_eye[2])),
                            (float(s_center[0]), float(s_center[1]), float(s_center[2])),
                        )
                        s_proj = orthographic_matrix(s_half * 2.2, 1.0, 0.1, s_half * 7.0)
                        sz_res = 512
                        sz_buf, _, _, _, _ = _rasterize_gbuffer(
                            verts_w, faces_w, s_view, s_proj, sz_res, sz_res, cull_backfaces=False
                        )
                        s_vp = s_proj @ s_view
                        p_clip = transform_points(s_vp, pos_w)
                        sx_f = (p_clip[:, 0] * 0.5 + 0.5) * sz_res - 0.5
                        sy_f = (0.5 - p_clip[:, 1] * 0.5) * sz_res - 0.5
                        sx_base = np.floor(sx_f).astype(np.int32)
                        sy_base = np.floor(sy_f).astype(np.int32)
                        in_sm = (
                            (sx_base >= 0)
                            & (sx_base < sz_res - 1)
                            & (sy_base >= 0)
                            & (sy_base < sz_res - 1)
                        )
                        idx_sm = np.nonzero(in_sm)[0]
                        depth_bias = 0.012
                        if len(idx_sm) > 0:
                            z_cmp = p_clip[idx_sm, 2] - depth_bias
                            occ = np.zeros((len(idx_sm),), dtype=np.float64)
                            for ox, oy in ((0, 0), (1, 0), (0, 1), (1, 1)):
                                sz_sample = sz_buf[sy_base[idx_sm] + oy, sx_base[idx_sm] + ox]
                                occ += (z_cmp > sz_sample).astype(np.float64)
                            sun_shadow_factor[idx_sm, 0] = 1.0 - 0.72 * (occ * 0.25)

                # Hemisphere ambient lighting + analytic IBL reflection for metallic surfaces
                hemi_t = 0.5 + 0.5 * nrm_oriented[:, 2:3]
                sky_c = np.asarray(scene.world.zenith_color, dtype=np.float64)[None, :] * 1.6 + 0.16
                gnd_c = (
                    np.asarray(scene.world.horizon_color, dtype=np.float64)[None, :] * 0.75 + 0.06
                )
                amb_col = gnd_c * (1.0 - hemi_t) + sky_c * hemi_t

                ndv = np.clip(np.sum(nrm_oriented * view_dir, axis=1, keepdims=True), 0.0, 1.0)
                f0 = 0.04 * (1.0 - metallic) + albedo_lin * metallic
                fresnel = f0 + (1.0 - f0) * ((1.0 - ndv) ** 5)

                refl = 2.0 * np.sum(nrm_oriented * view_dir, axis=1, keepdims=True) * nrm_oriented
                refl = refl - view_dir
                refl_t = np.clip(0.5 + 0.5 * refl[:, 2:3], 0.0, 1.0)
                env_refl = (
                    np.array([[0.22, 0.24, 0.28]]) * (1.0 - refl_t)
                    + np.array([[0.95, 0.96, 1.0]]) * refl_t
                ) * (0.45 + 0.55 * (1.0 - roughness))
                ibl_spec = (
                    fresnel
                    * env_refl
                    * (0.15 + 0.75 * metallic)
                    * (0.5 + scene.world.ambient_strength)
                )

                total_light = (
                    albedo_lin * amb_col * scene.world.ambient_strength * (1.0 - metallic)
                    + ibl_spec
                )

                shininess = np.clip((1.0 - roughness) ** 2 * 180.0 + 4.0, 4.0, 256.0)

                sun_used = False
                for lobj, lspec in lights:
                    lcol = np.asarray(lspec.color, dtype=np.float64)[None, :] * float(
                        lspec.intensity
                    )
                    if lspec.kind == "sun":
                        ldir = -np.asarray(lspec.direction or (-0.5, 0.4, -0.7), dtype=np.float64)
                        ldir /= max(1e-9, float(np.linalg.norm(ldir)))
                        l_vec = np.broadcast_to(ldir[None, :], (n_px, 3))
                        atten = sun_shadow_factor if (not sun_used and lspec.cast_shadows) else 1.0
                        sun_used = True
                    else:
                        lpos = np.asarray(lobj.transform.location, dtype=np.float64)[None, :]
                        delta = lpos - pos_w
                        dist2 = np.sum(delta * delta, axis=1, keepdims=True)
                        dist = np.sqrt(np.maximum(dist2, 1e-6))
                        l_vec = delta / dist
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

                    ndl = np.clip(np.sum(nrm_oriented * l_vec, axis=1, keepdims=True), 0.0, 1.0)
                    half_v = l_vec + view_dir
                    hv_len = np.linalg.norm(half_v, axis=1, keepdims=True)
                    half_v = half_v / np.where(hv_len < 1e-12, 1.0, hv_len)
                    ndh = np.clip(np.sum(nrm_oriented * half_v, axis=1, keepdims=True), 0.0, 1.0)
                    spec_pow = np.power(ndh, shininess) * (0.25 + 0.75 * (1.0 - roughness))
                    diff = albedo_lin * (1.0 - metallic) * ndl
                    spec = fresnel * spec_pow * ndl * 1.4
                    total_light += (diff + spec) * lcol * atten

                shaded = total_light + emission_lin

                if cfg.mode == "wireframe":
                    epx = edge_px_buf[hit_mask][:, None]
                    wire_mix = np.clip(1.0 - epx / 1.15, 0.0, 1.0)
                    shaded = (
                        shaded * 0.65 * (1.0 - wire_mix) + np.array([[0.12, 0.92, 0.98]]) * wire_mix
                    )

            # Reinhard tone mapping & sRGB gamma curve
            if cfg.mode in ("pbr", "clay", "wireframe"):
                shaded = shaded / (1.0 + shaded * 0.45)
                shaded = np.power(np.clip(shaded, 0.0, 1.0), 1.0 / 2.2)

            bg[hit_mask] = shaded.astype(np.float32)

        # Depth-tested ground grid on XY plane (Z=0)
        if cfg.show_grid and cfg.mode not in ("depth",):
            grid_extent = 5
            for i in range(-grid_extent, grid_extent + 1):
                col_x = (0.72, 0.25, 0.25) if i == 0 else (0.34, 0.36, 0.40)
                col_y = (0.25, 0.72, 0.28) if i == 0 else (0.34, 0.36, 0.40)
                alpha = 0.75 if i == 0 else 0.35
                _draw_segment_z(
                    bg,
                    z_buf,
                    np.array([-grid_extent, float(i), 0.0]),
                    np.array([grid_extent, float(i), 0.0]),
                    vp,
                    col_x,
                    alpha=alpha,
                )
                _draw_segment_z(
                    bg,
                    z_buf,
                    np.array([float(i), -grid_extent, 0.0]),
                    np.array([float(i), grid_extent, 0.0]),
                    vp,
                    col_y,
                    alpha=alpha,
                )

        u8 = np.clip(bg * 255.0, 0, 255).astype(np.uint8)
        return Image.fromarray(u8, mode="RGB")


def image_to_png_bytes(img: Image.Image) -> bytes:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()
