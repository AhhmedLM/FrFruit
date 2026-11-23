"""Optional C-accelerated triangle G-buffer rasterizer with automatic pure-Python fallback."""

from __future__ import annotations

import ctypes
import hashlib
import os
import shutil
import subprocess
from pathlib import Path

import numpy as np

_C_SOURCE = r"""
#include <math.h>
#include <stdint.h>

void rasterize_tris_c(
    int32_t n_cand,
    const int32_t *cand_idx,
    const double *x0_all, const double *y0_all, const double *z0_all, const double *w0_all,
    const double *x1_all, const double *y1_all, const double *z1_all, const double *w1_all,
    const double *x2_all, const double *y2_all, const double *z2_all, const double *w2_all,
    int32_t width, int32_t height, int32_t cull_backfaces,
    float *z_buf, int32_t *tri_id_buf, float *u_buf, float *v_buf, float *edge_px_buf
) {
    for (int32_t ci = 0; ci < n_cand; ++ci) {
        int32_t ti = cand_idx[ci];
        double ax = x0_all[ti], ay = y0_all[ti], az = z0_all[ti], aw = w0_all[ti];
        double bx = x1_all[ti], by = y1_all[ti], bz = z1_all[ti], bw = w1_all[ti];
        double cx = x2_all[ti], cy = y2_all[ti], cz = z2_all[ti], cw = w2_all[ti];

        double a2 = (bx - ax) * (cy - ay) - (by - ay) * (cx - ax);
        if (cull_backfaces) {
            if (a2 >= -1e-3) continue;
        } else {
            if (fabs(a2) <= 1e-3) continue;
        }

        double minx = fmin(ax, fmin(bx, cx));
        double maxx = fmax(ax, fmax(bx, cx));
        double miny = fmin(ay, fmin(by, cy));
        double maxy = fmax(ay, fmax(by, cy));
        if (maxx <= 0.0 || minx >= (double)width || maxy <= 0.0 || miny >= (double)height) {
            continue;
        }

        int32_t ix0 = (int32_t)floor(minx); if (ix0 < 0) ix0 = 0;
        int32_t ix1 = (int32_t)ceil(maxx);  if (ix1 > width) ix1 = width;
        int32_t iy0 = (int32_t)floor(miny); if (iy0 < 0) iy0 = 0;
        int32_t iy1 = (int32_t)ceil(maxy);  if (iy1 > height) iy1 = height;
        if (ix1 <= ix0 || iy1 <= iy0) continue;

        double inv_a2 = 1.0 / a2;
        double inv_w0 = 1.0 / (fabs(aw) > 1e-6 ? aw : 1e-6);
        double inv_w1 = 1.0 / (fabs(bw) > 1e-6 ? bw : 1e-6);
        double inv_w2 = 1.0 / (fabs(cw) > 1e-6 ? cw : 1e-6);

        double abs_a2 = fabs(a2);
        double l0 = hypot(cx - bx, cy - by) + 1e-6;
        double l1 = hypot(ax - cx, ay - cy) + 1e-6;
        double l2 = hypot(bx - ax, by - ay) + 1e-6;
        double s_e0 = abs_a2 / l0;
        double s_e1 = abs_a2 / l1;
        double s_e2 = abs_a2 / l2;

        double d_cb_x = cx - bx, d_cb_y = cy - by;
        double d_ac_x = ax - cx, d_ac_y = ay - cy;

        for (int32_t y = iy0; y < iy1; ++y) {
            double py = (double)y + 0.5;
            double e0_base = d_cb_x * (py - by);
            double e1_base = d_ac_x * (py - cy);
            int32_t row_off = y * width;

            for (int32_t x = ix0; x < ix1; ++x) {
                double px = (double)x + 0.5;
                double e0 = (e0_base - d_cb_y * (px - bx)) * inv_a2;
                if (e0 < -1e-4) continue;
                double e1 = (e1_base - d_ac_y * (px - cx)) * inv_a2;
                if (e1 < -1e-4) continue;
                double e2 = 1.0 - e0 - e1;
                if (e2 < -1e-4) continue;

                float z_interp = (float)(e0 * az + e1 * bz + e2 * cz);
                if (z_interp < -1.0f || z_interp > 1.0f) continue;

                int32_t idx = row_off + x;
                if (z_interp < z_buf[idx]) {
                    z_buf[idx] = z_interp;
                    tri_id_buf[idx] = ti;

                    double pw0 = e0 * inv_w0;
                    double pw1 = e1 * inv_w1;
                    double pw2 = e2 * inv_w2;
                    double psum = pw0 + pw1 + pw2;
                    double inv_sum = (fabs(psum) < 1e-12) ? 1.0 : (1.0 / psum);
                    u_buf[idx] = (float)(pw1 * inv_sum);
                    v_buf[idx] = (float)(pw2 * inv_sum);

                    double d0 = e0 * s_e0;
                    double d1 = e1 * s_e1;
                    double d2 = e2 * s_e2;
                    double dm = d0 < d1 ? (d0 < d2 ? d0 : d2) : (d1 < d2 ? d1 : d2);
                    edge_px_buf[idx] = (float)dm;
                }
            }
        }
    }
}
"""

_C_LIB: ctypes.CDLL | None = None
_C_INIT_DONE = False


def _load_c_rasterizer() -> ctypes.CDLL | None:
    global _C_LIB, _C_INIT_DONE
    if _C_INIT_DONE:
        return _C_LIB
    _C_INIT_DONE = True

    if os.environ.get("FRFRUIT_DISABLE_C_RASTER", "0") == "1":
        return None
    cc = shutil.which("gcc") or shutil.which("clang")
    if not cc:
        return None

    h = hashlib.sha1(_C_SOURCE.encode("utf-8")).hexdigest()[:12]
    cache_dir = Path.home() / ".cache" / "frfruit" / "native"
    cache_dir.mkdir(parents=True, exist_ok=True)
    so_path = cache_dir / f"fast_raster_{h}.so"
    if not so_path.exists():
        c_path = cache_dir / f"fast_raster_{h}.c"
        tmp_so = cache_dir / f"fast_raster_{h}_{os.getpid()}.so"
        try:
            c_path.write_text(_C_SOURCE, encoding="utf-8")
            proc = subprocess.run(
                [cc, "-O3", "-shared", "-fPIC", str(c_path), "-lm", "-o", str(tmp_so)],
                capture_output=True,
                check=False,
            )
            if proc.returncode != 0 or not tmp_so.exists():
                return None
            tmp_so.replace(so_path)
        except Exception:
            return None

    try:
        lib = ctypes.CDLL(str(so_path))
        d_ptr = ctypes.POINTER(ctypes.c_double)
        f_ptr = ctypes.POINTER(ctypes.c_float)
        i_ptr = ctypes.POINTER(ctypes.c_int32)
        lib.rasterize_tris_c.argtypes = [
            ctypes.c_int32,
            i_ptr,
            d_ptr,
            d_ptr,
            d_ptr,
            d_ptr,
            d_ptr,
            d_ptr,
            d_ptr,
            d_ptr,
            d_ptr,
            d_ptr,
            d_ptr,
            d_ptr,
            ctypes.c_int32,
            ctypes.c_int32,
            ctypes.c_int32,
            f_ptr,
            i_ptr,
            f_ptr,
            f_ptr,
            f_ptr,
        ]
        lib.rasterize_tris_c.restype = None
        _C_LIB = lib
    except Exception:
        _C_LIB = None
    return _C_LIB


def rasterize_candidates_fast(
    cand_idx: np.ndarray,
    x0_all: np.ndarray,
    y0_all: np.ndarray,
    z0_all: np.ndarray,
    w0_all: np.ndarray,
    x1_all: np.ndarray,
    y1_all: np.ndarray,
    z1_all: np.ndarray,
    w1_all: np.ndarray,
    x2_all: np.ndarray,
    y2_all: np.ndarray,
    z2_all: np.ndarray,
    w2_all: np.ndarray,
    width: int,
    height: int,
    cull_backfaces: bool,
    z_buf: np.ndarray,
    tri_id_buf: np.ndarray,
    u_buf: np.ndarray,
    v_buf: np.ndarray,
    edge_px_buf: np.ndarray,
) -> bool:
    """Run C-accelerated triangle rasterization if available; returns True if executed."""
    lib = _load_c_rasterizer()
    if lib is None or len(cand_idx) == 0:
        return False

    c_cand = np.ascontiguousarray(cand_idx, dtype=np.int32)
    x0 = np.ascontiguousarray(x0_all, dtype=np.float64)
    y0 = np.ascontiguousarray(y0_all, dtype=np.float64)
    z0 = np.ascontiguousarray(z0_all, dtype=np.float64)
    w0 = np.ascontiguousarray(w0_all, dtype=np.float64)
    x1 = np.ascontiguousarray(x1_all, dtype=np.float64)
    y1 = np.ascontiguousarray(y1_all, dtype=np.float64)
    z1 = np.ascontiguousarray(z1_all, dtype=np.float64)
    w1 = np.ascontiguousarray(w1_all, dtype=np.float64)
    x2 = np.ascontiguousarray(x2_all, dtype=np.float64)
    y2 = np.ascontiguousarray(y2_all, dtype=np.float64)
    z2 = np.ascontiguousarray(z2_all, dtype=np.float64)
    w2 = np.ascontiguousarray(w2_all, dtype=np.float64)

    d_p = ctypes.POINTER(ctypes.c_double)
    f_p = ctypes.POINTER(ctypes.c_float)
    i_p = ctypes.POINTER(ctypes.c_int32)

    lib.rasterize_tris_c(
        len(c_cand),
        c_cand.ctypes.data_as(i_p),
        x0.ctypes.data_as(d_p),
        y0.ctypes.data_as(d_p),
        z0.ctypes.data_as(d_p),
        w0.ctypes.data_as(d_p),
        x1.ctypes.data_as(d_p),
        y1.ctypes.data_as(d_p),
        z1.ctypes.data_as(d_p),
        w1.ctypes.data_as(d_p),
        x2.ctypes.data_as(d_p),
        y2.ctypes.data_as(d_p),
        z2.ctypes.data_as(d_p),
        w2.ctypes.data_as(d_p),
        int(width),
        int(height),
        1 if cull_backfaces else 0,
        z_buf.ctypes.data_as(f_p),
        tri_id_buf.ctypes.data_as(i_p),
        u_buf.ctypes.data_as(f_p),
        v_buf.ctypes.data_as(f_p),
        edge_px_buf.ctypes.data_as(f_p),
    )
    return True
