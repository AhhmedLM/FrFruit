"""Non-destructive modifier stack evaluator."""

from __future__ import annotations

import math
from collections.abc import Callable

import numpy as np

from frfruit.core.math3d import compose_matrix, transform_points
from frfruit.core.scene import (
    ArrayModifier,
    BendModifier,
    BevelModifier,
    BooleanModifier,
    MirrorModifier,
    SceneDocument,
    SceneObject,
    SmoothModifier,
    SolidifyModifier,
    SubdivisionModifier,
    TaperModifier,
    TwistModifier,
)
from frfruit.geometry.edit_ops import (
    bevel_edges,
    boolean_meshes,
    repair_manifold,
    smooth_laplacian,
    subdivide_mesh,
)
from frfruit.geometry.mesh import GeometryError, MeshData, concat_meshes


def _apply_mirror(mesh: MeshData, mod: MirrorModifier) -> MeshData:
    if mesh.is_empty:
        return mesh
    ax = {"x": 0, "y": 1, "z": 2}[mod.axis]
    verts_m = mesh.vertices.copy()
    verts_m[:, ax] = 2.0 * mod.offset - verts_m[:, ax]
    # Flip triangle winding so mirrored faces preserve outward normals
    faces_m = mesh.faces[:, [0, 2, 1]].copy()
    mirrored = MeshData(vertices=verts_m, faces=faces_m)
    if mod.bisect:
        # Keep only positive side of plane before mirroring
        keep_orig = np.all(mesh.vertices[mesh.faces][:, :, ax] >= mod.offset - 1e-6, axis=1)
        orig_trimmed = MeshData(vertices=mesh.vertices.copy(), faces=mesh.faces[keep_orig])
        return repair_manifold(concat_meshes([orig_trimmed, mirrored]))
    return concat_meshes([mesh, mirrored])


def _apply_array(mesh: MeshData, mod: ArrayModifier) -> MeshData:
    if mesh.is_empty or mod.count <= 1:
        return mesh
    copies: list[MeshData] = []
    if mod.mode == "linear":
        ox, oy, oz = mod.offset
        for i in range(mod.count):
            v = mesh.vertices.copy()
            v[:, 0] += i * ox
            v[:, 1] += i * oy
            v[:, 2] += i * oz
            copies.append(MeshData(vertices=v, faces=mesh.faces.copy(), uvs=mesh.uvs))
    else:
        cx, cy, cz = mod.radial_center
        center = np.array([cx, cy, cz], dtype=np.float64)
        step = mod.radial_angle_deg / mod.count
        for i in range(mod.count):
            ang = i * step
            rot_deg = (
                (ang, 0.0, 0.0)
                if mod.radial_axis == "x"
                else ((0.0, ang, 0.0) if mod.radial_axis == "y" else (0.0, 0.0, ang))
            )
            mat = compose_matrix(rotation_deg=rot_deg)
            v = transform_points(mat, mesh.vertices - center[None, :]) + center[None, :]
            copies.append(MeshData(vertices=v, faces=mesh.faces.copy(), uvs=mesh.uvs))
    return concat_meshes(copies)


def _apply_solidify(mesh: MeshData, mod: SolidifyModifier) -> MeshData:
    if mesh.is_empty:
        return mesh
    vn = mesh.vertex_normals()
    n_verts = len(mesh.vertices)
    inner_off = (mod.offset - 0.5) * mod.thickness
    outer_off = (mod.offset + 0.5) * mod.thickness

    v_inner = mesh.vertices + vn * inner_off
    v_outer = mesh.vertices + vn * outer_off
    verts = np.vstack([v_outer, v_inner])

    faces_outer = mesh.faces.copy()
    faces_inner = mesh.faces[:, [0, 2, 1]] + n_verts

    # Find boundary directed edges (u -> v) where (v -> u) doesn't exist
    directed: set[tuple[int, int]] = set()
    for a, b, c in mesh.faces:
        directed.add((int(a), int(b)))
        directed.add((int(b), int(c)))
        directed.add((int(c), int(a)))
    boundary = [(u, v) for (u, v) in directed if (v, u) not in directed]

    side_faces: list[list[int]] = []
    for u, v in boundary:
        ui = u + n_verts
        vi = v + n_verts
        side_faces.append([v, u, ui])
        side_faces.append([v, ui, vi])

    all_faces = [faces_outer, faces_inner]
    if side_faces:
        all_faces.append(np.asarray(side_faces, dtype=np.int64))
    return MeshData(vertices=verts, faces=np.vstack(all_faces))


def _apply_taper(mesh: MeshData, mod: TaperModifier) -> MeshData:
    if mesh.is_empty:
        return mesh
    ax = {"x": 0, "y": 1, "z": 2}[mod.axis]
    other = [i for i in range(3) if i != ax]
    verts = mesh.vertices.copy()
    mn = float(np.min(verts[:, ax]))
    mx = float(np.max(verts[:, ax]))
    span = mx - mn
    if span < 1e-8:
        return mesh
    t = (verts[:, ax] - mn) / span
    s = np.maximum(0.05, 1.0 + mod.factor * (t - 0.5) * 2.0)
    center_other = np.mean(verts[:, other], axis=0)
    verts[:, other[0]] = center_other[0] + (verts[:, other[0]] - center_other[0]) * s
    verts[:, other[1]] = center_other[1] + (verts[:, other[1]] - center_other[1]) * s
    return MeshData(vertices=verts, faces=mesh.faces.copy(), uvs=mesh.uvs)


def _apply_twist(mesh: MeshData, mod: TwistModifier) -> MeshData:
    if mesh.is_empty:
        return mesh
    ax = {"x": 0, "y": 1, "z": 2}[mod.axis]
    o1, o2 = [i for i in range(3) if i != ax]
    verts = mesh.vertices.copy()
    mn = float(np.min(verts[:, ax]))
    mx = float(np.max(verts[:, ax]))
    span = mx - mn
    if span < 1e-8:
        return mesh
    t = (verts[:, ax] - mn) / span - 0.5
    ang = t * math.radians(mod.angle_deg)
    c = np.cos(ang)
    s = np.sin(ang)
    p1 = verts[:, o1].copy()
    p2 = verts[:, o2].copy()
    verts[:, o1] = p1 * c - p2 * s
    verts[:, o2] = p1 * s + p2 * c
    return MeshData(vertices=verts, faces=mesh.faces.copy(), uvs=mesh.uvs)


def _apply_bend(mesh: MeshData, mod: BendModifier) -> MeshData:
    if mesh.is_empty or abs(mod.angle_deg) < 1e-6:
        return mesh
    ax = {"x": 0, "y": 1, "z": 2}[mod.axis]
    bax = {"x": 0, "y": 1, "z": 2}[mod.bend_axis]
    if ax == bax:
        bax = (ax + 2) % 3
    verts = mesh.vertices.copy()
    mn = float(np.min(verts[:, ax]))
    mx = float(np.max(verts[:, ax]))
    span = mx - mn
    if span < 1e-8:
        return mesh
    theta_total = math.radians(mod.angle_deg)
    r = span / theta_total
    mid = 0.5 * (mn + mx)
    t = (verts[:, ax] - mid) / r
    dist = r - verts[:, bax]
    verts[:, ax] = mid + dist * np.sin(t)
    verts[:, bax] = r - dist * np.cos(t)
    return MeshData(vertices=verts, faces=mesh.faces.copy(), uvs=mesh.uvs)


def _apply_displace(mesh: MeshData, strength: float, scale: float, seed: int) -> MeshData:
    if mesh.is_empty:
        return mesh
    vn = mesh.vertex_normals()
    verts = mesh.vertices.copy()
    rng = np.random.default_rng(seed)
    phase = rng.uniform(0.0, 2.0 * math.pi, size=(3,))
    freq = scale * math.pi
    disp = (
        np.sin(verts[:, 0] * freq + phase[0])
        * np.cos(verts[:, 1] * freq + phase[1])
        * np.sin(verts[:, 2] * freq + phase[2] + 0.7)
    )
    verts = verts + vn * (disp[:, None] * strength)
    return MeshData(vertices=verts, faces=mesh.faces.copy(), uvs=mesh.uvs)


def evaluate_modifier_stack(
    obj: SceneObject,
    base_mesh: MeshData,
    scene: SceneDocument,
    world_matrix_lookup: Callable[[str], np.ndarray] | None = None,
    raw_mesh_lookup: Callable[[str], MeshData | None] | None = None,
    _visited: set[str] | None = None,
) -> MeshData:
    """Evaluate an object's ordered non-destructive modifier stack in local space."""
    cur = base_mesh.copy()
    visited = set(_visited or set())
    visited.add(obj.id)

    for mod in obj.modifiers:
        if not mod.enabled:
            continue
        if isinstance(mod, MirrorModifier):
            cur = _apply_mirror(cur, mod)
        elif isinstance(mod, ArrayModifier):
            cur = _apply_array(cur, mod)
        elif isinstance(mod, SolidifyModifier):
            cur = _apply_solidify(cur, mod)
        elif isinstance(mod, SubdivisionModifier):
            cur = subdivide_mesh(
                cur,
                levels=mod.levels,
                smooth=mod.smooth,
                algorithm=mod.algorithm,
                crease_angle_deg=mod.crease_angle_deg,
            )
        elif isinstance(mod, BevelModifier):
            cur = bevel_edges(
                cur,
                angle_threshold_deg=mod.angle_threshold_deg,
                width=mod.width,
                segments=mod.segments,
            )
        elif isinstance(mod, TaperModifier):
            cur = _apply_taper(cur, mod)
        elif isinstance(mod, TwistModifier):
            cur = _apply_twist(cur, mod)
        elif isinstance(mod, BendModifier):
            cur = _apply_bend(cur, mod)
        elif isinstance(mod, SmoothModifier):
            cur = smooth_laplacian(cur, iterations=mod.iterations, factor=mod.factor)
        elif isinstance(mod, BooleanModifier):
            target = scene.resolve_object(mod.target_object_id)
            if target is None:
                raise GeometryError(
                    f"Boolean modifier references missing object {mod.target_object_id!r}."
                )
            if target.id in visited:
                raise GeometryError(
                    f"Cyclic boolean dependency between {obj.id!r} and {target.id!r}."
                )
            if raw_mesh_lookup is not None:
                target_mesh = raw_mesh_lookup(target.id)
            elif target.mesh is not None:
                target_mesh = MeshData.from_payload(target.mesh)
            else:
                target_mesh = None
            if target_mesh is None or target_mesh.is_empty:
                raise GeometryError(f"Boolean target {target.name!r} has no mesh geometry.")

            # Transform target mesh into obj's local space
            if world_matrix_lookup is not None:
                mat_self = world_matrix_lookup(obj.id)
                mat_tgt = world_matrix_lookup(target.id)
                rel_mat = np.linalg.inv(mat_self) @ mat_tgt
                target_local = MeshData(
                    vertices=transform_points(rel_mat, target_mesh.vertices),
                    faces=target_mesh.faces.copy(),
                )
            else:
                target_local = target_mesh

            cur = boolean_meshes(cur, target_local, mod.operation)
        else:
            # DisplaceModifier
            cur = _apply_displace(cur, mod.strength, mod.scale, mod.seed)
    return cur
