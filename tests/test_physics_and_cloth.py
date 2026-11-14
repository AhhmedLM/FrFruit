"""Tests for rigid-body impulse physics and XPBD cloth simulation with wind & colliders."""

from __future__ import annotations

import numpy as np

from frfruit.animation.evaluator import apply_animation_frame
from frfruit.core.scene import ClothSpec, RigidBodySpec, SceneDocument, SceneObject
from frfruit.core.types import Transform
from frfruit.geometry.primitives import create_primitive_mesh
from frfruit.geometry.spatial import evaluate_object_mesh
from frfruit.physics.solver import bake_scene_physics, clear_physics_caches


def test_rigid_body_gravity_floor_bounce_and_pairwise_collision() -> None:
    sc = SceneDocument.new_default("RigidTest")
    sc.timeline.fps = 24
    sc.timeline.frame_start = 1
    sc.timeline.frame_end = 36

    s1 = create_primitive_mesh("uv_sphere", radius=0.4, segments=14)
    s2 = create_primitive_mesh("uv_sphere", radius=0.4, segments=14)

    sc.objects["ball_drop"] = SceneObject(
        id="ball_drop",
        name="DropBall",
        kind="mesh",
        transform=Transform(location=(0.0, 0.0, 2.8)),
        mesh=s1.to_payload(),
        rigid_body=RigidBodySpec(
            enabled=True,
            body_type="dynamic",
            mass=1.0,
            restitution=0.6,
            friction=0.4,
            collider_shape="sphere",
        ),
    )
    sc.objects["ball_floor"] = SceneObject(
        id="ball_floor",
        name="GroundBall",
        kind="mesh",
        transform=Transform(location=(0.15, 0.0, 0.4)),
        mesh=s2.to_payload(),
        rigid_body=RigidBodySpec(
            enabled=True,
            body_type="dynamic",
            mass=1.0,
            restitution=0.6,
            friction=0.4,
            initial_velocity=(0.5, 0.0, 0.0),
            collider_shape="sphere",
        ),
    )

    res = bake_scene_physics(sc, ground_z=0.0)
    assert res["rigid_bodies"]["simulated_bodies"] == 2
    assert len(res["rigid_bodies"]["dynamic_bodies_baked"]) == 2
    assert len(sc.timeline.tracks) >= 6

    # Drop ball must fall from Z=2.8 to floor (~0.4 radius above z=0.0)
    end_doc = apply_animation_frame(sc, 36)
    ev1 = evaluate_object_mesh(end_doc, "ball_drop")
    assert ev1.bounds.min_corner[2] >= -0.05  # Never falls through floor
    assert ev1.bounds.center[2] < 2.2  # Fell significantly under gravity

    # Ground ball with +X initial velocity moves horizontally
    ev2 = evaluate_object_mesh(end_doc, "ball_floor")
    assert ev2.bounds.center[0] > 0.25


def test_xpbd_cloth_drapes_over_sphere_and_reacts_to_wind() -> None:
    sc = SceneDocument.new_default("ClothTest")
    sc.timeline.fps = 24
    sc.timeline.frame_start = 1
    sc.timeline.frame_end = 24

    orb = create_primitive_mesh("uv_sphere", radius=0.75, segments=18)
    sc.objects["sphere_obs"] = SceneObject(
        id="sphere_obs",
        name="SphereObstacle",
        kind="mesh",
        transform=Transform(location=(0.0, 0.0, 0.85)),
        mesh=orb.to_payload(),
        rigid_body=RigidBodySpec(enabled=True, body_type="static", collider_shape="sphere"),
    )

    drape = create_primitive_mesh("plane", size=2.0, segments=10)
    sc.objects["drape_cloth"] = SceneObject(
        id="drape_cloth",
        name="DrapeCloth",
        kind="mesh",
        transform=Transform(location=(0.0, 0.0, 1.85)),
        mesh=drape.to_payload(),
        cloth=ClothSpec(
            enabled=True,
            pin_mode="none",
            stiffness=0.92,
            bend_stiffness=0.25,
            wind=(0.0, 0.0, 0.0),
            collision_margin=0.04,
            substeps=8,
        ),
    )

    banner = create_primitive_mesh("plane", size=1.4, segments=8)
    sc.objects["wind_banner"] = SceneObject(
        id="wind_banner",
        name="WindBanner",
        kind="mesh",
        transform=Transform(location=(3.5, 0.0, 1.8), rotation_deg=(90.0, 0.0, 0.0)),
        mesh=banner.to_payload(),
        cloth=ClothSpec(
            enabled=True,
            pin_mode="top_corners",
            stiffness=0.94,
            wind=(0.0, 3.5, 0.0),
            substeps=8,
        ),
    )

    res = bake_scene_physics(sc, ground_z=0.0)
    assert len(res["cloth_objects"]) == 2
    assert len(sc.objects["drape_cloth"].vertex_cache) == 24
    assert len(sc.objects["wind_banner"].vertex_cache) == 24

    # Evaluate cloth at frame 24
    f24 = apply_animation_frame(sc, 24)
    ev_drape = evaluate_object_mesh(f24, "drape_cloth")
    # Center drapes onto top of sphere (~0.85+0.75=1.60), corners hang much lower (< 1.5)
    z_min = float(ev_drape.world_mesh.vertices[:, 2].min())
    z_max = float(ev_drape.world_mesh.vertices[:, 2].max())
    assert z_max > 1.45 and z_min < z_max - 0.15

    # None of the drape cloth vertices penetrate inside the sphere obstacle center
    dists = np.linalg.norm(ev_drape.world_mesh.vertices - np.array([[0.0, 0.0, 0.85]]), axis=1)
    assert float(dists.min()) >= 0.72

    # Wind banner blows toward +Y at free vertices while keeping top pins anchored
    ev_banner_start = evaluate_object_mesh(apply_animation_frame(sc, 1), "wind_banner")
    ev_banner_end = evaluate_object_mesh(f24, "wind_banner")
    assert (
        float(ev_banner_end.world_mesh.vertices[:, 1].mean())
        > float(ev_banner_start.world_mesh.vertices[:, 1].mean()) + 0.05
    )

    # Clear cache
    clr = clear_physics_caches(sc)
    assert len(clr["cleared_vertex_caches"]) == 2
    assert not sc.objects["drape_cloth"].vertex_cache
