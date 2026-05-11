from __future__ import annotations

import sympy as sm

from multibodysim.flexible.flexible_ns_dynamics import FlexibleNonSymmetricDynamics


def assert_vector_equal(lhs, rhs, frame):
    diff = lhs - rhs
    assert all(sm.simplify(component) == 0 for component in diff.to_matrix(frame))


def build_kinematic_dynamics(config):
    dynamics = FlexibleNonSymmetricDynamics.__new__(FlexibleNonSymmetricDynamics)
    dynamics.config = config
    dynamics.graph = config["adjacency_graph"]
    dynamics.body_names = config["body_names"]
    dynamics.central_body = config["central_body"]
    dynamics.body_type = config["body_type"]
    dynamics.enable_gravity_gradient = config["enable_gravity_gradient"]
    dynamics.parents = dynamics._parents_from_adjacency(
        dynamics.graph,
        dynamics.body_names,
        dynamics.central_body,
    )
    dynamics.rigid_bodies = {
        name: None
        for name in dynamics.body_names
        if not dynamics.body_type[name].startswith("flexible-")
    }
    dynamics.flexible_bodies = {
        name: None
        for name in dynamics.body_names
        if dynamics.body_type[name].startswith("flexible-")
    }

    dynamics._define_symbols()
    dynamics._define_mode_shapes()
    dynamics._define_kinematics()
    dynamics._define_kinematic_equations()
    dynamics._setup_velocities()

    return dynamics


def test_flexible_parent_distal_joint_includes_modal_tip_velocity(
    single_angle_seven_part_config,
):
    dynamics = build_kinematic_dynamics(single_angle_seven_part_config)

    panel = "panel_2"
    frame = dynamics.frames[panel]
    joint = dynamics.points["joint_bus_1_panel_2"]
    expected_relative_velocity = dynamics._flexible_tip_velocity_sum(panel) * frame.y

    assert_vector_equal(joint.vel(frame), expected_relative_velocity, frame)


def test_flexible_parent_distal_joint_inertial_velocity_includes_tip_velocity(
    single_angle_seven_part_config,
):
    dynamics = build_kinematic_dynamics(single_angle_seven_part_config)
    inertial_frame = dynamics.frames["inertial"]

    panel = "panel_2"
    frame = dynamics.frames[panel]
    reference_joint = dynamics.points["joint_panel_2_bus_2"]
    distal_joint = dynamics.points["joint_bus_1_panel_2"]
    expected = (
        reference_joint.vel(inertial_frame)
        + dynamics._flexible_tip_velocity_sum(panel) * frame.y
        + frame.ang_vel_in(inertial_frame).cross(
            distal_joint.pos_from(reference_joint)
        )
    )

    assert_vector_equal(distal_joint.vel(inertial_frame), expected, inertial_frame)
