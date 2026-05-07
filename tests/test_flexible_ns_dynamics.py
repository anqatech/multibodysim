from __future__ import annotations

import sympy as sm

from multibodysim.flexible.flexible_ns_dynamics import FlexibleNonSymmetricDynamics


def assert_vector_equal(lhs, rhs, frame):
    diff = lhs - rhs
    assert all(sm.simplify(component) == 0 for component in diff.to_matrix(frame))


def seven_part_config():
    return {
        "body_names": [
            "bus_1",
            "bus_2",
            "bus_3",
            "panel_1",
            "panel_2",
            "panel_3",
            "panel_4",
        ],
        "central_body": "bus_2",
        "enable_gravity_gradient": False,
        "adjacency_graph": {
            "bus_1": ["panel_1", "panel_2"],
            "bus_2": ["panel_2", "panel_3"],
            "bus_3": ["panel_3", "panel_4"],
            "panel_1": ["bus_1"],
            "panel_2": ["bus_1", "bus_2"],
            "panel_3": ["bus_2", "bus_3"],
            "panel_4": ["bus_3"],
        },
        "body_type": {
            "bus_1": "rigid-left",
            "bus_2": "rigid-central",
            "bus_3": "rigid-right",
            "panel_1": "flexible-left",
            "panel_2": "flexible-left",
            "panel_3": "flexible-right",
            "panel_4": "flexible-right",
        },
        "flexible_types": {
            "panel_1": "cantilever",
            "panel_2": "cantilever",
            "panel_3": "cantilever",
            "panel_4": "cantilever",
        },
        "beam_parameters": {
            "cantilever": {"nb_modes": 1, "nb_points": 50},
        },
        "p_values": {
            "D": 1.0,
            "L": 3.0,
            "m_bus_1": 3.0,
            "m_bus_2": 3.0,
            "m_bus_3": 3.0,
            "m_panel_1": 2.0,
            "m_panel_2": 2.0,
            "m_panel_3": 2.0,
            "m_panel_4": 2.0,
            "E_mod": 140e9,
            "I_area": 2.5e-8,
            "planet_mu": 3.986004418e14,
            "orbit_semi_major_axis": 6778000.0,
            "orbit_eccentricity": 0.0,
        },
    }


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


def test_flexible_parent_distal_joint_includes_modal_tip_velocity():
    dynamics = build_kinematic_dynamics(seven_part_config())

    panel = "panel_2"
    frame = dynamics.frames[panel]
    joint = dynamics.points["joint_bus_1_panel_2"]
    expected_relative_velocity = dynamics._flexible_tip_velocity_sum(panel) * frame.y

    assert_vector_equal(joint.vel(frame), expected_relative_velocity, frame)


def test_flexible_parent_distal_joint_inertial_velocity_includes_tip_velocity():
    dynamics = build_kinematic_dynamics(seven_part_config())
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
