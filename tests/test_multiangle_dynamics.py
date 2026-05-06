from __future__ import annotations

import sympy as sm
import pytest

from multibodysim.beam.cantilever_beam import CantileverBeam
from multibodysim.beam.clamped_clamped_beam import ClampedClampedBeam
from multibodysim.multiangle import MultiAngleFlexibleDynamics


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
            "panel_2": "clamped-clamped",
            "panel_3": "clamped-clamped",
            "panel_4": "cantilever",
        },
        "beam_parameters": {
            "cantilever": {"nb_modes": 1},
            "clamped-clamped": {"nb_modes": 2},
        },
        "parameters": {
            "D": 1.0,
            "L": 3.0,
            "m_bus_1": 3.0,
            "m_bus_2": 3.0,
            "m_bus_3": 3.0,
            "m_panel_1": 2.0,
            "m_panel_2": 2.0,
            "m_panel_3": 30.0,
            "m_panel_4": 30.0,
            "E_mod": 140e9,
            "I_area": 2.5e-8,
            "planet_mu": 3.986004418e14,
            "orbit_semi_major_axis": 6778000.0,
            "orbit_eccentricity": 0.0,
        },
    }


def eleven_part_config():
    return {
        "body_names": [
            "bus_1",
            "bus_2",
            "bus_3",
            "bus_4",
            "bus_5",
            "panel_1",
            "panel_2",
            "panel_3",
            "panel_4",
            "panel_5",
            "panel_6",
        ],
        "central_body": "bus_3",
        "adjacency_graph": {
            "bus_1": ["panel_1", "panel_2"],
            "bus_2": ["panel_2", "panel_3"],
            "bus_3": ["panel_3", "panel_4"],
            "bus_4": ["panel_4", "panel_5"],
            "bus_5": ["panel_5", "panel_6"],
            "panel_1": ["bus_1"],
            "panel_2": ["bus_1", "bus_2"],
            "panel_3": ["bus_2", "bus_3"],
            "panel_4": ["bus_3", "bus_4"],
            "panel_5": ["bus_4", "bus_5"],
            "panel_6": ["bus_5"],
        },
        "body_type": {
            "bus_1": "rigid-left",
            "bus_2": "rigid-left",
            "bus_3": "rigid-central",
            "bus_4": "rigid-right",
            "bus_5": "rigid-right",
            "panel_1": "flexible-left",
            "panel_2": "flexible-left",
            "panel_3": "flexible-left",
            "panel_4": "flexible-right",
            "panel_5": "flexible-right",
            "panel_6": "flexible-right",
        },
        "flexible_types": {
            "panel_1": "cantilever",
            "panel_2": "cantilever",
            "panel_3": "cantilever",
            "panel_4": "cantilever",
            "panel_5": "cantilever",
            "panel_6": "cantilever",
        },
        "beam_parameters": {
            "cantilever": {"nb_modes": 1},
        },
        "parameters": {
            "D": 1.0,
            "L": 3.0,
            "m_bus_1": 3.0,
            "m_bus_2": 3.0,
            "m_bus_3": 3.0,
            "m_bus_4": 3.0,
            "m_bus_5": 3.0,
            "m_panel_1": 2.0,
            "m_panel_2": 2.0,
            "m_panel_3": 2.0,
            "m_panel_4": 2.0,
            "m_panel_5": 2.0,
            "m_panel_6": 2.0,
            "E_mod": 140e9,
            "I_area": 2.5e-8,
            "planet_mu": 3.986004418e14,
            "orbit_semi_major_axis": 6778000.0,
            "orbit_eccentricity": 0.0,
        },
    }


def assert_symbolic_equal(lhs, rhs):
    assert sm.simplify(lhs - rhs) == 0


def assert_vector_equal(lhs, rhs, frame):
    diff = lhs - rhs
    assert all(sm.simplify(component) == 0 for component in diff.to_matrix(frame))


def test_multiangle_bus_orientation_convention_for_seven_part_chain():
    dynamics = MultiAngleFlexibleDynamics(seven_part_config())

    q31 = dynamics.bus_angle_coordinates["bus_1"]
    q32 = dynamics.bus_angle_coordinates["bus_2"]
    q33 = dynamics.bus_angle_coordinates["bus_3"]

    assert str(q31.func) == "q3_1"
    assert str(q32.func) == "q3_2"
    assert str(q33.func) == "q3_3"

    assert_symbolic_equal(dynamics.orientation_angle("bus_1"), q32 + sm.pi + q31)
    assert_symbolic_equal(dynamics.orientation_angle("bus_2"), q32)
    assert_symbolic_equal(dynamics.orientation_angle("bus_3"), q32 + q33)


def test_multiangle_panel_orientation_uses_endpoint_bus_average_when_available():
    dynamics = MultiAngleFlexibleDynamics(seven_part_config())

    q31 = dynamics.bus_angle_coordinates["bus_1"]
    q32 = dynamics.bus_angle_coordinates["bus_2"]
    q33 = dynamics.bus_angle_coordinates["bus_3"]

    assert_symbolic_equal(dynamics.orientation_angle("panel_1"), q32 + sm.pi + q31)
    assert_symbolic_equal(
        dynamics.orientation_angle("panel_2"),
        q32 + sm.pi / 2 + q31 / 2,
    )
    assert_symbolic_equal(
        dynamics.orientation_angle("panel_3"),
        q32 + q33 / 2,
    )
    assert_symbolic_equal(dynamics.orientation_angle("panel_4"), q32 + q33)


def test_multiangle_convention_scales_to_more_buses():
    dynamics = MultiAngleFlexibleDynamics(eleven_part_config())

    q31 = dynamics.bus_angle_coordinates["bus_1"]
    q32 = dynamics.bus_angle_coordinates["bus_2"]
    q33 = dynamics.bus_angle_coordinates["bus_3"]
    q34 = dynamics.bus_angle_coordinates["bus_4"]
    q35 = dynamics.bus_angle_coordinates["bus_5"]

    assert_symbolic_equal(dynamics.orientation_angle("bus_1"), q33 + sm.pi + q31)
    assert_symbolic_equal(dynamics.orientation_angle("bus_2"), q33 + sm.pi + q32)
    assert_symbolic_equal(dynamics.orientation_angle("bus_3"), q33)
    assert_symbolic_equal(dynamics.orientation_angle("bus_4"), q33 + q34)
    assert_symbolic_equal(dynamics.orientation_angle("bus_5"), q33 + q35)

    assert_symbolic_equal(dynamics.orientation_angle("panel_1"), q33 + sm.pi + q31)
    assert_symbolic_equal(
        dynamics.orientation_angle("panel_2"),
        q33 + sm.pi + (q31 + q32) / 2,
    )
    assert_symbolic_equal(
        dynamics.orientation_angle("panel_3"),
        q33 + sm.pi / 2 + q32 / 2,
    )
    assert_symbolic_equal(
        dynamics.orientation_angle("panel_4"),
        q33 + q34 / 2,
    )
    assert_symbolic_equal(
        dynamics.orientation_angle("panel_5"),
        q33 + (q34 + q35) / 2,
    )
    assert_symbolic_equal(dynamics.orientation_angle("panel_6"), q33 + q35)


def test_multiangle_symbol_names_reject_non_bus_names():
    dynamics = MultiAngleFlexibleDynamics(seven_part_config())

    with pytest.raises(ValueError, match="bus_<number>"):
        dynamics._angle_symbol_name("wer")

    with pytest.raises(ValueError, match="bus_<number>"):
        dynamics._speed_symbol_name("wer")


def test_multiangle_defines_flexible_modal_symbols_and_state_vectors():
    dynamics = MultiAngleFlexibleDynamics(seven_part_config())

    q_names = [str(symbol.func) for symbol in dynamics.q]
    u_names = [str(symbol.func) for symbol in dynamics.u]

    assert q_names == [
        "q1",
        "q2",
        "q3_1",
        "q3_2",
        "q3_3",
        "eta1_1",
        "eta2_1",
        "eta2_2",
        "eta3_1",
        "eta3_2",
        "eta4_1",
    ]
    assert u_names == [
        "u1",
        "u2",
        "u3_1",
        "u3_2",
        "u3_3",
        "zeta1_1",
        "zeta2_1",
        "zeta2_2",
        "zeta3_1",
        "zeta3_2",
        "zeta4_1",
    ]

    assert len(dynamics.q) == len(dynamics.u) == 11
    assert dynamics.flexible_bodies["panel_2"]["beam_type"] == "clamped-clamped"
    assert len(dynamics.flexible_bodies["panel_2"]["eta_list"]) == 2
    assert len(dynamics.flexible_bodies["panel_2"]["zeta_list"]) == 2

    assert dynamics.flex_eta_index[("panel_1", 0)] == 0
    assert dynamics.flex_eta_index[("panel_2", 0)] == 1
    assert dynamics.flex_eta_index[("panel_2", 1)] == 2
    assert dynamics.flex_eta_index[("panel_3", 0)] == 3
    assert dynamics.flex_eta_index[("panel_3", 1)] == 4
    assert dynamics.flex_eta_index[("panel_4", 0)] == 5

    assert dynamics.flex_zeta_index[("panel_1", 0)] == 0
    assert dynamics.flex_zeta_index[("panel_2", 0)] == 1
    assert dynamics.flex_zeta_index[("panel_2", 1)] == 2
    assert dynamics.flex_zeta_index[("panel_3", 0)] == 3
    assert dynamics.flex_zeta_index[("panel_3", 1)] == 4
    assert dynamics.flex_zeta_index[("panel_4", 0)] == 5


def test_multiangle_defines_static_parameter_symbols_from_parameters():
    dynamics = MultiAngleFlexibleDynamics(seven_part_config())

    assert dynamics.parameter_symbols["D"] == sm.symbols("D")
    assert dynamics.parameter_symbols["L"] == sm.symbols("L")
    assert dynamics.D == sm.symbols("D")
    assert dynamics.L == sm.symbols("L")
    assert dynamics.mass_symbols["bus_2"] == sm.symbols("m_bus_2")
    assert dynamics.mass_symbols["panel_4"] == sm.symbols("m_panel_4")
    assert dynamics.E_mod == sm.symbols("E_mod")
    assert dynamics.I_area == sm.symbols("I_area")
    assert dynamics.planet_mu == sm.symbols("planet_mu")
    assert dynamics.orbit_semi_major_axis == sm.symbols("orbit_semi_major_axis")
    assert dynamics.orbit_eccentricity == sm.symbols("orbit_eccentricity")

    p_names = [str(symbol) for symbol in dynamics.p]
    assert p_names == [
        "D",
        "L",
        "m_bus_1",
        "m_bus_2",
        "m_bus_3",
        "m_panel_1",
        "m_panel_2",
        "m_panel_3",
        "m_panel_4",
        "E_mod",
        "I_area",
        "planet_mu",
        "orbit_semi_major_axis",
        "orbit_eccentricity",
    ]


def test_multiangle_parameter_symbols_scale_with_body_count():
    dynamics = MultiAngleFlexibleDynamics(eleven_part_config())

    assert dynamics.mass_symbols["bus_5"] == sm.symbols("m_bus_5")
    assert dynamics.mass_symbols["panel_6"] == sm.symbols("m_panel_6")
    assert len(dynamics.mass_symbols) == 11
    assert len(dynamics.p) == 18


def test_multiangle_requires_mass_parameter_for_each_body():
    config = seven_part_config()
    del config["parameters"]["m_panel_4"]

    with pytest.raises(KeyError, match="m_panel_4"):
        MultiAngleFlexibleDynamics(config)


def test_multiangle_defines_bus_torque_symbols_for_seven_part_chain():
    dynamics = MultiAngleFlexibleDynamics(seven_part_config())

    assert dynamics.bus_torque_symbols == {
        "bus_1": sm.symbols("tau_1"),
        "bus_2": sm.symbols("tau_2"),
        "bus_3": sm.symbols("tau_3"),
    }
    assert list(dynamics.tau) == [
        sm.symbols("tau_1"),
        sm.symbols("tau_2"),
        sm.symbols("tau_3"),
    ]


def test_multiangle_torque_symbols_scale_with_body_count():
    dynamics = MultiAngleFlexibleDynamics(eleven_part_config())

    assert dynamics.bus_torque_symbols["bus_5"] == sm.symbols("tau_5")
    assert list(dynamics.tau) == [
        sm.symbols("tau_1"),
        sm.symbols("tau_2"),
        sm.symbols("tau_3"),
        sm.symbols("tau_4"),
        sm.symbols("tau_5"),
    ]


def test_multiangle_defines_mode_shapes_for_flexible_bodies():
    dynamics = MultiAngleFlexibleDynamics(seven_part_config())

    panel_1 = dynamics.flexible_bodies["panel_1"]
    panel_2 = dynamics.flexible_bodies["panel_2"]

    assert isinstance(panel_1["beam"], CantileverBeam)
    assert isinstance(panel_2["beam"], ClampedClampedBeam)
    assert panel_1["beam"].L == 3.0
    assert panel_1["beam"].E == 140e9
    assert panel_1["beam"].I == 2.5e-8

    assert len(panel_1["phi_list"]) == 1
    assert len(panel_1["phi_mean_list"]) == 1
    assert len(panel_1["k_modal_list"]) == 1
    assert len(panel_1["phi_norm_list"]) == 1
    assert len(panel_1["phi_m1_list"]) == 1

    assert len(panel_2["phi_list"]) == 2
    assert len(panel_2["phi_mean_list"]) == 2
    assert len(panel_2["k_modal_list"]) == 2
    assert len(panel_2["phi_norm_list"]) == 2
    assert len(panel_2["phi_m1_list"]) == 2

    assert all(dynamics.s in phi.free_symbols for phi in panel_2["phi_list"])
    assert all(value > 0 for value in panel_2["phi_norm_list"])
    assert all(value > 0 for value in panel_2["k_modal_list"])


def test_multiangle_rejects_unrecognised_beam_type_when_defining_mode_shapes():
    config = seven_part_config()
    config["flexible_types"]["panel_1"] = "mystery-beam"
    config["beam_parameters"]["mystery-beam"] = {"nb_modes": 1}

    with pytest.raises(TypeError, match="Unrecognised beam type"):
        MultiAngleFlexibleDynamics(config)


def test_multiangle_defines_rigid_bus_inertia_matrices():
    dynamics = MultiAngleFlexibleDynamics(seven_part_config())

    m_bus_2 = dynamics.mass_symbols["bus_2"]
    expected = sm.Matrix(
        [
            [m_bus_2 * dynamics.D**2 / 12, 0, 0],
            [0, m_bus_2 * dynamics.D**2 / 12, 0],
            [0, 0, m_bus_2 * dynamics.D**2 / 6],
        ]
    )

    assert dynamics.inertia_matrices["bus_2"] == expected
    assert set(dynamics.inertia_matrices) == set(dynamics.body_names)


def test_multiangle_defines_flexible_panel_inertia_matrices():
    dynamics = MultiAngleFlexibleDynamics(seven_part_config())

    inertia = dynamics.inertia_matrices["panel_2"]
    m_panel_2 = dynamics.mass_symbols["panel_2"]
    eta21, eta22 = dynamics.flexible_bodies["panel_2"]["eta_list"]

    assert inertia.shape == (3, 3)
    assert sm.simplify(inertia[1, 1] - m_panel_2 * dynamics.L**2 / 12) == 0
    assert sm.simplify(inertia[0, 1] - inertia[1, 0]) == 0
    assert inertia[0, 2] == 0
    assert inertia[1, 2] == 0
    assert sm.simplify(inertia[2, 2] - (inertia[0, 0] + inertia[1, 1])) == 0
    assert inertia[0, 0].has(eta21)
    assert inertia[0, 0].has(eta22)
    assert inertia[0, 1].has(eta21)
    assert inertia[0, 1].has(eta22)


def test_multiangle_offset_from_rigid_bus_to_flexible_panel():
    dynamics = MultiAngleFlexibleDynamics(seven_part_config())

    left_offset = dynamics._get_offset_vector("bus_2", "panel_2")
    right_offset = dynamics._get_offset_vector("bus_2", "panel_3")

    assert left_offset == -dynamics.D / 2 * dynamics.frames["bus_2"].x
    assert right_offset == dynamics.D / 2 * dynamics.frames["bus_2"].x


def test_multiangle_offset_from_flexible_panel_to_rigid_bus_uses_tip_deflection():
    dynamics = MultiAngleFlexibleDynamics(seven_part_config())

    offset = dynamics._get_offset_vector("panel_2", "bus_1")
    eta_list = dynamics.flexible_bodies["panel_2"]["eta_list"]
    phi_list = dynamics.flexible_bodies["panel_2"]["phi_list"]
    expected_tip_deflection = sum(
        phi_k.subs(dynamics.s, dynamics.L) * eta_k
        for phi_k, eta_k in zip(phi_list, eta_list)
    )
    expected = (
        dynamics.L * dynamics.frames["panel_2"].x
        + expected_tip_deflection * dynamics.frames["panel_2"].y
    )

    assert offset == expected


def test_multiangle_offset_rejects_unsupported_body_type_pairs():
    dynamics = MultiAngleFlexibleDynamics(seven_part_config())

    with pytest.raises(NotImplementedError, match="two rigid bodies"):
        dynamics._get_offset_vector("bus_1", "bus_2")

    with pytest.raises(NotImplementedError, match="two flexible bodies"):
        dynamics._get_offset_vector("panel_1", "panel_2")


def test_multiangle_defines_expected_points_for_seven_part_chain():
    dynamics = MultiAngleFlexibleDynamics(seven_part_config())

    expected_points = {
        "N",
        "bus_1",
        "bus_2",
        "bus_3",
        "joint_panel_2_bus_2",
        "joint_panel_3_bus_2",
        "joint_bus_1_panel_2",
        "joint_bus_3_panel_3",
        "joint_panel_1_bus_1",
        "joint_panel_4_bus_3",
        "dm_panel_1",
        "dm_panel_2",
        "dm_panel_3",
        "dm_panel_4",
        "dm_center_of_mass_panel_1",
        "dm_center_of_mass_panel_2",
        "dm_center_of_mass_panel_3",
        "dm_center_of_mass_panel_4",
    }

    assert expected_points.issubset(dynamics.points)
    assert all(dynamics.inertial_position[body] is not None for body in dynamics.body_names)


def test_multiangle_places_central_bus_at_reference_translation():
    dynamics = MultiAngleFlexibleDynamics(seven_part_config())
    N = dynamics.frames["inertial"]

    expected = dynamics.q_translation["x"] * N.x + dynamics.q_translation["y"] * N.y

    assert dynamics.inertial_position["bus_2"] == expected


def test_multiangle_places_flexible_panel_mass_center_from_panel_joint():
    dynamics = MultiAngleFlexibleDynamics(seven_part_config())

    panel = "panel_3"
    joint = dynamics.points["joint_panel_3_bus_2"]
    panel_cm = dynamics.points["dm_center_of_mass_panel_3"]
    eta_list = dynamics.flexible_bodies[panel]["eta_list"]
    phi_mean_list = dynamics.flexible_bodies[panel]["phi_mean_list"]
    phi_mean_sum = sum(
        sm.Float(phi_mean_k) * eta_k
        for phi_mean_k, eta_k in zip(phi_mean_list, eta_list)
    )
    expected = (
        dynamics.L / 2 * dynamics.frames[panel].x
        + phi_mean_sum * dynamics.frames[panel].y
    )

    assert panel_cm.pos_from(joint) == expected


def test_multiangle_places_rigid_bus_mass_center_from_bus_joint():
    dynamics = MultiAngleFlexibleDynamics(seven_part_config())

    bus = "bus_3"
    joint = dynamics.points["joint_bus_3_panel_3"]
    bus_cm = dynamics.points[bus]
    expected = dynamics.D / 2 * dynamics.frames[bus].x

    assert bus_cm.pos_from(joint) == expected


def test_multiangle_defines_system_center_of_mass_from_body_mass_centers():
    dynamics = MultiAngleFlexibleDynamics(seven_part_config())

    expected_total_mass = sum(
        dynamics.mass_symbols[body]
        for body in dynamics.body_names
    )
    expected_position = sum(
        dynamics.mass_symbols[body] * dynamics.inertial_position[body]
        for body in dynamics.body_names
    ) / expected_total_mass

    assert dynamics.total_mass == expected_total_mass
    assert dynamics.points["center_of_mass"] is dynamics.G
    assert dynamics.G.pos_from(dynamics.O) == expected_position


def test_multiangle_defines_central_bus_position_relative_to_system_center_of_mass():
    dynamics = MultiAngleFlexibleDynamics(seven_part_config())

    expected = dynamics.points[dynamics.central_body].pos_from(dynamics.G)

    assert dynamics.r_GB == expected


def test_multiangle_defines_kinematic_differential_equations():
    dynamics = MultiAngleFlexibleDynamics(seven_part_config())

    assert dynamics.fk == dynamics.qd - dynamics.u
    assert dynamics.Mk == sm.eye(len(dynamics.q))
    assert dynamics.gk == -dynamics.u


def test_multiangle_defines_coordinate_derivative_replacements():
    dynamics = MultiAngleFlexibleDynamics(seven_part_config())

    assert dynamics.qd_repl == dict(zip(dynamics.qd, dynamics.u))
    assert dynamics.qdd_repl == {
        q.diff(dynamics.t): u.diff(dynamics.t)
        for q, u in zip(dynamics.qd, dynamics.u)
    }


def test_multiangle_defines_bus_angular_velocities():
    dynamics = MultiAngleFlexibleDynamics(seven_part_config())
    N = dynamics.frames["inertial"]

    u31 = dynamics.bus_speed_coordinates["bus_1"]
    u32 = dynamics.bus_speed_coordinates["bus_2"]
    u33 = dynamics.bus_speed_coordinates["bus_3"]

    assert_vector_equal(dynamics.angular_velocities["bus_1"], (u32 + u31) * N.z, N)
    assert_vector_equal(dynamics.angular_velocities["bus_2"], u32 * N.z, N)
    assert_vector_equal(dynamics.angular_velocities["bus_3"], (u32 + u33) * N.z, N)


def test_multiangle_defines_panel_angular_velocities_from_orientation_convention():
    dynamics = MultiAngleFlexibleDynamics(seven_part_config())
    N = dynamics.frames["inertial"]

    u31 = dynamics.bus_speed_coordinates["bus_1"]
    u32 = dynamics.bus_speed_coordinates["bus_2"]
    u33 = dynamics.bus_speed_coordinates["bus_3"]

    assert_vector_equal(dynamics.angular_velocities["panel_1"], (u32 + u31) * N.z, N)
    assert_vector_equal(
        dynamics.angular_velocities["panel_2"],
        (u32 + u31 / 2) * N.z,
        N,
    )
    assert_vector_equal(
        dynamics.angular_velocities["panel_3"],
        (u32 + u33 / 2) * N.z,
        N,
    )
    assert_vector_equal(dynamics.angular_velocities["panel_4"], (u32 + u33) * N.z, N)


def test_multiangle_defines_central_bus_linear_velocity():
    dynamics = MultiAngleFlexibleDynamics(seven_part_config())
    N = dynamics.frames["inertial"]

    expected = (
        dynamics.u_translation["x"] * N.x
        + dynamics.u_translation["y"] * N.y
    )

    assert_vector_equal(dynamics.points[dynamics.central_body].vel(N), expected, N)
    assert_vector_equal(dynamics.linear_velocities[dynamics.central_body], expected, N)
