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
