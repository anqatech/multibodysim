from __future__ import annotations

import sympy as sm
import pytest

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
