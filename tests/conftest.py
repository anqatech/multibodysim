from __future__ import annotations

import copy

import pytest

from multibodysim.multiangle import MultiAngleFlexibleDynamics


def _make_short_test_config(config: dict) -> dict:
    short_config = copy.deepcopy(config)
    short_config["sim_parameters"]["t_end"] = 5.0
    short_config["sim_parameters"]["nb_timesteps"] = 5
    return short_config


def _as_multiangle_config(config: dict) -> dict:
    multiangle_config = copy.deepcopy(config)
    rigid_bodies = [
        body
        for body in multiangle_config["body_names"]
        if multiangle_config["body_type"][body].startswith("rigid-")
    ]
    central_body = multiangle_config["central_body"]
    central_suffix = central_body.split("_", maxsplit=1)[1]

    q_initial = multiangle_config.setdefault("q_initial", {})
    old_q3 = q_initial.pop("q3", 0.0)
    for body in rigid_bodies:
        suffix = body.split("_", maxsplit=1)[1]
        q_initial.setdefault(f"q3_{suffix}", 0.0)
    q_initial[f"q3_{central_suffix}"] = old_q3

    initial_speeds = multiangle_config.setdefault("initial_speeds", {})
    old_u3 = initial_speeds.pop("u3", 0.0)
    for body in rigid_bodies:
        suffix = body.split("_", maxsplit=1)[1]
        initial_speeds.setdefault(f"u3_{suffix}", 0.0)
    initial_speeds[f"u3_{central_suffix}"] = old_u3

    state_atol = multiangle_config.get("sim_parameters", {}).get("state_atol")
    if isinstance(state_atol, dict):
        q3_atol = state_atol.pop("q3", None)
        u3_atol = state_atol.pop("u3", None)
        for body in rigid_bodies:
            suffix = body.split("_", maxsplit=1)[1]
            if q3_atol is not None:
                state_atol.setdefault(f"q3_{suffix}", q3_atol)
            if u3_atol is not None:
                state_atol.setdefault(f"u3_{suffix}", u3_atol)

    return multiangle_config


def _three_part_single_angle_config(*, enable_gravity_gradient: bool) -> dict:
    return {
        "robot_name": "3-Link Arm",
        "central_body": "bus_1",
        "body_names": ["bus_1", "panel_1", "panel_2"],
        "body_type": {
            "bus_1": "rigid-central",
            "panel_1": "flexible-left",
            "panel_2": "flexible-right",
        },
        "adjacency_graph": {
            "bus_1": ["panel_1", "panel_2"],
            "panel_1": ["bus_1"],
            "panel_2": ["bus_1"],
        },
        "flexible_types": {
            "panel_1": "cantilever",
            "panel_2": "cantilever",
        },
        "forces": {
            "bus_1": {"x": 0, "y": 0, "z": 0},
            "panel_1": {"x": 0, "y": 0, "z": 0},
            "panel_2": {"x": 0, "y": 0, "z": 0},
        },
        "torques": {
            "bus_1": 0.0,
        },
        "torque_weights": {
            "bus_1": 1.0,
        },
        "enable_gravity_gradient": enable_gravity_gradient,
        "parameters": {
            "D": 1.0,
            "L": 3.0,
            "m_bus_1": 3.0,
            "m_panel_1": 2.0,
            "m_panel_2": 30.0,
            "E_mod": 140e9,
            "I_area": 2.5e-8,
            "planet_mu": 3.986004418e14,
            "orbit_semi_major_axis": 6778000.0,
            "orbit_eccentricity": 0.0,
        },
        "q_initial": {
            "q3": 0.0349066,
            "eta1_1": 0.0,
            "eta2_1": 0.0,
        },
        "initial_speeds": {
            "u3": 0.001131 if enable_gravity_gradient else 0.0,
            "zeta1_1": 0.0,
            "zeta2_1": 0.0,
        },
        "sim_parameters": {
            "t_start": 0.0,
            "t_end": 15000.0,
            "nb_timesteps": 2000,
            "simulation_type": "flexible",
            "method": "Radau",
            "rtol": 1e-5,
            "state_atol": {
                "q3": 1e-7,
                "u3": 1e-8,
                "eta": 1e-5,
                "zeta": 1e-6,
            },
        },
        "beam_parameters": {
            "cantilever": {
                "nb_modes": 1,
                "nb_points": 200,
            },
        },
    }


def _single_angle_seven_part_config() -> dict:
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
        "parameters": {
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


def _distributed_7part_single_angle_config(
    *,
    enable_gravity_gradient: bool,
    nonzero_initial_flexing: bool,
) -> dict:
    eta1_initial = 0.003 if nonzero_initial_flexing else 0.0

    return {
        "robot_name": "7-Link Arm",
        "central_body": "bus_2",
        "body_names": [
            "bus_1",
            "bus_2",
            "bus_3",
            "panel_1",
            "panel_2",
            "panel_3",
            "panel_4",
        ],
        "body_type": {
            "bus_1": "rigid-left",
            "bus_2": "rigid-central",
            "bus_3": "rigid-right",
            "panel_1": "flexible-left",
            "panel_2": "flexible-left",
            "panel_3": "flexible-right",
            "panel_4": "flexible-right",
        },
        "adjacency_graph": {
            "bus_1": ["panel_1", "panel_2"],
            "bus_2": ["panel_2", "panel_3"],
            "bus_3": ["panel_3", "panel_4"],
            "panel_1": ["bus_1"],
            "panel_2": ["bus_1", "bus_2"],
            "panel_3": ["bus_2", "bus_3"],
            "panel_4": ["bus_3"],
        },
        "flexible_types": {
            "panel_1": "cantilever",
            "panel_2": "clamped-clamped",
            "panel_3": "clamped-clamped",
            "panel_4": "cantilever",
        },
        "forces": {
            "bus_1": {"x": 0, "y": 0, "z": 0},
            "bus_2": {"x": 0, "y": 0, "z": 0},
            "bus_3": {"x": 0, "y": 0, "z": 0},
            "panel_1": {"x": 0, "y": 0, "z": 0},
            "panel_2": {"x": 0, "y": 0, "z": 0},
            "panel_3": {"x": 0, "y": 0, "z": 0},
            "panel_4": {"x": 0, "y": 0, "z": 0},
        },
        "torques": {
            "bus_1": 0.0,
            "bus_2": 0.0,
            "bus_3": 0.0,
        },
        "torque_weights": {
            "bus_1": 0.0,
            "bus_2": 1.0,
            "bus_3": 0.0,
        },
        "enable_gravity_gradient": enable_gravity_gradient,
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
        "q_initial": {
            "q3": 0.0349066,
            "eta1_1": eta1_initial,
            "eta2_1": 0.0,
            "eta3_1": 0.0,
            "eta4_1": 0.0,
        },
        "initial_speeds": {
            "u3": 0.0,
            "zeta1_1": 0.0,
            "zeta2_1": 0.0,
            "zeta3_1": 0.0,
            "zeta4_1": 0.0,
        },
        "sim_parameters": {
            "t_start": 0.0,
            "t_end": 15000.0,
            "nb_timesteps": 2000,
            "simulation_type": "flexible",
            "method": "Radau",
            "rtol": 1e-5,
            "state_atol": {
                "q3": 1e-7,
                "u3": 1e-8,
                "eta": 1e-5,
                "zeta": 1e-6,
            },
        },
        "beam_parameters": {
            "cantilever": {
                "nb_modes": 1,
                "nb_points": 200,
                "flexible_inertia_integration": "gauss-legendre",
                "inertia_quadrature_points": 8,
            },
            "clamped-clamped": {
                "nb_modes": 1,
                "nb_points": 200,
                "flexible_inertia_integration": "gauss-legendre",
                "inertia_quadrature_points": 8,
            },
        },
    }


def _seven_part_config() -> dict:
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


def _compact_gravity_gradient_config() -> dict:
    return {
        "body_names": [
            "bus_1",
            "bus_2",
            "bus_3",
            "panel_1",
            "panel_2",
        ],
        "central_body": "bus_2",
        "adjacency_graph": {
            "bus_1": ["panel_1"],
            "bus_2": ["panel_1", "panel_2"],
            "bus_3": ["panel_2"],
            "panel_1": ["bus_1", "bus_2"],
            "panel_2": ["bus_2", "bus_3"],
        },
        "body_type": {
            "bus_1": "rigid-left",
            "bus_2": "rigid-central",
            "bus_3": "rigid-right",
            "panel_1": "flexible-left",
            "panel_2": "flexible-right",
        },
        "flexible_types": {
            "panel_1": "cantilever",
            "panel_2": "cantilever",
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
            "m_panel_1": 2.0,
            "m_panel_2": 2.0,
            "E_mod": 140e9,
            "I_area": 2.5e-8,
            "planet_mu": 3.986004418e14,
            "orbit_semi_major_axis": 6778000.0,
            "orbit_eccentricity": 0.0,
        },
        "enable_gravity_gradient": True,
    }


def _eleven_part_config() -> dict:
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


@pytest.fixture(scope="module")
def seven_part_dynamics():
    return MultiAngleFlexibleDynamics(_seven_part_config())


@pytest.fixture(scope="module")
def eleven_part_dynamics():
    return MultiAngleFlexibleDynamics(_eleven_part_config())


@pytest.fixture(scope="module")
def gravity_gradient_dynamics():
    return MultiAngleFlexibleDynamics(_compact_gravity_gradient_config())


@pytest.fixture
def seven_part_config() -> dict:
    return _seven_part_config()


@pytest.fixture
def compact_gravity_gradient_config() -> dict:
    return _compact_gravity_gradient_config()


@pytest.fixture
def eleven_part_config() -> dict:
    return _eleven_part_config()


@pytest.fixture(scope="session")
def gg_off_config() -> dict:
    return _three_part_single_angle_config(enable_gravity_gradient=False)


@pytest.fixture(scope="session")
def gg_on_config() -> dict:
    return _three_part_single_angle_config(enable_gravity_gradient=True)


@pytest.fixture
def gg_off_short_config(gg_off_config: dict) -> dict:
    return _make_short_test_config(gg_off_config)


@pytest.fixture
def gg_on_short_config(gg_on_config: dict) -> dict:
    return _make_short_test_config(gg_on_config)


@pytest.fixture
def single_angle_seven_part_config() -> dict:
    return _single_angle_seven_part_config()


@pytest.fixture
def distributed_7part_zf_gg_off_single_angle_config() -> dict:
    return _distributed_7part_single_angle_config(
        enable_gravity_gradient=False,
        nonzero_initial_flexing=False,
    )


@pytest.fixture
def distributed_7part_zf_gg_on_single_angle_config() -> dict:
    return _distributed_7part_single_angle_config(
        enable_gravity_gradient=True,
        nonzero_initial_flexing=False,
    )


@pytest.fixture
def distributed_7part_nzf_gg_off_single_angle_config() -> dict:
    return _distributed_7part_single_angle_config(
        enable_gravity_gradient=False,
        nonzero_initial_flexing=True,
    )


@pytest.fixture
def distributed_7part_nzf_gg_on_single_angle_config() -> dict:
    return _distributed_7part_single_angle_config(
        enable_gravity_gradient=True,
        nonzero_initial_flexing=True,
    )


@pytest.fixture
def distributed_7part_zf_gg_off_multiangle_config(
    distributed_7part_zf_gg_off_single_angle_config: dict,
) -> dict:
    return _as_multiangle_config(distributed_7part_zf_gg_off_single_angle_config)


@pytest.fixture
def distributed_7part_zf_gg_on_multiangle_config(
    distributed_7part_zf_gg_on_single_angle_config: dict,
) -> dict:
    return _as_multiangle_config(distributed_7part_zf_gg_on_single_angle_config)


@pytest.fixture
def distributed_7part_nzf_gg_off_multiangle_config(
    distributed_7part_nzf_gg_off_single_angle_config: dict,
) -> dict:
    return _as_multiangle_config(distributed_7part_nzf_gg_off_single_angle_config)


@pytest.fixture
def distributed_7part_nzf_gg_on_multiangle_config(
    distributed_7part_nzf_gg_on_single_angle_config: dict,
) -> dict:
    return _as_multiangle_config(distributed_7part_nzf_gg_on_single_angle_config)
