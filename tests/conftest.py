from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from multibodysim.multiangle import MultiAngleFlexibleDynamics


CONFIG_DIRS = [
    Path(
        "/Users/jalalelhazzat/Documents/Packages/configuration/flexible/"
        "3-Parts-Spacecraft/Zero-Initial-Flexing"
    ),
    Path("/Users/jalalelhazzat/Documents/Packages/configuration/flexible/Zero-Initial-Flexing"),
    Path("/Users/jalalelhazzat/Documents/Packages/configuration/flexible/Best-Practices"),
]


def _load_config(*filenames: str) -> dict:
    for config_dir in CONFIG_DIRS:
        for filename in filenames:
            config_path = config_dir / filename
            if config_path.exists():
                return json.loads(config_path.read_text())
    names = ", ".join(filenames)
    dirs = ", ".join(str(config_dir) for config_dir in CONFIG_DIRS)
    raise FileNotFoundError(f"Missing test config. Tried: {names} in {dirs}")


def _make_short_test_config(config: dict) -> dict:
    short_config = copy.deepcopy(config)
    short_config["sim_parameters"]["t_end"] = 5.0
    short_config["sim_parameters"]["nb_timesteps"] = 5
    return short_config


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
    return _load_config(
        "GG_off_mode_1_flex_init_0_bodies_3_conf.json",
        "GG_off_mode_1_bodies_3_conf.json",
    )


@pytest.fixture(scope="session")
def gg_on_config() -> dict:
    return _load_config(
        "GG_on_mode_1_flex_init_0_bodies_3_conf.json",
        "GG_on_mode_1_bodies_3_conf.json",
    )


@pytest.fixture
def gg_off_short_config(gg_off_config: dict) -> dict:
    return _make_short_test_config(gg_off_config)


@pytest.fixture
def gg_on_short_config(gg_on_config: dict) -> dict:
    return _make_short_test_config(gg_on_config)
