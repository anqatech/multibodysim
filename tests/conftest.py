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


DIST_CONFIG_DIR = Path(
    "/Users/jalalelhazzat/Documents/Packages/configuration/flexible/distributed"
)


def _load_config(*filenames: str) -> dict:
    for config_dir in CONFIG_DIRS:
        for filename in filenames:
            config_path = config_dir / filename
            if config_path.exists():
                return json.loads(config_path.read_text())
    names = ", ".join(filenames)
    dirs = ", ".join(str(config_dir) for config_dir in CONFIG_DIRS)
    raise FileNotFoundError(f"Missing test config. Tried: {names} in {dirs}")


def _load_config_path(config_path: Path) -> dict:
    if not config_path.exists():
        raise FileNotFoundError(f"Missing test config: {config_path}")
    return json.loads(config_path.read_text())


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


@pytest.fixture
def distributed_7part_zf_gg_off_single_angle_config() -> dict:
    return _load_config_path(
        DIST_CONFIG_DIR
        / "Zero-Initial-Flexing"
        / "dist-GG_off_mode_1_flex_init_0_bodies_7_conf.json"
    )


@pytest.fixture
def distributed_7part_zf_gg_on_single_angle_config() -> dict:
    return _load_config_path(
        DIST_CONFIG_DIR
        / "Zero-Initial-Flexing"
        / "dist-GG_on_mode_1_flex_init_0_bodies_7_conf.json"
    )


@pytest.fixture
def distributed_7part_nzf_gg_off_single_angle_config() -> dict:
    return _load_config_path(
        DIST_CONFIG_DIR
        / "Non-Zero-Initial-Flexing"
        / "dist-GG_off_mode_1_flex_init_non_0_bodies_7_conf.json"
    )


@pytest.fixture
def distributed_7part_nzf_gg_on_single_angle_config() -> dict:
    return _load_config_path(
        DIST_CONFIG_DIR
        / "Non-Zero-Initial-Flexing"
        / "dist-GG_on_mode_1_flex_init_non_0_bodies_7_conf.json"
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
