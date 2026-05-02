from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest


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
