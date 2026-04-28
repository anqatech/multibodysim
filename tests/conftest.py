from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest


CONFIG_DIR = Path(
    "/Users/jalalelhazzat/Documents/Packages/configuration/flexible/Best-Practices"
)


def _load_config(*filenames: str) -> dict:
    for filename in filenames:
        config_path = CONFIG_DIR / filename
        if config_path.exists():
            return json.loads(config_path.read_text())
    names = ", ".join(filenames)
    raise FileNotFoundError(f"Missing test config. Tried: {names}")


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
