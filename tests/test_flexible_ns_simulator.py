from __future__ import annotations

import numpy as np
import pytest

from multibodysim import FlexibleNonSymmetricSimulator


def test_example_config_builds_simulator_and_initial_conditions(gg_off_short_config: dict):
    simulator = FlexibleNonSymmetricSimulator(gg_off_short_config)

    assert simulator.dynamics.enable_gravity_gradient is False
    assert simulator.dynamics.state_dimension == 5
    assert simulator.torque_vals.shape == (1,)

    x0 = simulator.setup_initial_conditions()

    assert x0.shape == (10,)
    assert np.all(np.isfinite(x0))


def test_default_solver_absolute_tolerances_match_existing_values(gg_off_short_config: dict):
    gg_off_short_config["sim_parameters"].pop("state_atol", None)
    simulator = FlexibleNonSymmetricSimulator(gg_off_short_config)

    atol = simulator._build_absolute_tolerances(gg_off_short_config["sim_parameters"])

    assert np.allclose(
        atol,
        np.array(
            [
                1e-2,  # q1
                1e-2,  # q2
                1e-8,  # q3
                1e-6,  # eta1_1
                1e-6,  # eta2_1
                1e-3,  # u1
                1e-3,  # u2
                1e-9,  # u3
                1e-6,  # zeta1_1
                1e-6,  # zeta2_1
            ]
        ),
    )


def test_solver_absolute_tolerances_are_configurable(gg_off_short_config: dict):
    gg_off_short_config["sim_parameters"].update(
        {
            "state_atol": {
                "q3": 1e-7,
                "u3": 1e-8,
                "eta": 1e-5,
                "zeta": 1e-4,
                "eta2_1": 2e-5,
            }
        }
    )
    simulator = FlexibleNonSymmetricSimulator(gg_off_short_config)

    atol = simulator._build_absolute_tolerances(gg_off_short_config["sim_parameters"])

    assert np.allclose(
        atol,
        np.array(
            [
                1e-2,  # q1
                1e-2,  # q2
                1e-7,  # q3
                1e-5,  # eta1_1 from eta family override
                2e-5,  # eta2_1 from exact-state override
                1e-3,  # u1
                1e-3,  # u2
                1e-8,  # u3
                1e-4,  # zeta1_1
                1e-4,  # zeta2_1
            ]
        ),
    )


@pytest.mark.parametrize(
    ("config_fixture_name", "gravity_gradient_expected"),
    [
        ("gg_off_short_config", False),
        ("gg_on_short_config", True),
    ],
)
def test_short_simulation_runs_for_example_configs(
    request: pytest.FixtureRequest,
    config_fixture_name: str,
    gravity_gradient_expected: bool,
):
    config = request.getfixturevalue(config_fixture_name)
    simulator = FlexibleNonSymmetricSimulator(config)

    assert simulator.dynamics.enable_gravity_gradient is gravity_gradient_expected

    results = simulator.run_simulation(eval_flag=True)

    assert results["success"] is True
    expected_timesteps = config["sim_parameters"]["nb_timesteps"]
    assert results["states"].shape == (expected_timesteps, 10)
    assert results["time"].shape == (expected_timesteps,)

    for key in ("q3", "u3", "eta1_1", "eta2_1", "J_eff", "rG_x", "rG_y"):
        assert key in results
        assert np.all(np.isfinite(results[key]))
