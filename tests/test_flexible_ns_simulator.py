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
