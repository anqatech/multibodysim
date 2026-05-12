from __future__ import annotations

import copy

import numpy as np
import pytest

from multibodysim.multiangle import MultiAngleFlexibleSimulator


def test_multiangle_simulator_builds_from_multiangle_config(
    distributed_7part_zf_gg_off_multiangle_config,
):
    simulator = MultiAngleFlexibleSimulator(
        distributed_7part_zf_gg_off_multiangle_config,
    )

    assert simulator.config is distributed_7part_zf_gg_off_multiangle_config
    assert simulator.dynamics.config is distributed_7part_zf_gg_off_multiangle_config
    assert simulator.parameter_values.shape == (
        len(simulator.dynamics.parameter_symbols),
    )
    assert simulator.initial_torque_values.shape == (
        len(simulator.dynamics.rigid_body_names),
    )

    np.testing.assert_allclose(
        simulator.get_torque_values(),
        simulator.initial_torque_values,
    )

    initial_conditions = simulator.setup_initial_conditions(verbose=False)
    assert initial_conditions.shape == (2 * simulator.dynamics.state_dimension,)

    rhs = simulator.evaluate_rhs(0.0, initial_conditions)
    assert rhs.shape == initial_conditions.shape
    assert np.all(np.isfinite(rhs))

    with pytest.raises(ValueError, match="Simulation has not been run yet"):
        simulator.get_results()


def test_multiangle_simulator_runs_short_integration(
    distributed_7part_zf_gg_off_multiangle_config,
):
    config = copy.deepcopy(distributed_7part_zf_gg_off_multiangle_config)
    config["sim_parameters"]["t_end"] = 0.1
    config["sim_parameters"]["nb_timesteps"] = 3
    config["sim_parameters"]["method"] = "DOP853"

    simulator = MultiAngleFlexibleSimulator(config)
    results = simulator.run_simulation(eval_flag=True, verbose=False)

    assert results["success"]
    assert results["time"].shape == (3,)
    assert results["states"].shape == (3, 2 * simulator.dynamics.state_dimension)
    assert "q3_2" in results
    assert "u3_2" in results
    assert simulator.get_results() is results
