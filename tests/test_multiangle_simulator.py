from __future__ import annotations

import copy

import numpy as np
import pytest

from multibodysim.controllers.pd_attitude import PlanarAttitudeController
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

    q = initial_conditions[:simulator.dynamics.state_dimension]
    u = initial_conditions[simulator.dynamics.state_dimension:]
    theta_initial = simulator.plant_view.theta(q)
    controller = PlanarAttitudeController(simulator.plant_view)
    controller.configure_attitude_pd(
        theta_target=theta_initial + 0.01,
        theta_dot_target=0.0,
        Kp=2.0,
        Kd=0.0,
        Tr=1.0,
    )
    simulator.set_controller(controller)

    simulator.evaluate_rhs(0.0, initial_conditions)
    simulator.evaluate_rhs(1.0, initial_conditions)

    expected_torques = (
        simulator.initial_torque_values
        + 0.02 * simulator.torque_weights
    )
    np.testing.assert_allclose(simulator.get_torque_values(), expected_torques)
    assert np.isclose(simulator.plant_view.theta(q), theta_initial)
    assert np.isclose(simulator.plant_view.theta_dot(u), u[3])

    with pytest.raises(ValueError, match="Simulation has not been run yet"):
        simulator.get_results()


def test_multiangle_absolute_tolerances_use_grouped_attitude_defaults(
    distributed_7part_zf_gg_off_multiangle_config,
):
    config = copy.deepcopy(distributed_7part_zf_gg_off_multiangle_config)
    config["sim_parameters"]["state_atol"] = {
        "q1": 1e-2,
        "q2": 1e-2,
        "q3": 1e-7,
        "eta": 1e-5,
        "u1": 1e-3,
        "u2": 1e-3,
        "u3": 1e-8,
        "zeta": 1e-6,
    }

    simulator = MultiAngleFlexibleSimulator(config)
    tolerances = simulator._build_absolute_tolerances(config["sim_parameters"])
    names = [
        *[symbol.name for symbol in simulator.dynamics.q],
        *[symbol.name for symbol in simulator.dynamics.u],
    ]
    tolerance_by_name = dict(zip(names, tolerances))

    for name in ("q3_1", "q3_2", "q3_3"):
        assert np.isclose(tolerance_by_name[name], 1e-7)

    for name in ("u3_1", "u3_2", "u3_3"):
        assert np.isclose(tolerance_by_name[name], 1e-8)


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
