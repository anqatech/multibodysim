from __future__ import annotations

import copy

import numpy as np
import pytest

from multibodysim.controllers.pd_attitude import PlanarAttitudeController
from multibodysim.multiangle import (
    MultiAngleFlexibleDynamics,
    MultiAngleFlexibleSimulator,
)


@pytest.fixture(autouse=True)
def fake_autowrap_preparation(monkeypatch):
    calls = []

    def prepare_autowrap_evaluators(dynamics, *, cache_root=None):
        calls.append(dynamics)
        dynamics._eval_kinematics = lambda q, u, torques: (
            np.eye(len(dynamics.q)),
            np.zeros((len(dynamics.q), 1)),
        )
        dynamics.eval_kinematics_backend = "autowrap"
        dynamics.eval_kinematics_generated_metadata = {"cache_key": "kinematics"}
        dynamics.eval_kinematics_codegen_timing = {"source": "cache"}
        dynamics._eval_differentials = lambda q, u, torques: (
            np.eye(len(dynamics.u)),
            np.zeros((len(dynamics.u), 1)),
        )
        dynamics.eval_differentials_backend = "autowrap"
        dynamics.eval_differentials_generated_metadata = {
            "cache_key": "differentials",
        }
        dynamics.eval_differentials_codegen_timing = {"source": "cache"}
        return {
            "kinematics": {
                "success": True,
                "metadata": dynamics.eval_kinematics_generated_metadata,
            },
            "differentials": {
                "success": True,
                "metadata": dynamics.eval_differentials_generated_metadata,
            },
        }

    monkeypatch.setattr(
        "multibodysim.multiangle.simulator.prepare_autowrap_evaluators",
        prepare_autowrap_evaluators,
    )
    return calls


def test_multiangle_simulator_builds_from_multiangle_config(
    distributed_7part_zf_gg_off_multiangle_config,
    fake_autowrap_preparation,
):
    simulator = MultiAngleFlexibleSimulator(
        distributed_7part_zf_gg_off_multiangle_config,
    )

    assert fake_autowrap_preparation == [simulator.dynamics]
    assert simulator.config is distributed_7part_zf_gg_off_multiangle_config
    assert simulator.dynamics.config is distributed_7part_zf_gg_off_multiangle_config
    assert simulator.parameter_values.shape == (
        len(simulator.dynamics.parameter_symbols),
    )
    assert simulator.initial_torque_values.shape == (
        len(simulator.dynamics.rigid_body_names),
    )

    assert simulator.dynamics.eval_differentials_backend == "autowrap"
    assert simulator.dynamics.eval_kinematics_backend == "autowrap"
    assert simulator.codegen_metadata["kinematics"]["success"]
    assert simulator.codegen_metadata["differentials"]["success"]
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
    controller.configure_inertial_pd(
        theta_target=theta_initial + 0.01,
        Kp=2.0,
        Kd=0.0,
        manoeuvre_duration=1.0,
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


def test_direct_multiangle_dynamics_construction_leaves_rhs_evaluators_unprepared(
    distributed_7part_zf_gg_off_multiangle_config,
):
    dynamics = MultiAngleFlexibleDynamics(
        distributed_7part_zf_gg_off_multiangle_config,
    )

    assert dynamics.eval_kinematics_backend == "unprepared"
    assert dynamics.eval_differentials_backend == "unprepared"
    assert dynamics._eval_kinematics is None
    assert dynamics._eval_differentials is None
    assert dynamics.eval_kinematics_generated_metadata is None
    assert dynamics.eval_differentials_generated_metadata is None
    assert dynamics.autowrap_codegen_metadata is None


def test_multiangle_absolute_tolerances_use_grouped_attitude_defaults(
    distributed_7part_zf_gg_off_multiangle_config,
):
    config = copy.deepcopy(distributed_7part_zf_gg_off_multiangle_config)
    config["sim_parameters"]["state_atol"] = {
        "q1": 1e-2,
        "q2": 1e-2,
        "q_central_angle": 1e-7,
        "eta": 1e-5,
        "u1": 1e-3,
        "u2": 1e-3,
        "u_central_angle": 1e-8,
        "zeta": 1e-6,
    }

    simulator = MultiAngleFlexibleSimulator(config)
    tolerances = simulator._build_absolute_tolerances(config["sim_parameters"])
    names = [
        *[symbol.name for symbol in simulator.dynamics.q],
        *[symbol.name for symbol in simulator.dynamics.u],
    ]
    tolerance_by_name = dict(zip(names, tolerances))

    for name in (
        "q_relative_angle_bus_1",
        "q_central_angle",
        "q_relative_angle_bus_3",
    ):
        assert np.isclose(tolerance_by_name[name], 1e-7)

    for name in (
        "u_relative_angle_bus_1",
        "u_central_angle",
        "u_relative_angle_bus_3",
    ):
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
    assert "q_central_angle" in results
    assert "u_central_angle" in results
    assert results["J_eff"].shape == (3,)
    assert results["tau_FF"].shape == (3,)
    assert results["tau_PD"].shape == (3,)
    assert results["rG_x"].shape == (3,)
    assert results["vG_y"].shape == (3,)
    assert np.all(np.isfinite(results["J_eff"]))
    assert simulator.get_results() is results
