from __future__ import annotations

import copy

import numpy as np
import pytest
import sympy as sm

from multibodysim.controllers.rigid_gravity_gradient import (
    RigidGravityGradientTorqueEstimator,
    TorqueAllocatedRigidGravityGradientFeedforward,
    compute_nominal_rigid_inertia,
    prepare_rigid_gravity_gradient_feedforward,
)
from multibodysim.multiangle import MultiAngleFlexibleSimulator


def test_nominal_rigid_inertia_is_symmetric_positive_definite(
    gravity_gradient_dynamics,
):
    inertia = compute_nominal_rigid_inertia(gravity_gradient_dynamics)

    assert inertia.shape == (3, 3)
    np.testing.assert_allclose(inertia, inertia.T, atol=1e-12)
    assert np.all(np.linalg.eigvalsh(inertia) > 0.0)


def test_rigid_estimator_matches_central_gravity_gradient_row(
    gravity_gradient_dynamics,
):
    dynamics = gravity_gradient_dynamics
    inertia = compute_nominal_rigid_inertia(dynamics)
    estimator = RigidGravityGradientTorqueEstimator(
        dynamics.parameter_values["planet_mu"],
        inertia,
    )
    central_row = list(dynamics.u).index(dynamics.central_speed)
    central_expression = dynamics._with_specialised_parameters(
        dynamics.gravity_gradient_generalised_forces[central_row]
    )
    central_function = sm.lambdify(
        dynamics.q,
        central_expression,
        "numpy",
        cse=True,
        docstring_limit=0,
    )

    q_values = np.zeros(dynamics.state_dimension, dtype=float)
    q_x_index = list(dynamics.q).index(dynamics.q_translation["x"])
    q_y_index = list(dynamics.q).index(dynamics.q_translation["y"])
    central_angle_index = list(dynamics.q).index(dynamics.central_angle)
    radius = dynamics.parameter_values["orbit_semi_major_axis"]

    for x_position, y_position, central_angle in (
        (radius, 0.0, 0.0),
        (0.8 * radius, 0.6 * radius, 0.2),
        (0.0, 1.2 * radius, 0.9),
    ):
        q_values[q_x_index] = x_position
        q_values[q_y_index] = y_position
        q_values[central_angle_index] = central_angle

        actual = estimator.evaluate(
            centre_of_mass_position=(x_position, y_position),
            central_attitude=central_angle,
        )
        expected = float(np.asarray(central_function(*q_values)))

        np.testing.assert_allclose(
            actual.rigid_gravity_gradient_torque,
            expected,
            rtol=1e-9,
            atol=1e-12,
        )


def test_rigid_feedforward_scales_and_cancels_rigid_torque():
    estimator = RigidGravityGradientTorqueEstimator(
        gravitational_parameter=4.0,
        nominal_inertia=np.diag([1.0, 3.0, 5.0]),
    )
    first = TorqueAllocatedRigidGravityGradientFeedforward(
        estimator,
        torque_weights=np.array([0.25, 0.75]),
        control_effectiveness=2.0,
    )
    second = TorqueAllocatedRigidGravityGradientFeedforward(
        estimator,
        torque_weights=np.array([0.5, 1.5]),
        control_effectiveness=4.0,
    )

    first_result = first.evaluate(
        centre_of_mass_position=(2.0, 1.0),
        central_attitude=0.3,
    )
    second_result = second.evaluate(
        centre_of_mass_position=(2.0, 1.0),
        central_attitude=0.3,
    )

    assert np.isclose(first.allocation_factor, 0.1)
    assert np.isclose(second.allocation_factor, 0.05)
    assert np.isclose(
        first_result.cancellation_torque,
        -first.allocation_factor
        * first_result.rigid_gravity_gradient_torque,
    )
    assert np.isclose(
        second_result.cancellation_torque,
        0.5 * first_result.cancellation_torque,
    )


def test_rigid_feedforward_rejects_invalid_inputs():
    estimator = RigidGravityGradientTorqueEstimator(
        gravitational_parameter=4.0,
        nominal_inertia=np.diag([1.0, 3.0, 5.0]),
    )

    with pytest.raises(ValueError, match="non-zero direction"):
        TorqueAllocatedRigidGravityGradientFeedforward(
            estimator,
            torque_weights=np.zeros(2),
            control_effectiveness=1.0,
        )

    with pytest.raises(ValueError, match="non-zero"):
        TorqueAllocatedRigidGravityGradientFeedforward(
            estimator,
            torque_weights=np.ones(2),
            control_effectiveness=0.0,
        )

    with pytest.raises(ValueError, match="non-zero magnitude"):
        estimator.evaluate(
            centre_of_mass_position=(0.0, 0.0),
            central_attitude=0.0,
        )


def test_prepare_rigid_feedforward_uses_configured_torque_weights(
    distributed_7part_zf_gg_on_multiangle_config,
):
    config = copy.deepcopy(distributed_7part_zf_gg_on_multiangle_config)
    config["sim_parameters"]["t_end"] = 0.1
    config["sim_parameters"]["nb_timesteps"] = 3

    simulator = MultiAngleFlexibleSimulator(config)
    first = prepare_rigid_gravity_gradient_feedforward(simulator)
    second = prepare_rigid_gravity_gradient_feedforward(
        simulator,
        torque_weights=2.0 * simulator.torque_weights,
    )

    np.testing.assert_allclose(first.torque_weights, simulator.torque_weights)
    np.testing.assert_allclose(
        second.torque_weights,
        2.0 * simulator.torque_weights,
    )
    assert np.isfinite(first.allocation_factor)
    assert np.isfinite(second.allocation_factor)
    np.testing.assert_allclose(
        second.allocation_factor,
        0.5 * first.allocation_factor,
        rtol=1e-12,
        atol=1e-15,
    )
