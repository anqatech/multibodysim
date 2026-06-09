from __future__ import annotations

import numpy as np
import pytest

from multibodysim.controllers.control_effectiveness import (
    ScalarControlEffectiveness,
    evaluate_scalar_control_effectiveness,
)


class CoupledDynamics:
    state_dimension = 2
    rigid_body_names = ["bus_1", "bus_2"]

    def __init__(self):
        self._eval_differentials = self.evaluate_differentials

    @staticmethod
    def evaluate_differentials(q, u, torques):
        del u
        mass_matrix = np.array(
            [
                [4.0 + 0.1 * q[0], 1.0],
                [1.0, 3.0 + 0.1 * q[1]],
            ]
        )
        base_forcing = np.array([[2.0], [-1.0]])
        torque_map = np.array(
            [
                [1.0, -2.0],
                [3.0, 1.0],
            ]
        )
        forcing = (
            base_forcing
            + torque_map
            @ np.asarray(torques, dtype=float).reshape(2, 1)
        )
        return mass_matrix, forcing


class PlantView:
    i_theta_u = 0


def test_evaluation_uses_full_coupled_control_effectiveness():
    dynamics = CoupledDynamics()
    q = np.array([0.2, -0.1])
    u = np.array([0.3, -0.4])
    weights = np.array([0.25, 0.75])

    result = evaluate_scalar_control_effectiveness(
        dynamics,
        PlantView(),
        weights,
        q,
        u,
    )

    mass_matrix, forcing_zero = dynamics._eval_differentials(
        q,
        u,
        np.zeros(2),
    )
    _, forcing_unit = dynamics._eval_differentials(q, u, weights)
    control_direction = forcing_unit - forcing_zero
    unit_acceleration = np.linalg.solve(
        mass_matrix,
        control_direction,
    )
    expected_effectiveness = unit_acceleration[0, 0]

    assert isinstance(result, ScalarControlEffectiveness)
    assert np.isclose(
        result.control_effectiveness,
        expected_effectiveness,
    )
    assert np.isclose(
        result.reciprocal_control_effectiveness,
        1.0 / expected_effectiveness,
    )


def test_evaluation_retains_control_channel_sign():
    result = evaluate_scalar_control_effectiveness(
        CoupledDynamics(),
        PlantView(),
        np.array([0.0, 1.0]),
        np.zeros(2),
        np.zeros(2),
    )

    assert result.control_effectiveness < 0.0
    assert result.reciprocal_control_effectiveness < 0.0
    assert np.isclose(
        (
            result.control_effectiveness
            * result.reciprocal_control_effectiveness
        ),
        1.0,
    )


def test_evaluation_accepts_nonzero_baseline_torques():
    dynamics = CoupledDynamics()
    common = dict(
        dynamics=dynamics,
        plant_view=PlantView(),
        torque_weights=np.array([0.25, 0.75]),
        q=np.zeros(2),
        u=np.zeros(2),
    )

    zero_baseline = evaluate_scalar_control_effectiveness(**common)
    nonzero_baseline = evaluate_scalar_control_effectiveness(
        **common,
        baseline_torques=np.array([5.0, -3.0]),
    )

    assert np.isclose(
        zero_baseline.control_effectiveness,
        nonzero_baseline.control_effectiveness,
    )


def test_evaluation_rejects_zero_control_effectiveness():
    with pytest.raises(ValueError, match="zero central-attitude"):
        evaluate_scalar_control_effectiveness(
            CoupledDynamics(),
            PlantView(),
            np.array([1.0, 0.0]),
            np.zeros(2),
            np.zeros(2),
        )


def test_evaluation_rejects_torque_dependent_mass_matrix():
    dynamics = CoupledDynamics()
    original = dynamics._eval_differentials

    def torque_dependent(q, u, torques):
        mass_matrix, forcing = original(q, u, torques)
        mass_matrix = mass_matrix.copy()
        mass_matrix[0, 0] += torques[0]
        return mass_matrix, forcing

    dynamics._eval_differentials = torque_dependent

    with pytest.raises(RuntimeError, match="mass matrix changed"):
        evaluate_scalar_control_effectiveness(
            dynamics,
            PlantView(),
            np.array([1.0, 0.0]),
            np.zeros(2),
            np.zeros(2),
        )
