from __future__ import annotations

import numpy as np
import pytest

from multibodysim.allocation import (
    ControlEffectivenessEvaluator,
    ControlEffectivenessVector,
)
from multibodysim.controllers.control_effectiveness import (
    ScalarControlEffectiveness,
    evaluate_scalar_control_effectiveness,
)


class CoupledDynamics:
    state_dimension = 2
    rigid_body_names = ["bus_1", "bus_2"]

    def __init__(self):
        self._eval_differentials = self.evaluate_differentials
        self._eval_control_force_matrix = self.evaluate_control_force_matrix

    @staticmethod
    def evaluate_differentials(q, u, torques):
        mass_matrix = np.array(
            [
                [4.0 + 0.1 * q[0], 1.0],
                [1.0, 3.0 + 0.1 * q[1]],
            ]
        )
        base_forcing = np.array([[2.0], [-1.0]])
        forcing = (
            base_forcing
            + CoupledDynamics.evaluate_control_force_matrix(q, u)
            @ np.asarray(torques, dtype=float).reshape(2, 1)
        )
        return mass_matrix, forcing

    @staticmethod
    def evaluate_control_force_matrix(q, u):
        del q, u
        return np.array(
            [
                [1.0, -2.0],
                [3.0, 1.0],
            ]
        )


class RecordingDirectControlMatrixDynamics(CoupledDynamics):
    def __init__(self):
        super().__init__()
        self.differential_call_count = 0
        self.control_force_matrix_call_count = 0

    def evaluate_differentials(self, q, u, torques):
        self.differential_call_count += 1
        return super().evaluate_differentials(q, u, torques)

    def evaluate_control_force_matrix(self, q, u):
        self.control_force_matrix_call_count += 1
        return super().evaluate_control_force_matrix(q, u)


class PlantView:
    i_theta_u = 0


def mass_matrix_for(dynamics, q, u):
    mass_matrix, _ = dynamics._eval_differentials(
        q,
        u,
        np.zeros(len(dynamics.rigid_body_names)),
    )
    return mass_matrix


def test_evaluator_returns_per_bus_control_effectiveness():
    dynamics = CoupledDynamics()
    q = np.array([0.2, -0.1])
    u = np.array([0.3, -0.4])
    mass_matrix = mass_matrix_for(dynamics, q, u)

    result = ControlEffectivenessEvaluator(
        dynamics,
        PlantView(),
    ).evaluate(q, u, mass_matrix)

    expected_control_directions = dynamics.evaluate_control_force_matrix(q, u)
    expected_accelerations = np.linalg.solve(
        mass_matrix,
        expected_control_directions,
    )

    assert isinstance(result, ControlEffectivenessVector)
    np.testing.assert_allclose(
        result.control_generalised_force_directions,
        expected_control_directions,
    )
    np.testing.assert_allclose(
        result.unit_control_accelerations,
        expected_accelerations,
    )
    np.testing.assert_allclose(
        result.effectiveness,
        expected_accelerations[0, :],
    )


def test_evaluator_does_not_call_differential_evaluator():
    dynamics = RecordingDirectControlMatrixDynamics()
    q = np.array([0.2, -0.1])
    u = np.array([0.3, -0.4])
    mass_matrix = CoupledDynamics.evaluate_differentials(
        q,
        u,
        np.zeros(2),
    )[0]

    result = ControlEffectivenessEvaluator(
        dynamics,
        PlantView(),
    ).evaluate(q, u, mass_matrix)

    assert dynamics.differential_call_count == 0
    assert dynamics.control_force_matrix_call_count == 1
    assert np.all(np.isfinite(result.effectiveness))


def test_scalar_evaluation_uses_full_coupled_control_effectiveness():
    dynamics = CoupledDynamics()
    q = np.array([0.2, -0.1])
    u = np.array([0.3, -0.4])
    weights = np.array([0.25, 0.75])
    mass_matrix = mass_matrix_for(dynamics, q, u)

    result = evaluate_scalar_control_effectiveness(
        dynamics,
        PlantView(),
        weights,
        q,
        u,
        mass_matrix=mass_matrix,
    )

    control_direction = (
        dynamics.evaluate_control_force_matrix(q, u)
        @ weights.reshape(-1, 1)
    )
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


@pytest.mark.parametrize(
    "weights",
    [
        np.array([1.0, 0.0]),
        np.array([0.0, 1.0]),
        np.array([0.25, 0.75]),
        np.array([0.5, 0.5]),
    ],
)
def test_scalar_effectiveness_is_projection_of_vector_effectiveness(weights):
    dynamics = CoupledDynamics()
    q = np.array([0.2, -0.1])
    u = np.array([0.3, -0.4])
    mass_matrix = mass_matrix_for(dynamics, q, u)

    vector_result = ControlEffectivenessEvaluator(
        dynamics,
        PlantView(),
    ).evaluate(q, u, mass_matrix)
    scalar_result = evaluate_scalar_control_effectiveness(
        dynamics,
        PlantView(),
        weights,
        q,
        u,
        mass_matrix=mass_matrix,
    )

    assert np.isclose(
        scalar_result.control_effectiveness,
        vector_result.effectiveness @ weights,
    )
    np.testing.assert_allclose(
        scalar_result.control_generalised_force_direction,
        vector_result.control_generalised_force_directions
        @ weights.reshape(-1, 1),
    )
    np.testing.assert_allclose(
        scalar_result.unit_control_acceleration,
        vector_result.unit_control_accelerations
        @ weights.reshape(-1, 1),
    )


def test_evaluation_retains_control_channel_sign():
    dynamics = CoupledDynamics()
    q = np.zeros(2)
    u = np.zeros(2)
    result = evaluate_scalar_control_effectiveness(
        dynamics,
        PlantView(),
        np.array([0.0, 1.0]),
        q,
        u,
        mass_matrix=mass_matrix_for(dynamics, q, u),
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


def test_evaluation_rejects_zero_control_effectiveness():
    dynamics = CoupledDynamics()
    q = np.zeros(2)
    u = np.zeros(2)
    with pytest.raises(ValueError, match="zero central-attitude"):
        evaluate_scalar_control_effectiveness(
            dynamics,
            PlantView(),
            np.array([1.0, 0.0]),
            q,
            u,
            mass_matrix=mass_matrix_for(dynamics, q, u),
        )


def test_evaluator_requires_direct_control_force_matrix():
    dynamics = CoupledDynamics()
    dynamics._eval_control_force_matrix = None

    with pytest.raises(RuntimeError, match="_eval_control_force_matrix"):
        ControlEffectivenessEvaluator(dynamics, PlantView())


def test_vector_evaluation_rejects_zero_control_effectiveness():
    class ZeroCentralEffectivenessDynamics(CoupledDynamics):
        @staticmethod
        def evaluate_control_force_matrix(q, u):
            del q, u
            return np.array(
                [
                    [0.0, 0.0],
                    [3.0, 1.0],
                ]
            )

    dynamics = ZeroCentralEffectivenessDynamics()
    q = np.zeros(2)
    u = np.zeros(2)

    with pytest.raises(ValueError, match="zero central-attitude"):
        ControlEffectivenessEvaluator(
            dynamics,
            PlantView(),
        ).evaluate(q, u, np.eye(2))
