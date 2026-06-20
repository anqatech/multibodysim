from __future__ import annotations

import numpy as np
import pytest

from multibodysim.allocation import (
    ControlEffectivenessVector,
    evaluate_control_effectiveness_vector,
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


class DirectControlMatrixDynamics(CoupledDynamics):
    def __init__(self):
        super().__init__()
        self.differential_call_count = 0
        self.control_force_matrix_call_count = 0

    def evaluate_differentials(self, q, u, torques):
        self.differential_call_count += 1
        return super().evaluate_differentials(q, u, torques)

    def evaluate_control_force_matrix(self, q, u):
        del q, u
        self.control_force_matrix_call_count += 1
        return np.array(
            [
                [1.0, -2.0],
                [3.0, 1.0],
            ]
        )


class PlantView:
    i_theta_u = 0


def test_vector_evaluation_returns_per_bus_control_effectiveness():
    dynamics = CoupledDynamics()
    q = np.array([0.2, -0.1])
    u = np.array([0.3, -0.4])

    result = evaluate_control_effectiveness_vector(
        dynamics,
        PlantView(),
        q,
        u,
    )

    mass_matrix, forcing_zero = dynamics._eval_differentials(
        q,
        u,
        np.zeros(2),
    )
    _, forcing_bus_1 = dynamics._eval_differentials(
        q,
        u,
        np.array([1.0, 0.0]),
    )
    _, forcing_bus_2 = dynamics._eval_differentials(
        q,
        u,
        np.array([0.0, 1.0]),
    )
    expected_control_directions = np.column_stack(
        (
            (forcing_bus_1 - forcing_zero).reshape(-1),
            (forcing_bus_2 - forcing_zero).reshape(-1),
        )
    )
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


def test_vector_evaluation_uses_direct_control_force_matrix_when_available():
    dynamics = DirectControlMatrixDynamics()
    dynamics._eval_control_force_matrix = dynamics.evaluate_control_force_matrix
    q = np.array([0.2, -0.1])
    u = np.array([0.3, -0.4])

    result = evaluate_control_effectiveness_vector(
        dynamics,
        PlantView(),
        q,
        u,
    )

    expected_control_directions = np.array(
        [
            [1.0, -2.0],
            [3.0, 1.0],
        ]
    )
    mass_matrix, _ = CoupledDynamics.evaluate_differentials(
        q,
        u,
        np.zeros(2),
    )
    expected_accelerations = np.linalg.solve(
        mass_matrix,
        expected_control_directions,
    )

    assert dynamics.differential_call_count == 1
    assert dynamics.control_force_matrix_call_count == 1
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

    vector_result = evaluate_control_effectiveness_vector(
        dynamics,
        PlantView(),
        q,
        u,
    )
    scalar_result = evaluate_scalar_control_effectiveness(
        dynamics,
        PlantView(),
        weights,
        q,
        u,
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

    zero_vector = evaluate_control_effectiveness_vector(
        dynamics,
        PlantView(),
        np.zeros(2),
        np.zeros(2),
    )
    nonzero_vector = evaluate_control_effectiveness_vector(
        dynamics,
        PlantView(),
        np.zeros(2),
        np.zeros(2),
        baseline_torques=np.array([5.0, -3.0]),
    )
    np.testing.assert_allclose(
        zero_vector.effectiveness,
        nonzero_vector.effectiveness,
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


def test_vector_evaluation_rejects_zero_control_effectiveness():
    class ZeroCentralEffectivenessDynamics(CoupledDynamics):
        @staticmethod
        def evaluate_differentials(q, u, torques):
            del q, u
            mass_matrix = np.eye(2)
            base_forcing = np.array([[2.0], [-1.0]])
            torque_map = np.array(
                [
                    [0.0, 0.0],
                    [3.0, 1.0],
                ]
            )
            forcing = (
                base_forcing
                + torque_map
                @ np.asarray(torques, dtype=float).reshape(2, 1)
            )
            return mass_matrix, forcing

    with pytest.raises(ValueError, match="zero central-attitude"):
        evaluate_control_effectiveness_vector(
            ZeroCentralEffectivenessDynamics(),
            PlantView(),
            np.zeros(2),
            np.zeros(2),
        )
