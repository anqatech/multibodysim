from __future__ import annotations

import numpy as np

from multibodysim.allocation import (
    StateBoundedMinimumEffortAllocation,
    allocate_bounded_minimum_effort_at_state,
    evaluate_control_effectiveness_vector,
    solve_bounded_minimum_effort_allocation,
)


class ThreeBusCoupledDynamics:
    state_dimension = 2
    rigid_body_names = ["bus_1", "bus_2", "bus_3"]

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
                [1.0, -2.0, 0.5],
                [3.0, 1.0, -1.5],
            ]
        )
        forcing = (
            base_forcing
            + torque_map
            @ np.asarray(torques, dtype=float).reshape(3, 1)
        )
        return mass_matrix, forcing


class PlantView:
    i_theta_u = 0


def test_state_bounded_allocation_matches_separate_helper_calls():
    dynamics = ThreeBusCoupledDynamics()
    plant_view = PlantView()
    q = np.array([0.2, -0.1])
    u = np.array([0.3, -0.4])
    commanded_acceleration = 0.1
    effort_penalty_matrix = np.diag([1.0, 2.0, 3.0])
    lower_bounds = np.full(3, -0.5)
    upper_bounds = np.full(3, 0.5)

    result = allocate_bounded_minimum_effort_at_state(
        dynamics,
        plant_view,
        q,
        u,
        commanded_acceleration,
        effort_penalty_matrix,
        lower_bounds,
        upper_bounds,
    )
    effectiveness = evaluate_control_effectiveness_vector(
        dynamics,
        plant_view,
        q,
        u,
    )
    allocation = solve_bounded_minimum_effort_allocation(
        commanded_acceleration,
        effectiveness.effectiveness,
        effort_penalty_matrix,
        lower_bounds,
        upper_bounds,
    )

    assert isinstance(result, StateBoundedMinimumEffortAllocation)
    np.testing.assert_allclose(
        result.control_effectiveness.effectiveness,
        effectiveness.effectiveness,
    )
    np.testing.assert_allclose(
        result.allocation.torque_increments,
        allocation.torque_increments,
    )
    np.testing.assert_allclose(
        result.torque_increments,
        allocation.torque_increments,
    )
    assert result.allocation.active_set == allocation.active_set
    assert np.isclose(
        effectiveness.effectiveness @ result.torque_increments,
        commanded_acceleration,
    )


def test_state_bounded_allocation_passes_baseline_torques_to_effectiveness():
    dynamics = ThreeBusCoupledDynamics()
    plant_view = PlantView()
    q = np.zeros(2)
    u = np.zeros(2)
    common = dict(
        dynamics=dynamics,
        plant_view=plant_view,
        q=q,
        u=u,
        commanded_acceleration=0.05,
        effort_penalty_matrix=np.eye(3),
        lower_bounds=np.full(3, -0.5),
        upper_bounds=np.full(3, 0.5),
    )

    zero_baseline = allocate_bounded_minimum_effort_at_state(**common)
    nonzero_baseline = allocate_bounded_minimum_effort_at_state(
        **common,
        baseline_torques=np.array([0.2, -0.1, 0.05]),
    )

    np.testing.assert_allclose(
        zero_baseline.control_effectiveness.effectiveness,
        nonzero_baseline.control_effectiveness.effectiveness,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        zero_baseline.torque_increments,
        nonzero_baseline.torque_increments,
        atol=1e-12,
    )


def test_state_bounded_allocation_forwards_preference_arguments():
    dynamics = ThreeBusCoupledDynamics()
    plant_view = PlantView()
    q = np.array([0.2, -0.1])
    u = np.array([0.3, -0.4])
    commanded_acceleration = 0.1
    effort_penalty_matrix = np.eye(3)
    lower_bounds = np.full(3, -0.5)
    upper_bounds = np.full(3, 0.5)
    preferred_weights = np.array([0.0, 0.56, 0.44])
    preferred_penalty_matrix = np.diag([0.0, 5.0, 5.0])

    result = allocate_bounded_minimum_effort_at_state(
        dynamics,
        plant_view,
        q,
        u,
        commanded_acceleration,
        effort_penalty_matrix,
        lower_bounds,
        upper_bounds,
        preferred_weights=preferred_weights,
        preferred_penalty_matrix=preferred_penalty_matrix,
    )
    effectiveness = evaluate_control_effectiveness_vector(
        dynamics,
        plant_view,
        q,
        u,
    )
    expected = solve_bounded_minimum_effort_allocation(
        commanded_acceleration,
        effectiveness.effectiveness,
        effort_penalty_matrix,
        lower_bounds,
        upper_bounds,
        preferred_weights=preferred_weights,
        preferred_penalty_matrix=preferred_penalty_matrix,
    )

    np.testing.assert_allclose(
        result.torque_increments,
        expected.torque_increments,
    )
    assert result.allocation.active_set == expected.active_set
