from __future__ import annotations

import numpy as np
import pytest

from multibodysim.allocation import (
    BoundedMinimumEffortAllocation,
    evaluate_feasible_acceleration_interval,
    solve_bounded_minimum_effort_allocation,
    solve_unconstrained_minimum_effort_allocation,
)


NOTEBOOK_EFFECTIVENESS = np.array(
    [
        -6.453571839024e-04,
        6.472287308408e-04,
        -4.407656308179e-04,
    ]
)


def test_feasible_acceleration_interval_uses_bound_corners():
    lower_bounds = np.full(3, -0.05)
    upper_bounds = np.full(3, 0.05)

    interval = evaluate_feasible_acceleration_interval(
        NOTEBOOK_EFFECTIVENESS,
        lower_bounds,
        upper_bounds,
    )

    expected_maximum = float(
        NOTEBOOK_EFFECTIVENESS
        @ np.array([-0.05, 0.05, -0.05])
    )
    expected_minimum = -expected_maximum
    np.testing.assert_allclose(
        interval,
        (expected_minimum, expected_maximum),
    )


def test_bounded_solution_matches_unconstrained_when_bounds_inactive():
    control_effectiveness = np.array([2.0, -1.0, 0.5])
    effort_penalty_matrix = np.diag([1.0, 4.0, 2.0])
    commanded_acceleration = 0.25
    lower_bounds = np.full(3, -10.0)
    upper_bounds = np.full(3, 10.0)

    result = solve_bounded_minimum_effort_allocation(
        commanded_acceleration,
        control_effectiveness,
        effort_penalty_matrix,
        lower_bounds,
        upper_bounds,
    )
    unconstrained_torques = solve_unconstrained_minimum_effort_allocation(
        commanded_acceleration,
        control_effectiveness,
        effort_penalty_matrix,
    )

    assert isinstance(result, BoundedMinimumEffortAllocation)
    np.testing.assert_allclose(result.torque_increments, unconstrained_torques)
    assert result.active_set == ("F", "F", "F")
    assert result.valid_candidate_count >= 1
    assert result.candidate_count == 27
    assert np.isclose(
        control_effectiveness @ result.torque_increments,
        commanded_acceleration,
    )


def test_bounded_solution_prefers_weighted_split_when_bounds_inactive():
    result = solve_bounded_minimum_effort_allocation(
        1.0,
        np.array([1.0, 1.0]),
        np.eye(2),
        np.full(2, -10.0),
        np.full(2, 10.0),
        preferred_weights=np.array([1.0, 0.0]),
        preferred_penalty_matrix=np.diag([10.0, 0.0]),
    )

    np.testing.assert_allclose(
        result.torque_increments,
        np.array([11.0 / 12.0, 1.0 / 12.0]),
    )
    assert result.active_set == ("F", "F")
    assert np.isclose(result.achieved_acceleration, 1.0)


def test_bounded_solution_rejects_incomplete_preference_arguments():
    with pytest.raises(ValueError, match="must be supplied together"):
        solve_bounded_minimum_effort_allocation(
            1.0,
            np.array([1.0, 1.0]),
            np.eye(2),
            np.full(2, -10.0),
            np.full(2, 10.0),
            preferred_weights=np.array([1.0, 0.0]),
        )

    with pytest.raises(ValueError, match="must be supplied together"):
        solve_bounded_minimum_effort_allocation(
            1.0,
            np.array([1.0, 1.0]),
            np.eye(2),
            np.full(2, -10.0),
            np.full(2, 10.0),
            preferred_penalty_matrix=np.eye(2),
        )


def test_bounded_solution_rejects_unreachable_preferred_direction():
    with pytest.raises(ValueError, match="control_effectiveness @ preferred_weights"):
        solve_bounded_minimum_effort_allocation(
            1.0,
            np.array([1.0, -1.0]),
            np.eye(2),
            np.full(2, -10.0),
            np.full(2, 10.0),
            preferred_weights=np.array([1.0, 1.0]),
            preferred_penalty_matrix=np.eye(2),
        )


def test_bounded_solution_matches_three_bus_notebook_case():
    lower_bounds = np.full(3, -0.05)
    upper_bounds = np.full(3, 0.05)
    commanded_acceleration = 8.0e-5

    result = solve_bounded_minimum_effort_allocation(
        commanded_acceleration,
        NOTEBOOK_EFFECTIVENESS,
        np.eye(3),
        lower_bounds,
        upper_bounds,
    )

    np.testing.assert_allclose(
        result.torque_increments,
        np.array([-0.05, 0.05, -0.03487273777295]),
    )
    assert result.active_set == ("L", "U", "F")
    assert result.candidate_count == 27
    assert result.valid_candidate_count == 1
    assert np.isclose(result.achieved_acceleration, commanded_acceleration)
    assert np.isclose(result.residual, 0.0)
    assert result.lower_multipliers[0] >= 0.0
    assert result.upper_multipliers[1] >= 0.0
    assert result.lower_multipliers[1] == 0.0
    assert result.upper_multipliers[0] == 0.0
    assert result.lower_multipliers[2] == 0.0
    assert result.upper_multipliers[2] == 0.0


def test_bounded_solution_supports_five_channels():
    control_effectiveness = np.array([1.0, -0.5, 0.25, 0.75, -0.2])
    effort_penalty_matrix = np.diag([1.0, 2.0, 3.0, 1.5, 4.0])

    result = solve_bounded_minimum_effort_allocation(
        0.2,
        control_effectiveness,
        effort_penalty_matrix,
        np.full(5, -1.0),
        np.full(5, 1.0),
    )

    assert result.candidate_count == 3**5
    assert np.isclose(
        control_effectiveness @ result.torque_increments,
        0.2,
    )
    assert np.all(result.torque_increments >= -1.0)
    assert np.all(result.torque_increments <= 1.0)


def test_bounded_solution_rejects_more_than_five_channels():
    with pytest.raises(ValueError, match="at most 5 torque channels"):
        solve_bounded_minimum_effort_allocation(
            1.0,
            np.ones(6),
            np.eye(6),
            -np.ones(6),
            np.ones(6),
        )


def test_bounded_solution_rejects_infeasible_command():
    with pytest.raises(ValueError, match="outside the feasible"):
        solve_bounded_minimum_effort_allocation(
            10.0,
            np.array([1.0, 1.0, 1.0]),
            np.eye(3),
            np.zeros(3),
            np.ones(3),
        )


@pytest.mark.parametrize(
    ("lower_bounds", "upper_bounds", "error_match"),
    [
        (np.array([0.0, 0.0]), np.ones(3), "lower_bounds"),
        (np.zeros(3), np.array([1.0, 1.0]), "upper_bounds"),
        (np.array([0.0, 2.0, 0.0]), np.ones(3), "less than or equal"),
    ],
)
def test_bounded_solution_rejects_invalid_bounds(
    lower_bounds,
    upper_bounds,
    error_match,
):
    with pytest.raises(ValueError, match=error_match):
        solve_bounded_minimum_effort_allocation(
            0.5,
            np.array([1.0, 1.0, 1.0]),
            np.eye(3),
            lower_bounds,
            upper_bounds,
        )
