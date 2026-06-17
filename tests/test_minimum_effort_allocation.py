from __future__ import annotations

import numpy as np
import pytest

from multibodysim.allocation import (
    solve_unconstrained_minimum_effort_allocation,
)


def test_unconstrained_minimum_effort_allocation_satisfies_command():
    control_effectiveness = np.array([2.0, -1.0, 0.5])
    effort_penalty_matrix = np.diag([1.0, 4.0, 2.0])
    commanded_acceleration = 0.25

    torque_increments = solve_unconstrained_minimum_effort_allocation(
        commanded_acceleration,
        control_effectiveness,
        effort_penalty_matrix,
    )

    assert np.isclose(
        control_effectiveness @ torque_increments,
        commanded_acceleration,
    )


def test_unconstrained_minimum_effort_allocation_matches_closed_form():
    control_effectiveness = np.array([1.0, 2.0])
    effort_penalty_matrix = np.array(
        [
            [2.0, 0.5],
            [0.5, 3.0],
        ]
    )
    commanded_acceleration = -0.4

    torque_increments = solve_unconstrained_minimum_effort_allocation(
        commanded_acceleration,
        control_effectiveness,
        effort_penalty_matrix,
    )

    weighted_effectiveness = np.linalg.solve(
        effort_penalty_matrix,
        control_effectiveness,
    )
    expected_torque_increments = (
        commanded_acceleration
        / (control_effectiveness @ weighted_effectiveness)
        * weighted_effectiveness
    )
    np.testing.assert_allclose(
        torque_increments,
        expected_torque_increments,
    )


def test_higher_effort_penalty_reduces_torque_on_expensive_bus():
    control_effectiveness = np.array([1.0, 1.0])
    commanded_acceleration = 1.0

    equal_penalty_torques = solve_unconstrained_minimum_effort_allocation(
        commanded_acceleration,
        control_effectiveness,
        np.diag([1.0, 1.0]),
    )
    expensive_first_bus_torques = (
        solve_unconstrained_minimum_effort_allocation(
            commanded_acceleration,
            control_effectiveness,
            np.diag([10.0, 1.0]),
        )
    )

    np.testing.assert_allclose(equal_penalty_torques, np.array([0.5, 0.5]))
    assert expensive_first_bus_torques[0] < equal_penalty_torques[0]
    assert expensive_first_bus_torques[1] > equal_penalty_torques[1]
    assert np.isclose(
        control_effectiveness @ expensive_first_bus_torques,
        commanded_acceleration,
    )


@pytest.mark.parametrize(
    ("control_effectiveness", "effort_penalty_matrix", "error_match"),
    [
        (np.array([0.0, 0.0]), np.eye(2), "zero vector"),
        (np.array([1.0, np.nan]), np.eye(2), "finite"),
        (np.array([1.0, 2.0]), np.eye(3), "shape"),
        (
            np.array([1.0, 2.0]),
            np.array(
                [
                    [1.0, 0.1],
                    [0.0, 1.0],
                ]
            ),
            "symmetric",
        ),
        (
            np.array([1.0, 2.0]),
            np.array(
                [
                    [1.0, 0.0],
                    [0.0, 0.0],
                ]
            ),
            "positive definite",
        ),
    ],
)
def test_unconstrained_minimum_effort_allocation_rejects_invalid_inputs(
    control_effectiveness,
    effort_penalty_matrix,
    error_match,
):
    with pytest.raises(ValueError, match=error_match):
        solve_unconstrained_minimum_effort_allocation(
            1.0,
            control_effectiveness,
            effort_penalty_matrix,
        )


def test_unconstrained_minimum_effort_allocation_rejects_non_scalar_command():
    with pytest.raises(ValueError, match="scalar"):
        solve_unconstrained_minimum_effort_allocation(
            np.array([1.0, 2.0]),
            np.array([1.0, 2.0]),
            np.eye(2),
        )
