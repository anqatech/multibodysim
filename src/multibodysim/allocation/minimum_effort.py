from __future__ import annotations

import numpy as np


def solve_unconstrained_minimum_effort_allocation(
    commanded_acceleration,
    control_effectiveness,
    effort_penalty_matrix,
):
    """Return the minimum-effort torque increments for one scalar command."""

    acceleration = _finite_scalar(
        commanded_acceleration,
        "commanded_acceleration",
    )
    effectiveness = _finite_vector(
        control_effectiveness,
        "control_effectiveness",
    )
    penalty_matrix = _symmetric_positive_definite_matrix(
        effort_penalty_matrix,
        effectiveness.size,
        "effort_penalty_matrix",
    )

    weighted_effectiveness = np.linalg.solve(
        penalty_matrix,
        effectiveness,
    )
    denominator = float(effectiveness @ weighted_effectiveness)
    if denominator <= np.finfo(float).eps:
        raise ValueError(
            "control_effectiveness must contain at least one controllable "
            "direction under effort_penalty_matrix."
        )

    return (
        acceleration
        / denominator
        * weighted_effectiveness
    )


def _finite_scalar(value, name):
    scalar = np.asarray(value, dtype=float)
    if scalar.shape != ():
        raise ValueError(f"{name} must be a scalar.")
    scalar = float(scalar)
    if not np.isfinite(scalar):
        raise ValueError(f"{name} must be finite.")
    return scalar


def _finite_vector(values, name):
    vector = np.asarray(values, dtype=float).reshape(-1)
    if vector.size == 0:
        raise ValueError(f"{name} must contain at least one value.")
    if not np.all(np.isfinite(vector)):
        raise ValueError(f"{name} must contain only finite values.")
    if not np.any(vector):
        raise ValueError(f"{name} must not be the zero vector.")
    return vector


def _symmetric_positive_definite_matrix(matrix, expected_size, name):
    matrix = np.asarray(matrix, dtype=float)
    if matrix.shape != (expected_size, expected_size):
        raise ValueError(
            f"{name} must have shape ({expected_size}, {expected_size}); "
            f"got {matrix.shape}."
        )
    if not np.all(np.isfinite(matrix)):
        raise ValueError(f"{name} must contain only finite values.")
    if not np.allclose(matrix, matrix.T, rtol=0.0, atol=1e-12):
        raise ValueError(f"{name} must be symmetric.")
    try:
        np.linalg.cholesky(matrix)
    except np.linalg.LinAlgError as error:
        raise ValueError(f"{name} must be positive definite.") from error
    return matrix
