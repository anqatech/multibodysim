from __future__ import annotations

from dataclasses import dataclass
from itertools import product

import numpy as np


MAX_ENUMERATED_CHANNELS = 5
ACTIVE_SET_STATES = ("F", "L", "U")


@dataclass(frozen=True)
class BoundedMinimumEffortAllocation:
    torque_increments: np.ndarray
    achieved_acceleration: float
    residual: float
    cost: float
    active_set: tuple[str, ...]
    lagrange_multiplier: float
    lower_multipliers: np.ndarray
    upper_multipliers: np.ndarray
    feasible_acceleration_interval: tuple[float, float]
    candidate_count: int
    valid_candidate_count: int


@dataclass(frozen=True)
class _Candidate:
    torque_increments: np.ndarray
    achieved_acceleration: float
    residual: float
    cost: float
    active_set: tuple[str, ...]
    lagrange_multiplier: float
    lower_multipliers: np.ndarray
    upper_multipliers: np.ndarray
    valid: bool


def evaluate_feasible_acceleration_interval(
    control_effectiveness,
    lower_bounds,
    upper_bounds,
):
    """Return the acceleration interval reachable inside torque bounds."""

    effectiveness = _finite_vector(
        control_effectiveness,
        "control_effectiveness",
    )
    lower, upper = _bounds(lower_bounds, upper_bounds, effectiveness.size)

    lower_contributions = effectiveness * lower
    upper_contributions = effectiveness * upper
    acceleration_min = float(
        np.minimum(lower_contributions, upper_contributions).sum()
    )
    acceleration_max = float(
        np.maximum(lower_contributions, upper_contributions).sum()
    )
    return acceleration_min, acceleration_max


def solve_bounded_minimum_effort_allocation(
    commanded_acceleration,
    control_effectiveness,
    effort_penalty_matrix,
    lower_bounds,
    upper_bounds,
    *,
    preferred_weights=None,
    preferred_penalty_matrix=None,
    tolerance=1e-10,
) -> BoundedMinimumEffortAllocation:
    """Return the bounded minimum-effort torque increments.

    The bounded solve enumerates all lower/free/upper active-set patterns.
    This is intentionally limited to small allocations with at most five
    torque channels.  When preferred weights and a preferred penalty matrix are
    supplied, the objective becomes

        0.5 * tau.T @ R @ tau + 0.5 * (tau - tau_pref).T @ P @ (tau - tau_pref)

    with tau_pref built from the commanded acceleration and preferred weights.
    """

    acceleration = _finite_scalar(
        commanded_acceleration,
        "commanded_acceleration",
    )
    effectiveness = _finite_vector(
        control_effectiveness,
        "control_effectiveness",
    )
    channel_count = effectiveness.size
    if channel_count > MAX_ENUMERATED_CHANNELS:
        raise ValueError(
            "bounded active-set enumeration supports at most "
            f"{MAX_ENUMERATED_CHANNELS} torque channels; "
            f"got {channel_count}."
        )
    lower, upper = _bounds(lower_bounds, upper_bounds, channel_count)
    penalty_matrix = _symmetric_positive_definite_matrix(
        effort_penalty_matrix,
        channel_count,
        "effort_penalty_matrix",
    )
    tolerance = _positive_tolerance(tolerance)
    (
        objective_matrix,
        objective_linear,
        objective_constant,
    ) = _objective_terms(
        acceleration,
        effectiveness,
        penalty_matrix,
        preferred_weights,
        preferred_penalty_matrix,
        tolerance,
    )

    feasible_interval = evaluate_feasible_acceleration_interval(
        effectiveness,
        lower,
        upper,
    )
    acceleration_min, acceleration_max = feasible_interval
    if (
        acceleration < acceleration_min - tolerance
        or acceleration > acceleration_max + tolerance
    ):
        raise ValueError(
            "commanded_acceleration is outside the feasible acceleration "
            "interval "
            f"[{acceleration_min}, {acceleration_max}]."
        )

    candidates = [
        _solve_active_set_pattern(
            pattern,
            acceleration,
            effectiveness,
            objective_matrix,
            objective_linear,
            objective_constant,
            lower,
            upper,
            tolerance,
        )
        for pattern in product(ACTIVE_SET_STATES, repeat=channel_count)
    ]
    valid_candidates = [candidate for candidate in candidates if candidate.valid]
    if not valid_candidates:
        raise RuntimeError(
            "No KKT-consistent bounded minimum-effort allocation was found."
        )

    selected = min(valid_candidates, key=lambda candidate: candidate.cost)
    return BoundedMinimumEffortAllocation(
        torque_increments=selected.torque_increments.copy(),
        achieved_acceleration=selected.achieved_acceleration,
        residual=selected.residual,
        cost=selected.cost,
        active_set=selected.active_set,
        lagrange_multiplier=selected.lagrange_multiplier,
        lower_multipliers=selected.lower_multipliers.copy(),
        upper_multipliers=selected.upper_multipliers.copy(),
        feasible_acceleration_interval=feasible_interval,
        candidate_count=len(candidates),
        valid_candidate_count=len(valid_candidates),
    )


def _solve_active_set_pattern(
    pattern,
    commanded_acceleration,
    effectiveness,
    objective_matrix,
    objective_linear,
    objective_constant,
    lower_bounds,
    upper_bounds,
    tolerance,
):
    channel_count = effectiveness.size
    torque_increments = np.zeros(channel_count, dtype=float)
    free_indices = []
    active_indices = []

    for index, state in enumerate(pattern):
        if state == "F":
            free_indices.append(index)
        elif state == "L":
            torque_increments[index] = lower_bounds[index]
            active_indices.append(index)
        elif state == "U":
            torque_increments[index] = upper_bounds[index]
            active_indices.append(index)

    free_indices = np.asarray(free_indices, dtype=int)
    active_indices = np.asarray(active_indices, dtype=int)

    if free_indices.size:
        free_torques, lagrange_multiplier = _solve_free_components(
            free_indices,
            active_indices,
            torque_increments[active_indices],
            commanded_acceleration,
            effectiveness,
            objective_matrix,
            objective_linear,
            tolerance,
        )
        if free_torques is None:
            return _invalid_candidate(pattern, channel_count)
        torque_increments[free_indices] = free_torques
    else:
        if abs(effectiveness @ torque_increments - commanded_acceleration) > (
            tolerance
        ):
            return _invalid_candidate(pattern, channel_count)
        interval = _all_active_lagrange_multiplier_interval(
            pattern,
            torque_increments,
            effectiveness,
            objective_matrix,
            objective_linear,
            tolerance,
        )
        if interval is None:
            return _invalid_candidate(pattern, channel_count)
        lagrange_multiplier = _choose_lagrange_multiplier(interval)

    return _evaluate_candidate(
        pattern,
        torque_increments,
        lagrange_multiplier,
        commanded_acceleration,
        effectiveness,
        objective_matrix,
        objective_linear,
        objective_constant,
        lower_bounds,
        upper_bounds,
        tolerance,
    )


def _solve_free_components(
    free_indices,
    active_indices,
    active_torques,
    commanded_acceleration,
    effectiveness,
    objective_matrix,
    objective_linear,
    tolerance,
):
    free_effectiveness = effectiveness[free_indices]
    free_penalty = objective_matrix[np.ix_(free_indices, free_indices)]
    effort_shift = objective_linear[free_indices].copy()

    if active_indices.size:
        free_active_penalty = objective_matrix[
            np.ix_(free_indices, active_indices)
        ]
        effort_shift += free_active_penalty @ active_torques
        residual_acceleration = commanded_acceleration - float(
            effectiveness[active_indices] @ active_torques
        )
    else:
        residual_acceleration = commanded_acceleration

    weighted_effectiveness = np.linalg.solve(
        free_penalty,
        free_effectiveness,
    )
    denominator = float(free_effectiveness @ weighted_effectiveness)
    if denominator <= tolerance:
        return None, None

    weighted_effort_shift = np.linalg.solve(
        free_penalty,
        effort_shift,
    )
    lagrange_multiplier = (
        residual_acceleration
        + float(free_effectiveness @ weighted_effort_shift)
    ) / denominator
    free_torques = np.linalg.solve(
        free_penalty,
        lagrange_multiplier * free_effectiveness - effort_shift,
    )
    return free_torques, float(lagrange_multiplier)


def _evaluate_candidate(
    pattern,
    torque_increments,
    lagrange_multiplier,
    commanded_acceleration,
    effectiveness,
    objective_matrix,
    objective_linear,
    objective_constant,
    lower_bounds,
    upper_bounds,
    tolerance,
):
    channel_count = effectiveness.size
    achieved_acceleration = float(effectiveness @ torque_increments)
    residual = achieved_acceleration - commanded_acceleration
    lower_violation = np.maximum(lower_bounds - torque_increments, 0.0)
    upper_violation = np.maximum(torque_increments - upper_bounds, 0.0)
    bounds_ok = (
        float(np.max(lower_violation)) <= tolerance
        and float(np.max(upper_violation)) <= tolerance
    )
    equality_ok = abs(residual) <= tolerance

    stationarity_residual = (
        objective_matrix @ torque_increments
        + objective_linear
        - lagrange_multiplier * effectiveness
    )
    lower_multipliers = np.zeros(channel_count, dtype=float)
    upper_multipliers = np.zeros(channel_count, dtype=float)
    free_stationarity_ok = True
    active_signs_ok = True

    for index, state in enumerate(pattern):
        if state == "F":
            if abs(stationarity_residual[index]) > tolerance:
                free_stationarity_ok = False
        elif state == "L":
            lower_multipliers[index] = stationarity_residual[index]
            if lower_multipliers[index] < -tolerance:
                active_signs_ok = False
        elif state == "U":
            upper_multipliers[index] = -stationarity_residual[index]
            if upper_multipliers[index] < -tolerance:
                active_signs_ok = False

    valid = bool(
        bounds_ok
        and equality_ok
        and free_stationarity_ok
        and active_signs_ok
    )
    return _Candidate(
        torque_increments=torque_increments.copy(),
        achieved_acceleration=achieved_acceleration,
        residual=residual,
        cost=(
            0.5 * float(torque_increments @ objective_matrix @ torque_increments)
            + float(objective_linear @ torque_increments)
            + float(objective_constant)
        ),
        active_set=tuple(pattern),
        lagrange_multiplier=float(lagrange_multiplier),
        lower_multipliers=lower_multipliers,
        upper_multipliers=upper_multipliers,
        valid=valid,
    )


def _all_active_lagrange_multiplier_interval(
    pattern,
    torque_increments,
    effectiveness,
    objective_matrix,
    objective_linear,
    tolerance,
):
    residual_without_multiplier = (
        objective_matrix @ torque_increments
        + objective_linear
    )
    lower_limit = -np.inf
    upper_limit = np.inf

    for index, state in enumerate(pattern):
        effectiveness_i = effectiveness[index]
        residual_i = residual_without_multiplier[index]
        if state == "L":
            interval = _lambda_interval_for_lower_bound(
                residual_i,
                effectiveness_i,
                tolerance,
            )
        elif state == "U":
            interval = _lambda_interval_for_upper_bound(
                residual_i,
                effectiveness_i,
                tolerance,
            )
        else:
            continue
        if interval is None:
            return None
        lower_limit = max(lower_limit, interval[0])
        upper_limit = min(upper_limit, interval[1])

    if lower_limit > upper_limit + tolerance:
        return None
    return lower_limit, upper_limit


def _lambda_interval_for_lower_bound(
    residual_without_multiplier,
    effectiveness,
    tolerance,
):
    if abs(effectiveness) <= tolerance:
        if residual_without_multiplier < -tolerance:
            return None
        return -np.inf, np.inf
    crossing = residual_without_multiplier / effectiveness
    if effectiveness > 0.0:
        return -np.inf, crossing
    return crossing, np.inf


def _lambda_interval_for_upper_bound(
    residual_without_multiplier,
    effectiveness,
    tolerance,
):
    if abs(effectiveness) <= tolerance:
        if residual_without_multiplier > tolerance:
            return None
        return -np.inf, np.inf
    crossing = residual_without_multiplier / effectiveness
    if effectiveness > 0.0:
        return crossing, np.inf
    return -np.inf, crossing


def _choose_lagrange_multiplier(interval):
    lower_limit, upper_limit = interval
    if lower_limit <= 0.0 <= upper_limit:
        return 0.0
    if np.isfinite(lower_limit):
        return float(lower_limit)
    return float(upper_limit)


def _objective_terms(
    commanded_acceleration,
    effectiveness,
    effort_penalty_matrix,
    preferred_weights,
    preferred_penalty_matrix,
    tolerance,
):
    channel_count = effectiveness.size
    if preferred_weights is None and preferred_penalty_matrix is None:
        return (
            effort_penalty_matrix,
            np.zeros(channel_count, dtype=float),
            0.0,
        )
    if preferred_weights is None or preferred_penalty_matrix is None:
        raise ValueError(
            "preferred_weights and preferred_penalty_matrix must be supplied "
            "together."
        )

    weights = _nonnegative_weight_vector(
        preferred_weights,
        channel_count,
        "preferred_weights",
        tolerance,
    )
    preferred_penalty = _symmetric_positive_semidefinite_matrix(
        preferred_penalty_matrix,
        channel_count,
        "preferred_penalty_matrix",
        tolerance,
    )

    preferred_acceleration_gain = float(effectiveness @ weights)
    if abs(preferred_acceleration_gain) <= tolerance:
        raise ValueError(
            "preferred_weights must satisfy "
            "abs(control_effectiveness @ preferred_weights) > tolerance."
        )

    preferred_torques = (
        commanded_acceleration
        / preferred_acceleration_gain
        * weights
    )
    objective_matrix = effort_penalty_matrix + preferred_penalty
    objective_linear = -(preferred_penalty @ preferred_torques)
    objective_constant = (
        0.5 * float(preferred_torques @ preferred_penalty @ preferred_torques)
    )
    return objective_matrix, objective_linear, objective_constant


def _invalid_candidate(pattern, channel_count):
    return _Candidate(
        torque_increments=np.full(channel_count, np.nan),
        achieved_acceleration=np.nan,
        residual=np.nan,
        cost=np.inf,
        active_set=tuple(pattern),
        lagrange_multiplier=np.nan,
        lower_multipliers=np.full(channel_count, np.nan),
        upper_multipliers=np.full(channel_count, np.nan),
        valid=False,
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


def _nonnegative_weight_vector(values, expected_size, name, tolerance):
    vector = np.asarray(values, dtype=float).reshape(-1)
    if vector.size != expected_size:
        raise ValueError(
            f"{name} must contain {expected_size} values; got {vector.size}."
        )
    if not np.all(np.isfinite(vector)):
        raise ValueError(f"{name} must contain only finite values.")
    if np.any(vector < -tolerance):
        raise ValueError(f"{name} must contain only non-negative values.")
    vector = np.maximum(vector, 0.0)
    if not np.any(vector):
        raise ValueError(f"{name} must not be the zero vector.")
    return vector


def _bounds(lower_bounds, upper_bounds, expected_size):
    lower = np.asarray(lower_bounds, dtype=float).reshape(-1)
    upper = np.asarray(upper_bounds, dtype=float).reshape(-1)
    if lower.size != expected_size:
        raise ValueError(
            "lower_bounds must contain "
            f"{expected_size} values; got {lower.size}."
        )
    if upper.size != expected_size:
        raise ValueError(
            "upper_bounds must contain "
            f"{expected_size} values; got {upper.size}."
        )
    if not np.all(np.isfinite(lower)):
        raise ValueError("lower_bounds must contain only finite values.")
    if not np.all(np.isfinite(upper)):
        raise ValueError("upper_bounds must contain only finite values.")
    if np.any(lower > upper):
        raise ValueError("lower_bounds must be less than or equal to upper_bounds.")
    return lower, upper


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


def _symmetric_positive_semidefinite_matrix(
    matrix,
    expected_size,
    name,
    tolerance,
):
    matrix = np.asarray(matrix, dtype=float)
    if matrix.shape != (expected_size, expected_size):
        raise ValueError(
            f"{name} must have shape ({expected_size}, {expected_size}); "
            f"got {matrix.shape}."
        )
    if not np.all(np.isfinite(matrix)):
        raise ValueError(f"{name} must contain only finite values.")
    if not np.allclose(matrix, matrix.T, rtol=0.0, atol=tolerance):
        raise ValueError(f"{name} must be symmetric.")
    eigenvalues = np.linalg.eigvalsh(matrix)
    if float(np.min(eigenvalues)) < -tolerance:
        raise ValueError(f"{name} must be positive semidefinite.")
    return matrix


def _positive_tolerance(value):
    tolerance = _finite_scalar(value, "tolerance")
    if tolerance <= 0.0:
        raise ValueError("tolerance must be positive.")
    return tolerance
