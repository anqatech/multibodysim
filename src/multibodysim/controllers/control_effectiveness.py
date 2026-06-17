from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from ..allocation import evaluate_control_effectiveness_vector


@dataclass(frozen=True)
class ScalarControlEffectiveness:
    control_effectiveness: float
    reciprocal_control_effectiveness: float
    central_speed_index: int
    torque_weights: np.ndarray
    mass_matrix: np.ndarray
    control_generalised_force_direction: np.ndarray
    unit_control_acceleration: np.ndarray


def evaluate_scalar_control_effectiveness(
    dynamics: Any,
    plant_view: Any,
    torque_weights,
    q,
    u,
    *,
    baseline_torques=None,
) -> ScalarControlEffectiveness:
    """Evaluate the local central-acceleration control channel.

    The scalar command is distributed through ``torque_weights``. Its local
    central-attitude effectiveness is extracted from the full coupled system,

        g_tau = s_theta.T @ M(q)^-1 @ B(q).

    The reciprocal converts a requested instantaneous central angular
    acceleration into scalar commanded torque. It is not a physical inertia
    and is not, by itself, a feedback gain-design model.
    """

    torque_count = len(dynamics.rigid_body_names)
    weights = _finite_vector(
        torque_weights,
        torque_count,
        "torque_weights",
    )
    if not np.any(weights):
        raise ValueError(
            "torque_weights must define a non-zero control direction."
        )

    vector_result = evaluate_control_effectiveness_vector(
        dynamics,
        plant_view,
        q,
        u,
        baseline_torques=baseline_torques,
    )
    control_effectiveness = float(
        vector_result.effectiveness @ weights
    )
    if abs(control_effectiveness) <= np.finfo(float).eps:
        raise ValueError(
            "The selected torque weights have zero central-attitude "
            "control effectiveness at this state."
        )

    control_direction = (
        vector_result.control_generalised_force_directions
        @ weights.reshape(-1, 1)
    )
    unit_control_acceleration = (
        vector_result.unit_control_accelerations
        @ weights.reshape(-1, 1)
    )

    return ScalarControlEffectiveness(
        control_effectiveness=control_effectiveness,
        reciprocal_control_effectiveness=1.0 / control_effectiveness,
        central_speed_index=vector_result.central_speed_index,
        torque_weights=weights.copy(),
        mass_matrix=vector_result.mass_matrix.copy(),
        control_generalised_force_direction=control_direction,
        unit_control_acceleration=unit_control_acceleration,
    )


def _finite_vector(values, expected_size, name):
    vector = np.asarray(values, dtype=float).reshape(-1)
    if vector.size != expected_size:
        raise ValueError(
            f"{name} must contain {expected_size} values; "
            f"got {vector.size}."
        )
    if not np.all(np.isfinite(vector)):
        raise ValueError(f"{name} must contain only finite values.")
    return vector
