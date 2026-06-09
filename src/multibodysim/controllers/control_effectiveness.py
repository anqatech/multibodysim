from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


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

    state_dimension = int(dynamics.state_dimension)
    q_values = _finite_vector(q, state_dimension, "q")
    u_values = _finite_vector(u, state_dimension, "u")

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

    if baseline_torques is None:
        baseline = np.zeros(torque_count, dtype=float)
    else:
        baseline = _finite_vector(
            baseline_torques,
            torque_count,
            "baseline_torques",
        )

    evaluator = getattr(dynamics, "_eval_differentials", None)
    if not callable(evaluator):
        raise RuntimeError(
            "The differential evaluator must be prepared before evaluating "
            "scalar control effectiveness."
        )

    mass_matrix, baseline_forcing = _evaluate_differentials(
        evaluator,
        q_values,
        u_values,
        baseline,
        state_dimension,
    )
    unit_mass_matrix, unit_forcing = _evaluate_differentials(
        evaluator,
        q_values,
        u_values,
        baseline + weights,
        state_dimension,
    )
    if not np.allclose(
        mass_matrix,
        unit_mass_matrix,
        rtol=1e-12,
        atol=1e-12,
    ):
        raise RuntimeError(
            "The differential mass matrix changed with applied torque; "
            "the forcing difference cannot be interpreted as B(q)."
        )

    control_direction = unit_forcing - baseline_forcing
    unit_control_acceleration = np.linalg.solve(
        mass_matrix,
        control_direction,
    )

    central_speed_index = int(plant_view.i_theta_u)
    if not 0 <= central_speed_index < state_dimension:
        raise ValueError(
            "The plant view has an invalid central-speed index."
        )
    control_effectiveness = float(
        unit_control_acceleration[central_speed_index, 0]
    )
    if abs(control_effectiveness) <= np.finfo(float).eps:
        raise ValueError(
            "The selected torque weights have zero central-attitude "
            "control effectiveness at this state."
        )

    return ScalarControlEffectiveness(
        control_effectiveness=control_effectiveness,
        reciprocal_control_effectiveness=1.0 / control_effectiveness,
        central_speed_index=central_speed_index,
        torque_weights=weights.copy(),
        mass_matrix=mass_matrix,
        control_generalised_force_direction=control_direction,
        unit_control_acceleration=unit_control_acceleration,
    )


def _evaluate_differentials(
    evaluator,
    q,
    u,
    torques,
    state_dimension,
):
    mass_matrix, forcing = evaluator(q, u, torques)
    mass_matrix = np.asarray(mass_matrix, dtype=float)
    if mass_matrix.shape != (state_dimension, state_dimension):
        raise ValueError(
            "The differential evaluator returned mass matrix shape "
            f"{mass_matrix.shape}; expected "
            f"({state_dimension}, {state_dimension})."
        )
    forcing = np.asarray(forcing, dtype=float)
    if forcing.size != state_dimension:
        raise ValueError(
            "The differential evaluator returned forcing with "
            f"{forcing.size} values; expected {state_dimension}."
        )
    forcing = forcing.reshape(state_dimension, 1)
    if not np.all(np.isfinite(mass_matrix)):
        raise ValueError(
            "The differential evaluator returned a non-finite mass matrix."
        )
    if not np.all(np.isfinite(forcing)):
        raise ValueError(
            "The differential evaluator returned non-finite forcing."
        )
    return mass_matrix, forcing


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
