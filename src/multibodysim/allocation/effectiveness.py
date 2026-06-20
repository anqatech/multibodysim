from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class ControlEffectivenessVector:
    effectiveness: np.ndarray
    central_speed_index: int
    mass_matrix: np.ndarray
    control_generalised_force_directions: np.ndarray
    unit_control_accelerations: np.ndarray


def evaluate_control_effectiveness_vector(
    dynamics: Any,
    plant_view: Any,
    q,
    u,
    *,
    baseline_torques=None,
) -> ControlEffectivenessVector:
    """Evaluate each bus torque's local central-acceleration effectiveness."""

    state_dimension = int(dynamics.state_dimension)
    q_values = _finite_vector(q, state_dimension, "q")
    u_values = _finite_vector(u, state_dimension, "u")

    torque_count = len(dynamics.rigid_body_names)
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
            "control effectiveness."
        )

    mass_matrix, baseline_forcing = _evaluate_differentials(
        evaluator,
        q_values,
        u_values,
        baseline,
        state_dimension,
    )

    central_speed_index = int(plant_view.i_theta_u)
    if not 0 <= central_speed_index < state_dimension:
        raise ValueError("The plant view has an invalid central-speed index.")

    control_matrix_evaluator = getattr(
        dynamics,
        "_eval_control_force_matrix",
        None,
    )
    if callable(control_matrix_evaluator):
        control_directions = _evaluate_control_force_matrix(
            control_matrix_evaluator,
            q_values,
            u_values,
            state_dimension,
            torque_count,
        )
    else:
        control_directions = _evaluate_control_force_matrix_by_differences(
            evaluator,
            q_values,
            u_values,
            baseline,
            mass_matrix,
            baseline_forcing,
            state_dimension,
            torque_count,
        )

    unit_control_accelerations = np.linalg.solve(
        mass_matrix,
        control_directions,
    )
    effectiveness = unit_control_accelerations[central_speed_index, :].copy()

    if not np.any(effectiveness):
        raise ValueError(
            "Bus torques have zero central-attitude control effectiveness "
            "at this state."
        )

    return ControlEffectivenessVector(
        effectiveness=effectiveness.copy(),
        central_speed_index=central_speed_index,
        mass_matrix=mass_matrix.copy(),
        control_generalised_force_directions=control_directions.copy(),
        unit_control_accelerations=unit_control_accelerations.copy(),
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


def _evaluate_control_force_matrix(
    evaluator,
    q,
    u,
    state_dimension,
    torque_count,
):
    control_directions = np.asarray(evaluator(q, u), dtype=float)
    if control_directions.shape != (state_dimension, torque_count):
        raise ValueError(
            "The control-force matrix evaluator returned shape "
            f"{control_directions.shape}; expected "
            f"({state_dimension}, {torque_count})."
        )
    if not np.all(np.isfinite(control_directions)):
        raise ValueError(
            "The control-force matrix evaluator returned non-finite values."
        )
    return control_directions


def _evaluate_control_force_matrix_by_differences(
    differential_evaluator,
    q,
    u,
    baseline_torques,
    mass_matrix,
    baseline_forcing,
    state_dimension,
    torque_count,
):
    control_directions = np.zeros(
        (state_dimension, torque_count),
        dtype=float,
    )

    for torque_index in range(torque_count):
        unit_torques = baseline_torques.copy()
        unit_torques[torque_index] += 1.0
        unit_mass_matrix, unit_forcing = _evaluate_differentials(
            differential_evaluator,
            q,
            u,
            unit_torques,
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

        control_directions[:, torque_index] = (
            unit_forcing - baseline_forcing
        ).reshape(-1)

    return control_directions


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
