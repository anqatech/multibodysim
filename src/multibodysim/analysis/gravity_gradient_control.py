from __future__ import annotations

from typing import Any

import numpy as np

from ..codegen import prepare_autowrap_gravity_gradient_evaluator


def gravity_gradient_control_diagnostic(
    simulator: Any,
    state: np.ndarray,
    *,
    prepared_evaluator: dict | None = None,
    torque_weights: np.ndarray | None = None,
) -> dict:
    """Reduce the full gravity-gradient loading to the scalar control channel.

    The simulator equations use

        M(q) ud = f(q, u, tau).

    This diagnostic evaluates the gravity-gradient contribution Q_GG(q) and
    constructs the control direction B(q) from the forcing change produced by
    one unit of scalar control torque distributed using ``torque_weights``.
    """
    dynamics = _require_prepared_gg_simulator(simulator)
    q, u = _split_state(state, dynamics.state_dimension)
    weights = _control_weights(simulator, dynamics, torque_weights)
    prepared = _gravity_gradient_evaluator(dynamics, prepared_evaluator)

    gravity_gradient_forces = _column_vector(
        prepared["function"](q),
        dynamics.state_dimension,
        "gravity-gradient evaluator output",
    )

    zero_torques = np.zeros(len(dynamics.rigid_body_names), dtype=float)
    unit_control_torques = weights.copy()

    mass_matrix, zero_torque_forcing = _evaluate_differentials(
        dynamics,
        q,
        u,
        zero_torques,
    )
    unit_mass_matrix, unit_control_forcing = _evaluate_differentials(
        dynamics,
        q,
        u,
        unit_control_torques,
    )

    if not np.allclose(
        mass_matrix,
        unit_mass_matrix,
        rtol=1e-12,
        atol=1e-12,
    ):
        raise RuntimeError(
            "The differential mass matrix changed with applied torque; "
            "the numerical forcing difference cannot be interpreted as B(q)."
        )

    control_direction = unit_control_forcing - zero_torque_forcing
    gravity_gradient_acceleration = np.linalg.solve(
        mass_matrix,
        gravity_gradient_forces,
    )
    unit_control_acceleration = np.linalg.solve(
        mass_matrix,
        control_direction,
    )

    central_speed_index = int(simulator.plant_view.i_theta_u)
    if not 0 <= central_speed_index < dynamics.state_dimension:
        raise ValueError(
            "The simulator plant view has an invalid central-speed index."
        )

    central_gg_acceleration = float(
        gravity_gradient_acceleration[central_speed_index, 0]
    )
    control_effectiveness = float(
        unit_control_acceleration[central_speed_index, 0]
    )
    if abs(control_effectiveness) <= np.finfo(float).eps:
        raise ValueError(
            "The selected torque weights have zero central-attitude control "
            "effectiveness at this state."
        )

    effective_attitude_inertia = 1.0 / control_effectiveness
    equivalent_gg_torque = central_gg_acceleration / control_effectiveness
    cancellation_torque = -equivalent_gg_torque

    cancellation_acceleration = np.linalg.solve(
        mass_matrix,
        gravity_gradient_forces
        + cancellation_torque * control_direction,
    )
    central_cancellation_residual = float(
        cancellation_acceleration[central_speed_index, 0]
    )

    return {
        "q": q.copy(),
        "u": u.copy(),
        "torque_weights": weights.copy(),
        "central_speed_index": central_speed_index,
        "mass_matrix": mass_matrix,
        "zero_torque_forcing": zero_torque_forcing,
        "unit_control_forcing": unit_control_forcing,
        "gravity_gradient_generalised_forces": gravity_gradient_forces,
        "control_generalised_force_direction": control_direction,
        "gravity_gradient_acceleration": gravity_gradient_acceleration,
        "unit_control_acceleration": unit_control_acceleration,
        "central_gravity_gradient_acceleration": central_gg_acceleration,
        "control_effectiveness": control_effectiveness,
        "effective_attitude_inertia": effective_attitude_inertia,
        "equivalent_gravity_gradient_torque": equivalent_gg_torque,
        "cancellation_torque": cancellation_torque,
        "cancellation_acceleration": cancellation_acceleration,
        "central_cancellation_residual_acceleration": (
            central_cancellation_residual
        ),
        "evaluator_metadata": prepared.get("metadata"),
        "evaluator_timing": prepared.get("timing"),
        "evaluator_artifact_dir": prepared.get("artifact_dir"),
    }


def _require_prepared_gg_simulator(simulator: Any):
    dynamics = getattr(simulator, "dynamics", None)
    plant_view = getattr(simulator, "plant_view", None)
    if dynamics is None or plant_view is None:
        raise TypeError(
            "gravity_gradient_control_diagnostic expects a "
            "MultiAngleFlexibleSimulator."
        )
    if not getattr(dynamics, "enable_gravity_gradient", False):
        raise ValueError(
            "Gravity-gradient control diagnostics require "
            "enable_gravity_gradient=True."
        )
    if getattr(dynamics, "_eval_differentials", None) is None:
        raise RuntimeError(
            "The simulator differential evaluator is not prepared."
        )
    return dynamics


def _split_state(state: np.ndarray, state_dimension: int):
    values = np.asarray(state, dtype=float).reshape(-1)
    expected_size = 2 * state_dimension
    if values.size != expected_size:
        raise ValueError(
            f"state must contain {expected_size} values; got {values.size}."
        )
    if not np.all(np.isfinite(values)):
        raise ValueError("state must contain only finite values.")
    return values[:state_dimension], values[state_dimension:]


def _control_weights(simulator, dynamics, torque_weights):
    if torque_weights is None:
        torque_weights = simulator.torque_weights

    weights = np.asarray(torque_weights, dtype=float).reshape(-1)
    expected_size = len(dynamics.rigid_body_names)
    if weights.size != expected_size:
        raise ValueError(
            f"torque_weights must contain {expected_size} values; "
            f"got {weights.size}."
        )
    if not np.all(np.isfinite(weights)):
        raise ValueError("torque_weights must contain only finite values.")
    if not np.any(weights):
        raise ValueError("torque_weights must define a non-zero control direction.")
    return weights


def _gravity_gradient_evaluator(dynamics, prepared_evaluator):
    if prepared_evaluator is None:
        prepared_evaluator = prepare_autowrap_gravity_gradient_evaluator(
            dynamics,
        )
    if not isinstance(prepared_evaluator, dict):
        raise TypeError("prepared_evaluator must be a prepared evaluator dictionary.")
    if not callable(prepared_evaluator.get("function")):
        raise ValueError(
            "prepared_evaluator must contain a callable 'function'."
        )
    return prepared_evaluator


def _evaluate_differentials(dynamics, q, u, torques):
    mass_matrix, forcing = dynamics._eval_differentials(q, u, torques)
    state_dimension = dynamics.state_dimension
    mass_matrix = np.asarray(mass_matrix, dtype=float)
    if mass_matrix.shape != (state_dimension, state_dimension):
        raise ValueError(
            "The differential evaluator returned mass matrix shape "
            f"{mass_matrix.shape}; expected "
            f"({state_dimension}, {state_dimension})."
        )
    forcing = _column_vector(
        forcing,
        state_dimension,
        "differential forcing",
    )
    return mass_matrix, forcing


def _column_vector(values, expected_size: int, name: str):
    vector = np.asarray(values, dtype=float)
    if vector.size != expected_size:
        raise ValueError(
            f"{name} must contain {expected_size} values; got {vector.size}."
        )
    vector = vector.reshape(expected_size, 1)
    if not np.all(np.isfinite(vector)):
        raise ValueError(f"{name} must contain only finite values.")
    return vector
