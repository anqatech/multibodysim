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


class ControlEffectivenessEvaluator:
    """Evaluate bus-torque central-acceleration effectiveness."""

    def __init__(self, dynamics: Any, plant_view: Any):
        self.dynamics = dynamics
        self.plant_view = plant_view
        self.state_dimension = int(dynamics.state_dimension)
        self.torque_count = len(dynamics.rigid_body_names)
        self.central_speed_index = int(plant_view.i_theta_u)

        if not 0 <= self.central_speed_index < self.state_dimension:
            raise ValueError("The plant view has an invalid central-speed index.")

        self._control_force_matrix_evaluator = getattr(
            dynamics,
            "_eval_control_force_matrix",
            None,
        )
        if not callable(self._control_force_matrix_evaluator):
            raise RuntimeError(
                "ControlEffectivenessEvaluator requires dynamics to expose "
                "_eval_control_force_matrix(q, u)."
            )

    def evaluate(self, q, u, mass_matrix) -> ControlEffectivenessVector:
        q_values = _finite_vector(q, self.state_dimension, "q")
        u_values = _finite_vector(u, self.state_dimension, "u")
        mass = _finite_matrix(
            mass_matrix,
            (self.state_dimension, self.state_dimension),
            "mass_matrix",
        )
        control_directions = self._evaluate_control_force_matrix(
            q_values,
            u_values,
        )

        unit_control_accelerations = np.linalg.solve(
            mass,
            control_directions,
        )
        effectiveness = (
            unit_control_accelerations[self.central_speed_index, :]
            .copy()
        )

        if not np.any(effectiveness):
            raise ValueError(
                "Bus torques have zero central-attitude control effectiveness "
                "at this state."
            )

        return ControlEffectivenessVector(
            effectiveness=effectiveness.copy(),
            central_speed_index=self.central_speed_index,
            mass_matrix=mass.copy(),
            control_generalised_force_directions=control_directions.copy(),
            unit_control_accelerations=unit_control_accelerations.copy(),
        )

    def _evaluate_control_force_matrix(self, q, u):
        control_directions = np.asarray(
            self._control_force_matrix_evaluator(q, u),
            dtype=float,
        )
        expected_shape = (self.state_dimension, self.torque_count)
        if control_directions.shape != expected_shape:
            raise ValueError(
                "The control-force matrix evaluator returned shape "
                f"{control_directions.shape}; expected {expected_shape}."
            )
        if not np.all(np.isfinite(control_directions)):
            raise ValueError(
                "The control-force matrix evaluator returned non-finite "
                "values."
            )
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


def _finite_matrix(values, expected_shape, name):
    matrix = np.asarray(values, dtype=float)
    if matrix.shape != expected_shape:
        raise ValueError(
            f"{name} must have shape {expected_shape}; got {matrix.shape}."
        )
    if not np.all(np.isfinite(matrix)):
        raise ValueError(f"{name} must contain only finite values.")
    return matrix
