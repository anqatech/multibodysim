from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class MultiAngleMappedState:
    q: np.ndarray
    u: np.ndarray
    centre_of_mass_position: np.ndarray
    centre_of_mass_velocity: np.ndarray

    @property
    def state(self) -> np.ndarray:
        return np.hstack((self.q, self.u))


class MultiAngleCoordinateMapper:
    """Map a physical COM state into multi-angle generalised coordinates.

    Internal attitude, relative-angle and flexible states are retained from
    the supplied templates. Only the global translation coordinates and
    speeds are solved so the complete model has the requested centre-of-mass
    position and velocity.
    """

    def __init__(self, dynamics: Any):
        self.dynamics = dynamics
        self.state_dimension = int(dynamics.state_dimension)
        self.q1_index = self._symbol_index(
            dynamics.q,
            dynamics.q_translation["x"],
        )
        self.q2_index = self._symbol_index(
            dynamics.q,
            dynamics.q_translation["y"],
        )
        self.u1_index = self._symbol_index(
            dynamics.u,
            dynamics.u_translation["x"],
        )
        self.u2_index = self._symbol_index(
            dynamics.u,
            dynamics.u_translation["y"],
        )

    def map(
        self,
        q_template: np.ndarray,
        u_template: np.ndarray,
        *,
        centre_of_mass_position: np.ndarray,
        centre_of_mass_velocity: np.ndarray,
    ) -> MultiAngleMappedState:
        q_values = self._state_vector(q_template, "q_template")
        u_values = self._state_vector(u_template, "u_template")
        target_position = self._planar_vector(
            centre_of_mass_position,
            "centre_of_mass_position",
        )
        target_velocity = self._planar_vector(
            centre_of_mass_velocity,
            "centre_of_mass_velocity",
        )

        q_values[self.q1_index] = 0.0
        q_values[self.q2_index] = 0.0
        centre_at_translation_origin = np.asarray(
            self.dynamics.rG_func(q_values, u_values),
            dtype=float,
        ).reshape(-1)
        q_values[self.q1_index] = (
            target_position[0] - centre_at_translation_origin[0]
        )
        q_values[self.q2_index] = (
            target_position[1] - centre_at_translation_origin[1]
        )

        u_values[self.u1_index] = 0.0
        u_values[self.u2_index] = 0.0
        velocity_without_translation = np.asarray(
            self.dynamics.vG_func(q_values, u_values),
            dtype=float,
        ).reshape(-1)
        u_values[self.u1_index] = (
            target_velocity[0] - velocity_without_translation[0]
        )
        u_values[self.u2_index] = (
            target_velocity[1] - velocity_without_translation[1]
        )

        mapped_position = np.asarray(
            self.dynamics.rG_func(q_values, u_values),
            dtype=float,
        ).reshape(-1)[:2]
        mapped_velocity = np.asarray(
            self.dynamics.vG_func(q_values, u_values),
            dtype=float,
        ).reshape(-1)[:2]

        return MultiAngleMappedState(
            q=q_values,
            u=u_values,
            centre_of_mass_position=mapped_position,
            centre_of_mass_velocity=mapped_velocity,
        )

    def _state_vector(self, values, name: str) -> np.ndarray:
        vector = np.asarray(values, dtype=float).reshape(-1).copy()
        if vector.size != self.state_dimension:
            raise ValueError(
                f"{name} must contain {self.state_dimension} values; "
                f"got {vector.size}."
            )
        if not np.all(np.isfinite(vector)):
            raise ValueError(f"{name} must contain only finite values.")
        return vector

    @staticmethod
    def _planar_vector(values, name: str) -> np.ndarray:
        vector = np.asarray(values, dtype=float).reshape(-1)
        if vector.size < 2:
            raise ValueError(f"{name} must contain at least two values.")
        vector = vector[:2].copy()
        if not np.all(np.isfinite(vector)):
            raise ValueError(f"{name} must contain only finite values.")
        return vector

    @staticmethod
    def _symbol_index(symbols, target) -> int:
        return list(symbols).index(target)
