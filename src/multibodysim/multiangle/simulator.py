from __future__ import annotations

import numpy as np

from .dynamics import MultiAngleFlexibleDynamics


class MultiAngleFlexibleSimulator:

    def __init__(self, config: dict):
        self.config = config
        self.dynamics = MultiAngleFlexibleDynamics(config)

        self.parameter_values = np.array(
            self.dynamics.get_parameter_values(),
            dtype=float,
        )
        self.initial_torque_values = np.array(
            self.dynamics.get_torque_values(),
            dtype=float,
        )
        self.torque_values = self.initial_torque_values.copy()

        self.results = None

    def get_parameter_values(self) -> np.ndarray:
        return self.parameter_values.copy()

    def get_torque_values(self) -> np.ndarray:
        return self.torque_values.copy()

    def reset_torque_values(self):
        self.torque_values = self.initial_torque_values.copy()

    def setup_initial_conditions(self, verbose: bool = True) -> np.ndarray:
        return self.dynamics.get_initial_conditions(verbose=verbose)

    def evaluate_rhs(
        self,
        t: float,
        state: np.ndarray,
        torque_values: np.ndarray | None = None,
    ) -> np.ndarray:
        state = np.asarray(state, dtype=float)
        state_dimension = self.dynamics.state_dimension
        q = state[:state_dimension]
        u = state[state_dimension:]
        torques = (
            self.torque_values
            if torque_values is None
            else np.asarray(torque_values, dtype=float)
        )

        Mk, gk = self.dynamics.eval_kinematics(
            q,
            u,
            self.parameter_values,
            torques,
        )
        qd = -np.linalg.solve(
            np.asarray(Mk, dtype=float),
            np.asarray(gk, dtype=float).squeeze(),
        )

        mass_matrix, forcing = self.dynamics.eval_differentials(
            q,
            u,
            self.parameter_values,
            torques,
        )
        ud = -np.linalg.solve(
            np.asarray(mass_matrix, dtype=float),
            np.asarray(forcing, dtype=float).squeeze(),
        )

        return np.hstack((qd, ud))

    def eval_rhs(self, t: float, state: np.ndarray) -> np.ndarray:
        return self.evaluate_rhs(t, state)

    def get_results(self):
        if self.results is None:
            raise ValueError("Simulation has not been run yet.")
        return self.results
