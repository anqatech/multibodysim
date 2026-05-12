from __future__ import annotations

import numpy as np

from .dynamics import MultiAngleFlexibleDynamics


class MultiAngleFlexibleSimulator:
    """Minimal simulator scaffold for the multi-angle flexible dynamics model."""

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

    def get_results(self):
        if self.results is None:
            raise ValueError("Simulation has not been run yet.")
        return self.results
