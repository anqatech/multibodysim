from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class PlanarCentreOfMassReferenceState:
    time: float
    position: np.ndarray
    velocity: np.ndarray
    mean_anomaly: float
    eccentric_anomaly: float
    true_anomaly: float


class PlanarKeplerianReference:
    """Propagate a planar elliptic centre-of-mass reference orbit.

    The reference uses the two-body Kepler problem in fixed, periapsis-aligned
    inertial axes. The orbit is prograde and restricted to ``0 <= e < 1``.
    """

    def __init__(
        self,
        gravitational_parameter: float,
        semi_major_axis: float,
        eccentricity: float,
        *,
        reference_time: float = 0.0,
        initial_true_anomaly: float = 0.0,
        tolerance: float = 1e-13,
        maximum_iterations: int = 50,
    ):
        self.gravitational_parameter = float(gravitational_parameter)
        self.semi_major_axis = float(semi_major_axis)
        self.eccentricity = float(eccentricity)
        self.reference_time = float(reference_time)
        self.initial_true_anomaly = float(initial_true_anomaly)
        self.tolerance = float(tolerance)
        self.maximum_iterations = int(maximum_iterations)

        if self.gravitational_parameter <= 0.0:
            raise ValueError("gravitational_parameter must be positive.")
        if self.semi_major_axis <= 0.0:
            raise ValueError("semi_major_axis must be positive.")
        if not 0.0 <= self.eccentricity < 1.0:
            raise ValueError(
                "eccentricity must satisfy 0 <= eccentricity < 1."
            )
        if self.tolerance <= 0.0:
            raise ValueError("tolerance must be positive.")
        if self.maximum_iterations < 1:
            raise ValueError("maximum_iterations must be at least 1.")

        self.mean_motion = np.sqrt(
            self.gravitational_parameter / self.semi_major_axis**3
        )
        self.period = 2.0 * np.pi / self.mean_motion
        self.initial_eccentric_anomaly = (
            self._eccentric_anomaly_from_true_anomaly(
                self.initial_true_anomaly
            )
        )
        self.initial_mean_anomaly = (
            self.initial_eccentric_anomaly
            - self.eccentricity
            * np.sin(self.initial_eccentric_anomaly)
        )

    def evaluate(self, t: float) -> PlanarCentreOfMassReferenceState:
        time = float(t)
        mean_anomaly = (
            self.initial_mean_anomaly
            + self.mean_motion * (time - self.reference_time)
        )
        eccentric_anomaly = self._solve_eccentric_anomaly(mean_anomaly)

        eccentricity = self.eccentricity
        semi_major_axis = self.semi_major_axis
        beta = np.sqrt(1.0 - eccentricity**2)
        cos_e = np.cos(eccentric_anomaly)
        sin_e = np.sin(eccentric_anomaly)
        radius_factor = 1.0 - eccentricity * cos_e

        position = np.array(
            [
                semi_major_axis * (cos_e - eccentricity),
                semi_major_axis * beta * sin_e,
            ],
            dtype=float,
        )
        velocity = (
            semi_major_axis
            * self.mean_motion
            / radius_factor
            * np.array(
                [
                    -sin_e,
                    beta * cos_e,
                ],
                dtype=float,
            )
        )
        true_anomaly = np.arctan2(
            beta * sin_e,
            cos_e - eccentricity,
        )

        return PlanarCentreOfMassReferenceState(
            time=time,
            position=position,
            velocity=velocity,
            mean_anomaly=float(self._wrap_angle(mean_anomaly)),
            eccentric_anomaly=float(eccentric_anomaly),
            true_anomaly=float(true_anomaly),
        )

    def _eccentric_anomaly_from_true_anomaly(
        self,
        true_anomaly: float,
    ) -> float:
        eccentricity = self.eccentricity
        return float(
            np.arctan2(
                np.sqrt(1.0 - eccentricity**2)
                * np.sin(true_anomaly),
                eccentricity + np.cos(true_anomaly),
            )
        )

    def _solve_eccentric_anomaly(self, mean_anomaly: float) -> float:
        wrapped_mean_anomaly = self._wrap_angle(mean_anomaly)
        if self.eccentricity == 0.0:
            return float(wrapped_mean_anomaly)

        eccentric_anomaly = (
            wrapped_mean_anomaly
            if self.eccentricity < 0.8
            else np.copysign(np.pi, wrapped_mean_anomaly or 1.0)
        )

        for _ in range(self.maximum_iterations):
            residual = (
                eccentric_anomaly
                - self.eccentricity * np.sin(eccentric_anomaly)
                - wrapped_mean_anomaly
            )
            derivative = (
                1.0
                - self.eccentricity * np.cos(eccentric_anomaly)
            )
            correction = residual / derivative
            eccentric_anomaly -= correction
            if abs(correction) <= self.tolerance:
                return float(eccentric_anomaly)

        raise RuntimeError(
            "Kepler equation did not converge within "
            f"{self.maximum_iterations} iterations."
        )

    @staticmethod
    def _wrap_angle(angle: float) -> float:
        return (angle + np.pi) % (2.0 * np.pi) - np.pi
