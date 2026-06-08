from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class PlanarAttitudeReferenceState:
    theta: float
    theta_dot: float
    theta_ddot: float


class InertialRestToRestReference:
    def __init__(self, theta_target: float, duration: float):
        self.theta_target = float(theta_target)
        self.duration = float(duration)
        self.start_time = None
        self.theta_initial = None

    @property
    def is_initialised(self) -> bool:
        return self.start_time is not None

    @property
    def delta_theta(self) -> float | None:
        if self.theta_initial is None:
            return None
        return self.theta_target - self.theta_initial

    def initialise(self, start_time: float, theta_initial: float) -> None:
        self.start_time = float(start_time)
        self.theta_initial = float(theta_initial)

    def reset(self) -> None:
        self.start_time = None
        self.theta_initial = None

    def evaluate(self, t: float) -> PlanarAttitudeReferenceState:
        self._require_initialised()
        elapsed_time = float(t) - self.start_time
        delta_theta = self.delta_theta

        return PlanarAttitudeReferenceState(
            theta=(
                self.theta_initial
                + delta_theta
                * self.smooth_step_5th_order(elapsed_time, self.duration)
            ),
            theta_dot=(
                delta_theta
                * self.smooth_step_5th_order_derivative(
                    elapsed_time,
                    self.duration,
                )
            ),
            theta_ddot=(
                delta_theta
                * self.smooth_step_5th_order_second_derivative(
                    elapsed_time,
                    self.duration,
                )
            ),
        )

    def theta(self, t: float) -> float:
        return self.evaluate(t).theta

    def theta_dot(self, t: float) -> float:
        return self.evaluate(t).theta_dot

    def theta_ddot(self, t: float) -> float:
        return self.evaluate(t).theta_ddot

    @staticmethod
    def smooth_step_5th_order(t: float, duration: float) -> float:
        if t <= 0.0:
            return 0.0
        if t >= duration:
            return 1.0
        s = t / duration
        return 10.0 * s**3 - 15.0 * s**4 + 6.0 * s**5

    @staticmethod
    def smooth_step_5th_order_derivative(
        t: float,
        duration: float,
    ) -> float:
        if t <= 0.0 or t >= duration:
            return 0.0
        s = t / duration
        return (30.0 * s**2 - 60.0 * s**3 + 30.0 * s**4) / duration

    @staticmethod
    def smooth_step_5th_order_second_derivative(
        t: float,
        duration: float,
    ) -> float:
        if t <= 0.0 or t >= duration:
            return 0.0
        s = t / duration
        return (
            60.0 * s
            - 180.0 * s**2
            + 120.0 * s**3
        ) / duration**2

    def _require_initialised(self) -> None:
        if not self.is_initialised:
            raise RuntimeError(
                "InertialRestToRestReference must be initialised before "
                "evaluation."
            )


class NadirPointingReference:
    def __init__(self, offset: float = -0.5 * np.pi):
        self.offset = float(offset)

    def evaluate(
        self,
        t: float,
        *,
        position,
        velocity,
    ) -> PlanarAttitudeReferenceState:
        del t

        position = np.asarray(position, dtype=float).reshape(-1)
        velocity = np.asarray(velocity, dtype=float).reshape(-1)
        if position.size < 2 or velocity.size < 2:
            raise ValueError(
                "NadirPointingReference requires planar position and velocity."
            )

        r_x, r_y = position[:2]
        v_x, v_y = velocity[:2]
        radius_squared = r_x**2 + r_y**2
        if radius_squared <= 0.0:
            raise ValueError(
                "NadirPointingReference requires a non-zero orbital radius."
            )

        radius = np.sqrt(radius_squared)
        angular_momentum = r_x * v_y - r_y * v_x
        radial_speed = (r_x * v_x + r_y * v_y) / radius
        theta = np.arctan2(-r_y, -r_x) + self.offset
        theta = (theta + np.pi) % (2.0 * np.pi) - np.pi

        return PlanarAttitudeReferenceState(
            theta=float(theta),
            theta_dot=float(angular_momentum / radius_squared),
            theta_ddot=float(
                -2.0
                * angular_momentum
                * radial_speed
                / radius**3
            ),
        )
