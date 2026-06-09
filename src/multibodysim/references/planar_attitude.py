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
        if self.duration <= 0.0:
            raise ValueError("duration must be positive.")
        self.reset()

    @property
    def is_initialised(self) -> bool:
        return self.start_time is not None

    @property
    def delta_theta(self) -> float | None:
        if self.theta_initial is None:
            return None
        return self.theta_target - self.theta_initial

    def initialise(
        self,
        start_time: float,
        theta_initial: float,
        theta_dot_initial: float = 0.0,
    ) -> None:
        delta_theta = self.theta_target - float(theta_initial)
        duration = self.duration
        coefficients = np.empty(6, dtype=float)
        coefficients[0] = 0.0
        coefficients[1] = duration * float(theta_dot_initial)
        coefficients[2] = 0.0
        terminal_system = np.array(
            [
                [1.0, 1.0, 1.0],
                [3.0, 4.0, 5.0],
                [6.0, 12.0, 20.0],
            ],
            dtype=float,
        )
        terminal_values = np.array(
            [
                delta_theta - np.sum(coefficients[:3]),
                -(coefficients[1] + 2.0 * coefficients[2]),
                -2.0 * coefficients[2],
            ],
            dtype=float,
        )
        coefficients[3:] = np.linalg.solve(
            terminal_system,
            terminal_values,
        )

        self.start_time = float(start_time)
        self.theta_initial = float(theta_initial)
        self.theta_dot_initial = float(theta_dot_initial)
        self.coefficients = coefficients

    def reset(self) -> None:
        self.start_time = None
        self.theta_initial = None
        self.theta_dot_initial = None
        self.coefficients = None

    def evaluate(self, t: float) -> PlanarAttitudeReferenceState:
        self._require_initialised()
        elapsed_time = float(t) - self.start_time
        if elapsed_time <= 0.0:
            return PlanarAttitudeReferenceState(
                theta=self.theta_initial,
                theta_dot=self.theta_dot_initial,
                theta_ddot=0.0,
            )
        if elapsed_time >= self.duration:
            return PlanarAttitudeReferenceState(
                theta=self.theta_target,
                theta_dot=0.0,
                theta_ddot=0.0,
            )

        normalised_time = elapsed_time / self.duration
        powers = normalised_time ** np.arange(6)
        return PlanarAttitudeReferenceState(
            theta=float(self.theta_initial + self.coefficients @ powers),
            theta_dot=float(
                np.arange(1, 6)
                @ (
                    self.coefficients[1:]
                    * normalised_time ** np.arange(5)
                )
                / self.duration
            ),
            theta_ddot=float(
                np.arange(2, 6)
                @ (
                    np.arange(1, 5)
                    * self.coefficients[2:]
                    * normalised_time ** np.arange(4)
                )
                / self.duration**2
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


class NadirAcquisitionReference:
    """Join the current attitude smoothly to a moving nadir reference."""

    def __init__(
        self,
        duration: float,
        offset: float = -0.5 * np.pi,
    ):
        self.duration = float(duration)
        self.offset = float(offset)
        if self.duration <= 0.0:
            raise ValueError("duration must be positive.")

        self.nadir_reference = NadirPointingReference(offset=self.offset)
        self.reset()

    @property
    def is_initialised(self) -> bool:
        return self.start_time is not None

    def initialise(
        self,
        start_time: float,
        theta_initial: float,
        theta_dot_initial: float,
        *,
        position,
        velocity,
    ) -> None:
        nadir_initial = self.nadir_reference.evaluate(
            start_time,
            position=position,
            velocity=velocity,
        )
        relative_angle_initial = self._wrap_angle(
            float(theta_initial) - nadir_initial.theta
        )
        relative_rate_initial = (
            float(theta_dot_initial) - nadir_initial.theta_dot
        )
        relative_acceleration_initial = -nadir_initial.theta_ddot

        duration = self.duration
        coefficients = np.empty(6, dtype=float)
        coefficients[0] = relative_angle_initial
        coefficients[1] = duration * relative_rate_initial
        coefficients[2] = (
            0.5
            * duration**2
            * relative_acceleration_initial
        )
        terminal_system = np.array(
            [
                [1.0, 1.0, 1.0],
                [3.0, 4.0, 5.0],
                [6.0, 12.0, 20.0],
            ],
            dtype=float,
        )
        terminal_values = np.array(
            [
                -np.sum(coefficients[:3]),
                -(coefficients[1] + 2.0 * coefficients[2]),
                -2.0 * coefficients[2],
            ],
            dtype=float,
        )
        coefficients[3:] = np.linalg.solve(
            terminal_system,
            terminal_values,
        )

        self.start_time = float(start_time)
        self.theta_initial = float(theta_initial)
        self.theta_dot_initial = float(theta_dot_initial)
        self.relative_angle_initial = relative_angle_initial
        self.relative_rate_initial = relative_rate_initial
        self.coefficients = coefficients

    def reset(self) -> None:
        self.start_time = None
        self.theta_initial = None
        self.theta_dot_initial = None
        self.relative_angle_initial = None
        self.relative_rate_initial = None
        self.coefficients = None

    def evaluate(
        self,
        t: float,
        *,
        position,
        velocity,
    ) -> PlanarAttitudeReferenceState:
        self._require_initialised()
        time = float(t)
        nadir = self.nadir_reference.evaluate(
            time,
            position=position,
            velocity=velocity,
        )
        elapsed_time = time - self.start_time

        if elapsed_time <= 0.0:
            return PlanarAttitudeReferenceState(
                theta=self.theta_initial,
                theta_dot=self.theta_dot_initial,
                theta_ddot=0.0,
            )
        if elapsed_time >= self.duration:
            return nadir

        normalised_time = elapsed_time / self.duration
        powers = normalised_time ** np.arange(6)
        relative_angle = float(self.coefficients @ powers)
        relative_rate = float(
            np.arange(1, 6)
            @ (
                self.coefficients[1:]
                * normalised_time ** np.arange(5)
            )
            / self.duration
        )
        relative_acceleration = float(
            np.arange(2, 6)
            @ (
                np.arange(1, 5)
                * self.coefficients[2:]
                * normalised_time ** np.arange(4)
            )
            / self.duration**2
        )

        return PlanarAttitudeReferenceState(
            theta=self._wrap_angle(nadir.theta + relative_angle),
            theta_dot=nadir.theta_dot + relative_rate,
            theta_ddot=nadir.theta_ddot + relative_acceleration,
        )

    def _require_initialised(self) -> None:
        if not self.is_initialised:
            raise RuntimeError(
                "NadirAcquisitionReference must be initialised before "
                "evaluation."
            )

    @staticmethod
    def _wrap_angle(angle: float) -> float:
        return (angle + np.pi) % (2.0 * np.pi) - np.pi
