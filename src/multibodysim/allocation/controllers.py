from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from ..controllers.base import ControlOutput
from ..controllers.planar_attitude_reference import (
    PlanarAttitudeCommandState,
    PlanarAttitudeReferenceTracker,
)
from .bounded_minimum_effort import (
    BoundedMinimumEffortAllocation,
    evaluate_feasible_acceleration_interval,
    solve_bounded_minimum_effort_allocation,
)
from .effectiveness import (
    ControlEffectivenessEvaluator,
    ControlEffectivenessVector,
)


@dataclass(frozen=True)
class AllocatedPlanarAttitudeDiagnostics:
    command_state: PlanarAttitudeCommandState
    control_effectiveness: ControlEffectivenessVector
    allocation: BoundedMinimumEffortAllocation
    reference_acceleration: float
    proportional_acceleration: float
    derivative_acceleration: float
    gravity_gradient_acceleration: float
    gravity_gradient_torque: float
    requested_acceleration: float
    saturated_acceleration: float
    feasible_acceleration_interval: tuple[float, float]
    clipped: bool


class AllocatedPlanarAttitudeController:
    """Planar PD controller that allocates scalar acceleration to bus torques."""

    def __init__(
        self,
        dynamics: Any,
        plant_view: Any,
        *,
        effort_penalty_matrix,
        lower_bounds,
        upper_bounds,
        preferred_weights=None,
        preferred_penalty_matrix=None,
        tolerance=1e-10,
    ):
        self.dynamics = dynamics
        self.plant_view = plant_view
        self.reference_tracker = PlanarAttitudeReferenceTracker(plant_view)
        self.control_effectiveness_evaluator = ControlEffectivenessEvaluator(
            dynamics,
            plant_view,
        )

        self.effort_penalty_matrix = np.asarray(
            effort_penalty_matrix,
            dtype=float,
        ).copy()
        self.lower_bounds = np.asarray(lower_bounds, dtype=float).reshape(-1)
        self.upper_bounds = np.asarray(upper_bounds, dtype=float).reshape(-1)
        self.preferred_weights = _optional_vector(preferred_weights)
        self.preferred_penalty_matrix = _optional_matrix(
            preferred_penalty_matrix,
        )
        self.tolerance = float(tolerance)

        self.Kp_acceleration = 0.0
        self.Kd_acceleration = 0.0
        self.gravity_gradient_feedforward = None
        self.gravity_gradient_acceleration_feedforward = None
        self.gravity_gradient_acceleration_gain = 0.0
        self.last_diagnostics: AllocatedPlanarAttitudeDiagnostics | None = None

    @property
    def mode(self):
        return self.reference_tracker.mode

    @property
    def reference(self):
        return self.reference_tracker.reference

    def reset(self):
        self.reference_tracker.reset()
        self.last_diagnostics = None

    def configure_inertial_pd(
        self,
        *,
        theta_target,
        Kp_acceleration,
        Kd_acceleration,
        manoeuvre_duration,
        use_input_shaping=False,
        shaper=None,
        omega=None,
        zeta=None,
        gravity_gradient_feedforward=None,
        gravity_gradient_acceleration_feedforward=None,
        gravity_gradient_acceleration_gain=0.0,
    ):
        self.Kp_acceleration = _finite_scalar(
            Kp_acceleration,
            "Kp_acceleration",
        )
        self.Kd_acceleration = _finite_scalar(
            Kd_acceleration,
            "Kd_acceleration",
        )
        self._configure_gravity_gradient_feedforward(
            gravity_gradient_feedforward=gravity_gradient_feedforward,
            gravity_gradient_acceleration_feedforward=(
                gravity_gradient_acceleration_feedforward
            ),
            gravity_gradient_acceleration_gain=gravity_gradient_acceleration_gain,
        )
        self.reference_tracker.configure_inertial(
            theta_target=theta_target,
            manoeuvre_duration=manoeuvre_duration,
            use_input_shaping=use_input_shaping,
            shaper=shaper,
            omega=omega,
            zeta=zeta,
        )

    def configure_nadir_pd(
        self,
        *,
        Kp_acceleration,
        Kd_acceleration,
        manoeuvre_duration,
        offset=-0.5 * np.pi,
        gravity_gradient_feedforward=None,
        gravity_gradient_acceleration_feedforward=None,
        gravity_gradient_acceleration_gain=0.0,
    ):
        self.Kp_acceleration = _finite_scalar(
            Kp_acceleration,
            "Kp_acceleration",
        )
        self.Kd_acceleration = _finite_scalar(
            Kd_acceleration,
            "Kd_acceleration",
        )
        self._configure_gravity_gradient_feedforward(
            gravity_gradient_feedforward=gravity_gradient_feedforward,
            gravity_gradient_acceleration_feedforward=(
                gravity_gradient_acceleration_feedforward
            ),
            gravity_gradient_acceleration_gain=gravity_gradient_acceleration_gain,
        )
        self.reference_tracker.configure_nadir(
            manoeuvre_duration=manoeuvre_duration,
            offset=offset,
        )

    def compute(self, t, q, u, Md=None):
        if Md is None:
            raise ValueError(
                "AllocatedPlanarAttitudeController requires Md="
                "mass_matrix from the simulator."
            )
        diagnostics = self.evaluate_allocation(t, q, u, mass_matrix=Md)
        self.last_diagnostics = diagnostics
        return ControlOutput(bus_torques=diagnostics.allocation.torque_increments)

    def evaluate_allocation(
        self,
        t,
        q,
        u,
        *,
        mass_matrix,
    ) -> AllocatedPlanarAttitudeDiagnostics:
        command_state = self.reference_tracker.evaluate(t, q, u)
        reference_acceleration = float(command_state.theta_ddot_ref)
        proportional_acceleration = float(
            self.Kp_acceleration * command_state.error
        )
        derivative_acceleration = float(
            self.Kd_acceleration * command_state.error_dot
        )
        gravity_gradient_torque, gravity_gradient_acceleration = (
            self._gravity_gradient_feedforward_terms(
                q,
                u,
                command_state.theta,
                mass_matrix,
            )
        )
        requested_acceleration = float(
            reference_acceleration
            + proportional_acceleration
            + derivative_acceleration
            + gravity_gradient_acceleration
        )

        effectiveness = self.control_effectiveness_evaluator.evaluate(
            q,
            u,
            mass_matrix,
        )
        feasible_interval = evaluate_feasible_acceleration_interval(
            effectiveness.effectiveness,
            self.lower_bounds,
            self.upper_bounds,
        )
        saturated_acceleration = float(
            np.clip(
                requested_acceleration,
                feasible_interval[0],
                feasible_interval[1],
            )
        )
        clipped = not np.isclose(
            saturated_acceleration,
            requested_acceleration,
            rtol=0.0,
            atol=self.tolerance,
        )

        allocation = solve_bounded_minimum_effort_allocation(
            saturated_acceleration,
            effectiveness.effectiveness,
            self.effort_penalty_matrix,
            self.lower_bounds,
            self.upper_bounds,
            preferred_weights=self.preferred_weights,
            preferred_penalty_matrix=self.preferred_penalty_matrix,
            tolerance=self.tolerance,
        )

        return AllocatedPlanarAttitudeDiagnostics(
            command_state=command_state,
            control_effectiveness=effectiveness,
            allocation=allocation,
            reference_acceleration=reference_acceleration,
            proportional_acceleration=proportional_acceleration,
            derivative_acceleration=derivative_acceleration,
            gravity_gradient_acceleration=gravity_gradient_acceleration,
            gravity_gradient_torque=gravity_gradient_torque,
            requested_acceleration=requested_acceleration,
            saturated_acceleration=saturated_acceleration,
            feasible_acceleration_interval=feasible_interval,
            clipped=bool(clipped),
        )

    def _gravity_gradient_feedforward_torque(self, q, u, theta):
        if self.gravity_gradient_feedforward is None:
            return 0.0

        rG_x, rG_y, _, _ = self.plant_view.com_state(q, u)
        result = self.gravity_gradient_feedforward.evaluate(
            centre_of_mass_position=(rG_x, rG_y),
            central_attitude=theta,
        )
        torque = float(result.cancellation_torque)
        if not np.isfinite(torque):
            raise ValueError(
                "gravity-gradient feedforward returned a non-finite "
                "cancellation torque."
            )
        return torque

    def _gravity_gradient_feedforward_terms(self, q, u, theta, mass_matrix):
        if self.gravity_gradient_acceleration_feedforward is not None:
            result = self.gravity_gradient_acceleration_feedforward.evaluate(
                q,
                u,
                mass_matrix,
            )
            acceleration = float(result.cancellation_acceleration)
            if not np.isfinite(acceleration):
                raise ValueError(
                    "gravity-gradient acceleration feedforward returned a "
                    "non-finite cancellation acceleration."
                )
            return 0.0, acceleration

        torque = self._gravity_gradient_feedforward_torque(q, u, theta)
        acceleration = float(
            self.gravity_gradient_acceleration_gain
            * torque
        )
        return torque, acceleration

    def _configure_gravity_gradient_feedforward(
        self,
        *,
        gravity_gradient_feedforward,
        gravity_gradient_acceleration_feedforward,
        gravity_gradient_acceleration_gain,
    ):
        if (
            gravity_gradient_feedforward is not None
            and gravity_gradient_acceleration_feedforward is not None
        ):
            raise ValueError(
                "Use either gravity_gradient_feedforward or "
                "gravity_gradient_acceleration_feedforward, not both."
            )
        if (
            gravity_gradient_acceleration_feedforward is not None
            and float(gravity_gradient_acceleration_gain) != 0.0
        ):
            raise ValueError(
                "gravity_gradient_acceleration_gain is only used with "
                "gravity_gradient_feedforward."
            )
        self.gravity_gradient_feedforward = (
            _validated_gravity_gradient_feedforward(
                gravity_gradient_feedforward,
            )
        )
        self.gravity_gradient_acceleration_feedforward = (
            _validated_gravity_gradient_acceleration_feedforward(
                gravity_gradient_acceleration_feedforward,
            )
        )
        self.gravity_gradient_acceleration_gain = _finite_scalar(
            gravity_gradient_acceleration_gain,
            "gravity_gradient_acceleration_gain",
        )


def _optional_vector(values):
    if values is None:
        return None
    return np.asarray(values, dtype=float).reshape(-1).copy()


def _optional_matrix(values):
    if values is None:
        return None
    return np.asarray(values, dtype=float).copy()


def _finite_scalar(value, name):
    scalar = float(value)
    if not np.isfinite(scalar):
        raise ValueError(f"{name} must be finite.")
    return scalar


def _validated_gravity_gradient_feedforward(value):
    if value is not None and not callable(getattr(value, "evaluate", None)):
        raise TypeError(
            "gravity_gradient_feedforward must expose an evaluate() method."
        )
    return value


def _validated_gravity_gradient_acceleration_feedforward(value):
    if value is not None and not callable(getattr(value, "evaluate", None)):
        raise TypeError(
            "gravity_gradient_acceleration_feedforward must expose an "
            "evaluate() method."
        )
    return value
