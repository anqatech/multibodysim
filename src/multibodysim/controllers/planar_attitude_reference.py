from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..inputshaping.zv_input_shaping import InputShaper
from ..references import (
    InertialRestToRestReference,
    NadirAcquisitionReference,
)


@dataclass(frozen=True)
class PlanarAttitudeCommandState:
    theta: float
    theta_dot: float
    theta_ref: float
    theta_dot_ref: float
    theta_ddot_ref: float
    error: float
    error_dot: float


class PlanarAttitudeReferenceTracker:
    """Shared planar reference/error generator for attitude control."""

    def __init__(self, plant_view):
        self.plant_view = plant_view
        self.mode = None
        self.theta_target = 0.0
        self.use_input_shaping = False
        self.shaper = None
        self.manoeuvre_duration = 0.0
        self.reference = None
        self.manoeuvre_start_time = None
        self.theta_start = None
        self.theta_final = None
        self.delta_theta = None

    def reset(self):
        if isinstance(
            self.reference,
            (InertialRestToRestReference, NadirAcquisitionReference),
        ):
            self.reference.reset()

        self.manoeuvre_start_time = None
        self.theta_start = None
        self.theta_final = None
        self.delta_theta = None

    @staticmethod
    def smooth_step_5th_order(t, Tr):
        return InertialRestToRestReference.smooth_step_5th_order(t, Tr)

    @staticmethod
    def derivative_smooth_step_5th_order(t, Tr):
        return (
            InertialRestToRestReference
            .smooth_step_5th_order_derivative(t, Tr)
        )

    def raw_theta_command(self, t):
        return self.reference.theta(t)

    def raw_theta_dot_command(self, t):
        return self.reference.theta_dot(t)

    def raw_theta_ddot_command(self, t):
        return self.reference.theta_ddot(t)

    def configure_inertial(
        self,
        *,
        theta_target,
        manoeuvre_duration,
        use_input_shaping=False,
        shaper=None,
        omega=None,
        zeta=None,
    ):
        self.mode = "inertial_pd"
        self.theta_target = float(theta_target)
        self.manoeuvre_duration = float(manoeuvre_duration)
        self.use_input_shaping = bool(use_input_shaping)
        self.reference = InertialRestToRestReference(
            theta_target=self.theta_target,
            duration=self.manoeuvre_duration,
        )

        if shaper is not None:
            self.shaper = shaper
        elif self.use_input_shaping:
            if omega is None or zeta is None:
                raise ValueError(
                    "Input shaping requested but omega or zeta not provided."
                )
            self.shaper = InputShaper.zvd(omega=float(omega), zeta=float(zeta))
        else:
            self.shaper = None

        self.reset()

    def configure_nadir(
        self,
        *,
        manoeuvre_duration,
        offset=-0.5 * np.pi,
    ):
        self.mode = "nadir_pd"
        self.manoeuvre_duration = float(manoeuvre_duration)
        self.reference = NadirAcquisitionReference(
            duration=self.manoeuvre_duration,
            offset=offset,
        )
        self.use_input_shaping = False
        self.shaper = None
        self.reset()

    def evaluate(self, t, q, u) -> PlanarAttitudeCommandState:
        if self.mode == "inertial_pd":
            return self._evaluate_inertial(t, q, u)
        if self.mode == "nadir_pd":
            return self._evaluate_nadir(t, q, u)
        raise ValueError("Planar attitude reference tracker is not configured.")

    def _evaluate_inertial(self, t, q, u) -> PlanarAttitudeCommandState:
        theta = self.plant_view.theta(q)
        theta_dot = self.plant_view.theta_dot(u)

        if not self.reference.is_initialised:
            self.reference.initialise(
                start_time=t,
                theta_initial=theta,
                theta_dot_initial=theta_dot,
            )
            self.manoeuvre_start_time = self.reference.start_time
            self.theta_start = self.reference.theta_initial
            self.theta_final = self.theta_target
            self.delta_theta = self.reference.delta_theta

        if self.use_input_shaping and self.shaper is not None:
            theta_ref = self.shaper.shape(t, self.raw_theta_command)
            theta_dot_ref = self.shaper.shape(t, self.raw_theta_dot_command)
            theta_ddot_ref = self.shaper.shape(
                t,
                self.raw_theta_ddot_command,
            )
        else:
            reference_state = self.reference.evaluate(t)
            theta_ref = reference_state.theta
            theta_dot_ref = reference_state.theta_dot
            theta_ddot_ref = reference_state.theta_ddot

        return PlanarAttitudeCommandState(
            theta=float(theta),
            theta_dot=float(theta_dot),
            theta_ref=float(theta_ref),
            theta_dot_ref=float(theta_dot_ref),
            theta_ddot_ref=float(theta_ddot_ref),
            error=float(theta_ref - theta),
            error_dot=float(theta_dot_ref - theta_dot),
        )

    def _evaluate_nadir(self, t, q, u) -> PlanarAttitudeCommandState:
        theta = self.plant_view.theta(q)
        theta_dot = self.plant_view.theta_dot(u)

        rGx, rGy, vGx, vGy = self.plant_view.com_state(q, u)
        if (
            isinstance(self.reference, NadirAcquisitionReference)
            and not self.reference.is_initialised
        ):
            self.reference.initialise(
                start_time=t,
                theta_initial=theta,
                theta_dot_initial=theta_dot,
                position=(rGx, rGy),
                velocity=(vGx, vGy),
            )
        reference_state = self.reference.evaluate(
            t,
            position=(rGx, rGy),
            velocity=(vGx, vGy),
        )

        error = (
            reference_state.theta - theta + np.pi
        ) % (2 * np.pi) - np.pi
        error_dot = reference_state.theta_dot - theta_dot

        return PlanarAttitudeCommandState(
            theta=float(theta),
            theta_dot=float(theta_dot),
            theta_ref=float(reference_state.theta),
            theta_dot_ref=float(reference_state.theta_dot),
            theta_ddot_ref=float(reference_state.theta_ddot),
            error=float(error),
            error_dot=float(error_dot),
        )
