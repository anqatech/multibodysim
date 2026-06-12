import numpy as np
from .base import ControlOutput
from ..inputshaping.zv_input_shaping import InputShaper
from ..references import (
    InertialRestToRestReference,
    NadirAcquisitionReference,
)


class PlanarAttitudeController:
    def __init__(self, plant_view):
        self.plant_view = plant_view

        self.mode = None

        self.theta_target = 0.0
        self.Kp = 0.0
        self.Kd = 0.0

        self.Kp_nadir = 0.0
        self.Kd_nadir = 0.0

        self.use_input_shaping = False
        self.shaper = None
        self.manoeuvre_duration = 0.0
        self.reference_acceleration_gain = 0.0
        self.gravity_gradient_feedforward = None
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

    def configure_inertial_pd(
        self,
        *,
        theta_target,
        Kp,
        Kd,
        manoeuvre_duration,
        reference_acceleration_gain=0.0,
        use_input_shaping=False,
        shaper=None,
        omega=None,
        zeta=None,
        gravity_gradient_feedforward=None,
    ):
        self.mode = "inertial_pd"

        self.theta_target = float(theta_target)
        self.Kp = float(Kp)
        self.Kd = float(Kd)
        self.reference_acceleration_gain = self._validated_acceleration_gain(
            reference_acceleration_gain
        )
        self.gravity_gradient_feedforward = (
            self._validated_gravity_gradient_feedforward(
                gravity_gradient_feedforward
            )
        )

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

    def configure_nadir_pd(
        self,
        *,
        offset=-0.5 * np.pi,
        Kp,
        Kd,
        manoeuvre_duration,
        reference_acceleration_gain=0.0,
        gravity_gradient_feedforward=None,
    ):
        self.mode = "nadir_pd"

        self.Kp_nadir = float(Kp)
        self.Kd_nadir = float(Kd)
        self.reference_acceleration_gain = self._validated_acceleration_gain(
            reference_acceleration_gain
        )
        self.gravity_gradient_feedforward = (
            self._validated_gravity_gradient_feedforward(
                gravity_gradient_feedforward
            )
        )
        self.manoeuvre_duration = float(manoeuvre_duration)
        self.reference = NadirAcquisitionReference(
            duration=self.manoeuvre_duration,
            offset=offset,
        )

        self.use_input_shaping = False
        self.shaper = None

        self.reset()

    def compute(self, t, q, u, Md=None):
        if self.mode == "inertial_pd":
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

            err = theta_ref - theta
            err_dot = theta_dot_ref - theta_dot

            tau_reference_ff = (
                self.reference_acceleration_gain * theta_ddot_ref
            )
            tau_gravity_gradient_ff = (
                self._gravity_gradient_feedforward_torque(q, u, theta)
            )
            tau_fb = self.Kp * err + self.Kd * err_dot
            return ControlOutput(
                tau_fb=float(tau_fb),
                tau_reference_ff=float(tau_reference_ff),
                tau_gravity_gradient_ff=float(
                    tau_gravity_gradient_ff
                ),
            )

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

        # --- errors ---
        err = (
            reference_state.theta - theta + np.pi
        ) % (2 * np.pi) - np.pi
        err_dot = reference_state.theta_dot - theta_dot

        tau_reference_ff = (
            self.reference_acceleration_gain
            * reference_state.theta_ddot
        )
        tau_gravity_gradient_ff = (
            self._gravity_gradient_feedforward_torque(q, u, theta)
        )
        tau_fb = self.Kp_nadir * err + self.Kd_nadir * err_dot

        return ControlOutput(
            tau_fb=float(tau_fb),
            tau_reference_ff=float(tau_reference_ff),
            tau_gravity_gradient_ff=float(tau_gravity_gradient_ff),
        )

    @staticmethod
    def _validated_acceleration_gain(value):
        gain = float(value)
        if not np.isfinite(gain):
            raise ValueError(
                "reference_acceleration_gain must be finite."
            )
        return gain

    @staticmethod
    def _validated_gravity_gradient_feedforward(value):
        if value is not None and not callable(
            getattr(value, "evaluate", None)
        ):
            raise TypeError(
                "gravity_gradient_feedforward must expose an "
                "evaluate() method."
            )
        return value

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
