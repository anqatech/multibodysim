import numpy as np

from .base import ControlOutput
from .planar_attitude_reference import PlanarAttitudeReferenceTracker


class PlanarAttitudeController:
    def __init__(self, plant_view):
        self.plant_view = plant_view
        self.reference_tracker = PlanarAttitudeReferenceTracker(plant_view)

        self.theta_target = 0.0
        self.Kp = 0.0
        self.Kd = 0.0

        self.Kp_nadir = 0.0
        self.Kd_nadir = 0.0

        self.manoeuvre_duration = 0.0
        self.reference_acceleration_gain = 0.0
        self.gravity_gradient_feedforward = None

    @property
    def mode(self):
        return self.reference_tracker.mode

    @property
    def reference(self):
        return self.reference_tracker.reference

    @property
    def use_input_shaping(self):
        return self.reference_tracker.use_input_shaping

    @property
    def shaper(self):
        return self.reference_tracker.shaper

    @property
    def manoeuvre_start_time(self):
        return self.reference_tracker.manoeuvre_start_time

    @property
    def theta_start(self):
        return self.reference_tracker.theta_start

    @property
    def theta_final(self):
        return self.reference_tracker.theta_final

    @property
    def delta_theta(self):
        return self.reference_tracker.delta_theta

    def reset(self):
        self.reference_tracker.reset()

    @staticmethod
    def smooth_step_5th_order(t, Tr):
        return PlanarAttitudeReferenceTracker.smooth_step_5th_order(t, Tr)

    @staticmethod
    def derivative_smooth_step_5th_order(t, Tr):
        return PlanarAttitudeReferenceTracker.derivative_smooth_step_5th_order(
            t,
            Tr,
        )

    def raw_theta_command(self, t):
        return self.reference_tracker.raw_theta_command(t)

    def raw_theta_dot_command(self, t):
        return self.reference_tracker.raw_theta_dot_command(t)

    def raw_theta_ddot_command(self, t):
        return self.reference_tracker.raw_theta_ddot_command(t)

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
        self.reference_tracker.configure_inertial(
            theta_target=self.theta_target,
            manoeuvre_duration=self.manoeuvre_duration,
            use_input_shaping=use_input_shaping,
            shaper=shaper,
            omega=omega,
            zeta=zeta,
        )

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
        self.reference_tracker.configure_nadir(
            manoeuvre_duration=self.manoeuvre_duration,
            offset=offset,
        )

    def compute(self, t, q, u, Md=None):
        command_state = self.reference_tracker.evaluate(t, q, u)
        if self.mode == "inertial_pd":
            tau_reference_ff = (
                self.reference_acceleration_gain
                * command_state.theta_ddot_ref
            )
            tau_gravity_gradient_ff = (
                self._gravity_gradient_feedforward_torque(
                    q,
                    u,
                    command_state.theta,
                )
            )
            tau_fb = (
                self.Kp * command_state.error
                + self.Kd * command_state.error_dot
            )
            return ControlOutput(
                tau_fb=float(tau_fb),
                tau_reference_ff=float(tau_reference_ff),
                tau_gravity_gradient_ff=float(
                    tau_gravity_gradient_ff
                ),
            )

        tau_reference_ff = (
            self.reference_acceleration_gain
            * command_state.theta_ddot_ref
        )
        tau_gravity_gradient_ff = (
            self._gravity_gradient_feedforward_torque(
                q,
                u,
                command_state.theta,
            )
        )
        tau_fb = (
            self.Kp_nadir * command_state.error
            + self.Kd_nadir * command_state.error_dot
        )

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
