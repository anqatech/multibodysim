import numpy as np
from .base import ControlOutput
from ..inputshaping.zv_input_shaping import InputShaper


class PlanarAttitudeController:
    def __init__(self, plant_view):
        self.plant_view = plant_view

        self.mode = None

        self.theta_target = 0.0
        self.theta_dot_target = 0.0
        self.Kp = 0.0
        self.Kd = 0.0

        self.Kp_nadir = 0.0
        self.Kd_nadir = 0.0

        self.use_input_shaping = False
        self.shaper = None
        self.Tr = 0.0

        self.manoeuvre_start_time = None
        self.theta_start = None
        self.theta_final = None
        self.delta_theta = None

    def reset(self):
        self.manoeuvre_start_time = None
        self.theta_start = None
        self.theta_final = None
        self.delta_theta = None

    @staticmethod
    def smooth_step_5th_order(t, Tr):
        if t <= 0.0:
            return 0.0
        if t >= Tr:
            return 1.0
        s = t / Tr
        return 10*s**3 - 15*s**4 + 6*s**5

    @staticmethod
    def derivative_smooth_step_5th_order(t, Tr):
        if t <= 0.0 or t >= Tr:
            return 0.0
        s = t / Tr
        return (30*s**2 - 60*s**3 + 30*s**4) / Tr
    
    def raw_theta_command(self, t):
        tau = t - self.manoeuvre_start_time
        sigma = self.smooth_step_5th_order(tau, self.Tr)
        return self.theta_start + self.delta_theta * sigma

    def raw_theta_dot_command(self, t):
        tau = t - self.manoeuvre_start_time
        sigma_dot = self.derivative_smooth_step_5th_order(tau, self.Tr)
        return self.delta_theta * sigma_dot

    def configure_attitude_pd(self, theta_target, theta_dot_target, Kp, Kd, Tr,
                              use_input_shaping=False, shaper=None,
                              omega=None, zeta=None):
        self.mode = "attitude_pd"

        self.theta_target = float(theta_target)
        self.theta_dot_target = float(theta_dot_target)
        self.Kp = float(Kp)
        self.Kd = float(Kd)

        self.Tr = float(Tr)
        self.use_input_shaping = bool(use_input_shaping)

        if shaper is not None:
            self.shaper = shaper
        elif self.use_input_shaping:
            if omega is None or zeta is None:
                raise ValueError("Input shaping requested but omega or zeta not provided.")
            self.shaper = InputShaper.zvd(omega=float(omega), zeta=float(zeta))
        else:
            self.shaper = None

        self.reset()

    def configure_nadir_pd(self, Kp, Kd):
        self.mode = "nadir_pd"

        self.Kp_nadir = float(Kp)
        self.Kd_nadir = float(Kd)

        self.use_input_shaping = False
        self.shaper = None

        self.reset()

    def compute(self, t, q, u, Md=None):
        if self.mode == "attitude_pd":
            theta = self.plant_view.theta(q)
            theta_dot = self.plant_view.theta_dot(u)

            if self.manoeuvre_start_time is None:
                self.manoeuvre_start_time = t
                self.theta_start = theta
                self.theta_final = self.theta_target
                self.delta_theta = self.theta_final - self.theta_start

            if self.use_input_shaping and self.shaper is not None:
                theta_ref = self.shaper.shape(t, self.raw_theta_command)
                theta_dot_ref = self.shaper.shape(t, self.raw_theta_dot_command)
            else:
                theta_ref = self.raw_theta_command(t)
                theta_dot_ref = self.raw_theta_dot_command(t)

            err = theta_ref - theta
            err_dot = theta_dot_ref - theta_dot

            tau_fb = self.Kp * err + self.Kd * err_dot
            return ControlOutput(tau_ff=0.0, tau_fb=float(tau_fb))

        if Md is None:
            raise ValueError("Nadir PD requires Md.")

        theta = self.plant_view.theta(q)
        theta_dot = self.plant_view.theta_dot(u)

        J = self.plant_view.J_theta(Md)

        rGx, rGy, vGx, vGy = self.plant_view.com_state(q, u)

        # --- LVLH pointing ---
        theta_k = np.arctan2(-rGy, -rGx)
        theta_ref = theta_k - 0.5 * np.pi

        # --- kinematics ---
        h = rGx * vGy - rGy * vGx
        r2 = rGx * rGx + rGy * rGy
        r = np.sqrt(r2)

        theta_dot_ref = h / r2

        rdot = (rGx * vGx + rGy * vGy) / r
        theta_ddot_ref = -2.0 * h * rdot / (r**3)

        # --- errors ---
        err = (theta_ref - theta + np.pi) % (2*np.pi) - np.pi
        err_dot = theta_dot_ref - theta_dot

        tau_ff = J * theta_ddot_ref
        tau_fb = self.Kp_nadir * err + self.Kd_nadir * err_dot

        return ControlOutput(tau_ff=float(tau_ff), tau_fb=float(tau_fb))
