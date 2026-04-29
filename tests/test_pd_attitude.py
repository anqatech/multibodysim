from __future__ import annotations

import numpy as np
import pytest

from multibodysim.controllers.pd_attitude import PlanarAttitudeController


class FakePlantView:
    def theta(self, q):
        return q[0]

    def theta_dot(self, u):
        return u[0]

    def J_theta(self, Md):
        return float(Md[0, 0])

    def com_state(self, q, u):
        return 1.0, 0.0, 0.0, 1.0


def test_smooth_step_boundary_values():
    assert PlanarAttitudeController.smooth_step_5th_order(-1.0, 10.0) == 0.0
    assert PlanarAttitudeController.smooth_step_5th_order(0.0, 10.0) == 0.0
    assert PlanarAttitudeController.smooth_step_5th_order(10.0, 10.0) == 1.0
    assert PlanarAttitudeController.smooth_step_5th_order(11.0, 10.0) == 1.0

    assert PlanarAttitudeController.derivative_smooth_step_5th_order(0.0, 10.0) == 0.0
    assert PlanarAttitudeController.derivative_smooth_step_5th_order(10.0, 10.0) == 0.0
    assert PlanarAttitudeController.derivative_smooth_step_5th_order(5.0, 10.0) > 0.0


def test_attitude_pd_returns_zero_torque_at_final_target():
    controller = PlanarAttitudeController(FakePlantView())
    controller.configure_attitude_pd(
        theta_target=1.0,
        theta_dot_target=0.0,
        Kp=2.0,
        Kd=3.0,
        Tr=10.0,
    )

    first = controller.compute(t=0.0, q=np.array([0.0]), u=np.array([0.0]))
    assert first.tau_total == 0.0

    final = controller.compute(t=10.0, q=np.array([1.0]), u=np.array([0.0]))
    assert np.isclose(final.tau_ff, 0.0)
    assert np.isclose(final.tau_fb, 0.0)


def test_attitude_pd_returns_expected_transient_torque():
    controller = PlanarAttitudeController(FakePlantView())
    controller.configure_attitude_pd(
        theta_target=1.0,
        theta_dot_target=0.0,
        Kp=2.0,
        Kd=3.0,
        Tr=10.0,
    )

    controller.compute(t=0.0, q=np.array([0.0]), u=np.array([0.0]))
    output = controller.compute(t=5.0, q=np.array([0.25]), u=np.array([0.1]))

    theta_ref = 0.5
    theta_dot_ref = 1.0 * PlanarAttitudeController.derivative_smooth_step_5th_order(5.0, 10.0)
    expected = 2.0 * (theta_ref - 0.25) + 3.0 * (theta_dot_ref - 0.1)

    assert np.isclose(output.tau_ff, 0.0)
    assert np.isclose(output.tau_fb, expected)


def test_input_shaping_requires_modal_parameters():
    controller = PlanarAttitudeController(FakePlantView())

    with pytest.raises(ValueError, match="Input shaping requested"):
        controller.configure_attitude_pd(
            theta_target=1.0,
            theta_dot_target=0.0,
            Kp=1.0,
            Kd=1.0,
            Tr=10.0,
            use_input_shaping=True,
        )


def test_nadir_pd_requires_mass_matrix():
    controller = PlanarAttitudeController(FakePlantView())
    controller.configure_nadir_pd(Kp=1.0, Kd=1.0)

    with pytest.raises(ValueError, match="Nadir PD requires Md"):
        controller.compute(t=0.0, q=np.array([0.0]), u=np.array([0.0]))


def test_nadir_pd_returns_finite_feedforward_and_feedback():
    controller = PlanarAttitudeController(FakePlantView())
    controller.configure_nadir_pd(Kp=2.0, Kd=3.0)

    output = controller.compute(
        t=0.0,
        q=np.array([0.0]),
        u=np.array([0.0]),
        Md=np.array([[5.0]]),
    )

    assert np.isfinite(output.tau_ff)
    assert np.isfinite(output.tau_fb)
