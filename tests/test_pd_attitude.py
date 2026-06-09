from __future__ import annotations

import numpy as np
import pytest

from multibodysim.controllers.pd_attitude import PlanarAttitudeController
from multibodysim.controllers.plant_view import MultiAnglePlantView
from multibodysim.references import (
    InertialRestToRestReference,
    NadirAcquisitionReference,
)


class FakePlantView:
    def theta(self, q):
        return q[0]

    def theta_dot(self, u):
        return u[0]

    def J_theta(self, Md):
        return float(Md[0, 0])

    def com_state(self, q, u):
        return 1.0, 0.0, 0.0, 1.0


class GeneralOrbitPlantView(FakePlantView):
    def com_state(self, q, u):
        return 2.0, 1.0, -0.2, 0.7


class FakeMultiAngleDynamics:
    central_body = "bus_2"
    q_reference = {"bus_1": None, "bus_2": None, "bus_3": None}
    u_reference = {"bus_1": None, "bus_2": None, "bus_3": None}

    def rG_func(self, q, u):
        return np.array([[1.0], [2.0], [3.0]])

    def vG_func(self, q, u):
        return np.array([[4.0], [5.0], [6.0]])


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
    controller.configure_inertial_pd(
        theta_target=1.0,
        Kp=2.0,
        Kd=3.0,
        manoeuvre_duration=10.0,
    )

    first = controller.compute(t=0.0, q=np.array([0.0]), u=np.array([0.0]))
    assert first.tau_total == 0.0

    final = controller.compute(t=10.0, q=np.array([1.0]), u=np.array([0.0]))
    assert np.isclose(final.tau_ff, 0.0)
    assert np.isclose(final.tau_fb, 0.0)
    assert isinstance(
        controller.reference,
        InertialRestToRestReference,
    )


def test_attitude_pd_returns_expected_transient_torque():
    controller = PlanarAttitudeController(FakePlantView())
    controller.configure_inertial_pd(
        theta_target=1.0,
        Kp=2.0,
        Kd=3.0,
        manoeuvre_duration=10.0,
        reference_acceleration_gain=5.0,
    )

    controller.compute(t=0.0, q=np.array([0.0]), u=np.array([0.0]))
    output = controller.compute(t=5.0, q=np.array([0.25]), u=np.array([0.1]))

    theta_ref = 0.5
    theta_dot_ref = (
        PlanarAttitudeController.derivative_smooth_step_5th_order(
            5.0,
            10.0,
        )
    )
    theta_ddot_ref = controller.reference.theta_ddot(5.0)
    expected = 2.0 * (theta_ref - 0.25) + 3.0 * (theta_dot_ref - 0.1)

    assert np.isclose(output.tau_ff, 5.0 * theta_ddot_ref)
    assert np.isclose(output.tau_fb, expected)


def test_input_shaping_requires_modal_parameters():
    controller = PlanarAttitudeController(FakePlantView())

    with pytest.raises(ValueError, match="Input shaping requested"):
        controller.configure_inertial_pd(
            theta_target=1.0,
            Kp=1.0,
            Kd=1.0,
            manoeuvre_duration=10.0,
            use_input_shaping=True,
        )


def test_reference_acceleration_gain_must_be_finite():
    controller = PlanarAttitudeController(FakePlantView())

    with pytest.raises(ValueError, match="reference_acceleration_gain"):
        controller.configure_nadir_pd(
            Kp=1.0,
            Kd=1.0,
            manoeuvre_duration=10.0,
            reference_acceleration_gain=np.nan,
        )


def test_nadir_pd_returns_finite_feedforward_and_feedback():
    controller = PlanarAttitudeController(FakePlantView())
    controller.configure_nadir_pd(
        Kp=2.0,
        Kd=3.0,
        manoeuvre_duration=10.0,
    )

    output = controller.compute(
        t=0.0,
        q=np.array([0.0]),
        u=np.array([0.0]),
        Md=np.array([[5.0]]),
    )

    assert np.isfinite(output.tau_ff)
    assert np.isfinite(output.tau_fb)
    assert isinstance(controller.reference, NadirAcquisitionReference)


def test_nadir_pd_accepts_reference_offset():
    controller = PlanarAttitudeController(FakePlantView())
    controller.configure_nadir_pd(
        offset=0.0,
        Kp=2.0,
        Kd=0.0,
        manoeuvre_duration=10.0,
    )

    controller.compute(
        t=0.0,
        q=np.array([np.pi]),
        u=np.array([1.0]),
        Md=np.array([[5.0]]),
    )

    assert controller.reference.offset == 0.0


def test_nadir_pd_matches_direct_nadir_equations_after_acquisition():
    controller = PlanarAttitudeController(GeneralOrbitPlantView())
    acceleration_gain = 7.0
    controller.configure_nadir_pd(
        Kp=2.0,
        Kd=3.0,
        manoeuvre_duration=4.0,
        reference_acceleration_gain=acceleration_gain,
    )

    theta = 0.3
    theta_dot = 0.1
    controller.compute(
        t=0.0,
        q=np.array([theta]),
        u=np.array([theta_dot]),
    )
    output = controller.compute(
        t=4.0,
        q=np.array([theta]),
        u=np.array([theta_dot]),
    )

    r_x, r_y, v_x, v_y = 2.0, 1.0, -0.2, 0.7
    theta_reference = np.arctan2(-r_y, -r_x) - 0.5 * np.pi
    angular_momentum = r_x * v_y - r_y * v_x
    radius_squared = r_x**2 + r_y**2
    radius = np.sqrt(radius_squared)
    radial_speed = (r_x * v_x + r_y * v_y) / radius
    theta_dot_reference = angular_momentum / radius_squared
    theta_ddot_reference = (
        -2.0 * angular_momentum * radial_speed / radius**3
    )
    error = (
        theta_reference - theta + np.pi
    ) % (2.0 * np.pi) - np.pi
    error_dot = theta_dot_reference - theta_dot

    assert np.isclose(
        output.tau_ff,
        acceleration_gain * theta_ddot_reference,
    )
    assert np.isclose(output.tau_fb, 2.0 * error + 3.0 * error_dot)


def test_nadir_pd_acquisition_starts_from_current_attitude_and_rate():
    controller = PlanarAttitudeController(FakePlantView())
    controller.configure_nadir_pd(
        Kp=2.0,
        Kd=3.0,
        manoeuvre_duration=10.0,
    )

    output = controller.compute(
        t=0.0,
        q=np.array([0.25]),
        u=np.array([0.1]),
        Md=np.array([[5.0]]),
    )

    assert isinstance(controller.reference, NadirAcquisitionReference)
    assert np.isclose(output.tau_ff, 0.0)
    assert np.isclose(output.tau_fb, 0.0)


def test_nadir_pd_acquisition_becomes_direct_nadir_tracking():
    controller = PlanarAttitudeController(FakePlantView())
    controller.configure_nadir_pd(
        Kp=2.0,
        Kd=3.0,
        manoeuvre_duration=10.0,
    )
    controller.compute(
        t=0.0,
        q=np.array([0.25]),
        u=np.array([0.1]),
        Md=np.array([[5.0]]),
    )

    output = controller.compute(
        t=10.0,
        q=np.array([0.5 * np.pi]),
        u=np.array([1.0]),
        Md=np.array([[5.0]]),
    )

    assert np.isclose(output.tau_ff, 0.0)
    assert np.isclose(output.tau_fb, 0.0)


def test_multiangle_plant_view_com_state_accepts_column_vectors():
    plant_view = MultiAnglePlantView(FakeMultiAngleDynamics())

    assert plant_view.com_state(q=np.zeros(3), u=np.zeros(3)) == (
        1.0,
        2.0,
        4.0,
        5.0,
    )
