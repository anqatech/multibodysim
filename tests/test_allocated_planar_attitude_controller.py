from __future__ import annotations

import numpy as np

from multibodysim.allocation import AllocatedPlanarAttitudeController


class OneDofThreeBusDynamics:
    state_dimension = 1
    rigid_body_names = ["bus_1", "bus_2", "bus_3"]

    def __init__(self):
        self._eval_differentials = self.evaluate_differentials
        self._eval_control_force_matrix = self.evaluate_control_force_matrix

    @staticmethod
    def evaluate_differentials(q, u, torques):
        torques = np.asarray(torques, dtype=float)
        mass_matrix = np.array([[2.0]])
        control_force_matrix = (
            OneDofThreeBusDynamics.evaluate_control_force_matrix(q, u)
        )
        forcing = np.array(
            [[torques @ control_force_matrix.reshape(-1)]]
        )
        return mass_matrix, forcing

    @staticmethod
    def evaluate_control_force_matrix(q, u):
        del q, u
        return np.array([[1.0, 2.0, -1.0]])


class PlantView:
    i_theta_u = 0

    @staticmethod
    def theta(q):
        return float(q[0])

    @staticmethod
    def theta_dot(u):
        return float(u[0])

    @staticmethod
    def com_state(q, u):
        del q, u
        return 2.0, 1.0, -0.1, 0.3


class RecordingGravityGradientFeedforward:
    def __init__(self, cancellation_torque):
        self.cancellation_torque = float(cancellation_torque)
        self.calls = []

    def evaluate(self, *, centre_of_mass_position, central_attitude):
        self.calls.append((centre_of_mass_position, central_attitude))
        return type(
            "GravityGradientFeedforwardResult",
            (),
            {"cancellation_torque": self.cancellation_torque},
        )()


def make_controller(
    *,
    lower_bounds=None,
    upper_bounds=None,
    gravity_gradient_feedforward=None,
    gravity_gradient_acceleration_gain=0.0,
):
    controller = AllocatedPlanarAttitudeController(
        OneDofThreeBusDynamics(),
        PlantView(),
        effort_penalty_matrix=np.eye(3),
        lower_bounds=(
            np.full(3, -10.0)
            if lower_bounds is None
            else lower_bounds
        ),
        upper_bounds=(
            np.full(3, 10.0)
            if upper_bounds is None
            else upper_bounds
        ),
    )
    controller.configure_inertial_pd(
        theta_target=1.0,
        Kp_acceleration=0.2,
        Kd_acceleration=0.0,
        manoeuvre_duration=10.0,
        gravity_gradient_feedforward=gravity_gradient_feedforward,
        gravity_gradient_acceleration_gain=gravity_gradient_acceleration_gain,
    )
    controller.compute(
        0.0,
        np.array([0.0]),
        np.array([0.0]),
        Md=np.array([[2.0]]),
    )
    return controller


def test_allocated_controller_returns_direct_bus_torques_for_feasible_command():
    controller = make_controller()

    output = controller.compute(
        10.0,
        np.array([0.0]),
        np.array([0.0]),
        Md=np.array([[2.0]]),
    )
    diagnostics = controller.last_diagnostics

    expected_effectiveness = np.array([0.5, 1.0, -0.5])
    expected_torques = (
        0.2
        / float(expected_effectiveness @ expected_effectiveness)
        * expected_effectiveness
    )
    np.testing.assert_allclose(output.bus_torques, expected_torques)
    assert output.tau_total == 0.0
    assert diagnostics is not None
    assert diagnostics.clipped is False
    assert np.isclose(diagnostics.requested_acceleration, 0.2)
    assert np.isclose(diagnostics.saturated_acceleration, 0.2)
    assert np.isclose(diagnostics.allocation.achieved_acceleration, 0.2)


def test_allocated_controller_clips_infeasible_acceleration_command():
    controller = make_controller(
        lower_bounds=np.full(3, -0.05),
        upper_bounds=np.full(3, 0.05),
    )
    controller.Kp_acceleration = 2.0

    output = controller.compute(
        10.0,
        np.array([0.0]),
        np.array([0.0]),
        Md=np.array([[2.0]]),
    )
    diagnostics = controller.last_diagnostics

    assert diagnostics is not None
    assert diagnostics.clipped is True
    np.testing.assert_allclose(
        diagnostics.feasible_acceleration_interval,
        (-0.1, 0.1),
    )
    assert np.isclose(diagnostics.requested_acceleration, 2.0)
    assert np.isclose(diagnostics.saturated_acceleration, 0.1)
    assert np.isclose(diagnostics.allocation.achieved_acceleration, 0.1)
    assert np.all(output.bus_torques >= -0.05)
    assert np.all(output.bus_torques <= 0.05)


def test_allocated_controller_includes_gravity_gradient_acceleration_term():
    feedforward = RecordingGravityGradientFeedforward(
        cancellation_torque=0.4,
    )
    controller = make_controller(
        gravity_gradient_feedforward=feedforward,
        gravity_gradient_acceleration_gain=0.5,
    )
    controller.Kp_acceleration = 0.0

    output = controller.compute(
        10.0,
        np.array([1.0]),
        np.array([0.0]),
        Md=np.array([[2.0]]),
    )
    diagnostics = controller.last_diagnostics

    assert feedforward.calls[-1] == ((2.0, 1.0), 1.0)
    assert diagnostics is not None
    assert np.isclose(diagnostics.gravity_gradient_torque, 0.4)
    assert np.isclose(diagnostics.gravity_gradient_acceleration, 0.2)
    assert np.isclose(diagnostics.requested_acceleration, 0.2)
    assert np.isclose(
        diagnostics.control_effectiveness.effectiveness
        @ output.bus_torques,
        0.2,
    )
