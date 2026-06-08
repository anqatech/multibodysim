from __future__ import annotations

import numpy as np
import pytest

from multibodysim.analysis import gravity_gradient_control_diagnostic


class FakeDynamics:
    def __init__(self, *, enable_gravity_gradient=True):
        self.enable_gravity_gradient = enable_gravity_gradient
        self.state_dimension = 2
        self.rigid_body_names = ["bus_1", "bus_2"]
        self._eval_differentials = self.eval_differentials

    @staticmethod
    def eval_differentials(q, u, torques):
        mass_matrix = np.array(
            [
                [2.0, 0.5],
                [0.5, 1.5],
            ]
        )
        base_forcing = np.array(
            [
                [10.0 + q[0] + u[0]],
                [-4.0 + q[1] - u[1]],
            ]
        )
        torque_map = np.array(
            [
                [2.0, 0.0],
                [0.0, 4.0],
            ]
        )
        forcing = base_forcing + torque_map @ np.asarray(
            torques,
            dtype=float,
        ).reshape(2, 1)
        return mass_matrix, forcing


class FakePlantView:
    i_theta_u = 0


class FakeSimulator:
    def __init__(self, *, enable_gravity_gradient=True):
        self.dynamics = FakeDynamics(
            enable_gravity_gradient=enable_gravity_gradient,
        )
        self.plant_view = FakePlantView()
        self.torque_weights = np.array([0.25, 0.75])


def prepared_evaluator():
    return {
        "success": True,
        "function": lambda q: np.array(
            [
                [4.0 + q[0]],
                [1.0 - q[1]],
            ]
        ),
        "metadata": {"cache_key": "gravity-gradient"},
        "timing": {"source": "cache"},
        "artifact_dir": "gravity-gradient-dir",
    }


def test_gravity_gradient_control_diagnostic_reduces_full_system():
    simulator = FakeSimulator()
    state = np.array([0.2, -0.1, 0.3, -0.4])

    diagnostic = gravity_gradient_control_diagnostic(
        simulator,
        state,
        prepared_evaluator=prepared_evaluator(),
    )

    mass_matrix = np.array([[2.0, 0.5], [0.5, 1.5]])
    gravity_gradient_forces = np.array([[4.2], [1.1]])
    control_direction = np.array([[0.5], [3.0]])
    gravity_gradient_acceleration = np.linalg.solve(
        mass_matrix,
        gravity_gradient_forces,
    )
    unit_control_acceleration = np.linalg.solve(
        mass_matrix,
        control_direction,
    )
    control_effectiveness = unit_control_acceleration[0, 0]
    equivalent_torque = (
        gravity_gradient_acceleration[0, 0] / control_effectiveness
    )

    np.testing.assert_allclose(diagnostic["mass_matrix"], mass_matrix)
    np.testing.assert_allclose(
        diagnostic["gravity_gradient_generalised_forces"],
        gravity_gradient_forces,
    )
    np.testing.assert_allclose(
        diagnostic["control_generalised_force_direction"],
        control_direction,
    )
    np.testing.assert_allclose(
        diagnostic["gravity_gradient_acceleration"],
        gravity_gradient_acceleration,
    )
    np.testing.assert_allclose(
        diagnostic["unit_control_acceleration"],
        unit_control_acceleration,
    )
    assert np.isclose(
        diagnostic["central_gravity_gradient_acceleration"],
        gravity_gradient_acceleration[0, 0],
    )
    assert np.isclose(
        diagnostic["control_effectiveness"],
        control_effectiveness,
    )
    assert np.isclose(
        diagnostic["effective_attitude_inertia"],
        1.0 / control_effectiveness,
    )
    assert np.isclose(
        diagnostic["equivalent_gravity_gradient_torque"],
        equivalent_torque,
    )
    assert np.isclose(diagnostic["cancellation_torque"], -equivalent_torque)
    assert np.isclose(
        diagnostic["central_cancellation_residual_acceleration"],
        0.0,
        atol=1e-15,
    )
    assert diagnostic["evaluator_metadata"]["cache_key"] == "gravity-gradient"


def test_gravity_gradient_control_diagnostic_prepares_evaluator_on_demand(
    monkeypatch,
):
    simulator = FakeSimulator()
    prepared = prepared_evaluator()
    calls = []

    def prepare(dynamics):
        calls.append(dynamics)
        return prepared

    monkeypatch.setattr(
        "multibodysim.analysis.gravity_gradient_control."
        "prepare_autowrap_gravity_gradient_evaluator",
        prepare,
    )

    diagnostic = gravity_gradient_control_diagnostic(
        simulator,
        np.zeros(4),
    )

    assert calls == [simulator.dynamics]
    assert diagnostic["evaluator_timing"]["source"] == "cache"


def test_gravity_gradient_control_diagnostic_accepts_weight_override():
    simulator = FakeSimulator()

    diagnostic = gravity_gradient_control_diagnostic(
        simulator,
        np.zeros(4),
        prepared_evaluator=prepared_evaluator(),
        torque_weights=np.array([1.0, 0.0]),
    )

    np.testing.assert_allclose(
        diagnostic["control_generalised_force_direction"],
        [[2.0], [0.0]],
    )
    np.testing.assert_allclose(diagnostic["torque_weights"], [1.0, 0.0])


def test_gravity_gradient_control_diagnostic_rejects_gg_off():
    simulator = FakeSimulator(enable_gravity_gradient=False)

    with pytest.raises(ValueError, match="enable_gravity_gradient=True"):
        gravity_gradient_control_diagnostic(
            simulator,
            np.zeros(4),
            prepared_evaluator=prepared_evaluator(),
        )


def test_gravity_gradient_control_diagnostic_rejects_invalid_state_size():
    with pytest.raises(ValueError, match="state must contain 4 values"):
        gravity_gradient_control_diagnostic(
            FakeSimulator(),
            np.zeros(3),
            prepared_evaluator=prepared_evaluator(),
        )


def test_gravity_gradient_control_diagnostic_rejects_zero_control_direction():
    with pytest.raises(ValueError, match="non-zero control direction"):
        gravity_gradient_control_diagnostic(
            FakeSimulator(),
            np.zeros(4),
            prepared_evaluator=prepared_evaluator(),
            torque_weights=np.zeros(2),
        )


def test_gravity_gradient_control_diagnostic_rejects_zero_effectiveness():
    simulator = FakeSimulator()

    with pytest.raises(ValueError, match="zero central-attitude"):
        gravity_gradient_control_diagnostic(
            simulator,
            np.zeros(4),
            prepared_evaluator=prepared_evaluator(),
            torque_weights=np.array([2.0, 3.0]),
        )
