from __future__ import annotations

import numpy as np
import pytest

from multibodysim.analysis import gravity_gradient_control_diagnostic
from multibodysim.controllers.gravity_gradient import (
    GravityGradientCompensationResult,
    GravityGradientCompensator,
)


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
                [2.0 + 0.1 * q[0], 0.5],
                [0.5, 1.5 + 0.1 * q[1]],
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
                [2.0 + 0.2 * q[0], 0.0],
                [0.0, 4.0 - 0.1 * q[1]],
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


@pytest.mark.parametrize(
    ("q", "u"),
    [
        (np.array([0.0, 0.0]), np.array([0.0, 0.0])),
        (np.array([0.2, -0.1]), np.array([0.3, -0.4])),
        (np.array([-0.3, 0.4]), np.array([-0.2, 0.5])),
    ],
)
def test_compensator_matches_independent_analysis_diagnostic(q, u):
    simulator = FakeSimulator()
    prepared = prepared_evaluator()
    compensator = GravityGradientCompensator(
        simulator.dynamics,
        simulator.plant_view,
        simulator.torque_weights,
        prepared_evaluator=prepared,
    )

    result = compensator.evaluate(q, u)
    diagnostic = gravity_gradient_control_diagnostic(
        simulator,
        np.hstack((q, u)),
        prepared_evaluator=prepared,
    )

    assert isinstance(result, GravityGradientCompensationResult)
    np.testing.assert_allclose(
        result.gravity_gradient_generalised_forces,
        diagnostic["gravity_gradient_generalised_forces"],
    )
    np.testing.assert_allclose(
        result.control_generalised_force_direction,
        diagnostic["control_generalised_force_direction"],
    )
    np.testing.assert_allclose(
        result.gravity_gradient_acceleration,
        diagnostic["gravity_gradient_acceleration"],
    )
    np.testing.assert_allclose(
        result.unit_control_acceleration,
        diagnostic["unit_control_acceleration"],
    )
    assert np.isclose(
        result.control_effectiveness,
        diagnostic["control_effectiveness"],
    )
    assert np.isclose(
        result.effective_attitude_inertia,
        diagnostic["effective_attitude_inertia"],
    )
    assert np.isclose(
        result.equivalent_gravity_gradient_torque,
        diagnostic["equivalent_gravity_gradient_torque"],
    )
    assert np.isclose(
        result.cancellation_torque,
        diagnostic["cancellation_torque"],
    )
    assert np.isclose(
        result.central_cancellation_residual_acceleration,
        0.0,
        atol=1e-15,
    )


def test_compensator_prepares_evaluator_once(monkeypatch):
    simulator = FakeSimulator()
    prepared = prepared_evaluator()
    calls = []

    def prepare(dynamics, *, cache_root=None):
        calls.append((dynamics, cache_root))
        return prepared

    monkeypatch.setattr(
        "multibodysim.controllers.gravity_gradient."
        "prepare_autowrap_gravity_gradient_evaluator",
        prepare,
    )

    compensator = GravityGradientCompensator(
        simulator.dynamics,
        simulator.plant_view,
        simulator.torque_weights,
        cache_root="cache-root",
    )
    compensator.evaluate(np.zeros(2), np.zeros(2))
    compensator.evaluate(np.ones(2), np.ones(2))

    assert calls == [(simulator.dynamics, "cache-root")]
    assert compensator.evaluator_metadata["cache_key"] == "gravity-gradient"
    assert compensator.evaluator_timing["source"] == "cache"
    assert compensator.evaluator_artifact_dir == "gravity-gradient-dir"


def test_compensator_copies_torque_weights():
    simulator = FakeSimulator()
    weights = simulator.torque_weights.copy()
    compensator = GravityGradientCompensator(
        simulator.dynamics,
        simulator.plant_view,
        weights,
        prepared_evaluator=prepared_evaluator(),
    )
    weights[:] = 99.0

    result = compensator.evaluate(np.zeros(2), np.zeros(2))

    np.testing.assert_allclose(result.torque_weights, [0.25, 0.75])


def test_compensator_rejects_gg_off_dynamics():
    simulator = FakeSimulator(enable_gravity_gradient=False)

    with pytest.raises(ValueError, match="enable_gravity_gradient=True"):
        GravityGradientCompensator(
            simulator.dynamics,
            simulator.plant_view,
            simulator.torque_weights,
            prepared_evaluator=prepared_evaluator(),
        )


def test_compensator_rejects_invalid_state_size():
    simulator = FakeSimulator()
    compensator = GravityGradientCompensator(
        simulator.dynamics,
        simulator.plant_view,
        simulator.torque_weights,
        prepared_evaluator=prepared_evaluator(),
    )

    with pytest.raises(ValueError, match="q must contain 2 values"):
        compensator.evaluate(np.zeros(1), np.zeros(2))


def test_compensator_rejects_zero_control_effectiveness():
    simulator = FakeSimulator()
    compensator = GravityGradientCompensator(
        simulator.dynamics,
        simulator.plant_view,
        np.array([2.0, 3.0]),
        prepared_evaluator=prepared_evaluator(),
    )

    with pytest.raises(ValueError, match="zero central-attitude"):
        compensator.evaluate(np.zeros(2), np.zeros(2))


def test_control_module_does_not_depend_on_analysis():
    module_path = (
        "src/multibodysim/controllers/gravity_gradient.py"
    )

    with open(module_path, encoding="utf-8") as module_file:
        source = module_file.read()

    assert "multibodysim.analysis" not in source
    assert "..analysis" not in source
