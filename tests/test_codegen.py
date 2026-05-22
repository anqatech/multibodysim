from __future__ import annotations

import copy

import sympy as sm
import pytest

from multibodysim.codegen import autowrap_eval_differentials_cache_key
from multibodysim.multiangle.dynamics import MultiAngleFlexibleDynamics
from multibodysim.multiangle.simulator import MultiAngleFlexibleSimulator


class FakeMultiAngleDynamics:
    def __init__(self):
        self.state_dimension = 2
        self.mass_matrix = sm.eye(2)
        self.forcing = sm.Matrix([sm.Symbol("tau_bus_1"), sm.S.Zero])
        self.enable_gravity_gradient = False
        self.body_names = ["bus_1", "panel_1"]
        self.body_type = {
            "bus_1": "rigid-central",
            "panel_1": "flexible-right",
        }
        self.graph = {
            "bus_1": ["panel_1"],
            "panel_1": ["bus_1"],
        }
        self.flexible_body_names = ["panel_1"]
        self.flexible_bodies = {
            "panel_1": {
                "eta_list": [sm.Symbol("eta1_1")],
            },
        }
        self.flexible_inertia_integration = {
            "method": "gauss-legendre",
            "quadrature_points": 8,
        }
        self.parameter_values = {
            "D": 1.0,
            "L": 3.0,
        }
        self.bus_torque_symbols = {
            "bus_1": sm.Symbol("tau_bus_1"),
        }


def test_autowrap_cache_key_changes_when_gravity_gradient_flag_changes():
    dynamics = FakeMultiAngleDynamics()
    baseline_key = autowrap_eval_differentials_cache_key(dynamics)

    dynamics.enable_gravity_gradient = True

    assert autowrap_eval_differentials_cache_key(dynamics) != baseline_key


def test_autowrap_cache_key_changes_when_parameters_change():
    dynamics = FakeMultiAngleDynamics()
    baseline_key = autowrap_eval_differentials_cache_key(dynamics)

    dynamics.parameter_values = copy.deepcopy(dynamics.parameter_values)
    dynamics.parameter_values["L"] = 4.0

    assert autowrap_eval_differentials_cache_key(dynamics) != baseline_key


def test_autowrap_cache_key_changes_when_topology_changes():
    dynamics = FakeMultiAngleDynamics()
    baseline_key = autowrap_eval_differentials_cache_key(dynamics)

    dynamics.graph = copy.deepcopy(dynamics.graph)
    dynamics.graph["panel_1"] = []

    assert autowrap_eval_differentials_cache_key(dynamics) != baseline_key


def test_autowrap_cache_key_changes_when_mode_counts_change():
    dynamics = FakeMultiAngleDynamics()
    baseline_key = autowrap_eval_differentials_cache_key(dynamics)

    dynamics.flexible_bodies = copy.deepcopy(dynamics.flexible_bodies)
    dynamics.flexible_bodies["panel_1"]["eta_list"].append(sm.Symbol("eta1_2"))

    assert autowrap_eval_differentials_cache_key(dynamics) != baseline_key


def test_dynamics_can_restore_numpy_eval_differentials_backend():
    dynamics = object.__new__(MultiAngleFlexibleDynamics)
    reference = lambda q, u, torques: (sm.eye(1), sm.zeros(1, 1))
    dynamics.eval_differentials_reference = reference
    dynamics.eval_differentials = object()
    dynamics.eval_differentials_backend = "autowrap"
    dynamics.eval_differentials_generated_metadata = {"cache_key": "fake"}
    dynamics.eval_differentials_generated_validation = {"success": True}

    dynamics.set_eval_differentials_backend("numpy")

    assert dynamics.eval_differentials_backend == "numpy"
    assert dynamics.eval_differentials is reference
    assert dynamics.eval_differentials_generated_metadata is None
    assert dynamics.eval_differentials_generated_validation is None


def test_dynamics_rejects_unknown_eval_differentials_backend():
    dynamics = object.__new__(MultiAngleFlexibleDynamics)

    with pytest.raises(ValueError, match="numpy.*autowrap"):
        dynamics.set_eval_differentials_backend("cython")


def test_dynamics_raises_when_autowrap_artifact_is_missing(monkeypatch):
    dynamics = object.__new__(MultiAngleFlexibleDynamics)

    monkeypatch.setattr(
        "multibodysim.codegen.load_validated_autowrap_eval_differentials",
        lambda dynamics: None,
    )

    with pytest.raises(RuntimeError, match="no valid generated evaluator"):
        dynamics.set_eval_differentials_backend("autowrap")


def test_dynamics_uses_loaded_autowrap_eval_differentials_backend(monkeypatch):
    dynamics = object.__new__(MultiAngleFlexibleDynamics)

    def fake_eval_differentials(q, u, torques):
        return sm.eye(1), sm.zeros(1, 1)

    def fake_loader(dynamics):
        return {
            "function": fake_eval_differentials,
            "metadata": {"cache_key": "fake"},
            "validation": {"success": True},
        }

    monkeypatch.setattr(
        "multibodysim.codegen.load_validated_autowrap_eval_differentials",
        fake_loader,
    )

    dynamics.set_eval_differentials_backend("autowrap")

    assert dynamics.eval_differentials_backend == "autowrap"
    assert dynamics.eval_differentials is fake_eval_differentials
    assert dynamics.eval_differentials_generated_metadata == {"cache_key": "fake"}
    assert dynamics.eval_differentials_generated_validation == {"success": True}


def test_simulator_eval_differentials_backend_switch_delegates_to_dynamics():
    simulator = object.__new__(MultiAngleFlexibleSimulator)

    class FakeDynamics:
        def __init__(self):
            self.backend = None

        def set_eval_differentials_backend(self, backend):
            self.backend = backend

    simulator.dynamics = FakeDynamics()

    simulator.set_eval_differentials_backend("numpy")

    assert simulator.dynamics.backend == "numpy"
