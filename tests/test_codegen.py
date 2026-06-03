from __future__ import annotations

import copy

import numpy as np
import sympy as sm

from multibodysim.codegen import (
    autowrap_eval_differentials_cache_key,
    autowrap_eval_kinematics_cache_key,
    prepare_autowrap_evaluators,
    validate_autowrap_evaluators,
)
from multibodysim.multiangle.dynamics import MultiAngleFlexibleDynamics
from multibodysim.multiangle.simulator import MultiAngleFlexibleSimulator


class FakeMultiAngleDynamics:
    def __init__(self):
        self.state_dimension = 2
        self.Mk = sm.eye(2)
        self.gk = sm.Matrix([sm.Symbol("u1"), sm.Symbol("u2")])
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


def test_autowrap_kinematics_cache_key_is_distinct_from_differentials_key():
    dynamics = FakeMultiAngleDynamics()

    assert autowrap_eval_kinematics_cache_key(dynamics) != (
        autowrap_eval_differentials_cache_key(dynamics)
    )


def test_autowrap_kinematics_cache_key_changes_when_parameters_change():
    dynamics = FakeMultiAngleDynamics()
    baseline_key = autowrap_eval_kinematics_cache_key(dynamics)

    dynamics.parameter_values = copy.deepcopy(dynamics.parameter_values)
    dynamics.parameter_values["L"] = 4.0

    assert autowrap_eval_kinematics_cache_key(dynamics) != baseline_key


def test_prepare_autowrap_uses_loaded_evaluators(monkeypatch):
    dynamics = object.__new__(MultiAngleFlexibleDynamics)

    def fake_eval_kinematics(q, u, torques):
        return sm.eye(1), sm.zeros(1, 1)

    def fake_eval_differentials(q, u, torques):
        return sm.eye(1), sm.zeros(1, 1)

    def fail_generation(dynamics, cache_root=None):
        raise AssertionError("cache hit should not generate")

    monkeypatch.setattr(
        "multibodysim.codegen.preparation.load_autowrap_eval_kinematics",
        lambda dynamics, cache_root=None: {
            "success": True,
            "function": fake_eval_kinematics,
            "metadata": {"cache_key": "kinematics"},
        },
    )
    monkeypatch.setattr(
        "multibodysim.codegen.preparation.load_autowrap_eval_differentials",
        lambda dynamics, cache_root=None: {
            "success": True,
            "function": fake_eval_differentials,
            "metadata": {"cache_key": "differentials"},
        },
    )
    monkeypatch.setattr(
        "multibodysim.codegen.preparation.generate_autowrap_eval_kinematics",
        fail_generation,
    )
    monkeypatch.setattr(
        "multibodysim.codegen.preparation.generate_autowrap_eval_differentials",
        fail_generation,
    )

    metadata = prepare_autowrap_evaluators(dynamics)

    assert dynamics.eval_kinematics_backend == "autowrap"
    assert dynamics.eval_differentials_backend == "autowrap"
    assert dynamics._eval_kinematics is fake_eval_kinematics
    assert dynamics._eval_differentials is fake_eval_differentials
    assert dynamics.eval_kinematics_codegen_timing["source"] == "cache"
    assert dynamics.eval_differentials_codegen_timing["source"] == "cache"
    assert metadata["kinematics"]["metadata"] == {"cache_key": "kinematics"}
    assert metadata["differentials"]["metadata"] == {"cache_key": "differentials"}


def test_prepare_autowrap_generates_when_cache_is_missing(monkeypatch):
    dynamics = object.__new__(MultiAngleFlexibleDynamics)
    generated = []

    def fake_eval_kinematics(q, u, torques):
        return sm.eye(1), sm.zeros(1, 1)

    def fake_eval_differentials(q, u, torques):
        return sm.eye(1), sm.zeros(1, 1)

    def fake_kinematics_generator(dynamics, cache_root=None):
        generated.append("kinematics")
        return {
            "success": True,
            "function": fake_eval_kinematics,
            "metadata": {"cache_key": "generated-kinematics"},
        }

    def fake_differentials_generator(dynamics, cache_root=None):
        generated.append("differentials")
        return {
            "success": True,
            "function": fake_eval_differentials,
            "metadata": {"cache_key": "generated-differentials"},
        }

    monkeypatch.setattr(
        "multibodysim.codegen.preparation.load_autowrap_eval_kinematics",
        lambda dynamics, cache_root=None: None,
    )
    monkeypatch.setattr(
        "multibodysim.codegen.preparation.load_autowrap_eval_differentials",
        lambda dynamics, cache_root=None: None,
    )
    monkeypatch.setattr(
        "multibodysim.codegen.preparation.generate_autowrap_eval_kinematics",
        fake_kinematics_generator,
    )
    monkeypatch.setattr(
        "multibodysim.codegen.preparation.generate_autowrap_eval_differentials",
        fake_differentials_generator,
    )

    prepare_autowrap_evaluators(dynamics)

    assert generated == ["kinematics", "differentials"]
    assert dynamics.eval_kinematics_backend == "autowrap"
    assert dynamics.eval_differentials_backend == "autowrap"
    assert dynamics.eval_kinematics_codegen_timing["source"] == "generated"
    assert dynamics.eval_differentials_codegen_timing["source"] == "generated"
    assert dynamics.eval_kinematics_generated_metadata == {
        "cache_key": "generated-kinematics",
    }
    assert dynamics._eval_differentials is fake_eval_differentials


def test_validate_autowrap_evaluators_builds_numpy_references_on_demand(
    monkeypatch,
):
    dynamics = object.__new__(MultiAngleFlexibleDynamics)
    dynamics.state_dimension = 1
    dynamics.get_torque_values = lambda: np.array([0.0], dtype=float)
    dynamics.get_initial_conditions = (
        lambda verbose=False: np.array([0.1, -0.2], dtype=float)
    )
    dynamics.eval_kinematics_backend = "autowrap"
    dynamics.eval_differentials_backend = "autowrap"
    dynamics._eval_kinematics = lambda q, u, torques: (
        np.eye(1),
        np.array([[-float(u[0])]]),
    )
    dynamics._eval_differentials = lambda q, u, torques: (
        np.eye(1),
        np.array([[float(q[0]) + float(torques[0])]]),
    )
    dynamics.eval_kinematics_generated_metadata = {"cache_key": "kinematics"}
    dynamics.eval_differentials_generated_metadata = {"cache_key": "differentials"}
    dynamics.autowrap_codegen_metadata = {
        "kinematics": {"artifact_dir": "kinematics-dir"},
        "differentials": {"artifact_dir": "differentials-dir"},
    }

    monkeypatch.setattr(
        "multibodysim.codegen.validation.make_numpy_eval_kinematics_reference",
        lambda dynamics: (
            lambda q, u, torques: (np.eye(1), np.array([[-float(u[0])]]))
        ),
    )
    monkeypatch.setattr(
        "multibodysim.codegen.validation.make_numpy_eval_differentials_reference",
        lambda dynamics: (
            lambda q, u, torques: (
                np.eye(1),
                np.array([[float(q[0]) + float(torques[0])]]),
            )
        ),
    )

    report = validate_autowrap_evaluators(dynamics)

    assert report["success"]
    assert report["kinematics"]["artifact_dir"] == "kinematics-dir"
    assert report["differentials"]["artifact_dir"] == "differentials-dir"
    assert report["kinematics"]["validation"]["success"]
    assert report["differentials"]["validation"]["success"]


def test_multiangle_backend_switching_api_is_removed():
    assert not hasattr(
        MultiAngleFlexibleDynamics,
        "set_eval_kinematics_backend",
    )
    assert not hasattr(
        MultiAngleFlexibleDynamics,
        "set_eval_differentials_backend",
    )
    assert not hasattr(
        MultiAngleFlexibleSimulator,
        "set_eval_kinematics_backend",
    )
    assert not hasattr(
        MultiAngleFlexibleSimulator,
        "set_eval_differentials_backend",
    )
