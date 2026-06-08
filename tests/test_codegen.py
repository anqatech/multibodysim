from __future__ import annotations

import copy

import numpy as np
import pytest
import sympy as sm

from multibodysim.codegen import (
    autowrap_eval_differentials_cache_key,
    autowrap_eval_gravity_gradient_cache_key,
    autowrap_eval_kinematics_cache_key,
    prepare_autowrap_evaluators,
    prepare_autowrap_gravity_gradient_evaluator,
    validate_autowrap_gravity_gradient_evaluator,
    validate_autowrap_evaluators,
)
from multibodysim.codegen.symbolic import (
    symbolic_eval_gravity_gradient_data,
    wrap_flat_autowrap_gravity_gradient_function,
)
from multibodysim.codegen.validation import (
    validate_gravity_gradient_candidate,
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
        self.gravity_gradient_generalised_forces = sm.Matrix(
            [
                sm.Symbol("D") * sm.Symbol("q1"),
                sm.Symbol("L") * sm.Symbol("q2"),
            ]
        )
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
        self.q = sm.Matrix([sm.Symbol("q1"), sm.Symbol("q2")])
        self.u = sm.Matrix([sm.Symbol("u1"), sm.Symbol("u2")])

    def _with_specialised_parameters(self, expression):
        replacements = {
            sm.Symbol(name): sm.Float(value)
            for name, value in self.parameter_values.items()
        }
        return expression.xreplace(replacements)

    def get_initial_conditions(self, verbose=False):
        return np.array([0.1, -0.2, 0.0, 0.0], dtype=float)


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


def test_autowrap_gravity_gradient_cache_key_is_distinct():
    dynamics = FakeMultiAngleDynamics()
    dynamics.enable_gravity_gradient = True

    gravity_gradient_key = autowrap_eval_gravity_gradient_cache_key(dynamics)

    assert gravity_gradient_key != autowrap_eval_differentials_cache_key(dynamics)
    assert gravity_gradient_key != autowrap_eval_kinematics_cache_key(dynamics)


@pytest.mark.parametrize(
    "mutation",
    [
        lambda dynamics: dynamics.parameter_values.__setitem__("L", 4.0),
        lambda dynamics: dynamics.graph.__setitem__("panel_1", []),
        lambda dynamics: dynamics.flexible_bodies["panel_1"]["eta_list"].append(
            sm.Symbol("eta1_2")
        ),
        lambda dynamics: setattr(dynamics, "enable_gravity_gradient", False),
    ],
)
def test_autowrap_gravity_gradient_cache_key_tracks_model_metadata(mutation):
    dynamics = FakeMultiAngleDynamics()
    dynamics.enable_gravity_gradient = True
    baseline_key = autowrap_eval_gravity_gradient_cache_key(dynamics)

    mutation(dynamics)

    assert autowrap_eval_gravity_gradient_cache_key(dynamics) != baseline_key


def test_gravity_gradient_wrapper_preserves_column_vector_shape():
    dynamics = FakeMultiAngleDynamics()
    dynamics.enable_gravity_gradient = True
    data = symbolic_eval_gravity_gradient_data(dynamics)
    function = wrap_flat_autowrap_gravity_gradient_function(
        lambda q1, q2: np.array([q1, 2.0 * q2]),
        data,
    )

    output = function(np.array([0.25, -0.5]))

    assert output.shape == (2, 1)
    np.testing.assert_allclose(output[:, 0], [0.25, -1.0])


def test_prepare_gravity_gradient_uses_loaded_evaluator(monkeypatch):
    dynamics = FakeMultiAngleDynamics()
    dynamics.enable_gravity_gradient = True
    cached_function = lambda q: np.asarray(q, dtype=float).reshape(2, 1)

    monkeypatch.setattr(
        "multibodysim.codegen.preparation.load_autowrap_eval_gravity_gradient",
        lambda dynamics, cache_root=None: {
            "success": True,
            "function": cached_function,
            "metadata": {"cache_key": "gravity-gradient"},
            "artifact_dir": "gravity-gradient-dir",
        },
    )
    monkeypatch.setattr(
        "multibodysim.codegen.preparation.generate_autowrap_eval_gravity_gradient",
        lambda dynamics, cache_root=None: (_ for _ in ()).throw(
            AssertionError("cache hit should not generate")
        ),
    )

    prepared = prepare_autowrap_gravity_gradient_evaluator(dynamics)

    assert prepared["function"] is cached_function
    assert prepared["timing"]["source"] == "cache"
    assert prepared["timing"]["generate_time_s"] == 0.0


def test_prepare_gravity_gradient_generates_when_cache_is_missing(monkeypatch):
    dynamics = FakeMultiAngleDynamics()
    dynamics.enable_gravity_gradient = True
    generated_function = lambda q: np.asarray(q, dtype=float).reshape(2, 1)

    monkeypatch.setattr(
        "multibodysim.codegen.preparation.load_autowrap_eval_gravity_gradient",
        lambda dynamics, cache_root=None: None,
    )
    monkeypatch.setattr(
        "multibodysim.codegen.preparation.generate_autowrap_eval_gravity_gradient",
        lambda dynamics, cache_root=None: {
            "success": True,
            "function": generated_function,
            "metadata": {"cache_key": "generated-gravity-gradient"},
            "artifact_dir": "generated-gravity-gradient-dir",
        },
    )

    prepared = prepare_autowrap_gravity_gradient_evaluator(dynamics)

    assert prepared["function"] is generated_function
    assert prepared["timing"]["source"] == "generated"
    assert prepared["timing"]["generate_time_s"] >= 0.0


def test_prepare_gravity_gradient_rejects_unsuccessful_generation(monkeypatch):
    dynamics = FakeMultiAngleDynamics()
    dynamics.enable_gravity_gradient = True

    monkeypatch.setattr(
        "multibodysim.codegen.preparation.load_autowrap_eval_gravity_gradient",
        lambda dynamics, cache_root=None: None,
    )
    monkeypatch.setattr(
        "multibodysim.codegen.preparation.generate_autowrap_eval_gravity_gradient",
        lambda dynamics, cache_root=None: {
            "success": False,
            "error": "validation failed",
        },
    )

    with pytest.raises(
        RuntimeError,
        match="Autowrap eval_gravity_gradient generation failed",
    ):
        prepare_autowrap_gravity_gradient_evaluator(dynamics)


def test_prepare_gravity_gradient_rejects_gg_off_model():
    dynamics = FakeMultiAngleDynamics()

    with pytest.raises(ValueError, match="enable_gravity_gradient=True"):
        prepare_autowrap_gravity_gradient_evaluator(dynamics)


def test_validate_gravity_gradient_candidate_detects_bad_output():
    dynamics = FakeMultiAngleDynamics()
    dynamics.enable_gravity_gradient = True
    reference = lambda q: np.array([[q[0]], [2.0 * q[1]]])

    valid_report = validate_gravity_gradient_candidate(
        dynamics,
        reference,
        reference=reference,
    )
    invalid_report = validate_gravity_gradient_candidate(
        dynamics,
        lambda q: reference(q) + 1e-3,
        reference=reference,
    )

    assert valid_report["success"]
    assert valid_report["max_absolute_difference"] == 0.0
    assert not invalid_report["success"]
    assert np.isclose(invalid_report["max_absolute_difference"], 1e-3)


def test_validate_autowrap_gravity_gradient_builds_reference_on_demand(
    monkeypatch,
):
    dynamics = FakeMultiAngleDynamics()
    dynamics.enable_gravity_gradient = True
    candidate = lambda q: np.array([[q[0]], [2.0 * q[1]]])

    monkeypatch.setattr(
        "multibodysim.codegen.preparation.prepare_autowrap_gravity_gradient_evaluator",
        lambda dynamics, cache_root=None: {
            "success": True,
            "function": candidate,
            "metadata": {"cache_key": "gravity-gradient"},
            "artifact_dir": "gravity-gradient-dir",
            "timing": {"source": "cache"},
        },
    )
    monkeypatch.setattr(
        "multibodysim.codegen.validation.make_numpy_eval_gravity_gradient_reference",
        lambda dynamics: candidate,
    )

    report = validate_autowrap_gravity_gradient_evaluator(dynamics)

    assert report["success"]
    assert report["artifact_dir"] == "gravity-gradient-dir"
    assert report["validation"]["success"]
    assert report["validation"]["max_absolute_difference"] == 0.0


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
