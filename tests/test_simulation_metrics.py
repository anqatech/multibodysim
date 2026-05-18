from __future__ import annotations

import numpy as np
import sympy as sm

from multibodysim.analysis import (
    compute_energy_diagnostics,
    simulation_diagnostics,
    simulation_diagnostics_table,
)


class FakeDynamics:
    def __init__(self):
        self.q = sm.Matrix([sm.Symbol("q1")])
        self.u = sm.Matrix([sm.Symbol("u1")])
        self.state_dimension = 1
        self.planet_mu = sm.Symbol("planet_mu")
        self.parameter_symbols = {"planet_mu": self.planet_mu}
        self.total_mass = sm.S(2)
        self.r_G_norm = self.q[0]
        self.V_strain = self.q[0] ** 2
        self.V_gg = sm.S.Zero

    def eval_differentials(self, q, u, parameter_values, torques):
        return np.array([[4.0]]), np.array([0.0])


class FakeSimulator:
    def __init__(self):
        self.dynamics = FakeDynamics()
        self.parameter_values = np.array([10.0])
        self.initial_torque_values = np.array([0.0])


def test_compute_energy_diagnostics_returns_energy_components_and_drift():
    results = {
        "time": np.array([0.0, 1.0, 2.0]),
        "states": np.array(
            [
                [2.0, 3.0],
                [3.0, 4.0],
                [4.0, 5.0],
            ],
        ),
    }

    energy = compute_energy_diagnostics(
        FakeSimulator(),
        results,
        sample_every=2,
    )

    assert energy["time"].to_list() == [0.0, 2.0]
    np.testing.assert_allclose(energy["kinetic"], [18.0, 50.0])
    np.testing.assert_allclose(energy["kepler_potential"], [-10.0, -5.0])
    np.testing.assert_allclose(energy["strain_potential"], [4.0, 16.0])
    np.testing.assert_allclose(energy["gravity_gradient_potential"], [0.0, 0.0])
    np.testing.assert_allclose(energy["total_energy"], [12.0, 61.0])
    np.testing.assert_allclose(energy["total_energy_drift"], [0.0, 49.0])
    np.testing.assert_allclose(
        energy["total_energy_relative_drift"],
        [0.0, 49.0 / 12.0],
    )


def test_compute_energy_diagnostics_rejects_invalid_sample_every():
    with np.testing.assert_raises(ValueError):
        compute_energy_diagnostics(FakeSimulator(), {}, sample_every=0)


def test_simulation_diagnostics_computes_attitude_and_flexible_metrics():
    results = {
        "success": True,
        "nfev": 10,
        "njev": 2,
        "nlu": 3,
        "q_central_angle": np.deg2rad(np.array([90.0, 91.0, 89.0])),
        "u_central_angle": np.deg2rad(np.array([0.0, 0.5, -1.0])),
        "eta1_1": np.array([0.0, 0.2, -0.1]),
        "zeta1_1": np.array([0.0, -0.3, 0.1]),
    }

    metrics = simulation_diagnostics(results)

    assert metrics["success"] is True
    assert metrics["nfev"] == 10
    assert np.isclose(metrics["central_angle_final_deg"], 89.0)
    assert np.isclose(metrics["central_angle_drift_deg"], -1.0)
    assert np.isclose(metrics["central_angle_peak_to_peak_deg"], 2.0)
    assert np.isclose(metrics["central_angle_speed_peak_abs_deg_s"], 1.0)
    assert np.isclose(metrics["eta1_1_peak_abs"], 0.2)
    assert np.isclose(metrics["eta1_1_rms"], np.sqrt((0.0**2 + 0.2**2 + 0.1**2) / 3.0))
    assert np.isclose(metrics["eta1_1_final_abs"], 0.1)
    assert np.isclose(metrics["zeta1_1_peak_abs"], 0.3)


def test_simulation_diagnostics_table_returns_display_rows():
    metrics = {
        "success": True,
        "nfev": 10,
        "central_angle_final_deg": 90.0,
        "eta1_1_peak_abs": 0.1,
    }

    rows, columns = simulation_diagnostics_table(metrics)

    assert columns == ["Metric", "Unit", "Value"]
    assert rows == [
        ("Solver success", "-", True),
        ("Function evaluations", "-", 10),
        ("Final attitude", "deg", 90.0),
        ("eta1_1_peak_abs", "-", 0.1),
    ]
