from __future__ import annotations

import numpy as np

from multibodysim import simulation_diagnostics, simulation_diagnostics_table


def test_simulation_diagnostics_computes_attitude_and_flexible_metrics():
    results = {
        "success": True,
        "nfev": 10,
        "njev": 2,
        "nlu": 3,
        "q3": np.deg2rad(np.array([90.0, 91.0, 89.0])),
        "u3": np.deg2rad(np.array([0.0, 0.5, -1.0])),
        "eta1_1": np.array([0.0, 0.2, -0.1]),
        "zeta1_1": np.array([0.0, -0.3, 0.1]),
    }

    metrics = simulation_diagnostics(results)

    assert metrics["success"] is True
    assert metrics["nfev"] == 10
    assert np.isclose(metrics["q3_final_deg"], 89.0)
    assert np.isclose(metrics["q3_drift_deg"], -1.0)
    assert np.isclose(metrics["q3_peak_to_peak_deg"], 2.0)
    assert np.isclose(metrics["u3_peak_abs_deg_s"], 1.0)
    assert np.isclose(metrics["eta1_1_peak_abs"], 0.2)
    assert np.isclose(metrics["eta1_1_rms"], np.sqrt((0.0**2 + 0.2**2 + 0.1**2) / 3.0))
    assert np.isclose(metrics["eta1_1_final_abs"], 0.1)
    assert np.isclose(metrics["zeta1_1_peak_abs"], 0.3)


def test_simulation_diagnostics_table_returns_display_rows():
    metrics = {
        "success": True,
        "nfev": 10,
        "q3_final_deg": 90.0,
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
