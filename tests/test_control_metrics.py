from __future__ import annotations

import numpy as np

from multibodysim import attitude_control_metrics, control_metrics_table


def test_attitude_control_metrics_for_simple_slew():
    t = np.linspace(0.0, 10.0, 11)
    theta_initial = 0.0
    theta_target = 1.0

    results = {
        "time": t,
        "q3": np.array([0.0, 0.05, 0.1, 0.3, 0.6, 0.9, 1.05, 1.02, 1.0, 1.0, 1.0]),
        "u3": np.full_like(t, 0.01),
        "tau_PD": np.full_like(t, 0.02),
        "eta1_1": np.linspace(0.0, 1.0e-6, len(t)),
    }

    metrics = attitude_control_metrics(
        results,
        theta_target=theta_target,
        theta_initial=theta_initial,
        Tr=5.0,
        settle_band_deg=3.0,
    )

    assert metrics["rise_time_10_90_s"] == 3.0
    assert np.isclose(metrics["overshoot_deg"], np.rad2deg(0.05))
    assert np.isclose(metrics["overshoot_percent"], 5.0)
    assert np.isclose(metrics["peak_u3_deg_s"], np.rad2deg(0.01))
    assert np.isclose(metrics["peak_tau_PD"], 0.02)
    assert np.isclose(metrics["impulse_abs_tau_PD"], 0.2)
    assert np.isclose(metrics["energy_tau_PD_sq"], 0.004)
    assert np.isclose(metrics["peak_eta1_1"], 1.0e-6)
    assert "post_Tr_rms_eta1_1" in metrics


def test_control_metrics_table_formats_known_labels():
    metrics = {
        "rise_time_10_90_s": 10.0,
        "peak_tau_PD": 0.2,
        "custom_metric": 3.0,
    }

    rows, columns = control_metrics_table(metrics)

    assert columns == ["Metric", "Unit", "Value"]
    assert rows[0] == ("Rise time 10-90%", "s", 10.0)
    assert rows[1] == ("Peak PD torque", "N.m", 0.2)
    assert rows[2] == ("custom_metric", "-", 3.0)
