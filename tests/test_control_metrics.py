from __future__ import annotations

import numpy as np

from multibodysim.analysis import (
    reference_tracking_metrics,
    reference_tracking_metrics_table,
)


def test_reference_tracking_metrics_for_fixed_reference():
    t = np.linspace(0.0, 4.0, 5)
    theta_reference = np.ones_like(t)
    theta = np.array([0.0, 0.4, 0.8, 1.1, 1.0])
    theta_dot_reference = np.zeros_like(t)
    theta_dot = np.array([0.0, 0.1, 0.2, -0.1, 0.0])

    results = {
        "time": t,
        "q_central_angle": theta,
        "u_central_angle": theta_dot,
        "theta_reference": theta_reference,
        "theta_dot_reference": theta_dot_reference,
        "theta_ddot_reference": np.zeros_like(t),
        "eta1_1": np.linspace(0.0, 1.0e-6, len(t)),
        "zeta1_1": np.array([0.0, -1.0e-5, 0.0, 1.0e-5, 0.0]),
    }

    metrics = reference_tracking_metrics(results, post_window_start=2.0)
    expected_error = theta_reference - theta
    expected_error_dot = theta_dot_reference - theta_dot

    assert np.isclose(
        metrics["attitude_error_final_deg"],
        np.rad2deg(expected_error[-1]),
    )
    assert np.isclose(
        metrics["attitude_error_rms_deg"],
        np.rad2deg(np.sqrt(np.mean(expected_error**2))),
    )
    assert np.isclose(
        metrics["attitude_error_dot_peak_abs_deg_s"],
        np.rad2deg(np.max(np.abs(expected_error_dot))),
    )
    assert np.isclose(
        metrics["central_angle_speed_peak_abs_deg_s"],
        np.rad2deg(0.2),
    )
    assert np.isclose(metrics["eta1_1_peak_abs"], 1.0e-6)
    assert np.isclose(
        metrics["eta1_1_post_window_rms"],
        np.sqrt(np.mean(results["eta1_1"][2:] ** 2)),
    )
    assert np.isclose(metrics["zeta1_1_final_abs"], 0.0)


def test_reference_tracking_metrics_wrap_angle_error():
    results = {
        "time": np.array([0.0]),
        "q_central_angle": np.deg2rad(np.array([179.0])),
        "u_central_angle": np.array([0.0]),
        "theta_reference": np.deg2rad(np.array([-179.0])),
        "theta_dot_reference": np.array([0.0]),
        "theta_ddot_reference": np.array([0.0]),
    }

    metrics = reference_tracking_metrics(results)

    assert np.isclose(metrics["attitude_error_final_deg"], 2.0)
    assert np.isclose(metrics["attitude_error_peak_abs_deg"], 2.0)


def test_reference_tracking_metrics_table_formats_known_labels():
    metrics = {
        "attitude_error_rms_deg": 1.5,
        "central_angle_speed_peak_abs_deg_s": 2.0,
        "custom_metric": 3.0,
    }

    rows, columns = reference_tracking_metrics_table(metrics)

    assert columns == ["Metric", "Unit", "Value"]
    assert rows[0] == ("RMS attitude error", "deg", 1.5)
    assert rows[1] == ("Peak angular velocity", "deg/s", 2.0)
    assert rows[2] == ("custom_metric", "-", 3.0)
