from __future__ import annotations

import numpy as np
import pytest

from multibodysim.analysis import (
    attitude_acquisition_metrics,
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
        "acquisition_overshoot_percent": 4.0,
        "custom_metric": 3.0,
    }

    rows, columns = reference_tracking_metrics_table(metrics)

    assert columns == ["Metric", "Unit", "Value"]
    assert rows[0] == ("RMS attitude error", "deg", 1.5)
    assert rows[1] == ("Peak angular velocity", "deg/s", 2.0)
    assert rows[2] == ("Acquisition overshoot", "%", 4.0)
    assert rows[3] == ("custom_metric", "-", 3.0)


def test_attitude_acquisition_metrics_for_monotone_error_decay():
    results = {
        "time": np.array([0.0, 1.0, 2.0, 3.0, 4.0]),
        "attitude_error": np.deg2rad(np.array([10.0, 8.0, 1.0, 0.4, 0.2])),
    }

    metrics = attitude_acquisition_metrics(
        results,
        settling_band_deg=0.5,
    )

    assert np.isclose(metrics["acquisition_initial_error_deg"], 10.0)
    assert np.isclose(metrics["acquisition_final_error_abs_deg"], 0.2)
    assert np.isclose(metrics["acquisition_error_reduction_percent"], 98.0)
    assert np.isclose(
        metrics["acquisition_time_10_percent_error_reduction_s"],
        1.0,
    )
    assert np.isclose(
        metrics["acquisition_time_90_percent_error_reduction_s"],
        2.0,
    )
    assert np.isclose(metrics["acquisition_decay_time_10_90_s"], 1.0)
    assert np.isnan(metrics["acquisition_zero_crossing_time_s"])
    assert np.isclose(metrics["acquisition_overshoot_deg"], 0.0)
    assert np.isclose(metrics["acquisition_settling_time_s"], 3.0)
    assert np.isclose(metrics["acquisition_settling_band_deg"], 0.5)


def test_attitude_acquisition_metrics_reports_zero_crossing_overshoot():
    results = {
        "time": np.array([0.0, 1.0, 2.0, 3.0]),
        "attitude_error": np.deg2rad(np.array([10.0, 3.0, -2.0, -1.0])),
    }

    metrics = attitude_acquisition_metrics(results)

    assert np.isclose(metrics["acquisition_zero_crossing_time_s"], 2.0)
    assert np.isclose(metrics["acquisition_overshoot_deg"], 2.0)
    assert np.isclose(metrics["acquisition_overshoot_percent"], 20.0)


def test_attitude_acquisition_metrics_uses_wrapped_reference_error():
    results = {
        "time": np.array([0.0, 1.0]),
        "q_central_angle": np.deg2rad(np.array([179.0, -179.5])),
        "theta_reference": np.deg2rad(np.array([-179.0, -179.0])),
    }

    metrics = attitude_acquisition_metrics(results)

    assert np.isclose(metrics["acquisition_initial_error_deg"], 2.0)
    assert np.isclose(metrics["acquisition_final_error_deg"], 0.5)


def test_attitude_acquisition_metrics_rejects_zero_initial_error():
    results = {
        "time": np.array([0.0, 1.0]),
        "attitude_error": np.array([0.0, 0.1]),
    }

    with pytest.raises(ValueError, match="non-zero initial attitude error"):
        attitude_acquisition_metrics(results)
