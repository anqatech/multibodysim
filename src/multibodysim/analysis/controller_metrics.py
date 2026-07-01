from __future__ import annotations

from typing import Any

import numpy as np


def _wrap_to_pi(angle):
    return (angle + np.pi) % (2 * np.pi) - np.pi


def _rms(values) -> float:
    values = np.asarray(values, dtype=float)
    return float(np.sqrt(np.mean(values**2))) if len(values) else np.nan


def _add_signal_metrics(
    metrics: dict[str, Any],
    prefix: str,
    values,
    *,
    suffix: str = "",
) -> None:
    values = np.asarray(values, dtype=float)
    metrics[f"{prefix}_final{suffix}"] = float(values[-1])
    metrics[f"{prefix}_rms{suffix}"] = _rms(values)
    metrics[f"{prefix}_peak_abs{suffix}"] = (
        float(np.max(np.abs(values))) if len(values) else np.nan
    )


def _add_torque_metrics(metrics: dict[str, Any], prefix: str, torque, time) -> None:
    torque = np.asarray(torque, dtype=float)
    time = np.asarray(time, dtype=float)
    metrics[f"{prefix}_peak_abs_Nm"] = float(np.max(np.abs(torque)))
    metrics[f"{prefix}_rms_Nm"] = _rms(torque)
    metrics[f"{prefix}_impulse_abs_Nms"] = float(np.trapezoid(np.abs(torque), time))
    metrics[f"{prefix}_energy_N2m2s"] = float(np.trapezoid(torque**2, time))


def _add_flexible_metrics(
    metrics: dict[str, Any],
    results: dict[str, Any],
    *,
    post_window_start: float | None = None,
) -> None:
    time = np.asarray(results["time"], dtype=float)
    post_mask = None
    if post_window_start is not None:
        post_mask = time >= float(post_window_start)

    for key in sorted(results):
        if not key.startswith(("eta", "zeta")):
            continue

        values = np.asarray(results[key], dtype=float)
        metrics[f"{key}_peak_abs"] = float(np.max(np.abs(values)))
        metrics[f"{key}_rms"] = _rms(values)
        metrics[f"{key}_final_abs"] = float(abs(values[-1]))

        if post_mask is not None:
            metrics[f"{key}_post_window_rms"] = _rms(values[post_mask])


def reference_tracking_metrics(
    results: dict[str, Any],
    *,
    post_window_start: float | None = None,
) -> dict[str, Any]:
    """Compute controller-neutral planar reference-tracking metrics."""
    theta = np.asarray(results["q_central_angle"], dtype=float)
    theta_dot = np.asarray(results["u_central_angle"], dtype=float)
    theta_reference = np.asarray(results["theta_reference"], dtype=float)
    theta_dot_reference = np.asarray(results["theta_dot_reference"], dtype=float)

    attitude_error = _wrap_to_pi(theta_reference - theta)
    attitude_error_dot = theta_dot_reference - theta_dot

    metrics: dict[str, Any] = {}
    _add_signal_metrics(
        metrics,
        "attitude_error",
        np.rad2deg(attitude_error),
        suffix="_deg",
    )
    _add_signal_metrics(
        metrics,
        "attitude_error_dot",
        np.rad2deg(attitude_error_dot),
        suffix="_deg_s",
    )
    _add_signal_metrics(
        metrics,
        "central_angle_speed",
        np.rad2deg(theta_dot),
        suffix="_deg_s",
    )
    _add_flexible_metrics(
        metrics,
        results,
        post_window_start=post_window_start,
    )
    return metrics


def reference_tracking_metrics_table(
    metrics: dict[str, Any],
) -> tuple[list[tuple[str, str, Any]], list[str]]:
    """Convert reference-tracking metrics to display-ready rows."""
    labels = {
        "attitude_error_final_deg": ("Final attitude error", "deg"),
        "attitude_error_rms_deg": ("RMS attitude error", "deg"),
        "attitude_error_peak_abs_deg": ("Peak abs attitude error", "deg"),
        "attitude_error_dot_final_deg_s": ("Final attitude-rate error", "deg/s"),
        "attitude_error_dot_rms_deg_s": ("RMS attitude-rate error", "deg/s"),
        "attitude_error_dot_peak_abs_deg_s": (
            "Peak abs attitude-rate error",
            "deg/s",
        ),
        "central_angle_speed_final_deg_s": ("Final angular velocity", "deg/s"),
        "central_angle_speed_rms_deg_s": ("RMS angular velocity", "deg/s"),
        "central_angle_speed_peak_abs_deg_s": ("Peak angular velocity", "deg/s"),
    }

    rows = []
    for key, value in metrics.items():
        metric, unit = labels.get(key, (key, "-"))
        rows.append((metric, unit, value))

    return rows, ["Metric", "Unit", "Value"]
