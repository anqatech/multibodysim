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


def attitude_acquisition_metrics(
    results: dict[str, Any],
    *,
    settling_band_deg: float = 0.5,
) -> dict[str, Any]:
    """Compute acquisition-style metrics for an attitude error driven to zero."""
    time = np.asarray(results["time"], dtype=float)
    attitude_error = _attitude_error(results)
    initial_error = float(attitude_error[0])
    initial_error_abs = abs(initial_error)

    if np.isclose(initial_error_abs, 0.0):
        raise ValueError(
            "attitude_acquisition_metrics requires a non-zero initial "
            "attitude error."
        )

    progress = 1.0 - attitude_error / initial_error
    t10 = _first_crossing_time(time, progress, 0.10)
    t90 = _first_crossing_time(time, progress, 0.90)

    initial_sign = np.sign(initial_error)
    opposite_side_error = -initial_sign * attitude_error
    overshoot = max(0.0, float(np.max(opposite_side_error)))
    zero_crossing_indices = np.where(opposite_side_error >= 0.0)[0]
    zero_crossing_time = (
        float(time[zero_crossing_indices[0]])
        if len(zero_crossing_indices)
        else np.nan
    )

    band = np.deg2rad(float(settling_band_deg))
    outside = np.where(np.abs(attitude_error) > band)[0]
    if len(outside) == 0:
        settling_time = float(time[0])
    elif outside[-1] + 1 < len(time):
        settling_time = float(time[outside[-1] + 1])
    else:
        settling_time = np.nan

    return {
        "acquisition_initial_error_deg": float(np.rad2deg(initial_error)),
        "acquisition_initial_error_abs_deg": float(np.rad2deg(initial_error_abs)),
        "acquisition_final_error_deg": float(np.rad2deg(attitude_error[-1])),
        "acquisition_final_error_abs_deg": float(
            np.rad2deg(abs(attitude_error[-1]))
        ),
        "acquisition_error_reduction_percent": float(
            100.0 * (1.0 - abs(attitude_error[-1]) / initial_error_abs)
        ),
        "acquisition_time_10_percent_error_reduction_s": t10,
        "acquisition_time_90_percent_error_reduction_s": t90,
        "acquisition_decay_time_10_90_s": t90 - t10,
        "acquisition_zero_crossing_time_s": zero_crossing_time,
        "acquisition_overshoot_deg": float(np.rad2deg(overshoot)),
        "acquisition_overshoot_percent": float(
            100.0 * overshoot / initial_error_abs
        ),
        "acquisition_settling_time_s": settling_time,
        "acquisition_settling_band_deg": float(settling_band_deg),
    }


def _attitude_error(results: dict[str, Any]) -> np.ndarray:
    if "attitude_error" in results:
        return _wrap_to_pi(np.asarray(results["attitude_error"], dtype=float))

    theta = np.asarray(results["q_central_angle"], dtype=float)
    theta_reference = np.asarray(results["theta_reference"], dtype=float)
    return _wrap_to_pi(theta_reference - theta)


def _first_crossing_time(time, values, level: float) -> float:
    indices = np.where(np.asarray(values, dtype=float) >= float(level))[0]
    return float(time[indices[0]]) if len(indices) else np.nan


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
        "acquisition_initial_error_deg": ("Initial acquisition error", "deg"),
        "acquisition_initial_error_abs_deg": (
            "Initial abs acquisition error",
            "deg",
        ),
        "acquisition_final_error_deg": ("Final acquisition error", "deg"),
        "acquisition_final_error_abs_deg": ("Final abs acquisition error", "deg"),
        "acquisition_error_reduction_percent": ("Acquisition error reduction", "%"),
        "acquisition_time_10_percent_error_reduction_s": (
            "Time to 10% error reduction",
            "s",
        ),
        "acquisition_time_90_percent_error_reduction_s": (
            "Time to 90% error reduction",
            "s",
        ),
        "acquisition_decay_time_10_90_s": ("Acquisition decay time 10-90%", "s"),
        "acquisition_zero_crossing_time_s": ("Acquisition zero-crossing time", "s"),
        "acquisition_overshoot_deg": ("Acquisition overshoot", "deg"),
        "acquisition_overshoot_percent": ("Acquisition overshoot", "%"),
        "acquisition_settling_time_s": ("Acquisition settling time", "s"),
        "acquisition_settling_band_deg": ("Acquisition settling band", "deg"),
    }

    rows = []
    for key, value in metrics.items():
        metric, unit = labels.get(key, (key, "-"))
        rows.append((metric, unit, value))

    return rows, ["Metric", "Unit", "Value"]
