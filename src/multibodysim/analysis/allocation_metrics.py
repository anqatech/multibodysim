from __future__ import annotations

from typing import Any

import numpy as np


def _wrap_to_pi(angle):
    return (angle + np.pi) % (2 * np.pi) - np.pi


def _rms(values) -> float:
    values = np.asarray(values, dtype=float)
    return float(np.sqrt(np.mean(values**2))) if len(values) else np.nan


def _nadir_angle_error(results: dict[str, Any], axis: str) -> np.ndarray:
    rG = np.column_stack((results["rG_x"], results["rG_y"]))
    rnorm = np.linalg.norm(rG, axis=1, keepdims=True)

    if np.any(rnorm == 0.0):
        raise ValueError("Zero COM radius encountered while computing nadir direction.")

    k_hat = -rG / rnorm
    alpha_nadir = np.arctan2(k_hat[:, 1], k_hat[:, 0])

    theta = np.asarray(results["q3"], dtype=float)
    if axis == "x":
        alpha_body = theta
    elif axis == "y":
        alpha_body = theta + np.pi / 2.0
    else:
        raise ValueError("pointing_axis must be 'x' or 'y'.")

    return _wrap_to_pi(alpha_body - alpha_nadir)


def _add_torque_metrics(metrics: dict[str, Any], prefix: str, torque, time) -> None:
    torque = np.asarray(torque, dtype=float)
    metrics[f"{prefix}_peak_abs_Nm"] = float(np.max(np.abs(torque)))
    metrics[f"{prefix}_rms_Nm"] = _rms(torque)
    metrics[f"{prefix}_impulse_abs_Nms"] = float(np.trapezoid(np.abs(torque), time))
    metrics[f"{prefix}_energy_N2m2s"] = float(np.trapezoid(torque**2, time))


def allocation_metrics(
    results: dict[str, Any],
    torque_weights: dict[str, float],
    *,
    pointing_axis: str = "y",
    post_window_start: float | None = None,
    theta_target: float | None = None,
) -> dict[str, Any]:
    """Compute metrics for comparing static torque-allocation choices."""
    time = np.asarray(results["time"], dtype=float)
    theta = np.asarray(results["q3"], dtype=float)
    u3 = np.asarray(results["u3"], dtype=float)
    tau_pd = np.asarray(results.get("tau_PD", np.zeros_like(time)), dtype=float)
    tau_ff = np.asarray(results.get("tau_FF", np.zeros_like(time)), dtype=float)
    tau_cmd = tau_pd + tau_ff

    metrics: dict[str, Any] = {
        "success": bool(results["success"]) if "success" in results else None,
        "nfev": results.get("nfev"),
        "torque_weight_sum": float(sum(torque_weights.values())),
        "q3_final_deg": float(np.rad2deg(theta[-1])),
        "u3_peak_abs_deg_s": float(np.rad2deg(np.max(np.abs(u3)))),
        "u3_rms_deg_s": float(np.rad2deg(_rms(u3))),
    }

    if theta_target is not None:
        theta_error = _wrap_to_pi(float(theta_target) - theta)
        metrics.update(
            {
                "attitude_error_final_deg": float(np.rad2deg(theta_error[-1])),
                "attitude_error_rms_deg": float(np.rad2deg(_rms(theta_error))),
                "attitude_error_peak_abs_deg": float(
                    np.rad2deg(np.max(np.abs(theta_error)))
                ),
            }
        )

    if {"rG_x", "rG_y"}.issubset(results):
        nadir_error = _nadir_angle_error(results, pointing_axis)
        metrics.update(
            {
                f"nadir_{pointing_axis}_error_final_deg": float(
                    np.rad2deg(nadir_error[-1])
                ),
                f"nadir_{pointing_axis}_error_rms_deg": float(
                    np.rad2deg(_rms(nadir_error))
                ),
                f"nadir_{pointing_axis}_error_peak_abs_deg": float(
                    np.rad2deg(np.max(np.abs(nadir_error)))
                ),
            }
        )

    _add_torque_metrics(metrics, "tau_PD", tau_pd, time)
    _add_torque_metrics(metrics, "tau_FF", tau_ff, time)
    _add_torque_metrics(metrics, "tau_cmd", tau_cmd, time)

    for bus_name, weight in torque_weights.items():
        metrics[f"{bus_name}_torque_weight"] = float(weight)
        _add_torque_metrics(metrics, f"tau_{bus_name}", float(weight) * tau_cmd, time)

    post_mask = None
    if post_window_start is not None:
        post_mask = time >= float(post_window_start)

    for key in sorted(results):
        if key.startswith(("eta", "zeta")):
            values = np.asarray(results[key], dtype=float)
            metrics[f"{key}_peak_abs"] = float(np.max(np.abs(values)))
            metrics[f"{key}_rms"] = _rms(values)
            metrics[f"{key}_final_abs"] = float(abs(values[-1]))

            if post_mask is not None:
                metrics[f"{key}_post_window_rms"] = _rms(values[post_mask])

    return metrics


def allocation_metrics_table(
    metrics: dict[str, Any],
) -> tuple[list[tuple[str, str, Any]], list[str]]:
    """Convert allocation metrics to display-ready rows and column names."""
    labels = {
        "success": ("Solver success", "-"),
        "nfev": ("Function evaluations", "-"),
        "torque_weight_sum": ("Torque-weight sum", "-"),
        "q3_final_deg": ("Final attitude", "deg"),
        "u3_peak_abs_deg_s": ("Peak angular velocity", "deg/s"),
        "u3_rms_deg_s": ("RMS angular velocity", "deg/s"),
        "attitude_error_final_deg": ("Final attitude error", "deg"),
        "attitude_error_rms_deg": ("RMS attitude error", "deg"),
        "attitude_error_peak_abs_deg": ("Peak abs attitude error", "deg"),
        "tau_PD_peak_abs_Nm": ("Peak abs PD torque", "N.m"),
        "tau_PD_rms_Nm": ("RMS PD torque", "N.m"),
        "tau_PD_impulse_abs_Nms": ("PD torque impulse", "N.m.s"),
        "tau_PD_energy_N2m2s": ("PD torque squared integral", "N^2.m^2.s"),
        "tau_FF_peak_abs_Nm": ("Peak abs FF torque", "N.m"),
        "tau_FF_rms_Nm": ("RMS FF torque", "N.m"),
        "tau_FF_impulse_abs_Nms": ("FF torque impulse", "N.m.s"),
        "tau_FF_energy_N2m2s": ("FF torque squared integral", "N^2.m^2.s"),
        "tau_cmd_peak_abs_Nm": ("Peak abs commanded torque", "N.m"),
        "tau_cmd_rms_Nm": ("RMS commanded torque", "N.m"),
        "tau_cmd_impulse_abs_Nms": ("Commanded torque impulse", "N.m.s"),
        "tau_cmd_energy_N2m2s": ("Commanded torque squared integral", "N^2.m^2.s"),
    }

    rows = []
    for key, value in metrics.items():
        if key.endswith("_torque_weight"):
            metric = key.removesuffix("_torque_weight").replace("_", " ")
            rows.append((f"{metric} torque weight", "-", value))
            continue

        if key.startswith("tau_bus_"):
            label = key.replace("_", " ")
            unit = "N.m"
            if key.endswith("_impulse_abs_Nms"):
                unit = "N.m.s"
            elif key.endswith("_energy_N2m2s"):
                unit = "N^2.m^2.s"
            rows.append((label, unit, value))
            continue

        if key.startswith("nadir_") and key.endswith("_error_final_deg"):
            axis = key.split("_")[1]
            rows.append((f"Final body {axis}-axis nadir error", "deg", value))
            continue
        if key.startswith("nadir_") and key.endswith("_error_rms_deg"):
            axis = key.split("_")[1]
            rows.append((f"RMS body {axis}-axis nadir error", "deg", value))
            continue
        if key.startswith("nadir_") and key.endswith("_error_peak_abs_deg"):
            axis = key.split("_")[1]
            rows.append((f"Peak abs body {axis}-axis nadir error", "deg", value))
            continue

        metric, unit = labels.get(key, (key, "-"))
        rows.append((metric, unit, value))

    return rows, ["Metric", "Unit", "Value"]
