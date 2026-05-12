from __future__ import annotations

from typing import Any

import numpy as np


def simulation_diagnostics(results: dict[str, Any]) -> dict[str, Any]:
    """Compute scalar diagnostics for a completed simulation run."""
    central_angle_deg = np.rad2deg(np.asarray(results["q_central_angle"], dtype=float))
    central_angle_speed_deg_s = np.rad2deg(
        np.asarray(results["u_central_angle"], dtype=float)
    )

    metrics: dict[str, Any] = {
        "success": bool(results["success"]),
        "nfev": results["nfev"],
        "njev": results.get("njev"),
        "nlu": results.get("nlu"),
        "central_angle_final_deg": float(central_angle_deg[-1]),
        "central_angle_drift_deg": float(central_angle_deg[-1] - central_angle_deg[0]),
        "central_angle_peak_to_peak_deg": float(np.ptp(central_angle_deg)),
        "central_angle_rms_about_mean_deg": float(
            np.sqrt(np.mean((central_angle_deg - np.mean(central_angle_deg)) ** 2))
        ),
        "central_angle_speed_peak_abs_deg_s": float(
            np.max(np.abs(central_angle_speed_deg_s))
        ),
        "central_angle_speed_rms_deg_s": float(
            np.sqrt(np.mean(central_angle_speed_deg_s**2))
        ),
    }

    for key in sorted(results):
        if key.startswith(("eta", "zeta")):
            values = np.asarray(results[key], dtype=float)
            metrics[f"{key}_peak_abs"] = float(np.max(np.abs(values)))
            metrics[f"{key}_rms"] = float(np.sqrt(np.mean(values**2)))
            metrics[f"{key}_final_abs"] = float(abs(values[-1]))

    return metrics


def simulation_diagnostics_table(
    metrics: dict[str, Any],
) -> tuple[list[tuple[str, str, Any]], list[str]]:
    """Convert simulation diagnostics to display-ready rows and column names."""
    labels = {
        "success": ("Solver success", "-"),
        "nfev": ("Function evaluations", "-"),
        "njev": ("Jacobian evaluations", "-"),
        "nlu": ("LU decompositions", "-"),
        "central_angle_final_deg": ("Final attitude", "deg"),
        "central_angle_drift_deg": ("Attitude drift", "deg"),
        "central_angle_peak_to_peak_deg": ("Attitude peak-to-peak", "deg"),
        "central_angle_rms_about_mean_deg": ("Attitude RMS about mean", "deg"),
        "central_angle_speed_peak_abs_deg_s": ("Peak angular velocity", "deg/s"),
        "central_angle_speed_rms_deg_s": ("RMS angular velocity", "deg/s"),
    }

    rows = []
    for key, value in metrics.items():
        metric, unit = labels.get(key, (key, "-"))
        rows.append((metric, unit, value))

    return rows, ["Metric", "Unit", "Value"]
