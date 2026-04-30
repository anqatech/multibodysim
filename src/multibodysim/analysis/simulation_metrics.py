from __future__ import annotations

from typing import Any

import numpy as np


def simulation_diagnostics(results: dict[str, Any]) -> dict[str, Any]:
    """Compute scalar diagnostics for a completed simulation run."""
    q3_deg = np.rad2deg(np.asarray(results["q3"], dtype=float))
    u3_deg_s = np.rad2deg(np.asarray(results["u3"], dtype=float))

    metrics: dict[str, Any] = {
        "success": bool(results["success"]),
        "nfev": results["nfev"],
        "njev": results.get("njev"),
        "nlu": results.get("nlu"),
        "q3_final_deg": float(q3_deg[-1]),
        "q3_drift_deg": float(q3_deg[-1] - q3_deg[0]),
        "q3_peak_to_peak_deg": float(np.ptp(q3_deg)),
        "q3_rms_about_mean_deg": float(np.sqrt(np.mean((q3_deg - np.mean(q3_deg)) ** 2))),
        "u3_peak_abs_deg_s": float(np.max(np.abs(u3_deg_s))),
        "u3_rms_deg_s": float(np.sqrt(np.mean(u3_deg_s**2))),
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
        "q3_final_deg": ("Final attitude", "deg"),
        "q3_drift_deg": ("Attitude drift", "deg"),
        "q3_peak_to_peak_deg": ("Attitude peak-to-peak", "deg"),
        "q3_rms_about_mean_deg": ("Attitude RMS about mean", "deg"),
        "u3_peak_abs_deg_s": ("Peak angular velocity", "deg/s"),
        "u3_rms_deg_s": ("RMS angular velocity", "deg/s"),
    }

    rows = []
    for key, value in metrics.items():
        metric, unit = labels.get(key, (key, "-"))
        rows.append((metric, unit, value))

    return rows, ["Metric", "Unit", "Value"]
