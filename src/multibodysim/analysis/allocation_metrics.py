from __future__ import annotations

from typing import Any

import numpy as np

from .controller_metrics import (
    _add_flexible_metrics,
    _add_signal_metrics,
    _add_torque_metrics,
)


def allocated_control_metrics(
    results: dict[str, Any],
    *,
    post_window_start: float | None = None,
) -> dict[str, Any]:
    """Compute metrics for the acceleration-command allocated controller."""
    time = np.asarray(results["time"], dtype=float)
    requested = np.asarray(
        results["central_acceleration_requested"],
        dtype=float,
    )
    saturated = np.asarray(
        results["central_acceleration_saturated"],
        dtype=float,
    )
    achieved = np.asarray(
        results["central_acceleration_achieved"],
        dtype=float,
    )
    feasible_min = np.asarray(
        results["central_acceleration_feasible_min"],
        dtype=float,
    )
    feasible_max = np.asarray(
        results["central_acceleration_feasible_max"],
        dtype=float,
    )
    clipped = np.asarray(results["allocation_clipped"], dtype=bool)

    metrics: dict[str, Any] = {
        "allocation_clipped_count": int(np.count_nonzero(clipped)),
        "allocation_clipped_fraction": (
            float(np.count_nonzero(clipped) / clipped.size)
            if clipped.size
            else np.nan
        ),
        "central_acceleration_feasible_min_min_rad_s2": float(
            np.min(feasible_min)
        ),
        "central_acceleration_feasible_max_max_rad_s2": float(
            np.max(feasible_max)
        ),
    }

    _add_signal_metrics(
        metrics,
        "central_acceleration_requested",
        requested,
        suffix="_rad_s2",
    )
    _add_signal_metrics(
        metrics,
        "central_acceleration_saturated",
        saturated,
        suffix="_rad_s2",
    )
    _add_signal_metrics(
        metrics,
        "central_acceleration_achieved",
        achieved,
        suffix="_rad_s2",
    )
    _add_signal_metrics(
        metrics,
        "central_acceleration_achieved_minus_saturated",
        achieved - saturated,
        suffix="_rad_s2",
    )

    for key in sorted(results):
        if key.startswith("tau_bus_"):
            _add_torque_metrics(metrics, key, results[key], time)

    _add_flexible_metrics(
        metrics,
        results,
        post_window_start=post_window_start,
    )
    return metrics


def allocated_control_metrics_table(
    metrics: dict[str, Any],
) -> tuple[list[tuple[str, str, Any]], list[str]]:
    """Convert allocated-control metrics to display-ready rows."""
    labels = {
        "allocation_clipped_count": ("Clipped allocation samples", "-"),
        "allocation_clipped_fraction": ("Clipped allocation fraction", "-"),
        "central_acceleration_feasible_min_min_rad_s2": (
            "Minimum feasible acceleration",
            "rad/s^2",
        ),
        "central_acceleration_feasible_max_max_rad_s2": (
            "Maximum feasible acceleration",
            "rad/s^2",
        ),
    }

    for prefix, label in (
        ("central_acceleration_requested", "Requested acceleration"),
        ("central_acceleration_saturated", "Saturated acceleration"),
        ("central_acceleration_achieved", "Achieved acceleration"),
        (
            "central_acceleration_achieved_minus_saturated",
            "Achieved-minus-saturated acceleration",
        ),
    ):
        labels[f"{prefix}_final_rad_s2"] = (f"Final {label.lower()}", "rad/s^2")
        labels[f"{prefix}_rms_rad_s2"] = (f"RMS {label.lower()}", "rad/s^2")
        labels[f"{prefix}_peak_abs_rad_s2"] = (
            f"Peak abs {label.lower()}",
            "rad/s^2",
        )

    rows = []
    for key, value in metrics.items():
        if key.startswith("tau_bus_"):
            rows.append((_torque_label(key), _torque_unit(key), value))
            continue

        metric, unit = labels.get(key, (key, "-"))
        rows.append((metric, unit, value))

    return rows, ["Metric", "Unit", "Value"]


def _torque_label(key: str) -> str:
    if key.endswith("_peak_abs_Nm"):
        prefix = key.removesuffix("_peak_abs_Nm")
        return f"Peak abs {prefix.replace('_', ' ')}"
    if key.endswith("_rms_Nm"):
        prefix = key.removesuffix("_rms_Nm")
        return f"RMS {prefix.replace('_', ' ')}"
    if key.endswith("_impulse_abs_Nms"):
        prefix = key.removesuffix("_impulse_abs_Nms")
        return f"Impulse abs {prefix.replace('_', ' ')}"
    if key.endswith("_energy_N2m2s"):
        prefix = key.removesuffix("_energy_N2m2s")
        return f"Squared integral {prefix.replace('_', ' ')}"
    return key.replace("_", " ")


def _torque_unit(key: str) -> str:
    if key.endswith(("_peak_abs_Nm", "_rms_Nm")):
        return "N.m"
    if key.endswith("_impulse_abs_Nms"):
        return "N.m.s"
    if key.endswith("_energy_N2m2s"):
        return "N^2.m^2.s"
    return "-"
