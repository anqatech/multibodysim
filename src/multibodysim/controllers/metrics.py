from __future__ import annotations

import numpy as np


def attitude_control_metrics(
    results: dict,
    theta_target: float,
    theta_initial: float,
    Tr: float | None = None,
    settle_band_deg: float = 0.5,
) -> dict[str, float]:
    """Compute scalar quality metrics for an attitude slew maneuver."""
    t = np.asarray(results["time"], dtype=float)
    theta = np.asarray(results["q3"], dtype=float)
    u3 = np.asarray(results["u3"], dtype=float)
    tau = np.asarray(results["tau_PD"], dtype=float)

    step = float(theta_target - theta_initial)
    if np.isclose(step, 0.0):
        raise ValueError("theta_target and theta_initial must define a non-zero maneuver.")

    theta_err = theta_target - theta
    progress = (theta - theta_initial) / step

    def first_crossing(level: float) -> float:
        if step > 0.0:
            indices = np.where(progress >= level)[0]
        else:
            indices = np.where(progress <= level)[0]
        return float(t[indices[0]]) if len(indices) else np.nan

    t10 = first_crossing(0.10)
    t90 = first_crossing(0.90)

    signed_overshoot = step * (progress - 1.0)
    overshoot = max(0.0, float(np.max(signed_overshoot)))
    overshoot_percent = 100.0 * overshoot / abs(step)

    band = np.deg2rad(settle_band_deg)
    outside = np.where(np.abs(theta_err) > band)[0]
    if len(outside) == 0:
        settling_time = float(t[0])
    elif outside[-1] + 1 < len(t):
        settling_time = float(t[outside[-1] + 1])
    else:
        settling_time = np.nan

    metrics = {
        "rise_time_10_90_s": t90 - t10,
        "settling_time_s": settling_time,
        "overshoot_deg": np.rad2deg(overshoot),
        "overshoot_percent": overshoot_percent,
        "steady_state_error_deg": np.rad2deg(float(theta_err[-1])),
        "peak_u3_deg_s": np.rad2deg(float(np.max(np.abs(u3)))),
        "peak_tau_PD": float(np.max(np.abs(tau))),
        "rms_tau_PD": float(np.sqrt(np.mean(tau**2))),
        "impulse_abs_tau_PD": float(np.trapezoid(np.abs(tau), t)),
        "energy_tau_PD_sq": float(np.trapezoid(tau**2, t)),
    }

    for eta_key in sorted(key for key in results if key.startswith("eta")):
        eta = np.asarray(results[eta_key], dtype=float)
        metrics[f"peak_{eta_key}"] = float(np.max(np.abs(eta)))

        if Tr is not None:
            post = eta[t >= Tr]
            metrics[f"post_Tr_rms_{eta_key}"] = (
                float(np.sqrt(np.mean(post**2))) if len(post) else np.nan
            )

    return metrics
