from __future__ import annotations

import numpy as np
import pytest

from multibodysim.analysis import allocation_metrics, allocation_metrics_table


def allocation_results():
    return {
        "success": True,
        "nfev": 12,
        "time": np.array([0.0, 1.0, 2.0]),
        "q3": np.deg2rad(np.array([90.0, 91.0, 92.0])),
        "u3": np.deg2rad(np.array([0.0, 1.0, -2.0])),
        "rG_x": np.array([1.0, 1.0, 1.0]),
        "rG_y": np.array([0.0, 0.0, 0.0]),
        "tau_PD": np.array([0.0, 2.0, -2.0]),
        "tau_FF": np.array([1.0, 1.0, 1.0]),
        "eta1_1": np.array([0.0, 0.2, -0.4]),
        "zeta1_1": np.array([0.0, -0.1, 0.3]),
    }


def test_allocation_metrics_compute_tracking_flex_and_torque_metrics():
    results = allocation_results()
    torque_weights = {"bus_1": 0.25, "bus_2": 0.75}

    metrics = allocation_metrics(
        results,
        torque_weights,
        pointing_axis="y",
        post_window_start=1.0,
        theta_target=np.deg2rad(93.0),
    )

    assert metrics["success"] is True
    assert metrics["nfev"] == 12
    assert np.isclose(metrics["torque_weight_sum"], 1.0)
    assert np.isclose(metrics["q3_final_deg"], 92.0)
    assert np.isclose(metrics["u3_peak_abs_deg_s"], 2.0)
    assert np.isclose(metrics["attitude_error_final_deg"], 1.0)
    assert np.isclose(metrics["attitude_error_peak_abs_deg"], 3.0)

    # Nadir is along -x, so body y-axis at 180/181/182 deg has errors 0/1/2 deg.
    assert np.isclose(metrics["nadir_y_error_final_deg"], 2.0)
    assert np.isclose(metrics["nadir_y_error_peak_abs_deg"], 2.0)

    tau_cmd = results["tau_PD"] + results["tau_FF"]
    assert np.isclose(metrics["tau_cmd_peak_abs_Nm"], np.max(np.abs(tau_cmd)))
    assert np.isclose(metrics["tau_cmd_impulse_abs_Nms"], np.trapezoid(np.abs(tau_cmd), results["time"]))
    assert np.isclose(metrics["tau_bus_1_peak_abs_Nm"], 0.25 * np.max(np.abs(tau_cmd)))
    assert np.isclose(
        metrics["tau_bus_2_energy_N2m2s"],
        np.trapezoid((0.75 * tau_cmd) ** 2, results["time"]),
    )

    assert np.isclose(metrics["eta1_1_peak_abs"], 0.4)
    assert np.isclose(metrics["eta1_1_post_window_rms"], np.sqrt((0.2**2 + 0.4**2) / 2.0))
    assert np.isclose(metrics["zeta1_1_final_abs"], 0.3)


def test_allocation_metrics_table_formats_known_and_dynamic_labels():
    metrics = {
        "torque_weight_sum": 1.0,
        "nadir_y_error_rms_deg": 2.0,
        "tau_cmd_peak_abs_Nm": 0.3,
        "bus_1_torque_weight": 0.25,
        "tau_bus_1_peak_abs_Nm": 0.1,
        "eta1_1_peak_abs": 0.2,
    }

    rows, columns = allocation_metrics_table(metrics)

    assert columns == ["Metric", "Unit", "Value"]
    assert rows[0] == ("Torque-weight sum", "-", 1.0)
    assert rows[1] == ("RMS body y-axis nadir error", "deg", 2.0)
    assert rows[2] == ("Peak abs commanded torque", "N.m", 0.3)
    assert rows[3] == ("bus 1 torque weight", "-", 0.25)
    assert rows[4] == ("tau bus 1 peak abs Nm", "N.m", 0.1)
    assert rows[5] == ("eta1_1_peak_abs", "-", 0.2)


def test_allocation_metrics_rejects_invalid_pointing_axis():
    with pytest.raises(ValueError, match="pointing_axis"):
        allocation_metrics(allocation_results(), {"bus_1": 1.0}, pointing_axis="z")
