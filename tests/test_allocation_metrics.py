from __future__ import annotations

import numpy as np

from multibodysim.analysis import (
    allocated_control_metrics,
    allocated_control_metrics_table,
)


def allocated_results():
    return {
        "time": np.array([0.0, 1.0, 2.0]),
        "central_acceleration_requested": np.array([0.0, 2.0, -1.0]),
        "central_acceleration_saturated": np.array([0.0, 0.5, -0.5]),
        "central_acceleration_achieved": np.array([0.0, 0.5, -0.49]),
        "central_acceleration_feasible_min": np.array([-0.4, -0.5, -0.6]),
        "central_acceleration_feasible_max": np.array([0.4, 0.5, 0.6]),
        "allocation_clipped": np.array([False, True, True]),
        "tau_bus_1": np.array([0.0, 0.2, -0.1]),
        "tau_bus_2": np.array([0.0, 0.1, -0.3]),
        "eta1_1": np.array([0.0, 0.2, -0.4]),
        "zeta1_1": np.array([0.0, -0.1, 0.3]),
    }


def test_allocated_control_metrics_compute_acceleration_and_torque_metrics():
    results = allocated_results()

    metrics = allocated_control_metrics(results, post_window_start=1.0)

    assert metrics["allocation_clipped_count"] == 2
    assert np.isclose(metrics["allocation_clipped_fraction"], 2.0 / 3.0)
    assert np.isclose(
        metrics["central_acceleration_requested_peak_abs_rad_s2"],
        2.0,
    )
    assert np.isclose(
        metrics["central_acceleration_saturated_final_rad_s2"],
        -0.5,
    )
    assert np.isclose(
        metrics["central_acceleration_achieved_minus_saturated_peak_abs_rad_s2"],
        0.01,
    )
    assert np.isclose(
        metrics["central_acceleration_feasible_min_min_rad_s2"],
        -0.6,
    )
    assert np.isclose(
        metrics["central_acceleration_feasible_max_max_rad_s2"],
        0.6,
    )
    assert np.isclose(metrics["tau_bus_1_peak_abs_Nm"], 0.2)
    assert np.isclose(
        metrics["tau_bus_2_energy_N2m2s"],
        np.trapezoid(results["tau_bus_2"] ** 2, results["time"]),
    )
    assert np.isclose(metrics["eta1_1_peak_abs"], 0.4)
    assert np.isclose(
        metrics["eta1_1_post_window_rms"],
        np.sqrt((0.2**2 + 0.4**2) / 2.0),
    )
    assert np.isclose(metrics["zeta1_1_final_abs"], 0.3)


def test_allocated_control_metrics_table_formats_known_and_dynamic_labels():
    metrics = {
        "allocation_clipped_fraction": 0.25,
        "central_acceleration_requested_peak_abs_rad_s2": 1.2,
        "tau_bus_1_peak_abs_Nm": 0.1,
        "tau_bus_2_impulse_abs_Nms": 0.4,
        "custom_metric": 3.0,
    }

    rows, columns = allocated_control_metrics_table(metrics)

    assert columns == ["Metric", "Unit", "Value"]
    assert rows[0] == ("Clipped allocation fraction", "-", 0.25)
    assert rows[1] == ("Peak abs requested acceleration", "rad/s^2", 1.2)
    assert rows[2] == ("Peak abs tau bus 1", "N.m", 0.1)
    assert rows[3] == ("Impulse abs tau bus 2", "N.m.s", 0.4)
    assert rows[4] == ("custom_metric", "-", 3.0)
