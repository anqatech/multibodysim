from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest

from multibodysim.plotting import (
    plot_control_torques,
    plot_flexible_modes,
    plot_nadir_angle_error,
    plot_planar_speeds,
    plot_planar_states,
    plot_relative_attitude_states,
)
from multibodysim.plotting.simulation_plots import compute_nadir_angle_error, wrap_to_pi


def motion_results():
    return {
        "time": np.array([0.0, 1_000.0, 2_000.0]),
        "q1": np.array([0.0, 1_000.0, 2_000.0]),
        "q2": np.array([3_000.0, 4_000.0, 5_000.0]),
        "q_central_angle": np.array([0.0, np.pi / 2.0, 3.0 * np.pi]),
        "u1": np.array([0.0, 10.0, 20.0]),
        "u2": np.array([30.0, 40.0, 50.0]),
        "u_central_angle": np.array(
            [0.0, np.pi / 180.0, 2.0 * np.pi / 180.0]
        ),
        "rG_x": np.array([1.0, 0.0, -1.0]),
        "rG_y": np.array([0.0, 1.0, 0.0]),
    }


def multiangle_motion_results():
    results = motion_results()
    results["q_relative_angle_bus_1"] = np.array([0.1, 0.2, 0.3])
    results["q_relative_angle_bus_3"] = np.array([-0.1, -0.2, -0.3])
    results["u_relative_angle_bus_1"] = np.array([0.01, 0.02, 0.03])
    results["u_relative_angle_bus_3"] = np.array([-0.01, -0.02, -0.03])
    return results


def test_wrap_to_pi_keeps_angles_in_principal_interval():
    wrapped = wrap_to_pi(np.array([-3.0 * np.pi, 0.0, 3.0 * np.pi]))

    assert np.all(wrapped >= -np.pi)
    assert np.all(wrapped < np.pi)
    assert np.allclose(wrapped, np.array([-np.pi, 0.0, -np.pi]))


def test_plot_states_scales_positions_and_wraps_angle():
    fig, axes = plot_planar_states(motion_results(), show=False)

    assert len(axes) == 3
    assert axes[0].get_ylabel() == "Position [km]"
    assert axes[2].get_ylabel() == "Angle [deg]"
    assert axes[2].get_xlabel() == "Time [s]"
    assert np.allclose(axes[0].lines[0].get_ydata(), np.array([0.0, 1.0, 2.0]))
    assert np.allclose(axes[2].lines[0].get_ydata(), np.array([0.0, 90.0, -180.0]))

    plt.close(fig)


def test_plot_states_accepts_data_slice():
    fig, axes = plot_planar_states(
        motion_results(),
        show=False,
        data_slice=slice(1, None),
    )

    assert np.allclose(axes[0].lines[0].get_xdata(), np.array([1_000.0, 2_000.0]))
    assert np.allclose(axes[0].lines[0].get_ydata(), np.array([1.0, 2.0]))
    assert np.allclose(axes[2].lines[0].get_ydata(), np.array([90.0, -180.0]))

    plt.close(fig)


def test_plot_speeds_converts_angular_speed_to_degrees():
    fig, axes = plot_planar_speeds(motion_results(), show=False)

    assert len(axes) == 3
    assert axes[0].get_ylabel() == "Velocity [m/s]"
    assert axes[2].get_ylabel() == "Angular Velocity [deg/s]"
    assert axes[2].get_xlabel() == "Time [s]"
    assert np.allclose(axes[2].lines[0].get_ydata(), np.array([0.0, 1.0, 2.0]))

    plt.close(fig)


def test_plot_speeds_accepts_data_slice():
    fig, axes = plot_planar_speeds(
        motion_results(),
        show=False,
        data_slice=slice(None, 2),
    )

    assert np.allclose(axes[0].lines[0].get_xdata(), np.array([0.0, 1_000.0]))
    assert np.allclose(axes[2].lines[0].get_ydata(), np.array([0.0, 1.0]))

    plt.close(fig)


def test_motion_plots_use_multiangle_central_attitude_keys():
    fig_states, state_axes = plot_planar_states(
        multiangle_motion_results(),
        show=False,
    )
    fig_speeds, speed_axes = plot_planar_speeds(
        multiangle_motion_results(),
        show=False,
    )

    assert state_axes[2].get_legend().get_texts()[0].get_text() == "q_central_angle"
    assert speed_axes[2].get_legend().get_texts()[0].get_text() == "u_central_angle"
    assert np.allclose(
        state_axes[2].lines[0].get_ydata(),
        np.array([0.0, 90.0, -180.0]),
    )
    assert np.allclose(speed_axes[2].lines[0].get_ydata(), np.array([0.0, 1.0, 2.0]))

    plt.close(fig_states)
    plt.close(fig_speeds)


def test_plot_relative_attitude_states_plots_multiangle_only_states():
    fig, axes = plot_relative_attitude_states(
        multiangle_motion_results(),
        show=False,
    )

    assert len(axes) == 2
    assert axes[0].get_ylabel() == "Relative angle [deg]"
    assert axes[1].get_ylabel() == "Relative angular velocity [deg/s]"
    assert axes[1].get_xlabel() == "Time [s]"
    assert [
        text.get_text()
        for text in axes[0].get_legend().get_texts()
    ] == [
        "q_relative_angle_bus_1",
        "q_relative_angle_bus_3",
    ]
    assert np.allclose(axes[0].lines[0].get_ydata(), np.rad2deg([0.1, 0.2, 0.3]))
    assert np.allclose(axes[1].lines[0].get_ydata(), np.rad2deg([0.01, 0.02, 0.03]))

    plt.close(fig)


def test_plot_relative_attitude_states_accepts_data_slice():
    fig, axes = plot_relative_attitude_states(
        multiangle_motion_results(),
        show=False,
        data_slice=slice(1, None),
    )

    assert np.allclose(axes[0].lines[0].get_xdata(), np.array([1_000.0, 2_000.0]))
    assert np.allclose(axes[0].lines[0].get_ydata(), np.rad2deg([0.2, 0.3]))
    assert np.allclose(axes[1].lines[0].get_ydata(), np.rad2deg([0.02, 0.03]))

    plt.close(fig)


def test_plot_relative_attitude_states_requires_relative_states():
    with pytest.raises(ValueError, match="No relative attitude states"):
        plot_relative_attitude_states(motion_results(), show=False)


def test_plot_flexible_modes_discovers_eta_and_zeta_keys():
    results = {
        "time": np.array([0.0, 1.0]),
        "zeta1_1": np.array([0.0, 0.1]),
        "eta2_1": np.array([0.2, 0.3]),
        "eta1_1": np.array([0.4, 0.5]),
    }

    fig, axes = plot_flexible_modes(results, show=False)

    assert len(axes) == 3
    assert [ax.get_legend().get_texts()[0].get_text() for ax in axes] == [
        "eta1_1",
        "eta2_1",
        "zeta1_1",
    ]
    assert axes[0].get_ylabel() == "Modal Amplitude [-]"
    assert axes[-1].get_ylabel() == "Modal Velocity [-]"

    plt.close(fig)


def test_plot_flexible_modes_accepts_data_slice():
    results = {
        "time": np.array([0.0, 1.0, 2.0]),
        "eta1_1": np.array([0.0, 0.1, 0.2]),
    }

    fig, axes = plot_flexible_modes(
        results,
        eta_keys=["eta1_1"],
        zeta_keys=[],
        show=False,
        data_slice=slice(1, None),
    )

    assert np.allclose(axes[0].lines[0].get_xdata(), np.array([1.0, 2.0]))
    assert np.allclose(axes[0].lines[0].get_ydata(), np.array([0.1, 0.2]))

    plt.close(fig)


def test_plot_flexible_modes_handles_single_coordinate():
    results = {"time": np.array([0.0, 1.0]), "eta1_1": np.array([0.0, 0.1])}

    fig, axes = plot_flexible_modes(
        results,
        eta_keys=["eta1_1"],
        zeta_keys=[],
        show=False,
    )

    assert len(axes) == 1
    assert axes[0].get_xlabel() == "Time [s]"

    plt.close(fig)


def test_plot_flexible_modes_requires_flexible_coordinates():
    with pytest.raises(ValueError, match="No flexible coordinates"):
        plot_flexible_modes({"time": np.array([0.0])}, show=False)


def test_compute_nadir_angle_error_supports_body_axes():
    results = motion_results()

    x_error = compute_nadir_angle_error(results, axis="x")
    y_error = compute_nadir_angle_error(results, axis="y")

    assert np.allclose(x_error, np.array([-np.pi, -np.pi, -np.pi]))
    assert np.allclose(y_error, np.array([-np.pi / 2.0, -np.pi / 2.0, -np.pi / 2.0]))


def test_compute_nadir_angle_error_rejects_invalid_inputs():
    zero_radius = motion_results()
    zero_radius["rG_x"] = np.array([0.0, 1.0, 1.0])
    zero_radius["rG_y"] = np.array([0.0, 1.0, 1.0])

    with pytest.raises(ValueError, match="Zero COM radius"):
        compute_nadir_angle_error(zero_radius)

    with pytest.raises(ValueError, match="axis must be"):
        compute_nadir_angle_error(motion_results(), axis="z")


def test_plot_nadir_angle_error_returns_labeled_axis():
    fig, ax = plot_nadir_angle_error(motion_results(), axis="y", show=False)

    assert ax.get_xlabel() == "Time [s]"
    assert ax.get_ylabel() == "$\\delta_y$ [deg]"
    assert ax.get_title() == "Body y-axis relative to nadir"

    plt.close(fig)


def test_plot_nadir_angle_error_accepts_data_slice():
    fig, ax = plot_nadir_angle_error(
        motion_results(),
        axis="y",
        show=False,
        data_slice=slice(1, None),
    )

    assert np.allclose(ax.lines[0].get_xdata(), np.array([1_000.0, 2_000.0]))
    assert np.allclose(ax.lines[0].get_ydata(), np.array([-90.0, -90.0]))

    plt.close(fig)


def test_plot_control_torques_returns_figure_and_axes():
    results = {
        "time": np.array([0.0, 1.0, 2.0]),
        "tau_PD": np.array([0.0, 0.1, 0.0]),
        "tau_FF": np.array([0.0, 0.0, 0.0]),
    }

    fig, axes = plot_control_torques(results, show=False)

    assert len(axes) == 2
    assert axes[0].get_ylabel() == "PD torque [N.m]"
    assert axes[1].get_ylabel() == "FF torque [N.m]"
    assert axes[1].get_xlabel() == "Time [s]"

    plt.close(fig)


def test_plot_control_torques_accepts_data_slice():
    results = {
        "time": np.array([0.0, 1.0, 2.0]),
        "tau_PD": np.array([0.0, 0.1, 0.2]),
        "tau_FF": np.array([1.0, 1.1, 1.2]),
    }

    fig, axes = plot_control_torques(results, show=False, data_slice=slice(None, 2))

    assert np.allclose(axes[0].lines[0].get_xdata(), np.array([0.0, 1.0]))
    assert np.allclose(axes[0].lines[0].get_ydata(), np.array([0.0, 0.1]))
    assert np.allclose(axes[1].lines[0].get_ydata(), np.array([1.0, 1.1]))

    plt.close(fig)
