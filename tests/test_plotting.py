from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest

from multibodysim.plotting import (
    plot_control_torques,
    plot_flexible_motion,
    plot_nadir_angle_error,
    plot_speeds_u1_u2_u3_motion,
    plot_states_q1_q2_q3_motion,
)
from multibodysim.plotting.simulation_plots import nadir_angle_error, wrap_to_pi


def motion_results():
    return {
        "time": np.array([0.0, 1_000.0, 2_000.0]),
        "q1": np.array([0.0, 1_000.0, 2_000.0]),
        "q2": np.array([3_000.0, 4_000.0, 5_000.0]),
        "q3": np.array([0.0, np.pi / 2.0, 3.0 * np.pi]),
        "u1": np.array([0.0, 10.0, 20.0]),
        "u2": np.array([30.0, 40.0, 50.0]),
        "u3": np.array([0.0, np.pi / 180.0, 2.0 * np.pi / 180.0]),
        "rG_x": np.array([1.0, 0.0, -1.0]),
        "rG_y": np.array([0.0, 1.0, 0.0]),
    }


def test_wrap_to_pi_keeps_angles_in_principal_interval():
    wrapped = wrap_to_pi(np.array([-3.0 * np.pi, 0.0, 3.0 * np.pi]))

    assert np.all(wrapped >= -np.pi)
    assert np.all(wrapped < np.pi)
    assert np.allclose(wrapped, np.array([-np.pi, 0.0, -np.pi]))


def test_plot_states_scales_positions_and_wraps_angle():
    fig, axes = plot_states_q1_q2_q3_motion(motion_results(), show=False)

    assert len(axes) == 3
    assert axes[0].get_ylabel() == "Position [km]"
    assert axes[2].get_ylabel() == "Angle [deg]"
    assert axes[2].get_xlabel() == "Time [s]"
    assert np.allclose(axes[0].lines[0].get_ydata(), np.array([0.0, 1.0, 2.0]))
    assert np.allclose(axes[2].lines[0].get_ydata(), np.array([0.0, 90.0, -180.0]))

    plt.close(fig)


def test_plot_speeds_converts_angular_speed_to_degrees():
    fig, axes = plot_speeds_u1_u2_u3_motion(motion_results(), show=False)

    assert len(axes) == 3
    assert axes[0].get_ylabel() == "Velocity [m/s]"
    assert axes[2].get_ylabel() == "Angular Velocity [deg/s]"
    assert axes[2].get_xlabel() == "Time [s]"
    assert np.allclose(axes[2].lines[0].get_ydata(), np.array([0.0, 1.0, 2.0]))

    plt.close(fig)


def test_plot_flexible_motion_discovers_eta_and_zeta_keys():
    results = {
        "time": np.array([0.0, 1.0]),
        "zeta1_1": np.array([0.0, 0.1]),
        "eta2_1": np.array([0.2, 0.3]),
        "eta1_1": np.array([0.4, 0.5]),
    }

    fig, axes = plot_flexible_motion(results, show=False)

    assert len(axes) == 3
    assert [ax.get_legend().get_texts()[0].get_text() for ax in axes] == [
        "eta1_1",
        "eta2_1",
        "zeta1_1",
    ]
    assert axes[0].get_ylabel() == "Modal Amplitude [-]"
    assert axes[-1].get_ylabel() == "Modal Velocity [-]"

    plt.close(fig)


def test_plot_flexible_motion_handles_single_coordinate():
    results = {"time": np.array([0.0, 1.0]), "eta1_1": np.array([0.0, 0.1])}

    fig, axes = plot_flexible_motion(results, eta_keys=["eta1_1"], zeta_keys=[], show=False)

    assert len(axes) == 1
    assert axes[0].get_xlabel() == "Time [s]"

    plt.close(fig)


def test_plot_flexible_motion_requires_flexible_coordinates():
    with pytest.raises(ValueError, match="No flexible coordinates"):
        plot_flexible_motion({"time": np.array([0.0])}, show=False)


def test_nadir_angle_error_supports_body_axes():
    results = motion_results()

    x_error = nadir_angle_error(results, axis="x")
    y_error = nadir_angle_error(results, axis="y")

    assert np.allclose(x_error, np.array([-np.pi, -np.pi, -np.pi]))
    assert np.allclose(y_error, np.array([-np.pi / 2.0, -np.pi / 2.0, -np.pi / 2.0]))


def test_nadir_angle_error_rejects_invalid_inputs():
    zero_radius = motion_results()
    zero_radius["rG_x"] = np.array([0.0, 1.0, 1.0])
    zero_radius["rG_y"] = np.array([0.0, 1.0, 1.0])

    with pytest.raises(ValueError, match="Zero COM radius"):
        nadir_angle_error(zero_radius)

    with pytest.raises(ValueError, match="axis must be"):
        nadir_angle_error(motion_results(), axis="z")


def test_plot_nadir_angle_error_returns_labeled_axis():
    fig, ax = plot_nadir_angle_error(motion_results(), axis="y", show=False)

    assert ax.get_xlabel() == "Time [s]"
    assert ax.get_ylabel() == "$\\delta_y$ [deg]"
    assert ax.get_title() == "Body y-axis relative to nadir"

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
