from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest

from multibodysim.plotting import (
    plot_angular_momentum_diagnostics,
    plot_control_torques,
    plot_energy_diagnostics,
    plot_flexible_modes,
    plot_nadir_angle_error,
    plot_planar_speeds,
    plot_planar_states,
    plot_relative_attitude_states,
    plot_state_envelopes,
    plot_state_spectra,
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


def test_plot_energy_diagnostics_plots_energy_dataframe():
    energy = {
        "time": np.array([0.0, 1.0, 2.0]),
        "kinetic": np.array([10.0, 11.0, 13.0]),
        "kepler_potential": np.array([-4.0, -4.5, -5.0]),
        "strain_potential": np.array([0.2, 0.3, 0.1]),
        "gravity_gradient_potential": np.array([0.0, 0.05, 0.02]),
        "total_energy_drift": np.array([0.0, 0.1, -0.1]),
        "total_energy_relative_drift": np.array([0.0, 0.01, -0.01]),
    }

    fig, axes = plot_energy_diagnostics(energy, show=False)

    assert len(axes) == 3
    assert axes[0].get_ylabel() == "Strain energy [mJ]"
    assert axes[0].get_title() == "Strain Energy"
    assert axes[1].get_ylabel() == "Energy exchange [mJ]"
    assert axes[1].get_title() == "Energy Exchange Components"
    assert axes[2].get_ylabel() == "Relative drift [-]"
    assert axes[2].get_title() == "Total Energy Relative Drift"
    assert axes[2].get_xlabel() == "Time [s]"
    assert np.allclose(axes[0].lines[0].get_ydata(), np.array([200.0, 300.0, 100.0]))
    assert np.allclose(axes[1].lines[0].get_ydata(), np.array([0.0, 500.0, 2000.0]))
    assert np.allclose(axes[1].lines[1].get_ydata(), np.array([0.0, -150.0, 80.0]))
    assert np.allclose(axes[2].lines[0].get_ydata(), np.array([0.0, 0.01, -0.01]))

    plt.close(fig)


def test_plot_energy_diagnostics_accepts_data_slice():
    energy = {
        "time": np.array([0.0, 1.0, 2.0]),
        "kinetic": np.array([10.0, 11.0, 13.0]),
        "kepler_potential": np.array([-4.0, -4.5, -5.0]),
        "strain_potential": np.array([0.2, 0.3, 0.1]),
        "gravity_gradient_potential": np.array([0.0, 0.05, 0.02]),
        "total_energy_drift": np.array([0.0, 0.1, -0.1]),
        "total_energy_relative_drift": np.array([0.0, 0.01, -0.01]),
    }

    fig, axes = plot_energy_diagnostics(
        energy,
        show=False,
        data_slice=slice(1, None),
    )

    assert np.allclose(axes[0].lines[0].get_xdata(), np.array([1.0, 2.0]))
    assert np.allclose(axes[0].lines[0].get_ydata(), np.array([300.0, 100.0]))
    assert np.allclose(axes[1].lines[0].get_ydata(), np.array([0.0, 1500.0]))
    assert np.allclose(axes[1].lines[1].get_ydata(), np.array([0.0, 230.0]))
    assert np.allclose(axes[2].lines[0].get_ydata(), np.array([0.01, -0.01]))

    plt.close(fig)


def test_plot_angular_momentum_diagnostics_plots_drift_dataframe():
    angular_momentum = {
        "time": np.array([0.0, 1.0, 2.0]),
        "H_origin_z_drift": np.array([0.0, 0.1, 0.2]),
        "H_cm_z_drift": np.array([0.0, -0.01, -0.02]),
        "H_origin_z_relative_drift": np.array([0.0, 1e-3, 2e-3]),
        "H_cm_z_relative_drift": np.array([0.0, -1e-4, -2e-4]),
    }

    fig, axes = plot_angular_momentum_diagnostics(
        angular_momentum,
        show=False,
    )

    assert len(axes) == 2
    assert axes[0].get_ylabel() == "H about O scaled drift [-]"
    assert axes[1].get_ylabel() == "H about G scaled drift [-]"
    assert axes[1].get_xlabel() == "Time [s]"
    assert np.allclose(axes[0].lines[0].get_ydata(), np.array([0.0, 1e-3, 2e-3]))
    assert np.allclose(axes[1].lines[0].get_ydata(), np.array([0.0, -1e-4, -2e-4]))

    plt.close(fig)


def test_plot_angular_momentum_diagnostics_accepts_data_slice():
    angular_momentum = {
        "time": np.array([0.0, 1.0, 2.0]),
        "H_origin_z_drift": np.array([0.0, 0.1, 0.2]),
        "H_cm_z_drift": np.array([0.0, -0.01, -0.02]),
        "H_origin_z_relative_drift": np.array([0.0, 1e-3, 2e-3]),
        "H_cm_z_relative_drift": np.array([0.0, -1e-4, -2e-4]),
    }

    fig, axes = plot_angular_momentum_diagnostics(
        angular_momentum,
        show=False,
        data_slice=slice(1, None),
    )

    assert np.allclose(axes[0].lines[0].get_xdata(), np.array([1.0, 2.0]))
    assert np.allclose(axes[0].lines[0].get_ydata(), np.array([1e-3, 2e-3]))
    assert np.allclose(axes[1].lines[0].get_ydata(), np.array([-1e-4, -2e-4]))

    plt.close(fig)


def test_plot_state_envelopes_uses_data_slice_and_draws_mean_and_band():
    results = {
        "time": np.array([0.0, 1.0, 2.0, 3.0]),
        "eta1_1": np.array([0.0, 2.0, 4.0, 6.0]),
    }

    fig, axes = plot_state_envelopes(
        results,
        ["eta1_1"],
        n_bins=1,
        show=False,
        data_slice=slice(1, None),
    )

    assert len(axes) == 1
    assert axes[0].get_xlabel() == "Time [s]"
    assert axes[0].get_title() == "eta1_1"
    assert len(axes[0].lines) == 2
    assert len(axes[0].collections) == 1
    np.testing.assert_allclose(axes[0].lines[0].get_xdata(), np.array([2.0]))
    np.testing.assert_allclose(axes[0].lines[0].get_ydata(), np.array([4.0]))

    plt.close(fig)


def test_plot_state_envelopes_applies_callable_transform_and_labels():
    results = {
        "time": np.array([0.0, 1.0, 2.0]),
        "q_relative_angle_bus_1": np.array([0.0, np.pi / 2.0, np.pi]),
    }

    fig, axes = plot_state_envelopes(
        results,
        ["q_relative_angle_bus_1"],
        transforms=np.rad2deg,
        labels={"q_relative_angle_bus_1": "bus 1"},
        ylabel="Angle [deg]",
        n_bins=1,
        show=False,
    )

    assert axes[0].get_ylabel() == "Angle [deg]"
    assert axes[0].get_title() == "bus 1"
    np.testing.assert_allclose(axes[0].lines[0].get_ydata(), np.array([90.0]))

    plt.close(fig)


def test_plot_state_spectra_uses_hertz_and_detects_sinusoid_peak():
    frequency_hz = 2.0
    time = np.linspace(0.0, 1.0, 101)[:-1]
    results = {
        "time": time,
        "eta1_1": np.sin(2.0 * np.pi * frequency_hz * time),
    }

    fig, axes = plot_state_spectra(
        results,
        ["eta1_1"],
        window=None,
        show=False,
    )

    plotted_frequency = axes[0].lines[0].get_xdata()
    plotted_amplitude = axes[0].lines[0].get_ydata()
    peak_frequency = plotted_frequency[np.argmax(plotted_amplitude)]

    assert axes[0].get_xlabel() == "Frequency [Hz]"
    np.testing.assert_allclose(peak_frequency, frequency_hz)

    plt.close(fig)


def test_plot_state_spectra_applies_per_key_transforms_and_frequency_limit():
    time = np.linspace(0.0, 1.0, 101)[:-1]
    results = {
        "time": time,
        "eta1_1": np.sin(2.0 * np.pi * 2.0 * time),
        "q_relative_angle_bus_1": np.sin(2.0 * np.pi * 4.0 * time),
    }

    fig, axes = plot_state_spectra(
        results,
        ["eta1_1", "q_relative_angle_bus_1"],
        transforms={
            "eta1_1": lambda values: 2.0 * values,
            "q_relative_angle_bus_1": np.rad2deg,
        },
        labels={"eta1_1": "eta", "q_relative_angle_bus_1": "angle"},
        max_frequency_hz=3.0,
        window=None,
        show=False,
    )

    assert len(axes) == 2
    assert axes[0].get_title() == "eta"
    assert axes[1].get_title() == "angle"
    assert np.max(axes[0].lines[0].get_xdata()) <= 3.0
    assert np.max(axes[1].lines[0].get_xdata()) <= 3.0
    np.testing.assert_allclose(np.max(axes[0].lines[0].get_ydata()), 2.0)

    plt.close(fig)


def test_state_envelope_and_spectrum_plots_reject_missing_results_keys():
    with pytest.raises(KeyError, match="time"):
        plot_state_envelopes({"eta1_1": np.array([1.0])}, ["eta1_1"], show=False)

    with pytest.raises(KeyError, match="missing"):
        plot_state_spectra(
            {"time": np.array([0.0, 1.0])},
            ["missing"],
            show=False,
        )
