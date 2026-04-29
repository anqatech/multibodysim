from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from multibodysim.plotting import plot_control_torques


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
