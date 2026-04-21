import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker


def wrap_to_pi(angle):
    return (angle + np.pi) % (2 * np.pi) - np.pi

def _format_large_number_axes(ax):
    ax.ticklabel_format(axis="y", style="plain", useOffset=False)
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda y, _: f"{int(y):,}"))

def plot_states_q1_q2_q3_motion(results, figsize=(10, 5), show=True):
    ts = results["time"]
    q1_km = results["q1"] / 1e3
    q2_km = results["q2"] / 1e3
    q3_wrapped = wrap_to_pi(results["q3"])

    fig, axes = plt.subplots(3, 1, sharex=True, figsize=figsize)

    axes[0].plot(ts, q1_km)
    axes[0].legend(["q1"])
    axes[0].set_ylabel("Position [km]")

    axes[1].plot(ts, q2_km)
    axes[1].legend(["q2"])
    axes[1].set_ylabel("Position [km]")

    axes[2].plot(ts, np.rad2deg(q3_wrapped))
    axes[2].legend(["q3"])
    axes[2].set_ylabel("Angle [deg]")
    axes[2].set_xlabel("Time [s]")

    _format_large_number_axes(axes[0])
    _format_large_number_axes(axes[1])
    axes[2].ticklabel_format(axis="y", style="plain", useOffset=False)

    fig.tight_layout()

    if show:
        plt.show()

    return fig, axes

def plot_speeds_u1_u2_u3_motion(results, figsize=(10, 5), show=True):
    ts = results["time"]

    fig, axes = plt.subplots(3, 1, sharex=True, figsize=figsize)

    axes[0].plot(ts, results["u1"])
    axes[0].legend(["u1"])
    axes[0].set_ylabel("Velocity [m/s]")

    axes[1].plot(ts, results["u2"])
    axes[1].legend(["u2"])
    axes[1].set_ylabel("Velocity [m/s]")

    axes[2].plot(ts, np.rad2deg(results["u3"]))
    axes[2].legend(["u3"])
    axes[2].set_ylabel("Angular Velocity [deg/s]")
    axes[2].set_xlabel("Time [s]")

    _format_large_number_axes(axes[0])
    _format_large_number_axes(axes[1])

    fig.tight_layout()

    if show:
        plt.show()

    return fig, axes

def plot_flexible_motion(results, eta_keys=None, zeta_keys=None, figsize=(10, 8), show=True):
    ts = results["time"]

    if eta_keys is None:
        eta_keys = sorted([k for k in results.keys() if k.startswith("eta")])

    if zeta_keys is None:
        zeta_keys = sorted([k for k in results.keys() if k.startswith("zeta")])

    keys = eta_keys + zeta_keys
    nplots = len(keys)

    if nplots == 0:
        raise ValueError("No flexible coordinates found in results.")

    fig, axes = plt.subplots(nplots, 1, sharex=True, figsize=figsize)

    if nplots == 1:
        axes = [axes]

    for ax, key in zip(axes, keys):
        ax.plot(ts, results[key])
        ax.legend([key])

        if key.startswith("eta"):
            ax.set_ylabel("Modal Amplitude [-]")
        else:
            ax.set_ylabel("Modal Velocity [-]")

    axes[-1].set_xlabel("Time [s]")

    fig.tight_layout()

    if show:
        plt.show()

    return fig, axes
