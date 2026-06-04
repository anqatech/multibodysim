import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker


def wrap_to_pi(angle):
    return (angle + np.pi) % (2 * np.pi) - np.pi

def _format_large_number_axes(ax):
    ax.ticklabel_format(axis="y", style="plain", useOffset=False)
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda y, _: f"{int(y):,}"))

def _as_plot_slice(data_slice):
    return slice(None) if data_slice is None else data_slice

def _as_state_keys(keys):
    if isinstance(keys, str):
        keys = [keys]
    else:
        keys = list(keys)

    if not keys:
        raise ValueError("At least one state key must be selected for plotting.")

    return keys

def _state_transform(transforms, key):
    if transforms is None:
        return None

    if callable(transforms):
        return transforms

    if isinstance(transforms, dict):
        transform = transforms.get(key)
        if transform is not None and not callable(transform):
            raise TypeError(f"Transform for '{key}' must be callable.")
        return transform

    raise TypeError("transforms must be None, a callable, or a dict of callables.")

def _state_label(labels, key):
    if labels is None:
        return key

    if not isinstance(labels, dict):
        raise TypeError("labels must be None or a dict of labels.")

    return labels.get(key, key)

def _state_values(results, key, plot_slice, transforms=None):
    if "time" not in results:
        raise KeyError("Expected 'time' in results.")

    if key not in results:
        raise KeyError(f"Expected '{key}' in results.")

    values = np.asarray(results[key], dtype=float)[plot_slice]
    transform = _state_transform(transforms, key)
    if transform is not None:
        values = np.asarray(transform(values), dtype=float)

    return values

def _default_state_figsize(nplots):
    return (10, 2.8 * nplots)

def _as_axes_list(axes):
    if isinstance(axes, np.ndarray):
        return list(axes.flat)

    return [axes]

def _state_time(results, plot_slice):
    if "time" not in results:
        raise KeyError("Expected 'time' in results.")

    return np.asarray(results["time"], dtype=float)[plot_slice]

def _envelope_statistics(time, values, n_bins):
    time = np.asarray(time, dtype=float)
    values = np.asarray(values, dtype=float)

    if n_bins <= 0:
        raise ValueError("n_bins must be positive.")

    if len(time) == 0:
        raise ValueError("Envelope plots require at least one time sample.")

    if len(time) != len(values):
        raise ValueError("time and values must have the same length.")

    edges = np.linspace(time[0], time[-1], n_bins + 1)
    centres = 0.5 * (edges[:-1] + edges[1:])
    lower = np.full(n_bins, np.nan)
    upper = np.full(n_bins, np.nan)
    mean = np.full(n_bins, np.nan)

    bin_index = np.searchsorted(edges, time, side="right") - 1
    bin_index = np.clip(bin_index, 0, n_bins - 1)

    for index in range(n_bins):
        mask = bin_index == index
        if not np.any(mask):
            continue
        chunk = values[mask]
        lower[index] = np.min(chunk)
        upper[index] = np.max(chunk)
        mean[index] = np.mean(chunk)

    valid = ~np.isnan(mean)
    return centres[valid], lower[valid], upper[valid], mean[valid]

def _frequency_spectrum(time, values, window="hann", demean=True):
    time = np.asarray(time, dtype=float)
    values = np.asarray(values, dtype=float)

    if len(time) < 2:
        raise ValueError("Spectrum plots require at least two time samples.")

    if len(time) != len(values):
        raise ValueError("time and values must have the same length.")

    dt = float(np.median(np.diff(time)))
    if dt <= 0.0:
        raise ValueError("time samples must be strictly increasing.")

    spectrum_values = values - np.mean(values) if demean else values.copy()
    if window in ("hann", "hanning"):
        weights = np.hanning(len(spectrum_values))
    elif window in (None, "boxcar", "rectangular"):
        weights = np.ones_like(spectrum_values)
    else:
        raise ValueError("window must be 'hann', 'hanning', 'boxcar', or None.")

    if np.allclose(weights.sum(), 0.0):
        weights = np.ones_like(spectrum_values)

    spectrum = np.fft.rfft(spectrum_values * weights)
    frequency_hz = np.fft.rfftfreq(len(spectrum_values), d=dt)
    amplitude = 2.0 * np.abs(spectrum) / weights.sum()

    return frequency_hz[1:], amplitude[1:]

def _attitude_key(results):
    if "q_central_angle" in results:
        return "q_central_angle"

    raise KeyError("Expected 'q_central_angle' in results.")

def _angular_speed_key(results):
    if "u_central_angle" in results:
        return "u_central_angle"

    raise KeyError("Expected 'u_central_angle' in results.")

def _relative_attitude_keys(results):
    angle_keys = sorted(
        key for key in results
        if key.startswith("q_relative_angle_")
    )
    speed_keys = sorted(
        key for key in results
        if key.startswith("u_relative_angle_")
    )

    if not angle_keys and not speed_keys:
        raise ValueError("No relative attitude states found in results.")

    return angle_keys, speed_keys

def _tight_layout_with_title(fig, title=None):
    if title is not None:
        fig.suptitle(title)
        fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.95))
        return

    fig.tight_layout()

def plot_planar_states(
    results,
    figsize=(10, 5),
    show=True,
    data_slice=None,
    title=None,
):
    plot_slice = _as_plot_slice(data_slice)
    ts = results["time"][plot_slice]
    q1_km = results["q1"][plot_slice] / 1e3
    q2_km = results["q2"][plot_slice] / 1e3
    attitude_key = _attitude_key(results)
    central_angle_wrapped = wrap_to_pi(results[attitude_key][plot_slice])

    fig, axes = plt.subplots(3, 1, sharex=True, figsize=figsize)

    axes[0].plot(ts, q1_km)
    axes[0].legend(["q1"])
    axes[0].set_ylabel("Position [km]")

    axes[1].plot(ts, q2_km)
    axes[1].legend(["q2"])
    axes[1].set_ylabel("Position [km]")

    axes[2].plot(ts, np.rad2deg(central_angle_wrapped))
    axes[2].legend([attitude_key])
    axes[2].set_ylabel("Angle [deg]")
    axes[2].set_xlabel("Time [s]")

    _format_large_number_axes(axes[0])
    _format_large_number_axes(axes[1])
    axes[2].ticklabel_format(axis="y", style="plain", useOffset=False)

    _tight_layout_with_title(fig, title=title)

    if show:
        plt.show()

    return fig, axes

def plot_relative_attitude_states(
    results,
    angle_keys=None,
    speed_keys=None,
    figsize=(10, 5),
    show=True,
    data_slice=None,
    title=None,
):
    plot_slice = _as_plot_slice(data_slice)
    ts = results["time"][plot_slice]

    discovered_angle_keys, discovered_speed_keys = _relative_attitude_keys(results)

    if angle_keys is None:
        angle_keys = discovered_angle_keys

    if speed_keys is None:
        speed_keys = discovered_speed_keys

    if not angle_keys and not speed_keys:
        raise ValueError("No relative attitude states selected for plotting.")

    fig, axes = plt.subplots(2, 1, sharex=True, figsize=figsize)

    if angle_keys:
        for key in angle_keys:
            axes[0].plot(ts, np.rad2deg(results[key][plot_slice]))
        axes[0].legend(angle_keys)
    axes[0].set_ylabel("Relative angle [deg]")
    axes[0].grid(True)

    if speed_keys:
        for key in speed_keys:
            axes[1].plot(ts, np.rad2deg(results[key][plot_slice]))
        axes[1].legend(speed_keys)
    axes[1].set_ylabel("Relative angular velocity [deg/s]")
    axes[1].set_xlabel("Time [s]")
    axes[1].grid(True)
    axes[1].xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))

    _tight_layout_with_title(fig, title=title)

    if show:
        plt.show()

    return fig, axes

def plot_planar_speeds(
    results,
    figsize=(10, 5),
    show=True,
    data_slice=None,
    title=None,
):
    plot_slice = _as_plot_slice(data_slice)
    ts = results["time"][plot_slice]
    angular_speed_key = _angular_speed_key(results)

    fig, axes = plt.subplots(3, 1, sharex=True, figsize=figsize)

    axes[0].plot(ts, results["u1"][plot_slice])
    axes[0].legend(["u1"])
    axes[0].set_ylabel("Velocity [m/s]")

    axes[1].plot(ts, results["u2"][plot_slice])
    axes[1].legend(["u2"])
    axes[1].set_ylabel("Velocity [m/s]")

    axes[2].plot(ts, np.rad2deg(results[angular_speed_key][plot_slice]))
    axes[2].legend([angular_speed_key])
    axes[2].set_ylabel("Angular Velocity [deg/s]")
    axes[2].set_xlabel("Time [s]")

    _format_large_number_axes(axes[0])
    _format_large_number_axes(axes[1])

    _tight_layout_with_title(fig, title=title)

    if show:
        plt.show()

    return fig, axes

def plot_flexible_modes(
    results,
    eta_keys=None,
    zeta_keys=None,
    figsize=(10, 8),
    show=True,
    data_slice=None,
    title=None,
):
    plot_slice = _as_plot_slice(data_slice)
    ts = results["time"][plot_slice]

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
        ax.plot(ts, results[key][plot_slice])
        ax.legend([key])

        if key.startswith("eta"):
            ax.set_ylabel("Modal Amplitude [-]")
        else:
            ax.set_ylabel("Modal Velocity [-]")

    axes[-1].set_xlabel("Time [s]")

    _tight_layout_with_title(fig, title=title)

    if show:
        plt.show()

    return fig, axes

def plot_state_envelopes(
    results,
    keys,
    *,
    transforms=None,
    labels=None,
    ylabel=None,
    title=None,
    n_bins=300,
    figsize=None,
    show=True,
    data_slice=None,
):
    plot_slice = _as_plot_slice(data_slice)
    time = _state_time(results, plot_slice)
    keys = _as_state_keys(keys)
    if figsize is None:
        figsize = _default_state_figsize(len(keys))

    fig, axes = plt.subplots(len(keys), 1, sharex=True, figsize=figsize)
    axes = _as_axes_list(axes)

    for ax, key in zip(axes, keys):
        values = _state_values(
            results,
            key,
            plot_slice,
            transforms=transforms,
        )
        centres, lower, upper, mean = _envelope_statistics(
            time,
            values,
            n_bins,
        )
        line = ax.plot(
            centres,
            mean,
            linewidth=1.6,
            label=f"{_state_label(labels, key)} mean",
        )[0]
        ax.fill_between(
            centres,
            lower,
            upper,
            color=line.get_color(),
            alpha=0.18,
            linewidth=0.0,
            label=f"{_state_label(labels, key)} min-max",
        )
        ax.axhline(0.0, color="black", linewidth=0.7, alpha=0.35)
        ax.set_ylabel(ylabel if ylabel is not None else key)
        ax.set_title(_state_label(labels, key))
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper right")

    axes[-1].set_xlabel("Time [s]")
    if title is not None:
        fig.suptitle(title)
    fig.tight_layout()

    if show:
        plt.show()

    return fig, axes

def plot_state_spectra(
    results,
    keys,
    *,
    transforms=None,
    labels=None,
    ylabel=None,
    title=None,
    max_frequency_hz=None,
    window="hann",
    demean=True,
    figsize=None,
    show=True,
    data_slice=None,
):
    plot_slice = _as_plot_slice(data_slice)
    time = _state_time(results, plot_slice)
    keys = _as_state_keys(keys)
    if figsize is None:
        figsize = _default_state_figsize(len(keys))

    fig, axes = plt.subplots(len(keys), 1, sharex=True, figsize=figsize)
    axes = _as_axes_list(axes)

    for ax, key in zip(axes, keys):
        values = _state_values(
            results,
            key,
            plot_slice,
            transforms=transforms,
        )
        frequency_hz, amplitude = _frequency_spectrum(
            time,
            values,
            window=window,
            demean=demean,
        )
        mask = np.ones_like(frequency_hz, dtype=bool)
        if max_frequency_hz is not None:
            mask = frequency_hz <= max_frequency_hz

        ax.plot(
            frequency_hz[mask],
            amplitude[mask],
            linewidth=1.0,
            label=_state_label(labels, key),
        )
        ax.set_ylabel(ylabel if ylabel is not None else "Amplitude")
        ax.set_title(_state_label(labels, key))
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper right")

    axes[-1].set_xlabel("Frequency [Hz]")
    if title is not None:
        fig.suptitle(title)
    fig.tight_layout()

    if show:
        plt.show()

    return fig, axes

def compute_nadir_angle_error(results, axis="x"):
    rG = np.column_stack((results["rG_x"], results["rG_y"]))
    rnorm = np.linalg.norm(rG, axis=1, keepdims=True)

    if np.any(rnorm == 0.0):
        raise ValueError("Zero COM radius encountered while computing nadir direction.")

    k_hat = -rG / rnorm
    alpha_nadir = np.arctan2(k_hat[:, 1], k_hat[:, 0])

    theta = results[_attitude_key(results)]

    if axis == "x":
        alpha_body = theta
    elif axis == "y":
        alpha_body = theta + np.pi / 2.0
    else:
        raise ValueError("axis must be 'x' or 'y'.")

    return wrap_to_pi(alpha_body - alpha_nadir)

def plot_nadir_angle_error(
    results,
    axis="x",
    figsize=(10, 3),
    show=True,
    data_slice=None,
    title=None,
):
    plot_slice = _as_plot_slice(data_slice)
    ts = results["time"][plot_slice]
    delta = compute_nadir_angle_error(results, axis=axis)[plot_slice]

    fig, ax = plt.subplots(1, 1, figsize=figsize)

    ax.plot(ts, np.rad2deg(delta))
    ax.set_xlabel("Time [s]")
    ax.set_ylabel(f"$\\delta_{axis}$ [deg]")
    if title is None:
        title = f"Body {axis}-axis relative to nadir"
    ax.set_title(title)
    ax.grid(True)
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))

    fig.tight_layout()

    if show:
        plt.show()

    return fig, ax

def plot_control_torques(
    results,
    figsize=(10, 5),
    show=True,
    data_slice=None,
    title=None,
):
    plot_slice = _as_plot_slice(data_slice)
    ts = results["time"][plot_slice]

    fig, axes = plt.subplots(2, 1, sharex=True, figsize=figsize)

    axes[0].plot(ts, results["tau_PD"][plot_slice])
    axes[0].set_ylabel("PD torque [N.m]")
    axes[0].legend(["tau_PD"])
    axes[0].grid(True)

    axes[1].plot(ts, results["tau_FF"][plot_slice])
    axes[1].set_ylabel("FF torque [N.m]")
    axes[1].set_xlabel("Time [s]")
    axes[1].legend(["tau_FF"])
    axes[1].grid(True)

    _tight_layout_with_title(fig, title=title)

    if show:
        plt.show()

    return fig, axes

def plot_energy_diagnostics(
    energy,
    figsize=(11, 10),
    show=True,
    data_slice=None,
):
    plot_slice = _as_plot_slice(data_slice)
    time = np.asarray(energy["time"])[plot_slice]
    kinetic = np.asarray(energy["kinetic"])[plot_slice]
    kepler_potential = np.asarray(energy["kepler_potential"])[plot_slice]
    strain_potential = np.asarray(energy["strain_potential"])[plot_slice]
    gravity_gradient_potential = np.asarray(
        energy["gravity_gradient_potential"]
    )[plot_slice]
    total_energy_relative_drift = np.asarray(
        energy["total_energy_relative_drift"]
    )[plot_slice]

    orbital_mechanical = kinetic + kepler_potential
    orbital_mechanical_drift = orbital_mechanical - orbital_mechanical[0]
    strain_gravity_potential = strain_potential + gravity_gradient_potential
    strain_gravity_potential_drift = (
        strain_gravity_potential
        - strain_gravity_potential[0]
    )

    fig, axes = plt.subplots(3, 1, figsize=figsize, sharex=True)

    axes[0].plot(time, 1e3 * strain_potential)
    axes[0].set_ylabel("Strain energy [mJ]")
    axes[0].set_title("Strain Energy")
    axes[0].grid(True)

    axes[1].plot(
        time,
        1e3 * orbital_mechanical_drift,
        label="Δ(T + V_kepler)",
    )
    axes[1].plot(
        time,
        -1e3 * strain_gravity_potential_drift,
        linestyle="--",
        label="-Δ(V_strain + V_gg)",
    )
    axes[1].set_ylabel("Energy exchange [mJ]")
    axes[1].set_title("Energy Exchange Components")
    axes[1].legend()
    axes[1].grid(True)

    axes[2].plot(time, total_energy_relative_drift)
    axes[2].set_ylabel("Relative drift [-]")
    axes[2].set_xlabel("Time [s]")
    axes[2].set_title("Total Energy Relative Drift")
    axes[2].grid(True)

    fig.tight_layout()

    if show:
        plt.show()

    return fig, axes

def plot_angular_momentum_diagnostics(
    angular_momentum,
    figsize=(10, 6),
    show=True,
    data_slice=None,
):
    plot_slice = _as_plot_slice(data_slice)
    time = np.asarray(angular_momentum["time"])[plot_slice]
    H_origin_drift = np.asarray(
        angular_momentum["H_origin_z_relative_drift"]
    )[plot_slice]
    H_cm_drift = np.asarray(
        angular_momentum["H_cm_z_relative_drift"]
    )[plot_slice]

    fig, axes = plt.subplots(2, 1, figsize=figsize, sharex=True)

    axes[0].plot(time, H_origin_drift)
    axes[0].set_ylabel("H about O scaled drift [-]")
    axes[0].grid(True)

    axes[1].plot(time, H_cm_drift)
    axes[1].set_ylabel("H about G scaled drift [-]")
    axes[1].set_xlabel("Time [s]")
    axes[1].grid(True)

    fig.tight_layout()

    if show:
        plt.show()

    return fig, axes
