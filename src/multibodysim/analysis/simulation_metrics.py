from __future__ import annotations

from typing import Any

import numpy as np
import sympy as sm


def _vector_from_matrix(components, frame):
    return (
        components[0] * frame.x
        + components[1] * frame.y
        + components[2] * frame.z
    )


def _build_angular_momentum_functions(simulator: Any, quadrature_points: int = 8):
    dyn = simulator.dynamics
    inertial_frame = dyn.frames["inertial"]
    parameters = list(dyn.parameter_symbols.values())

    rigid_H_origin = sm.S.Zero
    rigid_H_cm = sm.S.Zero

    for body in dyn.rigid_body_names:
        frame = dyn.frames[body]
        mass = dyn.mass_symbols[body]

        r = dyn.inertial_position[body].express(inertial_frame)
        v = dyn.linear_velocities[body].express(inertial_frame)

        r_rel = (dyn.inertial_position[body] - dyn.r_G).express(inertial_frame)
        v_rel = (dyn.linear_velocities[body] - dyn.v_G).express(inertial_frame)

        omega = dyn.angular_velocities[body]
        Iomega_components = dyn.inertia_matrices[body] * omega.to_matrix(frame)
        Iomega = _vector_from_matrix(Iomega_components, frame).express(inertial_frame)
        spin_z = Iomega.dot(inertial_frame.z)

        rigid_H_origin += mass * r.cross(v).dot(inertial_frame.z) + spin_z
        rigid_H_cm += mass * r_rel.cross(v_rel).dot(inertial_frame.z) + spin_z

    rigid_H_origin_func = sm.lambdify(
        (dyn.q, dyn.u, parameters),
        rigid_H_origin,
        "numpy",
        cse=True,
    )
    rigid_H_cm_func = sm.lambdify(
        (dyn.q, dyn.u, parameters),
        rigid_H_cm,
        "numpy",
        cse=True,
    )

    flexible_origin_funcs = []
    flexible_cm_funcs = []

    for body in dyn.flexible_body_names:
        mass_per_length = dyn.mass_symbols[body] / dyn.L

        dm = dyn.points[f"dm_{body}"]
        r_dm = dm.pos_from(dyn.O).express(inertial_frame)
        v_dm = dm.vel(inertial_frame).xreplace(dyn.qd_repl).express(inertial_frame)

        r_rel = (r_dm - dyn.r_G).express(inertial_frame)
        v_rel = (v_dm - dyn.v_G).express(inertial_frame)

        H_origin_density = mass_per_length * r_dm.cross(v_dm).dot(inertial_frame.z)
        H_cm_density = mass_per_length * r_rel.cross(v_rel).dot(inertial_frame.z)

        flexible_origin_funcs.append(
            sm.lambdify(
                (dyn.q, dyn.u, parameters, dyn.s),
                H_origin_density,
                "numpy",
                cse=True,
            )
        )
        flexible_cm_funcs.append(
            sm.lambdify(
                (dyn.q, dyn.u, parameters, dyn.s),
                H_cm_density,
                "numpy",
                cse=True,
            )
        )

    xi, wi = np.polynomial.legendre.leggauss(quadrature_points)
    L_value = float(dyn.parameter_values["L"])
    s_nodes = 0.5 * L_value * (xi + 1.0)
    s_weights = 0.5 * L_value * wi

    return {
        "rigid_H_origin_func": rigid_H_origin_func,
        "rigid_H_cm_func": rigid_H_cm_func,
        "flexible_origin_funcs": flexible_origin_funcs,
        "flexible_cm_funcs": flexible_cm_funcs,
        "s_nodes": s_nodes,
        "s_weights": s_weights,
    }


def compute_angular_momentum_diagnostics(
    simulator: Any,
    results: dict[str, Any],
    sample_every: int = 1,
    quadrature_points: int = 8,
):
    if sample_every < 1:
        raise ValueError("sample_every must be >= 1.")
    if quadrature_points < 1:
        raise ValueError("quadrature_points must be >= 1.")

    try:
        import pandas as pd
    except ImportError as exc:  # pragma: no cover - depends on local environment
        raise ImportError(
            "compute_angular_momentum_diagnostics requires pandas. "
            "Install the package with the dev extras or install pandas."
        ) from exc

    funcs = _build_angular_momentum_functions(
        simulator,
        quadrature_points=quadrature_points,
    )

    dyn = simulator.dynamics
    parameter_values = simulator.parameter_values

    states = np.asarray(results["states"], dtype=float)
    time = np.asarray(results["time"], dtype=float)
    state_dimension = dyn.state_dimension

    indices = np.arange(0, len(time), sample_every)
    if indices[-1] != len(time) - 1:
        indices = np.r_[indices, len(time) - 1]

    rows = []

    for index in indices:
        q = states[index, :state_dimension]
        u = states[index, state_dimension:]

        H_origin = float(
            np.asarray(funcs["rigid_H_origin_func"](q, u, parameter_values))
        )
        H_cm = float(np.asarray(funcs["rigid_H_cm_func"](q, u, parameter_values)))

        for flex_func in funcs["flexible_origin_funcs"]:
            H_origin += sum(
                weight * float(np.asarray(flex_func(q, u, parameter_values, s_value)))
                for s_value, weight in zip(funcs["s_nodes"], funcs["s_weights"])
            )

        for flex_func in funcs["flexible_cm_funcs"]:
            H_cm += sum(
                weight * float(np.asarray(flex_func(q, u, parameter_values, s_value)))
                for s_value, weight in zip(funcs["s_nodes"], funcs["s_weights"])
            )

        rows.append(
            {
                "time": time[index],
                "H_origin_z": H_origin,
                "H_cm_z": H_cm,
            }
        )

    angular_momentum = pd.DataFrame(rows)

    for key in ["H_origin_z", "H_cm_z"]:
        angular_momentum[f"{key}_drift"] = (
            angular_momentum[key] - angular_momentum[key].iloc[0]
        )
        scale = max(abs(angular_momentum[key].iloc[0]), 1.0)
        angular_momentum[f"{key}_relative_drift"] = (
            angular_momentum[f"{key}_drift"] / scale
        )

    return angular_momentum


def compute_energy_diagnostics(
    simulator: Any,
    results: dict[str, Any],
    sample_every: int = 1,
):
    if sample_every < 1:
        raise ValueError("sample_every must be >= 1.")

    try:
        import pandas as pd
    except ImportError as exc:  # pragma: no cover - depends on local environment
        raise ImportError(
            "compute_energy_diagnostics requires pandas. "
            "Install the package with the dev extras or install pandas."
        ) from exc

    dyn = simulator.dynamics
    parameter_values = simulator.parameter_values
    torques = simulator.initial_torque_values

    states = np.asarray(results["states"], dtype=float)
    time = np.asarray(results["time"], dtype=float)
    state_dimension = dyn.state_dimension

    indices = np.arange(0, len(time), sample_every)
    if indices[-1] != len(time) - 1:
        indices = np.r_[indices, len(time) - 1]

    parameters = list(dyn.parameter_symbols.values())

    V_kepler_expr = -dyn.planet_mu * dyn.total_mass / dyn.r_G_norm
    V_strain_expr = getattr(dyn, "V_strain", sm.S.Zero)
    V_gg_expr = getattr(dyn, "V_gg", sm.S.Zero)

    V_kepler_func = sm.lambdify(
        (dyn.q, dyn.u, parameters),
        V_kepler_expr,
        "numpy",
        cse=True,
    )
    V_strain_func = sm.lambdify(
        (dyn.q, dyn.u, parameters),
        V_strain_expr,
        "numpy",
        cse=True,
    )
    V_gg_func = sm.lambdify(
        (dyn.q, dyn.u, parameters),
        V_gg_expr,
        "numpy",
        cse=True,
    )

    rows = []

    for index in indices:
        q = states[index, :state_dimension]
        u = states[index, state_dimension:]

        mass_matrix, _ = dyn.eval_differentials(
            q,
            u,
            parameter_values,
            torques,
        )
        mass_matrix = np.asarray(mass_matrix, dtype=float)

        kinetic = 0.5 * float(u @ mass_matrix @ u)
        kepler = float(np.asarray(V_kepler_func(q, u, parameter_values), dtype=float))
        strain = float(np.asarray(V_strain_func(q, u, parameter_values), dtype=float))
        gravity_gradient = float(
            np.asarray(V_gg_func(q, u, parameter_values), dtype=float)
        )

        total = kinetic + kepler + strain + gravity_gradient

        rows.append(
            {
                "time": time[index],
                "kinetic": kinetic,
                "kepler_potential": kepler,
                "strain_potential": strain,
                "gravity_gradient_potential": gravity_gradient,
                "total_energy": total,
            }
        )

    energy = pd.DataFrame(rows)
    energy["total_energy_drift"] = (
        energy["total_energy"] - energy["total_energy"].iloc[0]
    )

    scale = max(abs(energy["total_energy"].iloc[0]), 1.0)
    energy["total_energy_relative_drift"] = (
        energy["total_energy_drift"] / scale
    )

    return energy


def simulation_diagnostics(results: dict[str, Any]) -> dict[str, Any]:
    central_angle_deg = np.rad2deg(np.asarray(results["q_central_angle"], dtype=float))
    central_angle_speed_deg_s = np.rad2deg(
        np.asarray(results["u_central_angle"], dtype=float)
    )

    metrics: dict[str, Any] = {
        "success": bool(results["success"]),
        "nfev": results["nfev"],
        "njev": results.get("njev"),
        "nlu": results.get("nlu"),
        "central_angle_final_deg": float(central_angle_deg[-1]),
        "central_angle_drift_deg": float(central_angle_deg[-1] - central_angle_deg[0]),
        "central_angle_peak_to_peak_deg": float(np.ptp(central_angle_deg)),
        "central_angle_rms_about_mean_deg": float(
            np.sqrt(np.mean((central_angle_deg - np.mean(central_angle_deg)) ** 2))
        ),
        "central_angle_speed_peak_abs_deg_s": float(
            np.max(np.abs(central_angle_speed_deg_s))
        ),
        "central_angle_speed_rms_deg_s": float(
            np.sqrt(np.mean(central_angle_speed_deg_s**2))
        ),
    }

    for key in sorted(results):
        if key.startswith(("eta", "zeta")):
            values = np.asarray(results[key], dtype=float)
            metrics[f"{key}_peak_abs"] = float(np.max(np.abs(values)))
            metrics[f"{key}_rms"] = float(np.sqrt(np.mean(values**2)))
            metrics[f"{key}_final_abs"] = float(abs(values[-1]))

    return metrics


def simulation_diagnostics_table(
    metrics: dict[str, Any],
) -> tuple[list[tuple[str, str, Any]], list[str]]:
    labels = {
        "success": ("Solver success", "-"),
        "nfev": ("Function evaluations", "-"),
        "njev": ("Jacobian evaluations", "-"),
        "nlu": ("LU decompositions", "-"),
        "central_angle_final_deg": ("Final attitude", "deg"),
        "central_angle_drift_deg": ("Attitude drift", "deg"),
        "central_angle_peak_to_peak_deg": ("Attitude peak-to-peak", "deg"),
        "central_angle_rms_about_mean_deg": ("Attitude RMS about mean", "deg"),
        "central_angle_speed_peak_abs_deg_s": ("Peak angular velocity", "deg/s"),
        "central_angle_speed_rms_deg_s": ("RMS angular velocity", "deg/s"),
    }

    rows = []
    for key, value in metrics.items():
        metric, unit = labels.get(key, (key, "-"))
        rows.append((metric, unit, value))

    return rows, ["Metric", "Unit", "Value"]
