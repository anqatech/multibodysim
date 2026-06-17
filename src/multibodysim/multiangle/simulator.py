from __future__ import annotations

import numpy as np
from scipy.integrate import solve_ivp

from ..codegen import prepare_autowrap_evaluators
from ..controllers.base import AttitudeController, ControlOutput
from ..controllers.plant_view import MultiAnglePlantView
from .dynamics import MultiAngleFlexibleDynamics


class MultiAngleFlexibleSimulator:
    DEFAULT_ABSOLUTE_TOLERANCES = {
        "q1": 1e-2,
        "q2": 1e-2,
        "q_central_angle": 1e-7,
        "eta": 1e-6,
        "u1": 1e-3,
        "u2": 1e-3,
        "u_central_angle": 1e-8,
        "zeta": 1e-6,
        "q_default": 1e-6,
        "u_default": 1e-6,
    }

    def __init__(self, config: dict):
        self.config = config
        self.dynamics = MultiAngleFlexibleDynamics(config)
        self.codegen_metadata = prepare_autowrap_evaluators(self.dynamics)

        self.parameter_values = np.array(
            self.dynamics.get_parameter_values(),
            dtype=float,
        )
        self.zero_torque_values = np.zeros(
            len(self.dynamics.rigid_body_names),
            dtype=float,
        )
        self.torque_values = self.zero_torque_values.copy()
        self.torque_weights = np.array(
            self.dynamics.get_torque_weights(),
            dtype=float,
        )
        self.plant_view = MultiAnglePlantView(self.dynamics)

        self.controller = None
        self.results = None

    def get_parameter_values(self) -> np.ndarray:
        return self.parameter_values.copy()

    def get_torque_values(self) -> np.ndarray:
        return self.torque_values.copy()

    def reset_torque_values(self):
        self.torque_values = self.zero_torque_values.copy()

    def set_controller(self, controller: AttitudeController | None):
        self.controller = controller

    def get_control_output(
        self,
        t: float,
        q: np.ndarray,
        u: np.ndarray,
        mass_matrix: np.ndarray | None = None,
    ) -> ControlOutput:
        if self.controller is None:
            return ControlOutput()

        return self.controller.compute(t, q, u, Md=mass_matrix)

    def setup_initial_conditions(self, verbose: bool = True) -> np.ndarray:
        return self.dynamics.get_initial_conditions(verbose=verbose)

    def evaluate_rhs(
        self,
        t: float,
        state: np.ndarray,
        torque_values: np.ndarray | None = None,
    ) -> np.ndarray:
        state = np.asarray(state, dtype=float)
        state_dimension = self.dynamics.state_dimension
        q = state[:state_dimension]
        u = state[state_dimension:]
        torques = self.zero_torque_values.copy()
        if torque_values is not None:
            torques = np.asarray(torque_values, dtype=float).copy()

        Mk, gk = self.dynamics._eval_kinematics(
            q,
            u,
            torques,
        )
        qd = -np.linalg.solve(
            np.asarray(Mk, dtype=float),
            np.asarray(gk, dtype=float).squeeze(),
        )

        mass_matrix, forcing = self.dynamics._eval_differentials(
            q,
            u,
            torques,
        )

        if torque_values is None and self.controller is not None:
            control_output = self.get_control_output(
                t,
                q,
                u,
                mass_matrix=np.asarray(mass_matrix, dtype=float),
            )
            if control_output.bus_torques is not None:
                if control_output.tau_total != 0.0:
                    raise ValueError(
                        "ControlOutput cannot provide both direct bus_torques "
                        "and a non-zero scalar torque command."
                    )
                torques = self._validated_bus_torques(
                    control_output.bus_torques,
                )
                mass_matrix, forcing = self.dynamics._eval_differentials(
                    q,
                    u,
                    torques,
                )
            elif control_output.tau_total != 0.0:
                torques = torques + control_output.tau_total * self.torque_weights
                mass_matrix, forcing = self.dynamics._eval_differentials(
                    q,
                    u,
                    torques,
                )

        self.torque_values = torques.copy()

        ud = np.linalg.solve(
            np.asarray(mass_matrix, dtype=float),
            np.asarray(forcing, dtype=float).squeeze(),
        )

        return np.hstack((qd, ud))

    def _validated_bus_torques(self, bus_torques) -> np.ndarray:
        torques = np.asarray(bus_torques, dtype=float).reshape(-1)
        expected_size = self.zero_torque_values.size
        if torques.size != expected_size:
            raise ValueError(
                "bus_torques must contain "
                f"{expected_size} values; got {torques.size}."
            )
        if not np.all(np.isfinite(torques)):
            raise ValueError("bus_torques must contain only finite values.")
        return torques.copy()

    def _absolute_tolerance_for_name(
        self,
        name: str,
        tolerance_map: dict,
        default_key: str,
    ) -> float:
        if name in tolerance_map:
            return tolerance_map[name]

        if name.startswith("q_relative_angle_"):
            return tolerance_map["q_central_angle"]

        if name.startswith("u_relative_angle_"):
            return tolerance_map["u_central_angle"]

        if name.startswith("eta"):
            return tolerance_map["eta"]

        if name.startswith("zeta"):
            return tolerance_map["zeta"]

        return tolerance_map[default_key]

    def _build_absolute_tolerances(self, sim_params: dict) -> np.ndarray:
        tolerance_map = self.DEFAULT_ABSOLUTE_TOLERANCES.copy()
        tolerance_map.update(sim_params.get("state_atol", {}))

        atol = []

        for q_sym in self.dynamics.q:
            atol.append(
                self._absolute_tolerance_for_name(
                    q_sym.name,
                    tolerance_map,
                    "q_default",
                )
            )

        for u_sym in self.dynamics.u:
            atol.append(
                self._absolute_tolerance_for_name(
                    u_sym.name,
                    tolerance_map,
                    "u_default",
                )
            )

        return np.array(atol, dtype=float)

    def run_simulation(
        self,
        eval_flag: bool = True,
        verbose: bool = True,
    ) -> dict:
        initial_conditions = self.setup_initial_conditions(verbose=verbose)
        sim_params = self.config["sim_parameters"]

        t_start = sim_params["t_start"]
        t_end = sim_params["t_end"]
        nb_timesteps = sim_params["nb_timesteps"]
        t_eval = np.linspace(t_start, t_end, nb_timesteps)

        integration_options = {
            "method": sim_params.get("method", "Radau"),
            "rtol": sim_params.get("rtol", 1e-5),
            "atol": self._build_absolute_tolerances(sim_params),
        }

        if verbose:
            print(f"Starting simulation from t={t_start} to t={t_end}")
            print(f"Integration method: {integration_options['method']}")
            print(
                "Tolerances: "
                f"rtol={integration_options['rtol']}, "
                f"atol={integration_options['atol']}\n"
            )

        result = solve_ivp(
            fun=self.evaluate_rhs,
            t_span=(t_start, t_end),
            y0=initial_conditions,
            t_eval=t_eval if eval_flag else None,
            dense_output=not eval_flag,
            **integration_options,
        )

        states = result.y.T
        times = result.t
        state_dimension = self.dynamics.state_dimension

        self.results = {
            "time": times,
            "states": states,
            "success": result.success,
            "message": result.message,
            "nfev": result.nfev,
            "njev": getattr(result, "njev", None),
            "nlu": getattr(result, "nlu", None),
            "config": self.config.copy(),
            "sim_object": result,
        }

        for index, q_sym in enumerate(self.dynamics.q):
            self.results[q_sym.name] = states[:, index]

        for index, u_sym in enumerate(self.dynamics.u):
            self.results[u_sym.name] = states[:, state_dimension + index]

        self._add_diagnostics_to_results(times, states)

        if verbose:
            print(f"Simulation completed: {result.success}")
            print(f"Message: {result.message}")
            print(f"Number of function evaluations: {result.nfev}\n")

        return self.results

    def _add_diagnostics_to_results(
        self,
        times: np.ndarray,
        states: np.ndarray,
    ) -> None:
        state_dimension = self.dynamics.state_dimension
        q = states[:, :state_dimension]
        u = states[:, state_dimension:]

        theta_index = self.plant_view.i_theta_u
        J_eff = np.zeros_like(times)
        tau_ff = np.zeros_like(times)
        tau_reference_ff = np.zeros_like(times)
        tau_gravity_gradient_ff = np.zeros_like(times)
        tau_pd = np.zeros_like(times)
        rG_x = np.zeros_like(times)
        rG_y = np.zeros_like(times)
        rG_z = np.zeros_like(times)
        vG_x = np.zeros_like(times)
        vG_y = np.zeros_like(times)
        vG_z = np.zeros_like(times)

        for index, (time, qk, uk) in enumerate(zip(times, q, u)):
            mass_matrix, _ = self.dynamics._eval_differentials(
                qk,
                uk,
                self.zero_torque_values,
            )
            mass_matrix = np.asarray(mass_matrix, dtype=float)
            J_eff[index] = abs(mass_matrix[theta_index, theta_index])

            control_output = self.get_control_output(
                time,
                qk,
                uk,
                mass_matrix=mass_matrix,
            )
            tau_ff[index] = control_output.tau_ff
            tau_reference_ff[index] = (
                control_output.tau_reference_ff
            )
            tau_gravity_gradient_ff[index] = (
                control_output.tau_gravity_gradient_ff
            )
            tau_pd[index] = control_output.tau_fb

            rG = np.asarray(
                self.dynamics.rG_func(qk, uk),
                dtype=float,
            ).reshape(-1)
            vG = np.asarray(
                self.dynamics.vG_func(qk, uk),
                dtype=float,
            ).reshape(-1)

            rG_x[index], rG_y[index], rG_z[index] = rG[:3]
            vG_x[index], vG_y[index], vG_z[index] = vG[:3]

        self.results["J_eff"] = J_eff
        self.results["tau_FF"] = tau_ff
        self.results["tau_reference_FF"] = tau_reference_ff
        self.results["tau_GG_FF"] = tau_gravity_gradient_ff
        self.results["tau_PD"] = tau_pd
        self.results["rG_x"] = rG_x
        self.results["rG_y"] = rG_y
        self.results["rG_z"] = rG_z
        self.results["vG_x"] = vG_x
        self.results["vG_y"] = vG_y
        self.results["vG_z"] = vG_z

    def get_results(self):
        if self.results is None:
            raise ValueError("Simulation has not been run yet.")
        return self.results
