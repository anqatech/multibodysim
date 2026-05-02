import numpy as np
from scipy.integrate import solve_ivp
from .flexible_ns_dynamics import FlexibleNonSymmetricDynamics
from ..controllers.plant_view import FlexibleNSPlantView
from ..controllers.base import AttitudeController, ControlOutput


class FlexibleNonSymmetricSimulator:
    DEFAULT_ABSOLUTE_TOLERANCES = {
        "q1": 1e-2,
        "q2": 1e-2,
        "q3": 1e-8,
        "eta": 1e-6,
        "u1": 1e-3,
        "u2": 1e-3,
        "u3": 1e-9,
        "zeta": 1e-6,
        "q_default": 1e-6,
        "u_default": 1e-6,
    }

    def __init__(self, config):
        self.config = config
        
        # ---------- Create symbolic dynamics model ---------- 
        self.dynamics = FlexibleNonSymmetricDynamics(config)
        
        # ---------- Extract parameter values ---------- 
        self.p_vals = np.array(list(self.dynamics.get_parameter_values()), dtype=float)

        # ---------- Extract torque values (one scalar per rigid body) ----------
        self.torque_intial_vals = np.array(list(self.dynamics.get_torque_values()), dtype=float)
        self.torque_vals = self.torque_intial_vals.copy()
        self.torque_weights = np.array(list(self.dynamics.get_torque_weights()), dtype=float)

        # ---------- Plant view for controllers ----------
        self.plant_view = FlexibleNSPlantView(self.dynamics, self.p_vals)

        # ------------- Initialisations -------------
        self.results = None
        self.controller = None

    def set_controller(self, controller: AttitudeController | None):
        self.controller = controller

    def get_control_output(self, t, q, u, Md=None) -> ControlOutput:
        if self.controller is not None:
            return self.controller.compute(t, q, u, Md=Md)

        return ControlOutput()

    def eval_rhs(self, t, x):
        q = x[:self.dynamics.state_dimension]
        u = x[self.dynamics.state_dimension:]

        try:
            # ---------- Evaluate kinematic equations ---------- 
            Mk, gk = self.dynamics.eval_kinematics(q, u, self.p_vals, self.torque_vals)
            qd = -np.linalg.solve(Mk, np.squeeze(gk))

            # ---------- Evaluate dynamic equations ---------- 
            Md, gd = self.dynamics.eval_differentials(q, u, self.p_vals, self.torque_vals)

            # reset to perturbation torques each RHS call
            self.torque_vals = self.torque_intial_vals.copy()

            ctrl_out = self.get_control_output(t, q, u, Md=Md)
            tau_ctrl = ctrl_out.tau_total

            if tau_ctrl != 0.0:
                # distribute control torque using weights
                self.torque_vals += tau_ctrl * self.torque_weights

                # IMPORTANT: recompute gd after updating torque values
                Md, gd = self.dynamics.eval_differentials(q, u, self.p_vals, self.torque_vals)
    
            ud = -np.linalg.solve(Md, np.squeeze(gd))
            
        except np.linalg.LinAlgError:
            print("Singular matrix encountered.")
            qd = np.zeros_like(q)
            ud = np.zeros_like(u)
        
        return np.hstack((qd, ud))

    def setup_initial_conditions(self, verbose=True):
        return self.dynamics.get_initial_conditions(verbose=verbose)

    def _absolute_tolerance_for_name(self, name, tolerance_map, default_key):
        if name in tolerance_map:
            return tolerance_map[name]

        if name.startswith("eta"):
            return tolerance_map["eta"]

        if name.startswith("zeta"):
            return tolerance_map["zeta"]

        return tolerance_map[default_key]

    def _build_absolute_tolerances(self, sim_params):
        tolerance_map = self.DEFAULT_ABSOLUTE_TOLERANCES.copy()

        tolerance_map.update(sim_params.get("state_atol", {}))

        atol = []

        for q_sym in self.dynamics.q:
            name = q_sym.name
            atol.append(self._absolute_tolerance_for_name(name, tolerance_map, "q_default"))

        for u_sym in self.dynamics.u:
            name = u_sym.name
            atol.append(self._absolute_tolerance_for_name(name, tolerance_map, "u_default"))

        return np.array(atol, dtype=float)

    def run_simulation(self, eval_flag, verbose=True):
        # ---------- Get initial conditions ---------- 
        x0 = self.setup_initial_conditions(verbose=verbose)
        
        # ---------- Extract simulation parameters ---------- 
        sim_params = self.config["sim_parameters"]
        t_start = sim_params["t_start"]
        t_end = sim_params["t_end"]
        nb_timesteps = sim_params["nb_timesteps"]
        
        # ---------- Create time evaluation points ---------- 
        t_eval = np.linspace(t_start, t_end, nb_timesteps)
        
        # ---------- Integration settings ---------- 
        atol = self._build_absolute_tolerances(sim_params)

        integration_options = {
            "rtol": sim_params.get("rtol", 1e-5),
            "atol": atol,
            "method": sim_params.get("method", "Radau"),
            # "max_step": sim_params.get("max_step", 5e-4)
        }
        
        if verbose:
            print(f"Starting simulation from t={t_start} to t={t_end}")
            print(f"Integration method: {integration_options["method"]}")
            print(f"Tolerances: rtol={integration_options["rtol"]}, atol={integration_options["atol"]}\n")

        # ---------- Integrate equations of motion ---------- 
        result = solve_ivp(
            fun=self.eval_rhs,
            t_span=(t_start, t_end),
            y0=x0,
            t_eval=t_eval if eval_flag else None,
            dense_output=not eval_flag,
            **integration_options
        )

        # ---------- Process results ----------
        xs = np.transpose(result.y)
        ts = result.t
        
        if verbose:
            print(f"Simulation completed: {result.success}")
            print(f"Message: {result.message}")
            print(f"Number of function evaluations: {result.nfev}\n")
        
        def _name(sym):
            return str(getattr(sym, "func", sym))
        
        q_syms = list(self.dynamics.q)
        u_syms = list(self.dynamics.u)
        q_names = [_name(s) for s in q_syms]
        u_names = [_name(s) for s in u_syms]
        
        nq = len(q_syms)
        nu = len(u_syms)
        
        self.results = {
            "time": ts,
            "states": xs,
            "success": result.success,
            "message": result.message,
            "nfev": result.nfev,
            "njev": result.njev if hasattr(result, "njev") else None,
            "nlu": result.nlu if hasattr(result, "nlu") else None,
            "config": self.config.copy(),
            "sim_object": result
        }
        
        # ---------- Save all generalized coordinates with their real names ----------
        for i, name in enumerate(q_names):
            self.results[name] = xs[:, i]
        
        # ---------- Save all generalized speeds with their real names ----------
        for i, name in enumerate(u_names):
            self.results[name] = xs[:, nq + i]

        q = xs[:, :self.dynamics.state_dimension]
        u = xs[:, self.dynamics.state_dimension:]

        u_ref = list(self.dynamics.u_reference.keys())
        theta_index = u_ref.index("theta")

        J_eff = np.zeros_like(ts)
        tau_ff = np.zeros_like(ts)
        tau_pd = np.zeros_like(ts)
        rG_x = np.zeros_like(ts)
        rG_y = np.zeros_like(ts)
        vG_x = np.zeros_like(ts)
        vG_y = np.zeros_like(ts)
        for k, (tk, qk, uk) in enumerate(zip(ts, q, u)):
            # Evaluate Kane's dynamic mass matrix at this state
            Md, gd = self.dynamics.eval_differentials(qk, uk, self.p_vals, self.torque_vals)
            Md = np.asarray(Md, dtype=float)

            # Effective inertia for the attitude DOF
            J_eff[k] = np.abs(Md[theta_index, theta_index])
            ctrl_out = self.get_control_output(tk, qk, uk, Md=Md)
            tau_ff[k] = ctrl_out.tau_ff
            tau_pd[k] = ctrl_out.tau_fb

            # --- COM position/velocity diagnostics ---
            xG, yG = self.dynamics.rG_func(qk, uk, self.p_vals)
            vxG, vyG = self.dynamics.vG_func(qk, uk, self.p_vals)

            rG_x[k] = float(xG)
            rG_y[k] = float(yG)
            vG_x[k] = float(vxG)
            vG_y[k] = float(vyG)

        self.results["J_eff"] = J_eff
        self.results["tau_FF"] = tau_ff
        self.results["tau_PD"] = tau_pd
        self.results["rG_x"] = np.array(rG_x)
        self.results["rG_y"] = np.array(rG_y)
        self.results["vG_x"] = np.array(vG_x)
        self.results["vG_y"] = np.array(vG_y)

        return self.results

    def get_results(self):
        if self.results is None:
            raise ValueError("Simulation has not been run yet. Call run_simulation() first.")
        return self.results
