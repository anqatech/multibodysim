import numpy as np
from scipy.integrate import solve_ivp
from .flexible_ns_dynamics import FlexibleNonSymmetricDynamics


class FlexibleNonSymmetricSimulator:
    def __init__(self, config):
        self.config = config
        
        # ---------- Create symbolic dynamics model ---------- 
        self.dynamics = FlexibleNonSymmetricDynamics(config)
        
        # ---------- Extract parameter values ---------- 
        # self.p_vals = self.dynamics.get_parameter_values()
        self.p_vals = np.array(list(self.dynamics.get_parameter_values()), dtype=float)

        # index of tau in the parameter vector
        self.p_keys = list(self.config["p_values"].keys())
        try:
            self.tau_index = self.p_keys.index("tau")
        except ValueError:
            raise KeyError("Config p_values must contain tau entry for the bus torque.")

        # --------- PD attitude control settings ---------
        self.use_attitude_pd    = False        # flag
        self.theta_target       = 0.0          # desired bus angle [rad]
        self.theta_dot_target   = 0.0          # desired bus angular velocity [rad/s]
        self.Kp                 = 0.0          # Proportional gain
        self.Kd                 = 0.0          # Derivative gain
        
        # ---------- Initialize results storage ---------- 
        self.results = None

    def set_attitude_manoeuver(self, theta_target, theta_dot_target, Kp, Kd):
        self.theta_target      = theta_target
        self.theta_dot_target  = theta_dot_target
        self.Kp                = Kp
        self.Kd                = Kd
        self.use_attitude_pd   = True


    def eval_rhs(self, t, x):
        q = x[:self.dynamics.state_dimension]
        u = x[self.dynamics.state_dimension:]

        # ---- compute PD torque if enabled ----
        if self.use_attitude_pd:
            # identify indices of the bus attitude and angular_velocity
            q_ref = list(self.dynamics.q_reference.keys())
            u_ref = list(self.dynamics.u_reference.keys())

            i_theta_q_index = q_ref.index("theta")
            i_theta_u_index = u_ref.index("theta")

            theta     = q[i_theta_q_index]
            theta_dot = u[i_theta_u_index]

            # Errors for PD law
            err     = self.theta_target - theta
            err_dot = self.theta_dot_target - theta_dot

            tau_pd  = self.Kp * err + self.Kd * err_dot

            # overwrite the tau parameter entry
            self.p_vals[self.tau_index] = tau_pd

        try:
            # ---------- Evaluate kinematic equations ---------- 
            Mk, gk = self.dynamics.eval_kinematics(q, u, self.p_vals)
            qd = -np.linalg.solve(Mk, np.squeeze(gk))

            # ---------- Evaluate dynamic equations ---------- 
            Md, gd = self.dynamics.eval_differentials(q, u, self.p_vals)
            ud = -np.linalg.solve(Md, np.squeeze(gd))
            
        except np.linalg.LinAlgError:
            print("Singular matrix encountered.")
            qd = np.zeros_like(q)
            ud = np.zeros_like(u)
        
        return np.hstack((qd, ud))

    def setup_initial_conditions(self):
        return self.dynamics.get_initial_conditions()

    def run_simulation(self):
        # ---------- Get initial conditions ---------- 
        x0 = self.setup_initial_conditions()
        
        # ---------- Extract simulation parameters ---------- 
        sim_params = self.config["sim_parameters"]
        t_start = sim_params["t_start"]
        t_end = sim_params["t_end"]
        nb_timesteps = sim_params["nb_timesteps"]
        
        # ---------- Create time evaluation points ---------- 
        t_eval = np.linspace(t_start, t_end, nb_timesteps)
        
        # ---------- Integration settings ---------- 
        integration_options = {
            "rtol": sim_params.get("rtol", 1e-6),
            "atol": sim_params.get("atol", 1e-9),
            "method": sim_params.get("method", "Radau")
        }
        
        print(f"Starting simulation from t={t_start} to t={t_end}")
        print(f"Integration method: {integration_options["method"]}")
        print(f"Tolerances: rtol={integration_options["rtol"]}, atol={integration_options["atol"]}\n")
    
        # ---------- Integrate equations of motion ---------- 
        result = solve_ivp(
            fun=self.eval_rhs,
            t_span=(t_start, t_end),
            y0=x0,
            t_eval=t_eval,
            **integration_options
        )

        # ---------- Process results ----------
        xs = np.transpose(result.y)
        ts = result.t
        
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
        tau_pd = np.zeros_like(ts)
        for k, (qk, uk) in enumerate(zip(q, u)):
            # Evaluate Kane's dynamic mass matrix at this state
            Md, gd = self.dynamics.eval_differentials(qk, uk, self.p_vals)
            Md = np.asarray(Md, dtype=float)

            # Effective inertia for the attitude DOF
            J_eff[k] = np.abs(Md[theta_index, theta_index])

            if self.use_attitude_pd:
                theta = qk[theta_index]
                theta_dot = uk[theta_index]

                err     = self.theta_target - theta
                err_dot = self.theta_dot_target - theta_dot

                tau_pd[k]  = self.Kp * err + self.Kd * err_dot
            else:
                tau_pd[k]  = self.p_vals[self.tau_index]

        self.results["J_eff"] = J_eff
        self.results["tau_PD"] = tau_pd

        return self.results

    def get_results(self):
        if self.results is None:
            raise ValueError("Simulation has not been run yet. Call run_simulation() first.")
        return self.results
