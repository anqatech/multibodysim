import numpy as np
from scipy.integrate import solve_ivp
from .flexible_ns_dynamics import FlexibleNonSymmetricDynamics
from ..inputshaping.zv_input_shaping import InputShaper


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

        # --------- Input Shaping settings ---------
        self.use_input_shaping = False
        self.shaper = None
        self.manoeuvre_start_time = None
        self.theta_start = None
        self.theta_final = None
        self.delta_theta = None
        self.Tr = 0.0
        
        # ---------- Initialize results storage ---------- 
        self.results = None

    def set_attitude_manoeuver(self, theta_target, theta_dot_target, Kp, Kd, Tr, omega, zeta, shaping_flag):
        self.theta_target      = theta_target
        self.theta_dot_target  = theta_dot_target
        self.Kp                = Kp
        self.Kd                = Kd
        self.use_attitude_pd   = True

        self.set_input_shaping(omega, zeta, Tr, shaping_flag)
    
    def smooth_step_5th_order(self, t, Tr):
        # return foes from 0 to 1 with zero velocity and zero acceleration at endpoints
        if t <= 0.0: return 0.0
        if t >= Tr:  return 1.0
        s = t/Tr
        return 10*s**3 - 15*s**4 + 6*s**5

    def derivative_smooth_step_5th_order(self, t, Tr):
        # derivative of smooth_step_5th_order
        if t <= 0.0: return 0.0
        if t >= Tr:  return 0.0
        s = t/Tr
        return (30*s**2 - 60*s**3 + 30*s**4) / Tr
    
    def set_input_shaping(self, omega, zeta, Tr, shaping_flag):
        self.shaper = InputShaper.zvd(omega=omega, zeta=zeta)
        self.Tr = float(Tr)
        self.use_input_shaping = shaping_flag

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

            # Initialize maneuver start on first call
            if self.manoeuvre_start_time is None:
                self.manoeuvre_start_time = t
                self.theta_start = theta
                self.theta_final = self.theta_target
                self.delta_theta = self.theta_final - self.theta_start

            def raw_theta_command(time_command):
                tau = time_command - self.manoeuvre_start_time
                return self.theta_start + self.delta_theta * self.smooth_step_5th_order(tau, self.Tr)

            def raw_theta_dot_command(time_command):
                tau = time_command - self.manoeuvre_start_time
                return self.delta_theta * self.derivative_smooth_step_5th_order(tau, self.Tr)

            if self.use_input_shaping and (self.shaper is not None):
                theta_ref = self.shaper.shape(t, raw_theta_command)
                theta_dot_ref = self.shaper.shape(t, raw_theta_dot_command)
            else:
                theta_ref = raw_theta_command(t)
                theta_dot_ref = raw_theta_dot_command(t)

            # Errors for PD law
            err     = theta_ref     - theta
            err_dot = theta_dot_ref - theta_dot

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

    def run_simulation(self, eval_flag):
        # ---------- Get initial conditions ---------- 
        x0 = self.setup_initial_conditions()
        
        # ---------- Extract simulation parameters ---------- 
        sim_params = self.config["sim_parameters"]
        t_start = sim_params["t_start"]
        t_end = sim_params["t_end"]
        nb_timesteps = sim_params["nb_timesteps"]
        
        # ---------- Create time evaluation points ---------- 
        t_eval = np.linspace(t_start, t_end, nb_timesteps)
        
# ----------------------------------------------------------------------------------------------------------------
        # ---------- Integration settings ---------- 
        atol = []

        # --- q states ---
        for q_sym in self.dynamics.q:
            name = q_sym.name

            if name in ("q1", "q2"):
                atol.append(1e-2)
            elif name == "q3":
                atol.append(1e-8)
            elif name.startswith("eta"):
                atol.append(1e-6)
            else:
                atol.append(1e-6)

        # --- u states ---
        for u_sym in self.dynamics.u:
            name = u_sym.name

            if name in ("u1", "u2"):
                atol.append(1e-3)
            elif name == "u3":
                atol.append(1e-9)
            elif name.startswith("zeta"):
                atol.append(1e-6)
            else:
                atol.append(1e-6)

        atol = np.array(atol)

        integration_options = {
            "rtol": sim_params.get("rtol", 1e-6),
            "atol": atol,
            "method": sim_params.get("method", "Radau"),
            # "max_step": sim_params.get("max_step", 5e-4)
        }
        
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
# ----------------------------------------------------------------------------------------------------------------

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
        tau_pd = np.zeros_like(ts)
        rG_x = np.zeros_like(ts)
        rG_y = np.zeros_like(ts)
        vG_x = np.zeros_like(ts)
        vG_y = np.zeros_like(ts)
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

            # --- COM position/velocity diagnostics ---
            xG, yG = self.dynamics.rG_func(qk, uk, self.p_vals)
            vxG, vyG = self.dynamics.vG_func(qk, uk, self.p_vals)

            rG_x[k] = float(xG)
            rG_y[k] = float(yG)
            vG_x[k] = float(vxG)
            vG_y[k] = float(vyG)

        self.results["J_eff"] = J_eff
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
