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
        
# ----------------------------------------------------------------------------------------------------
        # ---- Nadir pointing control (LVLH tracking) ----
        self.use_nadir_pd = False
        self.Kp_nadir = 0.0
        self.Kd_nadir = 0.0
# ----------------------------------------------------------------------------------------------------

        # ---------- Initialize results storage ---------- 
        self.results = None

    def set_attitude_manoeuver(self, theta_target, theta_dot_target, Kp, Kd, Tr, omega, zeta, shaping_flag):
        self.theta_target      = theta_target
        self.theta_dot_target  = theta_dot_target
        self.Kp                = Kp
        self.Kd                = Kd
        self.use_attitude_pd   = True

        self.set_input_shaping(omega, zeta, Tr, shaping_flag)

# ----------------------------------------------------------------------------------------------------
    def set_nadir_pointing(self, Kp, Kd):
        self.Kp_nadir = float(Kp)
        self.Kd_nadir = float(Kd)

        self.use_nadir_pd = True
        self.use_attitude_pd = False      # avoid conflicting controllers
        self.use_input_shaping = False    # recommended for tracking

    @staticmethod
    def wrap_to_domain_minus_pi_pi(angle):
        return (angle + np.pi) % (2*np.pi) - np.pi
# ----------------------------------------------------------------------------------------------------
    
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
    
    def second_derivative_smooth_step_5th_order(self, t, Tr):
        if t <= 0.0 or t >= Tr:
            return 0.0
        s = t / Tr
        return (60*s - 180*s**2 + 120*s**3) / (Tr**2)

    def set_input_shaping(self, omega, zeta, Tr, shaping_flag):
        self.shaper = InputShaper.zvd(omega=omega, zeta=zeta)
        self.Tr = float(Tr)
        self.use_input_shaping = shaping_flag

    def compute_control_tau(self, t, q, u, Md=None):
        tau_ff = 0.0
        tau_pd = self.p_vals[self.tau_index]

        # ---- attitude PD ----
        if self.use_attitude_pd:
            q_ref = list(self.dynamics.q_reference.keys())
            u_ref = list(self.dynamics.u_reference.keys())
            i_theta_q_index = q_ref.index("theta")
            i_theta_u_index = u_ref.index("theta")

            theta     = q[i_theta_q_index]
            theta_dot = u[i_theta_u_index]

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
                theta_ref     = self.shaper.shape(t, raw_theta_command)
                theta_dot_ref = self.shaper.shape(t, raw_theta_dot_command)
            else:
                theta_ref     = raw_theta_command(t)
                theta_dot_ref = raw_theta_dot_command(t)

            err     = theta_ref     - theta
            err_dot = theta_dot_ref - theta_dot

            tau_ff = 0.0
            tau_pd = self.Kp * err + self.Kd * err_dot

        # ---- nadir PD (optionally overrides attitude PD if both enabled) ----
        if self.use_nadir_pd:
            q_ref = list(self.dynamics.q_reference.keys())
            u_ref = list(self.dynamics.u_reference.keys())
            i_theta_q_index = q_ref.index("theta")
            i_theta_u_index = u_ref.index("theta")

            theta     = q[i_theta_q_index]
            theta_dot = u[i_theta_u_index]

            if Md is None:
                raise ValueError("Nadir PD requested but Md not provided to compute_control_tau().")

            J_instant = float(Md[i_theta_u_index, i_theta_u_index])

            rGx, rGy = self.dynamics.rG_func(q, u, self.p_vals)
            vGx, vGy = self.dynamics.vG_func(q, u, self.p_vals)

            theta_k   = np.arctan2(-rGy, -rGx)
            theta_ref = theta_k - 0.5*np.pi

            h  = rGx*vGy - rGy*vGx
            r2 = rGx*rGx + rGy*rGy

            theta_dot_ref = h / r2
            r    = np.sqrt(r2)
            rdot = (rGx*vGx + rGy*vGy) / r
            theta_ddot_ref = -2.0 * h * rdot / (r**3)

            err     = self.wrap_to_domain_minus_pi_pi(theta_ref - theta)
            err_dot = theta_dot_ref - theta_dot

            tau_ff = J_instant * theta_ddot_ref
            tau_pd = self.Kp_nadir * err + self.Kd_nadir * err_dot

        def soft_cap(x, x_max=1e-4):
            return x_max * np.tanh(x / x_max)
        
        # tau_ff = soft_cap(tau_ff, 1e-8)
        # tau_pd = soft_cap(tau_pd, 5e-4)
        tau_ff = 0.0

        return tau_ff, tau_pd

    def eval_rhs(self, t, x):
        q = x[:self.dynamics.state_dimension]
        u = x[self.dynamics.state_dimension:]

        try:
            # ---------- Evaluate kinematic equations ---------- 
            Mk, gk = self.dynamics.eval_kinematics(q, u, self.p_vals)
            qd = -np.linalg.solve(Mk, np.squeeze(gk))

            # ---------- Evaluate dynamic equations ---------- 
            Md, gd = self.dynamics.eval_differentials(q, u, self.p_vals)

            if self.use_attitude_pd or self.use_nadir_pd:
                tau_ff, tau_pd = self.compute_control_tau(t, q, u, Md=Md)
                self.p_vals[self.tau_index] = tau_ff + tau_pd

                # IMPORTANT: recompute gd after updating tau
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
        tau_ff = np.zeros_like(ts)
        tau_pd = np.zeros_like(ts)
        rG_x = np.zeros_like(ts)
        rG_y = np.zeros_like(ts)
        vG_x = np.zeros_like(ts)
        vG_y = np.zeros_like(ts)
        for k, (tk, qk, uk) in enumerate(zip(ts, q, u)):
            # Evaluate Kane's dynamic mass matrix at this state
            Md, gd = self.dynamics.eval_differentials(qk, uk, self.p_vals)
            Md = np.asarray(Md, dtype=float)

            # Effective inertia for the attitude DOF
            J_eff[k] = np.abs(Md[theta_index, theta_index])
            tau_ff_k, tau_pd_k = self.compute_control_tau(tk, qk, uk, Md=Md)
            tau_ff[k] = tau_ff_k
            tau_pd[k] = tau_pd_k

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
