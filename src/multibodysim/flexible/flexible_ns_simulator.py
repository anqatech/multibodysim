import numpy as np
from scipy.integrate import solve_ivp
from .flexible_ns_dynamics import FlexibleNonSymmetricDynamics


class FlexibleNonSymmetricSimulator:
    def __init__(self, config):
        self.config = config
        
        # ---------- Create symbolic dynamics model ---------- 
        self.dynamics = FlexibleNonSymmetricDynamics(config)
        
        # ---------- Extract parameter values ---------- 
        self.p_vals = self.dynamics.get_parameter_values()
        
        # ---------- Initialize results storage ---------- 
        self.results = None

    def eval_rhs(self, t, x):
        q = x[:self.dynamics.state_dimension]
        u = x[self.dynamics.state_dimension:]

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

        return self.results

    def get_results(self):
        if self.results is None:
            raise ValueError("Simulation has not been run yet. Call run_simulation() first.")
        return self.results
