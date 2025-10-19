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
        q = x[:5]
        u = x[5:]

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
        
        # ---------- Store results ---------- 
        self.results = {
            'time': ts,
            'states': xs,
            'success': result.success,
            'message': result.message,
            'nfev': result.nfev,
            'njev': result.njev if hasattr(result, 'njev') else None,
            'nlu': result.nlu if hasattr(result, 'nlu') else None,
            
            # ---------- Reference generalized coordinates ---------- 
            'q1': xs[:, 0],      # Central bus geometric center x-position [m]
            'q2': xs[:, 1],      # Central bus geometric center y-position [m]
            'q3': xs[:, 2],      # Central bus rotation [rad]
            
            # ---------- Reference generalized speeds ---------- 
            'u1': xs[:, 5],      # Bus x-velocity [m/s]
            'u2': xs[:, 6],      # Bus y-velocity [m/s]
            'u3': xs[:, 7],      # Bus angular velocity [rad/s]
            
            # ---------- Configuration ---------- 
            'config': self.config.copy()
        }

        # ---------- Flexible generalized coordinates and speeds ---------- 
        for i in range(self.dynamics.state_reference_dimension, self.dynamics.state_dimension):
            self.results[f"eta_{i-2}"] = xs[:, i]                                  # Solar panel modal amplitude [-]
            self.results[f"zeta_{i-2}"] = xs[:, i+self.dynamics.state_dimension]   # Solar panel modal velocity [-]

        return self.results

    def get_results(self):
        if self.results is None:
            raise ValueError("Simulation has not been run yet. Call run_simulation() first.")
        return self.results
