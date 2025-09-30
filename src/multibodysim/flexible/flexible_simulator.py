import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import fsolve
from .flexible_symbolic_model import FlexibleSymbolicDynamics


class FlexibleSimulator:
    def __init__(self, config):
        self.config = config
        
        # Create symbolic dynamics model
        self.dynamics = FlexibleSymbolicDynamics(config)
        
        # Extract parameter values
        self.p_vals = self.dynamics.get_parameter_values()
        
        # Initialize results storage
        self.results = None
        
    def eval_rhs(self, t, x):
        qN = x[:5]
        u = x[5:]

        # Correct the dependent coordinates
        eta_l = fsolve(
            lambda qr, q, p: np.squeeze(self.dynamics.eval_constraints(qr, q, p)),
            qN[4:5],
            args=(qN[0:4], self.p_vals)
        )
        qN = np.concatenate([qN[0:4], eta_l])

        try:
            Mk, gk = self.dynamics.eval_kinematics(qN, u, self.p_vals)
            qNd = -np.linalg.solve(Mk, np.squeeze(gk))

            Md, gd = self.dynamics.eval_differentials(qN, u, self.p_vals)
            ud = -np.linalg.solve(Md, np.squeeze(gd))
            
        except np.linalg.LinAlgError:
            print("Singular matrix encountered.")
            qNd = np.zeros_like(qN)
            ud = np.zeros_like(u)
        
        return np.hstack((qNd, ud))
    
    def setup_initial_conditions(self):
        return self.dynamics.get_initial_conditions()
    
    def run_simulation(self):
        # Get initial conditions
        x0 = self.setup_initial_conditions()
        
        # Extract simulation parameters
        sim_params = self.config['sim_parameters']
        t_start = sim_params['t_start']
        t_end = sim_params['t_end']
        nb_timesteps = sim_params['nb_timesteps']
        
        # Create time evaluation points
        t_eval = np.linspace(t_start, t_end, nb_timesteps)
        
        # Integration settings
        integration_options = {
            'rtol': sim_params.get('rtol', 1e-6),
            'atol': sim_params.get('atol', 1e-9),
            'method': sim_params.get('method', 'Radau')
        }
        
        print(f"Starting simulation from t={t_start} to t={t_end}")
        print(f"Integration method: {integration_options['method']}")
        print(f"Tolerances: rtol={integration_options['rtol']}, atol={integration_options['atol']}")
        
        # Integrate equations of motion
        result = solve_ivp(
            self.eval_rhs,
            (t_start, t_end),
            x0,
            t_eval=t_eval,
            **integration_options
        )
        
        # Process results
        xs = np.transpose(result.y)
        ts = result.t
        
        print(f"Simulation completed: {result.success}")
        print(f"Message: {result.message}")
        
        # Check constraint violations
        constraints = self.eval_simulated_constraints(xs)
        max_constraint_violation = np.max(np.abs(constraints))
        
        print(f"Maximum constraint violation: {max_constraint_violation:.2e}")
        
        # Store results
        self.results = {
            'time': ts,
            'states': xs,
            'success': result.success,
            'message': result.message,
            'nfev': result.nfev,
            'njev': result.njev if hasattr(result, 'njev') else None,
            'nlu': result.nlu if hasattr(result, 'nlu') else None,
            
            # Generalized coordinates
            'q1': xs[:, 0],      # Bus x-position [m]
            'q2': xs[:, 1],      # Bus y-position [m]
            'q3': xs[:, 2],      # Bus rotation [rad]
            'eta_r': xs[:, 3],   # Right panel modal amplitude [-]
            'eta_l': xs[:, 4],   # Left panel modal amplitude [-]
            
            # Generalized speeds
            'u1': xs[:, 5],      # Bus x-velocity [m/s]
            'u2': xs[:, 6],      # Bus y-velocity [m/s]
            'u3': xs[:, 7],      # Bus angular velocity [rad/s]
            'u4': xs[:, 8],      # Right panel modal velocity [-]
            
            # Constraint information
            'constraints': constraints,
            'max_constraint_violation': max_constraint_violation,
            
            # Configuration
            'config': self.config.copy()
        }
        
        return self.results
    
    def eval_simulated_constraints(self, xs):
        constraints = []
        for xi in xs:
            con = self.dynamics.eval_constraints(xi[4:5], xi[0:4], self.p_vals)
            constraints.append(con.squeeze())
        return np.array(constraints)
    
    def get_results(self):
        if self.results is None:
            raise ValueError("Simulation has not been run yet. Call run_simulation() first.")
        return self.results
    
    def save_results(self, filename):
        if self.results is None:
            raise ValueError("No results to save. Run simulation first.")
        
        # Save as numpy archive
        if filename.endswith('.npz'):
            np.savez(filename, **self.results)
        elif filename.endswith('.npy'):
            np.save(filename, self.results)
        else:
            # Default to npz
            np.savez(filename + '.npz', **self.results)
        
        print(f"Results saved to {filename}")
    
    def load_results(self, filename):
        if filename.endswith('.npz'):
            loaded = np.load(filename, allow_pickle=True)
            self.results = {key: loaded[key] for key in loaded.files}
        elif filename.endswith('.npy'):
            self.results = np.load(filename, allow_pickle=True).item()
        else:
            # Try npz first
            try:
                loaded = np.load(filename + '.npz', allow_pickle=True)
                self.results = {key: loaded[key] for key in loaded.files}
            except:
                self.results = np.load(filename + '.npy', allow_pickle=True).item()
        
        print(f"Results loaded from {filename}")
        return self.results
