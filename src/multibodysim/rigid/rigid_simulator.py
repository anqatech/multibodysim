import numpy as np
import sympy as sm
from scipy.integrate import solve_ivp
from .rigid_symbolic_model import RigidSymbolicDynamics


class RigidSimulator:
    def __init__(self, config):
        """Initialize the rigid satellite simulator with config"""
        self.config = config
        
        # Create symbolic dynamics model - KEY CHANGE
        self.dynamics = RigidSymbolicDynamics(config)
        
        # Extract parameter values
        self.p_vals = self.dynamics.get_parameter_values()
        
        # Initialize results storage
        self.results = None
        
    def eval_rhs(self, t, x):
        q = x[:3]  # First 3 elements are positions
        u = x[3:]  # Last 3 elements are velocities

        try:
            # Use dynamics object methods instead of embedded symbolic code
            Mk, gk = self.dynamics.eval_kinematics(q, u, self.p_vals)
            qd = -np.linalg.solve(Mk, np.squeeze(gk))

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
        # Get initial conditions from dynamics object
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
        
        # Integrate equations of motion
        result = solve_ivp(
            fun=self.eval_rhs,
            t_span=(t_start, t_end),
            y0=x0,
            t_eval=t_eval,
            **integration_options
        )
        
        # Process results
        xs = np.transpose(result.y)
        ts = result.t
        
        print(f"Simulation completed: {result.success}")
        print(f"Message: {result.message}")
        
        # Store results in same format as flexible simulator
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
            
            # Generalized speeds
            'u1': xs[:, 3],      # Bus x-velocity [m/s]
            'u2': xs[:, 4],      # Bus y-velocity [m/s]
            'u3': xs[:, 5],      # Bus angular velocity [rad/s]
            
            # Configuration
            'config': self.config.copy()
        }
        
        return self.results
    
    def get_results(self):
        if self.results is None:
            raise ValueError("Simulation has not been run yet. Call run_simulation() first.")
        return self.results
    
    def save_results(self, filename):
        if self.results is None:
            raise ValueError("No results to save. Run simulation first.")
        
        if filename.endswith('.npz'):
            np.savez(filename, **self.results)
        elif filename.endswith('.npy'):
            np.save(filename, self.results)
        else:
            np.savez(filename + '.npz', **self.results)
        
        print(f"Results saved to {filename}")
    
    def load_results(self, filename):
        if filename.endswith('.npz'):
            loaded = np.load(filename, allow_pickle=True)
            self.results = {key: loaded[key] for key in loaded.files}
        elif filename.endswith('.npy'):
            self.results = np.load(filename, allow_pickle=True).item()
        else:
            try:
                loaded = np.load(filename + '.npz', allow_pickle=True)
                self.results = {key: loaded[key] for key in loaded.files}
            except:
                self.results = np.load(filename + '.npy', allow_pickle=True).item()
        
        print(f"Results loaded from {filename}")
        return self.results
