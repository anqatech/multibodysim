import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import fsolve
from .flexible_ns_dynamics import FlexibleNonSymmetricDynamics


class FlexibleNonSymmetricSimulatorNew:
    def __init__(self, config):
        self.config = config
        
        # Create symbolic dynamics model
        self.dynamics = FlexibleNonSymmetricDynamics(config)
        
        # Extract parameter values
        self.p_vals = self.dynamics.get_parameter_values()
        
        # Initialize results storage
        self.results = None

    def eval_rhs(self, t, x):
        q = x[:5]
        u = x[5:]

        try:
            # Evaluate kinematic equations
            Mk, gk = self.dynamics.eval_kinematics(q, u, self.p_vals)
            qd = -np.linalg.solve(Mk, np.squeeze(gk))

            # Evaluate dynamic equations
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
        pass

    def get_results(self):
        pass
        # if self.results is None:
        #     raise ValueError("Simulation has not been run yet. Call run_simulation() first.")
        # return self.results
    
    def save_results(self, filename):
        pass
        # if self.results is None:
        #     raise ValueError("No results to save. Run simulation first.")
        
        # # Save as numpy archive
        # if filename.endswith('.npz'):
        #     np.savez(filename, **self.results)
        # elif filename.endswith('.npy'):
        #     np.save(filename, self.results)
        # else:
        #     # Default to npz
        #     np.savez(filename + '.npz', **self.results)
        
        # print(f"Results saved to {filename}")
    
    def load_results(self, filename):
        pass
        # if filename.endswith('.npz'):
        #     loaded = np.load(filename, allow_pickle=True)
        #     self.results = {key: loaded[key] for key in loaded.files}
        # elif filename.endswith('.npy'):
        #     self.results = np.load(filename, allow_pickle=True).item()
        # else:
        #     # Try npz first
        #     try:
        #         loaded = np.load(filename + '.npz', allow_pickle=True)
        #         self.results = {key: loaded[key] for key in loaded.files}
        #     except:
        #         self.results = np.load(filename + '.npy', allow_pickle=True).item()
        
        # print(f"Results loaded from {filename}")
        # return self.results
