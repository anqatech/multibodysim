import numpy as np
from scipy.integrate import solve_ivp
from .rigid_symbolic_7part import Rigid7PartSymbolicDynamics


class Rigid7PartSimulator:
    def __init__(self, config):
        self.config = config
        
        # Create symbolic dynamics model
        self.dynamics = Rigid7PartSymbolicDynamics(config)
        
        # Extract parameter values
        self.p_vals = self.dynamics.get_parameter_values()
        
        # Initialize results storage
        self.results = None
    
    def eval_rhs(self, t, x):
        q = x[:3]  # Positions
        u = x[3:]  # Velocities
        
        try:
            # Evaluate kinematic equations
            Mk, gk = self.dynamics.eval_kinematics(q, u, self.p_vals)
            qd = -np.linalg.solve(Mk, np.squeeze(gk))
            
            # Evaluate dynamic equations
            Md, gd = self.dynamics.eval_differentials(q, u, self.p_vals)
            ud = -np.linalg.solve(Md, np.squeeze(gd))
            
        except np.linalg.LinAlgError:
            print(f"Warning: Singular matrix encountered at t={t:.3f}")
            qd = np.zeros_like(q)
            ud = np.zeros_like(u)
        
        return np.hstack((qd, ud))
    
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
            'method': sim_params.get('method', 'RK45')
        }
        
        print(f"\nStarting simulation from t={t_start} to t={t_end}")
        print(f"Integration method: {integration_options['method']}")
        print(f"Tolerances: rtol={integration_options['rtol']}, atol={integration_options['atol']}\n")
        
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
        print(f"Number of function evaluations: {result.nfev}\n")
        
        # Calculate additional quantities
        self._calculate_derived_quantities(xs, ts)
        
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
            
            # Generalized speeds
            'u1': xs[:, 3],      # Bus x-velocity [m/s]
            'u2': xs[:, 4],      # Bus y-velocity [m/s]
            'u3': xs[:, 5],      # Bus angular velocity [rad/s]
            
            # Derived quantities
            'linear_momentum': self.linear_momentum,
            'angular_momentum': self.angular_momentum,
            'kinetic_energy': self.kinetic_energy,
            'center_of_mass': self.center_of_mass,
            
            # Configuration
            'config': self.config.copy()
        }
        
        return self.results
    
    def _calculate_derived_quantities(self, xs, ts):
        """
        Calculate derived quantities like momentum and energy.
        
        Parameters
        ----------
        xs : array_like
            State trajectory array
        ts : array_like
            Time points
        """
        # Extract parameters
        p = self.config['p_values']
        
        # Total mass
        M_total = (p['m_b1'] + p['m_b2'] + p['m_b3'] + 
                  p['m_p1'] + p['m_p2'] + p['m_p3'] + p['m_p4'])
        
        # Initialize arrays for derived quantities
        self.linear_momentum = np.zeros((len(ts), 3))
        self.angular_momentum = np.zeros(len(ts))
        self.kinetic_energy = np.zeros(len(ts))
        self.center_of_mass = np.zeros((len(ts), 2))
        
        for i, x in enumerate(xs):
            q1, q2, q3 = x[0:3]
            u1, u2, u3 = x[3:6]
            
            # Linear momentum (simplified - assuming all bodies move with central bus)
            # For exact calculation, we'd need individual body velocities
            self.linear_momentum[i, 0] = M_total * u1
            self.linear_momentum[i, 1] = M_total * u2
            self.linear_momentum[i, 2] = 0  # z-component is zero (planar motion)
            
            # Angular momentum about origin (simplified)
            # L = r × p + I_total * omega
            I_total = self._calculate_total_inertia(p)
            self.angular_momentum[i] = I_total * u3
            
            # Kinetic energy
            T_trans = 0.5 * M_total * (u1**2 + u2**2)
            T_rot = 0.5 * I_total * u3**2
            self.kinetic_energy[i] = T_trans + T_rot
            
            # Center of mass position (simplified)
            self.center_of_mass[i, 0] = q1  # Approximation
            self.center_of_mass[i, 1] = q2  # Approximation
    
    def _calculate_total_inertia(self, p):
        """
        Calculate total moment of inertia about z-axis.
        
        Parameters
        ----------
        p : dict
            Parameter dictionary
        
        Returns
        -------
        I_total : float
            Total moment of inertia
        """
        # Inertias about center of mass for each body
        Izz_bus1 = p['m_b1'] * p['D']**2 / 6
        Izz_bus2 = p['m_b2'] * p['D']**2 / 6
        Izz_bus3 = p['m_b3'] * p['D']**2 / 6
        Izz_p1 = p['m_p1'] * p['L']**2 / 12
        Izz_p2 = p['m_p2'] * p['L']**2 / 12
        Izz_p3 = p['m_p3'] * p['L']**2 / 12
        Izz_p4 = p['m_p4'] * p['L']**2 / 12
        
        # Sum of all inertias (simplified - neglecting parallel axis theorem)
        I_total = (Izz_bus1 + Izz_bus2 + Izz_bus3 + 
                  Izz_p1 + Izz_p2 + Izz_p3 + Izz_p4)
        
        return I_total
    
    def get_results(self):
        """
        Get simulation results.
        
        Returns
        -------
        results : dict
            Simulation results dictionary
        
        Raises
        ------
        ValueError
            If simulation has not been run yet
        """
        if self.results is None:
            raise ValueError("Simulation has not been run yet. Call run_simulation() first.")
        return self.results
    
    def save_results(self, filename):
        """
        Save simulation results to file.
        
        Parameters
        ----------
        filename : str
            Output filename (with or without extension)
        """
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
        """
        Load simulation results from file.
        
        Parameters
        ----------
        filename : str
            Input filename
        
        Returns
        -------
        results : dict
            Loaded results dictionary
        """
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