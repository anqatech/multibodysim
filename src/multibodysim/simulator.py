import numpy as np
import sympy as sm
from scipy.integrate import solve_ivp


class SatelliteSimulator:
    def __init__(self, symbolic_model):
        print("Step 5: Compiling symbolic EOM matrices into a numerical function...")
        self._model = symbolic_model
        self._lambdify_eom()
        self.result = None

    def _lambdify_eom(self):
        # Unpack symbols from the model for clarity
        q, u, p = self._model.q, self._model.u, self._model.p
        Mk, gk = self._model.Mk, self._model.gk
        Md, gd = self._model.Md, self._model.gd
        
        # Create a single function that evaluates all EOM matrices at once
        self.eval_eom_matrices = sm.lambdify(
            (q, u, p),
            [Mk, gk, Md, gd],
            'numpy'
        )

        # Lambdify the helper expression for calculating initial speeds
        rho = self._model.r_GB.express(self._model.B).simplify()
        rho_vector = sm.Matrix([
            [rho.dot(self._model.B.x)], 
            [rho.dot(self._model.B.y)]
        ])
        S = sm.Matrix([
            [0, -1], 
            [1, 0]
        ])
        R_theta = sm.Matrix([
            [sm.cos(self._model.q3), -sm.sin(self._model.q3)],
            [sm.sin(self._model.q3),  sm.cos(self._model.q3)]
        ])
        
        initial_speeds_expr = self._model.u3 * S * R_theta * rho_vector
        
        self.calculate_initial_speeds = sm.lambdify(
            (
                self._model.q3, 
                self._model.u3, 
                self._model.D, 
                self._model.L, 
                self._model.m_r, 
                self._model.m_l, 
                self._model.m_b
            ),
            [initial_speeds_expr[0], initial_speeds_expr[1]],
            'numpy'
        )

    def _rhs(self, t, x, p_vec):
        # Unpack the state and parameter vectors
        q = x[:3]
        u = x[3:]
        
        # Evaluate the EOM matrices with the current values of q, u, p
        Mk, gk, Md, gd = self.eval_eom_matrices(q, u, p_vec)
        
        # Solve for q_dot and u_dot
        qd = np.linalg.solve(-Mk, np.squeeze(gk))
        ud = np.linalg.solve(-Md, np.squeeze(gd))
        
        # Pack q_dot and u_dot into a new state time derivative vector xd
        xd = np.empty_like(x)
        xd[:3] = qd
        xd[3:] = ud
        return xd

    def run(self, config):
        # Unpack and pre-process the configuration data
        p_values = config['p_values']
        q_initial = config['q_initial']
        q_initial["q3"] = np.deg2rad(q_initial["q3"])
        u_initial = config['initial_speeds']
        sim_parameters = config['sim_parameters']

        # Complete the initial speeds dictionary if necessary
        u_final = u_initial.copy()
        if 'u1' not in u_final and 'u2' not in u_final and 'u3' in u_final:
            print("Calculating initial u1 and u2 for zero linear momentum...")
            u1_0, u2_0 = self.calculate_initial_speeds(
                q_initial["q3"], u_final["u3"],
                p_values["D"], p_values["L"], p_values["m_r"], p_values["m_l"], p_values["m_b"]
            )
            u_final['u1'] = u1_0
            u_final['u2'] = u2_0
        
        # Assemble the initial state vector x0 in the correct order
        x0 = np.array([
            q_initial['q1'], q_initial['q2'], q_initial['q3'],
            u_final['u1'], u_final['u2'], u_final['u3']
        ])

        # Assemble model parameters in the correct order
        p_vec = [
            p_values["D"], p_values["L"], p_values["m_r"], 
            p_values["m_l"], p_values["m_b"], p_values["tau"]
        ]

        # Generate the simulation timesteps vector
        t_start, t_end = sim_parameters["t_start"], sim_parameters["t_end"]
        t_points = np.linspace(
            t_start, 
            t_end, 
            num=sim_parameters["nb_timesteps"]
        )
        
        # Run the ODE solver
        print("Running the simulation...")
        self.result = solve_ivp(
            fun=self._rhs,
            t_span=(t_start, t_end),
            y0=x0,
            t_eval=t_points,
            args=(p_vec,),
            rtol=1e-6,
            atol=1e-6,
            method='RK45',
            vectorized=False
        )
        print(f"\nSimulation finished successfully: {self.result.message}")

