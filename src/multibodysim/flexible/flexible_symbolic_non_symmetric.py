import numpy as np
import sympy as sm
import sympy.physics.mechanics as me
from scipy.optimize import fsolve
from ..beam.cantilever_beam import CantileverBeam


class FlexibleSymbolicNonSymmetricDynamics:
    def __init__(self, config):
        self.config = config
        me.init_vprinting()

        self._define_symbols()
        self._define_mode_shapes()
        self._define_kinematics()
        self._define_kinematic_equations()
        self._setup_velocities()
        self._derive_generalized_forces()
        self._formulate_eom()

        # Create numerical evaluation functions
        self._create_lambdified_functions()

    def _define_symbols(self):
        # Generalized coordinates and speeds
        self.q1, self.q2, self.q3 = me.dynamicsymbols('q1, q2, q3')
        self.u1, self.u2, self.u3 = me.dynamicsymbols('u1, u2, u3')
        
        # Flexible body coordinates and speeds (one mode per panel)
        self.eta_r, self.eta_l = me.dynamicsymbols('eta_r, eta_l')
        self.u4, self.u5 = me.dynamicsymbols('u4, u5')
        
        # System parameters
        self.D, self.L = sm.symbols('D, L')           # Bus side D, Panel length L
        self.m_b = sm.symbols('m_b')                  # Satellite bus mass
        self.m_l, self.m_r = sm.symbols('m_l, m_r')   # Satellite left and right panel masses
        self.E_mod = sm.symbols('E')                  # Young's Modulus
        self.I_area = sm.symbols('I')                 # Area Moment of Inertia
        self.tau = me.dynamicsymbols('tau')           # External torque on the central bus
        self.t = me.dynamicsymbols._t                 # Time symbol
        
        # Position along the beam for integration
        self.s = sm.Symbol('s')
        
        # Vectors definition
        self.p_symbols = sm.Matrix([self.D, self.L, self.m_b, self.m_r, self.m_l, 
                                   self.E_mod, self.I_area, self.tau])

        self.q = sm.Matrix([self.q1, self.q2, self.q3, self.eta_r, self.eta_l])
        self.qd = self.q.diff(me.dynamicsymbols._t)
        self.u = sm.Matrix([self.u1, self.u2, self.u3, self.u4, self.u5])
        self.ud = self.u.diff(me.dynamicsymbols._t)

        # Zero replacement dictionaries
        self.qd_zero = {qdi: 0 for qdi in self.qd}
        self.ud_zero = {udi: 0 for udi in self.ud}

    def _define_mode_shapes(self):
        # Create cantilever beam instance
        beam_params = self.config.get('beam_parameters', {})
        self.beam = CantileverBeam(
            length=self.config['p_values']['L'],
            E=self.config['p_values']['E_mod'],
            I=self.config['p_values']['I_area'],
            beta1=beam_params['beta1'],
            sigma1=beam_params['sigma1'],
            s=self.s
        )
        
        # Shape function (First mode of a uniform cantilever beam) - symbolic
        self.beta1 = self.beam.beta1
        self.sigma1 = self.beam.sigma1
        arg = self.beta1 * self.s / self.L
        self.phi1 = (sm.cosh(arg) - sm.cos(arg) - 
                     self.sigma1 * (sm.sinh(arg) - sm.sin(arg)))
        
        # Calculate mass-averaged deflection using the beam class
        n_points = beam_params.get('n_integration_points', 200)
        self.phi1_mean = self.beam.mode_shape_mean(n_points)

    def _define_kinematics(self):
        # Reference frames (Part 3)
        self.N = me.ReferenceFrame('N')
        self.B = me.ReferenceFrame('B') 
        self.C = me.ReferenceFrame('C')
        self.E = me.ReferenceFrame('E')
        
        self.B.orient_axis(self.N, self.q3, self.N.z)
        self.C.orient_axis(self.N, self.q3, self.N.z)
        self.E.orient_axis(self.N, self.q3 + sm.pi, self.N.z)
        
        # Points
        self.O = me.Point('O')
        self.Bus_cm = me.Point('B_cm')
        self.Joint_Right = me.Point('J_R')
        self.Joint_Left = me.Point('J_L')
        
        # Position chain
        self.Bus_cm.set_pos(self.O, self.q1 * self.N.x + self.q2 * self.N.y)
        self.Joint_Right.set_pos(self.Bus_cm, (self.D / 2) * self.B.x)
        self.Joint_Left.set_pos(self.Bus_cm, -(self.D / 2) * self.B.x)

        # Points and position of infinitesimal mass elements on flexible panels
        self.dm_on_C = self.s * self.C.x + self.phi1 * self.eta_r * self.C.y
        self.dm_on_E = self.s * self.E.x + self.phi1 * self.eta_l * self.E.y
        self.Point_dm_C = self.Joint_Right.locatenew('P_dm_C', self.dm_on_C)
        self.Point_dm_E = self.Joint_Left.locatenew('P_dm_E', self.dm_on_E)
        
        # Panel centers of mass
        self.cm_dm_on_C = (self.L / 2) * self.C.x + self.phi1_mean * self.eta_r * self.C.y
        self.cm_dm_on_E = (self.L / 2) * self.E.x + self.phi1_mean * self.eta_l * self.E.y
        
        self.Point_C_cm = self.Joint_Right.locatenew('P_C_cm', self.cm_dm_on_C)
        self.Point_E_cm = self.Joint_Left.locatenew('P_E_cm', self.cm_dm_on_E)
        
        # System center of mass
        M = self.m_b + self.m_r + self.m_l
        r_G = (
            self.m_b * self.Bus_cm.pos_from(self.O) + 
            self.m_r * self.Point_C_cm.pos_from(self.O) + 
            self.m_l * self.Point_E_cm.pos_from(self.O)
        ) / M
        self.G = self.O.locatenew('G', r_G)
        self.r_GB = self.Bus_cm.pos_from(self.G)

    def _define_kinematic_equations(self):
        # Kinematical differential equations
        self.fk = sm.Matrix([
            self.q1.diff(self.t) - self.u1,
            self.q2.diff(self.t) - self.u2,
            self.q3.diff(self.t) - self.u3,
            self.eta_r.diff(self.t) - self.u4,
            self.eta_l.diff(self.t) - self.u5,
        ])
        
        self.Mk = self.fk.jacobian(self.qd)
        self.gk = self.fk.xreplace(self.qd_zero)
        
        qd_sol = -self.Mk.LUsolve(self.gk)
        self.qd_repl = dict(zip(self.qd, qd_sol))
        self.qdd_repl = {q.diff(self.t): u.diff(self.t) for q, u in self.qd_repl.items()}

    def _setup_velocities(self):
        # Angular velocities
        self.B.set_ang_vel(self.N, self.u3 * self.N.z)
        self.C.set_ang_vel(self.N, self.u3 * self.N.z)
        self.E.set_ang_vel(self.N, self.u3 * self.N.z)
        
        # Translational velocities
        self.O.set_vel(self.N, 0)
        self.Bus_cm.set_vel(self.N, self.u1 * self.N.x + self.u2 * self.N.y)
        _ = self.Joint_Right.v2pt_theory(self.Bus_cm, self.N, self.B)
        _ = self.Joint_Left.v2pt_theory(self.Bus_cm, self.N, self.B)

        # Flexible element velocities
        self.Point_dm_C.set_vel(self.C, self.phi1 * self.u4 * self.C.y)
        self.Point_dm_E.set_vel(self.E, self.phi1 * self.u5 * self.E.y)
        
        self.N_v_dm_C = self.Point_dm_C.v1pt_theory(self.Joint_Right, self.N, self.C)
        self.N_v_dm_E = self.Point_dm_E.v1pt_theory(self.Joint_Left, self.N, self.E)

        # Panel center of mass velocities
        self.Point_C_cm.set_vel(self.C, self.phi1_mean * self.u4 * self.C.y)
        self.Point_E_cm.set_vel(self.E, self.phi1_mean * self.u5 * self.E.y)
        self.N_v_C_cm = self.Point_C_cm.v1pt_theory(self.Joint_Right, self.N, self.C)
        self.N_v_E_cm = self.Point_E_cm.v1pt_theory(self.Joint_Left, self.N, self.E)
    
    def _derive_generalized_forces(self):
        # Collect velocities for partial velocity calculation
        velocities = (
            self.B.ang_vel_in(self.N),
            self.C.ang_vel_in(self.N), 
            self.E.ang_vel_in(self.N),
            self.Bus_cm.vel(self.N),
            self.N_v_dm_C,
            self.N_v_dm_E
        )
        
        # Partial velocities
        w_B, w_C, w_E, v_B, v_C, v_E = me.partial_velocity(velocities, self.u, self.N)
        
        # Forces and torques acting on the system
        R_B = 0 * self.B.x + 0 * self.B.y
        R_C = 0 * self.C.x + 0 * self.C.y
        R_E = 0 * self.E.x + 0 * self.E.y
        
        T_B = self.tau * self.B.z
        T_C = 0 * self.C.z
        T_E = 0 * self.E.z
        
        # Generalized active forces
        self.Generalised_Active_Forces = sm.zeros(len(self.u), 1)
        for i in range(3):
            Active_Force = 0
            for v, r, w, t in zip([v_B, v_C, v_E], [R_B, R_C, R_E], [w_B, w_C, w_E], [T_B, T_C, T_E]):
                Active_Force += v[i].dot(r) + w[i].dot(t)
            self.Generalised_Active_Forces[i] = Active_Force
        self.Generalised_Active_Forces = sm.Matrix(self.Generalised_Active_Forces)
        
        # Modal stiffness for the first mode of a cantilever beam
        # Use the beam class for modal stiffness calculation
        k_modal = self.beam.modal_stiffness_symbolic()

        k_r = k_modal
        k_l = k_modal
        
        # Strain potential energy stored in the flexible panels
        V_strain = (1/2) * k_r * self.eta_r**2 + (1/2) * k_l * self.eta_l**2
        
        # Restoring force from both panel stiffnesses
        self.Generalised_Active_Forces[3] = - V_strain.diff(self.eta_r)
        self.Generalised_Active_Forces[4] = - V_strain.diff(self.eta_l)
        
        # === Generalized Inertia Forces (Part 9) ===
        
        # Contribution from the rigid bus
        I_bus = (self.m_b * self.D**2) / 12
        I_b = me.inertia(self.B, I_bus, I_bus, 2 * I_bus)
        
        Rs_B = -self.m_b * self.Bus_cm.acc(self.N).xreplace(self.qdd_repl).xreplace(self.qd_repl)
        
        N_w_B = self.B.ang_vel_in(self.N).xreplace(self.qd_repl)
        N_alpha_B = N_w_B.dt(self.N).xreplace(self.qdd_repl).xreplace(self.qd_repl)
        Ts_B = -(N_alpha_B.dot(I_b) + me.cross(N_w_B, I_b.dot(N_w_B)))

        # Partial velocities
        v_B_partials = me.partial_velocity([self.Bus_cm.vel(self.N).xreplace(self.qd_repl)], self.u, self.N)[0]
        w_B_partials = me.partial_velocity([N_w_B], self.u, self.N)[0]
        
        Fr_star_B = sm.Matrix([
            v_B_partials[i].dot(Rs_B) + w_B_partials[i].dot(Ts_B) 
            for i in range(len(self.u))
        ])
        
        # Contribution from the flexible panels (via integration)
        mu_r = self.m_r / self.L
        mu_l = self.m_l / self.L
        
        # Accelerations of the flexible elements
        N_a_dm_C = self.N_v_dm_C.dt(self.N).xreplace(self.qdd_repl).xreplace(self.qd_repl)
        N_a_dm_E = self.N_v_dm_E.dt(self.N).xreplace(self.qdd_repl).xreplace(self.qd_repl)
        
        R_star_dm_C = -(mu_r * N_a_dm_C)
        R_star_dm_E = -(mu_l * N_a_dm_E)
        
        # Partial velocities
        v_dm_C_partials = me.partial_velocity([self.N_v_dm_C.xreplace(self.qd_repl)], self.u, self.N)[0]
        v_dm_E_partials = me.partial_velocity([self.N_v_dm_E.xreplace(self.qd_repl)], self.u, self.N)[0]
        
        Fr_star_C = sm.zeros(len(self.u), 1)
        Fr_star_E = sm.zeros(len(self.u), 1)
        
        for i in range(len(self.u)):
            integrand_C = v_dm_C_partials[i].dot(R_star_dm_C)
            integrand_E = v_dm_E_partials[i].dot(R_star_dm_E)
            Fr_star_C[i] = sm.integrate(integrand_C, (self.s, 0, self.L))
            Fr_star_E[i] = sm.integrate(integrand_E, (self.s, 0, self.L))
        
        self.Fr_star = Fr_star_B + Fr_star_C + Fr_star_E

    def _formulate_eom(self):
        # Dynamic differential equation
        self.kane_eq = (self.Generalised_Active_Forces + self.Fr_star)

        # Generation of Matrix Md and vector gd
        self.Md = self.kane_eq.jacobian(self.ud)
        self.Md.simplify()
        self.gd = self.kane_eq.xreplace(self.ud_zero)

    def _create_lambdified_functions(self):
        # Create lambdified functions matching the notebook exactly
        self.eval_kinematics = sm.lambdify((self.q, self.u, self.p_symbols), (self.Mk, self.gk))
        self.eval_differentials = sm.lambdify((self.q, self.u, self.p_symbols), (self.Md, self.gd))

    def get_parameter_values(self):
        return np.array([
            self.config['p_values']['D'],
            self.config['p_values']['L'],
            self.config['p_values']['m_b'], 
            self.config['p_values']['m_r'],
            self.config['p_values']['m_l'],
            self.config['p_values']['E_mod'],
            self.config['p_values']['I_area'],
            self.config['p_values']['tau']
        ])
    
    def get_initial_conditions(self):
        # Extract initial states
        initial_states = self.config.get("q_initial", {})
        eta_r_initial = initial_states["eta_r"]
        eta_l_initial = initial_states["eta_l"]

        # Extract initial speeds
        initial_speeds = self.config.get('initial_speeds', {})
        u3_initial = initial_speeds.get('u3', 0.0)
        u4_initial = initial_speeds.get('u4', 0.0)
        u5_initial = initial_speeds.get('u5', 0.0)

        # Extract parameters
        parameters = self.config.get("p_values", {})
        M = parameters["m_b"] + parameters["m_r"] + parameters["m_l"]
        
        # Center of mass position vector
        rho = self.r_GB.express(self.B).simplify()
        rho_vector = sm.Matrix([
            [rho.dot(self.B.x)],
            [rho.dot(self.B.y)],
        ])
                
        # Skew matrix S
        S = sm.Matrix([
            [0, -1],
            [1, 0],
        ])

        # Rotation matrix R_theta
        R_theta = np.array([
            [np.cos(initial_states["q3"]), -np.sin(initial_states["q3"])],
            [np.sin(initial_states["q3"]),  np.cos(initial_states["q3"])]
        ])
        
        # Calculate initial generalized speeds constraints
        rho_derivative = -( self.phi1_mean / M) * (parameters["m_r"] * eta_r_initial - parameters["m_l"] * eta_l_initial) * np.array([[0.0], [1.0]])
        initial_generalised_speeds_constraints = u3_initial * S @ R_theta @ rho_vector + R_theta @ rho_derivative

        u_init_func = sm.lambdify(
            (self.q1, self.q2, self.q3, self.eta_r, self.eta_l, self.u3, self.D, self.L, self.m_b, self.m_r, self.m_l, self.E_mod, self.I_area), 
            [initial_generalised_speeds_constraints[0], initial_generalised_speeds_constraints[1]], 
            'numpy'
        )

        u1_consistent, u2_consistent = u_init_func(
            initial_states["q1"], 
            initial_states["q2"], 
            initial_states["q3"], 
            initial_states["eta_r"], 
            initial_states["eta_l"], 
            u3_initial, 
            parameters["D"],
            parameters["L"],
            parameters["m_b"],
            parameters["m_r"],
            parameters["m_l"],
            parameters["E_mod"],
            parameters["I_area"],
        )

        # Combine into state vector
        x0 = np.array([
            initial_states["q1"], 
            initial_states["q2"], 
            initial_states["q3"], 
            initial_states["eta_r"], 
            initial_states["eta_l"], 
            u1_consistent, 
            u2_consistent,
            u3_initial,
            u4_initial,
            u5_initial
        ])
        
        print(f"\nInitial conditions set (with momentum conservation):")
        print(f"  Positions: q1={initial_states["q1"]:.3f}, q2={initial_states["q2"]:.3f}, q3={np.rad2deg(initial_states["q3"]):.3f}°, eta_r={np.rad2deg(initial_states["eta_r"]):.3f}, eta_l={np.rad2deg(initial_states["eta_l"]):.3f}")
        print(f"  Velocities: u1={u1_consistent:.6f}, u2={u2_consistent:.6f}, u3={u3_initial:.3f}, , u4={u4_initial:.3f}, , u5={u5_initial:.3f}\n")
        
        return x0
