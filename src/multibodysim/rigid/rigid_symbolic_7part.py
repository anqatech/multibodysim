import numpy as np
import sympy as sm
import sympy.physics.mechanics as me


class Rigid7PartSymbolicDynamics:
    def __init__(self, config):
        self.config = config
        me.init_vprinting()
        
        # Build symbolic model following package blueprint
        self._define_symbols()
        self._define_kinematics()
        self._define_kinematic_equations()
        self._setup_velocities()
        self._derive_generalized_forces()
        self._formulate_eom()
        
        # Create numerical evaluation functions
        self._create_lambdified_functions()
    
    def _define_symbols(self):
        # Generalized coordinates and speeds
        self.q1, self.q2, self.q3 = me.dynamicsymbols('q1 q2 q3')
        self.u1, self.u2, self.u3 = me.dynamicsymbols('u1 u2 u3')
        # External torque
        self.tau = me.dynamicsymbols('tau')
        self.t = me.dynamicsymbols._t
        
        # System parameters
        self.D, self.L = sm.symbols('D L', real=True, positive=True)
        # Bus masses
        self.m_b1, self.m_b2, self.m_b3 = sm.symbols('m_b1 m_b2 m_b3', real=True, positive=True)
        # Panel masses
        self.m_p1, self.m_p2, self.m_p3, self.m_p4 = sm.symbols('m_p1 m_p2 m_p3 m_p4', real=True, positive=True)
        
        # Vectors definition
        self.p_symbols = sm.Matrix([
            self.D, self.L, 
            self.m_b1, self.m_b2, self.m_b3,
            self.m_p1, self.m_p2, self.m_p3, self.m_p4,
            self.tau
        ])
        
        self.q = sm.Matrix([self.q1, self.q2, self.q3])
        self.qd = self.q.diff(self.t)
        self.u = sm.Matrix([self.u1, self.u2, self.u3])
        self.ud = self.u.diff(self.t)
        
        # Zero replacement dictionaries
        self.qd_zero = {qdi: 0 for qdi in self.qd}
        self.ud_zero = {udi: 0 for udi in self.ud}
    
    def _define_kinematics(self):
        # Reference Frames
        self.N = me.ReferenceFrame('N')  # Inertial frame
        
        # Bus frames
        self.B1 = me.ReferenceFrame('B_1')
        self.B2 = me.ReferenceFrame('B_2')  # Central bus
        self.B3 = me.ReferenceFrame('B_3')
        
        # Panel frames
        self.P1 = me.ReferenceFrame('P_1')
        self.P2 = me.ReferenceFrame('P_2')
        self.P3 = me.ReferenceFrame('P_3')
        self.P4 = me.ReferenceFrame('P_4')
        
        # Orient all frames with the same rotation
        self.B1.orient_axis(self.N, self.q3, self.N.z)
        self.B2.orient_axis(self.N, self.q3, self.N.z)
        self.B3.orient_axis(self.N, self.q3, self.N.z)
        
        # Panels P1 and P2 are on opposite side (rotated by π)
        self.P1.orient_axis(self.N, self.q3 + sm.pi, self.N.z)
        self.P2.orient_axis(self.N, self.q3 + sm.pi, self.N.z)
        self.P3.orient_axis(self.N, self.q3, self.N.z)
        self.P4.orient_axis(self.N, self.q3, self.N.z)
        
        self.frame_list = [self.B1, self.B2, self.B3, self.P1, self.P2, self.P3, self.P4]
        
        # Points
        self.O = me.Point('O')  # Origin
        
        # Bus centers of mass
        self.Bus1_cm = me.Point('B1_cm')
        self.Bus2_cm = me.Point('B2_cm')  # Central bus
        self.Bus3_cm = me.Point('B3_cm')
        
        # Joint points
        self.Joint1_Right = me.Point('J1_R')
        self.Joint1_Left = me.Point('J1_L')
        self.Joint2_Right = me.Point('J2_R')
        self.Joint2_Left = me.Point('J2_L')
        self.Joint3_Right = me.Point('J3_R')
        self.Joint3_Left = me.Point('J3_L')
        
        # Panel centers of mass
        self.P1_cm = me.Point('P1_cm')
        self.P2_cm = me.Point('P2_cm')
        self.P3_cm = me.Point('P3_cm')
        self.P4_cm = me.Point('P4_cm')
        
        # Position chain (starting from central bus)
        self.Bus2_cm.set_pos(self.O, self.q1 * self.N.x + self.q2 * self.N.y)
        
        # Joints on central bus
        self.Joint2_Right.set_pos(self.Bus2_cm, (self.D/2) * self.B2.x)
        self.Joint2_Left.set_pos(self.Bus2_cm, -(self.D/2) * self.B2.x)
        
        # Panels attached to central bus
        self.P3_cm.set_pos(self.Joint2_Right, (self.L/2) * self.P3.x)
        self.P2_cm.set_pos(self.Joint2_Left, (self.L/2) * self.P2.x)
        
        # Joints at panel ends
        self.Joint1_Right.set_pos(self.P2_cm, (self.L/2) * self.P2.x)
        self.Joint3_Left.set_pos(self.P3_cm, (self.L/2) * self.P3.x)
        
        # Outer buses
        self.Bus1_cm.set_pos(self.Joint1_Right, -(self.D/2) * self.B1.x)
        self.Bus3_cm.set_pos(self.Joint3_Left, (self.D/2) * self.B3.x)
        
        # Outer joints
        self.Joint3_Right.set_pos(self.Bus3_cm, (self.D/2) * self.B3.x)
        self.Joint1_Left.set_pos(self.Bus1_cm, -(self.D/2) * self.B1.x)
        
        # Outer panels
        self.P4_cm.set_pos(self.Joint3_Right, (self.L/2) * self.P4.x)
        self.P1_cm.set_pos(self.Joint1_Left, (self.L/2) * self.P1.x)
        
        self.body_list = [
            self.Bus1_cm, self.Bus2_cm, self.Bus3_cm,
            self.P1_cm, self.P2_cm, self.P3_cm, self.P4_cm
        ]
        
        # System center of mass
        M = self.m_b1 + self.m_b2 + self.m_b3 + self.m_p1 + self.m_p2 + self.m_p3 + self.m_p4
        r_G = (
            self.m_b1 * self.Bus1_cm.pos_from(self.O) +
            self.m_b2 * self.Bus2_cm.pos_from(self.O) +
            self.m_b3 * self.Bus3_cm.pos_from(self.O) +
            self.m_p1 * self.P1_cm.pos_from(self.O) +
            self.m_p2 * self.P2_cm.pos_from(self.O) +
            self.m_p3 * self.P3_cm.pos_from(self.O) +
            self.m_p4 * self.P4_cm.pos_from(self.O)
        ) / M
        G = self.O.locatenew('G', r_G)
        self.r_GB = self.Bus2_cm.pos_from(G)
    
    def _define_kinematic_equations(self):
        # Kinematic differential equation
        fk = sm.Matrix([
            self.u1 - self.q1.diff(),
            self.u2 - self.q2.diff(),
            self.u3 - self.q3.diff(),
        ])
        
        # Generation of Matrix Mk and vector gk
        self.Mk = fk.jacobian(self.qd)
        self.gk = fk.xreplace(self.qd_zero)
        
        qd_sol = -self.Mk.LUsolve(self.gk)
        self.qd_repl = dict(zip(self.qd, qd_sol))
        self.qdd_repl = {q.diff(self.t): u.diff(self.t) for q, u in self.qd_repl.items()}
    
    def _setup_velocities(self):
        # Angular velocities (all bodies rotate together)
        for frame in self.frame_list:
            frame.set_ang_vel(self.N, self.u3 * self.N.z)
        
        # Translational velocities
        self.O.set_vel(self.N, 0)
        self.Bus2_cm.set_vel(self.N, self.u1 * self.N.x + self.u2 * self.N.y)
        
        # Use two-point velocity theorem for connected points
        self.Joint2_Right.v2pt_theory(self.Bus2_cm, self.N, self.B2)
        self.Joint2_Left.v2pt_theory(self.Bus2_cm, self.N, self.B2)
        self.P3_cm.v2pt_theory(self.Joint2_Right, self.N, self.P3)
        self.P2_cm.v2pt_theory(self.Joint2_Left, self.N, self.P2)
        self.Joint1_Right.v2pt_theory(self.P2_cm, self.N, self.P2)
        self.Joint3_Left.v2pt_theory(self.P3_cm, self.N, self.P3)
        self.Bus1_cm.v2pt_theory(self.Joint1_Right, self.N, self.B1)
        self.Bus3_cm.v2pt_theory(self.Joint3_Left, self.N, self.B3)
        self.Joint1_Left.v2pt_theory(self.Bus1_cm, self.N, self.B1)
        self.Joint3_Right.v2pt_theory(self.Bus3_cm, self.N, self.B3)
        self.P4_cm.v2pt_theory(self.Joint3_Right, self.N, self.P4)
        self.P1_cm.v2pt_theory(self.Joint1_Left, self.N, self.P1)
    
    def _derive_generalized_forces(self):
        # Velocities for partial velocity calculation
        body_velocities = [cm.vel(self.N) for cm in self.body_list]
        frame_angular_velocities = [frame.ang_vel_in(self.N) for frame in self.frame_list]
        
        # Partial velocities
        v_partials = me.partial_velocity(body_velocities, self.u, self.N)
        w_partials = me.partial_velocity(frame_angular_velocities, self.u, self.N)
        
        # Inertia dyadics
        Izz_bus1 = self.m_b1 * self.D**2 / 6
        Izz_bus2 = self.m_b2 * self.D**2 / 6
        Izz_bus3 = self.m_b3 * self.D**2 / 6
        Izz_p1 = self.m_p1 * self.L**2 / 12
        Izz_p2 = self.m_p2 * self.L**2 / 12
        Izz_p3 = self.m_p3 * self.L**2 / 12
        Izz_p4 = self.m_p4 * self.L**2 / 12
        
        I_b1 = me.inertia(self.B1, 0, 0, Izz_bus1)
        I_b2 = me.inertia(self.B2, 0, 0, Izz_bus2)
        I_b3 = me.inertia(self.B3, 0, 0, Izz_bus3)
        I_p1 = me.inertia(self.P1, 0, 0, Izz_p1)
        I_p2 = me.inertia(self.P2, 0, 0, Izz_p2)
        I_p3 = me.inertia(self.P3, 0, 0, Izz_p3)
        I_p4 = me.inertia(self.P4, 0, 0, Izz_p4)
        
        self.inertias = [I_b1, I_b2, I_b3, I_p1, I_p2, I_p3, I_p4]
        
        # Accelerations
        body_accelerations = [cm.acc(self.N) for cm in self.body_list]
        body_angular_accelerations = [frame.ang_acc_in(self.N) for frame in self.frame_list]
        
        # Applied forces (all zero)
        R_B1 = 0 * self.B1.x + 0 * self.B1.y
        R_B2 = 0 * self.B2.x + 0 * self.B2.y
        R_B3 = 0 * self.B3.x + 0 * self.B3.y
        R_P1 = 0 * self.P1.x + 0 * self.P1.y
        R_P2 = 0 * self.P2.x + 0 * self.P2.y
        R_P3 = 0 * self.P3.x + 0 * self.P3.y
        R_P4 = 0 * self.P4.x + 0 * self.P4.y

        forces_list = [R_B1, R_B2, R_B3, R_P1, R_P2, R_P3, R_P4]
        
        # Applied torques (only on central bus)
        T_B1 = 0 * self.B1.z
        T_B2 = self.tau * self.B2.z  # External torque on central bus
        T_B3 = 0 * self.B3.z
        T_P1 = 0 * self.P1.z
        T_P2 = 0 * self.P2.z
        T_P3 = 0 * self.P3.z
        T_P4 = 0 * self.P4.z
        
        torques_list = [T_B1, T_B2, T_B3, T_P1, T_P2, T_P3, T_P4]
        
        # Generalized active forces
        self.Generalised_Active_Forces = []
        for i in range(3):
            Active_Force = 0
            for v, r, w, t in zip(v_partials, forces_list, w_partials, torques_list):
                Active_Force += v[i].dot(r) + w[i].dot(t)
            self.Generalised_Active_Forces.append(Active_Force)
        self.Generalised_Active_Forces = sm.Matrix(self.Generalised_Active_Forces)
        
        # Inertia forces
        masses_list = [self.m_b1, self.m_b2, self.m_b3, self.m_p1, self.m_p2, self.m_p3, self.m_p4]
        inertia_forces_list = [-m * a for m, a in zip(masses_list, body_accelerations)]
        
        # Inertia torques
        inertia_torques_list = []
        for i, (frame, inertia) in enumerate(zip(self.frame_list, self.inertias)):
            omega = frame_angular_velocities[i]
            alpha = body_angular_accelerations[i]
            T_star = -(alpha.dot(inertia) + me.cross(omega, inertia).dot(omega))
            inertia_torques_list.append(T_star)
        
        # Generalized inertia forces
        self.Generalised_Inertia_Forces = []
        for i in range(3):
            Inertia_Force = 0
            for v, rs, w, ts in zip(v_partials, inertia_forces_list, w_partials, inertia_torques_list):
                Inertia_Force += v[i].dot(rs) + w[i].dot(ts)
            self.Generalised_Inertia_Forces.append(Inertia_Force)
        self.Generalised_Inertia_Forces = sm.Matrix(self.Generalised_Inertia_Forces)
    
    def _formulate_eom(self):
        # Dynamic differential equation
        kane_eq = (self.Generalised_Active_Forces + self.Generalised_Inertia_Forces)
        
        # Generation of Matrix Md and vector gd
        self.Md = kane_eq.jacobian(self.ud)
        self.gd = kane_eq.xreplace(self.ud_zero)
    
    def _create_lambdified_functions(self):
        self.eval_kinematics = sm.lambdify((self.q, self.u, self.p_symbols), (self.Mk, self.gk))
        self.eval_differentials = sm.lambdify((self.q, self.u, self.p_symbols), (self.Md, self.gd))
    
    def get_parameter_values(self):
        p_values = self.config['p_values']
        
        return np.array([
            p_values['D'],      # Bus side length
            p_values['L'],      # Panel length
            p_values['m_b1'],   # Bus 1 mass
            p_values['m_b2'],   # Bus 2 mass (central)
            p_values['m_b3'],   # Bus 3 mass
            p_values['m_p1'],   # Panel 1 mass
            p_values['m_p2'],   # Panel 2 mass
            p_values['m_p3'],   # Panel 3 mass
            p_values['m_p4'],   # Panel 4 mass
            p_values['tau']     # External torque
        ])
    
    def get_initial_conditions(self):
        # Extract initial states
        initial_states = self.config.get("q_initial", {})
        
        # Extract initial speeds
        initial_speeds = self.config.get('initial_speeds', {})
        u3_initial = initial_speeds.get('u3', 0.0)
        
        # Extract parameters
        parameters = self.config.get("p_values", {})
        
        # Center of mass position vector
        rho = self.r_GB.express(self.B2).simplify()
        rho_vector = sm.Matrix([
            [rho.dot(self.B2.x)],
            [rho.dot(self.B2.y)],
        ])
        
        # Skew matrix S
        S = sm.Matrix([
            [0, -1],
            [1, 0],
        ])
        
        # Rotation matrix R_theta
        R_theta = sm.Matrix([
            [sm.cos(initial_states["q3"]), -sm.sin(initial_states["q3"])],
            [sm.sin(initial_states["q3"]), sm.cos(initial_states["q3"])]
        ])
        
        # Calculate initial generalized speeds constraints
        initial_generalised_speeds_constraints = u3_initial * S @ R_theta @ rho_vector
        
        u_init_func = sm.lambdify(
            (self.q1, self.q2, self.q3, self.u3, 
             self.D, self.L, 
             self.m_b1, self.m_b2, self.m_b3,
             self.m_p1, self.m_p2, self.m_p3, self.m_p4),
            [initial_generalised_speeds_constraints[0], 
             initial_generalised_speeds_constraints[1]],
            'numpy'
        )
        
        u1_consistent, u2_consistent = u_init_func(
            initial_states["q1"],
            initial_states["q2"],
            initial_states["q3"],
            u3_initial,
            parameters["D"],
            parameters["L"],
            parameters["m_b1"],
            parameters["m_b2"],
            parameters["m_b3"],
            parameters["m_p1"],
            parameters["m_p2"],
            parameters["m_p3"],
            parameters["m_p4"]
        )
        
        # Combine into state vector
        x0 = np.array([
            initial_states["q1"],
            initial_states["q2"],
            initial_states["q3"],
            u1_consistent,
            u2_consistent,
            u3_initial
        ])
        
        print(f"\nInitial conditions set (with momentum conservation):")
        print(f"  Positions: q1={initial_states['q1']:.3f}, q2={initial_states['q2']:.3f}, q3={np.rad2deg(initial_states['q3']):.3f}°")
        print(f"  Velocities: u1={u1_consistent:.6f}, u2={u2_consistent:.6f}, u3={u3_initial:.3f}\n")
        
        return x0
