import sympy as sm
import sympy.physics.mechanics as me
me.init_vprinting()

class RigidSymbolicDynamics:
    def __init__(self):
        self._define_symbols()
        self._define_kinematics()
        self._derive_generalized_forces()
        self._formulate_eom()

    def _define_symbols(self):
        print("Step 1: Setting up symbolic model...")
        # Generalized coordinates and speeds
        self.q1, self.q2, self.q3 = me.dynamicsymbols('q1, q2, q3')
        self.u1, self.u2, self.u3 = me.dynamicsymbols('u1, u2, u3')
        self.q = sm.Matrix([self.q1, self.q2, self.q3])
        self.qd = self.q.diff(me.dynamicsymbols._t)
        self.u = sm.Matrix([self.u1, self.u2, self.u3])
        self.ud = self.u.diff(me.dynamicsymbols._t)

        # System parameters
        self.D, self.L = sm.symbols('D, L')
        self.m_b, self.m_l, self.m_r = sm.symbols('m_b, m_l, m_r')
        self.tau = me.dynamicsymbols('tau')
        self.t = me.dynamicsymbols._t

        # Parameter vector for lambdification
        self.p = sm.Matrix([self.D, self.L, self.m_r, self.m_l, self.m_b, self.tau])

    def _define_kinematics(self):
        print("Step 2: Defining system kinematics...")
        # Reference Frames
        self.N = me.ReferenceFrame('N') # Inertial frame
        self.B = me.ReferenceFrame('B') # Bus frame
        self.C = me.ReferenceFrame('C') # Right panel frame
        self.E = me.ReferenceFrame('E') # Left panel frame

        self.B.orient_axis(self.N, self.q3, self.N.z)
        self.C.orient_axis(self.N, self.q3, self.N.z)
        self.E.orient_axis(self.N, self.q3 + sm.pi, self.N.z)

        # Points
        O = me.Point('O')
        self.Bus_cm = me.Point('B_cm')
        Joint_Right = me.Point('J_R')
        Joint_Left = me.Point('J_L')
        self.Panel_Right_cm = me.Point('R_cm')
        self.Panel_Left_cm = me.Point('L_cm')

        # Position Chain
        self.Bus_cm.set_pos(O, self.q1 * self.N.x + self.q2 * self.N.y)
        Joint_Right.set_pos(self.Bus_cm, (self.D / 2) * self.B.x)
        Joint_Left.set_pos(self.Bus_cm, -(self.D / 2) * self.B.x)
        self.Panel_Right_cm.set_pos(Joint_Right, (self.L / 2) * self.C.x)
        self.Panel_Left_cm.set_pos(Joint_Left, (self.L / 2) * self.E.x)

        # Velocities
        self.B.set_ang_vel(self.N, self.u3 * self.N.z)
        self.C.set_ang_vel(self.N, self.u3 * self.N.z)
        self.E.set_ang_vel(self.N, self.u3 * self.N.z)

        O.set_vel(self.N, 0)
        self.Bus_cm.set_vel(self.N, self.u1 * self.N.x + self.u2 * self.N.y)
        Joint_Right.v2pt_theory(self.Bus_cm, self.N, self.B)
        Joint_Left.v2pt_theory(self.Bus_cm, self.N, self.B)
        self.Panel_Right_cm.v2pt_theory(Joint_Right, self.N, self.C)
        self.Panel_Left_cm.v2pt_theory(Joint_Left, self.N, self.E)

        # System Center of Mass (useful for initial conditions later)
        M = self.m_b + self.m_r + self.m_l
        r_G = (self.m_b * self.Bus_cm.pos_from(O) +
               self.m_r * self.Panel_Right_cm.pos_from(O) +
               self.m_l * self.Panel_Left_cm.pos_from(O)) / M
        G = O.locatenew('G', r_G)
        self.r_GB = self.Bus_cm.pos_from(G)

    def _derive_generalized_forces(self):
        print("Step 3: Formulating generalized forces...")
        # Velocities of Interest
        velocities = (
            self.B.ang_vel_in(self.N),
            self.C.ang_vel_in(self.N),
            self.E.ang_vel_in(self.N),
            self.Bus_cm.vel(self.N),
            self.Panel_Right_cm.vel(self.N),
            self.Panel_Left_cm.vel(self.N)
        )

        # Partial Velocities
        partials = me.partial_velocity(velocities, self.u, self.N)
        w_B_partials, w_C_partials, w_E_partials = partials[0], partials[1], partials[2]
        v_B_partials, v_R_partials, v_L_partials = partials[3], partials[4], partials[5]

        # Inertia Dyadics
        I_b = me.inertia(self.B, 0, 0, self.m_b * self.D**2 / 6)
        I_c = me.inertia(self.C, 0, 0, self.m_r * self.L**2 / 12)
        I_e = me.inertia(self.E, 0, 0, self.m_l * self.L**2 / 12)

        # Accelerations
        alpha_B = self.B.ang_acc_in(self.N)
        alpha_C = self.C.ang_acc_in(self.N)
        alpha_E = self.E.ang_acc_in(self.N)

        a_Bus_cm = self.Bus_cm.acc(self.N)
        a_Panel_Right_cm = self.Panel_Right_cm.acc(self.N)
        a_Panel_Left_cm = self.Panel_Left_cm.acc(self.N)

        # Applied Forces and Torques
        R_B = 0 * self.B.x + 0 * self.B.y
        R_R = 0 * self.C.x + 0 * self.C.y
        R_L = 0 * self.E.x + 0 * self.E.y
        
        T_B = self.tau * self.B.z
        T_R = 0 * self.C.z
        T_L = 0 * self.E.z

        # Inertia Forces and Torques
        Rs_B = -self.m_b * a_Bus_cm
        Rs_R = -self.m_r * a_Panel_Right_cm
        Rs_L = -self.m_l * a_Panel_Left_cm

        Ts_B = -(alpha_B.dot(I_b) + me.cross(self.B.ang_vel_in(self.N), I_b).dot(self.B.ang_vel_in(self.N)))
        Ts_C = -(alpha_C.dot(I_c) + me.cross(self.C.ang_vel_in(self.N), I_c).dot(self.C.ang_vel_in(self.N)))
        Ts_E = -(alpha_E.dot(I_e) + me.cross(self.E.ang_vel_in(self.N), I_e).dot(self.E.ang_vel_in(self.N)))

        # Assemble Generalized Forces
        self.Generalised_Active_Forces = sm.Matrix([0, 0, 0])
        self.Generalised_Inertia_Forces = sm.Matrix([0, 0, 0])

        for i in range(3):
            # Active forces contribution
            Fr_active = (v_B_partials[i].dot(R_B) +
                         v_R_partials[i].dot(R_R) +
                         v_L_partials[i].dot(R_L) +
                         w_B_partials[i].dot(T_B) +
                         w_C_partials[i].dot(T_R) +
                         w_E_partials[i].dot(T_L))
            self.Generalised_Active_Forces[i] = Fr_active

            # Inertia forces contribution
            Fr_inertia = (v_B_partials[i].dot(Rs_B) +
                          v_R_partials[i].dot(Rs_R) +
                          v_L_partials[i].dot(Rs_L) +
                          w_B_partials[i].dot(Ts_B) +
                          w_C_partials[i].dot(Ts_C) +
                          w_E_partials[i].dot(Ts_E))
            self.Generalised_Inertia_Forces[i] = Fr_inertia

    def _formulate_eom(self):
        print("Step 4: Assembling final equations of motion...")
        # Utility dictionaries
        qd_zero = {qdi: 0 for qdi in self.qd}
        ud_zero = {udi: 0 for udi in self.ud}

        # Kinematic differential equation
        fk = sm.Matrix([
            self.u1 - self.q1.diff(),
            self.u2 - self.q2.diff(),
            self.u3 - self.q3.diff(),
        ])

        # Generation of Matrix Mk and vector gk
        self.Mk = fk.jacobian(self.qd)
        self.gk = fk.xreplace(qd_zero)

        # Dynamic differential equation
        kane_eq = (self.Generalised_Active_Forces + self.Generalised_Inertia_Forces)

        # Generation of Matrix Md and vector gd
        self.Md = kane_eq.jacobian(self.ud)
        self.gd = kane_eq.xreplace(ud_zero)
