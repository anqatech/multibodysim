class FlexibleNSPlantView:
    def __init__(self, dynamics, p_vals):
        self.dynamics = dynamics
        self.p_vals = p_vals

        q_ref = list(dynamics.q_reference.keys())
        u_ref = list(dynamics.u_reference.keys())
        self.i_theta_q = q_ref.index("theta")
        self.i_theta_u = u_ref.index("theta")

    def theta(self, q):
        return q[self.i_theta_q]

    def theta_dot(self, u):
        return u[self.i_theta_u]

    def J_theta(self, Md):
        return float(Md[self.i_theta_u, self.i_theta_u])

    def com_state(self, q, u):
        rGx, rGy = self.dynamics.rG_func(q, u, self.p_vals)
        vGx, vGy = self.dynamics.vG_func(q, u, self.p_vals)
        return float(rGx), float(rGy), float(vGx), float(vGy)


class MultiAnglePlantView:
    def __init__(self, dynamics, parameter_values):
        self.dynamics = dynamics
        self.parameter_values = parameter_values

        q_ref = list(dynamics.q_reference.keys())
        u_ref = list(dynamics.u_reference.keys())
        self.i_theta_q = q_ref.index(dynamics.central_body)
        self.i_theta_u = u_ref.index(dynamics.central_body)

    def theta(self, q):
        return q[self.i_theta_q]

    def theta_dot(self, u):
        return u[self.i_theta_u]

    def J_theta(self, Md):
        return float(Md[self.i_theta_u, self.i_theta_u])

    def com_state(self, q, u):
        rG = self.dynamics.rG_func(q, u, self.parameter_values)
        vG = self.dynamics.vG_func(q, u, self.parameter_values)

        return (
            float(rG[0]),
            float(rG[1]),
            float(vG[0]),
            float(vG[1]),
        )
