from collections import deque
import numpy as np
import sympy as sm
import sympy.physics.mechanics as me
from ..beam.cantilever_beam import CantileverBeam


class FlexibleNonSymmetricDynamics:
    def __init__(self, config):
        self.config = config
        me.init_vprinting()

        # ---------- Initialise satellite configuration ----------
        try:
            self.graph = config["adjacency_graph"]
            self.body_names = config["body_names"]
            self.central_body = config["central_body"]
            self.body_type = config["body_type"]
        except KeyError as e:
            raise KeyError(f"Missing config key: {e}")
        
        self.parents = self._parents_from_adjacency(self.graph, self.body_names, self.central_body)

        self.rigid_bodies = {name: None for name in self.body_names if not self.body_type[name].startswith("flexible-")}
        self.flexible_bodies = {name: None for name in self.body_names if self.body_type[name].startswith("flexible-")}
        
        # ---------- Creates satellite symbolic dynalmics  ----------
        self._define_symbols()
        self._define_mode_shapes()
        self._define_kinematics()
        self._define_kinematic_equations()
        self._setup_velocities()
        self._setup_accelerations()
        self._derive_generalized_forces()
        self._formulate_eom()
        self._create_lambdified_functions()

    def _parents_from_adjacency(self, graph, body_names, central_body):
        visited = {body: False for body in body_names}
        parents  = {}
    
        visited[central_body] = True
        parents[central_body] = "N"
        queue = deque([central_body])
        
        while queue:
            current_node = queue.popleft()
            
            for neighbor in graph[current_node]:
                if not visited[neighbor]:
                    visited[neighbor] = True
                    parents[neighbor]  = current_node
                    queue.append(neighbor)
                
        return parents
           
    def _define_symbols(self):
        # ---------- Base generalized coordinates and speeds ----------
        self.q_angle = me.dynamicsymbols('q3')
        self.q_reference = {
            "x": me.dynamicsymbols('q1'),
            "y": me.dynamicsymbols('q2'),
            "theta": self.q_angle
        }
        
        self.u_angle = me.dynamicsymbols('u3')
        self.u_reference = {
            "x": me.dynamicsymbols('u1'),
            "y": me.dynamicsymbols('u2'),
            "theta": self.u_angle
        }
        
         # ---------- Flexible bodies ----------
        eta_list = []
        zeta_list = []

        flexible_types = self.config["flexible_types"]
        for i, body in enumerate(self.flexible_bodies.keys()):
            eta_i = me.dynamicsymbols(f"eta{i+1}")
            eta_list.append(eta_i)
            zeta_i = me.dynamicsymbols(f"zeta{i+1}")
            zeta_list.append(zeta_i)
            self.flexible_bodies[body] = {"eta": eta_i, "zeta": zeta_i, "beam_type": flexible_types[body]}

         # ---------- Auxiliary variables ----------
        # Position along the beam for integration
        self.s = sm.Symbol('s')

        # Time symbol
        self.t = me.dynamicsymbols._t
        
        # ---------- Flat state vectors ----------
        self.q = sm.Matrix([*self.q_reference.values(), *eta_list])
        self.qd = self.q.diff(self.t)
        self.u = sm.Matrix([*self.u_reference.values(), *zeta_list])
        self.ud = self.u.diff(self.t)

        # Zero replacement dictionaries
        self.qd_zero = {qdi: 0 for qdi in self.qd}
        self.ud_zero = {udi: 0 for udi in self.ud}
        
         # ---------- System parameters ----------
        self.p_symbols = {}
        for key, val in self.config["p_values"].items():
            symbol = sm.symbols(key)
            self.p_symbols[key] = symbol

    def _define_mode_shapes(self):
        # ---------- Setup ----------
        beam_params = self.config["beam_parameters"]
        p_values = self.config["p_values"]
        
        for body, values in self.flexible_bodies.items():
            beam_type = values["beam_type"]
            params = beam_params[beam_type]
            if beam_type == "cantilever":
                beam = CantileverBeam(
                    length=p_values["L"],
                    E=p_values["E_mod"],
                    I=p_values["I_area"],
                    n=params["nb_modes"]
                )
            elif beam_type == "clamped-clamped":
                # !!!!!!!!!!!!!!!!!! To be replaced after refactoring ClampedClampedBeam !!!!!!!!!!!!!!!!!!
                # beam = ClampedClampedBeam(
                #     length=p_values["L"],
                #     E=p_values["E_mod"],
                #     I=p_values["I_area"],
                #     n=params["nb_modes"]
                # )
                beam = CantileverBeam(
                    length=p_values["L"],
                    E=p_values["E_mod"],
                    I=p_values["I_area"],
                    n=params["nb_modes"]
                )
            else:
                raise TypeError(f"Unrecognised beam type: {beam_type}")

            self.flexible_bodies[body]["beam"] = beam
            self.flexible_bodies[body]["phi"] = beam.mode_shape_symbolic(self.s, 1)
            self.flexible_bodies[body]["phi_mean"] = beam.mode_shape_mean(params["nb_points"])
            self.flexible_bodies[body]["k_modal"] = beam.modal_stiffness(1)
            

    def _get_offset_vector(self, parent, child):
        parent_type = self.body_type[parent]
        child_type = self.body_type[child]
    
        # default: no offset
        offset = 0 * self.frames[parent].x
    
        # bus (rigid) → panel (flexible-<side>)
        if parent_type.startswith("rigid") and child_type.startswith("flexible-"):
            sign = -1 if child_type.endswith("-left") else 1
            offset = sign * (self.p_symbols["D"] / 2) * self.frames[parent].x

        return offset

    def _define_kinematics(self):
        # ---------- Reference frames ----------
        N = me.ReferenceFrame('N')
        self.frames = {"inertial": N}
        
        for body in self.body_names:
            frame = me.ReferenceFrame(f"frame_{body}")
            self.frames[body] = frame

            angle = self.q_angle + sm.pi if self.body_type[body].endswith('-left') else self.q_angle
            frame.orient_axis(N, angle, N.z)
        
        # ---------- Points ----------
        self.points = {}
        self.inertial_position = {body: None for body in self.body_names}
        
        # Inertial origin point
        self.O = me.Point('O')
        self.O.set_vel(N, 0)
        self.points["N"] = self.O

        # Central point
        self.central_cm = me.Point(f"{self.central_body}_cm")
        self.central_cm.set_pos(self.O, self.q_reference["x"] * N.x + self.q_reference["y"] * N.y)
        self.points[self.central_body] = self.central_cm
        self.inertial_position[self.central_body] = self.central_cm.pos_from(self.O)

        # Remainder of points
        for child, parent in self.parents.items():
            if child == self.central_body:
                continue  # central_cm already placed

            attach = self._get_offset_vector(parent, child)
            parent_point = self.points[parent]
            child_point  = parent_point.locatenew(f"joint_{child}_{parent}", attach)
            self.points[f"joint_{child}_{parent}"] = child_point
        
            if self.body_type[child].startswith("flexible-"):
                frame      =  self.frames[child]
                amplitude  =  self.flexible_bodies[child]["eta"]
                dm         =  self.s * frame.x + self.flexible_bodies[child]["phi"] * amplitude * frame.y
                dm_point   =  self.points[f"joint_{child}_{parent}"].locatenew(f"dm_{child}", dm)
                self.points[f"dm_{child}"] = dm_point
        
                dm_cm = (self.p_symbols["L"] / 2) * frame.x + self.flexible_bodies[child]["phi_mean"] * amplitude * frame.y
                dm_cm_point = self.points[f"joint_{child}_{parent}"].locatenew(f"dm_center_of_mass_{child}", dm_cm)
                self.points[f"dm_center_of_mass_{child}"] = dm_cm_point
        
                if not self.inertial_position[child]:
                    self.inertial_position[child] = dm_cm_point.pos_from(self.O)
            else:
                self.points[child] = child_point

        # System center of mass
        M = sm.Float(0)
        for key, value in self.p_symbols.items():
            if key.startswith("m_"):
                M += value
        
        r_G = 0.0
        for key, value in self.inertial_position.items():
            r_G = r_G + self.p_symbols[f"m_{key}"] * value
        r_G /= M

        G = self.O.locatenew('G', r_G)
        self.r_GB = self.points[self.central_body].pos_from(G)
        self.points["center_of_mass"] = G


    def _define_kinematic_equations(self):
        # ---------- Kinematical differential equations ----------
        self.fk = self.q.diff(self.t) - self.u
        # self.fk = self.qd - self.u

        self.Mk = self.fk.jacobian(self.qd)
        self.gk = self.fk.xreplace(self.qd_zero)
        
        qd_sol = -self.Mk.LUsolve(self.gk)
        self.qd_repl = dict(zip(self.qd, qd_sol))
        self.qdd_repl = {q.diff(self.t): u.diff(self.t) for q, u in self.qd_repl.items()}

    def _setup_velocities(self):
        # ---------- Setup ----------
        self.angular_velocities = {name: None for name in self.body_names}
        self.linear_velocities = {name: None for name in self.body_names}
        inertial_frame = self.frames["inertial"]

        # ---------- Angular velocities ----------
        for name, frame in self.frames.items():
            if name != "inertial":
                frame.set_ang_vel(inertial_frame, self.u_angle * inertial_frame.z)
                self.angular_velocities[name] = frame.ang_vel_in(inertial_frame).xreplace(self.qdd_repl).xreplace(self.qd_repl)
        
        # ---------- Central Body velocity ----------
        self.points[self.central_body].set_vel(
            inertial_frame, self.u_reference["x"] * inertial_frame.x + self.u_reference["y"] * inertial_frame.y
        )
        self.linear_velocities[self.central_body] = self.points[self.central_body].vel(inertial_frame).xreplace(self.qdd_repl).xreplace(self.qd_repl)
        
        # ---------- Translational velocities ----------
        for child, parent in self.parents.items():
            if child == self.central_body:
                continue

            parent_point = self.points[parent]
            parent_frame = self.frames[parent]
        
            joint_name = f"joint_{child}_{parent}"
            child_point = self.points.get(joint_name)
            if child_point is None:
                raise KeyError(f"Missing joint point '{joint_name}' for child '{child}' (check _define_kinematics).")
        
            child_point.v2pt_theory(parent_point, inertial_frame, parent_frame)
        
            if self.body_type[child].startswith("flexible-"):
                self.points[f"dm_{child}"].set_vel(
                    self.frames[child], self.flexible_bodies[child]["phi"] * self.flexible_bodies[child]["zeta"] * self.frames[child].y
                )
                dm_vel = self.points[f"dm_{child}"].v1pt_theory(child_point, inertial_frame, self.frames[child])
                self.linear_velocities[child] = dm_vel.xreplace(self.qdd_repl).xreplace(self.qd_repl)
            else:
                self.linear_velocities[child] = child_point.vel(inertial_frame).xreplace(self.qdd_repl).xreplace(self.qd_repl)

        # ---------- Flexible center of mass velocities ----------
        # !!! Not used at the moment --> TBD if it is to be kept
        for item in self.flexible_bodies:
            self.points[f"dm_center_of_mass_{item}"].set_vel(
                self.frames[item], self.flexible_bodies[child]["phi_mean"] * self.flexible_bodies[item]["zeta"] * self.frames[item].y
            )
            dm_center_of_mass_velocity = self.points[f"dm_center_of_mass_{item}"].v1pt_theory(
                self.points[f"joint_{item}_{self.central_body}"], inertial_frame, self.frames[item]
            )

    def _setup_accelerations(self):
        # ---------- Setup ----------
        self.angular_accelerations = {name: None for name in self.body_names}
        self.linear_accelerations = {name: None for name in self.body_names}
        inertial_frame = self.frames["inertial"]

        # ---------- Angular accelerations ----------
        for name, frame in self.frames.items():
            if name != "inertial":
                self.angular_accelerations[name] = frame.ang_acc_in(inertial_frame).xreplace(self.qdd_repl).xreplace(self.qd_repl)
        
        # ---------- Central Body acceleration ----------
        self.linear_accelerations[self.central_body] = self.points[self.central_body].acc(inertial_frame).xreplace(self.qdd_repl).xreplace(self.qd_repl)

        # ---------- Translational accelerations ----------
        for child, parent in self.parents.items():
            if child == self.central_body:
                continue

            if self.body_type[child].startswith("flexible-"):
                self.linear_accelerations[child] = self.points[f"dm_{child}"].acc(inertial_frame).xreplace(self.qdd_repl).xreplace(self.qd_repl)
            else:
                self.linear_accelerations[child] = self.points[child].acc(inertial_frame).xreplace(self.qdd_repl).xreplace(self.qd_repl)

    def _get_forces(self):
        forces = {}
        for name, frame in self.frames.items():
            if name == "inertial":
                continue
            
            forces[name] = 0 * frame.x + 0 * frame.y
        
        return forces

    def _get_torques(self):
        torques = {}
        for name, frame in self.frames.items():
            if name == "inertial":
                continue

            if name == self.central_body:
                torques[name] = self.p_symbols["tau"] * frame.z
            else:
                torques[name] = 0 * frame.z
        
        return torques

    def _define_generalized_active_forces(self):
        # ---------- Setup ----------
        self.state_reference_dimension = len(self.u_reference)
        self.state_dimension = len(self.u)
        self.generalised_active_forces = sm.zeros(self.state_dimension, 1)
        
        for i in range(self.state_dimension):
            for body in self.parents.keys():
                v_partial = self.partial_linear_velocities[body][i]
                w_partial = self.partial_angular_velocities[body][i]
                
                force = self.forces[body]
                torque = self.torques[body]
    
                self.generalised_active_forces[i] += v_partial.dot(force) + w_partial.dot(torque)

        # ---------- Modal stiffness for the first mode of a cantilever beam ----------
        k_modal = self.flexible_bodies["panel_1"]["k_modal"]

        # ---------- Strain potential energy stored in the flexible panels ---------- 
        V_strain = sm.Float(0)
        for body, states in self.flexible_bodies.items():
            V_strain += (1/2) * k_modal * states["eta"]**2
        
        i = 0
        for body, states in self.flexible_bodies.items():
            self.generalised_active_forces[i+self.state_reference_dimension] += - V_strain.diff(states["eta"])
            i += 1


    def _define_generalized_inertia_forces(self):
        # ---------- Setup ----------
        self.generalised_inertia_forces = sm.zeros(self.state_dimension, 1)
        
        # ---------- Contribution from rigid bodies ---------- 
        for i in range(self.state_dimension):
            for body in self.rigid_bodies.keys():
                v_partial = self.partial_linear_velocities[body][i]
                w_partial = self.partial_angular_velocities[body][i]
            
                I_bus = (self.p_symbols[f"m_{body}"] * self.p_symbols["D"]**2) / 12
                Inertia_matrix = me.inertia(self.frames[body], I_bus, I_bus, 2 * I_bus)
    
                body_linear_acceleration = self.linear_accelerations[body]
                body_angular_velocity = self.angular_velocities[body]
                body_angular_acceleration = self.angular_accelerations[body]
                
                R_star_body = -self.p_symbols[f"m_{body}"] * body_linear_acceleration
                T_star_body = -(body_angular_acceleration.dot(Inertia_matrix) \
                         + me.cross(body_angular_velocity, Inertia_matrix.dot(body_angular_velocity)))

                self.generalised_inertia_forces[i] += v_partial.dot(R_star_body) + w_partial.dot(T_star_body)
           
        # ---------- Contribution from flexible bodies ---------- 
        for i in range(self.state_dimension):
            for body in self.flexible_bodies.keys():
                v_partial = self.partial_linear_velocities[body][i]
                w_partial = self.partial_angular_velocities[body][i]
            
                mu_body = self.p_symbols[f"m_{body}"] / self.p_symbols["L"]

                body_linear_acceleration = self.linear_accelerations[body]
                
                R_star_dm_body = -(mu_body * body_linear_acceleration)

                integrand_body = v_partial.dot(R_star_dm_body)
                R_star_body = sm.integrate(integrand_body, (self.s, 0, self.p_symbols["L"]))
            
                self.generalised_inertia_forces[i] += R_star_body

    def _derive_generalized_forces(self):
        # ---------- Partial velocities ----------
        partial_angular_velocities = me.partial_velocity(
            self.angular_velocities.values(), self.u, self.frames["inertial"]
        )
        self.partial_angular_velocities = dict(zip(self.angular_velocities.keys(), partial_angular_velocities))

        partial_linear_velocities = me.partial_velocity(
            self.linear_velocities.values(), self.u, self.frames["inertial"]
        )
        self.partial_linear_velocities = dict(zip(self.linear_velocities.keys(), partial_linear_velocities))

        # ---------- Forces and torques acting on the system ---------- 
        self.forces = self._get_forces()
        self.torques = self._get_torques()

        self._define_generalized_active_forces()
        self._define_generalized_inertia_forces()
        
    def _formulate_eom(self):
        # ---------- Dynamic differential equation ---------- 
        self.kane_eq = (self.generalised_active_forces + self.generalised_inertia_forces)

        # ---------- Generation of Matrix Md and vector gd ---------- 
        self.Md = self.kane_eq.jacobian(self.ud)
        self.Md.simplify()
        self.gd = self.kane_eq.xreplace(self.ud_zero)

    def _create_lambdified_functions(self):
        # ---------- Create lambdified functions matching the notebook exactly ---------- 
        self.eval_kinematics = sm.lambdify((self.q, self.u, self.p_symbols.values()), (self.Mk, self.gk))
        self.eval_differentials = sm.lambdify((self.q, self.u, self.p_symbols.values()), (self.Md, self.gd))

    def get_parameter_values(self):
        return self.config['p_values'].values()

    def get_initial_conditions(self):
        # ---------- Setup ---------- 
        x0 = np.zeros(2 * self.state_dimension)
        
        # ---------- Extract initial states ---------- 
        initial_states = self.config.get("q_initial", {})
        for i, value in enumerate(initial_states.values()):
            x0[i] = value
        
        # ---------- Extract initial speeds ----------
        initial_speeds = self.config.get('initial_speeds', {})
        for i, value in enumerate(initial_speeds.values()):
            x0[i+self.state_dimension+2] = value

        # ---------- Extract parameters ---------- 
        parameters = self.config.get("p_values", {})
        M = 0.0
        for key, value in parameters.items():
            if key.startswith("m_"):
                M += value

        # ---------- Center of mass position vector ---------- 
        central_body_frame = self.frames[self.central_body]
        rho = self.r_GB.express(central_body_frame).simplify()
        self.rho_vector = sm.Matrix([
            [rho.dot(central_body_frame.x)],
            [rho.dot(central_body_frame.y)],
        ])

        # ---------- Skew matrix S ---------- 
        S = sm.Matrix([
            [0, -1],
            [1,  0],
        ])

        # ---------- Rotation matrix R_theta ---------- 
        R_theta = np.array([
            [np.cos(initial_states["q3"]), -np.sin(initial_states["q3"])],
            [np.sin(initial_states["q3"]),  np.cos(initial_states["q3"])]
        ])
        
        # ---------- Calculate initial generalized speeds constraints ---------- 
        M_symbol = sm.Float(0)
        for key, value in self.p_symbols.items():
            if key.startswith("m_"):
                M_symbol += value
                
        self.rho_derivative = sm.Matrix([[0], [0]])
        for key, value in self.flexible_bodies.items():
            self.rho_derivative = self.rho_derivative + self.p_symbols[f"m_{key}"] * value["zeta"] * value["phi_mean"] * sm.Matrix([
                    [self.frames[key].y.dot(self.frames[self.central_body].x)],
                    [self.frames[key].y.dot(self.frames[self.central_body].y)],
                ])
        self.rho_derivative = self.rho_derivative / M

        self.initial_generalised_speeds_constraints = initial_speeds["u3"] * S @ R_theta @ self.rho_vector + R_theta @ self.rho_derivative

        state_symbols = [state for state in self.q]
        speed_symbols = [self.u_angle] + [param["zeta"] for param in self.flexible_bodies.values()]
        param_symbols  = [param for key, param in self.p_symbols.items() if key != "tau"]
        
        arg_syms = tuple(state_symbols + speed_symbols + param_symbols)

        self.u_init_func = sm.lambdify(
            arg_syms, 
            [self.initial_generalised_speeds_constraints[0], self.initial_generalised_speeds_constraints[1]], 
            'numpy'
        )

        args = list(initial_states.values()) + list(initial_speeds.values()) + [value for key, value in parameters.items() if key != "tau"]
        u1_consistent, u2_consistent = self.u_init_func(*args)
        
        x0[self.state_dimension] = u1_consistent
        x0[self.state_dimension+1] = u2_consistent

        print(f"\nInitial conditions set (with momentum conservation):")
        
        output_string = (
            f"  Positions: q1={initial_states["q1"]:.3f}, q2={initial_states["q2"]:.3f}, "
            f"q3={np.rad2deg(initial_states["q3"]):.3f}°, "
        )
        for i in range(len(self.flexible_bodies.keys())):
            output_string = output_string + f"eta{i+1}={initial_states[f"eta{i+1}"]:.3f}, \n"
        print(output_string)

        
        output_string = f"  Velocities: u1={u1_consistent:.6f}, u2={u2_consistent:.6f}, u3={initial_speeds["u3"]:.3f}, "
        for i in range(len(self.flexible_bodies.keys())):
            output_string = output_string + f"zeta{i+1}={initial_speeds[f"zeta{i+1}"]:.3f}, \n"
        print(output_string)
        
        return x0
