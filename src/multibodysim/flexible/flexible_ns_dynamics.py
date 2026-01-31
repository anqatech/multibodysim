from collections import deque
import numpy as np
import sympy as sm
import sympy.physics.mechanics as me
from ..beam.cantilever_beam import CantileverBeam
from ..beam.clamped_clamped_beam import ClampedClampedBeam


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
        eta_all = []
        zeta_all = []
        
        flexible_types = self.config["flexible_types"]
        beam_params = self.config["beam_parameters"]
        
        # keep a flat ordering + indices for (body,mode)
        self.flex_eta_index  = {} 
        self.flex_zeta_index = {}
        flat_eta_idx = 0
        flat_zeta_idx = 0
        
        for i_body, body in enumerate(self.flexible_bodies.keys()):
            btype = flexible_types[body]
            n_modes = beam_params[btype]["nb_modes"]
        
            eta_list  = [me.dynamicsymbols(f"eta{i_body+1}_{k+1}")  for k in range(n_modes)]
            zeta_list = [me.dynamicsymbols(f"zeta{i_body+1}_{k+1}") for k in range(n_modes)]
        
            self.flexible_bodies[body] = {
                "beam_type": btype,
                "eta_list": eta_list,
                "zeta_list": zeta_list
            }
        
            # record indices in the same order we extend the flat vectors
            for k in range(n_modes):
                self.flex_eta_index[(body, k)]  = flat_eta_idx
                self.flex_zeta_index[(body, k)] = flat_zeta_idx
                flat_eta_idx  += 1
                flat_zeta_idx += 1
        
            eta_all.extend(eta_list)
            zeta_all.extend(zeta_list)

        # ---------- Auxiliary variables ----------
        # Position along the beam for integration
        self.s = sm.Symbol('s')

        # Time symbol
        self.t = me.dynamicsymbols._t
        
        # # ---------- Flat state vectors ----------
        self.q = sm.Matrix([*self.q_reference.values(), *eta_all])
        self.qd = self.q.diff(self.t)
        self.u = sm.Matrix([*self.u_reference.values(), *zeta_all])
        self.ud = self.u.diff(self.t)

        # Zero replacement dictionaries
        self.qd_zero = {qdi: 0 for qdi in self.qd}
        self.ud_zero = {udi: 0 for udi in self.ud}
        
         # ---------- System parameters ----------
        self.p_symbols = {}
        for key, val in self.config["p_values"].items():
            symbol = sm.symbols(key)
            self.p_symbols[key] = symbol

        # ---------- Optional per-body torque actuators ----------
        self.torque_symbols = {}

        for body in self.rigid_bodies.keys():
            symbol = sm.symbols(f"tau_{body}")
            self.torque_symbols[body] = symbol

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
                beam = ClampedClampedBeam(
                    length=p_values["L"],
                    E=p_values["E_mod"],
                    I=p_values["I_area"],
                    n=params["nb_modes"]
                )
            else:
                raise TypeError(f"Unrecognised beam type: {beam_type}")

            self.flexible_bodies[body]["beam"] = beam
            n_modes = self.config["beam_parameters"][beam_type]["nb_modes"]
            nb_pts  = self.config["beam_parameters"][beam_type]["nb_points"]
            
            phi_list       = [beam.mode_shape_symbolic(self.s, k+1) for k in range(n_modes)]
            phi_mean_list  = [beam.mode_shape_mean(nb_pts, mode=k+1) for k in range(n_modes)]
            k_modal_list   = [beam.modal_stiffness(k+1) for k in range(n_modes)]
            
            self.flexible_bodies[body]["phi_list"]      = phi_list
            self.flexible_bodies[body]["phi_mean_list"] = phi_mean_list
            self.flexible_bodies[body]["k_modal_list"]  = k_modal_list

    def _get_offset_vector(self, parent, child):
        parent_type = self.body_type[parent]
        child_type = self.body_type[child]
    
        # ---------- Default: no offset ---------- 
        offset = 0 * self.frames[parent].x

        # Convenience flags (e.g., "flexible-left", "flexible-right")
        parent_is_rigid = parent_type.startswith("rigid-")
        parent_is_flex  = parent_type.startswith("flexible-")
        child_is_rigid  = child_type.startswith("rigid-")
        child_is_flex   = child_type.startswith("flexible-")

        # ---- Unsupported case: panel attaching to panel ----
        if parent_is_flex and child_is_flex:
            raise NotImplementedError(
                f"Offset between two flexible bodies (e.g., panel→panel) "
                f"not implemented: parent='{parent_type}', child='{child_type}'."
            )

        # ---- Unsupported case: bus attaching to bus ----
        if parent_is_rigid and child_is_rigid:
            raise NotImplementedError(
                f"Offset between two rigid bodies not implemented: "
                f"parent='{parent_type}', child='{child_type}'."
            )
        
        # ---- Supported case: bus attaching to panel ----
        if parent_is_flex and child_is_rigid:
            frame    = self.frames[parent]
            L        = self.p_symbols["L"]
            eta_list = self.flexible_bodies[parent]["eta_list"]
            phi_list = self.flexible_bodies[parent]["phi_list"]

            phi_L = 0
            for phi_k, eta_k in zip(phi_list, eta_list):
                phi_L += phi_k.subs(self.s, L) * eta_k

            offset = L * frame.x + phi_L * frame.y            
        
        # ---- Supported case: panel attaching to bus ----
        if parent_is_rigid and child_is_flex:
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

            parent_type = self.body_type[parent]
            child_type = self.body_type[child]

            attach = self._get_offset_vector(parent, child)

            if parent_type.startswith("rigid-"):
                parent_point = self.points[parent]
            else:
                joint_name = "joint_" + parent + "_" + self.parents[parent]
                parent_point = self.points[joint_name]

            child_point  = parent_point.locatenew(f"joint_{child}_{parent}", attach)
            self.points[f"joint_{child}_{parent}"] = child_point
        
            if self.body_type[child].startswith("flexible-"):
                frame     =  self.frames[child]
                eta_list  = self.flexible_bodies[child]["eta_list"]
                phi_list  = self.flexible_bodies[child]["phi_list"]
                phi_sum   = sum(phi_k * eta_k for (phi_k, eta_k) in zip(phi_list, eta_list))
                
                dm   = self.s * frame.x + phi_sum * frame.y
                dm_point = self.points[f"joint_{child}_{parent}"].locatenew(f"dm_{child}", dm)
                self.points[f"dm_{child}"] = dm_point
                
                phi_mean_list = self.flexible_bodies[child]["phi_mean_list"]
                phi_mean_sum  = sum(pm * eta_k for (pm, eta_k) in zip(phi_mean_list, eta_list))
                dm_cm = (self.p_symbols["L"] / 2) * frame.x + phi_mean_sum * frame.y
                dm_cm_point = self.points[f"joint_{child}_{parent}"].locatenew(f"dm_center_of_mass_{child}", dm_cm)
                self.points[f"dm_center_of_mass_{child}"] = dm_cm_point

                if not self.inertial_position[child]:
                    self.inertial_position[child] = dm_cm_point.pos_from(self.O)
            else:
                frame  =  self.frames[child]
                sign = -1 if child_type.endswith("-left") else 1
                offset = sign * (self.p_symbols["D"] / 2) * frame.x

                child_cm_point  = self.points[f"joint_{child}_{parent}"].locatenew(f"{child}_cm", offset)
                self.points[child] = child_cm_point

                if not self.inertial_position[child]:
                    self.inertial_position[child] = child_cm_point.pos_from(self.O)

        # System center of mass
        M = sm.Float(0)
        for key, value in self.p_symbols.items():
            if key.startswith("m_"):
                M += value
        
        r_G = 0.0
        for key, value in self.inertial_position.items():
            r_G = r_G + self.p_symbols[f"m_{key}"] * value
        r_G /= M

        self.G = self.O.locatenew('G', r_G)
        self.r_GB = self.points[self.central_body].pos_from(self.G)
        self.points["center_of_mass"] = self.G


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

            parent_type = self.body_type[parent]
            parent_is_rigid = parent_type.startswith("rigid-")
            parent_is_flex  = parent_type.startswith("flexible-")

            if parent_is_flex:
                parent_of_parent = self.parents[parent]
                reference_joint_name = f"joint_{parent}_{parent_of_parent}"
                reference_joint_point = self.points[reference_joint_name]
                child_joint_point = self.points[f"joint_{child}_{parent}"]
                child_joint_velocity = child_joint_point.v1pt_theory(reference_joint_point, inertial_frame, self.frames[parent])

                child_point = self.points[child]
                child_point.v2pt_theory(child_joint_point, inertial_frame, self.frames[child])
                self.linear_velocities[child] = child_point.vel(inertial_frame).xreplace(self.qdd_repl).xreplace(self.qd_repl)

            elif parent_is_rigid:
                parent_point = self.points[parent]
                parent_frame = self.frames[parent]
            
                joint_name = f"joint_{child}_{parent}"
                child_point = self.points[joint_name]
                if child_point is None:
                    raise KeyError(f"Missing joint point '{joint_name}' for child '{child}' (check _define_kinematics).")
            
                child_point.v2pt_theory(parent_point, inertial_frame, parent_frame)
            
                if self.body_type[child].startswith("flexible-"):
                    phi_list  = self.flexible_bodies[child]["phi_list"]
                    zeta_list = self.flexible_bodies[child]["zeta_list"]
                    phi_zeta_sum = sum(phi_k * zeta_k for (phi_k, zeta_k) in zip(phi_list, zeta_list))
                    
                    self.points[f"dm_{child}"].set_vel(
                        self.frames[child], phi_zeta_sum * self.frames[child].y
                    )

                    dm_vel = self.points[f"dm_{child}"].v1pt_theory(child_point, inertial_frame, self.frames[child])
                    self.linear_velocities[child] = dm_vel.xreplace(self.qdd_repl).xreplace(self.qd_repl)
                else:
                    self.linear_velocities[child] = child_point.vel(inertial_frame).xreplace(self.qdd_repl).xreplace(self.qd_repl)
            else:
                raise KeyError(f"Unknown parent type for '{parent}': {parent_type}")

        # ---------- Flexible center of mass velocities ----------
        # !!! Not used at the moment --> TBD if it is to be kept
        for item, fb in self.flexible_bodies.items():
            phi_mean_list = fb["phi_mean_list"]
            zeta_list     = fb["zeta_list"]
            phi_mean_zeta_sum = sum(pm * z for pm, z in zip(phi_mean_list, zeta_list))
        
            self.points[f"dm_center_of_mass_{item}"].set_vel(
                self.frames[item], phi_mean_zeta_sum * self.frames[item].y
            )
            parent = self.parents[item]  # not always central body once you generalize!
            _ = self.points[f"dm_center_of_mass_{item}"].v1pt_theory(
                self.points[f"joint_{item}_{parent}"], inertial_frame, self.frames[item]
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

    def _get_external_forces(self):
        forces = {}
        for name, frame in self.frames.items():
            if name == "inertial":
                continue
            
            forces[name] = 0 * frame.x + 0 * frame.y
        
        return forces

    def _get_external_torques(self):
        torques = {}
        for name, frame in self.frames.items():
            if name == "inertial":
                continue

            if name in self.torque_symbols.keys():
                torques[name] = self.torque_symbols[name] * frame.z
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

        # ---- Kepler-only gravity: apply net gravity at COM ----

        # Total mass (symbolic)
        M_symbol = sm.Float(0)
        for key, value in self.p_symbols.items():
            if key.startswith("m_"):
                M_symbol += value

        mu = self.p_symbols.get("planet_mu", None)

        # Net gravitational force at COM
        self.r_G = self.points["center_of_mass"].pos_from(self.O).express(self.frames["inertial"])
        r2 = self.r_G.dot(self.r_G)
        r = sm.sqrt(r2)
        F_g = -mu * M_symbol * self.r_G / (r**3)

        self.v_G = self.r_G.dt(self.frames["inertial"]).xreplace(self.qd_repl)
        vG_partials = me.partial_velocity([self.v_G], self.u, self.frames["inertial"])[0]

        for i in range(self.state_dimension):
            self.generalised_active_forces[i] += vG_partials[i].dot(F_g)
        
        # ---------- Strain potential energy stored in the flexible panels ---------- 
        V_strain = sm.S.Zero
        for body, fb in self.flexible_bodies.items():
            eta_list     = fb["eta_list"]
            k_modal_list = fb["k_modal_list"]
            for k, (eta_k, Kk) in enumerate(zip(eta_list, k_modal_list)):
                V_strain += sm.Rational(1, 2) * Kk * eta_k**2
        
        # ---------- Flexible contribution to generalised active forces ---------- 
        for body, fb in self.flexible_bodies.items():
            eta_list     = fb["eta_list"]
            k_modal_list = fb["k_modal_list"]
            for k, (eta_k, Kk) in enumerate(zip(eta_list, k_modal_list)):
                flat_eta = self.flex_eta_index[(body, k)]
                row = self.state_reference_dimension + flat_eta
                self.generalised_active_forces[row] += -(Kk * eta_k)

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
        self.forces = self._get_external_forces()
        self.torques = self._get_external_torques()

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
        self.eval_kinematics = sm.lambdify(
            (self.q, self.u, self.p_symbols.values(), self.torque_symbols.values()), 
            (self.Mk, self.gk), 
            cse=True
        )
        self.eval_differentials = sm.lambdify(
            (self.q, self.u, self.p_symbols.values(), self.torque_symbols.values()), 
            (self.Md, self.gd), 
            cse=True
        )

        self.rG_func = sm.lambdify(
            (self.q, self.u, self.p_symbols.values()),
            [
                self.r_G.dot(self.frames["inertial"].x),
                self.r_G.dot(self.frames["inertial"].y),
            ],
            "numpy",
            cse=True
        )

        self.vG_func = sm.lambdify(
            (self.q, self.u, self.p_symbols.values()),
            [
                self.v_G.dot(self.frames["inertial"].x),
                self.v_G.dot(self.frames["inertial"].y),
            ],
            "numpy",
            cse=True
        )

    def get_parameter_values(self):
        return self.config["p_values"].values()
    
    def get_torque_values(self):
        torques = self.config["torques"]

        torque_values = []
        for body in self.rigid_bodies:
            torque = torques[body]
            torque_values.append(torque)

        return torque_values

    def get_torque_weights(self):
        torque_weights = self.config["torque_weights"].values()

        return torque_weights

    def get_initial_conditions(self):
        # ---------- Setup ---------- 
        x0 = np.zeros(2 * self.state_dimension)
        
        # ---------- Extract initial states ---------- 
        initial_states = {}
        for state_symbol in self.q:
            name = state_symbol.name
            initial_states[name] = float(self.config["q_initial"].get(name, 0.0))

        for i, value in enumerate(initial_states.values()):
            x0[i] = value
        
        # ---------- Extract initial speeds ----------
        initial_speeds = {}
        for speed_symbol in self.u:
            name = speed_symbol.name
            initial_speeds[name] = float(self.config["initial_speeds"].get(name, 0.0))

        for i, value in enumerate(initial_speeds.values()):
            x0[i+self.state_dimension] = value

        # ---------- Extract parameters ---------- 
        parameters = self.config["p_values"]
        M = 0.0
        for key, value in parameters.items():
            if key.startswith("m_"):
                M += value

        # ---------- Center of mass position vector (central frame) ---------- 
        central_body_frame = self.frames[self.central_body]
        rho = self.r_GB.express(central_body_frame).simplify()
        self.rho_vector = sm.Matrix([
            [rho.dot(central_body_frame.x)],
            [rho.dot(central_body_frame.y)],
        ])

        # ---------- Evaluate rho(eta0) numerically ----------
        # rho_vector is expressed in the central body frame
        rho_func = sm.lambdify(
            tuple(list(self.q) + list(self.p_symbols.values())),
            self.rho_vector,
            "numpy"
        )

        # Build q and p arguments using the current config values
        q_args = []
        for q_symbol in self.q:
            name = q_symbol.name
            q_args.append( initial_states[name])

        p_args = []
        for p_symbol in self.p_symbols.values():
            pname = p_symbol.name
            p_args.append(self.config["p_values"][pname])

        rho0 = np.array(rho_func(*(q_args + p_args)), dtype=float)

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

        # ---------- Keplerian COM initial position/velocity ----------
        mu_planet = float(self.config["p_values"]["planet_mu"])
        a = float(self.config["p_values"]["orbit_semi_major_axis"])
        e = float(self.config["p_values"]["orbit_eccentricity"])

        # True anomaly at t0 by convention
        initial_true_anomaly = 0.0

        # Kepler radius at t0
        r0 = a * (1.0 - e**2) / (1.0 + e * np.cos(initial_true_anomaly))

        # Kepler speed components at t0 (planar)
        h = np.sqrt(mu_planet * a * (1.0 - e**2))
        vG0 = np.array([
            [-(mu_planet / h) * e * np.sin(initial_true_anomaly)],
            [ (mu_planet / h) * (1.0 + e * np.cos(initial_true_anomaly))],
        ], dtype=float)

        rG0 = np.array([[r0], [0.0]], dtype=float)

        # ---------- Set central-bus position from COM Kepler ICs ----------
        rB0 = rG0 + R_theta @ rho0

        # overwrite q1,q2 in both x0 and the local initial_states dict
        initial_states["q1"] = float(rB0[0, 0])
        initial_states["q2"] = float(rB0[1, 0])
        x0[0] = initial_states["q1"]
        x0[1] = initial_states["q2"]
        
        # ---------- Calculate initial generalized speeds constraints ---------- 
        M_symbol = sm.Float(0)
        for key, value in self.p_symbols.items():
            if key.startswith("m_"):
                M_symbol += value
                
        self.rho_derivative = sm.Matrix([[0], [0]])
        for body, fb in self.flexible_bodies.items():
            pm_list = fb["phi_mean_list"]
            z_list  = fb["zeta_list"]
            pmz_sum = sum(pm * z for pm, z in zip(pm_list, z_list))
            self.rho_derivative += self.p_symbols[f"m_{body}"] * pmz_sum * sm.Matrix([
                [self.frames[body].y.dot(self.frames[self.central_body].x)],
                [self.frames[body].y.dot(self.frames[self.central_body].y)],
            ])
        
        self.rho_derivative = self.rho_derivative / M
        
        self.initial_generalised_speeds_constraints = initial_speeds["u3"] * S @ R_theta @ self.rho_vector + R_theta @ self.rho_derivative

        state_symbols = [state for state in self.q]
        speed_symbols = [self.u_angle] + [z for fb in self.flexible_bodies.values() for z in fb["zeta_list"]]
        param_symbols  = [param for key, param in self.p_symbols.items() if key != "tau"]
        
        arg_syms = tuple(state_symbols + speed_symbols + param_symbols)

        self.u_init_func = sm.lambdify(
            arg_syms, 
            [self.initial_generalised_speeds_constraints[0], self.initial_generalised_speeds_constraints[1]], 
            "numpy"
        )

        # 1) states in the order of self.q
        state_args = []
        for q_symbol in self.q:                       
            name = str(q_symbol).replace("(t)", "")
            state_args.append(initial_states[name])
        
        # 2) speeds in the order of speed_symbols = [u3] + all zeta_list
        speed_args = []
        speed_args.append(self.config["initial_speeds"]["u3"])
        for fb in self.flexible_bodies.values():
            for z_symbol in fb["zeta_list"]:
                z_name = str(z_symbol).replace("(t)", "")
                speed_args.append(self.config["initial_speeds"][z_name])
        
        # 3) parameters in the SAME order used to build param_symbols
        param_args = []
        for key, symbol in self.p_symbols.items():
            if key == "tau":
                continue
            param_args.append(self.config["p_values"][key])
        
        args = state_args + speed_args + param_args

        u1_offset, u2_offset = self.u_init_func(*args)

        u1_consistent = float(vG0[0, 0] + u1_offset)
        u2_consistent = float(vG0[1, 0] + u2_offset)

        x0[self.state_dimension] = u1_consistent
        x0[self.state_dimension+1] = u2_consistent

        print("\nInitial conditions set (with momentum conservation):\n")

        # --- Positions ---
        pos_parts = [
            f"q1={initial_states["q1"]:.3f}",
            f"q2={initial_states["q2"]:.3f}",
            f"q3={np.degrees(initial_states["q3"]):.3f}°",
        ]
        for body, fb in self.flexible_bodies.items():
            for eta_sym in fb["eta_list"]:
                name = str(eta_sym).replace("(t)", "")
                pos_parts.append(f"{name}={initial_states[name]:.3f}")
        
        print("  Positions: " + ", ".join(pos_parts) + "\n")
        
        # --- Velocities ---
        vel_parts = [
            f"u1={u1_consistent:.6f}",
            f"u2={u2_consistent:.6f}",
            f"u3={initial_speeds["u3"]:.3f}",
        ]
        for body, fb in self.flexible_bodies.items():
            for zeta_sym in fb["zeta_list"]:
                name = str(zeta_sym).replace("(t)", "")
                vel_parts.append(f"{name}={initial_speeds[name]:.3f}")
        
        print("  Velocities: " + ", ".join(vel_parts) + "\n")
        
        return x0
