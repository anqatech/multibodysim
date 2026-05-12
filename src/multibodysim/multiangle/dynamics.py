from __future__ import annotations

from collections import deque
import re

import numpy as np
import sympy as sm
import sympy.physics.mechanics as me

from ..beam.cantilever_beam import CantileverBeam
from ..beam.clamped_clamped_beam import ClampedClampedBeam


class MultiAngleFlexibleDynamics:
    def __init__(self, config: dict):
        self.config = config
        me.init_vprinting()

        try:
            self.graph = config["adjacency_graph"]
            self.body_names = config["body_names"]
            self.central_body = config["central_body"]
            self.body_type = config["body_type"]
            self.flexible_types = config["flexible_types"]
            self.beam_parameters = config["beam_parameters"]
            self.parameter_values = config["parameters"]
            self.enable_gravity_gradient = config.get("enable_gravity_gradient", False)
        except KeyError as exc:
            raise KeyError(f"Missing config key: {exc}") from exc

        self.flexible_inertia_integration = (
            self._normalise_flexible_inertia_integration_config(
                config.get("flexible_inertia_integration", {})
            )
        )

        self.parents = self._parents_from_adjacency(
            self.graph, self.body_names, self.central_body
        )
        self.rigid_body_names = [
            name
            for name in self.body_names
            if self.body_type[name].startswith("rigid-")
        ]
        self.flexible_body_names = [
            name
            for name in self.body_names
            if self.body_type[name].startswith("flexible-")
        ]

        if self.central_body not in self.rigid_body_names:
            raise ValueError("central_body must be one of the rigid bodies.")

        self._define_symbols()

        self.state_dimension = len(self.u)

        self._define_mode_shapes()
        self._define_inertia_matrices()
        self._define_frame_orientations()
        self._define_points()
        self._define_system_center_of_mass()
        self._define_kinematic_equations()
        self._define_angular_velocities()
        self._define_linear_velocities()
        self._define_accelerations()

        self.external_forces = self._get_external_forces()
        self.external_torques = self._get_external_torques()

        self._define_partial_velocities()
        self._define_system_center_of_mass_kinematics()
        self._define_generalised_active_forces()
        self._define_generalised_inertia_forces()
        self._derive_equations_of_motion()
        self._create_lambdified_functions()

    def _parents_from_adjacency(self, graph, body_names, central_body):
        visited = {body: False for body in body_names}
        parents = {}

        visited[central_body] = True
        parents[central_body] = "N"
        queue = deque([central_body])

        while queue:
            current_node = queue.popleft()

            for neighbor in graph[current_node]:
                if not visited[neighbor]:
                    visited[neighbor] = True
                    parents[neighbor] = current_node
                    queue.append(neighbor)

        missing = [body for body, was_visited in visited.items() if not was_visited]
        if missing:
            raise ValueError(f"Disconnected bodies from central body: {missing}")

        return parents

    @staticmethod
    def _normalise_flexible_inertia_integration_config(settings: dict):
        valid_methods = {"gauss-legendre", "symbolic"}

        method = settings.get("method", "gauss-legendre")
        if method not in valid_methods:
            valid_method_names = ", ".join(sorted(valid_methods))
            raise ValueError(
                "flexible_inertia_integration.method must be one of "
                f"{valid_method_names}; got {method!r}."
            )

        quadrature_points = int(settings.get("quadrature_points", 8))
        if quadrature_points < 1:
            raise ValueError(
                "flexible_inertia_integration.quadrature_points must be >= 1."
            )

        return {
            "method": method,
            "quadrature_points": quadrature_points,
        }

    def _angle_symbol_name(self, body_name: str) -> str:
        match = re.fullmatch(r"bus_(\d+)", body_name)
        if match:
            return f"q3_{match.group(1)}"

        raise ValueError(f"Expected bus name of form 'bus_<number>', got {body_name!r}.")

    def _speed_symbol_name(self, body_name: str) -> str:
        match = re.fullmatch(r"bus_(\d+)", body_name)
        if match:
            return f"u3_{match.group(1)}"

        raise ValueError(f"Expected bus name of form 'bus_<number>', got {body_name!r}.")

    def _define_symbols(self):
        self.t = me.dynamicsymbols._t

        self._define_dynamic_symbols()
        self._define_parameter_symbols()
        self._define_torque_symbols()

    def _define_dynamic_symbols(self):
        self.q_translation = {
            "x": me.dynamicsymbols("q1"),
            "y": me.dynamicsymbols("q2"),
        }
        self.u_translation = {
            "x": me.dynamicsymbols("u1"),
            "y": me.dynamicsymbols("u2"),
        }

        self.bus_angle_coordinates = {}
        self.bus_speed_coordinates = {}
        for body in self.rigid_body_names:
            q_symbol = me.dynamicsymbols(self._angle_symbol_name(body))
            u_symbol = me.dynamicsymbols(self._speed_symbol_name(body))
            self.bus_angle_coordinates[body] = q_symbol
            self.bus_speed_coordinates[body] = u_symbol

        self.central_angle = self.bus_angle_coordinates[self.central_body]
        self.central_speed = self.bus_speed_coordinates[self.central_body]

        self.flexible_bodies = {}
        self.flex_eta_index = {}
        self.flex_zeta_index = {}
        eta_all = []
        zeta_all = []

        flat_eta_idx = 0
        flat_zeta_idx = 0
        for i_body, body in enumerate(self.flexible_body_names):
            beam_type = self.flexible_types[body]
            n_modes = self.beam_parameters[beam_type]["nb_modes"]

            eta_list = [
                me.dynamicsymbols(f"eta{i_body + 1}_{mode + 1}")
                for mode in range(n_modes)
            ]
            zeta_list = [
                me.dynamicsymbols(f"zeta{i_body + 1}_{mode + 1}")
                for mode in range(n_modes)
            ]

            self.flexible_bodies[body] = {
                "beam_type": beam_type,
                "eta_list": eta_list,
                "zeta_list": zeta_list,
            }

            for mode in range(n_modes):
                self.flex_eta_index[(body, mode)] = flat_eta_idx
                self.flex_zeta_index[(body, mode)] = flat_zeta_idx
                flat_eta_idx += 1
                flat_zeta_idx += 1

            eta_all.extend(eta_list)
            zeta_all.extend(zeta_list)

        self.q_reference = {
            **self.q_translation,
            **self.bus_angle_coordinates,
        }
        self.u_reference = {
            **self.u_translation,
            **self.bus_speed_coordinates,
        }

        self.q = sm.Matrix(
            [
                *self.q_translation.values(),
                *self.bus_angle_coordinates.values(),
                *eta_all,
            ]
        )
        self.u = sm.Matrix(
            [
                *self.u_translation.values(),
                *self.bus_speed_coordinates.values(),
                *zeta_all,
            ]
        )
        self.qd = self.q.diff(self.t)
        self.ud = self.u.diff(self.t)

    def _define_parameter_symbols(self):
        parameter_names = [
            "D",
            "L",
            *[f"m_{body}" for body in self.body_names],
            "E_mod",
            "I_area",
            "planet_mu",
            "orbit_semi_major_axis",
            "orbit_eccentricity",
        ]
        missing = [name for name in parameter_names if name not in self.parameter_values]
        if missing:
            raise KeyError(f"Missing parameters entries: {missing}")

        self.parameter_symbols = {
            name: sm.symbols(name)
            for name in parameter_names
        }
        self.p = sm.Matrix(list(self.parameter_symbols.values()))

        self.D = self.parameter_symbols["D"]
        self.L = self.parameter_symbols["L"]
        self.mass_symbols = {
            body: self.parameter_symbols[f"m_{body}"]
            for body in self.body_names
        }
        self.E_mod = self.parameter_symbols["E_mod"]
        self.I_area = self.parameter_symbols["I_area"]
        self.planet_mu = self.parameter_symbols["planet_mu"]
        self.orbit_semi_major_axis = self.parameter_symbols["orbit_semi_major_axis"]
        self.orbit_eccentricity = self.parameter_symbols["orbit_eccentricity"]

    def _define_torque_symbols(self):
        self.bus_torque_symbols = {
            body: sm.symbols(f"tau_{body.removeprefix('bus_')}")
            for body in self.rigid_body_names
        }
        self.tau = sm.Matrix(
            [
                self.bus_torque_symbols[body]
                for body in self.rigid_body_names
            ]
        )

    def _define_mode_shapes(self):
        self.s = sm.symbols("s")

        for body, values in self.flexible_bodies.items():
            beam_type = values["beam_type"]
            params = self.beam_parameters[beam_type]
            n_modes = params["nb_modes"]
            nb_points = params.get("nb_points", 200)

            if beam_type == "cantilever":
                beam = CantileverBeam(
                    length=self.parameter_values["L"],
                    E=self.parameter_values["E_mod"],
                    I=self.parameter_values["I_area"],
                    n=n_modes,
                )
            elif beam_type == "clamped-clamped":
                beam = ClampedClampedBeam(
                    length=self.parameter_values["L"],
                    E=self.parameter_values["E_mod"],
                    I=self.parameter_values["I_area"],
                    n=n_modes,
                )
            else:
                raise TypeError(f"Unrecognised beam type: {beam_type}")

            values["beam"] = beam
            values["phi_list"] = [
                beam.mode_shape_symbolic(self.s, mode + 1)
                for mode in range(n_modes)
            ]
            values["phi_mean_list"] = [
                beam.mode_shape_mean(nb_points, mode=mode + 1)
                for mode in range(n_modes)
            ]
            values["k_modal_list"] = [
                beam.modal_stiffness(mode + 1)
                for mode in range(n_modes)
            ]
            values["phi_norm_list"] = [
                beam.mode_shape_norm(nb_points, mode=mode + 1)
                for mode in range(n_modes)
            ]
            values["phi_m1_list"] = [
                beam.mode_shape_first_moment(nb_points, mode=mode + 1)
                for mode in range(n_modes)
            ]

    def _define_inertia_matrices(self):
        self.inertia_matrices = {}

        for body in self.rigid_body_names:
            mass = self.mass_symbols[body]
            I11 = sm.Rational(1, 12) * mass * self.D**2
            I22 = sm.Rational(1, 12) * mass * self.D**2
            I33 = I11 + I22

            self.inertia_matrices[body] = sm.Matrix(
                [
                    [I11, 0, 0],
                    [0, I22, 0],
                    [0, 0, I33],
                ]
            )

        for body in self.flexible_body_names:
            fb_dict = self.flexible_bodies[body]
            eta_list = fb_dict["eta_list"]
            phi_mean_list = fb_dict["phi_mean_list"]
            phi_norm_list = fb_dict["phi_norm_list"]
            phi_m1_list = fb_dict["phi_m1_list"]

            mass = self.mass_symbols[body]

            I22 = sm.Rational(1, 12) * mass * self.L**2

            I12 = sm.S.Zero
            for eta_k, phi_m1_k in zip(eta_list, phi_m1_list):
                I12 += -eta_k * sm.Float(phi_m1_k)
            I12 = mass * I12

            first_term = sm.S.Zero
            for eta_k, phi_norm_k in zip(eta_list, phi_norm_list):
                first_term += eta_k**2 * sm.Float(phi_norm_k)

            second_term = sm.S.Zero
            for eta_k, phi_mean_k in zip(eta_list, phi_mean_list):
                second_term += eta_k * sm.Float(phi_mean_k)

            I11 = mass * (first_term - second_term**2)
            I33 = I11 + I22

            self.inertia_matrices[body] = sm.Matrix(
                [
                    [I11, I12, 0],
                    [I12, I22, 0],
                    [0, 0, I33],
                ]
            )

    def _orientation_offset(self, body_name: str):
        body_type = self.body_type[body_name]
        if body_type.endswith("-left"):
            return sm.pi
        return sm.S.Zero

    def _bus_orientation_angle(self, body_name: str):
        if body_name == self.central_body:
            return self.central_angle

        return (
            self.central_angle
            + self._orientation_offset(body_name)
            + self.bus_angle_coordinates[body_name]
        )

    def _rigid_neighbors(self, body_name: str) -> list[str]:
        return [
            neighbor
            for neighbor in self.graph[body_name]
            if neighbor in self.rigid_body_names
        ]

    def _body_orientation_angle(self, body_name: str):
        if body_name in self.rigid_body_names:
            return self._bus_orientation_angle(body_name)

        rigid_neighbors = self._rigid_neighbors(body_name)
        if not rigid_neighbors:
            raise ValueError(f"Flexible body '{body_name}' is not attached to a rigid bus.")

        if len(rigid_neighbors) == 1:
            source_bus = rigid_neighbors[0]
            if source_bus == self.central_body:
                return self._bus_orientation_angle(source_bus) + self._orientation_offset(body_name)

            return self._bus_orientation_angle(source_bus)

        if len(rigid_neighbors) == 2:
            first_angle = self._bus_orientation_angle(rigid_neighbors[0])
            second_angle = self._bus_orientation_angle(rigid_neighbors[1])
            return sm.Rational(1, 2) * (first_angle + second_angle)

        raise ValueError(
            f"Flexible body '{body_name}' is attached to more than two rigid buses: "
            f"{rigid_neighbors}."
        )

    def _connection_kind(self, parent: str, child: str) -> str:
        parent_type = self.body_type[parent]
        child_type = self.body_type[child]

        parent_is_rigid = parent_type.startswith("rigid-")
        parent_is_flexible = parent_type.startswith("flexible-")
        child_is_rigid = child_type.startswith("rigid-")
        child_is_flexible = child_type.startswith("flexible-")

        if parent_is_rigid and child_is_flexible:
            return "rigid_to_flexible"

        if parent_is_flexible and child_is_rigid:
            return "flexible_to_rigid"

        if parent_is_flexible and child_is_flexible:
            raise NotImplementedError(
                "Connection between two flexible bodies is not implemented: "
                f"parent={parent!r}, child={child!r}."
            )

        if parent_is_rigid and child_is_rigid:
            raise NotImplementedError(
                "Connection between two rigid bodies is not implemented: "
                f"parent={parent!r}, child={child!r}."
            )

        raise NotImplementedError(
            "Unsupported body type pair: "
            f"parent={parent!r} ({parent_type!r}), child={child!r} ({child_type!r})."
        )

    def _get_offset_vector(self, parent: str, child: str):
        connection_kind = self._connection_kind(parent, child)
        child_type = self.body_type[child]

        if connection_kind == "rigid_to_flexible":
            sign = -1 if child_type.endswith("-left") else 1
            return sign * (self.D / 2) * self.frames[parent].x

        if connection_kind == "flexible_to_rigid":
            frame = self.frames[parent]
            eta_list = self.flexible_bodies[parent]["eta_list"]
            phi_list = self.flexible_bodies[parent]["phi_list"]

            phi_tip = sm.S.Zero
            for phi_k, eta_k in zip(phi_list, eta_list):
                phi_tip += phi_k.subs(self.s, self.L) * eta_k

            return self.L * frame.x + phi_tip * frame.y

        raise NotImplementedError(f"Unsupported connection kind: {connection_kind!r}.")

    def _define_points(self):
        inertial_frame = self.frames["inertial"]

        self.points = {}
        self.inertial_position = {body: None for body in self.body_names}

        self.O = me.Point("O")
        self.O.set_vel(inertial_frame, 0)
        self.points["N"] = self.O

        self.central_cm = me.Point(f"{self.central_body}_cm")
        self.central_cm.set_pos(
            self.O,
            self.q_translation["x"] * inertial_frame.x
            + self.q_translation["y"] * inertial_frame.y,
        )
        self.points[self.central_body] = self.central_cm
        self.inertial_position[self.central_body] = self.central_cm.pos_from(self.O)

        for child, parent in self.parents.items():
            if child == self.central_body:
                continue

            parent_type = self.body_type[parent]
            child_type = self.body_type[child]
            attach = self._get_offset_vector(parent, child)

            if parent_type.startswith("rigid-"):
                parent_point = self.points[parent]
            else:
                joint_name = f"joint_{parent}_{self.parents[parent]}"
                parent_point = self.points[joint_name]

            child_joint = parent_point.locatenew(f"joint_{child}_{parent}", attach)
            self.points[f"joint_{child}_{parent}"] = child_joint

            if child_type.startswith("flexible-"):
                frame = self.frames[child]
                eta_list = self.flexible_bodies[child]["eta_list"]
                phi_list = self.flexible_bodies[child]["phi_list"]
                phi_sum = sum(
                    phi_k * eta_k
                    for phi_k, eta_k in zip(phi_list, eta_list)
                )

                dm_offset = self.s * frame.x + phi_sum * frame.y
                dm_point = child_joint.locatenew(f"dm_{child}", dm_offset)
                self.points[f"dm_{child}"] = dm_point

                phi_mean_list = self.flexible_bodies[child]["phi_mean_list"]
                phi_mean_sum = sum(
                    sm.Float(phi_mean_k) * eta_k
                    for phi_mean_k, eta_k in zip(phi_mean_list, eta_list)
                )
                cm_offset = self.L / 2 * frame.x + phi_mean_sum * frame.y
                cm_point = child_joint.locatenew(
                    f"dm_center_of_mass_{child}",
                    cm_offset,
                )
                self.points[f"dm_center_of_mass_{child}"] = cm_point
                self.inertial_position[child] = cm_point.pos_from(self.O)

            elif child_type.startswith("rigid-"):
                frame = self.frames[child]
                sign = -1 if child_type.endswith("-left") else 1
                cm_offset = sign * self.D / 2 * frame.x
                cm_point = child_joint.locatenew(f"{child}_cm", cm_offset)
                self.points[child] = cm_point
                self.inertial_position[child] = cm_point.pos_from(self.O)

    def _define_system_center_of_mass(self):
        self.total_mass = sum(
            self.mass_symbols[body]
            for body in self.body_names
        )

        r_G = sm.S.Zero
        for body in self.body_names:
            r_G += self.mass_symbols[body] * self.inertial_position[body]
        r_G /= self.total_mass

        self.G = self.O.locatenew("G", r_G)
        self.r_GB = self.points[self.central_body].pos_from(self.G)
        self.points["center_of_mass"] = self.G

    def _define_system_center_of_mass_kinematics(self):
        inertial_frame = self.frames["inertial"]

        self.r_G = self.points["center_of_mass"].pos_from(self.O).express(
            inertial_frame
        )
        self.v_G = self.r_G.dt(inertial_frame).xreplace(self.qd_repl)
        self.partial_v_G = me.partial_velocity(
            [self.v_G],
            self.u,
            inertial_frame,
        )[0]

    def _define_kinematic_equations(self):
        self.qd_zero = {qdi: 0 for qdi in self.qd}
        self.ud_zero = {udi: 0 for udi in self.ud}

        self.fk = self.qd - self.u
        self.Mk = self.fk.jacobian(self.qd)
        self.gk = self.fk.xreplace(self.qd_zero)

        qd_sol = -self.Mk.LUsolve(self.gk)
        self.qd_repl = dict(zip(self.qd, qd_sol))
        self.qdd_repl = {
            q.diff(self.t): u.diff(self.t)
            for q, u in self.qd_repl.items()
        }

    def _define_angular_velocities(self):
        inertial_frame = self.frames["inertial"]
        self.angular_velocities = {}

        for body in self.body_names:
            frame = self.frames[body]
            angular_velocity = frame.ang_vel_in(inertial_frame).xreplace(self.qd_repl)
            frame.set_ang_vel(inertial_frame, angular_velocity)
            self.angular_velocities[body] = angular_velocity

    def _define_linear_velocities(self):
        inertial_frame = self.frames["inertial"]
        self.linear_velocities = {}
        self.joint_velocities = {}
        self.flexible_center_of_mass_velocities = {}

        central_velocity = (
            self.u_translation["x"] * inertial_frame.x
            + self.u_translation["y"] * inertial_frame.y
        )
        self.points[self.central_body].set_vel(inertial_frame, central_velocity)
        self.linear_velocities[self.central_body] = central_velocity

        for child, parent in self.parents.items():
            if child == self.central_body:
                continue

            connection_kind = self._connection_kind(parent, child)
            child_type = self.body_type[child]
            joint_name = f"joint_{child}_{parent}"
            child_joint = self.points[joint_name]

            if connection_kind == "rigid_to_flexible":
                parent_point = self.points[parent]
                parent_frame = self.frames[parent]
                child_joint.v2pt_theory(parent_point, inertial_frame, parent_frame)

            elif connection_kind == "flexible_to_rigid":
                parent_frame = self.frames[parent]
                parent_root_name = f"joint_{parent}_{self.parents[parent]}"
                parent_root = self.points[parent_root_name]
                phi_tip_zeta_sum = self._flexible_tip_velocity_sum(parent)

                child_joint.set_vel(
                    parent_frame,
                    phi_tip_zeta_sum * parent_frame.y,
                )
                child_joint.v1pt_theory(parent_root, inertial_frame, parent_frame)

            else:
                raise NotImplementedError(
                    f"Unsupported connection kind: {connection_kind!r}."
                )

            self.joint_velocities[joint_name] = child_joint.vel(inertial_frame)

            if child_type.startswith("flexible-"):
                frame = self.frames[child]
                dm_point = self.points[f"dm_{child}"]
                dm_point.set_vel(
                    frame,
                    self._flexible_distributed_velocity_sum(child) * frame.y,
                )
                dm_vel = dm_point.v1pt_theory(child_joint, inertial_frame, frame)
                self.linear_velocities[child] = dm_vel.xreplace(self.qd_repl)

                cm_point = self.points[f"dm_center_of_mass_{child}"]
                cm_point.set_vel(
                    frame,
                    self._flexible_center_of_mass_velocity_sum(child) * frame.y,
                )
                cm_vel = cm_point.v1pt_theory(child_joint, inertial_frame, frame)
                self.flexible_center_of_mass_velocities[child] = cm_vel.xreplace(
                    self.qd_repl
                )

            elif child_type.startswith("rigid-"):
                child_point = self.points[child]
                child_frame = self.frames[child]
                child_point.v2pt_theory(child_joint, inertial_frame, child_frame)
                self.linear_velocities[child] = child_point.vel(inertial_frame).xreplace(
                    self.qd_repl
                )

            else:
                raise KeyError(f"Unknown child type for '{child}': {child_type}")

    def _flexible_distributed_velocity_sum(self, body: str):
        phi_list = self.flexible_bodies[body]["phi_list"]
        zeta_list = self.flexible_bodies[body]["zeta_list"]

        return sum(
            phi_k * zeta_k
            for phi_k, zeta_k in zip(phi_list, zeta_list)
        )

    def _flexible_center_of_mass_velocity_sum(self, body: str):
        phi_mean_list = self.flexible_bodies[body]["phi_mean_list"]
        zeta_list = self.flexible_bodies[body]["zeta_list"]

        return sum(
            sm.Float(phi_mean_k) * zeta_k
            for phi_mean_k, zeta_k in zip(phi_mean_list, zeta_list)
        )

    def _flexible_tip_velocity_sum(self, body: str):
        phi_list = self.flexible_bodies[body]["phi_list"]
        zeta_list = self.flexible_bodies[body]["zeta_list"]

        return sum(
            phi_k.subs(self.s, self.L) * zeta_k
            for phi_k, zeta_k in zip(phi_list, zeta_list)
        )

    def _as_speed_acceleration_expression(self, vector):
        return vector.xreplace(self.qdd_repl).xreplace(self.qd_repl)

    def _define_accelerations(self):
        inertial_frame = self.frames["inertial"]
        self.angular_accelerations = {}
        self.linear_accelerations = {}
        self.joint_accelerations = {}
        self.flexible_center_of_mass_accelerations = {}

        for body in self.body_names:
            angular_acceleration = self.frames[body].ang_acc_in(inertial_frame)
            self.angular_accelerations[body] = (
                self._as_speed_acceleration_expression(angular_acceleration)
            )

        central_acceleration = self.points[self.central_body].acc(inertial_frame)
        self.linear_accelerations[self.central_body] = (
            self._as_speed_acceleration_expression(central_acceleration)
        )

        for child, parent in self.parents.items():
            if child == self.central_body:
                continue

            child_type = self.body_type[child]
            joint_name = f"joint_{child}_{parent}"
            child_joint = self.points[joint_name]
            joint_acceleration = child_joint.acc(inertial_frame)
            self.joint_accelerations[joint_name] = (
                self._as_speed_acceleration_expression(joint_acceleration)
            )

            if child_type.startswith("flexible-"):
                dm_point = self.points[f"dm_{child}"]
                dm_acceleration = dm_point.acc(inertial_frame)
                self.linear_accelerations[child] = (
                    self._as_speed_acceleration_expression(dm_acceleration)
                )

                cm_point = self.points[f"dm_center_of_mass_{child}"]
                cm_acceleration = cm_point.acc(inertial_frame)
                self.flexible_center_of_mass_accelerations[child] = (
                    self._as_speed_acceleration_expression(cm_acceleration)
                )

            elif child_type.startswith("rigid-"):
                child_point = self.points[child]
                child_acceleration = child_point.acc(inertial_frame)
                self.linear_accelerations[child] = (
                    self._as_speed_acceleration_expression(child_acceleration)
                )

            else:
                raise KeyError(f"Unknown child type for '{child}': {child_type}")

    def _get_external_forces(self):
        return {
            body: 0 * self.frames[body].x + 0 * self.frames[body].y
            for body in self.body_names
        }

    def _get_external_torques(self):
        torques = {}

        for body in self.body_names:
            frame = self.frames[body]
            if body in self.bus_torque_symbols:
                torques[body] = self.bus_torque_symbols[body] * frame.z
            else:
                torques[body] = 0 * frame.z

        return torques

    def _define_partial_velocities(self):
        inertial_frame = self.frames["inertial"]

        partial_angular_velocities = me.partial_velocity(
            list(self.angular_velocities.values()),
            self.u,
            inertial_frame,
        )
        self.partial_angular_velocities = dict(
            zip(self.angular_velocities.keys(), partial_angular_velocities)
        )

        partial_linear_velocities = me.partial_velocity(
            list(self.linear_velocities.values()),
            self.u,
            inertial_frame,
        )
        self.partial_linear_velocities = dict(
            zip(self.linear_velocities.keys(), partial_linear_velocities)
        )

    def _initialise_generalised_active_forces(self):
        self.generalised_active_forces = sm.zeros(self.state_dimension, 1)

    def _add_external_load_generalised_forces(self):
        for i in range(self.state_dimension):
            for body in self.body_names:
                v_partial = self.partial_linear_velocities[body][i]
                w_partial = self.partial_angular_velocities[body][i]
                force = self.external_forces[body]
                torque = self.external_torques[body]

                self.generalised_active_forces[i] += (
                    v_partial.dot(force)
                    + w_partial.dot(torque)
                )

    def _add_kepler_gravity_generalised_forces(self):
        self.r_G_squared = self.r_G.dot(self.r_G)
        self.r_G_norm = sm.sqrt(self.r_G_squared)
        self.F_gravity = (
            -self.planet_mu
            * self.total_mass
            * self.r_G
            / self.r_G_norm**3
        )

        for i in range(self.state_dimension):
            self.generalised_active_forces[i] += self.partial_v_G[i].dot(
                self.F_gravity
            )

    def _body_centre_offset_from_system_centre(self, body: str):
        return self.inertial_position[body] - self.r_G

    def _define_body_centre_offsets(self):
        self.body_centre_offsets = {
            body: self._body_centre_offset_from_system_centre(body)
            for body in self.body_names
        }

    def _body_gravity_gradient_energy(self, body: str):
        frame = self.frames[body]
        local_vertical_body = self.e3_hat_inertial.express(frame)
        self.e3_hat_body[body] = local_vertical_body

        e_body_components = local_vertical_body.to_matrix(frame)
        inertia_matrix = self.inertia_matrices[body]
        body_offset = self.body_centre_offsets[body].express(frame)

        local_directional_inertia = (
            e_body_components.T * inertia_matrix * e_body_components
        )[0]
        parallel_axis_directional_inertia = self.mass_symbols[body] * (
            body_offset.dot(body_offset)
            - local_vertical_body.dot(body_offset) ** 2
        )

        return local_directional_inertia + parallel_axis_directional_inertia

    def _body_gravity_gradient_trace_inertia(self, body: str):
        body_offset = self.body_centre_offsets[body]
        return (
            sm.trace(self.inertia_matrices[body])
            + 2 * self.mass_symbols[body] * body_offset.dot(body_offset)
        )

    def _build_gravity_gradient_potential(self):
        self._define_body_centre_offsets()
        self.gravity_gradient_directional_inertia = sm.S.Zero
        self.gravity_gradient_trace_inertia = sm.S.Zero

        for body in self.body_names:
            self.gravity_gradient_directional_inertia += (
                self._body_gravity_gradient_energy(body)
            )
            self.gravity_gradient_trace_inertia += (
                self._body_gravity_gradient_trace_inertia(body)
            )

        trace_prefix = -sm.Rational(1, 2) * self.planet_mu / self.r_G_norm**3
        directional_prefix = sm.Rational(3, 2) * self.planet_mu / self.r_G_norm**3

        self.V_gg_trace = trace_prefix * self.gravity_gradient_trace_inertia
        self.V_gg_directional = (
            directional_prefix * self.gravity_gradient_directional_inertia
        )
        self.V_gg = self.V_gg_trace + self.V_gg_directional

    def _add_gravity_gradient_generalised_forces(self):
        self.e3_hat_inertial = -self.r_G / self.r_G_norm
        self.e3_hat_body = {}
        self.body_centre_offsets = {}
        self.gravity_gradient_directional_inertia = sm.S.Zero
        self.gravity_gradient_trace_inertia = sm.S.Zero
        self.V_gg_trace = sm.S.Zero
        self.V_gg_directional = sm.S.Zero
        self.V_gg = sm.S.Zero

        if not self.enable_gravity_gradient:
            return

        self._build_gravity_gradient_potential()

        for body, angle_coordinate in self.bus_angle_coordinates.items():
            row = list(self.u).index(self.bus_speed_coordinates[body])
            self.generalised_active_forces[row] += -sm.diff(
                self.V_gg,
                angle_coordinate,
            )

        modal_offset = len(self.u_reference)
        for body, values in self.flexible_bodies.items():
            for mode, eta_k in enumerate(values["eta_list"]):
                row = modal_offset + self.flex_eta_index[(body, mode)]
                self.generalised_active_forces[row] += -sm.diff(self.V_gg, eta_k)

    def _add_flexible_strain_generalised_forces(self):
        self.V_strain = sm.S.Zero

        modal_offset = len(self.u_reference)
        for body, values in self.flexible_bodies.items():
            eta_list = values["eta_list"]
            k_modal_list = values["k_modal_list"]

            for mode, (eta_k, K_k) in enumerate(zip(eta_list, k_modal_list)):
                self.V_strain += sm.Rational(1, 2) * K_k * eta_k**2

                modal_row = modal_offset + self.flex_eta_index[(body, mode)]
                self.generalised_active_forces[modal_row] += -(K_k * eta_k)

    def _define_generalised_active_forces(self):
        self._initialise_generalised_active_forces()
        self._add_external_load_generalised_forces()
        self._add_kepler_gravity_generalised_forces()
        self._add_gravity_gradient_generalised_forces()
        self._add_flexible_strain_generalised_forces()

    def _body_inertia_times_vector(self, body: str, vector):
        frame = self.frames[body]
        components = self.inertia_matrices[body] * vector.to_matrix(frame)
        return (
            components[0] * frame.x
            + components[1] * frame.y
            + components[2] * frame.z
        )

    def _initialise_generalised_inertia_forces(self):
        self.generalised_inertia_forces = sm.zeros(self.state_dimension, 1)
        self.rigid_body_inertia_forces = {}
        self.rigid_body_inertia_torques = {}
        self.flexible_body_inertia_force_densities = {}
        self.flexible_body_generalised_inertia_forces = {}

    def _define_rigid_body_inertia_loads(self):
        for body in self.rigid_body_names:
            mass = self.mass_symbols[body]
            linear_acceleration = self.linear_accelerations[body]
            angular_velocity = self.angular_velocities[body]
            angular_acceleration = self.angular_accelerations[body]

            inertia_alpha = self._body_inertia_times_vector(
                body,
                angular_acceleration,
            )
            inertia_omega = self._body_inertia_times_vector(
                body,
                angular_velocity,
            )

            self.rigid_body_inertia_forces[body] = -mass * linear_acceleration
            self.rigid_body_inertia_torques[body] = -(
                inertia_alpha
                + angular_velocity.cross(inertia_omega)
            )

    def _add_rigid_body_generalised_inertia_forces(self):
        self._define_rigid_body_inertia_loads()

        for i in range(self.state_dimension):
            for body in self.rigid_body_names:
                v_partial = self.partial_linear_velocities[body][i]
                w_partial = self.partial_angular_velocities[body][i]
                inertia_force = self.rigid_body_inertia_forces[body]
                inertia_torque = self.rigid_body_inertia_torques[body]

                self.generalised_inertia_forces[i] += (
                    v_partial.dot(inertia_force)
                    + w_partial.dot(inertia_torque)
                )

    def _define_flexible_body_inertia_loads(self):
        for body in self.flexible_body_names:
            mass_per_length = self.mass_symbols[body] / self.L
            linear_acceleration = self.linear_accelerations[body]

            self.flexible_body_inertia_force_densities[body] = (
                -mass_per_length * linear_acceleration
            )

    def _flexible_body_inertia_quadrature_points(self, body: str) -> int:
        beam_type = self.flexible_bodies[body]["beam_type"]
        return int(
            self.beam_parameters[beam_type].get(
                "inertia_quadrature_points",
                self.flexible_inertia_integration["quadrature_points"],
            )
        )

    def _integrate_flexible_body_expression_symbolically(self, expression):
        return sm.integrate(expression, (self.s, 0, self.L))

    def _integrate_flexible_body_expression_by_quadrature(
        self,
        body: str,
        expression,
    ):
        n_points = self._flexible_body_inertia_quadrature_points(body)
        nodes, weights = np.polynomial.legendre.leggauss(n_points)

        length = sm.Float(self.parameter_values["L"])
        half_length = length / 2
        integral = sm.S.Zero

        for node, weight in zip(nodes, weights):
            s_value = half_length * (sm.Float(node) + 1)
            integral += sm.Float(weight) * expression.subs(self.s, s_value)

        return half_length * integral

    def _integrate_flexible_body_expression(self, body: str, expression):
        method = self.flexible_inertia_integration["method"]

        if method == "symbolic":
            return self._integrate_flexible_body_expression_symbolically(expression)

        if method == "gauss-legendre":
            return self._integrate_flexible_body_expression_by_quadrature(
                body,
                expression,
            )

        raise ValueError(f"Unsupported flexible inertia integration method: {method}")

    def _add_flexible_body_generalised_inertia_forces(self):
        self._define_flexible_body_inertia_loads()

        for body in self.flexible_body_names:
            force_density = self.flexible_body_inertia_force_densities[body]
            body_contributions = sm.zeros(self.state_dimension, 1)

            for i in range(self.state_dimension):
                v_partial = self.partial_linear_velocities[body][i]
                integrand = v_partial.dot(force_density)
                contribution = self._integrate_flexible_body_expression(
                    body,
                    integrand,
                )

                body_contributions[i] = contribution
                self.generalised_inertia_forces[i] += contribution

            self.flexible_body_generalised_inertia_forces[body] = body_contributions

    def _define_generalised_inertia_forces(self):
        self._initialise_generalised_inertia_forces()
        self._add_rigid_body_generalised_inertia_forces()
        self._add_flexible_body_generalised_inertia_forces()

    def _derive_equations_of_motion(self):
        self.kane_eq = self.generalised_active_forces + self.generalised_inertia_forces

        self.mass_matrix = -self.kane_eq.jacobian(self.ud)
        self.forcing = self.kane_eq.xreplace(self.ud_zero)

    def _create_lambdified_functions(self):
        parameters = list(self.parameter_symbols.values())
        torques = list(self.bus_torque_symbols.values())
        inertial_frame = self.frames["inertial"]

        self.eval_kinematics = sm.lambdify(
            (
                self.q,
                self.u,
                parameters,
                torques,
            ),
            (self.Mk, self.gk),
            "numpy",
            cse=True,
        )
        self.eval_differentials = sm.lambdify(
            (
                self.q,
                self.u,
                parameters,
                torques,
            ),
            (self.mass_matrix, self.forcing),
            "numpy",
            cse=True,
        )
        self.rG_func = sm.lambdify(
            (
                self.q,
                self.u,
                parameters,
            ),
            self.r_G.to_matrix(inertial_frame),
            "numpy",
            cse=True,
        )
        self.vG_func = sm.lambdify(
            (
                self.q,
                self.u,
                parameters,
            ),
            self.v_G.to_matrix(inertial_frame),
            "numpy",
            cse=True,
        )

    def get_parameter_values(self):
        return [
            self.parameter_values[name]
            for name in self.parameter_symbols
        ]

    def get_torque_values(self):
        configured_torques = self.config.get("torques", {})
        return [
            configured_torques.get(body, 0.0)
            for body in self.rigid_body_names
        ]

    def get_torque_weights(self):
        configured_weights = self.config.get("torque_weights", {})
        return [
            configured_weights.get(body, 0.0)
            for body in self.rigid_body_names
        ]

    def _kepler_initial_centre_of_mass_state(self):
        mu = float(self.parameter_values["planet_mu"])
        a = float(self.parameter_values["orbit_semi_major_axis"])
        e = float(self.parameter_values["orbit_eccentricity"])

        initial_true_anomaly = 0.0
        r0 = a * (1.0 - e**2) / (
            1.0 + e * np.cos(initial_true_anomaly)
        )
        h = np.sqrt(mu * a * (1.0 - e**2))

        r_G0 = np.array([r0, 0.0, 0.0], dtype=float)
        v_G0 = np.array(
            [
                -(mu / h) * e * np.sin(initial_true_anomaly),
                (mu / h) * (1.0 + e * np.cos(initial_true_anomaly)),
                0.0,
            ],
            dtype=float,
        )

        return r_G0, v_G0

    def get_initial_conditions(self, verbose=True):
        configured_states = self.config.get("q_initial", {})
        configured_speeds = self.config.get("initial_speeds", {})

        q_values = np.array(
            [
                float(configured_states.get(state_symbol.name, 0.0))
                for state_symbol in self.q
            ],
            dtype=float,
        )
        u_values = np.array(
            [
                float(configured_speeds.get(speed_symbol.name, 0.0))
                for speed_symbol in self.u
            ],
            dtype=float,
        )

        q1_index = list(self.q).index(self.q_translation["x"])
        q2_index = list(self.q).index(self.q_translation["y"])
        u1_index = list(self.u).index(self.u_translation["x"])
        u2_index = list(self.u).index(self.u_translation["y"])

        parameter_values = self.get_parameter_values()
        r_G0, v_G0 = self._kepler_initial_centre_of_mass_state()

        q_values[q1_index] = 0.0
        q_values[q2_index] = 0.0
        r_G_at_origin = np.asarray(
            self.rG_func(q_values, u_values, parameter_values),
            dtype=float,
        ).reshape(3)
        q_values[q1_index] = r_G0[0] - r_G_at_origin[0]
        q_values[q2_index] = r_G0[1] - r_G_at_origin[1]

        u_values[u1_index] = 0.0
        u_values[u2_index] = 0.0
        v_G_without_translation = np.asarray(
            self.vG_func(q_values, u_values, parameter_values),
            dtype=float,
        ).reshape(3)
        u_values[u1_index] = v_G0[0] - v_G_without_translation[0]
        u_values[u2_index] = v_G0[1] - v_G_without_translation[1]

        if verbose:
            print("\nInitial conditions set (with momentum conservation):\n")
            position_parts = []
            for state_symbol, value in zip(self.q, q_values):
                if state_symbol.name.startswith("q3_"):
                    position_parts.append(
                        f"{state_symbol.name}={np.degrees(value):.3f}°"
                    )
                else:
                    position_parts.append(f"{state_symbol.name}={value:.3f}")

            speed_parts = [
                f"{speed_symbol.name}={value:.6f}"
                for speed_symbol, value in zip(self.u, u_values)
            ]

            print("  Positions: " + ", ".join(position_parts) + "\n")
            print("  Velocities: " + ", ".join(speed_parts) + "\n")

        return np.concatenate([q_values, u_values])

    def _define_frame_orientations(self):
        inertial_frame = me.ReferenceFrame("N")
        self.frames = {"inertial": inertial_frame}
        self.bus_orientation_angles = {}
        self.body_orientation_angles = {}

        for body in self.rigid_body_names:
            self.bus_orientation_angles[body] = self._bus_orientation_angle(body)

        for body in self.body_names:
            angle = self._body_orientation_angle(body)
            frame = me.ReferenceFrame(f"frame_{body}")
            frame.orient_axis(inertial_frame, angle, inertial_frame.z)

            self.frames[body] = frame
            self.body_orientation_angles[body] = sm.simplify(angle)

    def orientation_angle(self, body_name: str):
        """Return the absolute frame orientation angle of a body."""
        return self.body_orientation_angles[body_name]
