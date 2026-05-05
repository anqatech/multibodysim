from __future__ import annotations

from collections import deque
import re

import sympy as sm
import sympy.physics.mechanics as me


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
        except KeyError as exc:
            raise KeyError(f"Missing config key: {exc}") from exc

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
        self._define_frame_orientations()

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
