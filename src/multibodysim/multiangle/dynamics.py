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

        self.q_reference = {
            "x": me.dynamicsymbols("q1"),
            "y": me.dynamicsymbols("q2"),
        }
        self.u_reference = {
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
            self.q_reference[body] = q_symbol
            self.u_reference[body] = u_symbol

        self.central_angle = self.bus_angle_coordinates[self.central_body]
        self.central_speed = self.bus_speed_coordinates[self.central_body]

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

    def _orientation_source_bus(self, body_name: str) -> str:
        parent = self.parents[body_name]
        if parent in self.rigid_body_names:
            return parent

        for neighbor in self.graph[body_name]:
            if neighbor in self.rigid_body_names:
                return neighbor

        raise ValueError(f"Flexible body '{body_name}' is not attached to a rigid bus.")

    def _body_orientation_angle(self, body_name: str):
        if body_name in self.rigid_body_names:
            return self._bus_orientation_angle(body_name)

        source_bus = self._orientation_source_bus(body_name)
        if source_bus == self.central_body:
            return self._bus_orientation_angle(source_bus) + self._orientation_offset(body_name)

        return self._bus_orientation_angle(source_bus)

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
