from __future__ import annotations

import numpy as np
import sympy as sm

from multibodysim.references import (
    MultiAngleCoordinateMapper,
    PlanarKeplerianReference,
)


class FakeDynamics:
    def __init__(self):
        self.q1, self.q2, self.theta = sm.symbols(
            "q1 q2 q_central_angle"
        )
        self.u1, self.u2, self.omega = sm.symbols(
            "u1 u2 u_central_angle"
        )
        self.q = sm.Matrix([self.q1, self.q2, self.theta])
        self.u = sm.Matrix([self.u1, self.u2, self.omega])
        self.state_dimension = 3
        self.q_translation = {"x": self.q1, "y": self.q2}
        self.u_translation = {"x": self.u1, "y": self.u2}

    @staticmethod
    def rG_func(q, u):
        del u
        return np.array(
            [
                [q[0] + 2.0 * np.cos(q[2])],
                [q[1] + 2.0 * np.sin(q[2])],
                [0.0],
            ]
        )

    @staticmethod
    def vG_func(q, u):
        return np.array(
            [
                [u[0] - 2.0 * np.sin(q[2]) * u[2]],
                [u[1] + 2.0 * np.cos(q[2]) * u[2]],
                [0.0],
            ]
        )


def test_coordinate_mapper_matches_requested_com_state():
    dynamics = FakeDynamics()
    mapper = MultiAngleCoordinateMapper(dynamics)
    theta = 0.4
    omega = 0.03
    q_template = np.array([99.0, -99.0, theta])
    u_template = np.array([88.0, -88.0, omega])
    target_position = np.array([10.0, -4.0])
    target_velocity = np.array([0.5, 0.8])

    mapped = mapper.map(
        q_template,
        u_template,
        centre_of_mass_position=target_position,
        centre_of_mass_velocity=target_velocity,
    )

    np.testing.assert_allclose(
        mapped.centre_of_mass_position,
        target_position,
    )
    np.testing.assert_allclose(
        mapped.centre_of_mass_velocity,
        target_velocity,
    )
    assert mapped.q[2] == theta
    assert mapped.u[2] == omega
    np.testing.assert_allclose(
        mapped.state,
        np.hstack((mapped.q, mapped.u)),
    )


def test_coordinate_mapper_does_not_modify_templates():
    dynamics = FakeDynamics()
    mapper = MultiAngleCoordinateMapper(dynamics)
    q_template = np.array([1.0, 2.0, 0.4])
    u_template = np.array([3.0, 4.0, 0.03])
    q_original = q_template.copy()
    u_original = u_template.copy()

    mapper.map(
        q_template,
        u_template,
        centre_of_mass_position=np.array([10.0, -4.0]),
        centre_of_mass_velocity=np.array([0.5, 0.8]),
    )

    np.testing.assert_allclose(q_template, q_original)
    np.testing.assert_allclose(u_template, u_original)


def test_coordinate_mapper_places_real_model_on_propagated_reference(
    seven_part_dynamics,
):
    dynamics = seven_part_dynamics
    mapper = MultiAngleCoordinateMapper(dynamics)
    reference = PlanarKeplerianReference(
        gravitational_parameter=dynamics.parameter_values["planet_mu"],
        semi_major_axis=dynamics.parameter_values[
            "orbit_semi_major_axis"
        ],
        eccentricity=dynamics.parameter_values["orbit_eccentricity"],
    )
    reference_state = reference.evaluate(0.23 * reference.period)

    q_template = np.zeros(dynamics.state_dimension)
    u_template = np.zeros(dynamics.state_dimension)
    q_template[list(dynamics.q).index(dynamics.central_angle)] = 0.4
    u_template[list(dynamics.u).index(dynamics.central_speed)] = 0.03
    q_internal = q_template.copy()
    u_internal = u_template.copy()

    mapped = mapper.map(
        q_template,
        u_template,
        centre_of_mass_position=reference_state.position,
        centre_of_mass_velocity=reference_state.velocity,
    )

    np.testing.assert_allclose(
        mapped.centre_of_mass_position,
        reference_state.position,
        atol=1e-8,
    )
    np.testing.assert_allclose(
        mapped.centre_of_mass_velocity,
        reference_state.velocity,
        atol=1e-9,
    )

    translation_q_indices = {mapper.q1_index, mapper.q2_index}
    translation_u_indices = {mapper.u1_index, mapper.u2_index}
    for index in range(dynamics.state_dimension):
        if index not in translation_q_indices:
            assert mapped.q[index] == q_internal[index]
        if index not in translation_u_indices:
            assert mapped.u[index] == u_internal[index]
