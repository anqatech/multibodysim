from __future__ import annotations

import numpy as np
import sympy as sm

from multibodysim.references import (
    InertialRestToRestReference,
    MultiAngleReferenceBuilder,
    NadirAcquisitionReference,
    NadirPointingReference,
    PlanarKeplerianReference,
)


MU = 3.986004418e14
A = 6_778_000.0


class FakeDynamics:
    def __init__(self):
        symbols = sm.symbols(
            "q1 q2 q_central_angle q_relative_angle_bus_1 eta1_1"
        )
        speeds = sm.symbols(
            "u1 u2 u_central_angle u_relative_angle_bus_1 zeta1_1"
        )
        self.q = sm.Matrix(symbols)
        self.u = sm.Matrix(speeds)
        self.state_dimension = len(symbols)
        self.q_translation = {"x": symbols[0], "y": symbols[1]}
        self.u_translation = {"x": speeds[0], "y": speeds[1]}
        self.central_angle = symbols[2]
        self.central_speed = speeds[2]

    @staticmethod
    def rG_func(q, u):
        del u
        offset = 2.0 + q[3] + q[4]
        return np.array(
            [
                [q[0] + offset * np.cos(q[2])],
                [q[1] + offset * np.sin(q[2])],
                [0.0],
            ]
        )

    @staticmethod
    def vG_func(q, u):
        offset = 2.0 + q[3] + q[4]
        internal_speed = u[3] + u[4]
        return np.array(
            [
                [
                    u[0]
                    - offset * np.sin(q[2]) * u[2]
                    + internal_speed * np.cos(q[2])
                ],
                [
                    u[1]
                    + offset * np.cos(q[2]) * u[2]
                    + internal_speed * np.sin(q[2])
                ],
                [0.0],
            ]
        )


def test_inertial_builder_maps_orbit_and_zeroes_internal_states():
    dynamics = FakeDynamics()
    orbit = PlanarKeplerianReference(MU, A, 0.0)
    attitude = InertialRestToRestReference(
        theta_target=0.6,
        duration=100.0,
    )
    attitude.initialise(start_time=0.0, theta_initial=0.2)
    builder = MultiAngleReferenceBuilder(dynamics, orbit, attitude)

    reference = builder.evaluate(50.0)

    np.testing.assert_allclose(
        reference.centre_of_mass.position,
        dynamics.rG_func(reference.q, reference.u).reshape(-1)[:2],
        atol=1e-8,
    )
    np.testing.assert_allclose(
        reference.centre_of_mass.velocity,
        dynamics.vG_func(reference.q, reference.u).reshape(-1)[:2],
        atol=1e-9,
    )
    assert reference.q[2] == reference.attitude.theta
    assert reference.u[2] == reference.attitude.theta_dot
    np.testing.assert_array_equal(reference.q[3:], 0.0)
    np.testing.assert_array_equal(reference.u[3:], 0.0)
    assert reference.theta_ddot == reference.attitude.theta_ddot
    np.testing.assert_array_equal(
        reference.state,
        np.hstack((reference.q, reference.u)),
    )


def test_nadir_builder_uses_propagated_orbit_for_attitude_reference():
    dynamics = FakeDynamics()
    orbit = PlanarKeplerianReference(MU, A, 0.0)
    attitude = NadirPointingReference()
    builder = MultiAngleReferenceBuilder(dynamics, orbit, attitude)

    reference = builder.evaluate(0.25 * orbit.period)

    np.testing.assert_allclose(
        reference.centre_of_mass.position,
        [0.0, A],
        atol=1e-8,
    )
    assert np.isclose(reference.attitude.theta, -np.pi)
    assert np.isclose(reference.attitude.theta_dot, orbit.mean_motion)
    assert np.isclose(reference.attitude.theta_ddot, 0.0)
    np.testing.assert_array_equal(reference.q[3:], 0.0)
    np.testing.assert_array_equal(reference.u[3:], 0.0)


def test_nadir_acquisition_builder_uses_moving_orbit_reference():
    dynamics = FakeDynamics()
    orbit = PlanarKeplerianReference(MU, A, 0.0)
    initial_orbit = orbit.evaluate(0.0)
    attitude = NadirAcquisitionReference(duration=100.0)
    attitude.initialise(
        start_time=0.0,
        theta_initial=0.2,
        theta_dot_initial=0.0,
        position=initial_orbit.position,
        velocity=initial_orbit.velocity,
    )
    builder = MultiAngleReferenceBuilder(dynamics, orbit, attitude)

    initial = builder.evaluate(0.0)
    final = builder.evaluate(100.0)
    final_orbit = orbit.evaluate(100.0)
    final_nadir = NadirPointingReference().evaluate(
        100.0,
        position=final_orbit.position,
        velocity=final_orbit.velocity,
    )

    assert initial.attitude.theta == 0.2
    assert initial.attitude.theta_dot == 0.0
    assert initial.attitude.theta_ddot == 0.0
    assert final.attitude == final_nadir
    np.testing.assert_array_equal(final.q[3:], 0.0)
    np.testing.assert_array_equal(final.u[3:], 0.0)


def test_builder_zeroes_real_model_internal_reference_states(
    seven_part_dynamics,
):
    dynamics = seven_part_dynamics
    orbit = PlanarKeplerianReference(
        dynamics.parameter_values["planet_mu"],
        dynamics.parameter_values["orbit_semi_major_axis"],
        dynamics.parameter_values["orbit_eccentricity"],
    )
    attitude = NadirPointingReference()
    builder = MultiAngleReferenceBuilder(dynamics, orbit, attitude)

    reference = builder.evaluate(0.17 * orbit.period)

    excluded_q = {
        builder.coordinate_mapper.q1_index,
        builder.coordinate_mapper.q2_index,
        builder.central_angle_index,
    }
    excluded_u = {
        builder.coordinate_mapper.u1_index,
        builder.coordinate_mapper.u2_index,
        builder.central_speed_index,
    }
    internal_q = [
        reference.q[index]
        for index in range(dynamics.state_dimension)
        if index not in excluded_q
    ]
    internal_u = [
        reference.u[index]
        for index in range(dynamics.state_dimension)
        if index not in excluded_u
    ]

    np.testing.assert_array_equal(internal_q, 0.0)
    np.testing.assert_array_equal(internal_u, 0.0)
    np.testing.assert_allclose(
        dynamics.rG_func(reference.q, reference.u).reshape(-1)[:2],
        reference.centre_of_mass.position,
        atol=1e-8,
    )
    np.testing.assert_allclose(
        dynamics.vG_func(reference.q, reference.u).reshape(-1)[:2],
        reference.centre_of_mass.velocity,
        atol=1e-9,
    )
