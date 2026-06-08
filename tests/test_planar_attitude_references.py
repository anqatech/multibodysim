from __future__ import annotations

import numpy as np
import pytest

from multibodysim.references import (
    InertialRestToRestReference,
    NadirPointingReference,
)


def test_inertial_reference_requires_initialisation():
    reference = InertialRestToRestReference(
        theta_target=1.0,
        duration=10.0,
    )

    with pytest.raises(RuntimeError, match="must be initialised"):
        reference.evaluate(0.0)


def test_inertial_reference_returns_rest_to_rest_trajectory():
    reference = InertialRestToRestReference(
        theta_target=1.25,
        duration=10.0,
    )
    reference.initialise(start_time=2.0, theta_initial=0.25)

    initial = reference.evaluate(2.0)
    midpoint = reference.evaluate(7.0)
    final = reference.evaluate(12.0)

    assert initial.theta == 0.25
    assert initial.theta_dot == 0.0
    assert initial.theta_ddot == 0.0
    assert np.isclose(midpoint.theta, 0.75)
    assert np.isclose(midpoint.theta_dot, 0.1875)
    assert np.isclose(midpoint.theta_ddot, 0.0)
    assert final.theta == 1.25
    assert final.theta_dot == 0.0
    assert final.theta_ddot == 0.0


def test_inertial_reference_reset_allows_new_initial_state():
    reference = InertialRestToRestReference(
        theta_target=1.0,
        duration=10.0,
    )
    reference.initialise(start_time=0.0, theta_initial=0.0)
    reference.reset()
    reference.initialise(start_time=5.0, theta_initial=0.5)

    state = reference.evaluate(5.0)

    assert state.theta == 0.5
    assert reference.delta_theta == 0.5


def test_nadir_reference_matches_existing_circular_orbit_kinematics():
    reference = NadirPointingReference()

    state = reference.evaluate(
        0.0,
        position=(1.0, 0.0),
        velocity=(0.0, 1.0),
    )

    assert np.isclose(state.theta, 0.5 * np.pi)
    assert np.isclose(state.theta_dot, 1.0)
    assert np.isclose(state.theta_ddot, 0.0)


def test_nadir_reference_applies_pointing_offset():
    reference = NadirPointingReference(offset=0.25)

    state = reference.evaluate(
        0.0,
        position=(-1.0, 0.0),
        velocity=(0.0, -1.0),
    )

    assert np.isclose(state.theta, 0.25)
    assert np.isclose(state.theta_dot, 1.0)


def test_nadir_reference_rejects_zero_orbit_radius():
    reference = NadirPointingReference()

    with pytest.raises(ValueError, match="non-zero orbital radius"):
        reference.evaluate(
            0.0,
            position=(0.0, 0.0),
            velocity=(0.0, 0.0),
        )
