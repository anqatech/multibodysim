from __future__ import annotations

import numpy as np
import pytest

from multibodysim.references import PlanarKeplerianReference


MU = 3.986004418e14
A = 6_778_000.0


def test_circular_kepler_reference_matches_expected_quarter_orbit():
    reference = PlanarKeplerianReference(
        gravitational_parameter=MU,
        semi_major_axis=A,
        eccentricity=0.0,
    )

    initial = reference.evaluate(0.0)
    quarter = reference.evaluate(0.25 * reference.period)

    speed = np.sqrt(MU / A)
    np.testing.assert_allclose(initial.position, [A, 0.0], atol=1e-8)
    np.testing.assert_allclose(initial.velocity, [0.0, speed], atol=1e-9)
    np.testing.assert_allclose(quarter.position, [0.0, A], atol=1e-8)
    np.testing.assert_allclose(
        quarter.velocity,
        [-speed, 0.0],
        atol=1e-9,
    )


def test_eccentric_kepler_reference_matches_periapsis_state():
    eccentricity = 0.2
    reference = PlanarKeplerianReference(
        gravitational_parameter=MU,
        semi_major_axis=A,
        eccentricity=eccentricity,
    )

    state = reference.evaluate(0.0)

    radius = A * (1.0 - eccentricity)
    speed = np.sqrt(
        MU * (1.0 + eccentricity)
        / (A * (1.0 - eccentricity))
    )
    np.testing.assert_allclose(state.position, [radius, 0.0], atol=1e-8)
    np.testing.assert_allclose(state.velocity, [0.0, speed], atol=1e-9)


def test_eccentric_kepler_reference_conserves_orbital_energy():
    reference = PlanarKeplerianReference(
        gravitational_parameter=MU,
        semi_major_axis=A,
        eccentricity=0.4,
        initial_true_anomaly=0.7,
    )
    expected_energy = -MU / (2.0 * A)

    for fraction in np.linspace(0.0, 1.0, 9):
        state = reference.evaluate(fraction * reference.period)
        radius = np.linalg.norm(state.position)
        speed_squared = state.velocity @ state.velocity
        energy = 0.5 * speed_squared - MU / radius
        assert np.isclose(energy, expected_energy, rtol=1e-13)


def test_kepler_reference_repeats_after_one_period():
    reference = PlanarKeplerianReference(
        gravitational_parameter=MU,
        semi_major_axis=A,
        eccentricity=0.3,
        reference_time=12.0,
        initial_true_anomaly=0.8,
    )

    initial = reference.evaluate(12.0)
    repeated = reference.evaluate(12.0 + reference.period)

    np.testing.assert_allclose(repeated.position, initial.position, atol=1e-8)
    np.testing.assert_allclose(repeated.velocity, initial.velocity, atol=1e-9)


@pytest.mark.parametrize(
    ("keyword", "value"),
    [
        ("gravitational_parameter", 0.0),
        ("semi_major_axis", 0.0),
        ("eccentricity", -0.1),
        ("eccentricity", 1.0),
    ],
)
def test_kepler_reference_rejects_invalid_orbit(keyword, value):
    arguments = {
        "gravitational_parameter": MU,
        "semi_major_axis": A,
        "eccentricity": 0.0,
    }
    arguments[keyword] = value

    with pytest.raises(ValueError):
        PlanarKeplerianReference(**arguments)
