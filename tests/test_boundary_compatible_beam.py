import numpy as np
import sympy as sm

from multibodysim.beam.boundary_compatible_beam import BoundaryCompatibleBeam


def assert_symbolic_matrix_equal(lhs, rhs):
    assert lhs.shape == rhs.shape
    diff = lhs - rhs
    assert all(sm.simplify(value) == 0 for value in diff)


def assert_symbolic_matrix_close(lhs, rhs, atol=1e-9):
    assert lhs.shape == rhs.shape
    residuals = [abs(float(sm.N(value))) for value in lhs - rhs]
    assert np.allclose(residuals, 0.0, atol=atol)


def test_boundary_shape_functions_satisfy_endpoint_interpolation():
    s = sm.Symbol("s")
    beam = BoundaryCompatibleBeam(length=sm.Integer(3), E=1, I=1, n=1)

    shapes = beam.boundary_shape_functions_symbolic(s)
    slopes = [shape.diff(s) for shape in shapes]

    assert [sm.simplify(shape.subs(s, 0)) for shape in shapes] == [1, 0, 0, 0]
    assert [sm.simplify(shape.subs(s, beam.L)) for shape in shapes] == [0, 0, 1, 0]

    assert [sm.simplify(slope.subs(s, 0)) for slope in slopes] == [0, 1, 0, 0]
    assert [sm.simplify(slope.subs(s, beam.L)) for slope in slopes] == [0, 0, 0, 1]


def test_boundary_shape_second_derivatives_match_direct_derivatives():
    s = sm.Symbol("s")
    beam = BoundaryCompatibleBeam(length=sm.Integer(3), E=1, I=1, n=1)

    shapes = beam.boundary_shape_functions_symbolic(s)
    direct_second_derivatives = [shape.diff(s, 2) for shape in shapes]
    helper_second_derivatives = beam.boundary_shape_second_derivatives_symbolic(s)

    for direct, helper in zip(direct_second_derivatives, helper_second_derivatives):
        assert sm.simplify(direct - helper) == 0


def test_boundary_stiffness_matrix_matches_euler_bernoulli_matrix():
    s = sm.Symbol("s")
    length = sm.Symbol("L", positive=True)
    E = sm.Symbol("E", positive=True)
    I = sm.Symbol("I", positive=True)
    beam = BoundaryCompatibleBeam(length=length, E=E, I=I, n=0)

    expected = E * I / length**3 * sm.Matrix(
        [
            [12, 6 * length, -12, 6 * length],
            [6 * length, 4 * length**2, -6 * length, 2 * length**2],
            [-12, -6 * length, 12, -6 * length],
            [6 * length, 2 * length**2, -6 * length, 4 * length**2],
        ]
    )

    actual = beam.boundary_stiffness_matrix_symbolic(s)
    assert_symbolic_matrix_equal(actual, expected)


def test_stiffness_blocks_match_full_stiffness_matrix_partition():
    s = sm.Symbol("s")
    beam = BoundaryCompatibleBeam(length=sm.Integer(3), E=sm.Integer(5), I=sm.Integer(7), n=1)

    blocks = beam.stiffness_blocks_symbolic(s)
    full_matrix = beam.stiffness_matrix_symbolic(s)

    assert blocks["K_bb"].shape == (4, 4)
    assert blocks["K_b_eta"].shape == (4, 1)
    assert blocks["K_eta_eta"].shape == (1, 1)

    assert_symbolic_matrix_close(full_matrix[:4, :4], blocks["K_bb"])
    assert_symbolic_matrix_close(full_matrix[:4, 4:], blocks["K_b_eta"])
    assert_symbolic_matrix_close(full_matrix[4:, :4], blocks["K_b_eta"].T)
    assert_symbolic_matrix_close(full_matrix[4:, 4:], blocks["K_eta_eta"])
    assert_symbolic_matrix_close(full_matrix, full_matrix.T)


def test_boundary_stiffness_does_not_penalise_rigid_body_motion():
    s = sm.Symbol("s")
    beam = BoundaryCompatibleBeam(length=sm.Integer(3), E=sm.Integer(5), I=sm.Integer(7), n=0)
    stiffness = beam.boundary_stiffness_matrix_symbolic(s)

    translation = sm.Matrix([2, 0, 2, 0])
    rotation = sm.Matrix([0, 2, 2 * beam.L, 2])

    translation_energy = (translation.T * stiffness * translation)[0] / 2
    rotation_energy = (rotation.T * stiffness * rotation)[0] / 2

    assert sm.simplify(translation_energy) == 0
    assert sm.simplify(rotation_energy) == 0


def test_internal_modes_satisfy_homogeneous_endpoint_conditions():
    s = sm.Symbol("s")
    beam = BoundaryCompatibleBeam(length=sm.Integer(3), E=1, I=1, n=1)

    mode = beam.internal_mode_shape_symbolic(s, mode=1)
    slope = mode.diff(s)

    endpoint_values = [
        mode.subs(s, 0),
        slope.subs(s, 0),
        mode.subs(s, beam.L),
        slope.subs(s, beam.L),
    ]

    assert np.allclose([float(sm.N(value)) for value in endpoint_values], 0.0, atol=1e-10)
