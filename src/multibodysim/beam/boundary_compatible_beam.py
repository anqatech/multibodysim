import sympy as sm

from .clamped_clamped_beam import ClampedClampedBeam


class BoundaryCompatibleBeam:
    """Euler-Bernoulli beam helper with Hermite boundary coordinates.

    The displacement field is

        w(s, t) = N(s) d_b(t) + psi(s) eta(t),

    where d_b = [v_i, theta_i, v_j, theta_j]^T is computed elsewhere from
    endpoint motion, and the internal modes satisfy homogeneous clamped
    endpoint conditions.
    """

    def __init__(self, length, E, I, n):
        self.L = sm.sympify(length)
        self.E = sm.sympify(E)
        self.I = sm.sympify(I)
        self.nb_modes = n
        self.internal_beam = ClampedClampedBeam(length=length, E=E, I=I, n=n)

    def boundary_shape_functions_symbolic(self, s):
        s = self._validate_material_coordinate(s)
        xi = s / self.L
        return [
            1 - 3 * xi**2 + 2 * xi**3,
            self.L * (xi - 2 * xi**2 + xi**3),
            3 * xi**2 - 2 * xi**3,
            self.L * (-xi**2 + xi**3),
        ]

    def boundary_shape_second_derivatives_symbolic(self, s):
        s = self._validate_material_coordinate(s)
        xi = s / self.L
        return [
            (-6 + 12 * xi) / self.L**2,
            (-4 + 6 * xi) / self.L,
            (6 - 12 * xi) / self.L**2,
            (-2 + 6 * xi) / self.L,
        ]

    def internal_mode_shape_symbolic(self, s, mode):
        return self.internal_beam.mode_shape_symbolic(s, mode)

    def internal_mode_shapes_symbolic(self, s):
        return [
            self.internal_mode_shape_symbolic(s, mode)
            for mode in range(1, self.nb_modes + 1)
        ]

    def internal_mode_second_derivatives_symbolic(self, s):
        return [
            mode_shape.diff(s, 2)
            for mode_shape in self.internal_mode_shapes_symbolic(s)
        ]

    def curvature_basis_symbolic(self, s):
        return (
            self.boundary_shape_second_derivatives_symbolic(s)
            + self.internal_mode_second_derivatives_symbolic(s)
        )

    def boundary_stiffness_matrix_symbolic(self, s=None):
        s = self._symbol(s)
        boundary_basis = sm.Matrix([self.boundary_shape_second_derivatives_symbolic(s)])
        return self.E * self.I * sm.integrate(
            boundary_basis.T * boundary_basis,
            (s, 0, self.L),
        )

    def boundary_modal_stiffness_matrix_symbolic(self, s=None):
        s = self._symbol(s)
        boundary_basis = sm.Matrix([self.boundary_shape_second_derivatives_symbolic(s)])
        modal_basis = sm.Matrix([self.internal_mode_second_derivatives_symbolic(s)])
        return self.E * self.I * sm.integrate(
            boundary_basis.T * modal_basis,
            (s, 0, self.L),
        )

    def modal_stiffness_matrix_symbolic(self, s=None):
        s = self._symbol(s)
        modal_basis = sm.Matrix([self.internal_mode_second_derivatives_symbolic(s)])
        return self.E * self.I * sm.integrate(
            modal_basis.T * modal_basis,
            (s, 0, self.L),
        )

    def stiffness_matrix_symbolic(self, s=None):
        s = self._symbol(s)
        curvature_basis = sm.Matrix([self.curvature_basis_symbolic(s)])
        return self.E * self.I * sm.integrate(
            curvature_basis.T * curvature_basis,
            (s, 0, self.L),
        )

    def stiffness_blocks_symbolic(self, s=None):
        return {
            "K_bb": self.boundary_stiffness_matrix_symbolic(s),
            "K_b_eta": self.boundary_modal_stiffness_matrix_symbolic(s),
            "K_eta_eta": self.modal_stiffness_matrix_symbolic(s),
        }

    def _symbol(self, s):
        if s is not None:
            return s
        return sm.Symbol("s")

    def _validate_material_coordinate(self, s):
        s = sm.sympify(s)
        length = sm.sympify(self.L)

        if s.free_symbols:
            return s

        if length.free_symbols:
            raise ValueError(
                "Cannot validate a numeric beam coordinate against symbolic length L. "
                "Use a symbolic coordinate, or instantiate the beam with a numeric "
                "length."
            )

        if not 0 <= float(s) <= float(length):
            raise ValueError(
                f"Beam coordinate s must satisfy 0 <= s <= L. Got s={s}, L={length}."
            )

        return s
