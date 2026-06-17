from __future__ import annotations

import numpy as np
import sympy as sm
import sympy.physics.mechanics as me

from multibodysim.analysis import (
    MultiAngleDiagnosticContext,
    compute_angular_momentum_diagnostics,
    compute_energy_diagnostics,
    diagnostic_context_from_simulator,
    initial_strain_energy_by_panel,
    simulation_diagnostics,
    simulation_diagnostics_table,
)


class FakeDynamics:
    def __init__(self):
        self.q = sm.Matrix([sm.Symbol("q1")])
        self.u = sm.Matrix([sm.Symbol("u1")])
        self.state_dimension = 1
        self.planet_mu = sm.Symbol("planet_mu")
        self.parameter_symbols = {"planet_mu": self.planet_mu}
        self.total_mass = sm.S(2)
        self.r_G_norm = self.q[0]
        self.V_strain = self.q[0] ** 2
        self.V_gg = sm.S.Zero

    def _eval_differentials(self, q, u, torques):
        return np.array([[4.0]]), np.array([0.0])


class FakeSimulator:
    def __init__(self):
        self.dynamics = FakeDynamics()
        self.parameter_values = np.array([10.0])
        self.torque_values = np.array([0.0])

    def get_torque_values(self):
        return self.torque_values.copy()


class FakeAngularMomentumDynamics:
    def __init__(self):
        q1 = sm.Symbol("q1")
        u1 = sm.Symbol("u1")
        N = me.ReferenceFrame("N")

        self.q = sm.Matrix([q1])
        self.u = sm.Matrix([u1])
        self.state_dimension = 1
        self.parameter_symbols = {}
        self.parameter_values = {"L": 1.0}
        self.frames = {"inertial": N, "bus": N}
        self.rigid_body_names = ["bus"]
        self.flexible_body_names = []
        self.mass_symbols = {"bus": sm.S(2)}
        self.inertia_matrices = {
            "bus": sm.diag(sm.S.Zero, sm.S.Zero, sm.S(3)),
        }
        self.inertial_position = {"bus": q1 * N.x}
        self.linear_velocities = {"bus": u1 * N.y}
        self.angular_velocities = {"bus": u1 * N.z}
        self.r_G = sm.S.Zero * N.x
        self.v_G = sm.S.Zero * N.x


class FakeAngularMomentumSimulator:
    def __init__(self):
        self.dynamics = FakeAngularMomentumDynamics()
        self.parameter_values = np.array([])
        self.torque_values = np.array([])

    def get_torque_values(self):
        return self.torque_values.copy()


def test_compute_angular_momentum_diagnostics_returns_momentum_and_drift():
    results = {
        "time": np.array([0.0, 1.0, 2.0]),
        "states": np.array(
            [
                [1.0, 2.0],
                [2.0, 3.0],
                [3.0, 4.0],
            ],
        ),
    }

    angular_momentum = compute_angular_momentum_diagnostics(
        diagnostic_context_from_simulator(FakeAngularMomentumSimulator()),
        results,
        sample_every=2,
    )

    assert angular_momentum["time"].to_list() == [0.0, 2.0]
    np.testing.assert_allclose(angular_momentum["H_origin_z"], [10.0, 36.0])
    np.testing.assert_allclose(angular_momentum["H_cm_z"], [10.0, 36.0])
    np.testing.assert_allclose(angular_momentum["H_origin_z_drift"], [0.0, 26.0])
    np.testing.assert_allclose(angular_momentum["H_cm_z_drift"], [0.0, 26.0])
    np.testing.assert_allclose(
        angular_momentum["H_origin_z_relative_drift"],
        [0.0, 2.6],
    )
    assert "H_origin_z_drift_ppm" not in angular_momentum
    assert "H_cm_z_drift_ppm" not in angular_momentum


def test_compute_angular_momentum_diagnostics_rejects_invalid_sampling():
    with np.testing.assert_raises(ValueError):
        compute_angular_momentum_diagnostics(
            diagnostic_context_from_simulator(FakeAngularMomentumSimulator()),
            {},
            sample_every=0,
        )

    with np.testing.assert_raises(ValueError):
        compute_angular_momentum_diagnostics(
            diagnostic_context_from_simulator(FakeAngularMomentumSimulator()),
            {},
            quadrature_points=0,
        )


def test_compute_angular_momentum_diagnostics_rejects_simulator_object():
    with np.testing.assert_raises(TypeError):
        compute_angular_momentum_diagnostics(
            FakeAngularMomentumSimulator(),
            {},
        )


def test_compute_energy_diagnostics_returns_energy_components_and_drift():
    results = {
        "time": np.array([0.0, 1.0, 2.0]),
        "states": np.array(
            [
                [2.0, 3.0],
                [3.0, 4.0],
                [4.0, 5.0],
            ],
        ),
    }

    energy = compute_energy_diagnostics(
        diagnostic_context_from_simulator(FakeSimulator()),
        results,
        sample_every=2,
    )

    assert energy["time"].to_list() == [0.0, 2.0]
    np.testing.assert_allclose(energy["kinetic"], [18.0, 50.0])
    np.testing.assert_allclose(energy["kepler_potential"], [-10.0, -5.0])
    np.testing.assert_allclose(energy["strain_potential"], [4.0, 16.0])
    np.testing.assert_allclose(energy["gravity_gradient_potential"], [0.0, 0.0])
    np.testing.assert_allclose(energy["total_energy"], [12.0, 61.0])
    np.testing.assert_allclose(energy["total_energy_drift"], [0.0, 49.0])
    np.testing.assert_allclose(
        energy["total_energy_relative_drift"],
        [0.0, 49.0 / 12.0],
    )


def test_compute_energy_diagnostics_rejects_invalid_sample_every():
    with np.testing.assert_raises(ValueError):
        compute_energy_diagnostics(
            diagnostic_context_from_simulator(FakeSimulator()),
            {},
            sample_every=0,
        )


def test_compute_energy_diagnostics_rejects_simulator_object():
    with np.testing.assert_raises(TypeError):
        compute_energy_diagnostics(FakeSimulator(), {})


def test_diagnostic_context_from_simulator_copies_numerical_arrays():
    simulator = FakeSimulator()
    context = diagnostic_context_from_simulator(simulator)

    simulator.parameter_values[0] = 99.0
    simulator.torque_values[0] = 88.0

    assert context.dynamics is simulator.dynamics
    np.testing.assert_allclose(context.parameter_values, [10.0])
    np.testing.assert_allclose(context.torque_values, [0.0])


def test_compute_energy_diagnostics_uses_context_torque_values():
    class TorqueSensitiveDynamics(FakeDynamics):
        def _eval_differentials(self, q, u, torques):
            return np.array([[float(torques[0])]]), np.array([0.0])

    results = {
        "time": np.array([0.0]),
        "states": np.array([[2.0, 3.0]]),
    }
    context = MultiAngleDiagnosticContext(
        dynamics=TorqueSensitiveDynamics(),
        parameter_values=np.array([10.0]),
        torque_values=np.array([6.0]),
    )

    energy = compute_energy_diagnostics(context, results)

    np.testing.assert_allclose(energy["kinetic"], [27.0])


def test_initial_strain_energy_by_panel_returns_panel_rows_and_total():
    class FakePanelDynamics:
        def __init__(self):
            q1, q2 = sm.symbols("q1 q2")
            self.q = sm.Matrix([q1, q2])
            self.u = sm.Matrix([])
            self.state_dimension = 2
            self.parameter_symbols = {}
            self.outer_flexible_panels = ["panel_outer"]
            self.inter_bus_flexible_panels = ["panel_inner"]
            self.boundary_compatible_stiffness_matrices = {
                "panel_inner": sm.Matrix([[4.0, 1.0], [1.0, 6.0]]),
            }
            self.element_coordinates = {
                "panel_inner": sm.Matrix([q1, q2]),
            }

        def _outer_panel_modal_strain_energy(self, panel):
            assert panel == "panel_outer"
            return sm.Rational(1, 2) * 10.0 * self.q[0] ** 2

    context = MultiAngleDiagnosticContext(
        dynamics=FakePanelDynamics(),
        parameter_values=np.array([]),
        torque_values=np.array([]),
    )

    strain_energy = initial_strain_energy_by_panel(
        context,
        np.array([2.0, 3.0]),
    )

    assert strain_energy["panel"].to_list() == [
        "panel_outer",
        "panel_inner",
        "total",
    ]
    assert strain_energy["panel_kind"].to_list() == [
        "outer_modal",
        "boundary_compatible",
        "total",
    ]
    assert "strain_energy_J" not in strain_energy
    np.testing.assert_allclose(
        strain_energy["strain_energy_mJ"],
        [20_000.0, 41_000.0, 61_000.0],
    )


def test_initial_strain_energy_by_panel_rejects_simulator_object():
    with np.testing.assert_raises(TypeError):
        initial_strain_energy_by_panel(FakeSimulator(), np.array([1.0, 2.0]))


def test_simulation_diagnostics_computes_attitude_and_flexible_metrics():
    results = {
        "success": True,
        "nfev": 10,
        "njev": 2,
        "nlu": 3,
        "q_central_angle": np.deg2rad(np.array([90.0, 91.0, 89.0])),
        "u_central_angle": np.deg2rad(np.array([0.0, 0.5, -1.0])),
        "eta1_1": np.array([0.0, 0.2, -0.1]),
        "zeta1_1": np.array([0.0, -0.3, 0.1]),
    }

    metrics = simulation_diagnostics(results)

    assert metrics["success"] is True
    assert metrics["nfev"] == 10
    assert np.isclose(metrics["central_angle_final_deg"], 89.0)
    assert np.isclose(metrics["central_angle_drift_deg"], -1.0)
    assert np.isclose(metrics["central_angle_peak_to_peak_deg"], 2.0)
    assert np.isclose(metrics["central_angle_speed_peak_abs_deg_s"], 1.0)
    assert np.isclose(metrics["eta1_1_peak_abs"], 0.2)
    assert np.isclose(metrics["eta1_1_rms"], np.sqrt((0.0**2 + 0.2**2 + 0.1**2) / 3.0))
    assert np.isclose(metrics["eta1_1_final_abs"], 0.1)
    assert np.isclose(metrics["zeta1_1_peak_abs"], 0.3)


def test_simulation_diagnostics_table_returns_display_rows():
    metrics = {
        "success": True,
        "nfev": 10,
        "central_angle_final_deg": 90.0,
        "eta1_1_peak_abs": 0.1,
    }

    rows, columns = simulation_diagnostics_table(metrics)

    assert columns == ["Metric", "Unit", "Value"]
    assert rows == [
        ("Solver success", "-", True),
        ("Function evaluations", "-", 10),
        ("Final attitude", "deg", 90.0),
        ("eta1_1_peak_abs", "-", 0.1),
    ]
