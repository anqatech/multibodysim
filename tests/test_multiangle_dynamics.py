from __future__ import annotations

import numpy as np
import sympy as sm
import pytest

from multibodysim.beam.cantilever_beam import CantileverBeam
from multibodysim.beam.boundary_compatible_beam import BoundaryCompatibleBeam
from multibodysim.beam.clamped_clamped_beam import ClampedClampedBeam
from multibodysim.multiangle import MultiAngleFlexibleDynamics


def assert_symbolic_equal(lhs, rhs):
    assert sm.simplify(lhs - rhs) == 0


def assert_vector_equal(lhs, rhs, frame):
    diff = lhs - rhs
    assert all(
        sm.trigsimp(sm.simplify(component)) == 0
        for component in diff.to_matrix(frame)
    )


def assert_vectors_do_not_contain_coordinate_derivatives(vectors, dynamics):
    inertial_frame = dynamics.frames["inertial"]
    forbidden = set(dynamics.qd)
    forbidden.update(qdi.diff(dynamics.t) for qdi in dynamics.qd)

    for vector in vectors.values():
        for component in vector.to_matrix(inertial_frame):
            assert not (component.atoms(sm.Derivative) & forbidden)


def test_multiangle_bus_orientation_convention_for_seven_part_chain(seven_part_dynamics):
    dynamics = seven_part_dynamics

    q31 = dynamics.bus_angle_coordinates["bus_1"]
    q32 = dynamics.bus_angle_coordinates["bus_2"]
    q33 = dynamics.bus_angle_coordinates["bus_3"]

    assert str(q31.func) == "q_relative_angle_bus_1"
    assert str(q32.func) == "q_central_angle"
    assert str(q33.func) == "q_relative_angle_bus_3"

    assert_symbolic_equal(dynamics.orientation_angle("bus_1"), q32 + sm.pi + q31)
    assert_symbolic_equal(dynamics.orientation_angle("bus_2"), q32)
    assert_symbolic_equal(dynamics.orientation_angle("bus_3"), q32 + q33)


def test_multiangle_panel_orientation_uses_directed_endpoint_tangent_average(seven_part_dynamics):
    dynamics = seven_part_dynamics

    q31 = dynamics.bus_angle_coordinates["bus_1"]
    q32 = dynamics.bus_angle_coordinates["bus_2"]
    q33 = dynamics.bus_angle_coordinates["bus_3"]

    assert_symbolic_equal(dynamics.orientation_angle("panel_1"), q32 + sm.pi + q31)
    assert_symbolic_equal(
        dynamics.orientation_angle("panel_2"),
        q32 + sm.pi + q31 / 2,
    )
    assert_symbolic_equal(
        dynamics.orientation_angle("panel_3"),
        q32 + q33 / 2,
    )
    assert_symbolic_equal(dynamics.orientation_angle("panel_4"), q32 + q33)


def test_multiangle_classifies_flexible_panels_from_adjacency(seven_part_dynamics):
    dynamics = seven_part_dynamics

    assert dynamics.outer_flexible_panels == ["panel_1", "panel_4"]
    assert dynamics.inter_bus_flexible_panels == ["panel_2", "panel_3"]
    assert dynamics.flexible_panel_connections == {
        "panel_1": {"kind": "outer", "buses": ("bus_1",)},
        "panel_2": {"kind": "inter-bus", "buses": ("bus_1", "bus_2")},
        "panel_3": {"kind": "inter-bus", "buses": ("bus_2", "bus_3")},
        "panel_4": {"kind": "outer", "buses": ("bus_3",)},
    }


def test_multiangle_convention_scales_to_more_buses(eleven_part_dynamics):
    dynamics = eleven_part_dynamics

    q31 = dynamics.bus_angle_coordinates["bus_1"]
    q32 = dynamics.bus_angle_coordinates["bus_2"]
    q33 = dynamics.bus_angle_coordinates["bus_3"]
    q34 = dynamics.bus_angle_coordinates["bus_4"]
    q35 = dynamics.bus_angle_coordinates["bus_5"]

    assert_symbolic_equal(dynamics.orientation_angle("bus_1"), q33 + sm.pi + q31)
    assert_symbolic_equal(dynamics.orientation_angle("bus_2"), q33 + sm.pi + q32)
    assert_symbolic_equal(dynamics.orientation_angle("bus_3"), q33)
    assert_symbolic_equal(dynamics.orientation_angle("bus_4"), q33 + q34)
    assert_symbolic_equal(dynamics.orientation_angle("bus_5"), q33 + q35)

    assert_symbolic_equal(dynamics.orientation_angle("panel_1"), q33 + sm.pi + q31)
    assert_symbolic_equal(
        dynamics.orientation_angle("panel_2"),
        q33 + sm.pi + (q31 + q32) / 2,
    )
    assert_symbolic_equal(
        dynamics.orientation_angle("panel_3"),
        q33 + sm.pi + q32 / 2,
    )
    assert_symbolic_equal(
        dynamics.orientation_angle("panel_4"),
        q33 + q34 / 2,
    )
    assert_symbolic_equal(
        dynamics.orientation_angle("panel_5"),
        q33 + (q34 + q35) / 2,
    )
    assert_symbolic_equal(dynamics.orientation_angle("panel_6"), q33 + q35)


def test_multiangle_panel_classification_scales_to_more_buses(eleven_part_dynamics):
    dynamics = eleven_part_dynamics

    assert dynamics.outer_flexible_panels == ["panel_1", "panel_6"]
    assert dynamics.inter_bus_flexible_panels == [
        "panel_2",
        "panel_3",
        "panel_4",
        "panel_5",
    ]
    assert dynamics.flexible_panel_connections["panel_4"] == {
        "kind": "inter-bus",
        "buses": ("bus_3", "bus_4"),
    }


def test_multiangle_symbol_names_reject_non_bus_names(seven_part_dynamics):
    dynamics = seven_part_dynamics

    with pytest.raises(ValueError, match="bus_<number>"):
        dynamics._angle_symbol_name("wer")

    with pytest.raises(ValueError, match="bus_<number>"):
        dynamics._speed_symbol_name("wer")


def test_multiangle_defines_flexible_modal_symbols_and_state_vectors(seven_part_dynamics):
    dynamics = seven_part_dynamics

    q_names = [str(symbol.func) for symbol in dynamics.q]
    u_names = [str(symbol.func) for symbol in dynamics.u]

    assert q_names == [
        "q1",
        "q2",
        "q_relative_angle_bus_1",
        "q_central_angle",
        "q_relative_angle_bus_3",
        "eta1_1",
        "eta2_1",
        "eta2_2",
        "eta3_1",
        "eta3_2",
        "eta4_1",
    ]
    assert u_names == [
        "u1",
        "u2",
        "u_relative_angle_bus_1",
        "u_central_angle",
        "u_relative_angle_bus_3",
        "zeta1_1",
        "zeta2_1",
        "zeta2_2",
        "zeta3_1",
        "zeta3_2",
        "zeta4_1",
    ]

    assert len(dynamics.q) == len(dynamics.u) == 11
    assert dynamics.flexible_bodies["panel_2"]["beam_type"] == "clamped-clamped"
    assert len(dynamics.flexible_bodies["panel_2"]["eta_list"]) == 2
    assert len(dynamics.flexible_bodies["panel_2"]["zeta_list"]) == 2

    assert dynamics.flex_eta_index[("panel_1", 0)] == 0
    assert dynamics.flex_eta_index[("panel_2", 0)] == 1
    assert dynamics.flex_eta_index[("panel_2", 1)] == 2
    assert dynamics.flex_eta_index[("panel_3", 0)] == 3
    assert dynamics.flex_eta_index[("panel_3", 1)] == 4
    assert dynamics.flex_eta_index[("panel_4", 0)] == 5

    assert dynamics.flex_zeta_index[("panel_1", 0)] == 0
    assert dynamics.flex_zeta_index[("panel_2", 0)] == 1
    assert dynamics.flex_zeta_index[("panel_2", 1)] == 2
    assert dynamics.flex_zeta_index[("panel_3", 0)] == 3
    assert dynamics.flex_zeta_index[("panel_3", 1)] == 4
    assert dynamics.flex_zeta_index[("panel_4", 0)] == 5


def test_multiangle_defines_static_parameter_symbols_from_parameters(seven_part_dynamics):
    dynamics = seven_part_dynamics

    assert dynamics.parameter_symbols["D"] == sm.symbols("D")
    assert dynamics.parameter_symbols["L"] == sm.symbols("L")
    assert dynamics.D == sm.symbols("D")
    assert dynamics.L == sm.symbols("L")
    assert dynamics.mass_symbols["bus_2"] == sm.symbols("m_bus_2")
    assert dynamics.mass_symbols["panel_4"] == sm.symbols("m_panel_4")
    assert dynamics.E_mod == sm.symbols("E_mod")
    assert dynamics.I_area == sm.symbols("I_area")
    assert dynamics.planet_mu == sm.symbols("planet_mu")
    assert dynamics.orbit_semi_major_axis == sm.symbols("orbit_semi_major_axis")
    assert dynamics.orbit_eccentricity == sm.symbols("orbit_eccentricity")

    p_names = [str(symbol) for symbol in dynamics.p]
    assert p_names == [
        "D",
        "L",
        "m_bus_1",
        "m_bus_2",
        "m_bus_3",
        "m_panel_1",
        "m_panel_2",
        "m_panel_3",
        "m_panel_4",
        "E_mod",
        "I_area",
        "planet_mu",
        "orbit_semi_major_axis",
        "orbit_eccentricity",
    ]


def test_multiangle_parameter_symbols_scale_with_body_count(eleven_part_dynamics):
    dynamics = eleven_part_dynamics

    assert dynamics.mass_symbols["bus_5"] == sm.symbols("m_bus_5")
    assert dynamics.mass_symbols["panel_6"] == sm.symbols("m_panel_6")
    assert len(dynamics.mass_symbols) == 11
    assert len(dynamics.p) == 18


def test_multiangle_requires_mass_parameter_for_each_body(seven_part_config):
    config = seven_part_config
    del config["parameters"]["m_panel_4"]

    with pytest.raises(KeyError, match="m_panel_4"):
        MultiAngleFlexibleDynamics(config)


@pytest.mark.parametrize(
    ("single_angle_fixture", "multiangle_fixture", "expected_gravity_gradient"),
    [
        (
            "distributed_7part_zf_gg_off_single_angle_config",
            "distributed_7part_zf_gg_off_multiangle_config",
            False,
        ),
        (
            "distributed_7part_zf_gg_on_single_angle_config",
            "distributed_7part_zf_gg_on_multiangle_config",
            True,
        ),
        (
            "distributed_7part_nzf_gg_off_single_angle_config",
            "distributed_7part_nzf_gg_off_multiangle_config",
            False,
        ),
        (
            "distributed_7part_nzf_gg_on_single_angle_config",
            "distributed_7part_nzf_gg_on_multiangle_config",
            True,
        ),
    ],
)
def test_distributed_7part_multiangle_fixtures_adapt_single_angle_initial_states(
    request,
    single_angle_fixture,
    multiangle_fixture,
    expected_gravity_gradient,
):
    single_angle_config = request.getfixturevalue(single_angle_fixture)
    multiangle_config = request.getfixturevalue(multiangle_fixture)

    assert single_angle_config["central_body"] == "bus_2"
    assert multiangle_config["central_body"] == "bus_2"
    assert multiangle_config["enable_gravity_gradient"] is expected_gravity_gradient

    assert "q_central_angle" in single_angle_config["q_initial"]
    assert "u_central_angle" in single_angle_config["initial_speeds"]
    assert "q3" not in multiangle_config["q_initial"]
    assert "u3" not in multiangle_config["initial_speeds"]

    assert multiangle_config["q_initial"]["q_relative_angle_bus_1"] == 0.0
    assert (
        multiangle_config["q_initial"]["q_central_angle"]
        == single_angle_config["q_initial"]["q_central_angle"]
    )
    assert multiangle_config["q_initial"]["q_relative_angle_bus_3"] == 0.0
    assert multiangle_config["initial_speeds"]["u_relative_angle_bus_1"] == 0.0
    assert (
        multiangle_config["initial_speeds"]["u_central_angle"]
        == single_angle_config["initial_speeds"]["u_central_angle"]
    )
    assert multiangle_config["initial_speeds"]["u_relative_angle_bus_3"] == 0.0

    state_atol = multiangle_config["sim_parameters"]["state_atol"]
    assert "q_central_angle" in state_atol
    assert "u_central_angle" in state_atol


def test_multiangle_defines_bus_torque_symbols_for_seven_part_chain(seven_part_dynamics):
    dynamics = seven_part_dynamics

    assert dynamics.bus_torque_symbols == {
        "bus_1": sm.symbols("tau_1"),
        "bus_2": sm.symbols("tau_2"),
        "bus_3": sm.symbols("tau_3"),
    }
    assert list(dynamics.tau) == [
        sm.symbols("tau_1"),
        sm.symbols("tau_2"),
        sm.symbols("tau_3"),
    ]


def test_multiangle_torque_symbols_scale_with_body_count(eleven_part_dynamics):
    dynamics = eleven_part_dynamics

    assert dynamics.bus_torque_symbols["bus_5"] == sm.symbols("tau_5")
    assert list(dynamics.tau) == [
        sm.symbols("tau_1"),
        sm.symbols("tau_2"),
        sm.symbols("tau_3"),
        sm.symbols("tau_4"),
        sm.symbols("tau_5"),
    ]


def test_multiangle_defines_mode_shapes_for_flexible_bodies(seven_part_dynamics):
    dynamics = seven_part_dynamics

    panel_1 = dynamics.flexible_bodies["panel_1"]
    panel_2 = dynamics.flexible_bodies["panel_2"]

    assert isinstance(panel_1["beam"], CantileverBeam)
    assert isinstance(panel_2["beam"], ClampedClampedBeam)
    assert panel_1["beam"].L == 3.0
    assert panel_1["beam"].E == 140e9
    assert panel_1["beam"].I == 2.5e-8

    assert len(panel_1["phi_list"]) == 1
    assert len(panel_1["phi_mean_list"]) == 1
    assert len(panel_1["k_modal_list"]) == 1
    assert len(panel_1["phi_norm_list"]) == 1
    assert len(panel_1["phi_m1_list"]) == 1

    assert len(panel_2["phi_list"]) == 2
    assert len(panel_2["phi_mean_list"]) == 2
    assert len(panel_2["k_modal_list"]) == 2
    assert len(panel_2["phi_norm_list"]) == 2
    assert len(panel_2["phi_m1_list"]) == 2

    assert all(dynamics.s in phi.free_symbols for phi in panel_2["phi_list"])
    assert all(value > 0 for value in panel_2["phi_norm_list"])
    assert all(value > 0 for value in panel_2["k_modal_list"])


def test_multiangle_rejects_unrecognised_beam_type_when_defining_mode_shapes(
    seven_part_config,
):
    config = seven_part_config
    config["flexible_types"]["panel_1"] = "mystery-beam"
    config["beam_parameters"]["mystery-beam"] = {"nb_modes": 1}

    with pytest.raises(TypeError, match="Unrecognised beam type"):
        MultiAngleFlexibleDynamics(config)


def test_multiangle_defines_rigid_bus_inertia_matrices(seven_part_dynamics):
    dynamics = seven_part_dynamics

    m_bus_2 = dynamics.mass_symbols["bus_2"]
    expected = sm.Matrix(
        [
            [m_bus_2 * dynamics.D**2 / 12, 0, 0],
            [0, m_bus_2 * dynamics.D**2 / 12, 0],
            [0, 0, m_bus_2 * dynamics.D**2 / 6],
        ]
    )

    assert dynamics.inertia_matrices["bus_2"] == expected
    assert set(dynamics.inertia_matrices) == set(dynamics.body_names)


def test_multiangle_outer_panel_inertia_matrices_use_modal_mass_properties(
    seven_part_dynamics,
):
    dynamics = seven_part_dynamics

    inertia = dynamics.inertia_matrices["panel_1"]
    m_panel_1 = dynamics.mass_symbols["panel_1"]
    eta11 = dynamics.flexible_bodies["panel_1"]["eta_list"][0]

    assert inertia.shape == (3, 3)
    assert sm.simplify(inertia[1, 1] - m_panel_1 * dynamics.L**2 / 12) == 0
    assert sm.simplify(inertia[0, 1] - inertia[1, 0]) == 0
    assert inertia[0, 2] == 0
    assert inertia[1, 2] == 0
    assert sm.simplify(inertia[2, 2] - (inertia[0, 0] + inertia[1, 1])) == 0
    assert inertia[0, 0].has(eta11)
    assert inertia[0, 1].has(eta11)


def test_multiangle_inter_bus_panel_inertia_matrices_use_boundary_compatible_mass_properties(
    seven_part_dynamics,
):
    dynamics = seven_part_dynamics

    panel = "panel_2"
    inertia = dynamics.inertia_matrices[panel]
    mass = dynamics.mass_symbols[panel]
    displacement = dynamics._inter_bus_panel_boundary_compatible_displacement(panel)
    mean_displacement = dynamics._flexible_center_of_mass_displacement_sum(panel)
    centred_displacement = displacement - mean_displacement
    relative_angle = dynamics.bus_angle_coordinates["bus_1"]

    expected_I11 = mass / dynamics.L * dynamics._integrate_flexible_body_expression(
        panel,
        centred_displacement**2,
    )
    expected_I12 = -mass / dynamics.L * dynamics._integrate_flexible_body_expression(
        panel,
        (dynamics.s - dynamics.L / 2) * centred_displacement,
    )

    assert inertia.shape == (3, 3)
    assert_symbolic_equal(inertia[0, 0], expected_I11)
    assert_symbolic_equal(inertia[1, 1], mass * dynamics.L**2 / 12)
    assert_symbolic_equal(inertia[0, 1], expected_I12)
    assert_symbolic_equal(inertia[1, 0], expected_I12)
    assert inertia[0, 2] == 0
    assert inertia[1, 2] == 0
    assert_symbolic_equal(inertia[2, 2], inertia[0, 0] + inertia[1, 1])
    assert inertia[0, 0].has(relative_angle)
    assert inertia[0, 1].has(relative_angle)


def test_multiangle_offset_from_rigid_bus_to_flexible_panel(seven_part_dynamics):
    dynamics = seven_part_dynamics

    left_offset = dynamics._get_offset_vector("bus_2", "panel_2")
    right_offset = dynamics._get_offset_vector("bus_2", "panel_3")
    outer_left_offset = dynamics._get_offset_vector("bus_1", "panel_1")
    outer_right_offset = dynamics._get_offset_vector("bus_3", "panel_4")

    assert left_offset == -dynamics.D / 2 * dynamics.frames["bus_2"].x
    assert right_offset == dynamics.D / 2 * dynamics.frames["bus_2"].x
    assert outer_left_offset == dynamics.D / 2 * dynamics.frames["bus_1"].x
    assert outer_right_offset == dynamics.D / 2 * dynamics.frames["bus_3"].x


def test_multiangle_offset_from_flexible_panel_to_rigid_bus_uses_tip_deflection(seven_part_dynamics):
    dynamics = seven_part_dynamics

    offset = dynamics._get_offset_vector("panel_2", "bus_1")
    eta_list = dynamics.flexible_bodies["panel_2"]["eta_list"]
    phi_list = dynamics.flexible_bodies["panel_2"]["phi_list"]
    expected_tip_deflection = sum(
        phi_k.subs(dynamics.s, dynamics.L) * eta_k
        for phi_k, eta_k in zip(phi_list, eta_list)
    )
    expected = (
        dynamics.L * dynamics.frames["panel_2"].x
        + expected_tip_deflection * dynamics.frames["panel_2"].y
    )

    assert offset == expected


def test_multiangle_classifies_supported_connection_kinds(seven_part_dynamics):
    dynamics = seven_part_dynamics

    assert dynamics._connection_kind("bus_2", "panel_2") == "rigid_to_flexible"
    assert dynamics._connection_kind("panel_2", "bus_1") == "flexible_to_rigid"


def test_multiangle_connection_kind_rejects_unsupported_body_type_pairs(seven_part_dynamics):
    dynamics = seven_part_dynamics

    with pytest.raises(NotImplementedError, match="two rigid bodies"):
        dynamics._connection_kind("bus_1", "bus_2")

    with pytest.raises(NotImplementedError, match="two flexible bodies"):
        dynamics._connection_kind("panel_1", "panel_2")


def test_multiangle_offset_rejects_unsupported_body_type_pairs(seven_part_dynamics):
    dynamics = seven_part_dynamics

    with pytest.raises(NotImplementedError, match="two rigid bodies"):
        dynamics._get_offset_vector("bus_1", "bus_2")

    with pytest.raises(NotImplementedError, match="two flexible bodies"):
        dynamics._get_offset_vector("panel_1", "panel_2")


def test_multiangle_defines_expected_points_for_seven_part_chain(seven_part_dynamics):
    dynamics = seven_part_dynamics

    expected_points = {
        "N",
        "bus_1",
        "bus_2",
        "bus_3",
        "joint_panel_2_bus_2",
        "joint_panel_3_bus_2",
        "joint_bus_1_panel_2",
        "joint_bus_3_panel_3",
        "joint_panel_1_bus_1",
        "joint_panel_4_bus_3",
        "dm_panel_1",
        "dm_panel_2",
        "dm_panel_3",
        "dm_panel_4",
        "dm_center_of_mass_panel_1",
        "dm_center_of_mass_panel_2",
        "dm_center_of_mass_panel_3",
        "dm_center_of_mass_panel_4",
    }

    assert expected_points.issubset(dynamics.points)
    assert all(dynamics.inertial_position[body] is not None for body in dynamics.body_names)


def test_multiangle_defines_boundary_points_for_inter_bus_panels(seven_part_dynamics):
    dynamics = seven_part_dynamics

    assert dynamics.boundary_points == {
        "panel_2": {
            "root_bus": "bus_2",
            "tip_bus": "bus_1",
            "root_joint": "joint_panel_2_bus_2",
            "tip_joint": "joint_bus_1_panel_2",
        },
        "panel_3": {
            "root_bus": "bus_2",
            "tip_bus": "bus_3",
            "root_joint": "joint_panel_3_bus_2",
            "tip_joint": "joint_bus_3_panel_3",
        },
    }


def test_multiangle_defines_boundary_coordinates_from_current_panel_geometry(
    seven_part_dynamics,
):
    dynamics = seven_part_dynamics

    for panel, relative_bus in [
        ("panel_2", "bus_1"),
        ("panel_3", "bus_3"),
    ]:
        eta_list = dynamics.flexible_bodies[panel]["eta_list"]
        phi_list = dynamics.flexible_bodies[panel]["phi_list"]
        expected_tip_deflection = sum(
            phi_k.subs(dynamics.s, dynamics.L) * eta_k
            for phi_k, eta_k in zip(phi_list, eta_list)
        )
        relative_angle = dynamics.bus_angle_coordinates[relative_bus]
        coordinates = dynamics.boundary_coordinates[panel]

        assert coordinates.shape == (4, 1)
        assert_symbolic_equal(coordinates[0], 0)
        assert_symbolic_equal(coordinates[1], -relative_angle / 2)
        assert_symbolic_equal(coordinates[2], expected_tip_deflection)
        assert_symbolic_equal(coordinates[3], relative_angle / 2)
        assert list(dynamics.element_coordinates[panel]) == (
            list(coordinates) + eta_list
        )


def test_multiangle_places_non_central_bus_centres_outward_from_panel_tip(
    seven_part_dynamics,
):
    dynamics = seven_part_dynamics

    for bus, joint_name in [
        ("bus_1", "joint_bus_1_panel_2"),
        ("bus_3", "joint_bus_3_panel_3"),
    ]:
        bus_offset = dynamics.points[bus].pos_from(dynamics.points[joint_name])
        expected = dynamics.D / 2 * dynamics.frames[bus].x

        assert_vector_equal(
            bus_offset,
            expected,
            dynamics.frames["inertial"],
        )


def test_multiangle_places_central_bus_at_reference_translation(seven_part_dynamics):
    dynamics = seven_part_dynamics
    N = dynamics.frames["inertial"]

    expected = dynamics.q_translation["x"] * N.x + dynamics.q_translation["y"] * N.y

    assert dynamics.inertial_position["bus_2"] == expected


def test_multiangle_outer_panel_material_point_uses_modal_displacement(
    seven_part_dynamics,
):
    dynamics = seven_part_dynamics

    panel = "panel_1"
    joint = dynamics.points["joint_panel_1_bus_1"]
    dm_point = dynamics.points[f"dm_{panel}"]
    eta_list = dynamics.flexible_bodies[panel]["eta_list"]
    phi_list = dynamics.flexible_bodies[panel]["phi_list"]
    expected_displacement = sum(
        phi_k * eta_k
        for phi_k, eta_k in zip(phi_list, eta_list)
    )
    expected = (
        dynamics.s * dynamics.frames[panel].x
        + expected_displacement * dynamics.frames[panel].y
    )

    assert_vector_equal(dm_point.pos_from(joint), expected, dynamics.frames[panel])


def test_multiangle_inter_bus_material_point_uses_boundary_compatible_displacement(
    seven_part_dynamics,
):
    dynamics = seven_part_dynamics

    for panel in ("panel_2", "panel_3"):
        joint = dynamics.points[dynamics.boundary_points[panel]["root_joint"]]
        dm_point = dynamics.points[f"dm_{panel}"]
        eta_list = dynamics.flexible_bodies[panel]["eta_list"]
        beam = BoundaryCompatibleBeam(
            length=dynamics.parameter_values["L"],
            E=dynamics.parameter_values["E_mod"],
            I=dynamics.parameter_values["I_area"],
            n=len(eta_list),
        )
        boundary_shapes = beam.boundary_shape_functions_symbolic(dynamics.s)
        internal_shapes = beam.internal_mode_shapes_symbolic(dynamics.s)
        expected_displacement = sum(
            shape * coordinate
            for shape, coordinate in zip(
                boundary_shapes,
                dynamics.boundary_coordinates[panel],
            )
        ) + sum(
            shape * eta_k
            for shape, eta_k in zip(internal_shapes, eta_list)
        )
        expected = (
            dynamics.s * dynamics.frames[panel].x
            + expected_displacement * dynamics.frames[panel].y
        )

        assert_vector_equal(dm_point.pos_from(joint), expected, dynamics.frames[panel])


def test_multiangle_inter_bus_boundary_compatible_displacement_reduces_to_internal_modes(
    seven_part_dynamics,
):
    dynamics = seven_part_dynamics

    panel = "panel_2"
    eta_list = dynamics.flexible_bodies[panel]["eta_list"]
    beam = BoundaryCompatibleBeam(
        length=dynamics.parameter_values["L"],
        E=dynamics.parameter_values["E_mod"],
        I=dynamics.parameter_values["I_area"],
        n=len(eta_list),
    )
    expected = sum(
        shape * eta_k
        for shape, eta_k in zip(
            beam.internal_mode_shapes_symbolic(dynamics.s),
            eta_list,
        )
    )
    zero_boundary_coordinates = {
        coordinate: sm.S.Zero
        for coordinate in dynamics.boundary_coordinates[panel]
    }
    actual = dynamics._inter_bus_panel_boundary_compatible_displacement(panel).subs(
        zero_boundary_coordinates
    )

    assert_symbolic_equal(actual, expected)


def test_multiangle_places_outer_panel_mass_center_from_modal_mean(
    seven_part_dynamics,
):
    dynamics = seven_part_dynamics

    panel = "panel_1"
    joint = dynamics.points["joint_panel_1_bus_1"]
    panel_cm = dynamics.points["dm_center_of_mass_panel_1"]
    eta_list = dynamics.flexible_bodies[panel]["eta_list"]
    phi_mean_list = dynamics.flexible_bodies[panel]["phi_mean_list"]
    phi_mean_sum = sum(
        sm.Float(phi_mean_k) * eta_k
        for phi_mean_k, eta_k in zip(phi_mean_list, eta_list)
    )
    expected = (
        dynamics.L / 2 * dynamics.frames[panel].x
        + phi_mean_sum * dynamics.frames[panel].y
    )

    assert panel_cm.pos_from(joint) == expected


def test_multiangle_places_inter_bus_panel_mass_center_from_boundary_compatible_mean(
    seven_part_dynamics,
):
    dynamics = seven_part_dynamics

    panel = "panel_3"
    joint = dynamics.points["joint_panel_3_bus_2"]
    panel_cm = dynamics.points["dm_center_of_mass_panel_3"]
    expected = (
        dynamics.L / 2 * dynamics.frames[panel].x
        + dynamics._flexible_center_of_mass_displacement_sum(panel)
        * dynamics.frames[panel].y
    )

    assert panel_cm.pos_from(joint) == expected


def test_multiangle_places_rigid_bus_mass_center_from_bus_joint(seven_part_dynamics):
    dynamics = seven_part_dynamics

    bus = "bus_3"
    joint = dynamics.points["joint_bus_3_panel_3"]
    bus_cm = dynamics.points[bus]
    expected = dynamics.D / 2 * dynamics.frames[bus].x

    assert bus_cm.pos_from(joint) == expected


def test_multiangle_defines_system_center_of_mass_from_body_mass_centers(seven_part_dynamics):
    dynamics = seven_part_dynamics

    expected_total_mass = sum(
        dynamics.mass_symbols[body]
        for body in dynamics.body_names
    )
    expected_position = sum(
        dynamics.mass_symbols[body] * dynamics.inertial_position[body]
        for body in dynamics.body_names
    ) / expected_total_mass

    assert dynamics.total_mass == expected_total_mass
    assert dynamics.points["center_of_mass"] is dynamics.G
    assert dynamics.G.pos_from(dynamics.O) == expected_position


def test_multiangle_defines_central_bus_position_relative_to_system_center_of_mass(seven_part_dynamics):
    dynamics = seven_part_dynamics

    expected = dynamics.points[dynamics.central_body].pos_from(dynamics.G)

    assert dynamics.r_GB == expected


def test_multiangle_defines_kinematic_differential_equations(seven_part_dynamics):
    dynamics = seven_part_dynamics

    assert dynamics.fk == dynamics.qd - dynamics.u
    assert dynamics.Mk == sm.eye(len(dynamics.q))
    assert dynamics.gk == -dynamics.u


def test_multiangle_defines_coordinate_derivative_replacements(seven_part_dynamics):
    dynamics = seven_part_dynamics

    assert dynamics.qd_repl == dict(zip(dynamics.qd, dynamics.u))
    assert dynamics.qdd_repl == {
        q.diff(dynamics.t): u.diff(dynamics.t)
        for q, u in zip(dynamics.qd, dynamics.u)
    }


def test_multiangle_defines_bus_angular_velocities(seven_part_dynamics):
    dynamics = seven_part_dynamics
    N = dynamics.frames["inertial"]

    u31 = dynamics.bus_speed_coordinates["bus_1"]
    u32 = dynamics.bus_speed_coordinates["bus_2"]
    u33 = dynamics.bus_speed_coordinates["bus_3"]

    assert_vector_equal(dynamics.angular_velocities["bus_1"], (u32 + u31) * N.z, N)
    assert_vector_equal(dynamics.angular_velocities["bus_2"], u32 * N.z, N)
    assert_vector_equal(dynamics.angular_velocities["bus_3"], (u32 + u33) * N.z, N)


def test_multiangle_defines_panel_angular_velocities_from_orientation_convention(seven_part_dynamics):
    dynamics = seven_part_dynamics
    N = dynamics.frames["inertial"]

    u31 = dynamics.bus_speed_coordinates["bus_1"]
    u32 = dynamics.bus_speed_coordinates["bus_2"]
    u33 = dynamics.bus_speed_coordinates["bus_3"]

    assert_vector_equal(dynamics.angular_velocities["panel_1"], (u32 + u31) * N.z, N)
    assert_vector_equal(
        dynamics.angular_velocities["panel_2"],
        (u32 + u31 / 2) * N.z,
        N,
    )
    assert_vector_equal(
        dynamics.angular_velocities["panel_3"],
        (u32 + u33 / 2) * N.z,
        N,
    )
    assert_vector_equal(dynamics.angular_velocities["panel_4"], (u32 + u33) * N.z, N)


def test_multiangle_defines_central_bus_linear_velocity(seven_part_dynamics):
    dynamics = seven_part_dynamics
    N = dynamics.frames["inertial"]

    expected = (
        dynamics.u_translation["x"] * N.x
        + dynamics.u_translation["y"] * N.y
    )

    assert_vector_equal(dynamics.points[dynamics.central_body].vel(N), expected, N)
    assert_vector_equal(dynamics.linear_velocities[dynamics.central_body], expected, N)


def test_multiangle_defines_rigid_parent_joint_velocity_with_two_point_theory(seven_part_dynamics):
    dynamics = seven_part_dynamics
    N = dynamics.frames["inertial"]

    joint = dynamics.points["joint_panel_2_bus_2"]
    parent_point = dynamics.points["bus_2"]
    parent_frame = dynamics.frames["bus_2"]
    relative_position = joint.pos_from(parent_point)
    expected = (
        parent_point.vel(N)
        + parent_frame.ang_vel_in(N).cross(relative_position)
    )

    assert "joint_panel_2_bus_2" in dynamics.joint_velocities
    assert_vector_equal(joint.vel(N), expected, N)
    assert_vector_equal(dynamics.joint_velocities["joint_panel_2_bus_2"], expected, N)


def test_multiangle_defines_flexible_parent_distal_joint_tip_velocity(seven_part_dynamics):
    dynamics = seven_part_dynamics

    panel = "panel_2"
    frame = dynamics.frames[panel]
    joint = dynamics.points["joint_bus_1_panel_2"]
    expected_relative_velocity = dynamics._flexible_tip_velocity_sum(panel) * frame.y

    assert_vector_equal(joint.vel(frame), expected_relative_velocity, frame)


def test_multiangle_defines_flexible_parent_distal_joint_inertial_velocity(seven_part_dynamics):
    dynamics = seven_part_dynamics
    N = dynamics.frames["inertial"]

    panel = "panel_2"
    frame = dynamics.frames[panel]
    parent_root = dynamics.points["joint_panel_2_bus_2"]
    joint = dynamics.points["joint_bus_1_panel_2"]
    expected = (
        parent_root.vel(N)
        + dynamics._flexible_tip_velocity_sum(panel) * frame.y
        + frame.ang_vel_in(N).cross(joint.pos_from(parent_root))
    )

    assert "joint_bus_1_panel_2" in dynamics.joint_velocities
    assert_vector_equal(joint.vel(N), expected, N)
    assert_vector_equal(dynamics.joint_velocities["joint_bus_1_panel_2"], expected, N)


def test_multiangle_defines_rigid_child_bus_velocity_from_distal_joint(seven_part_dynamics):
    dynamics = seven_part_dynamics
    N = dynamics.frames["inertial"]

    bus = "bus_1"
    joint = dynamics.points["joint_bus_1_panel_2"]
    bus_point = dynamics.points[bus]
    bus_frame = dynamics.frames[bus]
    expected = (
        joint.vel(N)
        + bus_frame.ang_vel_in(N).cross(bus_point.pos_from(joint))
    )

    assert_vector_equal(dynamics.linear_velocities[bus], expected, N)


def test_multiangle_defines_flexible_body_distributed_and_center_of_mass_velocities(seven_part_dynamics):
    dynamics = seven_part_dynamics

    panel = "panel_3"
    frame = dynamics.frames[panel]
    dm_point = dynamics.points[f"dm_{panel}"]
    cm_point = dynamics.points[f"dm_center_of_mass_{panel}"]

    expected_dm_relative_velocity = (
        dynamics._flexible_distributed_velocity_sum(panel) * frame.y
    )
    expected_cm_relative_velocity = (
        dynamics._flexible_center_of_mass_velocity_sum(panel) * frame.y
    )

    assert_vector_equal(dm_point.vel(frame), expected_dm_relative_velocity, frame)
    assert_vector_equal(cm_point.vel(frame), expected_cm_relative_velocity, frame)
    assert panel in dynamics.linear_velocities
    assert panel in dynamics.flexible_center_of_mass_velocities


def test_multiangle_outer_panel_distributed_velocity_remains_modal_only(
    seven_part_dynamics,
):
    dynamics = seven_part_dynamics

    panel = "panel_1"
    expected = sum(
        phi_k * zeta_k
        for phi_k, zeta_k in zip(
            dynamics.flexible_bodies[panel]["phi_list"],
            dynamics.flexible_bodies[panel]["zeta_list"],
        )
    )

    assert_symbolic_equal(dynamics._flexible_distributed_velocity_sum(panel), expected)


def test_multiangle_inter_bus_distributed_velocity_differentiates_boundary_compatible_displacement(
    seven_part_dynamics,
):
    dynamics = seven_part_dynamics

    panel = "panel_3"
    expected = (
        dynamics._inter_bus_panel_boundary_compatible_displacement(panel)
        .diff(dynamics.t)
        .xreplace(dynamics.qd_repl)
    )

    assert_symbolic_equal(dynamics._flexible_distributed_velocity_sum(panel), expected)


def test_multiangle_defines_bus_angular_accelerations(seven_part_dynamics):
    dynamics = seven_part_dynamics
    N = dynamics.frames["inertial"]
    t = dynamics.t

    u31 = dynamics.bus_speed_coordinates["bus_1"]
    u32 = dynamics.bus_speed_coordinates["bus_2"]
    u33 = dynamics.bus_speed_coordinates["bus_3"]

    assert_vector_equal(
        dynamics.angular_accelerations["bus_1"],
        (u32.diff(t) + u31.diff(t)) * N.z,
        N,
    )
    assert_vector_equal(
        dynamics.angular_accelerations["bus_2"],
        u32.diff(t) * N.z,
        N,
    )
    assert_vector_equal(
        dynamics.angular_accelerations["bus_3"],
        (u32.diff(t) + u33.diff(t)) * N.z,
        N,
    )


def test_multiangle_defines_central_bus_linear_acceleration(seven_part_dynamics):
    dynamics = seven_part_dynamics
    N = dynamics.frames["inertial"]
    t = dynamics.t

    expected = (
        dynamics.u_translation["x"].diff(t) * N.x
        + dynamics.u_translation["y"].diff(t) * N.y
    )

    assert_vector_equal(
        dynamics.points[dynamics.central_body].acc(N),
        expected,
        N,
    )
    assert_vector_equal(
        dynamics.linear_accelerations[dynamics.central_body],
        expected,
        N,
    )


def test_multiangle_defines_rigid_parent_joint_acceleration_with_two_point_theory(seven_part_dynamics):
    dynamics = seven_part_dynamics
    N = dynamics.frames["inertial"]

    joint = dynamics.points["joint_panel_2_bus_2"]
    parent_point = dynamics.points["bus_2"]
    parent_frame = dynamics.frames["bus_2"]
    relative_position = joint.pos_from(parent_point)
    omega = parent_frame.ang_vel_in(N)
    alpha = parent_frame.ang_acc_in(N)
    expected = (
        parent_point.acc(N)
        + alpha.cross(relative_position)
        + omega.cross(omega.cross(relative_position))
    ).xreplace(dynamics.qd_repl)
    actual = joint.acc(N).xreplace(dynamics.qd_repl)

    assert "joint_panel_2_bus_2" in dynamics.joint_accelerations
    assert_vector_equal(actual, expected, N)
    assert_vector_equal(
        dynamics.joint_accelerations["joint_panel_2_bus_2"],
        expected,
        N,
    )


def test_multiangle_defines_flexible_parent_distal_joint_acceleration(seven_part_dynamics):
    dynamics = seven_part_dynamics
    N = dynamics.frames["inertial"]

    panel = "panel_2"
    frame = dynamics.frames[panel]
    parent_root = dynamics.points["joint_panel_2_bus_2"]
    joint = dynamics.points["joint_bus_1_panel_2"]
    relative_position = joint.pos_from(parent_root)
    relative_velocity = dynamics._flexible_tip_velocity_sum(panel) * frame.y
    omega = frame.ang_vel_in(N)
    alpha = frame.ang_acc_in(N)
    expected = (
        parent_root.acc(N)
        + relative_velocity.dt(frame)
        + 2 * omega.cross(relative_velocity)
        + alpha.cross(relative_position)
        + omega.cross(omega.cross(relative_position))
    ).xreplace(dynamics.qd_repl)
    actual = joint.acc(N).xreplace(dynamics.qd_repl)

    assert "joint_bus_1_panel_2" in dynamics.joint_accelerations
    assert_vector_equal(actual, expected, N)
    assert_vector_equal(
        dynamics.joint_accelerations["joint_bus_1_panel_2"],
        expected,
        N,
    )


def test_multiangle_accelerations_are_stored_in_speed_variables(seven_part_dynamics):
    dynamics = seven_part_dynamics

    assert_vectors_do_not_contain_coordinate_derivatives(
        dynamics.angular_accelerations,
        dynamics,
    )
    assert_vectors_do_not_contain_coordinate_derivatives(
        dynamics.linear_accelerations,
        dynamics,
    )
    assert_vectors_do_not_contain_coordinate_derivatives(
        dynamics.joint_accelerations,
        dynamics,
    )
    assert_vectors_do_not_contain_coordinate_derivatives(
        dynamics.flexible_center_of_mass_accelerations,
        dynamics,
    )


def test_multiangle_external_forces_are_zero_for_all_bodies(seven_part_dynamics):
    dynamics = seven_part_dynamics

    assert set(dynamics.external_forces) == set(dynamics.body_names)
    for body in dynamics.body_names:
        assert_vector_equal(
            dynamics.external_forces[body],
            0 * dynamics.frames[body].x,
            dynamics.frames[body],
        )


def test_multiangle_external_torques_are_symbolic_on_rigid_buses_only(seven_part_dynamics):
    dynamics = seven_part_dynamics

    assert set(dynamics.external_torques) == set(dynamics.body_names)

    for bus in dynamics.rigid_body_names:
        expected = dynamics.bus_torque_symbols[bus] * dynamics.frames[bus].z
        assert_vector_equal(
            dynamics.external_torques[bus],
            expected,
            dynamics.frames[bus],
        )

    for panel in dynamics.flexible_body_names:
        assert_vector_equal(
            dynamics.external_torques[panel],
            0 * dynamics.frames[panel].z,
            dynamics.frames[panel],
        )


def test_multiangle_defines_partial_velocity_vectors_for_all_bodies(seven_part_dynamics):
    dynamics = seven_part_dynamics

    assert set(dynamics.partial_angular_velocities) == set(dynamics.body_names)
    assert set(dynamics.partial_linear_velocities) == set(dynamics.body_names)

    for body in dynamics.body_names:
        assert len(dynamics.partial_angular_velocities[body]) == len(dynamics.u)
        assert len(dynamics.partial_linear_velocities[body]) == len(dynamics.u)


def test_multiangle_partial_linear_velocities_reconstruct_body_velocities(seven_part_dynamics):
    dynamics = seven_part_dynamics
    N = dynamics.frames["inertial"]

    for body, velocity in dynamics.linear_velocities.items():
        reconstructed = sum(
            partial_velocity * speed
            for partial_velocity, speed in zip(
                dynamics.partial_linear_velocities[body],
                dynamics.u,
            )
        )

        assert_vector_equal(reconstructed, velocity, N)


def test_multiangle_central_bus_linear_partials_are_translation_basis_vectors(seven_part_dynamics):
    dynamics = seven_part_dynamics
    N = dynamics.frames["inertial"]

    u1 = dynamics.u_translation["x"]
    u2 = dynamics.u_translation["y"]
    i_u1 = list(dynamics.u).index(u1)
    i_u2 = list(dynamics.u).index(u2)
    partials = dynamics.partial_linear_velocities[dynamics.central_body]

    assert_vector_equal(partials[i_u1], N.x, N)
    assert_vector_equal(partials[i_u2], N.y, N)

    for index, speed in enumerate(dynamics.u):
        if speed not in {u1, u2}:
            assert_vector_equal(partials[index], 0 * N.x, N)


def test_multiangle_rigid_parent_panel_linear_partials_include_root_translation(seven_part_dynamics):
    dynamics = seven_part_dynamics
    N = dynamics.frames["inertial"]

    panel = "panel_3"
    u1 = dynamics.u_translation["x"]
    u2 = dynamics.u_translation["y"]
    i_u1 = list(dynamics.u).index(u1)
    i_u2 = list(dynamics.u).index(u2)
    partials = dynamics.partial_linear_velocities[panel]

    assert_vector_equal(partials[i_u1], N.x, N)
    assert_vector_equal(partials[i_u2], N.y, N)


def test_multiangle_rigid_parent_panel_linear_partials_include_modal_speeds(seven_part_dynamics):
    dynamics = seven_part_dynamics
    N = dynamics.frames["inertial"]

    panel = "panel_1"
    frame = dynamics.frames[panel]
    zeta_list = dynamics.flexible_bodies[panel]["zeta_list"]
    phi_list = dynamics.flexible_bodies[panel]["phi_list"]
    partials = dynamics.partial_linear_velocities[panel]

    for zeta_k, phi_k in zip(zeta_list, phi_list):
        index = list(dynamics.u).index(zeta_k)
        assert_vector_equal(partials[index], phi_k * frame.y, N)


def test_multiangle_inter_bus_panel_linear_partials_include_boundary_compatible_speeds(
    seven_part_dynamics,
):
    dynamics = seven_part_dynamics
    N = dynamics.frames["inertial"]

    panel = "panel_3"
    frame = dynamics.frames[panel]
    zeta_list = dynamics.flexible_bodies[panel]["zeta_list"]
    velocity_sum = dynamics._flexible_distributed_velocity_sum(panel)
    partials = dynamics.partial_linear_velocities[panel]

    for zeta_k in zeta_list:
        index = list(dynamics.u).index(zeta_k)
        expected = sm.diff(velocity_sum, zeta_k) * frame.y
        assert_vector_equal(partials[index], expected, N)


def test_multiangle_bus_angular_partials_distinguish_bus_torque_locations(seven_part_dynamics):
    dynamics = seven_part_dynamics
    N = dynamics.frames["inertial"]

    u31 = dynamics.bus_speed_coordinates["bus_1"]
    u32 = dynamics.bus_speed_coordinates["bus_2"]
    u33 = dynamics.bus_speed_coordinates["bus_3"]
    i31 = list(dynamics.u).index(u31)
    i32 = list(dynamics.u).index(u32)
    i33 = list(dynamics.u).index(u33)

    partials = dynamics.partial_angular_velocities

    assert_vector_equal(partials["bus_1"][i31], N.z, N)
    assert_vector_equal(partials["bus_2"][i31], 0 * N.z, N)
    assert_vector_equal(partials["bus_3"][i31], 0 * N.z, N)

    assert_vector_equal(partials["bus_1"][i32], N.z, N)
    assert_vector_equal(partials["bus_2"][i32], N.z, N)
    assert_vector_equal(partials["bus_3"][i32], N.z, N)

    assert_vector_equal(partials["bus_1"][i33], 0 * N.z, N)
    assert_vector_equal(partials["bus_2"][i33], 0 * N.z, N)
    assert_vector_equal(partials["bus_3"][i33], N.z, N)


def test_multiangle_panel_angular_partials_follow_level_1_5_orientation_model(seven_part_dynamics):
    dynamics = seven_part_dynamics
    N = dynamics.frames["inertial"]

    u31 = dynamics.bus_speed_coordinates["bus_1"]
    u32 = dynamics.bus_speed_coordinates["bus_2"]
    u33 = dynamics.bus_speed_coordinates["bus_3"]
    i31 = list(dynamics.u).index(u31)
    i32 = list(dynamics.u).index(u32)
    i33 = list(dynamics.u).index(u33)

    partials = dynamics.partial_angular_velocities

    assert_vector_equal(partials["panel_2"][i31], sm.Rational(1, 2) * N.z, N)
    assert_vector_equal(partials["panel_2"][i32], N.z, N)
    assert_vector_equal(partials["panel_2"][i33], 0 * N.z, N)

    assert_vector_equal(partials["panel_3"][i31], 0 * N.z, N)
    assert_vector_equal(partials["panel_3"][i32], N.z, N)
    assert_vector_equal(partials["panel_3"][i33], sm.Rational(1, 2) * N.z, N)


def test_multiangle_initialises_generalised_active_force_vector(seven_part_dynamics):
    dynamics = seven_part_dynamics

    assert dynamics.state_dimension == len(dynamics.u)
    assert dynamics.generalised_active_forces.shape == (len(dynamics.u), 1)


def test_multiangle_external_bus_torques_create_distinct_generalised_forces(seven_part_dynamics):
    dynamics = seven_part_dynamics

    tau1 = dynamics.bus_torque_symbols["bus_1"]
    tau2 = dynamics.bus_torque_symbols["bus_2"]
    tau3 = dynamics.bus_torque_symbols["bus_3"]

    u1 = dynamics.u_translation["x"]
    u2 = dynamics.u_translation["y"]
    u31 = dynamics.bus_speed_coordinates["bus_1"]
    u32 = dynamics.bus_speed_coordinates["bus_2"]
    u33 = dynamics.bus_speed_coordinates["bus_3"]

    expected = sm.zeros(len(dynamics.u), 1)
    expected[list(dynamics.u).index(u1)] = 0
    expected[list(dynamics.u).index(u2)] = 0
    expected[list(dynamics.u).index(u31)] = tau1
    expected[list(dynamics.u).index(u32)] = tau1 + tau2 + tau3
    expected[list(dynamics.u).index(u33)] = tau3

    for row, coordinate in enumerate(dynamics.q):
        expected[row] += -sm.diff(dynamics.V_strain, coordinate)

    for i in range(len(dynamics.u)):
        actual_without_gravity = (
            dynamics.generalised_active_forces[i]
            - dynamics.partial_v_G[i].dot(dynamics.F_gravity)
        )
        assert_symbolic_equal(actual_without_gravity, expected[i])


def test_multiangle_generalised_active_forces_match_implemented_force_blocks(seven_part_dynamics):
    dynamics = seven_part_dynamics

    expected = sm.zeros(len(dynamics.u), 1)

    for i in range(len(dynamics.u)):
        for body in dynamics.body_names:
            expected[i] += (
                dynamics.partial_linear_velocities[body][i].dot(
                    dynamics.external_forces[body]
                )
                + dynamics.partial_angular_velocities[body][i].dot(
                    dynamics.external_torques[body]
                )
            )

    for row, coordinate in enumerate(dynamics.q):
        expected[row] += -sm.diff(dynamics.V_strain, coordinate)

    for i in range(len(dynamics.u)):
        actual_without_gravity = (
            dynamics.generalised_active_forces[i]
            - dynamics.partial_v_G[i].dot(dynamics.F_gravity)
        )
        assert_symbolic_equal(actual_without_gravity, expected[i])


def test_multiangle_defaults_flexible_inertia_integration_to_quadrature(seven_part_dynamics):
    dynamics = seven_part_dynamics

    assert dynamics.flexible_inertia_integration == {
        "method": "gauss-legendre",
        "quadrature_points": 8,
    }


def test_multiangle_accepts_canonical_flexible_inertia_integration_config():
    settings = (
        MultiAngleFlexibleDynamics._normalise_flexible_inertia_integration_config(
            {
                "method": "symbolic",
                "quadrature_points": "4",
            }
        )
    )

    assert settings == {
        "method": "symbolic",
        "quadrature_points": 4,
    }


def test_multiangle_rejects_non_canonical_flexible_inertia_integration_method():
    with pytest.raises(ValueError, match="flexible_inertia_integration.method"):
        MultiAngleFlexibleDynamics._normalise_flexible_inertia_integration_config(
            {"method": "sympy"}
        )


def test_multiangle_initialises_generalised_inertia_force_vector(seven_part_dynamics):
    dynamics = seven_part_dynamics

    assert dynamics.state_dimension == len(dynamics.u)
    assert dynamics.generalised_inertia_forces.shape == (len(dynamics.u), 1)
    assert set(dynamics.rigid_body_inertia_forces) == set(dynamics.rigid_body_names)
    assert set(dynamics.rigid_body_inertia_torques) == set(dynamics.rigid_body_names)
    assert set(dynamics.flexible_body_inertia_force_densities) == set(
        dynamics.flexible_body_names
    )
    assert set(dynamics.flexible_body_generalised_inertia_forces) == set(
        dynamics.flexible_body_names
    )


def test_multiangle_central_bus_rigid_body_inertia_loads_have_newton_euler_form(seven_part_dynamics):
    dynamics = seven_part_dynamics
    N = dynamics.frames["inertial"]
    body = dynamics.central_body

    m_bus = dynamics.mass_symbols[body]
    u1_dot = dynamics.u_translation["x"].diff(dynamics.t)
    u2_dot = dynamics.u_translation["y"].diff(dynamics.t)
    u3_dot = dynamics.bus_speed_coordinates[body].diff(dynamics.t)
    Izz = dynamics.inertia_matrices[body][2, 2]

    expected_force = -m_bus * (u1_dot * N.x + u2_dot * N.y)
    expected_torque = -Izz * u3_dot * N.z

    assert_vector_equal(dynamics.rigid_body_inertia_forces[body], expected_force, N)
    assert_vector_equal(dynamics.rigid_body_inertia_torques[body], expected_torque, N)


def test_multiangle_flexible_body_inertia_density_has_expected_form(seven_part_dynamics):
    dynamics = seven_part_dynamics
    N = dynamics.frames["inertial"]
    body = "panel_1"

    expected_density = (
        -dynamics.mass_symbols[body]
        / dynamics.L
        * dynamics.linear_accelerations[body]
    )

    assert_vector_equal(
        dynamics.flexible_body_inertia_force_densities[body],
        expected_density,
        N,
    )


def test_multiangle_flexible_body_inertia_quadrature_integrates_polynomial(seven_part_dynamics):
    dynamics = seven_part_dynamics

    result = dynamics._integrate_flexible_body_expression("panel_1", dynamics.s**2)

    assert abs(float(result) - 9.0) < 1e-12


def test_multiangle_flexible_body_inertia_symbolic_integration_uses_sympy(
    seven_part_dynamics,
    monkeypatch,
):
    dynamics = seven_part_dynamics
    monkeypatch.setitem(
        dynamics.flexible_inertia_integration,
        "method",
        "symbolic",
    )

    result = dynamics._integrate_flexible_body_expression("panel_1", dynamics.s**2)

    assert_symbolic_equal(result, dynamics.L**3 / 3)


def test_multiangle_flexible_body_inertia_is_integrated_density_projection(seven_part_dynamics):
    dynamics = seven_part_dynamics
    body = "panel_1"
    row = len(dynamics.u_reference) + dynamics.flex_zeta_index[(body, 0)]

    v_partial = dynamics.partial_linear_velocities[body][row]
    force_density = dynamics.flexible_body_inertia_force_densities[body]
    expected_integrand = v_partial.dot(force_density)
    expected_contribution = dynamics._integrate_flexible_body_expression(
        body,
        expected_integrand,
    )

    assert_symbolic_equal(
        dynamics.flexible_body_generalised_inertia_forces[body][row],
        expected_contribution,
    )


def test_multiangle_generalised_inertia_forces_match_body_projections(seven_part_dynamics):
    dynamics = seven_part_dynamics

    expected = sm.zeros(len(dynamics.u), 1)
    for i in range(len(dynamics.u)):
        for body in dynamics.rigid_body_names:
            expected[i] += (
                dynamics.partial_linear_velocities[body][i].dot(
                    dynamics.rigid_body_inertia_forces[body]
                )
                + dynamics.partial_angular_velocities[body][i].dot(
                    dynamics.rigid_body_inertia_torques[body]
                )
            )

    for body in dynamics.flexible_body_names:
        expected += dynamics.flexible_body_generalised_inertia_forces[body]

    for i in range(len(dynamics.u)):
        assert_symbolic_equal(dynamics.generalised_inertia_forces[i], expected[i])


def test_multiangle_derives_equations_of_motion_from_kane_forces(seven_part_dynamics):
    dynamics = seven_part_dynamics

    assert dynamics.kane_eq == (
        dynamics.generalised_active_forces + dynamics.generalised_inertia_forces
    )
    assert dynamics.mass_matrix == -dynamics.kane_eq.jacobian(dynamics.ud)
    assert dynamics.forcing == dynamics.kane_eq.xreplace(dynamics.ud_zero)

    assert dynamics.kane_eq.shape == (len(dynamics.u), 1)
    assert dynamics.mass_matrix.shape == (len(dynamics.u), len(dynamics.u))
    assert dynamics.forcing.shape == (len(dynamics.u), 1)


def test_multiangle_creates_lambdified_equation_evaluators_at_initialisation(seven_part_dynamics):
    dynamics = seven_part_dynamics

    q_values = np.zeros(len(dynamics.q))
    u_values = np.zeros(len(dynamics.u))
    torque_values = dynamics.get_torque_values()

    Mk, gk = dynamics.eval_kinematics(
        q_values,
        u_values,
        torque_values,
    )
    mass_matrix, forcing = dynamics.eval_differentials(
        q_values,
        u_values,
        torque_values,
    )
    r_G = dynamics.rG_func(q_values, u_values)
    v_G = dynamics.vG_func(q_values, u_values)

    assert np.asarray(Mk).shape == (len(dynamics.q), len(dynamics.q))
    assert np.asarray(gk).shape == (len(dynamics.q), 1)
    assert np.asarray(mass_matrix).shape == (len(dynamics.u), len(dynamics.u))
    assert np.asarray(forcing).shape == (len(dynamics.u), 1)
    assert np.asarray(r_G).shape == (3, 1)
    assert np.asarray(v_G).shape == (3, 1)


def test_multiangle_initial_conditions_make_centre_of_mass_keplerian(seven_part_config):
    config = seven_part_config
    config["q_initial"] = {
        "q_relative_angle_bus_1": np.deg2rad(-1.5),
        "q_central_angle": np.deg2rad(2.0),
        "q_relative_angle_bus_3": np.deg2rad(1.0),
        "eta1_1": 1.0e-3,
        "eta2_1": -2.0e-4,
        "eta2_2": 3.0e-4,
        "eta3_1": 4.0e-4,
        "eta3_2": -1.0e-4,
        "eta4_1": -7.5e-4,
    }
    config["initial_speeds"] = {
        "u_relative_angle_bus_1": -1.5e-4,
        "u_central_angle": 1.131e-3,
        "u_relative_angle_bus_3": 2.0e-4,
        "zeta1_1": 1.0e-5,
        "zeta2_1": -2.0e-5,
        "zeta2_2": 1.5e-5,
        "zeta3_1": -1.2e-5,
        "zeta3_2": 2.5e-5,
        "zeta4_1": 7.0e-6,
    }
    dynamics = MultiAngleFlexibleDynamics(config)

    x0 = dynamics.get_initial_conditions(verbose=False)
    q0 = x0[:dynamics.state_dimension]
    u0 = x0[dynamics.state_dimension:]

    r_G = np.asarray(dynamics.rG_func(q0, u0), dtype=float).reshape(3)
    v_G = np.asarray(dynamics.vG_func(q0, u0), dtype=float).reshape(3)

    mu = config["parameters"]["planet_mu"]
    a = config["parameters"]["orbit_semi_major_axis"]
    e = config["parameters"]["orbit_eccentricity"]
    initial_true_anomaly = 0.0
    r0 = a * (1.0 - e**2) / (1.0 + e * np.cos(initial_true_anomaly))
    h = np.sqrt(mu * a * (1.0 - e**2))
    v_G_expected = np.array(
        [
            -(mu / h) * e * np.sin(initial_true_anomaly),
            (mu / h) * (1.0 + e * np.cos(initial_true_anomaly)),
            0.0,
        ]
    )

    assert x0.shape == (2 * dynamics.state_dimension,)
    np.testing.assert_allclose(r_G, np.array([r0, 0.0, 0.0]), atol=1e-8)
    np.testing.assert_allclose(v_G, v_G_expected, rtol=1e-12, atol=1e-9)


def test_multiangle_initial_conditions_match_explicit_centre_offset_formula(
    seven_part_config,
):
    config = seven_part_config
    theta0 = np.deg2rad(12.0)
    central_speed = 2.5e-4
    config["q_initial"] = {
        "q_relative_angle_bus_1": 0.0,
        "q_central_angle": theta0,
        "q_relative_angle_bus_3": 0.0,
        "eta1_1": 7.5e-4,
        "eta2_1": -2.0e-4,
        "eta2_2": 3.0e-4,
        "eta3_1": 4.0e-4,
        "eta3_2": -1.0e-4,
        "eta4_1": -6.0e-4,
    }
    config["initial_speeds"] = {
        "u_relative_angle_bus_1": 0.0,
        "u_central_angle": central_speed,
        "u_relative_angle_bus_3": 0.0,
        "zeta1_1": 1.0e-5,
        "zeta2_1": -2.0e-5,
        "zeta2_2": 1.5e-5,
        "zeta3_1": -1.2e-5,
        "zeta3_2": 2.5e-5,
        "zeta4_1": 7.0e-6,
    }
    dynamics = MultiAngleFlexibleDynamics(config)

    x0 = dynamics.get_initial_conditions(verbose=False)
    q0 = x0[:dynamics.state_dimension]
    u0 = x0[dynamics.state_dimension:]

    central_frame = dynamics.frames[dynamics.central_body]
    rho = dynamics.r_GB.express(central_frame)
    rho_vector = sm.Matrix(
        [
            rho.dot(central_frame.x),
            rho.dot(central_frame.y),
        ]
    )
    rho_dot = rho.dt(central_frame).xreplace(dynamics.qd_repl)
    rho_dot_vector = sm.Matrix(
        [
            rho_dot.dot(central_frame.x),
            rho_dot.dot(central_frame.y),
        ]
    )
    rho_func = sm.lambdify(
        (
            dynamics.q,
            dynamics.u,
            list(dynamics.parameter_symbols.values()),
        ),
        (rho_vector, rho_dot_vector),
        "numpy",
    )

    rho0, rho_dot0 = rho_func(q0, u0, dynamics.get_parameter_values())
    rho0 = np.asarray(rho0, dtype=float).reshape(2)
    rho_dot0 = np.asarray(rho_dot0, dtype=float).reshape(2)

    mu = config["parameters"]["planet_mu"]
    a = config["parameters"]["orbit_semi_major_axis"]
    e = config["parameters"]["orbit_eccentricity"]
    r0 = a * (1.0 - e**2) / (1.0 + e)
    h = np.sqrt(mu * a * (1.0 - e**2))
    r_G0 = np.array([r0, 0.0])
    v_G0 = np.array([0.0, (mu / h) * (1.0 + e)])
    rotation = np.array(
        [
            [np.cos(theta0), -np.sin(theta0)],
            [np.sin(theta0), np.cos(theta0)],
        ]
    )
    skew = np.array(
        [
            [0.0, -1.0],
            [1.0, 0.0],
        ]
    )

    expected_q_translation = r_G0 + rotation @ rho0
    expected_u_translation = (
        v_G0
        + central_speed * skew @ rotation @ rho0
        + rotation @ rho_dot0
    )

    np.testing.assert_allclose(q0[:2], expected_q_translation, rtol=1e-12, atol=1e-8)
    np.testing.assert_allclose(u0[:2], expected_u_translation, rtol=1e-12, atol=1e-9)


def test_multiangle_numeric_value_helpers_follow_symbol_order(
    seven_part_dynamics,
    monkeypatch,
):
    dynamics = seven_part_dynamics
    monkeypatch.setitem(
        dynamics.config,
        "torques",
        {
            "bus_1": 1.2,
            "bus_3": -0.4,
        },
    )

    assert dynamics.get_parameter_values() == [
        dynamics.parameter_values[name]
        for name in dynamics.parameter_symbols
    ]
    assert dynamics.get_torque_values() == [1.2, 0.0, -0.4]


def test_multiangle_equation_forcing_has_no_speed_derivatives(seven_part_dynamics):
    dynamics = seven_part_dynamics
    forcing_derivatives = dynamics.forcing.atoms(sm.Derivative)

    assert not set(dynamics.ud) & forcing_derivatives


def test_multiangle_translation_mass_matrix_contains_total_mass(seven_part_dynamics):
    dynamics = seven_part_dynamics
    length_subs = {dynamics.L: dynamics.parameter_values["L"]}

    assert_symbolic_equal(
        dynamics.mass_matrix[0, 0].subs(length_subs),
        dynamics.total_mass.subs(length_subs),
    )
    assert_symbolic_equal(
        dynamics.mass_matrix[1, 1].subs(length_subs),
        dynamics.total_mass.subs(length_subs),
    )


def test_multiangle_kepler_gravity_quantities_are_stored(seven_part_dynamics):
    dynamics = seven_part_dynamics
    N = dynamics.frames["inertial"]

    expected_r_G = dynamics.points["center_of_mass"].pos_from(dynamics.O).express(N)
    expected_r_norm = sm.sqrt(expected_r_G.dot(expected_r_G))
    expected_force = (
        -dynamics.planet_mu
        * dynamics.total_mass
        * expected_r_G
        / expected_r_norm**3
    )
    expected_v_G = expected_r_G.dt(N).xreplace(dynamics.qd_repl)

    assert dynamics.r_G == expected_r_G
    assert dynamics.r_G_squared == expected_r_G.dot(expected_r_G)
    assert dynamics.r_G_norm == expected_r_norm
    assert dynamics.F_gravity == expected_force
    assert dynamics.v_G == expected_v_G
    assert len(dynamics.partial_v_G) == len(dynamics.u)


def test_multiangle_gravity_gradient_is_disabled_by_default(seven_part_dynamics):
    dynamics = seven_part_dynamics

    assert dynamics.enable_gravity_gradient is False
    assert dynamics.V_gg == 0
    assert dynamics.V_gg_trace == 0
    assert dynamics.V_gg_directional == 0
    assert dynamics.gravity_gradient_trace_inertia == 0
    assert dynamics.gravity_gradient_directional_inertia == 0
    assert dynamics.e3_hat_inertial == -dynamics.r_G / dynamics.r_G_norm
    assert dynamics.e3_hat_body == {}
    assert dynamics.body_centre_offsets == {}


def test_multiangle_gravity_gradient_potential_energy_is_stored_when_enabled(
    gravity_gradient_dynamics,
):
    dynamics = gravity_gradient_dynamics

    expected_directional_inertia = sm.S.Zero
    expected_trace_inertia = sm.S.Zero
    for body in dynamics.body_names:
        expected_directional_inertia += dynamics._body_gravity_gradient_energy(body)
        expected_trace_inertia += dynamics._body_gravity_gradient_trace_inertia(body)

    expected_trace = (
        -sm.Rational(1, 2)
        * dynamics.planet_mu
        * expected_trace_inertia
        / dynamics.r_G_norm**3
    )
    expected_directional = (
        sm.Rational(3, 2)
        * dynamics.planet_mu
        * expected_directional_inertia
        / dynamics.r_G_norm**3
    )
    expected = expected_trace + expected_directional

    assert dynamics.enable_gravity_gradient is True
    assert dynamics.V_gg == expected
    assert dynamics.V_gg_trace == expected_trace
    assert dynamics.V_gg_directional == expected_directional
    assert dynamics.gravity_gradient_trace_inertia == expected_trace_inertia
    assert dynamics.gravity_gradient_directional_inertia == expected_directional_inertia
    assert set(dynamics.e3_hat_body) == set(dynamics.body_names)
    assert set(dynamics.body_centre_offsets) == set(dynamics.body_names)
    assert dynamics.V_gg.has(dynamics.planet_mu)


def test_multiangle_gravity_gradient_directional_inertia_includes_parallel_axis(
    gravity_gradient_dynamics,
):
    dynamics = gravity_gradient_dynamics
    body = "panel_1"
    frame = dynamics.frames[body]

    local_vertical_body = dynamics.e3_hat_inertial.express(frame)
    e_body_components = local_vertical_body.to_matrix(frame)
    local_inertia = dynamics.inertia_matrices[body]
    body_offset = dynamics._body_centre_offset_from_system_centre(body).express(frame)
    expected = (
        (e_body_components.T * local_inertia * e_body_components)[0]
        + dynamics.mass_symbols[body]
        * (
            body_offset.dot(body_offset)
            - local_vertical_body.dot(body_offset) ** 2
        )
    )

    assert_symbolic_equal(dynamics._body_gravity_gradient_energy(body), expected)
    assert dynamics._body_gravity_gradient_energy(body).has(
        *dynamics.flexible_bodies[body]["eta_list"]
    )


def test_multiangle_gravity_gradient_trace_inertia_includes_parallel_axis(
    gravity_gradient_dynamics,
):
    dynamics = gravity_gradient_dynamics
    body = "panel_1"

    body_offset = dynamics._body_centre_offset_from_system_centre(body)
    expected = (
        sm.trace(dynamics.inertia_matrices[body])
        + 2 * dynamics.mass_symbols[body] * body_offset.dot(body_offset)
    )

    assert_symbolic_equal(
        dynamics._body_gravity_gradient_trace_inertia(body),
        expected,
    )
    assert dynamics._body_gravity_gradient_trace_inertia(body).has(
        *dynamics.flexible_bodies[body]["eta_list"]
    )


def test_multiangle_gravity_gradient_adds_attitude_row_forces_when_enabled(
    gravity_gradient_dynamics,
):
    dynamics = gravity_gradient_dynamics

    for body, angle_coordinate in dynamics.bus_angle_coordinates.items():
        row = list(dynamics.u).index(dynamics.bus_speed_coordinates[body])
        external_and_kepler = (
            dynamics.partial_v_G[row].dot(dynamics.F_gravity)
        )
        for loaded_body in dynamics.body_names:
            external_and_kepler += (
                dynamics.partial_linear_velocities[loaded_body][row].dot(
                    dynamics.external_forces[loaded_body]
                )
                + dynamics.partial_angular_velocities[loaded_body][row].dot(
                    dynamics.external_torques[loaded_body]
                )
            )

        strain_force = -sm.diff(dynamics.V_strain, angle_coordinate)
        actual_gravity_gradient = (
            dynamics.generalised_active_forces[row]
            - external_and_kepler
            - strain_force
        )
        expected_gravity_gradient = -sm.diff(dynamics.V_gg, angle_coordinate)

        assert actual_gravity_gradient == expected_gravity_gradient
        assert actual_gravity_gradient.has(dynamics.planet_mu)


def test_multiangle_gravity_gradient_adds_modal_row_forces_when_enabled(
    gravity_gradient_dynamics,
):
    dynamics = gravity_gradient_dynamics

    modal_offset = len(dynamics.u_reference)
    for body, values in dynamics.flexible_bodies.items():
        for mode, eta_k in enumerate(values["eta_list"]):
            row = modal_offset + dynamics.flex_eta_index[(body, mode)]
            strain_force = -sm.diff(dynamics.V_strain, eta_k)
            actual_gravity_gradient = (
                dynamics.generalised_active_forces[row]
                - dynamics.partial_v_G[row].dot(dynamics.F_gravity)
                - strain_force
            )
            expected_gravity_gradient = -sm.diff(dynamics.V_gg, eta_k)

            assert actual_gravity_gradient == expected_gravity_gradient


def test_multiangle_kepler_gravity_force_is_com_partial_velocity_projection(seven_part_dynamics):
    dynamics = seven_part_dynamics

    for i in range(len(dynamics.u)):
        external_and_strain = sm.S.Zero

        for body in dynamics.body_names:
            external_and_strain += (
                dynamics.partial_linear_velocities[body][i].dot(
                    dynamics.external_forces[body]
                )
                + dynamics.partial_angular_velocities[body][i].dot(
                    dynamics.external_torques[body]
                )
            )

        external_and_strain += -sm.diff(dynamics.V_strain, dynamics.q[i])

        actual_gravity = dynamics.generalised_active_forces[i] - external_and_strain
        expected_gravity = dynamics.partial_v_G[i].dot(dynamics.F_gravity)
        assert_symbolic_equal(actual_gravity, expected_gravity)


def test_multiangle_flexible_strain_potential_energy_is_stored(seven_part_dynamics):
    dynamics = seven_part_dynamics

    expected = sm.S.Zero
    for body in dynamics.outer_flexible_panels:
        values = dynamics.flexible_bodies[body]
        for eta_k, K_k in zip(values["eta_list"], values["k_modal_list"]):
            expected += sm.Rational(1, 2) * K_k * eta_k**2

    for body in dynamics.inter_bus_flexible_panels:
        element_coordinates = dynamics.element_coordinates[body]
        stiffness_matrix = dynamics.boundary_compatible_stiffness_matrices[body]
        expected += sm.Rational(1, 2) * (
            element_coordinates.T * stiffness_matrix * element_coordinates
        )[0]

    assert_symbolic_equal(dynamics.V_strain, expected)


def test_multiangle_flexible_strain_forces_are_added_from_potential(seven_part_dynamics):
    dynamics = seven_part_dynamics

    for row, coordinate in enumerate(dynamics.q):
        non_strain = dynamics.partial_v_G[row].dot(dynamics.F_gravity)

        for body in dynamics.body_names:
            non_strain += (
                dynamics.partial_linear_velocities[body][row].dot(
                    dynamics.external_forces[body]
                )
                + dynamics.partial_angular_velocities[body][row].dot(
                    dynamics.external_torques[body]
                )
            )

        actual = dynamics.generalised_active_forces[row] - non_strain
        expected = -sm.diff(dynamics.V_strain, coordinate)

        assert_symbolic_equal(actual, expected)


def test_multiangle_inter_bus_panels_store_boundary_compatible_stiffness_matrices(
    seven_part_dynamics,
):
    dynamics = seven_part_dynamics

    assert set(dynamics.boundary_compatible_stiffness_matrices) == set(
        dynamics.inter_bus_flexible_panels
    )

    for panel in dynamics.inter_bus_flexible_panels:
        expected_size = 4 + len(dynamics.flexible_bodies[panel]["eta_list"])
        assert dynamics.boundary_compatible_stiffness_matrices[panel].shape == (
            expected_size,
            expected_size,
        )
