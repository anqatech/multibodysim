import pytest
import numpy as np
import sympy as sm


def test_rigid_7part_dynamics_initialization(rigid_7part_dynamics):
    """Test that 7-part rigid dynamics model initializes correctly."""
    # Check generalized coordinates (same as 3-part: q1, q2, q3)
    assert hasattr(rigid_7part_dynamics, 'q1'), "Missing coordinate q1"
    assert hasattr(rigid_7part_dynamics, 'q2'), "Missing coordinate q2"
    assert hasattr(rigid_7part_dynamics, 'q3'), "Missing coordinate q3"
    
    # Check generalized speeds (same as 3-part: u1, u2, u3)
    assert hasattr(rigid_7part_dynamics, 'u1'), "Missing speed u1"
    assert hasattr(rigid_7part_dynamics, 'u2'), "Missing speed u2"
    assert hasattr(rigid_7part_dynamics, 'u3'), "Missing speed u3"
    
    # Check geometric parameters
    assert hasattr(rigid_7part_dynamics, 'D'), "Missing parameter D"
    assert hasattr(rigid_7part_dynamics, 'L'), "Missing parameter L"
    
    # Check 7-part mass parameters
    bus_masses = ['m_b1', 'm_b2', 'm_b3']
    panel_masses = ['m_p1', 'm_p2', 'm_p3', 'm_p4']
    
    for mass in bus_masses + panel_masses:
        assert hasattr(rigid_7part_dynamics, mass), f"Missing mass parameter {mass}"
    
    # Check applied torque
    assert hasattr(rigid_7part_dynamics, 'tau'), "Missing parameter tau"
    
    print("✓ 7-part rigid dynamics model initialized with all required components")
    print(f"  Bus masses: {bus_masses}")
    print(f"  Panel masses: {panel_masses}")


def test_reference_frames_7part(rigid_7part_dynamics):
    """Test that all 7 reference frames are created correctly."""
    # Check inertial frame
    assert hasattr(rigid_7part_dynamics, 'N'), "Missing inertial frame N"
    
    # Check bus frames (3 buses)
    bus_frames = ['B1', 'B2', 'B3']
    for frame_name in bus_frames:
        assert hasattr(rigid_7part_dynamics, frame_name), f"Missing bus frame {frame_name}"
    
    # Check panel frames (4 panels)
    panel_frames = ['P1', 'P2', 'P3', 'P4']
    for frame_name in panel_frames:
        assert hasattr(rigid_7part_dynamics, frame_name), f"Missing panel frame {frame_name}"
    
    # Check frame list exists and has correct length
    assert hasattr(rigid_7part_dynamics, 'frame_list'), "Missing frame_list attribute"
    assert len(rigid_7part_dynamics.frame_list) == 7, f"Expected 7 frames, got {len(rigid_7part_dynamics.frame_list)}"
    
    print("✓ All 7 reference frames created correctly")


def test_body_centers_of_mass_7part(rigid_7part_dynamics):
    """Test that all 7 body centers of mass are defined."""
    # Check bus centers of mass
    bus_cms = ['Bus1_cm', 'Bus2_cm', 'Bus3_cm']
    for cm_name in bus_cms:
        assert hasattr(rigid_7part_dynamics, cm_name), f"Missing bus CM {cm_name}"
    
    # Check panel centers of mass
    panel_cms = ['P1_cm', 'P2_cm', 'P3_cm', 'P4_cm']
    for cm_name in panel_cms:
        assert hasattr(rigid_7part_dynamics, cm_name), f"Missing panel CM {cm_name}"
    
    # Check body list exists and has correct length
    assert hasattr(rigid_7part_dynamics, 'body_list'), "Missing body_list attribute"
    assert len(rigid_7part_dynamics.body_list) == 7, f"Expected 7 bodies, got {len(rigid_7part_dynamics.body_list)}"
    
    print("✓ All 7 body centers of mass defined correctly")


def test_joint_points_7part(rigid_7part_dynamics):
    """Test that joint points are properly defined for the 7-part system."""
    # Expected joint points based on the kinematic chain
    joint_points = [
        'Joint1_Right', 'Joint1_Left',    # Joints on Bus 1
        'Joint2_Right', 'Joint2_Left',    # Joints on Bus 2 (central)
        'Joint3_Right', 'Joint3_Left'     # Joints on Bus 3
    ]
    
    for joint_name in joint_points:
        assert hasattr(rigid_7part_dynamics, joint_name), f"Missing joint point {joint_name}"
    
    print("✓ All joint points defined for kinematic chain")


def test_frame_orientations_7part(rigid_7part_dynamics):
    """Test that frames are oriented correctly for the 7-part system."""
    # At q3 = 0, bus frames should align with inertial frame
    B2x_in_N = rigid_7part_dynamics.B2.x.express(rigid_7part_dynamics.N)
    
    x_component = B2x_in_N.dot(rigid_7part_dynamics.N.x).subs(rigid_7part_dynamics.q3, 0)
    y_component = B2x_in_N.dot(rigid_7part_dynamics.N.y).subs(rigid_7part_dynamics.q3, 0)
    
    assert x_component == 1, f"B2.x should align with N.x at q3=0, got x_component={x_component}"
    assert y_component == 0, f"B2.x should not have N.y component at q3=0, got y_component={y_component}"
    
    # Check that P1 and P2 point opposite to P3 and P4 (panels on opposite sides)
    P1x_in_N = rigid_7part_dynamics.P1.x.express(rigid_7part_dynamics.N)
    P3x_in_N = rigid_7part_dynamics.P3.x.express(rigid_7part_dynamics.N)
    
    # P1 and P3 should be opposite (P1 rotated by π)
    sum_x = sm.simplify(P1x_in_N.dot(rigid_7part_dynamics.N.x) + P3x_in_N.dot(rigid_7part_dynamics.N.x))
    sum_y = sm.simplify(P1x_in_N.dot(rigid_7part_dynamics.N.y) + P3x_in_N.dot(rigid_7part_dynamics.N.y))
    
    assert sum_x == 0, f"P1 and P3 should be opposite: sum_x={sum_x}"
    assert sum_y == 0, f"P1 and P3 should be opposite: sum_y={sum_y}"
    
    print("✓ 7-part frame orientations correct")


def test_eom_matrices_structure_7part(rigid_7part_dynamics):
    """Test that 7-part EOM matrices have correct structure."""
    # Check that matrices exist
    assert hasattr(rigid_7part_dynamics, 'Mk'), "Missing kinematic matrix Mk"
    assert hasattr(rigid_7part_dynamics, 'gk'), "Missing kinematic vector gk"
    assert hasattr(rigid_7part_dynamics, 'Md'), "Missing dynamic matrix Md"
    assert hasattr(rigid_7part_dynamics, 'gd'), "Missing dynamic vector gd"
    
    # Check dimensions (still 3×3 - same DOF as 3-part system)
    assert rigid_7part_dynamics.Mk.shape == (3, 3), f"Mk wrong shape: {rigid_7part_dynamics.Mk.shape}"
    assert rigid_7part_dynamics.gk.shape == (3, 1), f"gk wrong shape: {rigid_7part_dynamics.gk.shape}"
    assert rigid_7part_dynamics.Md.shape == (3, 3), f"Md wrong shape: {rigid_7part_dynamics.Md.shape}"
    assert rigid_7part_dynamics.gd.shape == (3, 1), f"gd wrong shape: {rigid_7part_dynamics.gd.shape}"
    
    # Kinematic matrix should be -I
    expected_Mk = -sm.eye(3)
    assert rigid_7part_dynamics.Mk.equals(expected_Mk), f"Mk should be -I, got {rigid_7part_dynamics.Mk}"
    
    # Mass matrix should be symmetric
    Md_diff = sm.simplify(rigid_7part_dynamics.Md - rigid_7part_dynamics.Md.T)
    assert Md_diff == sm.zeros(3, 3), f"Mass matrix not symmetric: {Md_diff}"
    
    print("✓ 7-part EOM matrices have correct structure")


def test_parameter_values_extraction_7part(rigid_7part_dynamics, rigid_nbody_config):
    """Test parameter value extraction for 7-part system."""
    p_vals = rigid_7part_dynamics.get_parameter_values()
    expected_params = rigid_nbody_config['p_values']
    
    # Check array length (10 parameters for 7-part system)
    assert len(p_vals) == 10, f"Expected 10 parameters, got {len(p_vals)}"
    
    # Check parameter values match config
    expected_order = ['D', 'L', 'm_b1', 'm_b2', 'm_b3', 'm_p1', 'm_p2', 'm_p3', 'm_p4', 'tau']
    for i, param_name in enumerate(expected_order):
        expected_val = expected_params[param_name]
        actual_val = p_vals[i]
        assert np.isclose(actual_val, expected_val), f"Parameter {param_name}: expected {expected_val}, got {actual_val}"
    
    print(f"✓ 7-part parameter values extracted correctly")


def test_initial_conditions_7part(rigid_7part_dynamics, rigid_nbody_config):
    """Test initial condition calculation for 7-part system."""
    x0 = rigid_7part_dynamics.get_initial_conditions()
    
    # Check dimensions (still 6 states: 3 positions + 3 velocities)
    assert len(x0) == 6, f"Initial state should have 6 elements, got {len(x0)}"
    
    # Check all values are finite
    assert np.all(np.isfinite(x0)), f"Initial conditions contain non-finite values: {x0}"
    
    # Check position coordinates match config
    q_initial = rigid_nbody_config['q_initial']
    assert np.isclose(x0[0], q_initial['q1']), f"q1: expected {q_initial['q1']}, got {x0[0]}"
    assert np.isclose(x0[1], q_initial['q2']), f"q2: expected {q_initial['q2']}, got {x0[1]}"
    assert np.isclose(x0[2], np.deg2rad(q_initial['q3'])), f"q3: expected {np.deg2rad(q_initial['q3'])}, got {x0[2]}"
    
    # Check angular velocity matches config
    initial_speeds = rigid_nbody_config.get('initial_speeds', {})
    if 'u3' in initial_speeds:
        assert np.isclose(x0[5], initial_speeds['u3']), f"u3: expected {initial_speeds['u3']}, got {x0[5]}"
    
    print(f"✓ 7-part initial conditions calculated correctly: {x0}")


def test_center_of_mass_calculation_7part(rigid_7part_dynamics):
    """Test center of mass calculation for 7-part system."""
    # Check that r_GB exists and has the right structure
    assert hasattr(rigid_7part_dynamics, 'r_GB'), "Missing center of mass vector 'r_GB'"
    
    # r_GB should be a Vector (SymPy mechanics vector)
    from sympy.physics.vector import Vector
    assert isinstance(rigid_7part_dynamics.r_GB, Vector), "r_GB should be a Vector"
    
    print(f"✓ Center of mass calculation implemented for 7-part system")


def test_inertia_calculations_7part(rigid_7part_dynamics):
    """Test that inertia calculations include all 7 bodies."""
    # Check that inertia list exists and has correct length
    assert hasattr(rigid_7part_dynamics, 'inertias'), "Missing inertias list"
    assert len(rigid_7part_dynamics.inertias) == 7, f"Expected 7 inertias, got {len(rigid_7part_dynamics.inertias)}"
    
    print("✓ Inertia calculations include all 7 bodies")


def test_numerical_evaluation_7part(rigid_7part_dynamics, sample_states, numerical_evaluation_helper):
    """Test numerical evaluation of 7-part EOM matrices."""
    rigid_state = sample_states['rigid_state']
    
    # Evaluate dynamics at sample state
    result = numerical_evaluation_helper(rigid_7part_dynamics, rigid_state, None)
    
    assert result['success'], f"Numerical evaluation failed: {result['error']}"
    
    Mk, gk, Md, gd = result['Mk'], result['gk'], result['Md'], result['gd']
    
    # Check return types and shapes
    assert isinstance(Mk, np.ndarray), f"Mk should be numpy array, got {type(Mk)}"
    assert isinstance(Md, np.ndarray), f"Md should be numpy array, got {type(Md)}"
    assert Mk.shape == (3, 3), f"Mk wrong shape: {Mk.shape}"
    assert Md.shape == (3, 3), f"Md wrong shape: {Md.shape}"
    
    # Check all values are finite
    assert np.all(np.isfinite(Mk)), "Mk contains non-finite values"
    assert np.all(np.isfinite(Md)), "Md contains non-finite values"
    assert np.all(np.isfinite(gk)), "gk contains non-finite values"
    assert np.all(np.isfinite(gd)), "gd contains non-finite values"
    
    # Check Mk is -I
    np.testing.assert_allclose(Mk, -np.eye(3), rtol=1e-12, err_msg="Mk should be -I")
    
    # Check mass matrix determinant is non-zero (invertible)
    det_Md = np.linalg.det(Md)
    assert abs(det_Md) > 1e-10, f"Mass matrix nearly singular: det={det_Md}"
    
    print("✓ 7-part numerical evaluation produces valid matrices")
    print(f"  Mass matrix determinant: {det_Md:.2e}")


def test_rigid_7part_simulator_initialization(rigid_7part_simulator):
    """Test that 7-part rigid simulator initializes correctly."""
    # Check simulator has necessary components
    assert hasattr(rigid_7part_simulator, 'config'), "Missing config attribute"
    assert hasattr(rigid_7part_simulator, 'dynamics'), "Missing dynamics attribute"
    assert hasattr(rigid_7part_simulator, 'p_vals'), "Missing parameter values"
    
    # Check parameter values
    assert isinstance(rigid_7part_simulator.p_vals, np.ndarray), "Parameter values should be numpy array"
    assert len(rigid_7part_simulator.p_vals) == 10, f"Expected 10 parameters, got {len(rigid_7part_simulator.p_vals)}"
    
    # Check dynamics is correct type
    from multibodysim.rigid.rigid_symbolic_7part import Rigid7PartSymbolicDynamics
    assert isinstance(rigid_7part_simulator.dynamics, Rigid7PartSymbolicDynamics), "Wrong dynamics type"
    
    print("✓ 7-part rigid simulator initialized correctly")


def test_simulator_rhs_evaluation_7part(rigid_7part_simulator, sample_states):
    """Test right-hand side function evaluation for 7-part system."""
    rigid_state = sample_states['rigid_state']
    t = sample_states['time']
    
    # Evaluate RHS
    xdot = rigid_7part_simulator.eval_rhs(t, rigid_state)
    
    # Check output structure
    assert isinstance(xdot, np.ndarray), f"RHS should return numpy array, got {type(xdot)}"
    assert len(xdot) == len(rigid_state), f"RHS output wrong size: {len(xdot)} vs {len(rigid_state)}"
    assert np.all(np.isfinite(xdot)), f"RHS contains non-finite values: {xdot}"
    
    # Check kinematic constraint: first 3 elements should equal last 3
    q_dot = xdot[:3]
    u_current = rigid_state[3:]
    np.testing.assert_allclose(q_dot, u_current, rtol=1e-12,
                              err_msg="Kinematic relationship qd = u not satisfied")
    
    print(f"✓ 7-part RHS evaluation successful: {xdot}")


def test_short_simulation_7part(rigid_7part_simulator, quick_simulation_helper):
    """Test running a short simulation of 7-part system."""
    # Run quick simulation
    result = quick_simulation_helper(rigid_7part_simulator, duration=0.5, timesteps=10)
    
    assert result['success'], f"7-part simulation failed: {result['error']}"
    
    t = result['time']
    states = result['states']
    
    # Check output structure
    assert len(t) == 10, f"Expected 10 time points, got {len(t)}"
    assert states.shape == (10, 6), f"Expected (10,6) state array, got {states.shape}"
    
    # Check all values are finite
    assert np.all(np.isfinite(t)), "Time vector contains non-finite values"
    assert np.all(np.isfinite(states)), "State vector contains non-finite values"
    
    # Check simulation progresses in time
    assert np.all(np.diff(t) > 0), "Time should be monotonically increasing"
    assert np.isclose(t[-1], 0.5, rtol=1e-10), f"Final time should be 0.5, got {t[-1]}"
    
    print(f"✓ 7-part short simulation completed successfully")
    print(f"  Final state: {states[-1, :]}")


def test_mass_comparison_3part_vs_7part(rigid_7part_simulator, rigid_7part_dynamics, rigid_nbody_config):
    """Test that 7-part system has higher total mass than 3-part system."""
    # Calculate total mass from config
    config_params = rigid_nbody_config['p_values']
    total_mass_7part = (config_params['m_b1'] + config_params['m_b2'] + config_params['m_b3'] +
                       config_params['m_p1'] + config_params['m_p2'] + config_params['m_p3'] + config_params['m_p4'])
    
    # Should be significantly larger than typical 3-part system
    typical_3part_mass = 35.0  # From 3-part config (30 + 2 + 3)
    assert total_mass_7part > typical_3part_mass, f"7-part mass ({total_mass_7part}) should exceed 3-part mass ({typical_3part_mass})"
    
    print(f"✓ 7-part total mass: {total_mass_7part} kg (vs typical 3-part: {typical_3part_mass} kg)")


def test_7part_conservation_properties(rigid_7part_simulator, rigid_nbody_config):
    """Test conservation properties for 7-part system with zero torque."""
    # Modify config for zero torque
    config_zero_torque = rigid_nbody_config.copy()
    config_zero_torque['p_values']['tau'] = 0.0
    config_zero_torque['sim_parameters']['t_end'] = 1.0
    config_zero_torque['sim_parameters']['nb_timesteps'] = 20
    
    # Create new simulator with zero torque
    from multibodysim.rigid.rigid_simulator_7parts import Rigid7PartSimulator
    sim_zero_torque = Rigid7PartSimulator(config_zero_torque)
    
    # Run simulation
    results = sim_zero_torque.run_simulation()
    
    assert results['success'], f"7-part zero torque simulation failed: {results['message']}"
    
    # Check angular velocity conservation
    u3_values = results['u3']
    u3_variation = np.std(u3_values) / np.abs(np.mean(u3_values))
    
    assert u3_variation < 1e-6, f"Angular velocity not conserved in 7-part system: variation = {u3_variation*100:.4f}%"
    
    print(f"✓ 7-part system conserves angular momentum: variation = {u3_variation*100:.2e}%")


def test_7part_derived_quantities(rigid_7part_simulator, quick_simulation_helper):
    """Test calculation of derived quantities (momentum, energy) for 7-part system."""
    # Run simulation to generate derived quantities
    result = quick_simulation_helper(rigid_7part_simulator, duration=0.2, timesteps=5)
    
    assert result['success'], f"Simulation for derived quantities failed: {result['error']}"
    
    # Check if derived quantities are calculated by the simulator
    # (These might be added by the _calculate_derived_quantities method)
    if 'linear_momentum' in result:
        momentum = result['linear_momentum']
        assert momentum.shape[1] == 3, f"Linear momentum should be 3D, got shape {momentum.shape}"
        assert np.all(np.isfinite(momentum)), "Linear momentum contains non-finite values"
        print(f"✓ Linear momentum calculated for 7-part system")
    
    if 'angular_momentum' in result:
        ang_momentum = result['angular_momentum']
        assert len(ang_momentum) == len(result['time']), "Angular momentum length mismatch"
        assert np.all(np.isfinite(ang_momentum)), "Angular momentum contains non-finite values"
        print(f"✓ Angular momentum calculated for 7-part system")
    
    if 'kinetic_energy' in result:
        kinetic_energy = result['kinetic_energy']
        assert len(kinetic_energy) == len(result['time']), "Kinetic energy length mismatch"
        assert np.all(kinetic_energy >= 0), "Kinetic energy should be non-negative"
        assert np.all(np.isfinite(kinetic_energy)), "Kinetic energy contains non-finite values"
        print(f"✓ Kinetic energy calculated for 7-part system")
    
    print("✓ Derived quantities calculation verified for 7-part system")


def test_7part_inertia_effects(rigid_7part_simulator, rigid_nbody_config):
    """Test that 7-part system shows different inertial behavior than 3-part."""
    # Create configuration with same applied torque as 3-part
    config_with_torque = rigid_nbody_config.copy()
    config_with_torque['p_values']['tau'] = 0.05  # Same as other tests
    config_with_torque['sim_parameters']['t_end'] = 1.0
    config_with_torque['sim_parameters']['nb_timesteps'] = 20
    
    from multibodysim.rigid.rigid_simulator_7parts import Rigid7PartSimulator
    sim_with_torque = Rigid7PartSimulator(config_with_torque)
    
    results = sim_with_torque.run_simulation()
    assert results['success'], f"7-part torque simulation failed: {results['message']}"
    
    # Check that angular acceleration is smaller than 3-part (higher inertia)
    u3_values = results['u3']
    u3_initial = u3_values[0]
    u3_final = u3_values[-1]
    angular_acceleration = (u3_final - u3_initial) / results['time'][-1]
    
    # Should be smaller than 3-part system due to higher moment of inertia
    typical_3part_accel = 0.2  # Rough estimate from 3-part tests
    assert abs(angular_acceleration) < typical_3part_accel, f"7-part angular acceleration ({angular_acceleration:.4f}) should be smaller than 3-part"
    
    print(f"✓ 7-part system shows lower angular acceleration due to higher inertia")
    print(f"  Angular acceleration: {angular_acceleration:.4f} rad/s² (vs typical 3-part: ~{typical_3part_accel} rad/s²)")
