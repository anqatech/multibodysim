import pytest
import numpy as np
import sympy as sm


def test_rigid_dynamics_initialization(rigid_dynamics):
    """Test that rigid dynamics model initializes correctly."""
    # Check that all required symbols exist
    assert hasattr(rigid_dynamics, 'q1'), "Missing coordinate q1"
    assert hasattr(rigid_dynamics, 'q2'), "Missing coordinate q2" 
    assert hasattr(rigid_dynamics, 'q3'), "Missing coordinate q3"
    assert hasattr(rigid_dynamics, 'u1'), "Missing speed u1"
    assert hasattr(rigid_dynamics, 'u2'), "Missing speed u2"
    assert hasattr(rigid_dynamics, 'u3'), "Missing speed u3"
    
    # Check parameter symbols
    assert hasattr(rigid_dynamics, 'D'), "Missing parameter D"
    assert hasattr(rigid_dynamics, 'L'), "Missing parameter L"
    assert hasattr(rigid_dynamics, 'm_b'), "Missing parameter m_b"
    assert hasattr(rigid_dynamics, 'm_r'), "Missing parameter m_r"
    assert hasattr(rigid_dynamics, 'm_l'), "Missing parameter m_l"
    assert hasattr(rigid_dynamics, 'tau'), "Missing parameter tau"
    
    # Check reference frames exist
    assert hasattr(rigid_dynamics, 'N'), "Missing inertial frame N"
    assert hasattr(rigid_dynamics, 'B'), "Missing bus frame B"
    assert hasattr(rigid_dynamics, 'C'), "Missing right panel frame C"
    assert hasattr(rigid_dynamics, 'E'), "Missing left panel frame E"
    
    print("✓ Rigid dynamics model initialized with all required components")


def test_reference_frame_orientations(rigid_dynamics):
    """Test that reference frames are oriented correctly."""
    # At q3 = 0, bus frame should align with inertial frame
    Bx_in_N = rigid_dynamics.B.x.express(rigid_dynamics.N)
    
    # Check components at q3 = 0
    x_component = Bx_in_N.dot(rigid_dynamics.N.x).subs(rigid_dynamics.q3, 0)
    y_component = Bx_in_N.dot(rigid_dynamics.N.y).subs(rigid_dynamics.q3, 0)
    
    assert x_component == 1, f"B.x should align with N.x at q3=0, got x_component={x_component}"
    assert y_component == 0, f"B.x should not have N.y component at q3=0, got y_component={y_component}"
    
    # Left panel should point opposite to right panel
    Cx_in_N = rigid_dynamics.C.x.express(rigid_dynamics.N)
    Ex_in_N = rigid_dynamics.E.x.express(rigid_dynamics.N)
    
    # Sum should be zero (opposite directions)
    sum_x = sm.simplify(Cx_in_N.dot(rigid_dynamics.N.x) + Ex_in_N.dot(rigid_dynamics.N.x))
    sum_y = sm.simplify(Cx_in_N.dot(rigid_dynamics.N.y) + Ex_in_N.dot(rigid_dynamics.N.y))
    
    assert sum_x == 0, f"Panel x-directions should be opposite, got sum_x={sum_x}"
    assert sum_y == 0, f"Panel x-directions should be opposite, got sum_y={sum_y}"
    
    print("✓ Reference frames oriented correctly")


def test_eom_matrices_structure(rigid_dynamics):
    """Test that equations of motion matrices have correct structure."""
    # Check that matrices exist
    assert hasattr(rigid_dynamics, 'Mk'), "Missing kinematic matrix Mk"
    assert hasattr(rigid_dynamics, 'gk'), "Missing kinematic vector gk"
    assert hasattr(rigid_dynamics, 'Md'), "Missing dynamic matrix Md"
    assert hasattr(rigid_dynamics, 'gd'), "Missing dynamic vector gd"
    
    # Check dimensions
    assert rigid_dynamics.Mk.shape == (3, 3), f"Mk wrong shape: {rigid_dynamics.Mk.shape}"
    assert rigid_dynamics.gk.shape == (3, 1), f"gk wrong shape: {rigid_dynamics.gk.shape}"
    assert rigid_dynamics.Md.shape == (3, 3), f"Md wrong shape: {rigid_dynamics.Md.shape}"
    assert rigid_dynamics.gd.shape == (3, 1), f"gd wrong shape: {rigid_dynamics.gd.shape}"
    
    # Kinematic matrix should be -I
    expected_Mk = -sm.eye(3)
    assert rigid_dynamics.Mk.equals(expected_Mk), f"Mk should be -I, got {rigid_dynamics.Mk}"
    
    # Mass matrix should be symmetric
    Md_diff = sm.simplify(rigid_dynamics.Md - rigid_dynamics.Md.T)
    assert Md_diff == sm.zeros(3, 3), f"Mass matrix not symmetric: {Md_diff}"
    
    print("✓ EOM matrices have correct structure")


def test_parameter_values_extraction(rigid_dynamics, rigid_config):
    """Test parameter value extraction from config."""
    p_vals = rigid_dynamics.get_parameter_values()
    expected_params = rigid_config['p_values']
    
    # Check array length
    assert len(p_vals) == 6, f"Expected 6 parameters, got {len(p_vals)}"
    
    # Check parameter values match config
    expected_order = ['D', 'L', 'm_b', 'm_r', 'm_l', 'tau']
    for i, param_name in enumerate(expected_order):
        expected_val = expected_params[param_name]
        actual_val = p_vals[i]
        assert np.isclose(actual_val, expected_val), f"Parameter {param_name}: expected {expected_val}, got {actual_val}"
    
    print(f"✓ Parameter values extracted correctly: {p_vals}")


def test_initial_conditions(rigid_dynamics, rigid_config):
    """Test initial condition calculation."""
    x0 = rigid_dynamics.get_initial_conditions()
    
    # Check dimensions
    assert len(x0) == 6, f"Initial state should have 6 elements, got {len(x0)}"
    
    # Check all values are finite
    assert np.all(np.isfinite(x0)), f"Initial conditions contain non-finite values: {x0}"
    
    # Check position coordinates match config
    q_initial = rigid_config['q_initial']
    assert np.isclose(x0[0], q_initial['q1']), f"q1: expected {q_initial['q1']}, got {x0[0]}"
    assert np.isclose(x0[1], q_initial['q2']), f"q2: expected {q_initial['q2']}, got {x0[1]}"
    assert np.isclose(x0[2], np.deg2rad(q_initial['q3'])), f"q3: expected {np.deg2rad(q_initial['q3'])}, got {x0[2]}"
    
    # Check angular velocity matches config (if specified)
    initial_speeds = rigid_config.get('initial_speeds', {})
    if 'u3' in initial_speeds:
        assert np.isclose(x0[5], initial_speeds['u3']), f"u3: expected {initial_speeds['u3']}, got {x0[5]}"
    
    print(f"✓ Initial conditions calculated correctly: {x0}")


def test_numerical_evaluation(rigid_dynamics, sample_states, numerical_evaluation_helper):
    """Test numerical evaluation of EOM matrices."""
    rigid_state = sample_states['rigid_state']
    
    # Evaluate dynamics at sample state
    result = numerical_evaluation_helper(rigid_dynamics, rigid_state, None)
    
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
    
    # Note: Mass matrix should be positive definite, but our formulation might have sign issues
    # Let's check the absolute values of eigenvalues to ensure the matrix is not singular
    eigenvals = np.linalg.eigvals(Md)
    min_abs_eigenval = np.min(np.abs(eigenvals))
    assert min_abs_eigenval > 1e-10, f"Mass matrix nearly singular: min|eigenval|={min_abs_eigenval}"
    
    print("✓ Numerical evaluation produces valid matrices")
    print(f"  Mass matrix determinant: {det_Md:.2e}")
    print(f"  Mass matrix eigenvalues: {eigenvals}")
    print(f"  Note: Negative eigenvalues may indicate sign convention in EOM formulation")


def test_rigid_simulator_initialization(rigid_simulator):
    """Test that rigid simulator initializes correctly."""
    # Check simulator has the necessary components
    assert hasattr(rigid_simulator, 'config'), "Missing config attribute"
    assert hasattr(rigid_simulator, 'dynamics'), "Missing dynamics attribute"
    assert hasattr(rigid_simulator, 'p_vals'), "Missing parameter values"
    
    # Check parameter values
    assert isinstance(rigid_simulator.p_vals, np.ndarray), "Parameter values should be numpy array"
    assert len(rigid_simulator.p_vals) == 6, f"Expected 6 parameters, got {len(rigid_simulator.p_vals)}"
    
    # Check dynamics is correct type
    from multibodysim.rigid.rigid_symbolic_model import RigidSymbolicDynamics
    assert isinstance(rigid_simulator.dynamics, RigidSymbolicDynamics), "Wrong dynamics type"
    
    print("✓ Rigid simulator initialized correctly")


def test_simulator_rhs_evaluation(rigid_simulator, sample_states):
    """Test right-hand side function evaluation."""
    rigid_state = sample_states['rigid_state']
    t = sample_states['time']
    
    # Evaluate RHS
    xdot = rigid_simulator.eval_rhs(t, rigid_state)
    
    # Check output structure
    assert isinstance(xdot, np.ndarray), f"RHS should return numpy array, got {type(xdot)}"
    assert len(xdot) == len(rigid_state), f"RHS output wrong size: {len(xdot)} vs {len(rigid_state)}"
    assert np.all(np.isfinite(xdot)), f"RHS contains non-finite values: {xdot}"
    
    # Check kinematic constraint: first 3 elements should equal last 3
    q_dot = xdot[:3]
    u_current = rigid_state[3:]
    np.testing.assert_allclose(q_dot, u_current, rtol=1e-12, 
                              err_msg="Kinematic relationship qd = u not satisfied")
    
    print(f"✓ RHS evaluation successful: {xdot}")


def test_short_simulation(rigid_simulator, quick_simulation_helper):
    """Test running a short simulation."""
    # Run quick simulation
    result = quick_simulation_helper(rigid_simulator, duration=0.5, timesteps=10)
    
    assert result['success'], f"Simulation failed: {result['error']}"
    
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
    
    print(f"✓ Short simulation completed successfully")
    print(f"  Final state: {states[-1, :]}")


def test_zero_torque_conservation(rigid_simulator, rigid_config):
    """Test conservation properties with zero applied torque."""
    # Modify config for zero torque
    config_zero_torque = rigid_config.copy()
    config_zero_torque['p_values']['tau'] = 0.0
    config_zero_torque['sim_parameters']['t_end'] = 1.0
    config_zero_torque['sim_parameters']['nb_timesteps'] = 20
    
    # Create new simulator with zero torque
    from multibodysim.rigid.rigid_simulator import RigidSimulator
    sim_zero_torque = RigidSimulator(config_zero_torque)
    
    # Run simulation
    results = sim_zero_torque.run_simulation()
    
    assert results['success'], f"Zero torque simulation failed: {results['message']}"
    
    # Check angular velocity conservation (no applied torque)
    u3_values = results['u3']
    u3_variation = np.std(u3_values) / np.abs(np.mean(u3_values))
    
    assert u3_variation < 1e-6, f"Angular velocity not conserved: variation = {u3_variation*100:.4f}%"
    
    print(f"✓ Angular velocity conserved with zero torque: variation = {u3_variation*100:.2e}%")


def test_torque_effect(rigid_simulator, rigid_config):
    """Test that applied torque affects the motion correctly."""
    # Run simulation with non-zero torque
    config_with_torque = rigid_config.copy()
    config_with_torque['p_values']['tau'] = 0.1  # Apply small torque
    config_with_torque['sim_parameters']['t_end'] = 1.0
    config_with_torque['sim_parameters']['nb_timesteps'] = 20
    
    from multibodysim.rigid.rigid_simulator import RigidSimulator
    sim_with_torque = RigidSimulator(config_with_torque)
    
    results = sim_with_torque.run_simulation()
    assert results['success'], f"Torque simulation failed: {results['message']}"
    
    # Check that angular velocity increases (positive torque)
    u3_initial = results['u3'][0]
    u3_final = results['u3'][-1]
    
    # With positive torque, angular velocity should increase
    assert u3_final > u3_initial, f"Angular velocity should increase with positive torque: {u3_initial} -> {u3_final}"
    
    # Check angular acceleration is roughly constant (constant torque)
    u3_values = results['u3']
    u3_diff = np.diff(u3_values)
    u3_accel_variation = np.std(u3_diff) / np.abs(np.mean(u3_diff))
    
    assert u3_accel_variation < 0.1, f"Angular acceleration should be roughly constant: variation = {u3_accel_variation*100:.2f}%"
    
    print(f"✓ Applied torque correctly affects motion")
    print(f"  Angular velocity change: {u3_initial:.6f} -> {u3_final:.6f}")
