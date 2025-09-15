import pytest
import numpy as np
from scipy.integrate import solve_ivp


def test_simulator_initialization(simulator):
    # Check that lambdified functions exist
    assert hasattr(simulator, 'eval_eom_matrices'), "Missing eval_eom_matrices function"
    assert hasattr(simulator, 'calculate_initial_speeds'), "Missing calculate_initial_speeds function"
    assert callable(simulator.eval_eom_matrices), "eval_eom_matrices not callable"
    assert callable(simulator.calculate_initial_speeds), "calculate_initial_speeds not callable"
    
    print(f"✓ Simulator initialized with compiled numerical functions")


def test_eom_matrices_evaluation(simulator, standard_params):
    # Test state vectors
    q_test = [0.0, 0.0, 0.0]  # [x, y, theta]
    u_test = [0.0, 0.0, 0.1]  # [vx, vy, omega]
    p_test = [
        standard_params["D"], standard_params["L"], 
        standard_params["m_r"], standard_params["m_l"], 
        standard_params["m_b"], standard_params["tau"]
    ]
    
    # Evaluate matrices
    Mk, gk, Md, gd = simulator.eval_eom_matrices(q_test, u_test, p_test)
    
    # Check return types and shapes
    assert isinstance(Mk, np.ndarray), f"Mk should be numpy array, got {type(Mk)}"
    assert isinstance(Md, np.ndarray), f"Md should be numpy array, got {type(Md)}"
    assert Mk.shape == (3, 3), f"Mk wrong shape: {Mk.shape}"
    assert Md.shape == (3, 3), f"Md wrong shape: {Md.shape}"
    
    # Check values are finite
    assert np.all(np.isfinite(Mk)), "Mk contains non-finite values"
    assert np.all(np.isfinite(Md)), "Md contains non-finite values"
    assert np.all(np.isfinite(gk)), "gk contains non-finite values"
    assert np.all(np.isfinite(gd)), "gd contains non-finite values"
    
    # Check Mk is -I (from your kinematic equations)
    expected_Mk = -np.eye(3)
    np.testing.assert_allclose(Mk, expected_Mk, rtol=1e-10, 
                              err_msg="Mk should be negative identity matrix")
    
    print(f"✓ EOM matrices evaluate correctly to finite numerical values")
    print(f"✓ Mk = -I as expected, det(Md) = {np.linalg.det(Md):.2e}")


def test_initial_speeds_calculation(simulator, standard_params):
    # Test parameters
    q3_test = 0.0  # Initial angle (radians)
    u3_test = 0.1  # Initial angular velocity
    
    u1_calc, u2_calc = simulator.calculate_initial_speeds(
        q3_test, u3_test,
        standard_params["D"], standard_params["L"],
        standard_params["m_r"], standard_params["m_l"], standard_params["m_b"]
    )
    
    # Results should be finite
    assert np.isfinite(u1_calc), f"u1 calculation failed: {u1_calc}"
    assert np.isfinite(u2_calc), f"u2 calculation failed: {u2_calc}"
    
    print(f"✓ Initial speeds calculated: u1={u1_calc:.6f}, u2={u2_calc:.6f}")
    
    # For symmetric satellite (equal panel masses), should be zero
    if abs(standard_params["m_r"] - standard_params["m_l"]) < 1e-10:
        assert abs(u1_calc) < 1e-10, f"u1 should be ~0 for symmetric satellite: {u1_calc}"
        assert abs(u2_calc) < 1e-10, f"u2 should be ~0 for symmetric satellite: {u2_calc}"
        print(f"✓ Symmetric satellite gives zero linear velocities as expected")


def test_rhs_function_evaluation(simulator, standard_params):
    # Initial state vector [q1, q2, q3, u1, u2, u3]
    x_test = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.1])
    t_test = 0.0
    p_vec = [
        standard_params["D"], standard_params["L"], 
        standard_params["m_r"], standard_params["m_l"], 
        standard_params["m_b"], standard_params["tau"]
    ]
    
    # Evaluate RHS
    xd = simulator._rhs(t_test, x_test, p_vec)
    
    # Check output structure
    assert isinstance(xd, np.ndarray), f"RHS should return numpy array, got {type(xd)}"
    assert len(xd) == len(x_test), f"RHS output wrong size: {len(xd)} vs {len(x_test)}"
    assert np.all(np.isfinite(xd)), f"RHS contains non-finite values: {xd}"
    
    # Check kinematic relationship: first 3 elements should equal last 3
    q_dot = xd[:3]
    u_current = x_test[3:]
    np.testing.assert_allclose(q_dot, u_current, rtol=1e-12,
                              err_msg="Kinematic relationship qd = u not satisfied")
    
    print(f"✓ RHS function evaluates correctly")
    print(f"✓ Kinematic constraint qd = u satisfied")
    print(f"  State derivative: {xd}")


def test_simulation_execution(simulator, test_config):
    # Run simulation
    simulator.run(test_config)
    
    # Check simulation completed successfully
    assert simulator.result is not None, "Simulation result is None"
    assert hasattr(simulator.result, 'success'), "Result missing 'success' attribute"
    assert simulator.result.success, f"Simulation failed: {simulator.result.message}"
    
    # Check result structure
    assert hasattr(simulator.result, 't'), "Result missing time vector"
    assert hasattr(simulator.result, 'y'), "Result missing state vector"
    
    t = simulator.result.t
    y = simulator.result.y
    
    # Check dimensions
    expected_timesteps = test_config['sim_parameters']['nb_timesteps']
    assert len(t) == expected_timesteps, f"Wrong number of timesteps: {len(t)} vs {expected_timesteps}"
    assert y.shape[0] == 6, f"Wrong number of states: {y.shape[0]} vs 6"
    assert y.shape[1] == expected_timesteps, f"State array wrong size: {y.shape[1]} vs {expected_timesteps}"
    
    # Check all values are finite
    assert np.all(np.isfinite(t)), "Time vector contains non-finite values"
    assert np.all(np.isfinite(y)), "State vector contains non-finite values"
    
    print(f"✓ Simulation completed successfully")
    print(f"  Duration: {t[-1]:.1f}s, timesteps: {len(t)}")
    print(f"  Final state: {y[:, -1]}")


def test_configuration_handling(simulator):
    # Test with manually specified initial speeds
    config_manual = {
        'p_values': {
            "D": 1.0, "L": 2.0, "m_b": 100.0, 
            "m_r": 10.0, "m_l": 10.0, "tau": 0.0
        },
        'q_initial': {'q1': 0.0, 'q2': 0.0, 'q3': 0.0},
        'initial_speeds': {'u1': 0.1, 'u2': 0.2, 'u3': 0.3},  # All speeds specified
        'sim_parameters': {'t_start': 0.0, 't_end': 1.0, 'nb_timesteps': 10}
    }
    
    simulator.run(config_manual)
    assert simulator.result.success, "Simulation with manual speeds failed"
    
    # Check initial conditions were applied correctly
    y_initial = simulator.result.y[:, 0]
    expected_initial = [0.0, 0.0, 0.0, 0.1, 0.2, 0.3]  # [q1,q2,q3,u1,u2,u3]
    np.testing.assert_allclose(y_initial, expected_initial, rtol=1e-10,
                              err_msg="Initial conditions not applied correctly")
    
    print(f"✓ Configuration with manual initial speeds works")
    
    # Test with degree-to-radian conversion
    config_degrees = config_manual.copy()
    config_degrees['q_initial']['q3'] = 45.0  # degrees
    
    simulator.run(config_degrees)
    assert simulator.result.success, "Simulation with degree input failed"
    
    # Check angle was converted to radians
    q3_initial = simulator.result.y[2, 0]
    expected_radians = np.deg2rad(45.0)
    np.testing.assert_allclose(q3_initial, expected_radians, rtol=1e-10,
                              err_msg="Degree to radian conversion failed")
    
    print(f"✓ Degree to radian conversion works correctly")


def test_simulation_physics_consistency(simulator, zero_torque_config):
    simulator.run(zero_torque_config)
    
    result = simulator.result
    assert result.success, "Zero torque simulation failed"
    
    t = result.t
    y = result.y
    
    # Test 1: Angular velocity should be approximately constant (no applied torque)
    u3_values = y[5, :]  # Angular velocity
    u3_variation = np.std(u3_values) / np.mean(np.abs(u3_values))
    
    assert u3_variation < 1e-6, f"Angular velocity not constant: variation = {u3_variation*100:.4f}%"
    
    print(f"✓ Angular velocity constant for zero torque: variation = {u3_variation*100:.2e}%")
    
    # Test 2: For zero initial linear momentum, CM should not drift significantly
    cm_positions = y[:2, :]  # x, y positions
    max_drift = np.max(np.abs(cm_positions))
    
    assert max_drift < 1e-8, f"Unexpected CM drift: {max_drift:.2e}"
    
    print(f"✓ Center of mass remains stationary: max drift = {max_drift:.2e}")
    
    # Test 3: Total angle should increase linearly with time for constant angular velocity
    q3_values = y[2, :]
    u3_mean = np.mean(u3_values)
    
    # Expected final angle: omega * time
    expected_final_angle = u3_mean * t[-1]
    actual_final_angle = q3_values[-1]
    
    angle_error = abs(actual_final_angle - expected_final_angle)
    assert angle_error < 1e-6, f"Angle integration error: {angle_error:.2e} rad"
    
    print(f"✓ Angle integrates correctly: error = {angle_error:.2e} rad")
