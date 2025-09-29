import pytest
import numpy as np
import sympy as sm


def test_flexible_dynamics_initialization(flexible_dynamics):
    """Test that flexible dynamics model initializes correctly."""
    # Check generalized coordinates (rigid + flexible)
    assert hasattr(flexible_dynamics, 'q1'), "Missing coordinate q1"
    assert hasattr(flexible_dynamics, 'q2'), "Missing coordinate q2"
    assert hasattr(flexible_dynamics, 'q3'), "Missing coordinate q3"
    assert hasattr(flexible_dynamics, 'eta_r'), "Missing flexible coordinate eta_r"
    assert hasattr(flexible_dynamics, 'eta_l'), "Missing flexible coordinate eta_l"
    
    # Check generalized speeds
    assert hasattr(flexible_dynamics, 'u1'), "Missing speed u1"
    assert hasattr(flexible_dynamics, 'u2'), "Missing speed u2"
    assert hasattr(flexible_dynamics, 'u3'), "Missing speed u3"
    assert hasattr(flexible_dynamics, 'u4'), "Missing speed u4"
    assert hasattr(flexible_dynamics, 'u5'), "Missing speed u5"
    
    # Check physical parameters
    assert hasattr(flexible_dynamics, 'D'), "Missing parameter D"
    assert hasattr(flexible_dynamics, 'L'), "Missing parameter L"
    assert hasattr(flexible_dynamics, 'm_b'), "Missing parameter m_b"
    assert hasattr(flexible_dynamics, 'm_r'), "Missing parameter m_r"
    assert hasattr(flexible_dynamics, 'm_l'), "Missing parameter m_l"
    assert hasattr(flexible_dynamics, 'E_mod'), "Missing parameter E_mod"
    assert hasattr(flexible_dynamics, 'I_area'), "Missing parameter I_area"
    
    # Check reference frames exist
    assert hasattr(flexible_dynamics, 'N'), "Missing inertial frame N"
    assert hasattr(flexible_dynamics, 'B'), "Missing bus frame B"
    assert hasattr(flexible_dynamics, 'C'), "Missing right panel frame C"
    assert hasattr(flexible_dynamics, 'E'), "Missing left panel frame E"
    
    print("✓ Flexible dynamics model initialized with all required components")


def test_beam_mode_shapes(flexible_dynamics, cantilever_beam):
    """Test beam mode shape calculations."""
    # Check that beam instance exists
    assert hasattr(flexible_dynamics, 'beam'), "Missing beam instance"
    
    # Test mode shape evaluation at several points
    s_points = np.array([0.0, flexible_dynamics.beam.L/4, flexible_dynamics.beam.L/2, flexible_dynamics.beam.L])
    mode_values = flexible_dynamics.beam.mode_shape(s_points)
    
    # Check output structure
    assert isinstance(mode_values, np.ndarray), "Mode shape should return numpy array"
    assert len(mode_values) == len(s_points), f"Mode shape output wrong size: {len(mode_values)} vs {len(s_points)}"
    assert np.all(np.isfinite(mode_values)), "Mode shape contains non-finite values"
    
    # Cantilever boundary conditions: fixed at s=0
    assert np.isclose(mode_values[0], 0.0, atol=1e-10), f"Mode shape should be zero at fixed end: {mode_values[0]}"
    
    # Should have maximum deflection at free end (s=L)
    max_deflection_index = np.argmax(np.abs(mode_values))
    assert max_deflection_index == len(mode_values) - 1, "Maximum deflection should be at free end"
    
    print(f"✓ Beam mode shapes calculated correctly")
    print(f"  Mode values: {mode_values}")


def test_beam_mean_deflection(cantilever_beam):
    """Test beam mean deflection calculation."""
    mean_deflection = cantilever_beam.mode_shape_mean()
    
    # Should be finite and non-zero
    assert np.isfinite(mean_deflection), f"Mean deflection not finite: {mean_deflection}"
    assert mean_deflection != 0.0, "Mean deflection should be non-zero for first mode"
    
    # For cantilever beam first mode, mean should be positive (deflection towards positive y)
    assert mean_deflection > 0, f"Mean deflection should be positive: {mean_deflection}"
    
    print(f"✓ Beam mean deflection calculated: {mean_deflection:.6f}")


def test_modal_stiffness_symbolic(cantilever_beam):
    """Test symbolic modal stiffness calculation."""
    k_modal = cantilever_beam.modal_stiffness_symbolic()
    
    # Should be a SymPy expression
    assert isinstance(k_modal, (sm.Basic, sm.Expr)), f"Modal stiffness should be SymPy expression, got {type(k_modal)}"
    
    # Check that k_modal is not zero/empty
    assert k_modal != 0, "Modal stiffness should be non-zero"
    
    # Evaluate numerically to check magnitude
    k_numeric = float(k_modal.evalf())
    assert k_numeric > 0, f"Modal stiffness should be positive: {k_numeric}"
    assert np.isfinite(k_numeric), f"Modal stiffness not finite: {k_numeric}"
    
    # The beam instance already has numerical values substituted for E and I,
    # so they appear as numbers rather than symbols in the expression
    print(f"✓ Modal stiffness calculated: {k_numeric:.2e} N⋅m²")
    print(f"  Expression contains symbols: {k_modal.free_symbols}")


def test_constraint_equations(flexible_dynamics):
    """Test holonomic constraint equations."""
    # Check constraint exists
    assert hasattr(flexible_dynamics, 'fh'), "Missing constraint equations fh"
    
    # Should be 1x1 matrix (one constraint)
    assert flexible_dynamics.fh.shape == (1, 1), f"Constraint wrong shape: {flexible_dynamics.fh.shape}"
    
    # Constraint should involve panel masses and flexible coordinates
    constraint_symbols = flexible_dynamics.fh[0].free_symbols
    
    # Check that constraint contains mass parameters
    mass_symbols = {flexible_dynamics.m_l, flexible_dynamics.m_r}
    mass_intersection = mass_symbols.intersection(constraint_symbols)
    assert len(mass_intersection) > 0, f"Constraint should contain mass symbols, found: {constraint_symbols}"
    
    # Check that constraint is not trivially zero
    assert flexible_dynamics.fh[0] != 0, "Constraint should be non-trivial"
    
    print("✓ Constraint equations properly formulated")
    print(f"  Constraint symbols: {constraint_symbols}")


def test_eom_matrices_structure(flexible_dynamics):
    """Test that flexible EOM matrices have correct structure."""
    # Check that matrices exist
    assert hasattr(flexible_dynamics, 'Mk'), "Missing kinematic matrix Mk"
    assert hasattr(flexible_dynamics, 'gk'), "Missing kinematic vector gk"
    assert hasattr(flexible_dynamics, 'Md'), "Missing dynamic matrix Md"
    assert hasattr(flexible_dynamics, 'gd'), "Missing dynamic vector gd"
    
    # Check dimensions - flexible system has 5 coordinates, 4 speeds
    assert flexible_dynamics.Mk.shape == (5, 5), f"Mk wrong shape: {flexible_dynamics.Mk.shape}"
    assert flexible_dynamics.gk.shape == (5, 1), f"gk wrong shape: {flexible_dynamics.gk.shape}"
    assert flexible_dynamics.Md.shape == (4, 4), f"Md wrong shape: {flexible_dynamics.Md.shape}"
    assert flexible_dynamics.gd.shape == (4, 1), f"gd wrong shape: {flexible_dynamics.gd.shape}"
    
    print("✓ Flexible EOM matrices have correct dimensions")


def test_parameter_values_extraction(flexible_dynamics, flexible_config):
    """Test parameter value extraction from config."""
    p_vals = flexible_dynamics.get_parameter_values()
    expected_params = flexible_config['p_values']
    
    # Check array length (8 parameters for flexible system)
    assert len(p_vals) == 8, f"Expected 8 parameters, got {len(p_vals)}"
    
    # Check parameter values match config
    expected_order = ['D', 'L', 'm_b', 'm_r', 'm_l', 'E_mod', 'I_area', 'tau']
    for i, param_name in enumerate(expected_order):
        expected_val = expected_params[param_name]
        actual_val = p_vals[i]
        assert np.isclose(actual_val, expected_val), f"Parameter {param_name}: expected {expected_val}, got {actual_val}"
    
    print(f"✓ Parameter values extracted correctly")


def test_initial_conditions(flexible_dynamics, flexible_config):
    """Test initial condition calculation with constraint satisfaction."""
    x0 = flexible_dynamics.get_initial_conditions()
    
    # Check dimensions (5 coordinates + 4 speeds = 9 states)
    assert len(x0) == 9, f"Initial state should have 9 elements, got {len(x0)}"
    
    # Check all values are finite
    assert np.all(np.isfinite(x0)), f"Initial conditions contain non-finite values: {x0}"
    
    # Check position coordinates match config
    q_initial = flexible_config['q_initial']
    assert np.isclose(x0[0], q_initial['q1']), f"q1: expected {q_initial['q1']}, got {x0[0]}"
    assert np.isclose(x0[1], q_initial['q2']), f"q2: expected {q_initial['q2']}, got {x0[1]}"
    assert np.isclose(x0[2], np.deg2rad(q_initial['q3'])), f"q3: expected {np.deg2rad(q_initial['q3'])}, got {x0[2]}"
    assert np.isclose(x0[3], q_initial['eta_r']), f"eta_r: expected {q_initial['eta_r']}, got {x0[3]}"
    
    # eta_l (x0[4]) should satisfy constraint - will be calculated, not from config
    
    print(f"✓ Initial conditions calculated correctly: {x0}")


def test_constraint_satisfaction(flexible_dynamics):
    """Test that initial conditions satisfy constraints."""
    x0 = flexible_dynamics.get_initial_conditions()
    p_vals = flexible_dynamics.get_parameter_values()
    
    # Extract coordinates
    qN = x0[:5]  # All 5 coordinates
    q = qN[:4]   # Independent coordinates  
    qr = qN[4:5] # Dependent coordinate (eta_l)
    
    # Evaluate constraint
    constraint_violation = flexible_dynamics.eval_constraints(qr, q, p_vals)
    
    # Should be nearly zero
    max_violation = np.max(np.abs(constraint_violation))
    assert max_violation < 1e-10, f"Constraint violation too large: {max_violation}"
    
    print(f"✓ Initial conditions satisfy constraints: violation = {max_violation:.2e}")


def test_numerical_evaluation(flexible_dynamics, sample_states, numerical_evaluation_helper):
    """Test numerical evaluation of flexible EOM matrices."""
    flexible_state = sample_states['flexible_state']
    
    # Evaluate dynamics at sample state
    result = numerical_evaluation_helper(flexible_dynamics, flexible_state, None)
    
    assert result['success'], f"Numerical evaluation failed: {result['error']}"
    
    Mk, gk, Md, gd = result['Mk'], result['gk'], result['Md'], result['gd']
    
    # Check return types and shapes
    assert isinstance(Mk, np.ndarray), f"Mk should be numpy array, got {type(Mk)}"
    assert isinstance(Md, np.ndarray), f"Md should be numpy array, got {type(Md)}"
    assert Mk.shape == (5, 5), f"Mk wrong shape: {Mk.shape}"
    assert Md.shape == (4, 4), f"Md wrong shape: {Md.shape}"
    
    # Check all values are finite
    assert np.all(np.isfinite(Mk)), "Mk contains non-finite values"
    assert np.all(np.isfinite(Md)), "Md contains non-finite values"
    assert np.all(np.isfinite(gk)), "gk contains non-finite values"  
    assert np.all(np.isfinite(gd)), "gd contains non-finite values"
    
    # Check matrices are invertible
    det_Mk = np.linalg.det(Mk)
    det_Md = np.linalg.det(Md)
    assert abs(det_Mk) > 1e-10, f"Kinematic matrix nearly singular: det={det_Mk}"
    assert abs(det_Md) > 1e-10, f"Dynamic matrix nearly singular: det={det_Md}"
    
    print("✓ Numerical evaluation produces valid matrices")
    print(f"  Kinematic matrix det: {det_Mk:.2e}")
    print(f"  Dynamic matrix det: {det_Md:.2e}")


def test_flexible_simulator_initialization(flexible_simulator):
    """Test that flexible simulator initializes correctly."""
    # Check simulator has necessary components
    assert hasattr(flexible_simulator, 'config'), "Missing config attribute"
    assert hasattr(flexible_simulator, 'dynamics'), "Missing dynamics attribute"
    assert hasattr(flexible_simulator, 'p_vals'), "Missing parameter values"
    
    # Check parameter values
    assert isinstance(flexible_simulator.p_vals, np.ndarray), "Parameter values should be numpy array"
    assert len(flexible_simulator.p_vals) == 8, f"Expected 8 parameters, got {len(flexible_simulator.p_vals)}"
    
    # Check dynamics is correct type
    from multibodysim.flexible.flexible_symbolic_model import FlexibleSymbolicDynamics
    assert isinstance(flexible_simulator.dynamics, FlexibleSymbolicDynamics), "Wrong dynamics type"
    
    print("✓ Flexible simulator initialized correctly")


def test_simulator_rhs_evaluation(flexible_simulator, sample_states):
    """Test right-hand side function evaluation for flexible system."""
    flexible_state = sample_states['flexible_state']
    t = sample_states['time']
    
    # Evaluate RHS
    xdot = flexible_simulator.eval_rhs(t, flexible_state)
    
    # Check output structure
    assert isinstance(xdot, np.ndarray), f"RHS should return numpy array, got {type(xdot)}"
    assert len(xdot) == len(flexible_state), f"RHS output wrong size: {len(xdot)} vs {len(flexible_state)}"
    assert np.all(np.isfinite(xdot)), f"RHS contains non-finite values: {xdot}"
    
    print(f"✓ RHS evaluation successful for flexible system: {xdot}")


def test_short_simulation(flexible_simulator, quick_simulation_helper):
    """Test running a short flexible simulation."""
    # Run quick simulation (shorter than default due to complexity)
    result = quick_simulation_helper(flexible_simulator, duration=0.1, timesteps=10)
    
    assert result['success'], f"Simulation failed: {result['error']}"
    
    t = result['time']
    states = result['states']
    
    # Check output structure
    assert len(t) == 10, f"Expected 10 time points, got {len(t)}"
    assert states.shape == (10, 9), f"Expected (10,9) state array, got {states.shape}"
    
    # Check all values are finite
    assert np.all(np.isfinite(t)), "Time vector contains non-finite values"
    assert np.all(np.isfinite(states)), "State vector contains non-finite values"
    
    # Check simulation progresses in time
    assert np.all(np.diff(t) > 0), "Time should be monotonically increasing"
    assert np.isclose(t[-1], 0.1, rtol=1e-10), f"Final time should be 0.1, got {t[-1]}"
    
    print(f"✓ Short flexible simulation completed successfully")
    print(f"  Final rigid coordinates: {states[-1, :3]}")
    print(f"  Final flexible coordinates: {states[-1, 3:5]}")


def test_flexible_modal_response(flexible_simulator, flexible_config):
    """Test that flexible modes respond correctly to initial conditions."""
    # Create config with initial modal excitation
    config_modal = flexible_config.copy()
    config_modal['q_initial']['eta_r'] = 0.01  # Small initial deflection
    config_modal['initial_speeds']['u4'] = 0.1  # Modal velocity
    config_modal['p_values']['tau'] = 0.0       # No external torque
    config_modal['sim_parameters']['t_end'] = 0.5
    config_modal['sim_parameters']['nb_timesteps'] = 50
    
    # Create new simulator with modal excitation
    from multibodysim.flexible.flexible_simulator import FlexibleSimulator
    sim_modal = FlexibleSimulator(config_modal)
    
    # Run simulation
    results = sim_modal.run_simulation()
    
    assert results['success'], f"Modal simulation failed: {results['message']}"
    
    # Check that flexible coordinates show oscillatory behavior
    eta_r_values = results['eta_r']
    
    # Should have some variation (oscillation)
    eta_r_variation = np.std(eta_r_values)
    assert eta_r_variation > 1e-6, f"Flexible coordinate shows no variation: std = {eta_r_variation}"
    
    # Check constraint is maintained throughout simulation
    max_constraint_violation = results.get('max_constraint_violation', 0)
    assert max_constraint_violation < 1e-4, f"Constraint violation too large: {max_constraint_violation}"
    
    print(f"✓ Flexible modal response simulation successful")
    print(f"  Modal coordinate variation: {eta_r_variation:.2e}")
    print(f"  Max constraint violation: {max_constraint_violation:.2e}")


def test_flexible_vs_rigid_comparison(flexible_simulator, flexible_config):
    """Test that flexible system reduces to rigid behavior with high stiffness."""
    # Create very stiff system (high E_mod)
    config_stiff = flexible_config.copy()
    config_stiff['p_values']['E_mod'] *= 1e6  # Very high stiffness
    config_stiff['q_initial']['eta_r'] = 0.0  # No initial deflection
    config_stiff['q_initial']['eta_l'] = 0.0
    config_stiff['initial_speeds']['u4'] = 0.0  # No modal velocity
    config_stiff['sim_parameters']['t_end'] = 0.2
    config_stiff['sim_parameters']['nb_timesteps'] = 20
    
    from multibodysim.flexible.flexible_simulator import FlexibleSimulator
    sim_stiff = FlexibleSimulator(config_stiff)
    
    results = sim_stiff.run_simulation()
    assert results['success'], f"Stiff simulation failed: {results['message']}"
    
    # Check that flexible coordinates remain small (rigid-like behavior)
    eta_r_values = results['eta_r']
    eta_l_values = results['eta_l']
    
    max_eta_r = np.max(np.abs(eta_r_values))
    max_eta_l = np.max(np.abs(eta_l_values))
    
    assert max_eta_r < 1e-6, f"Right panel deflection too large for stiff system: {max_eta_r}"
    assert max_eta_l < 1e-6, f"Left panel deflection too large for stiff system: {max_eta_l}"
    
    print(f"✓ High stiffness system behaves rigidly")
    print(f"  Max eta_r: {max_eta_r:.2e}, Max eta_l: {max_eta_l:.2e}")
