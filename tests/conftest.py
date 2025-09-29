import pytest
import numpy as np

# Import your classes
from multibodysim import (
    RigidSymbolicDynamics,
    RigidSimulator,
    Rigid7PartSymbolicDynamics,
    Rigid7PartSimulator,
    FlexibleSymbolicDynamics,
    FlexibleSimulator,
    CantileverBeam,
    ClampedClampedBeam
)


@pytest.fixture(scope="session")
def rigid_config():
    """Hardcoded rigid satellite configuration."""
    return {
        "p_values": {
            "D": 1.0,
            "L": 3.0,
            "m_r": 30.0,
            "m_l": 2.0,
            "m_b": 3.0,
            "tau": 0.05
        },
        "q_initial": {
            "q1": 2.0,
            "q2": 2.0,
            "q3": 0.0
        },
        "initial_speeds": {
            "u3": 0.1
        },
        "sim_parameters": {
            "t_start": 0.0,
            "t_end": 200.0,
            "nb_timesteps": 100,
            "simulation_type": "rigid",
            "method": "RK45",
            "rtol": 1e-6,
            "atol": 1e-9
        }
    }


@pytest.fixture(scope="session")
def rigid_nbody_config():
    """Hardcoded rigid n-body satellite configuration."""
    return {
        "p_values": {
            "D": 1.0,
            "L": 3.0,
            "m_b1": 3.0,
            "m_b2": 1.0,
            "m_b3": 50.0,
            "m_p1": 2.0,
            "m_p2": 2.0,
            "m_p3": 20.0,
            "m_p4": 20.0,
            "tau": 0.05
        },
        "q_initial": {
            "q1": 2.0,
            "q2": 2.0,
            "q3": 0.0
        },
        "initial_speeds": {
            "u3": 0.1
        },
        "sim_parameters": {
            "t_start": 0.0,
            "t_end": 200.0,
            "nb_timesteps": 100,
            "simulation_type": "rigid",
            "method": "RK45",
            "rtol": 1e-6,
            "atol": 1e-9
        }
    }


@pytest.fixture(scope="session")
def flexible_config():
    """Hardcoded flexible satellite configuration."""
    return {
        "p_values": {
            "D": 1.0,
            "L": 3.0,
            "m_r": 2.0,
            "m_l": 2.0,
            "m_b": 20.0,
            "E_mod": 140e9,
            "I_area": 2.5e-8,
            "tau": 0.05
        },
        "q_initial": {
            "q1": 2.0,
            "q2": 2.0,
            "q3": 0.0,
            "eta_r": 0.0,
            "eta_l": 0.0
        },
        "initial_speeds": {
            "u1": 0.0,
            "u2": 0.0,
            "u3": 0.05,
            "u4": 0.0
        },
        "sim_parameters": {
            "t_start": 0.0,
            "t_end": 200.0,
            "nb_timesteps": 2000,
            "simulation_type": "flexible",
            "method": "Radau",
            "rtol": 1e-6,
            "atol": 1e-9
        },
        "beam_parameters": {
            "beta1": 1.8751040687,
            "sigma1": 0.7340955,
            "n_integration_points": 200
        }
    }


@pytest.fixture
def rigid_dynamics(rigid_config):
    """Create rigid symbolic dynamics instance."""
    return RigidSymbolicDynamics(rigid_config)


@pytest.fixture
def rigid_simulator(rigid_config):
    """Create rigid simulator instance."""
    return RigidSimulator(rigid_config)


@pytest.fixture
def rigid_7part_dynamics(rigid_nbody_config):
    """Create 7-part rigid symbolic dynamics instance."""
    return Rigid7PartSymbolicDynamics(rigid_nbody_config)


@pytest.fixture
def rigid_7part_simulator(rigid_nbody_config):
    """Create 7-part rigid simulator instance."""
    return Rigid7PartSimulator(rigid_nbody_config)


@pytest.fixture
def flexible_dynamics(flexible_config):
    """Create flexible symbolic dynamics instance."""
    return FlexibleSymbolicDynamics(flexible_config)


@pytest.fixture
def flexible_simulator(flexible_config):
    """Create flexible simulator instance."""
    return FlexibleSimulator(flexible_config)


@pytest.fixture
def cantilever_beam(flexible_config):
    """Create cantilever beam instance."""
    params = flexible_config['p_values']
    beam_params = flexible_config['beam_parameters']
    
    import sympy as sm
    s = sm.Symbol('s')
    
    return CantileverBeam(
        length=params['L'],
        E=params['E_mod'],
        I=params['I_area'],
        beta1=beam_params['beta1'],
        sigma1=beam_params['sigma1'],
        s=s
    )


@pytest.fixture
def clamped_beam(flexible_config):
    """Create clamped-clamped beam instance."""
    params = flexible_config['p_values']
    beam_params = flexible_config['beam_parameters']
    
    import sympy as sm
    s = sm.Symbol('s')
    
    return ClampedClampedBeam(
        length=params['L'],
        E=params['E_mod'],
        I=params['I_area'],
        beta1=beam_params['beta1'],
        sigma1=beam_params['sigma1'],
        s=s
    )


@pytest.fixture
def sample_states():
    """Sample state vectors for testing numerical evaluations."""
    return {
        'rigid_state': np.array([1.0, 2.0, 0.1, 0.0, 0.0, 0.05]),  # [q1,q2,q3,u1,u2,u3]
        'flexible_state': np.array([2.0, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.05, 0.0]),  # flexible coords
        'time': 0.0
    }


@pytest.fixture
def numerical_evaluation_helper():
    """Helper function for numerical evaluations."""
    def evaluate_dynamics(dynamics, state, config):
        """Evaluate dynamics at given state."""
        if hasattr(dynamics, 'eval_kinematics'):
            # For rigid dynamics
            if len(state) == 6:  # rigid system
                q = state[:3]
                u = state[3:]
                p_vals = dynamics.get_parameter_values()
                
                try:
                    Mk, gk = dynamics.eval_kinematics(q, u, p_vals)
                    Md, gd = dynamics.eval_differentials(q, u, p_vals)
                    
                    return {
                        'Mk': Mk,
                        'gk': gk,
                        'Md': Md,
                        'gd': gd,
                        'success': True,
                        'error': None
                    }
                except Exception as e:
                    return {
                        'success': False,
                        'error': str(e)
                    }
                    
            # For flexible dynamics
            elif len(state) == 9:  # flexible system
                qN = state[:5]
                u = state[5:]
                p_vals = dynamics.get_parameter_values()
                
                try:
                    Mk, gk = dynamics.eval_kinematics(qN, u, p_vals)
                    Md, gd = dynamics.eval_differentials(qN, u, p_vals)
                    
                    return {
                        'Mk': Mk,
                        'gk': gk, 
                        'Md': Md,
                        'gd': gd,
                        'success': True,
                        'error': None
                    }
                except Exception as e:
                    return {
                        'success': False,
                        'error': str(e)
                    }
        
        return {
            'success': False,
            'error': 'Unsupported dynamics type or state dimension'
        }
    
    return evaluate_dynamics


@pytest.fixture
def quick_simulation_helper():
    """Helper for running quick simulations."""
    def run_quick_sim(simulator, duration=1.0, timesteps=10):
        """Run a short simulation for testing."""
        # Modify config for quick run
        original_config = simulator.config.copy()
        
        # Update simulation parameters
        simulator.config['sim_parameters'].update({
            't_end': duration,
            'nb_timesteps': timesteps
        })
        
        try:
            results = simulator.run_simulation()
            return {
                'success': results['success'],
                'time': results['time'],
                'states': results['states'],
                'error': None
            }
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }
        finally:
            # Restore original config
            simulator.config = original_config
    
    return run_quick_sim


@pytest.fixture
def beam_evaluation_helper():
    """Helper for evaluating beam properties."""
    def evaluate_beam(beam, s_points=None):
        """Evaluate beam mode shapes and properties."""
        if s_points is None:
            s_points = np.linspace(0, beam.L, 50)
        
        try:
            # Evaluate mode shape
            mode_values = beam.mode_shape(s_points)
            
            # Calculate mean deflection
            mean_deflection = beam.mode_shape_mean()
            
            # Get modal stiffness (symbolic)
            modal_stiffness = beam.modal_stiffness_symbolic()
            
            return {
                's_points': s_points,
                'mode_values': mode_values,
                'mean_deflection': mean_deflection,
                'modal_stiffness_expr': modal_stiffness,
                'success': True,
                'error': None
            }
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }
    
    return evaluate_beam
