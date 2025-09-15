import pytest
import numpy as np
from multibodysim import SatelliteSymbolicModel, SatelliteSimulator


@pytest.fixture
def symbolic_model():
    """Fixture providing a fresh symbolic model instance."""
    return SatelliteSymbolicModel()


@pytest.fixture
def simulator(symbolic_model):
    """Fixture providing a simulator instance with compiled symbolic model."""
    return SatelliteSimulator(symbolic_model)


@pytest.fixture
def standard_params():
    """Standard parameter set for testing."""
    return {
        "D": 1.0,      # Bus dimension (m)
        "L": 2.0,      # Panel length (m)
        "m_b": 100.0,  # Bus mass (kg)
        "m_r": 10.0,   # Right panel mass (kg)
        "m_l": 10.0,   # Left panel mass (kg)
        "tau": 0.0     # Applied torque (N⋅m)
    }


@pytest.fixture
def test_config(standard_params):
    """Complete simulation configuration for testing."""
    return {
        'p_values': standard_params,
        'q_initial': {
            'q1': 0.0,    # Initial x position (m)
            'q2': 0.0,    # Initial y position (m)  
            'q3': 0.0     # Initial angle (degrees)
        },
        'initial_speeds': {
            'u3': 0.1     # Initial angular velocity (rad/s)
        },
        'sim_parameters': {
            't_start': 0.0,       # Simulation start time (s)
            't_end': 10.0,        # Simulation end time (s)
            'nb_timesteps': 100   # Simulation number of time steps (-)
        }
    }


@pytest.fixture
def zero_torque_config(test_config):
    """Configuration for free rotation test (no applied torque)."""
    config = test_config.copy()
    config['p_values']['tau'] = 0.0
    return config


@pytest.fixture
def symmetric_satellite_params():
    """Parameters for a symmetric satellite (equal panel masses)."""
    return {
        "D": 1.0,        # Bus dimension (m)
        "L": 2.0,        # Panel length (m)
        "m_b": 100.0,    # Bus mass (kg)
        "m_r": 10.0,     # Right panel mass (kg)
        "m_l": 10.0,     # Left panel mass (kg)
        "tau": 0.0       # Applied torque (N⋅m)
    }