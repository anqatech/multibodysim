import pytest
from multibodysim.symbolic_model import SatelliteSymbolicModel


def test_model_creation_succeeds(symbolic_model):
    # The fixture already created the model, so if we get here, creation worked
    assert symbolic_model is not None
    print(f"✓ Model created successfully: {type(symbolic_model)}")


def test_coordinates_exist(symbolic_model):
    # Check the main coordinate vectors exist
    assert hasattr(symbolic_model, 'q'), "Model missing generalized coordinates 'q'"
    assert hasattr(symbolic_model, 'u'), "Model missing generalized speeds 'u'"
    
    # Check they have the right length (3 coordinates each)
    assert len(symbolic_model.q) == 3, f"Expected 3 coordinates, got {len(symbolic_model.q)}"
    assert len(symbolic_model.u) == 3, f"Expected 3 speeds, got {len(symbolic_model.u)}"
    
    print(f"✓ Coordinates q: {symbolic_model.q}")
    print(f"✓ Speeds u: {symbolic_model.u}")


def test_physical_parameters_exist(symbolic_model):
    # Check geometric parameters
    assert hasattr(symbolic_model, 'D'), "Missing bus dimension parameter 'D'"
    assert hasattr(symbolic_model, 'L'), "Missing panel length parameter 'L'"
    
    # Check mass parameters  
    assert hasattr(symbolic_model, 'm_b'), "Missing bus mass parameter 'm_b'"
    assert hasattr(symbolic_model, 'm_r'), "Missing right panel mass 'm_r'"
    assert hasattr(symbolic_model, 'm_l'), "Missing left panel mass 'm_l'"
    
    # Check applied torque
    assert hasattr(symbolic_model, 'tau'), "Missing applied torque parameter 'tau'"
    
    print(f"✓ Physical parameters defined: D, L, m_b, m_r, m_l, tau")
    
    # Bonus check: parameter vector for lambdification
    assert hasattr(symbolic_model, 'p'), "Missing parameter vector 'p'"
    assert len(symbolic_model.p) == 6, f"Expected 6 parameters, got {len(symbolic_model.p)}"