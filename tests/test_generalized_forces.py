import pytest
import sympy as sm
import numpy as np


def test_generalized_forces_structure(symbolic_model):
    # Check that both force vectors exist
    assert hasattr(symbolic_model, 'Generalised_Active_Forces'), "Missing generalized active forces"
    assert hasattr(symbolic_model, 'Generalised_Inertia_Forces'), "Missing generalized inertia forces"
    
    # Check they're matrices with correct dimensions
    active = symbolic_model.Generalised_Active_Forces
    inertia = symbolic_model.Generalised_Inertia_Forces
    
    assert active.shape == (3, 1), f"Active forces wrong shape: {active.shape}, expected (3, 1)"
    assert inertia.shape == (3, 1), f"Inertia forces wrong shape: {inertia.shape}, expected (3, 1)"
    
    # Check they're not all zero (that would indicate a problem)
    active_nonzero = any(force != 0 for force in active)
    inertia_nonzero = any(force != 0 for force in inertia)
    
    # At least inertia forces should be non-zero (they contain mass terms)
    assert inertia_nonzero, "All inertia forces are zero - this suggests an error"
    
    print(f"✓ Generalized forces have correct structure (3×1 each)")
    print(f"✓ Active forces non-zero: {active_nonzero}, Inertia forces non-zero: {inertia_nonzero}")


def test_zero_torque_active_forces(symbolic_model):
    # Substitute tau = 0 into the active forces
    active_forces_zero_torque = symbolic_model.Generalised_Active_Forces.subs(symbolic_model.tau, 0)
    
    # All should be zero when no torque is applied
    for i in range(3):
        force_component = sm.simplify(active_forces_zero_torque[i])
        assert force_component == 0, f"Active force {i} should be zero when tau=0, got: {force_component}"
    
    print(f"✓ With tau=0, all active forces are zero (correct - no other applied forces)")


def test_torque_coupling_fixed(symbolic_model):
    active = symbolic_model.Generalised_Active_Forces
    
    # Check which generalized forces contain tau
    # We need to check if tau appears in the expression, not just free_symbols
    contains_tau = []
    for i in range(3):
        force_expr = active[i]
        
        # Convert expression to string and check if 'tau' appears
        # This handles both symbols and functions
        force_str = str(force_expr)
        has_tau = 'tau' in force_str
        contains_tau.append(has_tau)
        
        print(f"Force {i}: {force_expr}, contains tau: {has_tau}")
    
    # For your system: tau should only appear in the rotational equation (index 2)
    # q1, q2 are translations, q3 is rotation
    assert not contains_tau[0], f"tau should not affect q1 (x-translation) generalized force"
    assert not contains_tau[1], f"tau should not affect q2 (y-translation) generalized force"  
    assert contains_tau[2], f"tau should affect q3 (rotation) generalized force"
    
    print(f"✓ Applied torque correctly affects only rotational generalized force")
    print(f"  Forces containing tau: q1={contains_tau[0]}, q2={contains_tau[1]}, q3={contains_tau[2]}")

def test_torque_effect_numerical(symbolic_model):
    active = symbolic_model.Generalised_Active_Forces
    
    # Test rotational force (index 2) with different tau values
    rotational_force = active[2]
    
    # Substitute tau = 0 and tau = 1
    force_at_zero = rotational_force.subs(symbolic_model.tau, 0)
    force_at_one = rotational_force.subs(symbolic_model.tau, 1)
    
    # They should be different
    difference = sm.simplify(force_at_one - force_at_zero)
    
    assert difference != 0, f"Torque has no effect on rotational force: difference = {difference}"
    assert difference == 1, f"Expected difference of 1, got {difference}"
    
    # Test translational forces (should be unaffected by tau)
    for i in [0, 1]:  # q1, q2 forces
        trans_force = active[i]
        force_at_zero = trans_force.subs(symbolic_model.tau, 0)
        force_at_one = trans_force.subs(symbolic_model.tau, 1)
        difference = sm.simplify(force_at_one - force_at_zero)
        
        assert difference == 0, f"Translational force {i} affected by torque: difference = {difference}"
    
    print(f"✓ Torque numerically affects only rotational generalized force")
    print(f"✓ Rotational force changes by 1 when tau changes from 0 to 1")

def test_inertia_forces_mass_dependence(symbolic_model):
    inertia = symbolic_model.Generalised_Inertia_Forces
    
    # Collect all symbols that appear in inertia forces
    all_symbols = set()
    for i in range(3):
        all_symbols.update(inertia[i].free_symbols)
    
    # Check that all masses appear
    mass_symbols = {symbolic_model.m_b, symbolic_model.m_r, symbolic_model.m_l}
    missing_masses = mass_symbols - all_symbols
    
    assert len(missing_masses) == 0, f"Masses missing from inertia forces: {missing_masses}"
    
    # Check that geometric parameters also appear (D, L affect inertia)
    geometric_symbols = {symbolic_model.D, symbolic_model.L}
    missing_geometry = geometric_symbols - all_symbols
    
    assert len(missing_geometry) == 0, f"Geometric parameters missing from inertia forces: {missing_geometry}"
    
    print(f"✓ Inertia forces correctly depend on all masses and geometry")
