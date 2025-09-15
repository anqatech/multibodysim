import pytest
import sympy as sm
import numpy as np


def test_eom_matrices_exist_and_structure(symbolic_model):
    # Check all EOM matrices exist
    assert hasattr(symbolic_model, 'Mk'), "Missing kinematic matrix 'Mk'"
    assert hasattr(symbolic_model, 'gk'), "Missing kinematic vector 'gk'"
    assert hasattr(symbolic_model, 'Md'), "Missing dynamic matrix 'Md'"
    assert hasattr(symbolic_model, 'gd'), "Missing dynamic vector 'gd'"
    
    # Check dimensions
    assert symbolic_model.Mk.shape == (3, 3), f"Mk wrong shape: {symbolic_model.Mk.shape}"
    assert symbolic_model.gk.shape == (3, 1), f"gk wrong shape: {symbolic_model.gk.shape}"
    assert symbolic_model.Md.shape == (3, 3), f"Md wrong shape: {symbolic_model.Md.shape}"
    assert symbolic_model.gd.shape == (3, 1), f"gd wrong shape: {symbolic_model.gd.shape}"
    
    print(f"✓ All EOM matrices exist with correct dimensions (3×3 and 3×1)")


def test_kinematic_constraint_matrix(symbolic_model):
    Mk = symbolic_model.Mk
    
    # Should be negative identity matrix
    expected_Mk = -sm.eye(3)
    difference = sm.simplify(Mk - expected_Mk)
    
    assert difference == sm.zeros(3, 3), f"Mk should be -I, got difference: {difference}"
    
    print(f"✓ Kinematic constraint matrix Mk = -I (correct for simple coordinate choice)")


def test_kinematic_constraint_vector(symbolic_model):
    gk = symbolic_model.gk
    
    # gk should be [u1, u2, u3] for the kinematic equations u_i - qd_i = 0
    expected_gk = sm.Matrix([symbolic_model.u1, symbolic_model.u2, symbolic_model.u3])
    difference = sm.simplify(gk - expected_gk)
    
    assert difference == sm.zeros(3, 1), f"gk should be [u1, u2, u3], got difference: {difference}"
    
    print(f"✓ Kinematic constraint vector gk = [u1, u2, u3] (correct)")


def test_mass_matrix_properties(symbolic_model):
    Md = symbolic_model.Md
    
    # Test symmetry
    difference = sm.simplify(Md - Md.T)
    assert difference == sm.zeros(3, 3), f"Mass matrix not symmetric: {difference}"
    
    print(f"✓ Mass matrix Md is symmetric")

def test_dynamic_vector_structure_fixed(symbolic_model):
    """
    Test 16 (Fixed): Check dynamic vector (gd) dependencies.
    
    Updated understanding: gd contains velocity-dependent terms (centrifugal/Coriolis),
    while mass terms appear in the mass matrix Md, not in gd.
    """
    gd = symbolic_model.gd
    Md = symbolic_model.Md
    
    # Collect all symbols in gd and Md
    gd_symbols = set()
    Md_symbols = set()
    
    for i in range(3):
        gd_symbols.update(gd[i].free_symbols)
        for j in range(3):
            Md_symbols.update(Md[i, j].free_symbols)
    
    print(f"gd symbols: {gd_symbols}")
    print(f"Md symbols: {Md_symbols}")
    
    # Check that all masses appear somewhere in the EOM (either Md or gd)
    mass_symbols = {symbolic_model.m_b, symbolic_model.m_r, symbolic_model.m_l}
    combined_symbols = gd_symbols.union(Md_symbols)
    missing_masses = mass_symbols - combined_symbols
    
    assert len(missing_masses) == 0, f"Masses missing from EOM: {missing_masses}"
    
    # For your specific system, expect:
    # - Panel masses (m_r, m_l) in gd due to centrifugal effects
    # - All masses (m_b, m_r, m_l) in Md due to inertia
    
    panel_masses = {symbolic_model.m_r, symbolic_model.m_l}
    panel_in_gd = panel_masses.intersection(gd_symbols)
    
    assert len(panel_in_gd) > 0, f"Expected panel masses in gd for centrifugal terms: {panel_in_gd}"
    
    # Check that bus mass appears in mass matrix
    assert symbolic_model.m_b in Md_symbols, "Bus mass should appear in mass matrix Md"
    
    print(f"✓ All masses appear in EOM (Md + gd combined)")
    print(f"✓ Panel masses in gd: {panel_in_gd} (centrifugal effects)")
    print(f"✓ Bus mass in Md: {symbolic_model.m_b in Md_symbols} (inertia effects)")

def test_mass_matrix_contains_all_masses(symbolic_model):
    Md = symbolic_model.Md
    
    # Collect all symbols in mass matrix
    Md_symbols = set()
    for i in range(3):
        for j in range(3):
            Md_symbols.update(Md[i, j].free_symbols)
    
    # All masses should appear in mass matrix
    mass_symbols = {symbolic_model.m_b, symbolic_model.m_r, symbolic_model.m_l}
    missing_masses = mass_symbols - Md_symbols
    
    assert len(missing_masses) == 0, f"Masses missing from mass matrix: {missing_masses}"
    
    print(f"✓ Mass matrix Md contains all masses: {mass_symbols.intersection(Md_symbols)}")
    
    # Check geometric parameters also appear (they affect inertia)
    geometric_symbols = {symbolic_model.D, symbolic_model.L}
    missing_geometry = geometric_symbols - Md_symbols
    
    assert len(missing_geometry) == 0, f"Geometric parameters missing from mass matrix: {missing_geometry}"
    
    print(f"✓ Mass matrix contains geometric parameters: {geometric_symbols.intersection(Md_symbols)}")

def test_eom_coupling_structure(symbolic_model):
    Md = symbolic_model.Md
    
    # Check if off-diagonal terms exist (indicating coupling)
    has_coupling = False
    for i in range(3):
        for j in range(3):
            if i != j:  # Off-diagonal element
                element = sm.simplify(Md[i, j])
                if element != 0:
                    has_coupling = True
                    print(f"  Coupling: Md[{i},{j}] = {element}")
    
    # For your satellite with offset panels, there should be some coupling
    # (Though the exact structure depends on your specific configuration)
    print(f"✓ Mass matrix coupling detected: {has_coupling}")


def test_eom_determinant_nonzero(symbolic_model, standard_params):
    Md = symbolic_model.Md
    
    # Substitute numerical parameter values
    Md_numeric = Md.subs([
        (symbolic_model.D, standard_params["D"]),
        (symbolic_model.L, standard_params["L"]),
        (symbolic_model.m_b, standard_params["m_b"]),
        (symbolic_model.m_r, standard_params["m_r"]),
        (symbolic_model.m_l, standard_params["m_l"])
    ])
    
    # Compute determinant
    det_Md = sm.simplify(Md_numeric.det())
    
    # Should be non-zero
    assert det_Md != 0, f"Mass matrix determinant is zero: {det_Md}"
    
    # Convert to float to check magnitude
    det_value = float(det_Md.evalf())
    assert abs(det_value) > 1e-10, f"Mass matrix nearly singular: det = {det_value}"
    
    print(f"✓ Mass matrix determinant = {det_value:.2e} (invertible)")


def test_eom_substitution_interface(symbolic_model):
    # Test that matrices can be lambdified (needed for simulation)
    q_test = [0.0, 0.0, 0.0]
    u_test = [0.0, 0.0, 0.1]
    p_test = [1.0, 2.0, 10.0, 10.0, 100.0, 0.0]  # D, L, m_r, m_l, m_b, tau
    
    try:
        # This is what the simulator does
        eval_func = sm.lambdify(
            (symbolic_model.q, symbolic_model.u, symbolic_model.p),
            [symbolic_model.Mk, symbolic_model.gk, symbolic_model.Md, symbolic_model.gd],
            'numpy'
        )
        
        # Try to evaluate
        Mk_val, gk_val, Md_val, gd_val = eval_func(q_test, u_test, p_test)
        
        # Check results are reasonable
        assert isinstance(Mk_val, np.ndarray), "Mk evaluation failed"
        assert isinstance(Md_val, np.ndarray), "Md evaluation failed"
        assert Mk_val.shape == (3, 3), f"Mk shape wrong: {Mk_val.shape}"
        assert Md_val.shape == (3, 3), f"Md shape wrong: {Md_val.shape}"
        
        print(f"✓ EOM matrices can be evaluated numerically")
        print(f"  Sample Md determinant: {np.linalg.det(Md_val):.2e}")
        
    except Exception as e:
        pytest.fail(f"EOM matrices cannot be lambdified/evaluated: {e}")
