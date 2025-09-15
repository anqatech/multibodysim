import pytest
import sympy as sm
import sympy.physics.mechanics as me


def test_reference_frames_exist(symbolic_model):
    # Check all frames exist
    assert hasattr(symbolic_model, 'N'), "Missing inertial frame 'N'"
    assert hasattr(symbolic_model, 'B'), "Missing bus frame 'B'" 
    assert hasattr(symbolic_model, 'C'), "Missing right panel frame 'C'"
    assert hasattr(symbolic_model, 'E'), "Missing left panel frame 'E'"
    
    # Check they are actually ReferenceFrame objects
    assert isinstance(symbolic_model.N, me.ReferenceFrame)
    assert isinstance(symbolic_model.B, me.ReferenceFrame)
    assert isinstance(symbolic_model.C, me.ReferenceFrame)
    assert isinstance(symbolic_model.E, me.ReferenceFrame)
    
    print(f"✓ All 4 reference frames created: N, B, C, E")

def test_frame_orientations_physical(symbolic_model):
    # Test 1: At q3 = 0, frames should be aligned
    # B.x should be parallel to N.x when q3 = 0
    Bx_in_N = symbolic_model.B.x.express(symbolic_model.N)
    Bx_components_at_zero = [
        Bx_in_N.dot(symbolic_model.N.x).subs(symbolic_model.q3, 0),
        Bx_in_N.dot(symbolic_model.N.y).subs(symbolic_model.q3, 0),
        Bx_in_N.dot(symbolic_model.N.z).subs(symbolic_model.q3, 0)
    ]
    
    # At q3=0: B.x should equal N.x (components: [1, 0, 0])
    assert Bx_components_at_zero[0] == 1, f"B.x·N.x at q3=0 should be 1, got {Bx_components_at_zero[0]}"
    assert Bx_components_at_zero[1] == 0, f"B.x·N.y at q3=0 should be 0, got {Bx_components_at_zero[1]}"
    assert Bx_components_at_zero[2] == 0, f"B.x·N.z at q3=0 should be 0, got {Bx_components_at_zero[2]}"
    
    # Test 2: At q3 = π/2, B.x should align with either +N.y or -N.y
    import sympy as sm
    Bx_components_at_90 = [
        Bx_in_N.dot(symbolic_model.N.x).subs(symbolic_model.q3, sm.pi/2),
        Bx_in_N.dot(symbolic_model.N.y).subs(symbolic_model.q3, sm.pi/2),
        Bx_in_N.dot(symbolic_model.N.z).subs(symbolic_model.q3, sm.pi/2)
    ]
    
    # At q3=π/2: B.x should be perpendicular to N.x
    assert sm.simplify(Bx_components_at_90[0]) == 0, f"B.x·N.x at q3=π/2 should be 0"
    # And should align with either +N.y or -N.y (depending on sign convention)
    y_component = sm.simplify(Bx_components_at_90[1])
    assert abs(y_component) == 1, f"B.x·N.y at q3=π/2 should be ±1, got {y_component}"
    
    print(f"✓ Frame orientations work correctly:")
    print(f"  At q3=0: B.x = N.x")
    print(f"  At q3=π/2: B.x·N.y = {y_component}")

def test_left_panel_orientation(symbolic_model):
    # Express both panel x-directions in the inertial frame
    Cx_in_N = symbolic_model.C.x.express(symbolic_model.N)  
    Ex_in_N = symbolic_model.E.x.express(symbolic_model.N)
    
    # They should be opposite: C.x + E.x should equal zero
    sum_vector = Cx_in_N + Ex_in_N
    
    # Check each component sums to zero
    for unit_vec, name in [(symbolic_model.N.x, 'x'), (symbolic_model.N.y, 'y'), (symbolic_model.N.z, 'z')]:
        component_sum = sm.simplify(sum_vector.dot(unit_vec))
        assert component_sum == 0, f"Panel orientations not opposite: {name}-component sum = {component_sum}"
    
    print(f"✓ Left panel E points opposite to right panel C")

def test_center_of_mass_calculation(symbolic_model):
    # Check that r_GB exists and has the right structure
    assert hasattr(symbolic_model, 'r_GB'), "Missing center of mass vector 'r_GB'"
    
    # r_GB should be a Vector (SymPy mechanics vector)
    from sympy.physics.vector import Vector
    assert isinstance(symbolic_model.r_GB, Vector), "r_GB should be a Vector"
    
    # Test symmetric case: if panel masses are equal, r_GB should be zero
    # Substitute equal masses
    r_GB_symmetric = symbolic_model.r_GB.subs(symbolic_model.m_r, symbolic_model.m_l)
    
    # Express in bus frame and check components
    r_GB_components = [
        r_GB_symmetric.dot(symbolic_model.B.x),
        r_GB_symmetric.dot(symbolic_model.B.y)
    ]
    
    # For symmetric case, both components should simplify to zero
    for i, component in enumerate(r_GB_components):
        simplified = sm.simplify(component)
        assert simplified == 0, f"CM calculation wrong: component {i} = {simplified} (should be 0 for symmetric case)"
    
    print(f"✓ Center of mass calculation is physically correct")