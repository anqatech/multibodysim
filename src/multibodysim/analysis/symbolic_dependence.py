from __future__ import annotations

from collections.abc import Iterable

import sympy as sm
import sympy.physics.mechanics as me
from sympy.core.function import AppliedUndef


def _as_sympy_expression(expr, frame: me.ReferenceFrame):
    if isinstance(expr, me.Vector):
        return sm.Matrix(expr.to_matrix(frame))

    if isinstance(expr, sm.MatrixBase):
        return expr

    if isinstance(expr, Iterable) and not isinstance(expr, (str, bytes)):
        parts = []
        for item in expr:
            item_expr = _as_sympy_expression(item, frame)
            if isinstance(item_expr, sm.MatrixBase):
                parts.extend(list(item_expr))
            else:
                parts.append(item_expr)

        return sm.Matrix(parts)

    return expr


def symbolic_dependence_report(expr, dynamics, frame: me.ReferenceFrame | None = None):
    """Report which dynamic variables, parameters, and torques an expression uses."""
    if frame is None:
        frame = dynamics.frames["inertial"]

    expr = _as_sympy_expression(expr, frame)

    symbols = expr.free_symbols
    functions = expr.atoms(AppliedUndef)
    derivatives = expr.atoms(sm.Derivative)

    q = set(dynamics.q)
    qd = set(dynamics.qd)
    u = set(dynamics.u)
    ud = set(dynamics.ud)

    parameters = set(dynamics.parameter_symbols.values())
    torques = set(dynamics.bus_torque_symbols.values())

    return {
        "q": sorted(functions & q, key=str),
        "u": sorted(functions & u, key=str),
        "qd": sorted(derivatives & qd, key=str),
        "ud": sorted(derivatives & ud, key=str),
        "parameters": sorted(symbols & parameters, key=str),
        "torques": sorted(symbols & torques, key=str),
        "other_symbols": sorted(symbols - parameters - torques, key=str),
        "other_functions": sorted(functions - q - u, key=str),
        "other_derivatives": sorted(derivatives - qd - ud, key=str),
    }
