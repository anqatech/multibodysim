from __future__ import annotations

import sympy as sm


def lambdify_numpy_reference(args, expression):
    return sm.lambdify(
        args,
        expression,
        "numpy",
        cse=True,
        docstring_limit=0,
    )


def make_numpy_eval_kinematics_reference(dyn):
    torques = list(dyn.bus_torque_symbols.values())
    return lambdify_numpy_reference(
        (
            dyn.q,
            dyn.u,
            torques,
        ),
        dyn._with_specialised_parameters((dyn.Mk, dyn.gk)),
    )


def make_numpy_eval_differentials_reference(dyn):
    torques = list(dyn.bus_torque_symbols.values())
    return lambdify_numpy_reference(
        (
            dyn.q,
            dyn.u,
            torques,
        ),
        dyn._with_specialised_parameters((dyn.mass_matrix, dyn.forcing)),
    )


def make_numpy_eval_gravity_gradient_reference(dyn):
    return lambdify_numpy_reference(
        (dyn.q,),
        dyn._with_specialised_parameters(
            dyn.gravity_gradient_generalised_forces,
        ),
    )
