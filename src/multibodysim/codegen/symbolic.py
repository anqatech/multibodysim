from __future__ import annotations

import re

import numpy as np
import sympy as sm


def safe_scalar_name(symbol, index: int) -> str:
    raw_name = getattr(symbol, "name", None) or str(symbol)
    clean_name = re.sub(r"\W+", "_", raw_name).strip("_")
    if not clean_name or clean_name[0].isdigit():
        clean_name = f"arg_{clean_name}"
    return f"aw_{index}_{clean_name}"


def symbolic_eval_differentials_data(dyn) -> dict:
    torques = list(dyn.bus_torque_symbols.values())
    scalar_args, replacements = scalar_args_and_replacements(dyn, torques)

    mass_matrix, forcing = dyn._with_specialised_parameters(
        (dyn.mass_matrix, dyn.forcing),
    )
    mass_matrix = sm.Matrix(mass_matrix).xreplace(replacements)
    forcing = sm.Matrix(forcing).xreplace(replacements)

    mass_rows, mass_cols = mass_matrix.shape
    forcing_rows, forcing_cols = forcing.shape

    flat_mass = [
        mass_matrix[row, col]
        for row in range(mass_rows)
        for col in range(mass_cols)
    ]
    flat_forcing = [
        forcing[row, col]
        for row in range(forcing_rows)
        for col in range(forcing_cols)
    ]

    return {
        "scalar_args": scalar_args,
        "mass_shape": (mass_rows, mass_cols),
        "forcing_shape": (forcing_rows, forcing_cols),
        "flat_outputs": [*flat_mass, *flat_forcing],
    }


def symbolic_eval_kinematics_data(dyn) -> dict:
    torques = list(dyn.bus_torque_symbols.values())
    scalar_args, replacements = scalar_args_and_replacements(dyn, torques)

    kinematic_matrix, kinematic_forcing = dyn._with_specialised_parameters(
        (dyn.Mk, dyn.gk),
    )
    kinematic_matrix = sm.Matrix(kinematic_matrix).xreplace(replacements)
    kinematic_forcing = sm.Matrix(kinematic_forcing).xreplace(replacements)

    matrix_rows, matrix_cols = kinematic_matrix.shape
    forcing_rows, forcing_cols = kinematic_forcing.shape

    flat_matrix = [
        kinematic_matrix[row, col]
        for row in range(matrix_rows)
        for col in range(matrix_cols)
    ]
    flat_forcing = [
        kinematic_forcing[row, col]
        for row in range(forcing_rows)
        for col in range(forcing_cols)
    ]

    return {
        "scalar_args": scalar_args,
        "matrix_shape": (matrix_rows, matrix_cols),
        "forcing_shape": (forcing_rows, forcing_cols),
        "flat_outputs": [*flat_matrix, *flat_forcing],
    }


def symbolic_eval_gravity_gradient_data(dyn) -> dict:
    scalar_args, replacements = scalar_args_and_replacements(
        dyn,
        [],
        state_symbols=list(dyn.q),
    )

    gravity_gradient = dyn._with_specialised_parameters(
        dyn.gravity_gradient_generalised_forces,
    )
    gravity_gradient = sm.Matrix(gravity_gradient).xreplace(replacements)
    output_rows, output_cols = gravity_gradient.shape

    flat_outputs = [
        gravity_gradient[row, col]
        for row in range(output_rows)
        for col in range(output_cols)
    ]

    return {
        "scalar_args": scalar_args,
        "output_shape": (output_rows, output_cols),
        "flat_outputs": flat_outputs,
    }


def scalar_args_and_replacements(
    dyn,
    torques: list,
    *,
    state_symbols=None,
) -> tuple[list, dict]:
    if state_symbols is None:
        state_symbols = [*dyn.q, *dyn.u]

    original_args = [*state_symbols, *torques]
    scalar_args = [
        sm.Symbol(safe_scalar_name(symbol, index))
        for index, symbol in enumerate(original_args)
    ]
    return scalar_args, dict(zip(original_args, scalar_args))


def wrap_flat_autowrap_function(function, data: dict):
    mass_shape = data["mass_shape"]
    forcing_shape = data["forcing_shape"]
    mass_size = mass_shape[0] * mass_shape[1]

    def eval_differentials_autowrap(q, u, torques):
        values = [
            *np.asarray(q, dtype=float).reshape(-1),
            *np.asarray(u, dtype=float).reshape(-1),
            *np.asarray(torques, dtype=float).reshape(-1),
        ]
        flat_output = np.asarray(function(*values), dtype=float).reshape(-1)
        mass_matrix = flat_output[:mass_size].reshape(mass_shape)
        forcing = flat_output[mass_size:].reshape(forcing_shape)
        return mass_matrix, forcing

    return eval_differentials_autowrap


def wrap_flat_autowrap_kinematics_function(function, data: dict):
    matrix_shape = data["matrix_shape"]
    forcing_shape = data["forcing_shape"]
    matrix_size = matrix_shape[0] * matrix_shape[1]

    def eval_kinematics_autowrap(q, u, torques):
        values = [
            *np.asarray(q, dtype=float).reshape(-1),
            *np.asarray(u, dtype=float).reshape(-1),
            *np.asarray(torques, dtype=float).reshape(-1),
        ]
        flat_output = np.asarray(function(*values), dtype=float).reshape(-1)
        kinematic_matrix = flat_output[:matrix_size].reshape(matrix_shape)
        kinematic_forcing = flat_output[matrix_size:].reshape(forcing_shape)
        return kinematic_matrix, kinematic_forcing

    return eval_kinematics_autowrap


def wrap_flat_autowrap_gravity_gradient_function(function, data: dict):
    output_shape = data["output_shape"]

    def eval_gravity_gradient_autowrap(q):
        values = np.asarray(q, dtype=float).reshape(-1)
        flat_output = np.asarray(function(*values), dtype=float).reshape(-1)
        return flat_output.reshape(output_shape)

    return eval_gravity_gradient_autowrap
