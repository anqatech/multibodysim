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
    original_args = [*dyn.q, *dyn.u, *torques]
    scalar_args = [
        sm.Symbol(safe_scalar_name(symbol, index))
        for index, symbol in enumerate(original_args)
    ]
    replacements = dict(zip(original_args, scalar_args))

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
