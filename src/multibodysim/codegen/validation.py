from __future__ import annotations

import numpy as np


def validate_eval_differentials_candidate(
    dyn,
    candidate,
    *,
    tolerance: float = 1e-8,
) -> dict:
    reference = getattr(
        dyn,
        "eval_differentials_reference",
        dyn.eval_differentials,
    )
    torques = dyn.get_torque_values()
    initial_state = dyn.get_initial_conditions(verbose=False)
    n = dyn.state_dimension

    samples = [(initial_state[:n], initial_state[n:])]

    q_perturbed = initial_state[:n].copy()
    u_perturbed = initial_state[n:].copy()
    q_perturbed += np.linspace(-1e-7, 1e-7, n)
    u_perturbed += np.linspace(1e-8, -1e-8, n)
    samples.append((q_perturbed, u_perturbed))

    max_mass_difference = 0.0
    max_forcing_difference = 0.0
    max_solve_difference = 0.0

    for q_values, u_values in samples:
        reference_mass, reference_forcing = reference(q_values, u_values, torques)
        candidate_mass, candidate_forcing = candidate(q_values, u_values, torques)

        reference_mass = np.asarray(reference_mass, dtype=float)
        reference_forcing = np.asarray(reference_forcing, dtype=float)
        candidate_mass = np.asarray(candidate_mass, dtype=float)
        candidate_forcing = np.asarray(candidate_forcing, dtype=float)

        mass_difference = float(
            np.max(np.abs(reference_mass - candidate_mass)),
        )
        forcing_difference = float(
            np.max(np.abs(reference_forcing - candidate_forcing)),
        )

        reference_solve = np.linalg.solve(
            reference_mass,
            reference_forcing.squeeze(),
        )
        candidate_solve = np.linalg.solve(
            candidate_mass,
            candidate_forcing.squeeze(),
        )
        solve_difference = float(
            np.max(np.abs(reference_solve - candidate_solve)),
        )

        max_mass_difference = max(max_mass_difference, mass_difference)
        max_forcing_difference = max(max_forcing_difference, forcing_difference)
        max_solve_difference = max(max_solve_difference, solve_difference)

    success = (
        max_mass_difference < tolerance
        and max_forcing_difference < tolerance
        and max_solve_difference < tolerance
    )

    return {
        "success": success,
        "tolerance": tolerance,
        "max_mass_difference": max_mass_difference,
        "max_forcing_difference": max_forcing_difference,
        "max_solve_difference": max_solve_difference,
    }
