from __future__ import annotations

import numpy as np

from .reference import (
    make_numpy_eval_differentials_reference,
    make_numpy_eval_gravity_gradient_reference,
    make_numpy_eval_kinematics_reference,
)


def validate_eval_differentials_candidate(
    dyn,
    candidate,
    *,
    reference=None,
    tolerance: float = 1e-8,
) -> dict:
    if reference is None:
        reference = make_numpy_eval_differentials_reference(dyn)

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
            reference_forcing.reshape(-1),
        )
        candidate_solve = np.linalg.solve(
            candidate_mass,
            candidate_forcing.reshape(-1),
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


def validate_eval_kinematics_candidate(
    dyn,
    candidate,
    *,
    reference=None,
    tolerance: float = 1e-8,
) -> dict:
    if reference is None:
        reference = make_numpy_eval_kinematics_reference(dyn)

    torques = dyn.get_torque_values()
    initial_state = dyn.get_initial_conditions(verbose=False)
    n = dyn.state_dimension

    samples = [(initial_state[:n], initial_state[n:])]

    q_perturbed = initial_state[:n].copy()
    u_perturbed = initial_state[n:].copy()
    q_perturbed += np.linspace(-1e-7, 1e-7, n)
    u_perturbed += np.linspace(1e-8, -1e-8, n)
    samples.append((q_perturbed, u_perturbed))

    max_matrix_difference = 0.0
    max_forcing_difference = 0.0
    max_solve_difference = 0.0

    for q_values, u_values in samples:
        reference_matrix, reference_forcing = reference(q_values, u_values, torques)
        candidate_matrix, candidate_forcing = candidate(q_values, u_values, torques)

        reference_matrix = np.asarray(reference_matrix, dtype=float)
        reference_forcing = np.asarray(reference_forcing, dtype=float)
        candidate_matrix = np.asarray(candidate_matrix, dtype=float)
        candidate_forcing = np.asarray(candidate_forcing, dtype=float)

        matrix_difference = float(
            np.max(np.abs(reference_matrix - candidate_matrix)),
        )
        forcing_difference = float(
            np.max(np.abs(reference_forcing - candidate_forcing)),
        )

        reference_solve = -np.linalg.solve(
            reference_matrix,
            reference_forcing.reshape(-1),
        )
        candidate_solve = -np.linalg.solve(
            candidate_matrix,
            candidate_forcing.reshape(-1),
        )
        solve_difference = float(
            np.max(np.abs(reference_solve - candidate_solve)),
        )

        max_matrix_difference = max(max_matrix_difference, matrix_difference)
        max_forcing_difference = max(max_forcing_difference, forcing_difference)
        max_solve_difference = max(max_solve_difference, solve_difference)

    success = (
        max_matrix_difference < tolerance
        and max_forcing_difference < tolerance
        and max_solve_difference < tolerance
    )

    return {
        "success": success,
        "tolerance": tolerance,
        "max_matrix_difference": max_matrix_difference,
        "max_forcing_difference": max_forcing_difference,
        "max_solve_difference": max_solve_difference,
    }


def validate_gravity_gradient_candidate(
    dyn,
    candidate,
    *,
    reference=None,
    tolerance: float = 1e-8,
) -> dict:
    if reference is None:
        reference = make_numpy_eval_gravity_gradient_reference(dyn)

    initial_state = dyn.get_initial_conditions(verbose=False)
    n = dyn.state_dimension
    samples = [initial_state[:n]]

    q_perturbed = initial_state[:n].copy()
    q_perturbed += np.linspace(-1e-7, 1e-7, n)
    samples.append(q_perturbed)

    max_absolute_difference = 0.0
    for q_values in samples:
        reference_output = np.asarray(
            reference(q_values),
            dtype=float,
        ).reshape(n, 1)
        candidate_output = np.asarray(
            candidate(q_values),
            dtype=float,
        ).reshape(n, 1)
        max_absolute_difference = max(
            max_absolute_difference,
            float(np.max(np.abs(reference_output - candidate_output))),
        )

    return {
        "success": max_absolute_difference < tolerance,
        "tolerance": tolerance,
        "max_absolute_difference": max_absolute_difference,
    }


def validate_autowrap_gravity_gradient_evaluator(
    dyn,
    *,
    cache_root=None,
    tolerance: float = 1e-8,
) -> dict:
    if not dyn.enable_gravity_gradient:
        raise ValueError(
            "Gravity-gradient evaluator validation requires "
            "enable_gravity_gradient=True."
        )

    from .preparation import prepare_autowrap_gravity_gradient_evaluator

    candidate = prepare_autowrap_gravity_gradient_evaluator(
        dyn,
        cache_root=cache_root,
    )
    validation = validate_gravity_gradient_candidate(
        dyn,
        candidate["function"],
        tolerance=tolerance,
    )

    return {
        "success": validation["success"],
        "tolerance": tolerance,
        "artifact_dir": candidate.get("artifact_dir"),
        "metadata": candidate.get("metadata"),
        "timing": candidate.get("timing"),
        "validation": validation,
    }


def validate_autowrap_evaluators(
    dyn,
    *,
    cache_root=None,
    tolerance: float = 1e-8,
) -> dict:
    kinematics = _autowrap_kinematics_candidate(
        dyn,
        cache_root=cache_root,
    )
    differentials = _autowrap_differentials_candidate(
        dyn,
        cache_root=cache_root,
    )

    kinematics_reference = make_numpy_eval_kinematics_reference(dyn)
    differentials_reference = make_numpy_eval_differentials_reference(dyn)

    kinematics_validation = validate_eval_kinematics_candidate(
        dyn,
        kinematics["function"],
        reference=kinematics_reference,
        tolerance=tolerance,
    )
    differentials_validation = validate_eval_differentials_candidate(
        dyn,
        differentials["function"],
        reference=differentials_reference,
        tolerance=tolerance,
    )

    return {
        "success": (
            kinematics_validation["success"]
            and differentials_validation["success"]
        ),
        "tolerance": tolerance,
        "kinematics": {
            "artifact_dir": kinematics.get("artifact_dir"),
            "metadata": kinematics.get("metadata"),
            "validation": kinematics_validation,
        },
        "differentials": {
            "artifact_dir": differentials.get("artifact_dir"),
            "metadata": differentials.get("metadata"),
            "validation": differentials_validation,
        },
    }


def _autowrap_kinematics_candidate(
    dyn,
    *,
    cache_root=None,
) -> dict:
    if (
        getattr(dyn, "eval_kinematics_backend", None) == "autowrap"
        and callable(getattr(dyn, "_eval_kinematics", None))
    ):
        return {
            "function": dyn._eval_kinematics,
            "metadata": getattr(dyn, "eval_kinematics_generated_metadata", None),
            "artifact_dir": _artifact_dir_from_codegen_metadata(
                dyn,
                "kinematics",
            ),
        }

    from .autowrap_eval_kinematics import (
        generate_autowrap_eval_kinematics,
        load_autowrap_eval_kinematics,
    )

    candidate = load_autowrap_eval_kinematics(dyn, cache_root=cache_root)
    if candidate is not None:
        return candidate

    return generate_autowrap_eval_kinematics(dyn, cache_root=cache_root)


def _autowrap_differentials_candidate(
    dyn,
    *,
    cache_root=None,
) -> dict:
    if (
        getattr(dyn, "eval_differentials_backend", None) == "autowrap"
        and callable(getattr(dyn, "_eval_differentials", None))
    ):
        return {
            "function": dyn._eval_differentials,
            "metadata": getattr(
                dyn,
                "eval_differentials_generated_metadata",
                None,
            ),
            "artifact_dir": _artifact_dir_from_codegen_metadata(
                dyn,
                "differentials",
            ),
        }

    from .autowrap_eval_differentials import (
        generate_autowrap_eval_differentials,
        load_autowrap_eval_differentials,
    )

    candidate = load_autowrap_eval_differentials(dyn, cache_root=cache_root)
    if candidate is not None:
        return candidate

    return generate_autowrap_eval_differentials(dyn, cache_root=cache_root)


def _artifact_dir_from_codegen_metadata(dyn, key: str):
    metadata = getattr(dyn, "autowrap_codegen_metadata", None)
    if metadata is None:
        return None

    evaluator = metadata.get(key)
    if evaluator is None:
        return None

    return evaluator.get("artifact_dir")
