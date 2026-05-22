from __future__ import annotations

import sys
import time
from pathlib import Path

import sympy as sm

from .artifacts import (
    autowrap_eval_differentials_artifact_dir,
    load_function_from_metadata,
    read_metadata,
    write_metadata,
)
from .constants import BACKEND_NAME, GENERATED_EVALUATOR_ROOT
from .metadata import autowrap_eval_differentials_cache_metadata
from .symbolic import (
    symbolic_eval_differentials_data,
    wrap_flat_autowrap_function,
)
from .validation import validate_eval_differentials_candidate


def load_validated_autowrap_eval_differentials(
    dyn,
    *,
    cache_root: Path | None = None,
) -> dict | None:
    expected_metadata = autowrap_eval_differentials_cache_metadata(dyn)
    artifact_dir = autowrap_eval_differentials_artifact_dir(
        dyn,
        cache_root=cache_root,
    )
    metadata = read_metadata(artifact_dir)
    if metadata is None:
        return None

    if metadata.get("cache_key") != expected_metadata["cache_key"]:
        return None

    validation_metadata = metadata.get("validation", {})
    if validation_metadata.get("success") is not True:
        return None

    data = symbolic_eval_differentials_data(dyn)
    raw_function = load_function_from_metadata(artifact_dir, metadata)
    function = wrap_flat_autowrap_function(raw_function, data)
    validation = validate_eval_differentials_candidate(dyn, function)
    if not validation["success"]:
        return None

    return {
        "function": function,
        "metadata": metadata,
        "validation": validation,
        "artifact_dir": artifact_dir,
    }


def generate_autowrap_eval_differentials(
    dyn,
    *,
    cache_root: Path | None = None,
) -> dict:
    from sympy.utilities.autowrap import autowrap

    artifact_dir = autowrap_eval_differentials_artifact_dir(
        dyn,
        cache_root=cache_root,
    )
    artifact_dir.mkdir(parents=True, exist_ok=True)

    metadata = autowrap_eval_differentials_cache_metadata(dyn)
    data = symbolic_eval_differentials_data(dyn)
    output_vector = sm.Matrix(data["flat_outputs"])

    build_start = time.perf_counter()
    raw_function = autowrap(
        output_vector,
        args=data["scalar_args"],
        backend="cython",
        language="C",
        tempdir=str(artifact_dir),
    )
    build_time_s = time.perf_counter() - build_start

    function = wrap_flat_autowrap_function(raw_function, data)
    validation = validate_eval_differentials_candidate(dyn, function)

    result = {
        "success": validation["success"],
        "backend": BACKEND_NAME,
        "build_time_s": build_time_s,
        "artifact_dir": artifact_dir,
        "metadata": metadata,
        "validation": validation,
        "function": function if validation["success"] else None,
    }

    if not validation["success"]:
        return result

    module = sys.modules[raw_function.__module__]
    module_file = Path(module.__file__)
    metadata.update(
        {
            "module_name": raw_function.__module__,
            "function_name": raw_function.__name__,
            "module_file": module_file.name,
            "validation": validation,
        }
    )
    write_metadata(artifact_dir, metadata)
    result["metadata"] = metadata

    return result


__all__ = [
    "GENERATED_EVALUATOR_ROOT",
    "generate_autowrap_eval_differentials",
    "load_validated_autowrap_eval_differentials",
]
