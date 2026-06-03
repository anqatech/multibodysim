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


def load_autowrap_eval_differentials(
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

    data = symbolic_eval_differentials_data(dyn)
    raw_function = load_function_from_metadata(artifact_dir, metadata)
    function = wrap_flat_autowrap_function(raw_function, data)

    return {
        "success": True,
        "function": function,
        "metadata": metadata,
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

    result = {
        "success": True,
        "backend": BACKEND_NAME,
        "build_time_s": build_time_s,
        "artifact_dir": artifact_dir,
        "metadata": metadata,
        "function": function,
    }

    module = sys.modules[raw_function.__module__]
    module_file = Path(module.__file__)
    metadata.update(
        {
            "module_name": raw_function.__module__,
            "function_name": raw_function.__name__,
            "module_file": module_file.name,
        }
    )
    write_metadata(artifact_dir, metadata)
    result["metadata"] = metadata

    return result
