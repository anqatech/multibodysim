from __future__ import annotations

import hashlib
import json
import platform
from pathlib import Path
from typing import Any

import numpy as np
import sympy as sm

from .constants import BACKEND_NAME, CODEGEN_VERSION


def make_json_serialisable(value: Any):
    if isinstance(value, dict):
        return {
            str(key): make_json_serialisable(value[key])
            for key in sorted(value, key=str)
        }

    if isinstance(value, (list, tuple)):
        return [make_json_serialisable(item) for item in value]

    if isinstance(value, np.generic):
        return value.item()

    if isinstance(value, Path):
        return str(value)

    return value


def cython_version() -> str:
    try:
        import Cython
    except Exception:
        return "not-installed"

    return Cython.__version__


def autowrap_eval_differentials_cache_metadata(dyn) -> dict:
    return _autowrap_evaluator_cache_metadata(
        dyn,
        evaluator_name="eval_differentials",
        output_shapes={
            "mass_matrix_shape": list(sm.Matrix(dyn.mass_matrix).shape),
            "forcing_shape": list(sm.Matrix(dyn.forcing).shape),
        },
    )


def autowrap_eval_kinematics_cache_metadata(dyn) -> dict:
    return _autowrap_evaluator_cache_metadata(
        dyn,
        evaluator_name="eval_kinematics",
        output_shapes={
            "kinematic_matrix_shape": list(sm.Matrix(dyn.Mk).shape),
            "kinematic_forcing_shape": list(sm.Matrix(dyn.gk).shape),
        },
    )


def autowrap_eval_gravity_gradient_cache_metadata(dyn) -> dict:
    return _autowrap_evaluator_cache_metadata(
        dyn,
        evaluator_name="eval_gravity_gradient",
        output_shapes={
            "gravity_gradient_shape": list(
                sm.Matrix(dyn.gravity_gradient_generalised_forces).shape
            ),
        },
    )


def _autowrap_evaluator_cache_metadata(
    dyn,
    *,
    evaluator_name: str,
    output_shapes: dict,
) -> dict:
    flexible_mode_counts = {
        body: len(dyn.flexible_bodies[body]["eta_list"])
        for body in dyn.flexible_body_names
    }

    metadata = {
        "model_kind": "multiangle",
        "backend": BACKEND_NAME,
        "evaluator_name": evaluator_name,
        "codegen_version": CODEGEN_VERSION,
        "state_dimension": dyn.state_dimension,
        "enable_gravity_gradient": bool(dyn.enable_gravity_gradient),
        "body_names": list(dyn.body_names),
        "body_type": dict(dyn.body_type),
        "adjacency_graph": {
            body: list(dyn.graph[body])
            for body in dyn.body_names
        },
        "flexible_mode_counts": flexible_mode_counts,
        "flexible_inertia_integration": dict(dyn.flexible_inertia_integration),
        "parameter_values": dict(dyn.parameter_values),
        "torque_symbol_order": [
            str(symbol)
            for symbol in dyn.bus_torque_symbols.values()
        ],
        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
        "sympy_version": sm.__version__,
        "cython_version": cython_version(),
        "platform_tag": platform.platform(),
    }
    metadata.update(output_shapes)
    metadata["cache_key"] = autowrap_evaluator_cache_key_from_metadata(
        metadata,
    )
    return metadata


def autowrap_evaluator_cache_key_from_metadata(metadata: dict) -> str:
    comparable_metadata = {
        key: value
        for key, value in metadata.items()
        if key != "cache_key"
    }
    payload = json.dumps(
        make_json_serialisable(comparable_metadata),
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def autowrap_eval_differentials_cache_key_from_metadata(metadata: dict) -> str:
    return autowrap_evaluator_cache_key_from_metadata(metadata)


def autowrap_eval_differentials_cache_key(dyn) -> str:
    return autowrap_eval_differentials_cache_metadata(dyn)["cache_key"]


def autowrap_eval_kinematics_cache_key(dyn) -> str:
    return autowrap_eval_kinematics_cache_metadata(dyn)["cache_key"]


def autowrap_eval_gravity_gradient_cache_key(dyn) -> str:
    return autowrap_eval_gravity_gradient_cache_metadata(dyn)["cache_key"]
