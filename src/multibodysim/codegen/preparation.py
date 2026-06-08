from __future__ import annotations

from pathlib import Path
import time

from .autowrap_eval_differentials import (
    generate_autowrap_eval_differentials,
    load_autowrap_eval_differentials,
)
from .autowrap_eval_kinematics import (
    generate_autowrap_eval_kinematics,
    load_autowrap_eval_kinematics,
)
from .autowrap_eval_gravity_gradient import (
    generate_autowrap_eval_gravity_gradient,
    load_autowrap_eval_gravity_gradient,
)


def prepare_autowrap_evaluators(
    dyn,
    *,
    cache_root: Path | None = None,
) -> dict:
    kinematics = _load_or_generate_autowrap_eval_kinematics(
        dyn,
        cache_root=cache_root,
    )
    differentials = _load_or_generate_autowrap_eval_differentials(
        dyn,
        cache_root=cache_root,
    )

    return dyn._install_autowrap_evaluators(
        kinematics=kinematics,
        differentials=differentials,
    )


def prepare_autowrap_gravity_gradient_evaluator(
    dyn,
    *,
    cache_root: Path | None = None,
) -> dict:
    if not dyn.enable_gravity_gradient:
        raise ValueError(
            "Gravity-gradient evaluator preparation requires "
            "enable_gravity_gradient=True."
        )

    load_start = time.perf_counter()
    generated_evaluator = load_autowrap_eval_gravity_gradient(
        dyn,
        cache_root=cache_root,
    )
    load_time_s = time.perf_counter() - load_start
    timing = {"load_time_s": load_time_s}

    if generated_evaluator is None:
        generate_start = time.perf_counter()
        generated_evaluator = generate_autowrap_eval_gravity_gradient(
            dyn,
            cache_root=cache_root,
        )
        timing["generate_time_s"] = time.perf_counter() - generate_start
        timing["source"] = "generated"
    else:
        timing["generate_time_s"] = 0.0
        timing["source"] = "cache"

    if not generated_evaluator["success"]:
        raise RuntimeError(
            "Autowrap eval_gravity_gradient generation failed: "
            f"{generated_evaluator}"
        )

    generated_evaluator = dict(generated_evaluator)
    generated_evaluator["timing"] = timing
    return generated_evaluator


def _load_or_generate_autowrap_eval_kinematics(
    dyn,
    *,
    cache_root: Path | None = None,
) -> dict:
    load_start = time.perf_counter()
    generated_evaluator = load_autowrap_eval_kinematics(
        dyn,
        cache_root=cache_root,
    )
    load_time_s = time.perf_counter() - load_start
    timing = {"load_time_s": load_time_s}

    if generated_evaluator is None:
        generate_start = time.perf_counter()
        generated_evaluator = generate_autowrap_eval_kinematics(
            dyn,
            cache_root=cache_root,
        )
        timing["generate_time_s"] = time.perf_counter() - generate_start
        timing["source"] = "generated"
    else:
        timing["generate_time_s"] = 0.0
        timing["source"] = "cache"

    if not generated_evaluator["success"]:
        raise RuntimeError(
            "Autowrap eval_kinematics generation failed: "
            f"{generated_evaluator}"
        )

    generated_evaluator = dict(generated_evaluator)
    generated_evaluator["timing"] = timing
    return generated_evaluator


def _load_or_generate_autowrap_eval_differentials(
    dyn,
    *,
    cache_root: Path | None = None,
) -> dict:
    load_start = time.perf_counter()
    generated_evaluator = load_autowrap_eval_differentials(
        dyn,
        cache_root=cache_root,
    )
    load_time_s = time.perf_counter() - load_start
    timing = {"load_time_s": load_time_s}

    if generated_evaluator is None:
        generate_start = time.perf_counter()
        generated_evaluator = generate_autowrap_eval_differentials(
            dyn,
            cache_root=cache_root,
        )
        timing["generate_time_s"] = time.perf_counter() - generate_start
        timing["source"] = "generated"
    else:
        timing["generate_time_s"] = 0.0
        timing["source"] = "cache"

    if not generated_evaluator["success"]:
        raise RuntimeError(
            "Autowrap eval_differentials generation failed: "
            f"{generated_evaluator}"
        )

    generated_evaluator = dict(generated_evaluator)
    generated_evaluator["timing"] = timing
    return generated_evaluator
