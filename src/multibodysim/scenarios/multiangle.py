from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import time
from typing import Any

import numpy as np

from ..analysis import diagnostic_context_from_simulator
from ..controllers.base import AttitudeController
from ..multiangle.simulator import MultiAngleFlexibleSimulator


@dataclass
class MultiAngleScenario:
    name: str
    q_initial: dict[str, float] | None = None
    initial_speeds: dict[str, float] | None = None
    torque_weights: dict[str, float] | None = None
    controller: AttitudeController | None = None
    metadata: dict[str, Any] | None = None


def run_scenarios(
    simulator: MultiAngleFlexibleSimulator,
    scenarios: MultiAngleScenario | list[MultiAngleScenario],
    *,
    eval_flag: bool = True,
    verbose: bool = False,
) -> list[dict[str, Any]]:
    scenario_list = _as_scenario_list(scenarios)
    baseline = _baseline_from_simulator(simulator)

    run_records = []
    for scenario in scenario_list:
        _apply_scenario(simulator, baseline, scenario)
        diagnostic_context = diagnostic_context_from_simulator(
            simulator,
        )
        start = time.perf_counter()
        results = simulator.run_simulation(
            eval_flag=eval_flag,
            verbose=verbose,
        )
        wall_time = time.perf_counter() - start
        run_records.append(
            {
                "scenario": scenario.name,
                "metadata": deepcopy(scenario.metadata),
                "diagnostic_context": diagnostic_context,
                "results": results,
                "wall_time_s": wall_time,
                "success": results["success"],
                "message": results["message"],
                "nfev": results["nfev"],
                "njev": results["njev"],
                "nlu": results["nlu"],
            }
        )

    _restore_baseline(simulator, baseline)
    return run_records


def _as_scenario_list(
    scenarios: MultiAngleScenario | list[MultiAngleScenario],
) -> list[MultiAngleScenario]:
    if isinstance(scenarios, MultiAngleScenario):
        return [scenarios]

    return list(scenarios)


def _baseline_from_simulator(
    simulator: MultiAngleFlexibleSimulator,
) -> dict[str, Any]:
    return {
        "q_initial": deepcopy(simulator.config.get("q_initial", {})),
        "initial_speeds": deepcopy(simulator.config.get("initial_speeds", {})),
        "torque_weights": deepcopy(simulator.config.get("torque_weights", {})),
        "controller": simulator.controller,
    }


def _apply_scenario(
    simulator: MultiAngleFlexibleSimulator,
    baseline: dict[str, Any],
    scenario: MultiAngleScenario,
) -> None:
    _restore_baseline(simulator, baseline)

    if scenario.q_initial is not None:
        simulator.config["q_initial"].update(deepcopy(scenario.q_initial))

    if scenario.initial_speeds is not None:
        simulator.config["initial_speeds"].update(
            deepcopy(scenario.initial_speeds),
        )

    if scenario.torque_weights is not None:
        simulator.config["torque_weights"].update(
            deepcopy(scenario.torque_weights),
        )
        simulator.torque_weights = np.array(
            simulator.dynamics.get_torque_weights(),
            dtype=float,
        )

    simulator.set_controller(scenario.controller)
    simulator.reset_torque_values()
    simulator.results = None


def _restore_baseline(
    simulator: MultiAngleFlexibleSimulator,
    baseline: dict[str, Any],
) -> None:
    simulator.config["q_initial"] = deepcopy(baseline["q_initial"])
    simulator.config["initial_speeds"] = deepcopy(
        baseline["initial_speeds"],
    )
    simulator.config["torque_weights"] = deepcopy(
        baseline["torque_weights"],
    )
    simulator.torque_weights = np.array(
        simulator.dynamics.get_torque_weights(),
        dtype=float,
    )
    simulator.set_controller(baseline["controller"])
    simulator.reset_torque_values()
    simulator.results = None
