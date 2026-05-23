from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import time
from typing import Any

import numpy as np

from ..analysis import diagnostic_context_from_simulator
from ..codegen import (
    generate_autowrap_eval_differentials,
    generate_autowrap_eval_kinematics,
)
from ..controllers.base import AttitudeController
from ..multiangle.simulator import MultiAngleFlexibleSimulator


@dataclass
class MultiAngleScenario:
    name: str
    q_initial: dict[str, float] | None = None
    initial_speeds: dict[str, float] | None = None
    torques: dict[str, float] | None = None
    torque_weights: dict[str, float] | None = None
    controller: AttitudeController | None = None
    metadata: dict[str, Any] | None = None


@dataclass
class PreparedMultiAngleSimulator:
    simulator: MultiAngleFlexibleSimulator
    baseline_q_initial: dict[str, float]
    baseline_initial_speeds: dict[str, float]
    baseline_torques: dict[str, float]
    baseline_torque_weights: dict[str, float]
    baseline_controller: AttitudeController | None
    setup_timing: dict[str, float]
    codegen_metadata: dict[str, Any]


def prepare_autowrapped_simulator(config: dict) -> PreparedMultiAngleSimulator:
    total_start = time.perf_counter()

    simulator_start = time.perf_counter()
    simulator = MultiAngleFlexibleSimulator(config)
    simulator_build_time = time.perf_counter() - simulator_start

    kinematics_start = time.perf_counter()
    kinematics_result = generate_autowrap_eval_kinematics(simulator.dynamics)
    kinematics_time = time.perf_counter() - kinematics_start
    if not kinematics_result["success"]:
        raise RuntimeError(kinematics_result["validation"])

    differentials_start = time.perf_counter()
    differentials_result = generate_autowrap_eval_differentials(
        simulator.dynamics,
    )
    differentials_time = time.perf_counter() - differentials_start
    if not differentials_result["success"]:
        raise RuntimeError(differentials_result["validation"])

    switch_start = time.perf_counter()
    simulator.set_eval_kinematics_backend("autowrap")
    simulator.set_eval_differentials_backend("autowrap")
    backend_switch_time = time.perf_counter() - switch_start

    setup_timing = {
        "simulator_build_time_s": simulator_build_time,
        "autowrap_kinematics_time_s": kinematics_time,
        "autowrap_differentials_time_s": differentials_time,
        "backend_switch_time_s": backend_switch_time,
        "total_setup_time_s": time.perf_counter() - total_start,
    }

    codegen_metadata = {
        "kinematics": kinematics_result,
        "differentials": differentials_result,
    }

    return PreparedMultiAngleSimulator(
        simulator=simulator,
        baseline_q_initial=deepcopy(config.get("q_initial", {})),
        baseline_initial_speeds=deepcopy(config.get("initial_speeds", {})),
        baseline_torques=deepcopy(config.get("torques", {})),
        baseline_torque_weights=deepcopy(config.get("torque_weights", {})),
        baseline_controller=simulator.controller,
        setup_timing=setup_timing,
        codegen_metadata=codegen_metadata,
    )


def run_scenarios(
    prepared: PreparedMultiAngleSimulator,
    scenarios: MultiAngleScenario | list[MultiAngleScenario],
    *,
    eval_flag: bool = True,
    verbose: bool = False,
) -> list[dict[str, Any]]:
    scenario_list = _as_scenario_list(scenarios)

    run_records = []
    for scenario in scenario_list:
        _apply_scenario(prepared, scenario)
        diagnostic_context = diagnostic_context_from_simulator(
            prepared.simulator,
        )
        start = time.perf_counter()
        results = prepared.simulator.run_simulation(
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

    _restore_baseline(prepared)
    return run_records


def _as_scenario_list(
    scenarios: MultiAngleScenario | list[MultiAngleScenario],
) -> list[MultiAngleScenario]:
    if isinstance(scenarios, MultiAngleScenario):
        return [scenarios]

    return list(scenarios)


def _apply_scenario(
    prepared: PreparedMultiAngleSimulator,
    scenario: MultiAngleScenario,
) -> None:
    simulator = prepared.simulator
    _restore_baseline(prepared)

    if scenario.q_initial is not None:
        simulator.config["q_initial"].update(deepcopy(scenario.q_initial))

    if scenario.initial_speeds is not None:
        simulator.config["initial_speeds"].update(
            deepcopy(scenario.initial_speeds),
        )

    if scenario.torques is not None:
        simulator.config["torques"].update(deepcopy(scenario.torques))
        simulator.initial_torque_values = np.array(
            simulator.dynamics.get_torque_values(),
            dtype=float,
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


def _restore_baseline(prepared: PreparedMultiAngleSimulator) -> None:
    simulator = prepared.simulator
    simulator.config["q_initial"] = deepcopy(prepared.baseline_q_initial)
    simulator.config["initial_speeds"] = deepcopy(
        prepared.baseline_initial_speeds,
    )
    simulator.config["torques"] = deepcopy(prepared.baseline_torques)
    simulator.config["torque_weights"] = deepcopy(
        prepared.baseline_torque_weights,
    )
    simulator.initial_torque_values = np.array(
        simulator.dynamics.get_torque_values(),
        dtype=float,
    )
    simulator.torque_weights = np.array(
        simulator.dynamics.get_torque_weights(),
        dtype=float,
    )
    simulator.set_controller(prepared.baseline_controller)
    simulator.reset_torque_values()
    simulator.results = None
