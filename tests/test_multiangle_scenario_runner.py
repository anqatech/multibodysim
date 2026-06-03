from __future__ import annotations

import copy

import numpy as np

from multibodysim.multiangle import (
    MultiAngleScenario,
    run_scenarios,
)


class FakeDynamics:
    def __init__(self, config):
        self.config = config
        self.eval_kinematics_backend = "numpy"
        self.eval_differentials_backend = "numpy"

    def get_torque_values(self):
        torques = self.config.get("torques", {})
        return [
            torques.get("bus_1", 0.0),
            torques.get("bus_2", 0.0),
            torques.get("bus_3", 0.0),
        ]

    def get_torque_weights(self):
        weights = self.config.get("torque_weights", {})
        return [
            weights.get("bus_1", 0.0),
            weights.get("bus_2", 0.0),
            weights.get("bus_3", 0.0),
        ]


class FakeSimulator:
    def __init__(self, config):
        self.config = config
        self.dynamics = FakeDynamics(config)
        self.parameter_values = np.array([1.0, 2.0], dtype=float)
        self.initial_torque_values = np.array(
            self.dynamics.get_torque_values(),
            dtype=float,
        )
        self.torque_values = self.initial_torque_values.copy()
        self.torque_weights = np.array(
            self.dynamics.get_torque_weights(),
            dtype=float,
        )
        self.controller = None
        self.results = None

    def set_controller(self, controller):
        self.controller = controller

    def reset_torque_values(self):
        self.torque_values = self.initial_torque_values.copy()

    def run_simulation(self, eval_flag=True, verbose=False):
        self.results = {
            "success": True,
            "message": "fake run",
            "nfev": 1,
            "njev": 0,
            "nlu": 0,
            "q_initial_snapshot": copy.deepcopy(self.config["q_initial"]),
            "initial_speeds_snapshot": copy.deepcopy(
                self.config["initial_speeds"],
            ),
            "torques_snapshot": copy.deepcopy(self.config["torques"]),
            "torque_weights_snapshot": copy.deepcopy(
                self.config["torque_weights"],
            ),
            "initial_torque_values_snapshot": (
                self.initial_torque_values.copy()
            ),
            "torque_weights_values_snapshot": self.torque_weights.copy(),
            "controller_snapshot": self.controller,
            "eval_flag": eval_flag,
            "verbose": verbose,
        }
        return self.results


def _base_config():
    return {
        "q_initial": {
            "q_central_angle": 0.1,
            "eta1_1": 0.0,
        },
        "initial_speeds": {
            "u_central_angle": 0.0,
            "zeta1_1": 0.0,
        },
        "torques": {
            "bus_1": 0.0,
            "bus_2": 0.0,
            "bus_3": 0.0,
        },
        "torque_weights": {
            "bus_1": 0.0,
            "bus_2": 1.0,
            "bus_3": 0.0,
        },
    }


def test_run_scenarios_accepts_single_scenario():
    simulator = FakeSimulator(_base_config())

    runs = run_scenarios(
        simulator,
        MultiAngleScenario(
            name="single",
            q_initial={"q_central_angle": 0.2},
            metadata={"candidate": 1},
        ),
    )

    assert len(runs) == 1
    assert runs[0]["scenario"] == "single"
    assert runs[0]["metadata"] == {"candidate": 1}
    assert runs[0]["diagnostic_context"].dynamics is simulator.dynamics
    assert runs[0]["success"]
    assert runs[0]["results"]["q_initial_snapshot"]["q_central_angle"] == 0.2


def test_run_scenarios_accepts_list_and_restores_initial_conditions():
    config = _base_config()
    baseline_q = copy.deepcopy(config["q_initial"])
    baseline_speeds = copy.deepcopy(config["initial_speeds"])
    simulator = FakeSimulator(config)

    runs = run_scenarios(
        simulator,
        [
            MultiAngleScenario(
                name="angle",
                q_initial={"q_central_angle": 0.3},
            ),
            MultiAngleScenario(
                name="speed",
                initial_speeds={"u_central_angle": 0.4},
            ),
        ],
    )

    assert [run["scenario"] for run in runs] == ["angle", "speed"]
    assert runs[0]["results"]["q_initial_snapshot"]["q_central_angle"] == 0.3
    assert (
        runs[1]["results"]["q_initial_snapshot"]["q_central_angle"]
        == baseline_q["q_central_angle"]
    )
    assert (
        runs[1]["results"]["initial_speeds_snapshot"]["u_central_angle"]
        == 0.4
    )
    assert simulator.config["q_initial"] == baseline_q
    assert simulator.config["initial_speeds"] == baseline_speeds


def test_run_scenarios_resets_torque_weights_between_scenarios():
    simulator = FakeSimulator(_base_config())
    baseline_torque_weights = copy.deepcopy(simulator.config["torque_weights"])

    runs = run_scenarios(
        simulator,
        [
            MultiAngleScenario(
                name="left",
                torque_weights={"bus_1": 1.0, "bus_2": 0.0},
            ),
            MultiAngleScenario(name="baseline"),
        ],
    )

    assert runs[0]["results"]["torque_weights_snapshot"]["bus_1"] == 1.0
    assert runs[0]["results"]["torque_weights_snapshot"]["bus_2"] == 0.0
    assert (
        runs[1]["results"]["torque_weights_snapshot"]
        == baseline_torque_weights
    )
    np.testing.assert_allclose(
        simulator.torque_weights,
        np.array([0.0, 1.0, 0.0], dtype=float),
    )


def test_run_scenarios_captures_scenario_torque_values():
    simulator = FakeSimulator(_base_config())

    runs = run_scenarios(
        simulator,
        [
            MultiAngleScenario(
                name="torque-a",
                torques={"bus_1": 1.0, "bus_2": 2.0, "bus_3": 3.0},
            ),
            MultiAngleScenario(
                name="torque-b",
                torques={"bus_1": 4.0, "bus_2": 5.0, "bus_3": 6.0},
            ),
        ],
    )

    np.testing.assert_allclose(
        runs[0]["diagnostic_context"].torque_values,
        [1.0, 2.0, 3.0],
    )
    np.testing.assert_allclose(
        runs[1]["diagnostic_context"].torque_values,
        [4.0, 5.0, 6.0],
    )
    np.testing.assert_allclose(
        simulator.initial_torque_values,
        [0.0, 0.0, 0.0],
    )


def test_run_scenarios_resets_controller_between_scenarios():
    simulator = FakeSimulator(_base_config())
    controller = object()

    runs = run_scenarios(
        simulator,
        [
            MultiAngleScenario(name="controlled", controller=controller),
            MultiAngleScenario(name="uncontrolled"),
        ],
    )

    assert runs[0]["results"]["controller_snapshot"] is controller
    assert runs[1]["results"]["controller_snapshot"] is None
    assert simulator.controller is None
