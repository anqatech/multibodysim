from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from ..codegen import prepare_autowrap_gravity_gradient_evaluator
from ..references import (
    MultiAngleReferenceBuilder,
    MultiAngleReferenceState,
)


@dataclass(frozen=True)
class GravityGradientCompensationResult:
    q: np.ndarray
    u: np.ndarray
    torque_weights: np.ndarray
    central_speed_index: int
    mass_matrix: np.ndarray
    gravity_gradient_generalised_forces: np.ndarray
    control_generalised_force_direction: np.ndarray
    gravity_gradient_acceleration: np.ndarray
    unit_control_acceleration: np.ndarray
    central_gravity_gradient_acceleration: float
    control_effectiveness: float
    effective_attitude_inertia: float
    equivalent_gravity_gradient_torque: float
    cancellation_torque: float
    central_cancellation_residual_acceleration: float


@dataclass(frozen=True)
class ReferenceGravityGradientCompensationResult:
    time: float
    reference_state: MultiAngleReferenceState
    compensation: GravityGradientCompensationResult

    @property
    def reference_torque(self) -> float:
        return self.compensation.equivalent_gravity_gradient_torque

    @property
    def feedforward_torque(self) -> float:
        return self.compensation.cancellation_torque

    @property
    def effective_attitude_inertia(self) -> float:
        return self.compensation.effective_attitude_inertia

    @property
    def control_effectiveness(self) -> float:
        return self.compensation.control_effectiveness

    @property
    def central_cancellation_residual_acceleration(self) -> float:
        return self.compensation.central_cancellation_residual_acceleration


@dataclass(frozen=True)
class GravityGradientStiffnessEstimate:
    delta_theta: float
    minus_compensation: GravityGradientCompensationResult
    plus_compensation: GravityGradientCompensationResult
    stiffness: float

    @property
    def minus_torque(self) -> float:
        return (
            self.minus_compensation.equivalent_gravity_gradient_torque
        )

    @property
    def plus_torque(self) -> float:
        return self.plus_compensation.equivalent_gravity_gradient_torque


@dataclass(frozen=True)
class ReferenceGravityGradientStiffnessResult:
    time: float
    reference_compensation: ReferenceGravityGradientCompensationResult
    estimates: tuple[GravityGradientStiffnessEstimate, ...]
    preferred_delta_theta: float

    @property
    def stiffness(self) -> float:
        for estimate in self.estimates:
            if estimate.delta_theta == self.preferred_delta_theta:
                return estimate.stiffness
        raise RuntimeError(
            "The preferred stiffness estimate is not available."
        )


class GravityGradientCompensator:
    """Evaluate gravity-gradient loading in a scalar control channel."""

    def __init__(
        self,
        dynamics: Any,
        plant_view: Any,
        torque_weights: np.ndarray,
        *,
        cache_root: Path | None = None,
        prepared_evaluator: dict | None = None,
    ):
        self.dynamics = dynamics
        self.plant_view = plant_view
        self.torque_weights = self._normalise_torque_weights(torque_weights)
        self.central_speed_index = self._central_speed_index()
        self.prepared_evaluator = self._prepare_evaluator(
            cache_root=cache_root,
            prepared_evaluator=prepared_evaluator,
        )

    @property
    def evaluator_metadata(self):
        return self.prepared_evaluator.get("metadata")

    @property
    def evaluator_timing(self):
        return self.prepared_evaluator.get("timing")

    @property
    def evaluator_artifact_dir(self):
        return self.prepared_evaluator.get("artifact_dir")

    def evaluate(
        self,
        q: np.ndarray,
        u: np.ndarray,
    ) -> GravityGradientCompensationResult:
        q_values = self._state_vector(q, "q")
        u_values = self._state_vector(u, "u")

        gravity_gradient_forces = self._column_vector(
            self.prepared_evaluator["function"](q_values),
            "gravity-gradient evaluator output",
        )

        zero_torques = np.zeros(
            len(self.dynamics.rigid_body_names),
            dtype=float,
        )
        mass_matrix, zero_torque_forcing = self._evaluate_differentials(
            q_values,
            u_values,
            zero_torques,
        )
        unit_mass_matrix, unit_control_forcing = (
            self._evaluate_differentials(
                q_values,
                u_values,
                self.torque_weights,
            )
        )

        if not np.allclose(
            mass_matrix,
            unit_mass_matrix,
            rtol=1e-12,
            atol=1e-12,
        ):
            raise RuntimeError(
                "The differential mass matrix changed with applied torque; "
                "the numerical forcing difference cannot be interpreted "
                "as B(q)."
            )

        control_direction = unit_control_forcing - zero_torque_forcing
        gravity_gradient_acceleration = np.linalg.solve(
            mass_matrix,
            gravity_gradient_forces,
        )
        unit_control_acceleration = np.linalg.solve(
            mass_matrix,
            control_direction,
        )

        central_gg_acceleration = float(
            gravity_gradient_acceleration[self.central_speed_index, 0]
        )
        control_effectiveness = float(
            unit_control_acceleration[self.central_speed_index, 0]
        )
        if abs(control_effectiveness) <= np.finfo(float).eps:
            raise ValueError(
                "The selected torque weights have zero central-attitude "
                "control effectiveness at this state."
            )

        effective_attitude_inertia = 1.0 / control_effectiveness
        equivalent_gg_torque = (
            central_gg_acceleration / control_effectiveness
        )
        cancellation_torque = -equivalent_gg_torque
        cancellation_acceleration = np.linalg.solve(
            mass_matrix,
            gravity_gradient_forces
            + cancellation_torque * control_direction,
        )
        central_cancellation_residual = float(
            cancellation_acceleration[self.central_speed_index, 0]
        )

        return GravityGradientCompensationResult(
            q=q_values.copy(),
            u=u_values.copy(),
            torque_weights=self.torque_weights.copy(),
            central_speed_index=self.central_speed_index,
            mass_matrix=mass_matrix,
            gravity_gradient_generalised_forces=gravity_gradient_forces,
            control_generalised_force_direction=control_direction,
            gravity_gradient_acceleration=gravity_gradient_acceleration,
            unit_control_acceleration=unit_control_acceleration,
            central_gravity_gradient_acceleration=central_gg_acceleration,
            control_effectiveness=control_effectiveness,
            effective_attitude_inertia=effective_attitude_inertia,
            equivalent_gravity_gradient_torque=equivalent_gg_torque,
            cancellation_torque=cancellation_torque,
            central_cancellation_residual_acceleration=(
                central_cancellation_residual
            ),
        )

    def _prepare_evaluator(
        self,
        *,
        cache_root: Path | None,
        prepared_evaluator: dict | None,
    ) -> dict:
        if not getattr(self.dynamics, "enable_gravity_gradient", False):
            raise ValueError(
                "GravityGradientCompensator requires "
                "enable_gravity_gradient=True."
            )
        if getattr(self.dynamics, "_eval_differentials", None) is None:
            raise RuntimeError(
                "The differential evaluator must be prepared before creating "
                "GravityGradientCompensator."
            )

        if prepared_evaluator is None:
            prepared_evaluator = (
                prepare_autowrap_gravity_gradient_evaluator(
                    self.dynamics,
                    cache_root=cache_root,
                )
            )
        if not isinstance(prepared_evaluator, dict):
            raise TypeError(
                "prepared_evaluator must be a prepared evaluator dictionary."
            )
        if not callable(prepared_evaluator.get("function")):
            raise ValueError(
                "prepared_evaluator must contain a callable 'function'."
            )
        return prepared_evaluator

    def _normalise_torque_weights(
        self,
        torque_weights: np.ndarray,
    ) -> np.ndarray:
        weights = np.asarray(torque_weights, dtype=float).reshape(-1)
        expected_size = len(self.dynamics.rigid_body_names)
        if weights.size != expected_size:
            raise ValueError(
                f"torque_weights must contain {expected_size} values; "
                f"got {weights.size}."
            )
        if not np.all(np.isfinite(weights)):
            raise ValueError(
                "torque_weights must contain only finite values."
            )
        if not np.any(weights):
            raise ValueError(
                "torque_weights must define a non-zero control direction."
            )
        return weights.copy()

    def _central_speed_index(self) -> int:
        index = int(self.plant_view.i_theta_u)
        if not 0 <= index < self.dynamics.state_dimension:
            raise ValueError(
                "The plant view has an invalid central-speed index."
            )
        return index

    def _state_vector(
        self,
        values: np.ndarray,
        name: str,
    ) -> np.ndarray:
        vector = np.asarray(values, dtype=float).reshape(-1)
        expected_size = self.dynamics.state_dimension
        if vector.size != expected_size:
            raise ValueError(
                f"{name} must contain {expected_size} values; "
                f"got {vector.size}."
            )
        if not np.all(np.isfinite(vector)):
            raise ValueError(f"{name} must contain only finite values.")
        return vector

    def _evaluate_differentials(self, q, u, torques):
        mass_matrix, forcing = self.dynamics._eval_differentials(
            q,
            u,
            torques,
        )
        state_dimension = self.dynamics.state_dimension
        mass_matrix = np.asarray(mass_matrix, dtype=float)
        if mass_matrix.shape != (state_dimension, state_dimension):
            raise ValueError(
                "The differential evaluator returned mass matrix shape "
                f"{mass_matrix.shape}; expected "
                f"({state_dimension}, {state_dimension})."
            )
        forcing = self._column_vector(
            forcing,
            "differential forcing",
        )
        return mass_matrix, forcing

    def _column_vector(self, values, name: str) -> np.ndarray:
        vector = np.asarray(values, dtype=float)
        expected_size = self.dynamics.state_dimension
        if vector.size != expected_size:
            raise ValueError(
                f"{name} must contain {expected_size} values; "
                f"got {vector.size}."
            )
        vector = vector.reshape(expected_size, 1)
        if not np.all(np.isfinite(vector)):
            raise ValueError(f"{name} must contain only finite values.")
        return vector


class ReferenceGravityGradientCompensator:
    """Evaluate gravity-gradient feedforward on a nominal reference state."""

    DEFAULT_STIFFNESS_PERTURBATIONS = (1e-5, 1e-4, 1e-3)
    DEFAULT_PREFERRED_PERTURBATION = 1e-4

    def __init__(
        self,
        reference_builder: MultiAngleReferenceBuilder,
        compensator: GravityGradientCompensator,
    ):
        if not isinstance(reference_builder, MultiAngleReferenceBuilder):
            raise TypeError(
                "reference_builder must be a MultiAngleReferenceBuilder."
            )
        if not isinstance(compensator, GravityGradientCompensator):
            raise TypeError(
                "compensator must be a GravityGradientCompensator."
            )
        if reference_builder.dynamics is not compensator.dynamics:
            raise ValueError(
                "reference_builder and compensator must use the same "
                "dynamics object."
            )

        self.reference_builder = reference_builder
        self.compensator = compensator

    def evaluate(
        self,
        t: float,
    ) -> ReferenceGravityGradientCompensationResult:
        reference_state = self.reference_builder.evaluate(t)
        compensation = self.compensator.evaluate(
            reference_state.q,
            reference_state.u,
        )
        return ReferenceGravityGradientCompensationResult(
            time=float(t),
            reference_state=reference_state,
            compensation=compensation,
        )

    def evaluate_stiffness(
        self,
        t: float,
        *,
        perturbations: tuple[float, ...] | None = None,
        preferred_delta_theta: float = DEFAULT_PREFERRED_PERTURBATION,
    ) -> ReferenceGravityGradientStiffnessResult:
        deltas = self._normalise_perturbations(perturbations)
        preferred_delta = float(preferred_delta_theta)
        if preferred_delta not in deltas:
            raise ValueError(
                "preferred_delta_theta must be included in perturbations."
            )

        reference_compensation = self.evaluate(t)
        reference_state = reference_compensation.reference_state
        estimates = []

        for delta_theta in deltas:
            minus_compensation = self._evaluate_perturbed_reference(
                reference_state,
                -delta_theta,
            )
            plus_compensation = self._evaluate_perturbed_reference(
                reference_state,
                delta_theta,
            )
            stiffness = (
                plus_compensation.equivalent_gravity_gradient_torque
                - minus_compensation.equivalent_gravity_gradient_torque
            ) / (2.0 * delta_theta)
            estimates.append(
                GravityGradientStiffnessEstimate(
                    delta_theta=delta_theta,
                    minus_compensation=minus_compensation,
                    plus_compensation=plus_compensation,
                    stiffness=float(stiffness),
                )
            )

        return ReferenceGravityGradientStiffnessResult(
            time=float(t),
            reference_compensation=reference_compensation,
            estimates=tuple(estimates),
            preferred_delta_theta=preferred_delta,
        )

    def _evaluate_perturbed_reference(
        self,
        reference_state: MultiAngleReferenceState,
        theta_offset: float,
    ) -> GravityGradientCompensationResult:
        q_perturbed = reference_state.q.copy()
        u_perturbed = reference_state.u.copy()
        central_index = self.reference_builder.central_angle_index
        q_perturbed[central_index] += theta_offset

        mapped = self.reference_builder.coordinate_mapper.map(
            q_perturbed,
            u_perturbed,
            centre_of_mass_position=(
                reference_state.centre_of_mass.position
            ),
            centre_of_mass_velocity=(
                reference_state.centre_of_mass.velocity
            ),
        )
        return self.compensator.evaluate(mapped.q, mapped.u)

    def _normalise_perturbations(
        self,
        perturbations: tuple[float, ...] | None,
    ) -> tuple[float, ...]:
        if perturbations is None:
            perturbations = self.DEFAULT_STIFFNESS_PERTURBATIONS

        values = np.asarray(perturbations, dtype=float).reshape(-1)
        if values.size == 0:
            raise ValueError("perturbations must not be empty.")
        if not np.all(np.isfinite(values)):
            raise ValueError(
                "perturbations must contain only finite values."
            )
        if np.any(values <= 0.0):
            raise ValueError("perturbations must be positive.")
        if np.unique(values).size != values.size:
            raise ValueError("perturbations must not contain duplicates.")
        return tuple(float(value) for value in values)
