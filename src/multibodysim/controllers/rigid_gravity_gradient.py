from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import sympy as sm

from .control_effectiveness import evaluate_scalar_control_effectiveness
from ..references import MultiAngleCoordinateMapper


@dataclass(frozen=True)
class RigidGravityGradientTorqueResult:
    centre_of_mass_position: np.ndarray
    central_attitude: float
    local_vertical_body: np.ndarray
    rigid_gravity_gradient_torque: float


@dataclass(frozen=True)
class TorqueAllocatedRigidGravityGradientResult:
    rigid_gravity_gradient_torque: float
    allocation_factor: float
    cancellation_torque: float
    estimator_result: RigidGravityGradientTorqueResult


class RigidGravityGradientTorqueEstimator:
    """Estimate planar GG torque from observable rigid-body quantities."""

    def __init__(
        self,
        gravitational_parameter: float,
        nominal_inertia: np.ndarray,
    ):
        self.gravitational_parameter = self._positive_scalar(
            gravitational_parameter,
            "gravitational_parameter",
        )
        self.nominal_inertia = _validated_inertia_tensor(nominal_inertia)

    def evaluate(
        self,
        *,
        centre_of_mass_position,
        central_attitude: float,
    ) -> RigidGravityGradientTorqueResult:
        position = np.asarray(
            centre_of_mass_position,
            dtype=float,
        ).reshape(-1)
        if position.size < 2:
            raise ValueError(
                "centre_of_mass_position must contain at least two values."
            )
        position = position[:2].copy()
        if not np.all(np.isfinite(position)):
            raise ValueError(
                "centre_of_mass_position must contain only finite values."
            )

        attitude = float(central_attitude)
        if not np.isfinite(attitude):
            raise ValueError("central_attitude must be finite.")

        radius = float(np.linalg.norm(position))
        if radius <= 0.0:
            raise ValueError(
                "centre_of_mass_position must have non-zero magnitude."
            )

        local_vertical_inertial = np.array(
            [-position[0] / radius, -position[1] / radius, 0.0],
            dtype=float,
        )
        cosine = np.cos(attitude)
        sine = np.sin(attitude)
        inertial_to_body = np.array(
            [
                [cosine, sine, 0.0],
                [-sine, cosine, 0.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=float,
        )
        local_vertical_body = inertial_to_body @ local_vertical_inertial
        torque = (
            3.0
            * self.gravitational_parameter
            / radius**3
            * np.cross(
                local_vertical_body,
                self.nominal_inertia @ local_vertical_body,
            )[2]
        )

        return RigidGravityGradientTorqueResult(
            centre_of_mass_position=position,
            central_attitude=attitude,
            local_vertical_body=local_vertical_body,
            rigid_gravity_gradient_torque=float(torque),
        )

    @staticmethod
    def _positive_scalar(value, name: str) -> float:
        scalar = float(value)
        if not np.isfinite(scalar) or scalar <= 0.0:
            raise ValueError(f"{name} must be positive and finite.")
        return scalar


class TorqueAllocatedRigidGravityGradientFeedforward:
    """Convert rigid GG torque into a selected scalar control channel."""

    def __init__(
        self,
        estimator: RigidGravityGradientTorqueEstimator,
        torque_weights,
        control_effectiveness: float,
    ):
        if not isinstance(estimator, RigidGravityGradientTorqueEstimator):
            raise TypeError(
                "estimator must be a RigidGravityGradientTorqueEstimator."
            )
        self.estimator = estimator
        self.torque_weights = _finite_nonzero_vector(
            torque_weights,
            "torque_weights",
        )
        self.control_effectiveness = float(control_effectiveness)
        if not np.isfinite(self.control_effectiveness):
            raise ValueError("control_effectiveness must be finite.")
        if abs(self.control_effectiveness) <= np.finfo(float).eps:
            raise ValueError(
                "control_effectiveness must be non-zero."
            )

        self.reciprocal_control_effectiveness = (
            1.0 / self.control_effectiveness
        )
        self.nominal_central_inertia = float(
            estimator.nominal_inertia[2, 2]
        )
        self.allocation_factor = (
            self.reciprocal_control_effectiveness
            / self.nominal_central_inertia
        )

    def evaluate(
        self,
        *,
        centre_of_mass_position,
        central_attitude: float,
    ) -> TorqueAllocatedRigidGravityGradientResult:
        estimator_result = self.estimator.evaluate(
            centre_of_mass_position=centre_of_mass_position,
            central_attitude=central_attitude,
        )
        rigid_torque = estimator_result.rigid_gravity_gradient_torque
        cancellation_torque = -self.allocation_factor * rigid_torque
        return TorqueAllocatedRigidGravityGradientResult(
            rigid_gravity_gradient_torque=rigid_torque,
            allocation_factor=self.allocation_factor,
            cancellation_torque=float(cancellation_torque),
            estimator_result=estimator_result,
        )


def compute_nominal_rigid_inertia(dynamics: Any) -> np.ndarray:
    """Return complete-system nominal inertia in the central-body frame."""
    required_attributes = (
        "body_names",
        "central_body",
        "frames",
        "inertial_position",
        "mass_symbols",
        "r_G",
        "q",
        "_body_inertia_times_vector",
        "_with_specialised_parameters",
    )
    missing = [
        name for name in required_attributes
        if not hasattr(dynamics, name)
    ]
    if missing:
        raise TypeError(
            "dynamics does not expose the multi-angle inertia model: "
            + ", ".join(missing)
        )

    central_frame = dynamics.frames[dynamics.central_body]
    axes = (central_frame.x, central_frame.y, central_frame.z)
    columns = []

    for axis in axes:
        inertia_times_axis = 0 * central_frame.x
        for body in dynamics.body_names:
            offset = dynamics.inertial_position[body] - dynamics.r_G
            inertia_times_axis += dynamics._body_inertia_times_vector(
                body,
                axis,
            )
            inertia_times_axis += dynamics.mass_symbols[body] * (
                offset.dot(offset) * axis
                - offset.dot(axis) * offset
            )
        columns.append(
            inertia_times_axis.express(central_frame).to_matrix(
                central_frame
            )
        )

    expression = sm.Matrix.hstack(*columns)
    expression = dynamics._with_specialised_parameters(expression)
    zero_internal_configuration = {
        coordinate: 0.0
        for coordinate in dynamics.q
    }
    inertia = np.asarray(
        expression.subs(zero_internal_configuration),
        dtype=float,
    )
    return _validated_inertia_tensor(inertia)


def prepare_rigid_gravity_gradient_feedforward(
    simulator: Any,
    torque_weights=None,
) -> TorqueAllocatedRigidGravityGradientFeedforward:
    """Prepare rigid GG feedforward for one configuration and allocation."""
    dynamics = simulator.dynamics
    if not getattr(dynamics, "enable_gravity_gradient", False):
        raise ValueError(
            "Rigid gravity-gradient feedforward requires "
            "enable_gravity_gradient=True."
        )

    if torque_weights is None:
        torque_weights = simulator.torque_weights
    weights = _finite_nonzero_vector(torque_weights, "torque_weights")
    expected_size = len(dynamics.rigid_body_names)
    if weights.size != expected_size:
        raise ValueError(
            f"torque_weights must contain {expected_size} values; "
            f"got {weights.size}."
        )

    initial_state = simulator.setup_initial_conditions(verbose=False)
    state_dimension = dynamics.state_dimension
    q_initial = np.asarray(
        initial_state[:state_dimension],
        dtype=float,
    )
    u_initial = np.asarray(
        initial_state[state_dimension:],
        dtype=float,
    )
    centre_position = np.asarray(
        dynamics.rG_func(q_initial, u_initial),
        dtype=float,
    ).reshape(-1)[:2]
    centre_velocity = np.asarray(
        dynamics.vG_func(q_initial, u_initial),
        dtype=float,
    ).reshape(-1)[:2]

    q_nominal = np.zeros(state_dimension, dtype=float)
    u_nominal = np.zeros(state_dimension, dtype=float)
    central_q_index = list(dynamics.q).index(dynamics.central_angle)
    central_u_index = list(dynamics.u).index(dynamics.central_speed)
    q_nominal[central_q_index] = q_initial[central_q_index]
    u_nominal[central_u_index] = u_initial[central_u_index]
    mapped = MultiAngleCoordinateMapper(dynamics).map(
        q_nominal,
        u_nominal,
        centre_of_mass_position=centre_position,
        centre_of_mass_velocity=centre_velocity,
    )
    mass_matrix, _ = dynamics._eval_differentials(
        mapped.q,
        mapped.u,
        simulator.zero_torque_values,
    )

    effectiveness = evaluate_scalar_control_effectiveness(
        dynamics,
        simulator.plant_view,
        weights,
        mapped.q,
        mapped.u,
        mass_matrix=mass_matrix,
    )
    nominal_inertia = compute_nominal_rigid_inertia(dynamics)
    estimator = RigidGravityGradientTorqueEstimator(
        dynamics.parameter_values["planet_mu"],
        nominal_inertia,
    )
    return TorqueAllocatedRigidGravityGradientFeedforward(
        estimator,
        weights,
        effectiveness.control_effectiveness,
    )


def _validated_inertia_tensor(values) -> np.ndarray:
    inertia = np.asarray(values, dtype=float)
    if inertia.shape != (3, 3):
        raise ValueError(
            f"nominal_inertia must have shape (3, 3); got {inertia.shape}."
        )
    if not np.all(np.isfinite(inertia)):
        raise ValueError("nominal_inertia must contain only finite values.")
    if not np.allclose(inertia, inertia.T, rtol=1e-12, atol=1e-9):
        raise ValueError("nominal_inertia must be symmetric.")
    eigenvalues = np.linalg.eigvalsh(0.5 * (inertia + inertia.T))
    if np.any(eigenvalues <= 0.0):
        raise ValueError("nominal_inertia must be positive definite.")
    return inertia.copy()


def _finite_nonzero_vector(values, name: str) -> np.ndarray:
    vector = np.asarray(values, dtype=float).reshape(-1)
    if not vector.size:
        raise ValueError(f"{name} must not be empty.")
    if not np.all(np.isfinite(vector)):
        raise ValueError(f"{name} must contain only finite values.")
    if not np.any(vector):
        raise ValueError(f"{name} must define a non-zero direction.")
    return vector.copy()
