from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .kepler import (
    PlanarCentreOfMassReferenceState,
    PlanarKeplerianReference,
)
from .multiangle_state import MultiAngleCoordinateMapper
from .planar_attitude import (
    InertialRestToRestReference,
    NadirPointingReference,
    PlanarAttitudeReferenceState,
)


@dataclass(frozen=True)
class MultiAngleReferenceState:
    time: float
    q: np.ndarray
    u: np.ndarray
    centre_of_mass: PlanarCentreOfMassReferenceState
    attitude: PlanarAttitudeReferenceState

    @property
    def state(self) -> np.ndarray:
        return np.hstack((self.q, self.u))

    @property
    def theta_ddot(self) -> float:
        return self.attitude.theta_ddot


class MultiAngleReferenceBuilder:
    """Build a nominal full-state reference for multi-angle control.

    Every evaluation imposes zero relative angles, flexible coordinates,
    relative angular speeds and modal speeds. Only the centre-of-mass
    translation and central attitude follow time-varying references.
    """

    _SUPPORTED_ATTITUDE_REFERENCES = (
        InertialRestToRestReference,
        NadirPointingReference,
    )

    def __init__(
        self,
        dynamics: Any,
        centre_of_mass_reference: PlanarKeplerianReference,
        attitude_reference: (
            InertialRestToRestReference | NadirPointingReference
        ),
    ):
        if not isinstance(
            centre_of_mass_reference,
            PlanarKeplerianReference,
        ):
            raise TypeError(
                "centre_of_mass_reference must be a "
                "PlanarKeplerianReference."
            )
        if not isinstance(
            attitude_reference,
            self._SUPPORTED_ATTITUDE_REFERENCES,
        ):
            raise TypeError(
                "attitude_reference must be an "
                "InertialRestToRestReference or NadirPointingReference."
            )

        self.dynamics = dynamics
        self.centre_of_mass_reference = centre_of_mass_reference
        self.attitude_reference = attitude_reference
        self.coordinate_mapper = MultiAngleCoordinateMapper(dynamics)
        self.state_dimension = int(dynamics.state_dimension)
        self.central_angle_index = self._symbol_index(
            dynamics.q,
            dynamics.central_angle,
        )
        self.central_speed_index = self._symbol_index(
            dynamics.u,
            dynamics.central_speed,
        )

    def evaluate(self, t: float) -> MultiAngleReferenceState:
        centre_of_mass = self.centre_of_mass_reference.evaluate(t)
        attitude = self._evaluate_attitude(t, centre_of_mass)

        q_reference = np.zeros(self.state_dimension, dtype=float)
        u_reference = np.zeros(self.state_dimension, dtype=float)
        q_reference[self.central_angle_index] = attitude.theta
        u_reference[self.central_speed_index] = attitude.theta_dot

        mapped = self.coordinate_mapper.map(
            q_reference,
            u_reference,
            centre_of_mass_position=centre_of_mass.position,
            centre_of_mass_velocity=centre_of_mass.velocity,
        )

        return MultiAngleReferenceState(
            time=float(t),
            q=mapped.q,
            u=mapped.u,
            centre_of_mass=centre_of_mass,
            attitude=attitude,
        )

    def _evaluate_attitude(
        self,
        t: float,
        centre_of_mass: PlanarCentreOfMassReferenceState,
    ) -> PlanarAttitudeReferenceState:
        if isinstance(self.attitude_reference, NadirPointingReference):
            return self.attitude_reference.evaluate(
                t,
                position=centre_of_mass.position,
                velocity=centre_of_mass.velocity,
            )
        return self.attitude_reference.evaluate(t)

    @staticmethod
    def _symbol_index(symbols, target) -> int:
        return list(symbols).index(target)
