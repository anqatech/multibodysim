from dataclasses import dataclass
from typing import Optional, Protocol
import numpy as np


@dataclass(init=False)
class ControlOutput:
    tau_fb: float
    tau_reference_ff: float
    tau_gravity_gradient_ff: float
    bus_torques: np.ndarray | None

    def __init__(
        self,
        tau_ff: float = 0.0,
        tau_fb: float = 0.0,
        *,
        tau_reference_ff: float | None = None,
        tau_gravity_gradient_ff: float = 0.0,
        bus_torques=None,
    ):
        self.tau_fb = float(tau_fb)
        self.tau_reference_ff = float(
            tau_ff if tau_reference_ff is None else tau_reference_ff
        )
        self.tau_gravity_gradient_ff = float(
            tau_gravity_gradient_ff
        )
        self.bus_torques = self._normalise_bus_torques(bus_torques)

    @property
    def tau_ff(self) -> float:
        return self.tau_reference_ff + self.tau_gravity_gradient_ff

    @property
    def tau_total(self) -> float:
        return self.tau_ff + self.tau_fb

    @staticmethod
    def _normalise_bus_torques(bus_torques):
        if bus_torques is None:
            return None
        torques = np.asarray(bus_torques, dtype=float).reshape(-1)
        if not np.all(np.isfinite(torques)):
            raise ValueError("bus_torques must contain only finite values.")
        return torques
    
class AttitudeController(Protocol):
    def reset(self) -> None: ...
    def compute(
        self, t: float, q: np.ndarray, u: np.ndarray, Md: Optional[np.ndarray] = None
    ) -> ControlOutput: ...
