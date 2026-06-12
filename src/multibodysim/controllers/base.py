from dataclasses import dataclass
from typing import Optional, Protocol
import numpy as np


@dataclass(init=False)
class ControlOutput:
    tau_fb: float
    tau_reference_ff: float
    tau_gravity_gradient_ff: float

    def __init__(
        self,
        tau_ff: float = 0.0,
        tau_fb: float = 0.0,
        *,
        tau_reference_ff: float | None = None,
        tau_gravity_gradient_ff: float = 0.0,
    ):
        self.tau_fb = float(tau_fb)
        self.tau_reference_ff = float(
            tau_ff if tau_reference_ff is None else tau_reference_ff
        )
        self.tau_gravity_gradient_ff = float(
            tau_gravity_gradient_ff
        )

    @property
    def tau_ff(self) -> float:
        return self.tau_reference_ff + self.tau_gravity_gradient_ff

    @property
    def tau_total(self) -> float:
        return self.tau_ff + self.tau_fb
    
class AttitudeController(Protocol):
    def reset(self) -> None: ...
    def compute(
        self, t: float, q: np.ndarray, u: np.ndarray, Md: Optional[np.ndarray] = None
    ) -> ControlOutput: ...
