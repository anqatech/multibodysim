from dataclasses import dataclass
from typing import Optional, Protocol
import numpy as np


@dataclass
class ControlOutput:
    tau_ff: float = 0.0
    tau_fb: float = 0.0

    @property
    def tau_total(self) -> float:
        return self.tau_ff + self.tau_fb
    
class AttitudeController(Protocol):
    def reset(self) -> None: ...
    def compute(
        self, t: float, q: np.ndarray, u: np.ndarray, Md: Optional[np.ndarray] = None
    ) -> ControlOutput: ...
