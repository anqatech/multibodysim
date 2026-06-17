from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .bounded_minimum_effort import (
    BoundedMinimumEffortAllocation,
    solve_bounded_minimum_effort_allocation,
)
from .effectiveness import (
    ControlEffectivenessVector,
    evaluate_control_effectiveness_vector,
)


@dataclass(frozen=True)
class StateBoundedMinimumEffortAllocation:
    control_effectiveness: ControlEffectivenessVector
    allocation: BoundedMinimumEffortAllocation

    @property
    def torque_increments(self):
        return self.allocation.torque_increments


def allocate_bounded_minimum_effort_at_state(
    dynamics: Any,
    plant_view: Any,
    q,
    u,
    commanded_acceleration,
    effort_penalty_matrix,
    lower_bounds,
    upper_bounds,
    *,
    baseline_torques=None,
    tolerance=1e-10,
) -> StateBoundedMinimumEffortAllocation:
    """Allocate bounded minimum-effort torques at one state."""

    effectiveness = evaluate_control_effectiveness_vector(
        dynamics,
        plant_view,
        q,
        u,
        baseline_torques=baseline_torques,
    )
    allocation = solve_bounded_minimum_effort_allocation(
        commanded_acceleration,
        effectiveness.effectiveness,
        effort_penalty_matrix,
        lower_bounds,
        upper_bounds,
        tolerance=tolerance,
    )
    return StateBoundedMinimumEffortAllocation(
        control_effectiveness=effectiveness,
        allocation=allocation,
    )
