from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Protocol

from .diagnostics import EventType


@dataclass(frozen=True)
class SolverOptions:
    max_phi: float = 12.0 * math.pi
    critical_exclusion: float = 1e-6
    quad_epsabs: float = 1e-10
    quad_epsrel: float = 1e-10
    root_atol: float = 1e-12
    root_rtol: float = 1e-12
    max_brackets: int = 512
    horizon_buffer: float = 1e-9
    inner_boundary_radius: float | None = None


class TracePolicy(Protocol):
    def is_terminal(self, event: EventType, current_region: str) -> bool:
        raise NotImplementedError

    def next_region(self, event: EventType, current_region: str) -> str | None:
        raise NotImplementedError


@dataclass(frozen=True)
class OutsideTracePolicy:
    def is_terminal(self, event: EventType, current_region: str) -> bool:
        return event in {
            EventType.HORIZON,
            EventType.INNER_BOUNDARY,
            EventType.OUTER_BOUNDARY,
            EventType.FORBIDDEN_REGION,
            EventType.ESCAPE,
            EventType.MAX_PHI,
            EventType.SOLVER_FAILURE,
        }

    def next_region(self, event: EventType, current_region: str) -> str | None:
        return None


@dataclass(frozen=True)
class ThroughTracePolicy:
    region_transitions: dict[tuple[str, EventType], str] = field(default_factory=dict)

    def is_terminal(self, event: EventType, current_region: str) -> bool:
        if event == EventType.HORIZON:
            return self.next_region(event, current_region) is None
        return OutsideTracePolicy().is_terminal(event, current_region)

    def next_region(self, event: EventType, current_region: str) -> str | None:
        return self.region_transitions.get((current_region, event))
