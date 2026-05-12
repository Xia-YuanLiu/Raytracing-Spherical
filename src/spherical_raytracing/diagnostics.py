from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Callable


class EventType(StrEnum):
    TURNING_POINT = "turning_point"
    HORIZON = "horizon"
    INNER_BOUNDARY = "inner_boundary"
    OUTER_BOUNDARY = "outer_boundary"
    REGION_TRANSITION = "region_transition"
    SHELL_CROSSING = "shell_crossing"
    FORBIDDEN_REGION = "forbidden_region"
    MAX_PHI = "max_phi"
    ESCAPE = "escape"
    SOLVER_FAILURE = "solver_failure"


@dataclass(frozen=True)
class CriticalCurve:
    id: str
    r_ph: float
    b_crit: float
    stability: str
    region: str
    reachable_from_observer: bool


@dataclass(frozen=True)
class RayEvent:
    event_type: EventType
    phi: float
    u: float
    region: str
    message: str = ""


@dataclass(frozen=True)
class RaySegment:
    region: str
    radial_direction: str
    phi_start: float
    phi_end: float
    u_start: float
    u_end: float
    endpoint_event: EventType
    evaluator: Callable[[float], float]

    def contains_phi(self, phi: float, *, atol: float = 1e-12) -> bool:
        return self.phi_start - atol <= phi <= self.phi_end + atol

    def u_at(self, phi: float) -> float:
        if not self.contains_phi(phi):
            raise ValueError(f"phi={phi} is outside segment [{self.phi_start}, {self.phi_end}]")
        return float(self.evaluator(phi))

    def r_at(self, phi: float) -> float:
        u = self.u_at(phi)
        if u <= 0.0:
            return float("inf")
        return 1.0 / u


@dataclass(frozen=True)
class RayDiagnostics:
    estimated_error: float
    near_critical: bool
    critical_curve_id: str | None
    hit_horizon: bool
    hit_inner_boundary: bool
    turning_point_count: int
    max_phi_reached: bool
    termination_reason: str
    residuals: dict[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class RayResult:
    b: float
    segments: list[RaySegment]
    events: list[RayEvent]
    critical_status: CriticalCurve | None
    diagnostics: RayDiagnostics
