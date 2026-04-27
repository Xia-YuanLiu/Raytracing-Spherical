from __future__ import annotations

import math
from dataclasses import dataclass

from .metrics import StaticSphericalMetric


@dataclass(frozen=True)
class InfinityObserver:
    coordinate_name: str = "b"

    def impact_parameter(self, coordinate: float) -> float:
        if coordinate < 0.0:
            raise ValueError("screen coordinate must be non-negative")
        return float(coordinate)

    def screen_radius(self, coordinate: float) -> float:
        return self.impact_parameter(coordinate)

    @property
    def u_start(self) -> float:
        return 0.0


@dataclass(frozen=True)
class FiniteStaticObserver:
    r_obs: float
    metric: StaticSphericalMetric
    coordinate_name: str = "alpha"

    def __post_init__(self) -> None:
        if self.r_obs <= 0.0:
            raise ValueError("r_obs must be positive")
        if self.metric.A(self.r_obs) <= 0.0:
            raise ValueError("static observer requires A(r_obs) > 0")

    def impact_parameter(self, alpha: float) -> float:
        if alpha < 0.0 or alpha >= math.pi / 2:
            raise ValueError("alpha must satisfy 0 <= alpha < pi/2")
        return self.r_obs * math.sin(alpha) / math.sqrt(self.metric.A(self.r_obs))

    def screen_radius(self, alpha: float) -> float:
        if alpha < 0.0 or alpha >= math.pi / 2:
            raise ValueError("alpha must satisfy 0 <= alpha < pi/2")
        return self.r_obs * math.tan(alpha)

    @property
    def u_start(self) -> float:
        return 1.0 / self.r_obs
