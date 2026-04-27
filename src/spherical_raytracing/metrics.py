from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Protocol

import numpy as np

from .diagnostics import CriticalCurve


class StaticSphericalMetric(Protocol):
    region: str

    def A(self, r: float) -> float:
        raise NotImplementedError

    def B(self, r: float) -> float:
        raise NotImplementedError

    def dA_dr(self, r: float) -> float:
        raise NotImplementedError

    def dB_dr(self, r: float) -> float:
        raise NotImplementedError

    def G(self, u: float, b: float) -> float:
        raise NotImplementedError

    def horizons(self) -> list[float]:
        raise NotImplementedError

    def photon_spheres(self) -> list[float]:
        raise NotImplementedError

    def critical_curves(self) -> list[CriticalCurve]:
        raise NotImplementedError

    def valid_radial_domain(self) -> tuple[float, float]:
        raise NotImplementedError

    def static_domains(self) -> list[tuple[float, float]]:
        raise NotImplementedError


def _require_positive_radius(r: float) -> None:
    if r <= 0.0:
        raise ValueError("r must be positive")


def _positive_real_roots(coefficients: list[float], *, atol: float = 1e-10) -> list[float]:
    roots = np.roots(coefficients)
    real_roots = sorted(float(root.real) for root in roots if abs(root.imag) < atol and root.real > atol)
    unique: list[float] = []
    for root in real_roots:
        if not unique or abs(root - unique[-1]) > 1e-8:
            unique.append(root)
    return unique


@dataclass(frozen=True)
class SchwarzschildMetric:
    mass: float = 1.0
    region: str = "external"

    def __post_init__(self) -> None:
        if self.mass <= 0.0:
            raise ValueError("mass must be positive")

    def A(self, r: float) -> float:
        _require_positive_radius(r)
        return 1.0 - 2.0 * self.mass / r

    def B(self, r: float) -> float:
        a = self.A(r)
        if a == 0.0:
            return math.inf
        return 1.0 / a

    def dA_dr(self, r: float) -> float:
        _require_positive_radius(r)
        return 2.0 * self.mass / r**2

    def dB_dr(self, r: float) -> float:
        a = self.A(r)
        return -self.dA_dr(r) / a**2

    def G(self, u: float, b: float) -> float:
        if b <= 0.0:
            raise ValueError("b must be positive")
        return 1.0 / b**2 - u**2 + 2.0 * self.mass * u**3

    def horizons(self) -> list[float]:
        return [2.0 * self.mass]

    def photon_spheres(self) -> list[float]:
        return [3.0 * self.mass]

    def critical_curves(self) -> list[CriticalCurve]:
        r_ph = self.photon_spheres()[0]
        return [
            CriticalCurve(
                id=f"{self.region}:photon_sphere:0",
                r_ph=r_ph,
                b_crit=r_ph / math.sqrt(self.A(r_ph)),
                stability="unstable",
                region=self.region,
                reachable_from_observer=True,
            )
        ]

    def valid_radial_domain(self) -> tuple[float, float]:
        return (self.horizons()[0], math.inf)

    def static_domains(self) -> list[tuple[float, float]]:
        return [self.valid_radial_domain()]


@dataclass(frozen=True)
class LQGMetric:
    mass: float = 1.0
    alpha: float = 0.0
    region: str = "external"

    def __post_init__(self) -> None:
        if self.mass <= 0.0:
            raise ValueError("mass must be positive")

    def f(self, r: float) -> float:
        _require_positive_radius(r)
        return 1.0 - 2.0 * self.mass / r + self.alpha * self.mass**2 / r**4

    def A(self, r: float) -> float:
        return self.f(r)

    def B(self, r: float) -> float:
        f = self.f(r)
        if f == 0.0:
            return math.inf
        return 1.0 / f

    def dA_dr(self, r: float) -> float:
        _require_positive_radius(r)
        return 2.0 * self.mass / r**2 - 4.0 * self.alpha * self.mass**2 / r**5

    def dB_dr(self, r: float) -> float:
        f = self.f(r)
        return -self.dA_dr(r) / f**2

    def G(self, u: float, b: float) -> float:
        if b <= 0.0:
            raise ValueError("b must be positive")
        return -self.alpha * self.mass**2 * u**6 + 2.0 * self.mass * u**3 - u**2 + 1.0 / b**2

    def horizons(self) -> list[float]:
        return _positive_real_roots([1.0, -2.0 * self.mass, 0.0, 0.0, self.alpha * self.mass**2])

    def photon_spheres(self) -> list[float]:
        candidates = _positive_real_roots([1.0, -3.0 * self.mass, 0.0, 0.0, 3.0 * self.alpha * self.mass**2])
        return [r for r in candidates if self.A(r) > 0.0]

    def critical_curves(self) -> list[CriticalCurve]:
        return [
            CriticalCurve(
                id=f"{self.region}:photon_sphere:{index}",
                r_ph=r_ph,
                b_crit=r_ph / math.sqrt(self.A(r_ph)),
                stability="unstable",
                region=self.region,
                reachable_from_observer=True,
            )
            for index, r_ph in enumerate(self.photon_spheres())
        ]

    def valid_radial_domain(self) -> tuple[float, float]:
        horizons = self.horizons()
        if horizons:
            return (horizons[-1], math.inf)
        return (0.0, math.inf)

    def static_domains(self) -> list[tuple[float, float]]:
        return [self.valid_radial_domain()]
