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
    real_roots = sorted(float(root.real) for root in roots if abs(root.imag) < atol and root.real > 0.0)
    unique: list[float] = []
    for root in real_roots:
        if not unique or not math.isclose(root, unique[-1], rel_tol=1e-8, abs_tol=0.0):
            unique.append(root)
    return unique


def _unique_positive_values(values: list[float], *, rel_tol: float = 1e-12) -> list[float]:
    unique: list[float] = []
    for value in sorted(value for value in values if value > 0.0):
        if not unique or not math.isclose(value, unique[-1], rel_tol=rel_tol, abs_tol=0.0):
            unique.append(value)
    return unique


def _clamp_near_zero_discriminant(discriminant: float, scale: float, *, rel_tol: float = 1e-12) -> float | None:
    tolerance = rel_tol * scale
    if discriminant < -tolerance:
        return None
    if abs(discriminant) <= tolerance:
        return 0.0
    return discriminant


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
class ReissnerNordstromMetric:
    mass: float = 1.0
    charge: float = 0.0
    region: str = "external"

    def __post_init__(self) -> None:
        if self.mass <= 0.0:
            raise ValueError("mass must be positive")
        if not math.isfinite(self.charge):
            raise ValueError("charge must be finite")

    @property
    def charge_squared(self) -> float:
        return self.charge**2

    def A(self, r: float) -> float:
        _require_positive_radius(r)
        return 1.0 - 2.0 * self.mass / r + self.charge_squared / r**2

    def B(self, r: float) -> float:
        a = self.A(r)
        if a == 0.0:
            return math.inf
        return 1.0 / a

    def dA_dr(self, r: float) -> float:
        _require_positive_radius(r)
        return 2.0 * self.mass / r**2 - 2.0 * self.charge_squared / r**3

    def dB_dr(self, r: float) -> float:
        a = self.A(r)
        return -self.dA_dr(r) / a**2

    def G(self, u: float, b: float) -> float:
        if b <= 0.0:
            raise ValueError("b must be positive")
        return 1.0 / b**2 - u**2 + 2.0 * self.mass * u**3 - self.charge_squared * u**4

    def horizons(self) -> list[float]:
        discriminant = self.mass**2 - self.charge_squared
        discriminant = _clamp_near_zero_discriminant(
            discriminant,
            max(self.mass**2, self.charge_squared),
        )
        if discriminant is None:
            return []
        root = math.sqrt(discriminant)
        outer = self.mass + root
        candidates = [outer]
        if self.charge_squared > 0.0 and outer > 0.0:
            candidates.append(self.charge_squared / outer)
        return _unique_positive_values(candidates)

    def photon_spheres(self) -> list[float]:
        discriminant = 9.0 * self.mass**2 - 8.0 * self.charge_squared
        discriminant = _clamp_near_zero_discriminant(
            discriminant,
            max(9.0 * self.mass**2, 8.0 * self.charge_squared),
        )
        if discriminant is None:
            return []
        root = math.sqrt(discriminant)
        candidates = [(3.0 * self.mass - root) / 2.0, (3.0 * self.mass + root) / 2.0]
        return [r for r in _unique_positive_values(candidates) if self.A(r) > 0.0]

    def critical_curves(self) -> list[CriticalCurve]:
        photon_spheres = self.photon_spheres()
        return [
            CriticalCurve(
                id=f"{self.region}:photon_sphere:{index}",
                r_ph=r_ph,
                b_crit=r_ph / math.sqrt(self.A(r_ph)),
                stability="unstable" if index == len(photon_spheres) - 1 else "stable",
                region=self.region,
                reachable_from_observer=index == len(photon_spheres) - 1,
            )
            for index, r_ph in enumerate(photon_spheres)
        ]

    def valid_radial_domain(self) -> tuple[float, float]:
        horizons = self.horizons()
        if horizons:
            return (horizons[-1], math.inf)
        return (0.0, math.inf)

    def static_domains(self) -> list[tuple[float, float]]:
        horizons = self.horizons()
        if len(horizons) >= 2:
            return [(0.0, horizons[0]), (horizons[-1], math.inf)]
        return [self.valid_radial_domain()]


@dataclass(frozen=True)
class ReissnerNordstromDeSitterMetric:
    mass: float = 1.0
    charge: float = 0.0
    cosmological_constant: float = 0.01
    region: str = "external"

    def __post_init__(self) -> None:
        if self.mass <= 0.0:
            raise ValueError("mass must be positive")
        if not math.isfinite(self.charge):
            raise ValueError("charge must be finite")
        if not math.isfinite(self.cosmological_constant):
            raise ValueError("cosmological_constant must be finite")
        if self.cosmological_constant < 0.0:
            raise ValueError("cosmological_constant must be non-negative")

    @property
    def charge_squared(self) -> float:
        return self.charge**2

    def A(self, r: float) -> float:
        _require_positive_radius(r)
        return 1.0 - 2.0 * self.mass / r + self.charge_squared / r**2 - self.cosmological_constant * r**2 / 3.0

    def B(self, r: float) -> float:
        a = self.A(r)
        if a == 0.0:
            return math.inf
        return 1.0 / a

    def dA_dr(self, r: float) -> float:
        _require_positive_radius(r)
        return 2.0 * self.mass / r**2 - 2.0 * self.charge_squared / r**3 - 2.0 * self.cosmological_constant * r / 3.0

    def dB_dr(self, r: float) -> float:
        a = self.A(r)
        return -self.dA_dr(r) / a**2

    def G(self, u: float, b: float) -> float:
        if b <= 0.0:
            raise ValueError("b must be positive")
        return 1.0 / b**2 - u**2 + 2.0 * self.mass * u**3 - self.charge_squared * u**4 + self.cosmological_constant / 3.0

    def horizons(self) -> list[float]:
        lam = self.cosmological_constant
        if lam == 0.0:
            return ReissnerNordstromMetric(
                mass=self.mass,
                charge=self.charge,
                region=self.region,
            ).horizons()
        return _positive_real_roots([-lam / 3.0, 0.0, 1.0, -2.0 * self.mass, self.charge_squared])

    def photon_spheres(self) -> list[float]:
        discriminant = 9.0 * self.mass**2 - 8.0 * self.charge_squared
        discriminant = _clamp_near_zero_discriminant(
            discriminant,
            max(9.0 * self.mass**2, 8.0 * self.charge_squared),
        )
        if discriminant is None:
            return []
        root = math.sqrt(discriminant)
        candidates = [(3.0 * self.mass - root) / 2.0, (3.0 * self.mass + root) / 2.0]
        return [r for r in _unique_positive_values(candidates) if self.A(r) > 0.0]

    def critical_curves(self) -> list[CriticalCurve]:
        photon_spheres = self.photon_spheres()
        return [
            CriticalCurve(
                id=f"{self.region}:photon_sphere:{index}",
                r_ph=r_ph,
                b_crit=r_ph / math.sqrt(self.A(r_ph)),
                stability="unstable" if index == len(photon_spheres) - 1 else "stable",
                region=self.region,
                reachable_from_observer=index == len(photon_spheres) - 1,
            )
            for index, r_ph in enumerate(photon_spheres)
        ]

    def static_domains(self) -> list[tuple[float, float]]:
        if self.cosmological_constant == 0.0:
            return ReissnerNordstromMetric(
                mass=self.mass,
                charge=self.charge,
                region=self.region,
            ).static_domains()
        points = [0.0] + self.horizons()
        domains: list[tuple[float, float]] = []
        for left, right in zip(points[:-1], points[1:]):
            midpoint = 0.5 * (left + right)
            if midpoint > 0.0 and self.A(midpoint) > 0.0:
                domains.append((left, right))
        horizons = self.horizons()
        if horizons:
            right_probe = horizons[-1] * 1.1
            if self.A(right_probe) > 0.0:
                domains.append((horizons[-1], math.inf))
        return domains

    def valid_radial_domain(self) -> tuple[float, float]:
        if self.cosmological_constant == 0.0:
            return ReissnerNordstromMetric(
                mass=self.mass,
                charge=self.charge,
                region=self.region,
            ).valid_radial_domain()
        domains = self.static_domains()
        if domains:
            return domains[-1]
        return (0.0, 0.0)


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
