from __future__ import annotations

import math
from collections.abc import Callable

from scipy.integrate import quad
from scipy.optimize import brentq

from .diagnostics import EventType, RaySegment
from .metrics import StaticSphericalMetric
from .policies import SolverOptions


def make_quad_segment(
    *,
    metric: StaticSphericalMetric,
    b: float,
    u0: float,
    u1: float,
    phi0: float,
    direction: str,
    endpoint_event: EventType,
    region: str,
    options: SolverOptions,
    safe_g: Callable[[StaticSphericalMetric, float, float], float],
    regularize_outward_start: bool,
    regularize_outward_turning_endpoint: bool = False,
) -> tuple[RaySegment, float]:
    outward = u1 < u0
    span = abs(u1 - u0)
    low, high = (u1, u0) if outward else (u0, u1)

    def integrand(u: float) -> float:
        return 1.0 / math.sqrt(max(safe_g(metric, u, b), 1e-300))

    def integrate_turning_endpoint(target_u: float) -> tuple[float, float]:
        target_x = 1.0 - math.sqrt(max(abs(u1 - target_u), 0.0) / span)

        def transformed_integrand(x: float) -> float:
            distance = 1.0 - x
            if outward:
                u = u1 + span * distance**2
            else:
                u = u1 - span * distance**2
            return 2.0 * span * distance * integrand(u)

        return quad(
            transformed_integrand,
            0.0,
            target_x,
            epsabs=options.quad_epsabs,
            epsrel=options.quad_epsrel,
            limit=200,
        )

    def integrate_outward_from_turning_start(target_u: float) -> tuple[float, float]:
        target_x = math.sqrt(max(u0 - target_u, 0.0) / span)

        def transformed_integrand(x: float) -> float:
            u = u0 - span * x**2
            return 2.0 * span * x * integrand(u)

        return quad(
            transformed_integrand,
            0.0,
            target_x,
            epsabs=options.quad_epsabs,
            epsrel=options.quad_epsrel,
            limit=200,
        )

    def integrate_to(target_u: float) -> tuple[float, float]:
        if endpoint_event == EventType.TURNING_POINT and span > 0.0:
            if not outward or regularize_outward_turning_endpoint:
                return integrate_turning_endpoint(target_u)
        if outward:
            if regularize_outward_start and span > 0.0:
                return integrate_outward_from_turning_start(target_u)
            return quad(
                integrand,
                target_u,
                high,
                epsabs=options.quad_epsabs,
                epsrel=options.quad_epsrel,
                limit=200,
            )
        return quad(
            integrand,
            u0,
            target_u,
            epsabs=options.quad_epsabs,
            epsrel=options.quad_epsrel,
            limit=200,
        )

    phi_width, error = integrate_to(u1)
    full_phi1 = phi0 + phi_width
    phi1 = min(options.max_phi, full_phi1)
    truncated = phi1 < full_phi1

    def angle_to_u(phi: float) -> float:
        target = phi - phi0
        if target <= 0.0:
            return u0
        if target >= phi_width:
            return u1

        def residual(u: float) -> float:
            value, _ = integrate_to(u)
            return value - target

        return brentq(residual, low, high, xtol=options.root_atol, rtol=options.root_rtol)

    u_end = angle_to_u(phi1)
    return (
        RaySegment(
            region=region,
            radial_direction=direction,
            phi_start=phi0,
            phi_end=phi1,
            u_start=u0,
            u_end=u_end,
            endpoint_event=EventType.MAX_PHI if truncated else endpoint_event,
            evaluator=angle_to_u,
        ),
        error,
    )
