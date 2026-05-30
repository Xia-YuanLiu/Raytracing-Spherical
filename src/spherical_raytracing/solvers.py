from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import brentq

from ._segments import make_quad_segment
from ._turning import _first_turning_point_scalar_scan, _first_turning_point_scan
from .diagnostics import CriticalCurve, EventType, RayDiagnostics, RayEvent, RayResult, RaySegment
from .metrics import StaticSphericalMetric
from .observers import FiniteStaticObserver, InfinityObserver
from .policies import OutsideTracePolicy, SolverOptions, ThroughTracePolicy, TracePolicy


def _guard_tabulated_metric_combinations(
    metric: StaticSphericalMetric,
    observer: InfinityObserver | FiniteStaticObserver,
    trace_policy: TracePolicy,
) -> None:
    """Reject ``TabulatedMetric`` + ``InfinityObserver`` / ``ThroughTracePolicy``.

    Plan decisions 10 and 11 forbid these combinations in the first phase
    because ``TabulatedMetric.G(u=0)`` is undefined and the numerical payload
    only carries an exterior static patch (no internal-region support).

    ``TabulatedMetric`` is imported lazily so callers that never touch the
    numerical pipeline do not pay for ``scipy.interpolate`` at solver
    module-load time.
    """
    from .numerical_metrics import TabulatedMetric

    if not isinstance(metric, TabulatedMetric):
        return
    if isinstance(observer, InfinityObserver):
        raise NotImplementedError(
            "TabulatedMetric requires FiniteStaticObserver; InfinityObserver is "
            "not supported (plan decision 11: u=0 has no tabulated support)"
        )
    if isinstance(trace_policy, ThroughTracePolicy):
        raise NotImplementedError(
            "TabulatedMetric + ThroughTracePolicy is not supported in the first "
            "phase (plan decision 10: no internal-region payload)"
        )


_BOUNDARY_HORIZON_REL_TOL = 1e-6
_BOUNDARY_HORIZON_ABS_TOL = 1e-8


class _FastPathUnavailable(Exception):
    """Raised when a private fast path cannot prove the segment is safe."""


_GL_CACHE: dict[int, tuple[np.ndarray, np.ndarray]] = {}
_FAST_GL_LOW_ORDER = 32
_FAST_GL_HIGH_ORDER = 64
_FAST_GL_TOL_MULTIPLIER = 50.0
_FAST_G_NEGATIVE_RTOL = 1e-13


def _legendre_nodes_weights(order: int) -> tuple[np.ndarray, np.ndarray]:
    cached = _GL_CACHE.get(order)
    if cached is None:
        nodes, weights = np.polynomial.legendre.leggauss(order)
        cached = (np.asarray(nodes, dtype=float), np.asarray(weights, dtype=float))
        _GL_CACHE[order] = cached
    return cached


def _fast_gl_tolerance(value_a: float, value_b: float, options: SolverOptions) -> float:
    scale = max(1.0, abs(value_a), abs(value_b))
    return _FAST_GL_TOL_MULTIPLIER * max(options.quad_epsabs, options.quad_epsrel * scale)


def _integrate_gl_pair(
    evaluator,
    lower: float,
    upper: float,
    options: SolverOptions,
) -> tuple[float, float]:
    if not (math.isfinite(lower) and math.isfinite(upper)):
        raise _FastPathUnavailable("non-finite integration bound")
    if upper < lower:
        raise _FastPathUnavailable("GL fast path expects increasing bounds")
    if upper == lower:
        return 0.0, 0.0

    low_value = _integrate_gl_order(evaluator, lower, upper, _FAST_GL_LOW_ORDER)
    high_value = _integrate_gl_order(evaluator, lower, upper, _FAST_GL_HIGH_ORDER)
    error = abs(high_value - low_value)
    if not (math.isfinite(high_value) and math.isfinite(error)):
        raise _FastPathUnavailable("non-finite GL integral")
    if error > _fast_gl_tolerance(low_value, high_value, options):
        raise _FastPathUnavailable("GL pair disagreement exceeds tolerance")
    return high_value, error


def _integrate_gl_order(evaluator, lower: float, upper: float, order: int) -> float:
    nodes, weights = _legendre_nodes_weights(order)
    half_width = 0.5 * (upper - lower)
    midpoint = 0.5 * (upper + lower)
    samples = midpoint + half_width * nodes
    values = np.asarray(evaluator(samples), dtype=float)
    if values.shape != samples.shape:
        raise _FastPathUnavailable("vector integrand returned unexpected shape")
    if np.any(~np.isfinite(values)):
        raise _FastPathUnavailable("vector integrand returned non-finite values")
    return float(half_width * np.dot(weights, values))


def _tabulated_fast_integrand(metric: StaticSphericalMetric, u_values: np.ndarray, b: float) -> np.ndarray:
    try:
        g = np.asarray(metric._G_array(u_values, b), dtype=float)
    except ValueError as exc:
        raise _FastPathUnavailable(str(exc)) from exc

    if np.any(~np.isfinite(g)):
        raise _FastPathUnavailable("non-finite G values")
    g_scale = max(1.0, float(np.max(np.abs(g))) if g.size else 1.0)
    if np.any(g < -_FAST_G_NEGATIVE_RTOL * g_scale):
        raise _FastPathUnavailable("negative G sample")
    if np.any(g <= 0.0):
        raise _FastPathUnavailable("non-positive G sample")
    return 1.0 / np.sqrt(g)


def _segment_fast_stats(solver: object) -> dict[str, int]:
    stats = getattr(solver, "_fast_segment_stats", None)
    if stats is None:
        stats = {
            "attempted": 0,
            "accepted": 0,
            "fallback": 0,
            "late_fallback": 0,
        }
        setattr(solver, "_fast_segment_stats", stats)
    return stats


def _safe_g(metric: StaticSphericalMetric, u: float, b: float) -> float:
    value = metric.G(float(u), float(b))
    if not math.isfinite(value):
        return -math.inf
    return value


def _near_critical(metric: StaticSphericalMetric, b: float, exclusion: float) -> CriticalCurve | None:
    for curve in metric.critical_curves():
        if curve.reachable_from_observer and abs(b - curve.b_crit) / curve.b_crit < exclusion:
            return curve
    return None


def _classify_radial_event(
    metric: StaticSphericalMetric,
    r_boundary: float,
    *,
    default: EventType,
) -> EventType:
    """Classify a raw boundary radius against ``metric.horizons()``.

    The tolerance compares the original radius against each horizon root
    (rather than back-projecting from a buffered ``u``), per the plan's
    decision 6 hard constraint.
    """
    if not math.isfinite(r_boundary) or r_boundary <= 0.0:
        return default
    for horizon in metric.horizons():
        if horizon <= 0.0 or not math.isfinite(horizon):
            continue
        if math.isclose(
            r_boundary,
            horizon,
            rel_tol=_BOUNDARY_HORIZON_REL_TOL,
            abs_tol=_BOUNDARY_HORIZON_ABS_TOL,
        ):
            return EventType.HORIZON
    return default


def horizon_termination_reason(metric: StaticSphericalMetric, event_type: EventType, u: float) -> str:
    if event_type != EventType.HORIZON:
        return event_type.value
    if u <= 0.0 or not math.isfinite(u):
        return event_type.value
    radius = 1.0 / u
    r_min, r_max = metric.valid_radial_domain()
    if math.isfinite(r_min) and math.isclose(
        radius,
        r_min,
        rel_tol=_BOUNDARY_HORIZON_REL_TOL,
        abs_tol=_BOUNDARY_HORIZON_ABS_TOL,
    ):
        return "black_hole_horizon"
    if math.isfinite(r_max) and math.isclose(
        radius,
        r_max,
        rel_tol=_BOUNDARY_HORIZON_REL_TOL,
        abs_tol=_BOUNDARY_HORIZON_ABS_TOL,
    ):
        return "cosmological_horizon"
    return event_type.value


def horizon_event(metric: StaticSphericalMetric, event_type: EventType, phi: float, u: float, region: str) -> RayEvent:
    return RayEvent(event_type, phi, u, region, horizon_termination_reason(metric, event_type, u))


def _mirror_escape_segment(inbound: RaySegment, max_phi: float) -> RaySegment:
    full_phi_end = 2.0 * inbound.phi_end - inbound.phi_start
    phi_end = min(max_phi, full_phi_end)
    endpoint_event = EventType.MAX_PHI if phi_end < full_phi_end else EventType.ESCAPE

    def evaluator(phi: float, inbound: RaySegment = inbound) -> float:
        return inbound.u_at(_clamp_phi_to_segment(2.0 * inbound.phi_end - phi, inbound))

    return RaySegment(
        region=inbound.region,
        radial_direction="outward",
        phi_start=inbound.phi_end,
        phi_end=phi_end,
        u_start=inbound.u_end,
        u_end=evaluator(phi_end),
        endpoint_event=endpoint_event,
        evaluator=evaluator,
    )


def _clamp_phi_to_segment(phi: float, segment: RaySegment, *, atol: float = 1e-10) -> float:
    if phi < segment.phi_start and segment.phi_start - phi <= atol:
        return segment.phi_start
    if phi > segment.phi_end and phi - segment.phi_end <= atol:
        return segment.phi_end
    return phi


@dataclass
class QuadTransferSolver:
    metric: StaticSphericalMetric
    observer: InfinityObserver | FiniteStaticObserver
    trace_policy: TracePolicy = OutsideTracePolicy()
    options: SolverOptions = SolverOptions()

    def trace_b(self, b: float) -> RayResult:
        if b <= 0.0:
            raise ValueError("b must be positive")
        _guard_tabulated_metric_combinations(self.metric, self.observer, self.trace_policy)
        if isinstance(self.observer, FiniteStaticObserver):
            r_obs = self.observer.r_obs
            a_obs = self.metric.A(r_obs)
            if a_obs > 0.0:
                b_max = r_obs / math.sqrt(a_obs)
                if b > b_max:
                    raise ValueError(
                        f"b={b} exceeds finite observer's screen radius limit b_max={b_max}"
                    )

        critical_curve = _near_critical(self.metric, b, self.options.critical_exclusion)
        if critical_curve is not None:
            return RayResult(
                b=b,
                segments=[],
                events=[],
                critical_status=critical_curve,
                diagnostics=RayDiagnostics(
                    estimated_error=math.inf,
                    near_critical=True,
                    critical_curve_id=critical_curve.id,
                    hit_horizon=False,
                    hit_inner_boundary=False,
                    turning_point_count=0,
                    max_phi_reached=False,
                    termination_reason="near_critical",
                    residuals={},
                ),
            )

        u_start = self.observer.u_start
        u_inward_stop, inward_event = self._inward_stop_u()
        u_turn = self._first_turning_point(b, u_start, u_inward_stop)

        if u_turn is None:
            next_region = (
                self._continuation_region(EventType.HORIZON)
                if inward_event == EventType.HORIZON
                else None
            )
            if next_region is not None:
                through_result = self._trace_through_horizon(b, u_start, u_inward_stop, next_region)
                if through_result is not None:
                    return through_result

            segment, error = self._make_segment(
                b=b,
                u0=u_start,
                u1=u_inward_stop,
                phi0=0.0,
                direction="inward",
                endpoint_event=inward_event,
                region=self.metric.region,
            )
            event_type = segment.endpoint_event
            events, termination_reason = self._events_for_endpoint(event_type, segment.phi_end, segment.u_end)
            diagnostics = RayDiagnostics(
                estimated_error=error,
                near_critical=critical_curve is not None,
                critical_curve_id=None if critical_curve is None else critical_curve.id,
                hit_horizon=event_type == EventType.HORIZON,
                hit_inner_boundary=event_type == EventType.INNER_BOUNDARY,
                turning_point_count=0,
                max_phi_reached=segment.phi_end >= self.options.max_phi,
                termination_reason=termination_reason,
                residuals={"G_end": _safe_g(self.metric, segment.u_end, b)},
            )
            return RayResult(b=b, segments=[segment], events=events, critical_status=critical_curve, diagnostics=diagnostics)

        inbound, err_in = self._make_segment(
            b=b,
            u0=u_start,
            u1=u_turn,
            phi0=0.0,
            direction="inward",
            endpoint_event=EventType.TURNING_POINT,
            region=self.metric.region,
        )
        if inbound.endpoint_event == EventType.MAX_PHI:
            events, termination_reason = self._events_for_endpoint(inbound.endpoint_event, inbound.phi_end, inbound.u_end)
            diagnostics = RayDiagnostics(
                estimated_error=err_in,
                near_critical=False,
                critical_curve_id=None,
                hit_horizon=False,
                hit_inner_boundary=False,
                turning_point_count=0,
                max_phi_reached=True,
                termination_reason=termination_reason,
                residuals={"G_end": _safe_g(self.metric, inbound.u_end, b)},
            )
            return RayResult(b=b, segments=[inbound], events=events, critical_status=None, diagnostics=diagnostics)

        u_outward_stop, outward_event = self._outward_stop_u()
        if u_outward_stop > 0.0:
            outbound, err_out = self._make_segment(
                b=b,
                u0=u_turn,
                u1=u_outward_stop,
                phi0=inbound.phi_end,
                direction="outward",
                endpoint_event=outward_event,
                region=self.metric.region,
            )
            estimated_error = max(err_in, err_out)
        else:
            outbound = _mirror_escape_segment(inbound, self.options.max_phi)
            estimated_error = err_in
        final_event = outbound.endpoint_event
        events = [
            RayEvent(EventType.TURNING_POINT, inbound.phi_end, u_turn, self.metric.region),
            horizon_event(self.metric, final_event, outbound.phi_end, outbound.u_end, self.metric.region),
        ]
        diagnostics = RayDiagnostics(
            estimated_error=estimated_error,
            near_critical=False,
            critical_curve_id=None,
            hit_horizon=final_event == EventType.HORIZON,
            hit_inner_boundary=False,
            turning_point_count=1,
            max_phi_reached=final_event == EventType.MAX_PHI,
            termination_reason=horizon_termination_reason(self.metric, final_event, outbound.u_end),
            residuals={"G_turn": _safe_g(self.metric, u_turn, b)},
        )
        return RayResult(b=b, segments=[inbound, outbound], events=events, critical_status=None, diagnostics=diagnostics)

    def _inward_stop_u(self) -> tuple[float, EventType]:
        """Compute the inward integration cutoff in ``u`` and its event type.

        Uses ``metric.valid_radial_domain()[0]`` as the authoritative inner edge,
        buffered inward by ``horizon_buffer``. The event type is classified by
        comparing the raw boundary radius against ``metric.horizons()`` per
        plan decision 6 (``rel_tol=1e-6, abs_tol=1e-8``). When the metric does
        not expose a positive inner radius, falls back to
        ``SolverOptions.inner_boundary_radius`` and raises ``ValueError`` if
        neither is available.
        """
        r_min, _ = self.metric.valid_radial_domain()
        if r_min > 0.0 and math.isfinite(r_min):
            u_stop = (1.0 / r_min) * (1.0 - self.options.horizon_buffer)
            event = _classify_radial_event(self.metric, r_min, default=EventType.INNER_BOUNDARY)
            return u_stop, event
        if self.options.inner_boundary_radius is not None:
            return 1.0 / self.options.inner_boundary_radius, EventType.INNER_BOUNDARY
        raise ValueError(
            "metrics without a positive inner boundary require SolverOptions.inner_boundary_radius"
        )

    def _outward_stop_u(self) -> tuple[float, EventType]:
        """Compute the outward integration cutoff in ``u`` and its event type.

        When ``r_max`` is infinite, returns ``(0.0, ESCAPE)`` so the existing
        mirror-escape path can stay in charge of the asymptotically flat case.
        When ``r_max`` is finite, buffers outward by ``horizon_buffer`` and
        classifies the event by comparing the raw ``r_max`` against
        ``metric.horizons()``.
        """
        _, r_max = self.metric.valid_radial_domain()
        if not math.isfinite(r_max):
            return 0.0, EventType.ESCAPE
        if r_max <= 0.0:
            raise ValueError("metric.valid_radial_domain()[1] must be positive when finite")
        u_stop = (1.0 / r_max) * (1.0 + self.options.horizon_buffer)
        event = _classify_radial_event(self.metric, r_max, default=EventType.OUTER_BOUNDARY)
        return u_stop, event

    def _events_for_endpoint(self, event_type: EventType, phi: float, u: float) -> tuple[list[RayEvent], str]:
        events = [horizon_event(self.metric, event_type, phi, u, self.metric.region)]
        if self.trace_policy.is_terminal(event_type, self.metric.region):
            return events, horizon_termination_reason(self.metric, event_type, u)
        next_region = self._continuation_region(event_type)
        if next_region is None:
            events.append(
                RayEvent(
                    EventType.SOLVER_FAILURE,
                    phi,
                    u,
                    self.metric.region,
                    "policy requested continuation without a next region",
                )
            )
            return events, EventType.SOLVER_FAILURE.value
        events.append(RayEvent(EventType.REGION_TRANSITION, phi, u, next_region))
        return events, EventType.REGION_TRANSITION.value

    def _continuation_region(self, event_type: EventType) -> str | None:
        if self.trace_policy.is_terminal(event_type, self.metric.region):
            return None
        return self.trace_policy.next_region(event_type, self.metric.region)

    def _trace_through_horizon(
        self,
        b: float,
        u_start: float,
        u_horizon: float,
        next_region: str,
    ) -> RayResult | None:
        u_turn = self._first_continuation_turning_point(b, u_horizon)
        if u_turn is None:
            return None

        external, err_external = self._make_segment(
            b=b,
            u0=u_start,
            u1=u_horizon,
            phi0=0.0,
            direction="inward",
            endpoint_event=EventType.HORIZON,
            region=self.metric.region,
        )
        if external.endpoint_event == EventType.MAX_PHI:
            events, termination_reason = self._events_for_endpoint(external.endpoint_event, external.phi_end, external.u_end)
            return RayResult(
                b=b,
                segments=[external],
                events=events,
                critical_status=None,
                diagnostics=RayDiagnostics(
                    estimated_error=err_external,
                    near_critical=False,
                    critical_curve_id=None,
                    hit_horizon=False,
                    hit_inner_boundary=False,
                    turning_point_count=0,
                    max_phi_reached=True,
                    termination_reason=termination_reason,
                    residuals={"G_end": _safe_g(self.metric, external.u_end, b)},
                ),
            )

        through_in, err_through = self._make_segment(
            b=b,
            u0=u_horizon,
            u1=u_turn,
            phi0=external.phi_end,
            direction="inward",
            endpoint_event=EventType.TURNING_POINT,
            region=next_region,
        )
        segments = [external, through_in]
        events = [
            RayEvent(EventType.HORIZON, external.phi_end, external.u_end, self.metric.region),
            RayEvent(EventType.REGION_TRANSITION, external.phi_end, external.u_end, next_region),
        ]
        if through_in.endpoint_event == EventType.MAX_PHI:
            events.append(RayEvent(EventType.MAX_PHI, through_in.phi_end, through_in.u_end, next_region))
            return self._through_result(
                b,
                segments,
                events,
                max(err_external, err_through),
                EventType.MAX_PHI,
                hit_horizon=True,
            )

        phi_turn = through_in.phi_end
        events.append(RayEvent(EventType.TURNING_POINT, phi_turn, u_turn, next_region))
        phi_companion_horizon = 2.0 * phi_turn - external.phi_end
        inner_out = self._reflected_segment(
            source=through_in,
            region=next_region,
            radial_direction="outward",
            phi_start=phi_turn,
            phi_end=min(self.options.max_phi, phi_companion_horizon),
            mirror_center=phi_turn,
            endpoint_event=EventType.HORIZON,
        )
        segments.append(inner_out)
        if inner_out.endpoint_event == EventType.MAX_PHI:
            events.append(RayEvent(EventType.MAX_PHI, inner_out.phi_end, inner_out.u_end, next_region))
            return self._through_result(
                b,
                segments,
                events,
                max(err_external, err_through),
                EventType.MAX_PHI,
                hit_horizon=True,
            )

        events.append(RayEvent(EventType.HORIZON, inner_out.phi_end, inner_out.u_end, next_region))
        phi_escape = 2.0 * phi_turn
        companion_out = self._reflected_segment(
            source=external,
            region=next_region,
            radial_direction="outward",
            phi_start=phi_companion_horizon,
            phi_end=min(self.options.max_phi, phi_escape),
            mirror_center=phi_turn,
            endpoint_event=EventType.ESCAPE,
        )
        segments.append(companion_out)
        final_event = companion_out.endpoint_event
        events.append(RayEvent(final_event, companion_out.phi_end, companion_out.u_end, next_region))
        return self._through_result(
            b,
            segments,
            events,
            max(err_external, err_through),
            final_event,
            hit_horizon=True,
        )

    def _through_result(
        self,
        b: float,
        segments: list[RaySegment],
        events: list[RayEvent],
        estimated_error: float,
        termination_event: EventType,
        *,
        hit_horizon: bool,
    ) -> RayResult:
        return RayResult(
            b=b,
            segments=segments,
            events=events,
            critical_status=None,
            diagnostics=RayDiagnostics(
                estimated_error=estimated_error,
                near_critical=False,
                critical_curve_id=None,
                hit_horizon=hit_horizon,
                hit_inner_boundary=False,
                turning_point_count=sum(1 for event in events if event.event_type == EventType.TURNING_POINT),
                max_phi_reached=termination_event == EventType.MAX_PHI,
                termination_reason=termination_event.value,
                residuals={"G_end": _safe_g(self.metric, segments[-1].u_end, b)},
            ),
        )

    def _reflected_segment(
        self,
        source: RaySegment,
        region: str,
        radial_direction: str,
        phi_start: float,
        phi_end: float,
        mirror_center: float,
        endpoint_event: EventType,
    ) -> RaySegment:
        truncated = phi_end < (2.0 * mirror_center - source.phi_start)

        def reflected_u(phi: float, source: RaySegment = source, mirror_center: float = mirror_center) -> float:
            return source.u_at(_clamp_phi_to_segment(2.0 * mirror_center - phi, source))

        return RaySegment(
            region=region,
            radial_direction=radial_direction,
            phi_start=phi_start,
            phi_end=phi_end,
            u_start=reflected_u(phi_start),
            u_end=reflected_u(phi_end),
            endpoint_event=EventType.MAX_PHI if truncated else endpoint_event,
            evaluator=reflected_u,
        )

    def _first_continuation_turning_point(self, b: float, u_horizon: float) -> float | None:
        max_u = self._initial_continuation_max_u(u_horizon)
        for _ in range(8):
            u_turn = self._first_turning_point(b, u_horizon, max_u)
            if u_turn is not None and u_turn > u_horizon:
                return u_turn
            max_u *= 2.0
        return None

    def _initial_continuation_max_u(self, u_horizon: float) -> float:
        horizons = sorted(r for r in self.metric.horizons() if r > 0.0)
        candidates = [2.0 * u_horizon]
        if len(horizons) >= 2:
            candidates.append(1.1 / horizons[0])
        if self.options.inner_boundary_radius is not None:
            candidates.append(1.0 / self.options.inner_boundary_radius)
        return max(candidates)

    def _first_turning_point(self, b: float, u_start: float, u_stop: float) -> float | None:
        return _first_turning_point_scan(
            self.metric,
            b,
            u_start,
            u_stop,
            self.options,
            clip_start_at_zero=True,
        )

    def _first_turning_point_scalar(self, b: float, u_start: float, u_stop: float) -> float | None:
        return _first_turning_point_scalar_scan(
            self.metric,
            b,
            u_start,
            u_stop,
            self.options,
            clip_start_at_zero=True,
        )

    def _make_segment(
        self,
        b: float,
        u0: float,
        u1: float,
        phi0: float,
        direction: str,
        endpoint_event: EventType,
        region: str,
    ) -> tuple[RaySegment, float]:
        from .numerical_metrics import TabulatedMetric

        if isinstance(self.metric, TabulatedMetric):
            stats = _segment_fast_stats(self)
            stats["attempted"] += 1
            try:
                segment, error = self._make_segment_tabulated_fast(
                    b=b,
                    u0=u0,
                    u1=u1,
                    phi0=phi0,
                    direction=direction,
                    endpoint_event=endpoint_event,
                    region=region,
                )
            except _FastPathUnavailable:
                stats["fallback"] += 1
            else:
                stats["accepted"] += 1
                return segment, error

        return self._make_segment_quad_reference(
            b=b,
            u0=u0,
            u1=u1,
            phi0=phi0,
            direction=direction,
            endpoint_event=endpoint_event,
            region=region,
        )

    def _make_segment_tabulated_fast(
        self,
        b: float,
        u0: float,
        u1: float,
        phi0: float,
        direction: str,
        endpoint_event: EventType,
        region: str,
    ) -> tuple[RaySegment, float]:
        outward = u1 < u0
        span = abs(u1 - u0)

        def integrate_direct(target_u: float) -> tuple[float, float]:
            if target_u < u0:
                raise _FastPathUnavailable("direct fast path target precedes start")

            def evaluator(u_values: np.ndarray) -> np.ndarray:
                return _tabulated_fast_integrand(self.metric, u_values, b)

            return _integrate_gl_pair(evaluator, u0, target_u, self.options)

        def integrate_inward_turn(target_u: float) -> tuple[float, float]:
            if span <= 0.0:
                return 0.0, 0.0
            target_x = 1.0 - math.sqrt(max(u1 - target_u, 0.0) / span)

            def evaluator(x_values: np.ndarray) -> np.ndarray:
                distance = 1.0 - x_values
                u_values = u1 - span * distance**2
                return 2.0 * span * distance * _tabulated_fast_integrand(
                    self.metric,
                    u_values,
                    b,
                )

            return _integrate_gl_pair(evaluator, 0.0, target_x, self.options)

        def integrate_outward(target_u: float) -> tuple[float, float]:
            if span <= 0.0:
                return 0.0, 0.0
            target_x = math.sqrt(max(u0 - target_u, 0.0) / span)

            def evaluator(x_values: np.ndarray) -> np.ndarray:
                u_values = u0 - span * x_values**2
                return 2.0 * span * x_values * _tabulated_fast_integrand(
                    self.metric,
                    u_values,
                    b,
                )

            return _integrate_gl_pair(evaluator, 0.0, target_x, self.options)

        def integrate_to(target_u: float) -> tuple[float, float]:
            if endpoint_event == EventType.TURNING_POINT and not outward:
                return integrate_inward_turn(target_u)
            if outward:
                return integrate_outward(target_u)
            return integrate_direct(target_u)

        phi_width, error = integrate_to(u1)
        if not math.isfinite(phi_width) or phi_width < 0.0:
            raise _FastPathUnavailable("invalid fast segment width")

        full_phi1 = phi0 + phi_width
        phi1 = min(self.options.max_phi, full_phi1)
        truncated = phi1 < full_phi1

        final_endpoint_event = EventType.MAX_PHI if truncated else endpoint_event
        reference_cache: dict[str, tuple[RaySegment, float]] = {}
        fast_u_end: float | None = None

        def reference_segment() -> RaySegment:
            cached = reference_cache.get("segment")
            if cached is None:
                cached = self._make_segment_quad_reference(
                    b=b,
                    u0=u0,
                    u1=u1,
                    phi0=phi0,
                    direction=direction,
                    endpoint_event=endpoint_event,
                    region=region,
                )
                reference_cache["segment"] = cached
            return cached[0]

        def ensure_reference_matches_fast(ref: RaySegment) -> None:
            if ref.endpoint_event != final_endpoint_event:
                raise _FastPathUnavailable("reference fallback endpoint event differs from fast segment")
            if not math.isclose(ref.phi_start, phi0, rel_tol=0.0, abs_tol=1e-12):
                raise _FastPathUnavailable("reference fallback start differs from fast segment")
            if not math.isclose(ref.phi_end, phi1, rel_tol=1e-8, abs_tol=1e-8):
                raise _FastPathUnavailable("reference fallback end differs from fast segment")
            if fast_u_end is not None and not math.isclose(ref.u_end, fast_u_end, rel_tol=1e-8, abs_tol=1e-8):
                raise _FastPathUnavailable("reference fallback u_end differs from fast segment")

        def angle_to_u(phi: float, *, allow_reference_fallback: bool) -> float:
            target = phi - phi0
            if target <= 0.0:
                return u0
            if target >= phi_width:
                return u1

            def residual(u: float) -> float:
                value, _ = integrate_to(u)
                return value - target

            low, high = (u1, u0) if outward else (u0, u1)
            try:
                return brentq(
                    residual,
                    low,
                    high,
                    xtol=self.options.root_atol,
                    rtol=self.options.root_rtol,
                )
            except (_FastPathUnavailable, ValueError, RuntimeError) as exc:
                if not allow_reference_fallback:
                    raise _FastPathUnavailable("fast angle inversion failed") from exc
                _segment_fast_stats(self)["late_fallback"] += 1
                ref = reference_segment()
                ensure_reference_matches_fast(ref)
                return ref.u_at(_clamp_phi_to_segment(phi, ref, atol=1e-8))

        u_end = angle_to_u(phi1, allow_reference_fallback=False)
        fast_u_end = u_end

        def evaluator(phi: float) -> float:
            return angle_to_u(phi, allow_reference_fallback=True)

        return (
            RaySegment(
                region=region,
                radial_direction=direction,
                phi_start=phi0,
                phi_end=phi1,
                u_start=u0,
                u_end=u_end,
                endpoint_event=final_endpoint_event,
                evaluator=evaluator,
            ),
            error,
        )

    def _make_segment_quad_reference(
        self,
        b: float,
        u0: float,
        u1: float,
        phi0: float,
        direction: str,
        endpoint_event: EventType,
        region: str,
    ) -> tuple[RaySegment, float]:
        return make_quad_segment(
            metric=self.metric,
            b=b,
            u0=u0,
            u1=u1,
            phi0=phi0,
            direction=direction,
            endpoint_event=endpoint_event,
            region=region,
            options=self.options,
            safe_g=_safe_g,
            regularize_outward_start=True,
        )


@dataclass
class OdeTrajectorySolver:
    metric: StaticSphericalMetric
    observer: InfinityObserver | FiniteStaticObserver
    trace_policy: TracePolicy = OutsideTracePolicy()
    options: SolverOptions = SolverOptions()

    def _integrate_outbound_ode(
        self,
        b: float,
        inbound: RaySegment,
        u_outward_stop: float,
        outward_event: EventType,
    ) -> RaySegment:
        """Outbound leg when ``r_max < math.inf`` (e.g. RNdS with ``Λ>0``).

        Delegates to :meth:`QuadTransferSolver._make_segment` with
        ``direction='outward'``, which regularises the ``sqrt(G)=0`` singularity
        at the turning point via the change of variables ``u = u_turn - span*x^2``.
        ``solve_ivp`` cannot launch from a turning point directly because the
        right-hand side ``du/dphi = ±sqrt(G(u, b))`` is zero there. Plan decision
        8 ("``OdeTrajectorySolver`` follows ``QuadTransferSolver``'s static-domain
        fix") is honoured by reusing this regularised quadrature for the outbound
        leg; the inbound leg remains an independent ``solve_ivp`` integration.
        """
        quad_solver = QuadTransferSolver(
            self.metric, self.observer, self.trace_policy, self.options
        )
        segment, _err = quad_solver._make_segment(
            b=b,
            u0=inbound.u_end,
            u1=u_outward_stop,
            phi0=inbound.phi_end,
            direction="outward",
            endpoint_event=outward_event,
            region=self.metric.region,
        )
        return segment

    def trace_b(self, b: float) -> RayResult:
        if not isinstance(self.trace_policy, OutsideTracePolicy):
            raise ValueError("OdeTrajectorySolver only supports OutsideTracePolicy in exterior static coordinates")
        _guard_tabulated_metric_combinations(self.metric, self.observer, self.trace_policy)

        quad_result = QuadTransferSolver(self.metric, self.observer, self.trace_policy, self.options).trace_b(b)
        if quad_result.diagnostics.near_critical or not quad_result.segments:
            return quad_result

        first = quad_result.segments[0]
        u0 = max(first.u_start, 1e-12)
        direction = 1.0 if first.radial_direction == "inward" else -1.0

        def rhs(phi: float, y: np.ndarray) -> np.ndarray:
            g = max(_safe_g(self.metric, float(y[0]), b), 0.0)
            return np.array([direction * math.sqrt(g)])

        try:
            u_inward_stop, inward_event = QuadTransferSolver(
                self.metric, self.observer, self.trace_policy, self.options
            )._inward_stop_u()
        except ValueError:
            u_inward_stop = None
            inward_event = None

        def event_inward(phi: float, y: np.ndarray) -> float:
            if u_inward_stop is None:
                return 1.0
            return u_inward_stop - float(y[0])

        event_inward.terminal = True
        event_inward.direction = -1.0

        def event_turning(phi: float, y: np.ndarray) -> float:
            return _safe_g(self.metric, float(y[0]), b)

        event_turning.terminal = True
        event_turning.direction = -1.0
        events = [event_turning] if u_inward_stop is None else [event_inward, event_turning]

        solution = solve_ivp(
            rhs,
            (0.0, self.options.max_phi),
            np.array([u0]),
            events=events,
            rtol=1e-9,
            atol=1e-11,
            dense_output=True,
            max_step=0.01,
        )
        inward_index = 0 if u_inward_stop is not None else None
        turning_index = 1 if u_inward_stop is not None else 0
        hit_inward = inward_index is not None and len(solution.t_events[inward_index]) > 0
        hit_turning = len(solution.t_events[turning_index]) > 0
        if hit_inward:
            event_type = inward_event if inward_event is not None else EventType.HORIZON
        elif hit_turning:
            event_type = EventType.TURNING_POINT
        else:
            event_type = EventType.MAX_PHI
        if hit_inward:
            phi_end = float(solution.t_events[inward_index][0])
        elif hit_turning:
            phi_end = float(solution.t_events[turning_index][0])
        else:
            phi_end = float(solution.t[-1])
        u_end = float(solution.sol(phi_end)[0])
        if len(solution.t) >= 4:
            du_dphi = np.gradient(solution.y[0], solution.t)
            residuals = [
                abs(float(du) ** 2 - max(_safe_g(self.metric, float(u), b), 0.0))
                for du, u in zip(du_dphi[1:-1], solution.y[0][1:-1])
            ]
        else:
            residuals = [math.inf]
        segment = RaySegment(
            region=self.metric.region,
            radial_direction="inward",
            phi_start=0.0,
            phi_end=phi_end,
            u_start=u0,
            u_end=u_end,
            endpoint_event=event_type,
            evaluator=lambda phi, sol=solution.sol: float(sol(phi)[0]),
        )
        diagnostics = RayDiagnostics(
            estimated_error=max(residuals),
            near_critical=quad_result.diagnostics.near_critical,
            critical_curve_id=quad_result.diagnostics.critical_curve_id,
            hit_horizon=event_type == EventType.HORIZON,
            hit_inner_boundary=event_type == EventType.INNER_BOUNDARY,
            turning_point_count=1 if hit_turning else 0,
            max_phi_reached=event_type == EventType.MAX_PHI,
            termination_reason=horizon_termination_reason(self.metric, event_type, u_end),
            residuals={"max_radial_equation_residual": max(residuals)},
        )
        if hit_turning:
            try:
                u_outward_stop, outward_event = QuadTransferSolver(
                    self.metric, self.observer, self.trace_policy, self.options
                )._outward_stop_u()
            except ValueError:
                u_outward_stop = 0.0
                outward_event = EventType.ESCAPE

            if u_outward_stop > 0.0:
                outbound = self._integrate_outbound_ode(
                    b, segment, u_outward_stop, outward_event
                )
            else:
                outbound = _mirror_escape_segment(segment, self.options.max_phi)
            final_event = outbound.endpoint_event
            diagnostics = RayDiagnostics(
                estimated_error=max(residuals),
                near_critical=quad_result.diagnostics.near_critical,
                critical_curve_id=quad_result.diagnostics.critical_curve_id,
                hit_horizon=final_event == EventType.HORIZON,
                hit_inner_boundary=False,
                turning_point_count=1,
                max_phi_reached=final_event == EventType.MAX_PHI,
                termination_reason=horizon_termination_reason(self.metric, final_event, outbound.u_end),
                residuals={"max_radial_equation_residual": max(residuals)},
            )
            return RayResult(
                b=b,
                segments=[segment, outbound],
                events=[
                    RayEvent(EventType.TURNING_POINT, segment.phi_end, segment.u_end, self.metric.region),
                    horizon_event(self.metric, final_event, outbound.phi_end, outbound.u_end, self.metric.region),
                ],
                critical_status=quad_result.critical_status,
                diagnostics=diagnostics,
            )
        return RayResult(
            b=b,
            segments=[segment],
            events=[horizon_event(self.metric, event_type, phi_end, u_end, self.metric.region)],
            critical_status=quad_result.critical_status,
            diagnostics=diagnostics,
        )


@dataclass
class HamiltonianTrajectorySolver:
    metric: StaticSphericalMetric
    observer: FiniteStaticObserver
    trace_policy: OutsideTracePolicy = OutsideTracePolicy()
    options: SolverOptions = SolverOptions()
    energy: float = 1.0

    def trace_b(self, b: float) -> RayResult:
        if not isinstance(self.observer, FiniteStaticObserver):
            raise TypeError("HamiltonianTrajectorySolver requires a finite static observer")
        if not isinstance(self.trace_policy, OutsideTracePolicy):
            raise ValueError("HamiltonianTrajectorySolver only supports OutsideTracePolicy in exterior static coordinates")
        if b <= 0.0:
            raise ValueError("b must be positive")
        _guard_tabulated_metric_combinations(self.metric, self.observer, self.trace_policy)

        critical_curve = _near_critical(self.metric, b, self.options.critical_exclusion)
        if critical_curve is not None:
            return RayResult(
                b=b,
                segments=[],
                events=[],
                critical_status=critical_curve,
                diagnostics=RayDiagnostics(
                    estimated_error=math.inf,
                    near_critical=True,
                    critical_curve_id=critical_curve.id,
                    hit_horizon=False,
                    hit_inner_boundary=False,
                    turning_point_count=0,
                    max_phi_reached=False,
                    termination_reason="near_critical",
                    residuals={},
                ),
            )
        e = self.energy
        l = b * e
        r0 = self.observer.r_obs
        a0 = self.metric.A(r0)
        b0 = self.metric.B(r0)
        radicand = b0 * (e**2 / a0 - l**2 / r0**2)
        if radicand < 0.0:
            raise ValueError("initial data is outside the null cone")
        pr0 = -math.sqrt(radicand)
        pt0 = -e
        pphi0 = l

        try:
            u_inward_stop, inward_event = QuadTransferSolver(
                self.metric, self.observer, self.trace_policy, self.options
            )._inward_stop_u()
        except ValueError:
            u_inward_stop = None
            inward_event = None
        r_stop = 1.0 / u_inward_stop if u_inward_stop is not None else None

        def hamiltonian(r: float, pr: float, pt: float, pphi: float) -> float:
            return 0.5 * (-pt**2 / self.metric.A(r) + pr**2 / self.metric.B(r) + pphi**2 / r**2)

        def rhs(lambda_value: float, y: np.ndarray) -> np.ndarray:
            r, pr, phi, pt, pphi = map(float, y)
            a = self.metric.A(r)
            b_metric = self.metric.B(r)
            dr = pr / b_metric
            dpr = (
                -pt**2 * self.metric.dA_dr(r) / (2.0 * a**2)
                + pr**2 * self.metric.dB_dr(r) / (2.0 * b_metric**2)
                + pphi**2 / r**3
            )
            dphi = pphi / r**2
            dpt = 0.0
            dpphi = 0.0
            return np.array([dr, dpr, dphi, dpt, dpphi])

        def make_u_evaluator(phi_values: np.ndarray, r_values: np.ndarray):
            phi_grid = np.asarray(phi_values, dtype=float)
            r_grid = np.asarray(r_values, dtype=float)
            finite = np.isfinite(phi_grid) & np.isfinite(r_grid) & (r_grid > 0.0)
            phi_grid = phi_grid[finite]
            r_grid = r_grid[finite]
            if len(phi_grid) == 0:
                raise RuntimeError("Hamiltonian integration produced no finite interpolation samples")
            order = np.argsort(phi_grid)
            phi_grid = phi_grid[order]
            r_grid = r_grid[order]
            keep = np.concatenate(([True], np.diff(phi_grid) > 1e-12))
            phi_grid = phi_grid[keep]
            r_grid = r_grid[keep]

            def evaluator(phi: float, phi_grid: np.ndarray = phi_grid, r_grid: np.ndarray = r_grid) -> float:
                return float(1.0 / np.interp(phi, phi_grid, r_grid))

            return evaluator

        def event_inward(lambda_value: float, y: np.ndarray) -> float:
            if r_stop is None:
                return 1.0
            return float(y[0]) - r_stop

        event_inward.terminal = True
        event_inward.direction = -1.0

        def event_turning(lambda_value: float, y: np.ndarray) -> float:
            return float(y[1])

        event_turning.terminal = True
        event_turning.direction = 1.0

        solution = solve_ivp(
            rhs,
            (0.0, 10_000.0),
            np.array([r0, pr0, 0.0, pt0, pphi0]),
            events=[event_inward, event_turning],
            rtol=1e-9,
            atol=1e-11,
            dense_output=True,
            max_step=0.5,
        )

        hit_inward = len(solution.t_events[0]) > 0
        hit_turning = len(solution.t_events[1]) > 0
        if hit_inward:
            event_type = inward_event if inward_event is not None else EventType.HORIZON
        elif hit_turning:
            event_type = EventType.TURNING_POINT
        else:
            event_type = EventType.MAX_PHI
        lambda_end = float(solution.t[-1])
        r_end, pr_end, phi_end, pt_end, pphi_end = map(float, solution.y[:, -1])
        constraints = [
            abs(hamiltonian(float(r), float(pr), float(pt), float(pphi)))
            for r, pr, pt, pphi in zip(solution.y[0], solution.y[1], solution.y[3], solution.y[4])
        ]
        energy_values = -solution.y[3]
        angular_momentum_values = solution.y[4]
        impact_parameter_values = angular_momentum_values / energy_values
        energy_residual = float(np.max(np.abs(energy_values - e)))
        angular_momentum_residual = float(np.max(np.abs(angular_momentum_values - l)))
        impact_parameter_residual = float(np.max(np.abs(impact_parameter_values - b)))

        segment = RaySegment(
            region=self.metric.region,
            radial_direction="inward",
            phi_start=0.0,
            phi_end=phi_end,
            u_start=1.0 / r0,
            u_end=1.0 / r_end,
            endpoint_event=event_type,
            evaluator=make_u_evaluator(solution.y[2], solution.y[0]),
        )
        diagnostics = RayDiagnostics(
            estimated_error=max(constraints),
            near_critical=critical_curve is not None,
            critical_curve_id=None if critical_curve is None else critical_curve.id,
            hit_horizon=event_type == EventType.HORIZON,
            hit_inner_boundary=event_type == EventType.INNER_BOUNDARY,
            turning_point_count=1 if hit_turning else 0,
            max_phi_reached=event_type == EventType.MAX_PHI,
            termination_reason=horizon_termination_reason(self.metric, event_type, 1.0 / r_end),
            residuals={
                "max_hamiltonian_constraint": max(constraints),
                "energy_residual": energy_residual,
                "angular_momentum_residual": angular_momentum_residual,
                "impact_parameter_residual": impact_parameter_residual,
                "turning_point_crossings": 1 if hit_turning else 0,
                "lambda_end": lambda_end,
            },
        )
        if hit_turning:
            lambda_turn = float(solution.t_events[1][0])
            y0_out = np.asarray(solution.sol(lambda_turn), dtype=float).copy()
            y0_out[1] = abs(float(y0_out[1]))

            def event_max_phi(lambda_value: float, y: np.ndarray) -> float:
                return self.options.max_phi - float(y[2])

            event_max_phi.terminal = True
            event_max_phi.direction = -1.0

            _, r_max = self.metric.valid_radial_domain()
            if not math.isfinite(r_max):
                outbound = _mirror_escape_segment(segment, self.options.max_phi)
                final_event = outbound.endpoint_event
                diagnostics = RayDiagnostics(
                    estimated_error=max(constraints),
                    near_critical=critical_curve is not None,
                    critical_curve_id=None if critical_curve is None else critical_curve.id,
                    hit_horizon=False,
                    hit_inner_boundary=False,
                    turning_point_count=1,
                    max_phi_reached=final_event == EventType.MAX_PHI,
                    termination_reason=horizon_termination_reason(self.metric, final_event, outbound.u_end),
                    residuals={
                        "max_hamiltonian_constraint": max(constraints),
                        "energy_residual": energy_residual,
                        "angular_momentum_residual": angular_momentum_residual,
                        "impact_parameter_residual": impact_parameter_residual,
                        "turning_point_crossings": 1,
                        "lambda_end": lambda_end,
                    },
                )
                return RayResult(
                    b=b,
                    segments=[segment, outbound],
                    events=[
                        RayEvent(EventType.TURNING_POINT, segment.phi_end, segment.u_end, self.metric.region),
                        horizon_event(self.metric, final_event, outbound.phi_end, outbound.u_end, self.metric.region),
                    ],
                    critical_status=critical_curve,
                    diagnostics=diagnostics,
                )

            outbound_events = [event_max_phi]
            outbound_event_types = [EventType.MAX_PHI]
            r_out_stop = r_max * (1.0 - self.options.horizon_buffer)
            outbound_boundary_event = _classify_radial_event(
                self.metric,
                r_max,
                default=EventType.OUTER_BOUNDARY,
            )

            def event_outward_boundary(lambda_value: float, y: np.ndarray) -> float:
                return float(y[0]) - r_out_stop

            event_outward_boundary.terminal = True
            event_outward_boundary.direction = 1.0
            outbound_events.append(event_outward_boundary)
            outbound_event_types.append(outbound_boundary_event)

            outbound_solution = solve_ivp(
                rhs,
                (lambda_turn, 10_000.0),
                y0_out,
                events=outbound_events,
                rtol=1e-9,
                atol=1e-11,
                dense_output=True,
                max_step=0.5,
            )
            event_hits = [
                (float(values[0]), outbound_event_types[index])
                for index, values in enumerate(outbound_solution.t_events)
                if len(values) > 0
            ]
            final_event = min(event_hits, key=lambda item: item[0])[1] if event_hits else EventType.SOLVER_FAILURE
            r_out_end, _pr_out_end, phi_out_end, _pt_out_end, _pphi_out_end = map(float, outbound_solution.y[:, -1])
            outbound_constraints = [
                abs(hamiltonian(float(r), float(pr), float(pt), float(pphi)))
                for r, pr, pt, pphi in zip(
                    outbound_solution.y[0],
                    outbound_solution.y[1],
                    outbound_solution.y[3],
                    outbound_solution.y[4],
                )
            ]
            all_constraints = constraints + outbound_constraints
            outbound_energy_values = -outbound_solution.y[3]
            outbound_angular_momentum_values = outbound_solution.y[4]
            outbound_impact_parameter_values = outbound_angular_momentum_values / outbound_energy_values
            energy_residual = max(energy_residual, float(np.max(np.abs(outbound_energy_values - e))))
            angular_momentum_residual = max(
                angular_momentum_residual,
                float(np.max(np.abs(outbound_angular_momentum_values - l))),
            )
            impact_parameter_residual = max(
                impact_parameter_residual,
                float(np.max(np.abs(outbound_impact_parameter_values - b))),
            )
            lambda_end = float(outbound_solution.t[-1])
            outbound = RaySegment(
                region=self.metric.region,
                radial_direction="outward",
                phi_start=float(y0_out[2]),
                phi_end=phi_out_end,
                u_start=1.0 / float(y0_out[0]),
                u_end=1.0 / r_out_end,
                endpoint_event=final_event,
                evaluator=make_u_evaluator(outbound_solution.y[2], outbound_solution.y[0]),
            )
            diagnostics = RayDiagnostics(
                estimated_error=max(all_constraints),
                near_critical=critical_curve is not None,
                critical_curve_id=None if critical_curve is None else critical_curve.id,
                hit_horizon=final_event == EventType.HORIZON,
                hit_inner_boundary=False,
                turning_point_count=1,
                max_phi_reached=final_event == EventType.MAX_PHI,
                termination_reason=horizon_termination_reason(self.metric, final_event, outbound.u_end),
                residuals={
                    "max_hamiltonian_constraint": max(all_constraints),
                    "energy_residual": energy_residual,
                    "angular_momentum_residual": angular_momentum_residual,
                    "impact_parameter_residual": impact_parameter_residual,
                    "turning_point_crossings": 1,
                    "lambda_end": lambda_end,
                },
            )
            return RayResult(
                b=b,
                segments=[segment, outbound],
                events=[
                    RayEvent(EventType.TURNING_POINT, segment.phi_end, segment.u_end, self.metric.region),
                    horizon_event(self.metric, final_event, outbound.phi_end, outbound.u_end, self.metric.region),
                ],
                critical_status=critical_curve,
                diagnostics=diagnostics,
            )
        return RayResult(
            b=b,
            segments=[segment],
            events=[horizon_event(self.metric, event_type, phi_end, 1.0 / r_end, self.metric.region)],
            critical_status=critical_curve,
            diagnostics=diagnostics,
        )


__all__ = [
    "HamiltonianTrajectorySolver",
    "OdeTrajectorySolver",
    "OutsideTracePolicy",
    "QuadTransferSolver",
    "SolverOptions",
    "ThroughTracePolicy",
    "TracePolicy",
]
