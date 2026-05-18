from __future__ import annotations

import math
from dataclasses import dataclass, replace

import numpy as np
from scipy.integrate import quad, solve_ivp
from scipy.optimize import brentq

from .diagnostics import CriticalCurve, EventType, RayDiagnostics, RayEvent, RaySegment
from .junctions import (
    JunctionRayResult,
    SegmentConstants,
    StaticJunctionSpacetime,
    compare_junction_results,
    match_static_shell,
)
from .observers import FiniteStaticObserver
from .policies import SolverOptions
from .solvers import _classify_radial_event


def _safe_g_metric(metric, u: float, b: float) -> float:
    value = metric.G(float(u), float(b))
    if not math.isfinite(value):
        return -math.inf
    return value


def _near_critical_metric(metric, b: float, exclusion: float) -> CriticalCurve | None:
    for curve in metric.critical_curves():
        if curve.reachable_from_observer and abs(b - curve.b_crit) / curve.b_crit < exclusion:
            return curve
    return None


def _near_critical_result(
    *,
    coordinate: float,
    coordinate_name: str,
    b_plus: float,
    backend_name: str,
    critical_curve: CriticalCurve,
    physics_warnings: list[str],
    residuals: dict[str, float] | None = None,
    segments: list[RaySegment] | None = None,
    segment_constants: list[SegmentConstants] | None = None,
    events: list[RayEvent] | None = None,
    shell_crossings: list[object] | None = None,
) -> JunctionRayResult:
    diagnostics = RayDiagnostics(
        estimated_error=math.inf,
        near_critical=True,
        critical_curve_id=critical_curve.id,
        hit_horizon=False,
        hit_inner_boundary=False,
        turning_point_count=0,
        max_phi_reached=False,
        termination_reason="near_critical",
        residuals={"unresolved": 1.0, **({} if residuals is None else residuals)},
    )
    return JunctionRayResult(
        input_coordinate=coordinate,
        coordinate_name=coordinate_name,
        b_plus=b_plus,
        segments=[] if segments is None else segments,
        segment_constants=[] if segment_constants is None else segment_constants,
        events=[] if events is None else events,
        shell_crossings=[] if shell_crossings is None else shell_crossings,
        physics_warnings=physics_warnings,
        diagnostics=diagnostics,
        backend_name=backend_name,
        critical_status=critical_curve,
    )


def _validate_inner_boundary_inside_shell(options: SolverOptions, shell_radius: float) -> None:
    if options.inner_boundary_radius is not None and not 0.0 < options.inner_boundary_radius < shell_radius:
        raise ValueError("inner_boundary_radius must lie inside the shell")


@dataclass
class StaticJunctionTransferSolver:
    junction: StaticJunctionSpacetime
    observer: FiniteStaticObserver
    options: SolverOptions = SolverOptions()
    energy: float = 1.0

    def trace_alpha(self, alpha: float) -> JunctionRayResult:
        return self.trace_b(self.observer.impact_parameter(alpha), input_coordinate=alpha, coordinate_name="alpha")

    def trace_screen_radius(self, rho: float) -> JunctionRayResult:
        if rho < 0.0:
            raise ValueError("rho must be non-negative")
        alpha = math.atan(rho / self.observer.r_obs)
        return self.trace_b(self.observer.impact_parameter(alpha), input_coordinate=rho, coordinate_name="rho")

    def _make_segment(
        self,
        *,
        metric,
        b: float,
        u0: float,
        u1: float,
        phi0: float,
        direction: str,
        endpoint_event: EventType,
        region: str,
    ) -> tuple[RaySegment, float]:
        if u1 < u0:
            low, high = u1, u0
        else:
            low, high = u0, u1

        def integrand(u: float) -> float:
            return 1.0 / math.sqrt(max(_safe_g_metric(metric, u, b), 1e-300))

        def integrate_to(target_u: float) -> tuple[float, float]:
            if endpoint_event == EventType.TURNING_POINT and abs(u1 - u0) > 0.0:
                span = abs(u1 - u0)
                distance_to_turn = abs(u1 - target_u)
                target_x = 1.0 - math.sqrt(max(distance_to_turn, 0.0) / span)

                def transformed_integrand(x: float) -> float:
                    distance = 1.0 - x
                    if u1 >= u0:
                        u = u1 - span * distance**2
                    else:
                        u = u1 + span * distance**2
                    return 2.0 * span * distance * integrand(u)

                return quad(
                    transformed_integrand,
                    0.0,
                    target_x,
                    epsabs=self.options.quad_epsabs,
                    epsrel=self.options.quad_epsrel,
                    limit=200,
                )
            value, error = quad(
                integrand,
                low,
                target_u if u1 >= u0 else high,
                epsabs=self.options.quad_epsabs,
                epsrel=self.options.quad_epsrel,
                limit=200,
            )
            if u1 < u0:
                value, error = quad(
                    integrand,
                    target_u,
                    high,
                    epsabs=self.options.quad_epsabs,
                    epsrel=self.options.quad_epsrel,
                    limit=200,
                )
            return value, error

        phi_width, error = integrate_to(u1)
        full_phi1 = phi0 + phi_width
        phi1 = min(self.options.max_phi, full_phi1)
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

            return brentq(residual, low, high, xtol=self.options.root_atol, rtol=self.options.root_rtol)

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

    def _first_turning_point(self, metric, b: float, u_start: float, u_stop: float) -> float | None:
        if math.isclose(u_start, u_stop):
            return None
        low, high = sorted([u_start, u_stop])
        grid = np.linspace(low, high, self.options.max_brackets)
        if u_stop < u_start:
            grid = grid[::-1]
        previous_u = float(grid[0])
        previous_g = _safe_g_metric(metric, previous_u, b)
        for current_u_value in grid[1:]:
            current_u = float(current_u_value)
            current_g = _safe_g_metric(metric, current_u, b)
            if previous_g > 0.0 and current_g <= 0.0:
                bracket = sorted([previous_u, current_u])
                return brentq(
                    lambda u: metric.G(u, b),
                    bracket[0],
                    bracket[1],
                    xtol=self.options.root_atol,
                    rtol=self.options.root_rtol,
                )
            previous_u = current_u
            previous_g = current_g
        return None

    def _region_boundary_u(self, region: str, direction: str) -> tuple[float, EventType]:
        metric = self.junction.metric_for_region(region)
        shell_u = 1.0 / self.junction.shell.radius
        r_min, r_max = metric.valid_radial_domain()
        if region == "outer" and direction == "inward":
            if r_min > 0.0:
                horizon_u = 1.0 / r_min * (1.0 - self.options.horizon_buffer)
                if shell_u <= horizon_u:
                    return shell_u, EventType.SHELL_CROSSING
                event = _classify_radial_event(metric, r_min, default=EventType.INNER_BOUNDARY)
                return horizon_u, event
            return shell_u, EventType.SHELL_CROSSING
        if region == "inner" and direction == "outward":
            if math.isfinite(r_max):
                outer_boundary_u = 1.0 / r_max * (1.0 + self.options.horizon_buffer)
                if shell_u >= outer_boundary_u:
                    return shell_u, EventType.SHELL_CROSSING
                event = _classify_radial_event(metric, r_max, default=EventType.OUTER_BOUNDARY)
                return outer_boundary_u, event
            return shell_u, EventType.SHELL_CROSSING
        if direction == "inward":
            if self.options.inner_boundary_radius is not None:
                return 1.0 / self.options.inner_boundary_radius, EventType.INNER_BOUNDARY
            if r_min > 0.0:
                event = _classify_radial_event(metric, r_min, default=EventType.INNER_BOUNDARY)
                return 1.0 / r_min * (1.0 - self.options.horizon_buffer), event
            return max(shell_u * 2.0, 1.0), EventType.INNER_BOUNDARY
        if math.isfinite(r_max):
            event = _classify_radial_event(metric, r_max, default=EventType.OUTER_BOUNDARY)
            return 1.0 / r_max * (1.0 + self.options.horizon_buffer), event
        return 0.0, EventType.ESCAPE

    def trace_b(
        self,
        b_plus: float,
        *,
        input_coordinate: float | None = None,
        coordinate_name: str = "b_plus",
    ) -> JunctionRayResult:
        if not isinstance(self.observer, FiniteStaticObserver):
            raise TypeError("StaticJunctionTransferSolver requires a finite static observer")
        if b_plus <= 0.0:
            raise ValueError("b_plus must be positive")
        if self.observer.r_obs <= self.junction.shell.radius:
            raise ValueError("observer must be outside the shell")
        _validate_inner_boundary_inside_shell(self.options, self.junction.shell.radius)

        coordinate = b_plus if input_coordinate is None else input_coordinate
        physics_warnings = list(self.junction.physics_diagnostics(self.observer.r_obs).warnings)
        outer_curve = _near_critical_metric(self.junction.outer_metric, b_plus, self.options.critical_exclusion)
        if outer_curve is not None:
            return _near_critical_result(
                coordinate=coordinate,
                coordinate_name=coordinate_name,
                b_plus=b_plus,
                backend_name="transfer",
                critical_curve=outer_curve,
                physics_warnings=physics_warnings,
            )

        region = "outer"
        radial_direction = "inward"
        phi = 0.0
        u = self.observer.u_start
        E = self.energy
        L = b_plus * E
        b = b_plus
        segments: list[RaySegment] = []
        segment_constants: list[SegmentConstants] = []
        events: list[RayEvent] = []
        shell_crossings = []
        matching_residuals: dict[str, float] = {}
        shell_matching_residuals: dict[str, float] = {}
        estimated_error = 0.0
        turning_count = 0
        terminal_event = EventType.MAX_PHI

        for _ in range(64):
            metric = self.junction.metric_for_region(region)
            boundary_u, boundary_event = self._region_boundary_u(region, radial_direction)
            u_turn = self._first_turning_point(metric, b, u, boundary_u)
            if u_turn is not None:
                target_u = u_turn
                target_event = EventType.TURNING_POINT
            else:
                target_u = boundary_u
                target_event = boundary_event
            segment, error = self._make_segment(
                metric=metric,
                b=b,
                u0=u,
                u1=target_u,
                phi0=phi,
                direction=radial_direction,
                endpoint_event=target_event,
                region=region,
            )
            segments.append(segment)
            segment_constants.append(SegmentConstants(region=region, E=E, L=L, b=b))
            estimated_error = max(estimated_error, abs(error))
            phi = segment.phi_end
            u = segment.u_end
            endpoint = segment.endpoint_event
            events.append(RayEvent(endpoint, phi, u, region))

            if endpoint == EventType.MAX_PHI:
                terminal_event = endpoint
                break
            if endpoint == EventType.TURNING_POINT:
                turning_count += 1
                radial_direction = "outward" if radial_direction == "inward" else "inward"
                continue
            if endpoint == EventType.SHELL_CROSSING:
                from_constants = SegmentConstants(region=region, E=E, L=L, b=b)
                next_region = self.junction.other_region(region)
                to_constants = match_static_shell(
                    self.junction,
                    from_region=region,
                    to_region=next_region,
                    E=E,
                    L=L,
                    b=b,
                )
                crossing = self.junction.shell_crossing_record(
                    phi=phi,
                    radial_direction=radial_direction,
                    before=from_constants,
                    after=to_constants,
                )
                shell_crossings.append(crossing)
                matching_residuals = self.junction.shell_matching_residuals(
                    from_region=region,
                    to_region=next_region,
                    before=from_constants,
                    after=to_constants,
                )
                crossing_index = len(shell_crossings) - 1
                for key, value in matching_residuals.items():
                    shell_matching_residuals[f"shell_{crossing_index}_{key}"] = value
                    shell_matching_residuals[f"max_{key}"] = max(shell_matching_residuals.get(f"max_{key}", 0.0), value)
                region = next_region
                E = to_constants.E
                L = to_constants.L
                b = to_constants.b
                next_curve = _near_critical_metric(
                    self.junction.metric_for_region(region),
                    b,
                    self.options.critical_exclusion,
                )
                if next_curve is not None:
                    residuals = {
                        "shell_crossing_count": float(len(shell_crossings)),
                        **shell_matching_residuals,
                        **matching_residuals,
                    }
                    return _near_critical_result(
                        coordinate=coordinate,
                        coordinate_name=coordinate_name,
                        b_plus=b_plus,
                        backend_name="transfer",
                        critical_curve=next_curve,
                        physics_warnings=physics_warnings,
                        residuals=residuals,
                        segments=segments,
                        segment_constants=segment_constants,
                        events=events,
                        shell_crossings=shell_crossings,
                    )
                continue
            terminal_event = endpoint
            break
        else:
            terminal_event = EventType.SOLVER_FAILURE
            events.append(RayEvent(EventType.SOLVER_FAILURE, phi, u, region, "maximum junction events exceeded"))

        diagnostics = RayDiagnostics(
            estimated_error=estimated_error,
            near_critical=False,
            critical_curve_id=None,
            hit_horizon=terminal_event == EventType.HORIZON,
            hit_inner_boundary=terminal_event == EventType.INNER_BOUNDARY,
            turning_point_count=turning_count,
            max_phi_reached=terminal_event == EventType.MAX_PHI,
            termination_reason=terminal_event.value,
            residuals={"shell_crossing_count": float(len(shell_crossings)), **shell_matching_residuals, **matching_residuals},
        )
        return JunctionRayResult(
            input_coordinate=coordinate,
            coordinate_name=coordinate_name,
            b_plus=b_plus,
            segments=segments,
            segment_constants=segment_constants,
            events=events,
            shell_crossings=shell_crossings,
            physics_warnings=physics_warnings,
            diagnostics=diagnostics,
            backend_name="transfer",
        )


@dataclass
class StaticJunctionHamiltonianSolver:
    junction: StaticJunctionSpacetime
    observer: FiniteStaticObserver
    options: SolverOptions = SolverOptions()
    energy: float = 1.0
    max_lambda: float = 10_000.0
    max_step: float = 0.5

    def trace_alpha(self, alpha: float) -> JunctionRayResult:
        return self.trace_b(self.observer.impact_parameter(alpha), input_coordinate=alpha, coordinate_name="alpha")

    def trace_screen_radius(self, rho: float) -> JunctionRayResult:
        if rho < 0.0:
            raise ValueError("rho must be non-negative")
        alpha = math.atan(rho / self.observer.r_obs)
        return self.trace_b(self.observer.impact_parameter(alpha), input_coordinate=rho, coordinate_name="rho")

    def _hamiltonian(self, metric, r: float, pr: float, pt: float, pphi: float) -> float:
        return 0.5 * (-pt**2 / metric.A(r) + pr**2 / metric.B(r) + pphi**2 / r**2)

    def _rhs(self, metric):
        def rhs(lambda_value: float, y: np.ndarray) -> np.ndarray:
            r, pr, phi, pt, pphi = map(float, y)
            a = metric.A(r)
            b_metric = metric.B(r)
            dr = pr / b_metric
            dpr = (
                -pt**2 * metric.dA_dr(r) / (2.0 * a**2)
                + pr**2 * metric.dB_dr(r) / (2.0 * b_metric**2)
                + pphi**2 / r**3
            )
            dphi = pphi / r**2
            return np.array([dr, dpr, dphi, 0.0, 0.0])

        return rhs

    def _event_shell(self, radial_direction: str):
        def event_shell(lambda_value: float, y: np.ndarray) -> float:
            return float(y[0]) - self.junction.shell.radius

        event_shell.terminal = True
        event_shell.direction = -1.0 if radial_direction == "inward" else 1.0
        return event_shell

    def _event_turning(self, radial_direction: str):
        def event_turning(lambda_value: float, y: np.ndarray) -> float:
            return float(y[1])

        event_turning.terminal = True
        event_turning.direction = 1.0 if radial_direction == "inward" else -1.0
        return event_turning

    def _event_inner_boundary(self, metric, radial_direction: str) -> tuple[object, EventType] | None:
        if radial_direction != "inward":
            return None
        if self.options.inner_boundary_radius is not None:
            r_stop = self.options.inner_boundary_radius
            event_type = EventType.INNER_BOUNDARY
        else:
            r_min, _ = metric.valid_radial_domain()
            if r_min <= 0.0:
                return None
            r_stop = r_min * (1.0 + self.options.horizon_buffer)
            event_type = _classify_radial_event(metric, r_min, default=EventType.INNER_BOUNDARY)

        def event_inner_boundary(lambda_value: float, y: np.ndarray) -> float:
            return float(y[0]) - r_stop

        event_inner_boundary.terminal = True
        event_inner_boundary.direction = -1.0
        return event_inner_boundary, event_type

    def _event_outer_boundary(self, metric, radial_direction: str) -> tuple[object, EventType] | None:
        _, r_max = metric.valid_radial_domain()
        if radial_direction != "outward" or not math.isfinite(r_max):
            return None
        r_stop = r_max * (1.0 - self.options.horizon_buffer)
        event_type = _classify_radial_event(metric, r_max, default=EventType.OUTER_BOUNDARY)

        def event_outer_boundary(lambda_value: float, y: np.ndarray) -> float:
            return float(y[0]) - r_stop

        event_outer_boundary.terminal = True
        event_outer_boundary.direction = 1.0
        return event_outer_boundary, event_type

    def _event_escape(self, metric, radial_direction: str):
        _, r_max = metric.valid_radial_domain()
        if radial_direction != "outward" or math.isfinite(r_max):
            return None
        r_stop = max(2.0 * self.observer.r_obs, 2.0 * self.junction.shell.radius, self.observer.r_obs + 1.0)

        def event_escape(lambda_value: float, y: np.ndarray) -> float:
            return float(y[0]) - r_stop

        event_escape.terminal = True
        event_escape.direction = 1.0
        return event_escape

    def _event_max_phi(self):
        def event_max_phi(lambda_value: float, y: np.ndarray) -> float:
            return self.options.max_phi - float(y[2])

        event_max_phi.terminal = True
        event_max_phi.direction = -1.0
        return event_max_phi

    def _segment_events(self, metric, radial_direction: str) -> tuple[list[object], list[EventType]]:
        events = [self._event_shell(radial_direction), self._event_turning(radial_direction), self._event_max_phi()]
        event_types = [EventType.SHELL_CROSSING, EventType.TURNING_POINT, EventType.MAX_PHI]
        inner = self._event_inner_boundary(metric, radial_direction)
        if inner is not None:
            event, event_type = inner
            events.append(event)
            event_types.append(event_type)
        outer = self._event_outer_boundary(metric, radial_direction)
        if outer is not None:
            event, event_type = outer
            events.append(event)
            event_types.append(event_type)
        escape = self._event_escape(metric, radial_direction)
        if escape is not None:
            events.append(escape)
            event_types.append(EventType.ESCAPE)
        return events, event_types

    def _endpoint_from_solution(self, solution, event_types: list[EventType]) -> EventType:
        hits = [(float(values[0]), event_types[index]) for index, values in enumerate(solution.t_events) if len(values) > 0]
        if not hits:
            return EventType.SOLVER_FAILURE
        return min(hits, key=lambda item: item[0])[1]

    def _matched_pr(self, metric, E: float, L: float, radial_direction: str) -> float:
        radius = self.junction.shell.radius
        radicand = metric.B(radius) * (E**2 / metric.A(radius) - L**2 / radius**2)
        if radicand < -1e-10:
            raise ValueError("matched shell state is outside the null cone")
        sign = -1.0 if radial_direction == "inward" else 1.0
        return sign * math.sqrt(max(radicand, 0.0))

    def _restart_radius_after_shell(self, region: str, radial_direction: str) -> float:
        radius = self.junction.shell.radius
        offset = max(self.options.root_atol, 1e-10 * radius)
        if region == "inner":
            return radius - offset
        return radius + offset

    def trace_b(
        self,
        b_plus: float,
        *,
        input_coordinate: float | None = None,
        coordinate_name: str = "b_plus",
    ) -> JunctionRayResult:
        if not isinstance(self.observer, FiniteStaticObserver):
            raise TypeError("StaticJunctionHamiltonianSolver requires a finite static observer")
        if b_plus <= 0.0:
            raise ValueError("b_plus must be positive")
        if self.observer.r_obs <= self.junction.shell.radius:
            raise ValueError("observer must be outside the shell")
        _validate_inner_boundary_inside_shell(self.options, self.junction.shell.radius)

        coordinate = b_plus if input_coordinate is None else input_coordinate
        physics_warnings = list(self.junction.physics_diagnostics(self.observer.r_obs).warnings)
        outer_curve = _near_critical_metric(self.junction.outer_metric, b_plus, self.options.critical_exclusion)
        if outer_curve is not None:
            return _near_critical_result(
                coordinate=coordinate,
                coordinate_name=coordinate_name,
                b_plus=b_plus,
                backend_name="hamiltonian",
                critical_curve=outer_curve,
                physics_warnings=physics_warnings,
            )

        region = "outer"
        radial_direction = "inward"
        E = self.energy
        L = b_plus * E
        r0 = self.observer.r_obs
        metric = self.junction.outer_metric
        pr0_radicand = metric.B(r0) * (E**2 / metric.A(r0) - L**2 / r0**2)
        if pr0_radicand < 0.0:
            raise ValueError("initial state is outside the null cone")
        y0 = np.array([r0, -math.sqrt(pr0_radicand), 0.0, -E, L])

        segments: list[RaySegment] = []
        segment_constants: list[SegmentConstants] = []
        events: list[RayEvent] = []
        shell_crossings = []
        matching_residuals: dict[str, float] = {}
        shell_matching_residuals: dict[str, float] = {}
        max_abs_H = 0.0
        max_energy_drift = 0.0
        max_angular_momentum_drift = 0.0
        turning_count = 0
        terminal_event = EventType.MAX_PHI
        lambda_start = 0.0

        for _ in range(64):
            metric = self.junction.metric_for_region(region)
            segment_events, event_types = self._segment_events(metric, radial_direction)
            solution = solve_ivp(
                self._rhs(metric),
                (lambda_start, self.max_lambda),
                y0,
                events=segment_events,
                rtol=1e-9,
                atol=1e-11,
                dense_output=True,
                max_step=self.max_step,
            )
            r_values = solution.y[0]
            phi_values = solution.y[2]
            pt_values = solution.y[3]
            pphi_values = solution.y[4]
            pr_values = solution.y[1]
            constraints = [
                abs(self._hamiltonian(metric, float(r), float(pr), float(pt), float(pphi)))
                for r, pr, pt, pphi in zip(r_values, pr_values, pt_values, pphi_values)
            ]
            max_abs_H = max(max_abs_H, max(constraints, default=0.0))
            max_energy_drift = max(max_energy_drift, float(np.max(np.abs(-pt_values - E))))
            max_angular_momentum_drift = max(max_angular_momentum_drift, float(np.max(np.abs(pphi_values - L))))

            phi_start = float(phi_values[0])
            phi_end = float(phi_values[-1])
            r_start = float(r_values[0])
            r_end = float(r_values[-1])
            endpoint = self._endpoint_from_solution(solution, event_types)
            segment = RaySegment(
                region=region,
                radial_direction=radial_direction,
                phi_start=phi_start,
                phi_end=phi_end,
                u_start=1.0 / r_start,
                u_end=1.0 / r_end,
                endpoint_event=endpoint,
                evaluator=lambda phi, phi_values=phi_values.copy(), r_values=r_values.copy(): float(
                    1.0 / np.interp(phi, phi_values, r_values)
                ),
            )
            segments.append(segment)
            segment_constants.append(SegmentConstants(region=region, E=E, L=L, b=L / E))
            events.append(RayEvent(endpoint, phi_end, 1.0 / r_end, region))

            if endpoint == EventType.MAX_PHI:
                terminal_event = endpoint
                break
            if endpoint in {
                EventType.HORIZON,
                EventType.INNER_BOUNDARY,
                EventType.OUTER_BOUNDARY,
                EventType.ESCAPE,
                EventType.SOLVER_FAILURE,
            }:
                terminal_event = endpoint
                break
            if endpoint == EventType.TURNING_POINT:
                turning_count += 1
                radial_direction = "outward" if radial_direction == "inward" else "inward"
                y0 = solution.y[:, -1].copy()
                y0[1] = -float(y0[1])
                lambda_start = float(solution.t[-1])
                continue
            if endpoint == EventType.SHELL_CROSSING:
                before = SegmentConstants(region=region, E=E, L=L, b=L / E)
                next_region = self.junction.other_region(region)
                after = match_static_shell(self.junction, from_region=region, to_region=next_region, E=E, L=L, b=L / E)
                crossing = self.junction.shell_crossing_record(
                    phi=phi_end,
                    radial_direction=radial_direction,
                    before=before,
                    after=after,
                )
                shell_crossings.append(crossing)
                matching_residuals = self.junction.shell_matching_residuals(
                    from_region=region,
                    to_region=next_region,
                    before=before,
                    after=after,
                )
                crossing_index = len(shell_crossings) - 1
                for key, value in matching_residuals.items():
                    shell_matching_residuals[f"shell_{crossing_index}_{key}"] = value
                    shell_matching_residuals[f"max_{key}"] = max(shell_matching_residuals.get(f"max_{key}", 0.0), value)
                region = next_region
                E = after.E
                L = after.L
                next_curve = _near_critical_metric(
                    self.junction.metric_for_region(region),
                    L / E,
                    self.options.critical_exclusion,
                )
                if next_curve is not None:
                    residuals = {
                        "max_abs_H": max_abs_H,
                        "max_energy_drift": max_energy_drift,
                        "max_angular_momentum_drift": max_angular_momentum_drift,
                        "shell_crossing_count": float(len(shell_crossings)),
                        **shell_matching_residuals,
                        **matching_residuals,
                    }
                    return _near_critical_result(
                        coordinate=coordinate,
                        coordinate_name=coordinate_name,
                        b_plus=b_plus,
                        backend_name="hamiltonian",
                        critical_curve=next_curve,
                        physics_warnings=physics_warnings,
                        residuals=residuals,
                        segments=segments,
                        segment_constants=segment_constants,
                        events=events,
                        shell_crossings=shell_crossings,
                    )
                metric_to = self.junction.metric_for_region(region)
                y0 = np.array(
                    [
                        self._restart_radius_after_shell(region, radial_direction),
                        self._matched_pr(metric_to, E, L, radial_direction),
                        phi_end,
                        -E,
                        L,
                    ]
                )
                lambda_start = float(solution.t[-1])
                continue
            terminal_event = endpoint
            break
        else:
            terminal_event = EventType.SOLVER_FAILURE
            events.append(RayEvent(EventType.SOLVER_FAILURE, segments[-1].phi_end, segments[-1].u_end, region))

        diagnostics = RayDiagnostics(
            estimated_error=max_abs_H,
            near_critical=False,
            critical_curve_id=None,
            hit_horizon=terminal_event == EventType.HORIZON,
            hit_inner_boundary=terminal_event == EventType.INNER_BOUNDARY,
            turning_point_count=turning_count,
            max_phi_reached=terminal_event == EventType.MAX_PHI,
            termination_reason=terminal_event.value,
            residuals={
                "max_abs_H": max_abs_H,
                "max_energy_drift": max_energy_drift,
                "max_angular_momentum_drift": max_angular_momentum_drift,
                "shell_crossing_count": float(len(shell_crossings)),
                **shell_matching_residuals,
                **matching_residuals,
            },
        )
        return JunctionRayResult(
            input_coordinate=coordinate,
            coordinate_name=coordinate_name,
            b_plus=b_plus,
            segments=segments,
            segment_constants=segment_constants,
            events=events,
            shell_crossings=shell_crossings,
            physics_warnings=physics_warnings,
            diagnostics=diagnostics,
            backend_name="hamiltonian",
        )


def trace_with_backend_compare(
    *,
    junction: StaticJunctionSpacetime,
    observer: FiniteStaticObserver,
    b_plus: float,
    options: SolverOptions = SolverOptions(),
) -> JunctionRayResult:
    primary = StaticJunctionTransferSolver(junction=junction, observer=observer, options=options).trace_b(b_plus)
    comparison_result = StaticJunctionHamiltonianSolver(junction=junction, observer=observer, options=options).trace_b(b_plus)
    comparison = compare_junction_results(primary, comparison_result)
    residuals = dict(primary.diagnostics.residuals)
    residuals["backend_disagreement"] = float(comparison["backend_disagreement"])
    diagnostics = replace(primary.diagnostics, residuals=residuals)
    return replace(primary, diagnostics=diagnostics, backend_comparison=comparison)
