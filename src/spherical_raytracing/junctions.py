from __future__ import annotations

import math
from dataclasses import dataclass, field, replace

from .diagnostics import EventType, RayDiagnostics, RayEvent, RaySegment
from .metrics import StaticSphericalMetric
from .observers import FiniteStaticObserver
from .policies import SolverOptions
from .transfer import IntersectionRecord


@dataclass(frozen=True)
class StaticShell:
    radius: float

    def __post_init__(self) -> None:
        if self.radius <= 0.0:
            raise ValueError("shell radius must be positive")


@dataclass(frozen=True)
class SegmentConstants:
    region: str
    E: float
    L: float
    b: float


@dataclass(frozen=True)
class ShellCrossingRecord:
    phi: float
    r: float
    u: float
    radial_direction: str
    from_region: str
    to_region: str
    E_before: float
    E_after: float
    L: float
    b_before: float
    b_after: float

    @property
    def backward_energy_ratio(self) -> float:
        return self.E_after / self.E_before

    @property
    def forward_energy_ratio(self) -> float:
        return self.E_before / self.E_after


@dataclass(frozen=True)
class JunctionPhysicsDiagnostics:
    warnings: tuple[str, ...]
    surface_energy_density: float | None
    surface_pressure: float | None = None
    static_domains: dict[str, list[tuple[float, float]]] = field(default_factory=dict)


@dataclass(frozen=True)
class JunctionRayResult:
    input_coordinate: float
    coordinate_name: str
    b_plus: float
    segments: list[RaySegment]
    segment_constants: list[SegmentConstants]
    events: list[RayEvent]
    shell_crossings: list[ShellCrossingRecord]
    physics_warnings: list[str]
    diagnostics: RayDiagnostics
    backend_name: str
    critical_status: object | None = None
    intersections: list[IntersectionRecord] = field(default_factory=list)
    backend_comparison: dict[str, float | int | str] = field(default_factory=dict)

    @property
    def b(self) -> float:
        return self.b_plus

    @property
    def near_critical(self) -> bool:
        return self.diagnostics.near_critical

    @property
    def unresolved(self) -> bool:
        return bool(self.diagnostics.residuals.get("unresolved", 0.0))

    @property
    def termination_reason(self) -> str:
        return self.diagnostics.termination_reason


@dataclass(frozen=True)
class StaticJunctionSpacetime:
    inner_metric: StaticSphericalMetric
    outer_metric: StaticSphericalMetric
    shell: StaticShell
    options: SolverOptions = SolverOptions()

    def __post_init__(self) -> None:
        if _metric_family(self.inner_metric) != _metric_family(self.outer_metric):
            raise ValueError("static junction v1 supports same-family metrics only")

    def metric_for_region(self, region: str) -> StaticSphericalMetric:
        if region == "inner":
            return self.inner_metric
        if region == "outer":
            return self.outer_metric
        raise ValueError(f"unknown junction region {region!r}")

    def other_region(self, region: str) -> str:
        if region == "inner":
            return "outer"
        if region == "outer":
            return "inner"
        raise ValueError(f"unknown junction region {region!r}")

    def shell_crossing_record(
        self,
        *,
        phi: float,
        radial_direction: str,
        before: SegmentConstants,
        after: SegmentConstants,
    ) -> ShellCrossingRecord:
        radius = self.shell.radius
        return ShellCrossingRecord(
            phi=phi,
            r=radius,
            u=1.0 / radius,
            radial_direction=radial_direction,
            from_region=before.region,
            to_region=after.region,
            E_before=before.E,
            E_after=after.E,
            L=before.L,
            b_before=before.b,
            b_after=after.b,
        )

    def physics_diagnostics(self, r_obs: float | None = None) -> JunctionPhysicsDiagnostics:
        warnings: list[str] = []
        radius = self.shell.radius
        a_minus = self.inner_metric.A(radius)
        a_plus = self.outer_metric.A(radius)
        if a_minus <= 0.0:
            warnings.append("inner_shell_radius_nonstatic")
        if a_plus <= 0.0:
            warnings.append("outer_shell_radius_nonstatic")
        if r_obs is not None and self.outer_metric.A(r_obs) <= 0.0:
            warnings.append("observer_nonstatic")
        if r_obs is not None:
            scale = max(abs(radius), 1.0)
            if abs(r_obs - radius) <= self.options.horizon_buffer * scale:
                warnings.append("observer_near_shell")
        for label, metric in [("inner", self.inner_metric), ("outer", self.outer_metric)]:
            naked_singularity = hasattr(metric, "charge") and _has_naked_singularity(metric)
            if naked_singularity:
                warnings.append(f"{label}_naked_singularity")
            if hasattr(metric, "cosmological_constant") and float(metric.cosmological_constant) > 0.0:
                clean_patch = clean_rnds_static_patch(metric)
                observer_reachable = clean_patch is not None
                if clean_patch is not None:
                    r_min, r_max = clean_patch
                    observer_reachable = r_min < radius < r_max
                if label == "outer" and r_obs is not None:
                    observer_reachable = (
                        observer_reachable
                        and clean_patch is not None
                        and clean_patch[0] < r_obs < clean_patch[1]
                    )
                if not observer_reachable:
                    warnings.append(f"{label}_rnds_no_clean_static_patch")
        for label, metric in [("inner", self.inner_metric), ("outer", self.outer_metric)]:
            for horizon in metric.horizons():
                scale = max(abs(horizon), 1.0)
                if abs(radius - horizon) <= self.options.horizon_buffer * scale:
                    warnings.append(f"{label}_shell_near_horizon")
        if hasattr(self.inner_metric, "charge") and hasattr(self.outer_metric, "charge"):
            if not math.isclose(float(self.inner_metric.charge), float(self.outer_metric.charge), rel_tol=0.0, abs_tol=0.0):
                warnings.append("charge_mismatch_surface_source")
        if hasattr(self.inner_metric, "cosmological_constant") and hasattr(self.outer_metric, "cosmological_constant"):
            if not math.isclose(
                float(self.inner_metric.cosmological_constant),
                float(self.outer_metric.cosmological_constant),
                rel_tol=0.0,
                abs_tol=0.0,
            ):
                warnings.append("lambda_mismatch_vacuum_discontinuity")
        sigma = None
        pressure = None
        if a_minus > 0.0 and a_plus > 0.0:
            sigma = (math.sqrt(a_minus) - math.sqrt(a_plus)) / (4.0 * math.pi * radius)
            if sigma < 0.0:
                warnings.append("negative_surface_energy_density")
            pressure = self._surface_pressure_same_orientation()
            if pressure is not None and not math.isfinite(pressure):
                warnings.append("non_finite_surface_pressure")
        return JunctionPhysicsDiagnostics(
            warnings=tuple(dict.fromkeys(warnings)),
            surface_energy_density=sigma,
            surface_pressure=pressure,
            static_domains={
                "inner": self.inner_metric.static_domains(),
                "outer": self.outer_metric.static_domains(),
            },
        )

    def _surface_pressure_same_orientation(self) -> float | None:
        radius = self.shell.radius
        a_minus = self.inner_metric.A(radius)
        a_plus = self.outer_metric.A(radius)
        if a_minus <= 0.0 or a_plus <= 0.0:
            return None
        k_tau_jump = (
            self.outer_metric.dA_dr(radius) / (2.0 * math.sqrt(a_plus))
            - self.inner_metric.dA_dr(radius) / (2.0 * math.sqrt(a_minus))
        )
        k_theta_jump = (math.sqrt(a_plus) - math.sqrt(a_minus)) / radius
        return (k_tau_jump + k_theta_jump) / (8.0 * math.pi)

    def shell_matching_residuals(
        self,
        *,
        from_region: str,
        to_region: str,
        before: SegmentConstants,
        after: SegmentConstants,
    ) -> dict[str, float]:
        radius = self.shell.radius
        a_from = self.metric_for_region(from_region).A(radius)
        a_to = self.metric_for_region(to_region).A(radius)
        expected_ratio = math.sqrt(a_to / a_from)
        expected_b = before.b * math.sqrt(a_from / a_to)
        return {
            "energy_ratio_abs": abs((after.E / before.E) - expected_ratio),
            "angular_momentum_abs": abs(after.L - before.L),
            "impact_parameter_abs": abs(after.b - expected_b),
            "local_definition_abs": abs(after.b - after.L / after.E),
        }


def match_static_shell(
    junction: StaticJunctionSpacetime,
    *,
    from_region: str,
    to_region: str,
    E: float,
    L: float,
    b: float,
) -> SegmentConstants:
    if not all(math.isfinite(value) for value in [E, L, b]):
        raise ValueError("E, L, and b must be finite")
    if E == 0.0:
        raise ValueError("E must be nonzero")
    if not math.isclose(b, L / E, rel_tol=1e-12, abs_tol=1e-12):
        raise ValueError("b must equal L / E")
    radius = junction.shell.radius
    a_from = junction.metric_for_region(from_region).A(radius)
    a_to = junction.metric_for_region(to_region).A(radius)
    if a_from <= 0.0 or a_to <= 0.0:
        raise ValueError("static shell matching requires A(R) > 0 on both sides")
    ratio = math.sqrt(a_to / a_from)
    E_to = E * ratio
    L_to = L
    return SegmentConstants(region=to_region, E=E_to, L=L_to, b=L_to / E_to)


def _metric_family(metric: StaticSphericalMetric) -> str:
    name = type(metric).__name__
    if name == "SchwarzschildMetric":
        return "schwarzschild"
    if name == "ReissnerNordstromMetric":
        return "rn"
    if name == "ReissnerNordstromDeSitterMetric":
        return "rnds"
    raise ValueError(f"unsupported static junction metric family {name!r}")


def clean_rnds_static_patch(metric: StaticSphericalMetric) -> tuple[float, float] | None:
    if not hasattr(metric, "cosmological_constant"):
        return metric.valid_radial_domain()
    if float(metric.cosmological_constant) == 0.0:
        return metric.valid_radial_domain()
    horizons = metric.horizons()
    if len(horizons) < 2:
        return None
    candidate = (horizons[-2], horizons[-1])
    midpoint = 0.5 * (candidate[0] + candidate[1])
    if metric.A(midpoint) > 0.0:
        return candidate
    return None


def _has_naked_singularity(metric: StaticSphericalMetric) -> bool:
    if not hasattr(metric, "charge"):
        return False
    horizons = metric.horizons()
    if not horizons:
        return True
    if hasattr(metric, "cosmological_constant"):
        return not any(left > 0.0 for left, _right in metric.static_domains())
    return False


def shell_product_forward(shell_crossings: list[ShellCrossingRecord], phi_emission: float, *, atol: float = 1e-12) -> float:
    product = 1.0
    for crossing in shell_crossings:
        if crossing.phi <= phi_emission + atol:
            product *= crossing.forward_energy_ratio
    return product


def junction_redshift_factor(
    intersection: IntersectionRecord,
    result: JunctionRayResult,
    observer: FiniteStaticObserver,
    region_metrics: dict[str, StaticSphericalMetric],
) -> float:
    metric = region_metrics[intersection.region]
    shell_product = shell_product_forward(result.shell_crossings, intersection.phi)
    return math.sqrt(metric.A(intersection.r) / observer.metric.A(observer.r_obs)) * shell_product


def annotate_junction_intersections(
    intersections: list[IntersectionRecord],
    result: JunctionRayResult,
    observer: FiniteStaticObserver,
    region_metrics: dict[str, StaticSphericalMetric],
) -> list[IntersectionRecord]:
    annotated: list[IntersectionRecord] = []
    for intersection in intersections:
        g = junction_redshift_factor(intersection, result, observer, region_metrics)
        diagnostics = dict(intersection.diagnostics)
        diagnostics["g"] = g
        diagnostics["redshift_weight"] = g**4
        diagnostics["backend_name"] = result.backend_name
        annotated.append(replace(intersection, diagnostics=diagnostics))
    return annotated


def compare_junction_results(
    left: JunctionRayResult,
    right: JunctionRayResult,
    *,
    r_intersection_abs: float = 1e-6,
    g_abs: float = 1e-6,
    phi_event_abs: float = 1e-6,
    b_local_abs: float = 1e-8,
) -> dict[str, float | int | str]:
    disagreement = 0
    shell_delta = len(left.shell_crossings) - len(right.shell_crossings)
    if shell_delta != 0:
        disagreement = 1
    if left.termination_reason != right.termination_reason:
        disagreement = 1
    left_regions = ",".join(segment.region for segment in left.segments)
    right_regions = ",".join(segment.region for segment in right.segments)
    if left_regions != right_regions:
        disagreement = 1

    if len(left.events) != len(right.events):
        disagreement = 1
    event_phi_delta = 0.0
    for left_event, right_event in zip(left.events, right.events):
        if left_event.event_type != right_event.event_type:
            disagreement = 1
        if left_event.region != right_event.region:
            disagreement = 1
        if left_event.event_type != right_event.event_type or left_event.event_type is not EventType.ESCAPE:
            event_phi_delta = max(event_phi_delta, abs(left_event.phi - right_event.phi))
            event_phi_delta = max(event_phi_delta, abs(left_event.u - right_event.u))
    if event_phi_delta > phi_event_abs:
        disagreement = 1

    b_delta = 0.0
    for left_constants, right_constants in zip(left.segment_constants, right.segment_constants):
        b_delta = max(b_delta, abs(left_constants.b - right_constants.b))
    if b_delta > b_local_abs:
        disagreement = 1

    intersection_delta = len(left.intersections) - len(right.intersections)
    if intersection_delta != 0:
        disagreement = 1

    r_delta = 0.0
    g_delta = 0.0
    for left_intersection, right_intersection in zip(left.intersections, right.intersections):
        if left_intersection.m != right_intersection.m:
            disagreement = 1
        if abs(left_intersection.phi - right_intersection.phi) > phi_event_abs:
            disagreement = 1
        if left_intersection.region != right_intersection.region:
            disagreement = 1
        if left_intersection.path_class != right_intersection.path_class:
            disagreement = 1
        r_delta = max(r_delta, abs(left_intersection.r - right_intersection.r))
        if "g" in left_intersection.diagnostics and "g" in right_intersection.diagnostics:
            g_delta = max(
                g_delta,
                abs(float(left_intersection.diagnostics["g"]) - float(right_intersection.diagnostics["g"])),
            )
    if r_delta > r_intersection_abs or g_delta > g_abs:
        disagreement = 1

    return {
        "backend_disagreement": disagreement,
        "primary_backend": left.backend_name,
        "comparison_backend": right.backend_name,
        "shell_crossing_count_delta": shell_delta,
        "max_event_phi_delta": event_phi_delta,
        "max_local_b_delta": b_delta,
        "intersection_count_delta": intersection_delta,
        "max_intersection_r_delta": r_delta,
        "max_redshift_g_delta": g_delta,
    }
