import math

from spherical_raytracing.metrics import SchwarzschildMetric
from spherical_raytracing.observers import FiniteStaticObserver, InfinityObserver
from spherical_raytracing.solvers import QuadTransferSolver
from spherical_raytracing.sources import ThinDiskSource, observed_intensity
from spherical_raytracing.diagnostics import EventType, RaySegment
from spherical_raytracing.transfer import DiskWindow, IntersectionRecord, compute_intersections


def constant_profile(r: float, region: str | None = None) -> float:
    return 2.0


def test_static_redshift_at_infinity_uses_A_squared_weight():
    metric = SchwarzschildMetric(mass=1.0)
    source = ThinDiskSource(intensity_profile=constant_profile)
    result = QuadTransferSolver(metric=metric, observer=InfinityObserver()).trace_b(30.0)
    intersection = compute_intersections(result, DiskWindow(r_min=6.0, r_max=200.0), max_order=1)[0]

    weight = source.redshift_weight(intersection, metric, InfinityObserver())

    assert math.isclose(weight, metric.A(intersection.r) ** 2)


def test_static_redshift_for_finite_observer_uses_ratio():
    metric = SchwarzschildMetric(mass=1.0)
    observer = FiniteStaticObserver(r_obs=100.0, metric=metric)
    source = ThinDiskSource(intensity_profile=constant_profile)
    result = QuadTransferSolver(metric=metric, observer=InfinityObserver()).trace_b(30.0)
    intersection = compute_intersections(result, DiskWindow(r_min=6.0, r_max=200.0), max_order=1)[0]

    weight = source.redshift_weight(intersection, metric, observer)

    assert math.isclose(weight, (metric.A(intersection.r) / metric.A(observer.r_obs)) ** 2)


def test_observed_intensity_keeps_decomposition():
    metric = SchwarzschildMetric(mass=1.0)
    observer = InfinityObserver()
    source = ThinDiskSource(intensity_profile=constant_profile)
    result = QuadTransferSolver(metric=metric, observer=observer).trace_b(30.0)
    intersections = compute_intersections(result, DiskWindow(r_min=6.0, r_max=200.0), max_order=1)

    observed = observed_intensity(intersections, source, metric, observer)

    assert observed.total > 0.0
    assert observed.radii == [intersections[0].r]
    assert observed.regions == ["external"]
    assert observed.path_classes == ["direct"]


def test_observed_intensity_uses_metric_matching_intersection_region():
    external = SchwarzschildMetric(mass=1.0, region="external")
    companion = SchwarzschildMetric(mass=0.5, region="companion")
    source = ThinDiskSource(intensity_profile=constant_profile)
    segment = RaySegment(
        region="companion",
        radial_direction="outward",
        phi_start=0.0,
        phi_end=1.0,
        u_start=0.2,
        u_end=0.1,
        endpoint_event=EventType.ESCAPE,
        evaluator=lambda phi: 0.1,
    )
    intersection = IntersectionRecord(
        m=1,
        r=10.0,
        phi=0.5,
        region="companion",
        ray_segment=segment,
        path_class="direct",
    )

    observed = observed_intensity(
        [intersection],
        source,
        external,
        InfinityObserver(),
        region_metrics={"companion": companion},
    )

    assert math.isclose(observed.total, 2.0 * companion.A(intersection.r) ** 2)
    assert math.isclose(observed.redshift_weights[0], companion.A(intersection.r) ** 2)


def test_observed_intensity_marks_unknown_region_invalid():
    external = SchwarzschildMetric(mass=1.0, region="external")
    source = ThinDiskSource(intensity_profile=constant_profile)
    segment = RaySegment(
        region="companion",
        radial_direction="outward",
        phi_start=0.0,
        phi_end=1.0,
        u_start=0.2,
        u_end=0.1,
        endpoint_event=EventType.ESCAPE,
        evaluator=lambda phi: 0.1,
    )
    intersection = IntersectionRecord(
        m=1,
        r=10.0,
        phi=0.5,
        region="companion",
        ray_segment=segment,
        path_class="direct",
    )

    observed = observed_intensity([intersection], source, external, InfinityObserver())

    assert observed.total == 0.0
    assert observed.contributions == []
    assert observed.diagnostics["invalid_intersection_count"] == 1
