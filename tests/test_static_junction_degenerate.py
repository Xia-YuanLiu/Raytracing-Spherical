import math

import pytest

from spherical_raytracing.junction_tracing import StaticJunctionTransferSolver
from spherical_raytracing.junctions import (
    SegmentConstants,
    StaticJunctionSpacetime,
    StaticShell,
    annotate_junction_intersections,
    junction_redshift_factor,
    match_static_shell,
)
from spherical_raytracing.metrics import SchwarzschildMetric
from spherical_raytracing.observers import FiniteStaticObserver
from spherical_raytracing.solvers import QuadTransferSolver
from spherical_raytracing.sources import ThinDiskSource, observed_intensity
from spherical_raytracing.transfer import DiskWindow, compute_intersections


def _identical_schwarzschild_junction():
    inner = SchwarzschildMetric(mass=1.0, region="inner")
    outer = SchwarzschildMetric(mass=1.0, region="outer")
    junction = StaticJunctionSpacetime(inner_metric=inner, outer_metric=outer, shell=StaticShell(radius=20.0))
    observer = FiniteStaticObserver(r_obs=50.0, metric=outer)
    return inner, outer, junction, observer


def test_identical_metric_shell_matching_is_identity():
    _inner, _outer, junction, _observer = _identical_schwarzschild_junction()
    before = SegmentConstants(region="outer", E=1.0, L=10.0, b=10.0)

    after = match_static_shell(
        junction,
        from_region="outer",
        to_region="inner",
        E=before.E,
        L=before.L,
        b=before.b,
    )
    crossing = junction.shell_crossing_record(
        phi=0.25,
        radial_direction="inward",
        before=before,
        after=after,
    )

    assert after.E / before.E == pytest.approx(1.0, abs=1e-12)
    assert after.b == pytest.approx(before.b, abs=1e-12)
    assert crossing.backward_energy_ratio == pytest.approx(1.0, abs=1e-12)
    assert crossing.forward_energy_ratio == pytest.approx(1.0, abs=1e-12)


def test_identical_metric_junction_redshift_is_direct_finite_observer_redshift():
    inner, outer, junction, observer = _identical_schwarzschild_junction()
    result = StaticJunctionTransferSolver(junction=junction, observer=observer).trace_b(10.0)
    intersections = compute_intersections(
        result,
        DiskWindow(r_min=2.5, r_max=40.0, enabled_regions=frozenset({"inner", "outer"})),
        max_order=1,
    )

    assert len(intersections) == 1
    intersection = intersections[0]
    assert intersection.path_class == "direct"

    g = junction_redshift_factor(intersection, result, observer, {"inner": inner, "outer": outer})
    direct_g = math.sqrt(inner.A(intersection.r) / outer.A(observer.r_obs))

    assert g == pytest.approx(direct_g, rel=1e-12, abs=1e-12)


def test_identical_metric_direct_intersection_intensity_matches_single_metric():
    inner, outer, junction, observer = _identical_schwarzschild_junction()
    source = ThinDiskSource(intensity_profile=lambda r, region: 1.0 / (r * r))
    single_result = QuadTransferSolver(metric=outer, observer=observer).trace_b(10.0)
    junction_result = StaticJunctionTransferSolver(junction=junction, observer=observer).trace_b(10.0)
    single_intersections = compute_intersections(
        single_result,
        DiskWindow(r_min=2.5, r_max=40.0, enabled_regions=frozenset({"outer"})),
        max_order=1,
    )
    junction_intersections = compute_intersections(
        junction_result,
        DiskWindow(r_min=2.5, r_max=40.0, enabled_regions=frozenset({"inner", "outer"})),
        max_order=1,
    )

    assert len(single_intersections) == 1
    assert len(junction_intersections) == 1
    assert single_intersections[0].path_class == "direct"
    assert junction_intersections[0].path_class == "direct"
    assert junction_intersections[0].r == pytest.approx(single_intersections[0].r, abs=1e-12)

    annotated_junction_intersections = annotate_junction_intersections(
        junction_intersections,
        junction_result,
        observer,
        {"inner": inner, "outer": outer},
    )
    single_observed = observed_intensity(single_intersections, source, outer, observer)
    junction_observed = observed_intensity(
        annotated_junction_intersections,
        source,
        outer,
        observer,
        {"inner": inner, "outer": outer},
    )

    assert junction_observed.redshift_weights == pytest.approx(single_observed.redshift_weights, rel=1e-12, abs=1e-12)
    assert junction_observed.contributions == pytest.approx(single_observed.contributions, rel=1e-12, abs=1e-12)
    assert junction_observed.total == pytest.approx(single_observed.total, rel=1e-12, abs=1e-12)
