import math

import pytest


EXPECTED_PUBLIC_EXPORTS = [
    "CanonicalPayload",
    "CriticalCurve",
    "DiskWindow",
    "EventType",
    "FiniteStaticObserver",
    "HamiltonianTrajectorySolver",
    "ImageGrid",
    "InfinityObserver",
    "IntersectionRecord",
    "JunctionPhysicsDiagnostics",
    "JunctionRayResult",
    "LQGMetric",
    "LiDMBHMetadata",
    "SegmentConstants",
    "ShellCrossingRecord",
    "ObservedIntensity",
    "OdeTrajectorySolver",
    "OutsideTracePolicy",
    "QuadTransferSolver",
    "RadialProfile",
    "RayDiagnostics",
    "RayEvent",
    "RayResult",
    "RaySegment",
    "RenderedImage",
    "SamplingOptions",
    "StaticDomainSelector",
    "StaticJunctionHamiltonianSolver",
    "StaticJunctionTransferSolver",
    "StaticJunctionSpacetime",
    "StaticShell",
    "ReissnerNordstromDeSitterMetric",
    "ReissnerNordstromMetric",
    "SchwarzschildMetric",
    "SolverOptions",
    "TabulatedMetric",
    "ThinDiskSource",
    "ValidationGate",
    "ValidationOptions",
    "annotate_junction_intersections",
    "compare_junction_results",
    "junction_redshift_factor",
    "load_metric_npz",
    "match_static_shell",
    "trace_with_backend_compare",
    "ThroughTracePolicy",
    "compute_intersections",
    "observed_intensity",
    "render_axisymmetric_image",
    "sample_radial_profile",
]


def test_package_imports():
    import spherical_raytracing

    assert spherical_raytracing.__all__


def test_public_api_exports_current_contract():
    import spherical_raytracing

    assert spherical_raytracing.__all__ == EXPECTED_PUBLIC_EXPORTS
    for name in EXPECTED_PUBLIC_EXPORTS:
        assert getattr(spherical_raytracing, name)


def test_public_api_exports_core_types():
    from spherical_raytracing import (
        CanonicalPayload,
        DiskWindow,
        FiniteStaticObserver,
        HamiltonianTrajectorySolver,
        InfinityObserver,
        JunctionRayResult,
        LQGMetric,
        LiDMBHMetadata,
        OdeTrajectorySolver,
        QuadTransferSolver,
        ReissnerNordstromDeSitterMetric,
        ReissnerNordstromMetric,
        SchwarzschildMetric,
        StaticDomainSelector,
        StaticJunctionHamiltonianSolver,
        StaticJunctionTransferSolver,
        StaticJunctionSpacetime,
        StaticShell,
        TabulatedMetric,
        ThinDiskSource,
        ValidationGate,
        ValidationOptions,
        compare_junction_results,
        load_metric_npz,
        trace_with_backend_compare,
    )

    assert SchwarzschildMetric
    assert LQGMetric
    assert ReissnerNordstromMetric
    assert ReissnerNordstromDeSitterMetric
    assert StaticShell
    assert StaticJunctionSpacetime
    assert JunctionRayResult
    assert InfinityObserver
    assert FiniteStaticObserver
    assert QuadTransferSolver
    assert StaticJunctionHamiltonianSolver
    assert StaticJunctionTransferSolver
    assert OdeTrajectorySolver
    assert HamiltonianTrajectorySolver
    assert ThinDiskSource
    assert DiskWindow
    assert compare_junction_results
    assert trace_with_backend_compare
    assert TabulatedMetric
    assert load_metric_npz
    assert CanonicalPayload
    assert LiDMBHMetadata
    assert ValidationOptions
    assert StaticDomainSelector
    assert ValidationGate


def test_public_single_metric_workflow_traces_intersects_and_observes():
    from spherical_raytracing import (
        DiskWindow,
        FiniteStaticObserver,
        QuadTransferSolver,
        SchwarzschildMetric,
        ThinDiskSource,
        compute_intersections,
        observed_intensity,
    )

    metric = SchwarzschildMetric(mass=1.0)
    observer = FiniteStaticObserver(r_obs=100.0, metric=metric)
    result = QuadTransferSolver(metric=metric, observer=observer).trace_b(18.0)
    assert result.b == 18.0
    assert result.segments
    assert result.diagnostics.termination_reason

    segment = result.segments[0]
    phi_mid = 0.5 * (segment.phi_start + segment.phi_end)
    assert segment.contains_phi(phi_mid)
    assert math.isfinite(segment.u_at(phi_mid))

    intersections = compute_intersections(
        result,
        DiskWindow(r_min=6.0, r_max=90.0),
        max_order=3,
    )
    assert intersections
    assert intersections[0].r > 0.0
    assert intersections[0].ray_segment is not None

    source = ThinDiskSource(lambda r, _region: 1.0 / (r * r))
    observed = observed_intensity(intersections, source, metric, observer)
    assert observed.total > 0.0
    assert observed.contributions
    assert observed.redshift_weights


def test_public_static_junction_workflow_returns_junction_result_fields():
    from spherical_raytracing import (
        FiniteStaticObserver,
        JunctionRayResult,
        SchwarzschildMetric,
        SolverOptions,
        StaticJunctionSpacetime,
        StaticJunctionTransferSolver,
        StaticShell,
    )

    outer = SchwarzschildMetric(mass=1.0, region="outer")
    inner = SchwarzschildMetric(mass=0.1, region="inner")
    shell = StaticShell(radius=5.0)
    junction = StaticJunctionSpacetime(inner_metric=inner, outer_metric=outer, shell=shell)
    observer = FiniteStaticObserver(r_obs=50.0, metric=outer)
    solver = StaticJunctionTransferSolver(
        junction=junction,
        observer=observer,
        options=SolverOptions(inner_boundary_radius=1.0),
    )

    result = solver.trace_b(4.0)

    assert isinstance(result, JunctionRayResult)
    assert result.b == pytest.approx(result.b_plus)
    assert result.backend_name == "transfer"
    assert result.segments
    assert result.segment_constants
    assert result.shell_crossings
    assert result.diagnostics.termination_reason
