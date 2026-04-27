import math

from spherical_raytracing.diagnostics import (
    CriticalCurve,
    EventType,
    RayDiagnostics,
    RayResult,
    RaySegment,
)


def test_ray_segment_evaluator_returns_radius_from_phi():
    segment = RaySegment(
        region="external",
        radial_direction="inward",
        phi_start=0.0,
        phi_end=math.pi / 2,
        u_start=0.0,
        u_end=0.2,
        endpoint_event=EventType.TURNING_POINT,
        evaluator=lambda phi: 0.1 * phi,
    )

    assert math.isclose(segment.u_at(math.pi / 4), math.pi / 40)
    assert math.isclose(segment.r_at(math.pi / 4), 40 / math.pi)


def test_ray_result_exposes_near_critical_status():
    curve = CriticalCurve(
        id="external:photon_sphere:0",
        r_ph=3.0,
        b_crit=3.0 * math.sqrt(3.0),
        stability="unstable",
        region="external",
        reachable_from_observer=True,
    )
    diagnostics = RayDiagnostics(
        estimated_error=0.0,
        near_critical=True,
        critical_curve_id=curve.id,
        hit_horizon=False,
        hit_inner_boundary=False,
        turning_point_count=0,
        max_phi_reached=False,
        termination_reason="near_critical",
        residuals={},
    )

    result = RayResult(
        b=curve.b_crit,
        segments=[],
        events=[],
        critical_status=curve,
        diagnostics=diagnostics,
    )

    assert result.diagnostics.near_critical is True
    assert result.critical_status == curve
