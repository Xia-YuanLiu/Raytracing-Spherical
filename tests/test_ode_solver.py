import math

import pytest

from spherical_raytracing.diagnostics import EventType
from spherical_raytracing.metrics import SchwarzschildMetric
from spherical_raytracing.observers import InfinityObserver
from spherical_raytracing.policies import ThroughTracePolicy
from spherical_raytracing.solvers import OdeTrajectorySolver, QuadTransferSolver


def test_ode_solver_matches_quad_turning_angle_for_noncritical_ray():
    metric = SchwarzschildMetric(mass=1.0)
    b = 8.0
    quad = QuadTransferSolver(metric=metric, observer=InfinityObserver()).trace_b(b)
    ode = OdeTrajectorySolver(metric=metric, observer=InfinityObserver()).trace_b(b)

    assert math.isclose(ode.segments[0].phi_end, quad.segments[0].phi_end, rel_tol=2e-3)
    midpoint = 0.5 * quad.segments[0].phi_end
    assert math.isclose(ode.segments[0].u_at(midpoint), quad.segments[0].u_at(midpoint), rel_tol=2e-3)


def test_ode_solver_returns_escape_segment_for_scattered_ray():
    metric = SchwarzschildMetric(mass=1.0)
    result = OdeTrajectorySolver(metric=metric, observer=InfinityObserver()).trace_b(8.0)

    assert [segment.endpoint_event for segment in result.segments] == [
        EventType.TURNING_POINT,
        EventType.ESCAPE,
    ]
    assert result.diagnostics.termination_reason == EventType.ESCAPE.value


def test_ode_solver_preserves_radial_residual():
    metric = SchwarzschildMetric(mass=1.0)
    result = OdeTrajectorySolver(metric=metric, observer=InfinityObserver()).trace_b(8.0)

    assert abs(result.diagnostics.residuals["max_radial_equation_residual"]) < 1e-3


def test_ode_solver_stops_captured_ray_at_horizon():
    metric = SchwarzschildMetric(mass=1.0)
    result = OdeTrajectorySolver(metric=metric, observer=InfinityObserver()).trace_b(4.0)

    assert result.diagnostics.hit_horizon is True
    assert result.events[-1].event_type == EventType.HORIZON


def test_ode_solver_rejects_through_policy():
    metric = SchwarzschildMetric(mass=1.0)
    policy = ThroughTracePolicy(region_transitions={("external", EventType.HORIZON): "companion"})

    with pytest.raises(ValueError, match="only supports OutsideTracePolicy"):
        OdeTrajectorySolver(metric=metric, observer=InfinityObserver(), trace_policy=policy).trace_b(4.0)
