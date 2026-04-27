import math

from spherical_raytracing.diagnostics import EventType
from spherical_raytracing.metrics import LQGMetric, SchwarzschildMetric
from spherical_raytracing.observers import InfinityObserver
from spherical_raytracing.policies import OutsideTracePolicy, SolverOptions, ThroughTracePolicy
from spherical_raytracing.solvers import QuadTransferSolver
from spherical_raytracing.transfer import DiskWindow, compute_intersections


def test_quad_solver_marks_near_critical_without_claiming_accuracy():
    metric = SchwarzschildMetric(mass=1.0)
    b_crit = metric.critical_curves()[0].b_crit
    solver = QuadTransferSolver(
        metric=metric,
        observer=InfinityObserver(),
        trace_policy=OutsideTracePolicy(),
        options=SolverOptions(critical_exclusion=1e-4),
    )

    result = solver.trace_b(b_crit * (1.0 + 1e-5))

    assert result.diagnostics.near_critical is True
    assert result.diagnostics.critical_curve_id == "external:photon_sphere:0"
    assert result.diagnostics.hit_horizon is False
    assert result.diagnostics.termination_reason == "near_critical"


def test_quad_solver_exact_critical_ray_does_not_claim_horizon_hit():
    metric = SchwarzschildMetric(mass=1.0)
    b_crit = metric.critical_curves()[0].b_crit
    solver = QuadTransferSolver(metric=metric, observer=InfinityObserver())

    result = solver.trace_b(b_crit)

    assert result.diagnostics.near_critical is True
    assert result.diagnostics.hit_horizon is False
    assert result.events == []


def test_quad_solver_max_phi_truncates_segment_state_consistently():
    metric = SchwarzschildMetric(mass=1.0)
    solver = QuadTransferSolver(
        metric=metric,
        observer=InfinityObserver(),
        options=SolverOptions(max_phi=0.1),
    )

    result = solver.trace_b(4.0)

    assert result.diagnostics.max_phi_reached is True
    assert result.diagnostics.hit_horizon is False
    assert result.diagnostics.termination_reason == EventType.MAX_PHI.value
    assert result.events[-1].event_type == EventType.MAX_PHI
    assert result.segments[-1].endpoint_event == EventType.MAX_PHI
    assert math.isclose(result.segments[-1].u_end, result.segments[-1].u_at(result.segments[-1].phi_end))


def test_quad_solver_splits_escape_ray_at_turning_point():
    metric = SchwarzschildMetric(mass=1.0)
    solver = QuadTransferSolver(metric=metric, observer=InfinityObserver())

    result = solver.trace_b(8.0)

    assert result.diagnostics.near_critical is False
    assert result.diagnostics.turning_point_count == 1
    assert [segment.endpoint_event for segment in result.segments] == [
        EventType.TURNING_POINT,
        EventType.ESCAPE,
    ]
    assert result.segments[0].radial_direction == "inward"
    assert result.segments[1].radial_direction == "outward"


def test_quad_solver_captured_ray_hits_horizon():
    metric = SchwarzschildMetric(mass=1.0)
    solver = QuadTransferSolver(metric=metric, observer=InfinityObserver())

    result = solver.trace_b(4.0)

    assert result.diagnostics.hit_horizon is True
    assert result.events[-1].event_type == EventType.HORIZON


def test_quad_solver_records_policy_region_transition_at_horizon():
    metric = SchwarzschildMetric(mass=1.0)
    policy = ThroughTracePolicy(region_transitions={("external", EventType.HORIZON): "companion"})
    solver = QuadTransferSolver(metric=metric, observer=InfinityObserver(), trace_policy=policy)

    result = solver.trace_b(4.0)

    assert result.diagnostics.hit_horizon is True
    assert result.diagnostics.termination_reason == EventType.REGION_TRANSITION.value
    assert result.events[-2].event_type == EventType.HORIZON
    assert result.events[-1].event_type == EventType.REGION_TRANSITION
    assert result.events[-1].region == "companion"


def test_quad_solver_through_policy_continues_lqg_ray_into_companion_region():
    metric = LQGMetric(mass=1.0, alpha=0.25)
    policy = ThroughTracePolicy(region_transitions={("external", EventType.HORIZON): "companion"})
    solver = QuadTransferSolver(
        metric=metric,
        observer=InfinityObserver(),
        trace_policy=policy,
        options=SolverOptions(critical_exclusion=0.0, max_phi=40.0),
    )

    result = solver.trace_b(4.0)
    companion_segments = [segment for segment in result.segments if segment.region == "companion"]
    intersections = compute_intersections(
        result,
        DiskWindow(r_min=metric.photon_spheres()[0], r_max=100.0, enabled_regions=frozenset({"companion"})),
        max_order=6,
    )

    assert result.diagnostics.termination_reason == EventType.ESCAPE.value
    assert companion_segments
    assert companion_segments[-1].endpoint_event == EventType.ESCAPE
    assert intersections
    assert {record.region for record in intersections} == {"companion"}


def test_quad_solver_weak_deflection_matches_schwarzschild_limit():
    metric = SchwarzschildMetric(mass=1.0)
    solver = QuadTransferSolver(metric=metric, observer=InfinityObserver())
    b = 500.0

    result = solver.trace_b(b)
    phi_turn = result.segments[0].phi_end
    bending = 2.0 * phi_turn - math.pi

    assert math.isclose(bending, 4.0 * metric.mass / b, rel_tol=0.02)
