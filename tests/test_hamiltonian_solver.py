import math

import pytest

from spherical_raytracing.diagnostics import EventType
from spherical_raytracing.metrics import ReissnerNordstromDeSitterMetric, SchwarzschildMetric
from spherical_raytracing.observers import FiniteStaticObserver
from spherical_raytracing.policies import ThroughTracePolicy
from spherical_raytracing.solvers import HamiltonianTrajectorySolver, _mirror_escape_segment


def test_hamiltonian_solver_preserves_constraint_outside_horizon():
    metric = SchwarzschildMetric(mass=1.0)
    observer = FiniteStaticObserver(r_obs=50.0, metric=metric)
    b = 8.0
    result = HamiltonianTrajectorySolver(metric=metric, observer=observer).trace_b(b)

    assert abs(result.diagnostics.residuals["max_hamiltonian_constraint"]) < 1e-6
    assert abs(result.diagnostics.residuals["energy_residual"]) < 1e-12
    assert abs(result.diagnostics.residuals["angular_momentum_residual"]) < 1e-12
    assert abs(result.diagnostics.residuals["impact_parameter_residual"]) < 1e-12


def test_hamiltonian_solver_detects_turning_point_with_pr_sign_change():
    metric = SchwarzschildMetric(mass=1.0)
    observer = FiniteStaticObserver(r_obs=50.0, metric=metric)

    result = HamiltonianTrajectorySolver(metric=metric, observer=observer).trace_b(8.0)

    assert result.diagnostics.turning_point_count == 1
    assert EventType.TURNING_POINT in [event.event_type for event in result.events]


def test_hamiltonian_solver_returns_escape_segment_after_turning_point():
    metric = SchwarzschildMetric(mass=1.0)
    observer = FiniteStaticObserver(r_obs=50.0, metric=metric)
    solver = HamiltonianTrajectorySolver(metric=metric, observer=observer)

    result = solver.trace_b(8.0)

    assert [segment.endpoint_event for segment in result.segments] == [
        EventType.TURNING_POINT,
        EventType.ESCAPE,
    ]
    assert result.diagnostics.termination_reason == EventType.ESCAPE.value
    assert result.diagnostics.residuals["turning_point_crossings"] == 1

    inbound, outbound = result.segments
    expected = _mirror_escape_segment(inbound, solver.options.max_phi)
    assert outbound.radial_direction == "outward"
    assert outbound.phi_start == pytest.approx(expected.phi_start, abs=1e-12)
    assert outbound.phi_end == pytest.approx(expected.phi_end, abs=1e-10)
    assert outbound.u_start == pytest.approx(expected.u_start, abs=1e-12)
    assert outbound.u_end == pytest.approx(expected.u_end, abs=1e-10)
    for fraction in (0.25, 0.50, 0.75):
        phi = outbound.phi_start + fraction * (outbound.phi_end - outbound.phi_start)
        assert outbound.u_at(phi) == pytest.approx(expected.u_at(phi), abs=1e-10)


def test_hamiltonian_solver_outbound_reaches_finite_outer_boundary():
    metric = ReissnerNordstromDeSitterMetric(
        mass=1.0,
        charge=0.0,
        cosmological_constant=0.01,
    )
    observer = FiniteStaticObserver(r_obs=6.0, metric=metric)
    b_crit = metric.critical_curves()[0].b_crit
    _r_h, r_c = metric.horizons()

    result = HamiltonianTrajectorySolver(metric=metric, observer=observer).trace_b(1.05 * b_crit)

    assert len(result.segments) >= 2
    outbound = result.segments[-1]
    assert outbound.radial_direction == "outward"
    assert outbound.endpoint_event in {EventType.HORIZON, EventType.OUTER_BOUNDARY}
    r_end = 1.0 / outbound.u_end
    assert math.isclose(r_end, r_c, rel_tol=2e-6, abs_tol=2e-6)
    assert not math.isclose(r_end, observer.r_obs, rel_tol=1e-3, abs_tol=1e-3)
    assert result.diagnostics.termination_reason == "cosmological_horizon"




def test_hamiltonian_solver_excludes_near_critical_ray():
    metric = SchwarzschildMetric(mass=1.0)
    observer = FiniteStaticObserver(r_obs=50.0, metric=metric)
    b_crit = metric.critical_curves()[0].b_crit

    result = HamiltonianTrajectorySolver(metric=metric, observer=observer).trace_b(b_crit)

    assert result.segments == []
    assert result.diagnostics.near_critical is True
    assert result.diagnostics.termination_reason == "near_critical"

def test_hamiltonian_solver_terminates_at_horizon_for_captured_ray():
    metric = SchwarzschildMetric(mass=1.0)
    observer = FiniteStaticObserver(r_obs=50.0, metric=metric)

    result = HamiltonianTrajectorySolver(metric=metric, observer=observer).trace_b(4.0)

    assert result.diagnostics.hit_horizon is True
    assert result.events[-1].event_type == EventType.HORIZON


def test_hamiltonian_solver_rejects_through_policy():
    metric = SchwarzschildMetric(mass=1.0)
    observer = FiniteStaticObserver(r_obs=50.0, metric=metric)

    with pytest.raises(ValueError, match="only supports OutsideTracePolicy"):
        HamiltonianTrajectorySolver(
            metric=metric,
            observer=observer,
            trace_policy=ThroughTracePolicy(region_transitions={("external", EventType.HORIZON): "companion"}),
        ).trace_b(4.0)
