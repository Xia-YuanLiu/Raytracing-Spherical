import math

import pytest

from spherical_raytracing.diagnostics import EventType
from spherical_raytracing.metrics import SchwarzschildMetric
from spherical_raytracing.observers import FiniteStaticObserver
from spherical_raytracing.policies import ThroughTracePolicy
from spherical_raytracing.solvers import HamiltonianTrajectorySolver


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

    result = HamiltonianTrajectorySolver(metric=metric, observer=observer).trace_b(8.0)

    assert [segment.endpoint_event for segment in result.segments] == [
        EventType.TURNING_POINT,
        EventType.ESCAPE,
    ]
    assert result.diagnostics.termination_reason == EventType.ESCAPE.value
    assert result.diagnostics.residuals["turning_point_crossings"] == 1


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
