import math

import pytest

from spherical_raytracing.junctions import StaticJunctionSpacetime, StaticShell
from spherical_raytracing.junction_tracing import StaticJunctionHamiltonianSolver
from spherical_raytracing.metrics import SchwarzschildMetric
from spherical_raytracing.observers import FiniteStaticObserver
from spherical_raytracing.policies import SolverOptions


def make_schwarzschild_junction(radius: float = 5.0) -> tuple[StaticJunctionSpacetime, FiniteStaticObserver]:
    inner = SchwarzschildMetric(mass=0.1, region="inner")
    outer = SchwarzschildMetric(mass=1.0, region="outer")
    junction = StaticJunctionSpacetime(inner_metric=inner, outer_metric=outer, shell=StaticShell(radius=radius))
    observer = FiniteStaticObserver(r_obs=50.0, metric=outer)
    return junction, observer


def test_hamiltonian_solver_records_shell_crossing_and_residuals():
    junction, observer = make_schwarzschild_junction()
    solver = StaticJunctionHamiltonianSolver(junction=junction, observer=observer)

    result = solver.trace_b(4.0)

    assert result.backend_name == "hamiltonian"
    assert result.shell_crossings
    assert result.diagnostics.residuals["max_abs_H"] < 1e-6
    assert result.diagnostics.residuals["max_energy_drift"] < 1e-6
    assert result.diagnostics.residuals["max_angular_momentum_drift"] < 1e-6


def test_hamiltonian_solver_trace_alpha_uses_b_plus():
    junction, observer = make_schwarzschild_junction()
    solver = StaticJunctionHamiltonianSolver(junction=junction, observer=observer)
    alpha = 0.08

    result = solver.trace_alpha(alpha)

    assert result.b_plus == pytest.approx(observer.impact_parameter(alpha))
    assert result.coordinate_name == "alpha"


def test_hamiltonian_solver_rejects_shell_outside_observer():
    junction, observer = make_schwarzschild_junction(radius=60.0)
    solver = StaticJunctionHamiltonianSolver(junction=junction, observer=observer)

    with pytest.raises(ValueError, match="observer must be outside the shell"):
        solver.trace_b(4.0)


def test_hamiltonian_solver_marks_inner_near_critical_after_shell_matching():
    junction, observer = make_schwarzschild_junction()
    inner_b_crit = junction.inner_metric.critical_curves()[0].b_crit
    scale = math.sqrt(junction.outer_metric.A(junction.shell.radius) / junction.inner_metric.A(junction.shell.radius))
    solver = StaticJunctionHamiltonianSolver(junction=junction, observer=observer)

    result = solver.trace_b(inner_b_crit / scale)

    assert result.near_critical is True
    assert result.termination_reason == "near_critical"
    assert result.shell_crossings
    assert result.diagnostics.residuals["max_abs_H"] < 1e-6
    assert result.diagnostics.residuals["max_energy_drift"] < 1e-6
    assert result.diagnostics.residuals["max_angular_momentum_drift"] < 1e-6
    assert result.diagnostics.residuals["shell_crossing_count"] == pytest.approx(1.0)
    assert result.diagnostics.residuals["unresolved"] == 1.0


def test_hamiltonian_solver_reaches_inner_boundary_when_requested():
    junction, observer = make_schwarzschild_junction(radius=5.0)
    solver = StaticJunctionHamiltonianSolver(
        junction=junction,
        observer=observer,
        options=SolverOptions(inner_boundary_radius=1.0),
    )

    result = solver.trace_b(0.5)

    assert result.termination_reason == "inner_boundary"
    assert result.diagnostics.hit_inner_boundary is True
    assert result.events[-1].event_type == "inner_boundary"


def test_hamiltonian_solver_rejects_inner_boundary_outside_shell():
    junction, observer = make_schwarzschild_junction(radius=5.0)
    solver = StaticJunctionHamiltonianSolver(
        junction=junction,
        observer=observer,
        options=SolverOptions(inner_boundary_radius=5.0),
    )

    with pytest.raises(ValueError, match="inner_boundary_radius must lie inside the shell"):
        solver.trace_b(0.5)


def test_hamiltonian_solver_reports_solver_failure_when_lambda_bound_is_reached():
    junction, observer = make_schwarzschild_junction()
    solver = StaticJunctionHamiltonianSolver(junction=junction, observer=observer, max_lambda=0.001)

    result = solver.trace_b(4.0)

    assert result.termination_reason == "solver_failure"
    assert result.diagnostics.max_phi_reached is False
    assert result.events[-1].event_type == "solver_failure"


def test_hamiltonian_solver_escapes_in_infinite_outer_domain():
    junction, observer = make_schwarzschild_junction()
    solver = StaticJunctionHamiltonianSolver(junction=junction, observer=observer)

    result = solver.trace_b(4.0)

    assert result.termination_reason == "escape"
    assert result.events[-1].event_type == "escape"
