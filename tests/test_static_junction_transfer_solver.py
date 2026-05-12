import math

import pytest

from spherical_raytracing.diagnostics import EventType
from spherical_raytracing.junctions import StaticJunctionSpacetime, StaticShell, annotate_junction_intersections
from spherical_raytracing.junction_tracing import StaticJunctionTransferSolver
from spherical_raytracing.metrics import ReissnerNordstromDeSitterMetric, SchwarzschildMetric
from spherical_raytracing.observers import FiniteStaticObserver
from spherical_raytracing.policies import SolverOptions
from spherical_raytracing.transfer import DiskWindow, compute_intersections


def make_schwarzschild_junction(radius: float = 5.0) -> tuple[StaticJunctionSpacetime, FiniteStaticObserver]:
    inner = SchwarzschildMetric(mass=0.1, region="inner")
    outer = SchwarzschildMetric(mass=1.0, region="outer")
    junction = StaticJunctionSpacetime(inner_metric=inner, outer_metric=outer, shell=StaticShell(radius=radius))
    observer = FiniteStaticObserver(r_obs=50.0, metric=outer)
    return junction, observer


def test_transfer_solver_trace_b_records_shell_crossing_and_updates_local_b():
    junction, observer = make_schwarzschild_junction()
    solver = StaticJunctionTransferSolver(junction=junction, observer=observer)

    result = solver.trace_b(4.0)

    assert result.backend_name == "transfer"
    assert result.b_plus == 4.0
    assert result.shell_crossings
    first = result.shell_crossings[0]
    assert first.from_region == "outer"
    assert first.to_region == "inner"
    assert first.r == pytest.approx(5.0)
    assert first.b_after == pytest.approx(4.0 * math.sqrt(junction.outer_metric.A(5.0) / junction.inner_metric.A(5.0)))
    assert any(event.event_type is EventType.SHELL_CROSSING for event in result.events)


def test_transfer_solver_handles_turn_inside_and_cross_back_out():
    junction, observer = make_schwarzschild_junction()
    solver = StaticJunctionTransferSolver(junction=junction, observer=observer)

    result = solver.trace_b(4.0)

    assert len(result.shell_crossings) >= 2
    assert [crossing.from_region for crossing in result.shell_crossings[:2]] == ["outer", "inner"]
    assert [crossing.to_region for crossing in result.shell_crossings[:2]] == ["inner", "outer"]
    assert result.diagnostics.turning_point_count >= 1
    assert "shell_0_energy_ratio_abs" in result.diagnostics.residuals
    assert "shell_1_energy_ratio_abs" in result.diagnostics.residuals
    assert "max_energy_ratio_abs" in result.diagnostics.residuals


def test_transfer_solver_marks_inner_near_critical_after_shell_matching():
    junction, observer = make_schwarzschild_junction()
    inner_b_crit = junction.inner_metric.critical_curves()[0].b_crit
    scale = math.sqrt(junction.outer_metric.A(junction.shell.radius) / junction.inner_metric.A(junction.shell.radius))
    solver = StaticJunctionTransferSolver(junction=junction, observer=observer)

    result = solver.trace_b(inner_b_crit / scale)

    assert result.near_critical is True
    assert result.termination_reason == "near_critical"
    assert result.shell_crossings
    assert result.diagnostics.residuals["unresolved"] == 1.0
    assert result.diagnostics.residuals["shell_crossing_count"] == pytest.approx(1.0)
    assert "shell_0_energy_ratio_abs" in result.diagnostics.residuals


def test_transfer_solver_reaches_inner_boundary_when_requested():
    junction, observer = make_schwarzschild_junction(radius=5.0)
    solver = StaticJunctionTransferSolver(
        junction=junction,
        observer=observer,
        options=SolverOptions(inner_boundary_radius=1.0),
    )

    result = solver.trace_b(0.5)

    assert result.termination_reason == "inner_boundary"
    assert result.diagnostics.hit_inner_boundary is True
    assert result.events[-1].event_type is EventType.INNER_BOUNDARY


def test_transfer_solver_rejects_inner_boundary_outside_shell():
    junction, observer = make_schwarzschild_junction(radius=5.0)
    solver = StaticJunctionTransferSolver(
        junction=junction,
        observer=observer,
        options=SolverOptions(inner_boundary_radius=5.0),
    )

    with pytest.raises(ValueError, match="inner_boundary_radius must lie inside the shell"):
        solver.trace_b(0.5)


def test_transfer_solver_rejects_shell_outside_observer():
    junction, observer = make_schwarzschild_junction(radius=60.0)
    solver = StaticJunctionTransferSolver(junction=junction, observer=observer)

    with pytest.raises(ValueError, match="observer must be outside the shell"):
        solver.trace_b(4.0)


def test_transfer_solver_ray_turning_outside_shell_has_no_shell_crossing():
    junction, observer = make_schwarzschild_junction(radius=2.2)
    solver = StaticJunctionTransferSolver(junction=junction, observer=observer)

    result = solver.trace_b(8.0)

    assert result.shell_crossings == []
    assert all(segment.region == "outer" for segment in result.segments)
    assert result.termination_reason in {"escape", "outer_boundary", "max_phi"}


def test_transfer_solver_stops_at_rnds_horizon_before_inaccessible_shell():
    inner = ReissnerNordstromDeSitterMetric(mass=1.0, charge=0.5, cosmological_constant=0.01, region="inner")
    outer = ReissnerNordstromDeSitterMetric(mass=1.0, charge=0.5, cosmological_constant=0.01, region="outer")
    junction = StaticJunctionSpacetime(inner_metric=inner, outer_metric=outer, shell=StaticShell(radius=0.1))
    observer = FiniteStaticObserver(r_obs=6.0, metric=outer)
    solver = StaticJunctionTransferSolver(junction=junction, observer=observer)

    result = solver.trace_b(0.01)

    assert result.shell_crossings == []
    assert result.termination_reason == "horizon"
    assert result.diagnostics.hit_horizon is True
    assert result.events[-1].event_type is EventType.HORIZON
    assert "outer_rnds_no_clean_static_patch" in result.physics_warnings


def test_transfer_solver_trace_alpha_uses_finite_observer_b_plus():
    junction, observer = make_schwarzschild_junction()
    solver = StaticJunctionTransferSolver(junction=junction, observer=observer)
    alpha = 0.08

    result = solver.trace_alpha(alpha)

    assert result.coordinate_name == "alpha"
    assert result.input_coordinate == alpha
    assert result.b_plus == pytest.approx(observer.impact_parameter(alpha))


def test_transfer_intersections_can_be_annotated_with_junction_redshift():
    junction, observer = make_schwarzschild_junction()
    solver = StaticJunctionTransferSolver(junction=junction, observer=observer)
    result = solver.trace_b(4.0)
    intersections = compute_intersections(
        result,
        DiskWindow(r_min=0.3, r_max=20.0, enabled_regions=frozenset({"inner", "outer"})),
        max_order=3,
    )

    annotated = annotate_junction_intersections(
        intersections,
        result,
        observer,
        {"inner": junction.inner_metric, "outer": junction.outer_metric},
    )

    assert annotated
    assert all("g" in item.diagnostics for item in annotated)
    assert all("redshift_weight" in item.diagnostics for item in annotated)
