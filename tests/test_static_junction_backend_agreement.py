from dataclasses import replace

import pytest

from spherical_raytracing.junctions import StaticJunctionSpacetime, StaticShell, compare_junction_results
from spherical_raytracing.junction_tracing import (
    StaticJunctionHamiltonianSolver,
    StaticJunctionTransferSolver,
    trace_with_backend_compare,
)
from spherical_raytracing.metrics import SchwarzschildMetric
from spherical_raytracing.observers import FiniteStaticObserver
from spherical_raytracing.policies import SolverOptions
from spherical_raytracing.transfer import DiskWindow, compute_intersections


def make_schwarzschild_junction() -> tuple[StaticJunctionSpacetime, FiniteStaticObserver]:
    inner = SchwarzschildMetric(mass=0.1, region="inner")
    outer = SchwarzschildMetric(mass=1.0, region="outer")
    junction = StaticJunctionSpacetime(inner_metric=inner, outer_metric=outer, shell=StaticShell(radius=5.0))
    observer = FiniteStaticObserver(r_obs=50.0, metric=outer)
    return junction, observer


def make_aligned_backend_results():
    junction, observer = make_schwarzschild_junction()
    options = SolverOptions(inner_boundary_radius=1.0)
    transfer = StaticJunctionTransferSolver(junction=junction, observer=observer, options=options).trace_b(0.5)
    hamiltonian = StaticJunctionHamiltonianSolver(junction=junction, observer=observer, options=options).trace_b(0.5)
    return junction, observer, transfer, hamiltonian


def make_aligned_intersection_results():
    junction, observer = make_schwarzschild_junction()
    options = SolverOptions(inner_boundary_radius=1.0)
    transfer = StaticJunctionTransferSolver(junction=junction, observer=observer, options=options).trace_b(4.0)
    transfer_intersections = compute_intersections(
        transfer,
        DiskWindow(r_min=0.0, r_max=float("inf"), enabled_regions=frozenset({"inner", "outer"})),
        max_order=3,
    )
    assert transfer_intersections
    base = replace(transfer_intersections[0], diagnostics={"g": 0.7})
    transfer = replace(
        transfer,
        intersections=[base],
    )
    comparison = replace(
        transfer,
        events=transfer.events,
        segment_constants=transfer.segment_constants,
        intersections=[replace(base, r=base.r + 5e-4, diagnostics={"g": 0.7005})],
    )
    return transfer, comparison


def test_compare_junction_results_accepts_matching_backends():
    _, _, transfer, hamiltonian = make_aligned_backend_results()

    comparison = compare_junction_results(transfer, hamiltonian)

    assert comparison["backend_disagreement"] == 0
    assert comparison["shell_crossing_count_delta"] == 0


def test_compare_junction_results_uses_intersection_and_redshift_tolerances():
    transfer, hamiltonian = make_aligned_intersection_results()

    comparison = compare_junction_results(transfer, hamiltonian, r_intersection_abs=1e-3, g_abs=1e-3)

    assert comparison["backend_disagreement"] == 0
    assert comparison["max_intersection_r_delta"] == pytest.approx(5e-4)
    assert comparison["max_redshift_g_delta"] == pytest.approx(5e-4)

    hamiltonian = replace(
        hamiltonian,
        intersections=[replace(hamiltonian.intersections[0], diagnostics={"g": 0.702})],
    )

    comparison = compare_junction_results(transfer, hamiltonian, r_intersection_abs=1e-3, g_abs=1e-3)

    assert comparison["backend_disagreement"] == 1
    assert comparison["max_redshift_g_delta"] > 1e-3


def test_compare_junction_results_detects_event_identity_mismatches():
    _, _, transfer, hamiltonian = make_aligned_backend_results()

    transfer = replace(transfer, events=[replace(event, u=event.u + 5e-4) for event in transfer.events])
    comparison = compare_junction_results(transfer, hamiltonian, phi_event_abs=1e-6)
    assert comparison["backend_disagreement"] == 1

    _, _, transfer, hamiltonian = make_aligned_backend_results()
    hamiltonian = replace(hamiltonian, events=[replace(event, region=f"{event.region}_alt") for event in hamiltonian.events])
    comparison = compare_junction_results(transfer, hamiltonian, phi_event_abs=1e-6)
    assert comparison["backend_disagreement"] == 1

    _, _, transfer, hamiltonian = make_aligned_backend_results()
    hamiltonian = replace(hamiltonian, events=[hamiltonian.events[0]])
    comparison = compare_junction_results(transfer, hamiltonian, phi_event_abs=1e-6)
    assert comparison["backend_disagreement"] == 1


def test_compare_junction_results_detects_intersection_identity_mismatches():
    transfer, hamiltonian = make_aligned_intersection_results()

    comparison = compare_junction_results(transfer, hamiltonian, r_intersection_abs=1e-3, g_abs=1e-3)
    assert comparison["backend_disagreement"] == 0

    hamiltonian = replace(
        hamiltonian,
        intersections=[replace(hamiltonian.intersections[0], m=hamiltonian.intersections[0].m + 1)],
    )
    comparison = compare_junction_results(transfer, hamiltonian, r_intersection_abs=1e-3, g_abs=1e-3)
    assert comparison["backend_disagreement"] == 1


def test_trace_with_backend_compare_marks_result_diagnostics():
    junction, observer = make_schwarzschild_junction()

    result = trace_with_backend_compare(junction=junction, observer=observer, b_plus=4.0)

    assert result.backend_name == "transfer"
    assert "backend_disagreement" in result.diagnostics.residuals
    assert result.diagnostics.residuals["backend_disagreement"] == float(result.backend_comparison["backend_disagreement"])
    assert result.backend_comparison["primary_backend"] == "transfer"
    assert result.backend_comparison["comparison_backend"] == "hamiltonian"


def test_compare_junction_results_treats_escape_cutoffs_as_matching_boundaries():
    junction, observer = make_schwarzschild_junction()

    result = trace_with_backend_compare(junction=junction, observer=observer, b_plus=4.0)

    assert result.termination_reason == "escape"
    assert result.backend_comparison["backend_disagreement"] == 0
