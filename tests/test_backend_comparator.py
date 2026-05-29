import math

from spherical_raytracing.metrics import (
    ReissnerNordstromDeSitterMetric,
    ReissnerNordstromMetric,
    SchwarzschildMetric,
)
from spherical_raytracing.observers import FiniteStaticObserver
from spherical_raytracing.solvers import (
    HamiltonianTrajectorySolver,
    OdeTrajectorySolver,
    QuadTransferSolver,
)


PROBE_FRACTIONS = (0.25, 0.5, 0.75)


def _assert_close(observable, actual, expected, *, rtol, atol):
    assert math.isclose(actual, expected, rel_tol=rtol, abs_tol=atol), (
        f"{observable} diverged: actual={actual!r}, expected={expected!r}, "
        f"delta={actual - expected!r}"
    )


def assert_backends_match(
    metric,
    *,
    r_obs,
    impact_parameters,
    phi_atol=3e-4,
    u_atol=7e-5,
    hamiltonian_constraint_atol=1e-6,
):
    observer = FiniteStaticObserver(r_obs=r_obs, metric=metric)
    reference_solver = QuadTransferSolver(metric=metric, observer=observer)

    for b in impact_parameters:
        reference = reference_solver.trace_b(b)
        assert not reference.diagnostics.near_critical, f"quad b={b} unexpectedly near critical"
        assert reference.segments, f"quad b={b} produced no segments"

        for backend_name, solver_cls in (
            ("ode", OdeTrajectorySolver),
            ("hamiltonian", HamiltonianTrajectorySolver),
        ):
            actual = solver_cls(metric=metric, observer=observer).trace_b(b)
            _assert_backend_result_matches_reference(
                backend_name,
                b,
                actual,
                reference,
                phi_atol=phi_atol,
                u_atol=u_atol,
            )
            _assert_backend_diagnostics(
                backend_name,
                b,
                actual,
                hamiltonian_constraint_atol=hamiltonian_constraint_atol,
            )


def _assert_backend_result_matches_reference(
    backend_name,
    b,
    actual,
    reference,
    *,
    phi_atol,
    u_atol,
):
    context = f"{backend_name} b={b}"

    assert actual.diagnostics.termination_reason == reference.diagnostics.termination_reason, (
        f"{context} termination_reason diverged: "
        f"{actual.diagnostics.termination_reason!r} != {reference.diagnostics.termination_reason!r}"
    )
    assert len(actual.segments) == len(reference.segments), (
        f"{context} segment count diverged: {len(actual.segments)} != {len(reference.segments)}"
    )
    assert [segment.endpoint_event for segment in actual.segments] == [
        segment.endpoint_event for segment in reference.segments
    ], f"{context} segment endpoint events diverged"

    last_actual = actual.segments[-1]
    last_reference = reference.segments[-1]
    _assert_close(
        f"{context} last phi_end",
        last_actual.phi_end,
        last_reference.phi_end,
        rtol=1e-4,
        atol=phi_atol,
    )
    _assert_close(
        f"{context} last u_end",
        last_actual.u_end,
        last_reference.u_end,
        rtol=1e-5,
        atol=u_atol,
    )

    for segment_index, (actual_segment, reference_segment) in enumerate(
        zip(actual.segments, reference.segments)
    ):
        phi_lo = max(actual_segment.phi_start, reference_segment.phi_start)
        phi_hi = min(actual_segment.phi_end, reference_segment.phi_end)
        assert phi_hi > phi_lo, f"{context} segment {segment_index} has no shared phi interval"

        for fraction in PROBE_FRACTIONS:
            phi = phi_lo + fraction * (phi_hi - phi_lo)
            _assert_close(
                f"{context} segment {segment_index} u_at fraction={fraction}",
                actual_segment.u_at(phi),
                reference_segment.u_at(phi),
                rtol=2e-3,
                atol=u_atol,
            )


def _assert_backend_diagnostics(
    backend_name,
    b,
    result,
    *,
    hamiltonian_constraint_atol,
):
    if backend_name == "ode":
        residual = result.diagnostics.residuals["max_radial_equation_residual"]
        assert math.isfinite(residual), f"ode b={b} radial residual is not finite"
        assert residual < 1e-4, f"ode b={b} radial residual too large: {residual!r}"
        return

    residuals = result.diagnostics.residuals
    for key in ("energy_residual", "angular_momentum_residual", "impact_parameter_residual"):
        residual = residuals[key]
        assert math.isfinite(residual), f"hamiltonian b={b} {key} is not finite"
        assert residual < 1e-10, f"hamiltonian b={b} {key} too large: {residual!r}"

    constraint = residuals["max_hamiltonian_constraint"]
    assert math.isfinite(constraint), f"hamiltonian b={b} constraint is not finite"
    if hamiltonian_constraint_atol is not None:
        assert abs(constraint) < hamiltonian_constraint_atol, (
            f"hamiltonian b={b} constraint too large: {constraint!r}"
        )


def test_schwarzschild_backends_agree_on_benign_finite_observer_rays():
    """ODE/Hamiltonian agreement smoke; not a fully independent physical oracle."""
    metric = SchwarzschildMetric(mass=1.0)

    assert_backends_match(metric, r_obs=50.0, impact_parameters=(8.0, 12.0, 20.0))


def test_reissner_nordstrom_backends_agree_on_warm_benign_finite_observer_rays():
    """ODE/Hamiltonian agreement smoke; not a fully independent physical oracle."""
    metric = ReissnerNordstromMetric(mass=1.0, charge=0.35)

    assert_backends_match(metric, r_obs=50.0, impact_parameters=(8.0, 12.0, 20.0))


def test_reissner_nordstrom_de_sitter_backends_agree_on_warm_benign_finite_observer_rays():
    """ODE/Hamiltonian agreement smoke; not a fully independent physical oracle."""
    metric = ReissnerNordstromDeSitterMetric(
        mass=1.0,
        charge=0.2,
        cosmological_constant=0.005,
    )

    assert_backends_match(
        metric,
        r_obs=10.0,
        impact_parameters=(8.0, 10.0),
        hamiltonian_constraint_atol=None,
    )
