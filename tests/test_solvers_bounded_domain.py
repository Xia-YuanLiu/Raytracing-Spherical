"""Regression tests for the bounded static patch (RNdS r_h..r_c) on QuadTransferSolver.

Decoupled from the numerical metric work; uses the analytic
``ReissnerNordstromDeSitterMetric`` so we can pin down inward/outbound termination
and the ``b <= b_max`` admissibility check on a deterministic patch.
"""

from __future__ import annotations

import math

import pytest

from spherical_raytracing.diagnostics import EventType
from spherical_raytracing.metrics import ReissnerNordstromDeSitterMetric
from spherical_raytracing.observers import FiniteStaticObserver
from spherical_raytracing.solvers import OdeTrajectorySolver, QuadTransferSolver


@pytest.fixture
def rnds_setup():
    metric = ReissnerNordstromDeSitterMetric(mass=1.0, charge=0.0, cosmological_constant=0.01)
    observer = FiniteStaticObserver(r_obs=6.0, metric=metric)
    solver = QuadTransferSolver(metric=metric, observer=observer)
    b_crit = metric.critical_curves()[0].b_crit
    r_h, r_c = metric.horizons()
    return {
        "metric": metric,
        "observer": observer,
        "solver": solver,
        "b_crit": b_crit,
        "r_h": r_h,
        "r_c": r_c,
    }


def test_rnds_inward_below_half_b_crit_terminates_at_horizon_with_positive_phi(rnds_setup):
    """b = 0.5 * b_crit cannot turn; it must fall into r_h with phi_end > 0."""
    solver = rnds_setup["solver"]
    b_crit = rnds_setup["b_crit"]
    r_h = rnds_setup["r_h"]

    result = solver.trace_b(0.5 * b_crit)

    assert result.segments, "ray must produce at least one segment"
    last = result.segments[-1]
    assert last.endpoint_event == EventType.HORIZON
    assert last.phi_end > 0.0
    assert last.radial_direction == "inward"
    r_end = 1.0 / last.u_end
    assert math.isclose(r_end, r_h, rel_tol=1e-6, abs_tol=1e-6)


def test_rnds_inward_just_below_critical_terminates_at_horizon_with_positive_phi(rnds_setup):
    """b = 0.95 * b_crit still plunges (no turning point) and must reach r_h."""
    solver = rnds_setup["solver"]
    b_crit = rnds_setup["b_crit"]
    r_h = rnds_setup["r_h"]

    result = solver.trace_b(0.95 * b_crit)

    assert result.segments, "ray must produce at least one segment"
    assert len(result.segments) == 1, "no turning point expected below b_crit"
    last = result.segments[-1]
    assert last.endpoint_event == EventType.HORIZON
    assert last.phi_end > 0.0
    assert last.radial_direction == "inward"
    r_end = 1.0 / last.u_end
    assert math.isclose(r_end, r_h, rel_tol=1e-6, abs_tol=1e-6)
    assert result.diagnostics.hit_horizon is True


def test_rnds_trace_b_rejects_impact_parameter_above_screen_limit(rnds_setup):
    """``b > b_max = r_obs / sqrt(A(r_obs))`` exceeds the screen radius and must hard-fail.

    Plan decision 6: ``FiniteStaticObserver`` admits at most ``b_max`` (the
    image-plane edge ``alpha = pi/2``); pretending to trace beyond that is a
    silent contract violation.
    """
    metric = rnds_setup["metric"]
    observer = rnds_setup["observer"]
    solver = rnds_setup["solver"]

    b_max = observer.r_obs / math.sqrt(metric.A(observer.r_obs))
    b_over = b_max * 1.000001

    with pytest.raises(ValueError, match="screen radius"):
        solver.trace_b(b_over)


def test_rnds_outbound_just_above_critical_terminates_at_cosmological_radius(rnds_setup):
    """b = 1.05 * b_crit turns and the outbound segment must reach r_c (not r_obs).

    Plan decision 6: when ``r_max < inf`` the outbound segment is integrated
    independently from ``u_turn`` down to ``u_outward_stop``; the prior
    ``phi``-mirror could never overshoot ``r_obs`` and therefore left the ray
    falsely escaping at the observer radius.
    """
    solver = rnds_setup["solver"]
    observer = rnds_setup["observer"]
    b_crit = rnds_setup["b_crit"]
    r_c = rnds_setup["r_c"]

    result = solver.trace_b(1.05 * b_crit)

    assert result.segments, "ray must produce at least one segment"
    assert len(result.segments) >= 2, "expected inbound + outbound segments"
    inbound, outbound = result.segments[0], result.segments[-1]
    assert inbound.radial_direction == "inward"
    assert inbound.endpoint_event == EventType.TURNING_POINT
    assert outbound.radial_direction == "outward"
    assert outbound.endpoint_event in {EventType.HORIZON, EventType.OUTER_BOUNDARY}
    r_end = 1.0 / outbound.u_end
    assert math.isclose(r_end, r_c, rel_tol=1e-6, abs_tol=1e-6)
    assert not math.isclose(r_end, observer.r_obs, rel_tol=1e-3, abs_tol=1e-3)
    assert outbound.phi_end > math.pi / 2
    assert outbound.phi_end < solver.options.max_phi


def test_ode_rnds_outbound_just_above_critical_terminates_at_cosmological_radius(rnds_setup):
    """OdeTrajectorySolver must follow QuadTransferSolver's bounded-domain outbound.

    Plan line 8: ``OdeTrajectorySolver follows QuadTransferSolver's static-domain
    fix``. When the outer region carries a finite ``r_max`` (e.g. Λ>0 cosmological
    horizon), the ODE solver must integrate outward independently to the outer
    edge rather than ``phi``-mirror back to ``r_obs``.
    """
    metric = rnds_setup["metric"]
    observer = rnds_setup["observer"]
    b_crit = rnds_setup["b_crit"]
    r_c = rnds_setup["r_c"]

    solver = OdeTrajectorySolver(metric=metric, observer=observer)
    result = solver.trace_b(1.05 * b_crit)

    assert result.segments, "ray must produce at least one segment"
    assert len(result.segments) >= 2, "expected inbound + outbound segments"
    outbound = result.segments[-1]
    assert outbound.radial_direction == "outward"
    assert outbound.endpoint_event in {EventType.HORIZON, EventType.OUTER_BOUNDARY}
    r_end = 1.0 / outbound.u_end
    assert math.isclose(r_end, r_c, rel_tol=1e-4, abs_tol=1e-4)
    assert not math.isclose(r_end, observer.r_obs, rel_tol=1e-3, abs_tol=1e-3)
    assert outbound.phi_end > math.pi / 2
    assert outbound.phi_end < solver.options.max_phi
