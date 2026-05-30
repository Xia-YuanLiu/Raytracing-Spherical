from __future__ import annotations

import math

import numpy as np
import pytest
import sympy as sp
from scipy.integrate import solve_ivp

from spherical_raytracing._formula_verification import substitutions_for_metric
from spherical_raytracing._symbolic_equations import derive_christoffel_symbols, r
from spherical_raytracing.diagnostics import EventType
from spherical_raytracing.metrics import (
    LQGMetric,
    ReissnerNordstromDeSitterMetric,
    ReissnerNordstromMetric,
    SchwarzschildMetric,
)
from spherical_raytracing.observers import FiniteStaticObserver
from spherical_raytracing.policies import SolverOptions
from spherical_raytracing.solvers import QuadTransferSolver


def _family_name(metric) -> str:
    name = type(metric).__name__
    if name == "SchwarzschildMetric":
        return "schwarzschild"
    if name == "ReissnerNordstromMetric":
        return "rn"
    if name == "ReissnerNordstromDeSitterMetric":
        return "rnds"
    if name == "LQGMetric":
        return "lqg"
    raise ValueError(name)


def _christoffel_functions(metric):
    substitutions = substitutions_for_metric(metric)
    functions = {}
    for key, expr in derive_christoffel_symbols(_family_name(metric)).items():
        functions[key] = sp.lambdify(r, expr.subs(substitutions), "math")
    return functions


def _integrate_symbolic_geodesic(metric, *, r_obs: float, b_value: float, options: SolverOptions):
    observer = FiniteStaticObserver(r_obs=r_obs, metric=metric)
    gamma = _christoffel_functions(metric)
    energy = 1.0
    angular_momentum = b_value * energy
    a0 = metric.A(r_obs)
    b0 = metric.B(r_obs)
    vt0 = energy / a0
    vphi0 = angular_momentum / (r_obs * r_obs)
    vr0 = -math.sqrt((energy * energy / a0 - angular_momentum * angular_momentum / (r_obs * r_obs)) / b0)
    y0 = np.array([0.0, r_obs, 0.0, vt0, vr0, vphi0], dtype=float)

    def rhs(_lambda, y):
        radius = float(y[1])
        velocity = [float(y[3]), float(y[4]), float(y[5])]
        acceleration = [0.0, 0.0, 0.0]
        for (upper, lower_a, lower_b), func in gamma.items():
            acceleration[upper] -= float(func(radius)) * velocity[lower_a] * velocity[lower_b]
        return np.array([velocity[0], velocity[1], velocity[2], *acceleration])

    r_min, r_max = metric.valid_radial_domain()
    events = []
    event_types = []
    if r_min > 0.0 and math.isfinite(r_min):
        inner_stop = r_min / (1.0 - options.horizon_buffer)

        def event_inner(_lambda, y):
            return float(y[1]) - inner_stop

        event_inner.terminal = True
        event_inner.direction = -1.0
        events.append(event_inner)
        event_types.append(EventType.HORIZON)
    if math.isfinite(r_max):
        outer_stop = r_max / (1.0 + options.horizon_buffer)

        def event_outer(_lambda, y):
            return float(y[1]) - outer_stop

        event_outer.terminal = True
        event_outer.direction = 1.0
        events.append(event_outer)
        event_types.append(EventType.HORIZON)

    def event_max_phi(_lambda, y):
        return options.max_phi - float(y[2])

    event_max_phi.terminal = True
    event_max_phi.direction = -1.0
    events.append(event_max_phi)
    event_types.append(EventType.MAX_PHI)

    solution = solve_ivp(
        rhs,
        (0.0, 10_000.0),
        y0,
        method="DOP853",
        events=events,
        rtol=1e-9,
        atol=1e-11,
        dense_output=True,
        max_step=0.1,
    )
    event_hits = [
        (float(values[0]), event_types[index])
        for index, values in enumerate(solution.t_events)
        if len(values) > 0
    ]
    event_type = min(event_hits, key=lambda item: item[0])[1] if event_hits else EventType.SOLVER_FAILURE
    y_end = np.asarray(solution.y[:, -1], dtype=float)
    phi_grid = np.asarray(solution.y[2], dtype=float)
    r_grid = np.asarray(solution.y[1], dtype=float)
    order = np.argsort(phi_grid)
    phi_grid = phi_grid[order]
    r_grid = r_grid[order]
    keep = np.concatenate(([True], np.diff(phi_grid) > 1e-10))
    phi_grid = phi_grid[keep]
    r_grid = r_grid[keep]

    residuals = []
    for radius, vt, vr, vphi in zip(solution.y[1], solution.y[3], solution.y[4], solution.y[5]):
        residuals.append(abs(-metric.A(float(radius)) * vt * vt + metric.B(float(radius)) * vr * vr + radius * radius * vphi * vphi))

    return {
        "event_type": event_type,
        "phi_end": float(y_end[2]),
        "u_end": 1.0 / float(y_end[1]),
        "u_at": lambda phi: float(1.0 / np.interp(phi, phi_grid, r_grid)),
        "max_null_residual": max(residuals),
    }


@pytest.mark.slow
@pytest.mark.parametrize(
    "metric,r_obs,b_multiplier",
    [
        (SchwarzschildMetric(mass=1.0), 50.0, 0.70),
        (ReissnerNordstromMetric(mass=1.0, charge=0.5), 50.0, 0.70),
        (
            ReissnerNordstromDeSitterMetric(mass=1.0, charge=0.5, cosmological_constant=0.01),
            6.0,
            1.05,
        ),
        (LQGMetric(mass=1.0, alpha=0.25), 50.0, 0.70),
    ],
)
def test_quad_solver_matches_independent_symbolic_geodesic_oracle(metric, r_obs, b_multiplier):
    options = SolverOptions(critical_exclusion=0.0, horizon_buffer=1e-4)
    observer = FiniteStaticObserver(r_obs=r_obs, metric=metric)
    b_value = b_multiplier * metric.critical_curves()[-1].b_crit
    quad_result = QuadTransferSolver(metric=metric, observer=observer, options=options).trace_b(b_value)
    oracle = _integrate_symbolic_geodesic(metric, r_obs=r_obs, b_value=b_value, options=options)

    assert quad_result.segments
    assert oracle["max_null_residual"] < 1e-5
    assert quad_result.segments[-1].endpoint_event == oracle["event_type"]
    assert quad_result.segments[-1].phi_end == pytest.approx(oracle["phi_end"], rel=3e-5, abs=3e-5)
    assert quad_result.segments[-1].u_end == pytest.approx(oracle["u_end"], rel=3e-5, abs=3e-7)

    for segment in quad_result.segments:
        span = segment.phi_end - segment.phi_start
        if span <= 0.0:
            continue
        for fraction in (0.25, 0.5, 0.75):
            phi = segment.phi_start + fraction * span
            assert segment.u_at(phi) == pytest.approx(oracle["u_at"](phi), rel=3e-4, abs=3e-6)
