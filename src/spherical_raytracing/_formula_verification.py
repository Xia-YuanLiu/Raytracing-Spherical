from __future__ import annotations

import math
from collections.abc import Iterable
from typing import Any

import sympy as sp

from ._symbolic_equations import M, Q, Lam, alpha, b, metric_family_formulas, r, u


def assert_metric_matches_symbolic_oracle(
    metric: Any,
    family: str,
    substitutions: dict[sp.Symbol, float],
    *,
    r_samples: Iterable[float],
    b_samples: Iterable[float],
    rel_tol: float = 1e-10,
    abs_tol: float = 1e-10,
) -> None:
    """Assert a runtime metric matches the symbolic formulas for a family."""
    formulas = metric_family_formulas(family)
    for radius in r_samples:
        expected_a = _eval(formulas.A, substitutions, r, radius)
        expected_b = _eval(formulas.B, substitutions, r, radius)
        expected_da = _eval(formulas.dA_dr, substitutions, r, radius)
        expected_db = _eval(formulas.dB_dr, substitutions, r, radius)

        _assert_close("A", metric.A(radius), expected_a, rel_tol, abs_tol)
        _assert_close("B", metric.B(radius), expected_b, rel_tol, abs_tol)
        _assert_close("dA_dr", metric.dA_dr(radius), expected_da, rel_tol, abs_tol)
        _assert_close("dB_dr", metric.dB_dr(radius), expected_db, rel_tol * 20, abs_tol * 20)

        for impact_parameter in b_samples:
            expected_g = _eval2(
                formulas.G,
                substitutions,
                u,
                1.0 / radius,
                b,
                impact_parameter,
            )
            _assert_close(
                "G",
                metric.G(1.0 / radius, impact_parameter),
                expected_g,
                rel_tol * 20,
                abs_tol * 20,
            )


def substitutions_for_metric(metric: Any) -> dict[sp.Symbol, float]:
    """Return symbolic parameter substitutions for a runtime metric instance."""
    name = type(metric).__name__
    if name == "SchwarzschildMetric":
        return {M: float(metric.mass)}
    if name == "ReissnerNordstromMetric":
        return {M: float(metric.mass), Q: float(metric.charge)}
    if name == "ReissnerNordstromDeSitterMetric":
        return {
            M: float(metric.mass),
            Q: float(metric.charge),
            Lam: float(metric.cosmological_constant),
        }
    if name == "LQGMetric":
        return {M: float(metric.mass), alpha: float(metric.alpha)}
    raise ValueError(f"unsupported metric type {name!r}")


def _eval(expr: sp.Expr, substitutions: dict[sp.Symbol, float], variable: sp.Symbol, value: float) -> float:
    return float(expr.subs(substitutions).subs(variable, value).evalf(30))


def _eval2(
    expr: sp.Expr,
    substitutions: dict[sp.Symbol, float],
    variable_a: sp.Symbol,
    value_a: float,
    variable_b: sp.Symbol,
    value_b: float,
) -> float:
    return float(
        expr.subs(substitutions)
        .subs(variable_a, value_a)
        .subs(variable_b, value_b)
        .evalf(30)
    )


def _assert_close(name: str, actual: float, expected: float, rel_tol: float, abs_tol: float) -> None:
    assert math.isfinite(actual), f"{name}: actual value is not finite: {actual!r}"
    assert math.isfinite(expected), f"{name}: symbolic value is not finite: {expected!r}"
    assert math.isclose(actual, expected, rel_tol=rel_tol, abs_tol=abs_tol), (
        f"{name}: runtime metric differs from symbolic oracle "
        f"(actual={actual!r}, expected={expected!r})"
    )
