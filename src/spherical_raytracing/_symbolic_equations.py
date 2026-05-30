from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Any

import sympy as sp


@dataclass(frozen=True)
class SymbolicMetricFormula:
    name: str
    A: sp.Expr
    B: sp.Expr
    G: sp.Expr
    dA_dr: sp.Expr
    dB_dr: sp.Expr
    photon_sphere_condition: sp.Expr
    b_crit: sp.Expr
    parameters: tuple[sp.Symbol, ...]


r, u, b = sp.symbols("r u b", positive=True, real=True)
M, Q, Lam, alpha = sp.symbols("M Q Lambda alpha", real=True)
E, L = sp.symbols("E L", positive=True, real=True)
R = sp.symbols("R", positive=True, real=True)
A_emit, A_obs, A_from, A_to = sp.symbols(
    "A_emit A_obs A_from A_to",
    positive=True,
    real=True,
)


def static_spherical_metric_tensor() -> sp.Matrix:
    """Return the symbolic static spherical metric tensor."""
    t, radius, theta, phi = sp.symbols("t r theta phi", real=True)
    a = sp.Function("A")(radius)
    b_metric = sp.Function("B")(radius)
    return sp.diag(-a, b_metric, radius**2, radius**2 * sp.sin(theta) ** 2)


def derive_reduced_null_equation(A_expr: sp.Expr, B_expr: sp.Expr) -> sp.Expr:
    """Derive ``(du/dphi)^2`` from the null condition and constants ``E, L``."""
    rdot_squared = (E**2 / A_expr - L**2 / r**2) / B_expr
    dphi_dlambda = L / r**2
    du_dlambda_squared = rdot_squared / r**4
    du_dphi_squared = sp.simplify(du_dlambda_squared / dphi_dlambda**2)
    expression_with_impact = sp.simplify(du_dphi_squared.subs(E**2 / L**2, 1 / b**2))
    return sp.simplify(expression_with_impact.subs(r, 1 / u))


def derive_inverse_metric_reduced_null_equation(A_expr: sp.Expr) -> sp.Expr:
    """Specialise the reduced equation to ``B=1/A``."""
    return sp.simplify(derive_reduced_null_equation(A_expr, 1 / A_expr))


def derive_photon_sphere_condition(A_expr: sp.Expr) -> sp.Expr:
    """Return the static spherical photon-sphere equation ``r A' - 2 A``."""
    return sp.simplify(r * sp.diff(A_expr, r) - 2 * A_expr)


def derive_b_crit(A_expr: sp.Expr) -> sp.Expr:
    """Return ``b_crit(r) = r / sqrt(A(r))``."""
    return sp.simplify(r / sp.sqrt(A_expr))


def derive_finite_static_redshift() -> sp.Expr:
    """Return the finite static observer redshift factor."""
    return sp.sqrt(A_emit / A_obs)


def derive_static_shell_matching() -> dict[str, sp.Expr]:
    """Return the static shell energy, angular-momentum, and impact-parameter rules."""
    energy_ratio = sp.sqrt(A_to / A_from)
    return {
        "energy_ratio": energy_ratio,
        "L_to": L,
        "b_to": sp.simplify(L / (E * energy_ratio)),
    }


def verify_symbolic_identity(left: sp.Expr, right: sp.Expr) -> bool:
    """Return ``True`` when two symbolic expressions simplify to the same value."""
    return sp.simplify(left - right) == 0


def _family_a_expr(name: str) -> tuple[sp.Expr, tuple[sp.Symbol, ...]]:
    if name == "schwarzschild":
        return 1 - 2 * M / r, (M,)
    if name == "rn":
        return 1 - 2 * M / r + Q**2 / r**2, (M, Q)
    if name == "rnds":
        return 1 - 2 * M / r + Q**2 / r**2 - Lam * r**2 / 3, (M, Q, Lam)
    if name == "lqg":
        return 1 - 2 * M / r + alpha * M**2 / r**4, (M, alpha)
    raise ValueError(f"unknown metric family {name!r}")


@lru_cache(maxsize=None)
def metric_family_formulas(name: str) -> SymbolicMetricFormula:
    """Return symbolic formulas derived from a metric family's ``A(r)``."""
    a_expr, parameters = _family_a_expr(name)
    b_expr = sp.simplify(1 / a_expr)
    return SymbolicMetricFormula(
        name=name,
        A=sp.simplify(a_expr),
        B=b_expr,
        G=derive_inverse_metric_reduced_null_equation(a_expr),
        dA_dr=sp.diff(a_expr, r),
        dB_dr=sp.diff(b_expr, r),
        photon_sphere_condition=derive_photon_sphere_condition(a_expr),
        b_crit=derive_b_crit(a_expr),
        parameters=parameters,
    )


def positive_real_roots(expression: sp.Expr, substitutions: dict[sp.Symbol, Any]) -> list[float]:
    """Return positive real roots of ``expression == 0`` after substitution."""
    poly_expr = sp.factor(sp.together(expression.subs(substitutions)).as_numer_denom()[0])
    roots = sp.nroots(poly_expr)
    values = sorted(
        float(sp.re(root))
        for root in roots
        if abs(float(sp.im(root))) < 1e-10 and float(sp.re(root)) > 0.0
    )
    unique: list[float] = []
    for value in values:
        if not unique or abs(value - unique[-1]) > 1e-8 * max(abs(value), 1.0):
            unique.append(value)
    return unique


@lru_cache(maxsize=None)
def derive_christoffel_symbols(name: str) -> dict[tuple[int, int, int], sp.Expr]:
    """Derive equatorial Christoffel symbols for coordinates ``(t, r, phi)``."""
    formulas = metric_family_formulas(name)
    t, phi = sp.symbols("t phi", real=True)
    coords = (t, r, phi)
    metric = sp.diag(-formulas.A, formulas.B, r**2)
    inverse_metric = sp.simplify(metric.inv())
    christoffels: dict[tuple[int, int, int], sp.Expr] = {}
    for upper in range(3):
        for lower_a in range(3):
            for lower_b in range(3):
                value = 0
                for sigma in range(3):
                    value += inverse_metric[upper, sigma] * (
                        sp.diff(metric[sigma, lower_b], coords[lower_a])
                        + sp.diff(metric[sigma, lower_a], coords[lower_b])
                        - sp.diff(metric[lower_a, lower_b], coords[sigma])
                    )
                value = sp.simplify(value / 2)
                if value != 0:
                    christoffels[(upper, lower_a, lower_b)] = value
    return christoffels
