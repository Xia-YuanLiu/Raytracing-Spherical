from __future__ import annotations

import math

import pytest
import sympy as sp

from spherical_raytracing._formula_verification import (
    assert_metric_matches_symbolic_oracle,
    substitutions_for_metric,
)
from spherical_raytracing._symbolic_equations import (
    A_emit,
    A_from,
    A_obs,
    A_to,
    E,
    L,
    M,
    Q,
    Lam,
    alpha,
    b,
    derive_finite_static_redshift,
    derive_inverse_metric_reduced_null_equation,
    derive_photon_sphere_condition,
    derive_reduced_null_equation,
    derive_static_shell_matching,
    metric_family_formulas,
    positive_real_roots,
    r,
    u,
    verify_symbolic_identity,
)
from spherical_raytracing.junctions import StaticJunctionSpacetime, StaticShell, match_static_shell
from spherical_raytracing.metrics import (
    LQGMetric,
    ReissnerNordstromDeSitterMetric,
    ReissnerNordstromMetric,
    SchwarzschildMetric,
)
from spherical_raytracing.observers import FiniteStaticObserver
from spherical_raytracing.sources import ThinDiskSource
from spherical_raytracing.transfer import IntersectionRecord
from spherical_raytracing.diagnostics import EventType, RaySegment


METRIC_CASES = [
    ("schwarzschild", SchwarzschildMetric(mass=1.0), [3.2, 8.0, 40.0], [4.0, 8.0, 16.0]),
    ("rn", ReissnerNordstromMetric(mass=1.0, charge=0.5), [2.4, 8.0, 40.0], [4.0, 8.0, 16.0]),
    (
        "rnds",
        ReissnerNordstromDeSitterMetric(mass=1.0, charge=0.5, cosmological_constant=0.01),
        [2.5, 6.0, 10.0],
        [4.0, 8.0, 12.0],
    ),
    ("lqg", LQGMetric(mass=1.0, alpha=0.25), [2.4, 8.0, 40.0], [4.0, 8.0, 16.0]),
]


def test_reduced_null_equation_is_derived_from_null_condition():
    A = sp.Function("A")(r)
    B = sp.Function("B")(r)

    actual = derive_reduced_null_equation(A, B)
    expected = 1 / (b**2 * A.subs(r, 1 / u) * B.subs(r, 1 / u)) - u**2 / B.subs(r, 1 / u)

    assert verify_symbolic_identity(actual, expected)


def test_inverse_metric_reduced_null_equation_specialization():
    A = sp.Function("A")(r)

    actual = derive_inverse_metric_reduced_null_equation(A)
    expected = 1 / b**2 - u**2 * A.subs(r, 1 / u)

    assert verify_symbolic_identity(actual, expected)


@pytest.mark.parametrize("family,metric,r_samples,b_samples", METRIC_CASES)
def test_runtime_metric_methods_match_symbolic_oracle(family, metric, r_samples, b_samples):
    assert_metric_matches_symbolic_oracle(
        metric,
        family,
        substitutions_for_metric(metric),
        r_samples=r_samples,
        b_samples=b_samples,
    )


@pytest.mark.parametrize("family,metric,_r_samples,_b_samples", METRIC_CASES)
def test_photon_spheres_and_critical_curves_match_symbolic_oracle(
    family,
    metric,
    _r_samples,
    _b_samples,
):
    formulas = metric_family_formulas(family)
    substitutions = substitutions_for_metric(metric)
    symbolic_roots = [
        root
        for root in positive_real_roots(formulas.photon_sphere_condition, substitutions)
        if metric.A(root) > 0.0
    ]
    runtime_roots = metric.photon_spheres()

    assert runtime_roots == pytest.approx(symbolic_roots, rel=1e-9, abs=1e-9)
    for curve in metric.critical_curves():
        assert curve.b_crit == pytest.approx(
            curve.r_ph / math.sqrt(metric.A(curve.r_ph)),
            rel=1e-12,
            abs=1e-12,
        )


def test_metric_family_limit_identities_are_symbolically_verified():
    schwarzschild = metric_family_formulas("schwarzschild")
    rn = metric_family_formulas("rn")
    rnds = metric_family_formulas("rnds")
    lqg = metric_family_formulas("lqg")

    assert verify_symbolic_identity(rn.A.subs(Q, 0), schwarzschild.A)
    assert verify_symbolic_identity(rn.G.subs(Q, 0), schwarzschild.G)
    assert verify_symbolic_identity(rnds.A.subs(Lam, 0), rn.A)
    assert verify_symbolic_identity(rnds.G.subs(Lam, 0), rn.G)
    assert verify_symbolic_identity(lqg.A.subs(alpha, 0), schwarzschild.A)
    assert verify_symbolic_identity(lqg.G.subs(alpha, 0), schwarzschild.G)


def test_metric_family_oracle_g_is_derived_from_a_and_b():
    for family in ("schwarzschild", "rn", "rnds", "lqg"):
        formulas = metric_family_formulas(family)
        assert verify_symbolic_identity(
            formulas.G,
            derive_reduced_null_equation(formulas.A, formulas.B),
        )
        assert verify_symbolic_identity(
            formulas.photon_sphere_condition,
            derive_photon_sphere_condition(formulas.A),
        )


def test_finite_static_redshift_matches_source_helper():
    metric = SchwarzschildMetric(mass=1.0)
    observer = FiniteStaticObserver(r_obs=50.0, metric=metric)
    segment = RaySegment(
        region=metric.region,
        radial_direction="inward",
        phi_start=0.0,
        phi_end=1.0,
        u_start=1.0 / observer.r_obs,
        u_end=1.0 / 10.0,
        endpoint_event=EventType.TURNING_POINT,
        evaluator=lambda _phi: 1.0 / 10.0,
    )
    intersection = IntersectionRecord(
        m=1,
        r=10.0,
        phi=0.5,
        region=metric.region,
        ray_segment=segment,
        path_class="direct",
    )
    source = ThinDiskSource(lambda _r, _region: 1.0)
    symbolic_g = derive_finite_static_redshift().subs(
        {
            A_emit: metric.A(intersection.r),
            A_obs: metric.A(observer.r_obs),
        }
    )

    assert source.redshift_weight(intersection, metric, observer) == pytest.approx(
        float(symbolic_g**4),
        rel=1e-12,
        abs=1e-12,
    )


def test_static_shell_matching_matches_symbolic_oracle():
    inner = SchwarzschildMetric(mass=0.5, region="inner")
    outer = SchwarzschildMetric(mass=1.0, region="outer")
    junction = StaticJunctionSpacetime(inner_metric=inner, outer_metric=outer, shell=StaticShell(radius=10.0))
    before_e = 2.0
    before_l = 7.0
    before_b = before_l / before_e

    after = match_static_shell(
        junction,
        from_region="outer",
        to_region="inner",
        E=before_e,
        L=before_l,
        b=before_b,
    )
    matching = derive_static_shell_matching()
    substitutions = {
        A_from: outer.A(junction.shell.radius),
        A_to: inner.A(junction.shell.radius),
        E: before_e,
        L: before_l,
    }

    assert after.E / before_e == pytest.approx(float(matching["energy_ratio"].subs(substitutions)), abs=1e-12)
    assert after.L == pytest.approx(float(matching["L_to"].subs(substitutions)), abs=1e-12)
    assert after.b == pytest.approx(float(matching["b_to"].subs(substitutions)), abs=1e-12)


@pytest.mark.parametrize(
    ("metric", "r", "b"),
    [
        (SchwarzschildMetric(mass=1.3), 9.0, 7.0),
        (ReissnerNordstromMetric(mass=1.0, charge=0.4), 9.0, 7.0),
        (ReissnerNordstromDeSitterMetric(mass=1.0, charge=0.4, cosmological_constant=0.005), 8.0, 7.0),
        (LQGMetric(mass=1.0, alpha=0.25), 8.0, 7.0),
    ],
)
def test_radial_potential_matches_metric_coefficients(metric, r, b):
    u = 1.0 / r

    expected = 1.0 / (b**2 * metric.A(r) * metric.B(r)) - u**2 / metric.B(r)

    assert metric.G(u, b) == pytest.approx(expected)
