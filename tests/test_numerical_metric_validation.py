"""Manufactured numerical-metric validation against analytical ground truth.

Default phase for ``docs/plans/numerical-metric-validation-plan.md``:

- L1: core metric-component validation for metadata-rich and standard r/A/B payloads.
- L2: small b-sweep ray validation on representative analytical families.
- L3: one thin-disk intersection plus observed_intensity smoke.
- L4: deterministic Schwarzschild grid-convergence smoke on benign rays.

TODO(L5): add a single-metric backend sanity comparator for Quad/ODE/Hamiltonian
once that public comparator exists.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pytest

from spherical_raytracing import (
    CanonicalPayload,
    DiskWindow,
    FiniteStaticObserver,
    LiDMBHMetadata,
    QuadTransferSolver,
    ReissnerNordstromDeSitterMetric,
    ReissnerNordstromMetric,
    SchwarzschildMetric,
    TabulatedMetric,
    ThinDiskSource,
    compute_intersections,
    observed_intensity,
)
from spherical_raytracing.metrics import StaticSphericalMetric


@dataclass(frozen=True)
class ManufacturedMetric:
    analytical: StaticSphericalMetric
    numerical: TabulatedMetric
    r_obs: float
    r_test: np.ndarray


METRIC_CASES = [
    pytest.param(SchwarzschildMetric, {"mass": 1.0}, id="schwarzschild"),
    pytest.param(ReissnerNordstromMetric, {"mass": 1.0, "charge": 0.5}, id="rn_q0p5"),
    pytest.param(ReissnerNordstromMetric, {"mass": 1.0, "charge": 0.9}, id="rn_q0p9"),
    pytest.param(
        ReissnerNordstromDeSitterMetric,
        {"mass": 1.0, "charge": 0.0, "cosmological_constant": 0.01},
        id="rnds_q0_lam0p01",
    ),
    pytest.param(
        ReissnerNordstromDeSitterMetric,
        {"mass": 1.0, "charge": 0.5, "cosmological_constant": 0.01},
        id="rnds_q0p5_lam0p01",
    ),
]

RAY_CASES = [
    pytest.param(SchwarzschildMetric, {"mass": 1.0}, id="schwarzschild"),
    pytest.param(ReissnerNordstromMetric, {"mass": 1.0, "charge": 0.5}, id="rn_q0p5"),
    pytest.param(
        ReissnerNordstromDeSitterMetric,
        {"mass": 1.0, "charge": 0.5, "cosmological_constant": 0.01},
        id="rnds_q0p5_lam0p01",
    ),
]

L4_GRID_CASES = [
    (800, 8.0e-7, 1.0e-7, 3.0e-5),
    (1600, 8.0e-8, 1.0e-8, 3.0e-6),
    (3200, 2.0e-8, 2.0e-9, 3.0e-7),
]


def _external_static_domain(metric: StaticSphericalMetric) -> tuple[float, float]:
    domains = metric.static_domains()
    if not domains:
        return metric.valid_radial_domain()
    return domains[-1]


def _grid_upper_bound(metric: StaticSphericalMetric, domain: tuple[float, float]) -> float:
    _r_lo, r_hi = domain
    if math.isfinite(r_hi):
        return r_hi
    return 200.0


def _observer_radius(domain: tuple[float, float], grid_hi: float) -> float:
    r_lo, r_hi = domain
    if math.isfinite(r_hi):
        return 0.5 * (r_lo + r_hi)
    return min(50.0, grid_hi * 0.5)


def _payload_metadata(
    metric: StaticSphericalMetric,
    r: np.ndarray,
    payload_style: str,
) -> LiDMBHMetadata:
    if payload_style == "standard-rab":
        return LiDMBHMetadata.from_params(
            params=None,
            source_format="standard-rab",
            original_keys=("r", "A", "B"),
            raw_B=None,
        )

    static_domains = [
        [lo, None if not math.isfinite(hi) else hi]
        for lo, hi in metric.static_domains()
    ]
    params = {
        "model": type(metric).__name__,
        "static_domains": static_domains,
        "horizons": metric.horizons(),
        "photon_spheres": metric.photon_spheres(),
        "g_convention": "g_rr_inverse",
        "B_convention": "g_rr",
    }
    return LiDMBHMetadata.from_params(
        params=params,
        source_format="li-dm-bh",
        original_keys=("r", "f", "g", "B", "params"),
        raw_B=1.0 / np.array([metric.A(float(value)) for value in r]),
    )


def manufacture_numerical_metric(
    analytical_metric: StaticSphericalMetric,
    *,
    n_points: int = 2200,
    payload_style: str = "metadata",
) -> ManufacturedMetric:
    """Sample an analytical metric into the public TabulatedMetric contract."""
    domain = _external_static_domain(analytical_metric)
    r_lo, r_hi = domain
    grid_hi = _grid_upper_bound(analytical_metric, domain)

    if payload_style == "metadata":
        r_start = r_lo + max(abs(r_lo), 1.0) * 1e-5
        r_stop = grid_hi - max(abs(grid_hi), 1.0) * 1e-5
        static_domain = domain
    else:
        r_start = r_lo + max(abs(r_lo), 1.0) * 2e-2
        r_stop = grid_hi - max(abs(grid_hi), 1.0) * 2e-2
        static_domain = (r_start, r_stop)

    if not r_start < r_stop:
        raise ValueError(f"manufactured grid collapsed: {r_start=} {r_stop=}")

    r = np.geomspace(r_start, r_stop, n_points)
    A = np.array([analytical_metric.A(float(value)) for value in r])
    C = A.copy()
    metadata = _payload_metadata(analytical_metric, r, payload_style)
    payload = CanonicalPayload(r=r, A=A, C=C, metadata=metadata)
    numerical = TabulatedMetric(payload=payload, static_domain=static_domain)

    r_obs = _observer_radius(static_domain, float(r[-1]))
    r_safe_lo = max(r_start * 1.08, r_lo + 0.12)
    r_safe_hi = min(r_stop * 0.92, r_obs * 0.95 if r_obs > r_safe_lo else r_stop * 0.92)
    if r_safe_hi <= r_safe_lo:
        r_safe_hi = r_stop * 0.85
    r_test = np.geomspace(r_safe_lo, r_safe_hi, 9)
    return ManufacturedMetric(analytical_metric, numerical, r_obs, r_test)


def _reachable_curves_in_domain(
    metric: StaticSphericalMetric,
    domain: tuple[float, float],
) -> list:
    r_lo, r_hi = domain
    return [
        curve
        for curve in metric.critical_curves()
        if curve.reachable_from_observer and r_lo < curve.r_ph < r_hi
    ]


@pytest.mark.parametrize("metric_cls,kwargs", METRIC_CASES)
@pytest.mark.parametrize("payload_style", ["metadata", "standard-rab"])
def test_l1_metric_components_match_analytical(metric_cls, kwargs, payload_style):
    analytical = metric_cls(**kwargs)
    manufactured = manufacture_numerical_metric(
        analytical,
        n_points=2600,
        payload_style=payload_style,
    )
    numerical = manufactured.numerical

    for r in manufactured.r_test:
        r = float(r)
        np.testing.assert_allclose(numerical.A(r), analytical.A(r), rtol=5e-5, atol=1e-8)
        np.testing.assert_allclose(1.0 / numerical.B(r), 1.0 / analytical.B(r), rtol=5e-5, atol=1e-8)
        np.testing.assert_allclose(numerical.dA_dr(r), analytical.dA_dr(r), rtol=5e-4, atol=1e-8)
        np.testing.assert_allclose(numerical.dB_dr(r), analytical.dB_dr(r), rtol=5e-3, atol=1e-7)

        b_local = 0.75 * r / math.sqrt(analytical.A(r))
        u = 1.0 / r
        np.testing.assert_allclose(numerical.G(u, b_local), analytical.G(u, b_local), rtol=5e-4, atol=1e-9)

    domain = _external_static_domain(analytical)
    analytical_curves = _reachable_curves_in_domain(analytical, domain)
    numerical_curves = numerical.critical_curves()
    assert len(numerical_curves) == len(analytical_curves)
    for expected, actual in zip(analytical_curves, numerical_curves):
        root_tol = 1e-8 if payload_style == "metadata" else 2e-3
        assert math.isclose(actual.r_ph, expected.r_ph, rel_tol=root_tol, abs_tol=root_tol)
        assert math.isclose(actual.b_crit, expected.b_crit, rel_tol=2e-4, abs_tol=1e-6)

    if payload_style == "metadata":
        np.testing.assert_allclose(numerical.horizons(), analytical.horizons(), rtol=0, atol=0)


def test_l1_standard_rab_infers_roots_without_fake_boundary_anchors():
    analytical = SchwarzschildMetric(mass=1.0)
    manufactured = manufacture_numerical_metric(
        analytical,
        n_points=1600,
        payload_style="standard-rab",
    )
    numerical = manufactured.numerical
    r_lo, r_hi = numerical.valid_radial_domain()

    assert numerical.horizons() == []
    assert all(not math.isclose(root, r_lo) for root in numerical.horizons())
    assert all(not math.isclose(root, r_hi) for root in numerical.horizons())
    assert math.isclose(numerical.photon_spheres()[0], 3.0, rel_tol=2e-3, abs_tol=2e-3)
    assert math.isclose(numerical.critical_curves()[0].b_crit, 3.0 * math.sqrt(3.0), rel_tol=2e-3)

    for r in (r_lo * 1.002, r_hi * 0.998):
        assert math.isclose(numerical.A(r), analytical.A(r), rel_tol=2e-3, abs_tol=2e-4)


@pytest.mark.parametrize("metric_cls,kwargs", RAY_CASES)
def test_l2_small_b_sweep_matches_analytical_rays(metric_cls, kwargs):
    analytical = metric_cls(**kwargs)
    manufactured = manufacture_numerical_metric(analytical, n_points=3600, payload_style="metadata")
    numerical = manufactured.numerical
    observer_a = FiniteStaticObserver(r_obs=manufactured.r_obs, metric=analytical)
    observer_n = FiniteStaticObserver(r_obs=manufactured.r_obs, metric=numerical)
    solver_a = QuadTransferSolver(metric=analytical, observer=observer_a)
    solver_n = QuadTransferSolver(metric=numerical, observer=observer_n)

    b_crit = _reachable_curves_in_domain(analytical, _external_static_domain(analytical))[-1].b_crit
    b_max = manufactured.r_obs / math.sqrt(analytical.A(manufactured.r_obs))
    b_values = [
        0.45 * b_crit,
        0.80 * b_crit,
        1.05 * b_crit,
        min(1.45 * b_crit, 0.75 * b_max),
    ]

    for b in b_values:
        result_a = solver_a.trace_b(float(b))
        result_n = solver_n.trace_b(float(b))

        assert not result_a.diagnostics.near_critical
        assert not result_n.diagnostics.near_critical
        assert result_n.diagnostics.termination_reason == result_a.diagnostics.termination_reason
        assert [segment.endpoint_event for segment in result_n.segments] == [
            segment.endpoint_event for segment in result_a.segments
        ]
        assert len(result_n.segments) == len(result_a.segments)

        last_a = result_a.segments[-1]
        last_n = result_n.segments[-1]
        np.testing.assert_allclose(last_n.phi_end, last_a.phi_end, rtol=2e-3, atol=2e-5)
        np.testing.assert_allclose(last_n.u_end, last_a.u_end, rtol=2e-3, atol=5e-5)

        for segment_a, segment_n in zip(result_a.segments, result_n.segments):
            span = segment_a.phi_end - segment_a.phi_start
            sample_phis = [
                segment_a.phi_start + 0.25 * span,
                segment_a.phi_start + 0.50 * span,
                segment_a.phi_start + 0.75 * span,
            ]
            for phi in sample_phis:
                if segment_n.contains_phi(phi):
                    np.testing.assert_allclose(
                        segment_n.u_at(phi),
                        segment_a.u_at(phi),
                        rtol=3e-3,
                        atol=5e-5,
                    )


def test_l3_thin_disk_intersection_and_observed_intensity_smoke():
    analytical = SchwarzschildMetric(mass=1.0)
    manufactured = manufacture_numerical_metric(analytical, n_points=4200, payload_style="metadata")
    numerical = manufactured.numerical
    observer_a = FiniteStaticObserver(r_obs=100.0, metric=analytical)
    observer_n = FiniteStaticObserver(r_obs=100.0, metric=numerical)
    ray_a = QuadTransferSolver(metric=analytical, observer=observer_a).trace_b(18.0)
    ray_n = QuadTransferSolver(metric=numerical, observer=observer_n).trace_b(18.0)
    disk = DiskWindow(r_min=6.0, r_max=90.0)

    intersections_a = compute_intersections(ray_a, disk, max_order=3)
    intersections_n = compute_intersections(ray_n, disk, max_order=3)
    assert intersections_a
    assert len(intersections_n) == len(intersections_a)

    for actual, expected in zip(intersections_n, intersections_a):
        assert actual.m == expected.m
        assert actual.region == expected.region
        assert actual.path_class == expected.path_class
        np.testing.assert_allclose(actual.r, expected.r, rtol=3e-3, atol=5e-3)

    source = ThinDiskSource(lambda r, _region: 1.0 / (r * r))
    observed_a = observed_intensity(intersections_a, source, analytical, observer_a)
    observed_n = observed_intensity(intersections_n, source, numerical, observer_n)

    assert observed_a.total > 0.0
    np.testing.assert_allclose(observed_n.total, observed_a.total, rtol=5e-3, atol=1e-10)
    np.testing.assert_allclose(observed_n.redshift_weights, observed_a.redshift_weights, rtol=5e-3, atol=1e-10)
    np.testing.assert_allclose(observed_n.contributions, observed_a.contributions, rtol=5e-3, atol=1e-10)


def _schwarzschild_l4_reference_cases():
    analytical = SchwarzschildMetric(mass=1.0)
    manufactured = manufacture_numerical_metric(
        analytical,
        n_points=6400,
        payload_style="metadata",
    )
    observer = FiniteStaticObserver(r_obs=manufactured.r_obs, metric=analytical)
    solver = QuadTransferSolver(metric=analytical, observer=observer)
    b_crit = _reachable_curves_in_domain(analytical, _external_static_domain(analytical))[-1].b_crit
    b_max = manufactured.r_obs / math.sqrt(analytical.A(manufactured.r_obs))
    b_values = [
        pytest.param(0.70 * b_crit, id="captured_0p70_bcrit"),
        pytest.param(1.10 * b_crit, id="turning_1p10_bcrit"),
        pytest.param(min(1.35 * b_crit, 0.70 * b_max), id="wide_turning_stable"),
    ]
    references = [(float(param.values[0]), solver.trace_b(float(param.values[0]))) for param in b_values]
    return analytical, manufactured.r_obs, references


def _assert_l4_result_matches_reference(
    *,
    density: int,
    b: float,
    actual,
    expected,
    phi_abs: float,
    u_probe_abs: float,
    intersection_r_abs: float,
) -> None:
    assert actual.diagnostics.termination_reason == expected.diagnostics.termination_reason, (
        f"L4 event classification mismatch at n={density}, b={b}: "
        f"{actual.diagnostics.termination_reason} != {expected.diagnostics.termination_reason}"
    )
    assert len(actual.segments) == len(expected.segments), (
        f"L4 event classification mismatch at n={density}, b={b}: segment count differs"
    )
    assert [segment.endpoint_event for segment in actual.segments] == [
        segment.endpoint_event for segment in expected.segments
    ], f"L4 event classification mismatch at n={density}, b={b}: endpoint events differ"

    actual_last = actual.segments[-1]
    expected_last = expected.segments[-1]
    np.testing.assert_allclose(
        actual_last.phi_end,
        expected_last.phi_end,
        rtol=0.0,
        atol=phi_abs,
        err_msg=f"L4 convergence phi_end drift at n={density}, b={b}",
    )
    np.testing.assert_allclose(
        actual_last.u_end,
        expected_last.u_end,
        rtol=0.0,
        atol=u_probe_abs,
        err_msg=f"L4 convergence u_end drift at n={density}, b={b}",
    )

    for segment_index, (actual_segment, expected_segment) in enumerate(zip(actual.segments, expected.segments)):
        span = expected_segment.phi_end - expected_segment.phi_start
        for fraction in (0.33, 0.67):
            phi = expected_segment.phi_start + fraction * span
            assert actual_segment.contains_phi(phi), (
                f"L4 fixture construction mismatch at n={density}, b={b}: "
                f"probe phi={phi} missing from segment {segment_index}"
            )
            np.testing.assert_allclose(
                actual_segment.u_at(phi),
                expected_segment.u_at(phi),
                rtol=0.0,
                atol=u_probe_abs,
                err_msg=f"L4 convergence u_at drift at n={density}, b={b}, segment={segment_index}",
            )

    disk = DiskWindow(r_min=6.0, r_max=45.0)
    actual_intersections = compute_intersections(actual, disk, max_order=3)
    expected_intersections = compute_intersections(expected, disk, max_order=3)
    assert len(actual_intersections) == len(expected_intersections), (
        f"L4 fixture construction mismatch at n={density}, b={b}: disk crossing count differs"
    )
    for actual_intersection, expected_intersection in zip(actual_intersections, expected_intersections):
        assert actual_intersection.m == expected_intersection.m
        assert actual_intersection.region == expected_intersection.region
        assert actual_intersection.path_class == expected_intersection.path_class
        np.testing.assert_allclose(
            actual_intersection.r,
            expected_intersection.r,
            rtol=0.0,
            atol=intersection_r_abs,
            err_msg=f"L4 convergence disk radius drift at n={density}, b={b}",
        )


def test_l4_schwarzschild_grid_convergence_smoke():
    analytical, r_obs, reference_cases = _schwarzschild_l4_reference_cases()

    for density, phi_abs, u_probe_abs, intersection_r_abs in L4_GRID_CASES:
        manufactured = manufacture_numerical_metric(
            analytical,
            n_points=density,
            payload_style="metadata",
        )
        assert manufactured.r_obs == r_obs, f"L4 fixture construction mismatch at n={density}: observer moved"
        observer = FiniteStaticObserver(r_obs=r_obs, metric=manufactured.numerical)
        solver = QuadTransferSolver(metric=manufactured.numerical, observer=observer)

        for b, expected in reference_cases:
            actual = solver.trace_b(b)
            _assert_l4_result_matches_reference(
                density=density,
                b=b,
                actual=actual,
                expected=expected,
                phi_abs=phi_abs,
                u_probe_abs=u_probe_abs,
                intersection_r_abs=intersection_r_abs,
            )


@pytest.mark.slow
def test_l4_schwarzschild_grid_convergence_slow_scaffold():
    analytical, r_obs, reference_cases = _schwarzschild_l4_reference_cases()

    for density in (800, 1600, 3200, 6400):
        manufactured = manufacture_numerical_metric(
            analytical,
            n_points=density,
            payload_style="metadata",
        )
        observer = FiniteStaticObserver(r_obs=r_obs, metric=manufactured.numerical)
        solver = QuadTransferSolver(metric=manufactured.numerical, observer=observer)

        for b, expected in reference_cases:
            actual = solver.trace_b(b)
            _assert_l4_result_matches_reference(
                density=density,
                b=b,
                actual=actual,
                expected=expected,
                phi_abs=8.0e-7,
                u_probe_abs=1.0e-7,
                intersection_r_abs=3.0e-5,
            )
