import math

import pytest

from spherical_raytracing.metrics import (
    LQGMetric,
    ReissnerNordstromDeSitterMetric,
    ReissnerNordstromMetric,
    SchwarzschildMetric,
)
from spherical_raytracing.observers import InfinityObserver
from spherical_raytracing.solvers import QuadTransferSolver


def test_schwarzschild_geometry_and_potential():
    metric = SchwarzschildMetric(mass=2.0)
    r = 20.0
    u = 1.0 / r
    b = 12.0

    assert math.isclose(metric.A(r), 0.8)
    assert math.isclose(metric.B(r), 1.25)
    assert math.isclose(metric.dA_dr(r), 0.01)
    assert math.isclose(metric.dB_dr(r), -0.015625)
    assert math.isclose(metric.G(u, b), 1.0 / b**2 - u**2 + 2.0 * metric.mass * u**3)


def test_schwarzschild_horizon_photon_sphere_and_critical_curve():
    metric = SchwarzschildMetric(mass=2.0)
    curve = metric.critical_curves()[0]

    assert metric.horizons() == [4.0]
    assert metric.photon_spheres() == [6.0]
    assert math.isclose(curve.r_ph, 6.0)
    assert math.isclose(curve.b_crit, 6.0 * math.sqrt(3.0))
    assert curve.stability == "unstable"
    assert curve.region == "external"
    assert curve.reachable_from_observer is True


def test_lqg_reference_potential_for_f_inverse_f_metric():
    metric = LQGMetric(mass=2.0, alpha=0.25)
    u = 0.07
    b = 11.0

    expected = -metric.alpha * metric.mass**2 * u**6 + 2.0 * metric.mass * u**3 - u**2 + 1.0 / b**2
    assert math.isclose(metric.G(u, b), expected)


def test_lqg_horizons_and_exterior_photon_sphere_are_found_from_roots():
    metric = LQGMetric(mass=1.0, alpha=0.25)

    horizons = metric.horizons()
    photon_spheres = metric.photon_spheres()
    curve = metric.critical_curves()[0]

    assert len(horizons) == 2
    assert math.isclose(metric.A(horizons[-1]), 0.0, abs_tol=1e-10)
    assert 1.9 < horizons[-1] < 2.0
    assert len(photon_spheres) == 1
    assert 2.9 < photon_spheres[0] < 3.0
    assert math.isclose(
        metric.dA_dr(photon_spheres[0]) / photon_spheres[0] ** 2
        - 2.0 * metric.A(photon_spheres[0]) / photon_spheres[0] ** 3,
        0.0,
        abs_tol=1e-10,
    )
    assert curve.region == "external"
    assert math.isfinite(curve.b_crit)


def test_metric_rejects_invalid_radius():
    metric = SchwarzschildMetric(mass=1.0)

    with pytest.raises(ValueError, match="r must be positive"):
        metric.A(0.0)


def test_rn_geometry_horizons_photon_spheres_and_g_consistency():
    metric = ReissnerNordstromMetric(mass=1.0, charge=0.5, region="outer")
    r = 10.0
    u = 1.0 / r
    b = 8.0

    assert math.isclose(metric.A(r), 1.0 - 2.0 / r + 0.25 / r**2)
    assert math.isclose(metric.B(r), 1.0 / metric.A(r))
    assert math.isclose(metric.dA_dr(r), 2.0 / r**2 - 2.0 * 0.25 / r**3)
    assert metric.horizons() == pytest.approx([1.0 - math.sqrt(0.75), 1.0 + math.sqrt(0.75)])
    assert metric.valid_radial_domain() == pytest.approx((1.0 + math.sqrt(0.75), math.inf))

    photon_spheres = metric.photon_spheres()
    expected_outer = (3.0 + math.sqrt(9.0 - 8.0 * 0.25)) / 2.0
    assert photon_spheres[-1] == pytest.approx(expected_outer)
    assert all(metric.A(r_ph) > 0.0 for r_ph in photon_spheres)
    expected_g = 1.0 / (b**2 * metric.A(r) * metric.B(r)) - u**2 / metric.B(r)
    assert math.isclose(metric.G(u, b), expected_g)


def test_rn_g_supports_infinity_observer_start():
    metric = ReissnerNordstromMetric(mass=1.0, charge=0.5)

    assert metric.G(0.0, 8.0) == pytest.approx(1.0 / 8.0**2)
    result = QuadTransferSolver(metric=metric, observer=InfinityObserver()).trace_b(8.0)

    assert result.segments


def test_rn_warns_about_overcharged_static_domain_without_horizon():
    metric = ReissnerNordstromMetric(mass=1.0, charge=1.2)

    assert metric.horizons() == []
    assert metric.valid_radial_domain() == (0.0, math.inf)
    assert metric.static_domains() == [(0.0, math.inf)]


def test_extremal_rn_horizon_is_not_duplicated():
    metric = ReissnerNordstromMetric(mass=1.0, charge=1.0)

    assert metric.horizons() == pytest.approx([1.0])
    assert metric.static_domains() == [(1.0, math.inf)]


def test_small_rn_horizon_is_not_dropped_by_root_deduplication():
    metric = ReissnerNordstromMetric(mass=1e-9, charge=0.0)

    assert metric.horizons() == pytest.approx([2e-9])
    assert metric.photon_spheres() == pytest.approx([3e-9])


def test_small_charge_rn_inner_horizon_is_not_lost_to_cancellation():
    metric = ReissnerNordstromMetric(mass=1.0, charge=1e-9)

    horizons = metric.horizons()

    assert len(horizons) == 2
    assert horizons[0] == pytest.approx(5e-19, rel=1e-12)
    assert horizons[1] == pytest.approx(2.0)


def test_degenerate_rn_photon_sphere_is_not_duplicated():
    metric = ReissnerNordstromMetric(mass=1.0, charge=3.0 / math.sqrt(8.0))

    assert metric.photon_spheres() == pytest.approx([1.5])
    curves = metric.critical_curves()
    assert len(curves) == 1
    assert curves[0].stability == "unstable"


def test_rn_near_zero_negative_photon_discriminant_is_clamped():
    charge = math.sqrt((9.0 + 1e-13) / 8.0)
    metric = ReissnerNordstromMetric(mass=1.0, charge=charge)

    assert metric.photon_spheres() == pytest.approx([1.5])


def test_rn_charge_is_signed_but_geometry_depends_on_charge_squared():
    positive = ReissnerNordstromMetric(mass=1.0, charge=0.5)
    negative = ReissnerNordstromMetric(mass=1.0, charge=-0.5)

    assert negative.charge == -0.5
    assert negative.charge_squared == pytest.approx(positive.charge_squared)
    assert negative.A(10.0) == pytest.approx(positive.A(10.0))


def test_rnds_zero_lambda_matches_rn_boundary_geometry_and_domains():
    rn = ReissnerNordstromMetric(mass=1.0, charge=0.5, region="outer")
    rnds = ReissnerNordstromDeSitterMetric(
        mass=1.0,
        charge=0.5,
        cosmological_constant=0.0,
        region="outer",
    )

    for r in [0.5, 2.0, 10.0]:
        assert rnds.A(r) == pytest.approx(rn.A(r))
        assert rnds.B(r) == pytest.approx(rn.B(r))
        assert rnds.dA_dr(r) == pytest.approx(rn.dA_dr(r))
    assert rnds.horizons() == pytest.approx(rn.horizons())
    assert rnds.valid_radial_domain() == pytest.approx(rn.valid_radial_domain())
    assert rnds.static_domains() == pytest.approx(rn.static_domains())


def test_rnds_black_hole_static_patch_is_between_event_and_cosmological_horizon():
    metric = ReissnerNordstromDeSitterMetric(mass=1.0, charge=0.5, cosmological_constant=0.01)

    horizons = metric.horizons()
    assert len(horizons) == 3
    assert metric.A(horizons[0]) == pytest.approx(0.0, abs=1e-9)
    assert metric.A(horizons[1]) == pytest.approx(0.0, abs=1e-9)
    assert metric.A(horizons[2]) == pytest.approx(0.0, abs=1e-9)
    assert metric.A(0.5 * (horizons[1] + horizons[2])) > 0.0
    assert metric.valid_radial_domain() == pytest.approx((horizons[1], horizons[2]))
    assert metric.static_domains()[-1] == pytest.approx((horizons[1], horizons[2]))
    assert any(domain == pytest.approx(metric.valid_radial_domain()) for domain in metric.static_domains())


def test_rnds_g_supports_infinity_observer_start():
    metric = ReissnerNordstromDeSitterMetric(mass=1.0, charge=0.5, cosmological_constant=0.01)

    assert metric.G(0.0, 8.0) == pytest.approx(1.0 / 8.0**2 + 0.01 / 3.0)
    result = QuadTransferSolver(metric=metric, observer=InfinityObserver()).trace_b(8.0)

    assert result.segments


def test_sds_limit_static_patch_with_two_positive_horizons():
    metric = ReissnerNordstromDeSitterMetric(mass=1.0, charge=0.0, cosmological_constant=0.01)

    horizons = metric.horizons()
    assert len(horizons) == 2
    assert metric.valid_radial_domain() == pytest.approx((horizons[0], horizons[1]))


def test_degenerate_rnds_valid_radial_domain_does_not_return_non_static_interval():
    metric = ReissnerNordstromDeSitterMetric(mass=1.0, charge=0.0, cosmological_constant=1.0 / 9.0)

    assert metric.static_domains() == []
    assert metric.valid_radial_domain() == (0.0, 0.0)


def test_rnds_keeps_tiny_positive_cauchy_horizon_root():
    metric = ReissnerNordstromDeSitterMetric(
        mass=1e-9,
        charge=0.5e-9,
        cosmological_constant=1e-6,
    )

    horizons = metric.horizons()

    assert len(horizons) == 3
    assert horizons[0] == pytest.approx((1.0 - math.sqrt(0.75)) * 1e-9, rel=1e-5)


def test_rnds_rejects_non_finite_cosmological_constant():
    with pytest.raises(ValueError, match="cosmological_constant must be finite"):
        ReissnerNordstromDeSitterMetric(cosmological_constant=math.inf)
