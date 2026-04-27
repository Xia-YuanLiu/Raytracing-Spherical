import math

import pytest

from spherical_raytracing.metrics import LQGMetric, SchwarzschildMetric


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
