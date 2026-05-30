import pytest

from spherical_raytracing.metrics import (
    LQGMetric,
    ReissnerNordstromDeSitterMetric,
    ReissnerNordstromMetric,
    SchwarzschildMetric,
)


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
