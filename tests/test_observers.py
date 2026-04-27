import math

import pytest

from spherical_raytracing.metrics import SchwarzschildMetric
from spherical_raytracing.observers import FiniteStaticObserver, InfinityObserver


def test_infinity_observer_maps_screen_radius_to_impact_parameter():
    observer = InfinityObserver()

    assert observer.coordinate_name == "b"
    assert math.isclose(observer.impact_parameter(8.5), 8.5)
    assert math.isclose(observer.screen_radius(8.5), 8.5)


def test_finite_static_observer_maps_angle_to_impact_parameter():
    metric = SchwarzschildMetric(mass=1.0)
    observer = FiniteStaticObserver(r_obs=20.0, metric=metric)
    alpha = 0.2

    expected = observer.r_obs * math.sin(alpha) / math.sqrt(metric.A(observer.r_obs))
    assert math.isclose(observer.impact_parameter(alpha), expected)
    assert math.isclose(observer.screen_radius(alpha), observer.r_obs * math.tan(alpha))


def test_finite_static_observer_rejects_non_static_radius():
    metric = SchwarzschildMetric(mass=1.0)

    with pytest.raises(ValueError, match="static observer requires A"):
        FiniteStaticObserver(r_obs=1.5, metric=metric)
