import math

from spherical_raytracing.diagnostics import EventType, RaySegment
from spherical_raytracing.metrics import SchwarzschildMetric
from spherical_raytracing.observers import FiniteStaticObserver
from spherical_raytracing.sources import ThinDiskSource, observed_intensity
from spherical_raytracing.transfer import IntersectionRecord


def constant_profile(r: float, region: str | None = None) -> float:
    return 2.0


def test_source_uses_precomputed_junction_redshift_weight():
    metric = SchwarzschildMetric(mass=1.0, region="outer")
    observer = FiniteStaticObserver(r_obs=50.0, metric=metric)
    source = ThinDiskSource(intensity_profile=constant_profile)
    segment = RaySegment(
        region="inner",
        radial_direction="inward",
        phi_start=0.0,
        phi_end=1.0,
        u_start=0.2,
        u_end=0.25,
        endpoint_event=EventType.MAX_PHI,
        evaluator=lambda phi: 0.25,
    )
    intersection = IntersectionRecord(
        m=1,
        r=4.0,
        phi=0.5,
        region="inner",
        ray_segment=segment,
        path_class="direct",
        diagnostics={"redshift_weight": 0.25, "g": math.sqrt(0.5)},
    )

    observed = observed_intensity(
        [intersection],
        source,
        metric,
        observer,
        region_metrics={"inner": metric, "outer": metric},
    )

    assert observed.redshift_weights == [0.25]
    assert observed.contributions == [0.5]
    assert observed.total == 0.5
