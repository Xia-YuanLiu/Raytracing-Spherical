from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

from .metrics import StaticSphericalMetric
from .observers import FiniteStaticObserver, InfinityObserver
from .transfer import IntersectionRecord


IntensityProfile = Callable[[float, str | None], float]


@dataclass(frozen=True)
class ThinDiskSource:
    intensity_profile: IntensityProfile

    def intensity(self, r: float, region: str | None = None) -> float:
        return float(self.intensity_profile(r, region))

    def redshift_weight(
        self,
        intersection: IntersectionRecord,
        metric: StaticSphericalMetric,
        observer: InfinityObserver | FiniteStaticObserver,
        region_metrics: dict[str, StaticSphericalMetric] | None = None,
    ) -> float:
        metric = _metric_for_region(intersection.region, metric, region_metrics)
        if isinstance(observer, InfinityObserver):
            g = metric.A(intersection.r) ** 0.5
        else:
            g = (metric.A(intersection.r) / metric.A(observer.r_obs)) ** 0.5
        return g**4

    def contribution(
        self,
        intersection: IntersectionRecord,
        metric: StaticSphericalMetric,
        observer: InfinityObserver | FiniteStaticObserver,
        region_metrics: dict[str, StaticSphericalMetric] | None = None,
    ) -> float:
        return self.redshift_weight(intersection, metric, observer, region_metrics) * self.intensity(
            intersection.r,
            intersection.region,
        )


@dataclass(frozen=True)
class ObservedIntensity:
    total: float
    contributions: list[float]
    radii: list[float]
    redshift_weights: list[float]
    regions: list[str]
    path_classes: list[str]
    diagnostics: dict[str, float | int | str] = field(default_factory=dict)


def observed_intensity(
    intersections: list[IntersectionRecord],
    source: ThinDiskSource,
    metric: StaticSphericalMetric,
    observer: InfinityObserver | FiniteStaticObserver,
    region_metrics: dict[str, StaticSphericalMetric] | None = None,
) -> ObservedIntensity:
    contributions: list[float] = []
    radii: list[float] = []
    weights: list[float] = []
    regions: list[str] = []
    path_classes: list[str] = []
    invalid_regions: list[str] = []
    for intersection in intersections:
        try:
            weight = source.redshift_weight(intersection, metric, observer, region_metrics)
        except ValueError:
            invalid_regions.append(intersection.region)
            continue
        contribution = weight * source.intensity(intersection.r, intersection.region)
        contributions.append(contribution)
        radii.append(intersection.r)
        weights.append(weight)
        regions.append(intersection.region)
        path_classes.append(intersection.path_class)
    return ObservedIntensity(
        total=sum(contributions),
        contributions=contributions,
        radii=radii,
        redshift_weights=weights,
        regions=regions,
        path_classes=path_classes,
        diagnostics={
            "intersection_count": len(intersections),
            "invalid_intersection_count": len(invalid_regions),
            "invalid_regions": ",".join(sorted(set(invalid_regions))),
        },
    )


def _metric_for_region(
    region: str,
    metric: StaticSphericalMetric,
    region_metrics: dict[str, StaticSphericalMetric] | None,
) -> StaticSphericalMetric:
    if region == metric.region:
        return metric
    if region_metrics is not None and region in region_metrics:
        return region_metrics[region]
    raise ValueError(f"no metric is defined for region {region!r}")
