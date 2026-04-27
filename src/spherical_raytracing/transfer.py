from __future__ import annotations

import math
from dataclasses import dataclass, field

from .diagnostics import RayResult, RaySegment


@dataclass(frozen=True)
class DiskWindow:
    r_min: float
    r_max: float
    enabled_regions: frozenset[str] = frozenset({"external"})

    def contains(self, r: float, region: str) -> bool:
        return self.r_min <= r <= self.r_max and region in self.enabled_regions


@dataclass(frozen=True)
class IntersectionRecord:
    m: int
    r: float
    phi: float
    region: str
    ray_segment: RaySegment
    path_class: str
    diagnostics: dict[str, float | str] = field(default_factory=dict)


def path_class_for_order(m: int) -> str:
    if m == 1:
        return "direct"
    if m == 2:
        return "lensing_ring"
    return "photon_ring"


def disk_crossing_angles(max_order: int) -> list[float]:
    if max_order < 1:
        return []
    return [math.pi / 2.0 + n * math.pi for n in range(max_order)]


def compute_intersections(ray: RayResult, disk_window: DiskWindow, max_order: int) -> list[IntersectionRecord]:
    records: list[IntersectionRecord] = []
    for m, phi in enumerate(disk_crossing_angles(max_order), start=1):
        for segment in ray.segments:
            if not segment.contains_phi(phi):
                continue
            r = segment.r_at(phi)
            if not math.isfinite(r):
                continue
            if not disk_window.contains(r, segment.region):
                continue
            records.append(
                IntersectionRecord(
                    m=m,
                    r=r,
                    phi=phi,
                    region=segment.region,
                    ray_segment=segment,
                    path_class=path_class_for_order(m),
                    diagnostics={"termination_reason": ray.diagnostics.termination_reason},
                )
            )
            break
    return records
