from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from scipy.interpolate import PchipInterpolator


@dataclass(frozen=True)
class SamplingOptions:
    initial_count: int = 64
    max_refine: int = 4
    absolute_gradient_threshold: float = 1.0
    relative_jump_threshold: float = 0.25


@dataclass(frozen=True)
class RadialProfile:
    coordinates: np.ndarray
    intensities: np.ndarray
    diagnostics: dict[str, float | int | str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.coordinates.ndim != 1 or self.intensities.ndim != 1:
            raise ValueError("coordinates and intensities must be one-dimensional")
        if len(self.coordinates) != len(self.intensities):
            raise ValueError("coordinates and intensities must have the same length")
        if len(self.coordinates) < 2:
            raise ValueError("at least two radial samples are required")

    def evaluate(self, radius: np.ndarray) -> np.ndarray:
        interpolator = PchipInterpolator(self.coordinates, self.intensities, extrapolate=False)
        values = interpolator(radius)
        return np.where(np.isfinite(values), values, 0.0)


def _relative_jump(left: float, right: float) -> float:
    scale = max(abs(left), abs(right), 1e-300)
    return abs(right - left) / scale


@dataclass(frozen=True)
class _Sample:
    intensity: float
    termination_reason: str | None
    path_classes: tuple[str, ...]
    unresolved: bool
    shell_crossing_count: int | None = None
    backend_disagreement: bool = False


def _coerce_sample(value: Any) -> _Sample:
    if hasattr(value, "total"):
        diagnostics = getattr(value, "diagnostics", {})
        path_classes = tuple(str(item) for item in getattr(value, "path_classes", ()))
        unresolved = bool(
            diagnostics.get("near_critical", False)
            or diagnostics.get("unresolved", False)
            or diagnostics.get("unresolved_intervals", 0)
        )
        return _Sample(
            intensity=float(value.total),
            termination_reason=diagnostics.get("termination_reason"),
            path_classes=path_classes,
            unresolved=unresolved,
            shell_crossing_count=int(diagnostics["shell_crossing_count"])
            if "shell_crossing_count" in diagnostics
            else None,
            backend_disagreement=bool(diagnostics.get("backend_disagreement", False)),
        )
    return _Sample(
        intensity=float(value),
        termination_reason=None,
        path_classes=(),
        unresolved=False,
        shell_crossing_count=None,
        backend_disagreement=False,
    )


def sample_radial_profile(
    coordinate_min: float,
    coordinate_max: float,
    intensity_fn,
    critical_points: list[float],
    options: SamplingOptions = SamplingOptions(),
) -> RadialProfile:
    if coordinate_max <= coordinate_min:
        raise ValueError("coordinate_max must be greater than coordinate_min")
    if options.initial_count < 2:
        raise ValueError("initial_count must be at least 2")

    coordinates = list(np.linspace(coordinate_min, coordinate_max, options.initial_count))
    for point in critical_points:
        if coordinate_min <= point <= coordinate_max:
            coordinates.append(float(point))
    coordinates = sorted(set(coordinates))

    refine_rounds = 0
    unresolved_intervals = 0
    status_change_intervals = 0
    path_class_change_intervals = 0
    shell_crossing_change_intervals = 0
    backend_disagreement_intervals = 0
    sample_cache: dict[float, _Sample] = {}

    def sample_at(x: float) -> _Sample:
        key = float(x)
        if key not in sample_cache:
            sample_cache[key] = _coerce_sample(intensity_fn(key))
        return sample_cache[key]

    for _ in range(options.max_refine):
        samples = [sample_at(x) for x in coordinates]
        new_points: list[float] = []
        round_unresolved_intervals = 0
        round_status_changes = 0
        round_path_changes = 0
        round_shell_crossing_changes = 0
        round_backend_disagreements = 0
        for left_x, right_x, left_sample, right_sample in zip(
            coordinates[:-1],
            coordinates[1:],
            samples[:-1],
            samples[1:],
        ):
            width = right_x - left_x
            gradient = abs(right_sample.intensity - left_sample.intensity) / width
            jump = _relative_jump(left_sample.intensity, right_sample.intensity)
            status_changed = left_sample.termination_reason != right_sample.termination_reason
            path_changed = left_sample.path_classes != right_sample.path_classes
            unresolved = left_sample.unresolved or right_sample.unresolved
            shell_crossing_changed = left_sample.shell_crossing_count != right_sample.shell_crossing_count
            backend_disagreement = left_sample.backend_disagreement or right_sample.backend_disagreement
            if status_changed:
                round_status_changes += 1
            if path_changed:
                round_path_changes += 1
            if unresolved:
                round_unresolved_intervals += 1
            if shell_crossing_changed:
                round_shell_crossing_changes += 1
            if backend_disagreement:
                round_backend_disagreements += 1
            if (
                gradient > options.absolute_gradient_threshold
                or jump > options.relative_jump_threshold
                or status_changed
                or path_changed
                or unresolved
                or shell_crossing_changed
                or backend_disagreement
            ):
                new_points.append(0.5 * (left_x + right_x))
        if not new_points:
            break
        coordinates = sorted(set(coordinates + new_points))
        refine_rounds += 1
        unresolved_intervals = round_unresolved_intervals
        status_change_intervals += round_status_changes
        path_class_change_intervals += round_path_changes
        shell_crossing_change_intervals += round_shell_crossing_changes
        backend_disagreement_intervals += round_backend_disagreements

    final_samples = [sample_at(x) for x in coordinates]
    intensities_array = np.array([sample.intensity for sample in final_samples])
    coordinates_array = np.array(coordinates)
    return RadialProfile(
        coordinates=coordinates_array,
        intensities=intensities_array,
        diagnostics={
            "sample_count": len(coordinates_array),
            "refine_rounds": refine_rounds,
            "unresolved_intervals": unresolved_intervals if refine_rounds == options.max_refine else 0,
            "status_change_intervals": status_change_intervals,
            "path_class_change_intervals": path_class_change_intervals,
            "shell_crossing_change_intervals": shell_crossing_change_intervals,
            "backend_disagreement_intervals": backend_disagreement_intervals,
            "unresolved_sample_count": sum(1 for sample in final_samples if sample.unresolved),
        },
    )


@dataclass(frozen=True)
class ImageGrid:
    width: int
    height: int
    radius: float


@dataclass(frozen=True)
class RenderedImage:
    x: np.ndarray
    y: np.ndarray
    pixels: np.ndarray
    diagnostics: dict[str, float | int | str] = field(default_factory=dict)


def render_axisymmetric_image(profile: RadialProfile, grid: ImageGrid) -> RenderedImage:
    if grid.width <= 0 or grid.height <= 0:
        raise ValueError("image dimensions must be positive")
    x = np.linspace(-grid.radius, grid.radius, grid.width)
    y = np.linspace(-grid.radius, grid.radius, grid.height)
    xx, yy = np.meshgrid(x, y)
    rr = np.sqrt(xx**2 + yy**2)
    pixels = profile.evaluate(rr)
    return RenderedImage(x=x, y=y, pixels=pixels, diagnostics={"width": grid.width, "height": grid.height})
