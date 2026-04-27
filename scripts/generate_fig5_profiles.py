from __future__ import annotations

import argparse
import csv
import math
import os
import sys
import tempfile
import warnings
from collections.abc import Callable
from pathlib import Path

import numpy as np
from scipy.integrate import IntegrationWarning

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from spherical_raytracing import (
    DiskWindow,
    ImageGrid,
    InfinityObserver,
    QuadTransferSolver,
    RadialProfile,
    SchwarzschildMetric,
    SolverOptions,
    ThinDiskSource,
    compute_intersections,
    observed_intensity,
    render_axisymmetric_image,
)


ProfileFn = Callable[[float, str | None], float]


def fig5_top_emission(r: float, region: str | None = None) -> float:
    if r < 6.0:
        return 0.0
    return 1.0 / (r - 5.0) ** 2


def fig5_middle_emission(r: float, region: str | None = None) -> float:
    if r < 3.0:
        return 0.0
    return (2.0 - math.tanh(r - 3.0)) / (2.0 * math.exp(r - 3.0))


def fig5_bottom_emission(r: float, region: str | None = None) -> float:
    if r <= 2.0:
        return 0.0
    return (math.pi / 2.0 - math.atan(r - 5.0)) / (math.pi / 2.0 + math.atan(3.0))


PROFILES: dict[str, ProfileFn] = {
    "top": fig5_top_emission,
    "middle": fig5_middle_emission,
    "bottom": fig5_bottom_emission,
}


def _b_grid(samples: int, b_max: float, b_crit: float) -> np.ndarray:
    base = np.linspace(0.0, b_max, samples)
    lensing = np.linspace(5.02, 6.17, max(samples, 64))
    photon_offsets = np.geomspace(1e-5, 4e-2, max(samples // 2, 32))
    photon = np.concatenate([b_crit - photon_offsets, b_crit + photon_offsets])
    values = np.concatenate([base, lensing, photon])
    values = values[(values >= 0.0) & (values <= b_max)]
    return np.unique(np.round(values, 12))


def _contributions_by_class(intersections, observed) -> dict[str, float]:
    contributions = {"direct": 0.0, "lensing_ring": 0.0, "photon_ring": 0.0}
    for intersection, contribution in zip(intersections, observed.contributions):
        contributions[intersection.path_class] += contribution
    return contributions


def generate_rows(
    profile_name: str,
    samples: int,
    b_max: float,
    max_order: int,
    show_integration_warnings: bool,
) -> list[dict[str, object]]:
    metric = SchwarzschildMetric(mass=1.0)
    observer = InfinityObserver()
    solver = QuadTransferSolver(
        metric=metric,
        observer=observer,
        options=SolverOptions(critical_exclusion=0.0, max_phi=8.0 * math.pi),
    )
    source = ThinDiskSource(PROFILES[profile_name])
    disk = DiskWindow(r_min=2.0 + 1e-6, r_max=100.0)
    rows: list[dict[str, object]] = []

    for b in _b_grid(samples, b_max, metric.critical_curves()[0].b_crit):
        if b <= 0.0:
            rows.append(
                {
                    "profile": profile_name,
                    "b_over_M": 0.0,
                    "Iobs_over_I0": 0.0,
                    "direct": 0.0,
                    "lensing_ring": 0.0,
                    "photon_ring": 0.0,
                    "intersection_count": 0,
                    "termination_reason": "axis",
                }
            )
            continue
        with warnings.catch_warnings():
            if not show_integration_warnings:
                warnings.simplefilter("ignore", IntegrationWarning)
            ray = solver.trace_b(float(b))
            intersections = compute_intersections(ray, disk, max_order=max_order)
            observed = observed_intensity(intersections, source, metric, observer)
        contributions = _contributions_by_class(intersections, observed)
        rows.append(
            {
                "profile": profile_name,
                "b_over_M": float(b),
                "Iobs_over_I0": observed.total,
                "direct": contributions["direct"],
                "lensing_ring": contributions["lensing_ring"],
                "photon_ring": contributions["photon_ring"],
                "intersection_count": len(intersections),
                "termination_reason": ray.diagnostics.termination_reason,
            }
        )
    return rows


def write_csv(rows: list[dict[str, object]], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "profile",
        "b_over_M",
        "Iobs_over_I0",
        "direct",
        "lensing_ring",
        "photon_ring",
        "intersection_count",
        "termination_reason",
    ]
    with output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _configure_matplotlib():
    os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "spherical_raytracing_matplotlib"))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def _profile_rows(rows: list[dict[str, object]], profile_name: str) -> list[dict[str, object]]:
    profile_rows = [row for row in rows if row["profile"] == profile_name]
    profile_rows.sort(key=lambda row: float(row["b_over_M"]))
    return profile_rows


def _ordered_profile_names(rows: list[dict[str, object]]) -> list[str]:
    available = {str(row["profile"]) for row in rows}
    return [profile_name for profile_name in PROFILES if profile_name in available]


def write_plot(rows: list[dict[str, object]], output: Path) -> None:
    plt = _configure_matplotlib()

    output.parent.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(8.0, 4.8))
    profile_names = _ordered_profile_names(rows)
    for profile_name in profile_names:
        profile_rows = _profile_rows(rows, profile_name)
        axis.plot(
            [float(row["b_over_M"]) for row in profile_rows],
            [float(row["Iobs_over_I0"]) for row in profile_rows],
            label=profile_name,
            linewidth=1.4,
        )
    axis.set_xlabel("b/M")
    axis.set_ylabel("I_obs/I0")
    axis.set_xlim(left=0.0)
    axis.set_ylim(bottom=0.0)
    axis.legend()
    axis.grid(alpha=0.25)
    figure.tight_layout()
    figure.savefig(output, dpi=180)
    plt.close(figure)


def _radial_profile_from_rows(profile_rows: list[dict[str, object]]) -> RadialProfile:
    return RadialProfile(
        coordinates=np.array([float(row["b_over_M"]) for row in profile_rows]),
        intensities=np.array([float(row["Iobs_over_I0"]) for row in profile_rows]),
        diagnostics={"sample_count": len(profile_rows)},
    )


def write_image(rows: list[dict[str, object]], output: Path, image_size: int, image_radius: float) -> None:
    if image_size <= 0:
        raise ValueError("image_size must be positive")
    if image_radius <= 0.0:
        raise ValueError("image_radius must be positive")

    plt = _configure_matplotlib()

    output.parent.mkdir(parents=True, exist_ok=True)
    profile_names = _ordered_profile_names(rows)
    figure, axes = plt.subplots(len(profile_names), 1, figsize=(4.0, 4.0 * len(profile_names)), squeeze=False)
    for axis, profile_name in zip(axes[:, 0], profile_names):
        profile = _radial_profile_from_rows(_profile_rows(rows, profile_name))
        image = render_axisymmetric_image(profile, ImageGrid(width=image_size, height=image_size, radius=image_radius))
        vmax = max(float(np.max(image.pixels)), 1e-300)
        axis.imshow(
            image.pixels,
            extent=[-image_radius, image_radius, -image_radius, image_radius],
            origin="lower",
            cmap="afmhot",
            vmin=0.0,
            vmax=vmax,
        )
        axis.set_title(profile_name)
        axis.set_aspect("equal")
        axis.set_axis_off()
    figure.tight_layout(pad=0.15)
    figure.savefig(output, dpi=180, facecolor="black")
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate Fig. 5 thin-disk intensity profiles from Gralla-Holz-Wald.")
    parser.add_argument("--output", required=True, help="CSV output path")
    parser.add_argument("--plot-output", help="Optional PNG output path for the observed intensity profile")
    parser.add_argument("--image-output", help="Optional PNG output path for the axisymmetric image-plane rendering")
    parser.add_argument("--profile", choices=["top", "middle", "bottom", "all"], default="all")
    parser.add_argument("--samples", type=int, default=1200)
    parser.add_argument("--b-max", type=float, default=13.0)
    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument("--image-radius", type=float, default=13.0)
    parser.add_argument("--max-order", type=int, default=3)
    parser.add_argument("--show-integration-warnings", action="store_true")
    args = parser.parse_args()

    if args.samples < 2:
        raise SystemExit("--samples must be at least 2")
    if args.b_max <= 0.0:
        raise SystemExit("--b-max must be positive")
    if args.image_size <= 0:
        raise SystemExit("--image-size must be positive")
    if args.image_radius <= 0.0:
        raise SystemExit("--image-radius must be positive")
    if args.max_order < 1:
        raise SystemExit("--max-order must be at least 1")

    profile_names = list(PROFILES) if args.profile == "all" else [args.profile]
    rows: list[dict[str, object]] = []
    for profile_name in profile_names:
        rows.extend(
            generate_rows(
                profile_name,
                args.samples,
                args.b_max,
                args.max_order,
                args.show_integration_warnings,
            )
        )
    write_csv(rows, Path(args.output))
    if args.plot_output:
        write_plot(rows, Path(args.plot_output))
    if args.image_output:
        write_image(rows, Path(args.image_output), args.image_size, args.image_radius)


if __name__ == "__main__":
    main()
