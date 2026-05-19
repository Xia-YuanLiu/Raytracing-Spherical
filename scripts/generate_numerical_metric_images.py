from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
import tempfile
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.integrate import IntegrationWarning

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from spherical_raytracing import (
    DiskWindow,
    EventType,
    FiniteStaticObserver,
    ImageGrid,
    QuadTransferSolver,
    RadialProfile,
    SolverOptions,
    StaticDomainSelector,
    TabulatedMetric,
    ThinDiskSource,
    compute_intersections,
    load_metric_npz,
    observed_intensity,
    render_axisymmetric_image,
)


@dataclass(frozen=True)
class NumericalImageCase:
    path: Path
    metric: TabulatedMetric
    observer: FiniteStaticObserver
    solver: QuadTransferSolver
    disk: DiskWindow
    source: ThinDiskSource
    r_grid_min: float
    r_grid_max: float
    r_domain_lo: float
    r_domain_hi: float
    r_ph: float
    b_crit: float
    rho_crit: float
    image_radius: float
    load_warnings: tuple[str, ...]


def _configure_matplotlib():
    os.environ.setdefault(
        "MPLCONFIGDIR",
        str(Path(tempfile.gettempdir()) / "spherical_raytracing_matplotlib"),
    )
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def _screen_radius_for_b(observer: FiniteStaticObserver, b: float) -> float:
    a_obs = observer.metric.A(observer.r_obs)
    sin_alpha = b * math.sqrt(a_obs) / observer.r_obs
    sin_alpha = min(max(sin_alpha, 0.0), 1.0 - 1e-12)
    return observer.screen_radius(math.asin(sin_alpha))


def _choose_observer_radius(
    *,
    r_ph: float,
    r_lo: float,
    r_hi: float,
    r_grid_max: float,
) -> float:
    upper = min(r_hi, r_grid_max) if math.isfinite(r_hi) else r_grid_max
    if not math.isfinite(upper):
        raise ValueError("finite tabulated r_grid_max is required")

    if math.isfinite(r_hi):
        r_obs = 0.5 * (r_ph + r_hi)
    else:
        r_obs = min(5.0 * r_ph, 0.9 * r_grid_max)

    margin = max(1e-6 * max(abs(upper), 1.0), 1e-6)
    r_obs = min(r_obs, upper - margin)
    lower = max(r_lo + margin, r_ph * 1.05)
    if r_obs <= lower:
        r_obs = 0.5 * (lower + upper)
    if r_obs <= r_lo or r_obs >= upper:
        raise ValueError(
            f"could not choose observer radius inside support: r_obs={r_obs}, "
            f"domain=({r_lo}, {r_hi}), r_grid_max={r_grid_max}"
        )
    return r_obs


def _build_case(path: Path, *, max_phi: float, radius_scale: float) -> NumericalImageCase:
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        payload = load_metric_npz(path)
    load_warnings = tuple(str(item.message) for item in caught)

    r_lo, r_hi = StaticDomainSelector().choose(payload)
    metric = TabulatedMetric(payload=payload, static_domain=(r_lo, r_hi))
    curves = [curve for curve in metric.critical_curves() if curve.reachable_from_observer]
    if not curves:
        raise ValueError(f"{path.name}: no reachable critical curve")
    curve = curves[0]

    photons = [value for value in metric.photon_spheres() if r_lo < value < r_hi]
    r_ph = photons[-1] if photons else curve.r_ph
    r_grid_min = float(np.min(payload.r))
    r_grid_max = float(np.max(payload.r))
    r_obs = _choose_observer_radius(
        r_ph=r_ph,
        r_lo=r_lo,
        r_hi=r_hi,
        r_grid_max=r_grid_max,
    )
    observer = FiniteStaticObserver(r_obs=r_obs, metric=metric)
    solver = QuadTransferSolver(
        metric=metric,
        observer=observer,
        options=SolverOptions(critical_exclusion=0.0, max_phi=max_phi),
    )

    disk_r_min = max(r_lo + 1e-6, r_ph * 1.05)
    disk_r_max = min(r_obs - 0.5, r_grid_max)
    if math.isfinite(r_hi):
        disk_r_max = min(disk_r_max, r_hi - 1e-6)
    if disk_r_max <= disk_r_min:
        disk_r_max = 0.5 * (disk_r_min + r_obs)
    if disk_r_max <= disk_r_min:
        raise ValueError(
            f"{path.name}: cannot place disk inside observer radius "
            f"(r_min={disk_r_min}, r_max={disk_r_max}, r_obs={r_obs})"
        )

    rho_crit = _screen_radius_for_b(observer, curve.b_crit)
    image_radius = max(radius_scale * rho_crit, rho_crit + 1e-6)
    source = ThinDiskSource(lambda r, region=None: 1.0 / (r * r))

    return NumericalImageCase(
        path=path,
        metric=metric,
        observer=observer,
        solver=solver,
        disk=DiskWindow(r_min=disk_r_min, r_max=disk_r_max),
        source=source,
        r_grid_min=r_grid_min,
        r_grid_max=r_grid_max,
        r_domain_lo=float(r_lo),
        r_domain_hi=float(r_hi),
        r_ph=float(r_ph),
        b_crit=float(curve.b_crit),
        rho_crit=float(rho_crit),
        image_radius=float(image_radius),
        load_warnings=load_warnings,
    )


def _rho_grid(samples: int, rho_max: float, rho_crit: float) -> np.ndarray:
    base = np.linspace(0.0, rho_max, samples)
    if rho_crit <= 0.0 or rho_crit >= rho_max:
        return np.unique(np.round(base, 12))

    offset_min = max(rho_crit * 1e-6, 1e-8)
    offset_max = max(min(0.25 * rho_crit, rho_max - rho_crit), offset_min)
    offsets = np.geomspace(offset_min, offset_max, max(samples // 3, 48))
    critical = np.concatenate(([rho_crit], rho_crit - offsets, rho_crit + offsets))
    values = np.concatenate((base, critical))
    values = values[(values >= 0.0) & (values <= rho_max)]
    return np.unique(np.round(values, 12))


def _contributions_by_class(intersections, observed) -> dict[str, float]:
    contributions = {"direct": 0.0, "lensing_ring": 0.0, "photon_ring": 0.0}
    for intersection, contribution in zip(intersections, observed.contributions):
        contributions[intersection.path_class] += contribution
    return contributions


def _sample_case(case: NumericalImageCase, *, samples: int, max_order: int, show_warnings: bool) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for rho in _rho_grid(samples, case.image_radius, case.rho_crit):
        rho = float(rho)
        if rho <= 0.0:
            rows.append(
                {
                    "rho": 0.0,
                    "alpha": 0.0,
                    "b": 0.0,
                    "Iobs_over_I0": 0.0,
                    "direct": 0.0,
                    "lensing_ring": 0.0,
                    "photon_ring": 0.0,
                    "intersection_count": 0,
                    "termination_reason": "axis",
                    "max_phi_reached": False,
                    "phi_end": 0.0,
                }
            )
            continue

        alpha = math.atan(rho / case.observer.r_obs)
        b = case.observer.impact_parameter(alpha)
        with warnings.catch_warnings():
            if not show_warnings:
                warnings.simplefilter("ignore", IntegrationWarning)
            ray = case.solver.trace_b(b)
            intersections = [
                intersection
                for intersection in compute_intersections(ray, case.disk, max_order=max_order)
                if intersection.ray_segment.endpoint_event != EventType.MAX_PHI
            ]
            observed = observed_intensity(intersections, case.source, case.metric, case.observer)
        contributions = _contributions_by_class(intersections, observed)
        phi_end = ray.segments[-1].phi_end if ray.segments else math.nan
        rows.append(
            {
                "rho": rho,
                "alpha": alpha,
                "b": b,
                "Iobs_over_I0": observed.total,
                "direct": contributions["direct"],
                "lensing_ring": contributions["lensing_ring"],
                "photon_ring": contributions["photon_ring"],
                "intersection_count": len(intersections),
                "termination_reason": ray.diagnostics.termination_reason,
                "max_phi_reached": ray.diagnostics.max_phi_reached,
                "phi_end": phi_end,
            }
        )
    return rows


def _write_csv(rows: list[dict[str, object]], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "rho",
        "alpha",
        "b",
        "Iobs_over_I0",
        "direct",
        "lensing_ring",
        "photon_ring",
        "intersection_count",
        "termination_reason",
        "max_phi_reached",
        "phi_end",
    ]
    with output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _radial_profile_from_rows(rows: list[dict[str, object]]) -> RadialProfile:
    rows = sorted(rows, key=lambda row: float(row["rho"]))
    return RadialProfile(
        coordinates=np.array([float(row["rho"]) for row in rows]),
        intensities=np.array([float(row["Iobs_over_I0"]) for row in rows]),
        diagnostics={"sample_count": len(rows)},
    )


def _write_profile_plot(rows: list[dict[str, object]], case: NumericalImageCase, output: Path) -> None:
    plt = _configure_matplotlib()
    output.parent.mkdir(parents=True, exist_ok=True)
    sorted_rows = sorted(rows, key=lambda row: float(row["rho"]))
    figure, axis = plt.subplots(figsize=(8.0, 4.8))
    axis.plot(
        [float(row["rho"]) for row in sorted_rows],
        [float(row["Iobs_over_I0"]) for row in sorted_rows],
        linewidth=1.25,
        color="tab:blue",
    )
    axis.axvline(case.rho_crit, color="tab:orange", linewidth=1.0, alpha=0.75)
    axis.set_title(case.path.stem)
    axis.set_xlabel("screen radius rho")
    axis.set_ylabel("I_obs/I0")
    axis.set_xlim(0.0, case.image_radius)
    axis.set_ylim(bottom=0.0)
    axis.grid(alpha=0.25)
    figure.tight_layout()
    figure.savefig(output, dpi=180)
    plt.close(figure)


def _write_image(rows: list[dict[str, object]], case: NumericalImageCase, output: Path, image_size: int) -> None:
    plt = _configure_matplotlib()
    output.parent.mkdir(parents=True, exist_ok=True)
    profile = _radial_profile_from_rows(rows)
    image = render_axisymmetric_image(
        profile,
        ImageGrid(width=image_size, height=image_size, radius=case.image_radius),
    )
    vmax = max(float(np.max(image.pixels)), 1e-300)
    figure, axis = plt.subplots(figsize=(4.2, 4.2))
    axis.imshow(
        image.pixels,
        extent=[-case.image_radius, case.image_radius, -case.image_radius, case.image_radius],
        origin="lower",
        cmap="afmhot",
        vmin=0.0,
        vmax=vmax,
    )
    axis.set_aspect("equal")
    axis.set_axis_off()
    figure.tight_layout(pad=0.05)
    figure.savefig(output, dpi=180, facecolor="black")
    plt.close(figure)


def _write_contact_sheet(summaries: list[dict[str, object]], output: Path) -> None:
    if not summaries:
        return
    plt = _configure_matplotlib()
    output.parent.mkdir(parents=True, exist_ok=True)
    columns = 3
    rows = math.ceil(len(summaries) / columns)
    figure, axes = plt.subplots(rows, columns, figsize=(4.0 * columns, 4.3 * rows), squeeze=False)
    for axis in axes.ravel():
        axis.set_axis_off()
    for axis, summary in zip(axes.ravel(), summaries):
        image_path = Path(str(summary["image_png"]))
        pixels = plt.imread(image_path)
        axis.imshow(pixels)
        axis.set_title(str(summary["name"]), color="white", fontsize=10)
        axis.set_axis_off()
    figure.tight_layout(pad=0.4)
    figure.savefig(output, dpi=180, facecolor="black")
    plt.close(figure)


def _summary(case: NumericalImageCase, rows: list[dict[str, object]], output_dir: Path) -> dict[str, object]:
    intensities = [float(row["Iobs_over_I0"]) for row in rows]
    return {
        "name": case.path.stem,
        "input_npz": str(case.path),
        "profile_csv": str(output_dir / f"{case.path.stem}_profile.csv"),
        "profile_plot_png": str(output_dir / f"{case.path.stem}_profile.png"),
        "image_png": str(output_dir / f"{case.path.stem}_image.png"),
        "r_grid": [case.r_grid_min, case.r_grid_max],
        "static_domain": [
            case.r_domain_lo,
            None if not math.isfinite(case.r_domain_hi) else case.r_domain_hi,
        ],
        "r_obs": case.observer.r_obs,
        "disk_window": [case.disk.r_min, case.disk.r_max],
        "r_ph": case.r_ph,
        "b_crit": case.b_crit,
        "rho_crit": case.rho_crit,
        "image_radius": case.image_radius,
        "sample_count": len(rows),
        "intensity_max": max(intensities) if intensities else 0.0,
        "load_warnings": list(case.load_warnings),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate thin-disk images from numerical metric .npz files.")
    parser.add_argument(
        "--input-dir",
        default="/home/yuanpang/Work/Li-DM-BH/outputs/sanity/data",
        help="Directory containing Li-DM-BH or standard r/A/B .npz files.",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/numerical_metrics_sanity",
        help="Directory for CSV, profile plots, images, and manifest.json.",
    )
    parser.add_argument("--samples", type=int, default=360)
    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument("--radius-scale", type=float, default=1.8)
    parser.add_argument("--max-order", type=int, default=4)
    parser.add_argument("--max-phi", type=float, default=12.0 * math.pi)
    parser.add_argument("--show-integration-warnings", action="store_true")
    args = parser.parse_args()

    if args.samples < 2:
        raise SystemExit("--samples must be at least 2")
    if args.image_size <= 0:
        raise SystemExit("--image-size must be positive")
    if args.radius_scale <= 1.0:
        raise SystemExit("--radius-scale must be greater than 1")
    if args.max_order < 1:
        raise SystemExit("--max-order must be at least 1")
    if args.max_phi <= 0.0:
        raise SystemExit("--max-phi must be positive")

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    paths = sorted(input_dir.glob("*.npz"))
    if not paths:
        raise SystemExit(f"no .npz files found in {input_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)
    summaries: list[dict[str, object]] = []
    for path in paths:
        print(f"[numerical-image] {path.name}")
        case = _build_case(path, max_phi=args.max_phi, radius_scale=args.radius_scale)
        rows = _sample_case(
            case,
            samples=args.samples,
            max_order=args.max_order,
            show_warnings=args.show_integration_warnings,
        )
        csv_path = output_dir / f"{path.stem}_profile.csv"
        plot_path = output_dir / f"{path.stem}_profile.png"
        image_path = output_dir / f"{path.stem}_image.png"
        _write_csv(rows, csv_path)
        _write_profile_plot(rows, case, plot_path)
        _write_image(rows, case, image_path, args.image_size)
        summaries.append(_summary(case, rows, output_dir))

    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps({"cases": summaries}, indent=2), encoding="utf-8")
    contact_sheet = output_dir / "all_numerical_metric_images.png"
    _write_contact_sheet(summaries, contact_sheet)
    print(f"[numerical-image] wrote {manifest_path}")
    print(f"[numerical-image] wrote {contact_sheet}")


if __name__ == "__main__":
    main()
