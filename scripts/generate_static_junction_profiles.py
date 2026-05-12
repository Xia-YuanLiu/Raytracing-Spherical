from __future__ import annotations

import argparse
import json
import math
import os
import sys
import tempfile
from dataclasses import replace
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from spherical_raytracing.imaging import ImageGrid, RadialProfile, SamplingOptions, render_axisymmetric_image, sample_radial_profile
from spherical_raytracing.junctions import StaticJunctionSpacetime, StaticShell, annotate_junction_intersections, compare_junction_results
from spherical_raytracing.junction_tracing import StaticJunctionHamiltonianSolver, StaticJunctionTransferSolver
from spherical_raytracing.metrics import ReissnerNordstromDeSitterMetric, ReissnerNordstromMetric, SchwarzschildMetric
from spherical_raytracing.observers import FiniteStaticObserver
from spherical_raytracing.sources import ThinDiskSource, observed_intensity
from spherical_raytracing.transfer import DiskWindow, compute_intersections


def emissivity(r: float, region: str | None = None) -> float:
    return 1.0 / max(r**2, 1e-300)


def build_family(
    family: str,
    *,
    m_minus: float | None = None,
    m_plus: float | None = None,
    q_minus: float | None = None,
    q_plus: float | None = None,
    lambda_minus: float | None = None,
    lambda_plus: float | None = None,
    shell_radius: float | None = None,
    observer_radius: float | None = None,
) -> tuple[StaticJunctionSpacetime, FiniteStaticObserver, dict[str, float]]:
    if family == "schwarzschild":
        params = {
            "m_minus": 0.1 if m_minus is None else m_minus,
            "m_plus": 1.0 if m_plus is None else m_plus,
            "shell_radius": 5.0 if shell_radius is None else shell_radius,
            "observer_radius": 50.0 if observer_radius is None else observer_radius,
        }
        inner = SchwarzschildMetric(mass=params["m_minus"], region="inner")
        outer = SchwarzschildMetric(mass=params["m_plus"], region="outer")
    elif family == "rn":
        params = {
            "m_minus": 0.8 if m_minus is None else m_minus,
            "m_plus": 1.0 if m_plus is None else m_plus,
            "q_minus": 0.2 if q_minus is None else q_minus,
            "q_plus": 0.2 if q_plus is None else q_plus,
            "shell_radius": 5.0 if shell_radius is None else shell_radius,
            "observer_radius": 50.0 if observer_radius is None else observer_radius,
        }
        inner = ReissnerNordstromMetric(mass=params["m_minus"], charge=params["q_minus"], region="inner")
        outer = ReissnerNordstromMetric(mass=params["m_plus"], charge=params["q_plus"], region="outer")
    elif family == "rnds":
        params = {
            "m_minus": 0.8 if m_minus is None else m_minus,
            "m_plus": 1.0 if m_plus is None else m_plus,
            "q_minus": 0.2 if q_minus is None else q_minus,
            "q_plus": 0.2 if q_plus is None else q_plus,
            "lambda_minus": 0.01 if lambda_minus is None else lambda_minus,
            "lambda_plus": 0.01 if lambda_plus is None else lambda_plus,
            "shell_radius": 5.0 if shell_radius is None else shell_radius,
            "observer_radius": 6.0 if observer_radius is None else observer_radius,
        }
        inner = ReissnerNordstromDeSitterMetric(
            mass=params["m_minus"],
            charge=params["q_minus"],
            cosmological_constant=params["lambda_minus"],
            region="inner",
        )
        outer = ReissnerNordstromDeSitterMetric(
            mass=params["m_plus"],
            charge=params["q_plus"],
            cosmological_constant=params["lambda_plus"],
            region="outer",
        )
    else:
        raise ValueError(f"unknown family {family!r}")
    junction = StaticJunctionSpacetime(inner_metric=inner, outer_metric=outer, shell=StaticShell(radius=params["shell_radius"]))
    observer = FiniteStaticObserver(r_obs=params["observer_radius"], metric=outer)
    return junction, observer, params


def build_profile(family: str, args: argparse.Namespace) -> dict[str, object]:
    junction, observer, params = build_family(
        family,
        m_minus=args.m_minus,
        m_plus=args.m_plus,
        q_minus=args.q_minus,
        q_plus=args.q_plus,
        lambda_minus=args.lambda_minus,
        lambda_plus=args.lambda_plus,
        shell_radius=args.shell_radius,
        observer_radius=args.observer_radius,
    )
    inner = junction.inner_metric
    outer = junction.outer_metric
    solver = StaticJunctionTransferSolver(junction=junction, observer=observer)
    comparison_solver = StaticJunctionHamiltonianSolver(junction=junction, observer=observer) if args.compare_backends else None
    source = ThinDiskSource(intensity_profile=emissivity)
    disk = DiskWindow(r_min=args.disk_r_min, r_max=args.disk_r_max, enabled_regions=frozenset({"inner", "outer"}))

    rows_by_alpha: dict[float, dict[str, object]] = {}

    def intensity_at(alpha: float):
        ray = solver.trace_alpha(alpha)
        intersections = compute_intersections(ray, disk, max_order=args.max_order)
        annotated = annotate_junction_intersections(intersections, ray, observer, {"inner": inner, "outer": outer})
        backend_comparison: dict[str, float | int | str] | None = None
        if comparison_solver is not None:
            comparison_ray = comparison_solver.trace_alpha(alpha)
            comparison_intersections = compute_intersections(comparison_ray, disk, max_order=args.max_order)
            comparison_annotated = annotate_junction_intersections(
                comparison_intersections,
                comparison_ray,
                observer,
                {"inner": inner, "outer": outer},
            )
            ray = replace(ray, intersections=annotated)
            comparison_ray = replace(comparison_ray, intersections=comparison_annotated)
            backend_comparison = compare_junction_results(ray, comparison_ray)
        observed = observed_intensity(annotated, source, outer, observer, region_metrics={"inner": inner, "outer": outer})
        diagnostics = dict(observed.diagnostics)
        diagnostics["termination_reason"] = ray.termination_reason
        diagnostics["shell_crossing_count"] = len(ray.shell_crossings)
        diagnostics["backend_disagreement"] = 0.0 if backend_comparison is None else float(backend_comparison["backend_disagreement"])
        observed = replace(observed, diagnostics=diagnostics)
        row = {
            "alpha": float(alpha),
            "rho": observer.screen_radius(float(alpha)),
            "b_plus": ray.b_plus,
            "intensity": observed.total,
            "termination_reason": ray.termination_reason,
            "shell_crossing_count": len(ray.shell_crossings),
            "intersections": [
                {
                    "m": item.m,
                    "r": item.r,
                    "region": item.region,
                    "path_class": item.path_class,
                    "g": item.diagnostics.get("g", 0.0),
                }
                for item in annotated
            ],
        }
        if backend_comparison is not None:
            row["backend_comparison"] = backend_comparison
        rows_by_alpha[float(alpha)] = row
        return observed

    profile = sample_radial_profile(
        args.alpha_min,
        args.alpha_max,
        intensity_at,
        critical_points=[],
        options=SamplingOptions(initial_count=args.samples, max_refine=args.max_refine),
    )
    rows = [rows_by_alpha[float(alpha)] for alpha in profile.coordinates]
    return {
        "model": f"{family}_static_junction",
        "parameters": params,
        "warnings": list(junction.physics_diagnostics(observer.r_obs).warnings),
        "sampling_diagnostics": profile.diagnostics,
        "samples": rows,
    }


def _configure_matplotlib():
    os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "spherical_raytracing_matplotlib"))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def write_image(profile: dict[str, object], output: Path, image_size: int) -> None:
    if image_size <= 0:
        raise ValueError("image_size must be positive")
    rows = list(profile["samples"])
    coordinates = np.array([float(row["rho"]) for row in rows])
    intensities = np.array([float(row["intensity"]) for row in rows])
    radius = max(float(np.max(coordinates)), 1.0)
    radial_profile = RadialProfile(
        coordinates=coordinates,
        intensities=intensities,
        diagnostics={"sample_count": len(rows)},
    )
    image = render_axisymmetric_image(radial_profile, ImageGrid(width=image_size, height=image_size, radius=radius))
    vmax = max(float(np.max(image.pixels)), 1e-300)
    plt = _configure_matplotlib()
    output.parent.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(4.0, 4.0))
    axis.imshow(
        image.pixels,
        extent=[-radius, radius, -radius, radius],
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


def enforce_strict_physical(profiles: dict[str, dict[str, object]]) -> None:
    failures = []
    for family, profile in profiles.items():
        warnings = [str(warning) for warning in profile.get("warnings", [])]
        if warnings:
            failures.append(f"{family}: {', '.join(warnings)}")
    if failures:
        raise SystemExit("strict physical mode rejected outputs: " + "; ".join(failures))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--samples", type=int, default=64)
    parser.add_argument("--max-order", type=int, default=3)
    parser.add_argument("--max-refine", type=int, default=2)
    parser.add_argument("--families", nargs="+", default=["schwarzschild", "rn", "rnds"])
    parser.add_argument("--m-minus", type=float)
    parser.add_argument("--m-plus", type=float)
    parser.add_argument("--q-minus", type=float)
    parser.add_argument("--q-plus", type=float)
    parser.add_argument("--lambda-minus", type=float)
    parser.add_argument("--lambda-plus", type=float)
    parser.add_argument("--shell-radius", type=float)
    parser.add_argument("--observer-radius", type=float)
    parser.add_argument("--disk-r-min", type=float, default=0.25)
    parser.add_argument("--disk-r-max", type=float, default=100.0)
    parser.add_argument("--alpha-min", type=float, default=0.01)
    parser.add_argument("--alpha-max", type=float, default=0.25)
    parser.add_argument("--image-output-dir", help="Optional directory for per-family PNG images")
    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument("--strict-physical", action="store_true")
    parser.add_argument("--compare-backends", action="store_true")
    args = parser.parse_args()
    if args.samples <= 0:
        raise SystemExit("--samples must be positive")
    if args.max_order <= 0:
        raise SystemExit("--max-order must be positive")
    if args.max_refine < 0:
        raise SystemExit("--max-refine must be non-negative")
    if args.image_size <= 0:
        raise SystemExit("--image-size must be positive")
    if args.disk_r_min <= 0.0 or args.disk_r_max <= args.disk_r_min:
        raise SystemExit("disk window must satisfy 0 < --disk-r-min < --disk-r-max")
    if args.alpha_min < 0.0 or args.alpha_max <= args.alpha_min:
        raise SystemExit("alpha window must satisfy 0 <= --alpha-min < --alpha-max")
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    try:
        profiles = {family: build_profile(family, args) for family in args.families}
    except ValueError as exc:
        prefix = "strict physical mode rejected outputs" if args.strict_physical else "profile generation rejected outputs"
        raise SystemExit(f"{prefix}: {exc}") from exc
    if args.strict_physical:
        enforce_strict_physical(profiles)
    output.write_text(json.dumps({"families": profiles}, indent=2, sort_keys=True))
    if args.image_output_dir:
        image_dir = Path(args.image_output_dir)
        for family, profile in profiles.items():
            write_image(profile, image_dir / f"{family}.png", args.image_size)


if __name__ == "__main__":
    main()
