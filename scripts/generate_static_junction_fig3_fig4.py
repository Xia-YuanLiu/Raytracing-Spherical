from __future__ import annotations

import argparse
import json
import math
import os
import sys
import tempfile
import warnings
from pathlib import Path

import numpy as np
from scipy.integrate import IntegrationWarning

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from spherical_raytracing.imaging import ImageGrid, RadialProfile, render_axisymmetric_image
from spherical_raytracing.junction_analysis import paper_style_emissivity_factory
from spherical_raytracing.junctions import StaticJunctionSpacetime, StaticShell, annotate_junction_intersections
from spherical_raytracing.junction_tracing import StaticJunctionTransferSolver
from spherical_raytracing.metrics import SchwarzschildMetric
from spherical_raytracing.observers import FiniteStaticObserver
from spherical_raytracing.policies import SolverOptions
from spherical_raytracing.sources import ThinDiskSource, observed_intensity
from spherical_raytracing.transfer import DiskWindow, compute_intersections


COLORS = {1: "black", 2: "tab:orange", 3: "red"}
LABELS = {1: "First", 2: "Second", 3: "Third"}


def paper_emissivity_factory(m_minus: float):
    return paper_style_emissivity_factory(
        SchwarzschildMetric(mass=m_minus, region="inner"),
        shell_radius=math.inf,
    )


def _configure_matplotlib():
    os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "spherical_raytracing_matplotlib"))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def build_spacetime(
    *,
    m_minus: float,
    m_plus: float,
    shell_radius: float,
    observer_radius: float,
    max_phi: float,
    critical_exclusion: float,
) -> tuple[StaticJunctionSpacetime, FiniteStaticObserver, StaticJunctionTransferSolver]:
    inner = SchwarzschildMetric(mass=m_minus, region="inner")
    outer = SchwarzschildMetric(mass=m_plus, region="outer")
    junction = StaticJunctionSpacetime(inner_metric=inner, outer_metric=outer, shell=StaticShell(radius=shell_radius))
    observer = FiniteStaticObserver(r_obs=observer_radius, metric=outer)
    solver = StaticJunctionTransferSolver(
        junction=junction,
        observer=observer,
        options=SolverOptions(critical_exclusion=critical_exclusion, max_phi=max_phi),
    )
    return junction, observer, solver


def mapped_inner_critical_b(m_minus: float, m_plus: float, shell_radius: float) -> float:
    inner = SchwarzschildMetric(mass=m_minus, region="inner")
    outer = SchwarzschildMetric(mass=m_plus, region="outer")
    inner_b_crit = inner.critical_curves()[0].b_crit
    return inner_b_crit * math.sqrt(inner.A(shell_radius) / outer.A(shell_radius))


def b_grid(samples: int, b_max: float, *, m_minus: float, m_plus: float, shell_radius: float) -> np.ndarray:
    base = np.linspace(0.0, b_max, samples)
    outer_critical = 3.0 * math.sqrt(3.0) * m_plus
    inner_critical = mapped_inner_critical_b(m_minus, m_plus, shell_radius)

    focused: list[np.ndarray] = [
        np.linspace(max(0.0, inner_critical - 0.35), min(b_max, inner_critical + 0.35), max(samples, 32)),
        np.linspace(max(0.0, outer_critical - 0.45), min(b_max, outer_critical + 0.45), max(samples, 48)),
    ]
    for center in [inner_critical, outer_critical, shell_radius]:
        offsets = np.geomspace(1e-5, 8e-2, max(samples // 2, 24))
        focused.append(center - offsets)
        focused.append(center + offsets)

    values = np.concatenate([base, *focused])
    values = values[(values >= 0.0) & (values <= b_max)]
    return np.unique(np.round(values, 12))


def sample_rows(
    *,
    solver: StaticJunctionTransferSolver,
    junction: StaticJunctionSpacetime,
    observer: FiniteStaticObserver,
    source: ThinDiskSource,
    disk: DiskWindow,
    b_values: np.ndarray,
    max_order: int,
    show_integration_warnings: bool,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for b_plus in b_values:
        b = float(b_plus)
        if b <= 0.0:
            rows.append(
                {
                    "b_plus": 0.0,
                    "intensity": 0.0,
                    "termination_reason": "axis",
                    "shell_crossing_count": 0,
                    "intersections": [],
                }
            )
            continue
        with warnings.catch_warnings():
            if not show_integration_warnings:
                warnings.simplefilter("ignore", IntegrationWarning)
                warnings.filterwarnings("ignore", category=IntegrationWarning)
            ray = solver.trace_b(b)
            intersections = compute_intersections(ray, disk, max_order=max_order)
            annotated = annotate_junction_intersections(
                intersections,
                ray,
                observer,
                {"inner": junction.inner_metric, "outer": junction.outer_metric},
            )
            observed = observed_intensity(
                annotated,
                source,
                junction.outer_metric,
                observer,
                region_metrics={"inner": junction.inner_metric, "outer": junction.outer_metric},
            )

        intersection_rows = []
        for intersection, contribution, weight in zip(annotated, observed.contributions, observed.redshift_weights):
            intersection_rows.append(
                {
                    "m": intersection.m,
                    "r": intersection.r,
                    "phi": intersection.phi,
                    "region": intersection.region,
                    "path_class": intersection.path_class,
                    "g": intersection.diagnostics.get("g", 0.0),
                    "redshift_weight": weight,
                    "contribution": contribution,
                }
            )
        rows.append(
            {
                "b_plus": b,
                "intensity": observed.total,
                "termination_reason": ray.termination_reason,
                "shell_crossing_count": len(ray.shell_crossings),
                "intersections": intersection_rows,
            }
        )
    rows.sort(key=lambda row: float(row["b_plus"]))
    return rows


def rows_to_profile(rows: list[dict[str, object]]) -> RadialProfile:
    return RadialProfile(
        coordinates=np.array([float(row["b_plus"]) for row in rows]),
        intensities=np.array([float(row["intensity"]) for row in rows]),
        diagnostics={"sample_count": len(rows)},
    )


def _value_for_order(row: dict[str, object], order: int, key: str) -> float | None:
    for intersection in row["intersections"]:
        if int(intersection["m"]) == order:
            value = float(intersection[key])
            if math.isfinite(value):
                return value
            return None
    return None


def _values_by_order(
    rows: list[dict[str, object]],
    order: int,
    key: str,
    *,
    missing_value: float | None = None,
) -> tuple[list[float], list[float]]:
    xs: list[float] = []
    ys: list[float] = []
    for row in rows:
        value = _value_for_order(row, order, key)
        if value is None:
            if missing_value is None:
                continue
            value = missing_value
        xs.append(float(row["b_plus"]))
        ys.append(value)
    return xs, ys


def _plot_fig3_insets(axes, rows: list[dict[str, object]], *, r_plot_max: float) -> None:
    from mpl_toolkits.axes_grid1.inset_locator import inset_axes

    bcrit = 3.0 * math.sqrt(3.0)
    r_insets = [
        ((0.52, 0.57, 0.27, 0.33), (bcrit - 0.06, bcrit + 0.10), (0.0, r_plot_max)),
        ((0.52, 0.15, 0.27, 0.33), (0.82, 1.08), (0.0, r_plot_max)),
    ]
    for bounds, xlim, ylim in r_insets:
        inset = inset_axes(
            axes[1],
            width="100%",
            height="100%",
            bbox_to_anchor=bounds,
            bbox_transform=axes[1].transAxes,
            borderpad=0.0,
        )
        for order in [1, 2, 3]:
            xs, ys = _values_by_order(rows, order, "r", missing_value=0.0)
            inset.plot(xs, ys, color=COLORS[order], linewidth=0.8)
        inset.set_xlim(*xlim)
        inset.set_ylim(*ylim)
        inset.tick_params(direction="in", labelsize=6, pad=1)

    g_insets = [
        ((0.56, 0.58, 0.25, 0.31), (bcrit - 0.06, bcrit + 0.10)),
        ((0.56, 0.16, 0.25, 0.31), (0.82, 1.08)),
    ]
    for bounds, xlim in g_insets:
        inset = inset_axes(
            axes[2],
            width="100%",
            height="100%",
            bbox_to_anchor=bounds,
            bbox_transform=axes[2].transAxes,
            borderpad=0.0,
        )
        for order in [1, 2, 3]:
            xs, ys = _values_by_order(rows, order, "g", missing_value=0.0)
            inset.plot(xs, ys, color=COLORS[order], linewidth=0.8)
        inset.set_xlim(*xlim)
        inset.set_ylim(0.0, 1.05)
        inset.tick_params(direction="in", labelsize=6, pad=1)


def write_json(
    *,
    rows: list[dict[str, object]],
    output: Path,
    m_minus: float,
    m_plus: float,
    shell_radius: float,
    observer_radius: float,
    disk_r_min: float,
    disk_r_max: float,
) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model": "schwarzschild_static_junction",
        "paper": "Imaging Signatures of the Israel Junction Photon Ring Evolution in Dynamical Thin Shell",
        "figure_targets": ["Fig. 3", "Fig. 4"],
        "parameters": {
            "m_minus": m_minus,
            "m_plus": m_plus,
            "shell_radius": shell_radius,
            "observer_radius": observer_radius,
            "disk_r_min": disk_r_min,
            "disk_r_max": disk_r_max,
        },
        "source_model": {
            "name": "paper_eq_4_1",
            "inner_horizon_radius": 2.0 * m_minus,
            "formula": "I_em(r)=1/(r-r_h_minus+1)^3 for r>=r_h_minus, else 0",
        },
        "samples": rows,
    }
    output.write_text(json.dumps(payload, indent=2, sort_keys=True))


def write_fig3(
    rows: list[dict[str, object]],
    output: Path,
    *,
    b_max: float,
    fig3_b_max: float,
    shell_radius: float,
    r_plot_max: float,
) -> None:
    plt = _configure_matplotlib()
    output.parent.mkdir(parents=True, exist_ok=True)

    figure, axes = plt.subplots(1, 3, figsize=(12.5, 3.6), constrained_layout=True)
    b = [float(row["b_plus"]) for row in rows]
    intensity = [float(row["intensity"]) for row in rows]

    axes[0].plot(b, intensity, color="blue", linewidth=1.1)
    axes[0].set_xlabel("b")
    axes[0].set_ylabel("I_obs")
    axes[0].set_xlim(0.0, b_max)
    axes[0].set_ylim(bottom=0.0)

    for order in [1, 2, 3]:
        xs, ys = _values_by_order(rows, order, "r", missing_value=0.0)
        axes[1].plot(xs, ys, color=COLORS[order], linewidth=1.0, label=LABELS[order])
    axes[1].axhline(shell_radius, color="0.75", linestyle=":", linewidth=0.8)
    axes[1].set_xlabel("b")
    axes[1].set_ylabel("r")
    axes[1].set_xlim(0.0, fig3_b_max)
    axes[1].set_ylim(0.0, r_plot_max)
    axes[1].legend(frameon=False)

    for order in [1, 2, 3]:
        xs, ys = _values_by_order(rows, order, "g", missing_value=0.0)
        axes[2].plot(xs, ys, color=COLORS[order], linewidth=1.0)
    axes[2].set_xlabel("b")
    axes[2].set_ylabel("g_n")
    axes[2].set_xlim(0.0, b_max)
    axes[2].set_ylim(0.0, 1.05)
    _plot_fig3_insets(axes, rows, r_plot_max=r_plot_max)

    for label, axis in zip(["(a)", "(b)", "(c)"], axes):
        axis.text(0.5, -0.25, label, transform=axis.transAxes, ha="center", va="top", fontsize=9)
        axis.tick_params(direction="in", top=True, right=True)

    figure.savefig(output, dpi=220)
    plt.close(figure)


def write_fig4_image(rows: list[dict[str, object]], output: Path, *, image_size: int, image_radius: float) -> None:
    plt = _configure_matplotlib()
    output.parent.mkdir(parents=True, exist_ok=True)
    image = render_axisymmetric_image(rows_to_profile(rows), ImageGrid(width=image_size, height=image_size, radius=image_radius))

    figure, axis = plt.subplots(figsize=(4.6, 4.1))
    vmax = max(float(np.max(image.pixels)), 1e-300)
    im = axis.imshow(
        image.pixels,
        extent=[-image_radius, image_radius, -image_radius, image_radius],
        origin="lower",
        cmap="inferno",
        vmin=0.0,
        vmax=vmax,
    )
    axis.set_aspect("equal")
    axis.set_xlim(-image_radius, image_radius)
    axis.set_ylim(-image_radius, image_radius)
    figure.colorbar(im, ax=axis, fraction=0.046, pad=0.04)
    figure.tight_layout()
    figure.savefig(output, dpi=220)
    plt.close(figure)


def _trajectory_points(ray, *, samples_per_segment: int, plot_radius: float) -> list[tuple[np.ndarray, np.ndarray]]:
    paths: list[tuple[np.ndarray, np.ndarray]] = []
    for segment in ray.segments:
        phis = np.linspace(segment.phi_start, segment.phi_end, samples_per_segment)
        xs: list[float] = []
        ys: list[float] = []
        for phi in phis:
            r = segment.r_at(float(phi))
            if not math.isfinite(r) or r > plot_radius:
                if xs:
                    paths.append((np.array(xs), np.array(ys)))
                    xs = []
                    ys = []
                continue
            xs.append(r * math.cos(float(phi)))
            ys.append(r * math.sin(float(phi)))
        if xs:
            paths.append((np.array(xs), np.array(ys)))
    return paths


def _choose_trajectory_b(rows: list[dict[str, object]], *, shell_crossing: bool, fallback: float) -> float:
    candidates = [
        row
        for row in rows
        if len(row["intersections"]) >= 2 and (int(row["shell_crossing_count"]) > 0) == shell_crossing
    ]
    if not candidates:
        return fallback
    return float(min(candidates, key=lambda row: abs(float(row["b_plus"]) - fallback))["b_plus"])


def write_fig4_trajectories(
    rows: list[dict[str, object]],
    output: Path,
    *,
    solver: StaticJunctionTransferSolver,
    m_minus: float,
    shell_radius: float,
    trajectory_samples: int,
    trajectory_radius: float,
    show_integration_warnings: bool,
) -> None:
    plt = _configure_matplotlib()
    output.parent.mkdir(parents=True, exist_ok=True)

    b_crossing = _choose_trajectory_b(rows, shell_crossing=True, fallback=4.9)
    b_no_crossing = _choose_trajectory_b(rows, shell_crossing=False, fallback=5.35)
    with warnings.catch_warnings():
        if not show_integration_warnings:
            warnings.simplefilter("ignore", IntegrationWarning)
            warnings.filterwarnings("ignore", category=IntegrationWarning)
        ray_crossing = solver.trace_b(b_crossing)
        ray_no_crossing = solver.trace_b(b_no_crossing)

    figure, axis = plt.subplots(figsize=(4.3, 4.3))
    theta = np.linspace(0.0, 2.0 * math.pi, 512)
    axis.plot(shell_radius * np.cos(theta), shell_radius * np.sin(theta), color="tab:blue", linestyle="--", linewidth=1.0)
    horizon = 2.0 * m_minus
    axis.fill(horizon * np.cos(theta), horizon * np.sin(theta), color="black", alpha=0.9)
    axis.axvline(0.0, color="black", linestyle="--", linewidth=0.9)

    for ray, color, label in [
        (ray_no_crossing, "tab:orange", f"b={b_no_crossing:.3f}, no shell crossing"),
        (ray_crossing, "tab:blue", f"b={b_crossing:.3f}, shell crossing"),
    ]:
        with warnings.catch_warnings():
            if not show_integration_warnings:
                warnings.simplefilter("ignore", IntegrationWarning)
                warnings.filterwarnings("ignore", category=IntegrationWarning)
            paths = _trajectory_points(ray, samples_per_segment=trajectory_samples, plot_radius=trajectory_radius)
        first = True
        for x, y in paths:
            axis.plot(x, y, color=color, linewidth=1.4, label=label if first else None)
            first = False

    axis.set_aspect("equal")
    axis.set_xlim(-trajectory_radius, trajectory_radius)
    axis.set_ylim(-trajectory_radius, trajectory_radius)
    axis.legend(loc="upper right", fontsize=7, frameon=False)
    axis.tick_params(direction="in", top=True, right=True)
    figure.tight_layout()
    figure.savefig(output, dpi=220)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate static Schwarzschild-junction reproductions of Fig. 3 and Fig. 4."
    )
    parser.add_argument("--output", required=True, help="JSON table output path")
    parser.add_argument("--fig3-output", help="Optional PNG path for the Fig. 3 three-panel plot")
    parser.add_argument("--fig4-image-output", help="Optional PNG path for the Fig. 4 image-plane panel")
    parser.add_argument("--fig4-trajectories-output", help="Optional PNG path for the Fig. 4 trajectory panel")
    parser.add_argument("--m-minus", type=float, default=0.1)
    parser.add_argument("--m-plus", type=float, default=1.0)
    parser.add_argument("--shell-radius", type=float, default=3.1)
    parser.add_argument("--observer-radius", type=float, default=50.0)
    parser.add_argument("--samples", type=int, default=600)
    parser.add_argument("--b-max", type=float, default=10.0)
    parser.add_argument("--max-order", type=int, default=3)
    parser.add_argument("--disk-r-min", type=float)
    parser.add_argument("--disk-r-max", type=float, default=100.0)
    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument("--image-radius", type=float, default=7.0)
    parser.add_argument("--fig3-r-max", type=float, default=20.0)
    parser.add_argument("--fig3-b-max", type=float, default=15.0)
    parser.add_argument("--trajectory-samples", type=int, default=256)
    parser.add_argument("--trajectory-radius", type=float, default=7.0)
    parser.add_argument("--max-phi", type=float, default=8.0 * math.pi)
    parser.add_argument("--critical-exclusion", type=float, default=0.0)
    parser.add_argument("--show-integration-warnings", action="store_true")
    args = parser.parse_args()

    if args.m_minus <= 0.0 or args.m_plus <= 0.0:
        raise SystemExit("masses must be positive")
    if args.shell_radius <= 0.0 or args.observer_radius <= args.shell_radius:
        raise SystemExit("observer must be outside a positive shell radius")
    if args.samples < 2:
        raise SystemExit("--samples must be at least 2")
    if args.b_max <= 0.0:
        raise SystemExit("--b-max must be positive")
    if args.max_order < 1:
        raise SystemExit("--max-order must be at least 1")
    if args.image_size <= 0:
        raise SystemExit("--image-size must be positive")
    if args.image_radius <= 0.0 or args.trajectory_radius <= 0.0:
        raise SystemExit("plot radii must be positive")
    if args.fig3_r_max <= 0.0:
        raise SystemExit("--fig3-r-max must be positive")
    if args.fig3_b_max <= 0.0:
        raise SystemExit("--fig3-b-max must be positive")
    if args.trajectory_samples < 2:
        raise SystemExit("--trajectory-samples must be at least 2")

    disk_r_min = 2.0 * args.m_minus + 1e-6 if args.disk_r_min is None else args.disk_r_min
    if disk_r_min <= 0.0 or args.disk_r_max <= disk_r_min:
        raise SystemExit("disk window must satisfy 0 < r_min < r_max")

    junction, observer, solver = build_spacetime(
        m_minus=args.m_minus,
        m_plus=args.m_plus,
        shell_radius=args.shell_radius,
        observer_radius=args.observer_radius,
        max_phi=args.max_phi,
        critical_exclusion=args.critical_exclusion,
    )
    source = ThinDiskSource(paper_emissivity_factory(args.m_minus))
    disk = DiskWindow(r_min=disk_r_min, r_max=args.disk_r_max, enabled_regions=frozenset({"inner", "outer"}))
    rows = sample_rows(
        solver=solver,
        junction=junction,
        observer=observer,
        source=source,
        disk=disk,
        b_values=b_grid(
            args.samples,
            max(args.b_max, args.fig3_b_max),
            m_minus=args.m_minus,
            m_plus=args.m_plus,
            shell_radius=args.shell_radius,
        ),
        max_order=args.max_order,
        show_integration_warnings=args.show_integration_warnings,
    )

    write_json(
        rows=rows,
        output=Path(args.output),
        m_minus=args.m_minus,
        m_plus=args.m_plus,
        shell_radius=args.shell_radius,
        observer_radius=args.observer_radius,
        disk_r_min=disk_r_min,
        disk_r_max=args.disk_r_max,
    )
    if args.fig3_output:
        write_fig3(
            rows,
            Path(args.fig3_output),
            b_max=args.b_max,
            fig3_b_max=args.fig3_b_max,
            shell_radius=args.shell_radius,
            r_plot_max=args.fig3_r_max,
        )
    if args.fig4_image_output:
        write_fig4_image(rows, Path(args.fig4_image_output), image_size=args.image_size, image_radius=args.image_radius)
    if args.fig4_trajectories_output:
        write_fig4_trajectories(
            rows,
            Path(args.fig4_trajectories_output),
            solver=solver,
            m_minus=args.m_minus,
            shell_radius=args.shell_radius,
            trajectory_samples=args.trajectory_samples,
            trajectory_radius=args.trajectory_radius,
            show_integration_warnings=args.show_integration_warnings,
        )


if __name__ == "__main__":
    main()
