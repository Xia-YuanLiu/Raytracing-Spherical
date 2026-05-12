from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
import warnings
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import numpy as np
from scipy.integrate import IntegrationWarning

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from spherical_raytracing.junction_analysis import (
    DEFAULT_REPRESENTATIVE_CATEGORIES,
    AtlasCase,
    AtlasParameters,
    PhysicalFilterConfig,
    admit_parameters,
    augment_case_tags_from_profile,
    build_junction,
    category_distance,
    classify_photon_spheres,
    diagnose_profile,
    diagnostic_signal_score,
    paper_style_emissivity_factory,
    select_representative_cases,
)
from spherical_raytracing.imaging import ImageGrid, RadialProfile, SamplingOptions, render_axisymmetric_image, sample_radial_profile
from spherical_raytracing.junctions import annotate_junction_intersections, compare_junction_results
from spherical_raytracing.junction_tracing import StaticJunctionHamiltonianSolver, StaticJunctionTransferSolver
from spherical_raytracing.sources import ThinDiskSource, observed_intensity
from spherical_raytracing.transfer import DiskWindow, compute_intersections
from generate_static_junction_fig3_fig4 import (
    b_grid,
    build_spacetime,
    paper_emissivity_factory,
    sample_rows,
    write_fig3,
    write_fig4_image,
    write_fig4_trajectories,
)


PHYSICAL_FILTER = PhysicalFilterConfig(horizon_buffer=1e-3)
PRIMARY_FIGURE_TAGS = set(DEFAULT_REPRESENTATIVE_CATEGORIES) | {
    "double_sphere_single_ring",
    "single_sphere_double_peak",
}
ATLAS_SCREEN_RADIUS_MIN = 0.0
ATLAS_SCREEN_RADIUS_FLOOR = 8.0
MIN_PHASE_FIGURE_PIXELS = 640
MIN_CASE_IMAGE_PIXELS = 384
PROFILE_DPI = 180
PHASE_MAP_FIELDNAMES = (
    "family",
    "map_name",
    "m_minus",
    "m_plus",
    "q_minus",
    "q_plus",
    "lambda_minus",
    "lambda_plus",
    "shell_radius",
    "observer_radius",
    "admitted",
    "rejection_reasons",
    "warnings",
    "surface_energy_density",
    "surface_pressure",
    "q_jump",
    "lambda_jump",
    "photon_sphere_classification",
    "inner_photon_spheres",
    "outer_photon_spheres",
)


@dataclass(frozen=True)
class PresetMetadata:
    name: str
    default_grid_size: int
    default_samples: int
    default_max_order: int
    default_image_size: int
    per_category_limit: int
    description: str


@dataclass(frozen=True)
class PhaseMapSpec:
    name: str
    family: str
    x_field: str
    y_field: str
    x_values: Callable[[int], np.ndarray]
    y_values: Callable[[int], np.ndarray]
    parameters: Callable[[float, float], AtlasParameters]


PRESETS: dict[str, PresetMetadata] = {
    "quick": PresetMetadata(
        name="quick",
        default_grid_size=4,
        default_samples=6,
        default_max_order=2,
        default_image_size=24,
        per_category_limit=1,
        description="Smoke-test atlas preset with only RN mass/shell and RNdS lambda/shell phase maps.",
    ),
    "paper": PresetMetadata(
        name="paper",
        default_grid_size=16,
        default_samples=128,
        default_max_order=4,
        default_image_size=384,
        per_category_limit=2,
        description="Report-scale phase-map and representative-case preset for report drafts.",
    ),
    "dense": PresetMetadata(
        name="dense",
        default_grid_size=16,
        default_samples=64,
        default_max_order=5,
        default_image_size=192,
        per_category_limit=3,
        description="Dense phase-map and representative-case preset for larger atlas runs.",
    ),
    "schwarzschild-reference": PresetMetadata(
        name="schwarzschild-reference",
        default_grid_size=8,
        default_samples=12,
        default_max_order=3,
        default_image_size=24,
        per_category_limit=1,
        description="Reference Schwarzschild reproduction preset for the static-junction figure path.",
    ),
}


def _linspace(start: float, stop: float, count: int) -> np.ndarray:
    return np.linspace(start, stop, count, dtype=float)


PHASE_MAPS: tuple[PhaseMapSpec, ...] = (
    PhaseMapSpec(
        name="rn_mass_shell",
        family="rn",
        x_field="m_minus",
        y_field="shell_radius",
        x_values=lambda count: _linspace(0.2, 0.95, count),
        y_values=lambda count: _linspace(2.4, 8.0, count),
        parameters=lambda m_minus, shell_radius: AtlasParameters(
            family="rn",
            m_minus=float(m_minus),
            m_plus=1.0,
            q_minus=0.2,
            q_plus=0.2,
            shell_radius=float(shell_radius),
            observer_radius=50.0,
        ),
    ),
    PhaseMapSpec(
        name="rnds_lambda_shell",
        family="rnds",
        x_field="lambda_plus",
        y_field="shell_radius",
        x_values=lambda count: _linspace(0.0, 0.02, count),
        y_values=lambda count: _linspace(3.0, 8.0, count),
        parameters=lambda lambda_plus, shell_radius: AtlasParameters(
            family="rnds",
            m_minus=0.8,
            m_plus=1.0,
            q_minus=0.2,
            q_plus=0.2,
            lambda_minus=0.01,
            lambda_plus=float(lambda_plus),
            shell_radius=float(shell_radius),
            observer_radius=10.0,
        ),
    ),
)


def _configure_matplotlib():
    os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "spherical_raytracing_matplotlib"))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def _reference_profile_args(args: argparse.Namespace) -> argparse.Namespace:
    return argparse.Namespace(
        m_minus=0.1,
        m_plus=1.0,
        shell_radius=3.1,
        observer_radius=50.0,
        samples=args.samples,
        b_max=10.0,
        max_order=args.max_order,
        disk_r_min=None,
        disk_r_max=100.0,
        image_size=args.image_size,
        image_radius=7.0,
        fig3_r_max=20.0,
        fig3_b_max=15.0,
        trajectory_samples=256,
        trajectory_radius=7.0,
        max_phi=8.0 * math.pi,
        critical_exclusion=0.0,
        show_integration_warnings=False,
    )


def _json_ready(value: object) -> object:
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, tuple):
        return [_json_ready(item) for item in value]
    if isinstance(value, list):
        return [_json_ready(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    return value


def _write_json(payload: dict[str, object], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(_json_ready(payload), indent=2, sort_keys=True))


def _format_sequence(values: Sequence[object]) -> str:
    return ";".join(str(value) for value in values)


def _format_float_sequence(values: Sequence[float]) -> str:
    return ";".join(f"{float(value):.12g}" for value in values)


def _emissivity_modes(value: str) -> tuple[str, ...]:
    normalized = value.replace("-", "_")
    if normalized == "both":
        return ("paper", "inverse_square")
    if normalized in {"paper", "inverse_square"}:
        return (normalized,)
    raise ValueError(f"unsupported emissivity mode {value!r}")


def _case_id(parameters: AtlasParameters) -> str:
    payload = json.dumps(asdict(parameters), sort_keys=True, separators=(",", ":"))
    digest = hashlib.sha1(payload.encode("utf-8")).hexdigest()[:12]
    return f"{parameters.family}_{digest}"


def _parameters_from_row(row: dict[str, object]) -> AtlasParameters:
    return AtlasParameters(
        family=str(row["family"]),
        m_minus=float(row["m_minus"]),
        m_plus=float(row["m_plus"]),
        q_minus=float(row["q_minus"]),
        q_plus=float(row["q_plus"]),
        lambda_minus=float(row["lambda_minus"]),
        lambda_plus=float(row["lambda_plus"]),
        shell_radius=float(row["shell_radius"]),
        observer_radius=float(row["observer_radius"]),
    )


def _rows_to_bplus_radial_profile(rows: list[dict[str, object]]) -> RadialProfile:
    return RadialProfile(
        coordinates=np.array([float(row["b_plus"]) for row in rows], dtype=float),
        intensities=np.array([float(row["intensity"]) for row in rows], dtype=float),
        diagnostics={"sample_count": len(rows)},
    )


def _case_tags_from_row(row: dict[str, object], parameters: AtlasParameters) -> tuple[str, ...]:
    tags = []
    classification = str(row.get("photon_sphere_classification", "none"))
    if classification in DEFAULT_REPRESENTATIVE_CATEGORIES:
        tags.append(classification)
    if max(abs(parameters.q_minus), abs(parameters.q_plus)) / max(abs(parameters.m_plus), 1e-300) >= 0.4:
        tags.append("high_charge")
    if abs(float(row.get("q_jump") or 0.0)) >= 0.05:
        tags.append("large_q_jump")
    if abs(float(row.get("lambda_jump") or 0.0)) >= 0.002:
        tags.append("large_lambda_jump")
    if (
        abs(parameters.m_minus - parameters.m_plus) < 0.05
        and abs(parameters.q_minus) < 0.05
        and abs(parameters.q_plus) < 0.05
        and abs(parameters.lambda_minus) < 0.001
        and abs(parameters.lambda_plus) < 0.001
    ):
        tags.append("near_schwarzschild")
    return tuple(dict.fromkeys(tags))


def _cases_from_phase_rows(rows: list[dict[str, object]]) -> list[AtlasCase]:
    cases = []
    for row in rows:
        if not bool(row["admitted"]):
            continue
        parameters = _parameters_from_row(row)
        tags = _case_tags_from_row(row, parameters)
        if not tags:
            continue
        cases.append(
            AtlasCase(
                case_id=_case_id(parameters),
                family=parameters.family,
                parameters=parameters,
                category_tags=tags,
                category_distances={
                    category: category_distance(parameters, category)
                    for category in DEFAULT_REPRESENTATIVE_CATEGORIES
                },
                signal_score=diagnostic_signal_score(row),
            )
        )
    return cases


def _row_for_parameters(map_name: str, parameters: AtlasParameters) -> dict[str, object]:
    admission = admit_parameters(parameters, PHYSICAL_FILTER)
    photon_spheres = classify_photon_spheres(parameters)
    row = asdict(parameters)
    row.update(
        {
            "map_name": map_name,
            "admitted": admission.admitted,
            "rejection_reasons": _format_sequence(admission.rejection_reasons),
            "warnings": _format_sequence(admission.warnings),
            "surface_energy_density": admission.surface_energy_density,
            "surface_pressure": admission.surface_pressure,
            "q_jump": admission.q_jump,
            "lambda_jump": admission.lambda_jump,
            "photon_sphere_classification": photon_spheres.classification,
            "inner_photon_spheres": _format_float_sequence(photon_spheres.inner_radii),
            "outer_photon_spheres": _format_float_sequence(photon_spheres.outer_radii),
        }
    )
    return row


def _write_phase_csv(rows: list[dict[str, object]], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=PHASE_MAP_FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)


def _write_phase_png(rows: list[dict[str, object]], spec: PhaseMapSpec, output: Path, image_size: int) -> None:
    if image_size <= 0:
        raise ValueError("image_size must be positive")
    plt = _configure_matplotlib()
    output.parent.mkdir(parents=True, exist_ok=True)

    colors = {
        "none": "#8b8b8b",
        "inner_only": "#1f77b4",
        "outer_only": "#2ca02c",
        "double": "#d62728",
    }
    markers = {True: "o", False: "x"}
    figure_pixels = max(int(image_size), MIN_PHASE_FIGURE_PIXELS)
    dpi = 160
    figure_inches = figure_pixels / dpi
    figure, axis = plt.subplots(figsize=(figure_inches, figure_inches), dpi=dpi)
    for admitted in (True, False):
        subset = [row for row in rows if bool(row["admitted"]) is admitted]
        for classification, color in colors.items():
            classified = [row for row in subset if row["photon_sphere_classification"] == classification]
            if not classified:
                continue
            axis.scatter(
                [float(row[spec.x_field]) for row in classified],
                [float(row[spec.y_field]) for row in classified],
                c=color,
                marker=markers[admitted],
                s=36 if admitted else 42,
                linewidths=1.0,
                label=f"{classification}{'' if admitted else ' rejected'}",
            )
    axis.set_xlabel(spec.x_field)
    axis.set_ylabel(spec.y_field)
    axis.set_title(spec.name)
    axis.grid(True, alpha=0.25, linewidth=0.6)
    if rows:
        axis.legend(fontsize=7, loc="best", framealpha=0.9)
    figure.tight_layout(pad=0.55)
    figure.savefig(output)
    plt.close(figure)


def write_reference_profile_json(
    rows: list[dict[str, object]],
    output: Path,
    *,
    m_minus: float,
    m_plus: float,
    shell_radius: float,
    observer_radius: float,
) -> None:
    payload = {
        "model": "schwarzschild_static_junction_reference",
        "figure": "fig3_fig4",
        "parameters": {
            "m_minus": m_minus,
            "m_plus": m_plus,
            "shell_radius": shell_radius,
            "observer_radius": observer_radius,
        },
        "profile_diagnostics": asdict(diagnose_profile(rows, shell_radius=shell_radius)),
        "samples": rows,
    }
    output.write_text(json.dumps(_json_ready(payload), indent=2, sort_keys=True))


def write_reference_fig3_fig4(output_dir: Path, args: argparse.Namespace) -> dict[str, object]:
    ref_args = _reference_profile_args(args)
    junction, observer, solver = build_spacetime(
        m_minus=ref_args.m_minus,
        m_plus=ref_args.m_plus,
        shell_radius=ref_args.shell_radius,
        observer_radius=ref_args.observer_radius,
        max_phi=ref_args.max_phi,
        critical_exclusion=ref_args.critical_exclusion,
    )
    source = ThinDiskSource(paper_emissivity_factory(ref_args.m_minus))
    disk = DiskWindow(r_min=2.0 * ref_args.m_minus + 1e-6, r_max=ref_args.disk_r_max, enabled_regions=frozenset({"inner", "outer"}))
    rows = sample_rows(
        solver=solver,
        junction=junction,
        observer=observer,
        source=source,
        disk=disk,
        b_values=b_grid(
            ref_args.samples,
            max(ref_args.b_max, ref_args.fig3_b_max),
            m_minus=ref_args.m_minus,
            m_plus=ref_args.m_plus,
            shell_radius=ref_args.shell_radius,
        ),
        max_order=ref_args.max_order,
        show_integration_warnings=ref_args.show_integration_warnings,
    )
    write_reference_profile_json(
        rows,
        output_dir / "fig3_profile.json",
        m_minus=ref_args.m_minus,
        m_plus=ref_args.m_plus,
        shell_radius=ref_args.shell_radius,
        observer_radius=ref_args.observer_radius,
    )
    write_fig3(
        rows,
        output_dir / "fig3_profile.png",
        b_max=ref_args.b_max,
        fig3_b_max=ref_args.fig3_b_max,
        shell_radius=ref_args.shell_radius,
        r_plot_max=ref_args.image_radius,
    )
    write_fig4_image(rows, output_dir / "fig4_image.png", image_size=ref_args.image_size, image_radius=ref_args.image_radius)
    write_fig4_trajectories(
        rows,
        output_dir / "fig4_trajectories.png",
        solver=solver,
        m_minus=ref_args.m_minus,
        shell_radius=ref_args.shell_radius,
        trajectory_samples=ref_args.trajectory_samples,
        trajectory_radius=ref_args.trajectory_radius,
        show_integration_warnings=ref_args.show_integration_warnings,
    )
    return {
        "fig3_profile_json": str((output_dir / "fig3_profile.json").relative_to(output_dir.parent)),
        "fig3_profile_png": str((output_dir / "fig3_profile.png").relative_to(output_dir.parent)),
        "fig4_image_png": str((output_dir / "fig4_image.png").relative_to(output_dir.parent)),
        "fig4_trajectories_png": str((output_dir / "fig4_trajectories.png").relative_to(output_dir.parent)),
    }


def reference_sweep_cases(
    *,
    m_minus: float,
    radii: Sequence[float],
    args: argparse.Namespace,
) -> list[dict[str, object]]:
    cases: list[dict[str, object]] = []
    for shell_radius in radii:
        ref_args = argparse.Namespace(
            m_minus=m_minus,
            m_plus=1.0,
            shell_radius=float(shell_radius),
            observer_radius=50.0,
            samples=args.samples,
            b_max=10.0,
            max_order=args.max_order,
            disk_r_min=None,
            disk_r_max=100.0,
            image_size=args.image_size,
            image_radius=7.0,
            fig3_r_max=20.0,
            fig3_b_max=15.0,
            trajectory_samples=256,
            trajectory_radius=7.0,
            max_phi=8.0 * math.pi,
            critical_exclusion=0.0,
            show_integration_warnings=False,
        )
        junction, observer, solver = build_spacetime(
            m_minus=ref_args.m_minus,
            m_plus=ref_args.m_plus,
            shell_radius=ref_args.shell_radius,
            observer_radius=ref_args.observer_radius,
            max_phi=ref_args.max_phi,
            critical_exclusion=ref_args.critical_exclusion,
        )
        source = ThinDiskSource(paper_emissivity_factory(ref_args.m_minus))
        disk = DiskWindow(r_min=2.0 * ref_args.m_minus + 1e-6, r_max=ref_args.disk_r_max, enabled_regions=frozenset({"inner", "outer"}))
        rows = sample_rows(
            solver=solver,
            junction=junction,
            observer=observer,
            source=source,
            disk=disk,
            b_values=b_grid(
                ref_args.samples,
                max(ref_args.b_max, ref_args.fig3_b_max),
                m_minus=ref_args.m_minus,
                m_plus=ref_args.m_plus,
                shell_radius=ref_args.shell_radius,
            ),
            max_order=ref_args.max_order,
            show_integration_warnings=ref_args.show_integration_warnings,
        )
        cases.append(
            {
                "m_minus": m_minus,
                "shell_radius": float(shell_radius),
                "rows": rows,
                "profile_diagnostics": asdict(diagnose_profile(rows, shell_radius=float(shell_radius))),
            }
        )
    return cases


def _write_reference_sweep_plot(cases: list[dict[str, object]], output: Path, *, title: str) -> None:
    plt = _configure_matplotlib()
    output.parent.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(1, len(cases), figsize=(3.0 * len(cases), 2.4), constrained_layout=True)
    if len(cases) == 1:
        axes = [axes]
    for axis, case in zip(axes, cases):
        rows = list(case["rows"])
        axis.plot([float(row["b_plus"]) for row in rows], [float(row["intensity"]) for row in rows], color="black", linewidth=1.0)
        axis.set_title(f"r={float(case['shell_radius']):.1f}")
        axis.set_xlabel("b")
        axis.set_ylabel("I_obs")
        axis.grid(True, alpha=0.2, linewidth=0.5)
    figure.suptitle(title)
    figure.savefig(output, dpi=220)
    plt.close(figure)


def _write_reference_sweep_images(cases: list[dict[str, object]], output: Path, *, image_size: int) -> None:
    plt = _configure_matplotlib()
    output.parent.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(1, len(cases), figsize=(2.2 * len(cases), 2.2), constrained_layout=True)
    if len(cases) == 1:
        axes = [axes]
    for axis, case in zip(axes, cases):
        rows = list(case["rows"])
        radial_profile = _rows_to_bplus_radial_profile(rows)
        radius = max(float(np.max(radial_profile.coordinates)), 1.0)
        image = render_axisymmetric_image(radial_profile, ImageGrid(width=image_size, height=image_size, radius=radius))
        vmax = max(float(np.max(image.pixels)), 1e-300)
        axis.imshow(image.pixels, origin="lower", cmap="inferno", vmin=0.0, vmax=vmax)
        axis.set_title(f"r={float(case['shell_radius']):.1f}")
        axis.set_axis_off()
    figure.savefig(output, dpi=220)
    plt.close(figure)


def write_reference_sweep(output_dir: Path, args: argparse.Namespace, *, name: str, m_minus: float, radii: Sequence[float]) -> dict[str, object]:
    cases = reference_sweep_cases(m_minus=m_minus, radii=radii, args=args)
    json_path = output_dir / f"{name}_sweep.json"
    png_path = output_dir / f"{name}_sweep.png"
    images_path = output_dir / ("fig6_images.png" if name == "fig5" else "fig8_images.png")
    payload = {
        "name": name,
        "m_minus": m_minus,
        "cases": cases,
    }
    _write_json(payload, json_path)
    _write_reference_sweep_plot(cases, png_path, title=name)
    _write_reference_sweep_images(cases, images_path, image_size=args.image_size)
    root_dir = output_dir.parent
    return {
        json_path.name: str(json_path.relative_to(root_dir)),
        png_path.name: str(png_path.relative_to(root_dir)),
        images_path.name: str(images_path.relative_to(root_dir)),
    }


def _schwarzschild_reference_args(args: argparse.Namespace, preset: PresetMetadata) -> argparse.Namespace:
    return argparse.Namespace(
        output_dir=args.output_dir,
        samples=args.samples if args.samples is not None else preset.default_samples,
        max_order=args.max_order if args.max_order is not None else preset.default_max_order,
        image_size=args.image_size if args.image_size is not None else preset.default_image_size,
        grid_size=args.grid_size if args.grid_size is not None else preset.default_grid_size,
    )


def run_schwarzschild_reference(args: argparse.Namespace) -> dict[str, object]:
    output_dir = Path(args.output_dir) / "schwarzschild_reference"
    output_dir.mkdir(parents=True, exist_ok=True)
    fig3 = write_reference_fig3_fig4(output_dir, args)
    fig5 = write_reference_sweep(output_dir, args, name="fig5", m_minus=0.1, radii=[5.0, 3.1, 2.9, 2.1])
    fig7 = write_reference_sweep(output_dir, args, name="fig7", m_minus=0.8, radii=[5.0, 3.1, 2.9, 2.6])
    manifest = {
        "preset": "schwarzschild-reference",
        "families": ["schwarzschild"],
        "reference_artifacts": {**fig3, **fig5, **fig7},
        "figure_index": {},
        "selected_cases": [],
        "selected_case_summaries": [],
    }
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    (Path(args.output_dir) / "manifest.json").write_text(json.dumps(_json_ready(manifest), indent=2, sort_keys=True))
    return manifest


def write_schwarzschild_reference_artifacts(args: argparse.Namespace) -> dict[str, object]:
    output_dir = Path(args.output_dir) / "schwarzschild_reference"
    output_dir.mkdir(parents=True, exist_ok=True)
    fig3 = write_reference_fig3_fig4(output_dir, args)
    fig5 = write_reference_sweep(output_dir, args, name="fig5", m_minus=0.1, radii=[5.0, 3.1, 2.9, 2.1])
    fig7 = write_reference_sweep(output_dir, args, name="fig7", m_minus=0.8, radii=[5.0, 3.1, 2.9, 2.6])
    return {**fig3, **fig5, **fig7}


def _screen_radius_from_impact_parameter(observer, impact_parameter: float) -> float:
    sine = impact_parameter * math.sqrt(observer.metric.A(observer.r_obs)) / observer.r_obs
    alpha = math.asin(min(max(sine, 0.0), 0.999999999))
    return observer.screen_radius(alpha)


def _impact_parameter_from_screen_radius(observer, rho: float) -> float:
    alpha = math.atan(float(rho) / observer.r_obs)
    return observer.impact_parameter(alpha)


def _mapped_inner_critical_screen_radii(parameters: AtlasParameters, *, junction, observer) -> list[float]:
    radii = []
    shell = parameters.shell_radius
    outer_A = junction.outer_metric.A(shell)
    if outer_A <= 0.0:
        return radii
    for curve in junction.inner_metric.critical_curves():
        inner_A = junction.inner_metric.A(shell)
        if inner_A <= 0.0:
            continue
        mapped_b_plus = curve.b_crit * math.sqrt(inner_A / outer_A)
        try:
            rho = _screen_radius_from_impact_parameter(observer, mapped_b_plus)
        except (ValueError, ZeroDivisionError):
            continue
        if math.isfinite(rho):
            radii.append(rho)
    return radii


def _atlas_screen_radius_max(parameters: AtlasParameters, *, junction, observer) -> float:
    candidate_radii = [parameters.shell_radius]
    for curve in [*junction.inner_metric.critical_curves(), *junction.outer_metric.critical_curves()]:
        try:
            candidate_radii.append(_screen_radius_from_impact_parameter(observer, curve.b_crit))
        except (ValueError, ZeroDivisionError):
            continue
    candidate_radii.extend(_mapped_inner_critical_screen_radii(parameters, junction=junction, observer=observer))
    return max(ATLAS_SCREEN_RADIUS_FLOOR, *[value for value in candidate_radii if math.isfinite(value)])


def _atlas_screen_critical_points(*, junction, observer, rho_max: float) -> list[float]:
    points = [ATLAS_SCREEN_RADIUS_MIN]
    for curve in [*junction.inner_metric.critical_curves(), *junction.outer_metric.critical_curves()]:
        try:
            rho = _screen_radius_from_impact_parameter(observer, curve.b_crit)
        except (ValueError, ZeroDivisionError):
            continue
        if ATLAS_SCREEN_RADIUS_MIN < rho < rho_max:
            points.append(rho)
    return sorted(set(points))


def _atlas_focus_screen_points(parameters: AtlasParameters, *, junction, observer, rho_max: float, samples: int) -> list[float]:
    centers = list(_atlas_screen_critical_points(junction=junction, observer=observer, rho_max=rho_max))
    centers.extend(_mapped_inner_critical_screen_radii(parameters, junction=junction, observer=observer))
    centers = [center for center in centers if ATLAS_SCREEN_RADIUS_MIN < center < rho_max]
    points: list[float] = []
    for center in centers:
        width = max(0.08 * max(center, 1.0), 0.12)
        local_count = max(samples // 2, 64)
        points.extend(np.linspace(max(ATLAS_SCREEN_RADIUS_MIN, center - width), min(rho_max, center + width), local_count))
        offsets = np.geomspace(1e-5, min(width, 0.25), max(samples // 3, 48))
        points.extend(center - offsets)
        points.extend(center + offsets)
    return sorted({float(point) for point in points if ATLAS_SCREEN_RADIUS_MIN <= float(point) <= rho_max})


def _inverse_square_emissivity(r: float, region: str | None = None) -> float:
    return 1.0 / max(r**2, 1e-300)


def _trace_profile(
    parameters: AtlasParameters,
    *,
    emissivity_name: str,
    samples: int,
    max_order: int,
    compare_backends: bool,
) -> dict[str, object]:
    junction, observer = build_junction(parameters)
    transfer_solver = StaticJunctionTransferSolver(junction=junction, observer=observer)
    comparison_solver = StaticJunctionHamiltonianSolver(junction=junction, observer=observer) if compare_backends else None
    if emissivity_name == "paper":
        emissivity = paper_style_emissivity_factory(junction.inner_metric, shell_radius=parameters.shell_radius)
    elif emissivity_name == "inverse_square":
        emissivity = _inverse_square_emissivity
    else:
        raise ValueError(f"unsupported emissivity mode {emissivity_name!r}")
    source = ThinDiskSource(intensity_profile=emissivity)
    disk = DiskWindow(r_min=0.0, r_max=max(parameters.observer_radius * 2.0, 100.0), enabled_regions=frozenset({"inner", "outer"}))
    rho_max = _atlas_screen_radius_max(parameters, junction=junction, observer=observer)
    mapped_inner_critical_points = _mapped_inner_critical_screen_radii(parameters, junction=junction, observer=observer)
    critical_points = _atlas_screen_critical_points(junction=junction, observer=observer, rho_max=rho_max)
    focus_points = _atlas_focus_screen_points(parameters, junction=junction, observer=observer, rho_max=rho_max, samples=samples)
    rows_by_rho: dict[float, dict[str, object]] = {}

    def intensity_at(rho: float):
        rho = float(rho)
        if rho <= 0.0:
            rows_by_rho[rho] = {
                "rho": rho,
                "b_plus": 0.0,
                "intensity": 0.0,
                "termination_reason": "axis",
                "shell_crossing_count": 0,
                "intersections": [],
            }
            return 0.0

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", IntegrationWarning)
            warnings.filterwarnings("ignore", category=IntegrationWarning)
            ray = transfer_solver.trace_screen_radius(rho)
            intersections = compute_intersections(ray, disk, max_order=max_order)
            annotated = annotate_junction_intersections(
                intersections,
                ray,
                observer,
                {"inner": junction.inner_metric, "outer": junction.outer_metric},
            )
            backend_comparison = None
            if comparison_solver is not None:
                comparison_ray = comparison_solver.trace_screen_radius(rho)
                comparison_intersections = compute_intersections(comparison_ray, disk, max_order=max_order)
                comparison_annotated = annotate_junction_intersections(
                    comparison_intersections,
                    comparison_ray,
                    observer,
                    {"inner": junction.inner_metric, "outer": junction.outer_metric},
                )
                ray = replace(ray, intersections=annotated)
                comparison_ray = replace(comparison_ray, intersections=comparison_annotated)
                backend_comparison = compare_junction_results(ray, comparison_ray)
        observed = observed_intensity(
            annotated,
            source,
            junction.outer_metric,
            observer,
            region_metrics={"inner": junction.inner_metric, "outer": junction.outer_metric},
        )
        diagnostics = dict(observed.diagnostics)
        diagnostics["termination_reason"] = ray.termination_reason
        diagnostics["shell_crossing_count"] = len(ray.shell_crossings)
        diagnostics["backend_disagreement"] = 0.0 if backend_comparison is None else float(backend_comparison["backend_disagreement"])
        observed = replace(observed, diagnostics=diagnostics)
        row = {
            "rho": rho,
            "b_plus": float(ray.b_plus),
            "intensity": float(observed.total),
            "termination_reason": ray.termination_reason,
            "shell_crossing_count": len(ray.shell_crossings),
            "intersections": [
                {
                    "m": int(item.m),
                    "r": float(item.r),
                    "phi": float(item.phi),
                    "region": item.region,
                    "path_class": item.path_class,
                    "g": float(item.diagnostics.get("g", 0.0)),
                }
                for item in annotated
            ],
        }
        if backend_comparison is not None:
            row["backend_comparison"] = backend_comparison
        rows_by_rho[rho] = row
        return observed

    profile = sample_radial_profile(
        ATLAS_SCREEN_RADIUS_MIN,
        rho_max,
        intensity_at,
        critical_points=[*critical_points, *focus_points],
        options=SamplingOptions(initial_count=samples, max_refine=2 if samples >= 32 else 0),
    )
    rows_with_critical = [rows_by_rho[float(rho)] for rho in profile.coordinates]
    rows = [row for row in rows_with_critical if row.get("termination_reason") != "near_critical"]
    diagnostics = diagnose_profile(rows, shell_radius=parameters.shell_radius)
    sampling_diagnostics = dict(profile.diagnostics)
    sampling_diagnostics["dropped_near_critical_samples"] = len(rows_with_critical) - len(rows)
    sampling_diagnostics["rho_max"] = float(rho_max)
    sampling_diagnostics["screen_critical_points"] = [float(point) for point in critical_points]
    sampling_diagnostics["mapped_inner_critical_screen_radii"] = [
        float(point) for point in mapped_inner_critical_points
    ]
    sampling_diagnostics["focus_point_count"] = len(focus_points)
    return {
        "parameters": asdict(parameters),
        "emissivity": emissivity_name,
        "samples": rows,
        "profile_diagnostics": asdict(diagnostics),
        "sampling_diagnostics": sampling_diagnostics,
        "backend_comparison_summary": _backend_comparison_summary(compare_backends, rows),
    }


def _backend_comparison_summary(compare_backends: bool, rows: Sequence[dict[str, object]]) -> dict[str, object]:
    if not compare_backends:
        return {
            "enabled": False,
            "backend_disagreement_intervals": 0,
        }
    disagreement_count = 0
    for row in rows:
        comparison = row.get("backend_comparison")
        if not isinstance(comparison, dict):
            continue
        if float(comparison.get("backend_disagreement", 0.0)) > 0.0:
            disagreement_count += 1
    return {
        "enabled": True,
        "backend_disagreement_intervals": disagreement_count,
    }


def _write_profile_csv(profile: dict[str, object], output: Path) -> None:
    rows = list(profile["samples"])
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["rho", "b_plus", "intensity", "termination_reason", "shell_crossing_count"],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row[field] for field in writer.fieldnames})


def _rows_to_radial_profile(rows: list[dict[str, object]]) -> RadialProfile:
    return RadialProfile(
        coordinates=np.array([float(row["rho"]) for row in rows], dtype=float),
        intensities=np.array([float(row["intensity"]) for row in rows], dtype=float),
        diagnostics={"sample_count": len(rows)},
    )


def _high_order_screen_window(rows: list[dict[str, object]]) -> tuple[float, float] | None:
    all_x = [float(row["rho"]) for row in rows if "rho" in row]
    if len(all_x) < 3:
        return None
    high_order_x = []
    for row in rows:
        intersections = row.get("intersections", [])
        if not isinstance(intersections, Sequence):
            continue
        for item in intersections:
            if not isinstance(item, dict):
                continue
            if int(item.get("m", 0)) >= 2:
                high_order_x.append(float(row["rho"]))
                break
    if len(high_order_x) < 3:
        return None
    x_min = min(high_order_x)
    x_max = max(high_order_x)
    full_min = min(all_x)
    full_max = max(all_x)
    full_span = full_max - full_min
    if x_max <= x_min or full_span <= 0.0:
        return None
    if (x_max - x_min) / full_span > 0.25:
        return None
    pad = max((x_max - x_min) * 0.18, 1e-3)
    return max(full_min, x_min - pad), min(full_max, x_max + pad)


def _write_profile_png(profile: dict[str, object], output: Path) -> None:
    rows = list(profile["samples"])
    plt = _configure_matplotlib()
    from mpl_toolkits.axes_grid1.inset_locator import inset_axes

    output.parent.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(6.0, 4.0), dpi=PROFILE_DPI)
    xs = [float(row["rho"]) for row in rows]
    ys = [float(row["intensity"]) for row in rows]
    axis.plot(xs, ys, color="black", linewidth=1.0)
    axis.set_xlabel("rho")
    axis.set_ylabel("I_obs")
    axis.grid(True, alpha=0.2, linewidth=0.5)
    focus_window = _high_order_screen_window(rows)
    if focus_window is not None:
        x_left, x_right = focus_window
        inset_rows = [row for row in rows if x_left <= float(row["rho"]) <= x_right]
        if len(inset_rows) >= 3:
            inset_x = [float(row["rho"]) for row in inset_rows]
            inset_y = [float(row["intensity"]) for row in inset_rows]
            inset = inset_axes(axis, width="42%", height="42%", loc="upper right", borderpad=1.0)
            inset.plot(inset_x, inset_y, color="black", linewidth=0.8)
            inset.scatter(inset_x, inset_y, s=3.0, color="black", alpha=0.35)
            inset.set_xlim(x_left, x_right)
            y_min = min(inset_y)
            y_max = max(inset_y)
            y_pad = max((y_max - y_min) * 0.12, 1e-3)
            inset.set_ylim(y_min - y_pad, y_max + y_pad)
            inset.tick_params(labelsize=5, direction="in", top=True, right=True)
            inset.grid(True, alpha=0.15, linewidth=0.4)
            inset.set_title("m>=2 window", fontsize=6)
    figure.subplots_adjust(left=0.08, right=0.98, bottom=0.15, top=0.96, wspace=0.24)
    figure.savefig(output)
    plt.close(figure)


def _write_image_png(profile: dict[str, object], output: Path, *, image_size: int) -> None:
    rows = list(profile["samples"])
    radial_profile = _rows_to_radial_profile(rows)
    radius = max(float(np.max(radial_profile.coordinates)), 1.0)
    image = render_axisymmetric_image(radial_profile, ImageGrid(width=image_size, height=image_size, radius=radius))
    vmax = max(float(np.max(image.pixels)), 1e-300)
    plt = _configure_matplotlib()
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas_pixels = max(int(image_size), MIN_CASE_IMAGE_PIXELS)
    dpi = 160
    figure, axis = plt.subplots(figsize=(canvas_pixels / dpi, canvas_pixels / dpi), dpi=dpi)
    axis.imshow(image.pixels, origin="lower", cmap="afmhot", vmin=0.0, vmax=vmax, interpolation="bilinear")
    axis.set_axis_off()
    figure.subplots_adjust(left=0.0, right=1.0, bottom=0.0, top=1.0)
    figure.savefig(output, facecolor="black", pad_inches=0.0)
    plt.close(figure)


def _write_transfer_redshift_png(profile: dict[str, object], output: Path) -> None:
    rows = list(profile["samples"])
    plt = _configure_matplotlib()
    from mpl_toolkits.axes_grid1.inset_locator import inset_axes

    output.parent.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(1, 2, figsize=(8.0, 3.2), dpi=PROFILE_DPI)
    values_by_order: dict[int, tuple[list[float], list[float], list[float]]] = {}
    for order in range(1, 5):
        xs = []
        radii = []
        redshifts = []
        for row in rows:
            match = next((item for item in row["intersections"] if int(item["m"]) == order), None)
            if match is None:
                continue
            xs.append(float(row["rho"]))
            radii.append(float(match["r"]))
            redshifts.append(float(match.get("g", 0.0)))
        if xs:
            values_by_order[order] = (xs, radii, redshifts)
            axes[0].plot(xs, radii, linewidth=0.8, label=f"m={order}")
            axes[1].plot(xs, redshifts, linewidth=0.8, label=f"m={order}")
    parameters = dict(profile.get("parameters", {}))
    shell_radius = parameters.get("shell_radius")
    if shell_radius is not None:
        axes[0].axhline(float(shell_radius), color="0.65", linestyle=":", linewidth=0.8)
    axes[0].set_xlabel("rho")
    axes[0].set_ylabel("r")
    axes[1].set_xlabel("rho")
    axes[1].set_ylabel("g")
    for axis in axes:
        axis.grid(True, alpha=0.2, linewidth=0.5)
        if axis.lines:
            axis.legend(fontsize=6, loc="best")
    high_order_xs = [
        x
        for order, (xs, _radii, _redshifts) in values_by_order.items()
        if order >= 2
        for x in xs
    ]
    if len(high_order_xs) >= 3:
        x_min = min(high_order_xs)
        x_max = max(high_order_xs)
        pad = max((x_max - x_min) * 0.15, 1e-3)
        xlim = (x_min - pad, x_max + pad)
        inset_specs = [(axes[0], 1), (axes[1], 2)]
        for axis, value_index in inset_specs:
            inset = inset_axes(axis, width="42%", height="42%", loc="upper center", borderpad=1.0)
            plotted_values: list[float] = []
            for order, (xs, radii, redshifts) in values_by_order.items():
                if order < 2:
                    continue
                values = radii if value_index == 1 else redshifts
                inset.plot(xs, values, linewidth=0.7)
                plotted_values.extend(value for x, value in zip(xs, values) if xlim[0] <= x <= xlim[1])
            inset.set_xlim(*xlim)
            if plotted_values:
                y_min = min(plotted_values)
                y_max = max(plotted_values)
                y_pad = max((y_max - y_min) * 0.12, 1e-3)
                inset.set_ylim(y_min - y_pad, y_max + y_pad)
            inset.tick_params(labelsize=5, direction="in", top=True, right=True)
            inset.grid(True, alpha=0.15, linewidth=0.4)
    figure.subplots_adjust(left=0.08, right=0.98, bottom=0.15, top=0.96, wspace=0.24)
    figure.savefig(output)
    plt.close(figure)


def _write_case_artifacts(
    case: AtlasCase,
    output_dir: Path,
    *,
    emissivity_modes: tuple[str, ...],
    compare_backends: bool,
    image_size: int,
    sample_count: int,
    max_order: int,
) -> dict[str, object]:
    case_dir = output_dir / "cases" / case.case_id
    case_dir.mkdir(parents=True, exist_ok=True)
    profile_payloads = {}
    for emissivity_name in emissivity_modes:
        print(f"case trace: {case.case_id} ({emissivity_name})", file=sys.stderr)
        profile = _trace_profile(
            case.parameters,
            emissivity_name=emissivity_name,
            samples=sample_count,
            max_order=max_order,
            compare_backends=compare_backends and emissivity_name == "paper",
        )
        _write_json(profile, case_dir / f"profile_{emissivity_name}.json")
        _write_profile_csv(profile, case_dir / f"profile_{emissivity_name}.csv")
        _write_profile_png(profile, case_dir / f"profile_{emissivity_name}.png")
        _write_image_png(profile, case_dir / f"image_{emissivity_name}.png", image_size=image_size)
        profile_payloads[emissivity_name] = profile

    primary_mode = "paper" if "paper" in profile_payloads else emissivity_modes[0]
    primary_profile = profile_payloads[primary_mode]
    _write_transfer_redshift_png(primary_profile, case_dir / "transfer_redshift.png")
    final_case = augment_case_tags_from_profile(case, primary_profile)
    figure_index = {
        f"fig_case_{final_case.case_id}_{tag}.png": {
            "path": f"figures/fig_case_{final_case.case_id}_{tag}.png",
            "case_id": final_case.case_id,
            "case_dir": str(case_dir.relative_to(output_dir)),
            "tag": tag,
            "kind": "representative_case",
        }
        for tag in final_case.category_tags
        if tag in PRIMARY_FIGURE_TAGS
    }
    summary = {
        "case_id": final_case.case_id,
        "case_dir": str(case_dir.relative_to(output_dir)),
        "family": final_case.family,
        "parameters": asdict(final_case.parameters),
        "category_tags": list(final_case.category_tags),
        "category_distances": final_case.category_distances,
        "signal_score": final_case.signal_score,
        "primary_emissivity": primary_mode,
        "emissivity_modes": list(emissivity_modes),
        "profile_diagnostics": primary_profile["profile_diagnostics"],
        "backend_comparison_summary": primary_profile["backend_comparison_summary"],
        "figure_index": figure_index,
    }
    _write_json(summary, case_dir / "diagnostics.json")
    return summary


def _atlas_case_from_summary(summary: dict[str, object]) -> AtlasCase:
    parameters = AtlasParameters(**summary["parameters"])
    return AtlasCase(
        case_id=str(summary["case_id"]),
        family=str(summary["family"]),
        parameters=parameters,
        category_tags=tuple(str(tag) for tag in summary["category_tags"]),
        category_distances={str(key): float(value) for key, value in summary["category_distances"].items()},
        signal_score=float(summary["signal_score"]),
    )


def _finalize_selected_case_summaries(
    seed_summaries: list[dict[str, object]],
    output_dir: Path,
    *,
    per_category_limit: int,
) -> tuple[list[dict[str, object]], tuple[str, ...], dict[str, dict[str, object]]]:
    cases = [_atlas_case_from_summary(summary) for summary in seed_summaries]
    selected_cases, empty_categories = select_representative_cases(
        cases,
        per_category_limit=per_category_limit,
        return_empty=True,
    )
    by_id = {str(summary["case_id"]): summary for summary in seed_summaries}
    selected_summaries = [by_id[case.case_id] for case in selected_cases]
    figure_index = {}
    figures_dir = output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    for summary in selected_summaries:
        source = output_dir / str(summary["case_dir"]) / f"image_{summary['primary_emissivity']}.png"
        for figure_name, record in summary["figure_index"].items():
            target = figures_dir / figure_name
            shutil.copyfile(source, target)
            figure_index[figure_name] = record
    return selected_summaries, empty_categories, figure_index


def _generate_phase_map(
    spec: PhaseMapSpec,
    output_dir: Path,
    grid_size: int,
    image_size: int,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    print(f"phase map: {spec.name} ({grid_size}x{grid_size})", file=sys.stderr)
    x_values = spec.x_values(grid_size)
    y_values = spec.y_values(grid_size)
    rows = [
        _row_for_parameters(spec.name, spec.parameters(float(x_value), float(y_value)))
        for y_value in y_values
        for x_value in x_values
    ]
    csv_path = output_dir / "phase_maps" / f"{spec.name}.csv"
    png_path = output_dir / "phase_maps" / f"{spec.name}.png"
    _write_phase_csv(rows, csv_path)
    _write_phase_png(rows, spec, png_path, image_size)

    admitted_count = sum(1 for row in rows if row["admitted"])
    classifications: dict[str, int] = {}
    for row in rows:
        classification = str(row["photon_sphere_classification"])
        classifications[classification] = classifications.get(classification, 0) + 1
    return {
        "name": spec.name,
        "family": spec.family,
        "csv": str(csv_path.relative_to(output_dir)),
        "png": str(png_path.relative_to(output_dir)),
        "x_field": spec.x_field,
        "y_field": spec.y_field,
        "grid_size": grid_size,
        "row_count": len(rows),
        "admitted_count": admitted_count,
        "rejected_count": len(rows) - admitted_count,
        "photon_sphere_classifications": dict(sorted(classifications.items())),
    }, rows


def _git_commit() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    value = result.stdout.strip()
    return value or None


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate static-junction atlas phase maps and manifest.")
    parser.add_argument("--families", nargs="+", default=["rn", "rnds"], choices=["rn", "rnds", "schwarzschild"])
    parser.add_argument("--preset", default="quick", choices=sorted(PRESETS))
    parser.add_argument("--output-dir", default="outputs/junction_atlas")
    parser.add_argument("--grid-size", type=int)
    parser.add_argument("--samples", type=int)
    parser.add_argument("--max-order", type=int)
    parser.add_argument("--image-size", type=int)
    parser.add_argument("--compare-backends", action="store_true")
    parser.add_argument("--include-schwarzschild-reference", action="store_true")
    parser.add_argument("--emissivity", default="paper", choices=["paper", "inverse_square", "inverse-square", "both"])
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    preset = PRESETS[args.preset]
    if args.preset == "schwarzschild-reference":
        if args.include_schwarzschild_reference:
            raise SystemExit("--include-schwarzschild-reference cannot be combined with --preset schwarzschild-reference")
        if args.grid_size is not None and args.grid_size <= 0:
            raise SystemExit("--grid-size must be positive")
        if args.samples is not None and args.samples < 2:
            raise SystemExit("--samples must be at least 2")
        if args.max_order is not None and args.max_order <= 0:
            raise SystemExit("--max-order must be positive")
        if args.image_size is not None and args.image_size <= 0:
            raise SystemExit("--image-size must be positive")
        resolved_args = argparse.Namespace(
            output_dir=args.output_dir,
            samples=args.samples if args.samples is not None else preset.default_samples,
            max_order=args.max_order if args.max_order is not None else preset.default_max_order,
            image_size=args.image_size if args.image_size is not None else preset.default_image_size,
            grid_size=args.grid_size if args.grid_size is not None else preset.default_grid_size,
        )
        run_schwarzschild_reference(resolved_args)
        return
    if args.grid_size is not None and args.grid_size <= 0:
        raise SystemExit("--grid-size must be positive")
    grid_size = args.grid_size if args.grid_size is not None else preset.default_grid_size
    samples = args.samples if args.samples is not None else preset.default_samples
    max_order = args.max_order if args.max_order is not None else preset.default_max_order
    image_size = args.image_size if args.image_size is not None else preset.default_image_size

    if samples < 2:
        raise SystemExit("--samples must be at least 2")
    if max_order <= 0:
        raise SystemExit("--max-order must be positive")
    if image_size <= 0:
        raise SystemExit("--image-size must be positive")

    families = tuple(dict.fromkeys(args.families))
    output_dir = Path(args.output_dir)
    phase_dir = output_dir / "phase_maps"
    figures_dir = output_dir / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)
    phase_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    emissivity_modes = _emissivity_modes(args.emissivity)
    phase_summaries = []
    phase_rows_by_map = {}
    figure_index: dict[str, dict[str, str]] = {}
    for spec in PHASE_MAPS:
        if spec.family not in families:
            continue
        summary, rows = _generate_phase_map(spec, output_dir, grid_size, image_size)
        phase_summaries.append(summary)
        phase_rows_by_map[spec.name] = rows
        stable_figure = figures_dir / f"fig_{spec.name}_phase.png"
        shutil.copyfile(output_dir / summary["png"], stable_figure)
        stable_name = stable_figure.name
        figure_index[stable_name] = {
            "path": str(stable_figure.relative_to(output_dir)),
            "case_id": "",
            "case_dir": "",
            "tag": spec.name,
            "kind": "phase_map",
        }

    if not phase_summaries:
        raise SystemExit(f"preset {args.preset!r} has no Task 5 phase maps for families: {', '.join(families)}")

    candidates_by_id: dict[str, AtlasCase] = {}
    for rows in phase_rows_by_map.values():
        for candidate in _cases_from_phase_rows(rows):
            existing = candidates_by_id.get(candidate.case_id)
            if existing is None:
                candidates_by_id[candidate.case_id] = candidate
            else:
                merged_tags = tuple(sorted(set(existing.category_tags) | set(candidate.category_tags)))
                merged_distances = {
                    category: min(
                        existing.category_distances.get(category, float("inf")),
                        candidate.category_distances.get(category, float("inf")),
                    )
                    for category in DEFAULT_REPRESENTATIVE_CATEGORIES
                }
                candidates_by_id[candidate.case_id] = replace(
                    existing,
                    category_tags=merged_tags,
                    category_distances=merged_distances,
                    signal_score=max(existing.signal_score, candidate.signal_score),
                )
    per_category_limit = preset.per_category_limit
    seed_limit = max(per_category_limit * 2, per_category_limit + 1)
    seed_cases, seed_empty = select_representative_cases(
        candidates_by_id.values(),
        per_category_limit=seed_limit,
        return_empty=True,
    )
    seed_summaries = []
    skipped_cases = []
    for case in seed_cases:
        try:
            seed_summaries.append(
                _write_case_artifacts(
                    case,
                    output_dir,
                    emissivity_modes=emissivity_modes,
                    compare_backends=bool(args.compare_backends),
                    image_size=image_size,
                    sample_count=samples,
                    max_order=max_order,
                )
            )
        except (ValueError, RuntimeError, FloatingPointError, OverflowError) as exc:
            print(f"skipped case {case.case_id}: {exc}", file=sys.stderr)
            skipped_cases.append({"case_id": case.case_id, "reason": str(exc)})

    if seed_cases and not seed_summaries:
        raise SystemExit("representative tracing failed for all selected seed cases")

    selected_case_summaries, empty_categories, case_figure_index = _finalize_selected_case_summaries(
        seed_summaries,
        output_dir,
        per_category_limit=per_category_limit,
    )
    figure_index.update(case_figure_index)
    selected_cases = [str(summary["case_id"]) for summary in selected_case_summaries]
    backend_performed = any(
        bool(summary["backend_comparison_summary"].get("enabled", False))
        for summary in selected_case_summaries
    )
    backend_disagreement_count = sum(
        int(summary["backend_comparison_summary"].get("backend_disagreement_intervals", 0))
        for summary in selected_case_summaries
    )
    reference_artifacts = {}
    families_for_manifest = list(families)
    if args.include_schwarzschild_reference:
        reference_artifacts = write_schwarzschild_reference_artifacts(
            _schwarzschild_reference_args(args, PRESETS["schwarzschild-reference"])
        )
        families_for_manifest = list(dict.fromkeys([*families_for_manifest, "schwarzschild"]))

    manifest = {
        "preset": args.preset,
        "preset_metadata": asdict(preset),
        "families": families_for_manifest,
        "grid_size": grid_size,
        "samples": samples,
        "max_order": max_order,
        "image_size": image_size,
        "emissivity": args.emissivity.replace("-", "_"),
        "physical_filter": asdict(PHYSICAL_FILTER),
        "phase_map_summaries": phase_summaries,
        "figure_index": figure_index,
        "selected_cases": selected_cases,
        "selected_case_summaries": selected_case_summaries,
        "reference_artifacts": reference_artifacts,
        "skipped_representative_cases": skipped_cases,
        "seed_empty_representative_categories": list(seed_empty),
        "empty_representative_categories": list(empty_categories),
        "backend_comparison_summary": {
            "requested": bool(args.compare_backends),
            "enabled": backend_performed,
            "case_count": len(selected_case_summaries),
            "backend_disagreement_intervals": backend_disagreement_count,
        },
        "git_commit": _git_commit(),
    }
    (output_dir / "manifest.json").write_text(json.dumps(_json_ready(manifest), indent=2, sort_keys=True))
    print(f"wrote manifest: {output_dir / 'manifest.json'}", file=sys.stderr)


if __name__ == "__main__":
    main()
