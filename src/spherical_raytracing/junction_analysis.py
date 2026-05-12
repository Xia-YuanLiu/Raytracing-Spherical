from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, replace

from .junctions import StaticJunctionSpacetime, StaticShell, clean_rnds_static_patch
from .metrics import (
    ReissnerNordstromDeSitterMetric,
    ReissnerNordstromMetric,
    SchwarzschildMetric,
    StaticSphericalMetric,
)
from .observers import FiniteStaticObserver
from .policies import SolverOptions


@dataclass(frozen=True)
class AtlasParameters:
    family: str
    m_minus: float
    m_plus: float
    q_minus: float = 0.0
    q_plus: float = 0.0
    lambda_minus: float = 0.0
    lambda_plus: float = 0.0
    shell_radius: float = 5.0
    observer_radius: float = 50.0


@dataclass(frozen=True)
class PhysicalFilterConfig:
    horizon_buffer: float = 1e-3
    require_clean_rnds_patch: bool = True


@dataclass(frozen=True)
class AdmissionResult:
    admitted: bool
    rejection_reasons: tuple[str, ...]
    warnings: tuple[str, ...]
    surface_energy_density: float | None
    surface_pressure: float | None
    q_jump: float | None
    lambda_jump: float | None


@dataclass(frozen=True)
class PhotonSphereSummary:
    inner_radii: tuple[float, ...]
    outer_radii: tuple[float, ...]
    visible_regions: tuple[str, ...]
    classification: str


@dataclass(frozen=True)
class ProfilePeak:
    x: float
    y: float
    prominence: float
    index: int


@dataclass(frozen=True)
class ProfileDiagnostics:
    peak_count: int
    peaks: tuple[ProfilePeak, ...]
    redshift_cusp_orders: tuple[int, ...]
    v_shape_orders: tuple[int, ...]
    shell_crossing_count_max: int


@dataclass(frozen=True)
class AtlasCase:
    case_id: str
    family: str
    parameters: AtlasParameters
    category_tags: tuple[str, ...]
    category_distances: dict[str, float]
    signal_score: float


DEFAULT_REPRESENTATIVE_CATEGORIES = (
    "inner_only",
    "outer_only",
    "double",
    "double_sphere_single_ring",
    "single_sphere_double_peak",
    "shell_near_outer_photon_sphere",
    "high_charge",
    "rnds_observer_near_cosmological_patch",
    "large_q_jump",
    "large_lambda_jump",
    "near_schwarzschild",
)


CATEGORY_ANCHORS: dict[str, AtlasParameters] = {
    "inner_only": AtlasParameters(family="rn", m_minus=0.8, m_plus=1.0, q_minus=0.1, q_plus=0.2, shell_radius=2.4, observer_radius=50.0),
    "outer_only": AtlasParameters(family="rn", m_minus=0.8, m_plus=1.0, q_minus=0.1, q_plus=0.2, shell_radius=6.0, observer_radius=50.0),
    "double": AtlasParameters(family="rn", m_minus=0.8, m_plus=1.0, q_minus=0.1, q_plus=0.2, shell_radius=3.1, observer_radius=50.0),
    "double_sphere_single_ring": AtlasParameters(family="rn", m_minus=0.8, m_plus=1.0, q_minus=0.1, q_plus=0.2, shell_radius=3.1, observer_radius=50.0),
    "single_sphere_double_peak": AtlasParameters(family="rn", m_minus=0.8, m_plus=1.0, q_minus=0.1, q_plus=0.2, shell_radius=2.9, observer_radius=50.0),
    "shell_near_outer_photon_sphere": AtlasParameters(family="rn", m_minus=0.8, m_plus=1.0, q_minus=0.1, q_plus=0.2, shell_radius=5.0, observer_radius=50.0),
    "high_charge": AtlasParameters(family="rn", m_minus=0.8, m_plus=1.0, q_minus=0.2, q_plus=0.95, shell_radius=3.1, observer_radius=50.0),
    "rnds_observer_near_cosmological_patch": AtlasParameters(family="rnds", m_minus=0.8, m_plus=1.0, q_minus=0.2, q_plus=0.2, lambda_minus=0.01, lambda_plus=0.01, shell_radius=5.0, observer_radius=6.0),
    "large_q_jump": AtlasParameters(family="rn", m_minus=0.8, m_plus=1.0, q_minus=0.0, q_plus=0.8, shell_radius=3.1, observer_radius=50.0),
    "large_lambda_jump": AtlasParameters(family="rnds", m_minus=0.8, m_plus=1.0, q_minus=0.2, q_plus=0.2, lambda_minus=0.0, lambda_plus=0.02, shell_radius=5.0, observer_radius=6.0),
    "near_schwarzschild": AtlasParameters(family="schwarzschild", m_minus=0.1, m_plus=1.0, shell_radius=3.1, observer_radius=50.0),
}


DIAGNOSTIC_Q_JUMP_WEIGHT = 10.0
DIAGNOSTIC_LAMBDA_JUMP_WEIGHT = 200.0


def atlas_parameter_sort_key(parameters: AtlasParameters) -> tuple[object, ...]:
    return (
        parameters.family,
        parameters.m_minus,
        parameters.m_plus,
        parameters.q_minus,
        parameters.q_plus,
        parameters.lambda_minus,
        parameters.lambda_plus,
        parameters.shell_radius,
        parameters.observer_radius,
    )


def normalized_parameter_distance(parameters: AtlasParameters, anchor: AtlasParameters) -> float:
    family_penalty = 0.0 if parameters.family == anchor.family else 10.0
    values = [
        (parameters.m_minus - anchor.m_minus) / max(abs(anchor.m_minus), 1.0),
        (parameters.m_plus - anchor.m_plus) / max(abs(anchor.m_plus), 1.0),
        (parameters.q_minus - anchor.q_minus) / max(abs(anchor.q_minus), 1.0),
        (parameters.q_plus - anchor.q_plus) / max(abs(anchor.q_plus), 1.0),
        (parameters.lambda_minus - anchor.lambda_minus) / max(abs(anchor.lambda_minus), 1e-3),
        (parameters.lambda_plus - anchor.lambda_plus) / max(abs(anchor.lambda_plus), 1e-3),
        (parameters.shell_radius - anchor.shell_radius) / max(abs(anchor.shell_radius), 1.0),
        (parameters.observer_radius - anchor.observer_radius) / max(abs(anchor.observer_radius), 1.0),
    ]
    return family_penalty + math.sqrt(sum(value * value for value in values))


def category_distance(parameters: AtlasParameters, category: str) -> float:
    return normalized_parameter_distance(parameters, CATEGORY_ANCHORS[category])


def diagnostic_signal_score(row: Mapping[str, object]) -> float:
    warnings_value = row.get("warnings", "")
    if isinstance(warnings_value, str):
        warning_count = len([item for item in warnings_value.split(";") if item])
    elif isinstance(warnings_value, Sequence):
        warning_count = len([item for item in warnings_value if item])
    else:
        warning_count = 0
    q_jump = abs(float(row.get("q_jump") or 0.0))
    lambda_jump = abs(float(row.get("lambda_jump") or 0.0))
    sphere_score = {
        "none": 0.0,
        "inner_only": 1.0,
        "outer_only": 1.0,
        "double": 2.0,
    }.get(str(row.get("photon_sphere_classification")), 0.0)
    return sphere_score + warning_count + DIAGNOSTIC_Q_JUMP_WEIGHT * q_jump + DIAGNOSTIC_LAMBDA_JUMP_WEIGHT * lambda_jump


def merge_atlas_cases(left: AtlasCase, right: AtlasCase) -> AtlasCase:
    if left.case_id != right.case_id:
        raise ValueError("can only merge atlas cases with the same case_id")
    if left.family != right.family:
        raise ValueError(f"case_id {left.case_id!r} maps to multiple families")
    if atlas_parameter_sort_key(left.parameters) != atlas_parameter_sort_key(right.parameters):
        raise ValueError(f"case_id {left.case_id!r} maps to multiple parameter tuples")
    distances = dict(left.category_distances)
    for category, distance in right.category_distances.items():
        distances[category] = min(float(distance), float(distances.get(category, distance)))
    return replace(
        left,
        category_tags=tuple(sorted(set(left.category_tags) | set(right.category_tags))),
        category_distances=distances,
        signal_score=max(left.signal_score, right.signal_score),
    )


def augment_case_tags_from_profile(case: AtlasCase, profile: Mapping[str, object]) -> AtlasCase:
    tags = set(case.category_tags)
    diagnostics = profile["profile_diagnostics"]
    if not isinstance(diagnostics, Mapping):
        raise ValueError("profile_diagnostics must be a mapping")
    peak_count = int(diagnostics["peak_count"])
    samples = profile["samples"]
    if "double" in tags and peak_count == 1:
        tags.add("double_sphere_single_ring")
    if tags.intersection({"inner_only", "outer_only"}) and peak_count >= 2:
        tags.add("single_sphere_double_peak")
    if isinstance(samples, Sequence):
        for row in samples:
            if not isinstance(row, Mapping):
                continue
            intersections = row.get("intersections", [])
            if not isinstance(intersections, Sequence):
                continue
            if any(
                isinstance(item, Mapping) and abs(float(item["r"]) - case.parameters.shell_radius) <= 1e-3
                for item in intersections
            ):
                tags.add("shell_near_outer_photon_sphere")
                break
    if (
        case.family == "rnds"
        and case.parameters.lambda_plus > 0.0
        and diagnostics.get("redshift_cusp_orders")
    ):
        tags.add("rnds_observer_near_cosmological_patch")
    return replace(case, category_tags=tuple(sorted(tags)))


def select_representative_cases(
    candidates: Iterable[AtlasCase],
    *,
    per_category_limit: int,
    required_categories: Sequence[str] = DEFAULT_REPRESENTATIVE_CATEGORIES,
    return_empty: bool = False,
):
    if per_category_limit <= 0:
        raise ValueError("per_category_limit must be positive")
    by_category: dict[str, list[AtlasCase]] = {category: [] for category in required_categories}
    merged_candidates: dict[str, AtlasCase] = {}
    for candidate in candidates:
        existing = merged_candidates.get(candidate.case_id)
        merged_candidates[candidate.case_id] = candidate if existing is None else merge_atlas_cases(existing, candidate)
    for candidate in merged_candidates.values():
        for tag in candidate.category_tags:
            if tag in by_category:
                by_category[tag].append(candidate)

    selected_by_id: dict[str, AtlasCase] = {}
    empty = []
    for category in required_categories:
        ranked = sorted(
            by_category[category],
            key=lambda item: (
                item.category_distances.get(category, float("inf")),
                -item.signal_score,
                atlas_parameter_sort_key(item.parameters),
                item.case_id,
            ),
        )
        selected_for_category = 0
        seen_for_category: set[str] = set()
        for candidate in ranked:
            if candidate.case_id in seen_for_category:
                continue
            seen_for_category.add(candidate.case_id)
            existing = selected_by_id.get(candidate.case_id)
            selected_by_id[candidate.case_id] = candidate if existing is None else merge_atlas_cases(existing, candidate)
            selected_for_category += 1
            if selected_for_category >= per_category_limit:
                break
        if selected_for_category == 0:
            empty.append(category)

    selected = tuple(sorted(selected_by_id.values(), key=lambda item: item.case_id))
    if return_empty:
        return selected, tuple(empty)
    return selected


def build_metrics(parameters: AtlasParameters) -> tuple[StaticSphericalMetric, StaticSphericalMetric]:
    if parameters.family == "schwarzschild":
        inner = SchwarzschildMetric(mass=parameters.m_minus, region="inner")
        outer = SchwarzschildMetric(mass=parameters.m_plus, region="outer")
    elif parameters.family == "rn":
        inner = ReissnerNordstromMetric(mass=parameters.m_minus, charge=parameters.q_minus, region="inner")
        outer = ReissnerNordstromMetric(mass=parameters.m_plus, charge=parameters.q_plus, region="outer")
    elif parameters.family == "rnds":
        inner = ReissnerNordstromDeSitterMetric(
            mass=parameters.m_minus,
            charge=parameters.q_minus,
            cosmological_constant=parameters.lambda_minus,
            region="inner",
        )
        outer = ReissnerNordstromDeSitterMetric(
            mass=parameters.m_plus,
            charge=parameters.q_plus,
            cosmological_constant=parameters.lambda_plus,
            region="outer",
        )
    else:
        raise ValueError(f"unknown atlas family {parameters.family!r}")
    return inner, outer


def build_junction(
    parameters: AtlasParameters,
    *,
    options: SolverOptions | None = None,
) -> tuple[StaticJunctionSpacetime, FiniteStaticObserver]:
    inner, outer = build_metrics(parameters)
    junction = StaticJunctionSpacetime(
        inner_metric=inner,
        outer_metric=outer,
        shell=StaticShell(radius=parameters.shell_radius),
        options=options or SolverOptions(),
    )
    observer = FiniteStaticObserver(r_obs=parameters.observer_radius, metric=outer)
    return junction, observer


def admit_parameters(
    parameters: AtlasParameters,
    config: PhysicalFilterConfig = PhysicalFilterConfig(),
) -> AdmissionResult:
    inner, outer = build_metrics(parameters)
    junction = StaticJunctionSpacetime(
        inner_metric=inner,
        outer_metric=outer,
        shell=StaticShell(radius=parameters.shell_radius),
        options=SolverOptions(horizon_buffer=config.horizon_buffer),
    )

    diagnostics = junction.physics_diagnostics(parameters.observer_radius)
    warnings = list(diagnostics.warnings)
    rejection_reasons: list[str] = []

    observer_static = outer.A(parameters.observer_radius) > 0.0
    if not observer_static:
        rejection_reasons.append("observer_nonstatic")
    if parameters.observer_radius <= parameters.shell_radius:
        rejection_reasons.append("observer_inside_shell")

    for warning in diagnostics.warnings:
        if warning in {
            "inner_shell_radius_nonstatic",
            "outer_shell_radius_nonstatic",
            "observer_nonstatic",
            "inner_naked_singularity",
            "outer_naked_singularity",
        }:
            rejection_reasons.append(warning)

    if config.require_clean_rnds_patch:
        for warning in diagnostics.warnings:
            if warning in {
                "inner_rnds_no_clean_static_patch",
                "outer_rnds_no_clean_static_patch",
            }:
                rejection_reasons.append(warning)

    if parameters.family == "rnds" and config.require_clean_rnds_patch:
        patch = clean_rnds_static_patch(outer)
        if patch is None:
            rejection_reasons.append("outer_rnds_observer_outside_clean_static_patch")
        else:
            lower, upper = patch
            buffer = config.horizon_buffer * max(abs(lower), 1.0)
            if math.isfinite(upper):
                buffer = config.horizon_buffer * max(abs(lower), abs(upper), 1.0)
                inside_clean_patch = lower + buffer < parameters.observer_radius < upper - buffer
            else:
                inside_clean_patch = parameters.observer_radius > lower + buffer
            if not inside_clean_patch:
                rejection_reasons.append("outer_rnds_observer_outside_clean_static_patch")

    if diagnostics.surface_pressure is None or not math.isfinite(diagnostics.surface_pressure):
        rejection_reasons.append("non_finite_surface_pressure")

    q_jump: float | None = None
    if hasattr(junction.inner_metric, "charge") and hasattr(junction.outer_metric, "charge"):
        q_jump = float(junction.outer_metric.charge) - float(junction.inner_metric.charge)

    lambda_jump: float | None = None
    if hasattr(junction.inner_metric, "cosmological_constant") and hasattr(junction.outer_metric, "cosmological_constant"):
        lambda_jump = float(junction.outer_metric.cosmological_constant) - float(junction.inner_metric.cosmological_constant)

    return AdmissionResult(
        admitted=not rejection_reasons,
        rejection_reasons=tuple(dict.fromkeys(rejection_reasons)),
        warnings=tuple(dict.fromkeys(warnings)),
        surface_energy_density=diagnostics.surface_energy_density,
        surface_pressure=diagnostics.surface_pressure,
        q_jump=q_jump,
        lambda_jump=lambda_jump,
    )


def classify_photon_spheres(parameters: AtlasParameters) -> PhotonSphereSummary:
    inner, outer = build_metrics(parameters)
    shell = parameters.shell_radius
    inner_radii = tuple(r for r in inner.photon_spheres() if r < shell)
    outer_radii = tuple(r for r in outer.photon_spheres() if r > shell)
    visible = []
    if inner_radii:
        visible.append("inner")
    if outer_radii:
        visible.append("outer")
    if visible == ["inner"]:
        classification = "inner_only"
    elif visible == ["outer"]:
        classification = "outer_only"
    elif visible == ["inner", "outer"]:
        classification = "double"
    else:
        classification = "none"
    return PhotonSphereSummary(
        inner_radii=inner_radii,
        outer_radii=outer_radii,
        visible_regions=tuple(visible),
        classification=classification,
    )


def detect_peaks(xs: Sequence[float], ys: Sequence[float], *, min_prominence: float = 0.0) -> tuple[ProfilePeak, ...]:
    if len(xs) != len(ys):
        raise ValueError("xs and ys must have the same length")
    peaks = []
    for index in range(1, len(xs) - 1):
        left = float(ys[index - 1])
        center = float(ys[index])
        right = float(ys[index + 1])
        if center > left and center > right:
            prominence = center - max(left, right)
            if prominence >= min_prominence:
                peaks.append(ProfilePeak(float(xs[index]), center, prominence, index))
    return tuple(peaks)


def _profile_coordinate(row: Mapping[str, object]) -> float:
    if "rho" in row:
        return float(row["rho"])
    return float(row["b_plus"])


def _profile_ordinate(row: Mapping[str, object]) -> float:
    if "intensity" in row:
        return float(row["intensity"])
    g_values = [
        float(item.get("g", 0.0))
        for item in row.get("intersections", [])
        if isinstance(item, Mapping)
    ]
    return max(g_values, default=0.0)


def diagnose_profile(rows: Sequence[Mapping[str, object]], *, shell_radius: float) -> ProfileDiagnostics:
    xs = [_profile_coordinate(row) for row in rows]
    ys = [_profile_ordinate(row) for row in rows]
    peaks = detect_peaks(xs, ys, min_prominence=0.0)

    by_order: dict[int, list[tuple[float, float, float]]] = {}
    for row in rows:
        for item in row.get("intersections", []):
            order = int(item["m"])
            by_order.setdefault(order, []).append(
                (_profile_coordinate(row), float(item["r"]), float(item.get("g", 0.0)))
            )

    cusp_orders = []
    v_orders = []
    for order, values in by_order.items():
        values.sort()
        radii = [value[1] for value in values]
        if any(abs(radius - shell_radius) <= 1e-8 for radius in radii):
            cusp_orders.append(order)
        v_shape_prominence = 1e-3 * max(shell_radius, 1.0)
        for index in range(1, len(radii) - 1):
            is_local_extremum = (radii[index] < radii[index - 1] and radii[index] < radii[index + 1]) or (
                radii[index] > radii[index - 1] and radii[index] > radii[index + 1]
            )
            if (
                is_local_extremum
                and min(abs(radii[index] - radii[index - 1]), abs(radii[index] - radii[index + 1]))
                >= v_shape_prominence
            ):
                v_orders.append(order)
                break

    return ProfileDiagnostics(
        peak_count=len(peaks),
        peaks=peaks,
        redshift_cusp_orders=tuple(sorted(cusp_orders)),
        v_shape_orders=tuple(sorted(v_orders)),
        shell_crossing_count_max=max((int(row.get("shell_crossing_count", 0)) for row in rows), default=0),
    )


def paper_style_inner_edge(metric: StaticSphericalMetric, *, shell_radius: float) -> float:
    if hasattr(metric, "cosmological_constant") and float(metric.cosmological_constant) > 0.0:
        patch = clean_rnds_static_patch(metric)
        if patch is None:
            raise ValueError("paper emissivity requires at least one positive horizon or static patch edge")
        left, right = patch
        if left < shell_radius < right:
            return float(left)

    horizons = metric.horizons()
    if not horizons:
        raise ValueError("paper emissivity requires at least one positive horizon or static patch edge")

    for left, right in metric.static_domains():
        if left > 0.0 and left < shell_radius < right:
            return float(left)

    below_shell = [horizon for horizon in horizons if horizon < shell_radius]
    if below_shell:
        return float(max(below_shell))
    return float(horizons[0])


def paper_style_emissivity_factory(metric: StaticSphericalMetric, *, shell_radius: float):
    inner_edge = paper_style_inner_edge(metric, shell_radius=shell_radius)

    def emissivity(r: float, region: str | None = None) -> float:
        if r < inner_edge:
            return 0.0
        return 1.0 / max((r - inner_edge + 1.0) ** 3, 1e-300)

    return emissivity
