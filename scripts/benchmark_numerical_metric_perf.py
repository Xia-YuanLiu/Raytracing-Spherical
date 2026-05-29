from __future__ import annotations

import argparse
import math
import platform
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import scipy

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from spherical_raytracing import (
    CanonicalPayload,
    FiniteStaticObserver,
    LiDMBHMetadata,
    QuadTransferSolver,
    ReissnerNordstromDeSitterMetric,
    SchwarzschildMetric,
    SolverOptions,
    StaticDomainSelector,
    TabulatedMetric,
    load_metric_npz,
)


class ReferenceQuadTransferSolver(QuadTransferSolver):
    def _first_turning_point(self, b: float, u_start: float, u_stop: float) -> float | None:
        return self._first_turning_point_scalar(b, u_start, u_stop)

    def _make_segment(
        self,
        b: float,
        u0: float,
        u1: float,
        phi0: float,
        direction: str,
        endpoint_event,
        region: str,
    ):
        return self._make_segment_quad_reference(
            b=b,
            u0=u0,
            u1=u1,
            phi0=phi0,
            direction=direction,
            endpoint_event=endpoint_event,
            region=region,
        )


def percentile(values: list[float], q: float) -> float:
    if not values:
        return math.nan
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, math.ceil(q * len(ordered)) - 1))
    return ordered[index]


def summarize_times(times: list[float]) -> str:
    if not times:
        return "ray_count=0"
    median_ms = 1e3 * percentile(times, 0.50)
    p95_ms = 1e3 * percentile(times, 0.95)
    return f"ray_count={len(times)} median={median_ms:.3f} ms p95={p95_ms:.3f} ms"


def sample_metric(analytical, *, n_points: int = 2400) -> TabulatedMetric:
    r_lo, r_hi = analytical.valid_radial_domain()
    grid_hi = r_hi if math.isfinite(r_hi) else 200.0
    r_start = r_lo + max(abs(r_lo), 1.0) * 1e-5
    r_stop = grid_hi - max(abs(grid_hi), 1.0) * 1e-5
    r = np.geomspace(r_start, r_stop, n_points)
    A = np.array([analytical.A(float(value)) for value in r])
    C = A.copy()
    static_domains = [
        [lo, None if not math.isfinite(hi) else hi]
        for lo, hi in analytical.static_domains()
    ]
    metadata = LiDMBHMetadata.from_params(
        params={
            "model": type(analytical).__name__,
            "static_domains": static_domains,
            "horizons": analytical.horizons(),
            "photon_spheres": analytical.photon_spheres(),
            "g_convention": "g_rr_inverse",
            "B_convention": "g_rr",
        },
        source_format="li-dm-bh",
        original_keys=("r", "f", "g", "B", "params"),
        raw_B=1.0 / C,
    )
    payload = CanonicalPayload(r=r, A=A, C=C, metadata=metadata)
    return TabulatedMetric(payload=payload, static_domain=(r_lo, r_hi))


def make_real_metric(path: Path) -> TabulatedMetric | None:
    if not path.exists():
        return None
    payload = load_metric_npz(path)
    r_lo, r_hi = StaticDomainSelector().choose(payload)
    return TabulatedMetric(payload=payload, static_domain=(r_lo, r_hi))


def observer_radius(metric: TabulatedMetric) -> float:
    r_lo, r_hi = metric.valid_radial_domain()
    r_grid_max = float(metric._r_grid_max)
    if math.isfinite(r_hi):
        return 0.5 * (r_lo + r_hi)
    return min(max(4.0 * r_lo, 30.0), 0.5 * r_grid_max)


def candidate_b_values(metric: TabulatedMetric, observer: FiniteStaticObserver) -> dict[str, list[float]]:
    curves = [curve for curve in metric.critical_curves() if curve.reachable_from_observer]
    if not curves:
        return {}
    b_crit = curves[-1].b_crit
    b_max = observer.r_obs / math.sqrt(metric.A(observer.r_obs))
    turning_hi = min(1.40 * b_crit, 0.85 * b_max)
    near_hi = min(1.001 * b_crit, 0.98 * b_max)
    return {
        "captured": [0.45 * b_crit, 0.70 * b_crit, 0.95 * b_crit],
        "turning": [1.03 * b_crit, 1.12 * b_crit, turning_hi],
        "near_critical": [1.0001 * b_crit, near_hi],
    }


def probe_segments(result) -> int:
    probe_count = 0
    for segment in result.segments:
        span = segment.phi_end - segment.phi_start
        if span <= 0.0:
            continue
        for fraction in (0.25, 0.50, 0.75):
            phi = segment.phi_start + fraction * span
            segment.u_at(phi)
            probe_count += 1
    return probe_count


def time_category(
    solver: QuadTransferSolver,
    category: str,
    b_values: list[float],
    *,
    repeats: int,
    include_u_at: bool,
    solver_label: str,
) -> tuple[list[float], Counter]:
    timings: list[float] = []
    counts: Counter = Counter()
    for _ in range(repeats):
        for b in b_values:
            start = time.perf_counter()
            result = solver.trace_b(float(b))
            probe_count = probe_segments(result) if include_u_at else 0
            timings.append(time.perf_counter() - start)
            counts[f"termination:{result.diagnostics.termination_reason}"] += 1
            counts[f"turning_points:{result.diagnostics.turning_point_count}"] += 1
            counts[f"category:{category}"] += 1
            counts[f"solver:{solver_label}"] += 1
            counts["u_at_probes"] += probe_count
    return timings, counts


def benchmark_metric(
    label: str,
    metric: TabulatedMetric,
    *,
    repeats: int,
    include_u_at: bool,
    solver_cls=QuadTransferSolver,
    solver_label: str = "fast",
    display_solver_label: bool = False,
) -> None:
    r_obs = observer_radius(metric)
    observer = FiniteStaticObserver(r_obs=r_obs, metric=metric)
    solver = solver_cls(
        metric=metric,
        observer=observer,
        options=SolverOptions(critical_exclusion=0.0),
    )
    b_by_category = candidate_b_values(metric, observer)
    if not b_by_category:
        print(f"\n[{label}] skipped: no reachable critical curve")
        return

    display_label = f"{label}:{solver_label}" if display_solver_label else label
    print(f"\n[{display_label}] r_obs={r_obs:.6g} repeats={repeats} include_u_at={include_u_at}")
    total_counts: Counter = Counter()
    all_timings: list[float] = []
    for category, b_values in b_by_category.items():
        timings, counts = time_category(
            solver,
            category,
            b_values,
            repeats=repeats,
            include_u_at=include_u_at,
            solver_label=solver_label,
        )
        all_timings.extend(timings)
        total_counts.update(counts)
        print(f"  {category:14s} {summarize_times(timings)}")
    print(f"  {'all':14s} {summarize_times(all_timings)}")
    for key, value in sorted(total_counts.items()):
        print(f"    {key}={value}")
    fast_stats = getattr(solver, "_fast_segment_stats", None)
    if fast_stats:
        attempted = fast_stats.get("attempted", 0)
        accepted = fast_stats.get("accepted", 0)
        fallback = fast_stats.get("fallback", 0)
        late_fallback = fast_stats.get("late_fallback", 0)
        print(f"    fast_segments={accepted}/{attempted} accepted, fallback={fallback}, late_fallback={late_fallback}")


def benchmark_bounded_outward(
    *,
    repeats: int,
    include_u_at: bool,
    solver_cls=QuadTransferSolver,
    solver_label: str = "fast",
    display_solver_label: bool = False,
) -> None:
    metric = sample_metric(
        ReissnerNordstromDeSitterMetric(
            mass=1.0,
            charge=0.0,
            cosmological_constant=0.01,
        ),
        n_points=2600,
    )
    observer = FiniteStaticObserver(r_obs=6.0, metric=metric)
    solver = solver_cls(
        metric=metric,
        observer=observer,
        options=SolverOptions(critical_exclusion=0.0),
    )
    b_crit = metric.critical_curves()[0].b_crit
    b_values = [1.03 * b_crit, 1.05 * b_crit, 1.12 * b_crit]
    timings, counts = time_category(
        solver,
        "bounded_outward",
        b_values,
        repeats=repeats,
        include_u_at=include_u_at,
        solver_label=solver_label,
    )
    label = "manufactured_rnds_bounded_outward"
    display_label = f"{label}:{solver_label}" if display_solver_label else label
    print(f"\n[{display_label}] {summarize_times(timings)}")
    for key, value in sorted(counts.items()):
        print(f"    {key}={value}")
    fast_stats = getattr(solver, "_fast_segment_stats", None)
    if fast_stats:
        attempted = fast_stats.get("attempted", 0)
        accepted = fast_stats.get("accepted", 0)
        fallback = fast_stats.get("fallback", 0)
        late_fallback = fast_stats.get("late_fallback", 0)
        print(f"    fast_segments={accepted}/{attempted} accepted, fallback={fallback}, late_fallback={late_fallback}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Lightweight benchmark for TabulatedMetric QuadTransferSolver tracing.",
    )
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument(
        "--real-data-dir",
        type=Path,
        default=Path("/home/yuanpang/Work/Li-DM-BH/outputs/sanity/data"),
    )
    parser.add_argument("--real-file", default="nfw_L0.npz")
    parser.add_argument("--no-u-at", action="store_true")
    parser.add_argument(
        "--compare-reference",
        action="store_true",
        help="Also run a scalar-reference QuadTransferSolver for same-case comparison.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    include_u_at = not args.no_u_at
    print(
        f"[run] python={platform.python_version()} numpy={np.__version__} "
        f"scipy={scipy.__version__} repeats={args.repeats} "
        f"include_u_at={include_u_at} compare_reference={args.compare_reference}"
    )
    solver_variants = [(QuadTransferSolver, "fast")]
    if args.compare_reference:
        solver_variants.append((ReferenceQuadTransferSolver, "reference"))

    for solver_cls, solver_label in solver_variants:
        benchmark_metric(
            "manufactured_schwarzschild",
            sample_metric(SchwarzschildMetric(mass=1.0), n_points=2600),
            repeats=args.repeats,
            include_u_at=include_u_at,
            solver_cls=solver_cls,
            solver_label=solver_label,
            display_solver_label=args.compare_reference,
        )
        benchmark_bounded_outward(
            repeats=args.repeats,
            include_u_at=include_u_at,
            solver_cls=solver_cls,
            solver_label=solver_label,
            display_solver_label=args.compare_reference,
        )

    real_path = args.real_data_dir / args.real_file
    real_metric = make_real_metric(real_path)
    if real_metric is None:
        print(f"\n[real:{real_path.name}] skipped: {real_path} not found")
    else:
        for solver_cls, solver_label in solver_variants:
            benchmark_metric(
                f"real:{real_path.name}",
                real_metric,
                repeats=args.repeats,
                include_u_at=include_u_at,
                solver_cls=solver_cls,
                solver_label=solver_label,
                display_solver_label=args.compare_reference,
            )


if __name__ == "__main__":
    main()
