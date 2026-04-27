from __future__ import annotations

import argparse
import csv
import math
import os
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.optimize import brentq

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from spherical_raytracing import (
    DiskWindow,
    EventType,
    ImageGrid,
    InfinityObserver,
    LQGMetric,
    QuadTransferSolver,
    RadialProfile,
    SolverOptions,
    ThroughTracePolicy,
    compute_intersections,
    render_axisymmetric_image,
)


DEFAULT_GAMMA = 0.2375
DEFAULT_BETAV = 0.98
DEFAULT_MASS_OVER_SQRT_ALPHA = math.sqrt(4.0 * DEFAULT_BETAV**4 / (1.0 - DEFAULT_BETAV**2) ** 3)
REFERENCE_RING_EDGES = [
    (0.072, 0.076),
    (1.077, 1.512),
    (2.974, 4.373),
    (4.414, 5.146),
    (4.995, 5.194),
]


@dataclass(frozen=True)
class RingEdge:
    index: int
    b_n: float
    bprime_n: float
    reference_b_n: float | None = None
    reference_bprime_n: float | None = None


@dataclass(frozen=True)
class RaySample:
    b: float
    intensity: float
    radii: tuple[float, ...]
    phi_out: float
    phi_tot: float
    termination_reason: str


class LQGFig3Model:
    def __init__(
        self,
        *,
        gamma: float = DEFAULT_GAMMA,
        mass_over_sqrt_alpha: float = DEFAULT_MASS_OVER_SQRT_ALPHA,
        max_contributions: int = 3,
        show_integration_warnings: bool = False,
    ) -> None:
        if gamma <= 0.0:
            raise ValueError("gamma must be positive")
        if mass_over_sqrt_alpha <= 0.0:
            raise ValueError("mass_over_sqrt_alpha must be positive")
        if max_contributions < 1:
            raise ValueError("max_contributions must be at least 1")

        self.gamma = gamma
        self.alpha = 16.0 * math.sqrt(3.0) * math.pi * gamma**3
        self.mass_over_sqrt_alpha = mass_over_sqrt_alpha
        self.mass = mass_over_sqrt_alpha * math.sqrt(self.alpha)
        self.q = 1.0 / mass_over_sqrt_alpha**2
        self.max_contributions = max_contributions
        self.show_integration_warnings = show_integration_warnings
        self.metric = LQGMetric(mass=1.0, alpha=self.q)
        self.solver = QuadTransferSolver(
            metric=self.metric,
            observer=InfinityObserver(),
            trace_policy=ThroughTracePolicy(region_transitions={("external", EventType.HORIZON): "companion"}),
            options=SolverOptions(critical_exclusion=0.0, max_phi=100.0 * math.pi),
        )
        self.x_horizons = self.metric.horizons()
        if not self.x_horizons:
            raise ValueError("parameters do not produce a real positive horizon")
        self.x_plus = self.x_horizons[-1]
        self.u_plus = 1.0 / self.x_plus
        photon_candidates = self.metric.photon_spheres()
        if not photon_candidates:
            raise ValueError("parameters do not produce an external photon sphere")
        self.x_ph = photon_candidates[-1]
        self.u_ph = 1.0 / self.x_ph
        self.b_crit = self.x_ph / math.sqrt(self.f(self.x_ph))
        self._phi_cache: dict[float, tuple[float, float]] = {}
        self._ray_cache = {}

    def f(self, x: float) -> float:
        return self.metric.A(x)

    def G(self, u: float, b: float) -> float:
        return self.metric.G(u, b)

    def emission(self, x: float) -> float:
        if x <= self.x_ph:
            return 0.0
        return 1.0 / (x - (self.x_ph - 1.0)) ** 3

    def _trace(self, b: float):
        key = round(float(b), 14)
        cached = self._ray_cache.get(key)
        if cached is None:
            cached = self.solver.trace_b(float(b))
            self._ray_cache[key] = cached
        return cached

    def phi_angles(self, b: float) -> tuple[float, float]:
        key = round(float(b), 14)
        cached = self._phi_cache.get(key)
        if cached is not None:
            return cached
        ray = self._trace(b)
        companion_segments = [
            segment
            for segment in ray.segments
            if segment.region == "companion" and segment.endpoint_event == EventType.ESCAPE
        ]
        if not companion_segments:
            angles = (math.nan, math.nan)
        else:
            segment = companion_segments[-1]
            angles = (segment.phi_start, segment.phi_end)
        self._phi_cache[key] = angles
        return angles

    def companion_radii(self, b: float) -> tuple[float, ...]:
        if b <= 0.0:
            return ()
        if b >= self.b_crit * (1.0 - 1e-12):
            return ()

        phi_out, phi_tot = self.phi_angles(b)
        if not math.isfinite(phi_out) or not math.isfinite(phi_tot):
            return ()
        max_order = max(self.max_contributions + 8, int(math.floor(phi_tot / math.pi + 2.0)))
        records = compute_intersections(
            self._trace(b),
            DiskWindow(r_min=self.x_ph, r_max=math.inf, enabled_regions=frozenset({"companion"})),
            max_order=max_order,
        )
        return tuple(record.r for record in records)

    def sample(self, b: float) -> RaySample:
        if b <= 0.0:
            return RaySample(b=b, intensity=0.0, radii=(), phi_out=0.0, phi_tot=0.0, termination_reason="axis")
        if b >= self.b_crit * (1.0 - 1e-12):
            return RaySample(
                b=b,
                intensity=0.0,
                radii=(),
                phi_out=math.nan,
                phi_tot=math.nan,
                termination_reason="outside_companion_domain",
            )

        phi_out, phi_tot = self.phi_angles(b)
        radii = self.companion_radii(b)
        contributions = [self.f(x) ** 2 * self.emission(x) for x in radii[: self.max_contributions]]
        reason = "horizon_through" if radii else "no_disk_intersection"
        return RaySample(b=b, intensity=sum(contributions), radii=radii, phi_out=phi_out, phi_tot=phi_tot, termination_reason=reason)

    def solve_ring_edges(self, count: int) -> list[RingEdge]:
        edges: list[RingEdge] = []
        for index in range(1, count + 1):
            target = (2 * index - 1) * math.pi / 2.0
            b_n = self._solve_angle_root(lambda value: self.phi_angles(value)[1], target)
            bprime_n = self._solve_angle_root(lambda value: self.phi_angles(value)[0], target)
            reference = REFERENCE_RING_EDGES[index - 1] if index <= len(REFERENCE_RING_EDGES) else (None, None)
            edges.append(RingEdge(index=index, b_n=b_n, bprime_n=bprime_n, reference_b_n=reference[0], reference_bprime_n=reference[1]))
        return edges

    def _solve_angle_root(self, angle_fn, target: float) -> float:
        low = 1e-8
        high = self.b_crit * (1.0 - 1e-8)
        low_value = angle_fn(low) - target
        high_value = angle_fn(high) - target
        if low_value > 0.0 or high_value < 0.0:
            return math.nan
        return float(brentq(lambda value: angle_fn(value) - target, low, high, xtol=1e-8, rtol=1e-8, maxiter=80))


def _b_grid(samples: int, b_max: float, b_crit: float, ring_edges: list[RingEdge]) -> np.ndarray:
    base = np.linspace(0.0, b_max, samples)
    focused: list[float] = []
    for edge in ring_edges:
        if math.isfinite(edge.b_n) and math.isfinite(edge.bprime_n):
            focused.extend([edge.b_n, edge.bprime_n, 0.5 * (edge.b_n + edge.bprime_n)])
            width = max(edge.bprime_n - edge.b_n, 1e-3)
            for offset in (0.1 * width, 0.25 * width, 0.5 * width, 1e-3, 5e-3):
                focused.extend([edge.b_n - offset, edge.b_n + offset, edge.bprime_n - offset, edge.bprime_n + offset])
    critical_offsets = np.geomspace(1e-5, 5e-2, max(samples // 4, 16))
    focused.extend((b_crit - critical_offsets).tolist())
    values = np.concatenate([base, np.array(focused, dtype=float)])
    values = values[(values >= 0.0) & (values <= b_max)]
    return np.unique(np.round(values, 12))


def generate_rows(model: LQGFig3Model, samples: int, b_max: float, ring_edges: list[RingEdge]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for b in _b_grid(samples, b_max, model.b_crit, ring_edges):
        ray = model.sample(float(b))
        radii = list(ray.radii[:3])
        while len(radii) < 3:
            radii.append(math.nan)
        rows.append(
            {
                "b_over_M": ray.b,
                "Iobs_over_I0": ray.intensity,
                "intersection_count": len(ray.radii),
                "r1_over_M": radii[0],
                "r2_over_M": radii[1],
                "r3_over_M": radii[2],
                "phi_out": ray.phi_out,
                "phi_tot": ray.phi_tot,
                "termination_reason": ray.termination_reason,
            }
        )
    return rows


def write_csv(rows: list[dict[str, object]], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "b_over_M",
        "Iobs_over_I0",
        "intersection_count",
        "r1_over_M",
        "r2_over_M",
        "r3_over_M",
        "phi_out",
        "phi_tot",
        "termination_reason",
    ]
    with output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_comparison(edges: list[RingEdge], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "n",
        "b_n_over_M",
        "bprime_n_over_M",
        "reference_b_n_over_M",
        "reference_bprime_n_over_M",
        "delta_b_n_over_M",
        "delta_bprime_n_over_M",
    ]
    with output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for edge in edges:
            delta_b = edge.b_n - edge.reference_b_n if edge.reference_b_n is not None else math.nan
            delta_bp = edge.bprime_n - edge.reference_bprime_n if edge.reference_bprime_n is not None else math.nan
            writer.writerow(
                {
                    "n": edge.index,
                    "b_n_over_M": edge.b_n,
                    "bprime_n_over_M": edge.bprime_n,
                    "reference_b_n_over_M": edge.reference_b_n,
                    "reference_bprime_n_over_M": edge.reference_bprime_n,
                    "delta_b_n_over_M": delta_b,
                    "delta_bprime_n_over_M": delta_bp,
                }
            )


def _configure_matplotlib():
    os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "spherical_raytracing_matplotlib"))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def _radial_profile_from_rows(rows: list[dict[str, object]]) -> RadialProfile:
    rows = sorted(rows, key=lambda row: float(row["b_over_M"]))
    return RadialProfile(
        coordinates=np.array([float(row["b_over_M"]) for row in rows]),
        intensities=np.array([float(row["Iobs_over_I0"]) for row in rows]),
        diagnostics={"sample_count": len(rows)},
    )


def write_plot(rows: list[dict[str, object]], edges: list[RingEdge], output: Path, b_max: float) -> None:
    plt = _configure_matplotlib()
    output.parent.mkdir(parents=True, exist_ok=True)
    sorted_rows = sorted(rows, key=lambda row: float(row["b_over_M"]))
    figure, axis = plt.subplots(figsize=(8.0, 4.8))
    axis.plot(
        [float(row["b_over_M"]) for row in sorted_rows],
        [float(row["Iobs_over_I0"]) for row in sorted_rows],
        color="tab:blue",
        linewidth=1.4,
        label="horizon-through LQG",
    )
    for edge in edges:
        if math.isfinite(edge.b_n) and math.isfinite(edge.bprime_n):
            axis.axvspan(edge.b_n, edge.bprime_n, color="tab:orange", alpha=0.12)
            axis.axvline(edge.b_n, color="tab:orange", linewidth=0.7, alpha=0.65)
            axis.axvline(edge.bprime_n, color="tab:orange", linewidth=0.7, alpha=0.65)
    axis.set_xlabel("b/M")
    axis.set_ylabel("I_obs/I0")
    axis.set_xlim(0.0, b_max)
    axis.set_ylim(bottom=0.0)
    axis.grid(alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(output, dpi=180)
    plt.close(figure)


def write_image(rows: list[dict[str, object]], output: Path, image_size: int, image_radius: float) -> None:
    if image_size <= 0:
        raise ValueError("image_size must be positive")
    if image_radius <= 0.0:
        raise ValueError("image_radius must be positive")

    plt = _configure_matplotlib()
    output.parent.mkdir(parents=True, exist_ok=True)
    profile = _radial_profile_from_rows(rows)
    image = render_axisymmetric_image(profile, ImageGrid(width=image_size, height=image_size, radius=image_radius))
    vmax = max(float(np.max(image.pixels)), 1e-300)
    figure, axis = plt.subplots(figsize=(4.2, 4.2))
    axis.imshow(
        image.pixels,
        extent=[-image_radius, image_radius, -image_radius, image_radius],
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


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate the horizon-through LQG thin-disk intensity profile corresponding to Fig. 3."
    )
    parser.add_argument("--output", required=True, help="CSV output path for the observed intensity profile")
    parser.add_argument("--plot-output", help="Optional PNG output path for the observed intensity profile")
    parser.add_argument("--image-output", help="Optional PNG output path for the axisymmetric image-plane rendering")
    parser.add_argument("--comparison-output", help="Optional CSV output path comparing ring edges with the Fig. 3 caption")
    parser.add_argument("--samples", type=int, default=800)
    parser.add_argument("--b-max", type=float, default=5.5)
    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument("--image-radius", type=float, default=5.5)
    parser.add_argument("--gamma", type=float, default=DEFAULT_GAMMA)
    parser.add_argument("--mass-over-sqrt-alpha", type=float, default=DEFAULT_MASS_OVER_SQRT_ALPHA)
    parser.add_argument("--ring-count", type=int, default=5)
    parser.add_argument("--max-contributions", type=int, default=3)
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
    if args.ring_count < 1:
        raise SystemExit("--ring-count must be at least 1")
    if args.max_contributions < 1:
        raise SystemExit("--max-contributions must be at least 1")

    model = LQGFig3Model(
        gamma=args.gamma,
        mass_over_sqrt_alpha=args.mass_over_sqrt_alpha,
        max_contributions=args.max_contributions,
        show_integration_warnings=args.show_integration_warnings,
    )
    ring_edges = model.solve_ring_edges(args.ring_count)
    rows = generate_rows(model, args.samples, args.b_max, ring_edges)
    write_csv(rows, Path(args.output))
    if args.comparison_output:
        write_comparison(ring_edges, Path(args.comparison_output))
    if args.plot_output:
        write_plot(rows, ring_edges, Path(args.plot_output), args.b_max)
    if args.image_output:
        write_image(rows, Path(args.image_output), args.image_size, args.image_radius)


if __name__ == "__main__":
    main()
