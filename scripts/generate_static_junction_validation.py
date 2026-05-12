from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from spherical_raytracing.junctions import StaticJunctionSpacetime, StaticShell, compare_junction_results
from spherical_raytracing.junction_tracing import StaticJunctionHamiltonianSolver, StaticJunctionTransferSolver
from spherical_raytracing.metrics import ReissnerNordstromDeSitterMetric, ReissnerNordstromMetric, SchwarzschildMetric
from spherical_raytracing.observers import FiniteStaticObserver


def sample_model(junction: StaticJunctionSpacetime, observer: FiniteStaticObserver, samples: int) -> dict[str, object]:
    transfer = StaticJunctionTransferSolver(junction=junction, observer=observer)
    hamiltonian = StaticJunctionHamiltonianSolver(junction=junction, observer=observer)
    rows = []
    for alpha in np.linspace(0.02, 0.16, samples):
        b_plus = observer.impact_parameter(float(alpha))
        left = transfer.trace_b(b_plus)
        right = hamiltonian.trace_b(b_plus)
        comparison = compare_junction_results(left, right)
        rows.append(
            {
                "alpha": float(alpha),
                "b_plus": b_plus,
                "transfer_termination": left.termination_reason,
                "hamiltonian_termination": right.termination_reason,
                "transfer_shell_crossings": len(left.shell_crossings),
                "hamiltonian_shell_crossings": len(right.shell_crossings),
                "backend_disagreement": comparison["backend_disagreement"],
            }
        )
    return {"samples": rows, "warnings": list(junction.physics_diagnostics(observer.r_obs).warnings)}


def build_validation(samples: int) -> dict[str, object]:
    schwarzschild = StaticJunctionSpacetime(
        inner_metric=SchwarzschildMetric(mass=0.1, region="inner"),
        outer_metric=SchwarzschildMetric(mass=1.0, region="outer"),
        shell=StaticShell(radius=5.0),
    )
    rn = StaticJunctionSpacetime(
        inner_metric=ReissnerNordstromMetric(mass=0.8, charge=0.2, region="inner"),
        outer_metric=ReissnerNordstromMetric(mass=1.0, charge=0.2, region="outer"),
        shell=StaticShell(radius=5.0),
    )
    rnds = StaticJunctionSpacetime(
        inner_metric=ReissnerNordstromDeSitterMetric(mass=0.8, charge=0.2, cosmological_constant=0.01, region="inner"),
        outer_metric=ReissnerNordstromDeSitterMetric(mass=1.0, charge=0.2, cosmological_constant=0.01, region="outer"),
        shell=StaticShell(radius=5.0),
    )
    return {
        "schwarzschild": sample_model(schwarzschild, FiniteStaticObserver(r_obs=50.0, metric=schwarzschild.outer_metric), samples),
        "rn": sample_model(rn, FiniteStaticObserver(r_obs=50.0, metric=rn.outer_metric), samples),
        "rnds": sample_model(rnds, FiniteStaticObserver(r_obs=6.0, metric=rnds.outer_metric), samples),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--samples", type=int, default=8)
    args = parser.parse_args()
    if args.samples <= 0:
        raise SystemExit("--samples must be positive")
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(build_validation(args.samples), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
