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

from spherical_raytracing.metrics import SchwarzschildMetric
from spherical_raytracing.observers import InfinityObserver
from spherical_raytracing.solvers import QuadTransferSolver
from spherical_raytracing.transfer import DiskWindow, compute_intersections


def build_table(samples: int) -> dict[str, object]:
    metric = SchwarzschildMetric(mass=1.0)
    observer = InfinityObserver()
    solver = QuadTransferSolver(metric=metric, observer=observer)
    b_crit = metric.critical_curves()[0].b_crit
    b_values = np.linspace(1.05 * b_crit, 20.0, samples)
    rows = []
    for b in b_values:
        ray = solver.trace_b(float(b))
        intersections = compute_intersections(ray, DiskWindow(r_min=6.0, r_max=100.0), max_order=3)
        rows.append(
            {
                "b": float(b),
                "near_critical": ray.diagnostics.near_critical,
                "termination_reason": ray.diagnostics.termination_reason,
                "intersections": [
                    {"m": item.m, "r": item.r, "phi": item.phi, "path_class": item.path_class}
                    for item in intersections
                ],
            }
        )
    return {"metric": "schwarzschild", "mass": 1.0, "samples": rows}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--samples", type=int, default=32)
    args = parser.parse_args()
    if args.samples <= 0:
        raise SystemExit("--samples must be positive")
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(build_table(args.samples), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
