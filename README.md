# Raytracing Spherical

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

English | [中文](README.zh-CN.md)

Raytracing Spherical is a Python package for null-ray tracing in static,
spherically symmetric spacetimes. It uses the conserved quantities of spherical
symmetry to reduce face-on image generation to a one-dimensional
impact-parameter profile `I(b)`, then rotates that profile into a two-dimensional
image.

## Features

- General static spherical metric interface in areal-radius coordinates:
  `ds^2 = -A(r)dt^2 + B(r)dr^2 + r^2 dOmega^2`
- Schwarzschild and LQG-like metric examples
- Observer models at infinity and at finite static radius
- Quadrature, sampled ODE, and exterior Hamiltonian trajectory solvers
- Thin-disk transfer intersections and observed-intensity decomposition
- Axisymmetric radial-profile sampling and image rendering
- Validation and figure-generation scripts

## Installation

Python 3.11 or newer is required.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[test]"
```

The core package depends on NumPy and SciPy. Plot output from the scripts also
requires Matplotlib:

```bash
pip install matplotlib
```

## Quick Start

```python
from spherical_raytracing import (
    DiskWindow,
    InfinityObserver,
    QuadTransferSolver,
    SchwarzschildMetric,
    ThinDiskSource,
    compute_intersections,
    observed_intensity,
)

metric = SchwarzschildMetric(mass=1.0)
observer = InfinityObserver()
solver = QuadTransferSolver(metric=metric, observer=observer)

ray = solver.trace_b(8.0)
disk = DiskWindow(r_min=6.0, r_max=100.0)
intersections = compute_intersections(ray, disk, max_order=3)

source = ThinDiskSource(lambda r, region=None: 1.0 / r**2)
intensity = observed_intensity(intersections, source, metric, observer)

print(ray.diagnostics.termination_reason)
print(intensity.total)
```

## Project Layout

```text
src/spherical_raytracing/
  diagnostics.py    # Events, ray segments, diagnostics dataclasses
  imaging.py        # Radial profile sampling and axisymmetric rendering
  metrics.py        # Metric protocols and Schwarzschild/LQG-like metrics
  observers.py      # Infinity and finite static observer mappings
  policies.py       # Outside and through-trace policies
  solvers.py        # Quadrature, ODE, and Hamiltonian ray solvers
  sources.py        # Thin-disk sources and observed-intensity helpers
  transfer.py       # Thin-disk crossing records
```

## Scripts

Generate a Schwarzschild validation table:

```bash
python scripts/generate_validation_tables.py \
  --output outputs/schwarzschild_transfer.json \
  --samples 32
```

Generate Gralla-Holz-Wald Fig. 5 style thin-disk profiles:

```bash
python scripts/generate_fig5_profiles.py \
  --profile all \
  --output outputs/fig5_profiles.csv \
  --plot-output outputs/fig5_profiles.png \
  --image-output outputs/fig5_images.png
```

Generate the LQG Fig. 3 style profile and ring-edge comparison:

```bash
python scripts/generate_lqg_fig3_profiles.py \
  --output outputs/lqg_fig3_profile.csv \
  --plot-output outputs/lqg_fig3_profile.png \
  --image-output outputs/lqg_fig3_image.png \
  --comparison-output outputs/lqg_fig3_ring_edges.csv
```

Pre-generated CSV and PNG results are included under `outputs/`.

## Testing

```bash
pytest -q
```

The tests cover analytic Schwarzschild quantities, observer mappings, solver
events, transfer intersections, source redshift weights, image sampling, and the
public generation scripts.

## Notes

- Dimensionless geometric units are used throughout, with `M = 1` by default.
- `QuadTransferSolver` is the main path for face-on thin-disk image production.
- `OdeTrajectorySolver` and `HamiltonianTrajectorySolver` are intended for
  trajectory diagnostics and cross-checks.

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE).
