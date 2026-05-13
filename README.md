# Raytracing Spherical

[![License: BSD-3-Clause](https://img.shields.io/badge/License-BSD--3--Clause-blue.svg)](LICENSE)

English | [中文](README.zh-CN.md)

Raytracing Spherical is a Python package for null-ray tracing in static,
spherically symmetric spacetimes. It supports both single-metric thin-disk
imaging and static thin-shell junction spacetimes, where rays can cross a shell
between matched inner and outer metrics.

The package is built around face-on, axisymmetric image generation. For each
screen coordinate or impact parameter, it traces the ray, records disk
intersections, applies redshift weights, and samples the resulting one-
dimensional intensity profile before rendering a two-dimensional image.

## Features

- Static spherical metric interface in areal-radius coordinates:
  `ds^2 = -A(r)dt^2 + B(r)dr^2 + r^2 dOmega^2`
- Built-in Schwarzschild, LQG-like, Reissner-Nordstrom, and
  Reissner-Nordstrom-de Sitter metrics
- Infinity and finite-radius static observer models
- Quadrature transfer, sampled ODE, and Hamiltonian trajectory solvers
- Thin-disk intersections, path classification, redshift weights, radial
  profile sampling, and axisymmetric image rendering
- Static thin-shell junction tracing with shell-crossing records, energy and
  impact-parameter matching, and junction physics diagnostics
- Transfer and Hamiltonian backend comparison for junction cases
- Generation scripts for Schwarzschild, LQG-like, static-junction, RN, RN-dS,
  and atlas-style outputs

## Installation

Python 3.11 or newer is required.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[test]"
```

The core package depends on NumPy and SciPy. Plot and image output from the
generation scripts also requires Matplotlib:

```bash
pip install matplotlib
```

## Quick Start: Schwarzschild Thin Disk

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

## Quick Start: Static Junction

```python
from spherical_raytracing import (
    FiniteStaticObserver,
    ReissnerNordstromMetric,
    StaticJunctionSpacetime,
    StaticJunctionTransferSolver,
    StaticShell,
)

inner = ReissnerNordstromMetric(mass=0.8, charge=0.2, region="inner")
outer = ReissnerNordstromMetric(mass=1.0, charge=0.2, region="outer")
junction = StaticJunctionSpacetime(
    inner_metric=inner,
    outer_metric=outer,
    shell=StaticShell(radius=4.0),
)
observer = FiniteStaticObserver(r_obs=50.0, metric=outer)
solver = StaticJunctionTransferSolver(junction=junction, observer=observer)

ray = solver.trace_b(4.0)

print(ray.termination_reason)
print(len(ray.shell_crossings))
print(ray.physics_warnings)
```

`StaticJunctionSpacetime` currently supports same-family static junctions:
Schwarzschild-Schwarzschild, RN-RN, or RN-dS-RN-dS.

## Project Layout

```text
src/spherical_raytracing/
  diagnostics.py          # Ray events, segments, diagnostics, critical curves
  imaging.py              # Radial profile sampling and image rendering
  junction_analysis.py    # Junction case summaries and classification helpers
  junction_tracing.py     # Static junction transfer and Hamiltonian solvers
  junctions.py            # Shell matching, crossings, and junction diagnostics
  metrics.py              # Static spherical metrics
  observers.py            # Infinity and finite static observers
  policies.py             # Trace policy and solver options
  solvers.py              # Single-metric quadrature, ODE, and Hamiltonian solvers
  sources.py              # Thin-disk sources and observed intensity helpers
  transfer.py             # Disk windows and intersection records
```

## Scripts

Generate a Schwarzschild validation table:

```bash
python scripts/generate_validation_tables.py \
  --output outputs/schwarzschild_transfer.json \
  --samples 32
```

Generate Gralla-Holz-Wald Fig. 5 style Schwarzschild thin-disk profiles:

```bash
python scripts/generate_fig5_profiles.py \
  --profile all \
  --output outputs/schwarzschild_fig5/fig5_profiles.csv \
  --plot-output outputs/schwarzschild_fig5/fig5_profiles.png \
  --image-output outputs/schwarzschild_fig5/fig5_images.png
```

Generate the LQG Fig. 3 style profile and ring-edge comparison:

```bash
python scripts/generate_lqg_fig3_profiles.py \
  --output outputs/lqg_fig3/lqg_fig3_profile.csv \
  --plot-output outputs/lqg_fig3/lqg_fig3_profile.png \
  --image-output outputs/lqg_fig3/lqg_fig3_image.png \
  --comparison-output outputs/lqg_fig3/lqg_fig3_ring_edges.csv
```

Generate static-junction validation data:

```bash
python scripts/generate_static_junction_validation.py \
  --output outputs/static_junction_validation.json \
  --samples 8
```

Generate the static-junction Fig. 3/Fig. 4 reproduction outputs:

```bash
python scripts/generate_static_junction_fig3_fig4.py \
  --output outputs/static_junction/fig3_fig4/static_junction_fig3_fig4.json \
  --fig3-output outputs/static_junction/fig3_fig4/static_junction_fig3.png \
  --fig4-image-output outputs/static_junction/fig3_fig4/static_junction_fig4_image.png \
  --fig4-trajectories-output outputs/static_junction/fig3_fig4/static_junction_fig4_trajectories.png
```

Generate curated RN static-junction images:

```bash
python scripts/generate_rn_junction_images.py \
  --output-dir outputs/rn_junction_sweep \
  --compare-backends
```

Generate the RN/RN-dS static-junction atlas:

```bash
python scripts/generate_junction_atlas.py \
  --preset quick \
  --output-dir outputs/junction_atlas \
  --compare-backends \
  --include-schwarzschild-reference \
  --emissivity both
```

Write a Markdown atlas report from a generated manifest:

```bash
python scripts/write_junction_atlas_report.py \
  --manifest outputs/junction_atlas/manifest.json \
  --output docs/junction-atlas/rn-rnds-static-junction-atlas.md
```

## Generated Outputs

Generated artifacts are grouped under `outputs/` by experiment:

- `outputs/junction_atlas/`: RN/RN-dS atlas manifest, phase maps,
  representative case profiles, images, transfer-redshift plots, and optional
  Schwarzschild reference artifacts
- `outputs/junction_atlas_schwarzschild_reference/`: standalone
  Schwarzschild reference run from the atlas generator
- `outputs/rn_junction_sweep/`: curated RN static-junction sweep images and
  per-case JSON summaries
- `outputs/static_junction/fig3_fig4/`: static-junction Fig. 3/Fig. 4
  reproduction data and figures
- `outputs/schwarzschild_fig5/`: Schwarzschild Fig. 5 profile and image outputs
- `outputs/lqg_fig3/`: LQG Fig. 3 profile, image, and ring-edge outputs

Some legacy root-level output files are still present for compatibility with
earlier runs. New generated results should generally use the grouped output
directories.

## Testing

```bash
pytest -q
```

The tests cover metric quantities, observer mappings, solver events,
Hamiltonian and transfer backend agreement, disk intersections, redshift
weights, image sampling, static-junction shell matching, junction diagnostics,
and the public generation scripts.

## Notes

- Dimensionless geometric units are used throughout, with `M = 1` as the
  default scale in most examples.
- `QuadTransferSolver` is the main path for single-metric face-on thin-disk
  image production.
- `StaticJunctionTransferSolver` is the main path for static-junction image
  production; `StaticJunctionHamiltonianSolver` is used for diagnostics and
  backend comparison.
- RN-dS junctions must lie in clean static patches. Diagnostics report cases
  such as nonstatic shell or observer positions, nearby horizons, charge or
  cosmological-constant jumps, and negative surface energy density.
- Atlas and image-generation scripts can produce large output directories.

## Citation

If you use this code in academic work, please cite this repository using the
metadata in [CITATION.cff](CITATION.cff). Please also cite the original
scientific papers for the spacetime model, emission model, and comparison
figures used in your analysis.

## License

This project is licensed under the BSD 3-Clause License. See [LICENSE](LICENSE).
