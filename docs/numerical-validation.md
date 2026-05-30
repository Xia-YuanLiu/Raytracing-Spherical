# Numerical Validation

Date: 2026-05-29

The validation strategy is manufactured-solution testing: sample analytical
metrics into synthetic numerical tables, trace the same benign rays through the
analytical and tabulated paths, and compare stable public observables.

Core analytical formulas are checked by a private symbolic oracle module. The
symbolic oracle is executable test code, not a public API and not a manual
documentation contract.

## Default Guardrails

- Public API contract and representative workflows:
  `pytest tests/test_package.py -q`
- Symbolic formula oracle for analytical metrics, redshift, and shell matching:
  `pytest tests/test_symbolic_equation_oracles.py -q`
- Synthetic `.npz` ingestion and `TabulatedMetric` behavior:
  `pytest tests/test_numerical_metrics.py -q`
- Manufactured analytical-vs-tabulated validation:
  `pytest tests/test_numerical_metric_validation.py -q`
- Analytical backend comparator:
  `pytest tests/test_backend_comparator.py -q`
- Identical-metric junction degeneration:
  `pytest tests/test_static_junction_degenerate.py -q`

Default `pytest -q` excludes `@pytest.mark.slow` tests.

## Slow Scaffold

Run the slow numerical validation scaffold explicitly:

```bash
pytest -m slow tests/test_symbolic_geodesic_oracle.py -q
pytest -m slow tests/test_numerical_metric_validation.py -q
```

Slow tests may expand to more grid densities or metric families, but they should
not become a default CI requirement without a separate decision.

The symbolic geodesic oracle derives Christoffel symbols from the metric tensor
and integrates the full geodesic equation on benign finite-observer rays. It is
intended as an independent check against mistakes in reduced ray equations.

## Compared Observables

Validation compares observable behavior rather than raw implementation details:

- termination reason;
- segment count and endpoint events;
- `phi_end` and `u_end`;
- representative `RaySegment.u_at(...)` probes;
- thin-disk intersection radii when intersection count/order are stable;
- redshift weights and observed-intensity contributions where meaningful;
- backend residuals, including Hamiltonian constraint diagnostics for analytical
  Hamiltonian cases.

## Exclusions

Routine validation avoids near-critical, horizon-adjacent, table-boundary, and
`max_phi` truncation cases unless a test is explicitly about one of those
behaviors. These cases can flip event classification for physically legitimate
reasons and need targeted tests.

## Benchmark

The numerical metric benchmark is a measurement tool, not a CI threshold:

```bash
python3 scripts/benchmark_numerical_metric_perf.py --repeats 1 --no-u-at --compare-reference --real-data-dir /tmp/missing-li-data
```

It reports run metadata, ray counts, category counts, median/p95 timings, and
fast-path fallback counts. The `--compare-reference` mode runs the guarded fast
path and the scalar reference path in the same process for manual comparison.
