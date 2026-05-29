# Review: P0 Stabilization

Date: 2026-05-29

## Completed

- Repository hygiene risk made explicit: durable `docs/adr`, `docs/plans`, and
  `docs/review` files are no longer hidden by the broad `docs/` ignore rule.
- Public API contract locks the current 50 `spherical_raytracing.__all__`
  exports and representative single-metric/junction workflows.
- Synthetic numerical metric coverage exercises standard `r/A/B` and
  Li-DM-BH-style `r/f/g/B/params` payloads through load, domain selection, and
  `TabulatedMetric`.
- L4 manufactured Schwarzschild grid-convergence smoke and slow scaffold added.
- L5 analytical backend comparator added for Schwarzschild, RN, and RN-dS.
- Identical-metric static-junction degeneration tests added.
- Benchmark smoke and minimal GitHub Actions CI added.

## Validation Run

- `pytest tests/test_package.py -q`
- `pytest tests/test_numerical_metrics.py -q`
- `pytest tests/test_numerical_metric_validation.py -q`
- `pytest -m slow tests/test_numerical_metric_validation.py -q`
- `pytest tests/test_backend_comparator.py -q`
- `pytest tests/test_static_junction_degenerate.py -q`
- `pytest tests/test_benchmark_script_smoke.py -q`
- `pytest -q`

Full baseline before P1 work: `252 passed, 1 deselected, 30 warnings`.

## Plan Fit

P0 is complete relative to the stabilization plan. No public API symbols were
renamed, removed, or added. Default tests do not require private Li-DM-BH data.

## Review Findings

- No blocking API risk found.
- No over-designed public abstraction was introduced.
- Numerical warnings remain in difficult quadrature cases; they pre-existed and
  are now covered by passing tests.
- Commit boundary remains a manual action: new docs and private helper modules
  are visible as untracked until explicitly staged. No git commit was made.

## Human Confirmation Needed

- Confirm whether tracked historical `outputs/` files are examples, regression
  references, or removable generated artifacts before any cleanup.
