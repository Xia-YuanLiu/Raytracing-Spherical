# Review: P1 Simplification And Measured Speed Work

Date: 2026-05-29

## Completed

- Extracted a private shared quadrature segment/evaluator kernel into
  `src/spherical_raytracing/_segments.py`.
- Preserved single-metric fast path dispatch and junction-specific tracing
  control flow.
- Kept `RaySegment` and `JunctionRayResult` public behavior unchanged.
- Kept `TabulatedMetric` fast path private and fail-closed to scalar reference
  behavior.
- Added benchmark `--compare-reference` mode for fast-vs-scalar measurement.
- Added benchmark run metadata: Python, NumPy, and SciPy versions.

## Validation Run

- `pytest tests/test_package.py tests/test_backend_comparator.py -q`
- `pytest tests/test_static_junctions.py tests/test_static_junction_backend_agreement.py -q`
- `pytest tests/test_numerical_metrics.py -q`
- `pytest tests/test_numerical_metric_validation.py tests/test_numerical_metrics.py tests/test_benchmark_script_smoke.py -q`
- `python3 scripts/benchmark_numerical_metric_perf.py --repeats 1 --no-u-at --compare-reference --real-data-dir /tmp/missing-li-data`

Benchmark snapshot:

- manufactured Schwarzschild fast all median about `0.209 ms`, p95 about
  `28.835 ms`, fast segments `6/8` accepted, fallback `2`.
- manufactured Schwarzschild scalar-reference all median about `3.237 ms`, p95
  about `30.552 ms`.
- manufactured RN-dS bounded outward fast median about `0.293 ms`, reference
  median about `2.571 ms`.

## Plan Fit

P1.1 and the already-present P1.2 fast path are consistent with the plan:
private helpers only, no public API expansion, scalar reference behavior retained
as oracle/fallback, and benchmark data reports fallback categories.

P1.3 was executed conservatively. Ruff was not added because it is not installed
in the current environment, and adding an unverified tool dependency would be
CI-risky. Benchmark metadata polish was added instead.

## Review Findings

- No blocking regression found in P1 target tests.
- The new `_segments` helper is private and narrow; it removes duplicated
  quadrature/evaluator construction without taking over diagnostics or tracing
  policy.
- Numerical warnings now point at `_segments.py`, which is expected after moving
  the same quadrature calls.

## Human Confirmation Needed

- Decide later whether to introduce linting as a project dependency.
- Decide later whether imaging/profile benchmarks justify internal batch tracing.
