# Final Stabilization Review

Date: 2026-05-29

## Fixed In Final Pass

- Updated `.gitignore` so the README-linked governance docs are no longer
  hidden by the broad `docs/*` rule:
  - `docs/artifacts.md`
  - `docs/api-compatibility.md`
  - `docs/numerical-validation.md`
  - `docs/physics-conventions.md`
- Did not clean `outputs/` or delete historical artifacts.

## Final Phase Status

- P0 is complete. The public API contract, synthetic numerical metric coverage,
  L4 convergence smoke/slow scaffold, L5 analytical backend comparator,
  identical-metric junction degeneration tests, benchmark smoke, and minimal CI
  are present.
- P1 is complete for the reasonable non-breaking scope. The shared private
  segment/evaluator helper is present, and the single-metric `TabulatedMetric`
  fast path remains private, measured, and fail-closed to scalar reference
  behavior. Internal batch tracing was intentionally not added.
- P2 is complete for governance/documentation decisions that can be made without
  artifact deletion or a new public API decision. The Hamiltonian numerical
  metric policy and batch tracing deferral are recorded as ADRs.

## Commands Run

```bash
git status --short --untracked-files=all --ignored=matching docs CHANGELOG.md src/spherical_raytracing/_segments.py src/spherical_raytracing/_turning.py
```

Result after the ignore fix and after this final review document was written:

```text
?? CHANGELOG.md
?? docs/adr/0001-upstream-computation-core-api-contract.md
?? docs/adr/0002-tabulated-metric-hamiltonian-policy.md
?? docs/adr/0003-defer-internal-batch-tracing.md
?? docs/api-compatibility.md
?? docs/artifacts.md
?? docs/numerical-validation.md
?? docs/physics-conventions.md
?? docs/plans/implementation-prompts.md
?? docs/plans/numerical-metric-perf-plan.md
?? docs/plans/numerical-metric-validation-plan.md
?? docs/plans/upstream-computation-core-stabilization-plan.md
?? "docs/plans/\346\225\260\345\200\274\345\272\246\350\247\204\346\216\245\345\205\245\350\256\241\345\210\222.md"
?? docs/review/final-stabilization-review.md
?? docs/review/p0-stabilization-review.md
?? docs/review/p1-stabilization-review.md
?? docs/review/p2-governance-review.md
?? docs/review/upstream-computation-core-stabilization-plan-review.md
?? src/spherical_raytracing/_segments.py
?? src/spherical_raytracing/_turning.py
```

```bash
PYTHONPATH=src python3 -c "import spherical_raytracing as s; print(len(s.__all__)); print('\n'.join(s.__all__))"
```

Result: `50` exports. The list matches the P0 contract:

```text
CanonicalPayload
CriticalCurve
DiskWindow
EventType
FiniteStaticObserver
HamiltonianTrajectorySolver
ImageGrid
InfinityObserver
IntersectionRecord
JunctionPhysicsDiagnostics
JunctionRayResult
LQGMetric
LiDMBHMetadata
SegmentConstants
ShellCrossingRecord
ObservedIntensity
OdeTrajectorySolver
OutsideTracePolicy
QuadTransferSolver
RadialProfile
RayDiagnostics
RayEvent
RayResult
RaySegment
RenderedImage
SamplingOptions
StaticDomainSelector
StaticJunctionHamiltonianSolver
StaticJunctionTransferSolver
StaticJunctionSpacetime
StaticShell
ReissnerNordstromDeSitterMetric
ReissnerNordstromMetric
SchwarzschildMetric
SolverOptions
TabulatedMetric
ThinDiskSource
ValidationGate
ValidationOptions
annotate_junction_intersections
compare_junction_results
junction_redshift_factor
load_metric_npz
match_static_shell
trace_with_backend_compare
ThroughTracePolicy
compute_intersections
observed_intensity
render_axisymmetric_image
sample_radial_profile
```

```bash
pytest -q
```

Result: `253 passed, 1 deselected, 30 warnings in 111.56s`.

```bash
pytest -m slow tests/test_numerical_metric_validation.py -q
```

Result: `1 passed, 16 deselected in 0.30s`.

```bash
python3 scripts/benchmark_numerical_metric_perf.py --repeats 1 --no-u-at --compare-reference --real-data-dir /tmp/missing-li-data
```

Result: passed. Snapshot:

- manufactured Schwarzschild fast all median `0.210 ms`, p95 `27.723 ms`,
  fast segments `6/8` accepted, fallback `2`, late fallback `0`.
- manufactured Schwarzschild scalar reference all median `3.090 ms`, p95
  `29.388 ms`.
- manufactured RN-dS bounded outward fast median `0.283 ms`, p95 `0.303 ms`,
  fast segments `6/6` accepted.
- manufactured RN-dS bounded outward reference median `2.458 ms`, p95
  `4.312 ms`.
- real data path `/tmp/missing-li-data/nfw_L0.npz` skipped as expected.

Additional final checks:

- `git check-ignore -v` for the four README-linked docs now resolves to the
  explicit negation rules in `.gitignore`.
- `git ls-files -i -o --exclude-standard docs CHANGELOG.md src/spherical_raytracing/_segments.py src/spherical_raytracing/_turning.py`
  produced no output.
- `git diff --check` produced no output.

## Public API

The public API remains at 50 exports. No public export was added, removed, or
renamed in the final pass. New helpers `_segments.py` and `_turning.py` are
private implementation modules and are not exported through `__all__`.

## Scientific And Numerical Review

- L4 keeps scientific value as a deterministic manufactured-Schwarzschild
  convergence guard over stable observables: event classification, `phi_end`,
  `u_end`, representative `u_at(...)`, and stable disk intersections.
- L5 keeps scientific value as an analytical backend comparator across benign
  Schwarzschild, RN, and RN-dS cases. It does not over-claim ODE as a proof of
  Quad correctness and excludes near-critical and boundary-sensitive cases.
- Junction degeneration remains meaningful because identical metrics should
  preserve impact parameter and redshift matching invariants across the shell.
- The `TabulatedMetric` fast path remains numerically guarded by scalar
  reference comparisons, direct `RaySegment.u_at(...)` checks, bounded-domain
  checks, and fallback tests.
- Benchmark output is evidence for comparison, not a CI performance threshold.

## Warnings And Non-Blocking Risks

- Current warnings are known numerical warnings from quadrature in difficult
  ray categories and soft `ValidationGate`/loader warnings for finite tables or
  intentionally imperfect synthetic metadata. They are covered by passing tests
  and do not indicate a blocking delivery issue.
- A small number of tests exercise private helpers to compare vectorized and
  scalar turning-point behavior. That is an implementation guard for a numerical
  correctness path, not a public API contract.
- Real Li-DM-BH data remains optional; default tests and the benchmark command
  used here do not require private data.
- Historical `outputs/` remain untouched. Their eventual cleanup still requires
  human artifact classification.

## Files That Must Be Included When Staging

At minimum, commit staging must include the untracked delivery files that are
referenced by code, README, CI, tests, or review history:

- `.github/workflows/test.yml`
- `CHANGELOG.md`
- `REPO_OVERVIEW.md`
- `raytracing_spherical_review_roadmap.md`
- `scripts/benchmark_numerical_metric_perf.py`
- `src/spherical_raytracing/_segments.py`
- `src/spherical_raytracing/_turning.py`
- `tests/test_backend_comparator.py`
- `tests/test_benchmark_script_smoke.py`
- `tests/test_static_junction_degenerate.py`
- all new files under `docs/adr/`, `docs/plans/`, and `docs/review/`
- `docs/api-compatibility.md`
- `docs/artifacts.md`
- `docs/numerical-validation.md`
- `docs/physics-conventions.md`

## Final Decision

The stabilization plan is complete to the reasonable, non-breaking scope
requested. The repository can enter human review and commit staging. No blocking
scientific, numerical, API, ignore-rule, or clean-checkout issue was found in
this final pass.
