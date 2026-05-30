# Upstream Computation Core Stabilization Plan

Date: 2026-05-29

This is an execution plan for turning `spherical-raytracing` into a stable,
scientifically checkable upstream computation core. It is based on:

- `REPO_OVERVIEW.md`
- `docs/review/raytracing-spherical-review-roadmap.md`
- `docs/adr/0001-upstream-computation-core-api-contract.md`
- `docs/plans/numerical-metric-validation-plan.md`
- `docs/plans/numerical-metric-perf-plan.md`
- code-level validation of the current repository state

This is not a broad wishlist. It is the order in which stabilization work should
be executed.

## Core Priority Rule

Optimize in this order:

1. Scientific correctness
2. Downstream compatibility
3. Simplicity
4. Performance

P0 is about making the repository stable and credible. Performance acceleration
starts in P1.

## Repository Role

This repository is an upstream computation core for downstream projects that
reuse ray tracing, metric, thin-disk imaging, junction, and numerical-metric
semantics.

The downstream compatibility contract includes:

- public API exports
- public constructors and method shapes used by normal workflows
- result data structures and important fields
- physics conventions, including redshift and junction matching semantics
- numerical termination behavior

Internal implementation can change, but behavior-changing refactors must be
guarded by tests and justified by physical or numerical reasoning.

## Bug-Fix Policy During P0

P0 may fix bugs discovered by the new validation work, but only under these
conditions:

- the bug is reproduced by a new or existing test
- the behavior change explains the physical or numerical reason
- public API changes are forbidden unless covered by a separate ADR
- no drive-by formatting, renaming, cleanup, or speculative abstraction
- no unrelated performance optimization in the same change

## Current Findings To Respect

The review roadmap is directionally correct, with these important refinements:

- L1-L3 numerical metric validation already exists in
  `tests/test_numerical_metric_validation.py`; L4 and L5 are missing.
- The current package smoke test imports 25 of 50 public exports but does not
  lock the full public API contract.
- Numerical metric tests already create synthetic payloads dynamically. The
  remaining gap is reproducible coverage for paths that currently depend only on
  local real Li-DM-BH files.
- `scripts/benchmark_numerical_metric_perf.py` already runs and reports useful
  timing/category/fallback data. P0 should keep this as a baseline smoke, not
  spend time polishing benchmark format.
- Segment construction duplication between `solvers.py` and
  `junction_tracing.py` is real.
- `StaticJunctionSpacetime` currently supports analytical same-family junctions;
  it does not currently support `TabulatedMetric` junctions.
- `TabulatedMetric + Hamiltonian` is not a P0 target. It is a later decision
  point after validation is stronger.
- Repository hygiene matters before CI: planning docs may be ignored by
  `.gitignore`, and imported helper modules must not remain untracked when code
  that imports them is committed.

## P0: Stabilize The Contract And Scientific Guardrails

P0 should be implemented before structural refactors or speed work.

### P0.0 Confirm Commit Boundary And Repository Hygiene

Goal: prevent stabilization work from depending on ignored docs or untracked
runtime modules.

Expected files:

- possibly `.gitignore`
- possibly git index only, if files are force-added intentionally

Checks:

- run `git status --short --untracked-files=all`
- ensure every imported runtime module is tracked before CI is expected to pass
- ensure docs intended as durable project context are not silently ignored
- leave unrelated dirty files alone
- do not clean up tracked `outputs/` in P0

Current risk to verify before implementation:

- `docs/` may be ignored by `.gitignore`
- helper modules imported by committed code must not remain untracked

Acceptance criteria:

- implementation agents know which files are part of the intended change
- no committed code imports a missing untracked module
- plan/ADR/review docs that should be versioned are explicitly included or the
  ignore policy is consciously left as-is
- `outputs/` cleanup is deferred to P2 because ignoring already-tracked files
  does not reduce clone size

Verification:

```bash
git status --short --untracked-files=all
git ls-files docs/plans/upstream-computation-core-stabilization-plan.md
```

### P0.1 Add Medium Public API Contract Test

Goal: protect downstream callers without freezing every internal diagnostic.

Expected files:

- preferred: extend or rename `tests/test_package.py`
- acceptable: add `tests/test_public_api_contract.py` only if it does not
  duplicate a second hand-maintained API list

The test should lock `spherical_raytracing.__all__` to the current 50 exported
names:

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

It should also cover two public workflows:

1. Single-metric workflow:
   `metric -> observer -> QuadTransferSolver -> trace_b -> compute_intersections
   -> observed_intensity`.
2. Junction workflow:
   `StaticShell -> StaticJunctionSpacetime -> StaticJunctionTransferSolver
   -> JunctionRayResult`.

Acceptance criteria:

- all 50 exports are present and importable
- the existing 25-export smoke coverage is extended rather than duplicated
- key public result objects expose expected fields
- a representative `RaySegment.u_at(...)` call works
- no private helper names are locked
- no exact diagnostic strings are locked unless already part of documented API

Verification:

```bash
pytest tests/test_package.py -q
# If a separate contract file is created:
pytest tests/test_package.py tests/test_public_api_contract.py -q
```

### P0.2 Add Synthetic `.npz` Coverage Without Private Data

Goal: make numerical metric tests reproducible without relying on private local
data.

Expected files:

- preferred: `tests/test_numerical_metrics.py`
- preferred: `tests/test_numerical_metric_validation.py`
- optional only if a path-based script/benchmark needs a real file:
  `tests/fixtures/metrics/*.npz`
- optional if static fixtures are added: `tests/fixtures/metrics/README.md`

Fixture policy:

- dynamic synthetic payloads are preferred over checked-in binary fixtures
- standard payload coverage must include `r/A/B`
- Li-DM-BH-like payload coverage must include `r/f/g/B/params`
- real Li-DM-BH files remain optional smoke tests and should still skip if absent
- checked-in `.npz` files should be tiny and used only when a path-based
  workflow cannot be tested cleanly with `tmp_path`
- no new fixture should be written under `outputs/`

Acceptance criteria:

- `load_metric_npz` loads both synthetic styles
- `TabulatedMetric` can be built from both synthetic styles
- synthetic coverage exercises the same canonical payload path used by production
  code
- currently real-data-only smoke coverage has a synthetic equivalent where it
  protects core behavior
- tests pass without `/home/yuanpang/Work/Li-DM-BH`

Verification:

```bash
pytest tests/test_numerical_metrics.py -q
```

### P0.3 Add L4 Grid-Convergence Smoke And Slow Scaffold

Goal: start validating convergence behavior while keeping default pytest fast.

Expected files:

- preferred: extend `tests/test_numerical_metric_validation.py`
- acceptable if it grows too large: `tests/test_numerical_metric_convergence.py`
- `pyproject.toml`

Default pytest scope:

- Schwarzschild only
- fixed grid densities such as `800, 1600, 3200`
- fixed non-critical rays such as `0.70*b_crit`, `1.10*b_crit`,
  and `min(1.35*b_crit, 0.70*b_max)`
- compare `phi_end`
- compare `u_at(...)` at representative interior probes
- compare thin-disk intersection radii at fixed disk crossings when the
  intersection count and order are stable
- use explicit tolerances from an initial measured baseline, not vague
  "non-worsening" language

Slow validation scaffold:

- marked with `@pytest.mark.slow`
- may cover more grid densities and metric families later
- should not run in default quick tests unless explicitly selected
- marker registration in `pyproject.toml` is enough unless shared fixtures truly
  need `tests/conftest.py`

Exclusions:

- near-critical rays
- horizon-adjacent rays
- table-boundary rays
- `max_phi` truncation cases

Acceptance criteria:

- default test is small and deterministic
- convergence checks compare stable observables, not raw pointwise trajectories
- default smoke asserts fixed numerical tolerances on a stable manufactured
  Schwarzschild case
- slow marker is registered in `pyproject.toml`
- failures classify whether the issue is convergence, event classification, or
  fixture construction

Verification:

```bash
pytest tests/test_numerical_metric_validation.py -q
# If a separate convergence file is created:
pytest tests/test_numerical_metric_convergence.py -q
pytest -m slow tests/test_numerical_metric_validation.py tests/test_numerical_metric_convergence.py -q
```

### P0.4 Add L5 Analytical Backend Comparator

Goal: make backend agreement checkable on benign analytical cases.

Expected files:

- `tests/test_backend_comparator.py`

Comparator scope:

- strict: Schwarzschild
- warm coverage: Reissner-Nordstrom
- warm coverage: Reissner-Nordstrom-de Sitter
- finite static observer only
- two or three benign impact parameters per metric

Compare:

- termination reason
- segment endpoint event types
- segment count
- `phi_end`
- `u_end`
- two or three `u_at(...)` probes
- Hamiltonian residual or constraint diagnostic where available

Do not compare:

- full trajectory point-by-point
- near-critical cases
- boundary-sensitive cases
- `max_phi` cases
- `TabulatedMetric + Hamiltonian`

Acceptance criteria:

- comparator clearly states that ODE is not a fully independent oracle
- failures point to the backend and observable that diverged
- Schwarzschild cases are strict enough to catch obvious regressions
- RN and RN-dS cases provide warm coverage without over-claiming support

Verification:

```bash
pytest tests/test_backend_comparator.py -q
```

### P0.5 Add Identical-Metric Junction Degeneration Test

Goal: protect the physical meaning of static shell matching.

Expected files:

- `tests/test_static_junction_degenerate.py`

Scenario:

- inner and outer metrics are identical
- shell crossing should not change energy ratio or impact parameter
- redshift through the junction helper should reduce to the direct finite
  observer redshift with no extra shell factor
- a minimal direct-intersection smoke may compare single-metric and junction
  intensity only when the selected ray has stable matching intersections

Acceptance criteria:

- energy ratio is 1
- `b_after == b_before` within numerical tolerance
- `junction_redshift_factor` has no extra shell product in this degenerate case
- direct intersection strength agrees with single-metric finite-observer result
  only for a stable matched case
- exact event sequence equality is not required
- no broad imaging comparison is required in P0

Verification:

```bash
pytest tests/test_static_junction_degenerate.py -q
```

### P0.6 Keep Benchmark Baseline Runnable

Goal: make performance discussions measurable before optimizing.

Expected files:

- `scripts/benchmark_numerical_metric_perf.py`
- optional test: `tests/test_benchmark_script_smoke.py`

P0 required benchmark fields:

- metric or fixture name
- ray count
- ray category counts
- median timing
- p95 timing
- fallback count or fallback-compatible category if applicable

Nice-to-have fields such as run date, Python/NumPy/SciPy versions, grid size, and
total timing may be added in P1 when performance work starts. P0 does not
require hard timing thresholds.

Acceptance criteria:

- script runs on manufactured synthetic metrics
- script does not require private Li-DM-BH data
- output is stable enough to compare before/after changes
- timing thresholds are not enforced in CI
- do not spend P0 time on benchmark format polish beyond what is needed for a
  smoke run

Verification:

```bash
python scripts/benchmark_numerical_metric_perf.py
pytest tests/test_benchmark_script_smoke.py -q
```

### P0.7 Add Minimal CI

Goal: ensure the public contract and validation smoke tests run in a clean
environment.

Expected files:

- `.github/workflows/test.yml`

Scope:

- install package with test extras
- run `pytest -q`
- no ruff
- no coverage threshold
- no mypy
- no private data dependency

Acceptance criteria:

- CI uses Python 3.11 or a small 3.11/3.12 matrix
- all default tests pass without local absolute paths
- slow tests are not required in CI unless explicitly configured later

Verification:

```bash
pytest -q
```

## P0 Completion Definition

P0 is complete only when:

- commit boundary and ignored/tracked-file risks are explicit
- `pytest -q` passes
- the public API contract test passes
- synthetic numerical metric coverage covers both supported payload styles
- L4 default smoke exists and is fast
- L5 analytical comparator exists
- identical-metric junction degeneration is tested
- benchmark script runs without private data
- minimal CI runs default pytest
- no public API has changed without a separate ADR

## P1: Simplify Internals And Begin Measured Speed Work

P1 starts after P0 passes.

### P1.1 Extract Shared Segment Integration/Evaluator Kernel

Goal: reduce duplication between single-metric and junction tracing while
preserving behavior.

Expected files:

- `src/spherical_raytracing/solvers.py`
- `src/spherical_raytracing/junction_tracing.py`
- candidate new private module: `src/spherical_raytracing/_segments.py`
- tests that should remain green:
  - `tests/test_package.py`
  - `tests/test_static_junctions.py`
  - `tests/test_static_junction_backend_agreement.py`
  - `tests/test_backend_comparator.py`

Scope:

- extract segment integration and `phi -> u` evaluator construction only
- preserve public `RaySegment` and `JunctionRayResult`
- preserve endpoint event semantics
- keep diagnostics builders local unless duplication becomes proven

Non-goals:

- no public API changes
- no performance fast path in this extraction
- no junction support for `TabulatedMetric`
- no rewrite of event/diagnostic models

Acceptance criteria:

- behavior-preserving refactor
- before/after P0 tests match
- segment endpoint and `u_at(...)` behavior remain stable
- code duplication is reduced in the segment construction path

Verification:

```bash
pytest tests/test_package.py tests/test_backend_comparator.py -q
pytest tests/test_static_junctions.py tests/test_static_junction_backend_agreement.py -q
pytest -q
```

### P1.2 Improve Single-Metric `TabulatedMetric` Fast Path And Benchmark

Goal: speed up numerical metric tracing only after scalar behavior is guarded.

Expected files:

- `src/spherical_raytracing/numerical_metrics.py`
- `src/spherical_raytracing/solvers.py`
- `scripts/benchmark_numerical_metric_perf.py`
- candidate tests: `tests/test_numerical_metric_fast_path.py`

Scope:

- single-metric `TabulatedMetric` only
- vectorized metric evaluation may be used internally
- guarded fast path must fail closed to scalar reference behavior
- benchmark must report both speed and fallback rate

Non-goals:

- no junction fast path in this phase
- no public API or `SolverOptions` expansion unless unavoidable and justified
- no `TabulatedMetric + InfinityObserver`
- no `TabulatedMetric + ThroughTracePolicy`
- no `TabulatedMetric + Hamiltonian`

Acceptance criteria:

- fast path matches scalar reference on P0 stable cases
- difficult rays fall back instead of silently clamping
- benchmark reports before/after speed and fallback categories
- `RaySegment.u_at(...)` accuracy is directly tested

Verification:

```bash
pytest tests/test_numerical_metric_validation.py tests/test_numerical_metric_fast_path.py -q
python scripts/benchmark_numerical_metric_perf.py
```

### P1.3 Add Light Engineering Guardrails And Benchmark Polish

Goal: add basic quality checks after the scientific guardrails exist.

Expected files:

- `pyproject.toml`
- optional: `.github/workflows/test.yml`
- optional: `scripts/benchmark_numerical_metric_perf.py`

Scope:

- ruff or equivalent lightweight linting may be introduced
- coverage reporting may be introduced without a hard threshold at first
- benchmark output may gain run date, Python/NumPy/SciPy versions, grid size, and
  total timing if useful for comparing P1 optimization work
- keep config minimal

Non-goals:

- no mypy requirement yet
- no `py.typed` yet
- no large formatting churn
- no internal batch tracing until profiling shows it is worth the complexity

Acceptance criteria:

- lint config matches existing style with minimal churn
- CI remains fast
- no unrelated code is reformatted
- benchmark output changes are backward-compatible enough for manual comparison

## P2: Governance, Documentation, And Larger Decisions

P2 should not block P0/P1.

### P2.1 Classify Artifacts And Outputs

Goal: make generated data and historical artifacts understandable.

Expected files:

- `docs/artifacts.md`
- `.gitignore`
- optional updates under `outputs/`

Classify artifacts as:

- checked-in examples
- regression fixtures
- generated outputs
- historical references
- external/private data

P0 only prevents new fixtures from going under `outputs/`; full cleanup waits
until P2.

### P2.2 Decide `TabulatedMetric + Hamiltonian`

Goal: decide whether Hamiltonian support for numerical metrics is production,
diagnostic-only, or unsupported.

Expected output:

- ADR under `docs/adr/`
- tests only after the decision is made

Decision inputs:

- P0 L5 analytical comparator results
- P1 scalar/fast-path behavior
- derivative robustness of `TabulatedMetric`
- constraint drift on representative cases

### P2.3 Document Physics And Numerical Conventions

Goal: make scientific assumptions explicit for downstream users.

Expected files:

- `docs/physics-conventions.md`
- `docs/numerical-validation.md`
- README updates if needed

Topics:

- coordinate conventions
- areal radius
- redshift weights
- finite vs infinity observers
- static shell matching
- termination event semantics
- near-critical exclusion policy
- numerical table support boundaries

### P2.4 Add Release And Typing Policy

Goal: make downstream compatibility manageable over time.

Expected files:

- `CHANGELOG.md`
- `docs/api-compatibility.md`
- optional `src/spherical_raytracing/py.typed`

Scope:

- deprecation policy
- versioning expectations
- public API change process
- typing support decision

### P2.5 Decide Whether Internal Batch Tracing Is Worth It

Goal: avoid adding batch tracing complexity before benchmark data shows that
imaging/profile repeated scalar calls are a real bottleneck.

Expected files if approved later:

- `src/spherical_raytracing/transfer.py`
- `src/spherical_raytracing/rendering.py`
- candidate private module: `src/spherical_raytracing/_batch.py`
- candidate tests: `tests/test_batch_tracing_internal.py`

Decision inputs:

- P1 benchmark data on `sample_radial_profile` and `render_axisymmetric_image`
- current single-ray solver cost after `TabulatedMetric` fast-path work
- expected downstream workload size

If approved:

- keep helper private
- add no new `__all__` export
- preserve scalar public outputs
- require benchmark evidence of meaningful speedup

## No-Go List

Do not do these as part of P0 or early P1:

- do not rename public API symbols
- do not remove public exports
- do not add broad plugin/configuration systems
- do not introduce JAX, GPU, C++, or other heavy dependencies
- do not rewrite dataclasses into a large inheritance hierarchy
- do not make `TabulatedMetric` silently extrapolate to infinity
- do not treat ODE agreement as proof that Quad is correct
- do not require private Li-DM-BH data in default tests or CI
- do not enforce performance timing thresholds in CI during P0
- do not clean up `outputs/` until artifact policy exists
- do not add internal batch tracing until profiling justifies it

## Recommended Execution Order

1. P0.0 commit boundary and repository hygiene
2. P0.1 API contract test
3. P0.2 synthetic numerical metric coverage without private data
4. P0.3 L4 convergence smoke and slow scaffold
5. P0.4 L5 analytical backend comparator
6. P0.5 identical-metric junction degeneration test
7. P0.6 benchmark baseline smoke
8. P0.7 minimal CI
9. P1.1 shared segment integration/evaluator kernel
10. P1.2 single-metric `TabulatedMetric` fast path
11. P1.3 light engineering guardrails and benchmark polish
12. P2 governance and larger decisions

## Handoff Prompt For An Implementation Agent

Use this prompt when asking another agent to implement P0:

```text
Please implement P0 from
docs/plans/upstream-computation-core-stabilization-plan.md.

Hard constraints:
- Preserve the current public API.
- Start with tests and keep each change small.
- P0 may fix bugs found by validation, but every behavior change must be
  reproduced by a test and explained physically or numerically.
- Do not do performance optimization in P0.
- Do not require private Li-DM-BH data in default tests or CI.
- Do not rename exports, result fields, or public constructors.
- Do not refactor unrelated code.
- Do not add internal batch tracing in P0/P1 unless benchmark evidence first
  proves it is the right bottleneck.
- Before implementation, check ignored/untracked files so durable docs and
  imported runtime modules are not accidentally omitted.

Priority order:
scientific correctness > downstream compatibility > simplicity > performance.

Stop after P0 is complete and report:
- tests added
- files changed
- any behavior changes and their tests
- commands run
- remaining risks before P1
```
