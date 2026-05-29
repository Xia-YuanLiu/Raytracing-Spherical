# Numerical Metric Ray Tracing Performance Optimization Plan

## Goal

Tracing 500 rays on `TabulatedMetric` (`nfw_L0`, about 2000 grid points) currently
takes about 1.4 s serial, compared with about 0.025 s for analytical
`SchwarzschildMetric`. The goal is to remove avoidable scalar PCHIP overhead while
preserving numerical behavior, keeping the public API stable, and retaining the
current quadrature implementation as the correctness oracle and fallback.

## Current Architecture Constraints

- `TabulatedMetric` stores `A(r)` and `C(r)=1/B(r)` in `PchipInterpolator`
  instances with `extrapolate=False`. Scalar `G(u, b)` hard-fails for `b <= 0`,
  `u <= 0`, `r=1/u` outside the static domain, or `r` beyond the finite tabulated
  grid support. Any vectorized metric path must preserve these guard semantics.
- `QuadTransferSolver._make_segment()` is not just an integration routine. It
  creates the `RaySegment.evaluator` used by `RaySegment.u_at()`, which is then
  used by `compute_intersections()`, observed intensity, and image generation.
  Accuracy of the inverse map `phi -> u` is therefore user-visible, not merely
  diagnostic.
- `TabulatedMetric + InfinityObserver` and `TabulatedMetric + ThroughTracePolicy`
  are intentionally unsupported. For unbounded domains backed by finite tables,
  the solver must not integrate numerical metrics to `u=0`; current escape rays
  use the mirror escape path.
- `StaticJunctionTransferSolver` has its own `_first_turning_point()` and
  `_make_segment()` implementations. It does not automatically share the
  single-metric `QuadTransferSolver` fast path.
- `OdeTrajectorySolver` calls `QuadTransferSolver` for the preliminary trace and
  outbound turning-point leg, so changes to the quadrature path can affect ODE
  behavior indirectly.
- `scripts/generate_numerical_metric_images.py` uses
  `SolverOptions(critical_exclusion=0.0)`, so validation must include near-critical
  rays instead of assuming they are excluded.

## Profiling Summary

The dominant cost is scalar `PchipInterpolator.__call__` overhead:

| Component | Turning-point ray | Horizon/captured ray |
|---|---:|---:|
| `_first_turning_point` | about 1.3 ms | about 0.4 ms |
| `_make_segment` / `quad` | about 0.3 ms | about 8.1 ms |
| overhead | about 0.02 ms | about 0.02 ms |
| total `trace_b` | about 1.6 ms | about 10.5 ms |

Representative measurement:

| Method | 1000 G evaluations | Per value |
|---|---:|---:|
| Scalar `metric.G(u, b)` | about 4.4 ms | about 4.4 us |
| Vectorized PCHIP evaluation | about 0.07 ms | about 0.07 us |

The performance opportunity is real, but the fast path must be guarded because
PCHIP extrapolation, turning-point singularities, horizon endpoints, and `u_at()`
accuracy are all part of the current behavioral contract.

## Design Principles

1. Keep the existing scalar quadrature implementation as a private reference path.
   The fast path must be able to fall back to it per segment.
2. Add no public API surface in the first implementation. Prefer private helpers
   and module-level caches over new `SolverOptions` fields.
3. Preserve scalar failure semantics in vectorized helpers. Silent `nan` from PCHIP
   is not acceptable.
4. Optimize only `TabulatedMetric` initially. Analytical metrics stay on the
   current path unless a later benchmark shows a separate need.
5. Treat `RaySegment.u_at()` as correctness-critical. Any approximation to
   `phi -> u` needs direct tests against the reference evaluator.
6. Report fallback rates in benchmarks. A fast path that silently falls back on
   difficult rays is acceptable; a fast path that silently clamps through invalid
   regions is not.

## Phase 0: Stabilize the Reference and Benchmark Harness

Before changing algorithms, refactor enough of the current implementation to make
comparisons easy.

1. Keep the current `_make_segment()` logic available as a private reference helper
   such as `_make_segment_quad_reference()`.
2. Keep the current scalar turning scan available as a private helper such as
   `_first_turning_point_scalar()`.
3. Add a small benchmark script, for example
   `scripts/benchmark_numerical_metric_perf.py`, that reports median/p95 timings
   and ray category counts for:
   - captured rays below `b_crit`
   - ordinary turning rays above `b_crit`
   - near-critical rays with `critical_exclusion=0.0`
   - bounded-domain outward rays
   - `RaySegment.u_at()` probes used by disk intersections
4. Use the same real Li-DM-BH files already exercised in tests, plus manufactured
   Schwarzschild/RN/RNdS tables where analytical ground truth exists.

Acceptance:

- No behavior change.
- Existing tests pass.
- Benchmark output records timing, termination reasons, and fallback-compatible
  categories.

## Phase 1: Vectorized Turning-Point Search

This is the lowest-risk high-value change.

### Implementation

1. Add a private vector method to `TabulatedMetric`, for example
   `_G_array(u_values, b)`.
2. `_G_array` should:
   - require `b > 0`
   - require every `u > 0`
   - map to `r = 1/u`
   - reject values outside `valid_radial_domain()` or outside
     `[self._r_grid_min, self._r_grid_max]`
   - evaluate `_A_pchip` and `_C_pchip` once each on the full array
   - raise if interpolation returns non-finite values
3. Add a shared internal helper for turning-point scans that accepts
   `(metric, b, u_start, u_stop, options)` and uses `_G_array` only when the
   metric is `TabulatedMetric` and the scan interval is valid for the table.
4. Preserve the current scan semantics:
   - use `options.max_brackets` (currently 512 by default), not a hard-coded 200
   - preserve scan direction for junction rays
   - detect the first transition `previous_g > 0` and `current_g <= 0`
   - refine with the existing scalar `metric.G` inside `brentq`
5. Use the helper from both `QuadTransferSolver` and, if numerical junctions are
   in scope, `StaticJunctionTransferSolver`.

### Tests

- `_G_array` equals scalar `G` on representative arrays.
- `_G_array` raises for the same out-of-support cases as scalar `G`.
- Vectorized and scalar turning searches return the same roots for:
  - no root
  - captured rays
  - ordinary turning rays
  - descending scan direction used by junction code
  - finite bounded domains

Expected impact:

- Turning-point rays should improve substantially, likely cutting the 500-ray
  serial benchmark from about 1.4 s to roughly 0.3-0.5 s depending on ray mix.

## Phase 2: Guarded Vectorized Gauss-Legendre Segment Widths

Replace scalar `quad` only when the segment is demonstrably safe for a vectorized
fixed-order rule. Otherwise fall back to the reference helper.

### Implementation

1. Add internal cached Gauss-Legendre nodes/weights at module scope, initially for
   two orders such as 32 and 64. Do not add public `SolverOptions` fields yet.
2. Implement a private fast segment-integral helper for `TabulatedMetric`:
   - direct finite-interval integration for non-turning segments
   - current transformed-coordinate integration for inbound turning segments
   - current transformed-coordinate integration for outbound turning segments
3. For each attempted fast integral:
   - evaluate all GL nodes through `_G_array`
   - reject and fall back if any `G` is non-finite
   - reject and fall back if any sampled `G` is negative beyond a tiny roundoff
     tolerance
   - avoid `max(G, eps)` as the primary strategy; clamping can hide forbidden
     regions and alter horizon/captured-ray angles
   - compare 32-node and 64-node results and accept only when the difference is
     below a tolerance derived from `quad_epsabs` / `quad_epsrel`
4. Keep `angle_to_u(phi)` exact enough by reusing `brentq`, but have its residual
   call the same guarded fast integral from `u0` to the target. If any residual
   evaluation fails the fast checks, use the reference evaluator for that segment.
5. Return the GL pair difference as `estimated_error` for accepted fast segments.
   For fallback segments, return the original `quad` error estimate.

### Horizon and Captured Rays

Captured rays are still physically meaningful for termination, diagnostics, and
image scripts. They are not "diagnostics only". Therefore:

- do not accept a fast result merely because the final event is a horizon
- do not clamp through `G < 0` samples
- fall back to `quad` near horizons unless the GL pair check passes cleanly
- include captured rays below `b_crit` in fast-vs-reference tests

### Tests

Compare fast-vs-reference `RayResult` values for:

- `phi_end`, `u_end`, endpoint events, and termination reason
- `RaySegment.u_at()` at several interior `phi` probes
- captured rays below `b_crit`
- turning rays above `b_crit`
- `MAX_PHI` truncated rays
- bounded-domain outward rays
- real Li-DM-BH smoke files

Use tight tolerances for ordinary rays and explicit fallback expectations for
near-critical or horizon-adjacent cases. The fast path should fail closed.

Expected impact:

- Segment-width integration can become 10-30x faster on accepted rays.
- Overall benchmark speedup should be reported together with fallback rate. A
  realistic target is 10-30x for easy tabulated rays; difficult horizon and
  near-critical rays may remain closer to the scalar reference cost.

## Phase 3: Optional Fast `RaySegment.u_at()` Evaluator

The original plan proposed a cumulative `phi(u)` table built from GL nodes and
inverted with `np.interp`. That should not be treated as low risk. GL nodes are
excellent for a total integral, but a trapezoidal cumulative sum over sparse GL
nodes is a different lower-order approximation and can move disk intersections.

Only attempt this after Phase 2 is correct and benchmarked.

Safer options:

1. Build a monotone cumulative table on an adaptive panel grid in the same
   transformed coordinate used by the integral.
2. Use GL per panel for cumulative increments instead of trapezoids over raw GL
   nodes.
3. Invert with monotone interpolation only after validating random `u_at()` probes
   against the reference `brentq + quad` evaluator.
4. Keep the Phase 2 brentq evaluator as fallback.

Acceptance:

- `compute_intersections()` radii agree with the reference path within existing
  numerical metric validation tolerances.
- Rendered radial profiles and images agree with reference outputs within an
  explicit tolerance.
- Benchmarks show that `u_at()` calls are a meaningful remaining cost.

If `u_at()` is not a major bottleneck after Phase 2, skip this phase.

## Phase 4: Batch Imaging Path

After the single-ray fast path is stable, consider a separate batch API for image
generation.

Possible design:

1. Keep existing `trace_b`, `trace_alpha`, and `trace_screen_radius` unchanged.
2. Add an optional internal or public batch layer only after profiling shows the
   single-ray path has plateaued.
3. Batch only homogeneous ray categories where the same transformed-coordinate
   strategy applies.
4. Preserve per-ray fallback to the reference solver.

This phase is larger than a local optimization because it changes orchestration in
the imaging pipeline. It should be handled separately from the safe fast path.

## Validation Matrix

Use three levels of validation.

### Unit

- `_G_array` vs scalar `G`
- scalar vs vector turning-point root
- GL pair acceptance and fallback behavior
- no silent PCHIP extrapolation

### Solver

- manufactured Schwarzschild, RN, and RNdS tables compared to analytical metrics
- real Li-DM-BH `.npz` smoke files
- captured, turning, near-critical, bounded outward, and truncated rays
- `RaySegment.u_at()` probes

### Imaging

- existing thin-disk intersection tests
- one representative real numerical metric profile
- rendered image comparison against the reference path with documented tolerance
- fallback rate and timing in the manifest or benchmark output

## Compatibility Notes

- Public API remains unchanged in Phases 0-3.
- Analytical metrics stay on the current solver path.
- `TabulatedMetric + InfinityObserver` remains unsupported.
- `TabulatedMetric + ThroughTracePolicy` remains unsupported unless a separate
  numerical-internal-region design is added.
- Junction optimization requires either shared helpers or explicit changes in
  `junction_tracing.py`; it is not automatic.

## Updated Risk Assessment

| Risk | Mitigation |
|---|---|
| Vectorized helper silently accepts out-of-support values | Preserve scalar guard semantics and raise on non-finite PCHIP output |
| Fixed-order GL inaccurate near horizons or critical curves | Use GL pair residual checks and fallback to reference `quad` |
| Clamping changes captured-ray angles | Do not accept clamp-only fast results; fallback on negative/non-finite `G` |
| `u_at()` interpolation moves disk intersections | Keep brentq evaluator first; make cumulative table optional and separately validated |
| Junction solver not optimized | Use shared helpers or explicitly patch `junction_tracing.py` |
| Public API drift through new options | Use private helpers and module-level caches in the first pass |

## Recommended Implementation Order

1. Phase 0: refactor reference helpers and add benchmark harness.
2. Phase 1: `_G_array` plus vectorized turning-point scan.
3. Run full numerical metric validation and benchmark.
4. Phase 2: guarded GL segment widths with fallback.
5. Run validation matrix and benchmark with fallback rates.
6. Decide whether Phase 3 is worth implementing based on measured `u_at()` cost.

