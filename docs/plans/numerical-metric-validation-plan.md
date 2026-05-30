# Numerical Metric Ray Tracing Validation Plan

## Core Idea

**Manufactured Solution Validation**: sample known analytical metrics (Schwarzschild, RN, RN-dS)
on grids to create "synthetic numerical metrics", trace identical rays through the
analytical metric path and the tabulated metric path, and compare results
point-by-point.

The analytical metric path is the ground truth. The tabulated path should
converge toward it as grid density increases. Discrepancies are expected to come
from a mix of PCHIP interpolation, turning-point root finding, quadrature, and
`max_phi`/termination classification near sensitive boundaries; the validation
should separate those effects instead of assuming every difference is purely
interpolation error.

## Current State

The repo already has one test of this kind:
- `test_numerical_metrics.py::test_tabulated_metric_schwarzschild_round_trip_matches_analytic_phi_within_1e8`
- Tests 5 b values on Schwarzschild with `r_obs=100`, 4000-point grid
- Asserts `rel_err(phi_end) < 1e-8`

This is a good start but has gaps:
- Only Schwarzschild, only one metric family
- Only checks `phi_end`, not `u_end` or the `u(phi)` evaluator
- Only 5 impact parameters, no systematic sweep
- Doesn't test disk intersections (the actual imaging output)
- Doesn't test near-critical rays (most sensitive to interpolation error)
- Doesn't quantify how grid density affects accuracy
- Doesn't explicitly check `C = 1/B`, even though `TabulatedMetric` interpolates
  `A(r)` and `C(r)` rather than `B(r)` directly
- Doesn't cover both metadata-rich payloads and bare standard `r/A/B` payloads
  where horizons/photon spheres must be inferred

## Proposed Validation Framework

### Level 1: Metric Component Validation

Verify that the interpolated metric functions match the analytical originals.

```python
def validate_metric_components(metric_analytical, metric_numerical, r_test_points):
    """Compare metric components and derived roots point by point."""
    # A(r) relative error
    # C(r) = 1/B(r) relative error (primary interpolated quantity)
    # dA/dr and dC/dr relative error away from discontinuities / roots
    # B(r), dB/dr only away from horizons where |C| is comfortably above c_tol
    # G(u, b) relative error for several (u, b) pairs
    # Horizon positions (if metadata provided)
    # Critical impact parameter b_crit
    # Photon sphere position r_ph
```

This is the cheapest check and catches most interpolation issues early.
Run this level for two manufactured payload styles:

1. **Metadata-rich Li-DM-BH style**: includes `static_domains`, `horizons`,
   and `photon_spheres`; validates metadata priority and horizon anchoring.
2. **Bare standard `r/A/B` style**: no metadata roots; validates root inference
   and guards against fake zero anchors at non-horizon grid/support boundaries.

### Level 2: Ray-by-Ray Validation

Trace identical rays with the same solver backend but two metric
implementations: analytical metric vs manufactured `TabulatedMetric`.

**Metrics to compare per ray:**
1. **Termination reason** — must match exactly outside declared exclusion zones
2. **Segment count** — must match
3. **Segment endpoint events** — each segment should end at the same event type
4. **phi_end** — relative error < tolerance
5. **u_end** — relative error < tolerance (for horizon/boundary rays, compare absolute)
6. **u(phi) evaluator** — at several phi values along each segment (this is what
   determines disk intersection positions)

**Test matrix:**

| Metric | Q/Λ | Grid density | r_obs | b range |
|---|---|---|---|---|
| Schwarzschild | — | 500, 1000, 2000 | 50, 100 | 0.3 b_crit → 0.9 b_max |
| RN | 0.3, 0.5, 0.7, 0.9, 0.99 | 500 | 10-20 | 0.3 b_crit → 0.9 b_max |
| RN-dS | Λ=0.01, Q=0, 0.5 | 500 | 6-10 | 0.5 b_crit → 0.95 b_max |

**Near-critical rays** (most sensitive):
- First validate `r_ph` and `b_crit` directly in L1.
- For ray classification, do not sample inside the uncertainty band
  `abs(b - b_crit) / b_crit <= max(critical_exclusion, 5 * rel_err(b_crit))`;
  tiny interpolation shifts in `b_crit` can otherwise flip classification while
  the implementation is still correct.
- Use separate subcritical and supercritical sweeps, e.g. `0.95 -> 0.999` and
  `1.001 -> 1.1` times each metric implementation's own `b_crit`, plus a few matched absolute
  `b` values that are safely outside the uncertainty band.
- Mark true `near_critical` returns as a deliberate solver exclusion, not a
  failed ray trace.
- These rays orbit many times near the photon sphere, amplifying interpolation errors

### Level 3: Disk Intersection Validation

For imaging, the critical output is where rays cross the thin disk
(`phi = pi/2 + n*pi`). In the current implementation, `phi` itself is fixed by
`disk_crossing_angles`; the computed observable is the radius (and downstream
redshift/intensity), not a solved `phi`.

```python
def validate_disk_intersections(metric_a, metric_n, b_values, disk_window, source):
    """Compare thin-disk intersection points between analytical and numerical."""
    for b in b_values:
        intersections_a = compute_intersections(trace_a(b), disk_window)
        intersections_n = compute_intersections(trace_n(b), disk_window)
        observed_a = observed_intensity(intersections_a, source, metric_a, observer_a)
        observed_n = observed_intensity(intersections_n, source, metric_n, observer_n)

        # Compare: same number of intersections, same order m
        # Compare: same region/path_class, matching termination diagnostics
        # For each intersection: r_rel_err / r_abs_err
        # For imaging: redshift_weight_rel_err and contribution_rel_err
```

This is the most meaningful validation for the actual imaging pipeline.

### Level 4: Grid Convergence Study

As grid density increases, numerical results should converge to analytical, but
max error is not guaranteed to be strictly monotone. PCHIP shape preservation,
turning-point bracketing, near-critical dwell time, and `max_phi` truncation can
move the single worst ray around between grids.

```
n_grid:  100 -> 200 -> 500 -> 1000 -> 2000 -> 5000
Expected p95 rel_err: roughly decreases with grid density
Expected max rel_err: stays within a shrinking envelope, except excluded rays
```

This validates that:
1. PCHIP interpolation/root-finding error is the dominant observed error source
2. No systematic bias or grid-edge bug is hiding in the pipeline
3. Accuracy improves on stable rays before spending time on real exotic metrics

Acceptance should use:
- A stable ray set with unchanged termination and segment count across grids
- Median / p95 trends for convergence
- Max error only as an envelope check
- A finest-grid absolute acceptance threshold for representative observables

### Level 5: Backend Sanity Checks

For numerical metrics where no analytical solution exists, use backend checks as
sanity checks, not as a proof of correctness.

Important caveat: the existing `trace_with_backend_compare` is for static
junction spacetimes, not a single numerical metric. A single-metric comparator
should be added if this level is implemented.

1. **Quad transfer vs ODE/Hamiltonian**: Compare `QuadTransferSolver`,
   `OdeTrajectorySolver`, and, where stable, `HamiltonianTrajectorySolver` on
   the same metric and observer.
2. **Independence caveat**: `OdeTrajectorySolver` currently delegates setup and
   some outbound behavior to `QuadTransferSolver`, and both escape paths may use
   mirror symmetry. Agreement is useful but not fully independent.
3. **Conserved quantity monitoring**: Hamiltonian H ≈ 0, energy E = const, L = const
4. **Symmetry checks**: For Schwarzschild, verify that the mirror-escape segment
   is the time-reverse of the inbound segment

### Tolerance Strategy

Based on profiling results with Schwarzschild n=1000 grid:

| Quantity | Expected accuracy | Recommended tolerance |
|---|---|---|
| Termination reason | exact match | must match |
| Segment count | exact match | must match |
| phi_end (median) | ~1e-7 | 1e-4 |
| phi_end (max) | ~1e-3 | 1e-3 |
| u(phi) evaluator (median) | ~1e-6 | 1e-4 |
| u(phi) evaluator (max) | ~1e-3 | 1e-2 |
| b_crit | ~1e-11 | 1e-6 |
| r_ph | ~1e-12 | 1e-6 |

The large gap between median and max comes from rays whose turning points
fall in sparse grid regions. Max-error tolerance should be treated as an
envelope check, not as a strict convergence statistic.
Do not apply these tolerances to rays inside the near-critical exclusion band,
rays whose segment count differs because one path hits `max_phi`, or queries
outside the finite tabulated support.

For the manufactured validation to work cleanly:
- Use n ≥ 1000 grid points (geometric spacing: dense near horizon, sparser at large r)
- Use `r_obs` well inside the grid support
- Exclude rays with `b > 0.9 * b_max` for routine tests (close to screen edge);
  screen-edge behavior can be covered by a separate explicit boundary test
- For unbounded analytical domains represented by finite grids, keep
  `r_obs < r_grid_max` and never require `TabulatedMetric.G(u=0, b)`

## Implementation Plan

### File: `tests/test_numerical_metric_validation.py`

```python
"""Comprehensive validation of numerical metric ray tracing against analytical ground truth.

Strategy: sample analytical metrics on grids to create manufactured numerical metrics,
then compare ray tracing results point-by-point.

Levels:
  L1: Metric component accuracy (A, C, G, horizons, b_crit)
  L2: Ray-by-ray validation (phi, u, termination, segments)
  L3a: Disk intersection validation
  L3b: Redshift/intensity observable validation
  L4: Grid convergence (optional, slow)
  L5: Backend sanity checks (optional, slow)
"""
```

### Helper: manufactured metric builder

```python
def manufacture_numerical_metric(analytical_metric, r_lo, r_hi, n_points,
                                  r_obs=None, payload_style="metadata") -> tuple[TabulatedMetric, float]:
    """Sample an analytical metric on a grid and return (TabulatedMetric, recommended_r_obs).

    Uses geometric spacing: dense near the inner boundary, sparser at large r.

    payload_style="metadata" provides horizons/photon_spheres/static_domains.
    payload_style="standard-rab" omits metadata roots to exercise inference.
    """
```

### Test functions

```python
# ── L1: Component validation ──────────────────────────────────────
@pytest.mark.parametrize("metric_cls,kwargs", [
    (SchwarzschildMetric, {"mass": 1.0}),
    (ReissnerNordstromMetric, {"mass": 1.0, "charge": 0.5}),
    (ReissnerNordstromMetric, {"mass": 1.0, "charge": 0.9}),
])
@pytest.mark.parametrize("payload_style", ["metadata", "standard-rab"])
def test_l1_metric_components_match_analytical(metric_cls, kwargs, payload_style):
    ...

def test_l1_standard_rab_infers_roots_without_fake_boundary_anchors():
    ...

# ── L2: Ray-by-ray validation ─────────────────────────────────────
@pytest.mark.parametrize("metric_cls,kwargs", METRIC_PARAMS)
def test_l2_termination_reasons_match(metric_cls, kwargs):
    """Every non-excluded ray must classify the same way (horizon/escape/turning)."""

def test_l2_segment_counts_match():
    ...

def test_l2_phi_end_within_tolerance():
    ...

def test_l2_u_evaluator_within_tolerance():
    """The u(phi) evaluator (used for disk intersections) matches analytical."""

# ── L2: Near-critical rays ────────────────────────────────────────
def test_l2_near_critical_rays_classify_correctly():
    """Near-critical rays agree outside the explicit b_crit uncertainty band."""

# ── L3a: Disk intersections ───────────────────────────────────────
def test_l3_disk_intersections_match_analytical():
    """Compare thin-disk intersection count/order/region/path_class/r."""

# ── L3b: Imaging observables ──────────────────────────────────────
def test_l3_observed_intensity_matches_analytical():
    """Compare redshift weights and source-weighted contributions."""

# ── L4: Grid convergence (optional) ───────────────────────────────
@pytest.mark.slow
def test_l4_grid_convergence_improves_on_stable_ray_set():
    """Median/p95 errors decrease on rays with unchanged classification."""

# ── L5: Backend sanity checks (optional) ───────────────────────────
@pytest.mark.slow
def test_l5_single_metric_backend_sanity():
    """Compare Quad/ODE/Hamiltonian where each backend is supported."""
```

## What This Validates

| Failure mode | Caught by |
|---|---|
| PCHIP interpolation bug | L1 (component check) |
| Interpolating/validating the wrong radial component (`B` vs `C`) | L1 |
| Wrong horizon position | L1 (horizon comparison) |
| Wrong b_crit | L1 (critical curve comparison) |
| Fake boundary horizon from standard `r/A/B` payload | L1 |
| Turning point at wrong radius | L2 (phi/u mismatch) |
| Wrong termination classification | L2 (termination reason mismatch) |
| Disk intersection at wrong r | L3 (intersection comparison) |
| Redshift/intensity regression | L3b (observed intensity comparison) |
| Grid too coarse for accuracy | L4 (convergence failure) |
| Regressions from code changes | All levels (CI) |

## Complementary Approaches

If manufactured metrics are not sufficient (e.g., for exotic numerical metrics
with no analytical analogue):

1. **Backend comparison**: Add a single-metric comparator for `QuadTransferSolver`
   vs `OdeTrajectorySolver` vs `HamiltonianTrajectorySolver` on the same numerical
   metric. Treat this as corroborating evidence; the current implementations are
   not fully independent because some setup/outbound/escape logic is shared.

2. **Hamiltonian constraint**: For the Hamiltonian solver, verify |H| ≈ 0 along the ray.
   This is already tracked in `diagnostics.residuals["max_hamiltonian_constraint"]`.

3. **Energy/momentum conservation**: Verify E and L stay constant. Already tracked in
   `diagnostics.residuals["energy_residual"]`, `["angular_momentum_residual"]`.

4. **Li-DM-BH consistency**: The `.npz` files from Li-DM-BH carry `B * C ≈ 1` and
   metadata horizons. The `ValidationGate` already checks these.
