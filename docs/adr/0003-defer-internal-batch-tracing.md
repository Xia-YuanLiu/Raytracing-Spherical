# ADR 0003: Defer Internal Batch Tracing

Date: 2026-05-29

## Status

Accepted

## Context

The performance plan identifies scalar `TabulatedMetric` PCHIP and segment
integration overhead as the first measurable bottleneck. P1 work now has a
guarded single-ray fast path and a benchmark mode comparing fast and scalar
reference solvers.

Batch tracing would add orchestration complexity across imaging/profile code and
would need per-ray fallback semantics to preserve scalar public outputs.

## Decision

Do not add internal batch tracing in this stabilization pass. Reconsider only
after benchmarks on `sample_radial_profile` and `render_axisymmetric_image`
show that repeated scalar orchestration remains the dominant bottleneck after
the single-ray fast path.

If approved later, batch tracing must remain private initially, add no
`__all__` export, preserve scalar result semantics, and include benchmark
evidence of meaningful speedup.

## Consequences

- The public API stays stable.
- The codebase avoids a speculative batching layer.
- Future optimization work has a clear evidence threshold.
