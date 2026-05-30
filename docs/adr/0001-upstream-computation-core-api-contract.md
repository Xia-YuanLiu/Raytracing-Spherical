# Treat This Repository as an Upstream Computation Core

This repository is an upstream computation core for downstream projects that reuse its ray tracing, metric, thin-disk imaging, junction, and numerical-metric semantics. We will treat the public API, result data structures, physics conventions, and numerical termination behavior as a downstream compatibility contract, so stabilization work must put API contract tests and scientific validation ahead of behavior-changing refactors or performance optimization.

Consequences: internal implementation can still change, but P0 work should first add compatibility and validation guardrails; new public APIs should be introduced cautiously; performance work should prove both speed and numerical equivalence against the guarded behavior.
