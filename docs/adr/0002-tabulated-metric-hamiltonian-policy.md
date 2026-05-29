# ADR 0002: Keep `TabulatedMetric + Hamiltonian` Unsupported

Date: 2026-05-29

## Status

Accepted

## Context

`TabulatedMetric` derives metric derivatives from PCHIP interpolation of finite
tables. The Hamiltonian backend depends directly on derivative robustness and
constraint preservation. The P0/P1 guardrails validate analytical Hamiltonian
behavior and tabulated Quad behavior, but they do not establish production
stability for tabulated Hamiltonian trajectories.

The current public guard rejects `TabulatedMetric + Hamiltonian`, and validation
coverage explicitly avoids treating it as supported.

## Decision

Keep `TabulatedMetric + Hamiltonian` unsupported for public workflows. Use
`QuadTransferSolver` for production tabulated metric tracing and
`OdeTrajectorySolver` only as a backend sanity check where its documented
dependencies on Quad behavior are acceptable.

Future Hamiltonian support for tabulated metrics requires a new ADR plus tests
covering derivative robustness, Hamiltonian constraint drift, endpoint event
agreement, and representative `u_at(...)` behavior.

## Consequences

- No public API changes are required.
- Downstream users get a clear failure mode instead of silently unstable
  Hamiltonian traces.
- Analytical Hamiltonian tests remain valid and continue to protect backend
  agreement on closed-form metrics.
