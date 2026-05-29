# Review: P2 Governance

Date: 2026-05-29

## Completed

- Added artifact classification and `outputs/` policy.
- Documented physics and numerical conventions.
- Documented numerical validation commands and benchmark usage.
- Documented public API compatibility and typing policy.
- Added changelog seed.
- Added ADR 0002 keeping `TabulatedMetric + Hamiltonian` unsupported.
- Added ADR 0003 deferring internal batch tracing until benchmark evidence
  justifies it.

## Validation Run

P2 changes are documentation/governance only. Final verification after all code
changes:

- `pytest -q`: `253 passed, 1 deselected, 30 warnings`
- `pytest -m slow tests/test_numerical_metric_validation.py -q`: `1 passed,
  16 deselected`
- `python3 scripts/benchmark_numerical_metric_perf.py --repeats 1 --no-u-at --compare-reference --real-data-dir /tmp/missing-li-data`: passed with manufactured fast/reference output.

`scripts/docs_lint.sh` was requested as a best-effort docs check, but the script
does not exist in this repository.

## Plan Fit

P2.1, P2.2, P2.3, P2.4, and P2.5 are addressed to the extent reasonable without
human artifact cleanup or new public API decisions.

## Review Findings

- No public API change was made.
- No generated outputs were deleted or rewritten.
- ADRs document deferrals where the plan required a decision but available
  evidence did not justify implementation.

## Human Confirmation Needed

- Confirm classification and eventual treatment of already tracked `outputs/`.
- Confirm whether release/versioning policy should become stricter before the
  first external release.
