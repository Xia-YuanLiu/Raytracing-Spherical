# Changelog

## Unreleased

- Stabilized the public API contract with a 50-export `__all__` test and public
  single-metric/junction workflow smoke tests.
- Added synthetic numerical metric coverage for both standard `r/A/B` and
  Li-DM-BH-style `r/f/g/B/params` payloads.
- Added manufactured Schwarzschild grid-convergence smoke coverage and a slow
  validation scaffold.
- Added analytical backend comparator coverage for Schwarzschild, RN, and RN-dS.
- Added identical-metric static-junction degeneration tests.
- Added a benchmark smoke test, scalar-reference comparison mode, and minimal CI.
- Extracted shared private quadrature segment construction for single-metric and
  junction transfer paths.
- Documented artifact, numerical validation, physics convention, and API
  compatibility policies.
- Kept generated outputs and local reference PDFs out of new PR content, and
  moved the long review roadmap into `docs/review/`.
