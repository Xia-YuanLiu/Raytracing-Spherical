# API Compatibility Policy

Date: 2026-05-29

This package is treated as an upstream computation core. Downstream-compatible
behavior includes:

- names exported from `spherical_raytracing.__all__`;
- public constructors and method shapes used by normal workflows;
- result data structures and important fields;
- physics conventions such as redshift and shell matching;
- numerical termination behavior and event semantics.

## Change Process

- Do not remove or rename public exports without an ADR.
- Do not change result field meanings without an ADR and migration note.
- Prefer adding tests around existing behavior before changing internals.
- Behavior-changing bug fixes must include a reproducing test and a physical or
  numerical explanation.
- Deprecations should be documented for at least one minor release before
  removal once releases exist.

## Versioning

The current project version is pre-1.0. Until release automation exists, changes
should still be reviewed as if downstream users depend on the public contract.

## Typing

The package currently does not declare `py.typed`. Type hints are useful for
maintainers, but typed-package support should be a deliberate compatibility
decision after the public API and result models settle. No `py.typed` marker is
added in this stabilization pass.
