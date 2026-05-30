# Artifact Policy

Date: 2026-05-29

This repository is an upstream computation core. Generated artifacts can be
useful as examples or historical references, but they should not become an
implicit test dependency or expand the repository without review.

## Classes

- Checked-in examples: small, durable files referenced from README or docs.
- Regression fixtures: minimal files required by default tests. Prefer dynamic
  synthetic fixtures in `tmp_path`; use checked-in binary fixtures only when a
  path-based workflow cannot be tested otherwise.
- Generated outputs: images, CSV, JSON reports, manifests, and benchmark outputs
  produced by scripts. These belong under `outputs/` during local work.
- Historical references: previous generated outputs retained to document a paper
  figure, review, or known result. These need a note explaining why they remain.
- External/private data: Li-DM-BH files and other upstream datasets outside this
  repository. Default tests and CI must not require them.

## Current Policy

- Do not write new fixtures under `outputs/`.
- Do not require `/home/yuanpang/Work/Li-DM-BH` or any private path in default
  pytest or CI.
- Keep `outputs/` ignored for new generated files. Already tracked outputs are
  not removed by this policy; removal would be a separate reviewed change.
- New regression fixtures should be tiny, documented, and justified by a test
  that cannot use dynamic synthetic data.

## Open Review Item

The repository currently has many tracked files under `outputs/`. They may be
historical references or generated examples. Classifying or removing those files
requires human confirmation of downstream/paper dependencies.
