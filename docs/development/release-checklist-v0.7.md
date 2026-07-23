# v0.7.0 release checklist

This checklist records issue
[#18](https://github.com/DeloneCommons/pyvoro2/issues/18) release
qualification without duplicating the release contract in the
[archived v0.7 plan](plans/archive/v0.7.md) or the publication process in the
[development workflow](development-workflow.md). The completed plan is
included in the final release source; this checklist is the completed
operational record for release-commit CI, integration, the tag, tagged
artifacts, PyPI publication, and post-publication verification. By approved
maintainer decision, v0.7.0 has no GitHub Release or Zenodo archive.

## 1. Prerequisite and contract review

- [x] The issue-#16 baseline is committed, the amended API inventory records
      maintainer approval on 2026-07-23, and the completed v0.7 plan is archived
      in the final release source.
- [x] Required milestone issues #6–#17, #20, and #21 are closed; #18 is the
      only open milestone issue. Issues #19 and #23 are unmilestoned and do not
      block v0.7.
- [x] Issue #18, WP-10, the release gates, lifecycle inventory, migration
      contract, ADRs 0004–0006, contributor guidance, release tools, examples,
      benchmark scope, and CI workflows have been reviewed together.

## 2. Local deterministic qualification

- [x] A fresh editable environment installs successfully with `.[all]`.
- [x] `python tools/release_check.py` passes.
- [x] The deterministic test suite and explicit lint checks pass independently.
- [x] The preferred chemvoro-shaped and paper-regression workflows pass.

## 3. Optional, fuzz, and cross-check qualification

- [x] `pytest -m fuzz --fuzz-n 100` passes.
- [x] The optional `pyvoro` cross-check passes, or its unavailability is
      recorded without being reported as a passed gate.

No compatible `pyvoro` installation was available in the CPython 3.13
qualification environment. The optional comparison is unavailable, not
passed, and is not a release blocker.

## 4. Notebook and generated-file synchronization

- [x] Committed notebooks pass clean-kernel validation.
- [x] Notebook exports and the generated README are synchronized.

## 5. Strict documentation validation

- [x] `mkdocs build --strict` passes with final v0.7.0 metadata.

## 6. Source distribution and wheel validation

- [x] The sdist and local wheel build successfully at version 0.7.0.
- [x] Twine metadata and `tools/check_dist.py` content checks pass for every
      artifact.
- [x] Artifact contents, filenames, metadata, sizes, and hashes are recorded;
      caches, local files, secrets, and unintended build products are absent.
- [x] The sdist produces a functional wheel in a fresh source-build
      environment.

## 7. Clean wheel installation without SciPy

- [x] A fresh environment installs the wheel with ordinary runtime
      dependencies only, without SciPy.
- [x] Version/import, lazy-native-load, 2D/3D forward, separator-fit, preferred
      import, compatibility import, and deprecation-warning smoke checks pass.

## 8. Sparse installation and sparse-contract validation

- [x] A second fresh environment with SciPy passes the sparse solver tests and
      the preferred paper regression workflow with explicit sparse reporting.
- [x] Dense/sparse agreement, unsupported-branch rejection, disconnected
      component gauges, and lazy optional-SciPy import behavior are verified.
- [x] The static benchmark suite and the sparse-only `large_knn` case pass;
      instrumentation confirms that the large case does not allocate a dense
      global normal matrix.

## 9. Multi-Python and cross-platform CI

- [x] The final release commit passes Linux, macOS, and Windows tests on
      Python 3.10, 3.11, 3.12, and 3.13.
- [x] The final release commit passes CI lint/generated-file,
      docs/notebook, and distribution jobs.
- [x] Tagged cibuildwheel jobs produce and test the supported wheel matrix.

## 10. Release metadata and changelog preparation

- [x] The single-source package version is 0.7.0 and built metadata agrees.
- [x] The accumulated changes are a coherent dated `[0.7.0]` section beneath a
      fresh empty `[Unreleased]` section.
- [x] README, guides, inventory, examples, benchmark scope, and changelog agree
      on preferred APIs, compatibility routes, v0.8 removals, lifecycle status,
      and the static sparse boundary.
- [x] The completed v0.7 plan is archived in the final source. Its Outcome
      records the approved release model and local evidence without claiming
      that unchecked CI, tag, PyPI, or public-smoke operations have passed.

## 11. Maintainer review

- [x] Independent review accepted the repository diff, qualification evidence,
      release-source corrections, and amended API inventory.
- [x] On 2026-07-23 the maintainer authorized integration, tagging, tagged
      artifact validation, and PyPI publication under the approved v0.7 release
      model.

## 12. Integration, tag, and PyPI publication

- [x] The reviewed release commit is integrated according to the `dev` to
      `main` branch model.
- [x] The annotated or signed `v0.7.0` tag is created and pushed.
- [x] Tagged wheel/sdist artifacts are collected and independently validated.
- [x] Version 0.7.0 is published to PyPI.
- [x] No GitHub Release or Zenodo version record is created for v0.7.0. The
      next full GitHub/Zenodo archival release is planned for v0.8.0.

## 13. Post-publication verification

- [x] A clean install from PyPI passes the public smoke workflow.
- [x] Published artifact hashes/metadata and documentation deployment are
      verified.

## 14. Plan outcome, completion, and archival

- [x] The v0.7 Outcome records the delivered contract, sparse-path result,
      compatibility horizon, approved release model, deviations, and deferrals.
- [x] The plan status is Completed, the plan is moved to the archive in the
      final release source, and the plan indexes are updated.
- [x] PyPI publication and public verification are complete; issue #18 and the
      v0.7 milestone are ready to close.
- [x] The v0.8 cleanup plan remains Draft through v0.7 closure. Activation is a
      separate post-v0.7 maintainer decision after its milestone, issue set,
      approval date, and scope are recorded.

## 15. Publication completion record

- Final v0.7.0 source and tag commit:
  `5335d6fa201eaabe025b6bc70c8e71ccb9286b11`.
- At publication, `main` and `v0.7.0` resolved to the same reviewed commit;
  this checklist-completion change is post-release documentation and does not
  alter the tagged source.
- Final branch CI and the tagged wheel/sdist workflow completed successfully.
- Tagged artifacts were independently validated before upload, and v0.7.0 was
  published to PyPI.
- Clean public no-SciPy and SciPy-backed sparse installation smoke checks
  passed, and deployed documentation was verified.
- No GitHub Release or Zenodo version record was created for v0.7.0, as
  approved. The next full package, GitHub Release, and Zenodo archival release
  is planned for v0.8.0.
