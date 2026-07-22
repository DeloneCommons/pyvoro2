# v0.7.0 release checklist

This checklist records issue
[#18](https://github.com/DeloneCommons/pyvoro2/issues/18) release
qualification without duplicating the release contract in the
[v0.7 plan](plans/v0.7.md) or the publication process in the
[development workflow](development-workflow.md). Checkmarks record completed
evidence for the release-candidate commit. Publication and post-publication
items remain unchecked until the maintainer approves and performs them.

## 1. Prerequisite and contract review

- [x] The issue-#16 baseline is committed, the API inventory records maintainer
      approval on 2026-07-22, and the v0.7 plan remains Active.
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

- [x] `mkdocs build --strict` passes with current release-candidate metadata.

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

- [ ] The release-candidate commit passes Linux, macOS, and Windows tests on
      Python 3.10, 3.11, 3.12, and 3.13.
- [ ] The release-candidate commit passes CI lint/generated-file,
      docs/notebook, and distribution jobs.
- [ ] Tagged cibuildwheel jobs produce and test the supported wheel matrix.

## 10. Release metadata and changelog preparation

- [x] The single-source package version is 0.7.0 and built metadata agrees.
- [x] The accumulated changes are a coherent dated `[0.7.0]` section beneath a
      fresh empty `[Unreleased]` section.
- [x] README, guides, inventory, examples, benchmark scope, and changelog agree
      on preferred APIs, compatibility routes, v0.8 removals, lifecycle status,
      and the static sparse boundary.
- [x] The v0.7 plan remains Active and its Outcome does not claim publication.

## 11. Maintainer review

- [ ] An independent review has accepted the repository diff, qualification
      evidence, artifacts, and candidate CI state.
- [ ] The maintainer has explicitly authorized irreversible release actions.

## 12. Tag, GitHub release, PyPI, and Zenodo publication

- [ ] The reviewed release commit is integrated according to the `dev` to
      `main` branch model.
- [ ] The annotated or signed `v0.7.0` tag is created and pushed.
- [ ] Tagged wheel/sdist artifacts are collected and independently validated.
- [ ] Version 0.7.0 is published to PyPI and a GitHub release is created.
- [ ] Zenodo archive ownership/integration is confirmed and the archive is
      created or verified.

## 13. Post-publication verification

- [ ] A clean install from PyPI passes the public smoke workflow.
- [ ] Published artifact hashes/metadata, the GitHub release, documentation
      deployment, and Zenodo record are verified.

## 14. Plan outcome, completion, and archival

- [ ] Issue #18 and the v0.7 milestone are closed after publication
      verification.
- [ ] The v0.7 Outcome records the tag, delivered contract, sparse-path result,
      compatibility horizon, deviations, and deferrals.
- [ ] The plan status is changed to Completed, the plan is moved to the archive,
      and the plan indexes are updated.
- [ ] The draft v0.8 cleanup plan is activated only through its documented
      maintainer-approval mechanics.
