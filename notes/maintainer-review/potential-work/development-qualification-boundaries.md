# Development iteration versus native and release qualification

**Status:** proposed process/architecture clarification for the v0.8-based reboot;
**not** a new accepted CI policy, implementation task, or release gate. Distinguish
what is **already approved** from what still needs a maintainer decision.

## Motivation

A maintainer should be able to edit ordinary Python implementation code in an
editable installation, rerun a focused test, and continue working **without**
rebuilding C++, certifying native artifacts, or running a release-grade matrix
for every local change. Repeated qualification must not become the cost of
ordinary source editing.

This principle is substantially **already implemented and documented** in
[ADR 0027](../../../docs/development/decisions/0027-v0.9-reboot-from-v0.8.md)
and the [development workflow](../../../docs/development/development-workflow.md):
the standard environment uses `python -m pip install -e ".[dev]"`, edits to
existing Python files normally take effect without reinstalling or rebuilding
the extensions, and focused tests are the first feedback step.
[CONTRIBUTING.md](../../../CONTRIBUTING.md) documents the same setup.
Native geometry still requires the installed compiled `_core` / `_core2d`
modules; editable mode does not magically reinterpret edited C++.

**Candidate additional principle:** keep *development iteration*, *compatibility
verification of the Python/native boundary*, and *PR/release qualification*
distinct. Do not make a development-time operation implicitly require the
checks and artifacts of a later integration/release gate.

## Risk-based layers (proposal, not a replacement for current gates)

| Change / stage | Expected local feedback | What must not be inferred |
| --- | --- | --- |
| Existing **Python-only** implementation with unchanged native contract | Normal editable reload/new Python process; focused relevant Python/geometry tests, broaden on evidence | No C++ rebuild or native-artifact qualification for every edit |
| **Python consumer ↔ native contract** (ID, coordinate/shift meaning, output schema, result interpretation, numerical precondition semantics) | Focused cross-boundary/geometry regressions and independent numerical oracles as appropriate; fresh native build **if the native interface/source also changed** | A Python-only diff does **not** imply a purely Python correctness impact |
| **Native implementation, bindings, vendored source, CMake, ABI/build inputs** | Rebuild the edited extensions; native safety/geometry tests of that **fresh binary**, then broader checks at integration | Running tests against an older extension is not evidence for changed native code |
| **PR/integration** | Applicable required CI and maintainer review of the exact candidate | Local editable success does not satisfy merge gates |
| **Release/final artifacts** | Isolated wheel/sdist builds, supported-platform and installed-artifact validation according to an *approved* release plan | Editable/source tests do not qualify the shipped binaries |

The proposed distinctions primarily govern **which checks are necessary during
an iterative local edit**, not which checks may be skipped before integration.
In particular, the current [CI workflow](../../../.github/workflows/ci.yml)
**does not route PR jobs by changed paths**: it runs the required test,
sanitizer, and distribution jobs even for Python-only PRs. Changing that
behavior would be a **separate explicit CI/maintainer decision**, not a
consequence of this note.

Avoid equating “does not require a native rebuild” with “does not affect native
semantics.” A Python change can alter coordinate transformations, validation,
periodic image labels, or interpretation of a perfectly unchanged C++ result.
Choose verification from the *behavioral dependency*, not only the file suffix.

## History and boundary of reuse

- [Archived v0.8 remediation donor rules](../../../docs/development/plans/archive/v0.8-remediation.md#12-native-donor-rules-during-remediation)
  explicitly permitted matching existing native artifacts for Python-only
  iteration, but required fresh builds after native/contract changes and for
  final release qualification.
- The abandoned v0.9 attempt introduced much heavier external native
  qualification ([historical ADR 0024](https://github.com/DeloneCommons/pyvoro2/blob/legacy/v0.9-attempt-2026-10-08/docs/development/decisions/0024-external-native-artifact-qualification.md)).
  Its measured closure included some Python consumers of native results:
  “native-sensitive” is not identical to “file in cpp/”.
- The historical [change classifier](https://github.com/DeloneCommons/pyvoro2/blob/legacy/v0.9-attempt-2026-10-08/tools/ci_classify_changes.py)
  selected PR jobs by paths, but many Python changes were conservatively
  routed to full qualification. **Do not import that mechanism automatically.**
- Reboot [requirements R-E01 and R-E03](../../../docs/development/reboot/requirements-inventory.md#numerical-engineering-and-operational-lessons)
  separate artifact/certificate evidence from iteration costs; [PA-003](../../../docs/development/reboot/requirements-inventory.md#r-o06)
  preserves the unresolved choice between full and selective/scheduled
  integration qualification. [ADR 0027](../../../docs/development/decisions/0027-v0.9-reboot-from-v0.8.md)
  did **not** adopt the historical qualification framework.

## Open questions before any policy change

1. Is the *current* editable behavior reliable for existing Python sources
   across supported development environments? Verify with a small controlled
   edit-and-test workflow; do not add custom loaders unless a real limitation
   is demonstrated.
2. Can a dependency-aware scope map distinguish pure-Python edits from
   native-contract consumers without hiding significant numerical risks?
   Prefer simple documented expectations to another elaborate certification
   subsystem.
3. Should expensive PR CI checks remain universal (current accepted policy)
   or eventually become selectively triggered with a periodic/full
   qualification backstop? Any reduction needs evidence and an explicit
   approval, including a fail-closed strategy for ambiguous changes.
4. Which *native source/build* changes invalidate a previous development
   extension, and how can stale-binary testing be made obvious to contributors
   without imposing release qualifications on local edits?

**No implementation authorization:** do not alter CI, native loading,
installed provenance checking, manifest policies, PR gates, or release gates
based solely on this research note. The immediate goal is **fast, ordinary
editable Python development with proportionate local tests** while preserving
honest qualification at the gates that actually require it.
