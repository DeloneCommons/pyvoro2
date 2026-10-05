# Issue #88 historical implementation handoff

**Current status:** #88 was implemented, independently accepted and squash-merged
through [PR #90](https://github.com/DeloneCommons/pyvoro2/pull/90) as
`f2f9b3161c3b1fbd9ccfba3b563fbe4057cc13f1`. Its original source-review procedure
was superseded by [#109](https://github.com/DeloneCommons/pyvoro2/issues/109).
Current policy and responsibilities live in
[ADR 0024](decisions/0024-external-native-artifact-qualification.md) and the
[native qualification workflow](native-qualification.md).

The status, evidence identities and checklists below are the historical handoff,
not current pending work or issuance authority. They are preserved as recorded.

Historical handoff status: runtime/tooling corrections and independent source review are complete.
The corrected source closure has separate reviewed approval. Native and platform
qualification, exact-head CI, and the review PR/evidence handoff remain pending.
No production artifact is qualified by source approval or by this note.

## Authority and scope

The closed contract is issue #88 and the maintainer's
`ASTRA_MAX_ISSUE88_IMPLEMENTATION_PROMPT(1).md` of 2026-09-28. Live `dev`
was checked at commit `032c0d5befeb88b4a903274d082029c30fcc882f`, root tree
`2c8bc2268d22cc8745d25f85a3018b99e8c66eeb`. The predecessor archive
`pyvoro2-issue88-native-qualification-032c0d5-20260928.zip` has SHA-256
`b2d5180741f2733a2f2e1e843a4944851c7d0d3ac5fa3a4c7262cec924939d39`;
all 570 entries in its internal checksum manifest were verified.

This work changes qualification authority and safe entry, retaining the
accepted WP5 N/E/S, WP6 occurrence, WP7 selected-ghost and WP8 enclosure
contracts. It does not approve Checkpoint B, merge or close #88, change the
public API, edit functional vendor source, or enter Phase C.

## Component map

| Authority | Implementation responsibility |
|---|---|
| Reviewed source and consumers | Separately owned approval manifest; conservative native/vendor/build closure and Python consumer identity; explicit schema and policy revision. Measuring a digest does not approve it. |
| Effective build | External controlled driver and platform adapters; actual process/argument, response-file, environment, dependency, object and link records; ordered effective settings and independent arithmetic discriminators. |
| Issuance | External finalizer consumes successful complete component evidence and the final repaired/installed native payload, then writes a detached record and its trusted installation anchor. |
| Artifact verification | Private Python verifier checks trusted anchor, approved source/schema/policy, component claims, exact loaded native payload and dependency binding. Immutable installation identity may be cached; current FP state may not. |
| Safe execution | Preloaded private `_fpguard` and shared binding-owned raw integer inspection before lazy geometry loading, raw argument validation and typed casters, with checks after foreign callback boundaries; no caller-state normalization. |
| Route composition | WP5 spatial; independently owned WP6 planar; WP7 spatial = WP5 + selected spatial; WP7 planar = WP6 + selected planar; separate spatial/planar WP8 source/enclosure components. |

The record anchor belongs to the trusted package finalization/installation
path, not to native candidate metadata or a JSON file discovered beside it.
Replacing the verifier and trusted installation together is outside the
documented numerical/build trust boundary. This is not a malicious-compiler
or native-code security sandbox.

[ADR 0024](decisions/0024-external-native-artifact-qualification.md) now owns the
durable qualification policy; the [native qualification workflow](native-qualification.md)
maps its implementation and required evidence. The separate source approval binds
the independently reviewed closure and review-report hashes. Measurement alone
cannot approve a source change, and source approval cannot qualify an artifact.

## Global constraints

- No new mandatory runtime dependency or public qualification switch.
- Unqualified ordinary planar compute refuses; no geometry-only replacement.
- Geometry-only ghost and ID-only locate use their actual safety obligations,
  without acquiring unused certificate components.
- Zero-query paths construct no native geometry and issue no native certificate.
- Current GNU x86-64 requires separate x87/MXCSR nearest rounding, FTZ/DAZ off,
  all six exception classes masked, and PC64 for declared extended long double.
- Sticky flags neither prevent admission nor get cleared by the guard.
- AppleClang/MSVC ordinary WP5/WP8 support must survive with coherent adapters.
- WP8's bounded final NumPy/BLAS transform continues to permit association/FMA.
- Qualification-only hooks must not be present in production distributions.

## Implementation and remaining evidence

Checked items identify implementation work or focused development checks already
performed. They do not establish final-head acceptance. Independent source review
is complete; affected artifact checks must run again on the corrected head.

### 1. Raw runtime boundary and guarded coercion

- [x] Add failing split-rounding, PC24/PC53, unmasked-trap and callback-state
  tests using disposable child processes and exact surviving-harness restoration.
- [x] Introduce the shared raw-control inspector and preloaded `_fpguard`, with
  integer-only inspection before lazy geometry import and native argument binding.
- [x] Wire guarded public/native entry, profile inspection and replay/enclosure
  continuation, including per-callback coercion, warning delivery and domain
  subclass attribute/method checks before native-facing continuation. Independent
  entry review also verified eager production module registration during import.
- [x] Preserve structured WP5/WP6/GHOST/LOCATE failure ownership.
- [x] Fresh-build and run focused entry/precondition regressions during development.
- [x] Complete independent entry-boundary source review.
- [ ] Inspect the final optimized guard/dispatch instructions and rerun
  hostile-state controls on final artifacts.

### 2. Source approval, effective build, and artifact record

- [x] Add failing missing/forged/copied-record, source/schema mismatch and
  late-unsafe-launcher tests before issuing any qualification record.
- [x] Implement conservative source/schema/consumer measurement, separate approval
  and policy anchors, immutable payload registration and detached-record checks.
- [x] Implement controlled build recording with atomic per-process records,
  executable/backend/linker identity, ordered effective argv, response files,
  source/header inputs, environment and toolchain settings, and final objects.
- [x] Add effective-option, external-provider, linker-input and source-control
  checks with independent negative tests; keep approval distinct from captured
  hashes and reject incomplete or unsafe evidence.
- [x] Implement independent arithmetic/disassembly discrimination and finalization
  tooling that requires complete component evidence and postprocessing lineage.
- [x] Complete independent review and approve the corrected source closure.
- [ ] Run the discriminators and finalizer on the actual final production payloads,
  binding module, dependency, record and evidence identities.
- [ ] Verify a distinct conforming rebuild can receive a distinct valid record.

### 3. Route admission and qualification corpora

- [x] Replace version/ISA cohort authority with component qualification while
  keeping native metadata as consistency data and retaining per-call checks.
- [x] Preserve original archived corpus bytes and implement controlled route-suite
  and targeted missing-component/refusal checks without regenerating expectations.
- [ ] Run current WP6 48-case/1718-occurrence and original archived
  92-case/2298-occurrence corpora on the final qualification artifacts.
- [ ] Run WP7 planar 30-case/287-occurrence and accepted ordinary/selected
  spatial, ghost semantic and separate WP8 enclosure/source tests on those artifacts.
- [ ] Complete native missing-component and unqualified-platform refusal evidence
  for the final head alongside required positive paths.

### 4. Final installed artifacts and documentation

- [x] Implement controlled build/repair/install and distribution validation for
  `_core`, `_core2d`, `_fpguard`, private consumers and qualification records.
- [x] Add CI paths separating source/build admission, optimized native, sanitizers,
  Python compatibility, negative controls and direct/sdist-wheel checks.
- [x] Add focused qualification ADR; narrowly amend ADRs 0021–0023, active plan,
  native/build/support documentation and evidence navigation.
- [ ] Obtain actual final repaired/installed GNU14.2.1 manylinux positive evidence.
- [ ] Obtain final GNU13.3 and strict AVX/FMA controls, unsafe discriminators,
  AppleClang/MSVC WP5/WP8 positives and Python 3.10–3.14/full-suite evidence.
- [ ] Run final exact-head CI and retain complete commands, manifests, run IDs,
  imported module paths, native/dependency/wheel hashes and checksums.
- [ ] Finish independent branch review and hand off the review PR; do not
  merge, close issues or mark integration/release gates accepted.

## Review focus

- Foreign scalar/array/index, warning-handler and domain subclass callbacks that
  change FP state before the next package-owned operation must be refused at
  their individual return boundaries, including method acquisition.
- Direct native calls and profile inspection must be safe even when Python
  preparation and normal import-time checks are bypassed.
- A correct record copied to another module, or a modified dependency after
  repair, must not reuse the old qualification.
- Unknown relevant source or consumer edits must invalidate approval rather
  than become approved through automatic digest regeneration.
- Wheel acceptance must name the installed repaired payload; a pre-repair
  donor, candidate-only module or refusal-only smoke does not establish it.

## Evidence discipline and stop conditions

Predecessor characterization is immutable historical evidence, not a test run
on the implementation head. Overlapping test selections are not summed.
Final evidence will record the exact base/head/root tree and artifact identity.
Required CI or platform evidence that cannot be obtained is an explicit blocker.
Any necessary public API, mathematical or functional vendor change, unsafe
guard ordering, self-certifying trust loop, or unsupported existing platform
triggers the maintainer stop conditions in #88.
