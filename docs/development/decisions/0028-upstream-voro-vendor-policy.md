# 0028 — Transitional maintenance of vendored Voro++ while upstream dev evolves

- **Status:** Accepted
- **Date:** 2026-11-19
- **Scope:** policy for evaluating and, when justified, temporarily patching the vendored Voro++ snapshot
- **Related:** [ADR 0027](0027-v0.9-reboot-from-v0.8.md),
  [functional sequencing](../plans/functional-sequencing.md),
  [maintainer review register](https://github.com/DeloneCommons/pyvoro2/blob/dev/notes/maintainer-review/review-register.md)

## Context

pyvoro2's v0.8-based development line retains an older, independently tested
vendored Voro++ snapshot. Moving to upstream's substantially changed
[`chr1shr/voro` `dev`](https://github.com/chr1shr/voro/tree/dev)
requires a separate integration and compatibility effort; it is not a drop-in
vendor update. The upstream maintainer has indicated that the current `dev`
will eventually be merged into `master`, and prefers not to patch the current
`master` in the interim. No date or compatibility guarantee is assumed.

Historical D9 and the reboot's R-O02a left upstream-functional immutability
versus a bounded downstream patchset undecided. The previous reassessment
register imposed a broad upstream-investigation/reporting gate on planned C++
edits. That gate is too broad: it couples otherwise independent binding changes
to two specific Voro++ defects and requires upstream reporting even when the
maintainer's active development branch is the relevant reference.

This decision is **temporary vendor-maintenance policy**, not authorization to
adopt the upstream `dev` backend or to maintain an indefinite functional fork.

## Decision

1. **Keep the current validated vendor snapshot** until a separately approved
   upstream-`dev` migration, with its own API, ABI, numerical and regression
   qualification. Do not silently switch Voro++ branches or import unrelated
   changes.
2. **Triage each concrete defect independently.** Establish a minimal reproducer
   and an expected scientific or safety outcome against the currently vendored
   code. Then test the same case on a *recorded upstream `dev` commit*. A
   historical `master` result alone does not answer whether `dev` fixes it.
3. **If upstream `dev` no longer has the defect**, a small local backport is
   permitted as a candidate when current pyvoro2 users need the fix. Prefer the
   minimal equivalent correction compatible with the vendored source rather
   than copying a broad upstream refactor. Require a focused approved issue/PR,
   independent regression or fault-injection evidence, native checks and
   a documented patch inventory.
4. **If upstream `dev` still has the defect**, first assess its severity and
   actual impact. A bounded/noncritical defect may remain documented until the
   future migration. But silent wrong geometry, corruption or other serious
   supported-use failures are not automatically deferrable: evaluate a focused
   local correction, a sound detection/refusal contract, or a supported
   mitigation, each requiring a separate decision.
5. **If status is uncertain**, investigate before selecting a patch or a
   deferral. Record both revisions and the reproducer. Do not infer that a bug is
   fixed merely because source structure changed in `dev`.
6. **Upstream communication is case-specific.** Share independently reproduced
   findings and candidate corrections when useful and welcome; submitting an
   issue against today's upstream `master` or obtaining an upstream merge is
   **not a blanket prerequisite** to a separately approved pyvoro2 change.
   Do not assume that the upstream maintainer agreed to carry a particular
   patch on either branch.
7. **Maintain a narrow, removable patchset.** Each local Voro++ modification
   must identify the upstream/base revision, defect, test oracle, exact patch,
   affected 2D/3D modes and known numerical limits. Reassess the patch when
   upstream `dev` is integrated: carry it only if still necessary.

The former **global pre-C++ gate** requiring prior reporting of both R-O02b
(allocation failure) and R-O03 (extreme power range) is **replaced by this
per-defect triage**. Both investigations retain their own research priority;
they no longer block unrelated C++ binding work. This decision does not
authorize any particular vendor patch, new public guarantee, or changed native
behavior without its own scope and review.

## Consequences and validation

This policy improves near-term safety and avoids a premature backend rewrite,
at the cost of potentially carrying a small temporary delta. Source compatibility
alone is not sufficient: regressions need independent mathematics or fault
injection where practical, plus relevant ordinary/sanitizer/native tests.

For numerical failures, distinguish wrong native geometry from unsupported
binary64 accuracy; do not advertise a universal fix from one example. For
allocation-failure paths, specify the supported exception/recovery contract
before asserting global leak freedom.

A later migration to upstream `dev` (or its descendant `master`) needs its
own decision, qualification plan and reconciliation of every temporary patch.
The one-repository/distribution policy and general long-term Voro++ fork policy
remain open beyond this transitional exception.

## Alternatives considered

- **Never patch vendored Voro++:** rejected as a blanket rule because known,
  reproducible defects could affect the current supported library while an
  upstream migration remains expensive.
- **Migrate to upstream `dev` now:** not selected; the necessary port and
  regression scope are not established.
- **Backport every upstream `dev` fix or maintain a broad fork:** rejected;
  only demonstrated user-facing defects justify bounded patches.
- **Wait for upstream fixes in every case:** rejected for potentially silent
  scientific/safety failures; acceptable for explicitly documented,
  noncritical problems.

## Related work

- [R-O02b — native allocation failure](https://github.com/DeloneCommons/pyvoro2/blob/dev/notes/maintainer-review/potential-work/native-allocation-failure.md)
- [R-O03 — large-weight numerical failure](https://github.com/DeloneCommons/pyvoro2/blob/dev/notes/maintainer-review/potential-work/large-weight-numerics.md)
- [Current functional-sequencing plan](../plans/functional-sequencing.md)
