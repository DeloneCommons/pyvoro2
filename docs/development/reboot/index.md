# Historical v0.9 knowledge

This is a reading guide to what pyvoro2 learned during its former v0.9 attempt.
It preserves requirements, algorithms, counterexamples and review outcomes so
that later work can assess them without reconstructing the entire issue history.
It is **historical evidence, not a replacement architecture or release plan**.

[ADR 0027](../decisions/0027-v0.9-reboot-from-v0.8.md) governs the reboot from
published v0.8.0. Historical ADRs 0018–0026 are not adopted by this inventory.
A historically accepted implementation can still be an **unreviewed candidate**
for the reboot. Existing v0.8 behavior has a separate classification.

## Start here

| Question | Read |
|---|---|
| What happened, and where did the attempt stop? | This overview and the checkpoint table below. |
| Which scientific or API problem was each change addressing? | [Requirements inventory](requirements-inventory.md): capability matrix, baseline impact and candidate dispositions. |
| What can independently test a claim? | [Evidence index](evidence-index.md): exact oracles, numerical comparisons, adversarial inputs and review provenance. |
| What is callable on the reboot branch? | [Current v0.8 API inventory](../api-inventory.md), [architecture](../architecture.md) and current source. Historical signatures are not current API documentation. |
| What authorizes subsequent implementation? | A new explicit maintainer decision under ADR 0027; an inventory entry alone does not. |

### Source identities

The reconstruction uses GitHub issue/review history observed on 2026-10-08 UTC
and the following immutable source identities. Later issue administration does
not retroactively change the checkpoint outcomes recorded here.

| Source | Commit / reference |
|---|---|
| Published v0.8.0, the numerical/API baseline | [`db0884c641de0998d190de8aeee1d45154e46aff`][baseline] |
| Reboot integration base after PR1 | [`d0034e91e7619c9631fd9da38ae8baddb08e311c`][reboot-base] |
| Final historical v0.9 snapshot | [`b5e1aa6f53cd5edd94c0ae376047cb53278d2158`][archive]; tree `31afef7a8c043bb0bc56765c798ad18516549f1d` |
| Permanent protected archive tag | [`archive/v0.9-attempt-2026-10-08`][archive-tag], resolving to that historical commit |
| Transition coordination | [#120][transition]; PR1 [#121][pr121] |

Historical `dev` still named that snapshot at the reconstruction cutoff. Its
planned name, `legacy/v0.9-attempt-2026-10-08`, is not needed to resolve the
commit-pinned links in these pages. Branch cutover, issue closure, integration
of this inventory and any release remain separate maintainer operations.

## What happened

The published v0.8 baseline already had direct mathematical weights for
`compute`, a common forward result, certified periodic nearest-image machinery,
mandatory native preflight and duplicate safety, a graph-based fixed separator
fit, source/observation identity and an experimental realization-aware outer
loop. The historical attempt was therefore a stabilization and extension of an
existing scientific core, rather than its first implementation.

The [historical plan][h-plan], activated through [#46][i46], grouped the work into
periodic foundations (A), periodic topology and metadata (B), inverse
stabilization (C), and public qualification/API audit (D):

- **Phase A, WP1–WP4:** complete weight-first query/ghost input; distinguish the
  caller's lattice from the backend frame; introduce exact private lattice
  reduction; make certified image search and native translation recovery more
  robust to poor equivalent bases. Review also found source-rounding, ghost
  batch-state and coupled-remapping defects that needed focused remediation.
- **Phase B, WP5–WP9:** reconstruct native boundary provenance and periodic image
  labels; distinguish persistent owners, temporary ghost self-images and walls;
  make query/owner metadata coherent; remove obsolete reconstruction controls.
  Degeneracy investigations then established that native occurrences, exact
  contacts and normalized numerical topology need different contracts.
  Native qualification and its operational costs became a substantial separate
  engineering concern.
- **Phase C, WP10–WP11:** separate observation, mismatch, hard-bound and penalty
  spaces; add a bounded row policy; expose an ordinary realization-aware facade.
  A parameter-level gate selected no further row-wise shape implementation.
  Integrated review subsequently found final-state and diagnostic defects,
  repaired through PRs #117 and #119 with separate remediation reviews.
- **Phase D and release:** the whole-code recovery gate, WP12 public-workflow
  qualification, WP13 API/lifecycle audit and [#48][i48] release gate were not
  completed. The historical attempt was **not released as v0.9.0**.

The plan's [revision log][h-revisions] explains how the work changed: physical
rather than coefficient-space tie selection, the N/E/S boundary contracts,
qualification hardening, the post-B row-policy gate, and moving whole-code
recovery before Phase D. Frozen documents still contain historical labels such
as “Active” and target-release language. Read those labels in their original
context, alongside the later specific decisions and dated acceptance records.

### Accepted checkpoints and unfinished work

| Gate | Historical outcome | Evidence and boundary |
|---|---|---|
| Checkpoint A | **Accepted** at `1af2c79bc33a8d8238274c62992e83ff10fdb6dc`, tree `6868be264bb114ce4c7ffbaf97bfd09fcef7a75d` | [Explicit integrated acceptance][accept-a] after #60/#61, #62/#63 and #65/#66. PR #67 was subsequent cleanup. |
| Checkpoint B | **Accepted** at `351f53e2e5fab8978d7c3f3d28dd7beb1b847357`, tree `7fc6a0e2ef8986981b6f2da382f94dbc28abf679` | [Final integrated review][accept-b] and [tracker update][accept-b-tracker]. Merging individual degeneracy fixes was not this acceptance. |
| WP10, WP11, parameter gate and C remediations | Individually accepted under the historical plan | [#47][i47], [PR #115][p115], [#116][i116] and [#118][i118]. These are narrower outcomes than Checkpoint C. |
| Checkpoint C | **Pending / unaccepted** at the final archived snapshot | #118 records PR #119 acceptance at `863d56e27e3a03e2cf85f375b7889af3aad408b6`, then merge as `b5e1aa6f…`, while explicitly leaving C unaccepted. |
| Pre-Phase-D recovery; WP12; WP13 | **Unfinished** | The unchecked gates in #47 and [historical plan][h-plan]; no recovery closure or final API freeze is implied. |
| v0.9 release qualification/publication | **Unfinished; no v0.9.0 release** | #48 and ADR 0027. Successful CI and the archive tag are not release acceptance. |

Acceptance is attached to the reviewed state and scope. Neither this table nor
later remediation merges claim that all subsequent changes received another
integrated A/B/C review.

## Workstream map

The inventory is organized by capability; this table preserves the work-package
route into it. The evidence page records additional dependencies and review
chains, including findings outside the original work packages.

| Historical work | Capability entries |
|---|---|
| WP1 — #49 | [Weight-first queries](requirements-inventory.md#r-q01) |
| WP2 — #54 | [User lattice and backend frame](requirements-inventory.md#r-l01) |
| WP3 — #56; WP4 — #58 | [Exact reduction and image geometry](requirements-inventory.md#r-l02), [source and numerical translations](requirements-inventory.md#r-l03) |
| A remediations — #60, #62, #65; PR #67 | [Source/remapping failures](requirements-inventory.md#r-l03), [ghost execution safety](requirements-inventory.md#r-g01) |
| WP5 — #68 | [3D native attribution and exact ideals](requirements-inventory.md#r-t01) |
| WP6 — #74 | [Planar provenance](requirements-inventory.md#r-t02) |
| WP7 — #77 | [Ghost execution safety](requirements-inventory.md#r-g01), [ghost boundary meaning](requirements-inventory.md#r-g02) |
| WP8 — #79; WP9 — #85 | [Metadata/output](requirements-inventory.md#r-m01), [API removals](requirements-inventory.md#r-a01) |
| B follow-ups — #82, #84, #88, #92–#99 | [Performance](requirements-inventory.md#r-e02), [qualification](requirements-inventory.md#r-e01), [planar](requirements-inventory.md#r-t03) and [spatial degeneracy](requirements-inventory.md#r-t04) |
| WP10 — #107, entry gate #104 | [Independent spaces](requirements-inventory.md#r-i01), [row policy](requirements-inventory.md#r-i02) |
| WP11 — #111; gates #109, #113, PR #115 | [Realization-aware facade](requirements-inventory.md#r-i04), [qualification](requirements-inventory.md#r-e01), [process lessons](requirements-inventory.md#r-e03), [shape-parameter decision](requirements-inventory.md#r-i03) |
| C remediation — #116/#117; #118/#119 | [Final-state correctness](requirements-inventory.md#r-i05), [exact diagnostic availability](requirements-inventory.md#r-i06) |

## Why restart from v0.8?

ADR 0027 selects an understandable development workflow on the published source
before another substantial geometry architecture is chosen. It retains the
historical attempt as a permanent source of tests, numerical counterexamples,
mathematics and rationale. PR1 restored operations; this inventory supplies the
knowledge needed for an independent maintainer reading of the code.

Returning to v0.8 also restores its limitations and some historical defect
paths. It does not make all later fixes unnecessary. The inventory records
those impacts explicitly, while keeping the proposed fixes separate from
reboot authorization. The historical native qualification machinery is not a
mandatory reboot dependency. Former promises about v0.9, 1.0, prescribed
measures and mixed inversion are superseded; they do not become a new schedule.

## Suggested reading order

1. Read ADR 0027 and the [baseline entries](requirements-inventory.md#baseline)
   to establish what the current source already does.
2. Read the periodic entries in order: weights, user/backend coordinates,
   exact images, native boundary attribution, ghost identity, metadata and
   normalization. The [N/E/S comparison](requirements-inventory.md#geometry-layers)
   is the key to interpreting the degeneracy evidence.
3. Inspect the [independent geometry oracles](evidence-index.md#geometry-evidence),
   especially the missing-positive-face, equal-rounded-coordinate and
   nonzero-winding cases, before drawing conclusions from a normalizer's output.
4. Read the inverse entries, then the [final-state and binary64 evidence](evidence-index.md#separator-evidence).
   Keep fixed-observation fitting separate from the outer realization heuristic.
5. Read the [open questions](requirements-inventory.md#open-questions), including
   D9 and PA-001–003. Incidence-first reconstruction and direct semantic
   construction from Laguerre halfspaces remain competing research hypotheses.
6. Follow the [source corpus and review links](evidence-index.md#source-corpus)
   only for the capability being studied. Consult historical API names in the
   archived inventory, not as imports promised by the reboot.

## Maintaining the inventory

Keep identifiers stable. Add a new explicit decision link when a candidate is
adopted, rejected or deferred for the reboot; preserve the separate historical
status and counterexamples. Extend an existing capability entry before creating
another document. Historical tests remain in the protected source archive;
large kits, binary artifacts and duplicated source trees do not belong here.

The evidence index states which records were inspected, which claims depend on
reported historical execution, and which external artifacts are unavailable or
not publicly durable. This migration performs no new integrated Checkpoint-C
review and makes no new numerical qualification claim.

[baseline]: https://github.com/DeloneCommons/pyvoro2/tree/db0884c641de0998d190de8aeee1d45154e46aff
[reboot-base]: https://github.com/DeloneCommons/pyvoro2/tree/d0034e91e7619c9631fd9da38ae8baddb08e311c
[archive]: https://github.com/DeloneCommons/pyvoro2/tree/b5e1aa6f53cd5edd94c0ae376047cb53278d2158
[archive-tag]: https://github.com/DeloneCommons/pyvoro2/tree/archive/v0.9-attempt-2026-10-08
[transition]: https://github.com/DeloneCommons/pyvoro2/issues/120
[pr121]: https://github.com/DeloneCommons/pyvoro2/pull/121
[h-plan]: https://github.com/DeloneCommons/pyvoro2/blob/b5e1aa6f53cd5edd94c0ae376047cb53278d2158/docs/development/plans/v0.9.md
[h-revisions]: https://github.com/DeloneCommons/pyvoro2/blob/b5e1aa6f53cd5edd94c0ae376047cb53278d2158/docs/development/plans/v0.9.md#plan-revisions
[i46]: https://github.com/DeloneCommons/pyvoro2/issues/46
[i47]: https://github.com/DeloneCommons/pyvoro2/issues/47
[i48]: https://github.com/DeloneCommons/pyvoro2/issues/48
[accept-a]: https://github.com/DeloneCommons/pyvoro2/issues/47#issuecomment-5745444670
[accept-b]: https://github.com/DeloneCommons/pyvoro2/issues/95#issuecomment-5969300168
[accept-b-tracker]: https://github.com/DeloneCommons/pyvoro2/issues/47#issuecomment-5969309140
[p115]: https://github.com/DeloneCommons/pyvoro2/pull/115
[i116]: https://github.com/DeloneCommons/pyvoro2/issues/116
[i118]: https://github.com/DeloneCommons/pyvoro2/issues/118
