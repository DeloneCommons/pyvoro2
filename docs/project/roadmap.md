# Roadmap

This roadmap records version-level outcomes and long-term direction. It is not a
timeline, release plan, or list of every implementation task.

- Active release scope and gates live in
  [development plans](../development/plans/index.md).
- GitHub issues and milestones track concrete work and current progress.
- Decision records explain durable architectural choices.
- The changelog records completed user-visible behavior.

## Current direction: operational reboot

Development restarts from published v0.8.0 under
[ADR 0027](../development/decisions/0027-v0.9-reboot-from-v0.8.md) and
[issue #120](https://github.com/DeloneCommons/pyvoro2/issues/120).
The approved transition is PR1 operational bootstrap, followed by PR2 curated
historical knowledge migration. Preserve implemented v0.8 behavior while
restoring a manageable development process.

The previous v0.9 attempt is preserved at
`archive/v0.9-attempt-2026-10-08`; its designs and ADRs 0018–0026 are not
adopted automatically. Historical Checkpoint C remains pending/unaccepted;
pre-Phase-D recovery and WP12/WP13 remain unfinished.

Future geometry architecture and release scope are undecided. ADR 0017's
v0.9, 1.0, v1.1, and v1.2 sequence is superseded, and ADR 0006's earlier
sequence is not reactivated. Historical rationale remains in
[ADR 0017](../development/decisions/0017-v0.9-functional-stabilization-before-1.0.md).
Research candidates below do not authorize implementation.

## v0.7 — Forward and separator API stabilization (completed)

The v0.7 line established the forward and separator-inverse architecture and a
chemvoro-shaped downstream contract, without establishing a final downstream-readiness
claim for the reboot.

Delivered outcomes include:

- one common `TessellationResult` default in 2D and 3D;
- direct forward power computation from mathematical weights;
- stable site/ID/result association;
- canonical separator inverse ownership under `pyvoro2.inverse.separator`;
- explicit gauge and disconnected-component semantics;
- inspectable graph/Laplacian diagnostics and layered inverse results;
- an optional explicit SciPy sparse quadratic path;
- one bounded compatibility release and migration path for historical APIs;
- paper- and chemvoro-shaped integration validation.

See the [archived v0.7 development plan](../development/plans/archive/v0.7.md).

## v0.8 — Technical maintenance and Python 3.14 (published)

v0.8 is intentionally a **technical-maintenance release without new numerical
functionality**. Its R1–R9 remediation and post-R9 `COPYING` distribution
correction are part of the published v0.8.0 baseline at
`db0884c641de0998d190de8aeee1d45154e46aff`. Issue #33 is its historical release
qualification record.

Delivered outcomes include:

- remove compatibility-only inverse/planar routes retained for v0.7;
- reorganize tests and private Python helpers without changing the public
  architecture;
- support Python 3.10–3.14 and the approved wheel matrix;
- qualify source distributions through an isolated sdist-to-wheel round trip;
- resolve accepted correctness, backend-safety, source-identity, active-state,
  diagnostic, and maintenance findings;
- preserve stable numerical behavior except for separately approved correctness
  fixes.

See the completed [v0.8 plan](../development/plans/archive/v0.8.md),
[remediation plan](../development/plans/archive/v0.8-remediation.md),
[pre-release audit](../development/audits/v0.8-pre-release.md), and
[ADR 0006](../development/decisions/0006-v0.8-cleanup-release.md).

## Possible future work

Periodic geometry, weight-first queries, representation robustness, downstream
usability, and separator-workflow stabilization remain subjects for assessment.
The previous attempt's detailed requirements, outcomes, and defects belong in
PR2's curated inventory. They are not a new implementation checklist here.
A development version or historical success does not settle a geometry design,
API promotion, release gate, or date.

### Possible inverse research candidates

- **Prescribed-cell-measure inversion:** infer power weights from cell areas
  or volumes at fixed sites and domain.
- **Mixed separator-plus-measure inversion:** investigate combining these two
  kinds of observations.

Neither candidate has approved implementation, public API, architectural
relationship, release target, or pre-1.0 inclusion. Do not replace the former
v1.1/v1.2 allocation with another version promise. Promoting either candidate
requires separate mathematical analysis, representative use cases, a validation
strategy, and maintainer approval. Substantive inverse-architecture reassessment
belongs to later work, not PR1.

### Demand-driven engineering candidates

Potential directions without release commitments include improved periodic
representation, explicit scaling, clipping/wall domains, coincident-site
policies, persistent trajectories, performance work, and package-boundary
reassessment. Concrete demand and a separately approved scope must precede
implementation.

## Future research directions

Possible later research workstreams include:

- cell-centroid observations and separator/measure/centroid combinations;
- inverse fitting from planar sections or slices;
- richer regular-triangulation and dual diagnostics;
- optional solver backend plugins;
- bounded site-coordinate optimization as an explicit new unknown family;
- anisotropic or non-Euclidean models driven by a concrete research project.

## Informational and upstream-oriented limitations

Record these accurately without implying a committed pyvoro2 implementation:

- partial triclinic periodicity in 3D or oblique periodicity in 2D when correct
  support would require substantive backend work or a second periodic layer;
- unbounded cells/domains, which require a different result geometry contract;
- functionality requiring substantive Voro++ source changes rather than a
  bounded wrapper correction;
- guarantees beyond binary64/backend numerical resolution.

pyvoro2 should expose structured failures and adopt useful upstream Voro++
improvements when practical, but informational limitations are not scheduled
features.

## Decisions and scope boundaries

Current v0.8 uses one repository and distribution with vendored Voro++.
Whether to turn those facts into permanent package/backend policy remains an
explicit maintainer decision. Future architecture and release targets require
separate approval rather than inheritance from historical planning.

The operational transition does not authorize geometry or numerical rewrites,
API changes, new inverse models, site motion, arbitrary objective callbacks,
new domain families, GPU work, release qualification, or API freeze.

## Planning responsibilities

- This page records version-level direction and preserves candidate workstreams.
- [Development plans](../development/plans/index.md) define active release
  outcomes, dependencies, validation, and gates.
- Decision records explain durable choices.
- GitHub milestones group release outcomes.
- GitHub issues define implementation tasks and acceptance criteria.
- The changelog records completed user-visible changes.
- Completed plans are archived with their outcome and deferrals.
