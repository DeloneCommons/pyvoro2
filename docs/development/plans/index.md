# Development plans

Development plans connect the long-term [roadmap](../../project/roadmap.md) to
concrete GitHub issues. A plan defines the outcome, boundaries, dependencies,
validation, and release gates for one release or substantial workstream.

Plans are not daily task trackers. Current progress belongs in GitHub issues and
milestones.

## Current plan

The [v0.9.0 functional/API stabilization plan](v0.9.md) is currently a **Draft**.
It captures the reviewed implementation blueprint, accepted maintainer decisions,
dependency graph, validation strategy, and release gates, but it is not active
until the repository text is approved and a GitHub milestone is linked.

No post-v0.8 plan is currently active. Future substantial work therefore still
requires explicit plan activation; the existence of the draft does not authorize
implementation of unresolved public or architectural choices.

The roadmap reserves v0.9 for functional/API stabilization and downstream
readiness, followed by the stable 1.0 core; prescribed cell measures move to
v1.1 and mixed separator-plus-measure work to v1.2.

## Archived plans

Completed plans are preserved in the [plan archive](archive/index.md). The
completed [v0.8 technical-maintenance plan](archive/v0.8.md) and
[v0.8 remediation plan](archive/v0.8-remediation.md) record the work that
produced v0.8.0. R1–R9 and the post-R9 `COPYING` distribution correction are
complete. After source finalization and independent review, issue #33 qualifies
the exact frozen source commit and its artifacts before the public tag is
created. v0.8.0 uses Git tag, GitHub Release, and PyPI distribution without a
new pyvoro2 Zenodo software-version record.

The completed [v0.7 forward and separator API stabilization plan](archive/v0.7.md)
is also preserved there.

## Plan lifecycle

- **Draft** — scope and design are being reviewed, or activation mechanics remain.
- **Active** — approved for implementation and linked to a milestone.
- **Completed** — final release source approved, outcome recorded, and moved to the archive; external publication checks may remain in a versioned release checklist.
- **Superseded** — replaced by another named plan.

See [Development workflow](../development-workflow.md) for the complete process.

## Starting a plan

1. Copy the [plan template](template.md).
2. Use a version or descriptive workstream name.
3. Define outcome, scope, non-goals, decisions, work packages, validation, and
   release acceptance criteria.
4. Review the draft before creating the full issue set.
5. Activate it only after explicit maintainer approval and milestone linkage.

## Archive

Completed and superseded plans are kept in the [plan archive](archive/index.md).
They complement the changelog by preserving intent, dependencies, decisions,
and deferrals.
