# Development plans

Development plans connect the long-term [roadmap](../../project/roadmap.md) to
concrete GitHub issues. A plan defines the outcome, boundaries, dependencies,
validation, and release gates for one release or substantial workstream.

Plans are not daily task trackers. Current progress belongs in GitHub issues and
milestones.

## Current plan

No development plan is active. The completed
[v0.7 forward and separator API stabilization plan](archive/v0.7.md) is in the
archive and the v0.7 release checklist separately tracks the remaining external
CI, tag, PyPI, and verification operations.

## Next draft plan

| Plan | Status | Target | Purpose |
|---|---|---|---|
| [v0.8 cleanup and compatibility removal](v0.8.md) | Draft | v0.8.0 | Remove the bounded v0.7 compatibility layer, reorganize tests and private helpers, and resolve deferred maintenance findings without adding features. |

The v0.8 draft remains inactive until the v0.7.0 tag and PyPI publication are
verified and the v0.8 milestone, issue set, and maintainer approval date are
recorded. v0.8.0 is intended to receive the next GitHub Release and Zenodo
archive. Prescribed cell measures move to v0.9 and mixed
separator-plus-measure work to v0.10.

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
