# Development plans

Development plans connect the long-term [roadmap](../../project/roadmap.md) to
concrete GitHub issues. A plan defines the outcome, boundaries, dependencies,
validation, and release gates for one release or substantial workstream.

Plans are not daily task trackers. Current progress belongs in GitHub issues and
milestones.

## Current plan

The [v0.9.0 functional/API stabilization plan](v0.9.md) is **Active**. It is
maintainer-approved and linked to the
[`v0.9.0` milestone](https://github.com/DeloneCommons/pyvoro2/milestone/3),
[activation issue #46](https://github.com/DeloneCommons/pyvoro2/issues/46), and
[substantive execution tracker #47](https://github.com/DeloneCommons/pyvoro2/issues/47).
It is the source-controlled implementation authority for WP1–WP13; focused
child issues are prepared just-in-time from the current `dev` state.

Implementation must remain inside the active plan and accepted decision/API
contracts. The plan does not authorize unresolved scope expansion, and its D9
backend-fork policy remains the explicit later decision gate described there.

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
4. Review the draft before creating substantive implementation child issues.
5. Activate it only after explicit maintainer approval and milestone linkage;
   prepare focused execution issues from the current implementation state.

## Archive

Completed and superseded plans are kept in the [plan archive](archive/index.md).
They complement the changelog by preserving intent, dependencies, decisions,
and deferrals.
