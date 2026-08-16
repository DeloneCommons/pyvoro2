# Development plans

Development plans connect the long-term [roadmap](../../project/roadmap.md) to
concrete GitHub issues. A plan defines the outcome, boundaries, dependencies,
validation, and release gates for one release or substantial workstream.

Plans are not daily task trackers. Current progress belongs in GitHub issues and
milestones.

## Current plan

| Plan | Status | Target | Purpose |
|---|---|---|---|
| [v0.8 technical maintenance and Python 3.14](v0.8.md) | Active | v0.8.0 | Original technical-maintenance scope and release contract. |
| [v0.8 pre-release audit remediation](v0.8-remediation.md) | Active | v0.8.0 | Correctness, native-safety, geometry, result-integrity, diagnostics, public-contract synchronization, and handoff to clean qualification. |

The plans are linked to [milestone 2](https://github.com/DeloneCommons/pyvoro2/milestone/2). The remediation work is tracked by [issue #35](https://github.com/DeloneCommons/pyvoro2/issues/35), which blocks final qualification in [issue #33](https://github.com/DeloneCommons/pyvoro2/issues/33).
R1–R8 are complete. R9 ([issue #45](https://github.com/DeloneCommons/pyvoro2/issues/45))
is the active final source-contract cleanup. Issue #35 closes only after
independent R9 acceptance; issue #33 then begins from that exact clean accepted
commit.
v0.8.0 is intended to receive the next GitHub Release and Zenodo archive.
The roadmap now reserves v0.9 for functional/API stabilization and downstream
readiness, followed by the stable 1.0 core; prescribed cell measures move to
v1.1 and mixed separator-plus-measure work to v1.2.

The completed [v0.7 forward and separator API stabilization plan](archive/v0.7.md)
is preserved in the archive.

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
