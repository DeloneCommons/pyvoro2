# Development plans

Development plans connect the long-term [roadmap](../../project/roadmap.md) to
concrete GitHub issues. A plan defines the outcome, boundaries, dependencies,
validation, and release gates for one release or substantial workstream.

Plans are not daily task trackers. Current progress belongs in GitHub issues and
milestones.

## Current plan

| Plan | Status | Target | Purpose |
|---|---|---|---|
| [v0.7 forward and separator API stabilization](v0.7.md) | Active | v0.7.0 | Establish the common forward/result and canonical separator-inverse contract needed by chemvoro and later inverse methods. |

The v0.7 namespace and result decisions are accepted. The plan links the
milestone and issue sequence; implementation proceeds issue by issue under its
dependencies and release gates.

Notebook execution and published-output work is tracked by
[#20](https://github.com/DeloneCommons/pyvoro2/issues/20). It follows downstream
workflow validation in
[#15](https://github.com/DeloneCommons/pyvoro2/issues/15) and blocks final
documentation in [#16](https://github.com/DeloneCommons/pyvoro2/issues/16) and
release qualification in
[#18](https://github.com/DeloneCommons/pyvoro2/issues/18).

## Next draft plan

| Plan | Status | Target | Purpose |
|---|---|---|---|
| [v0.8 cleanup and compatibility removal](v0.8.md) | Draft | v0.8.0 | Remove the bounded v0.7 compatibility layer, reorganize tests and private helpers, and resolve deferred maintenance findings without adding features. |

The v0.8 draft is intentionally inactive until v0.7.0 is released and its
milestone and issue set are approved. Prescribed cell measures move to v0.9 and
mixed separator-plus-measure work to v0.10.

## Plan lifecycle

- **Draft** — scope and design are being reviewed, or activation mechanics remain.
- **Active** — approved for implementation and linked to a milestone.
- **Completed** — released and moved to the archive.
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
