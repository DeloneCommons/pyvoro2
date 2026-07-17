# Development plans

Development plans connect the long-term [roadmap](../../project/roadmap.md) to
concrete GitHub issues. A plan defines the outcome, boundaries, dependencies,
validation, and release gates for one release or substantial workstream.

Plans are not daily task trackers. Current progress belongs in GitHub issues and
milestones.

## Current plan

| Plan | Status | Target | Purpose |
|---|---|---|---|
| [v0.7 forward and separator API stabilization](v0.7.md) | Draft | v0.7.0 | Establish the common forward/result and separator-inverse contract needed by chemvoro and later inverse methods. |

The v0.7 plan remains a draft until the maintainer approves its scope,
decision process, and initial issue decomposition. The blocking API decisions
may then be resolved as the first planned issues before dependent
implementation begins.

## Plan lifecycle

- **Draft** — scope and design gates are under review.
- **Active** — approved for implementation and linked to a milestone.
- **Completed** — released and moved to the archive.
- **Superseded** — replaced by another named plan.

See [Development workflow](../development-workflow.md) for the complete process.

## Starting a plan

1. Copy the [plan template](template.md).
2. Use a version or descriptive workstream name.
3. Define outcome, scope, non-goals, decision gates, work packages, validation,
   and release acceptance criteria.
4. Review the draft before creating the full issue set.
5. Activate it only after explicit maintainer approval.

## Archive

Completed and superseded plans are kept in the [plan archive](archive/index.md).
They complement the changelog by preserving intent, dependencies, decisions,
and deferrals.
