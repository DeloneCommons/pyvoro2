# Development plans

Development plans connect the long-term [roadmap](../../project/roadmap.md) to
concrete GitHub issues. A plan defines the outcome, boundaries, dependencies,
validation, and release gates for one release or substantial workstream.

Plans are not daily task trackers. Current progress belongs in GitHub issues and
milestones.

## Current plans

The approved current workstream is the operational reboot under
[issue #120](https://github.com/DeloneCommons/pyvoro2/issues/120) and
[ADR 0027](../decisions/0027-v0.9-reboot-from-v0.8.md): PR1 operational bootstrap,
then PR2 curated historical knowledge migration. This is not a new geometric
implementation plan or release qualification. No future release scope is fixed.

Published v0.8.0 is the source baseline. The completed
[v0.8 technical-maintenance plan](archive/v0.8.md),
[v0.8 remediation plan](archive/v0.8-remediation.md), and
[v0.7 stabilization plan](archive/v0.7.md) remain archived history.

The former v0.9 attempt is preserved at the protected archive named in ADR
0027. Its Checkpoint C remains pending/unaccepted, and pre-Phase-D recovery,
WP12, and WP13 remain unfinished. Its plans and ADRs 0018–0026 are not active
reboot authority. The [historical knowledge guide](../reboot/index.md) maps
their capabilities, acceptance boundaries, evidence and unreviewed adoption
candidates. It is a reading resource, not a replacement release plan.

ADR 0017's future v0.9/1.0/v1.1/v1.2 sequence is superseded without reviving
ADR 0006's earlier sequence. Prescribed measures and mixed inversion are
possible research candidates only, with no implementation, API, architectural
relationship, release target, or pre-1.0 inclusion approved. Promotion requires
separate mathematical analysis, representative use cases, validation strategy,
and maintainer approval.

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
