# Decision records

Decision records capture durable architectural choices and their trade-offs.
They complement release plans and GitHub issues: an active plan defines approved
scope, an issue tracks implementation, and a decision record explains why a
long-lived choice was made.

Each record has a status:

- **Proposed**: under discussion;
- **Accepted**: governs current development;
- **Superseded**: replaced by a later record;
- **Rejected**: considered but not adopted.

## Records

1. [Project scope and layering](0001-project-scope-and-layering.md)
2. [Weights, radii, gauge, and component offsets](0002-weights-radii-and-gauge.md)
3. [Compatibility-first API evolution](0003-compatibility-first-evolution.md)

New records should describe context, decision, consequences, alternatives, and
links to the active plan and relevant issues. See the
[development workflow](../development-workflow.md). Do not create a record for
every small implementation choice.
