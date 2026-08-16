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
4. [Canonical inverse namespace and separator organization](0004-canonical-inverse-namespace.md)
5. [Common tessellation result contract](0005-tessellation-result-contract.md)
6. [v0.8 is a cleanup-only compatibility-removal release](0006-v0.8-cleanup-release.md)
7. [Separator inverse objective contract](0007-separator-objective-contract.md)
8. [Separator solver and linear-backend selection](0008-separator-solver-and-linear-backend.md)
9. [Certified scalar proximal solver](0009-certified-scalar-proximal-solver.md)
10. [Strict native construction preconditions](0010-native-construction-preconditions.md)
11. [Strict public input and ownership contract](0011-strict-input-and-ownership-contract.md)
12. [Certified periodic nearest-image geometry](0012-certified-periodic-image-geometry.md)
13. [Central generator preparation and mandatory backend safety](0013-central-generator-preparation-and-backend-safety.md)
14. [Two-layer separator observation and source identity](0014-separator-observation-and-source-identity.md)
15. [Atomic separator active-set final state](0015-atomic-separator-active-state.md)
16. [Severity-complete tessellation diagnostics](0016-severity-complete-tessellation-diagnostics.md)
17. [Functional stabilization precedes 1.0 and later inverse families](0017-v0.9-functional-stabilization-before-1.0.md)

New records should describe context, decision, consequences, alternatives, and
links to the active plan and relevant issues. See the
[development workflow](../development-workflow.md). Do not create a record for
every small implementation choice.
