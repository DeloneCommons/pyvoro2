# Development plan template

Copy this page when creating a release or major workstream plan. Remove guidance
that does not apply, but preserve the metadata, scope, decision, validation, and
outcome sections.

A plan should describe release structure rather than duplicate issue-by-issue
progress. Link issues after they are created.

---

# <Release or workstream> development plan

- **Status:** Draft
- **Plan revision:** 0.1
- **Target release:** <vX.Y.Z or not release-bound>
- **Base release:** <version>
- **Integration branch:** `dev`
- **Maintainer:** <name>
- **Approved:** Not yet approved
- **Last revised:** <YYYY-MM-DD>
- **GitHub milestone:** <link or to be created>

## Intended outcome

Describe the user-visible and architectural state that should exist when this
plan is complete. Prefer a small number of testable outcomes over a list of
proposed classes.

## Baseline and motivation

Summarize the implemented starting point, the problem being solved, and the
reason the work belongs in one plan.

## Accepted constraints

Link applicable decision records and list invariants that are not being
reopened by this plan.

## Scope

List required release outcomes.

## Non-goals

List adjacent work that this plan deliberately does not include.

## Decision gates

| ID | Question | Why it blocks or shapes work | Resolution artifact | Status |
|---|---|---|---|---|
| D-1 | <decision> | <impact> | <ADR, issue, or maintainer decision> | Open |

For each important decision, describe the realistic alternatives and a
preliminary recommendation without presenting it as accepted.

## Work packages

| ID | Work package | Required for release? | Depends on | Issue(s) |
|---|---|---|---|---|
| WP-1 | <coherent outcome> | Yes | — | To be created |

For each work package, define:

- purpose;
- deliverables;
- compatibility and lifecycle requirements;
- validation;
- acceptance criteria.

Do not use this table as a daily progress tracker. GitHub issues and the
milestone show completion state.

## Dependency and implementation order

Describe which work can proceed in parallel and which decisions or packages
must land first.

## Compatibility and migration

State which existing APIs, defaults, outputs, and scientific semantics must
remain available, and how preferred replacements will be introduced.

## Validation strategy

Define the required test matrix, numerical equivalence checks, downstream
integration, benchmarks, documentation builds, and release checks.

## Documentation outputs

List guides, references, migration material, architecture updates, decision
records, examples, and generated files required by the plan.

## Release acceptance criteria

List the gates that must all be satisfied before release. Separate required
criteria from conditional improvements that may be deferred.

## Risks and mitigations

| Risk | Consequence | Mitigation |
|---|---|---|
| <risk> | <impact> | <response> |

## Proposed issue decomposition

List durable issue titles and dependencies. Final issue bodies live on GitHub.
For issue-scoped human or coding-agent work, each issue should state the
observable outcome, in/out of scope behavior, accepted decisions, baseline to
preserve, tests/docs required, and stop conditions for maintainer review.

## Public API inventory

Link the release inventory when the plan changes public names, defaults, return
schemas, record fields, or scientific semantics. The inventory should be updated
during implementation and approved before release, not reconstructed afterward.

## Linked decisions and issues

Add links as the plan is activated and implemented.

## Plan revisions

| Date | Change | Reason / link |
|---|---|---|
| <date> | Initial draft | <context> |

## Outcome

Complete this section during release review before archiving the plan:

- release and date;
- delivered outcomes;
- compatibility result;
- validation summary;
- deferred work;
- links to release notes, milestone, and follow-up plans.
