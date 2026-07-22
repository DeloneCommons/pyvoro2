# Development workflow

This document defines how pyvoro2 work moves from an idea to an accepted
release. It is the shared process for maintainers, contributors, and coding
agents.

The workflow is intentionally lightweight enough for a single-maintainer
research package, while leaving a visible record of scope, design, validation,
and release decisions.

## Branch model

| Branch | Role |
|---|---|
| `main` | Latest stable public release. |
| `dev` | Integration branch for the active release plan. |
| Feature branches | Optional issue-scoped branches for work that should be reviewed before integration into `dev`. |

Direct changes to `main` should be limited to exceptional release or repository
administration fixes. Normal development is integrated through `dev` and
released from a reviewed state.

## Planning levels

| Artifact | Question it answers |
|---|---|
| [Roadmap](../project/roadmap.md) | Where is the project going across releases? |
| Active release plan | What must the current release achieve, and how will completion be judged? |
| Decision record | Why was a durable architectural or policy choice made? |
| GitHub milestone | Which issues belong to one release outcome? |
| GitHub issue | What concrete unit of work is being implemented and accepted? |
| Pull request or commit | What exact code and documentation changed? |
| [Changelog](../about/changelog.md) | What completed user-visible behavior was delivered? |

A substantial change should be traceable through these levels without requiring
access to a private chat or personal notes.

## When a release plan is required

A release plan is required when work does one or more of the following:

- changes several public APIs or result schemas;
- establishes a new compatibility boundary;
- adds an inverse observation family or solver;
- changes backend, periodic-image, or numerical semantics across modules;
- spans several dependent issues;
- defines a release whose acceptance requires coordinated documentation,
  migration, benchmarks, or downstream validation.

A small bug fix, documentation correction, additional regression test, or
isolated maintenance change can use the shorter issue or pull-request path. It
still needs tests, documentation, and a changelog entry when user-visible.

## Plan statuses

| Status | Meaning |
|---|---|
| **Draft** | Scope and decisions are being reviewed. It is not blanket authorization to implement unresolved design choices. |
| **Active** | The maintainer has approved the release scope and issue decomposition. |
| **Completed** | The final release source was approved, its outcome was recorded, and the plan was archived. A versioned checklist may still track external CI, tag, publication, and verification operations. |
| **Superseded** | Another plan replaced the work before completion; the replacement is linked. |

For the current lead-maintainer model, activation can be recorded directly in
the plan metadata with an approval date. If the project later adds formal
reviewers, the same field can link to the approving issue or pull request.

## Lifecycle of planned work

### 1. Capture the proposal

Start with a roadmap outcome, design issue, bug report, or research need.
Describe the user or scientific problem before proposing classes or module
names.

If the proposal changes a durable boundary, identify the decision record that
must be created or updated.

### 2. Draft the release or workstream plan

Create a file under `docs/development/plans/` from the
[plan template](plans/template.md). The draft should define:

- intended outcome and baseline;
- scope and non-goals;
- accepted architectural constraints;
- unresolved decision gates;
- work packages and dependencies;
- compatibility and migration expectations;
- validation strategy;
- documentation outputs;
- release acceptance criteria;
- risks and likely deferrals.

Do not turn the plan into a daily checklist. It should remain useful if issue
order changes.

### 3. Review and activate the plan

Before activation:

- resolve or explicitly schedule blocking design decisions;
- confirm that work packages are coherent and independently reviewable;
- identify which outcomes are required and which are conditional;
- check that the plan does not promise unsupported dates or speculative APIs;
- create or select the GitHub milestone.

Set `Status: Active`, record the approval date, and update the plans index.

### 4. Decompose work into issues

Create small issues with:

- a clear user or scientific outcome;
- links to the plan, API inventory, and relevant decision records;
- dependencies;
- explicit in-scope and out-of-scope behavior;
- baseline behavior that must be characterized or preserved;
- tests and documentation requirements;
- acceptance criteria stated as observable behavior;
- lifecycle, compatibility, warning, and removal impact where relevant;
- stop conditions for choices that require maintainer review.

Issues track progress. The plan tracks release structure. Avoid copying the full
plan into every issue.

### 5. Implement and validate

Work issue by issue on `dev` or an issue-scoped branch.

Every implementation change should:

- preserve accepted architecture and compatibility rules;
- add tests for the behavior or invariant it changes;
- update guides, reference, and current-architecture text when applicable;
- update generated files with their sources;
- record durable decisions in decision records;
- report the validation commands that were run.

For notebook changes, keep execution, validation, and publication separate:
refresh selected source notebooks with `python tools/execute_notebooks.py`, run
`python tools/check_notebooks.py` for non-mutating metadata and clean-kernel
validation, then regenerate stored-output pages with
`python tools/export_notebooks.py`. Ordinary MkDocs builds do not execute
notebooks.

A coding agent must read the active plan, linked issue, and relevant decision
records before changing public behavior. Unresolved design gates require a
maintainer decision rather than an invented implementation choice.

### 6. Accept a change into the release

When a user-visible change is accepted into `dev`:

- close or update the corresponding issue;
- add or refine its entry under `[Unreleased]` in `CHANGELOG.md`;
- ensure current documentation describes the implemented behavior;
- retain migration notes and compatibility tests;
- update the release plan only when scope, dependencies, risks, or release gates
  changed materially.

Do not delete completed work packages from the plan. GitHub issues show detailed
completion; the plan preserves the intended structure of the release.

### 7. Perform release review

Before releasing:

- check every required release gate in the active plan;
- complete the public API lifecycle inventory;
- resolve, defer, or rescope every open milestone issue;
- move deferred work to a later milestone or plan and record the reason;
- synchronize current-versus-target documentation;
- review migration and deprecation behavior;
- run the full release validation;
- convert `[Unreleased]` entries into the dated release section.

Conditional work that was not required for release should be recorded as
omitted or deferred, not silently removed from history.

### 8. Finalize the release source and archive the plan

After repository-local qualification and maintainer approval, include the
completed plan in the final release source:

1. add an outcome section to the plan;
2. link the intended release tag, changelog section, milestone, important
   decision records, and deferred follow-up issues;
3. set `Status: Completed`;
4. move the file to `docs/development/plans/archive/`;
5. update the plans index and roadmap if the next release direction changed.

A version-specific release checklist may remain open for external operations
that cannot be proven inside the source commit: supported CI, tag creation,
tagged artifact validation, package-index publication, deployed documentation,
and public smoke tests. Archiving the plan does not imply that unchecked
operations passed. A failed operational gate requires a corrective release
commit before publication or an explicit documented recovery.

The changelog remains the concise user-facing record of delivered behavior. The
archived plan preserves why the release was structured as it was and what was
deferred.

## Changelog timing

Use `[Unreleased]` continuously rather than reconstructing release notes at the
end.

Add an entry when a change is accepted into `dev` if it affects:

- public API or result semantics;
- scientific or numerical behavior;
- user-visible diagnostics or failures;
- packaging, installation, supported environments, or release tooling;
- public documentation or project policy in a meaningful way.

Do not add an entry for every internal refactor. Group related changes into
clear user-facing statements during release review.

## Decision-record triggers

Create or update a decision record when work fixes a durable choice about:

- project scope or layer boundaries;
- public namespace ownership;
- a common result or observation contract;
- gauge, units, orientation, periodic-image, or ID semantics;
- optional versus required dependencies;
- compatibility or deprecation policy;
- backend replacement or a major solver strategy.

An issue discussion is sufficient for local implementation choices that do not
create a durable public or architectural constraint.

## Scope changes and deferrals

When release scope changes:

- increment the plan revision and update the plan revision log;
- state whether the outcome was added, removed, or made conditional;
- link the maintainer decision or issue;
- move deferred issues to a later milestone;
- preserve the original rationale in Git history and, when useful, in the plan.

Do not rewrite the roadmap to mirror every scope adjustment. Update it only when
the version-level direction changes.

## Requirements for agent-assisted work

Agent-assisted changes follow the same process as human-authored changes.

Agents must:

- read `AGENTS.md`, the active plan, linked issue, relevant ADRs, and API
  inventory before implementation;
- use repository documents rather than private conversation as authority;
- work issue by issue unless the issue explicitly combines inseparable work;
- choose clean internal details within the accepted public contract rather than
  asking for approval on every local refactor;
- surface conflicts with ADRs, new mandatory dependencies, unexplained numerical
  changes, scope expansion, or unplanned public API choices;
- keep current and planned language distinct;
- include tests, documentation, generated outputs, inventory, and changelog
  changes where required;
- report public behavior changed, compatibility/lifecycle impact, checks passed,
  and any deviations or follow-ups;
- never mark a plan or decision as approved without an explicit maintainer
  instruction.

Any important decision reached in a chat must be transferred to an issue,
release plan, API inventory, or decision record before it governs later work.
The process should not produce duplicate status documents when the issue, tests,
docs, changelog, and plan already provide the needed record.

## Completion checklist for substantial work

A planned change is complete when:

- its issue acceptance criteria are met;
- implementation and deterministic tests agree;
- compatibility behavior is tested and documented;
- guides, reference, architecture, and decision records are consistent;
- generated files are synchronized;
- the relevant `[Unreleased]` changelog entry exists;
- required validation commands pass;
- no unresolved decision is hidden in implementation details.
