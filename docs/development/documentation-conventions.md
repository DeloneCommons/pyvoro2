# Documentation conventions

This page defines the language and maintenance rules used across the pyvoro2
repository. It applies to text written by maintainers, contributors, and coding
agents. The same standard applies whether a draft was written manually or with
AI assistance.

The aim is to keep each document useful outside the conversation or development
session in which it was created.

## Document roles

Each kind of document has one primary responsibility.

| Location | Primary responsibility |
|---|---|
| `docs/guide/` | Explain current user workflows and callable behavior. |
| `docs/theory/` | Explain the mathematics and scientific interpretation independently of provisional API names. |
| `docs/reference/` | Record exact current imports, signatures, fields, and docstrings. |
| `docs/development/architecture.md` | Describe the implemented architecture and accepted target responsibilities. |
| `docs/development/api-lifecycle.md` | Define stability, compatibility, and deprecation policy. |
| `docs/development/api-inventory.md` | Assign lifecycle status to concrete public names, defaults, result fields, and scientific semantics for the active release. |
| `docs/development/decisions/` | Record durable choices, alternatives, and consequences. |
| `docs/development/plans/` | Define the scope, dependencies, validation, and release gates for active or archived development work. |
| `docs/project/roadmap.md` | Describe version-level outcomes and long-term direction. |
| GitHub issues and milestones | Track concrete implementation work and its current status. |
| `CHANGELOG.md` | Record completed user-visible changes. |

Do not use one document as a substitute for another. In particular:

- the roadmap is not a task checklist;
- a release plan is not an API reference;
- an issue is not a durable architecture decision;
- the changelog is not a list of intentions;
- an archived plan does not replace release notes.

## Public project language

Use release names, workstream names, or issue titles instead of private planning
labels.

Preferred:

- “the v0.7 forward and separator API work”;
- “the prescribed-cell-measure workstream”;
- “the common forward-result design issue.”

Avoid:

- “Stage 0,” “Stage 1,” or similar chat-specific numbering;
- “as discussed earlier” without a repository link;
- “our current plan” when the relevant plan can be named;
- “the next step” when a release, work package, or issue can be identified;
- references to private brainstorming as authority.

The word *stage* is still appropriate for an algorithm, experiment, or data
processing pipeline. The restriction concerns internal project-management
labels that have no stable meaning to repository readers.

## Status vocabulary

Use the following terms consistently.

| Term | Meaning |
|---|---|
| **Implemented** | Present in the branch or release being documented and covered by tests or direct inspection. |
| **Targeted for vX.Y** | Accepted as an outcome for that release, but not necessarily implemented yet. |
| **Proposed** | Under discussion and not yet accepted as a requirement. |
| **Stable** | Implemented public API covered by the normal compatibility policy. |
| **Provisional** | Implemented and public, but still being evaluated before a stronger compatibility commitment. |
| **Experimental** | Implemented research functionality with explicitly weaker compatibility or convergence guarantees. |
| **Compatibility-only** | Retained for existing callers but not preferred for new code. |
| **Deprecated** | Still available temporarily with a documented replacement and migration path. |
| **Future research direction** | Worth recording, but not committed to a release. |
| **Deferred** | Removed from the active release scope and linked to a later issue, milestone, or plan. |

`Stable`, `Provisional`, `Experimental`, `Compatibility-only`, and `Deprecated`
refer to public API lifecycle. `Targeted`, `Proposed`, and `Deferred` refer to
planning status. See [API lifecycle](api-lifecycle.md) for the full compatibility
policy.

When text combines current and future behavior, label the boundary explicitly.
For example:

> The current v0.6.3 API accepts `radii=`. Direct `weights=` input is targeted
> for v0.7; the exact result metadata remains provisional.

Avoid words such as “soon,” “eventually,” or “later” when a more precise status
is available.

## Tense and claims

- Use present tense for implemented behavior: “`compute` returns …”.
- Use future or target language for accepted release work: “v0.7 will provide
  …” or “the v0.7 plan requires …”.
- Use conditional language for unresolved design choices: “the result may use
  …”.
- Do not describe an object as implemented before it exists. A preferred target
  may be named before implementation only when an accepted decision and active
  plan define it; stability still requires the API lifecycle inventory.
- Do not infer implementation from a roadmap, plan, or decision proposal.

Source code and tests define implemented behavior. Documentation should be
corrected when it claims more or less than the current code provides.

## User-guide style

User guides are task-oriented and should answer:

1. what problem the workflow solves;
2. which current public objects to use;
3. what the important inputs and outputs mean;
4. which failure modes or limitations need attention;
5. where to find the exact reference and deeper theory.

Examples in guides must run against the documented branch or release. Planned
imports and signatures belong in a development plan, not in executable user
examples.

## Theory style

Theory pages should use a paper-like mathematical style while remaining
accessible to a broad scientific audience.

- Begin with the geometric or scientific question.
- Introduce a two-site example before the general pair law.
- Introduce a three-site cycle before incidence matrices and cycle spaces.
- Define terminology before shorthand or symbols.
- Explain the meaning of an equation before or immediately after presenting it.
- Separate exact statements, numerical algorithms, heuristics, and empirical
  observations.
- Keep periodic extensions after the nonperiodic idea when that ordering is
  clearer.
- Prefer “separator observation,” “observed pair,” and “observation graph” in
  introductory prose; introduce row and matrix terminology only when needed.
- Link to the manuscript for full proofs, literature discussion, and benchmark
  detail rather than copying large passages.

A theory page may simplify notation, but it must preserve the scientific
meaning. In particular, it must not conflate:

- power weights with backend radii;
- global gauge with unidentified offsets between disconnected observation
  components;
- algebraic separator fit with realized face support;
- connector visibility with full face realization;
- a practical active-set algorithm with a proved global solver.

## Architecture and decision-record style

Architecture documentation should distinguish **current implementation** from
**target responsibility**. Concrete class names are examples until an accepted
decision and implementation make them authoritative.

Decision records should contain:

- status and date;
- context;
- the decision;
- consequences;
- alternatives considered;
- links to the release plan or issues that implement the decision.

Do not rewrite accepted historical context merely because the implementation
has moved on. Supersede the record and link to its replacement.

## Release-plan style

A release plan describes outcomes, constraints, work packages, dependencies,
validation, and release gates. It should not duplicate issue-by-issue progress.

- Use a version or named workstream in the filename and title.
- Record `Draft`, `Active`, `Completed`, or `Superseded` status.
- Increment the plan revision when scope, decision gates, or release criteria
  change materially.
- Mark unresolved choices as decision gates.
- Link concrete work to GitHub issues once they exist.
- Do not delete completed work packages from the plan.
- Record scope changes and deferrals in the plan revision log.
- Add an outcome summary before archiving the plan after release.

See [Development workflow](development-workflow.md) and the
[plan template](plans/template.md).

## Roadmap and changelog style

The roadmap uses version-level outcomes and long-term directions. It should be
stable under issue reordering and implementation detail changes.

The changelog contains completed, user-visible changes. Add entries when a
change is accepted into the integration branch, then finalize the release
heading and wording during release review. Purely internal refactors need a
changelog entry only when they materially affect users, contributors, build
behavior, or scientific interpretation.

## Cross-references and duplication

Prefer links to repeated policy text. A short summary may appear in
`AGENTS.md` or `CONTRIBUTING.md`, but this page is authoritative for writing
conventions.

When a change affects several document types, update each according to its role:

- implementation semantics → guide, reference, and tests;
- durable architecture → decision record and architecture;
- release scope → active plan and issues;
- long-term direction → roadmap;
- completed user-visible behavior → changelog.

## Generated documents

- Edit `docs/index.md`, then regenerate `README.md` with
  `python tools/gen_readme.py`.
- Edit source notebooks under `notebooks/`, then regenerate
  `docs/notebooks/*.md` with `python tools/export_notebooks.py`.

Generated outputs must be committed with their sources.

## Documentation review checklist

Before merging a documentation change, check that:

- current and planned behavior are labelled correctly;
- private planning language has been removed;
- terminology is defined before it is used;
- guides use executable current APIs;
- theory does not depend on provisional class names;
- the roadmap, active plan, issues, and changelog are not duplicating one
  another;
- links point to authoritative repository documents;
- generated README and notebook pages are synchronized;
- `mkdocs build --strict` passes.
