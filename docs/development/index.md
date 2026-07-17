# Development documentation

This section explains how pyvoro2 is structured, how planned work is approved
and tracked, which parts of the public API are intended to remain stable, and
why major design decisions were made. It is written for maintainers,
contributors, reviewers, coding agents, and downstream package authors.

## Where to look

| Question | Authoritative source |
|---|---|
| How do I use the current package? | [User guide](../guide/concepts.md) and [API reference](../reference/index.md) |
| What mathematics does it implement? | [Theory](../theory/index.md) |
| How does work move from proposal to release? | [Development workflow](development-workflow.md) |
| What work is planned for the current release? | [Development plans](plans/index.md), currently the [draft v0.7 plan](plans/v0.7.md) |
| How should repository documentation be written? | [Documentation conventions](documentation-conventions.md) |
| How are modules and layers organized? | [Architecture](architecture.md) |
| What compatibility can I rely on? | [API lifecycle](api-lifecycle.md) |
| Why was a durable choice made? | [Decision records](decisions/index.md) |
| What is planned over several releases? | [Roadmap](../project/roadmap.md) |
| What concrete work is in progress? | GitHub issues and milestones |
| How do I prepare a change? | [`CONTRIBUTING.md`](https://github.com/DeloneCommons/pyvoro2/blob/main/CONTRIBUTING.md) |
| What changed historically? | [Changelog](../about/changelog.md) |

## Authority and status

The current source code and tests remain the source of truth for implemented
behavior. User guides and reference pages describe that behavior for callers.

Accepted decision records and architecture documentation define durable
boundaries. An **active** development plan defines approved release scope and
gates; a **draft** plan records proposals and open decisions but is not blanket
implementation approval. The roadmap describes version-level direction rather
than current functionality.

Detailed progress belongs in GitHub issues. Completed user-visible behavior is
recorded under `[Unreleased]` in the changelog and finalized at release.
