# Development documentation

This section explains how pyvoro2 is structured, which parts of its public API
are intended to remain stable, and why major design decisions were made. It is
written for maintainers, contributors, reviewers, and downstream package
authors.

## Where to look

| Question | Authoritative source |
|---|---|
| How do I use the current package? | [User guide](../guide/concepts.md) and API reference |
| What mathematics does it implement? | [Theory](../theory/index.md) |
| How are modules and layers organized? | [Architecture](architecture.md) |
| What compatibility can I rely on? | [API lifecycle](api-lifecycle.md) |
| Why was a durable choice made? | [Decision records](decisions/index.md) |
| What is planned over several releases? | [Roadmap](../project/roadmap.md) |
| What is being implemented now? | GitHub issues and milestones |
| How do I prepare a change? | [`CONTRIBUTING.md`](https://github.com/DeloneCommons/pyvoro2/blob/main/CONTRIBUTING.md) |
| What changed historically? | [Changelog](../about/changelog.md) |

The current source code and tests remain the source of truth for implemented
behavior. Target-architecture sections describe constraints on future work; they
do not imply that every named object already exists.
