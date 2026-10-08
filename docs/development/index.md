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
| What work is planned next? | [Reboot transition / ADR 0027](decisions/0027-v0.9-reboot-from-v0.8.md) and [development plans](plans/index.md); future release scope is undecided |
| Which concrete APIs are stable, provisional, experimental, removed, or internal? | [v0.8 API inventory](api-inventory.md) and [API lifecycle](api-lifecycle.md) |
| How should repository documentation be written? | [Documentation conventions](documentation-conventions.md) |
| How are modules and layers organized? | [Architecture](architecture.md) |
| Why was a durable choice made? | [Decision records](decisions/index.md) |
| What is planned over several releases? | [Roadmap](../project/roadmap.md) |
| What historical work produced v0.8? | [v0.8 audit/remediation record](audits/v0.8-pre-release.md), [completed remediation plan](plans/archive/v0.8-remediation.md), GitHub issues, and milestones |
| What did the former v0.9 attempt establish? | [Historical knowledge](reboot/index.md), [requirements inventory](reboot/requirements-inventory.md), and [evidence index](reboot/evidence-index.md); historical acceptance is separate from reboot adoption |
| How do I prepare a change? | [`CONTRIBUTING.md`](https://github.com/DeloneCommons/pyvoro2/blob/main/CONTRIBUTING.md) |
| What changed historically? | [Changelog](../about/changelog.md) |
| What is included in v0.8.0? | [v0.8.0 release notes](../project/release-notes-v0.8.md) |

## Authority and status

The current source code and tests remain the source of truth for implemented
behavior. User guides and reference pages describe that behavior for callers.

[ADR 0027](decisions/0027-v0.9-reboot-from-v0.8.md) governs the operational
reboot from published v0.8.0. PR1 restores development operations; PR2 owns
curated historical knowledge migration. The former v0.9 attempt and ADRs
0018–0026 are evidence, not adopted architecture. Its Checkpoint C remains
unaccepted. Future geometry architecture and release scope are undecided;
prescribed measures and mixed inversion are research candidates only.

The v0.7 and v0.8 plans are completed and archived. Existing v0.8 behavior and
its API inventory remain the baseline; the development version does not imply
new numerical behavior or API promotion. The roadmap distinguishes current
approved work from future candidates.

Detailed progress belongs in GitHub issues. The API inventory is updated with
public changes, and completed user-visible behavior is recorded in the dated
changelog section while a fresh `[Unreleased]` section remains for future work.

User-facing lifecycle and migration decisions are summarized in
[Choosing an API](../guide/choosing-api.md) and
[Migrating from v0.6.3 through v0.8](../guide/migration-v0.7.md).
