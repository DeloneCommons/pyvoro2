# Development documentation

This section explains how pyvoro2 is structured, how planned work is approved
and tracked, which parts of the public API are intended to remain stable, and
why major design decisions were made. It is written for maintainers,
contributors, reviewers, coding agents, and downstream package authors.

## Where to look

| Question | Authoritative source |
|---|---|
| How do I use the current package? | [User Guide](../guide/index.md) and [API reference](../reference/index.md) |
| What mathematics does it implement? | [Theory](../theory/index.md) |
| How does work move from proposal to release? | [Development workflow](development-workflow.md) |
| What work is planned next? | [Reboot transition / ADR 0027](decisions/0027-v0.9-reboot-from-v0.8.md) and [development plans](plans/index.md); future release scope is undecided |
| Which concrete APIs are stable, provisional, experimental, removed, or internal? | [v0.8 API inventory](api-inventory.md) and [API lifecycle](api-lifecycle.md) |
| How do I maintain example notebooks? | [Notebook maintenance](notebook-workflow.md); user-facing catalogue in [Examples](../examples/index.md) |
| Where are project metadata and release history? | [Project overview](../project/index.md) |
| How should repository documentation be written? | [Documentation conventions](documentation-conventions.md) |
| How are modules and layers organized? | [Architecture](architecture.md) |
| Why was a durable choice made? | [Decision records](decisions/index.md) |
| What is planned over several releases? | [Roadmap](../project/roadmap.md) |
| Which capabilities are being considered for implementation? | [Maintainer reassessment register](https://github.com/DeloneCommons/pyvoro2/blob/dev/notes/maintainer-review/review-register.md) on `dev`; these are proposals or selected research directions, not approved release scope. |
| What remains to examine? | [Non-blocking functional coverage checklist](https://github.com/DeloneCommons/pyvoro2/blob/dev/notes/maintainer-review/functional-coverage.md). |
| What is the documentation workstream? | [Documentation refresh draft](plans/documentation-refresh.md). |
| What historical work produced v0.8? | [v0.8 audit/remediation record](audits/v0.8-pre-release.md), [completed remediation plan](plans/archive/v0.8-remediation.md), GitHub issues, and milestones |
| What did the former v0.9 attempt establish? | [Historical knowledge](reboot/index.md), [requirements inventory](reboot/requirements-inventory.md), and [evidence index](reboot/evidence-index.md); historical acceptance is separate from reboot adoption |
| How do I prepare a change? | [`CONTRIBUTING.md`](https://github.com/DeloneCommons/pyvoro2/blob/main/CONTRIBUTING.md) |
| What changed historically? | [Changelog](../project/changelog.md) |
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

## Known documentation UI limitations

These are documentation-interface limitations, not pyvoro2 computational or
public API defects. **Native Search:** with Zensical 0.1.2 / `zensical-ui` 0.0.42,
light-mode result breadcrumbs have approximately **3.33:1** contrast, below
the WCAG AA normal-text threshold of **4.5:1**, affecting 12 elements in the
recorded results. Search fixes foreground opacity at 45% inside its Shadow DOM
and exposes no documented breadcrumb-specific styling interface. An upstream
accessible default or supported styling mechanism is needed.

**Scrollable API tables:** at a 320 px viewport, axe reports
`scrollable-region-focusable` for one native `.md-typeset__scrollwrap` in the
API Reference, which lacks the expected keyboard-focus semantics. This may
particularly affect Safari keyboard navigation; Safari was not tested directly.
The same finding reproduces on the unchanged DOC-W4-C1 baseline. An upstream
focusability/accessibility correction is needed. Both findings are tracked in
[#142](https://github.com/DeloneCommons/pyvoro2/issues/142); see the
[commit-pinned evidence](https://github.com/DeloneCommons/pyvoro2/blob/9006390a9f6807aad48e48cb7136c489629be523/review/w4-c2/native-findings.json)
and [PR #141](https://github.com/DeloneCommons/pyvoro2/pull/141).
