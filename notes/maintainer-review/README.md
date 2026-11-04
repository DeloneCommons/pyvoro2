# Maintainer reassessment notebook

Public working notes for an independent maintainer-level reassessment of pyvoro2, its Voro++ backends, scientific algorithms, architecture, and documentation.

**Temporary working materials, not normative project documentation.** These notes preserve the maintainer's reassessment, observations, hypotheses, rejected ideas, candidate improvements and documentation drafts during the v0.9 reboot. They do not authorize implementation, change an accepted API contract, or commit work to a release. Durable decisions belong in accepted ADRs; implementation belongs in focused PRs against `dev`.

## How these notes are retired

This directory is a **transitional staging area**, not a permanent parallel documentation system. When a topic is addressed, transfer its enduring content to the appropriate maintained home: `docs/guide/` or `docs/theory/` for user-facing explanations, `docs/development/architecture.md` and accepted ADRs for architecture/contracts, GitHub issues and tests for actionable requirements and regressions, and relevant development plans for approved work. The register remains the single **candidate/status index** while reassessment is ongoing; it is not an implementation checklist.

After migration, verify that significant caveats, evidence references, numerical counterexamples and open questions remain accessible, fix incoming links, then remove the superseded working note. Some independently valuable deferred research can remain until its own decision or archival destination is identified; **there is no promise that all notes disappear by v0.9 or v1.0**. Before deleting the former review branch, verify that current notes and any valuable earlier-only evidence are reachable from `dev` or deliberately archived; a squash merge alone does not preserve the source branch's intermediate history. Git history is not a substitute for preserving active scientific requirements in their maintained location.

## Review decisions

[`review-register.md`](review-register.md) is the single current status index for the reassessment. Selected capability directions are **not** implementation approval; historical evidence stays in its source documents.

The [functional coverage checklist](functional-coverage.md) records **non-blocking questions for later review**. It is not a release gate and need not precede higher-priority work on point/edge identity or inverse row-wise parameters.

## Browse by subject

| Folder | What is inside |
| --- | --- |
| [`overview/`](overview/README.md) | Short descriptions of the software and underlying algorithms. |
| [`contracts/`](contracts/README.md) | Mathematical expectations, implementation limitations, and known failure classes. |
| [`potential-work/`](potential-work/README.md) | Substantial independent research notes only; shorter candidate decisions and comments belong in the register. |
| [`documentation-notes/`](documentation-notes/README.md) | Ideas for eventual user and developer documentation. |

Each directory has its own short index. Prefer concise, focused notes and links to existing evidence over duplicating full reports.

Important conclusions should reference source revisions, tests, historical evidence, and mathematical oracles where possible.

Context: [ADR 0027](../../docs/development/decisions/0027-v0.9-reboot-from-v0.8.md) · [Historical knowledge inventory](../../docs/development/reboot/index.md).
