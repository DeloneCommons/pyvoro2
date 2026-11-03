# Maintainer reassessment notebook

Public working notes for an independent maintainer-level reassessment of pyvoro2, its Voro++ backends, scientific algorithms, architecture, and documentation.

**Research notebook, not normative project documentation.** Notes may contain observations, hypotheses, rejected ideas, candidate improvements, and documentation drafts. They do not authorize implementation, change an accepted API contract, or commit work to a release. Durable decisions belong in accepted ADRs; implementation belongs in focused PRs against `dev`.

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
