# Maintainer reassessment notebook

This directory contains public working notes from an independent maintainer-level reassessment of pyvoro2, its Voro++ backends, scientific algorithms, code architecture, and documentation.

**This is a research notebook, not normative project documentation.**

The goal is to understand the important code paths, mathematical assumptions, numerical limitations, and design choices before selecting further changes.

Notes may include verified observations, preliminary hypotheses, rejected ideas, proposed improvements, and drafts of future documentation.

A note does not authorize implementation or change an accepted API contract. Durable decisions belong in accepted ADRs, and production changes belong in focused pull requests against `dev`.

## Initial notebooks

- `documentation-notes.md` — ideas for explaining pyvoro2 to users and developers.
- `potential-work.md` — candidate improvements and questions requiring investigation.

The notebook may grow into more focused files as necessary. Important conclusions should reference the relevant code, tests, historical evidence, and source revisions whenever practical.

Historical context: [ADR 0027](../../docs/development/decisions/0027-v0.9-reboot-from-v0.8.md) and the [historical knowledge inventory](../../docs/development/reboot/index.md).
