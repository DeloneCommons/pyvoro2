# Development audits

Development audits record evidence discovered after ordinary implementation work and
before release qualification. They are not substitutes for tests, issues, or release
artifacts; they preserve the reasoning, reproductions, and remediation contracts that
connect those layers.

## Historical audits

- [v0.8.0 pre-release audit and remediation record](v0.8-pre-release.md) —
  historical integrated correctness, native-safety, geometry, result-integrity,
  diagnostic, and release-contract audit for an earlier source snapshot. Its
  R1–R9 remediation and the post-R9 `COPYING` correction were subsequently
  completed and accepted.

The audit page contains the detailed R1–R9 implementation issue contracts. The
[completed remediation execution plan](../plans/archive/v0.8-remediation.md)
preserves their dependency order and the Chat-to-Codex workflow. Issue #33
qualifies the exact final source commit and artifacts after source finalization
and independent review and before the public tag is created.
