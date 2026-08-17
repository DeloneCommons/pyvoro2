# Archived development plans

Completed and superseded development plans are kept here after they receive an
outcome summary.

The archive preserves:

- the intended release outcome and original scope;
- accepted decisions and important trade-offs;
- the validation and release gates used;
- what was delivered;
- what was deferred and where it moved;
- links to the release tag, changelog, milestone, and follow-up work.

| Plan | Status | Target | Finalized | Release model |
|---|---|---|---|---|
| [v0.8 technical maintenance and Python 3.14](v0.8.md) | Completed | v0.8.0 | 2026-08-17 | Git tag, GitHub Release, and PyPI; intentionally no new Zenodo software-version record. |
| [v0.8 pre-release audit remediation](v0.8-remediation.md) | Completed | v0.8.0 | 2026-08-17 | R1–R9 plus accepted post-R9 `COPYING` correction; exact frozen commit to be qualified under issue #33 before tagging. |
| [v0.7 forward and separator API stabilization](v0.7.md) | Completed | v0.7.0 | 2026-07-23 | Git tag and PyPI transition release; no GitHub Release or Zenodo record. |

The archived v0.7 plan is included in the final v0.7.0 source. Its separate
[release checklist](../../release-checklist-v0.7.md) remains the operational
record for external CI, tagging, PyPI publication, public verification, and
closure of issue #18 and the milestone.
