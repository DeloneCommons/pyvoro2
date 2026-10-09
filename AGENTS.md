# Agent entry point

## Establish scope

- Check the current branch, base commit, linked issue, and approved task scope.
  The protected reboot integration branch is `dev`, restarted from published
  v0.8.0. `main` remains the published baseline; the former v0.9 attempt is
  frozen under `legacy/v0.9-attempt-2026-10-08` and the protected archive tag.
- Read [ADR 0027](docs/development/decisions/0027-v0.9-reboot-from-v0.8.md),
  the applicable decision records, and any approved plan before editing.
  Source and tests describe implemented behavior; approved decisions describe
  authorized changes. A proposal or historical implementation is not approval.
- Preserve v0.8 behavior unless the task explicitly changes it. Historical
  ADRs 0018–0026 and the former v0.9 release sequence do not govern the reboot.
  Future geometry architecture and release scope remain undecided.
- Escalate unresolved architectural/public-API choices, new mandatory
  dependencies, or scope contradictions; do not hide them in implementation.

## Implement and validate

- Prefer focused changes and regression tests. For numerical correctness,
  prefer independent analytic or mathematical oracles over implementation
  self-consistency alone. Do not silently accept unexplained numerical changes.
- Preserve external site IDs, periodic-image meaning, units, and public
  semantics unless an approved contract explicitly changes them.
- Use the standard editable setup: `python -m pip install -e ".[dev]"`.
  Start with a focused test, then broaden validation as appropriate; native
  geometry tests still require compiled extensions. See the
  [development workflow](docs/development/development-workflow.md).
- Do not repeatedly launch broader CI after a deterministic local failure is
  established. Resolve the cause or report the blocker and stop that test family.
- Edit `docs/index.md` and regenerate `README.md` with
  `python tools/gen_readme.py`. Edit source notebooks, then follow the execution
  and export process in [CONTRIBUTING.md](CONTRIBUTING.md).
- Report actual commands, outcomes, skips, and environment limitations. Local
  checks do not establish remote CI success or maintainer acceptance.

## Keep authority understandable

Use existing issues, tests, documentation, and changelog entries; avoid extra
plans, handoffs, and status artifacts when they add no necessary information.
Keep implemented behavior, approved work, historical evidence, and research
candidates distinct. Do not mark a decision, checkpoint, or release accepted
without explicit maintainer authority.

Detailed guidance:

- [Architecture and repository responsibilities](docs/development/architecture.md)
- [API lifecycle](docs/development/api-lifecycle.md) and
  [v0.8 API inventory](docs/development/api-inventory.md)
- [Development workflow](docs/development/development-workflow.md)
- [Documentation conventions](docs/development/documentation-conventions.md)
- [Plans](docs/development/plans/index.md) and
  [roadmap](docs/project/roadmap.md)
