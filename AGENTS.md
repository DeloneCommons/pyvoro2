# AGENTS.md

This file is the operational contract for coding agents and maintainers working
on pyvoro2. Durable rationale and process live in the linked repository
documents:

- [Architecture](docs/development/architecture.md)
- [Development workflow](docs/development/development-workflow.md)
- [Documentation conventions](docs/development/documentation-conventions.md)
- [API lifecycle](docs/development/api-lifecycle.md)
- [Decision records](docs/development/decisions/)
- [Development plans](docs/development/plans/)

## Project direction

pyvoro2 is a scientific package for forward and inverse weighted
tessellations:

- `pyvoro2` is the explicit 3D namespace;
- `pyvoro2.planar` is the explicit 2D namespace;
- the forward core computes standard Voronoi and power/Laguerre diagrams;
- the implemented inverse layer fits power weights from pairwise separator
  observations;
- v0.8 is a feature-free cleanup and compatibility-removal release;
- prescribed cell measures begin in v0.9 and mixed separator-plus-measure
  problems in v0.10, not as unrelated one-off solvers.

v0.7.0 finalized the forward/result contract and separator inverse API. ADR
0004 and ADR 0005 fix the canonical inverse namespace and common result
direction. The completed
[v0.7 development plan](docs/development/plans/archive/v0.7.md) and
[v0.7 API inventory](docs/development/api-inventory.md) record that release.
The v0.8 cleanup plan remains Draft until its milestone, issue set, approval
date, and Active status are recorded; do not begin v0.8 implementation or
present draft target functionality as implemented before activation.

## Authoritative sources

Use the following order when sources disagree:

1. Tests and current source code define implemented behavior.
2. User guides and API reference define documented public behavior.
3. Accepted decision records define durable choices.
4. `docs/development/architecture.md` defines current architecture and accepted
   target responsibilities.
5. The active development plan defines approved release scope, dependencies,
   and release gates.
6. The current release API inventory assigns lifecycle and migration status to
   concrete public surfaces.
7. `docs/project/roadmap.md` defines version-level direction, not current
   behavior.
8. GitHub issues define concrete work and current progress.

A draft plan is not implementation approval. Do not invent surface choices that
are not covered by an accepted ADR, active issue, or inventory entry; wait for
maintainer direction.

Update the relevant documentation when a change makes any of these sources
inconsistent.

## Repository map

- `cpp/`: pybind11 bindings for the vendored 3D and 2D Voro++ backends.
- `vendor/voro++/`: vendored upstream sources. Avoid local backend changes
  unless the issue explicitly requires them and documents the reason.
- `src/pyvoro2/`: 3D forward API, shared utilities, diagnostics, topology,
  visualization, and public package exports.
- `src/pyvoro2/planar/`: explicit 2D API and planar result/diagnostic layer.
- `src/pyvoro2/inverse/separator/`: canonical separator implementation.
- `src/pyvoro2/powerfit/`: v0.7 compatibility-only shim; ADR 0006 removes it
  in v0.8.
- `tests/`: deterministic tests plus opt-in fuzz and cross-check groups.
- `examples/`: repository-owned preferred-API workflows and deterministic
  public regression inputs.
- `notebooks/`: source notebooks.
- `docs/notebooks/`: generated notebook exports; do not edit directly.
- `docs/guide/`: task-oriented documentation for the current API.
- `docs/theory/`: API-independent mathematical concepts.
- `docs/reference/`: exact current API reference.
- `docs/development/`: architecture, API policy, workflow, plans, and decisions.
- `docs/development/plans/`: draft, active, and archived release/workstream plans.
- `docs/project/`: public project identity, roadmap, license, and AI policy.
- `tools/`: repository generation, validation, and release helpers.

## Generated files

Do not edit these files directly:

- `README.md` is generated from `docs/index.md` by
  `python tools/gen_readme.py`.
- `docs/notebooks/*.md` are generated from `notebooks/*.ipynb` by
  `python tools/export_notebooks.py`.

Notebook execution is separate from export. Refresh changed source notebooks
with `python tools/execute_notebooks.py NAME.ipynb`, validate all committed
notebooks with `python tools/check_notebooks.py`, then regenerate the Markdown
pages. When changing a source, include its executed source and generated page in
the same change.

## Development setup and checks

Install the full local stack:

```bash
python -m pip install -e ".[all]"
```

Fast checks for ordinary Python/documentation changes:

```bash
flake8 src tests tools benchmarks examples
pytest -q
python tools/export_notebooks.py --check
python tools/gen_readme.py --check
mkdocs build --strict
```

Notebook changes additionally require:

```bash
python tools/execute_notebooks.py NAME.ipynb
python tools/check_notebooks.py
python tools/export_notebooks.py
python tools/export_notebooks.py --check
```

Cells tagged `skip-execution` are intentionally omitted from automated refresh
and validation execution. Use that tag only for reviewed rich output that must
be refreshed manually, never to hide an execution failure.

Before a release or a broad refactor, run:

```bash
python tools/release_check.py
```

Fuzz and optional `pyvoro` cross-checks are opt-in:

```bash
pytest -m fuzz --fuzz-n 100
pytest -m pyvoro --fuzz-n 100
```

## Architectural invariants

Preserve these unless an accepted decision record supersedes them:

1. **Weights are mathematical; radii are a backend representation.**
   Do not make Voro++ radius shifts the scientific meaning of an inverse
   solution.
2. **One global additive weight shift is geometric gauge.** Independent shifts
   of disconnected observation components are additional unidentified offsets
   and may change the complete realized diagram.
3. **Observation fit and geometric realization are separate.** A small
   separator residual does not prove that the requested pair or periodic image
   is a realized boundary.
4. **Exact inner problems and heuristic outer loops are separate.** The
   fixed-observation separator fit has graph/Laplacian structure; active-set
   refinement is a realization-aware algorithm with inspectable failure modes.
5. **Dimensions remain explicit.** Do not hide materially different 2D and 3D
   capabilities behind misleading overloads.
6. **New inverse methods must share the inverse architecture.** Do not attach a
   prescribed-volume or mixed solver as an unrelated utility.
7. **Failures should be inspectable.** Prefer structured status, diagnostics,
   and witnesses over silent fallback or a bare convergence exception.
8. **Compatibility is deliberate and bounded.** v0.7 provides the documented
   transition paths; ADR 0006 removes compatibility-only inverse and planar
   routes in v0.8 rather than extending them indefinitely.
9. **One forward result, explicit capability differences.** v0.7 uses one
   `TessellationResult` in both dimensions without pretending that every
   optional geometry or normalization capability is shared.
10. **Immutability is pragmatic.** Prefer frozen outer containers and read-only
    owned arrays when clean; do not deep-freeze or copy raw nested records solely
    to claim immutability.

## Planning and change tracking

Follow [Development workflow](docs/development/development-workflow.md).

For substantial work:

1. read the active plan and linked issue;
2. confirm that accepted ADRs and the API inventory cover the public behavior;
3. implement only the issue scope;
4. add tests and current documentation with the code;
5. update or add a decision record for durable choices;
6. add a `[Unreleased]` changelog entry when accepted user-visible behavior
   changes;
7. report the validation commands that passed.

Do not use private chat history as lasting authority. Transfer important scope
or design decisions into the plan, an issue, or a decision record.

Do not remove completed work packages from an active plan. Issues show detailed
progress; plans preserve release structure. During release review, record
outcomes and deferrals, then archive the completed plan.


## Issue-scoped agent handoff

For substantial v0.7 implementation, one coding-agent chain should normally own
one linked issue. Before editing, read the issue, its work package, relevant
ADRs, and the API inventory. The issue defines observable outcomes and
boundaries; choose clean internal implementation details without inventing new
public policy.

Stop and request maintainer input when implementation would require:

- contradicting an accepted ADR;
- adding a new mandatory dependency;
- accepting an unexplained numerical change;
- expanding into a plan non-goal;
- introducing a public name/default not covered by the issue and inventory;
- weakening or extending a documented compatibility/removal schedule.

At handoff, report public behavior changed, compatibility/lifecycle impact,
validation commands run, and any deviation or follow-up. Do not create extra
process artifacts when the issue, tests, docs, and changelog already record the
work adequately.

## Public API changes

Before adding or changing a public name, signature, default, return schema, or
field meaning:

1. identify its lifecycle status using `docs/development/api-lifecycle.md`
   and the active-release API inventory;
2. confirm that the change belongs to the active plan or an approved maintenance
   issue;
3. open or reference an issue for architectural or breaking-change risk;
4. add or update tests for both preferred and compatibility paths;
5. update the user guide and API reference/docstrings;
6. add a decision record when the choice fixes a durable architectural
   boundary;
7. update the changelog.

Do not freeze speculative names merely because they appear in a roadmap or
brainstorming document. Names fixed by accepted ADRs and the active issue are
part of the implementation contract; secondary names remain subject to the API
inventory.

## Forward computation changes

For changes in `api.py`, planar API code, domains, output processing, periodic
shifts, or normalization:

- test both standard and power modes when relevant;
- test bounded and periodic domains when relevant;
- preserve external IDs and periodic image labels;
- check empty-cell behavior for power diagrams;
- keep 2D and 3D semantics aligned where possible without claiming unsupported
  parity;
- update diagnostics when a new failure mode becomes observable.

## Inverse changes

For changes under the canonical `inverse/` namespace or the v0.7-only
`powerfit/` compatibility package:

- keep measurement-space and algebraic edge-space quantities distinct;
- preserve confidence-zero/effective-graph semantics;
- report identifiability and component-offset policy explicitly;
- validate gauge-invariant quantities in tests;
- test infeasible hard constraints and contradiction witnesses;
- test realization separately from algebraic fit;
- include periodic wrong-image cases when shifts are involved;
- avoid chemistry-specific assumptions in pyvoro2 core code.

## Documentation rules

The authoritative writing policy is
[Documentation conventions](docs/development/documentation-conventions.md).
In brief:

- user guides describe callable behavior in the current tree;
- theory pages use accessible paper-like language and avoid provisional class
  names;
- architecture pages distinguish current implementation from target
  responsibility;
- release plans define scope and gates; issues track progress;
- the roadmap uses version-level outcomes, not private “Stage 0/1” labels;
- v0.8 is cleanup-only, v0.9 is prescribed measures, and v0.10 is mixed
  separator-plus-measure fitting;
- the changelog records completed user-visible changes;
- use `separator observation` in explanatory prose and historical API names
  only where code requires them;
- do not copy large parts of the manuscript into the docs.

## Scope control

Do not add the following to near-term work without a separate design decision
and concrete research need:

- moving-site optimization;
- arbitrary user callback objectives;
- anisotropic, spherical, or non-Euclidean diagrams;
- GPU acceleration;
- a general computational-geometry abstraction over Voro++;
- a planar oblique-periodic domain promised only for API symmetry.

## Change completion checklist

A change is complete when:

- its issue acceptance criteria are met;
- implementation and tests agree;
- generated files are synchronized;
- current/target wording in the docs is accurate;
- compatibility behavior is explicit;
- the changelog is updated when user-visible behavior changed;
- the narrowest relevant validation commands pass;
- no unresolved design decision is hidden in the implementation.
