# AGENTS.md

This file is the operational contract for coding agents and maintainers working
on pyvoro2. It is intentionally concise. The durable design rationale lives in
[`docs/development/architecture.md`](docs/development/architecture.md), and
major decisions live in
[`docs/development/decisions/`](docs/development/decisions/).

## Project direction

pyvoro2 is a scientific package for forward and inverse weighted
tessellations:

- `pyvoro2` is the explicit 3D namespace;
- `pyvoro2.planar` is the explicit 2D namespace;
- the forward core computes standard Voronoi and power/Laguerre diagrams;
- the implemented inverse layer fits power weights from pairwise separator
  observations;
- prescribed cell measures and mixed inverse problems are later extensions,
  not separate one-off solvers.

The v0.7 development line is stabilizing the forward/result contract and the
separator inverse API. Do not present roadmap functionality as implemented.

## Authoritative sources

Use the following order when sources disagree:

1. Tests and current source code define implemented behavior.
2. User guides and API reference define the documented public behavior.
3. `docs/development/architecture.md` defines architectural responsibilities
   and the target direction.
4. Accepted decision records define durable design choices.
5. `docs/project/roadmap.md` defines sequencing, not current behavior.
6. GitHub issues define concrete work in progress.

Update the relevant documentation when a change makes any of these sources
inconsistent.

## Repository map

- `cpp/`: pybind11 bindings for the vendored 3D and 2D Voro++ backends.
- `vendor/voro++/`: vendored upstream sources. Avoid local backend changes
  unless the issue explicitly requires them and documents the reason.
- `src/pyvoro2/`: 3D forward API, shared utilities, diagnostics, topology,
  visualization, and public package exports.
- `src/pyvoro2/planar/`: explicit 2D API and planar result/diagnostic layer.
- `src/pyvoro2/powerfit/`: current separator-based inverse implementation.
- `tests/`: deterministic tests plus opt-in fuzz and cross-check groups.
- `notebooks/`: source notebooks.
- `docs/notebooks/`: generated notebook exports; do not edit directly.
- `docs/guide/`: task-oriented documentation for the current API.
- `docs/theory/`: API-independent mathematical concepts.
- `docs/development/`: architecture, API policy, and decision records.
- `docs/project/`: public project identity, roadmap, license, and AI policy.
- `tools/`: repository generation, validation, and release helpers.

## Generated files

Do not edit these files directly:

- `README.md` is generated from `docs/index.md` by
  `python tools/gen_readme.py`.
- `docs/notebooks/*.md` are generated from `notebooks/*.ipynb` by
  `python tools/export_notebooks.py`.

When changing a source, regenerate the corresponding output and include both in
the same change.

## Development setup and checks

Install the full local stack:

```bash
python -m pip install -e ".[all]"
```

Fast checks for ordinary Python/documentation changes:

```bash
flake8 src tests tools
pytest -q
python tools/export_notebooks.py --check
python tools/gen_readme.py --check
mkdocs build --strict
```

Notebook changes additionally require:

```bash
python tools/check_notebooks.py
```

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
8. **Compatibility is deliberate.** Public renames require aliases/adapters,
   migration documentation, tests, and release-note coverage.

## Public API changes

Before adding or changing a public name, signature, default, return schema, or
field meaning:

1. identify its lifecycle status using `docs/development/api-lifecycle.md`;
2. open or reference an issue for architectural or breaking-change risk;
3. add or update tests for both preferred and compatibility paths;
4. update the user guide and API reference/docstrings;
5. add a decision record when the choice fixes a durable architectural
   boundary;
6. update the changelog.

Do not freeze speculative class names merely because they appear in the
roadmap. Architectural requirements and implementation names are different.

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

For changes under `powerfit/` or the future inverse namespace:

- keep measurement-space and algebraic edge-space quantities distinct;
- preserve confidence-zero/effective-graph semantics;
- report identifiability and component-offset policy explicitly;
- validate gauge-invariant quantities in tests;
- test infeasible hard constraints and contradiction witnesses;
- test realization separately from algebraic fit;
- include periodic wrong-image cases when shifts are involved;
- avoid chemistry-specific assumptions in pyvoro2 core code.

## Documentation rules

- User guides describe callable behavior in the current tree.
- Theory pages avoid dependence on provisional Python class names.
- Architecture pages distinguish **current implementation** from **target
  architecture**.
- The roadmap describes durable phases; GitHub issues hold actionable task
  lists.
- Do not copy large parts of the manuscript into the docs. Link to it for proofs
  and benchmark details.
- Use `separator observation` in explanatory prose. Keep historical API names
  such as `PairBisectorConstraints` where code requires them.

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

- implementation and tests agree;
- generated files are synchronized;
- current/target wording in the docs is accurate;
- compatibility behavior is explicit;
- the changelog is updated when user-visible behavior changed;
- the narrowest relevant validation commands pass.
