# Contributing to pyvoro2

pyvoro2 is research software with a native C++ backend, numerical geometry
code, and a developing inverse weighted-tessellation API. Contributions are
welcome when they are focused, testable, and consistent with the documented
architecture.

This contribution guide is being prepared with the v0.7.0 stabilization line.
Before that line reaches `main`, the project may still prioritize internal API
work over broad feature contributions.

## Before starting

For a small bug fix, documentation correction, or additional test, a pull
request can be sufficient. Open an issue first when the change:

- adds or changes public API;
- changes numerical or periodic-image semantics;
- adds an inverse observation family or solver;
- modifies the vendored/backend layer;
- changes packaging, licensing, or release behavior;
- is large enough that multiple designs are plausible.

Read these documents before architectural work:

- [Architecture](docs/development/architecture.md)
- [API lifecycle](docs/development/api-lifecycle.md)
- [Decision records](docs/development/decisions/)
- [Roadmap](docs/project/roadmap.md)

## Development environment

pyvoro2 requires Python 3.10 or newer and a working C++ build toolchain when it
is built from source. A normal editable development install is:

```bash
python -m pip install --upgrade pip
python -m pip install -e ".[all]"
```

The `all` extra installs the test, lint, documentation, notebook, visualization,
and release-check dependencies. Platform-specific CMake/Ninja installation may
still be needed when no suitable compiler toolchain is already available.

For a wheel-core plus repository-source workflow, see `tools/README.md` and
`tools/install_wheel_overlay.py`.

## Repository conventions

### Source and generated files

Edit:

- `docs/index.md`, then regenerate `README.md`;
- `notebooks/*.ipynb`, then regenerate `docs/notebooks/*.md`.

Do not edit the generated copies directly.

Regeneration commands:

```bash
python tools/gen_readme.py
python tools/export_notebooks.py
```

### Documentation roles

- `docs/guide/` explains how to use the current public API.
- `docs/theory/` explains mathematics independently of API names.
- `docs/development/` explains architecture, compatibility, and decisions.
- `docs/project/roadmap.md` gives milestone-level direction.
- GitHub issues and milestones contain concrete implementation tasks.

### Public API and compatibility

Public API changes must follow
[`docs/development/api-lifecycle.md`](docs/development/api-lifecycle.md).
In particular:

- do not remove or silently repurpose a documented name;
- add migration aliases or adapters where practical;
- document changed defaults and return schemas;
- test both the preferred path and the compatibility path;
- distinguish stable, provisional, experimental, compatibility-only, and
  internal surfaces.

The current `pyvoro2.powerfit` API is an important compatibility surface. A
clearer inverse namespace may be introduced, but existing users should not be
forced through an abrupt rename.

## Coding expectations

- Keep changes small enough to review.
- Prefer explicit numerical behavior over hidden normalization or fallback.
- Validate shapes, finiteness, dimensions, and domain compatibility at public
  boundaries.
- Preserve explicit 2D/3D capability differences.
- Add structured diagnostics for meaningful numerical or geometric failure
  modes.
- Keep chemistry-specific policy in downstream packages rather than pyvoro2.
- Add comments for non-obvious geometry, not for straightforward syntax.
- Follow the existing style and pass `flake8`.

For inverse code, preserve the distinctions documented in
[`docs/theory/separator-inverse.md`](docs/theory/separator-inverse.md):
measurement residuals versus implied weight-difference residuals, global gauge
versus disconnected-component offsets, and algebraic fit versus realized
boundaries.

## Tests

Run the deterministic suite for ordinary changes:

```bash
pytest -q
```

Run lint and generated-file checks:

```bash
flake8 src tests tools
python tools/export_notebooks.py --check
python tools/gen_readme.py --check
```

Build the documentation strictly:

```bash
mkdocs build --strict
```

Notebook changes require execution checks:

```bash
python tools/check_notebooks.py
```

Optional randomized and cross-wrapper checks are available for geometry-heavy
changes:

```bash
pytest -m fuzz --fuzz-n 100

# Requires an independently installed pyvoro package.
pytest -m pyvoro --fuzz-n 100
```

Before a release or broad refactor, run the combined validation:

```bash
python tools/release_check.py
```

## Test design guidance

A good geometry or inverse regression test should state which invariant it
protects. Depending on the change, consider:

- bounded and periodic cases;
- standard and power modes;
- 2D and 3D paths;
- external IDs and periodic shifts;
- empty or hidden cells;
- reciprocal adjacency/shift bookkeeping;
- disconnected observation graphs;
- gauge-invariant weight comparisons;
- infeasible difference constraints;
- requested-shift versus other-shift realization;
- deterministic failure/status reporting.

Avoid asserting incidental vertex order or floating-point details when a more
stable geometric invariant is available.

## Pull requests

A pull request should include:

- the problem and intended behavior;
- links to the issue or decision record when applicable;
- tests that fail without the change;
- documentation for user-visible behavior;
- compatibility notes;
- the validation commands that were run.

A concise checklist:

- [ ] implementation is scoped to the stated problem;
- [ ] deterministic tests pass;
- [ ] lint passes;
- [ ] generated files are synchronized;
- [ ] docs build with `--strict`;
- [ ] public API status and migration are documented;
- [ ] changelog entry is present when appropriate.

## Support and questions

Use GitHub issues for reproducible bugs and focused feature proposals. Include a
minimal example, Python/pyvoro2 versions, platform, domain type, and relevant
numerical inputs when possible.

Support is best-effort. The project is currently led and maintained by one
person, so there is no guaranteed response time. General scientific consulting,
interpretation of private datasets, and debugging workflows that cannot be
reproduced outside a private environment are outside the project's support
commitment.

## Governance and decisions

pyvoro2 currently uses a lead-maintainer model:

- routine implementation and release decisions are made by the maintainer;
- public API and architecture changes should be discussed in a visible issue or
  decision record;
- the maintainer makes the final technical and release decision;
- governance can be separated into a dedicated policy if a stable contributor
  or co-maintainer community develops.

Software contribution and scientific authorship are related but not identical.
Authorship decisions depend on the intellectual and scholarly contribution to
a specific research output, not only on the number of code changes.

## Security-sensitive reports

pyvoro2 is not a network service, but it distributes native code. Suspected
security vulnerabilities involving memory safety, package installation, build
artifacts, or release infrastructure should not initially be reported in a
public issue. Contact the maintainer privately using the email address in
`pyproject.toml`, and include enough detail to reproduce and assess the issue.
