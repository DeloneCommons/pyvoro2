# Tooling helpers

This directory contains repository-maintenance helpers used in local
publishability checks and CI.

The root `README.md` is authored independently for GitHub/PyPI, while
`docs/index.md` is the site landing page. Keep their claims consistent without
mechanical synchronization. Notebook sources and generated Markdown pages
remain synchronized through the execution/export tools below.

Main entry points:

- `python tools/execute_notebooks.py` — clear ordinary stored outputs, execute
  each selected source notebook in a fresh Jupyter kernel, and save only after
  successful execution.
- `python tools/export_notebooks.py` — regenerate `docs/notebooks/*.md` from
  stored outputs in the repo-root `notebooks/` directory without executing.
- `python tools/check_notebooks.py` — validate committed notebook structure,
  execution counts, and stored-error absence, then execute a clean in-memory
  copy in a fresh Jupyter kernel without rewriting repository files.
- `python tools/release_check.py` — run the combined local release-preparation
  checks, including an isolated sdist-to-wheel round trip.
- `python tools/build_wheel_from_sdist.py dist` — select exactly one generated
  sdist and rebuild its wheel with pip build isolation.
- `python tools/check_installed_package.py --require-scipy` — verify installed
  module provenance, installed license/metadata payload, the private `_internal`
  layout, lazy and explicit native loading, public transforms, periodic
  behavior, and representative 2D, 3D, canonical inverse, and schema-1 report
  workflows. `--require-scipy` also exercises an explicit sparse fit; use
  `--forbid-scipy` for a base installation.
- `python tools/check_dist.py dist` — verify that built sdists and wheels
  contain byte-identical project `LICENSE` and `COPYING`, `NOTICE.md`, and
  Voro++ license payloads, current platform metadata, exact package paths and
  native-module identities, one archive root, and other key files while
  rejecting duplicate/unsafe members and obsolete private and compatibility
  paths. Explicit artifact paths are also accepted.
- `python tools/check_dist_metadata.py dist` — discover `.whl` and `.tar.gz`
  artifacts in deterministic order and run Twine metadata validation without
  shell-dependent glob expansion. Explicit artifact paths are also accepted;
  equivalent resolved paths are checked only once.
- `python tools/check_wheel_matrix.py release-dist` — require the complete
  CPython 3.10–3.14 release matrix (20 supported native wheels and one matching
  sdist), validate wheel tags, Python/runtime dependency metadata, and project
  identity, and require both native extension modules in every wheel.

Install the focused notebook stack with:

```bash
python -m pip install -e ".[notebooks]"
```

It contains `nbformat`, `nbclient`, and `ipykernel` without changing ordinary
pyvoro2 runtime requirements. The `all` extra includes the same stack.

Execution always starts a fresh kernel per notebook with the repository root as
its working directory. Both execution and checking accept positional notebook
filenames. With none supplied, both discover every `notebooks/*.ipynb` file in
sorted filename order, matching the exporter's source set. Pass `--use-src` in
the wheel-overlay developer workflow; it prepends `repo/src` to `PYTHONPATH` in
the spawned kernel, where the compiled extensions must still be available.

Every nonempty code cell in a committed notebook needs an execution count,
except a cell tagged exactly `skip-execution`. Ordinary refresh/check commands
do not run those cells, and refresh preserves their stored output and count.
Notebook 05 uses the tag only on its five py3Dmol display cells so their
reviewed environment-sensitive HTML/custom MIME output is not regenerated.
When one of those cells changes, refresh it manually in a reviewed rich-output
environment and restore the tag. A failing ordinary cell must never be hidden
with the tag.

The normal maintenance order is:

```bash
python tools/execute_notebooks.py NAME.ipynb
python tools/check_notebooks.py
python tools/export_notebooks.py
python tools/export_notebooks.py --check
```

Refresh and validation omit cell timing and widget state and preserve reviewed
notebook kernel metadata. Markdown export and Zensical never start a kernel.

Documentation uses Zensical with `mkdocs.yml` and writes to `site/`:

```bash
zensical build --clean --strict
```

The documentation builder requires Python 3.11+. The `docs` and `all` extras
omit Zensical on Python 3.10 without changing pyvoro2's supported Python range.
The release checker uses the same strict command; `--skip-docs` skips only that
build, preserving notebook validation and export checks.

For a full local pre-release pass on Python 3.11+ after installing all optional
extras, run:

```bash
pip install -e ".[all]"
python tools/release_check.py
```
