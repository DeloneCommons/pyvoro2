# Tooling helpers

This directory contains repository-maintenance helpers used in local
publishability checks and CI.

Main entry points:

- `python tools/execute_notebooks.py` — clear ordinary stored outputs, execute
  each selected source notebook in a fresh Jupyter kernel, and save only after
  successful execution.
- `python tools/export_notebooks.py` — regenerate `docs/notebooks/*.md` from
  stored outputs in the repo-root `notebooks/` directory without executing.
- `python tools/check_notebooks.py` — validate committed notebook structure,
  execution counts, and stored-error absence, then execute a clean in-memory
  copy in a fresh Jupyter kernel without rewriting repository files.
- `python tools/gen_readme.py` — regenerate `README.md` from the MkDocs source.
- `python tools/release_check.py` — run the combined local release-preparation
  checks, including an isolated sdist-to-wheel round trip.
- `python tools/build_wheel_from_sdist.py dist` — select exactly one generated
  sdist and rebuild its wheel with pip build isolation.
- `python tools/check_installed_package.py --require-scipy` — verify installed
  module provenance, native loading, and representative 2D, 3D, and canonical
  inverse workflows. Use `--forbid-scipy` for a base installation.
- `python tools/check_dist.py dist` — verify that built sdists and wheels
  contain the expected key files.
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
notebook kernel metadata. Markdown export and MkDocs never start a kernel.

For a full local pre-release pass after installing the project with all optional
extras, run:

```bash
pip install -e ".[all]"
python tools/release_check.py
```
