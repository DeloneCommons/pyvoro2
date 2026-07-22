# Notebooks

The example notebooks are kept in the repository-root `notebooks/` directory so
that users can browse them directly on GitHub without going through the docs
site.

For the published docs, each notebook is also exported to a generated Markdown
page under `docs/notebooks/`.

## Source notebooks in the repository

The repository source notebooks are:

- `notebooks/01_basic_compute.ipynb`
- `notebooks/02_periodic_graph.ipynb`
- `notebooks/03_locate_and_ghost.ipynb`
- `notebooks/04_powerfit.ipynb` — canonical external-ID, weight-first periodic
  inverse workflow plus advanced fitting
- `notebooks/05_visualization.ipynb`
- `notebooks/06_powerfit_reports.ipynb` — ID-labelled records and reports
- `notebooks/07_powerfit_infeasibility.ipynb` — contradiction witnesses
- `notebooks/08_powerfit_active_path.ipynb` — experimental outer-loop path
  diagnostics

## Published notebook pages

The generated documentation pages are:

- [01 basic compute](../notebooks/01_basic_compute.md)
- [02 periodic graph](../notebooks/02_periodic_graph.md)
- [03 locate and ghost cells](../notebooks/03_locate_and_ghost.md)
- [04 powerfit workflow](../notebooks/04_powerfit.md)
- [05 visualization](../notebooks/05_visualization.md)
- [06 powerfit reports](../notebooks/06_powerfit_reports.md)
- [07 powerfit infeasibility](../notebooks/07_powerfit_infeasibility.md)
- [08 active-set path diagnostics](../notebooks/08_powerfit_active_path.md)

## Install the notebook tools

Notebook maintenance uses a focused optional dependency group:

```bash
python -m pip install -e ".[notebooks]"
```

It provides `nbformat`, `nbclient`, and `ipykernel`. These packages are also in
the complete `all` contributor installation, but are not runtime dependencies
of pyvoro2.

## Execution, validation, and export

The three responsibilities are intentionally separate.

Refresh stored outputs after changing a source notebook:

```bash
python tools/execute_notebooks.py 04_powerfit.ipynb
```

The command clears ordinary executable cells, starts a fresh Jupyter kernel for
each selected notebook, executes from the repository root, and saves the source
only after successful completion. With no filenames it discovers and refreshes
every `notebooks/*.ipynb` file in sorted filename order.

Validate all committed notebooks without rewriting them:

```bash
python tools/check_notebooks.py
```

The checker first requires an execution count on every nonempty executable code
cell and rejects stored error outputs. It then clears ordinary cells in an
in-memory copy and executes that copy through a fresh kernel. Execution failures
report the notebook, cell source, and traceback context. With no filenames it
discovers every `notebooks/*.ipynb` file, matching the exporter so newly added
source notebooks cannot bypass validation.

Finally, regenerate and check the published pages:

```bash
python tools/export_notebooks.py
python tools/export_notebooks.py --check
```

The Markdown exporter does not execute notebook code. It renders code and
outputs already stored in each source `.ipynb`, so an unexecuted source notebook
produces a page without its print or final-expression output.

The normal MkDocs build consumes those committed pages and never starts a
kernel.

## Source-overlay imports

In a wheel-overlay developer setup, make the repository source visible inside
the spawned kernel with:

```bash
python tools/execute_notebooks.py --use-src 04_powerfit.ipynb
python tools/check_notebooks.py --use-src
```

This prepends `repo/src` to the kernel's `PYTHONPATH`; changing only the parent
process import path would not affect Jupyter execution. The matching compiled
extensions must already be available in the overlay environment.

## Skipped rich output

The only execution-control tag is `skip-execution`. Use it solely for a cell
that is intentionally not run by ordinary automation. Skipped cells are exempt
from the execution-count requirement, are not executed by refresh or checking,
and keep their reviewed stored output during refresh. The tag must not be used
to conceal an error.

Notebook 05 applies the tag to its five py3Dmol-producing cells. Their useful
stored HTML and custom MIME bundles contain environment-specific viewer IDs and
plain-text object addresses, so routine regeneration would create large,
meaningless diffs. If one of those cells changes, execute and review it manually
in a visualization-capable Jupyter environment, save the new rich output, and
restore `skip-execution` before committing. Ordinary maintenance deliberately
leaves those payloads untouched.
