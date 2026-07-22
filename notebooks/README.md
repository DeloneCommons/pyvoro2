# Notebooks

These notebooks are the source examples for the documentation site and are kept
at the repository root so they can be browsed directly on GitHub.

Generated Markdown copies for the docs live under `docs/notebooks/` and are
produced by `python tools/export_notebooks.py`. Export is intentionally
non-executing and renders outputs already stored in each source notebook.

Source notebooks are committed in an executed state. Refresh selected sources
with `python tools/execute_notebooks.py NAME.ipynb`, validate committed metadata
and clean execution with `python tools/check_notebooks.py`, then export them.
With no filename, execution and validation discover every `.ipynb` file in this
directory in sorted order. The commands use a fresh kernel per notebook and run
from the repository root.

The `skip-execution` cell tag is reserved for deliberately preserved rich
output. The py3Dmol display cells in notebook 05 use it so routine maintenance
does not replace their reviewed HTML/custom MIME bundles with environment-
specific payloads. Refresh those cells manually in a reviewed visualization
environment when their source changes, then restore the tag before committing.

Included notebooks:

- `01_basic_compute.ipynb` — basic 3D tessellation usage.
- `02_periodic_graph.ipynb` — periodic topology and neighbor-graph workflows.
- `03_locate_and_ghost.ipynb` — point-location and ghost-cell queries.
- `04_powerfit.ipynb` — canonical external-ID, weight-first periodic workflow
  plus advanced separator fitting.
- `05_visualization.ipynb` — optional 2D/3D visualization helpers.
- `06_powerfit_reports.ipynb` — preferred report/export and weight-first
  realization surfaces.
- `07_powerfit_infeasibility.ipynb` — infeasibility witnesses and reporting.
- `08_powerfit_active_path.ipynb` — active-set path diagnostics.
