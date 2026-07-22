# Notebooks

These notebooks are the source examples for the documentation site and are kept
at the repository root so they can be browsed directly on GitHub.

Generated Markdown copies for the docs live under `docs/notebooks/` and are
produced by `python tools/export_notebooks.py`. Export is intentionally
non-executing and renders outputs already stored in each source notebook.

For v0.7,
[issue #20](https://github.com/DeloneCommons/pyvoro2/issues/20) establishes that
source notebooks are committed in an executed state, except for explicitly
tagged skipped cells. It will add clean Jupyter-kernel execution and metadata
validation after the notebooks have migrated to the preferred v0.7 API. Do not
regenerate the currently missing outputs against the pre-v0.7 API.

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
