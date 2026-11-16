# Examples

These eight notebooks provide longer workflows using the implemented API.
Follow a rendered page to read its stored output, or open the original `.ipynb`
on GitHub to download and run it in a local Jupyter environment. Install
`pyvoro2`; visualization examples additionally use `pyvoro2[viz]`.
Run code cells in order in a fresh kernel, since later sections may reuse
objects created earlier.

## Tessellations and periodic geometry

These examples are **3D-oriented**. For **2D**, start with the
[planar guide](../guide/planar.md) and its
[Matplotlib plotting recipe](../guide/planar.md#planar-plotting); this catalogue
does not contain a comprehensive planar notebook.

| Notebook | What it demonstrates | Original |
|---|---|---|
| [01 — Basic 3D computations](../notebooks/01_basic_compute.md) | Finite and triclinic periodic cells, standard/power modes, empty cells, face shifts, normalization and face properties. | [`.ipynb`](https://github.com/DeloneCommons/pyvoro2/blob/main/notebooks/01_basic_compute.ipynb) |
| [02 — Periodic neighbor graphs](../notebooks/02_periodic_graph.md) | Directed contacts, image shifts and graph extraction from a periodic sample. For a general reciprocal key retaining self-images, use the [periodic-contact recipe](../guide/periodic-contacts.md#building-a-periodic-graph-in-practice). | [`.ipynb`](https://github.com/DeloneCommons/pyvoro2/blob/main/notebooks/02_periodic_graph.ipynb) |
| [03 — Locate and ghost cells](../notebooks/03_locate_and_ghost.md) | Find owners of query points and inspect hypothetical probe cells. | [`.ipynb`](https://github.com/DeloneCommons/pyvoro2/blob/main/notebooks/03_locate_and_ghost.ipynb) |
| [05 — 3D visualization](../notebooks/05_visualization.md) | Optional py3Dmol inspection, styles, triclinic wrapping and global vertex labels. Interactive viewing requires a compatible notebook frontend. | [`.ipynb`](https://github.com/DeloneCommons/pyvoro2/blob/main/notebooks/05_visualization.ipynb) |

## Inverse fitting

Start with the [high-level inverse quickstart](../guide/inverse/index.md).
The notebooks then use `pyvoro2.inverse` and the advanced
`pyvoro2.inverse.separator` namespace. Advanced models and report/realization
helpers are provisional; active-set refinement is experimental.

| Notebook | What it demonstrates | Original |
|---|---|---|
| [04 — Separator-fitting workflow](../notebooks/04_powerfit.md) | External IDs, explicit periodic images, weight-first fitting and forward realization; then advanced objectives and active sets. | [`.ipynb`](https://github.com/DeloneCommons/pyvoro2/blob/main/notebooks/04_powerfit.ipynb) |
| [06 — Fit reports and exports](../notebooks/06_powerfit_reports.md) | ID-labelled records, realization diagnostics, final-state reports and JSON export. | [`.ipynb`](https://github.com/DeloneCommons/pyvoro2/blob/main/notebooks/06_powerfit_reports.ipynb) |
| [07 — Infeasibility diagnostics](../notebooks/07_powerfit_infeasibility.md) | Contradictory hard constraints, status inspection and contradiction witnesses. | [`.ipynb`](https://github.com/DeloneCommons/pyvoro2/blob/main/notebooks/07_powerfit_infeasibility.ipynb) |
| [08 — Active-set paths (experimental)](../notebooks/08_powerfit_active_path.md) | Transient outer-loop events and path diagnostics, separate from the final fit and realization. | [`.ipynb`](https://github.com/DeloneCommons/pyvoro2/blob/main/notebooks/08_powerfit_active_path.ipynb) |

The filenames retain their historical `powerfit` label; they use the current
inverse namespaces, not the removed `pyvoro2.powerfit` package.

## How the published examples are maintained

The site reads committed Markdown exports of stored notebook outputs. **Export
does not execute notebooks**, and a documentation build does not start a
kernel. Separate maintenance tools refresh source outputs and validate
in-memory copies in clean kernels. Reviewed rich-output cells can carry the
explicit `skip-execution` tag; their stored output is maintained manually.
See [Notebook maintenance](../development/notebook-workflow.md) for the exact
execution, validation and export commands.
