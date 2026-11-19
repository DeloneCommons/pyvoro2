<div class="pv-overview" markdown="1">

# Examples

<div class="pv-overview-lead" markdown="1">

Explore eight notebooks by the computation you want to try. Each demonstrates
an independent workflow; for a connected introduction, follow the
[User Guide](../guide/index.md).

</div>

## Tessellations and periodic geometry

These examples emphasize **3D**. New to tessellations? Start with **Basic 3D
computations**. For **2D**, use the [planar guide](../guide/planar.md) and its
[Matplotlib plotting recipe](../guide/planar.md#planar-plotting).

<div class="pv-overview-grid pv-overview-grid--catalogue" markdown="1">
<div class="pv-overview-card" markdown="1">

<p class="pv-overview-label">Notebook 01 · Start here</p>

### [Basic 3D computations](../notebooks/01_basic_compute.md)

Compute finite and fully periodic triclinic tessellations. Compare standard
Voronoi and weighted power cells, inspect geometry and empty cells, then
construct shared vertices, edges and faces through normalization.

[Source notebook 01 (.ipynb)](https://github.com/DeloneCommons/pyvoro2/blob/main/notebooks/01_basic_compute.ipynb)

</div>
<div class="pv-overview-card" markdown="1">

<p class="pv-overview-label">Notebook 02 · Periodic contacts</p>

### [Periodic neighbor graphs](../notebooks/02_periodic_graph.md)

Turn cell contacts into a graph while keeping the lattice-image shift of each
neighbor. Inspect reciprocal contacts and build adjacency lists. For an
undirected graph that also retains self-images, use the
[Guide recipe](../guide/periodic-contacts.md#building-a-periodic-graph-in-practice).

[Source notebook 02 (.ipynb)](https://github.com/DeloneCommons/pyvoro2/blob/main/notebooks/02_periodic_graph.ipynb)

</div>
<div class="pv-overview-card" markdown="1">

<p class="pv-overview-label">Notebook 03 · Point queries</p>

### [Locate and probe cells](../notebooks/03_locate_and_ghost.md)

Find which site owns a query point in a finite box. Compute the hypothetical
cell around an inserted probe without changing the original sites, and see
how out-of-domain queries are handled.

[Source notebook 03 (.ipynb)](https://github.com/DeloneCommons/pyvoro2/blob/main/notebooks/03_locate_and_ghost.ipynb)

</div>
<div class="pv-overview-card" markdown="1">

<p class="pv-overview-label">Notebook 05 · Provisional visualization helpers</p>

### [Inspect 3D geometry](../notebooks/05_visualization.md)

Explore cell wireframes with optional py3Dmol tools. Adjust styles, wrap
triclinic cell geometry for viewing and label shared vertices. Interactive
viewing needs a compatible notebook frontend.

[Source notebook 05 (.ipynb)](https://github.com/DeloneCommons/pyvoro2/blob/main/notebooks/05_visualization.ipynb)

</div>
</div>

## Inverse fitting

Begin with the [inverse quickstart](../guide/inverse/index.md) for the stable
high-level fitting workflow. These notebooks also use advanced models and
report/realization helpers, which are **provisional**. Active-set refinement
is **experimental**; see [API choice and lifecycle](../guide/choosing-api.md#feature-and-lifecycle-status).

<div class="pv-overview-grid pv-overview-grid--catalogue" markdown="1">
<div class="pv-overview-card" markdown="1">

<p class="pv-overview-label">Notebook 04 · Advanced workflow</p>

### [Fit separators, then check geometry](../notebooks/04_powerfit.md)

Fit weights from periodic separator observations with external site IDs and
explicit image shifts. Compute the power diagram and check actual contacts.
Continue into provisional fitting models and experimental active-set refinement.

[Source notebook 04 (.ipynb)](https://github.com/DeloneCommons/pyvoro2/blob/main/notebooks/04_powerfit.ipynb)

</div>
<div class="pv-overview-card" markdown="1">

<p class="pv-overview-label">Notebook 06 · Provisional report helpers</p>

### [Fit reports and exports](../notebooks/06_powerfit_reports.md)

Inspect fit residuals and realized contacts in records labelled by site ID.
Export reports as JSON, and distinguish the final state from iteration history
in an experimental active-set run.

[Source notebook 06 (.ipynb)](https://github.com/DeloneCommons/pyvoro2/blob/main/notebooks/06_powerfit_reports.ipynb)

</div>
<div class="pv-overview-card" markdown="1">

<p class="pv-overview-label">Notebook 07 · Provisional constraint models</p>

### [Diagnose incompatible constraints](../notebooks/07_powerfit_infeasibility.md)

Build a set of separator restrictions that cannot all hold at once. Inspect
the infeasible status and the conflicting observations, then export the
contradiction witness in a report.

[Source notebook 07 (.ipynb)](https://github.com/DeloneCommons/pyvoro2/blob/main/notebooks/07_powerfit_infeasibility.ipynb)

</div>
<div class="pv-overview-card" markdown="1">

<p class="pv-overview-label">Notebook 08 · Experimental</p>

### [Follow an active-set path](../notebooks/08_powerfit_active_path.md)

Track a fit that starts with no active observations and reconnects during
iteration. Compare temporary disconnections with the final result. This
experimental refinement has no universal convergence guarantee.

[Source notebook 08 (.ipynb)](https://github.com/DeloneCommons/pyvoro2/blob/main/notebooks/08_powerfit_active_path.ipynb)

</div>
</div>

<span id="how-the-published-examples-are-maintained"></span>

## Reading and running notebooks

The linked titles open rendered pages with **committed outputs**; the separate
source links open `.ipynb` files on GitHub. To run a notebook locally, install
`pyvoro2` (or `pyvoro2[viz]` for visualization) and run its cells in order in a
fresh Jupyter kernel.

**Notebook export does not execute code**, and a documentation build does not
start a kernel. See [Notebook maintenance](../development/notebook-workflow.md)
for the separate execution, validation and export procedures. The historical
`powerfit` filenames use the current inverse namespaces.

This is the existing collection. Its final selection and any replacements
remain deferred until the v0.9 functionality is clearer.

</div>
