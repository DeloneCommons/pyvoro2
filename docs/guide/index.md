<div class="pv-overview" markdown="1">

# User Guide

<div class="pv-overview-lead" markdown="1">

Start with the inputs you have: sites and a domain to compute a tessellation,
or separator observations to fit power weights. Follow either path through
the current 2D and 3D workflows.

</div>

<div class="pv-overview-row" markdown="1">
<div markdown="1">

## Tessellations and periodic geometry

Define a supported domain and compute standard Voronoi or weighted power
cells. Examine their geometry and measures, then analyze shared topology and
periodic contacts, keeping each neighbor's lattice-image shift. Finish by
validating or visualizing the result.

The quickstart begins with four sites in a fully periodic triclinic 3D cell
and connects to domain choices, planar 2D, geometric output and diagnostics.

**[Start the tessellation path](forward/index.md)**

</div>
<figure>
  <img src="../assets/landing/triclinic-voronoi.png#only-light" width="1600" height="900" alt="Four complete Voronoi polyhedra surround four sites and extend across a dashed skew lattice frame.">
  <img src="../assets/landing/triclinic-voronoi-dark.png#only-dark" width="1600" height="900" alt="Four complete Voronoi polyhedra surround four sites and extend across a dashed skew lattice frame.">
  <figcaption>Four complete Voronoi cells in a fully periodic triclinic lattice. Their volumes sum to 3 cubic units. The dashed frame marks one fundamental domain, not a clipping boundary.</figcaption>
</figure>
</div>

<div class="pv-overview-row pv-overview-row--reverse" markdown="1">
<div markdown="1">

## Inverse fitting

Begin with fixed sites and observed pairwise separator positions. Fit power
weights, inspect residuals, then compute a power diagram and check which
requested contacts and periodic images actually occur. A fitted separator
need not become a cell boundary.

Start with the stable high-level fitting API. The detailed guide introduces
provisional models, report and realization helpers; active-set refinement
remains experimental.

**[Start the inverse fitting path](inverse/index.md)**

</div>
<figure>
  <img src="../assets/landing/separator-cycle.png#only-light" width="820" height="444" alt="Three fixed sites with observed separator ticks on the left; fitted power cells on the right. The original ticks do not coincide exactly with the fitted edges.">
  <img src="../assets/landing/separator-cycle-dark.png#only-dark" width="820" height="444" alt="Three fixed sites with observed separator ticks on the left; fitted power cells on the right. The original ticks do not coincide exactly with the fitted edges.">
  <figcaption>Three observations are fitted approximately: they are not exactly compatible. Ticks mark observations; edges show computed boundaries. All three pairs meet in this particular result.</figcaption>
</figure>
</div>

## Find a task

<div class="pv-overview-links pv-overview-links--tasks" markdown="1">
<div markdown="1">

### Domains and computation

[Finite 3D boxes](domains.md#box-finite-non-periodic)

[Full or partial 3D periodicity](domains.md)

[Finite and rectangular-periodic 2D](planar.md)

[Compute standard or power cells](operations.md#1-compute-tessellate-all-sites)

</div>
<div markdown="1">

### Geometry and contacts

[Cell results and normalization](forward/index.md#geometric-output-and-normalization)

[Periodic neighbors and image shifts](periodic-contacts.md)

[Planar normalization](planar.md#planar-normalization)

</div>
<div markdown="1">

### Validation and visualization

[Tessellation diagnostics](periodic-contacts.md#diagnostics-catching-subtle-issues-early)

[Capabilities and numerical limits](capabilities.md)

[Inspect 3D geometry](visualization.md)

[Plot planar cells](planar.md#planar-plotting)

</div>
<div markdown="1">

### Inverse fitting

[Start from separator observations](inverse/index.md)

[Detailed fitting and diagnostics](inverse/separator-fitting.md)

[From fitted weights to actual contacts](inverse/index.md#from-a-fit-to-geometry)

</div>
</div>

**Choose domains by dimension.** Triclinic 3D cells are fully periodic; partial
periodicity in 3D requires a supported orthorhombic domain. Planar periodic
domains are rectangular. The 2D and 3D interfaces do not expose every identical
domain or output feature.

Use [Choosing an API](choosing-api.md) for namespaces, result layers and
lifecycle status. [Examples](../examples/index.md) offers independent notebook
workflows, [Theory](../theory/index.md) explains the mathematics, and
[API Reference](../reference/index.md) records exact signatures.

</div>
