<div class="pv-overview" markdown="1">

# About pyvoro2

<div class="pv-overview-lead" markdown="1">

pyvoro2 is an open-source Python library for computing 2D and 3D Voronoi and
power/Laguerre tessellations, including supported periodic domains, and for
fitting power weights from pairwise separator observations.

</div>

## What pyvoro2 provides

pyvoro2 uses **Voro++** to compute Voronoi and power cells. Beyond the underlying
tessellation engine, it provides Python tools for working with their geometry
and results:

- **Domains.** Convenient Python objects for describing bounded boxes and
  supported periodic cells, including fully periodic triclinic cells in 3D.
- **Cell geometry.** Structured results containing areas or volumes, with cell
  shapes, vertices, boundaries, and other geometric properties available when
  requested.
- **Topology.** Tools for identifying shared vertices and edges across cells,
  as well as shared faces in 3D, rather than working only with separate cell
  geometries.
- **Periodic neighbors.** Information about which sites share a boundary and
  which translated periodic image of a neighboring site forms that boundary.
- **Validation and diagnostics.** Checks for invalid inputs, geometric
  consistency, and potential problems in computed results.
- **Inverse fitting.** Estimation of power weights from observed pairwise
  separator positions at fixed sites, with information about how well the
  observations are matched.

The 2D and 3D interfaces have different supported domains and some different
output capabilities. See [Capabilities and limitations](../guide/capabilities.md)
for details.

Forward tessellation and inverse fitting can be used independently. In inverse
fitting, matching the observed separator positions does not necessarily mean
that the corresponding cell boundaries exist in the resulting power diagram.
See [Inverse fitting](../guide/inverse/index.md) and [Theory](../theory/index.md)
for an explanation.

## Project status

<div class="pv-overview-grid" markdown="1">
<div class="pv-overview-card" markdown="1">

<p class="pv-overview-label">Published · v0.8.0</p>

**v0.8.0 is the latest published release.** Some APIs are still provisional or
experimental; see API lifecycle for details.

[Release notes](release-notes-v0.8.md) ·
[API lifecycle](../development/api-lifecycle.md)

</div>
<div class="pv-overview-card" markdown="1">

<p class="pv-overview-label">In development · v0.9</p>

**Development of v0.9 is based on v0.8.0.** Its final feature set and release
schedule have not yet been established.

[Roadmap](roadmap.md) · [Changelog](changelog.md) ·
[Development basis (ADR 0027)](../development/decisions/0027-v0.9-reboot-from-v0.8.md)

</div>
</div>

## Project information

<div class="pv-overview-links" markdown="1">
<div markdown="1">

**License and attribution**

pyvoro2-authored code is LGPLv3+ from v0.6.0; earlier releases remain MIT.
Vendored Voro++ retains its separate upstream license.

[License and notices](license.md)

</div>
<div markdown="1">

**Contributing**

Development guidance covers tests, notebook maintenance, architecture, and
the review process.

[Contributor setup](https://github.com/DeloneCommons/pyvoro2/blob/dev/CONTRIBUTING.md) ·
[Development](../development/index.md)

</div>
<div markdown="1">

**Policy and provenance**

AI assistance is disclosed; the maintainer reviews integrated changes and
remains responsible for the software and scientific claims.

[AI-assisted development](ai.md) ·
[Decision records](../development/decisions/index.md)

</div>
</div>
</div>
