# User Guide

Choose a task below to start with the geometry you have and the output you need.
pyvoro2 computes standard Voronoi and weighted power/Laguerre tessellations for
fixed sites in two or three dimensions. The forward workflow is useful on its
own: compute cells, inspect measures and boundaries, or extract periodic
contacts. Inverse fitting instead estimates power weights from observed
pairwise separator positions.

Start with **Tessellations & periodic geometry** for a runnable triclinic 3D
example and an explanation of geometric output. Choose **Inverse fitting**
when the separator observations are your inputs. Both paths use the current
implemented API; a development version does not imply additional geometry or
inverse models.

| Your task | Start here |
|---|---|
| Compute finite 3D cells in a `Box` | [3D domains](domains.md#box-finite-non-periodic) and [compute](operations.md#1-compute-tessellate-all-sites) |
| Compute fully periodic triclinic 3D cells | [Triclinic quickstart](forward/index.md) |
| Use full or partial orthorhombic periodicity | [3D domains and periodicity](domains.md) |
| Compute finite or rectangular-periodic 2D cells | [Planar 2D](planar.md) |
| Choose standard or power cells, with mathematical `weights=` | [Compute options](operations.md#1-compute-tessellate-all-sites) |
| Keep periodic contacts and their image shifts | [Periodic contacts](periodic-contacts.md) |
| Construct global vertices, edges and (3D) faces | [Geometric output and normalization](forward/index.md#geometric-output-and-normalization) |
| Plot 2D cells or inspect 3D geometry | [Visualization](visualization.md) and [planar plotting](planar.md#planar-plotting) |
| Fit weights from fixed separator observations | [Inverse fitting quickstart](inverse/index.md) |
| Check unsupported domains or numerical limits | [Capabilities and limitations](capabilities.md) |

Periodic support differs by dimension. Triclinic `PeriodicCell` is fully
periodic; partial periodicity requires an orthorhombic cell. Planar periodic
cells are rectangular. See the linked domain guides before preparing inputs.

Use the [Examples catalogue](../examples/index.md) for longer executable
workflows, [Theory](../theory/index.md) for the mathematics, and
[API Reference](../reference/index.md) for exact signatures. When selecting
namespaces or interpreting result layers and lifecycle status, continue to
[Choosing an API](choosing-api.md).
