# Introducing Voro++ and pyvoro2

**Status:** draft direction for future introductory documentation.

Start visually and in simple terms: Voro++ constructs one convex Voronoi/Laguerre cell at a time by clipping an initial region against separating half-spaces. It does **not** start from a shared global vertex/edge/face complex.

Explain consequences immediately:

- Local cells are useful as-is for volumes, faces, neighbors, and straightforward pyvoro-style computation.
- Independently computed boundaries may disagree numerically; local validity is not a certified global topology.
- Periodic neighbor site identity alone does not identify the specific translated image.
- A global complex can be reconstructed as an additional derived layer, with a separately stated correctness contract.

Then show what pyvoro2 currently adds: user-facing domains, weight-first forward computation, 2D/3D paths, periodic image metadata, diagnostics/normalization, and separator-driven inverse fitting. Distinguish implemented capabilities from proposals.

**Suggested quick start:** lead with a simple `compute(..., output="cells")` example. Introduce `TessellationResult` and advanced geometry only afterwards. An eventual “Migrating from pyvoro” guide should compare actual signatures and record shapes rather than claiming drop-in compatibility.

Related notes: [Voro++ overview](../overview/voro.md) · [geometry layers](../potential-work/geometry-layers.md).
