# 0001 — Project scope and layering

- **Status:** Accepted
- **Date:** 2026-07-17

## Context

pyvoro2 began as a Python interface to Voro++, then gained periodic image
bookkeeping, diagnostics, normalization, a planar namespace, and a substantial
separator-based inverse-fitting implementation. Future plans include fitting weights
from prescribed cell measures and combining multiple observation families.

Without an explicit scope, the package could evolve in two unhelpful directions:
remain described as only a generic wrapper despite its research contribution,
or accumulate unrelated inverse solvers and speculative geometry features.

## Decision

pyvoro2 is a package for **forward and inverse weighted tessellations**.

- The forward 2D/3D Voronoi and power/Laguerre computation layer remains a
  first-class, independently useful core.
- `pyvoro2` remains the explicit 3D namespace.
- `pyvoro2.planar` remains the explicit 2D namespace.
- Separator observations, prescribed cell measures, and later mixed objectives
  belong to one inverse weighted-tessellation layer.
- New inverse methods must reuse common geometry, result, gauge, and diagnostic
  concepts rather than becoming isolated utilities.
- Site-coordinate optimization, arbitrary callback objectives, anisotropic or
  non-Euclidean diagrams, and GPU work are outside the near-term core.
- pyvoro2 does not aim to become a general computational-geometry framework or
  to compete with broad libraries such as CGAL.

## Consequences

- The v0.7 line must stabilize the forward/result contract before adding a second
  inverse family.
- Documentation should present the implemented separator method as one inverse
  observation model, while clearly marking cell-measure and mixed fitting as future
  work.
- Dimension-specific capabilities can remain different, but common concepts
  should use aligned terminology.
- Architecture and JOSS-facing narratives can explain a scholarly contribution
  beyond ordinary forward wrapping without diminishing the forward core.
- Scope expansion requires a concrete research need and a separate design
  decision.

## Alternatives considered

### Keep pyvoro2 as only a Voro++ wrapper

Rejected because the separator inverse layer, graph and gauge diagnostics, and
realization-aware workflows are already substantial research functionality and
need a maintainable public identity.

### Split every inverse method into a separate package

Rejected for separator and cell-measure fitting because they share sites,
domains, power weights, gauge, forward realization, and diagnostics. Downstream
application packages such as chemvoro can remain separate.

### Build a fully generic inverse-geometry framework immediately

Rejected because a generic protocol designed before the second real observation
family would be speculative and likely over-abstracted.
