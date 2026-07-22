# Roadmap

This roadmap records version-level outcomes and long-term direction. It is not a
timeline, release plan, or list of every implementation task.

- Active release scope and gates live in
  [development plans](../development/plans/index.md).
- GitHub issues and milestones track concrete work and current progress.
- Decision records explain durable architectural choices.
- The changelog records completed user-visible behavior.

## Project direction

pyvoro2 will remain a forward 2D/3D Voronoi and power/Laguerre package while
developing a coherent inverse weighted-tessellation layer.

The sequencing rule is:

> Stabilize shared forward geometry and separator fitting, remove the bounded
> transition architecture, then add prescribed-measure and mixed inverse
> methods as separate releases.

## v0.7 — Forward and separator API stabilization (completed)

The v0.7 line establishes a chemvoro-ready forward and separator-inverse
contract.

Delivered outcomes:

- one dimension-neutral `TessellationResult` returned by default in 2D and 3D,
  with explicit capability differences and raw `output='cells'` access;
- direct forward power computation from mathematical weights;
- stable association among sites, external IDs, and output cells;
- canonical inverse ownership under `pyvoro2.inverse.separator` and
  math-aligned separator terminology;
- one bounded compatibility release for `pyvoro2.powerfit`, broad top-level
  separator imports, historical separator aliases, and planar result selectors;
- explicit global gauge and disconnected-component-offset semantics;
- inspectable graph, incidence, and Laplacian problem data without making SciPy
  a runtime dependency;
- separator results that distinguish state, observations, algebraic structure,
  realization, and solver diagnostics;
- an optional explicit SciPy sparse-direct path for large static quadratic
  locality graphs, with dense execution retained;
- migration documentation, a finalized lifecycle inventory, and
  paper-/chemvoro-shaped integration validation.

The detailed scope, dependencies, decisions, and release gates are in the
[archived v0.7 development plan](../development/plans/archive/v0.7.md).

## v0.8 — Cleanup and compatibility removal

v0.8 is intentionally a **feature-free maintenance release**. It cleans the
architecture established in v0.7 before a second inverse observation family is
added.

Target outcomes:

- remove `pyvoro2.powerfit`, broad top-level separator exports, historical
  separator aliases, `PlanarComputeResult`, planar `return_result=`, and other
  compatibility-only routes that are not useful current API;
- reorganize the flat test suite into responsibility-based subdirectories;
- move root private Python helpers into `pyvoro2._internal`, without renaming
  compiled `_core` and `_core2d` extensions or adding a public `core` namespace;
- resolve non-critical consistency and maintenance findings deferred from the
  v0.7 final audit;
- preserve stable v0.7 numerical behavior and canonical public workflows;
- simplify reference navigation, distribution checks, imports, and regression
  ownership after compatibility removal.

The removal decision is based on clean architecture and usability, not on
preserving hypothetical historical callers. `output='cells'` remains an
explicit useful raw-data mode unless a separate future decision changes it.

See the draft [v0.8 cleanup plan](../development/plans/v0.8.md) and
[ADR 0006](../development/decisions/0006-v0.8-cleanup-release.md).

## v0.9 — Prescribed cell measures

The second inverse family targets fixed sites and domain, unknown power weights,
and prescribed cell areas in 2D or volumes in 3D.

Expected development order:

1. common cell-measure extraction;
2. target validation and explicit mass-balance policies;
3. residual evaluation without solving;
4. graph-structured sensitivity/Jacobian diagnostics;
5. finite-difference validation on stable generated cases;
6. damped Newton, Gauss–Newton, or trust-region weight updates;
7. empty and near-empty cell diagnostics;
8. generated-data recovery benchmarks;
9. partial and noisy targets;
10. expansion from simpler 2D domains to stable 3D and periodic cases.

The first public prescribed-measure solver should remain experimental until it
recovers generated diagrams modulo global gauge across representative cases and
returns structured diagnostics for failure.

## v0.10 — Mixed inverse problems

The mixed-problem line combines real observation families without creating
unrelated solver APIs.

The first mixed problem should support:

- fixed sites;
- unknown weights only;
- separator observations;
- cell-measure observations;
- explicit block and row scaling;
- per-block objective and diagnostic reporting;
- a separate final realization report.

Only after separator and measure implementations both exist should a generic
public observation-block protocol be finalized. Moving sites remains a separate
later unknown family, not an option hidden in the first mixed solver.

## 1.0 — Stable research-software contract and JOSS readiness

Version 1.0 should publish a stable core backed by real downstream use and a
visible development history.

Expected gates:

- public API inventory and lifecycle audit after v0.8 cleanup;
- forward and separator contracts validated by downstream use;
- prescribed-measure and mixed experimental boundaries documented honestly;
- complete install, test, documentation, notebook, and release path;
- examples and benchmarks suitable for reviewers;
- research-impact and reproducibility documentation;
- maintained issue, decision, plan, and release history showing iterative
  development;
- stable citation and archival metadata;
- JOSS paper and statement of need.

Version 1.0 does not require every research extension to be stable. Measure and
mixed solvers may remain explicitly experimental if the forward core and
separator API have a credible stable contract.

## Future research directions

Possible later workstreams include:

- cell-centroid observations;
- combined separator, measure, and centroid fitting;
- inverse fitting from planar sections or slices;
- regular-triangulation and dual diagnostics;
- optional solver backend plugins;
- bounded site-coordinate optimization;
- trajectory-scale and parallel tessellation workflows after profiling and
  ownership analysis;
- anisotropic or non-Euclidean models when a concrete research project requires
  them.

These directions are not commitments for 1.0.

## Explicit near-term non-goals

- spherical-surface tessellations;
- a general-purpose replacement for CGAL or other broad geometry frameworks;
- arbitrary user-defined nonlinear callbacks before built-in inverse families
  establish a stable protocol;
- GPU acceleration;
- site motion in the first mixed solver;
- planar oblique-periodic support solely for API symmetry;
- trajectory, repeated-frame, or parallel APIs before the post-1.0 design work
  is activated;
- preservation of historical inverse namespaces after v0.7.

## Planning responsibilities

- This page records version-level direction and scope.
- [Development plans](../development/plans/index.md) define active release
  outcomes, dependencies, validation, and gates.
- Decision records explain durable choices.
- GitHub milestones group release outcomes.
- GitHub issues define implementation tasks and acceptance criteria.
- The changelog records completed user-visible changes.
- Completed plans are archived with their outcome and deferrals.
