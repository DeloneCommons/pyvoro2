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

> Stabilize shared forward geometry and results first; express the implemented
> separator method through that contract; validate it in downstream use; only
> then add prescribed-measure and mixed inverse methods.

## v0.7 — Forward and separator API stabilization

The v0.7 line establishes a chemvoro-ready forward and separator-inverse
contract.

Target outcomes:

- one dimension-neutral `TessellationResult` returned by default in 2D and 3D,
  with explicit capability differences and raw compatibility output;
- direct forward power computation from mathematical weights;
- stable association among sites, external IDs, and output cells;
- canonical inverse ownership under `pyvoro2.inverse.separator` and
  math-aligned separator terminology;
- a bounded v0.7 compatibility period for existing `pyvoro2.powerfit` and broad
  top-level separator imports;
- explicit global gauge and disconnected-component-offset semantics;
- inspectable graph, incidence, and Laplacian problem data without requiring one
  sparse-matrix dependency as the public representation;
- separator results that distinguish state, observations, algebraic structure,
  realization, and solver diagnostics;
- migration documentation and an explicit API lifecycle inventory;
- paper-style and chemvoro-shaped integration validation.

A sparse quadratic execution path is desirable only when benchmarks justify its
performance and dependency cost. It is not part of the core compatibility
promise by default.

The detailed scope, dependencies, decisions, and release gates are in the
[v0.7 development plan](../development/plans/v0.7.md).

## v0.8 — Prescribed cell measures

The next inverse family targets fixed sites and domain, unknown power weights,
and prescribed cell areas in 2D or volumes in 3D. v0.8 is also the planned
removal release for the v0.7 `pyvoro2.powerfit` and broad top-level separator
compatibility paths, unless an explicit release decision extends the transition.

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

## v0.9 — Mixed inverse problems

The mixed-problem line combines observation families without creating unrelated
solver APIs.

The first mixed problem should support:

- fixed sites;
- unknown weights only;
- separator observations;
- cell-measure observations;
- explicit block and row scaling;
- per-block objective and diagnostic reporting;
- a separate final realization report.

Only after separator and measure implementations both exist should a generic
public observation-block protocol be finalized.

Moving sites remains a separate later unknown family, not an option hidden in
the first mixed solver.

## 1.0 — Stable research-software contract and JOSS readiness

Version 1.0 should publish a stable core backed by real downstream use and a
visible development history.

Expected gates:

- public API inventory and lifecycle audit;
- forward and separator contracts validated by downstream use;
- complete install, test, documentation, notebook, and release path;
- migration and deprecation status reviewed;
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
- indefinite preservation of historical inverse namespaces after the documented
  v0.7 transition.

## Planning responsibilities

- This page records version-level direction and scope.
- [Development plans](../development/plans/index.md) define active release
  outcomes, dependencies, validation, and gates.
- Decision records explain durable choices.
- GitHub milestones group release outcomes.
- GitHub issues define implementation tasks and acceptance criteria.
- The changelog records completed user-visible changes.
- Completed plans are archived with their outcome and deferrals.
