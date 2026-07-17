# Roadmap

This roadmap records durable development phases. It is not a timeline or a list
of every implementation task. GitHub issues and milestones hold actionable work
and current status.

## Project direction

pyvoro2 will remain a forward 2D/3D Voronoi and power/Laguerre package while
developing a coherent inverse weighted-tessellation layer.

The sequencing rule is:

> Stabilize shared forward geometry and results first; express the implemented
> separator method through that contract; validate it in downstream use; only
> then add prescribed-measure and mixed inverse methods.

## Stage 0 — Documentation and architectural contract

Purpose: make the current package, target boundaries, and compatibility policy
explicit before code is reorganized.

Deliverables:

- maintainer/agent instructions;
- current and target architecture documentation;
- theory pages for power diagrams and separator inversion;
- API lifecycle and deprecation policy;
- initial decision records;
- a durable public roadmap;
- contributor guidance appropriate to a single-maintainer research package;
- retirement of the obsolete root `DEV_PLAN.md`.

Stage 0 documents requirements and terminology. It does not claim that target
v0.7 classes or namespaces already exist.

## Stage 1 — v0.7 forward and separator API stabilization

Purpose: provide a chemvoro-ready contract and prepare the package for future
inverse families.

Expected outcomes:

- common 2D/3D forward result concepts for cells, measures, boundaries,
  periodic shifts, empty cells, and diagnostics;
- direct forward computation from mathematical power weights;
- stable site/external-ID association;
- a preferred inverse namespace and separator-observation terminology;
- compatibility for existing `pyvoro2.powerfit` workflows;
- explicit global gauge and disconnected-component-offset semantics;
- inspectable graph/incidence/Laplacian diagnostics without requiring one sparse
  dependency;
- a structured separator result that keeps algebraic, realization, and solver
  diagnostics distinct;
- an optional sparse quadratic solve path where it is beneficial;
- migration documentation and a public API inventory;
- a small downstream integration example shaped like chemvoro usage.

Stages 0 and 1 may be developed on the `dev` branch and merged to `main`
together for v0.7.0. Before release, documentation must be audited so planned
features are rewritten as implemented behavior where appropriate.

## Stage 2 — Prescribed cell measures

Purpose: add the second genuine inverse observation family: fixed sites and
domain, unknown weights, target cell areas in 2D or volumes in 3D.

Implementation order:

1. common cell-measure extraction;
2. target validation and explicit mass-balance policies;
3. residual evaluation without solving;
4. graph-structured sensitivity/Jacobian diagnostics;
5. finite-difference validation on stable generated cases;
6. damped Newton, Gauss–Newton, or trust-region weight updates;
7. empty/near-empty cell diagnostics;
8. generated-data recovery benchmarks;
9. partial and noisy targets;
10. expansion from simpler 2D domains to stable 3D and periodic cases.

The first public solver should remain experimental until generated diagrams can
be recovered modulo global gauge across representative cases and failures return
structured diagnostics.

## Stage 3 — Mixed inverse problems

Purpose: combine multiple observation families without creating unrelated
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

## Stage 4 — Stable 1.0 and JOSS readiness

Purpose: publish a stable research-software contract backed by real use and a
visible development history.

Expected gates:

- public API inventory and lifecycle audit;
- forward and separator contracts validated by downstream use;
- complete install, test, docs, notebook, and release path;
- migration and deprecation status reviewed;
- examples and benchmarks suitable for reviewers;
- research-impact and reproducibility documentation;
- maintained issue/release history showing iterative development;
- stable citation and archival metadata;
- JOSS paper and statement of need.

Version 1.0 does not require every research extension to be stable. Measure and
mixed solvers may remain explicitly experimental if the forward core and
separator API have a credible stable contract.

## Long-horizon research directions

Possible later branches include:

- cell-centroid observations;
- combined separator, measure, and centroid fitting;
- inverse fitting from planar sections or slices;
- regular-triangulation/dual diagnostics;
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
- breaking removal of the current `pyvoro2.powerfit` API.

## Planning responsibilities

- This page records phases and scope.
- Decision records explain durable choices.
- GitHub milestones group release outcomes.
- GitHub issues define small implementation tasks and acceptance criteria.
- The changelog records completed user-visible changes.
