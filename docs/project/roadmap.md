# Roadmap

This roadmap records version-level outcomes and long-term direction. It is not a
timeline, release plan, or list of every implementation task.

- Active release scope and gates live in
  [development plans](../development/plans/index.md).
- GitHub issues and milestones track concrete work and current progress.
- Decision records explain durable architectural choices.
- The changelog records completed user-visible behavior.

## Project direction

pyvoro2 will remain a forward 2D/3D Voronoi and power/Laguerre package with a
first-class separator-based inverse layer.

The sequencing rule is:

> Complete technical cleanup in v0.8, use development through v0.9.0 for
> functional/API stabilization, use released v0.9.x for downstream-readiness
> soak, stabilize the existing forward and separator inverse workflows in 1.0,
> then add new inverse observation families after 1.0.

The roadmap preserves potentially valuable workstreams so they are not lost.
Exact issue grouping, implementation order within a release, and acceptance
criteria belong to later development plans and GitHub issues. The post-v0.8
sequence is recorded in
[ADR 0017](../development/decisions/0017-v0.9-functional-stabilization-before-1.0.md).

## v0.7 — Forward and separator API stabilization (completed)

The v0.7 line established the forward and separator-inverse architecture and a
chemvoro-shaped downstream contract, but not the final downstream-readiness
claim now assigned to v0.9.

Delivered outcomes include:

- one common `TessellationResult` default in 2D and 3D;
- direct forward power computation from mathematical weights;
- stable site/ID/result association;
- canonical separator inverse ownership under `pyvoro2.inverse.separator`;
- explicit gauge and disconnected-component semantics;
- inspectable graph/Laplacian diagnostics and layered inverse results;
- an optional explicit SciPy sparse quadratic path;
- one bounded compatibility release and migration path for historical APIs;
- paper- and chemvoro-shaped integration validation.

See the [archived v0.7 development plan](../development/plans/archive/v0.7.md).

## v0.8 — Technical maintenance and Python 3.14 (source finalized)

v0.8 is intentionally a **technical-maintenance release without new numerical
functionality**. Its R1–R9 remediation and post-R9 `COPYING` distribution
correction are complete in the finalized source. Issue #33 qualifies the exact
source commit frozen after independent review before the public tag is created.

Delivered outcomes include:

- remove compatibility-only inverse/planar routes retained for v0.7;
- reorganize tests and private Python helpers without changing the public
  architecture;
- support Python 3.10–3.14 and the approved wheel matrix;
- qualify source distributions through an isolated sdist-to-wheel round trip;
- resolve accepted correctness, backend-safety, source-identity, active-state,
  diagnostic, and maintenance findings;
- preserve stable numerical behavior except for separately approved correctness
  fixes.

See the completed [v0.8 plan](../development/plans/archive/v0.8.md),
[remediation plan](../development/plans/archive/v0.8-remediation.md),
[pre-release audit](../development/audits/v0.8-pre-release.md), and
[ADR 0006](../development/decisions/0006-v0.8-cleanup-release.md).

## v0.9 — v0.9.0 functional stabilization and v0.9.x downstream soak

v0.9.0 is the last planned broad pre-1.0 functional/API refinement release. Its
purpose is to make the existing package a clean substrate for downstream
molecule, crystal, and independent MD-frame workflows while deliberate
pre-1.0 API changes are still inexpensive. Releasing v0.9.0 starts the v0.9.x
downstream-readiness/soak phase; it does not assert that the external soak has
already occurred.

### Required release outcomes

- Remove artificial wrapper/API restrictions that materially affect intended
  downstream workflows, especially periodic crystal use.
- Complete the preferred weight-first and periodic-query/output semantics needed
  by downstream code.
- Use chemvoro-shaped workflows as a primary API qualification oracle without
  moving chemistry-specific models or structure parsing into pyvoro2.
- Promote the normal realization-aware separator workflow from experimental to
  a **supported primary inverse contract**.
- Perform a final broad API refinement pass before the stronger 1.0 promise.
- Complete factual API/capability documentation alignment as a v0.9.0 gate.
  Run **content and information architecture**, then **visual/navigation design**
  or possible engine replacement as separately activated v0.9.x soak-period
  work, not as conditions for releasing v0.9.0.

The separator promotion does not turn the empirical outer active-set algorithm
into a convergence theorem. The fixed-observation inner problem remains the
exact/convex mathematical layer. Cycles, iteration limits, infeasibility, or
numerical failure may remain structured outcomes. Stable API means supported
inputs, semantics, provenance, result/status vocabulary, and failure reporting;
it does not mean every admissible problem converges.

The active v0.9 plan and target API inventory now fix the supported high-level
realization-aware entry-point shape. Ordinary callers should not need an API
labelled experimental merely to perform "points + separator observations ->
fitted weighted tessellation". Advanced path/iteration internals may remain
provisional or experimental if the normal workflow does not depend on them.

### Candidate v0.9 workstreams to preserve

These are candidates for later issue design, not a frozen issue list:

| Workstream | Intended correction |
|---|---|
| Weight-first query parity | Add mathematically consistent `weights=` support to `locate` and `ghost_cells`, including common generator/ghost gauge conversion. |
| Orientation-neutral `PeriodicCell` | Accept meaningful left- and right-handed user bases while preserving user vectors and integer shift labels. |
| Fractional/geometric cell helpers | Add Cartesian↔fractional conversion and exact half-open user-parallelepiped wrapping without changing backend-primary `remap_cart` semantics. |
| Representation-robust minimum images | Use exact private unimodular basis reduction in certified minimum-image geometry and map shifts back to the user basis. |
| Certified boundary image labels | Remove bounded-search correctness from face/edge reconstruction; require positive-measure semantic records, exact provenance handling, and backend-effective power-plane compatibility. |
| Periodic query/ghost metadata | Make original/user-wrapped/backend-site coordinates and owner/query shifts coherent; certify approximate-native translations and prevent indeterminate 3D ghost-ID reads. |
| Shift metadata without visible vertices | Permit internal temporary geometry for shift reconstruction without forcing detailed vertices into public output. |
| Search/tolerance/repair API lifecycle | Reassess controls such as face/edge search windows once they no longer determine correctness. |
| Separator inverse promotion | Stabilize the supported realization-aware high-level workflow while keeping empirical termination semantics explicit. |
| Downstream contract suite | Exercise nonperiodic molecules, orthorhombic/triclinic crystals, equivalent lattice representations, stable IDs/weights/measures/shifts, inverse fitting, and repeated independent frames. |

v0.9 does **not** mean implementing every mathematically possible wrapper
extension. Broad wall-domain work, persistent trajectories, automatic global
scaling, and similar expensive changes remain demand-driven unless real
chemvoro/downstream use promotes them.

## 1.0 — Stable main release and JOSS-ready core

Version 1.0 follows a successful released-v0.9.x soak and stabilizes the
functionality pyvoro2 already has. The soak must reveal no critical correctness
or API hole requiring another incompatible redesign, and downstream callers
should not need private APIs or manual workarounds. It does **not** wait for
prescribed-measure or mixed inverse solvers.

Expected gates:

- forward and periodic contracts validated by real downstream use;
- the normal separator inverse workflow supported through a non-experimental
  public contract with structured realization and termination diagnostics;
- a final public API, schema, capability, and lifecycle audit;
- no correctness-critical public search knob whose value changes the scientific
  answer for supported geometry;
- complete install, test, documentation, notebook, and release paths;
- reviewer-grade examples, benchmarks, reproducibility, citation, and archival
  metadata;
- JOSS preparation when repository history and release state are ready.

The 1.0 promise stabilizes supported inputs, meanings, outputs, and failure
semantics. It does not guarantee that every empirical iterative solve succeeds.

## v1.1 — Prescribed cell measures

The next inverse family targets fixed sites and domain, unknown power weights,
and prescribed cell areas in 2D or volumes in 3D.

Expected development order:

1. common cell-measure extraction;
2. target validation and explicit mass-balance policies;
3. residual evaluation without solving;
4. graph-structured sensitivity/Jacobian diagnostics;
5. finite-difference validation on stable generated cases;
6. select and implement an appropriate damped structure-aware nonlinear update;
7. empty and near-empty cell diagnostics;
8. generated-data recovery benchmarks;
9. partial and noisy targets;
10. expand from simpler 2D domains to stable 3D and periodic cases.

The solver should be judged by generated-data recovery modulo gauge and
structured failure diagnostics. Difficult periodic claims should reuse the
periodic representation/image contracts stabilized before 1.0 rather than add a
new correctness-sensitive search inside the measure solver.

## v1.2 — Mixed inverse problems

The first mixed problem should combine fixed-site, weights-only separator and
cell-measure observations with explicit block/row scaling and separate final
realization diagnostics.

Before freezing a generic composition protocol, consider built-in controls that
existing inverse workflows already need:

- per-row/group hard bounds and robust-loss scales;
- row-specific soft-penalty strengths where justified;
- per-site regularization or anchor controls;
- inspectable block and row objective contributions.

A shared private composition interface may be sufficient initially. A generic
**public** `ObservationBlock`-style protocol should be created only if the real
separator and measure implementations justify it. Arbitrary user callbacks and
site motion are not prerequisites for v1.2.

## Post-1.0 demand-driven engineering

Potentially valuable workstreams that should remain visible but are not release
commitments include:

- backend execution on a reduced equivalent periodic basis with geometry/shifts
  mapped back to the user's basis;
- exact clipping when generators lie outside the requested box;
- dominated coincident-site preprocessing for unequal power weights;
- optional automatic similarity scaling of coordinates, domains, weights/radii,
  tolerances, and outputs;
- public convex/wall domains;
- persistent native containers, trajectory APIs, parallelism, or other
  throughput work after profiling;
- package/distribution split reassessment if a real inverse-only native-free
  audience, second forward backend, divergent release cadence, or separate
  maintainership appears.

## Future research directions

Possible later research workstreams include:

- cell-centroid observations and separator/measure/centroid combinations;
- inverse fitting from planar sections or slices;
- richer regular-triangulation and dual diagnostics;
- optional solver backend plugins;
- bounded site-coordinate optimization as an explicit new unknown family;
- anisotropic or non-Euclidean models driven by a concrete research project.

## Informational and upstream-oriented limitations

Record these accurately without implying a committed pyvoro2 implementation:

- partial triclinic periodicity in 3D or oblique periodicity in 2D when correct
  support would require substantive backend work or a second periodic layer;
- unbounded cells/domains, which require a different result geometry contract;
- functionality requiring substantive Voro++ source changes rather than a
  bounded wrapper correction;
- guarantees beyond binary64/backend numerical resolution.

pyvoro2 should expose structured failures and adopt useful upstream Voro++
improvements when practical, but informational limitations are not scheduled
features.

## Pre-1.0 decision still to confirm

ADR 0017 settles **one repository and one distribution through 1.0**, with
strong internal forward/inverse boundaries and post-1.0 reassessment only under
a concrete trigger. One backend policy remains open:

1. whether to adopt a formal **no persistent pyvoro2-specific functional
   Voro++ fork** policy, including the treatment of narrowly carried
   correctness/upstream-backport patches.

The policy must be settled before accepting any v0.9 vendored Voro++ source
patch; a binding-only correction does not itself constitute a functional fork.

## Explicit near-term non-goals

- prescribed-measure or mixed inversion before 1.0;
- spherical-surface tessellations;
- a general-purpose replacement for broad geometry frameworks such as CGAL;
- arbitrary user-defined nonlinear callbacks before built-in inverse families
  establish a useful composition model;
- GPU acceleration without demonstrated need;
- site-coordinate optimization in the stable 1.0 inverse contract;
- planar oblique-periodic support solely for API symmetry;
- persistent trajectory/parallel APIs solely because MD is a downstream use
  case;
- preservation of historical inverse namespaces removed after v0.7.

## Planning responsibilities

- This page records version-level direction and preserves candidate workstreams.
- [Development plans](../development/plans/index.md) define active release
  outcomes, dependencies, validation, and gates.
- Decision records explain durable choices.
- GitHub milestones group release outcomes.
- GitHub issues define implementation tasks and acceptance criteria.
- The changelog records completed user-visible changes.
- Completed plans are archived with their outcome and deferrals.
