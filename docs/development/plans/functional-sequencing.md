# Functional research and implementation sequencing

- **Status:** Draft — maintainer agreed the dependency order; implementation
  scopes and release gates are **not yet approved**
- **Plan revision:** 0.1
- **Target release:** Not fixed; candidate work for the v0.8-based v0.9
  development line and, when justified, later releases
- **Base release:** published v0.8.0; historical v0.9 attempt is evidence only
- **Integration branch:** `dev`
- **Planning agreement:** 2026-11-19
- **GitHub milestone:** none; create only after selecting actual release outcomes

## Intended outcome

Choose the smallest scientifically coherent, useful forward/periodic and
separator-inverse improvements without reconstructing the historical v0.9
qualification machinery or forcing a premature public API. Prioritize the
**mathematical architecture of image-qualified contacts and global periodic
topology**, then implement its real prerequisites. Independently useful
forward/query work and algebraic inverse work can advance in parallel.

This plan establishes **research order and dependency gates**, not an approved
algorithm, feature list, release timeline, public contract or permission to
rewrite numerical/native code. [ADR 0027](../decisions/0027-v0.9-reboot-from-v0.8.md)
continues to govern the reboot, and the
[review register](https://github.com/DeloneCommons/pyvoro2/blob/dev/notes/maintainer-review/review-register.md)
remains the source of per-capability decisions.

## Baseline and architectural constraints

The current library provides local 2D/3D Voronoi/power cells, periodic
image-labelled contacts when requested, normalized result helpers and stable
fixed-observation separator fitting. The preserved historical v0.9 work has
useful tests and counterexamples but is **not** automatically adopted.

Keep three geometric responsibilities separate:

1. **Local cells:** usable forward geometry and measures regardless of whether
   optional adjacency/globalization succeeds.
2. **Image-qualified realized contacts:** neighbor identity, periodic image
   shift and face/edge evidence, including self-images and multiple images per
   pair.
3. **Global periodic complex:** vertex/edge/face correspondence, incidence and
   winding across cell and lattice images.

The **separator-observation graph** is not the realized-contact graph. A fitted
pairwise separator equation need not correspond to an actual boundary.
A realized-contact checker need not require the entire global complex.

The historical N/E/S certificates, native-operation replay and blanket
native-artifact qualification are not adopted as mandatory runtime machinery.
This does **not** assert that the current Voro++ face list is complete or that
a future global-complex algorithm has an unconditional correctness guarantee.

## Sequenced tracks and dependencies

| Track | Priority and purpose | Dependency / decision gate |
|---|---|---|
| **G0 — geometry architecture research** (R-O01, P-01a) | **Start first.** Compare incidence-first contact/topology reconstruction with direct semantic Laguerre-halfspace construction. Identify exact mathematical contracts and a feasible minimal layer boundary. | No preceding feature implementation; independent analysis and prototypes first. |
| **F1 — essential periodic/identity contracts** (R-L01, R-L02a, R-L03a, R-T02a, R-M01b) | Specify source/primary frames, lattice transforms, integer image shifts, physical ties, preserved original operands, genuine wall versus unresolved identity and output ownership. | Shape after initial G0 requirements are clear; approve each API/semantic change separately. |
| **F2 — independent forward/query extensions** (R-Q01, R-G01a/b, R-G02a, R-M01a) | Weight-first `locate`/`ghost_cells`, independent ghost batches, initialized IDs, query-owner and ghost-image metadata. | Can start alongside G0; synchronize any shared image/frame contracts with F1. No need to wait for G1b. |
| **G1a — derived image-qualified adjacency** | Implement the independently validated realized-contact layer or improve existing functions only after choosing the G0 strategy. | G0 decision and required subset of F1; local cell computation never depends on success here. |
| **G1b — optional global periodic complex** | Global vertex/edge/face identities, incidence, lattice winding and explicit limitations; no invented missing facets or zero shifts. | G0 decision, F1 prerequisites and a trustworthy G1a interface; separate acceptance. |
| **I0 — inverse correctness first** (R-I05, R-I06) | Reproduce historical final-state, hard-feasibility, affine-residual and numeric-diagnostic failures on the *current* v0.8-based source, repair only confirmed defects with independent oracles. | May proceed alongside G0/F1/F2; necessary before expanding unstable result semantics. |
| **I1 — algebraic inverse model/API** (R-I01–R-I03) | Independently review observation, mismatch, hard-bound and penalty spaces; row bounds/applicability and shape-parameter candidates, with source-row identity and zero-strength semantics. | Can design alongside G0 and after I0 baseline review. Each model/API change needs a separate contract and tests. |
| **I2 — realization-aware facade** (R-I04) | Design a normal high-level workflow over fitting, actual contacts/images, final-state and outer termination while retaining appropriate advanced/experimental controls. | Requires reliable G1a contact semantics and validated inverse state; does **not** require full G1b. |
| **U — upstream Voro++ triage** (R-O02a/b, R-O03) | Compare confirmed defects against pinned upstream `dev`; consider bounded local backports only where warranted. | Parallel to other research; any native patch follows [ADR 0028](../decisions/0028-upstream-voro-vendor-policy.md) and its own approval. |

**Critical path:** G0 research → selected F1 contracts → G1a contact layer →
G1b global complex. I2 depends on G1a, **not** on G1b. F2 and I0/I1 have
independent work and should not be delayed until G1b finishes.

These are **decision and dependency labels**, not an attempt to restore the
historical WP1–WP13 implementation/release schedule.

## G0 — required research deliverable before choosing geometry implementation

Produce a focused comparison of at least:

- **Incidence-first:** establish boundary identity/correspondence and periodic
  image transport before merging nearby vertex coordinates. Explain how
  positive native facets missing from returned local data would be detected or
  explicitly left unresolved.
- **Direct semantic construction:** derive relevant contacts from Voronoi/
  Laguerre halfspaces and sufficiently complete periodic images. Specify image
  search bounds, computational cost, degeneracy/empty and lower-dimensional
  cases, and the independent evidence needed for completeness.
- **Hybrid or restricted-domain alternatives** when neither can satisfy a
  useful general contract at acceptable cost.

Test representative finite/orthorhombic/triclinic geometries, ordinary/power
cases, multiple images, self-images, ambiguous point correspondence, nearly
degenerate edges/faces and nonzero winding. Use historical
[requirements R-O01](../reboot/requirements-inventory.md#r-o01) and
[evidence E06, E09 and E10](../reboot/evidence-index.md) as input, not as
pre-accepted proofs. Specify whether a result is exact, tolerance-dependent,
incomplete, unavailable, or an unsupported configuration; preserve local
geometry in all those outcomes.

The **G0 exit gate** is an independently reviewed research recommendation
and precise interface requirements for local cells, contacts and globalization.
It is acceptable to conclude that only G1a is currently feasible. There is no
automatic permission to add new public classes, demand global certification
on every `compute()`, or discard baseline diagnostics.

## F1/F2 — minimum foundation, not a blanket prerequisite list

Before G1a implementation, decide only the F1 contracts actually necessary
for its correctness: consistent lattice/image reference frame, exact integer
shift transport where decisions are discrete, physical tie semantics, reliable
source coordinate handling, and honest unknown-versus-wall/zero distinctions.
Do not first complete every selected query/ghost enhancement.

F2's weight-first `locate` and `ghost_cells` are valuable public API
improvements but do not mathematically define the global complex. Ghost state,
owner-image metadata and mutable-output aliasing must be independently safe,
yet unrelated F2 work should advance without waiting for G1b.

## Inverse development in parallel

Separate **algebraic inverse contracts** (observation graph, row policies,
penalties and final fitted state) from **geometric realization** (actual
image-qualified contacts). Reproduce R-I05/R-I06 counterexamples on the
current source before importing historical fixes or broadening the model.
Design R-I01–R-I03 with explicit units, row identity, coefficient/measurement
transforms, zero-confidence and zero-strength behavior, and transparent solver
limitations. The realization-aware facade may become a stable callable
interface only after its final-state and contact-status semantics are reliable;
a stable API does **not** promise universal active-set convergence.

## Validation, compatibility and stop conditions

- Preserve v0.8 numerical/public behavior unless a focused approval changes
  it; require independent analytic, geometric or fault-injection oracles where
  feasible, including relevant historical counterexamples.
- Keep `TessellationResult`, dimensional support and existing normalized
  output semantics until a specific migration proposal is accepted. Unknown
  image identity is never silently replaced with shift zero.
- No requirement that ordinary `compute()` run a global topology algorithm
  or the historical N/E/S proof-admission framework. A global result must
  explicitly distinguish verified, estimated/approximate and unavailable
  information as its eventual contract specifies.
- A G0 research dead end must **not** block F2, I0, I1, baseline maintenance
  or an independently approved release scope.
- Any vendored/native change follows the per-defect policy of
  [ADR 0028](../decisions/0028-upstream-voro-vendor-policy.md).
- Implementation PRs must update affected API/Guide/Theory pages and examples
  as their semantics change. Save a full documentation content/consistency pass
  for the eventual DOC-W99, after the functional contract stabilizes.

## Next explicit decisions

1. Scope and approve a **G0 research issue** with independent mathematical
   counterexamples and prototype/evaluation expectations.
2. Separately confirm transitional Voro++ policy in ADR 0028; triage specific
   R-O02b/R-O03 cases on upstream `dev` instead of treating them as a universal
   pre-C++ gate.
3. Translate G0 requirements into individually approved F1 tasks and research
   decision(s) for G1a/G1b. Split F2 and I0/I1 into useful independent issues
   as contracts are settled.
4. **Only after feasibility and downstream needs are clearer**, choose the
   actual v0.9 release scope, milestone, qualifications and deferred work.

## Non-goals

No immediate implementation approval for a global periodic complex, certified
native clipping, solver rewrites, new inverse observation families, persistent
trajectory engine, new domain/backend families, wholesale upstream-`dev`
migration, unconditional Voro++ vendor patching or renewed release promises.
Prescribed cell-measure and mixed separator/measure inversion remain
post-1.0 research possibilities, not gates for this plan.

## Record

| Date | Change |
|---|---|
| 2026-11-19 | Agreed research-first geometry ordering, parallel independent forward/inverse tracks, and the separate transitional Voro++ decision. |
