# 0029 — Image-qualified geometry and optional global periodic complex

- **Status:** Accepted
- **Date:** 2026-11-22
- **Scope:** architectural direction from G0, not an implementation or public-API specification
- **Research and acceptance record:** [G0 issue #149](https://github.com/DeloneCommons/pyvoro2/issues/149)
- **Frozen public evidence:** [G0 archive and SHA-256](https://github.com/DeloneCommons/pyvoro2/issues/149#issuecomment-7079961449)
- **Related:** [ADR 0027](0027-v0.9-reboot-from-v0.8.md),
  [ADR 0028](0028-upstream-voro-vendor-policy.md),
  [functional sequencing](../plans/functional-sequencing.md),
  [maintainer review register](../../../notes/maintainer-review/review-register.md)

## Context and evidence

The v0.8-based reboot needs image-qualified periodic contacts and may later need
global vertex/edge/face/cell incidence. Its published local Voronoi/Laguerre
functionality must not become dependent on expensive or fallible global
reconstruction. The historical v0.9 attempt is evidence, not adopted code or
mandatory native-certification machinery (ADR 0027).

G0 compared incidence-first assembly from Voro++ local occurrences with
independent source-power-halfspace construction and hybrid/restricted choices.
Three mathematical studies and two research prototypes were independently
audited, corrected and re-audited. The accepted research references are:

- **G0-P1:** `f8572cb02656d9763ae71f86c2754f10169cc3d3` — bounded observed-face
  image resolution, reciprocal polygon matching, consistent image transport
  and optional translated V/E/F/C attachments on a restricted 3D domain.
- **G0-D1:** `62864f014880a2d76ae6c38a984f7658a618c722` — separate source
  diagnostics, exact witnesses and bounded checks for wrong/missing geometry;
  includes the final documentation-only closure correction.
- **Original source baseline:** `b10d5739d4eddc412f6d4d9ca46cae6f53710f1b`.
  The independently reviewed ZIP and Git history bundle are attached to #149.

The periodic one-cell cube reconstructs a quotient with 1 vertex, 3 winding
edges, 3 faces and 1 cell, including translated attachments. A weighted
four-site example also establishes a necessary separation: at weight
`2^-52`, saved Voro++ cells are identical to the zero-weight-limit output
although the specified source power diagram has a positive omitted facet.
An exact assembly of the **observed** complex cannot by itself certify
source-semantic completeness.

The results are rigorous within their declared exact-operand and local-input
hypotheses. They do not qualify arbitrary floating native output or establish
representative production performance.

## Decision

1. **Keep Voro++ as the local-cell engine.** Local geometry and measures are
   the foundational forward product. The target design must preserve their
   independent availability when optional image resolution, contacts, global
   assembly or stronger diagnostics are unsupported, ambiguous or fail.
   This is a future layering requirement, not a claim that every existing
   `compute()` option already isolates derived-stage failures.

2. **Separate image-qualified contacts (G1a) from global assembly (G1b).**
   G1a is the first separately scoped production capability: identify neighbor
   owners and periodic images from valid local boundary evidence, checking
   reciprocal observed interfaces where the selected contract requires it.
   Its operation and results must not require successful G1b assembly.
   Multiple images per owner pair and nonzero-shift self contacts are legitimate.
   No user-basis shift, wall or neighbor is guessed after ambiguous or exhausted
   reconstruction.

3. **Allow optional restricted face-first assembly (G1b).** Establish global
   vertex/edge/face/cell identity through validated whole-face occurrence
   correspondences and exact integer transport, not proximity-based coordinate
   pooling. Preserve oriented translated attachments, parallel edges and
   legitimate nonzero winding. Initially target admitted bounded,
   full-dimensional convex 3D local cells with simple strictly convex face
   cycles and unique complete reciprocal matches: finite boxes, rectangular
   partial periodicity and nonsingular full periodic lattices. Unsupported
   degeneracies, subdivisions and inconclusive matching must explicitly refuse
   or return appropriately limited evidence; no source-complete or globally
   injective embedding is implied.

4. **Keep source-semantic diagnostics separate and optional.** A candidate
   image label, a validated reciprocal **observed** facet, and an independently
   confirmed positive **source** contact are different propositions.
   Independent source checks may establish positive contacts or exhibit
   missing native facets, but are not required on every ordinary computation.
   They must state the source/effective operand model, assumptions, region,
   verification scope and resource limits. A successful observed quotient
   does not become a source-complete diagram merely because its attachments
   are consistent; likewise a source-cell proof alone does not validate raw
   native polygon cycles.

5. **Use explicit evidence and failure semantics.** Distinguish exact
   stored-coordinate consistency, tolerance-dependent abstract gluing,
   certified source properties, a witnessed discrepancy, unresolved numerical
   disagreement, ambiguous images, exhausted budgets and unsupported inputs.
   Source (`S`), ideal stored backend operands (`E`) and native observations
   (`N`) remain useful *analytical distinctions*, not three mandatory
   runtime certification frameworks. No automatic insertion, collapse or
   repair of omitted positive facets is approved.

6. **Approve only essential F1 design prerequisites, not a full legacy port.**
   Future implementations must decide explicit source/primary/native/reporting
   frames, preserved original operands and prepared weights/radii, source-lift
   integers, user-basis shifts and exact unimodular transport, genuine domain
   walls versus unknown labels, stable local occurrence IDs, and mutation-safe
   snapshots for derived computations. Complete finite image searches or
   honest budget refusal are required for the claims each algorithm makes.
   Historical WP2–WP4 reduction and periodic-image algorithms are candidate
   evidence for F1, not automatically reinstated implementations or public
   interfaces. Numerical tolerances and production API shapes remain separate
   design/approval questions.

## Consequences and staging

- **G0 research is accepted as complete.** The public issue #149 holds the
  immutable evidence archive, two independent joint/closure reviews and
  revision pins; the private research workspace may remain available.
- **F1** selects the smallest necessary frame, operand, occurrence, image,
  ownership and result-lifetime contracts in separate reviewed scopes.
- **G1a** is the preferred first contact implementation. **G1b** remains a
  separately accepted, optional restricted capability. Source diagnostics
  can evolve independently as targeted checks.
- **F2 and inverse I0/I1** are independent. Realization-aware inverse **I2**
  needs trustworthy G1a contacts but not G1b.
- Baseline v0.8 public behavior remains in force unless a specific approved
  change says otherwise. No public class, schema, release version, CI gate or
  vendor patch is authorized by this ADR.
- Full source-topology reconstruction, noisy-coordinate recovery, general
  nonconforming/collinear faces, mixed-dimensional/coincident-owner complexes,
  oblique partial periodicity, automatic geometric repair and universal deep
  diagnostics remain outside the initial scope. Finite exact algorithms may
  refuse expensive highly skewed representations; practical basis reduction
  requires separate cost and correctness decisions.

## Alternatives considered

- **Unconditionally globalize all native output:** rejected; observed native
  face sets can omit genuine positive source interfaces, and invalid cycles
  must not be admitted.
- **Make independent source construction or full certification mandatory:**
  rejected; valuable as targeted diagnostics but not justified as the normal
  local-cell cost or a second production tessellation engine.
- **Use coordinate closeness to identify global vertices:** rejected as an
  identity rule; distinct exact vertices may round to the same float coordinates.
- **Reinstate the historical v0.9 native-operation/N–E–S proof stack:** rejected
  as a blanket prerequisite; preserve its counterexamples and sound ideas
  without importing unneeded runtime machinery.
- **Require G1b before useful G1a contacts or inverse I2:** rejected; the
  responsibility boundaries and research evidence support G1a independently.
