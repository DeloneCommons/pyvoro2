# Functional research and implementation sequencing

- **Status:** Draft — G0 geometry layering selected in ADR 0029; implementation
  scopes and release gates are **not yet approved**
- **Plan revision:** 0.2
- **Target release:** Not fixed; candidate work for the v0.8-based v0.9
  development line and, when justified, later releases
- **Base release:** published v0.8.0; historical v0.9 attempt is evidence only
- **Integration branch:** `dev`
- **Planning agreement:** 2026-11-19
- **GitHub milestone:** none; create only after selecting actual release outcomes

## Intended outcome

Build a bounded, coherent periodic foundation before production image-qualified
contacts. The post-G0 refinement in
[#151](https://github.com/DeloneCommons/pyvoro2/issues/151) prioritizes, in order,
scientific/mathematical correctness, long-term architectural reliability and
avoiding rework, implementation and verification cost, then earliest G1a
delivery. Independently useful forward/query work and algebraic inverse work
can advance in parallel, without reconstructing the historical v0.9
qualification machinery or forcing a premature public API.

Preserve experimental scope and existing compatibility boundaries. No
intermediate release is planned before the future v0.9 line; no v0.9 scope,
deadline or milestone is selected.

This plan establishes **research order and dependency gates**, not an approved
implementation, feature list, release timeline, public contract or permission
to rewrite numerical/native code. The G0 architecture direction is selected in
[ADR 0029](../decisions/0029-image-qualified-geometry-and-optional-global-complex.md);
[ADR 0027](../decisions/0027-v0.9-reboot-from-v0.8.md)
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
| **G0 — geometry architecture research** (R-O01, P-01a) | **Completed research; direction accepted in ADR 0029.** Independent incidence-first and source-semantic prototypes establish a restricted layering and its limits. | Exit evidence archived in [#149](https://github.com/DeloneCommons/pyvoro2/issues/149); no production feature approved by G0. |
| **P0 — contract and baseline counterexamples** | Settle frame/operand/shift/failure vocabulary, candidate public semantics, compatibility and G1a's minimal input-evidence needs; independently reproduce current defects. | Future separately scoped issue before F1 implementation; no production algorithm changes or G1a implementation. |
| **F1-A — user lattice, source coordinates and native frame** (R-L01, R-L03a) | A1 exact user-basis core; A2 separately qualified numerical native frame; A3 source/preparation separation and backend-primary remapping. | P0; focused numerical/source-semantics review. Pure user operations remain separate from native admissibility. |
| **F1-B — private computational basis** (R-L02b) | B1 bounded reducer comparison/decision, then a deterministic exact rank-3 reducer with exact transforms and independent postconditions. | Accepted A1 exact core; may proceed alongside A2/A3. R-L02b remains pending; no algorithm approved by this plan. |
| **F1-C — integration, evidence and G1-ready foundation** (R-L02a, R-L03b, R-T02a, R-M01b) | C1 complete nearest-image integration; C2 duplicate safety/insertion evidence; C3 occurrence/snapshot contracts; C4 mandatory translation-recovery analysis and disposition. | Applicable A/B interfaces; C3 preparation may use frozen interfaces. B/C integration and F1→G1a evidence reviews remain required. |
| **F2 — independent forward/query extensions** (R-Q01, R-G01a/b, R-G02a, R-M01a) | Weight-first `locate`/`ghost_cells`, independent ghost batches, initialized IDs, query-owner and ghost-image metadata. | Can start alongside G0; synchronize any shared image/frame contracts with F1. No need to wait for G1b. |
| **G1a — derived image-qualified adjacency** | Separately implement image-qualified contacts from observed local face/edge evidence, with bounded/complete matching for the admitted claim or honest refusal. | F1 integration/admission review, including explicit C4 disposition; separately approve precise 3D scope and independently review the 2D adapter. Local cells remain independently usable. |
| **G1b — optional global periodic complex** | Separately authorized restricted face-first observed-complex assembly with oriented incidence/winding; no invented missing facets or zero shifts. | Separate G1a→G1b admission review of face cycles, reciprocal polygons, transport and V/E/F/C attachments; contacts alone do not establish these. |
| **I0 — inverse correctness first** (R-I05, R-I06) | Reproduce historical final-state, hard-feasibility, affine-residual and numeric-diagnostic failures on the *current* v0.8-based source, repair only confirmed defects with independent oracles. | May proceed alongside G0/F1/F2; necessary before expanding unstable result semantics. |
| **I1 — algebraic inverse model/API** (R-I01–R-I03) | Independently review observation, mismatch, hard-bound and penalty spaces; row bounds/applicability and shape-parameter candidates, with source-row identity and zero-strength semantics. | Can design alongside G0 and after I0 baseline review. Each model/API change needs a separate contract and tests. |
| **I2 — realization-aware facade** (R-I04) | Design a normal high-level workflow over fitting, actual contacts/images, final-state and outer termination while retaining appropriate advanced/experimental controls. | Requires reliable G1a contact semantics and validated inverse state; does **not** require full G1b. |
| **U — upstream Voro++ triage** (R-O02a/b, R-O03) | Compare confirmed defects against pinned upstream `dev`; consider bounded local backports only where warranted. | Parallel to other research; any native patch follows [ADR 0028](../decisions/0028-upstream-voro-vendor-policy.md) and its own approval. |

**Critical path:** G0 (done) → P0 (separate future issue) → F1-A → F1-B → F1-C
→ F1 integration/admission review → G1a → G1a-to-G1b admission review →
optional G1b.

This is a dependency structure, not one PR per letter or purely serial
implementation: after A1's accepted exact core, B1 may run alongside A2/A3;
suitable C3 preparation may use frozen interfaces. F2, I0/I1 and per-defect
upstream/native triage remain independent/parallel. I2 needs G1a and validated
inverse state, **not** G1b.

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

The **G0 exit gate** was met by independently reviewed research and the
[archived evidence in issue #149](https://github.com/DeloneCommons/pyvoro2/issues/149#issuecomment-7079961449).
[ADR 0029](../decisions/0029-image-qualified-geometry-and-optional-global-complex.md) selects the restricted layered
direction: local cells, independent image-qualified observed contacts (G1a),
optional face-first observed-complex assembly (G1b), and separate source
checks. This does not authorize new public classes, global certification on
every `compute()`, or removal of baseline diagnostics.

## F1-A/B/C foundation and review boundaries

These tentative reviewable units define a bounded foundation, not a legacy
port or implementation authorization. **P0 is future work in a separate
issue:** a small contract and baseline-counterexample package settling exact
frame/operand/shift/failure vocabulary, candidate public semantics,
compatibility boundaries and G1a's minimal input-evidence needs. It must
independently reproduce current [#60](https://github.com/DeloneCommons/pyvoro2/issues/60),
[#65](https://github.com/DeloneCommons/pyvoro2/issues/65),
[PR #67](https://github.com/DeloneCommons/pyvoro2/pull/67) and native-frame
arithmetic defects; it changes no production algorithms and implements no G1a.
This plan revision neither performs P0 nor creates its issue.

### Three lattice representations and coordinate evidence

Use the **row-basis** convention $x=o+fA$:

1. **User/source $A$:** preserve supplied vector order, handedness,
   fractional/wrapping conventions, original binary64 operands and public
   image coefficients. Mathematical source validity is distinct from backend
   feasibility.
2. **Private computational $B=UA$:** exact unimodular integer $U$, with
   $s_{\mathrm{user}}=s_{\mathrm{reduced}}U$. This preconditions complete exact
   periodic search; it is neither a canonical/Niggli representation nor
   automatically the Voro++ basis.
3. **Native Voro++ frame:** separately prepared numerical lower-triangular
   representation, submitted/stored coordinates and output-to-Cartesian
   reporting transform, with independent numerical errors and admissibility.

Distinguish original source, prepared/primary, submitted native, actually
stored native and reported coordinates, and mathematical weights from actual
stored backend radii. A known integer change of representatives alone does
not certify source/native geometry equivalence. If insertion changes a chart,
source-reference coefficient transport needs a justified insertion transition;
missing evidence means unavailable, never guessed arithmetic or a zero image.

### F1-A — user lattice, source coordinates and native frame

- **A1 — exact user basis:** plan orientation-neutral finite binary64 lattices
  with exactly nonzero determinant, both handedness signs and original vector
  order; public fractional/Cartesian transformations and exact-decision
  wrapping with rounded float views. Pure user-coordinate operations need not
  require native Voro++. Retain native numerical safeguards as a separate
  boundary; mathematical validity does not imply native executability.
- **A2 — native frame qualification:** investigate one stable, sign-normalized
  nonpivoted QR frame, parity/reflection and face-cycle orientation, reconciling
  duplicated Gram–Schmidt implementations. The current frame has an
  independently reproduced near-parallel-vector nonisometry. QR is the
  preferred design to approve in A2, not an approved rewrite, native-container
  rebasing onto $B$, or relaxation of existing native safety.
- **A3 — source/preparation separation:** correct backend-primary remapping
  and source consumers, covering #65's coupled x-residual loss, #67's
  boundary/zero-epsilon/subnormal cases and #60's source snapping. Exact user
  wrapping and intentionally numerical backend-primary remapping remain
  distinct. The narrow existing separator source-coordinate consumer
  correction may proceed early in A3 or a focused related correction without
  waiting for C1; no inverse solver/model redesign.

### F1-B — deterministic internal computational basis

**B1** first requires a bounded comparison/decision, then selection and
implementation of a private *good deterministic* exact rank-3 reducer with
exact $B$, $U$, $U^{-1}$ and independent postconditions. Exact LLL with
$\delta=3/4$ is the preferred candidate, supported by historical experiments
and a rank-three counterexample to simplistic pairwise reduction, **not an
approved or mandatory algorithm**. Current R-L02b remains pending. Historical
[WP2 #54](https://github.com/DeloneCommons/pyvoro2/issues/54),
[WP3 #56](https://github.com/DeloneCommons/pyvoro2/issues/56),
[WP4 #58](https://github.com/DeloneCommons/pyvoro2/issues/58) and
[E02–E04](../reboot/evidence-index.md#e02) are evidence, not adopted contracts.

Do not require canonical/optimal reduction, expose a public reducer, replace
the full closest-vector algorithm or feed $B$ to the native container. Use
arbitrary-precision integer transport; check requested public fixed-width
materialization only after exact decisions. Preserve rectangular fast paths
where appropriate. Complete unreduced exact search after a reducer resource
limit is permissible only if that fallback policy, remaining resource budget
and correctness are separately approved; never mask invariant failures.

### F1-C — integration, evidence and G1-ready foundation

- **C1 — nearest-image integration:** integrate reduced exact proof geometry
  into the current complete search, with physical Cartesian tie semantics,
  pair reversal and translation covariance. Preserve source operands and
  existing orientation authority; materialize int64/float views late. No
  duplicate closest-vector solvers or arbitrary search stencils.
- **C2 — duplicate and insertion safety:** reuse exact geometry for complete
  duplicate detection and distance-only classification; retain the mandatory
  native insertion-safety backstop. Assess actual insertion disposition/chart
  transitions, including a current-build reproduction of the historical
  near-endpoint case, before asserting source-reference identities. This
  analysis authorizes no native patch.
- **C3 — occurrence and lifetime contracts:** establish stable local occurrence
  IDs scoped to an immutable consumed-field snapshot. Distinguish actual walls,
  known neighbor images, unknown identities and exhausted budgets; preserve
  raw mutable cell outputs and existing in-place semantics. Separate native
  observations from transformed reporting coordinates. Do not force expensive
  extraction/global assembly into ordinary `compute()` or implement the face
  matcher here.
- **C4 — required analysis and explicit disposition:** evaluate a reusable exact
  finite-box lattice-translation recovery primitive, named G1a/G1b consumers,
  alternatives, numerical assumptions, proof completeness, resource policy and
  independent tests. Before F1→G1a admission, record one reviewed disposition:
  implement in F1-C; satisfy the actual consumer through a G1a-specific
  alternative; or defer code with concrete evidence that the selected G1a
  contract does not require it. This evaluation cannot be omitted or moved to
  an unexamined backlog. Full exact compatibility-box search distinguishes
  zero/one/many/incomplete; uniqueness within a supplied box does **not** certify
  the native producer's error bound. No universal epsilon or automatic native
  attribution is approved. **C4 decision is mandatory; C4 code is conditional.**

### Stage reviews and G1 admission

Require a focused **F1-A numerical/source-semantics gate**, a **B/C
mathematical/integration gate**, and an **F1→G1a evidence/provenance gate**,
including mandatory C4 disposition. These reviews are boundaries, not extra
monolithic feature projects.

Separately scope G1a image-qualified contacts from **observed** local face/edge
evidence. Nearest-image CVP does not enumerate every legitimate contact image;
multiple contacts per owner pair and nonzero self images remain valid. Require
independently bounded/complete matching for the admitted claim or honest
refusal, without mandatory global reconstruction or repair of source-omitted
facets. Define the precise initial 3D scope and independently review the 2D
adapter; G0-P1 did not establish all 2D cases.

The separate **G1a→G1b admission review** must establish simple strictly convex
face cycles, unique complete reciprocal polygon correspondence, coherent
integer transport and oriented V/E/F/C attachments; verified contacts alone
do not establish these. Include G0-P1's star/self-intersection control and the
$2^{-52}$ omitted-source-facet negative control. G1b remains separately
authorized optional restricted face-first observed-complex assembly with
incidence/winding. A correct observed quotient is not complete source power
geometry; source diagnostics remain separate and optional.

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
- Require independent frame/source, exact transport/search and insertion-chart
  evidence at the F1-A, B/C and F1→G1a reviews. Native safety remains mandatory;
  mathematical lattice validity is not native admission. Reducer comparison,
  resource/failure policy and any fallback need their own approval.
- F1→G1a admission requires a recorded C4 analysis/disposition; G1a→G1b has a
  separate polygon/transport/attachment gate. Preserve the star and omitted
  source-facet negative controls; observed consistency is not source
  completeness, and an exact box result does not certify its producer's bound.
- No requirement that ordinary `compute()` run a global topology algorithm
  or the historical N/E/S proof-admission framework. A global result must
  explicitly distinguish verified, estimated/approximate and unavailable
  information as its eventual contract specifies.
- A G0 research dead end must **not** block F2, I0, I1, baseline maintenance
  or an independently approved release scope.
- Any vendored/native change follows the per-defect policy of
  [ADR 0028](../decisions/0028-upstream-voro-vendor-policy.md).
- Stop for maintainer input if an accepted ADR conflicts with a proposed
  statement, current `dev` has revised the plan differently, or a planning edit
  would implicitly authorize implementation. Preserve ADR 0027/0028/0029 and
  register statuses; no provisional API candidate is approved by this plan.
- Implementation PRs must update affected API/Guide/Theory pages and examples
  as their semantics change. Save a full documentation content/consistency pass
  for the eventual DOC-W99, after the functional contract stabilizes.

## Next explicit decisions

1. **G0 completed:** preserve the [#149 evidence archive](https://github.com/DeloneCommons/pyvoro2/issues/149#issuecomment-7079961449) and apply [ADR 0029](../decisions/0029-image-qualified-geometry-and-optional-global-complex.md) as the architectural direction.
2. **P0 next, in a separate future issue:** settle contract vocabulary,
   candidate semantics/compatibility, minimal G1a input evidence and independent
   current-baseline counterexamples. P0 has not been performed by this revision.
3. Separately scope F1-A/B/C units and approve their stage contracts: qualify
   the preferred A2 QR design, decide the B1 reducer after bounded comparison
   (LLL remains a candidate; R-L02b pending), and record C4's mandatory
   disposition before the F1→G1a evidence/provenance admission review.
4. Scope and review G1a independently, including its precise 3D contract and
   2D adapter; require the separate G1a→G1b polygon/transport/attachment review
   before optional G1b. F2, I0/I1 and per-defect native triage remain parallel;
   I2 needs G1a and validated inverse state, not G1b.
5. **Only after feasibility and downstream needs are clearer**, choose the
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
| 2026-11-22 | G0 research audited and archived in issue #149; ADR 0029 selects the layer boundary, preserving separate F1/G1a/G1b implementation gates. |
| 2026-11-22 | Draft revision 0.2 ([#151](https://github.com/DeloneCommons/pyvoro2/issues/151)): post-G0 foundation-first refinement adds future P0, tentative F1-A/B/C units, mandatory C4 analysis/disposition and separate G1 admission reviews; preserves parallel tracks and all implementation/release approval boundaries. |
