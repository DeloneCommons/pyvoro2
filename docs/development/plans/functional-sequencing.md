# Functional research and implementation sequencing

- **Status:** Draft — G0/P0 research complete; A1/A2/A3-R/A3-S accepted; remaining
  implementation scopes and release gates are **not yet approved**
- **Plan revision:** 0.3
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
to rewrite numerical/native code. Prefer the smallest useful scientific
contract and shared implementation. Extreme counterexamples can test a general
algorithm without making every finite input executable; judge added complexity
and runtime cost against demonstrated failures and representative workloads.
The G0 architecture direction is selected in
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
| **P0 — contract and baseline counterexamples** | **Completed research:** vocabulary, candidate contracts and independent baseline characterization. | [#153](https://github.com/DeloneCommons/pyvoro2/issues/153) / [PR #154](https://github.com/DeloneCommons/pyvoro2/pull/154); delivery is not blanket acceptance of its candidates. |
| **F1-A — user lattice, source coordinates and native frame** (R-L01, R-L03a) | **A1/A2/R/S accepted**; A3-P inserted-point preparation implemented under [#166](https://github.com/DeloneCommons/pyvoro2/issues/166), pending independent review. | Review P and focused F1-A composition; no F1 closure or release approval. Pure user operations remain separate from native admission. |
| **F1-B — private computational basis** (R-L02b) | B1 profile named search workloads and compare reducers; implement a private deterministic exact rank-3 reducer only if justified. | Accepted A1 exact core; independent of A3. R-L02b remains pending; no algorithm approved or unconditional G1a prerequisite. |
| **F1-C — integration, evidence and G1-ready foundation** (R-L02a, R-L03b, R-T02a, R-M01b) | C1 physical ties and conditional reducer integration; C2 duplicate/insertion evidence; C3 occurrence/snapshot contracts; C4 translation-recovery disposition within G1a selection. | Relevant A interfaces; B1 only for reducer integration. Review actual changed interfaces and evidence needed by the selected G1a contract. |
| **F2 — independent forward/query extensions** (R-Q01, R-G01a/b, R-G02a, R-M01a) | Independent ghost batches, initialized IDs, safe uninserted query handling, weight-first `locate`/`ghost_cells`, query-owner and ghost-image metadata. | Prioritize confirmed correctness defects independently; synchronize shared image/frame contracts with F1. Remote-query recovery is separately scoped; no need to wait for G1b. |
| **G1a — derived image-qualified adjacency** | Separately implement image-qualified contacts from observed local face/edge evidence, with bounded/complete matching for the admitted claim or honest refusal. | Select useful scope and actual F1 prerequisites, including explicit C4 disposition; approve precise 3D scope and independently review the 2D adapter. Local cells remain independently usable. |
| **G1b — optional global periodic complex** | Candidate restricted face-first observed-complex assembly with oriented incidence/winding; no invented missing facets or zero shifts. | Requires a separate G1a→G1b admission review and explicit implementation approval; assess face cycles, reciprocal polygons, transport and V/E/F/C attachments. Contacts alone do not establish these. |
| **I0 — inverse correctness first** (R-I05, R-I06) | Reproduce historical final-state, hard-feasibility, affine-residual and numeric-diagnostic failures on the *current* v0.8-based source, repair only confirmed defects with independent oracles. | May proceed alongside G0/F1/F2; necessary before expanding unstable result semantics. |
| **I1 — algebraic inverse model/API** (R-I01–R-I03) | Independently review observation, mismatch, hard-bound and penalty spaces; row bounds/applicability and shape-parameter candidates, with source-row identity and zero-strength semantics. | Can design independently after relevant I0 baseline review. Each model/API change needs a separate contract and tests. |
| **I2 — realization-aware facade** (R-I04) | Design a normal high-level workflow over fitting, actual contacts/images, final-state and outer termination while retaining appropriate advanced/experimental controls. | Requires reliable G1a contact semantics and validated inverse state; does **not** require full G1b. |
| **U — upstream Voro++ triage** (R-O02a/b, R-O03) | Compare confirmed defects against pinned upstream `dev`; consider bounded local backports only where warranted. | Parallel to other research; any native patch follows [ADR 0028](../decisions/0028-upstream-voro-vendor-policy.md) and its own approval. |

**Preferred integration order for A3:** R → S → P, then a focused F1-A
composition review. This order reduces conflicts in shared Python files; S
does not mathematically require P, and no A3 unit requires B1.

**Dependencies toward G1a:** select a concrete observed-contact contract and
implement the source/frame, search, insertion and snapshot capabilities it
actually requires. C1 physical ties can use the current complete search; its
reducer integration depends on B1 only if selected. C4 analysis and an explicit
disposition remain required during G1a selection. Review applicable F1 evidence
before G1a implementation, not every possible B/C extension as a blanket gate.
G1b is optional and has a separate admission review.

F2, I0/I1 and per-defect upstream/native triage remain independent; I2 needs
G1a and validated inverse state, **not** G1b. These are decision and dependency
labels, not an attempt to restore the historical WP1–WP13 release schedule.

## G0 — required research deliverable before choosing geometry implementation

**Completed research.** The brief required a focused comparison of at least:

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

The brief also required representative finite/orthorhombic/triclinic geometries,
ordinary/power cases, multiple images, self-images, ambiguous point
correspondence, nearly degenerate edges/faces and nonzero winding. Historical
[requirements R-O01](../reboot/requirements-inventory.md#r-o01) and
[evidence E06, E09 and E10](../reboot/evidence-index.md) were research inputs,
not pre-accepted proofs. The brief required distinguishing exact,
tolerance-dependent, incomplete, unavailable and unsupported results, while
preserving local geometry.

The **G0 exit gate** was met by independently reviewed research and the
[archived evidence in issue #149](https://github.com/DeloneCommons/pyvoro2/issues/149#issuecomment-7079961449).
[ADR 0029](../decisions/0029-image-qualified-geometry-and-optional-global-complex.md) selects the restricted layered
direction: local cells, independent image-qualified observed contacts (G1a),
optional face-first observed-complex assembly (G1b), and separate source
checks. This does not authorize new public classes, global certification on
every `compute()`, or removal of baseline diagnostics.

## F1-A/B/C foundation and review boundaries

These reviewable units define a bounded foundation, not a legacy port or
blanket implementation authorization. P0 was delivered through
[#153](https://github.com/DeloneCommons/pyvoro2/issues/153) and
[PR #154](https://github.com/DeloneCommons/pyvoro2/pull/154): its
[baseline observations and candidate contracts](https://github.com/DeloneCommons/pyvoro2/blob/dev/notes/maintainer-review/contracts/f1-periodic-foundation.md)
are historical evidence, not the current implementation or collectively
approved policy. A1 and A2 subsequently landed under ADRs 0030 and 0031.
Reconcile the original P0 oracles record by record when changing A3; the
18 remaining discrepancy records after A2 describe observations, not 18
independent implementation tasks.

### Three lattice representations and coordinate evidence

Use the **row-basis** convention $x=o+fA$:

1. **User/source $A$:** preserve supplied vector order, handedness,
   fractional/wrapping conventions, original binary64 operands and public
   image coefficients. Mathematical source validity is distinct from backend
   feasibility.
2. **Candidate private computational $B=UA$:** exact unimodular integer $U$,
   with $s_{\mathrm{user}}=s_{\mathrm{reduced}}U$. If selected, this preconditions
   complete exact periodic search; it is neither a canonical/Niggli
   representation nor automatically the Voro++ basis.
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

- **A1 — exact user basis (accepted):** [ADR 0030](../decisions/0030-exact-user-lattice-and-native-admission.md),
  [#155](https://github.com/DeloneCommons/pyvoro2/issues/155) /
  [PR #156](https://github.com/DeloneCommons/pyvoro2/pull/156). Original finite
  binary64 rows/order, exactly nonzero determinant of either sign, exact
  user-coordinate decisions with rounded views and separate native admission.
- **A2 — qualified native frame (accepted):** [ADR 0031](../decisions/0031-qualified-native-periodic-qr-frame.md),
  [#159](https://github.com/DeloneCommons/pyvoro2/issues/159) /
  [PR #160](https://github.com/DeloneCommons/pyvoro2/pull/160). Shared qualified
  QR, signed fully periodic 3D support and reflection-safe reporting. Its
  narrow R1 remap correction resolved seven of the 25 pre-remediation A2
  discrepancy records; neither general remapping nor source/preparation
  integration was accepted by A2.
- **A3-R — shared numerical remapping:** replace duplicated rectangular
  arithmetic and repeated coupled triangular reduction with a small shared
  algorithm; retain full-row translations, well-defined half-open numerical
  views, bounded snapping and final checked shifts. Prepare one frame per
  public composite remap instead of the current three preparations.
- **A3-S — source connectors and realization reference:** evaluate inferred
  and explicit connectors using original source operands; evaluate the complete
  fixed-shift affine expression before rounding. Ship this with the
  necessary same-operation numerical image-label translation in the existing
  matcher, including consistent reverse labels and measure keys. Unavailable
  correspondence must not become a positive/negative match or drive an
  incorrect active-set change. No new inverse model or full contact
  certification is implied.
- **A3-P — stable inserted-point preparation:** remove remote source lifts
  before projection through the accepted QR frame, reuse A1 arithmetic and
  R's remap, and keep final preparation shifts in the original row basis.
  Cover persistent and temporarily inserted ghost generators and the
  corresponding Cartesian remap helpers. Uninserted locate-query handling
  is separate F2 work.

**Design rationale and remaining decisions.** The post-A2 reassessment
(2026-11-27, baseline `5649e1b8`) confirmed the remaining P0 mechanisms and
identified three additional constraints on a proportionate solution:

- A double period-model agreement check is unnecessarily restrictive for
  ordinary rectangular input: for bounds `[-0.1, 0.2]` and x = 0.4,
  the rounded period and exact endpoint difference produce respectively
  `0.09999999999999998` and `0.1`, both contained and with shift 1.
  R should use the existing represented period `float(hi-lo)` alone,
  with explicit containment/refusal rules. Rounded-upper endpoint chart
  selection (including zero epsilon), nonoverlapping snap bands and final
  integer range checks are proposed R policies to approve in its issue.
  A nonoverlapping-band rule can refuse strongly anisotropic cells even
  when the frame passes A2 admission; make that compatibility effect explicit.
  A1's literal exact floors and rounded views remain unchanged.
- For unit-period source sites x = (1/8, 3 + 3/8), the nearest source image
  uses shift -3 whereas the corresponding raw contact has shift 0 after
  preparation. The existing preparation shifts and same-operation native
  anchors suffice for a bounded effective-model label bridge where
  correspondence is established. This is not source-facet certification;
  correcting the resolver alone is unsafe, but full P is not a prerequisite.
  See [connector construction](https://github.com/DeloneCommons/pyvoro2/blob/5649e1b8ac770d809014d7e963f5e05b3e7bd19b/src/pyvoro2/inverse/separator/constraints.py)
  and [realization matching](https://github.com/DeloneCommons/pyvoro2/blob/5649e1b8ac770d809014d7e963f5e05b3e7bd19b/src/pyvoro2/inverse/separator/realize.py).
- On the ordinary unimodular R1 basis with rows `(0,-1,-1)`, `(1,0,0)`,
  `(-1,-1,0)`, origin `(1.25,-0.5,0.125)` and source points at that origin
  plus `(x,1/2,1/2)` for x = 1/4, 1/2, 3/4, an exactly represented
  translation of only the middle generator by $2^{50}$ copies of the first
  row corrupted the geometry. Expected ID-associated standard volumes
  `(3/8,1/4,3/8)` became approximately `(0.265639,0.336407,0.397953)` in
  the native control. Exact reduction of the already rounded native input
  cannot undo the pre-remap loss of local phase. P addresses this mechanism without changing QR or
  requiring B1. Compact integer prototypes showed modest additional
  preparation cost relative to complete compute on two 4,000-point
  workloads; this is bounded research evidence, not a speed promise.
  Measure the actual pipeline before introducing optimization branches.

The native research controls used current Python with a preserved donor
binary and matching native source, not a fresh current build. Future issues
must specify exact reproducers, their mathematical oracles, and the selected
public numerical contract. Fix serious wrong results or justify refusal or
mitigation; neither universal execution of finite inputs nor a new standing
qualification framework is a goal. Preserve existing native safety and
A1/A2 contracts. No algorithm, epsilon-policy change or new public guarantee
is approved by this draft.

### F1-B — deterministic internal computational basis

**B1** first profiles named workloads and consumers of the existing complete
exact search. If the measured cost or resource refusal warrants reduction,
compare bounded alternatives and select a private *good deterministic* exact
rank-3 reducer with exact $B$, $U$, $U^{-1}$ and independent postconditions.
Exact LLL with $\delta=3/4$ remains a candidate, supported by historical
experiments and a rank-three counterexample to simplistic pairwise reduction,
**not an approved or mandatory algorithm**. Current R-L02b remains pending. Historical
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

- **C1 — nearest-image semantics and conditional integration:** establish
  physical Cartesian tie semantics, pair reversal and translation covariance
  in the existing complete search, independently of B1. Integrate reduced
  proof geometry only if a B1 reducer is selected. Preserve source operands
  and orientation authority; materialize int64/float views late. No duplicate
  closest-vector solvers or arbitrary search stencils.
- **C2 — duplicate and insertion safety:** reuse exact geometry for complete
  duplicate detection and distance-only classification; retain the mandatory
  native insertion-safety backstop. Assess actual insertion disposition/chart
  transitions, including a current-build reproduction of the historical
  near-endpoint case, where needed by a selected source-reference consumer.
  This analysis authorizes no native patch or universal insertion-history
  certificate.
- **C3 — occurrence and lifetime contracts:** establish stable local occurrence
  IDs scoped to an immutable snapshot of the fields consumed by the selected
  contact operation. Distinguish actual walls,
  known neighbor images, unknown identities and exhausted budgets; preserve
  raw mutable cell outputs and existing in-place semantics. Separate native
  observations from transformed reporting coordinates. Do not force expensive
  extraction/global assembly into ordinary `compute()` or implement the face
  matcher here.
- **C4 — required analysis and explicit disposition within G1a selection:**
  evaluate the actual G1a consumer's translation-recovery needs, including a
  reusable exact finite-box primitive if relevant, smaller alternatives,
  numerical assumptions, proof completeness, resource policy and independent
  tests. Before F1→G1a admission, record one reviewed disposition: implement in
  F1-C; satisfy the actual consumer through a G1a-specific alternative; or defer
  code with concrete evidence that the selected G1a contract does not require
  it. Reassess additional G1b consumers only if that capability is selected.
  This evaluation cannot be omitted or moved to an unexamined backlog. Full exact compatibility-box search distinguishes
  zero/one/many/incomplete; uniqueness within a supplied box does **not** certify
  the native producer's error bound. No universal epsilon or automatic native
  attribution is approved. **C4 decision is mandatory; C4 code is conditional.**

### Stage reviews and G1 admission

Require a focused **F1-A numerical/source-semantics review**, review B/C
changes if selected, and require **F1→G1a evidence/provenance admission**,
including mandatory C4 disposition. Reviews can reuse evidence and be combined
where they concern the same interfaces; they are not extra monolithic feature
projects. G1a design should also settle migration from the current heuristic
face/edge image attribution, not retain parallel algorithms indefinitely.

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
$2^{-52}$ omitted-source-facet negative control. G1b remains an optional
restricted face-first observed-complex capability, subject to a separate
G1a→G1b admission review and explicit implementation approval. Its intended
scope includes incidence and winding. A correct observed quotient is not
complete source power geometry; source diagnostics remain separate and optional.

F2's weight-first `locate` and `ghost_cells` are valuable public API
improvements but do not mathematically define the global complex. The existing
R-G01a ghost-batch defect on ordinary coordinates is a high-priority independent
correctness candidate; reproduce it with a fresh current native build before
choosing a minimal isolation fix. Ghost-ID initialization, query weights and
metadata remain separately scoped unless a shared correction requires them.

Uninserted locate queries must not inherit generator snapping or duplicate
insertion checks. Scope safe native stepping/index admission first; a concrete
unsafe dispatch path requires correction or justified refusal. Remote-query
recovery and owner-position transport are conditional on a separate useful
contract, not part of inserted-point A3-P. Preserve ordinary no-snap seam
behavior, missing-owner sentinels, ID-only calls and nonperiodic outside-query
semantics. A3 closure must state any open query limitations. Unrelated F2 work
should advance without waiting for G1a/G1b.

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
- Require independent frame/source and composition evidence for A3; validate
  search and insertion-chart claims actually selected for B/C and F1→G1a.
  Native safety remains mandatory; mathematical lattice validity is not native
  admission. Reducer comparison, resource/failure policy and any fallback need
  their own approval if selected. Reuse existing required CI and relevant
  evidence; no new permanent qualification or timing gate is implied.
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

1. **G0/P0 completed; A1/A2/R/S accepted:** preserve their research and decision
   records, including the narrow R1 exception and unmodified P0 replay.
2. Review A3-P's implemented inserted-point stabilization under #166 and
   Proposed ADR 0034, then complete the focused F1-A composition audit using
   accepted A1/A2/R/S evidence. Do not claim all F2 queries are qualified.
3. Independently prioritize F2 ghost-state correctness and the bounded
   locate-query safety decision, and reproduce and address confirmed I0 defects.
   Scope further F2 and I1 API work by useful contracts.
4. Choose concrete G1a consumers and prerequisites. Decide B1 against named
   search workloads; C1 physical ties do not require a reducer. Select
   necessary C2/C3 evidence and explicitly dispose of C4 before G1a admission.
   Preserve the separate G1b polygon/transport/attachment review and I2's
   dependency on G1a and validated inverse state, not G1b.
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
| 2026-11-27 | Draft revision 0.3: record P0/A1/A2 completion, propose bounded R/S/P work and independent F2 query safety, condition reducer integration on demonstrated need, and scope C2–C4 around the selected G1a contract. Existing ADRs, safety requirements and issue-level approval remain in force; no release scope selected. |
