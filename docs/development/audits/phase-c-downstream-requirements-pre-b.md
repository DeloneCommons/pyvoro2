# Pre-Checkpoint-B Phase C downstream-requirements review

- **Status:** Evidence note; not an accepted Phase C contract
- **Review scope:** separator-inverse downstream requirements before Phase C
- **Historical source baseline:** `e3745a81a912d191206a0066f534a55b9727d223` on `dev`
- **Recorded in repository:** 2026-09-24
- **Execution status of the original review:** source/contract inspection only; no build or test suite was executed
- **Related tracker:** [#47 — Complete v0.9.0 functional and API stabilization](https://github.com/DeloneCommons/pyvoro2/issues/47)
- **Current governing contract until a later amendment is accepted:** [ADR 0019](../decisions/0019-separator-measurement-spaces-and-supported-realization.md) and the [active v0.9 plan](../plans/v0.9.md)

This note preserves the factual content and conclusions of an independent
downstream-requirements review that was performed before the requested
post-Checkpoint-B boundary existed. Its purpose is to prevent the ChemVoro
requirements, source observations, and identified contract questions from
remaining only in a private conversation.

It is deliberately **not** an ADR and does not itself authorize a public API
change. The original review concluded that a bounded row-wise extension of
existing separator terms was probably required for the declared downstream
readiness objective, but it also concluded that the final source-grounded
decision had to be refreshed after Checkpoint B. Until that revalidation is
accepted, ADR 0019's current heterogeneity boundary remains authoritative.

## Executive finding

The inspected pre-B Phase C target was not quite sufficient for the stated
ChemVoro programme. The principal gap was not a new inverse family, new
topological constraint system, or chemistry-specific abstraction. It was the
inability of the existing scalar hard restrictions and scalar penalty
parameters to express heterogeneous fixed separator problems without
workarounds.

The review therefore recommended reconsidering, before WP10 implementation,
a **bounded row-wise parameterization of existing separator terms** together
with the hard-applicability, row binding/projection, and report semantics
needed to make those parameters safe.

Three conclusions were already supported by the inspected source:

1. a bounded heterogeneous extension was motivated by concrete downstream
   scenarios;
2. the extension could preserve the existing convex fixed-separator
   optimization family rather than introduce a new observation family; and
3. merely broadcasting arrays through the scalar implementation would be
   unsafe because row association, subsetting, coupling, feasibility, and
   reporting also have to change coherently.

Three integration conclusions were explicitly left for the post-B review:

1. whether the completed periodic contract provides sufficiently complete and
   certified final face/image identity for the intended downstream inspection;
2. whether periodic structural failures propagate through the inverse facade
   with the right semantics; and
3. whether forward and inverse final geometry remain representation-consistent
   under the completed boundary contracts.

The resulting repository decision gate is therefore: **after Checkpoint B and
before WP10 is prepared or implemented, revalidate Phase C against the
completed periodic source and these downstream requirements.**

## Downstream scenario audit preserved from the early review

"Covered" in this table means covered by the inspected implementation or the
then-planned WP1-WP11 contract; it does not mean that later Phase B/C work had
already passed implementation review.

| Scenario | Existing/planned basis | Remaining requirement or qualification | Primary responsibility |
|---|---|---|---|
| U1 — Chemistry-informed partition | separator observations, fixed fit, planned realization-aware workflow, final geometry | heterogeneous bounds/penalties and safe model propagation | pyvoro2 contract; target generation downstream |
| U2 — Comparison baselines | public standard/power forward computation | qualify against the same scientific source and backend-effective representation | existing pyvoro2 acceptance obligation |
| U3 — Covalent connectivity | realized faces and generic geometry | chemistry-specific classification/validation | downstream |
| U4 — Intermolecular contacts | certified owner/image identity and final cells | fragment lookup and contact criteria | downstream |
| U5 — Partition properties | cell/boundary measures | domain-dependent interpretation | primarily downstream |
| U6 — Custom target model | resolved generic separator observations | no additional primitive beyond the bounded numerical policy identified here | downstream target model + upstream fit |
| U7 — Direct separator targets | resolver, explicit shifts, fixed fit, realization diagnostics | do not interpret "best realizable" as a proved global topology optimum | contract clarification |
| U8 — Different regimes together | row targets and confidence | row bounds, hard applicability, penalty strengths, preferred intervals | pyvoro2 |
| U9 — Generated plus supplied information | duplicate/distinct observation rows and provenance | heterogeneous restrictions; distinguish active-conditional from unconditional hard information | shared policy/contract boundary |
| U10 — Annotations as priors | target/model construction | explicit downstream policy | downstream |
| U11 — Annotations as reference only | stable source/site identity | keep reference data out of model construction when not used as a prior | downstream |
| U12 — Inspect one pair | observation IDs, candidate diagnostics, shifts, residuals | join resolved model policy; distinguish image-level faces from pair summaries | upstream model record + downstream view |
| U13 — Inspect one atom | final cells, boundaries, empty-cell information | convenience aggregation and ID mapping | downstream |
| U14 — Unexpected faces | complete final cells and current unaccounted-pair diagnostics | current summaries are not a complete unexpected-image table | geometry upstream; traversal downstream |
| U15 — Self-image faces | periodic boundary contract | retain self-images in full topology; they are not fit-able ordinary observation rows | existing upstream responsibility |
| U16 — Controlled model comparison | deterministic inputs and public solves | preserve physical candidate keys even though observation IDs change with target/confidence | downstream experiment design |
| U17 — External-model compatibility | residuals, objective breakdown, conflicts, realization matching | compare models on a common full candidate set, not only their accepted active subsets | mostly downstream; contract clarity |
| U18 — Large datasets | structured fit outcomes, optional final layers, reports | model serialization/error propagation; output flags must not suppress geometry required internally | upstream integration |
| U19 — Graph/table export | IDs, shifts, geometry, records | pbcgraph/table adapters | downstream |
| U20 — Chemical feedback | repeated independently parameterized solves | no generic additional mechanism demonstrated | downstream |
| O1 — Disconnected observation graphs | connectivity, gauge, and L2 machinery | relative component offsets remain conventions/priors unless identified by data | existing diagnostics + downstream scientific policy |
| O2 — Empty cells and finite molecular domains | empty-cell output and explicit domains | do not assume every atom has positive cell measure or that clipped measure is intrinsic | downstream interpretation |
| O3 — General periodicity masks | supported domain families | an arbitrary atomic-system cell/mask is not automatically a supported pyvoro2 domain | adapter validation / possible later geometry work |

The audit found no evidence that ChemVoro required, for v0.9, generic mixed
observation blocks, topology constraints, chemistry classes, prescribed cell
measures, site motion, or another inverse family.

## Source-level corrections that matter to downstream interpretation

### Full final topology is the authority, not current pair summaries

The inspected realization diagnostics were useful but not a complete table of
all realized image-qualified boundaries:

- `unaccounted_pairs` was defined at unordered site-pair level;
- self-pairs were excluded;
- extra images of a pair were not reported as unaccounted merely because that
  unordered pair already had an observation;
- `realized_other_shift` indicated that some other image realized while the
  requested image did not;
- a requested image and an additional image could therefore both realize while
  `realized_other_shift` remained false;
- the selected `boundary_measure` was not a substitute for traversing all
  realized image-qualified faces.

The early recommendation was to preserve those summary meanings rather than
silently redefine them. The post-B review should instead verify that the
supported realization-aware result can expose complete final cells with every
certified image boundary, including self-images, when that layer is requested.
A separate "all-face table" could be an additive convenience later; it was not
identified as a candidate-API requirement.

### Observation row identity is not a permanent physical candidate key

ADR 0014 makes target and confidence part of observation/source identity.
Changing either can therefore change a row ID even when the same physical
candidate pair is being reconsidered. Subsetting preserves retained row IDs;
changing the observation set does not.

ChemVoro therefore needs its own downstream physical candidate key based on
stable site identity, orientation/image convention, and where necessary
occurrence/provenance. The early review did **not** recommend a new pyvoro2
identity class for this purpose.

### Realization-aware fitting is not a proved global topology optimizer

The inspected outer algorithm repeatedly fit an active subset, tessellated, and
updated that subset. Its hard restrictions and penalties applied to the rows
currently being fit. It also began with all rows active by default; an
infeasible initial hard system was not automatically repaired by searching for
a different topology.

Consequently, a hard restriction configured for a candidate row should not be
silently interpreted as an unconditional promise that the row must remain in
the final realized topology. If ChemVoro later requires unconditional
algebraic restrictions while other rows are independently realization-pruned,
that would be a distinct requirement and a stop condition for the bounded
design described here.

### Some chemical expectations remain policy, not mathematical guarantees

For disconnected observation components, independent component weight offsets
may preserve internal observed differences while changing competition between
components in the complete diagram. Existing gauge/L2 diagnostics matter, but
a default relative fragment offset is not chemical information.

Likewise, finite-domain cell measures require an explicit domain policy and
empty cells must remain visible. Neither observation justifies adding
prescribed measures or guaranteed-nonempty-cell constraints to Phase C.

## Row-wise mathematical finding

For each oriented separator row, every existing separator coordinate is an
affine function of the corresponding fitted weight difference. The fixed
problem therefore remains in the same mathematical family when selected scalar
parameters are replaced by valid row-wise values:

- hard intervals/equalities remain affine constraints;
- nonnegative row penalty strengths preserve convexity for the existing convex
  penalty families;
- heterogeneous parameters do not by themselves introduce another observation
  family or nonconvex topology objective.

The hard part is consequently not the existence of arrays. It is preserving the
semantic relationship between row parameters, observations, compiled affine
terms, solver selection, subsetting, final active state, and strict reports.

## Parameter-level findings to reconsider after Checkpoint B

The early review intentionally did **not** recommend "every numerical parameter
accepts arrays." It distinguished parameters with direct downstream evidence
from parameters that could remain scalar.

| Term / parameter | Early review finding |
|---|---|
| `SquaredLoss` | no new row parameter needed; observation confidence already supplies row-wise mismatch importance |
| `HuberLoss.delta` | mathematically useful for heterogeneous robust scales, but not established as a v0.9 gate |
| `Interval.lower/upper` | essential candidate for row-wise support |
| hard applicability | essential; some rows need no hard restriction at all |
| `FixedValue.value` | essential candidate where selected rows carry fixed information |
| `SoftIntervalPenalty.lower/upper` | essential for row-dependent preferred ranges |
| `SoftIntervalPenalty.strength` | essential for row-dependent soft weighting; zero strength must be true absence before dangerous evaluation |
| `ExponentialBoundaryPenalty.lower/upper/strength` | useful where repulsion follows row-dependent preferred/admissible ranges |
| exponential `margin/tau` | possible later extension; not established as a v0.9 gate |
| `ReciprocalBoundaryPenalty.lower/upper/strength` | same motivation as exponential boundary penalties |
| reciprocal `margin/epsilon` | possible later extension; not established as a v0.9 gate |
| `L2Regularization.strength` | a vector would be site-wise rather than observation-wise and was not justified by these scenarios |
| `L2Regularization.reference` | already site-vector-valued; no new row semantics |
| observation `confidence` | already row-wise and should retain its existing identity/zero-confidence contract |

Confidence is not a substitute for every other row-wise parameter. In
particular, confidence does not remove a hard equality, and changing a Huber
threshold changes the ratio between quadratic curvature and tail slope rather
than simply multiplying the loss.

## Hard applicability and mixed equality/bound semantics

The inspected implementation already compiled scalar `Interval` or
`FixedValue` terms into row arrays and then into difference-constraint
machinery. Row-dependent endpoints were therefore structurally natural.

The missing semantic requirement was **absence**. If one candidate row has no
hard restriction, that row must contribute:

- no feasibility edge/equality;
- no hard-tolerance classification;
- no hard-conflict witness; and
- no hard-induced coupling.

Zero observation confidence cannot stand in for this because confidence affects
the mismatch contribution, not the independent hard system.

The early review also noted that merely vectorizing `FixedValue` and a
strictly positive-width `Interval` would not represent a model in which some
rows are fixed, some bounded, and some unrestricted. One possible bounded
contract was to admit degenerate closed intervals (`lower == upper`) with the
existing fixed-value numerical acceptance semantics plus an explicit
applicability mask. That was a recommendation for later contract review, not an
accepted API choice.

One-sided public infinite bounds were **not** recommended. Existing
constructors and strict records use finite model parameters, and the presence
of internal unbounded solver paths was not considered sufficient evidence that
`+/-inf` sentinels would be safe for validation, exact arithmetic, witnesses,
serialization, and reports.

## Integration consequences of row-wise policy

If the post-B review accepts any row-wise extension, it must cover at least the
following integration points rather than only constructor signatures.

### Row-local structural coupling

The inspected coupling logic could promote every row when any hard term or
positive penalty was present. Heterogeneous terms require the structural
coupling mask to reflect which rows actually have applicable hard restrictions
or positive-strength penalties.

An all-masked hard term or all-zero penalty must not by itself force a stronger
solver path or connect graph components.

### Compilation and solver certificates

Current scalar compilation contains exact/scalar parameter assumptions and
shared breakpoint/certificate structures. Row-dependent endpoints or strengths
must be compiled row-wise without weakening:

- exact/source affine arithmetic;
- zero-strength-before-evaluation rules;
- boundary activation semantics;
- reciprocal tangent-continuation semantics;
- scalar/quadratic certificates; or
- established direct-versus-ADMM applicability rules.

This is why "replace `float(...)` with arrays" was explicitly rejected as an
adequate implementation strategy.

### Projection through every row subset

The fixed solver may decompose observations into connected components, and the
realization-aware engine may repeatedly select active subsets. Scalar model
objects are safe to reuse across such subsets; raw unsliced row arrays are not.

A resolved/bound problem therefore has to project row parameters through
exactly the same ordered row selection as the observations. Re-entry, repeated
subsetting, duplicate physical pairs, and multiple periodic shifts of one pair
must preserve the correct policy association.

### Full candidate policy versus current active membership

Configured model policy and realization-active membership are separate masks.
The full candidate policy must remain inspectable even after a row leaves the
current active fit. A hard applicability mask must not be reinterpreted as
"this face must exist" or "this row must remain active."

### Model policy versus observation/source identity

The early review recommended preserving the current identity boundary:

- bounds, applicability, penalty parameters, model spaces, and regularization
  belong to model/problem identity;
- they should not alter observation row IDs or observation-set/source
  fingerprints;
- observation confidence remains the existing special case that already belongs
  to observation identity.

A separate model fingerprint was considered optional. If one is eventually
added, it must cover resolved parameters and row association rather than
replace exact association checks.

## Provisional input/report semantics identified for later decision

These details were recommended by the early review because they make a bounded
row-wise design coherent, but they remain **provisional until the post-B gate
accepts or replaces them**:

- accepted row parameters use scalar-or-exact-length-`m` semantics;
- a scalar broadcasts to all observation rows;
- a vector is positional in the ordered observation set to which the problem is
  bound;
- accidental matrices/column vectors and wrong lengths are rejected;
- term measurement space remains term-global (`fraction`, `position`, or
  inherited `None`), not per-row;
- a reusable `FitModel` is policy, while a constructed problem owns the
  resolved association between that policy and one ordered observation set;
- accepted arrays are copied into owned read-only storage;
- row selection projects both observations and resolved model policy;
- hard applicability has explicit Boolean absence semantics rather than large
  finite bounds;
- strict finite validation remains in force even for masked/zero-strength rows;
- zero strength and all-masked hard terms are mathematically absent before
  numerical evaluation;
- result/report structures should expose a resolved model description
  sufficient to recover ordered row association, effective spaces, applicable
  hard restrictions, penalty families/parameters, and L2 semantics;
- absent declared bounds should be represented as absence/null rather than
  infinity;
- realization-aware results should retain full-candidate policy and identify
  the exact active-row projection used by the nested final fit.

The early review did **not** require per-row objective contributions or hard
slacks as part of this minimal surface.

## Recommended Phase C structure if the gate accepts the bounded extension

The historical review suggested that, if accepted, row-wise policy should be
implemented together with measurement-space compilation rather than as a
second compiler rewrite after WP10.

A possible execution split was:

1. **fixed-problem integration:** independent model spaces plus the accepted
   bounded row parameters, validation/ownership, row binding, affine
   compilation, feasibility, row-local coupling, component projection,
   objective/proximal/direct/ADMM integration, result views, and strict report
   migration;
2. **realization-refinement propagation:** candidate-to-active projection,
   drop/re-entry, final refit, gauge/component analysis, full-candidate policy,
   and atomic accepted-state association;
3. **supported facade qualification:** expose the stabilized model through the
   planned WP11 supported realization-aware facade without promoting advanced
   path/research controls.

This decomposition is evidence for the post-B planning discussion, not an
already accepted rename or split of WP10.

## Mandatory post-Checkpoint-B revalidation

After Checkpoint B is accepted and before the first WP10 implementation issue is
prepared, the reviewer must re-read the completed periodic source, tests,
relevant ADRs, current API inventory, and this evidence note.

The review must decide at least:

1. **Need:** do the completed ChemVoro-shaped workflows still require bounded
   row-wise parameters before the v0.9 candidate surface is frozen?
2. **Whitelist:** if yes, which specific existing term parameters are required
   now versus safe to defer to the v0.9.x soak or later mixed-inverse work?
3. **Hard semantics:** is explicit per-row hard applicability required, and how
   are equality/bounded/unrestricted rows represented without public nonfinite
   sentinels?
4. **Binding:** where is ordered row association owned, and how is it preserved
   through component and realization-active subsets?
5. **Active-conditional meaning:** do hard restrictions remain conditional on
   current active membership, or has downstream evidence established a genuine
   requirement for unconditional restrictions across independently pruned rows?
6. **Identity/reporting:** what model description and report-schema change is
   required without changing observation/source identity?
7. **Periodic integration:** do final cells provide complete certified
   owner/image/self-image information needed by the downstream inspection
   workflows?
8. **Failure propagation:** are periodic structural/resource failures carried
   through fixed and realization-aware inverse results without being hidden or
   misclassified?
9. **Representation consistency:** do forward and inverse geometry checks use
   the same accepted source and backend-effective representation semantics?
10. **Scope boundary:** can the requirement still be satisfied without generic
    mixed observation blocks, per-row callables/loss dispatch, topology
    constraints, prescribed measures, chemistry-specific types, site motion, or
    a new inverse family?

### Possible outcomes

The gate may:

- **retain current ADR 0019 unchanged** if post-B evidence shows that the
  scalar model is sufficient for v0.9;
- **accept a bounded Phase C amendment** and update the necessary ADRs/plan/API
  inventory before WP10 implementation; or
- **stop for a broader maintainer decision** if the real downstream requirement
  crosses the bounded separator-stabilization boundary.

No implementation should quietly start accepting row arrays before this gate is
closed.

## Documentation/decision surfaces if a bounded amendment is accepted

The early review identified the likely authority set that would need coherent
changes:

- **ADR 0019:** replace the blanket exclusion of per-row strengths with the
  accepted whitelist; define hard applicability, row binding/projection, and
  model reporting while keeping spaces term-global;
- **ADR 0007:** extend zero-term, coupling, and hard-feasibility rules row-wise
  without changing the underlying source objective;
- **ADR 0014:** clarify that new model parameters do not enter observation
  identity; retain confidence's existing identity role;
- **ADR 0015:** require accepted-state model association to match the active
  observation projection as well as source/weight state;
- **ADR 0017:** at most clarify that a bounded existing-separator extension is
  stabilization rather than the deferred generic mixed-observation
  architecture;
- **active plan and #47:** revise Phase C dependencies, acceptance criteria, and
  non-goals;
- **API inventory:** record accepted shapes, ownership/binding, bounds views,
  report schema, active projection, and lifecycle status;
- **user/migration documentation:** explain hard absence, equality/bounded
  rows, model policy versus observation identity, and summary-versus-full
  topology semantics.

A new ADR was not considered necessary if a tightly bounded dated amendment to
ADR 0019 is sufficient.

## Independent tests/oracles required if the extension is accepted

The early review proposed the following evidence classes. They remain useful as
a checklist for the post-B specification but are not tests required by this
evidence-note change.

### Independent affine/unit-space checks

Use small exact-coordinate systems with different connector lengths. Assemble
term affine coefficients independently from production compilation. Exercise
fraction observations with position-space mismatch, fraction hard intervals,
and penalties in both spaces.

### Hard-feasibility examples

Use rational cycle examples with known consistency/inconsistency. Cover masked
rows, mixed equality/interval rows, duplicate observations, reversed rows, and
periodic parallel rows. Verify that masked restrictions never appear in
witnesses.

### Small convex optimization oracles

For small squared-loss interval/soft-interval systems, enumerate relevant
active regions and solve independent linear/KKT systems. Add Huber
quadratic/tail transitions and independent checks of exponential/reciprocal
activation and continuation behavior where those families are in the accepted
whitelist.

### Row-association properties

Permute observations and row parameters together; physical solutions should be
equivalent where uniquely identified. Exercise noncontiguous subsets, repeated
subsetting, duplicate pairs, multiple shifts, active drop/re-entry, and final
refits.

### Absence/zero behavior

A zero-strength row must match omission even when evaluating the inactive term
would overflow. An all-masked hard term must match no hard restriction and must
not alone alter coupling or solver eligibility. Small positive supported
strengths must not be silently treated as zero.

### Connectivity/gauge checks

Selectively connect disconnected informative components with one hard
restriction or one positive penalty row. Verify structural coupling, data
identification, and gauge treatment separately.

### Identity, ownership, and strict reports

Verify scalar/constant-vector mathematical equivalence, mutation isolation,
serialization, source/observation identity preservation under model-policy
changes, and correct distinction between full candidate policy and active-fit
policy.

### Post-B geometry integration

Exercise at least one periodic case with:

- the requested image and an additional image both realized;
- a self-image face;
- an ordinary candidate absent from final topology;
- real walls; and
- an empty cell.

Inspect complete final cells rather than relying only on pair summaries.

## Scope explicitly excluded by the early review

The following were intentionally outside the proposed Phase C revision:

- arbitrary per-row callables, arbitrary loss objects, or a model DSL;
- per-row measurement spaces or generic mixed observation blocks;
- chemistry classes, annotation-policy parsing, bond orders, or QTAIM-specific
  APIs;
- "this face must exist" topology constraints or unconditional constraints
  across independently pruned rows without new evidence;
- prescribed cell measures or general nonempty-cell constraints;
- site-wise regularization strengths/anchors;
- chemistry-loop convergence/cycle machinery;
- warm-start, reusable-problem, trajectory, or batch architecture added without
  performance evidence;
- generalized I/O, symmetry reconstruction, or graph-export convenience;
- a new inverse family.

Row-wise Huber thresholds and boundary shape scales were classified as deferred
capabilities, not rejected mathematics.

## Residual risks and open questions

The original review could not evaluate the then-future completed Phase B
implementation. The post-B pass must therefore treat periodic completeness,
image identity, structural failure semantics, and forward/inverse representation
consistency as fresh source questions rather than inherit an early assumption.

The meaning of externally "fixed" information also remains a decision boundary.
The bounded design described here treats hard restrictions as
active-conditional. If ChemVoro instead requires some restrictions to remain
algebraically unconditional while other candidate rows are realization-pruned,
that is a distinct capability and must not be hidden behind a Boolean hard mask.

Heterogeneous endpoints and strengths may also create substantially broader
numerical scales and breakpoint diversity. The review recommended preserving
current ordinary fast paths and certification boundaries rather than inventing a
new performance architecture pre-emptively.

Finally, finite-state arguments for downstream chemistry loops depend on the
complete iteration state. Class labels alone need not define that state when
continuous weights, relaxation, or history affect the next step. That belongs
to ChemVoro research methodology rather than pyvoro2's solver API.

## Authority boundary

This note records evidence and a required future question. It does not amend
ADR 0019, API signatures, report schemas, WP10 decomposition, or the v1.2 mixed
inverse roadmap by itself.

The active plan now requires the post-Checkpoint-B revalidation to occur before
WP10. If that gate accepts a bounded extension, the accepted contract must be
moved into the appropriate ADRs, plan sections, API inventory, and
implementation issue before production code relies on it.
