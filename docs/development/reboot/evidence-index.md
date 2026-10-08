# Historical evidence index

This index preserves evidence for the [requirements inventory](requirements-inventory.md).
It does not adopt the historical implementation. Under
[ADR 0027](../decisions/0027-v0.9-reboot-from-v0.8.md), every historical addition
starts **unreviewed for reboot adoption**. The [overview](index.md) records exact
checkpoint acceptances: A and B were accepted; C remained pending. Accepted
WP10/WP11 and later remediation do not complete C, pre-Phase-D work, WP12/WP13,
API freeze, or release qualification.

“Independent oracle” below means expected mathematics is constructed separately
from the production algorithm. A regression protects a known behavior; parity
can share a backend defect; a native witness identifies an execution; artifact
qualification establishes bounded build/runtime properties. None substitutes
for the others. This reconstruction inspected source, discussions and supplied
artifact bytes. **No historical build, native test, attached script, qualification,
or integrated checkpoint review was rerun.**

<a id="source-corpus"></a>

## Source corpus and review chains

All historical source links pin
`b5e1aa6f53cd5edd94c0ae376047cb53278d2158`, protected by
`archive/v0.9-attempt-2026-10-08`. Baseline links explicitly pin published v0.8.0
`db0884c641de0998d190de8aeee1d45154e46aff`. An archived “Active,” “Accepted,”
“Preferred,” or mandatory future gate is a **frozen historical label**, not
current authority. Final archive tests sometimes incorporate later repairs;
they are not necessarily byte-identical to an earlier checkpoint's tests.

| Preserved source | What it establishes; limits |
|---|---|
| [Historical AGENTS][h-agents], [v0.9 plan][h-plan], [API inventory][h-api], [API lifecycle][h-lifecycle], [roadmap][h-roadmap] | Historical authority, execution/lifecycle ledger, boundaries and deferred work. Plan activation was [issue #46][i46]; [#47][i47] is the execution/evidence tracker. |
| [ADR 0018][adr18] | User lattice, backend frame, physical image ties and boundary semantics. |
| [ADR 0019][adr19] | Measurement/model spaces, row policy, supported realization workflow and parameter dispositions. |
| [ADR 0020][adr20] | Exact private rank-three reduction, proof obligations and resource accounting. |
| [ADR 0021][adr21] | WP5 native occurrence attribution and separate exact E/S audits. |
| [ADR 0022][adr22] | WP6 planar source provenance, exact contacts and refusal boundaries. |
| [ADR 0023][adr23] | WP7 initialized ghost route, stored-ghost chart and N+S boundary eligibility. |
| [ADR 0024][adr24] | External native-artifact qualification; later source-identity amendment is included. |
| [ADR 0025][adr25] | Numerical occurrence normalization and bounded proof-assisted planar identities. |
| [ADR 0026][adr26] | Final returned-state certification, schema v3 and exact diagnostic availability. |
| [WP5 implementation][wp5], [native witness derivation][witness], [WP7 implementation][wp7], [WP8 implementation][wp8] | Source-operation, image-envelope and consumer derivations; evidence obligations rather than independent reruns. |
| [Qualification workflow][qualification], [issue88 implementation][issue88] | Build/installation evidence and preservation instructions, separate from geometric truth. |
| [Pre-B downstream audit][audit-pre], [entry row-policy audit][audit-entry], [shape-parameter audit][audit-shape] | Earlier motivation, accepted A+B gate, then parameter-specific dispositions. The early audit alone authorized no implementation. |
| [Spatial dispositions][spatial-dispositions], [architecture][architecture], [post-functional intake][backlog] | Preserved counterexamples and open hypotheses. Incidence-first construction, backend policy and PA-001/002/003 remain unresolved choices. |

The chains below distinguish **issues** from **PRs**. Their linked bodies,
available comments and source reconciliation were inspected; an empty review
timeline supplies no missing verdict. Later explicit acceptance controls over
stale candidate-handoff wording.

| Work and issue chain | PR chain and decisive context |
|---|---|
| Baseline foundations [#36][i36], [#37][i37], [#38][i38], [#39][i39], [#40][i40], [#41][i41] | Native preconditions, strict ownership, exact image geometry and preparation already existed at v0.8; see baseline ADRs [0010][base10], [0011][base11], [0012][base12], [0013][base13]. |
| A: [#49][i49], [#54][i54], [#56][i56], [#58][i58] | WP1 [#50][p50]/portability [#51][p51]; WP2 [#55][p55]; WP3 [#57][p57]; WP4 [#59][p59]. WP3 supplies reduction; WP4 integrates consumers and physical ties. |
| A blockers [#60][i60], [#62][i62], [#65][i65]; separate [#64][i64] | Repairs [#61][p61], [#63][p63], [#66][p66]; [A acceptance][accept-a] precedes cleanup [#67][p67]. #64 allocation-exhaustion ownership stayed open. |
| WP5 [#68][i68]; WP6 [#74][i74]; WP7 [#77][i77]; WP8 [#79][i79] | WP5 prerequisite [#69][p69], G0 [#70][p70], action split [#71][p71], implementation [#72][p72]; then [#76][p76], [#78][p78], [#80][p80]. |
| Performance [#82][i82]/[#84][i84]; removals [#85][i85]; qualification [#88][i88] | [#87][p87], [#86][p86], [#90][p90]; [#91][p91] records open audit intake, not new runtime policy. |
| B composition [#92][i92]; characterization [#94][i94], contract [#95][i95], atlas [#96][i96], ADR [#97][i97], planar [#98][i98], spatial [#99][i99] | [#93][p93] repairs integration; [#100][p100] records open architecture; [#101][p101] ADR25; [#102][p102] planar; [#103][p103] spatial. [Final B acceptance][accept-b] is separate. [#105][p105] documents a remaining weak-view diagnostic limitation. |
| C entry [#104][i104]; WP10 [#107][i107]; source identity [#109][i109]; WP11 [#111][i111]; infrastructure [#113][i113] | [#106][p106], [#108][p108], [#110][p110], [#112][p112], [#114][p114]; [#115][p115] accepts parameter decisions with optional C implementation **NONE**. |
| Final-state [#116][i116]; exact range [#118][i118] | [#117][p117] accepted remediation; [#119][p119] first rejected then accepted at a replacement head. Exact identities and supplied evidence are in [E17](#e17). |

<a id="geometry-evidence"></a>

## Geometry, provenance and execution evidence

Unless stated otherwise, the linked sources/fixtures are preserved in the
archive. Pure arithmetic can be reconstructed from their inputs; public native
tests additionally require the historical package and applicable qualified
artifact. Preservation does not establish that they run unchanged on v0.8.

<a id="e01"></a>

### E01 — Weight families, parity and an analytic query

[Weight-first query tests][weight-tests] cover 2D boxes/rectangular periodicity
and 3D boxes/orthorhombic/general periodic cells. In
`test_ghost_weights_match_one_combined_gauge_radius_oracle`, independently
concatenating persistent and ghost weights before minimum subtraction and
square root checks the conversion policy. The resulting geometry still uses
the same backend. `test_ghost_common_weight_shift_preserves_geometry` and
`test_ghost_independent_weight_shift_changes_geometry` distinguish a global
gauge from changing competition between roles.

An independent physical owner is supplied by [locate tests][locate-tests],
`test_locate_equivalent_cubic_bases_have_unique_physical_owner`: generators
`(-3/16,-3/16,3/8)` and `(1/16,3/16,5/16)`, query `(0,0,1/8)`, IDs 101/202.
Standard scores are `34/256,19/256`; weights `(3/64,0)` give `22/256,19/256`.
Owner 202 follows directly, across exact cubic basis representations.
WP1 query-weight APIs and `ghost_radius` removal were additions; baseline
compute already supported weights. Parity alone did not detect E05's defect.

<a id="e02"></a>

### E02 — Exact user wrapping and a separate backend frame

[User-coordinate tests][user-tests] use independent Fraction Gaussian
elimination, rather than production adjugates. Exact nonsingularity includes
both determinant signs and cancellation to determinant ±1 at scale `2^27`.
`test_exact_floor_is_not_chosen_from_a_rounded_fractional_view`,
`test_wrap_fractional_preserves_exact_upper_endpoint_remainders`, and
`test_wrap_shift_int64_boundaries_are_exact` protect exact discrete decisions:
an interior remainder may round to 1 without changing its integer shift.
These are 3D coordinate oracles, not native-admission claims.

[Frame tests][frame-tests] check positive diagonal, parity and scale-aware
validation; public/snapshot agreement is consistency. [Reflection tests][reflection-tests],
especially `test_transport_helper_has_native_independent_analytic_oracle`,
independently specify coordinate transport and one cycle reversal for improper
Q, including faces-only and empty power outputs. QR transport remains floating
and validated, not exact S geometry. [Visualization tests][viz-tests] preserve
the integer shift `2^53+1`; premature float conversion produces x=2 instead of
x=1. Baseline lacked these coordinate methods and rejected left-handed cells.

<a id="e03"></a>

### E03 — Reduction, complete CVP and declared-box translation

[Reduction tests][lll-tests] independently check Gram minors, determinant,
unimodular inverses and a decreasing positive integral potential. Anchors are
`test_three_vector_cancellation_does_not_stall_at_pairwise_half_ties`,
`test_strict_swaps_follow_an_independent_positive_integer_potential_trace`,
and `test_256_seed_policy_oracle_and_diagnostic_maxima`. The historical policy
is rank-three exact LLL, delta 3/4, half ties toward zero, strict Lovasz swaps;
resource caps are not termination proofs or validity criteria.

[Reduced geometry tests][reduced-tests] combine analytic cubic/known-inverse
answers with an independent Fraction/cofactor CVP oracle. For reduced basis B,
`q=d B^-1`, incumbent radius R and inverse-column bound Cj, every candidate
satisfies `ceil(-qj-R*Cj) <= sj <= floor(-qj+R*Cj)`. Physical exact ties are
ordered by Cartesian displacement; user shift mapping precedes int64 checks.
Huge `2^80` shears are proof-only; native smoke uses modest shears.
[Periodic-image tests][image-tests] prove finite oracle-family completeness;
the baseline test helper's interior-cube winner check alone did not do so.
Baseline **production** CVP was already certified.

[Native-translation tests][translation-tests], including
`test_general_basis_counts_match_complete_independent_cofactor_oracle`,
independently enumerate a declared exact closed Cartesian box and distinguish
zero, one and multiple shifts. Scope is full 3D and signed/partial diagonal
2D/3D. This validates the consumer, not the producer's error enclosure.

<a id="e04"></a>

### E04 — Source-coordinate and coupled-remap counterexamples

[Source-geometry tests][source-tests],
`test_nearest_image_preserves_source_endpoint_below_backend_snap`, retain
`pi=(2^-41,1/4,1/4)`, `pj=(1/2,1/4,1/4)`. The unique source displacement is
`0x1.fffffffffe000p-2`; the alternative image has squared-distance gap
`2^-40`. Backend snapping created a different tied problem. Explicit/mixed
rows and 2D/3D partial periodicity extend the association regressions.

[Remap tests][remap-tests],
`test_coupled_upper_snap_preserves_cartesian_tangential_residual`, use basis
rows `(1,0,0),(-1/2,1,0),(0,0,1)` and point `(3/16,-2^-55,3/8)`.
Coupled y translation exposes x=`19/16`; the old one-sided snap erased
residual `3/16`. Zero-epsilon canonicalization is a separate backend-remap
regression, not permission to clamp exact user wrapping. [#60 acceptance][accept60]
and [A acceptance][accept-a] preserve review conclusions. Baseline source
contains these problematic paths; no fresh baseline failure was executed here.

<a id="e05"></a>

### E05 — Independent ghosts and initialized native IDs

[Ghost-cell tests][ghost-cells], `test_periodic_ghost_batch_analytic_volume`,
use persistent `(1/8,1/8,1/8)` and separate queries with x=`3/8,7/8`.
Standard/equal-weight volumes are exactly 1/2; persistent weight `-1/8` and
ghost weight `1/4` make each volume 1. Earlier second-query results were
approximately 0.8854167 or 0.6927083. [#62][i62] attributes this to retained
periodic images after deleting the primary ghost. Singleton/batch parity is
secondary; the analytic volume is decisive. A SciPy halfspace/ConvexHull
singleton oracle in the same file is independent but floating and optional.

The later [WP7 C++ reference][ghost-reference] is distinct safety evidence:
defined poisoned storage demonstrates temporary-ID read/copy on four legacy
3D periodic routes, with a nonperiodic negative control. Sanitizer success
alone cannot establish absence of uninitialized reads. The selected route
initializes checked internal ID n in a fresh augmented container per query,
retaining one batch-wide gauge. Source-reference parity is not a mathematical
cell oracle. Open [#64][i64] concerns allocation-exhaustion ownership, not the
fixed ghost-state contamination.

<a id="e06"></a>

### E06 — WP5 independent ideals and observed native operations

The [WP5 derivation][wp5] distinguishes **N**, actual native supports and
occurrences; **E**, exact geometry of stored operands and exact squares of
actual binary64 radii received by native code; and **S**, exact source geometry
with original mathematical weights or exact squares of supplied radii.
Ghost S has a specific exception: persistent sites stay original, but its
temporary site is the actual stored/materialized anchor g; see [ADR23][adr23]
and [E08](#e08).
Exactifying a rounded `r*r` defines neither the intended E nor S.
Unique N attribution does not prove complete S coverage or E=S.

[Ideal tests][wp5-ideal-tests] use independent rational Cramer intersections.
`test_weighted_triclinic_cells_match_independent_exact_supercell_oracle`
proves its `[-3,3]^3` family complete from self slabs and inverse bounds;
the two cells have 20/32 vertices. Rectangular/nonperiodic tests cover all
eight masks, power, walls, hidden/lower-dimensional cells and coincident cuts.
[Transition fixtures][wp5-fixture] and [characterization tests][wp5-character-tests]
give a prism cut `X+Y<=3/2+delta`: positive area squared `8*delta^2` for
negative delta, a line at zero, absent above zero. Fifteen ideal/input cases
and one historical projected native cycle are labeled separately.

[Native-witness tests][native-witness-tests],
`test_power_plane_offset_operation_order_is_observable`, show equal radii
`2^27` give different source-associated offsets in two native routines.
Shared strict arithmetic policy across producer/observer translation units
and actual occurrence tokens are execution obligations. Bit/topology parity,
primitive rounding-bin tests and candidate-budget refusals complement the
independent ideal; they do not make N an exact mesh.

The [decimal-cross regression][wp5-certificate-tests],
`test_decimal_cross_tiny_exact_facet_is_never_erased_by_area_tolerance`, uses
`(.1,.5,.5),(.9,.5,.5),(.5,.1,.5),(.5,.9,.5)` with x/y periodicity.
It stores a positive rational area-squared expectation and checks missing
native coverage. This exercises production semantic construction rather than
a second oracle. Native omission is conditional on the observed packet, so it
does not freeze one compiler's marginal topology or erase positivity by tolerance.

<a id="e07"></a>

### E07 — Planar interval oracle and source provenance

[wp6_interval_oracle.py][planar-oracle] converts each support line into an
independent rational feasibility interval. Its floor-centered four-image
family per periodic axis contains the production centered-three family.
[test_wp6_ideal.py][planar-ideal-tests],
`test_all_masks_against_independent_larger_interval_family`, compares contacts,
ranks and vertices for all four rectangular masks, three sites and two weight
families. Identical, empty, lower-dimensional and exact supplied-radius cases
have separate expectations. This is not an oblique 2D theorem.

[ADR22][adr22] and [native tests][planar-native-tests] connect all seven
ordinary cuts to actual storage slots and outgoing-edge tokens, including
growth/deletion. A coincident no-op preserves prior provenance. Complete E/S
contact classification distinguishes positive, point and absent contacts;
rounded endpoint equality cannot establish private collapse. Native profile
refusal applies independently of whether periodic metadata is requested.
The source oracle is preserved; native provenance needs the corresponding
historical artifact and is absent from the v0.8 implementation.

<a id="e08"></a>

### E08 — Complete ghost oracle and missing microfacets

[wp7_rational_oracle.py][ghost-oracle] supplies independent 2D line intervals,
3D plane intersections, affine ranks, areas/volumes and coincident-facet
groups. Rectangular domination and triclinic self-polytope/inverse bounds
prove complete image families. It imports no production geometry. Bounded
candidate/intersection budgets refuse rather than return a prefix.
[Public oracle tests][ghost-tests] cover all rectangular masks, full triclinic
3D, standard/power, hidden/lower-dimensional ghosts and batch-wide gauges.

[Adversarial tests][ghost-adversarial],
`test_triclinic_stored_roundtrip_has_exact_microfacets_missing_from_n`, use
the shear lattice `((1,1,0),(0,1,0),(0,0,1))` and its left-handed companion.
The actual stored ghost introduces positive microfacets below `2^-40`:
S has eight facets/four generator images; N covers six/two. Geometry-only
volume remains approximately 1/2; boundary certification refuses missing
positive facets. The invocation's stored ghost, not the original query, is
the S anchor; original persistent sites, public lattice/walls and original
weights remain its other inputs.
`test_observed_spatial_public_vertices_all_round_to_one_positive_site`
preserves eight private vertices and six positive references although every
public vertex rounds to one point. These refute tolerance-based deletion.

<a id="e09"></a>

### E09 — Planar degeneracy and proof-assisted normalization

[Planar normalization tests][planar-norm-tests] promote 36 characterization
cases. `test_square_preserves_all_native_occurrences` expects four vertices,
**12 raw global edges and all 20 native occurrences**, while exact S has
eight positive edges. The earlier shorthand “4V/8E/4F normalized target” is
superseded by [ADR25][adr25]. Exact Pythagorean five-/six-way center controls
prove 45/105 pair identities, not every whole-case slot.

`test_distinct_semantic_endpoints_with_colliding_public_coordinates` uses
x translation `2^40` and unequal radii to refute coordinate-only merging.
Artifact exemption requires exact private collapse and complete owning E/S
audits; identity additionally requires singleton E/S contact and exact chart
transport. [Context][planar-context] and [proof composition][normalization-proof]
bind actual consumed snapshots and test all alternative lift paths.
Mutation fails; copied/serialized results retain numerical fields but lose
live proof authority. A standalone square conservatively retains 11V/17E
and strict refusal. These are analytic facts plus proof-lifetime regressions,
not a generic independent N-to-S solver. [PR #102][p102] also repaired
evidence capture/EOF receipt ordering; that tooling repair changed no geometry.

<a id="e10"></a>

### E10 — Spatial atlas, missing positivity, collisions and winding

[Independent degeneracy oracle][deg-oracle] uses rational clipping, exact cap
hulls, complete image bounds and quotient geometry under one common allowed
translation. Independently wrapping each vertex would destroy edge winding.
The [33-case input corpus][deg-data] covers 16 archetypes and seven saved
strict-pass raw/S count disagreements. It extracts useful inputs/oracle from
the external #96 atlas; the full campaign and provenance graph are not tracked.

[test_degeneracy_scope.py][spatial-tests] preserves distinct failures:

| Test | Mathematical fact and limit |
|---|---|
| `test_tiny_exact_positive_facet_loss_remains_wp5_coverage_failure` | `spatial_bipyramid5__radial_out_2m36` has independently proved positive contact area omitted by N. Normalization cannot create it. |
| `test_equal_public_triples_do_not_identify_exact_semantic_vertices` | Perturbed quarter/three-quarter cube: three distinct exact vertex pairs share float triples; S counts 26/64/46/8. Strict raw success does not establish S completeness. |
| `test_partial_periodic_ridge_retains_two_distinct_quotient_endpoints` | A zero-area ridge from `(1/2,1/2,0)` to `(1/2,1/2,1)` retains two vertices when z is nonperiodic. It is exact contact evidence, not an observed zero-area native face. |
| `test_successful_power_five_audit_is_not_e_s_vertex_bijection` | E counts 29/59/35/5 and S 28/58/35/5 despite successful audit; dyadic-radius control restores E=S. |

The finite atlas found no retained lower-dimensional N face among 5,634 packet
and 304 pilot faces; this is not impossibility. The external atlas recorded
twelve strict failures, with dispositions preserved in the [regression
catalogue][spatial-dispositions]; only the representative bipyramid failure
was retained as a maintained regression in the 33-case subset. The oracle
assumes full-dimensional cells for this corpus. Synthetic
fragment controls are labeled synthetic. [PR #100][p100]'s incidence-first
proposal and direct semantic construction remain open, with no spatial
projection engine adopted.

<a id="e11"></a>

### E11 — Metadata and output ownership

[WP8 metadata tests][wp8-tests] use independent Fraction Gauss–Jordan wrapping.
`test_native_owner_float_is_preserved_while_exact_image_is_exposed` keeps
native owner position 0 while the exact image is `-2^-53`. The [matrix][wp8-matrix]
covers 2D/3D masks, IDs, powers and exact cubic shears. This proves wrapping
and specified image identity, not nearest-owner correctness.

[Checkpoint-B integration tests][b-integration],
`test_periodic_self_identity_composes`, independently specify one-/two-site
slab quotient counts and image-qualified self incidences. Swapping lifts can
evade bare-ID-set checks. [#92][i92]/[PR #93][p93] also fixed absent shifts
being invented as zero and raw-output stripping mutating shared normalized
dictionaries. Ownership/mutation tests are regressions, not exact geometry.
[API tests][api-tests] retain immediate removed-keyword rejection from WP9;
restoring that removal would change the baseline API. [PR #105][p105] leaves
a documented malformed weak-3D-view diagnostic limitation; it does not claim
a runtime repair or invalidate supported normalized views.

<a id="e12"></a>

### E12 — Native qualification and evidence provenance

[ADR24][adr24], [qualification][qualification] and
[test_native_qualification.py][qualification-tests] bind source/consumer/schema
closure, actual effective compile/link commands and consumed dependencies,
arithmetic discriminators, raw guards, repaired final bytes, detached records
and installation anchor. Requested flags or native self-metadata are not
sufficient evidence. Separate WP5, WP6, WP7-spatial/planar and WP8-spatial/planar
components do not inherit admission merely by sharing a module.

The [tracked predecessor corpus][qualification-fixtures] preserves a
10,151,270-byte ZIP containing 94 original raw input/packet files; the index
selects 92 ordinary WP6 computations. Strict arithmetic discriminator bits
`4013fffffb000000` versus fused `4013fffffb000001` establish operation
behavior, not geometric truth. Runtime guards preserve caller state while
checking the executing thread and callback boundaries. Sanitizers remain
separate safety evidence.

Historical positives included actual repaired GNU manylinux artifacts and
retained Apple/MSVC WP5/WP8 routes; other component/adaptor combinations could
refuse. Linux aarch64 was not qualified. A refusal-only wheel smoke cannot
replace positive installed-artifact evidence.

[#109][i109]/[PR #110][p110] later replaced procedural source approval with
implementer-owned canonical source identity; candidate technical qualification
precedes final independent review. Filesystem omission and special ZIP-member
blockers were repaired. [#113][i113]/[PR #114][p114] subsequently hardened
evidence/CI infrastructure. Neither amendment retroactively changes the B
acceptance source. Availability, import instrumentation and integration cost
questions remain open in [PA-001/002/003][backlog]. Baseline did not contain
this qualification architecture; no fresh artifact admission is claimed here.

<a id="e13"></a>

### E13 — Workload formulas and bounded performance measurements

[Independent lattice workload helper][lll-workload] uses Fraction/cofactor
formulas with no production reduction/geometry imports. The thin-case
source-basis formula counts 10,692,900 candidates; WP4's actual reduced consumer
test counts 3,270. Intrinsic anisotropy remains real work. Formula estimates
and actual consumer instrumentation in [reduced tests][reduced-tests] are
separate evidence classes; neither is timing or universal native support.

[#82 investigation][perf-investigation] located the interpreter slowdown in
ordinary WP5 Fraction clipping, not the initially suspected ghost oracle.
3.10 performed 4,103,028 modular inverses for that many hashes; 3.13 performed
26,651. Cache-disabled replay supplied a causal control. [#84][i84]/[PR #87][p87]
reduced hash requests to 298,878 without changing arithmetic/order/budgets;
[clipping tests][clipping-tests] preserve invariants, while 207 historical
differential cases establish implementation parity. Same-host full-suite
samples were 1002.89→834.10 seconds on 3.10 and 496.26→499.38 on 3.13.
Single samples and interrupted investigation runs are not general speed
guarantees. The maintained Python floor was retained. This optimization is
conditional on adopting the exact-audit kernel, which v0.8 lacks.

<a id="separator-evidence"></a>

## Separator, final-state and diagnostic evidence

<a id="e14"></a>

### E14 — Spaces, bound row policy and supported realization

For `z=wi-wj`, the affine separator coordinates are

$$
f(z)=\frac12+\frac{z}{2d^2},\qquad
p(z)=\frac d2+\frac{z}{2d}.
$$

Stored binary64 `distance` and `distance2` are separate operands. Source
measurement/target/confidence identity stays separate from mismatch, hard and
penalty spaces; a rounded `p=d*f` is not an exact computational identity.
[Problem][separator-problem], [policy][separator-policy] and [ADR19][adr19]
retain this distinction. The normal RHS uses complete
`c*alpha_model*(target_model-beta_model)`, not reconstructed `rho*z_obs`.

| Preserved test | Independent fact or regression; scope |
|---|---|
| [test_row_fit.py][row-fit] `test_mixed_space_hand_computable_anchor` | 2D, d=2, source fraction 1/4; position squared loss plus fraction interval/soft penalty. On the lower active branch `p < 3/4`, independent arithmetic gives `J(p)=(p-.5)^2/2+8*(.375-p/2)^2`, minimized at p=7/10 with z=-6/5 and J=1/40; explicit ADMM. |
| [test_row_prox.py][row-prox] `test_policy_complete_prox_key_separates_the_analytic_strength_anchor` | Rational minima 3/8 versus 3/10 distinguish strengths 1 and 4. Separate nondyadic scale 3/7 and offset -1/11 controls; batching sentries are structural regressions. |
| [test_row_oracles.py][row-oracles] `test_unequal_distance_mixed_prox_matches_original_unit_decimal_minimum` | Independent 80-digit Decimal derivative bisection, d=sqrt(2),2; 24 combinations of source/mismatch spaces, three penalties and squared/Huber losses. |
| [test_row_active.py][row-active] `test_active_drop_reentry_and_final_refit_use_original_candidate_policy` | Controlled schedule verifies projection/re-entry association; separate native 2D/3D image cases exercise real geometry. |
| [test_row_reports.py][row-reports] `test_v2_complete_policy_wrappers_and_source_identity_are_exact` | Historical test name; archive asserts schema v3. Literal grammar reconstruction plus independently computed Huber objective. |
| [test_facade.py][facade-tests] `test_native_mixed_space_independent_quadratic_oracle` | Genuine 2D/3D box and periodic workflows; final-refit/cycle/limit and failure-propagation regressions supplement the small independent objective. |

Historical A+B accepted exact-length row values and Boolean hard applicability, with
owned binding projected alongside observations. Zero confidence removes only
mismatch; zero strength is absent before dangerous evaluation. PR115 keeps
all five shape parameters scalar: Huber delta, exponential tau and reciprocal
margin deferred; exponential margin and reciprocal epsilon retained
term-global. The outer realization loop has no proved global topology optimum.
A common mathematical weight gauge preserves S; separate component offsets
can alter cross-component competition. Full image-qualified cells, including self-images,
remain more informative than pair summaries. These extensions/facade/report
changes are absent from the scalar, same-space v0.8 baseline.

<a id="e15"></a>

### E15 — F1–F4: certify the actual returned state

[#116][i116]/[PR #117][p117], [ADR26][adr26] and the archived
[solver][separator-solver]/[problem][separator-problem] move certification to
the actual returned binary64 vector and preserve separately owned diagnostics.

| Finding and precise maintained test | Decisive witness; evidence type |
|---|---|
| F1: [test_final_separator_state.py][finalstate-tests] `test_final_reference_shift_cannot_return_false_hard_success` | 2D x=0,1,3, `FixedValue(.3)`, zero L2, reference `[2^20,2^20,0]`, confidence 0/1. A reference shift can lose a nonrepresentable contrast; independent Fraction hard evaluation forbids false success. `.125` supplies a representable success control. |
| F1 row/builder guards: same file `test_final_guard_uses_each_rows_tolerance`, `test_builder_rejects_every_false_success_claim` | Per-row tolerances, either success claim, and canonicalization modes. Comparing maximum violation with an unrelated maximum tolerance is insufficient. Native facade has a separate test. |
| F2: [test_numerical_availability.py][availability-tests] `test_dyadic_source_encodings_keep_model_and_source_owners_separate` | d=.5, source fraction .75 versus position .375, c=8: model RHS `[1,-1]` and matrix `[[8,-8],[-8,8]]` agree while source coefficients/IDs differ. Source curvature overflow need not invalidate finite model work. |
| F3: [test_complete_affine_diagnostics.py][affine-tests] `test_native_active_cancellation_uses_complete_affine_residual_everywhere` | d=16, position target 8, c=`2^112`, fraction mismatch, L2=`2^95`: rounded prediction equals target while complete residual is nonzero. Independent Fraction original-expression checks cover real native active/history vectors. |
| F4: [final-state tests][finalstate-tests] `test_no_work_components_use_exact_reference_entries`, `test_isolated_reference_survives_hard_dispatch`, `test_empty_native_solution_preserves_supplied_reference` | Squared/Huber, zero/positive L2, isolated and empty n=0/1/3 cases. Exact supplied reference entries survive no-work branches; component conventions do not invent data. |

Reconstruction/corruption and copy/provenance controls are contract regressions.
[Report v3 source][separator-report] allows only enumerated diagnostic nulls
with producer-owned reasons and exhaustive JSON Pointers; inputs, objective,
hard controls, successful weights and geometry remain strict. History stores
summary/iteration bindings, not old weight snapshots, so discarded historical
vectors cannot be independently recertified from the result alone.

<a id="e16"></a>

### E16 — Exact range: B1/B2 and the later R1/R2

Let M be the largest finite binary64 value. Availability is decided from the
**exact complete expression**, not whether rounded evaluation happens to
return M. [Diagnostic owners][separator-diagnostics] use ordinary exponent
bounds and exceptional Fraction arithmetic. Available aggregate rounding uses
private high-precision arithmetic; this is not a universal correctly rounded
binary64-value theorem.

Public witnesses are 2D. `test_native_active_range_propagation_and_provenance`
in [the B1/B2 tests][exact-range-tests] separately covers actual planar selected
and marginal paths, final-state association and history.

| Finding / maintained test | Exact distinction and coverage |
|---|---|
| B1: [test_exact_diagnostic_range.py][exact-range-tests] `test_public_weighted_range_uses_original_confidence` | Residuals `[M,0]`, confidence `[2,0]`: weighted RMSE=M, L2 unavailable. `[M]` with `nextafter(1,+inf)` confidence makes both unavailable although rounded sqrt(confidence)=1. Denominator is row count. |
| B2: same file `test_public_row_and_max_cannot_round_away_range_exit` | Exact residual `M+2^969+1/2` is out of range despite rounding to M; four-row RMS can remain finite. `M^2+1>M^2` distinguishes weighted L2 from finite control `[M,0]`. |
| R1: [test_source_edge_diagnostic_range.py][source-range-tests] `test_source_fitted_difference_range_precedes_rounding`, `test_source_observation_difference_range_precedes_rounding`, `test_source_curvature_range_uses_original_confidence` | At d=1, `[M,-1]` gives unavailable fitted difference M+1 and algebraic residual `-M-1`, but finite source-position residual `(M+1)/2`; target -M with alpha=1,beta=.5 gives unavailable observed difference `-M-.5`. Exact original-confidence curvature can exceed M while rounded products equal M. |
| R1 recovery: same file `test_both_unavailable_edge_leaves_can_have_finite_residual` | Both difference leaves may be unavailable while complete algebraic residual is .5. Range follows `(target-beta)/alpha-wi+wj`, not rounded intermediates. |
| R2: same file `test_strict_diagnostic_cancellation_reaches_exact_dispatch_first` | d=`0x1.6a09e667f3bd2p-513`, equal weights `0x1.0000000000008p+0`: alpha*w exceeds M, but `.5+alpha*w-alpha*w=.5` and residual=0. Exact dispatch must precede unsafe products under NumPy warn/raise. |

The R1 curvature witness uses d=`0x1.199999999999ap-200`,
c=`0x1.76cf41f212d78p+226`, source alpha=`0x1.a723f789854a0p+398`;
position-model curvature and the zero optimum remain finite. Adjacent
confidence values provide controls.

B1/B2's 84 maintained cases did not catch R1/R2. The rejected review also
reproduced them on the predecessor, so they were surviving defects, not shown
new regressions. Final R1/R2 tests add 96 cases, signed/subnormal/adjacent
controls, product-prefix checks and a 576-configuration sweep per FP policy.
A 1,000-row sentry protects ordinary vectorization; it is not a benchmark.

Earlier [algebraic aggregate][algebraic-range-tests] and
[affine RMS][affine-rms-tests] regressions preserve 59-row RMSE/105-row MAE
boundaries and finite aggregates despite unavailable rows. Fraction threshold
oracles and 1,000-digit Decimal finite norms are independent of producer
rounding. Conditioned hard predictions and soft-penalty values retain their
existing semantics; unavailable descriptive diagnostics do not become control
values. None of this constitutes fresh Checkpoint-C acceptance.

<a id="e17"></a>

### E17 — Supplied attachments and preservation gaps

Five local copies were supplied. Numeric prefixes below identify those copies;
the remaining filename is the original artifact name. ZIP listings, manifests
and selected text/source bytes were inspected without installation or script
execution. These are **not publicly hosted repository artifacts**. The hashes
identify the supplied bytes; they do not imply authenticated remote checksums
or a durable download URL.

| Copy / exact filename | Source association, contents and role |
|---|---|
| 01 — `pyvoro2-review-kit-v3.12-pr117-7356157b-cp313-linux_x86_64.zip` | 117 members; source ZIP, manifest, cp313 Linux wheel, wheelhouse, build/CI/qualification logs and JUnit. Construction kit for accepted PR117 head `7356157b0b2955bae5c1d431690c8e2f2acbab63`, tree `a33b20057122627a7c1373348efc3615eebb0ceb`; merged `d2785d36f00320422afbccd53eb05e530450de47`. Historical full result 6059 passed/13 skipped. |
| 02 — `pyvoro2-review-kit-v3.13-pr119-84ad1a98-cp313-linux_x86_64.zip` | 109 members; corresponding construction kit for **rejected** head `84ad1a984aa5d86fbc0417a5f385fbd5f645fbb8`, tree `e90b44dd54b102b641308bdc67dcdb657943ade2`. Historical full result 6143/13 and green CI did not remove independent remaining defects. |
| 03 — `pyvoro2-pr119-reproduce-remaining-84ad1a98.py` | 7,525 bytes; independent Fraction public R1/R2 reproducer, 22 case/policy combinations, `--assert-fixed`. Inspected, never run here; byte-identical to 04's embedded script. |
| 04 — `pyvoro2-pr119-review-evidence-84ad1a98.zip` | 1,296 members, no wheels. Independent **CHANGES REQUIRED** HTML review, remediation prompt, reproducer, inventories/checksums, numerical/identity/schema evidence and selected construction records. Historical installed-current and predecessor JSON each report 12 failed checks; review execution is distinct from kit construction. |
| 05 — `pyvoro2-review-kit-v3.14-pr119-863d56e2-cp313-linux_x86_64.zip` | 118 members; final construction kit for **accepted replacement** `863d56e27e3a03e2cf85f375b7889af3aad408b6`, tree `31afef7a8c043bb0bc56765c798ad18516549f1d`. Historical full result 6239/13. Squash merge/archive b5e1aa6 has the identical tree. This kit is not the independent final-review report. |

| Copy | Bytes | Computed local SHA-256 |
|---|---:|---|
| 01 | 124054587 | `92f7eca02970688b0d63fad0a4d6a85289c70b21e62a54970ce724589fe5b040` |
| 02 | 123999135 | `8cb6ef3f27dbe349b283247eb467340e725fa9a1adff0b866e7f82bbcb89de52` |
| 03 | 7525 | `1ccdbc991a1790e8aa734950a41d3920ae5475430fc621229ec9355516dc0be4` |
| 04 | 12999052 | `e1e3c88b2d0d9d6b8bd30c5fcce595d8f81fe578ebbb927e518621dd4f476ad9` |
| 05 | 124035401 | `43c094d8da5135cf18a5ea7cdef7479f4eb504ba399b76fb007441afa563ee67` |

Nested source ZIP hashes for 02 and 05 were independently recomputed and
matched their manifests. Seven final source/test/ADR files matched archived
Git bytes. Both old and new wheels use version filename `pyvoro2-0.8.0`;
that filename alone identifies neither source nor qualification.

[Issue #116's closure][i116] preserves PR117 acceptance; [#118's final comment][accept118]
preserves PR119 replacement acceptance and explicitly leaves C pending.
That comment reports a final-review limitation: local fresh-build qualification
refused loader-evidence byte changes, so the review used the admitted
current-source kit route together with full CI. It does not establish a
successful fresh reviewer build under every loader.
Standalone **accepted final PR117 and PR119 review reports were not among the
five copies**. The rejected review records unavailable earlier C materials.
Do not substitute its rejected-head verdict for final acceptance, or claim
that final acceptance preserves an unavailable independent report.

Other named external evidence was not supplied, downloaded or reverified here:

| Artifact name | Owning evidence record / preserved subset |
|---|---|
| `pyvoro2-WP5-G0N-characterization-e2e520e(1).zip` | [WP5 / #68][i68]; compact transition fixtures in [E06](#e06). |
| `wp6-prerequisite-a98ac0d-20260924.zip` | [WP6 / #74][i74]; later committed raw input/packet subset in [E12](#e12). |
| `issue82-performance-2c33705094fe.zip` | [Investigation record][perf-investigation]; subsequent optimization is separate. |
| `pyvoro2-issue88-native-qualification-032c0d5-20260928.zip` | [#88][i88] and [implementation notes][issue88]; partial raw corpus is tracked. |
| `checkpoint-b-degeneracy-contract-ba9e58f.md` | [#95 contract record][i95]; ADR 0025 preserves the resulting bounded contract. |
| `CHECKPOINT_B_INTEGRATED_FINAL_REVIEW_351f53e.md`, `CHECKPOINT_B_INTEGRATED_REVIEW_EVIDENCE_351f53e.zip` | [Final B decision][accept-b]; public acceptance is accessible, the full bundles were not examined here. |

The [#94][i94]/[#96][i96] full characterization/atlas bundles likewise exceed
the committed extracts.
No archived full A review or standalone #62 native probe was located.
Public acceptance comments, exact input/oracle extracts, and inaccessible
execution bundles remain separately identified evidence.

GitHub Actions artifacts are retention-limited. At this reconstruction cutoff,
[final PR119 run metadata][final119-artifacts] listed `release-distributions`
as unexpired but scheduled to expire on 2027-01-05 UTC; no artifact bytes were
downloaded for that check. A successful run link does not promise permanent
artifact or log availability. The protected Git tag preserves tracked sources,
not untracked review kits or expiring Actions evidence.

<!-- Pinned sources: archived labels are historical, not current authority. -->

[wp5-certificate-tests]: https://github.com/DeloneCommons/pyvoro2/blob/b5e1aa6f53cd5edd94c0ae376047cb53278d2158/tests/forward/spatial/test_wp5_certificate.py
[final119-artifacts]: https://api.github.com/repos/DeloneCommons/pyvoro2/actions/runs/37683587444/artifacts?per_page=1
[accept-a]: https://github.com/DeloneCommons/pyvoro2/issues/47#issuecomment-5745444670
[accept-b]: https://github.com/DeloneCommons/pyvoro2/issues/95#issuecomment-5969300168
[accept118]: https://github.com/DeloneCommons/pyvoro2/issues/118#issuecomment-6049560737
[accept60]: https://github.com/DeloneCommons/pyvoro2/issues/60#issuecomment-5734452104
[adr18]: https://github.com/DeloneCommons/pyvoro2/blob/b5e1aa6f53cd5edd94c0ae376047cb53278d2158/docs/development/decisions/0018-periodic-user-lattice-and-boundary-semantics.md
[adr19]: https://github.com/DeloneCommons/pyvoro2/blob/b5e1aa6f53cd5edd94c0ae376047cb53278d2158/docs/development/decisions/0019-separator-measurement-spaces-and-supported-realization.md
[adr20]: https://github.com/DeloneCommons/pyvoro2/blob/b5e1aa6f53cd5edd94c0ae376047cb53278d2158/docs/development/decisions/0020-exact-private-lattice-reduction.md
[adr21]: https://github.com/DeloneCommons/pyvoro2/blob/b5e1aa6f53cd5edd94c0ae376047cb53278d2158/docs/development/decisions/0021-wp5-native-occurrence-and-exact-face-certification.md
[adr22]: https://github.com/DeloneCommons/pyvoro2/blob/b5e1aa6f53cd5edd94c0ae376047cb53278d2158/docs/development/decisions/0022-wp6-source-certified-planar-edge-provenance.md
[adr23]: https://github.com/DeloneCommons/pyvoro2/blob/b5e1aa6f53cd5edd94c0ae376047cb53278d2158/docs/development/decisions/0023-wp7-certified-ghost-boundaries.md
[adr24]: https://github.com/DeloneCommons/pyvoro2/blob/b5e1aa6f53cd5edd94c0ae376047cb53278d2158/docs/development/decisions/0024-external-native-artifact-qualification.md
[adr25]: https://github.com/DeloneCommons/pyvoro2/blob/b5e1aa6f53cd5edd94c0ae376047cb53278d2158/docs/development/decisions/0025-native-occurrence-normalization-and-proof-assisted-identities.md
[adr26]: https://github.com/DeloneCommons/pyvoro2/blob/b5e1aa6f53cd5edd94c0ae376047cb53278d2158/docs/development/decisions/0026-separator-final-state-and-diagnostic-availability.md
[affine-rms-tests]: https://github.com/DeloneCommons/pyvoro2/blob/b5e1aa6f53cd5edd94c0ae376047cb53278d2158/tests/inverse/separator/test_affine_rms_range_boundary.py
[affine-tests]: https://github.com/DeloneCommons/pyvoro2/blob/b5e1aa6f53cd5edd94c0ae376047cb53278d2158/tests/inverse/separator/test_complete_affine_diagnostics.py
[algebraic-range-tests]: https://github.com/DeloneCommons/pyvoro2/blob/b5e1aa6f53cd5edd94c0ae376047cb53278d2158/tests/inverse/separator/test_algebraic_aggregate_range_boundary.py
[api-tests]: https://github.com/DeloneCommons/pyvoro2/blob/b5e1aa6f53cd5edd94c0ae376047cb53278d2158/tests/forward/common/test_forward_api_contract.py
[architecture]: https://github.com/DeloneCommons/pyvoro2/blob/b5e1aa6f53cd5edd94c0ae376047cb53278d2158/docs/development/architecture.md
[audit-entry]: https://github.com/DeloneCommons/pyvoro2/blob/b5e1aa6f53cd5edd94c0ae376047cb53278d2158/docs/development/audits/phase-c-entry-row-policy-review.md
[audit-pre]: https://github.com/DeloneCommons/pyvoro2/blob/b5e1aa6f53cd5edd94c0ae376047cb53278d2158/docs/development/audits/phase-c-downstream-requirements-pre-b.md
[audit-shape]: https://github.com/DeloneCommons/pyvoro2/blob/b5e1aa6f53cd5edd94c0ae376047cb53278d2158/docs/development/audits/phase-c-row-wise-shape-refinement-review.md
[availability-tests]: https://github.com/DeloneCommons/pyvoro2/blob/b5e1aa6f53cd5edd94c0ae376047cb53278d2158/tests/inverse/separator/test_numerical_availability.py
[b-integration]: https://github.com/DeloneCommons/pyvoro2/blob/b5e1aa6f53cd5edd94c0ae376047cb53278d2158/tests/integration/test_checkpoint_b_periodic_integration.py
[backlog]: https://github.com/DeloneCommons/pyvoro2/blob/b5e1aa6f53cd5edd94c0ae376047cb53278d2158/docs/development/review-notes/v0.9-post-functional-audit-backlog.md
[base10]: https://github.com/DeloneCommons/pyvoro2/blob/db0884c641de0998d190de8aeee1d45154e46aff/docs/development/decisions/0010-native-construction-preconditions.md
[base11]: https://github.com/DeloneCommons/pyvoro2/blob/db0884c641de0998d190de8aeee1d45154e46aff/docs/development/decisions/0011-strict-input-and-ownership-contract.md
[base12]: https://github.com/DeloneCommons/pyvoro2/blob/db0884c641de0998d190de8aeee1d45154e46aff/docs/development/decisions/0012-certified-periodic-image-geometry.md
[base13]: https://github.com/DeloneCommons/pyvoro2/blob/db0884c641de0998d190de8aeee1d45154e46aff/docs/development/decisions/0013-central-generator-preparation-and-backend-safety.md
[clipping-tests]: https://github.com/DeloneCommons/pyvoro2/blob/b5e1aa6f53cd5edd94c0ae376047cb53278d2158/tests/forward/spatial/test_wp5_clipping.py
[deg-data]: https://github.com/DeloneCommons/pyvoro2/blob/b5e1aa6f53cd5edd94c0ae376047cb53278d2158/tests/forward/spatial/data/degeneracy_scope.json
[deg-oracle]: https://github.com/DeloneCommons/pyvoro2/blob/b5e1aa6f53cd5edd94c0ae376047cb53278d2158/tests/forward/spatial/_degeneracy_oracle.py
[exact-range-tests]: https://github.com/DeloneCommons/pyvoro2/blob/b5e1aa6f53cd5edd94c0ae376047cb53278d2158/tests/inverse/separator/test_exact_diagnostic_range.py
[facade-tests]: https://github.com/DeloneCommons/pyvoro2/blob/b5e1aa6f53cd5edd94c0ae376047cb53278d2158/tests/inverse/separator/test_facade.py
[finalstate-tests]: https://github.com/DeloneCommons/pyvoro2/blob/b5e1aa6f53cd5edd94c0ae376047cb53278d2158/tests/inverse/separator/test_final_separator_state.py
[frame-tests]: https://github.com/DeloneCommons/pyvoro2/blob/b5e1aa6f53cd5edd94c0ae376047cb53278d2158/tests/forward/spatial/test_periodic_backend_frame.py
[ghost-adversarial]: https://github.com/DeloneCommons/pyvoro2/blob/b5e1aa6f53cd5edd94c0ae376047cb53278d2158/tests/forward/test_wp7_adversarial.py
[ghost-cells]: https://github.com/DeloneCommons/pyvoro2/blob/b5e1aa6f53cd5edd94c0ae376047cb53278d2158/tests/forward/spatial/test_ghost_cells.py
[ghost-oracle]: https://github.com/DeloneCommons/pyvoro2/blob/b5e1aa6f53cd5edd94c0ae376047cb53278d2158/tests/forward/wp7_rational_oracle.py
[ghost-reference]: https://github.com/DeloneCommons/pyvoro2/blob/b5e1aa6f53cd5edd94c0ae376047cb53278d2158/tests/native/test_wp7_ghost_reference.cpp
[ghost-tests]: https://github.com/DeloneCommons/pyvoro2/blob/b5e1aa6f53cd5edd94c0ae376047cb53278d2158/tests/forward/test_wp7_oracle.py
[h-agents]: https://github.com/DeloneCommons/pyvoro2/blob/b5e1aa6f53cd5edd94c0ae376047cb53278d2158/AGENTS.md
[h-api]: https://github.com/DeloneCommons/pyvoro2/blob/b5e1aa6f53cd5edd94c0ae376047cb53278d2158/docs/development/api-inventory.md
[h-lifecycle]: https://github.com/DeloneCommons/pyvoro2/blob/b5e1aa6f53cd5edd94c0ae376047cb53278d2158/docs/development/api-lifecycle.md
[h-plan]: https://github.com/DeloneCommons/pyvoro2/blob/b5e1aa6f53cd5edd94c0ae376047cb53278d2158/docs/development/plans/v0.9.md
[h-roadmap]: https://github.com/DeloneCommons/pyvoro2/blob/b5e1aa6f53cd5edd94c0ae376047cb53278d2158/docs/project/roadmap.md
[i104]: https://github.com/DeloneCommons/pyvoro2/issues/104
[i107]: https://github.com/DeloneCommons/pyvoro2/issues/107
[i109]: https://github.com/DeloneCommons/pyvoro2/issues/109
[i111]: https://github.com/DeloneCommons/pyvoro2/issues/111
[i113]: https://github.com/DeloneCommons/pyvoro2/issues/113
[i116]: https://github.com/DeloneCommons/pyvoro2/issues/116
[i118]: https://github.com/DeloneCommons/pyvoro2/issues/118
[i36]: https://github.com/DeloneCommons/pyvoro2/issues/36
[i37]: https://github.com/DeloneCommons/pyvoro2/issues/37
[i38]: https://github.com/DeloneCommons/pyvoro2/issues/38
[i39]: https://github.com/DeloneCommons/pyvoro2/issues/39
[i40]: https://github.com/DeloneCommons/pyvoro2/issues/40
[i41]: https://github.com/DeloneCommons/pyvoro2/issues/41
[i46]: https://github.com/DeloneCommons/pyvoro2/issues/46
[i47]: https://github.com/DeloneCommons/pyvoro2/issues/47
[i49]: https://github.com/DeloneCommons/pyvoro2/issues/49
[i54]: https://github.com/DeloneCommons/pyvoro2/issues/54
[i56]: https://github.com/DeloneCommons/pyvoro2/issues/56
[i58]: https://github.com/DeloneCommons/pyvoro2/issues/58
[i60]: https://github.com/DeloneCommons/pyvoro2/issues/60
[i62]: https://github.com/DeloneCommons/pyvoro2/issues/62
[i64]: https://github.com/DeloneCommons/pyvoro2/issues/64
[i65]: https://github.com/DeloneCommons/pyvoro2/issues/65
[i68]: https://github.com/DeloneCommons/pyvoro2/issues/68
[i74]: https://github.com/DeloneCommons/pyvoro2/issues/74
[i77]: https://github.com/DeloneCommons/pyvoro2/issues/77
[i79]: https://github.com/DeloneCommons/pyvoro2/issues/79
[i82]: https://github.com/DeloneCommons/pyvoro2/issues/82
[i84]: https://github.com/DeloneCommons/pyvoro2/issues/84
[i85]: https://github.com/DeloneCommons/pyvoro2/issues/85
[i88]: https://github.com/DeloneCommons/pyvoro2/issues/88
[i92]: https://github.com/DeloneCommons/pyvoro2/issues/92
[i94]: https://github.com/DeloneCommons/pyvoro2/issues/94
[i95]: https://github.com/DeloneCommons/pyvoro2/issues/95
[i96]: https://github.com/DeloneCommons/pyvoro2/issues/96
[i97]: https://github.com/DeloneCommons/pyvoro2/issues/97
[i98]: https://github.com/DeloneCommons/pyvoro2/issues/98
[i99]: https://github.com/DeloneCommons/pyvoro2/issues/99
[image-tests]: https://github.com/DeloneCommons/pyvoro2/blob/b5e1aa6f53cd5edd94c0ae376047cb53278d2158/tests/forward/common/test_periodic_images.py
[issue88]: https://github.com/DeloneCommons/pyvoro2/blob/b5e1aa6f53cd5edd94c0ae376047cb53278d2158/docs/development/issue88-implementation.md
[lll-tests]: https://github.com/DeloneCommons/pyvoro2/blob/b5e1aa6f53cd5edd94c0ae376047cb53278d2158/tests/forward/common/test_exact_lattice_reduction.py
[lll-workload]: https://github.com/DeloneCommons/pyvoro2/blob/b5e1aa6f53cd5edd94c0ae376047cb53278d2158/tests/forward/common/_lattice_reduction_workload.py
[locate-tests]: https://github.com/DeloneCommons/pyvoro2/blob/b5e1aa6f53cd5edd94c0ae376047cb53278d2158/tests/forward/spatial/test_locate.py
[native-witness-tests]: https://github.com/DeloneCommons/pyvoro2/blob/b5e1aa6f53cd5edd94c0ae376047cb53278d2158/tests/forward/spatial/test_native_witness.py
[normalization-proof]: https://github.com/DeloneCommons/pyvoro2/blob/b5e1aa6f53cd5edd94c0ae376047cb53278d2158/src/pyvoro2/_internal/normalization_proof.py
[p100]: https://github.com/DeloneCommons/pyvoro2/pull/100
[p101]: https://github.com/DeloneCommons/pyvoro2/pull/101
[p102]: https://github.com/DeloneCommons/pyvoro2/pull/102
[p103]: https://github.com/DeloneCommons/pyvoro2/pull/103
[p105]: https://github.com/DeloneCommons/pyvoro2/pull/105
[p106]: https://github.com/DeloneCommons/pyvoro2/pull/106
[p108]: https://github.com/DeloneCommons/pyvoro2/pull/108
[p110]: https://github.com/DeloneCommons/pyvoro2/pull/110
[p112]: https://github.com/DeloneCommons/pyvoro2/pull/112
[p114]: https://github.com/DeloneCommons/pyvoro2/pull/114
[p115]: https://github.com/DeloneCommons/pyvoro2/pull/115
[p117]: https://github.com/DeloneCommons/pyvoro2/pull/117
[p119]: https://github.com/DeloneCommons/pyvoro2/pull/119
[p50]: https://github.com/DeloneCommons/pyvoro2/pull/50
[p51]: https://github.com/DeloneCommons/pyvoro2/pull/51
[p55]: https://github.com/DeloneCommons/pyvoro2/pull/55
[p57]: https://github.com/DeloneCommons/pyvoro2/pull/57
[p59]: https://github.com/DeloneCommons/pyvoro2/pull/59
[p61]: https://github.com/DeloneCommons/pyvoro2/pull/61
[p63]: https://github.com/DeloneCommons/pyvoro2/pull/63
[p66]: https://github.com/DeloneCommons/pyvoro2/pull/66
[p67]: https://github.com/DeloneCommons/pyvoro2/pull/67
[p69]: https://github.com/DeloneCommons/pyvoro2/pull/69
[p70]: https://github.com/DeloneCommons/pyvoro2/pull/70
[p71]: https://github.com/DeloneCommons/pyvoro2/pull/71
[p72]: https://github.com/DeloneCommons/pyvoro2/pull/72
[p76]: https://github.com/DeloneCommons/pyvoro2/pull/76
[p78]: https://github.com/DeloneCommons/pyvoro2/pull/78
[p80]: https://github.com/DeloneCommons/pyvoro2/pull/80
[p86]: https://github.com/DeloneCommons/pyvoro2/pull/86
[p87]: https://github.com/DeloneCommons/pyvoro2/pull/87
[p90]: https://github.com/DeloneCommons/pyvoro2/pull/90
[p91]: https://github.com/DeloneCommons/pyvoro2/pull/91
[p93]: https://github.com/DeloneCommons/pyvoro2/pull/93
[perf-investigation]: https://github.com/DeloneCommons/pyvoro2/issues/82#issuecomment-5860084447
[planar-context]: https://github.com/DeloneCommons/pyvoro2/blob/b5e1aa6f53cd5edd94c0ae376047cb53278d2158/src/pyvoro2/_internal/planar/normalization_context.py
[planar-ideal-tests]: https://github.com/DeloneCommons/pyvoro2/blob/b5e1aa6f53cd5edd94c0ae376047cb53278d2158/tests/forward/planar/test_wp6_ideal.py
[planar-native-tests]: https://github.com/DeloneCommons/pyvoro2/blob/b5e1aa6f53cd5edd94c0ae376047cb53278d2158/tests/forward/planar/test_wp6_native.py
[planar-norm-tests]: https://github.com/DeloneCommons/pyvoro2/blob/b5e1aa6f53cd5edd94c0ae376047cb53278d2158/tests/forward/planar/test_wp6_normalization.py
[planar-oracle]: https://github.com/DeloneCommons/pyvoro2/blob/b5e1aa6f53cd5edd94c0ae376047cb53278d2158/tests/forward/planar/wp6_interval_oracle.py
[qualification]: https://github.com/DeloneCommons/pyvoro2/blob/b5e1aa6f53cd5edd94c0ae376047cb53278d2158/docs/development/native-qualification.md
[qualification-fixtures]: https://github.com/DeloneCommons/pyvoro2/blob/b5e1aa6f53cd5edd94c0ae376047cb53278d2158/tools/native/qualification/fixtures/README.md
[qualification-tests]: https://github.com/DeloneCommons/pyvoro2/blob/b5e1aa6f53cd5edd94c0ae376047cb53278d2158/tests/tooling/test_native_qualification.py
[reduced-tests]: https://github.com/DeloneCommons/pyvoro2/blob/b5e1aa6f53cd5edd94c0ae376047cb53278d2158/tests/forward/common/test_reduced_periodic_geometry.py
[reflection-tests]: https://github.com/DeloneCommons/pyvoro2/blob/b5e1aa6f53cd5edd94c0ae376047cb53278d2158/tests/forward/spatial/test_periodic_reflection.py
[remap-tests]: https://github.com/DeloneCommons/pyvoro2/blob/b5e1aa6f53cd5edd94c0ae376047cb53278d2158/tests/forward/spatial/test_periodic_remap.py
[row-active]: https://github.com/DeloneCommons/pyvoro2/blob/b5e1aa6f53cd5edd94c0ae376047cb53278d2158/tests/inverse/separator/test_row_active.py
[row-fit]: https://github.com/DeloneCommons/pyvoro2/blob/b5e1aa6f53cd5edd94c0ae376047cb53278d2158/tests/inverse/separator/test_row_fit.py
[row-oracles]: https://github.com/DeloneCommons/pyvoro2/blob/b5e1aa6f53cd5edd94c0ae376047cb53278d2158/tests/inverse/separator/test_row_oracles.py
[row-prox]: https://github.com/DeloneCommons/pyvoro2/blob/b5e1aa6f53cd5edd94c0ae376047cb53278d2158/tests/inverse/separator/test_row_prox.py
[row-reports]: https://github.com/DeloneCommons/pyvoro2/blob/b5e1aa6f53cd5edd94c0ae376047cb53278d2158/tests/inverse/separator/test_row_reports.py
[separator-diagnostics]: https://github.com/DeloneCommons/pyvoro2/blob/b5e1aa6f53cd5edd94c0ae376047cb53278d2158/src/pyvoro2/inverse/separator/_diagnostics.py
[separator-policy]: https://github.com/DeloneCommons/pyvoro2/blob/b5e1aa6f53cd5edd94c0ae376047cb53278d2158/src/pyvoro2/inverse/separator/_policy.py
[separator-problem]: https://github.com/DeloneCommons/pyvoro2/blob/b5e1aa6f53cd5edd94c0ae376047cb53278d2158/src/pyvoro2/inverse/separator/problem.py
[separator-report]: https://github.com/DeloneCommons/pyvoro2/blob/b5e1aa6f53cd5edd94c0ae376047cb53278d2158/src/pyvoro2/inverse/separator/report.py
[separator-solver]: https://github.com/DeloneCommons/pyvoro2/blob/b5e1aa6f53cd5edd94c0ae376047cb53278d2158/src/pyvoro2/inverse/separator/solver.py
[source-range-tests]: https://github.com/DeloneCommons/pyvoro2/blob/b5e1aa6f53cd5edd94c0ae376047cb53278d2158/tests/inverse/separator/test_source_edge_diagnostic_range.py
[source-tests]: https://github.com/DeloneCommons/pyvoro2/blob/b5e1aa6f53cd5edd94c0ae376047cb53278d2158/tests/inverse/separator/test_source_geometry.py
[spatial-dispositions]: https://github.com/DeloneCommons/pyvoro2/blob/b5e1aa6f53cd5edd94c0ae376047cb53278d2158/docs/development/spatial-degeneracy-regressions.md
[spatial-tests]: https://github.com/DeloneCommons/pyvoro2/blob/b5e1aa6f53cd5edd94c0ae376047cb53278d2158/tests/forward/spatial/test_degeneracy_scope.py
[translation-tests]: https://github.com/DeloneCommons/pyvoro2/blob/b5e1aa6f53cd5edd94c0ae376047cb53278d2158/tests/forward/common/test_native_translation.py
[user-tests]: https://github.com/DeloneCommons/pyvoro2/blob/b5e1aa6f53cd5edd94c0ae376047cb53278d2158/tests/forward/spatial/test_periodic_user_coordinates.py
[viz-tests]: https://github.com/DeloneCommons/pyvoro2/blob/b5e1aa6f53cd5edd94c0ae376047cb53278d2158/tests/forward/spatial/test_viz3d_helpers.py
[weight-tests]: https://github.com/DeloneCommons/pyvoro2/blob/b5e1aa6f53cd5edd94c0ae376047cb53278d2158/tests/forward/common/test_weight_first_queries.py
[witness]: https://github.com/DeloneCommons/pyvoro2/blob/b5e1aa6f53cd5edd94c0ae376047cb53278d2158/docs/development/native-face-witness.md
[wp5]: https://github.com/DeloneCommons/pyvoro2/blob/b5e1aa6f53cd5edd94c0ae376047cb53278d2158/docs/development/wp5-implementation.md
[wp5-character-tests]: https://github.com/DeloneCommons/pyvoro2/blob/b5e1aa6f53cd5edd94c0ae376047cb53278d2158/tests/forward/spatial/test_wp5_characterization.py
[wp5-fixture]: https://github.com/DeloneCommons/pyvoro2/blob/b5e1aa6f53cd5edd94c0ae376047cb53278d2158/tests/fixtures/wp5_characterization.json
[wp5-ideal-tests]: https://github.com/DeloneCommons/pyvoro2/blob/b5e1aa6f53cd5edd94c0ae376047cb53278d2158/tests/forward/spatial/test_wp5_ideal.py
[wp7]: https://github.com/DeloneCommons/pyvoro2/blob/b5e1aa6f53cd5edd94c0ae376047cb53278d2158/docs/development/wp7-implementation.md
[wp8]: https://github.com/DeloneCommons/pyvoro2/blob/b5e1aa6f53cd5edd94c0ae376047cb53278d2158/docs/development/wp8-implementation.md
[wp8-matrix]: https://github.com/DeloneCommons/pyvoro2/blob/b5e1aa6f53cd5edd94c0ae376047cb53278d2158/tests/forward/common/test_wp8_matrix.py
[wp8-tests]: https://github.com/DeloneCommons/pyvoro2/blob/b5e1aa6f53cd5edd94c0ae376047cb53278d2158/tests/forward/common/test_wp8_metadata.py
