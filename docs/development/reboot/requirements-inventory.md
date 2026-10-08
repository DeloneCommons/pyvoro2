# Requirements and capability inventory

This inventory separates **what was learned**, **what the historical attempt
implemented**, and **what the reboot authorizes**. Read the [overview](index.md)
for source identities and checkpoint outcomes, and the [evidence index](evidence-index.md)
for independent oracles, precise fixtures and review provenance.

An observable requirement below is something a possible capability should be
assessed against; describing it does not approve that capability, its old
implementation, or a new release commitment. Current authority is
[ADR 0027](../decisions/0027-v0.9-reboot-from-v0.8.md). Historical ADRs 0018–0026
remain evidence. Current callable behavior is in the [v0.8 API inventory](../api-inventory.md).

## Classification and coverage matrix

**Kind** and **reboot disposition** are independent dimensions. The compact kind
codes mean: **S** — scientific invariant/property; **N** — native or numerical
safety requirement; **A** — public API capability; **H** — historical
implementation detail; **C** — candidate improvement; **R** — research hypothesis.
An entry can contain more than one kind; mathematical correctness does not by
itself select an implementation.

Disposition vocabulary is **Existing v0.8 baseline behavior**, **Unreviewed
candidate**, **Deferred**, **Rejected by explicit new decision**, and **Adopted
by explicit new decision**. The table shortens the first two to *Baseline* and
*Unreviewed*. Historical deferrals and acceptances are recorded separately.
This inventory makes no new adoption, rejection or scheduling decision.

| ID / capability | Kind | Historical status or workstream | Reboot disposition |
|---|---|---|---|
| [R-B01 — weights and gauge](#r-b01) | S, A | Published v0.8 | Baseline |
| [R-B02 — fixed inverse, identity and results](#r-b02) | S, A | Published v0.8 | Baseline |
| [R-B03 — inputs and native preflight](#r-b03) | N, A | Published v0.8 | Baseline |
| [R-Q01 — weight-first queries](#r-q01) | A, S | WP1; accepted A | Unreviewed |
| [R-L01 — user lattice/backend frame](#r-l01) | S, A, H | WP2; accepted A | Unreviewed |
| [R-L02 — reduction and image search](#r-l02) | S, H, C | WP3–WP4; accepted A | Unreviewed additions to baseline CVP |
| [R-L03 — source geometry and remapping](#r-l03) | S, N | WP4; #60/#65; A and later #67 | Unreviewed remedies |
| [R-G01 — ghost execution safety](#r-g01) | N, H | #62; WP7 | Unreviewed remedies |
| [R-T01 — spatial occurrence attribution](#r-t01) | S, H, A | WP5; accepted B | Unreviewed |
| [R-T02 — planar occurrence attribution](#r-t02) | S, H, A | WP6; accepted B | Unreviewed |
| [R-G02 — ghost boundary semantics](#r-g02) | S, A | WP7; accepted B | Unreviewed |
| [R-M01 — metadata/output ownership](#r-m01) | A, S, N | WP8; #92/#93; accepted B | Unreviewed additions/remedies |
| [R-A01 — obsolete reconstruction controls](#r-a01) | A | WP9 removals; accepted B | Unreviewed removals |
| [R-T03 — planar degeneracy/identities](#r-t03) | S, H | #94–#98; accepted bounded B | Unreviewed |
| [R-T04 — spatial degeneracy limits](#r-t04) | S, C | #96/#99; accepted bounded B | Unreviewed |
| [R-E01 — native qualification](#r-e01) | N, H | #88; later #109 accepted separately | Unreviewed; not mandatory |
| [R-E02 — exact-kernel performance](#r-e02) | C, H | #82 investigation; #84/PR #87 merged | Unreviewed |
| [R-E03 — operational lessons](#r-e03) | H | #113 historical; PR1 reboot | Adopted by explicit new decision: PR1 scope only |
| [R-I01 — independent separator spaces](#r-i01) | S, A | WP10 accepted; C unaccepted | Unreviewed |
| [R-I02 — row values/applicability](#r-i02) | S, A | #104/WP10 accepted; C unaccepted | Unreviewed |
| [R-I03 — shape parameters](#r-i03) | C | PR #115 gate accepted; no implementation | Unreviewed historical dispositions |
| [R-I04 — realization-aware facade](#r-i04) | A, H | WP11 accepted; C unaccepted | Unreviewed facade; baseline advanced engine retained |
| [R-I05 — final-state correctness](#r-i05) | S, N | #116/PR #117 accepted remediation | Unreviewed remedies |
| [R-I06 — exact diagnostic availability](#r-i06) | S, N, A | #118/PR #119 accepted remediation | Unreviewed remedies/schema |
| [R-O01 — future periodic complex](#r-o01) | R | Open hypotheses | Unreviewed |
| [R-O02 — vendor policy and allocation failure](#r-o02) | N, C | D9 / #64 open | Unreviewed |
| [R-O03 — extreme power range](#r-o03) | S, C | Historical deferred investigation | Unreviewed |
| [R-O04 — measures and mixed inversion](#r-o04) | R | Former roadmap; superseded scope | Unreviewed research only |
| [R-O05 — other candidate capabilities](#r-o05) | C, R | Open/deferred roadmap directions | Unreviewed |
| [R-O06 — PA backlog and unfinished gates](#r-o06) | C, H | PA-001–003 open; recovery/WP12/WP13 unfinished | Unreviewed; no inherited gates |

<a id="baseline"></a>
## Existing v0.8 baseline

<a id="r-b01"></a>
### R-B01 — Mathematical weights, backend radii and component offsets

The power score is $\|x-p_i\|^2-w_i$. A single common addition to all weights
preserves the complete diagram; backend radii encode shifted weights and are
not unique physical sizes. This mathematical symmetry does not guarantee
bitwise-identical native results under arbitrary rounded radius conversion.
Baseline `compute` already accepts mathematical
weights, with shared conversion utilities and explicit overflow rejection.
The conversion is not permission to square arbitrary enormous radii safely.

A separate constant on each disconnected observation component preserves its
internal contrasts but can change competition between components. Such offsets
need a convention or prior; a global gauge cannot determine them. Positive L2
regularization fixes a reference-dependent problem and removes gauge freedom.
These are baseline scientific distinctions, applicable to every power or
inverse capability. See [baseline ADR 0002][b-adr2] and [E01](evidence-index.md#e01).
Later no-work/singleton corrections are separately inventoried in R-I05.

<a id="r-b02"></a>
### R-B02 — Fixed separator fitting, source identity and coherent results

The baseline already fits supplied separator observations on a graph, with
explicit solver/backend selection, convex penalties, hard constraints and a
certified scalar proximal route. It also has an experimental active outer loop.
A fixed fit does not prove that every observed pair supports a realized face,
and the outer heuristic has no global topology/convergence guarantee.

The baseline distinguishes source records from fit rows, external site IDs
from internal indices, and final result availability from success labels.
Its common forward result and numerical normalization do not promise an exact
periodic global complex. Returning to v0.8 preserves these capabilities and
contracts, including report schema v1, but does not preserve all later
remediation. See baseline [objective][b-adr7], [solver][b-adr8], [proximal][b-adr9],
[source identity][b-adr14], [atomic state][b-adr15] and [diagnostics][b-adr16]
decisions; [E14–E16](evidence-index.md#separator-evidence) distinguish later changes.

<a id="r-b03"></a>
### R-B03 — Strict input, preparation and native safety boundaries

Baseline validation rejects lossy IDs, non-Boolean flags and nonfinite numeric
inputs; canonical domains and prepared arrays own their data. Mandatory
inserted-generator checks are independent of optional diagnostics. Periodic
minimum-image comparison uses the inclusive exact represented squared floor
`1e-10`; this implementation safety floor is not a universal physical scale.
Locate queries are not inserted generators; ghosts are.

Direct-native preflight independently checks 18 constructor paths, integer and
allocation arithmetic, finite expressions and a source-derived eager-allocation
cap of 1 GiB. It does not guarantee total memory usage or recovery from every
allocation failure. Conservative native interval refusal is distinct from
exact Python geometric decisions. Baseline certified CVP is already
proof-bounded; the new reducer was an efficiency/robustness addition, not the
first correctness proof. See [ADR 0010][b-adr10], [0011][b-adr11], [0012][b-adr12]
and [0013][b-adr13], and [E03](evidence-index.md#e03). These safeguards remain
baseline; ghost ID and allocation findings below identify additional gaps.

## Weights, coordinates and periodic image geometry

<a id="r-q01"></a>
### R-Q01 — Weight-first locate and ghost queries

**Problem and requirement.** Baseline locate/ghost APIs require radii even
though compute accepts weights. A caller should be able to use one mathematical
weight convention across 2D/3D compute, locate and ghost operations.

**Historical solution.** WP1 (#49/PR #50, portability correction PR #51) added
weight-first query families. Persistent and all temporary ghost weights used
one common gauge; mixed or incomplete weight/radius families were rejected,
as were power-only arguments in standard mode. The singular ghost-radius
compatibility spelling was removed historically. Manual radius conversion was
an alternative representation, not an independent native oracle.

**Assessment.** Accepted within A. Return to v0.8 restores radius-only query
signatures; implementation and removal choices need new approval. Analytic
owner/volume examples and gauge tests in [E01](evidence-index.md#e01) preserve
useful expectations independently of the desired API spelling.

<a id="r-l01"></a>
### R-L01 — Caller lattice versus internal backend frame

For row-basis $A$, source coordinates satisfy $x=o+fA$. A public lattice need
not be a backend-compatible triangular frame. WP2 (#54/PR #55) accepted either
orientation of an exactly nonsingular binary64 lattice and added exact
fractional transforms/wrapping. Exact rational floor decides the shift; the
returned binary64 remainder may round to the excluded upper endpoint without
changing that exact decision.

The internal QR relation $A^T=QR$, $L=R^T=AQ$ and $x_b=(x-o)Q$ maps to a backend
frame. A reflection requires orientation/winding care. Floating QR checks are
not exact geometry certificates. Backend-primary remapping is a different
operation from user-cell wrapping; backend reduction was not automatically
selected. WP2 still used coefficient-space tie selection; WP4 changed it.

Accepted in A under [ADR 0018][h-adr18]. Returning to v0.8 restores its
right-handedness/conditioning restrictions and lacks these public coordinate
methods. [E02](evidence-index.md#e02) indexes independent Fraction solves and
frame controls. Equivalent mathematical bases need not make every native
floating computation equally admissible or bitwise identical.

<a id="r-l02"></a>
### R-L02 — Exact reduction, closest images and duplicate distances

The baseline's exact closest-vector search can be expensive in a poor source
basis. WP3 introduced private exact rank-three LLL reduction; WP4 integrated
it into image, duplicate-distance and translation consumers. For $B=UA$,
integer unimodular $U$ and its exact inverse transport shifts by
$s_{user}=s_{reduced}U$.

The [ADR 0020][h-adr20] algorithm uses $\delta=3/4$, full size reduction,
half-integer rounding toward zero, strict Lovász swaps and deterministic sign
normalization. Integer Gram-potential descent proves termination; resource
limits still permit honest refusal. Pairwise reduction alone misses
three-vector cancellation. No public Niggli/Minkowski canonicalizer was adopted.

WP4 selected closest-image ties by physical Cartesian displacement, then the
declared extremal choice; this is invariant under equivalent lattice bases,
not arbitrary rotations. `image_search` remains a correctness-neutral seed.
Distance-only duplicate checks avoid unnecessary finite-vector/int64 views.
Accepted in A; all additions are unreviewed for the reboot. [E03](evidence-index.md#e03)
separates independent completeness bounds, formula-only workload estimates and
actual consumer counts. Huge-shear proof fixtures do not certify native
geometry over the same entire input range.

<a id="r-l03"></a>
### R-L03 — Preserve source geometry through numerical remapping

Exact inference must consume original source operands. In #60/PR #61, snapping
an endpoint before nearest-image inference turned a unique source separation
`1/2 - 2^-41` into a tie and selected the wrong image. A correct exact kernel
cannot restore source bits already discarded by preparation.

In #65/PR #66, a coupled triangular remap pushed a coordinate beyond one cell
width; a one-sided snap then erased the legitimate `3/16` residual. Two-sided
endpoint neighborhoods preserved it. PR #67 later repaired `eps=0` endpoint
and negative-quotient-underflow cases. These are backend numerical repairs,
not replacements for exact user wrapping or a general rounded-idempotence law.

WP4's exact native-translation kernel enumerates the entire declared closed
Cartesian error box: zero candidates means inconsistent, one unique, more than
one ambiguous. It neither establishes the producer's error box nor chooses the
nearest residual when several candidates survive. The source and coupled-remap
repairs were required for A; #67 followed acceptance. Baseline retains the old
source/remap paths, so these regressions matter to any reuse. No fresh baseline
native reproduction is claimed here. See [E03–E04](evidence-index.md#e04),
[#60][i60], [#65][i65] and [PR #67][p67].

<a id="r-g01"></a>
### R-G01 — Independent ghost calls and initialized native identities

Two different native hazards were found. #62/PR #63 showed that deleting a
primary temporary ghost did not remove cached periodic images: later ghosts
could see earlier state. Fresh containers per query restored batch/single-call
independence, at the cost of repeated construction and insertion. Analytic
unit-cube volumes refuted the contaminated result; comparing two shared
backend paths alone would not have sufficed.

WP7 separately found a temporary-neighbor ID could be read/copied before
initialization. It used a selected augmented container with a real initialized
internal ghost ID, before translating to public meaning. A defined poison
harness establishes the read/copy hazard; ASan/UBSan alone cannot prove absence
of uninitialized reads. Reused public ID `-1` is not an identity model.

The state fix was accepted in A; the selected ghost route in B. Both are
unreviewed remedies relevant to baseline ghost execution. Neither settles
allocation-exhaustion ownership (#64), nor defines boundary eligibility
(R-G02). [E05](evidence-index.md#e05) preserves the independent and structural
controls without importing their old container architecture.

<a id="geometry-layers"></a>
## Boundaries, identity and normalized topology

The historical notation is essential when interpreting a “certificate”:

| Layer | What it describes | What success does not establish |
|---|---|---|
| **N — native occurrences** | Actual floating cuts, source occurrences, stored coordinates, labels and indexed cycles | Complete positive-measure Laguerre boundaries or exact public geometry |
| **E — exact backend ideal** | Exact geometry of stored backend binary64 sites/lattice and exact squares of the actual backend radius operands | Equality with original source geometry; exactifying rounded native `r*r` gives a different definition |
| **S — exact source/public-semantic ideal** | Original persistent public sites/lattice and mathematical weights, or exact squares of explicitly supplied radii; WP7 ghosts use the actual stored ghost anchor described below | That the native producer retained every positive contact |

Numerical matching, exact image attribution, exact contact positivity and
complete semantic coverage are separate claims. Neither E nor S is allowed to
choose an otherwise ambiguous N provenance label. Historical [ADRs 0021–0025](evidence-index.md#source-corpus)
define particular actions at these boundaries; they are not current reboot
architecture.

<a id="r-t01"></a>
### R-T01 — Spatial native occurrence attribution and exact contacts

Baseline 3D reconstruction uses numerical geometry and a finite search control;
a plausible plane fit is not proof of the native source image. WP5
(#68, PRs #69–#72) introduced source-coupled witnesses, binary64 rounding-bin
preimages and complete source-compatible translation regions. Its source
replay follows actual arithmetic/branches, including negative native truncation
behavior; equivalent cut geometry alone is insufficient provenance.

The historical witness validates every final cycle, seed replay and occurrence.
Separate bounded exact-halfspace constructions decide E/S contact dimension,
positivity and coverage, retaining equality cases in finite candidate bounds.
Public shifts transport insertion images by $\sigma+K_i-K_j$; frame defects
remain observable. Unique N attribution can coexist with failed optional E/S
diagnostics, governed by `none/diagnose/warn/raise`. Ambiguous or incomplete N
attribution fails atomically. No missing reverse contact is fabricated.

Accepted in bounded B under [ADR 0021][h-adr21]. PR #69's shared noncontracting
FP policy changed 46/60 ordinary computations, including 13 topology changes,
in the recorded FMA-capable `-march=haswell` comparison. The generic Linux/GCC
13.3 comparison matched all 60. The observer addition was therefore not
universally behavior-neutral. Reboot adoption of any witness, arithmetic policy or refusal needs
separate assessment. [E06](evidence-index.md#e06) separates independent ideals
from ordinary/observer parity. The baseline lacks this certified provenance
and diagnostic framework, while still providing numerical cells and shifts.

<a id="r-t02"></a>
### R-T02 — Planar edge provenance without semantic invention

WP6 (#74/PR #76) propagated source tokens through seven planar cut sites,
vertex growth/compaction and final outgoing-edge slots. A complete rectangular
image region and independent rational line-interval oracle support image and
contact reasoning. Inserted coordinates must pass the actual native insertion
check: a valid public point just below an endpoint can round outside a native
block calculation.

Raw collapsed/zero-length edges can carry genuine native provenance. They
must not be deleted solely because a numerical length vanishes or an S contact
is absent. As in ordinary spatial calls, diagnostic policy is distinct from
unique N attribution. A missing/invalid shift remains unavailable, not a
fabricated zero displacement (#92/PR #93).

Accepted in B under [ADR 0022][h-adr22]. The final historical planar route
required qualification even for ordinary nonperiodic calls; no unqualified
fallback was selected. This is an unreviewed availability choice for the
reboot, not an inevitable consequence of exact planar mathematics.
[E07](evidence-index.md#e07) and [E09](evidence-index.md#e09) locate the source,
oracles and normalization counterexamples. Baseline numerical edge recovery
remains current pending a new choice.

<a id="r-g02"></a>
### R-G02 — Ghost owners, walls, self-images and complete boundaries

A temporary ghost, a persistent owner, a physical wall and a nonzero periodic
self-image are different entities. WP7's tagged boundary references preserve
that distinction instead of overloading a public integer. In the stored ghost
chart, a persistent reference uses shift $\sigma-K_j$; a self-reference uses
nonzero $\sigma$. Here $g$ is the actual stored/materialized ghost site, which
anchors returned geometry, not the original unwrapped query. Ghost S combines
that $g$ with original persistent sites, public lattice/walls and mathematical
weights. Native raw collapse may have no semantic reference.

Boundary-bearing ghost output required unique N attribution **and complete S
positivity/coverage**, or a proved empty disposition; E success was not required
on every ghost call. Failure of that stronger contract fails the entire batch.
Geometry-only availability is separate. Independent ghost queries do not form
a partition to be globally normalized together.

Accepted in B under [ADR 0023][h-adr23]. The stored triclinic ghost example has
six native facets but eight source-ideal facets, so a volume of `1/2` does not
prove complete boundary metadata. Coordinates may also all round to one
public point while distinct private vertices and positive references survive.
[E08](evidence-index.md#e08) preserves these facts. The baseline lacks this
tagged/certified boundary contract; its adoption and output schema remain
unreviewed alongside alternative availability policies (PA-001).

<a id="r-m01"></a>
### R-M01 — Query wrapping, owner images and output ownership

WP8 (#79/PR #80) separated the query's exact user-cell wrap from the native
owner's numerical position. It preserved native owner selection and position
bits, then used source-derived enclosures to certify the corresponding image
of the original owner. This is not an exact nearest-owner theorem: native
`owner_pos=0` can legitimately correspond to source image coordinate `-2^-53`.

Optional internal geometry had to survive until metadata/normalization
consumers finished, then be stripped according to selectors. ID-only routes
did not need unused certificates. #92/PR #93 repaired three integration seams:
image-qualified self-incidence, unavailable planar annotations, and normalized
nested records sharing mutable storage with raw output later stripped.

Accepted in B; [WP8 notes][h-wp8] and [E11](evidence-index.md#e11) define the
historical fields and matrix. Returning to v0.8 removes these metadata
extensions and restores old ownership paths. Caller IDs and user images remain
baseline meaning; a new schema or proof route is unreviewed. Numerical
coordinate equality alone is never evidence of periodic-image identity.

<a id="r-a01"></a>
### R-A01 — Removing reconstruction knobs is an API decision

WP9 (#85/PR #86) removed the applicable face/edge search, validation, repair and
tolerance controls made obsolete by the historical certified routes, following
earlier ghost spelling changes. Finite-window reconstruction could remain a
seed or diagnostic but could not define exhaustive attribution.

The historical removals were accepted in B. Those controls remain in the
current v0.8 API; this inventory does not deprecate them. Any replacement must
first settle its actual behavior, resource limits and migration needs.
[Historical API inventory][h-api] and [E11](evidence-index.md#e11) distinguish
signature checks from proof that a new geometry algorithm is correct.

<a id="r-t03"></a>
### R-T03 — Planar degeneracy and proof-assisted identities

Integrated B first failed despite green CI (#92). The repairs in PR #93 did
not resolve all pre-existing planar degeneracy. #94 characterization and #95–#97
contract work led to #98/PR #102 and [ADR 0025][h-adr25]. The accepted square
retains **all 20 N occurrences**, organized as **4 global vertices / 12 raw edge
classes**; its exact S complex has **4 vertices / 8 positive edges / 4 cells**.
These counts answer different questions. Earlier shorthand in #94 must be read
through the final contract.

The bounded adapter distinguishes proof of artifact exemption from proof of
identity. Exact E/S singleton/member bridges and typed lift potentials justify
specific aliases before irreversible numerical pooling. Incidence, equal
coordinates and zero measure alone do not. Alternate transport paths must
agree; a corrupted proof is refused rather than silently downgraded. Copying or
serialization discards private proof context while retaining numerical fields.

Accepted in bounded B, with five-/six-way central identity checks (45/45 and
105/105) that do not certify every whole map. Reboot baseline numerical
normalization lacks this proof adapter. [E09](evidence-index.md#e09) preserves
both positive controls and the remaining ambiguity; future incidence-first
construction is a separate hypothesis, not this implementation's conclusion.

<a id="r-t04"></a>
### R-T04 — Spatial missing faces, coordinate collisions and winding

Issue #96's independent atlas and #99/PR #103 fixed a bounded numerical spatial
contract, without importing a 3D exact identity adapter. The maintained subset
has 33 input-only cases spanning 16 archetypes, including seven strict-pass
semantic disagreements. The external atlas recorded twelve strict
failures; their dispositions remain documented, but only the representative
bipyramid failure was retained as a maintained regression. Eighty-four further
physical variants were not promoted because they repeated mechanisms. A search finding
no retained lower-dimensional native face among 5,634 plus 304 examined faces
is a bounded negative observation, not a theorem.

Counterexamples include a bipyramid radial perturbation `2^-36` with a missing
positive S face; cube radial perturbation `2^-52` with three pairs of distinct
exact vertices sharing public coordinate triples; and power geometry with
E=29V/59E but S=28V/58E, both 35F/5C. Complete audit need not imply E=S.

A one-generator periodic cube has quotient 1V/3E/3F/1C: an edge can join different
lifts of the same global vertex. Nonzero winding is not an inconsistent identity
cycle. Partial-periodic ridge endpoints can remain physically distinct.
Accepted B preserves these limits, not a globally exact complex. PR #105 later
documented a narrow malformed weak-view wall-bookkeeping failure; it did not
repair runtime code or reopen B. [E10](evidence-index.md#e10) and the [regression
catalogue][h-spatial] give inputs and expected dispositions. All candidate
normalization improvements remain unreviewed for the reboot.

## Numerical engineering and operational lessons

<a id="r-e01"></a>
### R-E01 — Geometry certificates and artifact qualification are different

Issue #88/PR #90 qualified six native components/routes with external source closure,
observed effective build commands/objects, arithmetic discriminators, immutable
loaded-payload identity and current-thread FP checks before/after callbacks.
Self-reported metadata, compiler names, requested flags, source hashes or final
geometry parity alone were insufficient. Properties of an actual build mattered,
not a universal compiler-name whitelist; partial platform profiles and sanitizer
runs had narrower scope than optimized numerical qualification.

Issue #109/PR #110 later separated mechanical source identity from reviewer approval:
an implementer-owned `native_source_manifest.json` replaced procedural approval
metadata, with record-v2 and enumeration/ZIP-entry negative controls. This was
accepted after B, not part of B's exact reviewed source. Geometry correctness,
source/build/installation identity and human acceptance remain separate.

[ADR 0024][h-adr24], [qualification documentation][h-qualification] and
[E12](evidence-index.md#e12) preserve the evidence. Returning to v0.8 removes this
framework and its route restrictions; importing it is not mandatory. Any
future certificate must state its real assumptions, while its implementation
and result-availability tradeoffs remain unreviewed (PA-001–003).

<a id="r-e02"></a>
### R-E02 — Profile exact arithmetic before optimizing the wrong subsystem

Issue #82 traced the Python 3.10/3.13 disparity to ordinary WP5 Fraction-based
clipping, not SciPy or the WP7 ghost oracle. #84/PR #87 carried active-constraint
and gap records with exact points, reducing recorded Fraction hashes from
4,103,028 to 298,878 (92.7%). The 207-case differential comparison preserved
observed semantics but shared the same geometric algorithm.

The reported full-suite timings on one host were 1002.89→834.10 seconds for
3.10 and 496.26→499.38 for 3.13. These are bounded observations, not universal
speedups or a reason to drop 3.10. The optimization was merged; a distinct
final acceptance comment was not located. [E13](evidence-index.md#e13) separates
profiles, timings and parity evidence. Baseline lacks this exact clipping
kernel, so the optimization is relevant only if similar work is selected.

<a id="r-e03"></a>
### R-E03 — Understandable development and review operations

Historical #113/PR #114 addressed slow process feedback and moved whole-code
recovery before Phase D; PR #83 had earlier CI risk routing. Those historical
workflows did not establish mathematical correctness or close recovery.

For the reboot, ADR 0027 and [PR1 #121][p121] explicitly adopt ordinary editable
development and routine fail-closed `CI gate`, with separately requested
extended CI. This is the only new adoption recorded in this inventory, and it
is limited to the already approved operational scope. It does not adopt
historical qualification, release gates or architecture. See the [current
workflow](../development-workflow.md) and [evidence status records](evidence-index.md#source-corpus).

<a id="separator-capabilities"></a>
## Separator fitting and realization

<a id="r-i01"></a>
### R-I01 — Observation space versus mismatch/model space

Baseline models use the observation measurement space for every term. WP10
(#107/PR #108; [ADR 0019][h-adr19]) allowed independent term-global choices for
mismatch, hard bounds and each penalty, while preserving source targets,
confidence, row IDs and measurement labels. Omitted spaces inherit the source.
This did not introduce arbitrary per-row spaces or generic mixed observations.

For an oriented pair, $z=w_i-w_j$ gives fraction and position coordinates

$$
f=\frac12+\frac{z}{2d^2},\qquad p=\frac d2+\frac{z}{2d}.
$$

Accepted stored binary64 `distance` and `distance2` are separate operands;
rounded `p=d*f` is not an implementation identity. The complete affine row is
$\beta+\alpha w_i-\alpha w_j$. With row-incidence matrix $D$, the quadratic
model uses $D^T\operatorname{diag}(c\alpha_{model}^2)D$ and the complete right-hand
side $D^T[c\alpha_{model}(t_{model}-\beta_{model})]$, not a lossy reconstruction
from rounded curvature and observed contrast. Algebraic edge diagnostics remain
source-derived, including source curvature; they are not the mixed-model graph
or a newly inferred variance model.

Convexity and row-local proximal structure remain under nonnegative confidence,
strengths and the supported convex families. Direct solving stays limited to
compatible quadratic problems; hard/nonquadratic work requires explicit ADMM,
and sparse backends remain explicit. WP10 was accepted, C was not. These spaces
and public fields are absent from the reboot baseline and unreviewed for
adoption. [E14](evidence-index.md#e14) includes hand-computable and independent
high-precision proximal oracles, beyond two paths through shared production code.

<a id="r-i02"></a>
### R-I02 — Bounded row values, hard applicability and stable association

The pre-B audit and #104/PR #106 selected a bounded A+B policy for WP10.
**A** allowed scalars or exact-length 1D row vectors for hard `Interval.lower/upper`,
`FixedValue.value`, and penalty `lower/upper/strength`. **B** added Boolean hard
`applicable`. Scalars broadcast; length-one arrays do not broadcast to other
row counts. Shape/type/finite validation remains strict even on inactive rows.
Hard equal endpoints are equality; soft widths remain positive.

An inapplicable hard row performs no conversion, feasibility/statistics work or
coupling. Zero confidence removes mismatch, not applicable hard policy. Zero
penalty strength suppresses dangerous evaluation and coupling while preserving
configured policy. Positive strength can couple even when the current penalty
is zero. L2 references remain site-indexed, including deceptive equal site/row
counts.

The implementation bound ordered row IDs once and projected observations and
policy together through subsets, re-entry, components and final refit. Duplicate
pairs/images remain distinct rows; equal lengths or pair sets do not prove
association. Full candidate policy and selected-fit policy have different
owners. Historical `resolved_policy` and schema v2 made this inspectable;
Issue #116 later moved reports to v3. Baseline scalar models, strict-width intervals
and v1 reports remain current. Accepted historical A+B is an unreviewed API and
ownership candidate; [E14](evidence-index.md#e14) covers omissions, zero controls,
row reorderings and native integration separately.

<a id="r-i03"></a>
### R-I03 — Shape parameters were decided individually, with no implementation

The accepted [PR #115][p115] gate selected **NONE** for additional row-wise
shape implementation. Its historical conclusions were:

| Parameter | Historical disposition | Reason / previously available alternative |
|---|---|---|
| Huber `delta` | DEFER | Changes robust transition; confidence scaling cannot emulate it. |
| Exponential `tau` | DEFER | Genuine decay/shape scale. |
| Reciprocal `margin` | DEFER | Genuine activation/boundary-layer width. |
| Exponential `margin` | RETAIN TERM-GLOBAL | At fixed `tau`, largely duplicates amplitude through `strength * exp(margin/tau)`. |
| Reciprocal `epsilon` | RETAIN TERM-GLOBAL | Common continuation/regularization policy within a term. |

All remained scalar. A few penalty instances with row-masked strengths can
express distinct regimes without generalizing every shape parameter. These
historical deferrals are not reboot scheduling or rejection decisions.
The [shape audit][h-shape] and [E14](evidence-index.md#e14) preserve the reasoning;
new freedom requires representative downstream evidence and explicit approval.
The plan's D8 separately deferred a public point-centered/symmetric-bound
convenience type or spelling: internal expressibility through measurement-space
models did not establish a need for another public API.

<a id="r-i04"></a>
### R-I04 — A convenient outer facade is not a new global solver

WP11 (#111/PR #112) added the historical Provisional
`fit_self_consistent_weights_from_separators(..., max_outer_iter=25)` facade,
reusing the existing active engine/result. It starts all candidate observations
active, disables history and retains default add/drop hysteresis, relaxation
and cycle controls; advanced controls/history remain Experimental.

Fixed fitting optimizes supplied observations. The realization-aware loop
recomputes geometry and updates an empirical active subset; hard policy is
active-conditional and does not promise face existence. It cannot repair every
initial hard infeasibility or create new candidate pairs. Full final cells and
image-qualified boundaries are needed to inspect extra/self images; pair-level
unaccounted summaries can omit self-pairs or an extra image of an already
realized pair.

Outer status, final inner convergence and final-state availability are separate.
A cycle/iteration limit can retain a coherent state; failed final refit cannot
borrow stale geometry. Native structural/resource/certificate failure must
propagate even with diagnostics disabled. WP11 was accepted, C remains
unaccepted. Baseline retains its advanced experimental engine, not this
preferred facade. [E14](evidence-index.md#e14) distinguishes analytic native
controls from facade/engine parity. No new supported import is authorized here.

<a id="r-i05"></a>
### R-I05 — Final weights, feasibility, diagnostics and geometry must agree

Issue #116/PR #117 repaired four integrated-C findings under [ADR 0026][h-adr26]:

| Finding | Requirement and historical correction | Relevant baseline impact |
|---|---|---|
| F1 — false hard success after final reference alignment | Re-evaluate each unchanged hard predicate on the actual returned weights; unrelated global maxima of violation/tolerance are not a valid test. Impossible binary64 representatives may yield `numerical_failure`; false successful public reconstruction is rejected even without canonicalization. | Baseline final-state contract exists, but later guards are absent; assess each reproducer before asserting the same baseline failure. |
| F2 — diagnostic representability and reportability | Keep source diagnostics source-owned; a genuinely unavailable diagnostic need not invalidate finite model weights/objective/geometry. Nullable output is limited to typed producer-owned fields. | Independent spaces and later availability/schema handling are absent. |
| F3 — cancellation hidden by rounded prediction | Compute the complete affine residual from original operands and each actual evaluated weight vector, including relaxed history; rounded prediction minus target can erase a real residual. Reconstructing two mutually wrong result layers is not validation. | Preserve the counterexample independently of old result builders. |
| F4 — zero-L2 and no-work states | At zero L2, separate coupling components use the declared reference/zero conventions; singleton values equal reference exactly, and empty fits return reference/zeros. One connected multisite component retains its solver anchor; positive L2 does not permit gauge alignment. | Baseline component-offset mathematics remains; later edge-case repairs need explicit reassessment. |

Coupling includes positive-confidence observations, applicable hard constraints
and positive penalties, not only the informative graph. Failure handling clears
invalid success-dependent fields while preserving actual solver/iteration
provenance, configured policy and valid source-only diagnostics. Finite
nonconverged states can remain inspectable.

The remediation was historically accepted at `7356157b…`, merged as `d2785d36…`;
it did not accept C. [E15](evidence-index.md#e15) supplies exact hard-feasibility,
small-residual and singleton/no-work witnesses. A later private review's R1/R2
labels identify different findings from #116's internal consolidation label R1.

<a id="r-i06"></a>
### R-I06 — Genuine out-of-range diagnostics versus evaluation failure

Issue #118/PR #119 required exact range classification of the complete diagnostic
from accepted binary64 operands. Let $M$ be maximum finite binary64. Exact
$M+\epsilon$ is out of range even when rounding would produce $M$; conversely,
large intermediate terms do not make a finite complete expression unavailable.
For a nonempty $n$-row view with residuals $r$ and confidence $c$:

| Diagnostic | Exact finite-range condition |
|---|---|
| Row residual / maximum absolute residual | Every required absolute residual is at most $M$. |
| RMS | $\sum r_i^2\le nM^2$. |
| Weighted L2 | $\sum c_i r_i^2\le M^2$. |
| Weighted RMSE | $\sum c_i r_i^2\le nM^2$; denominator is row count, not summed confidence. |

Zero confidence removes its exact weighted contribution before dangerous
arithmetic, without deleting the row from source diagnostics or denominators.
Source `z_fit`, `z_obs`, `q=z_obs-w_i+w_j` and curvature `c*alpha_source^2`
are classified independently: unavailable leaves can cancel to a finite
complete `q`. Rounded `sqrt(c)` is not an exact confidence weight.

B1/B2 were fixed at PR #119's earlier head `84ad1a98…`, yet independent review
still found **R1** source-edge range errors and **R2** strict-policy intermediate
overflow on a finite affine cancellation. Green CI did not refute those cases.
The accepted replacement `863d56e27e3a03e2cf85f375b7889af3aad408b6` routed exceptional
lanes before unsafe products/prefixes, retaining ordinary vectorized work.
Rare exact-rational classification and high-precision aggregate square roots
are not a universal correctly-rounded-output proof. Hard constraints, objective
and solver tolerances retained their separate numerical contract.

Historical report schema v3 requires local `unavailable_diagnostics` JSON-pointer
maps and typed reasons (`out_of_binary64_range` / `unavailable_dependency`)
for eligible nulls. Structural nulls differ; inputs, returned weights and other
strict fields cannot admit arbitrary NaN/Inf or forged reasons. History summaries
without historical weights cannot re-certify those discarded vectors.
Issue #118 explicitly accepted the final remediation while leaving C unaccepted.
Baseline has v1 and none of this later schema contract. [E16](evidence-index.md#e16)
and [E17](evidence-index.md#e17) separate exact witnesses, maintained regressions,
rejected-head review, final source and missing final review artifacts. Adoption
of remedies and serialization choices remains unreviewed.

<a id="open-questions"></a>
## Open architectural and research questions

<a id="r-o01"></a>
### R-O01 — Incidence-first reconstruction and direct semantic construction

The [historical roadmap][h-roadmap] and [PR #100][p100] preserve incidence-first
reconstruction as an open hypothesis: determine correspondence/incidence before
pooling numerical coordinates, with exact image transport and explicit identity
proofs. A second hypothesis constructs semantic cells/contacts directly from
Laguerre halfspaces rather than treating returned native faces as exhaustive.

Neither is selected. Incidence matching alone cannot recover omitted native
positive faces, settle degeneracy or prove global topology. Direct semantic
construction would need its own complete image bounds, robust incidence and
lower-dimensional treatment, domain scope, complexity and validation evidence.
[E09–E10](evidence-index.md#e09) provide counterexamples any candidate must address.
The reboot baseline remains numerical normalization; no new global-complex API,
representation, backend replacement or success guarantee is approved.

<a id="r-o02"></a>
### R-O02 — D9 and exceptional native allocation ownership

Historical D9 left **upstream-functionally-unmodified Voro++ versus a bounded
maintained downstream functional patchset** open. A concrete vendor change
would trigger the choice; patch lifetime alone does not define a fork. WP7's
binding-only fix did not settle D9. Its old WP13 deadline is superseded.

[#64][i64] records sequential raw allocations in periodic container
construction/growth that may lack ownership if `bad_alloc` interrupts them.
This inherited, nonblocking finding is distinct from normal-execution ghost
state contamination. Choices included defining/excluding recoverable leak-free
allocation exhaustion, then deterministic fault injection before any patch.
No such fault-injection artifact was located. The baseline shares the relevant
vendor source; the guarantee and remedy remain unreviewed. See [plan D9][h-plan]
and [E05](evidence-index.md#e05).

<a id="r-o03"></a>
### R-O03 — Large genuine power-weight range

Finite inputs and a common gauge do not bound the genuine weight differences
or protect every native radius-square/plane calculation. Uniform shifts cannot
cure a large physical dynamic range. The archived plan records investigation
rather than a universal safe cutoff, automatic rescaling or approved backend
patch. Source-semantic and rounded-backend ideals may diverge before obvious
nonfinite output. [E06](evidence-index.md#e06) supplies large-radius arithmetic
witnesses. Baseline power operations retain this candidate safety question;
new scaling, domain restrictions and rejection policy need independent analysis.

<a id="r-o04"></a>
### R-O04 — Prescribed measures and mixed inverse objectives

Prescribed-cell-measure inversion and mixed separator-plus-measure inversion
remain research hypotheses. The former plan/roadmap's proposed versions and
architectural relationships are superseded by ADR 0027. Neither a common
solver, modular API arrangement, public signature, dependency on the current
outer loop, release target nor pre-1.0 inclusion is approved.
The old suggestion of per-site anchors and inspectable block/row contributions
also did not select a public `ObservationBlock` protocol, arbitrary callbacks
or site motion as prerequisites; a private composition interface was an
alternative to assess.

Representative dimensions/domains, feasibility and nonempty-cell assumptions,
identifiability/gauge, differentiation or alternative algorithms, and independent
validation must be studied before selection. The preserved separator identities
and exact geometry counterexamples are useful inputs, not evidence that these
research problems have already been solved. No measure-fitting capability is
restored or promised by the v0.8 reboot.

<a id="r-o05"></a>
### R-O05 — Other candidates without an inherited schedule

The archived roadmap's other directions can be retained without their former
version assignments:

| Candidate family | Specific open work and boundary |
|---|---|
| Backend/throughput | Repeated frames and parallel execution ([#23][i23]); actually running the backend on a reduced basis and transporting results back, distinct from WP3/WP4's private proof reduction. |
| Domain/preparation | Separate storage and requested clipping domains for exterior generators; dominated coincident sites with unequal weights; optional similarity scaling of all affected quantities; convex/wall domains, oblique 2D or partial-triclinic 3D periodicity. Unbounded cells require a different geometry contract. |
| Descriptors and reconstruction | Solid angles ([#75][i75]), centroids, second moments, nonuniform-density masses, planar sections/slices and regular-triangulation/dual diagnostics. |
| Inverse research | Solver plugins, bounded site-coordinate optimization, anisotropic/non-Euclidean models, membership/power-comparison inequalities, visibility/support constraints, tied/structured weights and boundary-point observations. |
| Package boundaries and exposition | Theory/manuscript completion ([#19][i19]); reassess packaging on evidence of an inverse-only native-free audience, another backend, distinct release cadences or maintainership. Neither a split nor the old through-1.0 prescription is newly adopted. |

Each needs a concrete use case and bounded validation; none is a replacement
release plan.

For example, a $4\pi$ solid-angle sum requires the site strictly inside a closed
convex spatial cell; a power generator need not lie in its own cell. A descriptor
from native polygons is not an exact semantic certificate. Repeated-frame
speedups require profiling and explicit lifetime/invalidation behavior, not an
assumed persistent-container API. These unreviewed candidates have no new
reboot delivery scope. See [archived roadmap][h-roadmap].

<a id="r-o06"></a>
### R-O06 — PA-001–003 and unfinished recovery

The [post-functional-audit backlog][h-backlog] was non-normative intake:

| Item | Preserved question / alternatives | Applicability after returning to v0.8 |
|---|---|---|
| PA-001 | Separate necessary refusal, certified output and explicitly best-effort availability by surface; consider capability/trust metadata versus retaining refusal. | Conditional on a future architecture creating these guarantee boundaries; do not import qualification to recreate the problem. |
| PA-002 | Wrapped/instrumented loaders were refused while ordinary direct loading worked; investigate proof-preserving compatibility versus retaining/documenting refusal. | Historical loader/qualification integration is absent; no replacement loader is approved. |
| PA-003 | Whole-integration qualification cost versus closure-aware or scheduled validation. | PR1 has its own explicit CI decision; the old qualification-cost solution is unreviewed. |

Historical Checkpoint C, pre-Phase-D whole-code recovery, WP12 public-workflow
qualification, WP13 API/lifecycle audit and release gate #48 remained unfinished.
They are preserved status facts, not mandatory reboot gates. No final API freeze
or v0.9 release can be inferred from accepted remediations or the archive tag.
The historical documentation overhaul also remained unfinished: capability
verification, a limited-redesign MkDocs-to-Zensical migration preserving
Markdown/notebook workflows, then content/navigation redesign and verified
deployment changes. It is historical direction, not authorization to change
the reboot's documentation toolchain.
See [checkpoint boundaries](index.md#accepted-checkpoints-and-unfinished-work)
and [source corpus](evidence-index.md#source-corpus). These open questions do not
block this inventory; architectural adoption belongs to later explicit decisions.

[b-adr2]: https://github.com/DeloneCommons/pyvoro2/blob/db0884c641de0998d190de8aeee1d45154e46aff/docs/development/decisions/0002-weights-radii-and-gauge.md
[b-adr7]: https://github.com/DeloneCommons/pyvoro2/blob/db0884c641de0998d190de8aeee1d45154e46aff/docs/development/decisions/0007-separator-objective-contract.md
[b-adr8]: https://github.com/DeloneCommons/pyvoro2/blob/db0884c641de0998d190de8aeee1d45154e46aff/docs/development/decisions/0008-separator-solver-and-linear-backend.md
[b-adr9]: https://github.com/DeloneCommons/pyvoro2/blob/db0884c641de0998d190de8aeee1d45154e46aff/docs/development/decisions/0009-certified-scalar-proximal-solver.md
[b-adr10]: https://github.com/DeloneCommons/pyvoro2/blob/db0884c641de0998d190de8aeee1d45154e46aff/docs/development/decisions/0010-native-construction-preconditions.md
[b-adr11]: https://github.com/DeloneCommons/pyvoro2/blob/db0884c641de0998d190de8aeee1d45154e46aff/docs/development/decisions/0011-strict-input-and-ownership-contract.md
[b-adr12]: https://github.com/DeloneCommons/pyvoro2/blob/db0884c641de0998d190de8aeee1d45154e46aff/docs/development/decisions/0012-certified-periodic-image-geometry.md
[b-adr13]: https://github.com/DeloneCommons/pyvoro2/blob/db0884c641de0998d190de8aeee1d45154e46aff/docs/development/decisions/0013-central-generator-preparation-and-backend-safety.md
[b-adr14]: https://github.com/DeloneCommons/pyvoro2/blob/db0884c641de0998d190de8aeee1d45154e46aff/docs/development/decisions/0014-separator-observation-and-source-identity.md
[b-adr15]: https://github.com/DeloneCommons/pyvoro2/blob/db0884c641de0998d190de8aeee1d45154e46aff/docs/development/decisions/0015-atomic-separator-active-state.md
[b-adr16]: https://github.com/DeloneCommons/pyvoro2/blob/db0884c641de0998d190de8aeee1d45154e46aff/docs/development/decisions/0016-severity-complete-tessellation-diagnostics.md
[h-adr18]: https://github.com/DeloneCommons/pyvoro2/blob/b5e1aa6f53cd5edd94c0ae376047cb53278d2158/docs/development/decisions/0018-periodic-user-lattice-and-boundary-semantics.md
[h-adr19]: https://github.com/DeloneCommons/pyvoro2/blob/b5e1aa6f53cd5edd94c0ae376047cb53278d2158/docs/development/decisions/0019-separator-measurement-spaces-and-supported-realization.md
[h-adr20]: https://github.com/DeloneCommons/pyvoro2/blob/b5e1aa6f53cd5edd94c0ae376047cb53278d2158/docs/development/decisions/0020-exact-private-lattice-reduction.md
[h-adr21]: https://github.com/DeloneCommons/pyvoro2/blob/b5e1aa6f53cd5edd94c0ae376047cb53278d2158/docs/development/decisions/0021-wp5-native-occurrence-and-exact-face-certification.md
[h-adr22]: https://github.com/DeloneCommons/pyvoro2/blob/b5e1aa6f53cd5edd94c0ae376047cb53278d2158/docs/development/decisions/0022-wp6-source-certified-planar-edge-provenance.md
[h-adr23]: https://github.com/DeloneCommons/pyvoro2/blob/b5e1aa6f53cd5edd94c0ae376047cb53278d2158/docs/development/decisions/0023-wp7-certified-ghost-boundaries.md
[h-adr24]: https://github.com/DeloneCommons/pyvoro2/blob/b5e1aa6f53cd5edd94c0ae376047cb53278d2158/docs/development/decisions/0024-external-native-artifact-qualification.md
[h-adr25]: https://github.com/DeloneCommons/pyvoro2/blob/b5e1aa6f53cd5edd94c0ae376047cb53278d2158/docs/development/decisions/0025-native-occurrence-normalization-and-proof-assisted-identities.md
[h-adr26]: https://github.com/DeloneCommons/pyvoro2/blob/b5e1aa6f53cd5edd94c0ae376047cb53278d2158/docs/development/decisions/0026-separator-final-state-and-diagnostic-availability.md
[h-wp8]: https://github.com/DeloneCommons/pyvoro2/blob/b5e1aa6f53cd5edd94c0ae376047cb53278d2158/docs/development/wp8-implementation.md
[h-api]: https://github.com/DeloneCommons/pyvoro2/blob/b5e1aa6f53cd5edd94c0ae376047cb53278d2158/docs/development/api-inventory.md
[h-spatial]: https://github.com/DeloneCommons/pyvoro2/blob/b5e1aa6f53cd5edd94c0ae376047cb53278d2158/docs/development/spatial-degeneracy-regressions.md
[h-qualification]: https://github.com/DeloneCommons/pyvoro2/blob/b5e1aa6f53cd5edd94c0ae376047cb53278d2158/docs/development/native-qualification.md
[h-plan]: https://github.com/DeloneCommons/pyvoro2/blob/b5e1aa6f53cd5edd94c0ae376047cb53278d2158/docs/development/plans/v0.9.md
[h-shape]: https://github.com/DeloneCommons/pyvoro2/blob/b5e1aa6f53cd5edd94c0ae376047cb53278d2158/docs/development/audits/phase-c-row-wise-shape-refinement-review.md
[h-roadmap]: https://github.com/DeloneCommons/pyvoro2/blob/b5e1aa6f53cd5edd94c0ae376047cb53278d2158/docs/project/roadmap.md
[h-backlog]: https://github.com/DeloneCommons/pyvoro2/blob/b5e1aa6f53cd5edd94c0ae376047cb53278d2158/docs/development/review-notes/v0.9-post-functional-audit-backlog.md
[i19]: https://github.com/DeloneCommons/pyvoro2/issues/19
[i23]: https://github.com/DeloneCommons/pyvoro2/issues/23
[i60]: https://github.com/DeloneCommons/pyvoro2/issues/60
[i64]: https://github.com/DeloneCommons/pyvoro2/issues/64
[i65]: https://github.com/DeloneCommons/pyvoro2/issues/65
[i75]: https://github.com/DeloneCommons/pyvoro2/issues/75
[p67]: https://github.com/DeloneCommons/pyvoro2/pull/67
[p100]: https://github.com/DeloneCommons/pyvoro2/pull/100
[p115]: https://github.com/DeloneCommons/pyvoro2/pull/115
[p121]: https://github.com/DeloneCommons/pyvoro2/pull/121
