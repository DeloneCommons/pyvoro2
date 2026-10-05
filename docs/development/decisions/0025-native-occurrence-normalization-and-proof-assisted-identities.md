# 0025 — Native occurrence normalization, proof-assisted identities, and exact contact scope

- **Status:** Accepted
- **Date:** 2026-10-01
- **Related issues:** [#95 — Checkpoint B degeneracy closure](https://github.com/DeloneCommons/pyvoro2/issues/95),
  [#97 — contract freeze](https://github.com/DeloneCommons/pyvoro2/issues/97),
  [#98 — planar proof-assisted normalization](https://github.com/DeloneCommons/pyvoro2/issues/98),
  [#99 — bounded spatial normalization](https://github.com/DeloneCommons/pyvoro2/issues/99)
- **Evidence:** [#94 — planar characterization](https://github.com/DeloneCommons/pyvoro2/issues/94),
  [#96 — degeneracy atlas](https://github.com/DeloneCommons/pyvoro2/issues/96),
  [accepted independent contract review](https://github.com/DeloneCommons/pyvoro2/issues/95#issuecomment-5941808745)
- **Related plan:** [v0.9 Checkpoint B](../plans/v0.9.md#integration-checkpoint-b-periodic-topology-and-metadata)
- **Related decisions:** [ADR 0012](0012-certified-periodic-image-geometry.md),
  [ADR 0018](0018-periodic-user-lattice-and-boundary-semantics.md),
  [ADR 0021](0021-wp5-native-occurrence-and-exact-face-certification.md),
  [ADR 0022](0022-wp6-source-certified-planar-edge-provenance.md),
  [ADR 0023](0023-wp7-certified-ghost-boundaries.md), and
  [ADR 0024](0024-external-native-artifact-qualification.md)

## Context and authority

The accepted WP5/WP6 producers distinguish native occurrence provenance from
exact geometry. The #94 characterization showed that the compute-owned WP6
certificate already proves identities needed by a degenerate planar square,
but ordinary numerical normalization loses that authority. The #96 atlas
established higher-way planar identities, distinct collapse/contact roles,
spatial ridge/point incidence, and coordinate-rounding counterexamples. It did
not establish a complete native-to-semantic vertex map for every case.

The independent mathematical/architecture review reached
**CHECKPOINT-B DEGENERACY CONTRACT: CLOSED**. The maintainer accepted that
decision as design authority for #97–#99. Its evidence baseline is `dev`
commit `ba9e58faaf313001cc75f7c924c996707fdd51a9`, tree
`888f41fbda688cb20efd65736df31fe40c1fc850`. The accepted review is
`checkpoint-b-degeneracy-contract-ba9e58f.md`, SHA-256
`c3623cf93f1aa5aefb2cdcc8f551b1cc2c8262be45ef4eeac5470dd19dc02c9b`;
its companion verification archive is
`checkpoint-b-degeneracy-review-verification-ba9e58f.zip`, SHA-256
`e2ee7bf339dfd42ed757b3a0e624416f0bef1bb3a2357e7f4a41ba8c98aa449a`.
The evidence issues and accepted-review link above identify their provenance.

This ADR freezes that bounded decision for ordinary persistent-cell
normalization. It does not claim that the current normalizers implement the
new proof-assisted path. #98 owns that planar remediation; #99 owns spatial
scope/regressions. ADRs 0021/0022 retain WP5/WP6 mathematical authority, ADR 0023
retains the separate ghost policy, and ADR 0024 retains qualification authority.
The contract freeze does not accept Checkpoint B or authorize Phase C.

## Decision: four distinct layers

**N is preserved provenance-bearing occurrence data; E and S remain separate
exact ideals; normalization is a raw-inventory quotient representation with
numerical coordinates and, where available, explicitly proven identity
refinements.**

> Normalized topology is not an exact-S mesh.
>
> Successful strict normalization validation is not an exact-S reconstruction
> certificate.

| Layer | Object and authority |
|---|---|
| **N — native occurrence inventory** | Actual cell-local native occurrences and their source/token/slot/cycle association, owner/image or wall identity, ordering and multiplicity. N records the producing execution, including retained artifacts. |
| **E — native-effective exact ideal** | Exact mathematical geometry of the actual backend-effective binary64 problem: stored sites, native periods/lattice/walls and exact squares of actual backend radii in power mode. E is distinct from the floating cut support and clipping history in N. |
| **S — public-semantic exact ideal** | Exact mathematical geometry of the validated public input semantics: original sites, user lattice/walls and mathematical weights; explicit radii contribute the exact square of the supplied binary64 operand. A rounded native radius square is not that exact weight. |
| **Normalized raw quotient** | Numerical global organization of local native representations, with local-to-global mappings and image-qualified incidence, augmented where available by explicitly proven identities. Its global boundary records account for raw occurrences, rather than materializing S's positive-boundary complex. |

Preserve each consumed N occurrence even when E or S classifies it as
nonpositive or lower-dimensional. Pooling a global boundary record must not
erase its local occurrence identity or mapping. Existing public orientation
transforms and selectors remain valid with their source associations preserved;
private tokens need not become public fields. Neither E nor S chooses N's
provenance, replaces an owner/image, deletes an artifact or fabricates a reverse
occurrence. Independent `id=-1` ghost batches remain outside ordinary partition
normalization.

Standalone helpers remain conservative numerical/raw-record utilities.
Compute-owned normalization may consume private certified information absent
from stripped dictionaries. Their grouping and validation outcomes need not
match after authority is stripped. Numerical pooling is not semantic identity
authority. A scientific consumer requiring exact positive S boundaries or
dimensioned contact strata must use the owning certificate's complete successful
audit; `global_edges`, `global_faces` and strict normalized success do not
supply those facts. The proved relation to S is partial, not a promised bijection.

## Typed relations and identity closure

Every proof relation retains its execution/snapshot identity, dimension,
source references, layer, chart/basis, authority kind and proof domain.
Identical coordinates or integer tuples in different domains are not one proof.

The following table governs **proof-backed vertex-equivalence closure**.
Existing numerical bookkeeping may pool representations under its weaker
contract, but cannot contribute links to a certified identity path.

| Relation | Authority/domain | May enter vertex-equivalence closure? |
|---|---|---|
| Certified vertex identity | An admitted S-linked local-slot predicate, or separately proved reciprocal endpoint membership/correspondence and exact transport in the same bound domain | **Yes**, after all proof obligations are discharged; check exact lift potentials and alternative paths. |
| N-local endpoint equality | Exact source-associated internal endpoint operands; a local N fact | **No automatic admission.** It is an artifact-eligibility input, not an S-linked identity by itself. |
| N/E membership | Exact membership of a source-associated N endpoint in the named E geometry/chart | **No by itself.** It supports an adapter's separately proved vertex identity. |
| E/S bridge | Exact source/frame/image equations and endpoint agreement in the named E and S charts | **No by itself.** A bridge is not an indiscriminate equivalence edge. |
| Ideal/lattice lift transport | Exact ideal orbit equation in its named layer, lattice and chart | **Only within that proved domain.** Transport must support an admitted identity before joining N slots into an S-linked class. |
| Native lattice orbit | Exact N/E orbit equation | **Only within N/E.** No automatic public-S orbit follows without its exact bridge; no new spatial quotient consumer is required. |
| Boundary incidence | Native cycle/endpoint association or separately proved exact ideal containment, with layer and dimension | **No.** Containment/incidence composition never becomes vertex identity. |
| Positive semantic boundary obligation | Complete positive S contact map, owning E audit, coverage and applicable reciprocal-class requirements; contact dimension d−1 | **No.** This is a coverage/validation predicate. |
| Lower-dimensional contact incidence | Exact E or S point/ridge dimension, endpoints and support | **No.** A ridge retains its distinct endpoint roles. |
| Native provenance | Qualified source/token/slot/owner/image/wall association | **No by labels alone.** Preserve and validate the source association. |
| Numerical coincidence | Rounded coordinates, bins, tolerance or proximity in a numerical view | **No.** Algorithmic pooling establishes no certified identity. |

Only admitted proof-backed vertex identities enter the certified closure.
Incidence, numerical coincidence, tolerance/proximity, owner labels alone and
zero contact measure alone do not. An adapter must discharge membership and
chart/transport obligations before emitting an identity.

For an admitted equation `X(u) = X(v) + s @ A`, compose lift coefficients with
Python integers and retain the named chart/proof domain. Check alternative
paths and cycles; enforce public signed-int64 representation only when a
public shift is materialized. A periodic boundary loop between two lifts of one
global ID is incidence, not an inconsistent identity cycle. Preserve the
backend-primary representative convention from ADR 0018.

Proven relations must participate before irreversible numerical pooling.
Numerical or incidence-only links must never complete a certified proof path.
Known distinct certified S anchors must remain separate, even when their float
rows coincide; otherwise refuse the required unrepresentable view.

## Planar predicates: exemption and alias are separate

### Artifact exemption

An eligible retained planar N occurrence may be exempt from positive-boundary
reciprocity only when existing WP6 authority proves:

- trustworthy same-execution source attribution;
- exact internal endpoint equality;
- a complete successful owning E/S audit for the consumed scope;
- consistently nonpositive exact contacts; and
- satisfied positive-boundary coverage obligations.

The raw occurrence and its mapping remain present and structurally checked.
This exemption does **not** establish that its endpoint slots represent an S
vertex, and does not authorize an S-linked alias.

### S-linked point alias

This stronger predicate additionally requires all of the following:

1. The outgoing edge has an exact source/slot/`next` association. Its internal
   doubled-local endpoints are exactly equal, not merely equal after public
   rounding.
2. The attributed E and S contacts are both singleton dimension-zero points;
   absent contacts, positive segments and lower-dimensional owner cells do not
   substitute for those contact points.
3. The exact internal endpoint agrees with each point contact in the proper
   source chart. Exact source/frame/image transport proves the common S point;
   no frame or lattice defect is assumed to vanish.
4. Complete successful audit authority for the consumed scope is live and bound
   to the actual normalized snapshot.
5. Any extension across cells has a unique source-associated reciprocal endpoint
   correspondence, with separately checked membership and exact lift transport.

N-only endpoint equality remains evidence, not another automatic global-ID
merging rule. A valid local relation can be used when other slots lack exact
membership; whole-case N→S bijectivity is not a precondition.

Use the existing audit lazily when compute-owned normalization needs this
authority, including when public diagnostics are disabled. Missing authority
cannot bypass the standalone ambiguity guard or waive a check. A separately
valid ordinary numerical view may still be returned under existing actions
without a proof-dependent repair. An operation actually requiring missing
authority refuses; a failed proof-assisted operation has no weaker heuristic
retry.

| Planar situation | Required disposition |
|---|---|
| Internally unequal N fragments at an E/S point | No alias or artifact exemption. Preserve `WP6_NONPOSITIVE_NATIVE_EDGE` and semantic refusal; a numerical view may remain limited or refuse. |
| Internally equal N collapse displaced from S | Keep the N equality fact. Eligible audited nonpositive artifacts may receive the exemption, but no S-linked alias without exact point agreement. |
| E-absent/S-point or E-positive/S-point | S status alone grants neither alias nor exemption. Preserve conflict and positive-coverage findings. |
| Tiny positive S boundary missing from N | Preserve missing coverage. Do not synthesize an edge, merge its endpoints or call it a point because its float length vanishes. |
| Public-rounded positive endpoints coincide | No collapse permission. Preserve proved distinctions or use structured representation refusal. |

## Bounded spatial scope for v0.9

Preserve WP5 N attribution, independent E/S audits, positive-face coverage,
observed-support cycle checks and user-basis transport. Exact positive-face,
ridge and point incidence remains meaningful without an N→S vertex map.
Spatial normalization remains a numerical raw view; its strict success may
coexist with different S vertex/edge/face counts.

No new certificate-aware spatial N→S vertex projection or generic collapsed-face
normalization engine is required for v0.9. Zero area does not imply that all
vertices of a contact are identical. Exact ridge incidence preserves its two
distinct endpoints; contact dimension alone cannot generate identity.

The finite atlas found no retained lower-dimensional native N face among 5,634
packet faces and 304 pilot faces. This is a bounded negative result, not an
impossibility theorem. If such an artifact occurs, existing WP5 zero/absent,
cycle-collapse/invalidity, conflict, coverage and refusal semantics apply.
Preserve attributed N records/shifts when the owning action permits a return.
A numerical representation may return if valid under its own contract;
unsupported stronger projection is unavailable. The normalizer invents no
collapse rule, witness/schema extension or cross-dimensional collapse theorem.

## Architecture and proof-context lifetime

Select the bounded hybrid: certificate/dimension-specific adapters establish
typed relations; minimal shared machinery consumes only already-proven
identities, proof bindings, exact integer lift potentials and protected-class
bookkeeping. Dimension-specific code interprets point/segment/face geometry
and validation obligations. Shared closure code never derives identity from
contact dimension. Deliver the small shared mechanism with the planar consumer
under #98; no speculative spatial adapter or separate framework project is
required. This ADR does not prescribe a context data structure.

Private compute-owned authority must bind to the actual consumed state,
including relevant:

- raw/provenance snapshot and source occurrence associations;
- external/internal ID mapping;
- mathematical weight/radius interpretation;
- domain, chart and lattice operands;
- preparation/insertion transport;
- audit scope and completeness;
- qualification identity; and
- normalized mappings authorized by the proof.

Binding must establish the relevant canonical values/bytes, not object identity,
mutable dictionary flags, `has_periodic_shifts` or `semantic_consistent` alone.
Check the fields consumed by each claim. Mutating bound normalized state or
validating under a mismatching domain invalidates authority. Editing separately
owned parent raw dictionaries does not itself alter an unchanged normalized
snapshot. Existing raw mutability, selector ownership and result capabilities
retain their meanings.

Public extraction, direct reconstruction, deepcopy and serialization preserve
public data according to existing contracts but do not create or renew live
proof authority. Copied/serialized normalized outputs are numerical-only under
this decision; result capability metadata remains availability metadata.
Retained stale authority must fail explicitly through existing diagnostics,
without silently falling back to coordinate heuristics or repairing the view.

## Strict-validation contract

`level="strict"` remains an **error action over enabled and applicable
validation checks**, not a command to reconstruct exact S. Both levels collect
the enabled checks; strict raises the existing ValueError-compatible
`NormalizationError` for error-severity findings. Warning/info-only findings
retain their owning severity. Example truncation does not truncate checks,
counts or severity.

For ordinary normalized views, each enabled check requires its consumed schema
and applicable representation consistency: mapping/index consistency,
source-cell ownership, local occurrence alignment, nonperiodic zero shifts,
periodic image-qualified incidence, shift consistency and reciprocal raw-class
compatibility. Retain numerical polygon/incidence/Euler sanity checks and their
documented applicability and severity. This is not an exhaustive independent
schema or geometric audit. Unavailable input required by an enabled applicable
check is reported, not counted as checked success.

For a boundary shift `s` from cell i to j, compare vertex lifts with
`t_i = t_j + s`, using Python-integer intermediates. Preserve self-image and
wall distinctions. For multi-occurrence boundary classes, compare class-level
unions of image-qualified vertex occurrences, not arbitrary one-to-one fragment
pairs. Preserve spatial class-union semantics. Do not require raw global counts
to equal S or use point artifacts to manufacture positive incidence.

For compute-owned proof-assisted planar views, additionally check the bound
obligations actually consumed:

- current snapshot/domain binding and complete successful audit scope;
- every admitted identity, exact lift potential and protected S separation;
- complete positive-class coverage, applicable reciprocity and raw occurrence
  mappings, rather than ideal existence alone;
- exemption from positive-neighbor/reverse-edge obligations only for eligible
  retained nonpositive artifacts; and
- absence of numerical or incidence-only links from certified proof paths.

Positive reciprocity applies to positive S classes with the accepted E
consistency requirements. Ordinary standalone views have no artifact-exemption
authority and retain conservative raw reciprocity. A stripped copy can
therefore fail standalone strict validation while its original bound view
passes; report that scope/lost authority through existing issue/messages, with
no new public certification flag. Missing a whole-case vertex map does not by
itself fail this weaker representation contract.

Strict success does **not** certify exact S projection, E=S coordinates or
topology, a whole-case N→S bijection, proven ideal membership of all native
endpoints, exact manifold/cell-complex reconstruction, perturbation
independence, checks the caller disabled, native provenance newly inferred from
public fields, or native artifact qualification.

## Failure, limitation and resource semantics

Keep existing WP5/WP6, tessellation and normalization error/diagnostic families;
add no public exception hierarchy. #98 owns focused normalization reasons for
stale proof context and contradictory identity paths inside that framework,
including the accepted target reasons `NORMALIZATION_PROOF_CONTEXT_STALE` and
`NORMALIZATION_IDENTITY_CONFLICT`. This documentation freeze does not implement
those reasons.

| Condition | Required action |
|---|---|
| Missing native source/profile/insertion authority | Preserve the owning hard atomic failure; normalization cannot rescue attribution. |
| Incomplete or resource-refused audit | No exemption/identity operation needing that audit; preserve the audit-stage finding and raw diagnostic action. A required stronger operation refuses, never certifies a prefix. |
| E/S status conflict | Preserve both statuses and the owning conflict finding. No affected alias/exemption or successful semantic consumer. |
| E/S geometry differs but the accepted audit succeeds | No automatic failure; prove each proposed vertex bridge separately. Audit success is not bijection. |
| Unavailable N→S membership | A numerical raw view may remain valid. No semantic identity assertion or stronger projection is offered. |
| Stale/mutated retained proof context | Explicit normalization error; no silent downgrade, coordinate fallback or repair. |
| Contradictory identity/lift paths or merged distinct proved S anchors | Explicit identity-conflict error; never pick a path or discard a contradiction. |
| Missing positive exact coverage, including a tiny positive boundary | Preserve the owning coverage error; semantic consumers refuse. Raw output follows owning actions; never synthesize missing N. |
| Internally unequal planar point refinement | No point-collapse permission; preserve the nonpositive-native-edge error and any numerical representation refusal. |
| Collapsed planar artifact without semantic identity | Retain N; eligibility for an exemption remains separate from identity. Do not force an alias. |
| Hypothetical lower-dimensional spatial N face | Use existing WP5 audit/cycle findings and owning raw actions; refuse unsupported stronger use, without a new collapse engine. |
| Public coordinates cannot encode a required distinction | Distinct IDs with coincident float rows are allowed; otherwise structured representation refusal, not semantic merge. |
| Unqualified changed consumer/artifact | Preserve component admission refusal. Passing topology checks or copying an old record cannot grant qualification. |

A numerical-only limitation is not automatically a hard failure. An operation
consuming missing/stale authority cannot be silently downgraded. The existing
`tessellation_check` none/diagnose/warn/raise actions remain unchanged for
audit-only findings; they do not disable proof preconditions.

## Evidence examples and acceptance boundary

These examples constrain wording and the later regression obligations; they
are established evidence, not new executions or current remediation success.

| Evidence case | Exact/observed fact | Required normalization meaning |
|---|---|---|
| #94 four-owner periodic square | Exact S has 4 vertices, 8 positive undirected edges and 4 faces. | #98's target has 4 global vertex classes, **12 raw normalized edge classes** and all **20 N occurrences**: 8 positive classes plus 4 retained point-artifact classes. It is not an 8-edge normalized raw mesh. |
| #96 symmetric five-/six-way central classes | All 45 / 105 central pair identities are proved. | Apply the supported central closures; do not infer universal whole-case exact projection. |
| #96 partial-periodic spatial ridge | Zero-area contact endpoints are `(1/2,1/2,0)` and `(1/2,1/2,1)`; nonperiodic z makes them distinct quotient vertices. | Preserve both endpoint roles; dimension-blind zero-contact collapse is invalid. This is an exact contact, not an observed retained zero-area N face. |
| #96 asymmetric power-five | Complete successful audit with E=29V/59E and S=28V/58E, both 35F/5C. | E/S audit success does not establish vertex bijection or E=S topology. |
| #96 binary64 collisions | Large planar translation or a spatial `2^-52` perturbation makes distinct exact semantic vertices/boundary endpoints share public rounded coordinates. | Numerical coordinate equality does not authorize semantic identity. |

The planar child must support all 36 #94 controls on the compute-owned path
with default strict checks, while preserving raw data and the existing direct/
transitive coincident-cycle standalone guards. Other atlas families require
explicit supported, numerical-only or refusal dispositions; the decision does
not require universal successful normalization or proof of every unavailable
whole-case identity. #99 protects the spatial ridge and numerical/S boundary
without weakening diagnostics to make every saved failure pass.

#97's source-controlled wording requires independent mathematical/architecture
review before merge; #98 cannot merge before that acceptance. #99 may be
prepared after #97 independently of #98, with final integration following #98.
#95 requires independently accepted/merged children, reconciled exact integrated
source/qualification/FULL-CI evidence and a final integrated Checkpoint-B
return-to-closure review. Phase C remains blocked until that review accepts the
merged result.

## Qualification and public-surface consequences

No new public API/schema, exact mesh result, certification flag, proof/collapse
callback or correctness override is required. Functions, selectors and public
raw boundary schemas retain their contracts. WP5/WP6 mathematics, WP7/WP8
semantics and Phase C scope are unchanged.

The historical #97 procedure separated mathematical contract acceptance,
implementation source approval,
installed artifact qualification and integrated Checkpoint-B acceptance.
Under ADR 0024, measure the final changed-path closure rather than assuming
documentation is excluded. Pure documentation outside the measured closure
preserves qualification inputs only when global source, schema, consumer,
policy and all six component identities match. An unexpected measured change
in #97 stops for reconciliation rather than issuing a new record. Later
consumer/test changes under #109 require explicit implementer-owned source
manifest refresh and affected artifact qualification, followed by one independent
final review of the complete PR and exact-head evidence; unchanged native
implementation alone is insufficient. Independent source review is no longer
an issuance input.

## Alternatives rejected and deferred work

- Coordinate/tolerance identity or owner-label-only merging: lacks semantic
  membership/transport authority and can erase proved distinctions.
- Universal "collapsed edge ⇒ merge endpoints": conflates N-local equality,
  exemption and S-linked identity; unequal and displaced point refinements
  have different dispositions.
- Zero-area 3D contact ⇒ merge all vertices: contradicted by the two-ended ridge.
- Treating every retained N boundary as positive S topology, deleting artifacts
  or fabricating reverses: destroys raw provenance and confuses inventory with
  exact positive-boundary obligations.
- A perturbation-selected canonical triangulation: direction-dependent
  resolutions do not define one degenerate native or semantic representation.
- Independent dimension-specific engines duplicating closure/lifetime logic:
  unnecessary drift risk; share only the small proved-identity mechanism.
- A generic semantic geometry/collapse engine or exact-S mesh reconstruction:
  exceeds the bounded v0.9 promise and obscures the WP5/WP6 proof boundaries.

Stronger exact semantic topology reconstruction, including incidence-first
global-complex work, is future work requiring its own promised object and
proof. It is not a v0.9 deliverable or an implementation obligation of this ADR.
