# 0022 — WP6 source-certified ordinary planar edge provenance

- **Status:** Accepted contract; branch implementation present, independent production acceptance pending
- **Date:** 2026-09-24
- **Related issues:** [#74 — WP6](https://github.com/DeloneCommons/pyvoro2/issues/74),
  [#47 — v0.9 implementation](https://github.com/DeloneCommons/pyvoro2/issues/47)
- **Related plan:** [v0.9 WP6](../plans/v0.9.md#wp6-certify-periodic-2d-edge-owners-and-image-shifts)
- **Related decisions:** [ADR 0002](0002-weights-radii-and-gauge.md),
  [ADR 0013](0013-central-generator-preparation-and-backend-safety.md),
  [ADR 0016](0016-severity-complete-tessellation-diagnostics.md),
  [ADR 0018](0018-periodic-user-lattice-and-boundary-semantics.md), and
  [ADR 0021](0021-wp5-native-occurrence-and-exact-face-certification.md)

## Context and accepted boundary

Ordinary planar edge reconstruction previously selected owner/image labels
through a finite image window, floating line residuals and tolerances, and
fallback interpretations of negative native labels. Those methods cannot
establish the origin of every returned native occurrence. A collapsed edge can
retain a real source label, and a coincident no-op particle cut can leave an
initialization label unchanged.

The independent return-to-closure review closed G1 final-occurrence/source
association, G2 unchanged-vendor observation and noninterference, G3 the
qualified source/build/evaluation boundary, and G4 actual insertion and public
transport. This ADR transfers that accepted contract into repository authority.
It supersedes only the ordinary planar reconstruction and unspecified planar
collapse policy in earlier target text. WP5's accepted 3D mathematics is
unchanged. WP7 ghost identity, WP8 query metadata, residual WP9 cleanup, oblique
planar domains and the Phase C inverse contract remain separate.

The reviewed source was commit
`a98ac0d76d9c40286feef117e0fceaf31bad686f`, tree
`34a6b6f23f07deec6da49dfa8d280f3cf39f7aa8`. The prerequisite archive
`wp6-prerequisite-a98ac0d-20260924.zip` contained 43,430,944 bytes and had
SHA-256 `2e3da34058353f0611bd5c1676872e045a0b8eb6376babb7d50c805252440618`.
The return-to-closure bundle has SHA-256
`fa031f497d27936a380baa72cb3741e4fa939021a5cdcbf528cea61af744e111`.
These identities name reviewed bytes, not an unverified duplicate filename.

The prerequisite compared 92 fixtures under GCC 13.3/CPython 3.12 and an
independent GCC 14.2/CPython 3.13 rebuild, both Linux x86_64. An independent
rational interval oracle checked 10,178 labeled contacts in 402 exact cells.
This is prerequisite evidence, not qualification of the permanent production
witness or a cross-platform support claim. Production acceptance, #74 closure,
WP6 completion and Checkpoint B acceptance remain pending.

## Decision

### Native occurrence provenance comes from the producing execution

The qualified ordinary routes are `container_2d`/`container_poly_2d`,
`voro_compute_2d`, `voronoicell_neighbor_2d`, and `c_loop_all_2d`. Linked
nonconvex, boundary-container, quadtree, `quad_march`, and subset-loop routes
are outside this contract. No explicit wall objects are added by the ordinary
binding. Ghost and locate operations do not inherit ordinary persistent-cell
certification.

A binding/private adapter creates tokens for initialization sides and actual
particle cut attempts. Every one of the seven ordinary particle-cut call sites
observes the actual `region_index` image context and the radius-hook candidate
block/slot/owner before delegating unchanged clipping arithmetic. Stock `ne`
set/copy, capacity growth and deletion compaction propagate the tokens with
outgoing-edge topology. Final slot `k` associates endpoints `[k, ed[2*k]]` with
`ne[k]`. Token scope is source-cell-local. No-op cuts preserve old provenance.
Any newly introduced whole-cell copy must preserve that association explicitly.

The compact permanent witness retains:

- schema, reviewed source contract, effective build profile, bounds, periods
  and periodic mask;
- each persistent input's insertion disposition, actual stored site/radius,
  original/prepared values, preparation and insertion translations, and ID map;
- each stored source's computed/hidden disposition;
- each final outgoing-edge occurrence's source association and initialization
  side or persistent owner plus native image coefficient;
- internal exact collapse evidence, or internal doubled-local endpoints that
  determine it without public-coordinate rounding.

Shared snapshots need not be duplicated per edge. Full cut journals, queue
traces, before/after topology and applied floating supports are characterization
tools, not mandatory runtime payload. Compactness must not replace association
with unchecked matching between separate executions. Token exhaustion,
malformed association or missing population cannot return a certified prefix.

Known initialization side codes are `-1` x-low, `-2` x-high, `-3` y-low and
`-4` y-high. The side and mask identify either the source's axial periodic self
image or a real nonperiodic wall. Integer sign alone is not the proof. Unknown
or untrusted tokens fail; they do not become fallback walls. A particle token
records retained native origin, not proof of a positive ideal edge.

### Native completeness and profile support are explicit

For a periodic axis with `n` primary blocks, current primary index `c` lies in
`[0,n-1]`, the mask coordinate `e` lies in `[0,2n]`, and initialization uses
index `n`. Both guarded worklist phases and queue seeds/expansions preserve
those bounds. The three `region_index` branches satisfy

```text
c + (e-n) = candidate_primary_block + sigma*n
sigma in {-1, 0, +1}.
```

Nonperiodic coefficients are zero; central cuts skip the primary self particle.
There are no recursively generated images. This proves that every actual
producer is included. It does not prove that every image is visited or that
native pruning produces the exact ideal cell. Direct witnessed attribution
does not need reverse plane matching or per-call producer-family enumeration.

Checked integer/block/allocation and selector assumptions exclude signed
overflow and invalid floating-to-integer conversion. The accepted cohorts use
32-bit native signed/unsigned integers, binary64 with `FLT_EVAL_METHOD == 0`,
round-to-nearest, gradual underflow, no fast-math or LTO/IPO, and x86_64 targets
without enabled FMA instructions. Their flags did not explicitly disable
contraction; nonfused evidence is therefore cohort-specific. Insertion replay
must use source `Step` semantics, which differ from mathematical floor at
negative exact integers.

Production binds the Python/native schema and reviewed source contract to the
actual effective build/evaluation profile. A self-reported hash or compiler
name alone is insufficient. Changed source, FMA, reassociation, extended
evaluation, integer width or rounding requires affected-proposition
qualification or explicit refusal. The two prerequisite Linux cohorts do not
certify other package platforms. A binding-only implementation does not trigger
D9; a newly necessary vendor edit does.

### Insertion and public chart use actual storage

Every persistent input must actually be inserted in both standard and power
mode. Omitted insertion is a hard backend-stage failure before hidden/deleted
cell interpretation. A successfully inserted power cell may subsequently be
hidden. Half-open input containment alone does not prove insertion: the
contained coordinate `nextafter(1, 0)` in a one-block `[-1,1]` axis can round to
the out-of-range insertion quotient.

Planar preparation uses componentwise quotient/floor and Cartesian seam
snapping; it is not WP2's exact fractional-wrap operation. Retain its actual
integer `k` as an arbitrary-precision Python integer, validate the native
insertion branch and actual stored operands,
and retain the proven insertion translation `h`. Conditional default-preparation
`h=0` evidence does not remove those checks. The characterized nonzero `h=1`
case is a private `eps=0` stress, not a default-call history.

For original caller points `P`, public row lattice `A`, and native coefficient
`sigma`, transport uses arbitrary-precision integers:

```text
K_i = k_i + h_i
s_ij = sigma + K_i - K_j
boundary_image = P_j + s_ij @ A.
```

Private `k`, `K` and transported `s` have no signed-int64 gate. Large common
translations may cancel in a representable final edge shift. Only a required
public integer view enforces signed int64. Individual `K` values fitting does
not prove that transported `s` fits. Failure cannot clamp,
wrap or select another image. Owner/wall classification does not require an
unrequested public shift to be materialized.

The ordinary-compute preparation path shares the existing numerical
quotient/multiply/snap arithmetic with `RectangularCell.remap_cart`; only its
private integer storage differs. Public remapping and ghost/locate integer
policies remain unchanged, with no new public control.

With actual stored native sites `a` and native basis `B`, exactifying each
operand separately gives

```text
epsilon_i = a_i - (P_i - K_i @ A)
d_E = a_j + sigma @ B - a_i
d_S = P_j + s_ij @ A - P_i
d_E - d_S = epsilon_j - epsilon_i + sigma @ (B-A).
```

Neither stored/prepared equality nor a zero bridge defect is assumed. `A` uses
`RectangularCell`'s actual binary64 span values, not an exact endpoint
subtraction substituted silently. Public persistent `site` and `result.sites`
reference original `P_i`; requested source-centered native vertices are formed
from internal local coordinates and that anchor. Subtracting rounded global
vertices cannot recover proof coordinates. Public vertices remain numerical
native views, not exact S vertices.

Certification uses internal indices before external-ID remapping. Valid unique
nonnegative signed-int64 external IDs only relabel output. Generator edges keep
`adjacent_cell` and requested user-basis `adjacent_shift`; self shifts are
nonzero. Real walls keep their existing side code and omit `adjacent_shift`.
No ordinary `boundary_reference` schema is introduced.

### N, E and S remain independent authorities

| Authority | Definition |
|---|---|
| N | Actual native occurrences, source provenance and topology. |
| E | Exact ideal of actual stored native sites, native periods/walls and exact squares of actual backend binary64 radii; zero weights in standard mode. |
| S | Exact ideal of original caller sites, public periods/walls and mathematical `weights=`; for `radii=`, exact squares of supplied binary64 radii; zero weights in standard mode. |

Exactifying a rounded floating radius square does not produce the E/S weight.
Mathematical weights do not change to fit backend arithmetic. Exact common
weight shifts preserve geometry only when represented weight differences are
preserved; conversion and native source ordering can change E and N. Neither
ideal chooses, discards or repairs an attributed native image.

For either ideal, use its own source `g_i`, weights and basis consistently. In
local coordinates `z = X-g_i`, each image contributes

```text
d = g_j + s @ A - g_i
2*d·z <= d·d + w_i - w_j.
```

Self images bound periodic coordinates by `|z_l| <= L_l/2`; real walls bound
the remaining axes. For each owner choose an exact coefficient centering its
relative displacement in a half-period interval, then keep that coefficient
plus `{-1,0,+1}` on each periodic axis. Nonperiodic coefficients stay zero.
For an omitted positive coefficient `eta >= 2` versus coefficient one,

```text
(z-delta-eta*L)^2 - (z-delta-L)^2
    = (eta-1)*L*((eta+1)*L - 2*(z-delta)) > 0.
```

The negative side is analogous. Same-owner weights cancel, so all farther
images are strictly dominated on the bounded rectangle even in power mode.
This complete ideal family includes ties/point contacts and arbitrary translated
representatives; it is separate from the native `sigma` bound.

Reconstruct the complete bounded cell exactly. Distinguish a positive segment
on a full-dimensional cell, point contact, absent/redundant constraint,
lower-dimensional/empty cell, and identical-function/zero-normal degeneracy.
Preserve every coincident provenance label. A line test or contact with the
outer rectangle is insufficient. E/S contact statuses and complete coverage
are audited; equality of E/S vertex coordinates is not required.

### Raw multiplicity, collapse and semantic audit

An N occurrence is internally collapsed exactly when its internal endpoint
coordinates are equal; signed zeros are equal geometric coordinates. Public
rounding can separately collapse distinct internal endpoints. Numerical short
length thresholds establish neither kind of collapse nor ideal positivity.
Retain truthfully attributed raw occurrences, including artifacts and multiple
occurrences of one class. Different provenance on a coincident line stays
distinct.

A complete requested audit checks all E/S contacts, all positive provenance
classes, native coverage, degeneracy and required reciprocity:

| Finding | Policy |
|---|---|
| Extra internally collapsed N occurrence, consistently nonpositive E/S contact, otherwise complete positive coverage | Nonfatal artifact finding; not a positive semantic edge. |
| Noncollapsed N occurrence with nonpositive ideal contact | Error finding. |
| E/S contact-status disagreement | Error finding; preserve both meanings. |
| Missing positive E/S provenance, including a distinct coincident label | Error finding. |
| Positive provenance covered only by internally collapsed N occurrences | Error finding. |
| Invalid source association or untrusted occurrence | Hard attribution failure. |
| Incomplete exact audit | Resource/incompleteness finding; never successful consistency. |

Semantic reciprocity is `(i,j,s) <-> (j,i,-s)`, including opposite self images;
real walls are exempt. Check translated exact contact sets and complete class
coverage separately from native occurrence multiplicity. Do not require an
arbitrary one-to-one fragment pairing, select nearest fragments, pool distinct
provenance or mutate shifts. Required reciprocal failure is an error; optional
inspection keeps ADR 0016's nonfatal severity policy.

Retained line-offset and line-angle diagnostic tolerances may inspect numerical
unions of noncollapsed reciprocal native segments only after complete E/S
auditing and only for classes positive in both ideals. Exact native-local chart
translation precedes the numerical view; public vertices and public int64
shifts are unnecessary. This inspection cannot choose images or decide exact
positivity. Its separate work or representation refusal is an error finding,
not a failure of already established native attribution or a change to the
completed exact audit.

### Output, actions, resources and consumers

Complete attribution is mandatory whenever ordinary periodic owner-bearing
edges are returned or consumed, including `return_edge_shifts=False`. Calls
without an owner/image consumer need not construct unrequested proof geometry,
but insertion integrity remains mandatory. Attribution, source/profile,
insertion, necessary proof-resource and required public-representation failures
abort the requested provenance-bearing result atomically using the existing
planar structured failure mechanism, independently of `tessellation_check`.

`return_diagnostics=True` or `tessellation_check != 'none'` requests the
independent exact audit. `'none'` takes no diagnostic action; `'diagnose'`
attaches findings; `'warn'` warns when the completed diagnostic is not okay;
`'raise'` raises in the same case. Error issues make `ok=False` under ADR 0016.
An audit-only failure preserves attributed raw shifts when the requested action
permits a return. Strict semantic consumers require a complete successful audit.
Example truncation never truncates checks, counts or severity.

`has_periodic_shifts` means all requested generator-image metadata on returned
native edges is available, including available-but-empty output. It does not
assert E/S positivity. Proven walls need no shift. Private shifts stripped from
output do not set this capability. `return_vertices=False, return_edges=True,
return_edge_shifts=True` works with either adjacency setting. Invalid output
combinations fail before native work, and unrequested public-coordinate
materialization must not obstruct internal proof.

Keep native/witness, attribution, exact-audit, diagnostic-example and public
representation limits distinct. Complete mathematical families or proved upper
bounds are known before candidate guards apply. Runtime token/work/allocation
exhaustion cannot certify a prefix. Audit exhaustion after attribution need not
erase shifts, but fails strict semantic use. Private numerical ceilings are
engineering policy justified by planar workload evidence, not copied WP5 limits
or mathematical validity tests.

Result validation accepts proven planar walls without fake shifts. Normalization
preserves source/provenance and occurrence associations before geometric
pooling, or explicitly refuses a view unable to represent them. Its tolerances
are numerical-view controls. Diagnostics do not discard short edges as a
semantic test or reject every duplicate provenance key as an invalid fragment.
Raw orphan counts/annotations include harmless collapsed artifacts and do not
themselves assign severity or positivity. Numerical inspection refusal sets
`reciprocity_checked` and `ok_reciprocity` false without undoing a complete
exact audit. An unrepresentable normalization requested through compute is a
structured representation failure; standalone helpers retain `ValueError`.

Planar separator realization requires complete successful exact consistency and
consumes the complete positive S boundary-class set, not each raw N record.
Scientific length is derived once per exact S segment/class, with a checked
numerical view when requested. A positive exact segment whose length view
rounds to zero remains positive. Existing private realization/active paths carry
mathematical weights when backend radii are supplied separately. Failure cannot
become empty successful adjacency or an accepted partial active state. Numerical
edge descriptors remain descriptions of native output. Requested native
descriptor fields, including midpoint and tangent, must have finite numerical
views or raise a structured representation failure.

Standalone utilities given only mutable public dictionaries lack the original
weights, actual stored population and private witness. They retain their
documented numerical/raw-record scope and cannot claim full E/S or native
collapse certification. Result capabilities describe constructed availability,
not authenticity after arbitrary caller mutation.

### Immediate ordinary-compute lifecycle change

Remove `edge_shift_search`, `validate_edge_shifts`, `repair_edge_shifts`, and
`edge_shift_tol` from ordinary `pyvoro2.planar.compute` in WP6, without aliases,
ignored kwargs or no-op compatibility. Separator `image_search`, unrelated
diagnostic/normalization tolerances and valid resource/performance controls
remain. Planar ghost versions remain legacy WP7-owned controls; residual global
cleanup remains WP9-owned.

## Consequences and alternatives

The public raw record/result hierarchy stays intact while provenance becomes
source-grounded and semantic consistency becomes inspectable. Initial support
is bounded by explicit qualified profiles, with unsupported configurations
refused rather than silently inheriting a proof from a source hash.

Residual search, larger finite cubes, integer/owner ordering, E/S winner
selection and reciprocity repair are rejected because they can misattribute a
real native occurrence. Dropping artifacts or using epsilon positivity is
rejected because it erases the distinction between N and exact topology.
Keeping the full characterization journal is unnecessary: the compact witness
must retain the proof obligations, not every debugging trace. Reusing WP5's
3D reverse-plane proof or resource values by analogy is also rejected.

Independent production review must qualify the optimized witness, actual
source/build profiles, packed artifacts and narrow consumers. Prerequisite
closure alone does not mark this implementation or any later release gate
accepted.
