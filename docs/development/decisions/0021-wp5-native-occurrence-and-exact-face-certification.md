# 0021 — WP5 native occurrence and exact periodic face certification

- **Status:** Accepted for WP5 implementation; not implemented
- **Date:** 2026-09-23
- **Public-action amendment:** 2026-09-23 — native shift availability and
  exact E/S consistency have separate outcomes; G0-N/C/O remain closed.
- **Related issues:** [#68 — WP5](https://github.com/DeloneCommons/pyvoro2/issues/68),
  [#47 — v0.9 implementation](https://github.com/DeloneCommons/pyvoro2/issues/47)
- **Prerequisite:** [PR #69 — private native face witness](https://github.com/DeloneCommons/pyvoro2/pull/69)
- **Related decisions:** [ADR 0002](0002-weights-radii-and-gauge.md),
  [ADR 0012](0012-certified-periodic-image-geometry.md),
  [ADR 0018](0018-periodic-user-lattice-and-boundary-semantics.md),
  [ADR 0020](0020-exact-private-lattice-reduction.md)
- **Related plan:** [v0.9 WP5](../plans/v0.9.md)

## Status and scope

The independent WP5 mathematical gate is **G0-N CLOSED, G0-C CLOSED,
G0-O CLOSED**. Production WP5 implementation is authorized, but is pending.
Neither WP5 acceptance, issue #68 closure, Checkpoint B acceptance nor v0.9.0
readiness follows from this decision. This ADR is the detailed normative WP5
contract for **ordinary persistent 3D faces**; WP6 planar edges and WP7 ghost
boundaries have separate implementation gates. It supersedes the WP5-specific
envelope, native-area, zero-face deletion and optional-reciprocity assumptions
in earlier target documents. Existing source and tests still define current
implemented behavior until WP5 lands.

The reviewed starting point is `dev`
`e2e520e84619ce4300137a9b953cb9e6bddab785`, tree
`6f2241acbafab91517676d8e5f06a991eef90d22`. Characterization
`pyvoro2-WP5-G0N-characterization-e2e520e(1).zip` has SHA-256
`a33cf0450b2aa80594728f511edeb55ec1eb33994b9fdd54ca8de791545b1c72`.
The closing review checked the archive and internal hashes, ZIP integrity,
README, summary, scripts, raw packets, independent exact ideal derivations,
byte-identical reruns of all 15 characterized packets with the accepted wheel,
and focused native witness tests. This provenance describes evidence, not a
portable fixed native topology or a binary file to vendor.

## Three geometric objects and one attribution proof

All exact values below are exact rationals obtained from validated binary64
inputs, unless identified as integer coefficients. `RN` means one qualified
binary64 round-to-nearest, ties-to-even operation under the shared
ordinary/witness FP policy. Source expression association is part of the
contract.

**S — public semantic ideal.** Use original validated caller sites `p_i`,
public row lattice `A`, and public mathematical weights:

| Mode | Exact semantic weight `W_i^S` |
|---|---|
| standard | `0` |
| power with explicit `radii=` | `exact(actual supplied binary64 radius_i)^2` |
| power with `weights=` | `exact(validated caller binary64 weight_i)` |

For a public integer shift `s`, in doubled source-local coordinates
`y=2(x-p_i)`:

```text
d_S = p_j + s @ A - p_i
Delta_S = W_i^S - W_j^S
q_S = ||d_S||² + Delta_S
d_S · y <= q_S.
```

The square of a supplied radius is the **exact square of that binary64
operand**, not exactification of a rounded native `r*r`. S alone owns public
semantic positivity.

**E — native-effective exact ideal.** Use the actual witnessed/stored native
binary64 sites `b_i`, exactified native row lattice `L`, and `W_i^E=0`
in standard mode or `W_i^E=exact(actual backend binary64 radius_i)^2` in
power mode, including radii converted from `weights=`. For attributed native
coefficient `sigma`, in doubled native-local coordinates `Y`:

```text
d_E = b_j + sigma @ L - b_i
Delta_E = W_i^E - W_j^E
q_E = ||d_E||² + Delta_E
d_E · Y <= q_E.
```

E is the exact ideal represented by the actual native input data. It is not
the actual floating cut plane and does not replace `(n,h)`.

**N — actual applied native support/topology.** The accepted PR #69 witness
records each real cut or surviving seed origin, its owner and occurrence token,
the actual binary64 support `n · Y = h`, and final indexed native topology.
It checks the entire face edge cycle and noninterference with the ordinary
producer. Raw integer neighbor labels, especially zero and negative side
codes, do not establish generator/self/wall provenance. Distinct occurrences
remain distinct even when planes coincide.

**Attribution and joint rule.** For each N occurrence, construct the complete
finite set of periodic images whose *actual qualified binary64 source
operations* could have produced its `(n,h)`. Include the true image by the
one-sided theorem below. Do not use E or S positivity to select among
source-compatible images. Only a unique image/provenance class can supply a
public shift; compare that attributed native occurrence independently with
the full E and S ideals when diagnostics are requested. Native shift
attribution succeeds when its occurrence/provenance is trustworthy, its
complete producer-compatible set is known, exactly one non-equivalent
image/provenance class remains and the public shift is representable. A
successful **exact-consistency audit** additionally requires its E **and** S
contacts both to have exact affine dimension two, the exact projected native
observation audit, complete global coverage and required reciprocity. E/S
status disagreement (positive, zero or absent) is a structured representation
conflict in that audit; it does not invalidate an already attributed shift.
Neither
native radius/gauge arithmetic nor public weights are redefined to make
the other representation authoritative.

For either ideal, classify the attributed candidate's intersection with the
**fully reconstructed** corresponding cell by exact affine dimension:
positive two-dimensional boundary, nonempty lower-dimensional contact
(zero), or absent/redundant. Intersection with an outer bound alone proves
none of these statuses.

## Finite exact ideal reconstruction

For each ideal and source generator `i`, form a bounded exact outer polytope `P_i^0`
using the exact self-image slabs along periodic directions and exact real
walls along nonperiodic directions. In power mode the self-image weight
difference is zero. Choose an exact or outward-rigorous bound
`M >= max_{Y in P_i^0} ||Y||`; the maximum vertex 1-norm is a convenient exact
bound for the Euclidean norm. A candidate with `R=||d||` and
`q=R²+Delta` (using S or E consistently) that touches or restricts this
outer polytope necessarily has

```text
R² + Delta <= M R
R <= (M + sqrt(M² + 4 |Delta|)) / 2.
```

Include equality: it may be a zero-measure contact. Outward-bound the square
root rigorously. Convert the radius and source positions into complete exact
integer coefficient intervals; fix nonperiodic coefficients at zero. Use direct
axis intervals for orthorhombic partial/full periodicity. For full triclinic
periodicity, use the accepted private exact reduced basis, inverse-column
bounds and exact unimodular map back to the public basis. Enumerate the whole
proved region before exact polyhedral classification. A limit can refuse
enumeration after the complete region is known, never turn a searched prefix
into success. Reconstruct S and E independently. Their ideal candidate regions
are **not** the producer-attribution region.

## Producer-compatible attribution

The observed `(n,h)`, owner and occurrence kind constrain actual qualified
binary64 source arithmetic, not apparent returned polygon area. The
[witness record](../native-face-witness.md) specifies the shared noncontracting
ordinary/observer FP profile. The following exact preimage and route rules
give a **necessary-condition superset** of all generating histories. Final
forward replay of each complete source route, including the offset branch,
is the compatibility predicate. An unqualified FP profile is a failure.

### Binary64 preimage operators

For a finite binary64 `f`, let `B(f)` be its **closed rounding bin**. For
an ordinary finite interior value, its exact rational endpoints are midpoints
to the neighboring binary64 values. Include *both* endpoints even when
ties-to-even would reject one: the enclosure is conservative, and final
source replay resolves the tie. For numeric inversion of `+0` or `-0`,
use `[-2^-1075,+2^-1075]`; retain a signed-zero distinction whenever the
source expression/provenance requires it. At maximum finite `F`, use the
finite upper rounding threshold `F+2^970` (and its negative symmetric
threshold). No overflow branch is silently treated as finite.

For exact rational interval `I`, `P(I)` is the closed hull of `B(f)` for
all finite binary64 `f in I`. Find the first and last contained floats by
exact comparison; their bins determine the hull without iterating all
binary64 values. The governing implication is

```text
RN(x) in I  =>  x in P(I).
```

For a scalar `f`, `B(f)` is an interval and `P` also accepts that
interval. Interval additions/subtractions below use exact rational interval
arithmetic, and division by a positive source operand preserves endpoints.
All inferred integer intervals include both endpoints; final replay decides
exact ties. Empty interval means no candidate on that route.

### Source integer and insertion semantics

The actual `v_base.hh` / `c_loops.hh` operations for finite in-range
operands are

```text
Step(x) = trunc(x)-1 if x<0; otherwise trunc(x)
Div(a,N) = a/N by integer truncation if a>=0;
           -1 + trunc((a+1)/N) if a<0     (N>0)
Mod(a,N) = a%N if a>=0;
           N-1-(N-1-a)%N if a<0          (N>0).
```

`Step` deliberately differs from mathematical floor at a negative exact
integer. The qualified profile requires the producer's 32-bit signed
`int` width (check `sizeof(int)`/limits), and checks every relevant
intermediate sum/product and floating-to-integer cast before replay. A history
requiring signed overflow or out-of-range conversion is outside the
qualified producer profile, not another image candidate.

Replay actual preparation and insertion before reconstructing a face. For
rectangular `put_remap`, axis by axis in source x/y/z order, with `lower`
the axis lower bound, `N` the source block count, `boxk` its **block
width** and `ksp` the stored reciprocal block width:

```text
c = Step(RN(RN(x-lower)*ksp))
l = Mod(c,N)
x_stored = RN(x + RN(boxk*(l-c)))   # if periodic
kappa = (c-l)/N.
```

Nonperiodic axes check the source index range and do not acquire a periodic
`kappa`. `boxk` is not the whole period. The source-qualified stored
position and ID must match the witness, including the actual binary64
operands; an exact conceptual shift identity alone does not claim bitwise
equality of rounded coordinates.

For triclinic `put_locate_block`, perform the actual z -> y -> x sequence:
`k=Step(RN(z*zsp))`, `kappa_z=Div(k,nz)` if out of range, then
`z=RN(z-RN(kappa_z*bz))`,
`y=RN(y-RN(kappa_z*byz))`,
`x=RN(x-RN(kappa_z*bxz))`; next
`j=Step(RN(y*ysp))` and, if needed,
`kappa_y=Div(j,ny)`,
`y=RN(y-RN(kappa_y*by))`,
`x=RN(x-RN(kappa_y*bxy))`; finally
`c=Step(RN(x*xsp))` and, if needed,
`kappa_x=Div(c,nx)`,
`x=RN(x-RN(kappa_x*bx))`. Apply the source's conditional branches,
block updates and checked ranges. Accumulate the exact **integer**
removals in preparation/chart accounting; the replayed stored sites, IDs
and radii must match witnessed native storage. A mismatch is a source/profile
inconsistency, never a reason to choose a shift from nearest geometry.

### Rectangular particle, orthogonal seed and wall routes

`container.hh::region_index` restricts each periodic block translation
syntactically to `sigma_k in {-1,0,+1}` and each nonperiodic coefficient
to zero. This source bound is neither an arbitrary search cube nor an ideal
completeness bound. Replay the central direct cut and the periodic worklist
route separately. The periodic `region_index` operand is `q_k=0`,
`RN(upper_k-lower_k)` or the source negation of that rounded difference;
replay the actual source bounds and stored native sites:

```text
direct:     n_k = RN(b_j,k - b_i,k)
worklist:   n_k = RN(b_j,k - RN(b_i,k - q_k)).
```

Replay branch/block indices and exclude only the primary self particle that
`v_compute.cc` itself skips. Orthogonal initialized side support on a
periodic axis is trusted self-image provenance; a nonperiodic side is a real
wall, with its native side identity. Neither is a particle route or a
generator image chosen by a raw negative owner label.

### Triclinic particle routes

The exactified lower-triangular native row lattice and block operands are

```text
L = ((bx,0,0), (bxy,by,0), (bxz,byz,bz)); bx,by,bz > 0
N = nx
boxx = RN(bx/N)
xsp = RN(1/boxx).
```

For observed normal `n`, source `put_image` stores
`p_image,v=RN(b_j,v+dis_v)` and a worklist cut forms
`n_v=RN(p_image,v-b_i,v)` for `v=y,z`. Thus define the finite exact
necessary displacement intervals

```text
J_v = P(B(n_v)+b_i,v) - b_j,v,  v in {y,z}.
```

The direct primary block route has `dis=(0,0,0)` and is replayed
separately, **excluding** the own primary particle skipped by the producer.
For the remaining
routes, enumerate only the source-compatible integers below and replay the
actual block masks, primary source records, coordinate comparisons and
`put_image` operations. A generated image is copied from primary
storage, not recursively from generated images.

**Final native x translation.** The compute mask has `hx=2*N+1`, so its
admitted `ei` is `0..2*N`; the source primary `ci` is `0..N-1`.
Hence `qi=ci+(ei-N)` lies in `[-N,2*N-1]`, and
`v_x=Div(qi,N)` lies in `{-1,0,+1}`. `region_index` applies
`q=(RN(v_x*bx),0,0)` and adjusts the block index. This is a source
theorem, not a guessed neighborhood. For each image history:

```text
n = RN(p_image - RN(b_i-q)) componentwise
sigma = image_coefficient + (v_x,0,0).
```

**Side image.** The nonzero source y coefficient `beta=ima` gives
`dis_y=RN(beta*by)`. Retain exactly

```text
beta in Z ∩ (P(J_y)/by), beta != 0; di = 0,...,N-1.
qua   = di + Step(RN(RN(-beta*bxy)*xsp))
alpha = Div(qua,N)
fi    = qua-alpha*N
X     = RN(RN(beta*bxy)+RN(alpha*bx))
Y     = RN(beta*by).
```

Replay `create_side_image`'s source `switchx`, primary `fi/fijk`,
left/right tests, both neighboring output blocks and all x-wrap branches.
Specifically preserve the separately rounded source updates `RN(X+bx)`
and `RN(X-bx)`, the right-input block wrap `RN(X+bx)`, and
`RN(b_j,x+dis_x)` in `put_image`. Each emitted row carries the exact
native image coefficient induced by its source branch **and** its
source-produced binary64 displacement. Do not replace its arithmetic by
a single exact lattice sum.

**Vertical image.** The nonzero source z coefficient `k=ima` gives
`dis_z=RN(k*bz)` and finite exact bound

```text
k in Z ∩ (P(J_z)/bz), k != 0; T = RN(k*byz).
```

For the no-upper-y-wrap branch the source `Y0=RN(T+RN(beta*by))`
requires

```text
beta in Z ∩ (P(P(J_y)-T)/by).
```

For upper-y-wrap, source `Y=RN(Y0+by)` adds **one more** preimage stage:

```text
beta in Z ∩ (P(P(P(J_y)-by)-T)/by).
```

Both intervals include exact endpoints. For each retained `k,beta,di`
(`di=0..N-1`), replay the source x quotient in its actual association,
including each intermediate RN:

```text
qi0 = di + Step(source_ordered_RN((-k*bxz-beta*bxy)*xsp))
alpha0 = Div(qi0,N); fi0 = qi0-alpha0*N
X0 = source_ordered_RN(k*bxz+beta*bxy+alpha0*bx)
Y0 = RN(RN(k*byz)+RN(beta*by)); Z0 = RN(k*bz).
```

`source_ordered_RN` means the operation tree literally used by
`create_vertical_image`: in particular
`RN(RN(-k*bxz - RN(beta*bxy))*xsp)` for `qi0`, and
`RN(RN(RN(k*bxz)+RN(beta*bxy))+RN(alpha0*bx))` for `X0`.
These spellings are the qualified noncontracting source association, not
algebraic simplifications. Replay both left/right branches, switch
comparisons and x wraps.

For upper-y-wrap, the source recomputes `qi` using **beta+1** with the
different parenthesization `-(k*bxz+(beta+1)*bxy)*xsp`, then

```text
alpha1 = Div(qi1,N)
delta_alpha = alpha1-alpha0
correction = RN(bxy+RN(bx*delta_alpha))
X = RN(X0+correction)
XL = RN(XL0+correction)
XR = RN(XR0+correction)
Y = RN(Y0+by).
```

Replay the corresponding `switchx` subtraction, `switchy` and branch
comparisons as in `create_vertical_image`, then all up-left/up-right and
x-wrap variants. Do not use the final lattice coefficient to replace this
recomputation: that would lose the source-association theorem.

The producer particle family is finite: `J_y,J_z` are finite exact
rational intervals for finite witnessed support, `by,bz>0` bound their
integer quotients, `di` has `N` values, and the wrap/left/right/final-x
branch sets are finite. No image is recursively generated from another
image.

### Triclinic seed/self route

Use backwards rounding, rather than leaving a choice of seed enumeration
model. For each sign `chi in {-1,+1}`, set `m=chi*n` and retain

```text
k in Z ∩ P(B(m_z))/bz
j in Z ∩ P(P(B(m_y))-RN(k*byz))/by
i in Z ∩ P(P(P(B(m_x))-RN(k*bxz))-RN(j*bxy))/bx.
```

Replay `unitcell.cc::unit_voro_apply`'s actual source vector construction
`x=i*bx+j*bxy+k*bxz`, `y=j*by+k*byz`, `z=k*bz`,
then its `+/-` sign plane calls, offset and surviving checked seed token.
The semantic coefficient is `sigma=chi*(i,j,k)` and must be nonzero.
Construction-only bounds cannot survive as accepted face support. The
accepted witness's matched seed replay ties the surviving token to the
ordinary native seed; raw owner zero is not seed/self authority.

### Associated standard/power offset predicate

Given the fully replayed observed normal, the actual source squared-normal
association is

```text
D = RN(RN(RN(n_x*n_x)+RN(n_y*n_y))+RN(n_z*n_z))
S_i = RN(r_i*r_i); S_j = RN(r_j*r_j)

standard:            h = D
power r_scale:       h = RN(RN(D+S_i)-S_j)
power r_scale_check: h = RN(D+RN(S_i-S_j)).
```

Use the source's actual branch/overload association; a seed or wall uses its
own witnessed construction rather than an invented particle radius.
Compare **both** entire normal and actually permitted associated offset
with the witness. Retain both power expression histories if both remain
source-compatible and the witness does not identify the selected branch.
Never choose the numerically closer offset. The branch's culling decision
can be replayed as necessary source evidence; skipping a route by a
conservative bound must never exclude the actual applied cut.

### One-sided producer completeness theorem

**Theorem.** For every finite particle/seed/wall occurrence produced by a
qualified supported 3D execution, its actual image and provenance history
belongs to the finite producer-compatible set above.

**Proof.** The actual rectangular central/worklist or orthogonal
seed/wall route satisfies the syntactic block bounds and exact integer
replay. A triclinic particle is either primary, side or vertical; vertical
has the no-upper-wrap or upper-wrap source branch. For any actual rounded
normal, `RN(x) in I => x in P(I)` backwards through each associated
`put_image` and cut expression includes its actual `k` and `beta`.
The finite `di`, left/right/wrap tests and proven
`v_x in {-1,0,+1}` include its actual blocks. The separate seed/self
preimages include its source integers and sign. Preparation/insertion replay
includes the actual stored operand and remap. Final standard/power offset
replay retains the actually selected expression. Thus no actual qualified
source route is excluded. All route integer and branch sets are finite
before a resource cap. This is a *one-sided retention theorem*, not a claim
that every retained history actually ran.

Only **after** construction and replay may histories be quotiented, and only
when both their exact cut and semantic provenance (kind, owner, image
coefficient, chart) coincide. Occurrence multiplicity remains auditable.
If distinct image classes remain, classification is `unresolved`; the
only ideal-positive image must not be selected as a tie breaker. A complete
empty producer set is `inconsistent`; an incomplete capped enumeration
is a `resource failure`.

The WP4 exact translation enumerator may supply coefficient machinery, but
its caller's compatibility box is not WP5's final producer predicate. Never
use an arbitrary finite coefficient cube, centroid/one-vertex match, nearest
residual, minimum coefficient norm/L1 shift, ideal-positive filter, or
empirical error envelope.

In particular, the power source expression
`RN(RN(D + S_i) - S_j)` can differ from
`RN(D + RN(S_i - S_j))`, where `RN` denotes the qualified binary64 rounding.
Even equal large radii at `r=2**27` exhibit this association effect when
their squared radius is exactly representable. Do not attribute it merely to
rounding of `(2**27 + 1)**2`. The exact equal-weight diagram is invariant to
a common weight shift, but native binary64 topology need not be gauge
invariant. Native radii and planes are producer evidence; they never redefine
public mathematical weight semantics.

## Classification and public action

For an attributed label, classify the exact E and S **full-cell** contacts
independently. `zero` means a nonempty contact of affine dimension below
two; `absent` includes a strictly redundant cut. Native appearance does
not change either exact status.

| Observed N / producer / E / S | Native shift / exact-consistency audit |
|---|---|
| Unique compatible native occurrence, E positive, S positive | Return its shift; exact consistency passes locally only after projected-cycle audit and globally after required facet/cell and reciprocal coverage. |
| Unique occurrence, E zero, S zero | Return its shift; report `zero` as an exact-consistency finding even if native polygon area is positive. |
| Unique occurrence, S absent (whether E is absent or positive) | Return its shift; report absent/redundant contact or representation conflict as appropriate, regardless of native appearance. |
| Unique occurrence, E/S status disagreement among positive/zero/absent | Return its shift; report structured representation conflict without redefining public weights or native radii. |
| No native occurrence, S zero | No missing positive semantic facet solely from this contact. |
| No native occurrence, S positive | Report missing required positive native facet/cell coverage; retain other uniquely attributed shifts. |
| Multiple non-equivalent producer-compatible image/provenance classes | Hard attribution failure; do not choose the sole ideal-positive one. |
| Coincident support with different semantic provenance | Retain distinct classes and occurrences; if a requested face's class remains unresolved, fail attribution; never merge by plane equality alone. |
| Exactified native original cycle of affine rank three | Run projected native-support audit; rank three alone is not an error. |
| Hidden/lower-dimensional owner | Its native provenance can be real; missing required volumetric reverse coverage is an exact-consistency finding, not ownership repair. |

Every positive facet of the reconstructed S and E cells must have compatible
native occurrence coverage under its attributed source chart. Check incidence
and multiplicity: multiple native fragments/occurrences with one semantic
label cannot be silently merged; conflicting provenance remains unresolved.
Real nonperiodic walls retain wall identity and do not participate in
generator-image reciprocity.

### Exact projected-cycle observation audit

For a finite nonzero observed support `(n,h)` and each exactified doubled
native vertex `Y_v`, compute in exact rational arithmetic

```text
Z_v = Y_v - n*(n·Y_v-h)/(n·n).
```

`Z_v` lies on the **observed** support. It validates the observation,
not an exact native polygon, a public vertex, or either E/S ideal polygon.
Choose deterministically a nonzero component of `n` and drop that
coordinate to injectively identify the support plane with exact 2D
coordinates. Preserve the original cyclic occurrence order.

1. Collapse consecutive equal projected points (including the closing
   pair) and zero-length edges; if fewer than three distinct points or
   affine dimension below two remain, report a **collapsed observation**.
2. For each strictly between-collinear point on a straight edge, remove
   the redundant point. An opposed collinear turn/backtracking is
   **invalid**, not redundant. Do not remove a repeated nonconsecutive
   point to make a cycle valid.
3. Reject repeated nonconsecutive vertices, nonadjacent segment crossing
   or touching, nonadjacent overlapping edges, and malformed closure;
   adjacent edges may meet only at their common vertex.
4. Require a simple ordered boundary of a convex 2D polygon with
   nonzero exact oriented area. Every nonzero successive turn has one
   orientation; after redundant-point removal there are no opposed or
   collinear turns. Invalid cycles fail the audit rather than being fixed by
   a tolerance or reordering.

This private audit can pass when the **original** exactified binary64
vertices have affine rank three. Their projection corrects observational
nonplanarity for the cycle test only; it never changes returned vertices.
A projected cycle of dimension below two cannot support a positive exact-
consistency audit, but retains a uniquely source-attributed shift. Native
vector area, epsilon thresholds and exact coplanarity
of raw float vertices are not semantic measure tests. Exact semantic
measure is obtained solely from the corresponding E/S full ideal contacts.

### Structured reasons and resources

Use the repository's `TessellationIssue.code` uppercase-underscore style.
Exact-consistency failures may have `severity='error'`, making the completed
diagnostic not okay without unconditionally raising from `compute()`.
The following codes and mappings are the WP5 target contract; context
(`source_id`, occurrence token, route, candidate labels as applicable)
belongs in the issue examples/message without substituting a bare exception:

| Code | Distinct reason |
|---|---|
| `WP5_SOURCE_PROFILE_MISMATCH` | Native storage, source replay, witness noninterference or required FP/source invariant disagrees. |
| `WP5_IMAGE_UNRESOLVED` | Several non-equivalent producer-compatible image classes remain after complete enumeration. |
| `WP5_EXACT_ZERO` | Attributed E and S contacts are both exact zero. |
| `WP5_IDEAL_ABSENT` | Attributed E and S contacts are absent/redundant, including native-positive appearance. |
| `WP5_REPRESENTATION_CONFLICT` | E and S exact boundary statuses differ. |
| `WP5_NATIVE_CYCLE_INVALID` | Projected cycle is malformed, repeated, backtracking, self-touching/intersecting or nonconvex. |
| `WP5_NATIVE_CYCLE_COLLAPSED` | Projected observed cycle has affine dimension below two. |
| `WP5_OCCURRENCE_MULTIPLICITY` | Multiple fragments/occurrences for one semantic label fail unique coverage. |
| `WP5_PROVENANCE_COINCIDENT` | Coincident exact supports have unresolved distinct semantic provenance. |
| `WP5_POSITIVE_FACET_MISSING` | Exact positive S/E ideal facet lacks native coverage. |
| `WP5_OWNER_COVERAGE_MISSING` | Required returned volumetric owner/cell coverage is absent. |
| `WP5_RECIPROCAL_MISSING` | Required compatible reverse positive native/semantic coverage is missing. |
| `WP5_RESOURCE_LIMIT` | Producer attribution or exact semantic audit cannot complete within a guard; identify the stage in the issue context. |
| `WP5_SHIFT_REPRESENTATION` | Mathematically established public shift is outside public signed-int64 representation. |
| `WP5_UNSUPPORTED_FP_PROFILE` | Compiler/rounding/source profile is not qualified for the proof. |
| `WP5_NONFINITE_OUTPUT_VIEW` | Requested public coordinate/descriptor materialization is nonfinite. |
| `WP5_IMAGE_INCONSISTENT` | Completed producer enumeration has no compatible image. |

A malformed witness origin/topology uses `WP5_SOURCE_PROFILE_MISMATCH`;
its valid origin with an invalid *projected returned cycle* uses the cycle
codes. Different owners/coefficients on one support use
`WP5_PROVENANCE_COINCIDENT`, or `WP5_IMAGE_UNRESOLVED` if the ambiguity
is the image of one occurrence. Missing reverse volumetric coverage uses
`WP5_RECIPROCAL_MISSING`, even when hidden owner provenance is genuine.
No producer-compatible image uses `WP5_IMAGE_INCONSISTENT`; a compatible
image with S/E both absent uses `WP5_IDEAL_ABSENT`. When only one of E and S
is zero or absent, use `WP5_REPRESENTATION_CONFLICT` with both statuses in the
issue context. Resource and representation failures are separate from
inconsistency.

Every complete producer route's integer family is finite before limits.
Preflight the **whole** mathematical region, not its searched prefix. The
existing one-million candidate ceiling may refuse a known complete region;
additional exact bit/polytope guards may refuse work but cannot truncate a
successful proof. Completed empty producer set is inconsistent, completed
non-equivalent set unresolved, unfinished enumeration a resource failure.
Public integer overflow is representation failure; no smaller representable
coefficient may be substituted. Legacy finite search knobs never enter
these bounds.

Distinguish resources by stage. If the complete producer-compatible set cannot
be processed before a requested shift is uniquely established, the requested
shift output fails atomically: a searched prefix never establishes uniqueness.
If all requested shifts are known but independent E/S ideal enumeration or
exact geometry exceeds its budget, retain the shifts and report an incomplete
semantic audit with `WP5_RESOURCE_LIMIT` when diagnostics were requested. With
no diagnostic request the independent audit may be skipped. No zero/absent
native face is silently dropped, no reverse face fabricated, no shift repaired,
and no incomplete attribution presented as a known shift. An exact mismatch
alone does not declare the native producer's entire geometry invalid.

## Exact source-centered chart and frame bridge

Distinguish an exact conceptual chart from rounded stored native data:

```text
p_i = exactified original validated public site
K_i = exact accumulated integer user-lattice preparation/insertion translation
a_i = p_i - K_i @ A     # conceptual exact source-chart representative
b_i = actual witnessed/stored binary64 native site (exactified for proofs).
```

Consequently the public **integer** transport is exact:

```text
p_i = a_i + K_i @ A
s_ij = sigma + K_i - K_j
p_j + s_ij @ A - p_i = a_j + sigma @ A - a_i.
```

Let `Q` be the exactification of the accepted binary64 backend/user
frame transform, `o` its backend origin, and `L` the exactification
of the native row lattice. Define exact, calculable dyadic bridge defects

```text
epsilon_i = b_i - (a_i-o) @ Q
B = L - A @ Q
d = a_j + sigma @ A - a_i
(b_j+sigma @ L-b_i) - d @ Q
    = epsilon_j-epsilon_i + sigma @ B.
```

This identity retains the coefficient-amplified frame discrepancy.
`A @ Q == L`, `Q @ Q.T == I`, and
`b_i == (a_i-o) @ Q` are **not** assumed as exact equalities.
The defects are source-computable values, not tolerances and not a way to
choose `sigma`. The producer attribution establishes `sigma` first;
the exact `K_i,K_j` give the public `s_ij`. Public `cell['site']` and
`result.sites` anchor the original input `p_i`, never the rounded `b_i`.
Real walls have identity but no image coefficient; persistent self-neighbors
have the same owner and nonzero public shift.

## Public output, diagnostics and consumers

When `return_face_shifts=False`, every existing ordinary combination of
`return_faces`, `return_vertices`, and `return_adjacency` retains its
ordinary geometry/output behavior; a requested diagnostic still invokes
the relevant independent WP5 consistency audit for periodic 3D faces.
When `return_face_shifts=True`, validate
`return_faces=True` and a periodic domain **before native work**; both
Boolean values of `return_vertices` and `return_adjacency` are
supported. In particular, the ordered option tuple
`(return_face_shifts, return_faces, return_vertices, return_adjacency)
= (True, True, False, False)` must succeed when certification succeeds.
Temporary native observation/proof geometry is private and never
causes an unrequested public array or capability. Do not perform an
unrequested public-coordinate conversion merely to support an internal
proof if that conversion would add an avoidable representation failure.

On complete success a generator face has
`adjacent_cell = persistent owner` and `adjacent_shift = exact public
user-basis integer tuple`. A self-image may have the same source and owner
but must have nonzero shift. A wall retains its wall `adjacent_cell` and
omits the optional `adjacent_shift` key; a zero tuple is not a wall image.
Native face vertices and request-dependent indexing keep their ordinary
public meaning. `has_periodic_shifts` means all **requested** periodic shifts
on returned native generator faces were completely and uniquely source-
attributed and materialized, including the available-but-empty case. It
does not assert exact E/S positivity or full-tessellation consistency;
other capability flags reflect requested public fields. A result can have
`has_periodic_shifts == True` and `tessellation_diagnostics.ok == False`.
No new per-face certification-state field or separate capability flag is
required; requested diagnostics may identify problematic occurrences in
issues and existing analyzer-owned annotations.

The independent exact E/S/native consistency audit runs when
`return_diagnostics=True` or `tessellation_check != 'none'`, using the
existing `compute()` diagnostic lifecycle. With
`return_face_shifts=True, return_diagnostics=False, tessellation_check='none'`,
complete source-faithful shift attribution remains mandatory but the full
independent semantic audit may be skipped; running it must not change the
selected shifts. Exact zero/absent contacts, E/S status conflict, missing
positive ideal/native or required reciprocal coverage, projected-cycle
collapse/invalidity, and native/ideal topology mismatch are structured
findings in `TessellationDiagnostics`, not automatic shift-attribution
failures. Error-severity findings make `diagnostics.ok == False` under
ADR 0016's rule. Preserve ordinary diagnostic findings (volume closure,
hidden cells and severity policy), and honor the established
`tessellation_require_reciprocity` requirement for reciprocal coverage.

The existing `tessellation_check` value controls the action on the completed
diagnostic: `'none'` takes no diagnostic action (though
`return_diagnostics=True` still computes/returns diagnostics); `'diagnose'`
computes/attaches diagnostics without warning or exception; `'warn'` emits
the existing summary warning if `diagnostics.ok` is false; and `'raise'`
raises the existing `TessellationError` in that case. Structured output
attaches the diagnostic; raw `output='cells'` returns `(cells, diagnostics)`
only when `return_diagnostics=True`. No WP5-specific public strictness knob
is introduced.

If trustworthy occurrence/provenance or supported FP/source profile is
missing, source/insertion replay disagrees with witnessed storage, the
completed producer-compatible set is empty or has multiple non-equivalent
classes, attribution exhausts its resources before completion, or the
public shift cannot be represented, the requested shifts cannot be
truthfully produced. Such attribution failures remain hard failures
independent of `tessellation_check` and return no partly attributed result.
They use the existing spatial `TessellationError` mechanism with a
structured WP5 reason. A malformed *projected geometric cycle* alone is an
audit finding when occurrence attribution remains trustworthy.

The opt-in `annotate_face_properties` implementation currently uses
triangulated native float vertices: triangle magnitudes contribute to its
centroid, and the norm of the summed cross-product vector supplies its
`area`/normal. These are **numerical native surface descriptors**,
not exact E/S facet measures. When exactified returned vertices are
nonplanar, those triangulated quantities must not be described as a
best-fit *exact planar* face. Their `tol` and area must not decide
certification/topology. Keep the established public fields; no new
descriptor field is mandated merely by this documentation pass. An
explicitly requested nonfinite public view fails with
`WP5_NONFINITE_OUTPUT_VIEW`, while private exact work need not
materialize that view.

Separate raw native occurrence reciprocity, ideal semantic reciprocity and
complete-tessellation consistency. The semantic pairing is
`(i,j,s) <-> (j,i,-s)`, including distinct self pairs. Every exact
positive generator boundary needs compatible positive native/semantic reverse
coverage for a successful required audit. A missing returned volumetric
reverse occurrence is an error-severity consistency finding when reciprocity
is required, while any uniquely source-attributed native face retains its
shift. A hidden or lower-dimensional owner can still be genuine provenance
of a native cut.
Do not fabricate reverse faces, redirect ownership, repair shifts or suppress
faces by epsilon. Real walls are exempt.

Periodic 3D inverse/separator realization requires an explicitly completed,
successful exact E/S consistency audit before treating native topology as
scientifically realized geometry. It may inspect diagnostics or request an
equivalent internal strict check; default forward `compute()` is not made
strict by this consumer. Its scientific
boundary positivity and `boundary_measure` are determined from the exact
S ideal, with a documented finite numerical view when the existing report
field requires a float; the native triangulated face `area` is not that
measure. An audit failure is a structured realization failure, not
an empty realized adjacency, a successful fit, or convergence. Preserve the
existing atomic active-state behavior and unrelated nonperiodic inverse
behavior. The current realization implementation reads native face area;
WP5 production integration must replace that scientific authority for
periodic 3D without silently changing the public native descriptor.

`face_shift_search`, `face_shift_tol`, `validate_face_shifts` and
`repair_face_shifts` retain their existing input validation until WP9 but
are semantic no-ops for certified WP5, including a false validation flag or
true repair flag. They cannot alter success, failure, candidates or shifts.

## Rejected approaches and consequences

The earlier proposed returned-vertex envelope as the primary image and
positive-measure authority, native polygon-area positivity, automatic
zero-face deletion, optional reciprocity repair and backend-effective radii
as the exact public ideal are superseded **for WP5**. The characterization
showed why each can misclassify a native occurrence. The witness remains
private observation, not production certification. Implement WP5 in a
separate PR, test exact and producer-specific adversarial cases, then seek
independent mathematical/native/API acceptance before closing #68.
