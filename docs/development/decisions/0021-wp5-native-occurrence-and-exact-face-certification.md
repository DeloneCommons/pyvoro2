# 0021 — WP5 native occurrence and exact periodic face certification

- **Status:** Accepted for WP5 implementation; not implemented
- **Date:** 2026-09-23
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

## Three independent authorities

1. **Exact semantic ideal.** Exactify the public input coordinates, lattice
   and mathematical weights (or the public mathematical interpretation of
   supplied radii). Construct the periodic ideal cell independently of the
   native output. In doubled source-local coordinates `y = 2(x - p_i)`, a
   candidate image has

   ```text
   d = p_j + s @ A - p_i
   Delta = w_i - w_j
   d · y <= ||d||² + Delta
   ```

   Here `A` has the public lattice rows and `s` is a public integer row
   shift. An equivalent undoubled form has `2 d · (x-p_i)` on the left.
   Classify the candidate's intersection with the **fully reconstructed ideal
   cell** by exact affine dimension: positive two-dimensional boundary,
   nonempty lower-dimensional contact (zero), or absent/redundant. A candidate
   plane intersecting an outer bound alone is not proof of an ideal face.

2. **Actual native occurrence.** The accepted private witness records each
   real applied cut or surviving seed origin, its owner and occurrence token,
   actual binary64 plane normal `n` and offset `h`, and final indexed native
   topology. It checks the entire final edge cycle and noninterference with
   the ordinary producer. An integer neighbor label, especially raw zero or a
   negative side code, cannot by itself classify a generator, self image, or
   wall. Preserve distinct occurrences even when their planes coincide.

3. **Producer-complete attribution.** For each actual occurrence, construct a
   finite set of periodic images whose *actual qualified binary64 source
   operations* could have yielded its observed `(n,h)`. Include the true
   generating image by a completeness argument, then resolve equivalence by
   both exact cut and semantic provenance. Do not filter candidates through
   ideal positivity to select the image. Only after attribution compare that
   image against the exact ideal. Source-compatible attribution and ideal
   semantic classification are separate proofs.

## Finite exact ideal reconstruction

For a source generator `i`, form a bounded exact outer polytope `P_i^0`
using the exact self-image slabs along periodic directions and exact real
walls along nonperiodic directions. In power mode the self-image weight
difference is zero. Choose an exact or outward-rigorous bound
`M >= max_{Y in P_i^0} ||Y||`; the maximum vertex 1-norm is a convenient exact
bound for the Euclidean norm. A candidate with `R=||d||` and
`q=R²+Delta` that touches or restricts this outer polytope necessarily has

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
into success. The ideal region is **not** the producer-attribution region.

## Producer-compatible attribution

The observed `(n,h)`, owner and occurrence kind constrain source arithmetic,
not the returned polygon's apparent area. Qualify binary64 rounding and
noncontraction under the shared ordinary/observer 3D build policy described
in the [witness record](../native-face-witness.md). Build exact rounding
preimages for each *associated source expression*, including both actual
standard and power offset expressions; do not algebraically reassociate
operations before reasoning about rounding. The construction covers:

- stored/insertion-remapped site operands and persistent owner mapping;
- `v_compute.cc` local and worklist cuts, `region_index` block displacement,
  periodic `qx,qy,qz`, and the final native x translation;
- rectangular periodic image routes in `container.hh`/`container.cc`;
- triclinic primary, side, vertical and wrap image routes in
  `container_prd.hh`/`container_prd.cc`, plus `unitcell.hh`/
  `unitcell.cc` self-image seed routes;
- `rad_option.hh` standard/power radius state, `r_scale` and
  `r_scale_check`, with the observed plane's *actual* offset expression.

Compose route-specific finite integer bounds with the exact rounding-preimage
constraints and enumerate **all** source-compatible candidates. The proof
must retain the actual generating route, including seed/self origins and
insertion remapping; a generic Cartesian tolerance box without a justified
producer predicate is insufficient. The private WP4 exact translation
enumerator can supply coefficient machinery, but its caller's box is not by
itself the WP5 final compatibility predicate. Do not introduce an arbitrary
coefficient cube, centroid or one-vertex match, nearest/best residual, minimum
coefficient norm or L1 shift, ideal-active-candidate filter, or empirical
error envelope. Distinct owner/shift/kind occurrences remain distinct unless
both exact cut and provenance coincide; multiplicity must be audited.

In a producer-compatible construction, invert each observed finite rounded
normal component through the exact nearest-even rounding interval (respecting
tie endpoints). Pull that interval backward through the actual source order,
for example `x2 = RN(x - qx)`,
`x1 = RN(p_image_x - x2)` on a worklist route, including every rounding in
the route's `put_image` construction. Intersect the resulting exact Cartesian
translation intervals with the route's block/index and insertion constraints;
map them to complete integer bounds by direct rectangular axes or the exact
reduced inverse-column method. Then evaluate the *actual associated*
`h` expression against the witnessed offset and keep all compatible
occurrences. Local, periodic block and seed routes have different expression
graphs; qualifying one graph does not license the others. This is a
proof procedure, not an instruction to approximate inversion with a
floating residual or to assume an unbounded arithmetic branch can be
silently ignored. An unsupported/nonfinite source arithmetic condition
prevents successful certification.

In particular, the power source expression
`RN(RN(D + S_i) - S_j)` can differ from
`RN(D + RN(S_i - S_j))`, where `RN` denotes the qualified binary64 rounding.
Even equal large radii at `r=2**27` exhibit this association effect when
their squared radius is exactly representable. Do not attribute it merely to
rounding of `(2**27 + 1)**2`. The exact equal-weight diagram is invariant to
a common weight shift, but native binary64 topology need not be gauge
invariant. Native radii and planes are producer evidence; they never redefine
public mathematical weight semantics.

## Classification and failure

| Outcome | Required evidence |
|---|---|
| `positive` | A unique producer-compatible image/provenance class has an exact ideal boundary of affine dimension two, with complete native coverage and compatible reverse coverage. |
| `zero` | A uniquely attributed occurrence has a nonempty exact ideal contact of dimension below two, irrespective of its native polygon's numerical area. |
| `unresolved` | Complete producer-compatible enumeration leaves distinct image/provenance classes, or another precisely identified occurrence/multiplicity ambiguity remains. |
| `inconsistent` | No compatible image, absent/redundant exact ideal boundary for the attributed image, missing required exact positive native coverage, invalid native occurrence/topology, or required representation conflict. |
| `resource failure` | A known complete mathematical/source region cannot be processed within an explicit candidate, exact-arithmetic or polytope-work limit. |
| `representation failure` | A mathematically established public coefficient cannot fit the required public integer representation. |

The existing million-candidate ceiling may be retained as a structured
resource refusal. Successful certification cannot use a searched prefix.
When certified shifts are requested, **any** failed complete certificate
fails the call atomically. Do not drop zero or absent native faces, patch
reciprocity, substitute an image or return partially certified metadata.
An ordinary non-certified call may still return ordinary native geometry if
its ordinary validation succeeds: a certification mismatch does not by
itself condemn the producer's whole tessellation.

A positive returned native polygon need not represent a positive exact ideal
boundary: characterized cases include exact zero contacts and strictly
redundant/absent candidates. The exactified returned binary64 vertex cycle
can even have affine rank three. Therefore its vector area, epsilon threshold,
and assumed exact coplanarity are not semantic measure tests. A private exact
projection onto its observed support may audit cycle structure, but it
neither replaces public native vertices nor becomes the exact ideal polygon.
Returned face measures/annotations remain numerical native descriptors.

## Chart, public output and reciprocity

For the source-centered chart, with original persistent site `p_i`, native
prepared site `a_i`, and exactly accumulated preparation translation `k_i`:

```text
p_i = a_i + k_i @ A
s_ij = sigma + k_i - k_j
p_j + s_ij @ A - p_i = a_j + sigma @ A - a_i
```

Use the qualified backend/user-frame transport as well as the exact integer
translation accounting; do not infer `s_ij` afterward by nearest geometry.
Public `cell['site']` and `result.sites` anchor the original input sites.
Real walls have wall identity and no image coefficient; persistent self
neighbors have the same owner with a nonzero shift.

`return_face_shifts=True` requires `return_faces=True` and a periodic
domain. The combination `return_face_shifts=True, return_faces=True,
return_vertices=False, return_adjacency=False` must work. Compute temporary
private certification geometry when needed and strip all unrequested public
geometry. On complete success, generator `adjacent_cell` is the persistent
owner and `adjacent_shift` is the exact public user-basis integer tuple.
Walls retain their wall identity and have no applicable shift (omit the
optional `adjacent_shift` key in the ordinary face schema). Preserve native
face vertices and their request-dependent indexing; do not claim that native
`face_properties` annotations are exact ideal boundary measures.

Separate raw native occurrence reciprocity, ideal semantic reciprocity and
coverage of the complete successful certificate. The semantic pairing is
`(i,j,s) <-> (j,i,-s)`, including distinct self pairs. Every exact
positive generator boundary in a complete certified result needs compatible
certified-positive native/semantic reverse coverage. A missing returned
volumetric reverse occurrence is a structural certification failure; a hidden
or lower-dimensional owner can still be genuine provenance of a native cut.
Do not fabricate reverse faces, redirect ownership, repair shifts or suppress
faces by epsilon. Real walls are exempt.

Normalization, capability flags and diagnostics must reflect *requested,
successfully certified public* geometry, never temporary witness arrays.
Keep ordinary diagnostics and native geometry validity distinct from the new
certificate failure, with an inspectable structured reason. Audit
`face_properties` and the existing separator realization consumer against
the original-site chart; do not redefine inverse observation semantics.
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
