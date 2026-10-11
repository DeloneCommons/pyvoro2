# Numerical scale and mandatory generator separation: documentation idea

**Status:** proposed **user-facing explanation**, not yet public documentation or a changed numerical policy. Keep implementation observations separate from recommendations and unverified claims about upstream Voro++.

## The practical message

Voronoi/power tessellations are invariant under appropriate **mathematical** changes of units, but a finite-precision backend is not guaranteed to be numerically scale invariant. The current pyvoro2 v0.8-based implementation therefore refuses persistent-generator configurations that are too close for its fixed backend safety policy.

- The threshold is expressed in **input coordinate units**: native constants `backend_safety_distance = 1e-5` and `backend_safety_distance_squared = 1e-10`.
- The mandatory guard applies even with `duplicate_check='off'`, which disables only the optional user-threshold policy.
- Periodic comparisons use minimum-image separation. Temporary ghost generators are checked against permanent generators; ordinary `locate` query positions are **not** generator insertions and are not checked for duplicate proximity.
- Exact threshold decisions on public inputs and the conservative native lower-bound backstop are distinct: the native check may reject a just-safe near-boundary case if it cannot certify safety.
- A successful check does **not** guarantee geometrically correct output for arbitrary numerical scales, weights or ill-conditioned periodic bases.

Candidate introductory wording (subject to documentation review):

> pyvoro2 protects its Voro++ backend by rejecting generators whose separation cannot be established to exceed the fixed safety threshold, nominally `1e-5` in your coordinate units. If otherwise distinct generators are rejected solely because the numerical units are very small, consider **rescaling your entire geometry consistently** to a more numerically suitable range. Do not disable the mandatory safety guard. A passing preflight is not a guarantee of numerical accuracy.

See [backend contracts](../contracts/native-boundary.md) and [current constants](../../../cpp/native_preconditions.hpp#L24-L33).

## What must be scaled together

Under a positive uniform length rescaling `s > 0`:

| Quantity | Transformation |
| --- | --- |
| Generator coordinates, query coordinates, domain bounds, origin, lattice vectors | Multiply length-like coordinates by `s` (apply one consistent affine coordinate convention) |
| Backend radii / ghost radii | Multiply by `s` |
| Mathematical power weights `w = r²` | Multiply by `s²` |
| Returned lengths and areas / volumes | Scale by `s`, `s²` / `s³` respectively |

The ideal Voronoi/power ownership and topology are preserved under **consistent** scaling, subject to corresponding coordinate transformation. The numerical implementation may still have precision/overflow problems after rescaling; no universally optimal scale is claimed.

If the observed problem instead comes from **huge weight offsets relative to geometric squared lengths**, uniformly scaling coordinates and weights preserves their dimensionless ratio, so scaling alone does not solve the known large-weight numerical failure. The [large-weight investigation](../potential-work/large-weight-numerics.md) is separate. Extreme skew/conditioning may require a better lattice *representation*, not merely unit conversion.

## Helpful user-facing examples and caveats

- Explain `compute()` versus `locate()` versus `ghost_cells()`: close persistent generators raise for all three, while ordinary locate queries can coincide with sites.
- A two-site example separated by `1e-6` coordinate units can illustrate why a native safety error occurs; show that any rescaling example must transform the **domain** and, in power mode, radii/weights consistently.
- Explain that units are user-chosen and the floor is **absolute**, not automatically relative to the bounding box or lattice measure.
- Never suggest that all sites should have separation “of order one” as an explicit official Voro++ recommendation without locating and citing a matching upstream source. Such a statement has **not** been established in this review.
- Include `PeriodicCell`'s separate handedness/conditioning restrictions only as distinct input-policy topics, not consequences of the duplicate floor.
- Do not advertise this check as general robustness against squared-radius cancellation, pruning errors or nonreciprocal native faces.

Potential placement: future user guide under “numerical considerations / input scale”, linked from duplicate-safety error documentation. This is a documentation candidate only; no text has been moved into `docs/`.

Related: [Voro++ contract](../contracts/voro.md) · [geometry export notes](geometry-export.md).
