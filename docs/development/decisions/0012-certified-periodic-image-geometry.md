# 0012 — Certified periodic nearest-image geometry

- **Status:** Accepted
- **Date:** 2026-08-09
- **Related issue:** [#40 — v0.8 R4: certify periodic nearest-image and minimum-image geometry](https://github.com/DeloneCommons/pyvoro2/issues/40)
- **Related decisions:** [ADR 0011](0011-strict-input-and-ownership-contract.md)
- **Related plan:** [v0.8 remediation execution plan](../plans/v0.8-remediation.md)

## Context

Separator observations may name one periodic image explicitly or ask pyvoro2
to infer the nearest image. The former triclinic inference searched only the
coefficient cube `[-image_search, image_search]**3`. With the default value
one, the accepted cell

```text
PeriodicCell.from_params(1, 1.5, 1, 0, 0, 1)
```

and points

```text
pi = (0.63696169, 0.26978671, 0.04097352)
pj = (0.01652764, 0.81327024, 0.91275558)
```

selected `(1, 0, -1)` at squared distance `0.455884497918507`. The true
minimum is `(2, -1, -1)` at squared distance `0.23935148791850697`. A warning
that the selected coefficient touched the search boundary did not certify or
correct the result. When periodic wrapping is enabled (`wrap=True`, or
`duplicate_wrap=True` in forward operations), periodic duplicate checks also
need the same mathematical distance for every candidate pair that they
evaluate.

Periodic image choice is a scientific semantic under the API lifecycle
policy. R4 therefore needs one private exact source of geometry without adding
a public lattice API or changing the established shift convention.

## Decision

### Exact problem and sign convention

Lattice vectors are rows of `A`. For the ordered pair `(pi, pj)`, a returned
integer row shift `s` applies to site `j` and the displacement is

```text
d = pj - pi
r(s) = d + s @ A
```

The certified problem minimizes the Euclidean norm of `r(s)` over allowed
integer shifts, using the exact dyadic rational values represented by every
supplied binary64 coordinate and lattice component. Candidate selection does
not use a rounded NumPy inverse or a larger floating search cube. The returned
float64 displacement and squared distance are rounded numerical views of the
exact rational result, and a private exact squared-distance key remains
available for threshold comparisons.

An explicit user-supplied shift continues to identify a particular image. It
is validated and used unchanged even when another image is nearer. Only rows
without a shift use nearest-image inference.

### One private implementation

`pyvoro2._internal.periodic_images` owns the scalar and batch mathematics. Its
successful batch result contains owned read-only shifts, displacements,
squared distances, exact distance keys, method and work metadata, exact tie
counts, and `certified=True`. It never returns an uncertified result. The
spatial and planar domain adapters are thin geometry providers for this
implementation.

Rectangular 2D and orthorhombic 3D lattices, including every partially
periodic axis mask, use an exact per-axis fast path. Each periodic coordinate
examines the exact floor and ceiling nearest to `-d/L`; each non-periodic
coordinate has shift zero. At most `2**k` combinations are exact-compared for
`k` periodic axes.

Fully periodic non-orthogonal 3D lattices use an exact finite enumeration. For
each pair, all coordinates and basis components are aligned to one denominator
`2**Q`:

```text
d = D / 2**Q
A = B / 2**Q
r(s) = (D + s @ B) / 2**Q
N(s) = sum((D + s @ B)**2)
```

Thus every candidate distance is compared as a Python integer `N(s)`. The
exact inverse of `A` is prepared from rational binary64 values. With
`q = d @ inverse(A)`, an incumbent `s0`, and

```text
U = ceil_sqrt(N(s0)) / 2**Q
C_l = sum_j(abs(inverse(A)[j, l]))
```

every global minimizer lies in the exact interval

```text
ceil(-q_l - U*C_l) <= s_l <= floor(-q_l + U*C_l).
```

The implementation checked-multiplies the interval widths, enumerates every
tuple in that proof-derived box, and exact-compares all `N(s)`. This box, not
the seed, is the certification authority.

### Public `image_search` compatibility meaning

The public `image_search` parameter remains an exact non-negative integer with
default one. It is a bounded incumbent-seeding hint only. The implementation
may inspect a deterministic neighborhood around an exact fractional-coordinate
seed to tighten the proof box, but this work cannot limit certification or
change a successful shift, displacement, or distance. There is no approximate
mode and no advice to increase `image_search` for correctness.

Seed work is capped at 4,096 candidates per pair. A larger requested
neighborhood is truncated deterministically and recorded in private metadata.

### Deterministic exact ties

Private callers supply an orientation token derived from stable ordered
endpoint keys. For all exact minimizers, orientation `+1` selects the
lexicographically smallest shift and orientation `-1` selects the largest.
For separator inference, those stable keys are the resolved internal site
indices within the fixed resolved problem. External ID values remain metadata
and do not participate in geometric tie selection. Duplicate candidate pairs
likewise use their internal point indices. Each caller uses `+1` when the first
endpoint index is smaller and `-1` when it is larger. Reversing a pair therefore
reverses the orientation, negates the selected shift and displacement, and
preserves the distance. Translating `pj` by `t @ A` changes the selected shift
by `-t`, while translating `pi` changes it by `+t`; both preserve the physical
displacement. The rule does not introduce a point-array permutation invariant
or a public tie mode.

### Resource failure and cache policy

The private resource limits are:

```text
seed candidates                  4,096 per pair
exact candidates             1,000,000 per pair
exact candidates             5,000,000 cumulative per batch
exact lattice-basis cache           128 entries
```

Candidate widths and products are checked before exact enumeration. A selected
shift must fit the signed-int64 shift contract. Exceeding a candidate budget,
an unsupported exact basis, an unrepresentable shift, or an unavailable finite
binary64 result view raises the private structured
`MinimumImageCertificationError` with stage, pair, basis/conditioning,
interval, candidate, and limit metadata as applicable. There is no approximate
fallback.

The bounded LRU cache stores basis-derived exact data only. Its key contains
the dimension, periodic-axis tuple, and exact binary64 bit patterns of all
lattice components. The cell origin is absent because it does not change the
displacement lattice. Pair results are not cached.

### Call-site and workstream boundary

Inferred separator rows consume the certified shift and numerical displacement;
explicit rows retain their requested shift. When periodic wrapping is enabled
(`wrap=True`, or `duplicate_wrap=True` in forward operations), spatial and
planar duplicate checks consume the same primitive and exact distance key for
each candidate pair they already evaluate. With wrapping disabled, the
established unwrapped Cartesian check is preserved.

R4 does not change face/edge image reconstruction, public duplicate modes or
thresholds, generator containment, the mandatory native-safety floor, or the
periodic duplicate candidate scanner. In particular, using the certified
distance does not claim that the current scanner finds every pair across every
periodic seam. Candidate-generation and mandatory safety independent of
`duplicate_wrap` remain R5.

## Consequences

- Every successful inferred minimum image is certified against the exact
  binary64-input problem, including the known `(2, -1, -1)` regression.
- `image_search` keeps its public signature and default while becoming
  correctness-neutral; only private work metadata and runtime may vary.
- Exact tie behavior is reproducible and respects lattice translation and pair
  reversal.
- Accepted highly skewed cells may fail structurally when the conservative
  proof box exceeds the frozen resource budget. They do not return a heuristic
  image.
- Ordinary orthogonal cases avoid rational inverse work, while representative
  triclinic batches expose deterministic candidate counts for performance
  qualification.
- No mandatory dependency, public result field, public minimum-image class, or
  native/backend change is introduced.

## Alternatives considered

### Increase the coefficient cube

Rejected. No fixed heuristic radius certifies the closest image for all
accepted bases, and a boundary warning still permits a scientifically wrong
result.

### Use floating QR or a rounded inverse as proof authority

Rejected. Such a search can be a useful seed, but floating rounding does not
prove completeness for the exact binary64 values supplied by the caller.

### Ignore or remove `image_search`

Rejected. Keeping it as a bounded performance hint preserves the public
signature and default without retaining its former correctness dependence.

### Replace explicit shifts with nearer images

Rejected. A supplied shift identifies a requested periodic image, which is a
different contract from nearest-image inference and later realization checks.

### Complete duplicate safety and seam scanning in R4

Rejected. Exact pair distance is shared geometry. Mandatory backend safety,
containment, candidate-generation completeness, and public duplicate policy
are separately reviewable R5 decisions.
