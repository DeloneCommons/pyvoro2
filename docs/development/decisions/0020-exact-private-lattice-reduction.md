# 0020 — Exact private rank-3 lattice reduction

- **Status:** Accepted
- **Date:** 2026-09-17
- **Related issues:** [#56 — WP3 exact private lattice-basis reduction](https://github.com/DeloneCommons/pyvoro2/issues/56),
  [#47 — v0.9 functional implementation](https://github.com/DeloneCommons/pyvoro2/issues/47)
- **Related plan:** [active v0.9.0 development plan](../plans/v0.9.md)
- **Related decisions:** [ADR 0012](0012-certified-periodic-image-geometry.md),
  [ADR 0013](0013-central-generator-preparation-and-backend-safety.md), and
  [ADR 0018](0018-periodic-user-lattice-and-boundary-semantics.md)

## Context

The existing exact triclinic minimum-image proof is complete once it enumerates
its proved coefficient box, but poor unimodular representations can make that
box impractically large. The R5 duplicate-scanning bucket bounds are sensitive
to the same inverse-column norms. WP3 supplies the exact private reduction and
coefficient-mapping primitive needed by the later WP4 consumer work; it does
not change those consumers itself.

Planning compared exact rank-3 strategies against the existing proof formulas.
Pairwise reduction can stall despite a three-vector cancellation. Exact LLL
gave a compact reducedness certificate and removed the tested
representation-induced workloads. Stronger LLL parameters did not materially
improve those workloads, while Selling/Minkowski strengthening or local search
would add output-selection and certification obligations. The selected
production method is therefore exact LLL only, not a comparison framework or a
crystallographic canonicalizer.

## Decision

### Exact row-basis and coefficient conventions

For ordered user lattice rows `A`, the private reducer returns exact objects

```text
B = U @ A
U @ U_inverse = I
U_inverse @ U = I
det(U) in {-1, +1}.
```

Integer coefficient rows map without a transpose:

```text
s_user = s_reduced @ U
s_reduced = s_user @ U_inverse
s_reduced @ B = s_user @ A.
```

The mappings use arbitrary-precision Python integers. A future public consumer
must map to user coefficients before applying the public signed-int64 contract.
In particular, a reduced coefficient larger than signed int64 may cancel to a
small user shift.

The ordered source binary64 bits are interpreted directly as exact dyadic
rationals. A common power-of-two alignment permits integer reduction work, but
the authoritative result remains exact rational rows. Derived rows never
round-trip through binary64; for example, `1 - 2**-55` remains distinct from
the binary64 value `1.0`. Gram data and Gram--Schmidt coefficients are exact
integer/rational working values, not a second lattice authority.

### Deterministic exact LLL policy

The private policy is fixed as follows:

- rank is exactly three and `delta` is exactly `3/4`;
- start in source row order with `U = I`;
- perform full size reduction against earlier Gram--Schmidt rows in descending
  index order;
- round exact coefficients to the nearest integer with half ties toward zero,
  so exact `+/-1/2` makes no move;
- swap only on strict Lovasz violation; equality passes;
- after LLL, flip only those row signs needed to make the first nonzero exact
  Cartesian entry positive, applying every flip to `U`;
- do not sort, canonicalize, strengthen, make query-specific choices, or force
  `det(U) = +1`.

On aligned integer rows, the product of positive prefix Gram determinants is a
positive integer potential. Size reductions preserve it and every strict
Lovasz swap decreases it. Finite descending size-reduction passes and bounded
index movement therefore terminate. The step limit is a resource guard, not
the termination proof.

### Independent certification and structural limits

Normal successful execution recomputes the full certificate. The certificate
reconstructs `U @ A`, checks integer transforms, determinant and both inverse
products, rebuilds Gram--Schmidt data through an exact Gram-matrix route
separate from reducer state, checks positive squared norms, every size bound,
both Lovasz inequalities, the final signs, and coherent accounting. Certificate
failure is an invariant error, never a resource result.

Bit accounting applies to every explicitly materialized semantic integer and
normalized rational result, including the factors of compound Gram--Schmidt
terms and the square, subtraction, and product in each final Lovasz check
before later cancellation. Reducer-side rounding likewise observes its exact
remainder and selected integer. It does not claim to count temporary integers
used inside Python's `Fraction` normalization. Charged work is likewise a
deterministic formula-level score for fixed scalar arithmetic, comparison, and
row-update blocks, not a count of machine instructions or allocations.

The private defaults are one million charged operations, 100,000 LLL steps,
4,096 bits for each transform, 32,768 integer-operand bits, and 65,536 rational
numerator/denominator bits. Exhaustion reports method, policy, stage, resource,
observed value, limit, and bounded source-bit diagnostics. It returns no partial
or source-basis fallback. Successful immutable results alone enter a 128-entry
LRU cache keyed by ordered source bits, exact policy identity, and limits;
origin is irrelevant.

These limits are structural ceilings with substantial measured margin, not
validity tests. Across the frozen workload cohort below the maxima were 5
steps, 980 work, 81 transform bits, 81 inverse-transform bits, 181 integer
bits, and 138 rational bits. A separate deterministic 256-seed unimodular
composition qualification reached 22 steps, 3,270 work, 23/18 transform/
inverse bits, and 32/73 integer/rational bits. Tests also cover a much wider
exact exponent spread. A scale-only certificate witness reaches a 203-bit
rational denominator in the final Lovasz product and therefore fails
structurally under a 201-bit policy even though its aligned-integer reduction
uses tiny operands.

## Qualification method

The test-only evaluator imports no production exact-lattice, reduction,
periodic-image, or duplicate-scanning helper. It reconstructs source binary64
values with `Fraction.from_float`, uses direct cofactor inversion, and evaluates
the current proof-box and R5 bucket formulas arithmetically. It keeps physical
endpoints and one exact dyadic alignment fixed. Its default alignment derives
from the basis and both original endpoints, so cancellation in `pj - pi` cannot
discard endpoint scale. The qualification interface accepts only the frozen
`image_search` values zero and one rather than silently emulating a different
uncapped large-radius seed cube. Bounded tests cross-check its independent
arithmetic against the unchanged production triclinic preparation and bucket
formulas. For each seed it also uses the
better of the source/reduced seeded physical incumbents on both bases, isolating
the inverse-bound effect. Enormous boxes are counted from proved interval
widths rather than enumerated.

The decisive test-side reducedness certificate uses exact rank-3 Gram-minor
identities, while the supplementary policy oracle uses an exact projection
route and `divmod` rounding with explicit signed half cases. Determinants use a
Leibniz permutation sum; inverse evidence uses independently composed known
inverses and verifies both matrix products. A test-only, opt-in private trace
observes actual production size reductions, swaps, and sign changes without
adding normal trace collection: size reductions preserve
`Phi = Delta_1 * Delta_2`, and every recorded swap satisfies
`4 * Phi_after < 3 * Phi_before`, including a backtracking fixture. Exact
Lovasz equality records no swap.

The equivalent-composed cases are frozen exact integer unimodular images of the
cubic lattice. They are not floating matrix products rounded into supposed
equivalence. The seeded-random cohort composes elementary row operations from a
small fixed 64-bit xorshift generator while updating the known inverse after
each operation; source rows are converted to binary64 only after verifying
their exact integer construction.

### Existing proof workload

“Seeded” is the current proof formula with the stated `image_search`. “Shared”
holds the physical incumbent fixed. Every count is exact.

| Case | Seed | Seeded box: source -> reduced | Shared widths: source -> reduced | Shared box: source -> reduced |
|---|---:|---:|---|---:|
| `thin-3e-4` | 0 | 106,726,048 -> 3,270 | `(3,270, 3,270, 1)` -> `(3,270, 1, 1)` | 10,692,900 -> 3,270 |
| `thin-3e-4` | 1 | 10,692,900 -> 3,270 | `(3,270, 3,270, 1)` -> `(3,270, 1, 1)` | 10,692,900 -> 3,270 |
| `thin-1e-3` | 0 | 9,618,496 -> 980 | `(982, 980, 1)` -> `(980, 1, 1)` | 962,360 -> 980 |
| `thin-1e-3` | 1 | 962,360 -> 980 | `(982, 980, 1)` -> `(980, 1, 1)` | 962,360 -> 980 |
| `cubic-shear-2p8` | 0 | 1,816,657,920 -> 1 | `(257, 1, 1)` -> `(1, 1, 1)` | 257 -> 1 |
| `cubic-shear-2p8` | 1 | 1,760,452,600 -> 1 | `(257, 1, 1)` -> `(1, 1, 1)` | 257 -> 1 |
| `cubic-shear-2p32` | 0 | 143,556,623,567,053,834,980,748,007,458,461,450,240 -> 1 | `(4,294,967,297, 1, 1)` -> `(1, 1, 1)` | 4,294,967,297 -> 1 |
| `cubic-shear-2p32` | 1 | 143,556,623,299,658,786,612,703,861,550,149,009,400 -> 1 | `(4,294,967,297, 1, 1)` -> `(1, 1, 1)` | 4,294,967,297 -> 1 |
| `cubic-shear-2p53` | 0 | 2,776,788,940,479,535,401,660,178,245,223,751,849,551,546,373,983,965,658,516,291,584 -> 1 | `(9,007,199,254,740,993, 1, 1)` -> `(1, 1, 1)` | 9,007,199,254,740,993 -> 1 |
| `cubic-shear-2p53` | 1 | 2,776,788,940,479,532,935,376,165,249,325,594,582,278,284,282,852,180,644,037,918,712 -> 1 | `(9,007,199,254,740,993, 1, 1)` -> `(1, 1, 1)` | 9,007,199,254,740,993 -> 1 |
| `cubic-shear-2p80` | 0 | 901,119,530,779,133,941,010,400,279,216,016,767,268,608,665,671,634,971,267,240,754,779,844,943,081,539,703,742,681,243,975,680 -> 1 | `(1,208,925,819,614,629,174,706,177, 1, 1)` -> `(1, 1, 1)` | 1,208,925,819,614,629,174,706,177 -> 1 |
| `cubic-shear-2p80` | 1 | 901,119,530,779,133,941,010,394,316,107,173,140,221,496,322,051,887,036,855,234,343,125,241,478,676,122,347,572,058,451,345,400 -> 1 | `(1,208,925,819,614,629,174,706,177, 1, 1)` -> `(1, 1, 1)` | 1,208,925,819,614,629,174,706,177 -> 1 |
| `equivalent-composed-a` | 0 | 86,351,200 -> 8 | `(897, 27, 2)` -> `(2, 2, 2)` | 48,438 -> 8 |
| `equivalent-composed-a` | 1 | 66,311,622 -> 8 | `(897, 27, 2)` -> `(2, 2, 2)` | 48,438 -> 8 |
| `equivalent-composed-b` | 0 | 257,218,200 -> 8 | `(1,054, 22, 2)` -> `(2, 2, 2)` | 46,376 -> 8 |
| `equivalent-composed-b` | 1 | 75,021,200 -> 8 | `(1,054, 22, 2)` -> `(2, 2, 2)` | 46,376 -> 8 |
| `equivalent-composed-c` | 0 | 3,231,963 -> 1 | `(733, 12, 1)` -> `(1, 1, 1)` | 8,796 -> 1 |
| `equivalent-composed-c` | 1 | 1,328,910 -> 1 | `(733, 12, 1)` -> `(1, 1, 1)` | 8,796 -> 1 |
| `intrinsic-anisotropy` | 0 | 728,214,795 -> 46,341 | `(12, 1, 46,341)` -> `(46,341, 1, 1)` | 556,092 -> 46,341 |
| `intrinsic-anisotropy` | 1 | 331,967,811 -> 46,341 | `(12, 1, 46,341)` -> `(46,341, 1, 1)` | 556,092 -> 46,341 |
| `r5-sc-001` | 0 | 363,765 -> 2 | `(2,339, 1, 1)` -> `(2, 1, 1)` | 2,339 -> 2 |
| `r5-sc-001` | 1 | 355,989 -> 2 | `(2,339, 1, 1)` -> `(2, 1, 1)` | 2,339 -> 2 |

The frozen seeded-random cohort gives the following additional qualification:

| Composition seed | Proof seed | Seeded box: source -> reduced | Shared widths: source -> reduced | Shared box: source -> reduced |
|---:|---:|---:|---|---:|
| 0 | 0 | 64,537,200 -> 8 | `(79, 10, 2)` -> `(2, 2, 2)` | 1,580 -> 8 |
| 0 | 1 | 66,348 -> 8 | `(79, 10, 2)` -> `(2, 2, 2)` | 1,580 -> 8 |
| 1 | 0 | 6,725,203,976,616,300 -> 8 | `(210, 22, 1,409)` -> `(2, 2, 2)` | 6,509,580 -> 8 |
| 1 | 1 | 20,685,485,506,560 -> 8 | `(210, 22, 1,409)` -> `(2, 2, 2)` | 6,509,580 -> 8 |
| 56 | 0 | 13,000 -> 1 | `(13, 1, 1)` -> `(1, 1, 1)` | 13 -> 1 |
| 56 | 1 | 6,656 -> 1 | `(13, 1, 1)` -> `(1, 1, 1)` | 13 -> 1 |

Both thin regressions improve by more than 100x and finish below 4,096. All
cubic shears finish at one candidate independent of shear magnitude. The frozen
equivalent cohort finishes at most eight candidates and improves every
seeded over-budget case by more than 1,000x for both proof seeds; those six
checks are deliberately non-vacuous. Six thin-1e-3 rows total 5,880 reduced
candidates, below the existing cumulative batch budget. Shared-incumbent
counts remain separate evidence for the inverse-bound effect and are not used
as substitutes for seeded acceptance gates. Three seeded-random cases are also
over budget before reduction and meet the same non-vacuous 1,000x gate.
Intrinsic anisotropy remains separately reported: reduction removes
representation shear but cannot remove the physical `2**16` inverse scale.

### Inverse, bucket, length, and orthogonality diagnostics

The following display values are rounded only for readability; test assertions
and interval/bin decisions use exact fractions. Bucket tuples are exact counts.
Larger R5 bin counts mean finer certified sparse buckets; the product is not a
dense allocation.

| Case | inverse-column L1: source -> reduced | Bucket bins: source -> reduced | Gram diagonal: source -> reduced | off-diagonal L1: source -> reduced | Hadamard defect squared: source -> reduced |
|---|---|---|---|---:|---:|
| `thin-3e-4` | `(3334.33, 3333.33, 1)` -> `(3333.33, 1, 1)` | `(29, 29, 99,999)` -> `(29, 99,999, 99,999)` | `(1, 1, 1)` -> `(9e-8, 1, 1)` | 1 -> 0 | 1.1111e7 -> 1 |
| `thin-1e-3` | `(1001, 1000, 1)` -> `(1000, 1, 1)` | `(99, 99, 99,999)` -> `(99, 99,999, 99,999)` | `(1, 1, 1)` -> `(1e-6, 1, 1)` | 1 -> 0 | 1e6 -> 1 |
| `cubic-shear-2p8` | `(257, 1, 1)` -> `(1, 1, 1)` | `(389, 99,999, 99,999)` -> `(99,999, 99,999, 99,999)` | `(1, 65,537, 1)` -> `(1, 1, 1)` | 256 -> 0 | 65,537 -> 1 |
| `cubic-shear-2p32` | `(4.295e9, 1, 1)` -> `(1, 1, 1)` | `(1, 99,999, 99,999)` -> `(99,999, 99,999, 99,999)` | `(1, 1.8447e19, 1)` -> `(1, 1, 1)` | 4.295e9 -> 0 | 1.8447e19 -> 1 |
| `cubic-shear-2p53` | `(9.0072e15, 1, 1)` -> `(1, 1, 1)` | `(1, 99,999, 99,999)` -> `(99,999, 99,999, 99,999)` | `(1, 8.113e31, 1)` -> `(1, 1, 1)` | 9.0072e15 -> 0 | 8.113e31 -> 1 |
| `cubic-shear-2p80` | `(1.2089e24, 1, 1)` -> `(1, 1, 1)` | `(1, 99,999, 99,999)` -> `(99,999, 99,999, 99,999)` | `(1, 1.4615e48, 1)` -> `(1, 1, 1)` | 1.2089e24 -> 0 | 1.4615e48 -> 1 |
| `equivalent-composed-a` | `(598, 18, 1)` -> `(1, 1, 1)` | `(167, 5,555, 99,999)` -> `(99,999, 99,999, 99,999)` | `(1, 1,025, 731)` -> `(1, 1, 1)` | 708 -> 0 | 749,275 -> 1 |
| `equivalent-composed-b` | `(702, 14, 1)` -> `(1, 1, 1)` | `(142, 7,142, 99,999)` -> `(99,999, 99,999, 99,999)` | `(1, 2,305, 1,011)` -> `(1, 1, 1)` | 1,456 -> 0 | 2.3304e6 -> 1 |
| `equivalent-composed-c` | `(732, 12, 1)` -> `(1, 1, 1)` | `(136, 8,333, 99,999)` -> `(99,999, 99,999, 99,999)` | `(1, 4,097, 1,491)` -> `(1, 1, 1)` | 2,480 -> 0 | 6.1086e6 -> 1 |
| `intrinsic-anisotropy` | `(17, 1, 65,536)` -> `(65,536, 1, 1)` | `(5,882, 99,999, 1)` -> `(1, 99,999, 99,999)` | `(1, 257, 2.3283e-10)` -> `(2.3283e-10, 1, 1)` | 16 -> 0 | 257 -> 1 |
| `r5-sc-001` | `(25,000, .751891, .000167054)` -> `(19.4659, .473952, .000167054)` | `(3, 132,997, 598,610,259)` -> `(5,137, 210,991, 598,610,259)` | `(.00268896, 2.9648e6, 6.6738e7)` -> `(.00268896, 4.45251, 3.5833e7)` | 7.4147e6 -> .51328 | 1.2403e6 -> 1.00009 |
| `random-unimodular-seed-0` | `(53, 7, 1)` -> `(1, 1, 1)` | `(1,886, 14,285, 99,999)` -> `(99,999, 99,999, 99,999)` | `(37, 1,730, 926)` -> `(1, 1, 1)` | 1,703 -> 0 | 5.9273e7 -> 1 |
| `random-unimodular-seed-1` | `(140, 14, 939)` -> `(1, 1, 1)` | `(714, 7,142, 106)` -> `(99,999, 99,999, 99,999)` | `(19,618,209, 259,990, 433,235)` -> `(1, 1, 1)` | 5,509,406 -> 0 | 2.2097e18 -> 1 |
| `random-unimodular-seed-56` | `(13, 1, 1)` -> `(1, 1, 1)` | `(7,692, 99,999, 99,999)` -> `(99,999, 99,999, 99,999)` | `(1, 1, 145)` -> `(1, 1, 1)` | 12 -> 0 | 145 -> 1 |

For R5-SC-001 the exact product of inverse-column L1 bounds improves by
approximately 2,037.44x, exceeding the 100x gate. The corresponding exact bin
tuple changes on the first two axes as shown; this is reported rather than
collapsed into a scalar score.

### Transform and charged-work diagnostics

| Case | max abs `U` / `U_inverse` | bits `U` / `U_inverse` | integer / rational bits | steps / swaps / reductions | work / certificate |
|---|---:|---:|---:|---:|---:|
| `thin-3e-4` | 1 / 1 | 1 / 1 | 181 / 131 | 3 / 1 / 1 | 725 / 251 |
| `thin-1e-3` | 1 / 1 | 1 / 1 | 171 / 123 | 3 / 1 / 1 | 725 / 251 |
| `cubic-shear-2p8` | 256 / 256 | 9 / 9 | 9 / 9 | 2 / 0 / 1 | 622 / 251 |
| `cubic-shear-2p32` | 4,294,967,296 / 4,294,967,296 | 33 / 33 | 33 / 33 | 2 / 0 / 1 | 622 / 251 |
| `cubic-shear-2p53` | 9,007,199,254,740,992 / 9,007,199,254,740,992 | 54 / 54 | 54 / 54 | 2 / 0 / 1 | 622 / 251 |
| `cubic-shear-2p80` | 1,208,925,819,614,629,174,706,176 / 1,208,925,819,614,629,174,706,176 | 81 / 81 | 81 / 81 | 2 / 0 / 1 | 622 / 251 |
| `equivalent-composed-a` | 565 / 32 | 10 / 6 | 10 / 10 | 2 / 0 / 3 | 658 / 251 |
| `equivalent-composed-b` | 653 / 48 | 10 / 6 | 10 / 10 | 2 / 0 / 3 | 664 / 251 |
| `equivalent-composed-c` | 667 / 64 | 10 / 7 | 10 / 10 | 2 / 0 / 3 | 664 / 251 |
| `intrinsic-anisotropy` | 16 / 16 | 5 / 5 | 33 / 37 | 5 / 2 / 1 | 980 / 251 |
| `r5-sc-001` | 55,336,879 / 83,759 | 26 / 17 | 178 / 138 | 2 / 0 / 3 | 664 / 251 |
| `random-unimodular-seed-0` | 41 / 41 | 8 / 6 | 9 / 13 | 3 / 1 / 4 | 779 / 251 |
| `random-unimodular-seed-1` | 660 / 4,300 | 10 / 13 | 20 / 51 | 21 / 14 / 19 | 3,154 / 251 |
| `random-unimodular-seed-56` | 12 / 12 | 4 / 4 | 4 / 4 | 2 / 0 / 1 | 628 / 251 |

## Consequences and scope boundary

WP3 creates exact private machinery and measured evidence only. It does not
route Voro++ or any current proof consumer through `B`; alter
`PeriodicCell.vectors`, fractional coordinates, handedness/order, public
shifts, backend frames or snapshots, or prepared generators; change physical
tie decisions; or turn existing unreduced-consumer resource failures into
successes. Those integrations remain WP4 work.

Because the primitive is not yet a public consumer path, this decision makes no
claim of current public speed or geometry improvement and adds no changelog
entry. The measured gains concern the existing proof formulas, not merely
shorter vectors: the shared-incumbent rows isolate inverse-bound effects, while
the separate seeded rows expose incumbent-quality effects.

## Alternatives considered

### Pairwise/Gauss reduction

Rejected as production policy because exact pairwise half ties can leave a
three-vector cancellation unresolved.

### Stronger LLL delta or local strengthening

Rejected for WP3 because planning measurements did not justify a stronger
output contract or additional certification phase.

### Selling, Minkowski, Niggli, or query-specific selection

Rejected for WP3. They address different canonicalization/selection goals or
would require additional CVP, tie, and review obligations. They are not silent
fallbacks if this fixed policy reaches a resource limit.
