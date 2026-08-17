# 0013 — Central generator preparation and mandatory backend safety

- **Status:** Accepted
- **Date:** 2026-08-09
- **Related issue:** [#41 — v0.8 R5: centralize generator preparation and make native safety non-optional](https://github.com/DeloneCommons/pyvoro2/issues/41)
- **Related decisions:** [ADR 0010](0010-native-construction-preconditions.md), [ADR 0011](0011-strict-input-and-ownership-contract.md), [ADR 0012](0012-certified-periodic-image-geometry.md)
- **Related plan:** [v0.8 remediation execution plan](../plans/archive/v0.8-remediation.md)

## Context

Every forward operation constructs a native Voro++ container and inserts its
persistent generators. `ghost_cells` also inserts each query temporarily.
Before R5, these paths repeated validation and duplicate policy in their public
wrappers, periodic preparation was not a single operation snapshot, and public
duplicate options could disable the only Python protection from backend-unsafe
pairs. Voro++ also silently omits a non-periodic generator outside its container
or exactly on an upper bound.

R3 established strict source validation, immutable ownership, and native
constructor preconditions. R4 established the certified exact-binary64
minimum-image geometry used for periodic distance decisions. R5 must compose
those contracts without changing public names, signatures, defaults, result
schemas, or the mathematical meaning of periodic images.

## Decision

### One private generator-preparation boundary

`pyvoro2._internal.generator_preparation` owns the native-facing preparation
sequence for spatial and planar `compute`, `locate`, and `ghost_cells`:

```text
strict validation
→ periodic remapping
→ insertion containment
→ mandatory duplicate safety
→ optional user duplicate policy
→ internal ID assignment
→ native dispatch
```

Its frozen private `PreparedGenerators` snapshot keeps owned read-only input
Cartesian coordinates, primary Cartesian coordinates, native coordinates,
remap shifts, internal and external IDs, backend radii, periodic axes, domain
kind, and operation. It is not a public preparation API.

Box coordinates are not wrapped. Rectangular and orthorhombic periodic axes
use the domain remapper and preserve its integer shifts. A `PeriodicCell`
operation uses one validated geometry snapshot for Cartesian-to-internal
conversion, coupled primary remapping, shift provenance, and conversion of the
primary representative back to Cartesian coordinates.

Inverse realization and active-set code continues to call the public forward
operations and therefore inherits this boundary; it does not acquire a direct
native route.

### Half-open insertion containment

Every coordinate inserted into a non-periodic rectangular axis must satisfy

```text
lo <= x < hi
```

The lower bound is valid and the upper bound is invalid. Periodic rectangular
axes are remapped first. Native coordinates for a triclinic `PeriodicCell` are
remapped to `[0, bx) × [0, by) × [0, bz)`. Outside generators raise
`ValueError` with operation, generator role, input index, external ID, axis,
value, and required interval; they are never clipped or silently omitted.

This rule applies to persistent generators for all three operations and to
each temporary ghost generator. Locate query points are not inserted and keep
their existing query semantics. A contained distinct ghost can still produce
an empty cell, but an outside non-periodic ghost now raises before dispatch.

### Mandatory floor and optional policy

The fixed backend-safety boundary is

```text
BACKEND_SAFETY_DISTANCE_SQUARED = 1e-10
BACKEND_SAFETY_DISTANCE = 1e-5
```

A pair whose certified minimum-image squared distance is at most the exact
binary64 value `1e-10` is mandatory-unsafe. The inclusive comparison uses the
R4 exact squared-distance key for periodic geometry and deterministic
binary64-input Cartesian geometry otherwise. The Cartesian branch aligns and
subtracts the original source coordinates as exact binary64 dyadics before it
squares and sums their differences; it does not classify an inclusive boundary
from a rounded binary64 subtraction or rounded products. Failure to certify a
periodic pair fails before native dispatch.

This scan always runs and cannot be weakened by `duplicate_check`,
`duplicate_threshold`, `duplicate_wrap`, or `max_pairs`. Mandatory periodic
safety always wraps. The public options control only safe pairs above the
floor:

- `off` performs no additional user-threshold action;
- `warn` emits `RuntimeWarning` for a safe pair strictly closer than the user
  threshold;
- `raise` raises `DuplicateError` for such a pair;
- a threshold at or below `1e-5` adds no optional range;
- `duplicate_wrap=False` selects unwrapped Cartesian distance only for the
  optional policy.

`max_pairs` limits reported pairs, not native safety. A stopped scan records
truncation and says “at least” rather than implying an exact total.

### Complete local candidate generation

Candidate generation is separate from final distance and policy
classification. Cartesian and rectangular/orthorhombic scans use spatial
buckets in primary coordinates, wrapping neighbor keys only on periodic axes.
Bin widths are at least the requested radius, so a fixed neighboring-bin
stencil is complete even at seams, corners, and thresholds larger than a
domain span. The maps are sparse: numerical bin counts are not capped in a way
that widens ordinary cells. Python uses arbitrary-size integer keys; the native
backstop uses wide integer keys and raises structurally before insertion when
an extreme quotient or candidate workload is not representable.

Triclinic scans bucket primary fractional coordinates. For physical radius
`h`, exact source-binary64 inverse-basis column L1 bounds give

```text
|fractional_delta_l - integer_l|
    <= h * sum_j |(A_inverse)[j, l]|.
```

Python constructs both sides of this inequality from the exact dyadic values
represented by the input arrays. It reuses the R4 bit-keyed exact-basis cache,
forms fractional coordinates with integer arithmetic over a common inverse
denominator, and performs the modulo and bucket-floor operations exactly. With
`n_l = max(1, floor(1 / bound_l))`, each bin is therefore at least as wide as
the certified bound. The exact binary64 value `1e-5`, squared, is strictly
greater than the exact binary64 value `1e-10`, so the fixed-floor implication
needed by this candidate radius is inclusive. An unsafe pair has equal or
neighboring exact keys on every cyclic axis, including at a bucket boundary or
the zero/one seam and regardless of the size of its minimizing lattice shift.

These bounds decide candidates only. R4 certified geometry makes every final
periodic comparison. Candidate keys and neighboring bins are deduplicated;
ordinary well-separated clouds do not incur an unconditional all-pairs scan.
Subnormal positive thresholds remain representable for candidate generation,
including exact-duplicate discovery. Private work limits fail explicitly
rather than dispatching unsafely.

### Error provenance and internal IDs

`DuplicatePair(i, j, distance)` is unchanged. `DuplicateError` remains a
`ValueError` subclass with compatible positional `args`, string, `.pairs`, and
`.threshold`. It also records whether the error is `backend_safety` or
`user_threshold`, both safety constants, configured user threshold,
minimum-image and optional-wrap use, truncation, operation, and external-ID
pairs. Mandatory errors report `.threshold == 1e-5`; optional errors retain the
configured threshold.

Voro++ receives only internal IDs `0..n-1`; external IDs remain Python
provenance and are restored during packaging. Raw standard compute output must
contain every internal ID exactly once. Raw power output may contain a unique
subset because hidden cells are valid. Missing, duplicate, malformed, or
out-of-range raw IDs raise `RuntimeError` before public result packaging. The
maximum native integer remains reserved for the planar temporary ghost ID.

### Native backstop and workstream boundary

Every direct `_core` and `_core2d` construction path repeats insertion
containment and fixed-floor duplicate checks before any `put()`. Rectangular
paths use componentwise periodic minimum differences and local buckets.
Triclinic paths enclose inverse-basis entries, coefficient bounds, point keys,
shift boxes, image displacements, and squared-distance lower bounds with
outward-rounded binary64 intervals. Each primitive is forced through binary64
storage before its endpoints are widened with `nextafter`, and fast-math builds
are rejected. The runtime also requires round-to-nearest and verifies gradual
subnormal underflow instead of silently operating with flush-to-zero. A point
is indexed under every cyclic key allowed by its coefficient interval. Raw
neighbor-key expansion is separately bounded; cyclic key aliases and candidate
point indices are deduplicated before bucket lookup and candidate-comparison
accounting. The shift search enumerates every integer allowed by the outward
coefficient box. This proof does not rely on `long double` having more precision
than binary64. An interval or integer range that cannot be represented or
enumerated within the private work limits fails structurally before insertion;
it is never truncated or classified as safe. Boundary-near conservative
rejection is acceptable in this private last defense; Python R4 geometry
remains the public/provenance authority.
Locate queries are excluded from insertion containment, while ghost queries
are included. The private preflight role records query insertion separately
from reservation of the maximum native integer: spatial ghosts insert their
queries without claiming the planar synthetic ID, while planar ghosts require
both behaviors.

R5 does not add native minimum-image or generator-preparation APIs, change R4
shift mathematics or face/edge reconstruction, implement outside-generator
clipping, redesign R6 identity or R7 active state, or define R8 diagnostic
severity and `ok` policy. R8 may consume the now-safe forward boundary but does
not weaken or reinterpret it.

## Consequences

- Unsafe native insertion is rejected under default calls and under every
  public duplicate-option combination.
- Periodic persistent and ghost generators reach native code only in their
  primary representation, with original coordinates and remap shifts retained
  privately for provenance.
- Previously accepted outside non-periodic generators and ghosts now raise.
  This is an intentional correctness tightening for invalid native input.
- `warn` and `off` remain useful optional-policy choices without permitting a
  known backend duplicate failure.
- Complete candidate scanning remains local for ordinary data and fails
  explicitly under bounded pathological resource demand.
- Direct native callers receive a conservative pre-insertion defense even when
  they bypass Python preparation.

## Alternatives considered

### Make `duplicate_check='raise'` the safe mode

Rejected. Safety cannot depend on users discovering a non-default diagnostic
option, and `warn`, `off`, small thresholds, and disabled optional wrapping
must not permit a known backend failure.

### Clip or silently omit outside generators

Rejected. Clipping changes the scientific problem, while omission can change a
clipped tessellation without telling the caller.

### Use all-pairs scanning

Rejected for ordinary calls. The fixed safety floor must remain practical for
large well-separated inputs. Local complete candidates plus exact final
classification preserve correctness without unconditional quadratic work.

### Put final periodic classification in the native backstop

Rejected. Native doubles provide a last conservative guard, but R4 exact
binary64 geometry is the shared scientific authority and supplies public error
provenance.
