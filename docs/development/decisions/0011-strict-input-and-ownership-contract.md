# 0011 — Strict public input and ownership contract

- **Status:** Accepted
- **Date:** 2026-08-07
- **Related issue:** [#39 — v0.8 R3-B: complete package-wide strict input adoption and immutable ownership](https://github.com/DeloneCommons/pyvoro2/issues/39)
- **Related decisions:** [ADR 0005](0005-tessellation-result-contract.md), [ADR 0010](0010-native-construction-preconditions.md)
- **Related plan:** [v0.8 remediation execution plan](../plans/archive/v0.8-remediation.md)

## Context

ADR 0010 established native construction preconditions and the shared Python
validation foundation. Other public Python paths still converted exact
integers with `int(...)`, flags and masks through truthiness, and numerical
values through NumPy dtypes before checking their original categories. Frozen
domains and inverse values also retained caller-owned lists or writable arrays.
Caller mutation could therefore invalidate already checked state, while NaN,
infinity, or an out-of-range remap quotient could first reach a reduction,
linear-algebra operation, solver loop, or integer cast.

R3-B applies one package-wide contract without changing public names,
signatures, defaults, result schemas, forward geometry, or separator objective
and solver mathematics.

## Decision

### Exact scalar categories

Public integer fields use `operator.index` semantics and reject Python and
NumPy Booleans before conversion. Python integers, NumPy signed integers,
in-range NumPy unsigned integers, and other deliberate non-Boolean index
scalars are accepted. Each field then applies its positive, non-negative, or
destination-range constraint. Forward external IDs, including diagnostic
`expected_ids`, remain unique, non-negative, and representable by the
signed-int64 result contract. Separator external IDs retain their documented
wider exact non-negative range; direct observation indices, provenance
indices, and periodic shift components must fit signed int64. A documented
non-negative count such as `max_examples` continues to accept zero.

Public Boolean fields accept only `bool` and `numpy.bool_`. Integer 0/1,
strings, and arbitrary truthy or falsy objects are not Boolean input. A mask is
checked in its original shape and element categories before conversion, then
copied to an owned C-contiguous Boolean array. Retained masks are read-only;
algorithms make a separate writable copy only after validation.

Public string modes and stored string metadata accept only Python `str` or a
NumPy `str_` scalar. They reject NumPy arrays regardless of dimensionality or
size, bytes, numeric and Boolean values, and arbitrary equality objects before
comparison or containment. A NumPy string scalar is canonicalized to a
built-in `str`; enumerated choices are then compared exactly and
case-sensitively against their existing allowed values. Optional free-form
metadata additionally accepts `None` where already documented, without
narrowing the field to a new enumeration.

Public real scalars and arrays reject Boolean, complex, string, NaN, and
infinite values before reduction, flooring, linear algebra, or iteration.
Positive and non-negative ranges remain field-specific, so zero continues to
be valid only where the existing contract permits it. Validated stored model
and option scalars are canonical built-in Python `float` or `int` values.
Finite source coordinates must also produce representable finite separator
connector differences, squared distances, distances, and measurement
conversions; non-representable derived connector geometry raises `ValueError`
without leaking a numerical runtime warning.

### Canonical domains and retained numerical data

Spatial and planar boxes and rectangular periodic cells own nested tuples of
built-in finite floats. Periodicity is an owned tuple of built-in Booleans.
`PeriodicCell` owns nested float tuples for its vectors and origin. No domain
retains an alias to a caller list or array.

`Box.from_points` validates a non-empty finite real point matrix and finite
non-negative padding before reduction. Resulting bounds must still be finite
and strictly ordered, so zero padding is accepted only for data with positive
extent on every axis.

Retained numerical inputs such as separator observation arrays, exact shifts
and provenance indices, `L2Regularization.reference`, public problem arrays,
and retained masks are owned C-contiguous NumPy copies with writing disabled.
This is pragmatic ownership: solver-created result graphs and raw nested cell
records are not recursively frozen merely to broaden an immutability claim.
Direct problem construction is finite-strict. The canonical problem builder
still preserves R1/R2's reviewed handling of non-finite *derived* scaled-row
intermediates that can arise from finite extreme-scale source inputs; those
intermediates are not a route for accepting non-finite public source data.

### Periodic cells and remapping

`PeriodicCell` validates finite real vector and origin data before determinant,
singular-value, or basis work. The lattice must be right-handed with
`determinant > 0`; pyvoro2 does not silently flip vectors. The existing
conditioning policy remains unchanged: reject relative volume below `1e-12`,
reject condition number above `1e15`, and warn with `RuntimeWarning` above
`1e10`.

Spatial and planar remapping validates points, `return_shifts`, and explicit
`eps` before flooring. Every quotient and accumulated shift is proven
representable as signed int64 before conversion or addition. Rejected input
therefore cannot emit invalid-cast warnings or expose an int64 sentinel. The
existing remapped values, half-open convention, shear order, and shift-sign
meaning are unchanged.

Spatial and planar normalization treats caller-supplied raw cell records as a
new public boundary even when those records originally came from `compute`.
Cell IDs, local and global vertex indices, adjacent-cell IDs, and lattice
shifts are validated from their original categories before they enter sorting,
indexing, or topology keys. Coordinate quantization likewise requires every
finite coordinate-to-tolerance quotient and rounded key to be finite and
signed-int64 representable before conversion. Normalization preflights these
records and derived relative shifts before constructing topology or applying
in-place annotations, so rejection is independent of NumPy error settings and
cannot leave caller records partially mutated.

### Validation ordering and workstream boundary

Changed paths validate modes first, then exact integer and Boolean scalars,
then original numerical kinds, shapes, and finiteness. Only canonical owned
values enter relationship checks, reductions, solver loops, R3-A native
dispatch, or result packaging.

`r_min` and explicit `weight_shift` remain mutually exclusive when `r_min` is
nonzero. Public fit, active-set, and result-building boundaries establish that
relationship before solver, prediction, objective, or radius work; the shared
weight-to-radius transform retains the same defensive check.

This decision completes the implementation contract for R3-B but does not by
itself close R3; the integrated R3 patch still requires independent review. It
does not certify triclinic nearest-image selection (R4), make duplicate or
containment protection mandatory (R5), redesign observation identity (R6),
change active-set final-state semantics (R7), change diagnostic severity or
`ok` (R8), or qualify a release (R9 and issue #33).

## Consequences

- Previously accepted lossy integer, truthy Boolean, non-scalar string-mode,
  arbitrary equality, non-finite, and mutable-alias inputs now raise
  `ValueError` at their public boundary.
- Caller mutation after construction cannot alter canonical domains, model
  references, resolved observations, or public problem arrays.
- Directly constructed separator observations and problems receive the same
  category, range, ownership, and mask checks as builder-created instances.
- Valid inputs retain existing numerical values and output schemas. R1/R2
  objective and solver semantics and ADR 0010 native policy are unchanged.

## Alternatives considered

### Continue coercing values at each use site

Rejected. Lossy conversion destroys the source category needed to distinguish
an exact integer or Boolean from a merely convertible value, and repeated
local checks produce inconsistent messages and ranges.

### Preserve caller containers in frozen dataclasses

Rejected. Freezing only attribute assignment does not protect validation when
the referenced list or array remains mutable.

### Normalize a left-handed lattice automatically

Rejected. An implicit vector swap changes the caller's basis and periodic image
labels. Handedness is an input precondition, not a silent geometry repair.

### Fold R4, R5, or diagnostic policy into validation adoption

Rejected. Certified minimum-image geometry, mandatory backend safety policy,
and diagnostic severity are separate accepted workstreams with different
oracles and compatibility consequences.
