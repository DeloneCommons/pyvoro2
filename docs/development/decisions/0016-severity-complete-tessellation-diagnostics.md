# 0016 — Severity-complete tessellation diagnostics

- **Status:** Accepted
- **Date:** 2026-08-16
- **Related issue:** [#44 — v0.8 R8: make tessellation diagnostics and strict validation severity-complete](https://github.com/DeloneCommons/pyvoro2/issues/44)
- **Related decisions:** [ADR 0011](0011-strict-input-and-ownership-contract.md), [ADR 0013](0013-central-generator-preparation-and-backend-safety.md)
- **Related plan:** [v0.8 remediation execution plan](../plans/archive/v0.8-remediation.md)

## Context

Spatial and planar diagnostics previously had several competing meanings of a
successful tessellation. The analyzer based its overall result mainly on
measure closure and observed reciprocity, while strict validation and public
compute wrappers reconstructed smaller Boolean combinations. Missing expected
IDs could therefore be reported without failing standard-mode validation, and
an error issue was not guaranteed to make the overall diagnostic false.

Cell areas and volumes were converted through broad exception handling.
Malformed values could be skipped, non-finite values could poison the sum, and
negative values could be reduced to generic closure warnings. Periodic
annotations used `setdefault`, so a repaired second analysis could retain a
stale failure flag. Planar normalized-topology validation also made warning-only
subchecks fatal even though the corresponding issues were warnings.

These are diagnostic-policy defects, not forward-geometry defects. ADR 0013
already requires safe generator preparation and validates raw native IDs before
public result packaging. Standalone diagnostics must still describe malformed
or user-edited returned records without duplicating that native-safety layer.

## Decision

### One overall severity rule

Spatial and planar tessellation diagnostics use one final rule:

```text
ok = every call- and mode-required invariant passed
     and no issue has severity "error"
```

Every required failure emits an error issue. Warning- and info-only findings do
not make the overall diagnostic false. A small private shared module owns the
cross-dimensional measure, expected-ID, reciprocity-severity, annotation-reset,
stable-sum, and error-aggregation policy. Face/edge geometry and periodic shift
matching remain dimension-specific.

The descriptive `ok_volume`/`ok_area`, `ok_reciprocity`, and normalized-topology
subcheck fields remain. An optional or warning-only subcheck may be false while
the overall `ok` is true.

### Expected IDs depend on the declared mode

When `expected_ids` is omitted, diagnostics do not invent a completeness
requirement. When it is present:

| Mode | Absent ID classification | Severity | `missing_ids` | `empty_ids` | Overall effect |
|---|---|---|---|---|---|
| `"standard"` | `MISSING_IDS` | error | included | unchanged | fails |
| `"power"` | `HIDDEN_IDS` | info | included | included | nonfatal by itself |
| `None` | `MISSING_IDS` | warning | included | unchanged | nonfatal by itself |

The public mode vocabulary remains exactly `"standard"`, `"power"`, and
`None`. Diagnostics do not guess a mode from raw record shape.

### Cell measures are categorized before aggregation

A non-empty cell must contain an area or volume that is a non-Boolean real
scalar, finite, and non-negative. Empty cells may omit the measure or provide
exact finite zero. Both dimensions use these error codes:

| Code | Meaning |
|---|---|
| `MISSING_CELL_MEASURE` | A non-empty cell omitted its required measure. |
| `INVALID_CELL_MEASURE` | The value is not an accepted real scalar category. |
| `NONFINITE_CELL_MEASURE` | Conversion cannot produce a finite built-in float. |
| `NEGATIVE_CELL_MEASURE` | The finite value is negative. |
| `EMPTY_CELL_NONZERO_MEASURE` | An explicitly empty cell has a finite nonzero measure. |

Invalid values are never substituted with zero. Validated non-empty measures
are reduced with `math.fsum`. If a required measure is invalid, the measure
subcheck and overall diagnostic fail through the explicit measure issue; no
`GAP` or `OVERLAP` is inferred from a partial sum. When all terms are valid,
closure outside tolerance emits `GAP` or `OVERLAP` as an error.

If the sum of validated finite non-negative measures exceeds binary64 range,
the aggregate is represented as positive infinity and reported as
`OVERLAP`/error. The finite input terms are not reclassified as invalid cell
measures.

### Reciprocity can be inspected without being required

The private analyzer separates whether periodic reciprocity is inspected from
whether it is required. Public `analyze_tessellation(...,
check_reciprocity=True)` treats the requested check as required.
`validate_tessellation` and `compute` use their existing
`require_reciprocity` and `tessellation_require_reciprocity` choices.

| Finding | Required | Optional inspection |
|---|---|---|
| `NO_FACE_SHIFTS` / `NO_EDGE_SHIFTS` | error | info |
| `MISSING_RECIPROCAL` | error | warning |
| `RECIPROCAL_MISMATCH` | error | warning |

Existing domain-based defaults remain unchanged. Standard and power periodic
compute diagnostics require reciprocity by default. Optional findings retain a
false descriptive reciprocity subcheck where appropriate but do not fail the
overall diagnostic.

### Wrappers consume the completed diagnostic

`validate_tessellation(level="basic")` returns the completed diagnostic.
Strict validation raises `TessellationError` exactly when `diag.ok` is false
and retains that diagnostic on the exception. Public compute `diagnose` only
computes/attaches diagnostics, `warn` emits one summary warning exactly when
`diag.ok` is false, and `raise` raises exactly in the same case. These wrappers
do not reconstruct closure or reciprocity policy from subcheck fields.

### Mutable annotations are rerunnable

With marking enabled, each analysis first overwrites `orphan`,
`reciprocal_missing`, and `reciprocal_mismatch` to false on every face or edge
record, including records later excluded as walls or degenerate geometry. It
then marks current failures. With marking disabled, existing caller keys are
untouched.

### Normalized topology follows issue severity

Overall spatial and planar normalized-topology `ok` is false only when an error
issue exists. Planar `LOW_VERTEX_INCIDENCE` and `BAD_POLYGON_COUNT` remain
warnings; their descriptive subchecks can be false while strict validation
passes. Spatial `EULER_CHARACTERISTIC_MISMATCH` and existing warning-only
incidence findings retain the same nonfatal behavior. Existing error-level
shift, reciprocal-set, and incidence findings remain errors.

## Consequences

- Strict validation can no longer pass missing standard IDs or malformed cell
  measures.
- Hidden power sites are represented explicitly without rejecting a valid
  power diagram.
- Required closure and reciprocity failures have error issues that explain the
  false overall diagnostic.
- Callers choosing optional reciprocity still receive structured findings
  without warning promotion or a new public policy object.
- Reanalyzing repaired mutable records clears analyzer-owned stale state.
- Public names, signatures, defaults, result dataclass fields, and diagnostic
  mode values are unchanged; new issue codes and corrected severities are a
  v0.8 correctness tightening.
- R3 tolerance validation remains the input boundary and is not duplicated.
  R5 generator/native safety, R6 identity/reporting, R7 active-state semantics,
  and forward geometry are unchanged.

## Alternatives considered

### Let strict wrappers promote selected warnings

Rejected. It leaves issue severity unable to explain failure and recreates
different policy in each wrapper.

### Treat every absent expected ID as an error

Rejected. A missing power site may be a legitimate hidden cell, while raw
`mode=None` does not provide enough information to infer standard semantics.

### Treat malformed measures as zero or emit closure findings

Rejected. Substitution changes the reported scientific quantity, and a
closure issue derived from an incomplete sum hides the actual category error.

### Add a public diagnostic-policy registry or warning-promotion callback

Rejected. The required choices are already represented by the existing mode
and reciprocity parameters; v0.8 does not need a new configuration surface.
