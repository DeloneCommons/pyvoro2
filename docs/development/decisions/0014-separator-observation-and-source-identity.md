# 0014 — Two-layer separator observation and source identity

- **Status:** Accepted
- **Date:** 2026-08-15
- **Related issue:** [#42 — v0.8 R6: bind separator observations, results, realizations, and reports to canonical source data](https://github.com/DeloneCommons/pyvoro2/issues/42)
- **Related decisions:** [ADR 0007](0007-separator-objective-contract.md), [ADR 0011](0011-strict-input-and-ownership-contract.md), [ADR 0012](0012-certified-periodic-image-geometry.md), [ADR 0013](0013-central-generator-preparation-and-backend-safety.md)
- **Related plan:** [v0.8 remediation execution plan](../plans/archive/v0.8-remediation.md)

## Context

Separator rows can be scientifically meaningful without retaining the points
and domain from which their connector geometry was resolved. The public direct
`SeparatorObservations` constructor and the row-only chain from observations to
problem, fit result, and report therefore cannot require source geometry or
invent it. At the same time, source-aware fitting and realization must not
combine rows, predictions, or diagnostics merely because their lengths or
resolved numerical arrays happen to agree.

Before R6, private originating-observation checks were distributed across
consumers, some result and realization paths used only shape checks, and report
families lacked a common schema and source-provenance envelope. A single
mandatory source fingerprint would close those gaps only by breaking valid
directly constructed observations. R6 instead needs independent row identity
that is always available and exact source provenance that is acquired only
when the source is known and independently verified.

## Decision

### Source-independent observation identity

Every valid `SeparatorObservations` object has a canonical observation
namespace, one row identity per observation, and an ordered observation-set
identity. The canonical type and version strings are exactly:

```text
pyvoro2-separator-observation-namespace v1
pyvoro2-separator-row v1
pyvoro2-separator-observation-set v1
pyvoro2-separator-source v1
```

Each payload stores its corresponding `type` string and integer `version=1`;
the `v1` notation above names that exact pair.

The namespace payload contains `dimension`, `n_points`, `measurement`, and
`ids`. Each row payload contains, in canonical form, `i`, `j`, `shift`,
`measurement`, `target`, `confidence`, `distance`, `distance2`, `delta`,
`target_fraction`, `target_position`, and `explicit_shift`. Warnings are not
part of row identity.

A row identifier is exactly

```text
pyvoro2-separator-row-v1:<namespace_sha256_hex>:<input_index>:<row_sha256_hex>
```

The observation-set payload contains the namespace fingerprint and the ordered
row-ID sequence. Subsetting preserves the retained row IDs and `input_index`
values. Duplicate rows remain distinct through `input_index`; periodic
parallel rows remain distinct through their shifts. Reordering rows changes the
set identity. Binding a source later changes neither row IDs nor the
observation-set fingerprint.

Canonical payload fingerprints use one algorithm. Finite floats first
normalize signed zero and then use `float.hex()`. Arrays become row-major nested
lists and JSON primitives remain primitives. Serialization uses
`ensure_ascii=True`, `sort_keys=True`, `separators=(",", ":")`, and
`allow_nan=False`; UTF-8 bytes are hashed with SHA-256. The exposed form is
`sha256:` followed by 64 lowercase hexadecimal digits. Fingerprint agreement is
only an index into an exact comparison: runtime association still compares the
complete canonical values and does not trust a hash alone.

### Strict canonical observation values

The direct constructor remains public with its existing dataclass fields,
order, and public signature. It validates dimension, point count, endpoint and
shift integer categories and ranges, aligned shapes, distinct endpoints,
unique non-negative `input_index`, finite non-negative confidence, finite
nonzero connector geometry, IDs, and warnings. Its arrays are owned,
C-contiguous, and read-only.

`distance2` and `distance` are recomputed from `delta`, and
`target_fraction` and `target_position` are recomputed from the canonical
target and distance. A finite supplied redundant value is accepted only when

```python
np.allclose(
    supplied,
    derived,
    rtol=8 * np.finfo(np.float64).eps,
    atol=0.0,
)
```

and is then replaced by the recomputed binary64 value. This constructor
tolerance is not a source-equivalence rule.

### Optional monotonic source binding

Resolver-created observations are source-bound. Directly constructed valid
observations are unbound. The first source-aware use of an unbound object may
bind it only after independently recomputing every row's connector geometry
from the supplied points and domain and rejecting any inconsistency. Once
bound, the object cannot be rebound and requires exact canonical source
equality.

The source payload retains original caller-order points before periodic
remapping, exact dimension and count, exact ID provenance, and one exact domain
representation from this closed vocabulary:

| Kind | Fields |
|---|---|
| `none` | `kind` |
| `planar_box` | `kind`, `bounds` |
| `planar_rectangular_cell` | `kind`, `bounds`, `periodic` |
| `spatial_box` | `kind`, `bounds` |
| `spatial_orthorhombic_cell` | `kind`, `bounds`, `periodic` |
| `spatial_periodic_cell` | `kind`, `vectors`, `origin` |

Translated, periodically shifted, lattice-equivalent, or differently
represented domains are not the same source. Binding is private rather than a
public dataclass field or constructor argument, and it survives subsets,
shallow and deep copies, `dataclasses.replace`, `copy.replace` where available,
and same-version pickle round trips. A replacement inconsistent with an
existing source binding raises.

For an already resolved observation object passed to a fitting operation with
points, omitted/default `domain=None` verifies the exact source points but makes
no new domain assertion and does not erase an existing domain binding. An
explicit non-`None` domain must match exactly. For an unbound object,
`domain=None` establishes a bound source whose domain is `{"kind": "none"}`;
an explicit domain binds that exact representation. The observation object's
owned IDs remain authoritative on this path. Realization and active-set
operations establish or verify the complete source they actually use.

### One association policy

One private origin-helper family governs fit views, records, report builders,
realization, and active-set operations:

| Authoritative origin | Supplied observations | Result |
|---|---|---|
| unbound | unbound, exact same observation model | accept |
| bound | bound, exact same source | accept |
| bound | unbound | reject |
| unbound | bound | reject |
| bound | bound, different source | reject |

Length-only checks are forbidden. Reports take provenance from the
authoritative origin retained by their result or diagnostic, never from an
arbitrary supplied object that merely has matching rows.

### Versioned report envelope

Fit, realized-pair, and active-set reports retain their exact kind strings:

```text
power_weight_fit
realized_pair_diagnostics
self_consistent_power_fit
```

Every report adds these common top-level records:

```json
{
  "schema": {
    "name": "pyvoro2.inverse.separator.report",
    "version": 1
  },
  "producer": {
    "name": "pyvoro2",
    "version": "<pyvoro2.__version__>"
  },
  "source": {
    "binding": "unbound",
    "fingerprint": null,
    "dimension": 2,
    "n_points": 3,
    "points": null,
    "domain": null,
    "ids": null
  },
  "observation_set": {
    "fingerprint": "sha256:<hex>",
    "measurement": "fraction",
    "n_rows": 3,
    "row_ids": ["pyvoro2-separator-row-v1:..."]
  }
}
```

The source record always has exactly `binding`, `fingerprint`, `dimension`,
`n_points`, `points`, `domain`, and `ids`. An unbound source uses
`binding="unbound"` and null fingerprint, points, domain, and IDs. A bound
source uses `binding="bound"`, its exact source fingerprint, caller-order
points, exact domain record, and exact IDs provenance. A bound
`{"kind": "none"}` domain is distinct from an unbound null domain.

Every observation-aligned exported row adds `row_id` without removing or
renaming existing fields. Report builders return JSON-native values before
serialization. `dumps_report_json` uses `allow_nan=False`; finite fit,
realization, and active reports round-trip exactly through JSON.

R6 does not redesign active-set final-state availability. If an existing active
failure contains non-finite placeholders, report serialization fails closed.
R7 owns any future null/unavailable/reason representation and coherent
all-failure-state round trip.

### Private implementation boundary

Canonical encoding, exact comparisons, binding lifecycle, and association
rules belong to one private separator identity module. R6 adds no public
identity class, source argument, dependency, native input, or C++ behavior.

## Consequences

- The valid public row-only observation-to-report chain remains available and
  reports honest unbound provenance.
- Row and observation-set identities are stable before and after source
  binding, while exact source provenance can only become more informative.
- Copy, replacement, pickle, fit, realization, active-set, and reporting paths
  enforce one association policy instead of local shape checks.
- Exact-key report consumers must accept the schema, producer, source,
  observation-set, and row-ID additions in v0.8.
- Source validation may reject previously accepted combinations that silently
  joined unrelated rows and geometry; valid numerical fitting behavior is
  unchanged.

## Alternatives considered

### Require points and domains in every observation and report builder

Rejected. This would break the established direct constructor and row-only
builder chain, add public source arguments, and encourage fabricated
provenance.

### Treat matching fingerprints as sufficient equality

Rejected. SHA-256 is a compact public identifier, not a substitute for exact
runtime comparison of canonical scientific values.

### Use tolerant, translated, or lattice-equivalent source matching

Rejected. Constructor consistency tolerance has a different purpose, and
broad equivalence would erase the exact caller source required for provenance.

### Represent unavailable active failure data with nulls in R6

Rejected. That would begin the R7 final-state availability redesign. R6 fails
closed when existing placeholders are not valid JSON numbers.
