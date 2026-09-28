# High-level API

`compute(...)` returns `pyvoro2.TessellationResult` by default. Use
`output='cells'` only when a low-level workflow deliberately needs raw cell
dictionaries or the raw diagnostics tuple.

`tessellation_check='diagnose'` computes diagnostics without acting on the
result. `'warn'` emits one summary warning when `diagnostics.ok` is false, and
`'raise'` raises `TessellationError` in exactly the same case.

`ghost_cells(...)` returns `list[dict]`, not a `TessellationResult`. Every
requested 3D face has a four-field `boundary_reference` for source-attributed
N with positive exact public-semantic S contact, or outer `None` only for
proved internal collapse. Generator references identify an original persistent
site and user-basis image shift in periodic domains, including zero;
ghost-self references have a nonzero shift and no `adjacent_cell`, and physical
walls have side identity without a shift. `site` is the actual stored ghost's
Cartesian representative. The original `query` and `query_index` remain.
Certification needs no requested public vertices or adjacency. Hard
certificate refusals expose `code`, `stage`, `query_index` and bounded
`details` on a `ValueError`-compatible error. Certificate-bearing support is
initially qualified for Linux x86_64 GCC 13.3 strict binary64; other cohorts
refuse explicitly while safe geometry-only operation is separate. WP7 and WP8
are independently accepted and merged.

## Query results (spatial and planar)

| Field | Availability | Shape/type |
|---|---|---|
| `found`, `owner_id` | Every locate call | `(m,)` bool/integer; not-found ID -1 |
| `query`, `query_wrapped` | Periodic locate | `(m,d)` float64, original / RN64 exact wrap |
| `query_shift` | Periodic locate | `(m,d)` signed int64, zero on nonperiodic axes |
| `owner_pos` | Owner selector true | `(m,d)` native float64 Cartesian view |
| `owner_site`, `owner_shift` | Periodic and owner selector true | `(m,d)` original float64 / exact image int64 |

Owner coordinate sentinels are NaN; owner-shift sentinels are zero. Exact image
identity is `owner_site+owner_shift@A` in the original query chart. `owner_pos`
preserves qualified native arithmetic and is not a canonical exact-image
reconstruction. Query wrapping uses exact validated binary64 affine operands,
including the exposed binary64 rectangular spans. Empty arrays preserve their
shapes/dtypes. Arrays do not alias caller or private cached/native state.

Ghost records in both dimensions contain original `query` (float list),
`query_index`, `id=-1`, and the existing stored `site`. Periodic records add
`query_wrapped` (float list) and `query_shift` (integer tuple representable in
int64). No `site_shift` or aggregate result is added. All required WP7 checks
precede empty filtering, then new query views are materialized for retained rows.
An unrepresentable retained query raises `GHOST_SHIFT_UNREPRESENTABLE`, stage
`materialization`, with its input index and field in `details`.

The Provisional locate failure protocol is ValueError-compatible and exposes
`code`, `stage`, `query_index` (or None before selection), and bounded `details`:

| Code | Meaning |
|---|---|
| `LOCATE_BACKEND_INSERTION` | Actual persistent insertion/association failed |
| `LOCATE_NATIVE_UNSUPPORTED` | Unsafe integer execution or unsupported producer FP environment |
| `LOCATE_PROVENANCE_AMBIGUOUS` | Several compatible owner-image coefficients |
| `LOCATE_PROVENANCE_INCONSISTENT` | No compatible coefficient, malformed association, or explicit proof invariant |
| `LOCATE_CERTIFICATION_RESOURCE` | Complete proof/observer exceeds a private limit |
| `LOCATE_METADATA_UNREPRESENTABLE` | Requested float/int64 view cannot be materialized |

Ordinary input errors keep existing validation behavior. Invariant failures
retain their reason and cause; failures are atomic. ID-only locate does not
request owner-image certification. Existing Stable forward operations and
nonperiodic schemas remain unchanged apart from native integrity/safety checks.

::: pyvoro2.api
:::
