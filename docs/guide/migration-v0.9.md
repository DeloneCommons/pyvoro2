# v0.9 periodic, query, and separator migration

## Separate observation units from model units

Existing separator models inherit `SeparatorObservations.measurement` through
`space=None`. To optimize absolute position while restricting connector
fraction, use `SquaredLoss(space='position')` with
`Interval(0., 1., space='fraction')`. Observation-facing `target`, `predicted`,
and `residuals` retain source units. Use `mismatch_target`,
`mismatch_predicted`, and `mismatch_residuals` for optimization units.

Hard `lower`/`upper`/`value`/`applicable` and penalty
`lower`/`upper`/`strength` accept scalars or one-dimensional exact-length row
vectors. Scalars broadcast; a length-one vector does not broadcast to more
rows. Values are copied and strictly validated even on inactive or zero-strength
rows. `applicable=False` removes a hard restriction independently of confidence.
Hard `lower == upper` is valid. Shape parameters (`delta`, `margin`, `tau`,
`epsilon`) remain scalar, and spaces remain term-global.

Separator fit, realization, and active reports now declare schema version `2`.
Update exact-key consumers for the fit/active `model_spaces` and `model_policy`
blocks, fit summary `mismatch_space`, and four mismatch record fields. Realized
reports carry the v2 envelope without inventing model policy. Scalar and vector
configuration remain distinguishable as `uniform` and `rows`, including empty
selections. Active outer policy is candidate-aligned; its nested fit policy is
selected-row aligned. Unavailable predictions remain null.

Ordinary `pyvoro2.planar.compute` now attributes edge owners and images from the
native execution and audits exact geometry separately. The result class and
raw output selectors are unchanged.

## Remove obsolete reconstruction arguments

Delete `face_shift_search`, `validate_face_shifts`, `repair_face_shifts` and
`face_shift_tol` from spatial `pyvoro2.compute` (also `pyvoro2.api.compute`).
All four are removed immediately before 1.0. Even their former default values
raise ordinary Python argument-binding `TypeError` before native work.
There are no aliases, warning-only transitions or replacement controls.
Keep `return_face_shifts=True` to request public face-image metadata:

```python
result = pyvoro2.compute(points, domain=cell, return_face_shifts=True)
```

Native source attribution and the optional independent exact consistency audit
retain their WP5 meanings. Diagnostic tolerances cannot select or repair labels.


Delete these keywords from ordinary `planar.compute` and `planar.ghost_cells`
calls:

| Removed keyword | Replacement |
|---|---|
| `edge_shift_search` | None; a finite search radius no longer determines provenance. |
| `validate_edge_shifts` | None; required owner/image attribution cannot be disabled. |
| `repair_edge_shifts` | None; shifts are not mutated to force reciprocity. |
| `edge_shift_tol` | None; a matching tolerance no longer selects an image. |

There are no aliases or ignored compatibility arguments. Keep
`return_edge_shifts=True` when shifts belong in public output:

```python
result = pyvoro2.planar.compute(
    points,
    domain=cell,
    return_edges=True,
    return_edge_shifts=True,
    return_vertices=False,
    return_adjacency=False,
)
```

There are no ignored ghost aliases. Separator `image_search`, normalization
tolerances and tessellation diagnostic tolerances keep their existing purposes.

Ghost calls still return `list[dict]`. Each requested ghost edge now has
`boundary_reference`: a four-field `kind`/`generator_id`/`shift`/`wall_id`
record for a source-attributed positive public-semantic edge, or outer `None`
only for a proved internally collapsed raw edge. Positive generator references
carry user-basis shifts in a periodic domain even when
`return_edge_shifts=False`; `adjacent_shift` is an optional matching
compatibility view. Requesting it no longer requires public vertices. Ghost
self edges have a nonzero shift and omit `adjacent_cell`; walls have a side ID
and no shift. The public ghost `site` is the actual stored Cartesian anchor.
Both dimensions now include original `query`; periodic records add
`query_wrapped` and `query_shift` while retaining the WP7 stored-site anchor. Incomplete native/source/S certification raises a `ValueError`-compatible
`GHOST_*` failure with `code`, `stage`, `query_index` and bounded `details`.

## Interpret provenance and consistency separately

Generator `adjacent_shift` values refer to original caller sites in the user
lattice basis. Public persistent `site` also uses the original input. Real
walls omit `adjacent_shift`; do not require or insert a zero tuple for a wall.
Self-image shifts are nonzero. Omitted public shifts do not disable truthful
ordinary edge ownership.

`has_periodic_shifts` reports availability of requested native generator-image
metadata, including an empty result. It does not guarantee positive exact
contact. Raw collapsed occurrences remain visible, and repeated occurrences
need not represent distinct positive semantic edges.

Request `return_diagnostics=True` or a `tessellation_check` other than `'none'`
for the complete exact E/S consistency audit. `'diagnose'` attaches findings,
`'warn'` warns for a non-okay diagnostic and `'raise'` raises. Hard provenance,
insertion, profile and required representation failures raise regardless of
that setting. An incomplete resource-limited audit is not successful
consistency, even when attributed shifts can be returned.

Planar separator realization requires successful exact consistency and uses
positive public-semantic boundary classes and their exact-derived lengths.
Standalone raw-record utilities do not reconstruct the private native proof.
Edge and topology normalization refuse mappings that collapse distinct public
endpoints or cannot preserve their occurrence/provenance associations.

## Check the native support boundary

The initial ordinary planar certification profile is Linux x86_64 with GCC
13.3 and the qualified baseline-SSE2 binary64 build/evaluation policy.
Unsupported profiles refuse explicitly; the broader package wheel matrix is
not a planar-certification guarantee. Actual insertion omission is a hard
failure in both standard and power mode, separate from an inserted hidden cell.

See the [planar guide](planar.md) for current workflows and
[ADR 0022](../development/decisions/0022-wp6-source-certified-planar-edge-provenance.md)
for the ordinary scientific and failure contract. WP6 has been independently
accepted; [ADR 0023](../development/decisions/0023-wp7-certified-ghost-boundaries.md)
defines the independently accepted ghost contract. WP8 query/owner metadata
was independently accepted and merged through PR #80.

## Consume query and owner metadata directly

Periodic locate now always returns `(m,d)` `query`, `query_wrapped` and
`query_shift` arrays. `return_owner_position=True` adds `owner_site`,
`owner_pos` and `owner_shift` together. Use the exact image equation
`owner_site+owner_shift@A` for image identity; keep `owner_pos` when you need the
existing native Cartesian view. Do not add query wrapping to the owner shift
or canonicalize the native position to the exact-image float reconstruction.
Not-found owner fields use ID -1, NaN coordinates and zero shifts.

Ghost records retain their list/dict structure, `id=-1`, original input index
and stored-site boundary chart. Empty filtering follows all WP7 checks;
new query metadata materializes only retained rows. Code that enumerates dict
keys should allow these additive fields. Nonperiodic locate keys are unchanged.
Insertion omission and unsafe native queries now raise structured `LOCATE_*`
errors rather than silently losing a generator or entering unsafe conversion.
The added fields/failure protocol remain Provisional during v0.9.x; existing
forward operations stay Stable. The WP9 reconstruction-control removals above
do not change these metadata equations or ghost eligibility checks.
