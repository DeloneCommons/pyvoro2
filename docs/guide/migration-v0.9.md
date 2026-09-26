# v0.9 planar boundary migration

Ordinary `pyvoro2.planar.compute` now attributes edge owners and images from the
native execution and audits exact geometry separately. The result class and
raw output selectors are unchanged.

## Remove obsolete reconstruction arguments

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
Spatial records retain original `query`; planar records do not gain it until
WP8. Incomplete native/source/S certification raises a `ValueError`-compatible
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
defines the ghost contract, whose implementation acceptance remains pending.
