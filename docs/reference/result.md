# Common tessellation result

`TessellationResult` is the shared structured data contract for spatial and
planar tessellations. Both public `compute()` functions return it by default;
`output='result'` selects it explicitly. `output='cells'` is the supported
explicit low-level route for raw cell dictionaries and, when requested, the
raw diagnostics tuple.

Structured output is always one object. Diagnostics computed because of
`return_diagnostics=True` or `tessellation_check` are stored in
`tessellation_diagnostics`. Planar normalized objects are stored in
`normalized_vertices` and `normalized_topology`. The compatibility name
`PlanarComputeResult` is the identical class object, not a second wrapper; it
will be removed in v0.8.

Direct construction is provisional. It validates documented ID, measure,
empty-state, representation, and capability metadata, including the exact
weight/shift-to-radius relationship for weight-first power input. Normal
application code should obtain results from `compute(...)`; the complete
cross-field direct-construction contract may still be refined before 1.0. The
aligned measure and empty-state arrays are snapshots: later mutation of the
shared raw dictionaries does not update those arrays, while boundary access
revalidates mutable boundary records against the snapshot before returning
them. Empty cells always expose empty boundary collections; adding edge or face
records to a shared empty raw cell makes boundary access raise. Deep copies and pickle round trips retain the read-only
aligned-array and capability contracts without reinterpreting later permitted
raw-record mutation.

The provisional `global_vertices` and `global_edges` conveniences preserve the
historical planar result access pattern by forwarding to available normalized
planar objects.

::: pyvoro2.result.TessellationResult
:::
