# Planar normalization API

Periodic global-edge keys preserve boundary kind and owner/image provenance
before geometric pooling. Every local occurrence retains its association,
including repeats. Known real walls on nonperiodic axes need no
`adjacent_shift`; unknown negative labels and wall sides on periodic axes are
rejected. Supplied nonzero wall shifts and nonzero shifts on nonperiodic axes
are also rejected.

`normalize_edges` and `normalize_topology` refuse a mapping that collapses
distinct public edge endpoints, including tolerance pooling or periodic seam
snapping, before mutation. Equal public endpoints can be retained but do not
certify internal native collapse. `normalize_vertices` alone retains a numerical
vertex pool, raw vertices and local mappings. These numerical/raw-record
utilities do not establish exact E/S positivity.

When normalization requested through ordinary `compute` cannot be represented,
the call raises structured `TessellationError` with
`WP6_NORMALIZATION_REPRESENTATION`. Standalone normalization helpers retain
their `ValueError` behavior.

::: pyvoro2.planar.normalize
:::
