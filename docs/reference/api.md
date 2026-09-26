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
refuse explicitly while safe geometry-only operation is separate. WP7 awaits
independent exact-head acceptance.

::: pyvoro2.api
:::
