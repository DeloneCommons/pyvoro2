# Planar high-level API

`compute(...)` returns the common `pyvoro2.TessellationResult` by default.
Use `output='result'|'cells'` to select structured or raw output explicitly.

`tessellation_check='diagnose'|'warn'|'raise'` computes one final diagnostic;
warning and raising behavior is driven only by its `ok` value.

Ordinary compute source-attributes every published or consumed periodic edge
owner/image, including when public shifts are omitted. `return_edge_shifts=True`
requires periodic `return_edges=True` and supports either setting of public
vertices and vertex adjacency. Shifts use original caller sites in the user
basis; self images have nonzero shifts and real walls omit `adjacent_shift`.
`has_periodic_shifts` reports requested metadata availability, not exact ideal
positivity.

`return_diagnostics=True` or a non-`'none'` check also runs the independent full
exact E/S audit. Hard insertion, source/profile, provenance and required
representation failures raise `TessellationError` regardless of check action.
An audit-only failure can retain attributed shifts under a non-raising action.
The [planar guide](../../guide/planar.md) records the qualified native profile
and exact/native distinction.

`edge_shift_search`, `validate_edge_shifts`, `repair_edge_shifts` and
`edge_shift_tol` are removed from ordinary `compute`. Their legacy ghost
counterparts remain; the ghost route does not inherit ordinary certification.

::: pyvoro2.planar.api
:::
