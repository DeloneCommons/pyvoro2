# High-level API

`compute(...)` returns `pyvoro2.TessellationResult` by default. Use
`output='cells'` only when a low-level workflow deliberately needs raw cell
dictionaries or the raw diagnostics tuple.

`tessellation_check='diagnose'` computes diagnostics without acting on the
result. `'warn'` emits one summary warning when `diagnostics.ok` is false, and
`'raise'` raises `TessellationError` in exactly the same case.

::: pyvoro2.api
:::
