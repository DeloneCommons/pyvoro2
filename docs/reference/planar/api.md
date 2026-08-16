# Planar high-level API

`compute(...)` returns the common `pyvoro2.TessellationResult` by default.
Use `output='result'|'cells'` to select structured or raw output explicitly.

`tessellation_check='diagnose'|'warn'|'raise'` computes one final diagnostic;
warning and raising behavior is driven only by its `ok` value.

::: pyvoro2.planar.api
:::
