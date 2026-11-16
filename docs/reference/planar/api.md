# Planar high-level API

`pyvoro2.planar.compute(...)` returns the shared
[TessellationResult (2D/3D)](../result.md) by default.
Use `output='result'|'cells'` to select structured or raw output explicitly.

`tessellation_check='diagnose'|'warn'|'raise'` computes one final diagnostic;
warning and raising behavior is driven only by its `ok` value.

::: pyvoro2.planar.api
:::
