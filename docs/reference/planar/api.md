# Planar high-level API

`compute(...)` returns the common `pyvoro2.TessellationResult` by default. The
deprecated `return_result: bool | None = None` selector is retained for
compatibility. `None` means that the selector was omitted; passing either
boolean warns. New code uses `output='result'|'cells'`.

::: pyvoro2.planar.api
:::
