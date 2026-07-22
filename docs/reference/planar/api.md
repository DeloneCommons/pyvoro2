# Planar high-level API

`compute(...)` returns the common `pyvoro2.TessellationResult` by default. The
deprecated `return_result: bool | None = None` selector is retained for
compatibility during v0.7. `None` means that the selector was omitted; passing
either boolean warns. The selector is removed in v0.8. New code uses
`output='result'|'cells'`.

::: pyvoro2.planar.api
:::
