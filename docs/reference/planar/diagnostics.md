# Planar diagnostics API

Planar diagnostics share the spatial severity contract. Invalid cell areas and
closure failures are errors; expected-ID handling distinguishes standard,
power, and undeclared (`None`) modes; and explicitly requested periodic edge
reciprocity is required. Marked analyses reset analyzer-owned edge flags before
recording current failures. Strict tessellation validation consumes final
`TessellationDiagnostics.ok`; optional reciprocity findings remain nonfatal.

::: pyvoro2.planar.diagnostics
:::
