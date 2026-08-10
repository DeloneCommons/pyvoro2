# Duplicate check

Forward `compute`, `locate`, and `ghost_cells` always enforce a private
backend-safety floor before native insertion. The standalone helpers documented
below remain explicit user-threshold diagnostics; the forward
`duplicate_check`, `duplicate_threshold`, and `duplicate_wrap` arguments cannot
weaken mandatory safety.

::: pyvoro2.duplicates
:::
