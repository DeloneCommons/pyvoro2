# Inverse fitting

`pyvoro2.inverse` is the canonical high-level namespace for fixed-observation
separator fitting. Its deliberately small public surface contains the resolved
observation container, resolver, fit result, solver entry point, and neutral
weight/radius transforms.

Advanced objective models, problem objects, realization diagnostics, reports,
and experimental active-set refinement are documented under
[separator-specific inverse fitting](separator.md).

The high-level `SeparatorFitResult` keeps its flat compatibility fields and
provides `.state`, `.identification`, `.observation_view(...)`, `.objective`,
`.algebraic`, and `.solver_termination` access. The concrete provisional view
types live only in `pyvoro2.inverse.separator`, so this package's `__all__`
remains deliberately small.

::: pyvoro2.inverse
:::
