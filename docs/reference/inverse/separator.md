# Advanced separator fitting

`pyvoro2.inverse.separator` owns the separator implementation and exposes the
canonical core names alongside advanced model, problem, realization, report,
and diagnostic objects.

The fixed-observation fit is distinct from realization-aware active-set
refinement. The active-set API is experimental and separator-specific; it is a
practical outer algorithm without a universal convergence guarantee.

Historical names remain identity aliases during v0.7 and are planned for
removal in v0.8. New code should use the canonical five-name map documented in
the [compatibility reference](../powerfit/index.md).

::: pyvoro2.inverse.separator
:::
