# Deprecated power-fitting compatibility package

!!! warning "Deprecated in v0.7"
    `pyvoro2.powerfit` is retained only for historical imports during v0.7 and
    will be removed in v0.8. Use `pyvoro2.inverse` for the normal
    fixed-observation workflow or `pyvoro2.inverse.separator` for advanced
    separator-specific functionality.

| Historical name | Canonical v0.7 name |
|---|---|
| `PairBisectorConstraints` | `SeparatorObservations` |
| `resolve_pair_bisector_constraints` | `resolve_separator_observations` |
| `PowerFitProblem` | `SeparatorFitProblem` |
| `PowerWeightFitResult` | `SeparatorFitResult` |
| `fit_power_weights` | `fit_weights_from_separators` |

Because `PowerWeightFitResult` is an identity alias, its instances also expose
the v0.7 layered result properties. The new provisional view type names are
exported only by `pyvoro2.inverse.separator`; they do not broaden the historical
`pyvoro2.powerfit.__all__` surface.

::: pyvoro2.powerfit
:::
