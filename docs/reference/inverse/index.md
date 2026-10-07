# Inverse fitting

`pyvoro2.inverse` is the preferred namespace for fixed-observation and
realization-aware separator fitting. Its exact eight exports are:

```text
SeparatorObservations
resolve_separator_observations
SeparatorFitResult
fit_weights_from_separators
SelfConsistentPowerFitResult
fit_self_consistent_weights_from_separators
weights_to_radii
radii_to_weights
```

The original six exports remain Stable. The new realization-aware facade and
the shared result's final-state inspection are **Provisional** during v0.9.x
soak. Research/path controls remain Experimental under the advanced namespace.

Advanced objective models, problem objects, realization diagnostics, reports,
and Experimental active-set controls are documented under
[separator-specific inverse fitting](separator.md).

The high-level `SeparatorFitResult` keeps its flat compatibility fields and
provides `.state`, `.identification`, `.observation_view(...)`, `.objective`,
`.algebraic`, and `.solver_termination` access. The concrete provisional view
types live only in `pyvoro2.inverse.separator`, so this package's `__all__`
remains deliberately small.

## Realization-aware workflow

```python
result = inverse.fit_self_consistent_weights_from_separators(
    points, observations, domain=domain, max_outer_iter=25,
)
outer_status = result.outer_termination.status
inner_status = result.inner_fit.status
if result.final_state_available:
    weights = result.inner_fit.weights
    selected_rows = result.active_mask
    same_images = result.final_realization.realized_same_shift
else:
    reason = result.final_state_unavailable_reason
```

All candidates start active, using the accepted engine defaults. A required
2D Box/RectangularCell or 3D Box/OrthorhombicCell/PeriodicCell determines native
support. `max_outer_iter` uses the existing positive exact non-Boolean index
validation. The facade does not expose `active0`, `options`, hysteresis,
relaxation, cycle-window, weight-step or history controls. Models needing ADMM
require `fit_solver='admm'`; sparse selection is explicit and SciPy remains lazy.

`SelfConsistentPowerFitResult` is the identical advanced class, returned directly.
Its final fit, realization, final candidate diagnostics, direct `active_mask`,
observations, source/policy/mismatch views, outer metadata and availability are
Provisional through either import route. `history`, `path`, `path_summary`,
`marginal_constraints`, iteration objects, counters, marginal classifications
and path-derived status labels remain Experimental, also in records/reports.
The facade returns `history=None` and preserves other existing path data.

Outer outcomes are `self_consistent`, `cycle_detected`, `max_outer_iter`,
`infeasible_active_set` and `numerical_failure`. `converged` means only outer
self-consistency; `final_refit_converged` reports the final inner fit. Cycles and
outer limits can have available state, and finite weighted inner `max_iter`
states can be available without inner convergence. Final-only failure preserves
the outer stop and never reuses earlier geometry. No `success` Boolean is added.

Without usable weights, realization, candidate diagnostics, residual summaries,
records and optional tessellation diagnostics are `None`. Output switches do
not disable internal geometry or required semantic audits. Missing-row measures
may be NaN in arrays and structurally null in strict report-v3 records; a wrong-image measure
must be read with explicit image-matching flags. Full image-qualified cell
boundaries remain topology authority. Configuration, binding, diagnostic-raise,
dependency and mandatory certificate failures propagate as exceptions.

The fixed-observation solver remains a separate algorithm. Resolved observation
metadata is authoritative, unconditional argument checks still occur, and exact
points/domain source association cannot be replaced by lattice equivalence.

::: pyvoro2.inverse
:::
