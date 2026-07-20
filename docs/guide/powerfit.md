# Fit power weights from separator observations

`pyvoro2` can solve a fixed-site inverse problem for **power/Laguerre
tessellations**: fit power weights so that selected pairwise separators land at
desired locations along the connectors between sites.

New code should begin with the concise fixed-observation surface in
`pyvoro2.inverse`. Advanced objective models, realization checks, reports, and
the experimental active-set outer loop live in
`pyvoro2.inverse.separator`.

The API is geometry-first and domain-agnostic. The same high-level functions
work with supported 3D domains and planar `pyvoro2.planar` domains. Downstream
code decides:

- which site pairs are observed or proposed;
- which periodic image shift belongs to each observation;
- the target separator location;
- and the confidence of each observation.

`pyvoro2` then provides the mathematical and geometric layers:

- resolve and validate separator observations;
- fit power weights under a configurable convex model;
- expose graph, connectivity, and hard-feasibility diagnostics;
- compute the resulting power tessellation;
- detect which requested pairs and periodic images are realized;
- and optionally run a realization-aware active-set outer loop.

The fixed-observation inverse fit and the geometric realization check answer
different questions. For the API-independent derivation, see
[Inverse fitting from separator observations](../theory/separator-inverse.md).

## Geometry of one pair

For a pair of sites `i` and `j`, choose one specific image `q_j` of site `j`.
In a nonperiodic domain, `q_j = p_j`. Let

- `d = ||q_j - p_i||`,
- `z = w_i - w_j`,

where `w` are the fitted power weights.

Then the separator position along the connector is affine in `z`:

$$
 t(z) = \frac{1}{2} + \frac{z}{2 d^2}
$$

for normalized fraction, and

$$
 s(z) = \frac{d}{2} + \frac{z}{2 d}
$$

for absolute position measured from site `i`.

This is why `pyvoro2` exposes the measurement type explicitly: a loss in
fraction-space and a loss in position-space are **different optimization
problems**.

## Step 1: resolve separator observations once

```python
import numpy as np
import pyvoro2 as pv
import pyvoro2.inverse as inverse
import pyvoro2.inverse.separator as separator

points = np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]], dtype=float)
box = pv.Box(((-5, 5), (-5, 5), (-5, 5)))

observations = inverse.resolve_separator_observations(
    points,
    [(0, 1, 0.25)],
    measurement='fraction',
    domain=box,
)
```

Each raw tuple is `(i, j, value[, shift])`, where `shift=(na, nb, nc)` is the
integer lattice image applied to site `j`.

The resolved `SeparatorObservations` object stores the validated pair indices,
shifts, connector geometry, and targets in both fraction and position form.

## Step 2: define the fitting model

```python
model = separator.FitModel(
    mismatch=separator.SquaredLoss(),
    feasible=separator.Interval(0.0, 1.0),
    penalties=(
        separator.ExponentialBoundaryPenalty(
            lower=0.0,
            upper=1.0,
            margin=0.05,
            strength=1.0,
            tau=0.01,
        ),
    ),
)
```

The model separates three ideas:

- `mismatch=`: how target-vs-predicted separator locations are scored,
- `feasible=`: hard admissible sets such as an interval or fixed value,
- `penalties=`: soft penalties such as outside-interval or near-boundary
  repulsion.

Built-in pieces currently include:

- `SquaredLoss()`
- `HuberLoss(delta=...)`
- `Interval(lower, upper)`
- `FixedValue(value)`
- `SoftIntervalPenalty(lower, upper, strength=...)`
- `ExponentialBoundaryPenalty(...)`
- `ReciprocalBoundaryPenalty(...)`
- `L2Regularization(...)`

## Step 3: fit power weights

```python
fit = inverse.fit_weights_from_separators(
    points,
    observations,
    model=model,
)
```

The result contains:

- fitted `weights` and shifted `radii`,
- predicted separator locations in both fraction and position form,
- residuals in the chosen measurement space,
- `edge_diagnostics` with quantities such as `z_obs`, `z_fit`, and weighted
  difference-space inconsistency summaries,
- `objective_breakdown` with mismatch, penalty, and regularization totals for
  the packaged candidate weights,
- solver/termination metadata including optional `status_detail`,
- and explicit infeasibility reporting for contradictory hard constraints.

For example, if hard interval or equality restrictions cannot all hold
simultaneously, the fit returns:

- `status == 'infeasible_hard_constraints'`
- `hard_feasible == False`
- `weights is None`
- `conflict` with a compact contradiction witness
- `conflicting_constraint_indices` for the participating rows

instead of pretending the issue is merely slow convergence.

Both low-level fits and active-set results also provide `to_records(...)` helpers
that turn per-constraint diagnostics into plain Python rows for downstream
packages, table exporters, or custom reporting.

### Measurement-space and difference-space diagnostics

`SeparatorFitResult` exposes two complementary diagnostic views.

Measurement-space quantities live in the same space as the chosen separator
targets:

- `target`, `predicted`, `residuals`

Difference-space quantities live in the implied weight-difference model

\[
 y = \beta + \alpha (w_i - w_j),
\]

with

\[
 z_{\mathrm{obs}} = \frac{y_{\mathrm{target}} - \beta}{\alpha},
 \qquad
 z_{\mathrm{fit}} = w_i - w_j.
\]

The edge diagnostics expose `alpha`, `beta`, `z_obs`, `z_fit`, the difference-space
residual `z_obs - z_fit`, and edge weights

\[
 \omega = \mathrm{confidence} \cdot \alpha^2.
\]

The exported `weighted_rmse` is defined explicitly as

\[
 \sqrt{\mathrm{mean}(\omega r^2)},
\]

not as `sqrt(sum(w r^2) / sum(w))`. That distinction matters when you compare
results to other code that uses a normalized weighted RMSE convention.

For radii output, the API makes the **global representation shift** explicit:

- by default, `weights_to_radii(...)` uses the minimal common additive shift that
  makes all returned radii non-negative;
- `r_min=` remains available as a compatibility-oriented convenience when a
  specific minimum radius is required;
- `weight_shift=` lets downstream code request one explicit common shift.

Both conversion directions require finite inputs and finite representable
results. `weights_to_radii(...)` also requires a finite, non-negative `r_min`.
If squaring a radius or applying the common shift would overflow, the transform
raises `ValueError`; it does not clip or substitute a fallback value.

One common shift of every weight is the geometric gauge of the complete power
diagram. Disconnected observation graphs introduce an additional and different
ambiguity: each effective component can be shifted independently without
changing its observed separator equations, but relative shifts between
components can change the complete realized tessellation.

The current solver therefore chooses and reports a component-alignment policy:

- standalone fits center each disconnected effective component to mean zero;
- if a zero-strength regularization reference is supplied, each component is
  aligned to the reference mean on that component;
- `connectivity_check='none'|'diagnose'|'warn'|'raise'` controls whether the
  unidentified component offsets are reported, warned about, or raised.

These component offsets are conventions or prior information, not values
identified by the disconnected separator observations. Inspect realization
results when cross-component competition matters.

Connectivity is computed on the graph of **site unknowns**, not on a graph of
periodic images. A periodic shift changes the geometry of one observation row
and of realized-boundary matching, but it does not create an additional fitted
unknown. Interpenetrating periodic nets therefore remain disconnected unless an
observation actually couples their site indices.

## Step 4: check geometric realization

A requested pairwise separator is not automatically a realized face in the full
power tessellation. After fitting, you can ask which requested pairs became real
neighbors.

```python
realized = separator.match_realized_pairs(
    points,
    domain=box,
    radii=fit.radii,
    constraints=observations,
    return_boundary_measure=True,
    return_tessellation_diagnostics=True,
    unaccounted_pair_check='warn',
)
```

This returns purely geometric diagnostics:

- whether each pair is realized at all,
- whether it is realized with the **same** requested periodic shift,
- whether only some **other** image is realized,
- whether one of the endpoint cells is empty,
- an optional boundary measure of the matched boundary
  (**face area** in 3D, **edge length** in 2D),
- any realized unordered site pairs that were absent from the candidate set,
  exposed through `unaccounted_pairs`,
- and optional tessellation-wide diagnostics.

## Optional: refine the active set

For sparse or noisy candidate sets, the useful high-level workflow is often:

1. fit on a current active set;
2. run the actual power tessellation;
3. keep or re-add observations according to realized support;
4. repeat until active and realized sets agree.

This is a practical realization-aware outer algorithm. It is not part of the
exact graph/Laplacian theory of the fixed-observation inner fit, and its
termination status and path diagnostics should be inspected explicitly.

The explicitly experimental separator API provides this as:

```python
result = separator.solve_self_consistent_power_weights(
    points,
    observations,
    domain=box,
    model=model,
    options=separator.ActiveSetOptions(
        add_after=1,
        drop_after=2,
        relax=0.5,
        max_iter=25,
        cycle_window=8,
    ),
    return_history=True,
    return_boundary_measure=True,
    return_tessellation_diagnostics=True,
)
```

The solver is generic:

- it never invents candidate pairs,
- it never silently changes the user-supplied periodic image,
- it uses realized faces rather than any domain-specific contact logic,
- it supports hysteresis, under-relaxation, cycle detection, and marginal-pair
  reporting.

## Reading the final diagnostics

`solve_self_consistent_power_weights(...)` returns both a final low-level fit and
rich per-constraint diagnostics.

Useful fields include:

- `result.constraints`: the resolved pair set used throughout the solve,
- `result.active_mask`: final active-set membership,
- `result.realized`: realized-face matching diagnostics, including
  `unaccounted_pairs` when the final tessellation realizes candidate-absent
  pairs,
- `result.connectivity`: final candidate-graph and active-graph connectivity
  diagnostics plus the component-alignment policy used for disconnected
  components,
- `result.path_summary`: compact optimization-path diagnostics that answer
  questions such as whether the fit-active graph was **ever** disconnected,
  whether active-component offsets were ever not identified by the pairwise
  data, and whether the tessellation ever realized pairs that were absent from
  the candidate set,
- `result.history`: optional per-iteration rows; each row distinguishes the
  fit-active mask (`n_active_fit`) from the post-toggle mask used for the next
  iteration (`n_active`), and also records fit-active component counts and the
  number of realized pairs absent from the candidate set on that iteration,
- `result.diagnostics`: per-constraint targets, predictions, residuals,
  endpoint-empty flags, boundary measure, toggle counts, and generic status
  labels,
- `result.rms_residual_all` / `result.max_residual_all`: summaries over **all**
  candidate constraints,
- `result.tessellation_diagnostics`: final tessellation-wide checks,
- `result.marginal_constraints`: indices of toggling / cycle / wrong-shift
  pairs.

Transient path diagnostics are intentionally **inspectable** rather than
noisy: final-state `connectivity_check=` / `unaccounted_pair_check=` policies
still control warnings or exceptions, while `result.path_summary` and
`result.history` expose optimization-path events without turning every transient
component split into a default warning.

Status labels are intentionally generic, for example:

- `stable_active`
- `stable_inactive`
- `toggled_active`
- `toggled_inactive`
- `realized_other_shift`
- `active_unrealized`
- `cycle_member`

## Exporting diagnostics as plain records

Downstream packages often want rows rather than structured NumPy-heavy result
objects. The power-fitting package now exposes lightweight record exporters:

```python
rows = result.to_records(use_ids=True)
fit_rows = result.fit.to_records(result.constraints, use_ids=True)
realized_rows = result.realized.to_records(result.constraints, use_ids=True)
if result.fit.conflict is not None:
    conflict_rows = result.fit.conflict.to_records(ids=result.constraints.ids)
```

These helpers keep the core API numerical while making it straightforward to
feed results into custom logs, JSON encoders, or dataframe construction in a
downstream package.

## Full report bundles

When downstream code wants a single nested object rather than several row sets,
use the report helpers or the corresponding result methods:

```python
fit_report = fit.to_report(observations, use_ids=True)
realized_report = realized.to_report(observations, use_ids=True)
solve_report = result.to_report(use_ids=True)
```

The standalone helpers are also exported:

```python
fit_report = separator.build_fit_report(fit, observations, use_ids=True)
solve_report = separator.build_active_set_report(result, use_ids=True)
```

These report bundles stay plain-Python and JSON-friendly. They are useful when
a downstream package wants a complete diagnostic payload for logging, caching,
or UI work without manually unpacking NumPy-heavy result objects.

To serialize them directly:

```python
text = separator.dumps_report_json(solve_report, sort_keys=True)
separator.write_report_json(solve_report, 'solve_report.json', sort_keys=True)
```

## Native Huber fit on sparse outliers

A short robust-fitting example looks like this:

```python
model = separator.FitModel(mismatch=separator.HuberLoss(delta=0.03))
fit = inverse.fit_weights_from_separators(
    points,
    observations,
    model=model,
    solver='admm',
)

print(fit.status, fit.rms_residual)
print(fit.edge_diagnostics.weighted_rmse)
```

This is still the native `pyvoro2` solver path. The robust part comes from the
measurement-space Huber objective, while `edge_diagnostics` lets you inspect
the mathematical difference-space residuals directly.

## Advanced problem export and result packaging

For research workflows or external solvers, `pyvoro2` now exposes the resolved
inverse problem itself:

```python
observations = inverse.resolve_separator_observations(points, raw_observations)
problem = separator.build_power_fit_problem(observations, model=model)

weights = some_external_solver(problem)
result = separator.build_power_fit_result(
    problem,
    weights,
    solver='external',
    status='external_failure',
    status_detail='candidate iterate only',
)
```

This keeps `fit_weights_from_separators(...)` solver-owned while giving downstream
code a public export of the mathematics, prediction formulas, objective
evaluation, and result packaging.

## Current scope

The current implementation supports both **3D** domains through `pyvoro2` and
**2D planar** domains through `pyvoro2.planar`. The shared solver vocabulary is
intentionally dimension-safe: fitting is phrased in terms of separator
observations and generic boundary measure rather than chemistry-specific or
3D-only semantics.

The historical `pyvoro2.powerfit` package, broad top-level separator exports,
and the five historical core names remain deprecated compatibility routes for
v0.7. They are planned for removal in v0.8. See the
[compatibility reference](../reference/powerfit/index.md),
[architecture](../development/architecture.md), and
[API lifecycle](../development/api-lifecycle.md).

The main current restriction is geometric, not algebraic:

- 3D supports `Box`, `OrthorhombicCell`, and triclinic `PeriodicCell`;
- 2D currently supports `Box` and rectangular `RectangularCell`;
- there is **no** planar oblique-periodic `PeriodicCell` yet.

### Current objective-model scope

The built-in objective family remains compact:

- mismatch terms: `SquaredLoss`, `HuberLoss`
- hard feasibility: `Interval`, `FixedValue`
- soft penalties: `SoftIntervalPenalty`, `ExponentialBoundaryPenalty`,
  `ReciprocalBoundaryPenalty`
- regularization: `L2Regularization`

That set is broad enough for the current generic inverse workflow while keeping
hard-feasibility checks, residual diagnostics, and solver behavior easy to
reason about.

Additional mismatch or penalty families should wait until downstream packages
validate a concrete need for them. In particular, pyvoro2 does **not** try to
freeze an open-ended callback API for arbitrary user-defined objectives.

## Worked example notebooks

Three focused notebooks complement the guide:

- [`06_powerfit_reports`](../notebooks/06_powerfit_reports.md)
  shows how to export low-level fits, realized-pair diagnostics, and
  self-consistent active-set results as rows or JSON-friendly reports.
- [`07_powerfit_infeasibility`](../notebooks/07_powerfit_infeasibility.md)
  shows how contradictory hard restrictions are reported through
  `status`, `is_infeasible`, `conflict`, and report bundles.
- [`08_powerfit_active_path`](../notebooks/08_powerfit_active_path.md)
  shows how to inspect transient active-set path diagnostics separately from
  the final-state report objects.

These examples are aimed at downstream packages that want to keep the solver
API numerical while still producing human-readable logs, cached payloads, or UI
views.
