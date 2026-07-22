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

For namespace selection and lifecycle status, begin with
[Choosing an API](choosing-api.md). The [glossary](glossary.md) defines gauge,
component offsets, representation shift, realized face, and active set. Users
migrating from v0.6.3 should also read the
[v0.7 migration guide](migration-v0.7.md).

The v0.7 high-level resolver, observation container, fit result, fit entry
point, and neutral transforms are **stable**. Advanced models, problem and
operator views, report/realization helpers, and layered convenience views are
**provisional**. Active-set refinement is **experimental**. The optional
explicit sparse quadratic backend is **provisional** and supports large static
sparse observation graphs only.

## Canonical downstream integration

The repository-owned `examples/chemvoro_workflow.py` script is the canonical
chemistry-neutral downstream example. It uses only preferred v0.7 imports and
keeps application metadata outside pyvoro2 in an external-ID-keyed sidecar:

```python
import numpy as np
import pyvoro2 as pv
import pyvoro2.inverse as inverse
import pyvoro2.inverse.separator as separator

points = np.array([[0.1, 0.5, 0.5], [0.9, 0.5, 0.5]])
site_ids = np.array([205, 101], dtype=int)
metadata_by_id = {
    205: {'label': 'left-site', 'source_row': 0},
    101: {'label': 'right-site', 'source_row': 1},
}
cell = pv.PeriodicCell(
    vectors=((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))
)

observations = inverse.resolve_separator_observations(
    points,
    [(205, 101, 0.5, (-1, 0, 0))],
    ids=site_ids,
    index_mode='id',
    domain=cell,
    image='given_only',
)
fit = inverse.fit_weights_from_separators(
    points,
    observations,
    connectivity_check='diagnose',
)
weights = fit.state.mathematical_weights

result = pv.compute(
    points,
    domain=cell,
    ids=site_ids,
    mode='power',
    weights=weights,
    include_empty=True,
    return_faces=True,
    return_face_shifts=True,
)
boundaries_by_input = result.require_boundaries()

realized = separator.match_realized_pairs(
    points,
    domain=cell,
    weights=weights,
    constraints=observations,
)
fit_rows = fit.to_records(observations, use_ids=True)
fit_report = fit.to_report(observations, use_ids=True)
```

`result.ids`, `result.cell_measures`, `result.empty_mask`, and the collections
returned by `require_boundaries()` share input-site order. A downstream package
can therefore build its own ID-labelled rows with an explicit
`None if empty else measure` policy without reading raw backend order. Boundary
records preserve external neighbor IDs and `adjacent_shift`; fit and
realization records use external IDs when `use_ids=True`.

Inspect `fit.identification` before comparing fitted state across observation
components. `fit.state.global_representation_shift` records only the common
backend representation shift; it is not a fitted physical quantity. The
complete executable example also demonstrates same-image and wrong-image
realization reporting and JSON-friendly report export. The public deterministic
paper-style ladder is in `examples/paper_regressions.py`; both scripts and their
run instructions are described in `examples/README.md`.

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

For squared-loss static fits, the default `solver='auto'` continues to choose
the NumPy dense direct path. Select the optional SciPy sparse-direct path
explicitly when the fixed observation graph is large and local:

```bash
python -m pip install "pyvoro2[sparse]"
```

```python
sparse_fit = inverse.fit_weights_from_separators(
    points,
    observations,
    solver='sparse',
)
print(sparse_fit.solver_termination.backend)  # sparse
```

`solver='analytic'` explicitly selects the dense quadratic path. There is no
automatic size threshold in v0.7: `auto` remains dense for compatibility and
predictable dependency behavior. Sparse execution supports `SquaredLoss` with
optional L2 regularization and no scalar penalties or hard restrictions.
Huber mismatch, hard constraints, and scalar-penalty models continue to use the
existing ADMM path and reject `solver='sparse'`.

The sparse path changes only matrix storage and direct linear algebra. It uses
the same observation rows, periodic image labels, effective graph, component
anchors, and final component-alignment policy as the dense path. The returned
backend is inspectable through both `fit.solver` and
`fit.solver_termination.backend`.

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

### Read a fit through its scientific layers

The layers are deliberately one-directional rather than one monolithic result:

```text
resolved observations + sites
            |
            v
fixed-observation fit
    |-- fitted state and identification
    |-- observation predictions and objective
    |-- graph and operator diagnostics
    `-- fixed-solver termination
            |
            v  (explicit forward realization request)
realized geometry and requested-image matching
            |
            v  (experimental outer loop only)
active-set path and outer termination
```

A small algebraic residual does not imply that the requested pair is a realized
face. Realization and the active-set path are therefore separate objects and
separate lifecycle layers.

`SeparatorFitResult` keeps all existing flat fields and adds lightweight
layered views. The views reference existing arrays; they do not copy fitted or
observation data.

```python
state = fit.state
print(state.mathematical_weights)
print(state.backend_radii, state.global_representation_shift)

observation_fit = fit.observation_view(observations)
print(observation_fit.targets, observation_fit.confidence)
print(observation_fit.predictions, observation_fit.residuals)

identification = fit.identification
print(identification.effective_observation_components)
print(identification.relative_component_offsets_identified_by_data)
print(identification.component_offsets_selected_by_objective)
print(identification.component_alignment_policy)

termination = fit.solver_termination
print(termination.status, termination.backend, termination.converged)
```

The identification view always reports
`global_geometric_gauge_identified_by_data == False`: separator differences do
not identify one common additive constant. The informative observation graph
contains only positive-confidence separator rows. Its connected components are
reported by `effective_observation_components`, and
`relative_component_offsets_identified_by_data` is true exactly when this graph
is connected. A positive L2 regularization term guarantees selection of
otherwise free component offsets and is reported by
`component_offsets_selected_by_objective`. Other supported scalar penalties are
not classified as selecting offsets because they may have flat regions or zero
strength.

`identification.unconstrained_sites` reports sites isolated in that informative
graph. This can differ from the compatibility diagnostic
`fit.connectivity.unconstrained_points`, which retains its candidate-graph
meaning. A site mentioned only by zero-confidence rows is candidate-connected
but observationally unconstrained. Hard restrictions and penalties apply
independently of mismatch confidence and may constrain or bound offsets, but
they are not separator-observation data and never add informative graph edges.
An exact hard equality may fix an offset in a particular model; the current
identification view deliberately does not attempt to summarize that separate
constraint-identifiability question.

With `connectivity_check='none'`, connectivity-derived identification values
are `None`; accessing the view does not rebuild diagnostics that the caller
disabled.

The state view uses `global_representation_shift` for the common shift used to
form non-negative backend radii. This backend representation choice selects a
representative within the global geometric gauge; it is distinct from
independent component offsets and is not information recovered from separator
observations. The compatibility flat field `fit.weight_shift` has exactly this
meaning. Likewise, the compatibility field
`fit.connectivity.gauge_policy` contains the same string as the canonical
`component_alignment_policy`, despite its historical name.

The complete mapping is:

| Scientific layer | Canonical access | Existing flat fields or objects |
|---|---|---|
| Fitted state and backend representation | `fit.state` | `weights`, `radii`, `weight_shift` |
| Identification and component alignment | `fit.identification` | `connectivity` and its effective components, offset flags, effective-graph isolated sites, and `gauge_policy` |
| Observation-space fit | `fit.observation_view(observations)` | `measurement`, `target`, `predicted*`, `residuals`, residual summaries, `used_shifts`; confidence comes from `observations` |
| Objective contributions | `fit.objective` | `objective_breakdown` |
| Algebraic diagnostics | `fit.algebraic` | `edge_diagnostics`, `connectivity` |
| Fixed-solver termination | `fit.solver_termination` | `status`, `status_detail`, `solver`, `n_iter`, `converged`, `hard_feasible`, `conflict`, `warnings` |
| Requested-image matching and realized geometry | `realized.requested_image_matching`, `realized.geometry` | all `RealizedPairDiagnostics` fields |
| Experimental outer-loop termination and path | `result.outer_termination`, `result.path` | active-set termination fields, `active_mask`, `marginal_constraints`, `history`, `path_summary` |

The observation accessor accepts the originating resolved observations or an
independently resolved set with the same complete contents. It validates pair
indices, confidence, targets, requested shifts, and resolved geometry before
presenting observation-owned arrays beside the fit predictions. Its private
source binding survives shallow copies, deep copies, pickle round trips, and
`dataclasses.replace(...)`; a directly constructed result without source
observations cannot safely provide this view and raises `ValueError`.

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
ambiguity: each informative component can be shifted independently without
changing its observed separator equations, but relative shifts between
components can change the complete realized tessellation.

The current solver therefore chooses and reports a component-alignment policy:

- the numerical decomposition follows a model-coupling mask that includes
  positive-confidence rows and rows touched by hard restrictions or penalties;
- without positive regularization or a supplied reference, disconnected
  model-coupling components are centered to mean zero;
- if a zero-strength regularization reference is supplied, disconnected
  model-coupling components are aligned to the reference mean;
- positive L2 regularization selects weights relative to its zero or supplied
  reference;
- `connectivity_check='none'|'diagnose'|'warn'|'raise'` controls whether the
  observationally unidentified component offsets are reported, warned about,
  or raised.

The model-coupling mask is exposed under the historical problem-field name
`offset_identifying_constraint_mask`; despite that name, it is a solver
decomposition detail and is not an identifiability claim. Hard restrictions may
bound offsets, and penalties or numerical conventions may choose a returned
representative without guaranteeing a unique optimum. None of those values is
information identified by disconnected separator observations. Inspect
realization results when cross-component competition matters.

Connectivity is computed on the graph of **site unknowns**, not on a graph of
periodic images. A periodic shift changes the geometry of one observation row
and of realized-boundary matching, but it does not create an additional fitted
unknown. Interpenetrating periodic nets therefore remain disconnected unless an
observation actually couples their site indices.

### Inspect the observation graph and quadratic normal operator

Advanced workflows can inspect the fixed-observation mathematics directly from
the public problem. The graph and operator views are provisional and are
exported only from `pyvoro2.inverse.separator`.

```python
import numpy as np
import pyvoro2.inverse as inverse
import pyvoro2.inverse.separator as separator

points = np.array(
    [[0.0, 0.0], [2.0, 0.0], [5.0, 0.0]],
    dtype=float,
)
observations = inverse.resolve_separator_observations(
    points,
    [(0, 1, 0.20), (1, 2, 0.70), (0, 2, 0.40)],
    confidence=[1.0, 0.5, 2.0],
)
model = separator.FitModel(
    regularization=separator.L2Regularization(
        strength=0.25,
        reference=np.array([1.0, -1.0, 2.0]),
    )
)
problem = separator.build_power_fit_problem(observations, model=model)

graph = problem.observation_graph
operator = problem.quadratic_operator
B = graph.incidence_dense()
L_obs = operator.observation_laplacian_dense()
A = operator.regularized_normal_matrix_dense()
b_obs = operator.observation_rhs
b = operator.regularized_normal_rhs

fit = inverse.fit_weights_from_separators(
    points,
    observations,
    model=model,
    connectivity_check='diagnose',
)
assert np.allclose(B.T @ fit.weights, problem.predict_difference(fit.weights))
assert np.allclose(
    graph.beta + graph.alpha * (B.T @ fit.weights),
    fit.predicted,
)
assert np.allclose(A @ fit.weights, b)

# Optional conversion; SciPy is also used by solver='sparse'.
try:
    B_sparse = graph.incidence_sparse(format='csc')
    L_obs_sparse = operator.observation_laplacian_sparse(format='csr')
except ImportError:
    B_sparse = L_obs_sparse = None
```

For `n` sites and `m` observations, `B.shape == (n, m)`. Column `r` has
`+1` at `graph.site_i[r]` and `-1` at `graph.site_j[r]`, hence
`B.T @ weights == weights[site_i] - weights[site_j]`. Every observation remains
a column: repeated measurements and observations for different periodic images
of the same site pair are not deduplicated. `graph.observation_indices` maps
those columns back to the resolved input indices, while
`graph.requested_shifts` retains image identity.

The view names the two systems separately:

$$
L_{\mathrm{obs}}=B\operatorname{diag}(\rho)B^\mathsf{T},
\qquad
b_{\mathrm{obs}}=B(\rho z^{\mathrm{obs}}),
$$

and, for L2 strength $\lambda$ and reference $w^{\mathrm{ref}}$,

$$
A=L_{\mathrm{obs}}+\lambda I,
\qquad
b=b_{\mathrm{obs}}+\lambda w^{\mathrm{ref}}.
$$

There is no extra factor of two. Zero-confidence rows stay in `B` and in all
row-facing arrays, but their informative mask is false and `rho` is zero, so
they add nothing to $L_{\mathrm{obs}}$ or $b_{\mathrm{obs}}$ and do not connect
informative components.

Without positive L2 regularization, the observation-Laplacian nullity is the
number of informative components, including isolated sites. One common null
direction is global geometric gauge; additional component constants are
unidentified offsets that can affect the complete realized diagram. Positive
L2 regularization removes those null directions from `A`, but it does not make
the separator observations themselves connected or observationally identify
the offsets.

The repository benchmark harness
`benchmarks/benchmark_sparse_separator.py` covers small dense-favorable and
medium/large molecular-shaped locality graphs, including disconnected static
components. It records dense/sparse assembly, direct-solve and complete-fit
times, matrix storage, and numerical agreement. This scalability support is
for large **static** geometries. It does not provide trajectory processing, MD
frame reuse, prepared solvers across changing frames, parallel tessellation,
GPU/distributed execution, or scalable all-pairs observation construction.

The fixed normal system is available only for `SquaredLoss` with no scalar
penalties. Huber mismatch and scalar-penalty models still expose
`problem.observation_graph`, but `problem.quadratic_operator` raises
`ValueError` rather than claiming to represent their full objective. Hard
interval or equality restrictions may coexist with the quadratic view; they
remain separately visible through `problem.bounds`, and
`operator.normal_equations_characterize_fit` is false because a constrained
optimum need not solve the unconstrained normal equation.

## Step 4: check geometric realization

A requested pairwise separator is not automatically a realized face in the full
power tessellation. After fitting, you can ask which requested pairs became real
neighbors.

```python
realized = separator.match_realized_pairs(
    points,
    domain=box,
    weights=fit.state.mathematical_weights,
    constraints=observations,
    return_boundary_measure=True,
    return_tessellation_diagnostics=True,
    unaccounted_pair_check='warn',
)
```

`weights=` is the preferred realization input. `radii=` remains accepted as a
backend-compatible representation route; supply exactly one. The common shift
used to form radii is a geometric representation choice, not another fitted
scientific variable.

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

Realization remains an explicit, separate computation. Read periodic-image
matching and optional geometry independently:

```python
matching = realized.requested_image_matching
print(matching.any_realization, matching.same_requested_shift)
print(matching.another_periodic_shift, matching.realized_shifts)

geometry = realized.geometry
print(geometry.endpoint_i_empty, geometry.endpoint_j_empty)
print(geometry.boundary_measure, geometry.tessellation_diagnostics)
```

The fit result never computes or owns a tessellation automatically.

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

The layered aliases make the inner/outer boundary explicit:

```python
final_inner_fit = result.inner_fit
final_realization = result.final_realization
candidate_diagnostics = result.candidate_diagnostics
outer_termination = result.outer_termination
path = result.path
```

`path.active_mask`, `path.marginal_constraint_indices`, `path.history`, and
`path.summary` share the existing active-set result data. The outer termination
is experimental and does not change the exact fixed-observation meaning of
`final_inner_fit`.

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

Existing report sections map to the same layers without changing report keys:

| Report section | Layer |
|---|---|
| fit `weights`, `radii`, `weight_shift` | state |
| fit `connectivity` | identification and graph context for algebraic diagnostics |
| fit `constraints`, `fit_records`, and residual summaries in `summary` | observations |
| fit `objective_breakdown` | objective |
| fit `edge_diagnostics` | algebraic diagnostics |
| fit `summary`, `conflict`, and `warnings` | fixed solver termination |
| realized `records`, `unrealized` | requested-image matching |
| realized `unaccounted_pairs`, `tessellation_diagnostics`, and optional values in `records` | realized geometry |
| active-set `fit`, `realized`, `diagnostics`, `summary`, `history`, and `path_summary` | final inner fit, final realization, candidate diagnostics, outer termination, and active-set path |

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
v0.7. They will be removed in v0.8. See the
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

Four focused notebooks complement the guide:

- [`04_powerfit`](../notebooks/04_powerfit.md) presents the canonical
  external-ID, weight-first periodic workflow and then introduces advanced
  objective and active-set features.

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
