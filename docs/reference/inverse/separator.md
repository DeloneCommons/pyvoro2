# Advanced separator fitting

`pyvoro2.inverse.separator` owns the separator implementation and exposes the
canonical core names alongside advanced model, problem, realization, report,
and diagnostic objects.

The fixed-observation fit is distinct from realization-aware active-set
refinement. The active-set API is experimental and separator-specific; it is a
practical outer algorithm without a universal convergence guarantee.

Most advanced model, problem, operator, report, realization, and layered-view
objects on this page are **provisional**. Active-set refinement is
**experimental**. The optional explicit sparse quadratic backend is
**provisional** and is limited to the static squared-loss branch documented
below.

Historical names remain identity aliases during v0.7 and will be removed in
v0.8. New code should use the canonical five-name map documented in
the [compatibility reference](../powerfit/index.md).

## Layered result access

The provisional view types below organize existing result data without adding
fields to the established result dataclasses or copying their arrays.

| Owning result | Access | View or reused object |
|---|---|---|
| `SeparatorFitResult` | `.state` | `SeparatorFitStateView` |
| `SeparatorFitResult` | `.identification` | `SeparatorIdentificationView` |
| `SeparatorFitResult` | `.observation_view(observations)` | `SeparatorObservationView` |
| `SeparatorFitResult` | `.objective` | existing `PowerFitObjectiveBreakdown` or `None` |
| `SeparatorFitResult` | `.algebraic` | `SeparatorAlgebraicView` containing existing edge and connectivity diagnostics |
| `SeparatorFitResult` | `.solver_termination` | `SeparatorSolverTerminationView` |
| `RealizedPairDiagnostics` | `.requested_image_matching` | `RequestedImageMatchView` |
| `RealizedPairDiagnostics` | `.geometry` | `RealizedGeometryView` |
| `SelfConsistentPowerFitResult` | `.inner_fit`, `.final_realization`, `.candidate_diagnostics` | existing final objects |
| `SelfConsistentPowerFitResult` | `.outer_termination`, `.path` | `ActiveSetTerminationView`, `ActiveSetPathView` |

`observation_view(...)` accepts the originating resolved observations or a
fully equivalent independently resolved set. It checks pair indices,
confidence, targets, requested shifts, and resolved geometry before combining
observation-owned arrays with fit-owned predictions. The private binding is
retained by shallow copies, deep copies, pickle round trips, and
`dataclasses.replace(...)`. Directly constructed results without source
observations fail closed when this accessor is called.
`inspect.signature(SeparatorFitResult)` consequently includes the private
optional keyword-only parameter `_originating_observations_init=None`; it is an
init-only reconstruction channel, not a public result field or user input.

`SeparatorIdentificationView.unconstrained_sites` contains sites isolated in
the informative observation graph, which contains only positive-confidence
separator rows. Zero-confidence rows remain excluded even when hard
restrictions or penalties affect them: those terms may constrain or bound
component offsets, but they are not observational identification. Positive L2
regularization is the only currently supported additional objective reported as
guaranteed to select otherwise free component offsets. The compatibility
diagnostic `ConnectivityDiagnostics.unconstrained_points` retains its
established candidate-graph meaning.

The historical `SeparatorFitProblem.offset_identifying_constraint_mask` name
is preserved for compatibility. That mask includes rows touched by hard
restrictions or penalties because the numerical solver must keep coupled
variables in one subproblem; it does not define the informative observation
graph or claim unique offset selection. Exact hard equalities may fix offsets,
but the current identification view does not provide a general
constraint-identifiability classification.

`global_representation_shift` is a backend representation choice made by adding
one common constant, so it selects a representative within the global geometric
gauge. It is distinct from independent offsets between disconnected observation
components and is not inferred from observations.
`component_alignment_policy` is the canonical access to the policy string also
stored under the historical compatibility name `ConnectivityDiagnostics.gauge_policy`.

## Problem-owned graph and quadratic views

`SeparatorFitProblem` owns two additional provisional inspection views. They do
not change its dataclass fields or bind the problem into a fit result.

| Problem access | Public view | Main contents |
|---|---|---|
| `.observation_graph` | `SeparatorObservationGraphView` | `n_sites`, `n_observations`, `site_i`, `site_j`, `observation_indices`, `requested_shifts`, `alpha`, `beta`, `z_obs`, `rho`, `informative_mask`, components/connectivity, and incidence conversion |
| `.quadratic_operator` | `SeparatorQuadraticOperatorView` | `observation_rhs`, `regularized_normal_rhs`, regularization/reference, hard-constraint metadata, matrix-free products, dense/optional-sparse matrices, and nullity/gauge metadata |

For `m` observation rows and `n` sites, `graph.incidence_dense()` returns
`B.shape == (n, m)`. Column `r` is `+1` at `site_i[r]` and `-1` at
`site_j[r]`, so `B.T @ weights` gives the oriented fitted differences. The
column order is the resolved observation row order;
`graph.observation_indices` retains the originating input indices. Repeated
rows and rows for different periodic images are never collapsed.

The informative mask is true only for positive-confidence separator rows.
Zero-confidence rows remain in every row array and in `B`, but have `rho == 0`,
contribute nothing to the operators or right-hand sides, and do not connect
informative components. Isolated sites are therefore singleton informative
components.

The explicit operator names distinguish

```text
L_obs = B @ diag(rho) @ B.T
b_obs = B @ (rho * z_obs)
A     = L_obs + regularization_strength * I
b     = b_obs + regularization_strength * regularization_reference
```

Use `observation_laplacian_matvec(...)` and
`regularized_normal_matvec(...)` for matrix-free application,
`observation_laplacian_dense()` and
`regularized_normal_matrix_dense()` for NumPy matrices, and the corresponding
`*_sparse(format=...)` methods for optional SciPy conversion. SciPy is imported
lazily; requesting sparse conversion without it raises an actionable
`ImportError`. Matrix conversion alone does not select a solver.

The primary fixed quadratic fit additionally accepts `solver='sparse'` for an
optional SciPy sparse-direct solve. `solver='auto'` and `solver='analytic'`
retain the dense NumPy quadratic path; v0.7 does not choose a sparse backend
automatically. `SeparatorFitResult.solver` and
`SeparatorFitResult.solver_termination.backend` report the path actually
selected. Sparse execution is limited to unconstrained `SquaredLoss` with
optional L2 regularization and no scalar penalties. It is not exposed through
the experimental active-set outer solver or other inverse branches.

`match_realized_pairs(...)` accepts exactly one of mathematical `weights=` or
backend-compatible `radii=`. The weight-first route is preferred and uses the
same global representation conversion as forward `compute(...)`; the selected
shift is not a scientific inverse result. Existing radius-based calls remain
compatible.

`quadratic_operator` is available only for `SquaredLoss` with no scalar
penalties. Optional L2 regularization is represented exactly. Hard interval or
equality restrictions may coexist, but remain in `problem.bounds`; when they
are present, `normal_equations_characterize_fit` is false because a constrained
optimum need not satisfy the unconstrained equation. Huber mismatch and models
with scalar penalties retain `observation_graph` but reject
`quadratic_operator`.

::: pyvoro2.inverse.separator
:::
