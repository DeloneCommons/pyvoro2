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

Separator external IDs are unique non-negative integers aligned with input-site
order. Python integers and NumPy integer scalars are accepted consistently by
the resolver, fixed fit, and active-set entry points. Raw endpoints are strict
integers in both index and ID modes; lossy float conversion, numeric-string
parsing, and booleans are not accepted. Record and report conversion preserves
the integer IDs.

Direct observation indices, periodic shifts, provenance indices, search and
iteration counts, flags, and masks use the package-wide exact integer/Boolean
contract. Numerical model, confidence, and solver inputs must be finite and
respect their documented positive or non-negative ranges. Resolved
observations and regularization references own C-contiguous read-only arrays;
caller mutation after construction cannot change them. These validation and
ownership rules do not change the documented objective or active-set
mathematics.

The public direct `SeparatorObservations` constructor accepts only dimensions
two or three and requires exact point counts; aligned row shapes; in-range
integer endpoints with `i != j`; dimension-aligned integer shifts; unique
non-negative integer `input_index`; exact Boolean explicit-shift flags; finite
non-negative confidence; finite, nonzero connector geometry; and valid IDs and
warnings. It recomputes squared distance and distance from `delta`, then both
measurement forms from the canonical target and distance. Finite redundant
values are accepted only when
`np.allclose(supplied, derived,
rtol=8*np.finfo(np.float64).eps, atol=0.0)` and are replaced by the recomputed
binary64 values. This is constructor consistency tolerance, not source
equivalence.

## Observation and source identity

Every valid observation set has deterministic source-independent row IDs and
an ordered observation-set fingerprint. Row IDs have the exact form

```text
pyvoro2-separator-row-v1:<namespace_sha256_hex>:<input_index>:<row_sha256_hex>
```

and appear in every observation-aligned record. The namespace contains
dimension, point count, measurement, and IDs. A row includes its endpoints,
shift, measurement, target, confidence, distance values, connector, both
measurement forms, and explicit-shift flag; warnings do not affect identity.
Subsets retain row IDs and input indices, duplicate input rows remain distinct,
and reordering changes the observation-set fingerprint. Source binding never
changes these identities.

Canonical fingerprints normalize finite signed zero, encode floats with
`float.hex()`, convert arrays to row-major nested lists, and serialize with
sorted compact ASCII JSON that rejects non-finite values before SHA-256. Their
public form is `sha256:<64 lowercase hex>`. Runtime association compares exact
canonical values after fingerprint agreement rather than relying on the hash.

Resolver-created observations are bound to exact caller-order points, exact
domain representation, dimension/count, and ID provenance. Valid directly
constructed observations are source-unbound. A first source-aware operation
may bind them only after independently recomputing all row geometry. Binding is
monotonic and private: it survives subsets, shallow/deep copy,
`dataclasses.replace`, `copy.replace` where available, and same-version pickle;
an inconsistent replacement or attempted rebind raises.

Bound domain records use exactly `none`, `planar_box`,
`planar_rectangular_cell`, `spatial_box`, `spatial_orthorhombic_cell`, or
`spatial_periodic_cell`. Their fields are respectively `kind`; `kind,bounds`;
`kind,bounds,periodic`; `kind,bounds`; `kind,bounds,periodic`; and
`kind,vectors,origin`. Translated, periodically shifted, lattice-equivalent,
and differently represented domains are not exact source equality.

When points and an already resolved observation object are passed to a fitting
function, the points establish or verify exact source points. Omitted/default
`domain=None` makes no extra domain assertion and does not erase an existing
bound domain. For an unbound object it binds an exact `{"kind": "none"}`
domain; an explicit non-`None` domain establishes or verifies that exact
domain. The observation object's IDs remain authoritative. Realization and
active-set operations establish or verify the complete source they use.

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

`observation_view(...)` uses the shared exact origin policy before combining
observation-owned arrays with fit-owned predictions. Two unbound objects match
only when they have the exact same observation model; two bound objects match
only when they have the exact same source. A bound/unbound pair and two bound
objects from different sources are rejected. The private authoritative origin
is retained by shallow copies, deep copies, pickle round trips, and
`dataclasses.replace(...)`. Directly constructed results without authoritative
origin observations fail closed when this accessor is called.
`inspect.signature(SeparatorFitResult)` consequently includes the private
optional keyword-only parameter `_originating_observations_init=None`; it is an
init-only reconstruction channel, not a public result field or user input.

`SeparatorIdentificationView.unconstrained_sites` contains sites isolated in
the informative observation graph, which contains only positive-confidence
separator rows. Zero-confidence rows remain excluded even when hard
restrictions or positive-strength penalties affect them: those terms may
constrain or bound component offsets, but they are not observational
identification. Positive L2 regularization is the only currently supported
additional objective reported as guaranteed to select otherwise free component
offsets. The compatibility
diagnostic `ConnectivityDiagnostics.unconstrained_points` retains its
established candidate-graph meaning.

The historical `SeparatorFitProblem.offset_identifying_constraint_mask` name
is preserved for compatibility. That mask includes rows touched by hard
restrictions or positive-strength penalties because the numerical solver must
keep coupled variables in one subproblem. Zero-strength penalties are absent
and do not affect the mask. The mask does not define the informative
observation graph or claim unique offset selection. Exact hard equalities may
fix offsets, but the current identification view does not provide a general
constraint-identifiability classification.

`global_representation_shift` is a backend representation choice made by adding
one common constant, so it selects a representative within the global geometric
gauge. It is distinct from independent offsets between disconnected observation
components and is not inferred from observations.
`component_alignment_policy` is the canonical access to the policy string also
stored under the historical compatibility name `ConnectivityDiagnostics.gauge_policy`.

## Problem-owned graph and quadratic views

`SeparatorFitProblem` owns two additional provisional inspection views. They do
not change its dataclass fields. Its private authoritative observation origin
is retained when an external result is built from the problem.

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
rho_r = confidence_r * alpha_r**2
q_r   = confidence_r * alpha_r * (target_r - beta_r)
b_obs = B @ q
A     = L_obs + regularization_strength * I
b     = b_obs + regularization_strength * regularization_reference
```

`rho_r` and `q_r` are constructed directly with scale-safe products.
`z_obs` is retained for diagnostics but is not used to reconstruct `q_r`,
because the normal system can be finite when the implied difference is not.

Use `observation_laplacian_matvec(...)` and
`regularized_normal_matvec(...)` for matrix-free application,
`observation_laplacian_dense()` and
`regularized_normal_matrix_dense()` for NumPy matrices, and the corresponding
`*_sparse(format=...)` methods for optional SciPy conversion. SciPy is imported
lazily; requesting sparse conversion without it raises an actionable
`ImportError`. Matrix conversion alone does not select a solver.

`fit_weights_from_separators(...)` separates numerical method from linear
backend. `solver='direct'` is the default certified quadratic solve;
`solver='admm'` executes ADMM whenever a component solve is required and is
required for Huber mismatch, hard restrictions, or positive-strength scalar
penalties.
`linear_backend='dense'` uses NumPy without importing SciPy, while
`linear_backend='sparse'` explicitly requires SciPy. There is no site-count
backend switch. `SeparatorFitResult` and `solver_termination` report `solver`
and `linear_backend` separately. If no component solver or internal matrix
backend runs, a successful degenerate fit reports `solver='none'`,
`linear_backend=None`, and `n_iter=0`. The experimental active-set outer
solver forwards the same choices through its `fit_*` keyword parameters.
For a structured ADMM numerical failure after completed iterations, `n_iter`
retains that completed count, including failure of final quadratic
certification. Quadratic `status='optimal'` always refers to the continuous
source objective: exact helper thresholds and coordinatewise rounding of an
exact optimum do not define a separate binary64-lattice success mode.
Positive-strength scalar-penalty coordinates are solved by a private certified
bounded solver. Proved exact point signs and adjacent numeric-binary64 sign
brackets are the only successful scalar exits; expansion, iteration, or bounded
ambiguity exhaustion maps to the existing `numerical_failure` result. One
compiled term kernel supplies scalar, public objective, breakdown, report, and
JSON semantics. Failure detail includes original and component-local row
indices plus private scalar counts, last candidate and bracket, derivative or
localization evidence, and fallback count. Public solver options and
result/report fields are not extended.

`match_realized_pairs(...)` accepts exactly one of mathematical `weights=` or
backend-compatible `radii=`. The weight-first route is preferred and uses the
same global representation conversion as forward `compute(...)`; the selected
shift is not a scientific inverse result. Existing radius-based calls remain
compatible.

`quadratic_operator` is available only for `SquaredLoss` with no
positive-strength scalar penalties. Zero-strength penalties are exact no-ops
and do not hide the operator. Optional L2 regularization is represented
exactly. Hard interval or equality restrictions may coexist, but remain in
`problem.bounds`; when they are present,
`normal_equations_characterize_fit` is false because a constrained optimum
need not satisfy the unconstrained equation. Huber mismatch and models with
positive-strength scalar penalties retain `observation_graph` but reject
`quadratic_operator`.

The exact `PowerFitObjectiveBreakdown` fields are `total`, `mismatch`,
`penalties_total`, `penalty_terms`, `regularization`,
`hard_constraints_satisfied`, `hard_max_violation`, and
`hard_max_tolerance`. Mismatch uses the documented squared/Huber half-factor
convention, L2 is `0.5 * strength * ||weights - reference||**2`, and
`hard_max_tolerance` is the maximum shared float64 absolute-plus-relative
classification tolerance actually used, or zero without hard-bound rows.
See the
[separator-fitting guide](../../guide/powerfit.md#step-2-define-the-fitting-model)
and
[ADR 0007](../../development/decisions/0007-separator-objective-contract.md)
for the complete formulas.

## Report schema and source provenance

The direct row-only chain from `SeparatorObservations` through
`build_power_fit_problem`, `build_power_fit_result`, and `build_fit_report`
remains supported without points or a domain. It reports an honest unbound
source rather than fabricating geometry provenance.

Fit, realized-pair, and active-set reports retain the kinds
`power_weight_fit`, `realized_pair_diagnostics`, and
`self_consistent_power_fit`. Every report also has this versioned envelope:

```json
{
  "schema": {
    "name": "pyvoro2.inverse.separator.report",
    "version": 1
  },
  "producer": {
    "name": "pyvoro2",
    "version": "<pyvoro2.__version__>"
  },
  "source": {
    "binding": "unbound",
    "fingerprint": null,
    "dimension": 2,
    "n_points": 3,
    "points": null,
    "domain": null,
    "ids": null
  },
  "observation_set": {
    "fingerprint": "sha256:<64 lowercase hex>",
    "measurement": "fraction",
    "n_rows": 1,
    "row_ids": ["pyvoro2-separator-row-v1:..."]
  }
}
```

The source block shown is the exact unbound shape. A bound source has the same
keys with `binding="bound"`, its exact source
fingerprint, caller-order point rows, exact domain record, and IDs. Bound
`{"kind": "none"}` is distinct from unbound `domain=null`. Report provenance
comes from the result or diagnostic's authoritative origin, never from an
arbitrary same-length supplied object.

Report builders return JSON-native values. `dumps_report_json(...)` rejects
NaN and infinity. Finite fit, realized, and active reports therefore round-trip
exactly through JSON. An existing active failure state that contains
non-finite placeholders fails closed during serialization; null/unavailable
active-state semantics belong to R7.

::: pyvoro2.inverse.separator
:::
