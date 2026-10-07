# Advanced separator fitting

`pyvoro2.inverse.separator` owns the separator implementation and exposes the
canonical core names alongside advanced model, problem, realization, report,
and diagnostic objects.

The fixed-observation fit is distinct from realization-aware active-set
refinement. The active-set API is experimental and separator-specific; it is a
practical outer algorithm without a universal convergence guarantee.

Most advanced model, problem, operator, report, realization, and layered-view
objects on this page are **Provisional**. The active engine and its controls are
**Experimental**. The identical shared result's final-state inspection is
**Provisional**, also when imported here; its path/counter members remain
**Experimental**. Ordinary callers use the
[preferred facade](index.md#realization-aware-workflow), which is not an advanced
export. The optional explicit sparse quadratic backend is
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

## Independent spaces and bounded row values

The following keyword-only options preserve existing positional constructors:

```text
SquaredLoss(*, space=None)
HuberLoss(delta=1.0, *, space=None)
Interval(lower, upper, *, applicable=True, space=None)
FixedValue(value, *, applicable=True, space=None)
SoftIntervalPenalty(lower, upper, strength, *, space=None)
ExponentialBoundaryPenalty(lower=0., upper=1., margin=.02,
                           strength=1., tau=.01, *, space=None)
ReciprocalBoundaryPenalty(lower=0., upper=1., margin=.05,
                          strength=1., epsilon=1e-6, *, space=None)
```

Each space is `fraction`, `position`, or `None` to inherit observation units.
Hard endpoints/values/applicability and penalty endpoints/strengths accept
scalar or exact-length one-dimensional row values. Applicability must contain
actual Booleans. Numeric rows must be finite, strengths non-negative, and all
configured values satisfy family rules even when absent. A one-element vector
is a row vector, not a broadcasting scalar. Closed hard intervals allow equality;
soft and boundary intervals retain positive-width requirements. The C shape
fields `delta`, `margin`, `tau`, and `epsilon` remain public scalars.

`build_power_fit_problem` owns a positional model binding. Component and active
selections project that binding with the same ordered row selection. Site-indexed
L2 references retain full site order. Candidate policy survives active removal
and reentry. `.resolved_policy` is a recursively read-only mapping with exactly
`row_ids`, `model_spaces`, and `model_policy`; row IDs are an immutable tuple.
It is available on the problem, fixed result, and shared active result.
Copies, replacements, and same-version pickle preserve the association. Legacy
manually constructed results without policy provenance fail explicitly when a
policy-dependent view or report is requested.

Problems and results expose `mismatch_space`, `hard_constraint_space`, and
ordered `penalty_spaces`. Fixed results add `mismatch_target`,
`mismatch_predicted`, and `mismatch_residuals`. Source prediction and algebraic
diagnostics retain observation units; problem graph/normal coefficients use
mismatch units. Source edge `alpha`, `beta`, `z_obs`, and descriptive
`edge_weight = confidence * alpha**2` are not mixed-model curvature or converted
likelihood weights. The quadratic RHS evaluates the complete scaled
`confidence * alpha * (target - beta)` in mismatch space; it does not reconstruct
that value as `edge_weight * z_obs`. `PowerFitBounds.space` identifies declared hard units, and
its read-only `applicable` mask determines effective rows. Configured measurement
endpoints remain finite and row-aligned. Inapplicable difference endpoints are
NaN because they were not computed; absence is never an infinite public bound.
Without configured hard policy all four bound arrays are `None`, space is
`None`, and applicability is an all-false mask.

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
| `SelfConsistentPowerFitResult` | `.inner_fit`, `.final_realization`, `.candidate_diagnostics` | final fit and optional weights-dependent final objects |
| `SelfConsistentPowerFitResult` | `.outer_termination`, direct `.active_mask`, availability and final convergence | Provisional final-state inspection |
| `SelfConsistentPowerFitResult` | `.path`, `.history`, `.path_summary`, `.marginal_constraints` | Experimental path access |

The shared active result keeps its existing stored field names and private
reconstruction/ownership behavior; no new public constructor guarantee is made.
Candidate observations and source/policy/mismatch views, final fit, realization,
residuals, image/empty flags, connectivity and optional final geometry are
Provisional inspection. Toggle/first/last-realized counters, iteration objects,
marginal classifications and path-derived status labels remain Experimental,
including inside candidate diagnostics, records and reports. The preferred
facade returns `history=None` without filtering other research data or report v3.
`realized`, `diagnostics`, `rms_residual_all`, and `max_residual_all` are
optional and are simultaneously unavailable when the final fit has no usable
weights. In that state `final_realization`, `candidate_diagnostics`, and
`to_records()` also return `None`. The computed properties
`final_state_available`, `final_state_unavailable_reason`, and
`final_refit_converged` distinguish coherent final-layer availability, the
existing final fit status, and final inner-fit convergence. Outer `converged`
continues to mean `termination == 'self_consistent'`.

Present final layers may contain producer-established unavailable diagnostic
cells without losing finite weights/radii, genuine realization or their source
association. Raw arrays retain signed infinity for range overflow and NaN for a
proven unavailable dependency; export follows the bounded grammar below. Whole
`None` containers still mean an absent final state. Source residuals are
evaluated as complete affine expressions at final weights, not by subtracting
separately rounded predictions and targets. Full candidate summaries include
inactive and confidence-zero rows. Each stored history summary uses that
iteration's own `weights_eval`, after alignment and relaxation; private
iteration/source/value provenance survives copy, replacement and same-version
pickle without retaining historical weight vectors.

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

An `optimal` or converged native result certifies a finite soft objective and
each applicable hard row at the exact final returned representative.
`hard_max_violation <= hard_max_tolerance` is not a substitute for the per-row
predicate. A failed final success claim becomes `numerical_failure` with absent
weights-dependent layers, actual solver/backend and completed iteration count;
it does not change precheck feasibility or fabricate a conflict. The public
`build_power_fit_result` raises `ValueError` for false success even with
`canonicalize_gauge=False`. An explicitly nonconverged external candidate and a
genuine finite `max_iter` candidate remain inspectable.
Bound export and active reconstruction recheck this predicate independently;
copying or replacing status/objective metadata cannot create a successful fit.

At zero L2, standalone fits with multiple model-coupling components use supplied
reference means, or zero means by default; singleton entries equal their
reference exactly. Positive-confidence mismatch, applicable hard rows and
positive penalty rows establish coupling. One connected multi-site component
retains its solver anchor. Empty native fits use the full reference or zeros,
including one site; public one-site canonicalization remains a no-op. Positive
L2 retains its objective reference without gauge shifts. Optional active
alignment to the preceding relaxed state is allowed only when contrasts are
preserved exactly and L2 is zero. Final certification uses the selected policy,
without imposing excluded candidate hard rows.

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
    "version": 3
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
  },
  "unavailable_diagnostics": {}
}
```

The source block shown is the exact unbound shape. A bound source has the same
keys with `binding="bound"`, its exact source
fingerprint, caller-order point rows, exact domain record, and IDs. Bound
`{"kind": "none"}` is distinct from unbound `domain=null`. Report provenance
comes from the result or diagnostic's authoritative origin, never from an
arbitrary same-length supplied object.

Report builders return JSON-native values. `dumps_report_json(...)` rejects
NaN and infinity. Fit, realized, and active reports round-trip exactly through
JSON, including active no-weights failures. Active reports add an
`availability` block with `weights`, `realization`, `records`, and `reason`.
Unavailable weights-dependent sections and final realization/residual summary
values are JSON null, while the nested final fit and outer path/termination
metadata remain present.

Fit and active reports additionally have exactly these model blocks:

```text
model_spaces = {mismatch, hard_constraint, penalties}
model_policy = {mismatch, hard_constraint, penalties, regularization}
```

Each term has `family` (the public class name), resolved `space`, and
`parameters`. Every parameter is `{"kind":"uniform","value":scalar}` or
`{"kind":"rows","values":[...]}`; hard terms also have `applicable` in
that grammar. Unconfigured hard policy is null. Penalty order and configured
zero strengths are retained. L2 has `family="L2Regularization"`, scalar
`strength`, and `reference={"kind":"implicit_zero"}` or
`{"kind":"sites","values":[...]}` in full site order.

Fit summaries add `mismatch_space`; fit and active candidate records add
`mismatch_space`, `mismatch_target`, `mismatch_predicted`, and
`mismatch_residual`. Active outer blocks describe all candidate rows; the nested
fit describes the exact selected projection. Missing weights retain known policy
and targets with null predictions/residuals. Realization-only reports do not add
model blocks. Observation identity and observation-only records are unchanged.

## Numerical diagnostic availability

Schema v3 requires `unavailable_diagnostics` on every report root, including
nested fit/realized reports, and on every fixed, standalone candidate and
enriched active record. Ordinary output contains `{}`. Observation-only and
realized-only rows have no new map; history rows have no separate map.

Only these derived diagnostic leaves may become numerical nulls:

| Container | Eligible fields |
|---|---|
| Fixed record; report `/fit_records/*` | `predicted`, `predicted_fraction`, `predicted_position`, `residual`, `mismatch_predicted`, `mismatch_residual`, `alpha`, `beta`, `z_obs`, `z_fit`, `algebraic_residual`, `edge_weight` |
| Fixed `/edge_diagnostics` | Elements of `alpha`, `beta`, `z_obs`, `z_fit`, `residual`, `edge_weight`; scalars `weighted_l2`, `weighted_rmse`, `rmse`, `mae` |
| Fixed `/summary` | `rms_residual`, `max_residual` |
| Standalone candidate record | `predicted`, `predicted_fraction`, `predicted_position`, `residual` |
| Enriched active record; `/diagnostics/*`, `/marginal_records/*` | Candidate fields plus `mismatch_predicted`, `mismatch_residual` |
| Active `/summary` and `/history/*` | `rms_residual_all`, `max_residual_all` |
| Active `/fit` | Complete nested fixed-report whitelist, rebased with `/fit` |
| Realized report | No newly nullable scalar; empty root map |

Each map entry points to a null leaf with exactly one reason:
`out_of_binary64_range` means the finite mathematical diagnostic exceeds finite
binary64 magnitude; `unavailable_dependency` means an accepted operand is
unavailable. A typed producer establishes the reason. Caller-supplied infinity,
NaN or metadata is not proof. Unavailable `alpha` does not produce a spurious
zero `z_obs`; exact zero confidence removes weighted work before dangerous
evaluation. Finite values retain ordinary binary64 rounding and underflow.

Pointers are local to their containing record/report, start with `/`, use
standard `~0`/`~1` escaping and zero-based decimal indices without leading zeros.
A record might contain `{"/predicted": "out_of_binary64_range"}`; its report
also includes `/fit_records/0/predicted`. Every report map exhausts its subtree,
including nested reports, history and marginal-record copies. Nested roots and
records retain local maps. Enrichment merges maps and rebasing copies them.
Structural nulls, such as absent final weights, have no reason entry. Finite
leaves have no entry.

Source weighted norms use complete affine source residuals and apply
`sqrt(confidence)` before materializing a possibly overflowing residual.
Weighted RMSE divides squared weighted residuals by the number of rows, not
the sum of confidence. Aggregate availability is evaluated independently: a
row outside range can still have a representable weighted norm or RMS.
Unavailable dependencies propagate without dropping rows. Source algebraic
RMSE/MAE retain difference-space residuals.

Inputs, target representations, confidence, model policy, successful weights,
radii/shift, objective components, hard metrics, geometry and identity remain
strict. History `weight_step_norm` is also excluded because it affects stopping.
Final exports independently recompute from bound observations/policy and exact
final weights; equality between two copied wrong residual arrays is rejected.
Standalone candidate diagnostics and history use private immutable row/value
or iteration/source/value bindings preserved by supported reconstruction.
Ordinary finite manual history remains supported; nonfinite manual values
without producer provenance are rejected.

The general JSON normalizer still rejects arbitrary NaN/infinity and writers
use `allow_nan=False`; typed conversion precedes that check. Schema name, report
kinds and source/row identity versions are unchanged. All other v2 key, type
and absence rules remain; no v2 compatibility writer is provided. See
[ADR 0026](../../development/decisions/0026-separator-final-state-and-diagnostic-availability.md)
for the decision and migration boundary.

::: pyvoro2.inverse.separator
:::
