# v0.7 public API inventory

- **Status:** Draft; finalized before the v0.7.0 release candidate
- **Baseline:** v0.6.3
- **Target:** v0.7.0
- **Policy:** [API lifecycle and compatibility](api-lifecycle.md)
- **Plan:** [v0.7 development plan](plans/v0.7.md)

This inventory records the intended lifecycle status of public imports, return
routes, record schemas, defaults, and scientific semantics for v0.7. It is a
release artifact, not a retrospective document to be written after the release.

The initial categories below guide implementation. They must be checked against
actual code, tests, documentation, and downstream integration before v0.7.0 is
published.

## How to maintain this inventory

For every v0.7 issue that changes public behavior:

1. update the relevant row or section in the same change;
2. distinguish the v0.6.3 baseline from the intended v0.7 state;
3. record aliases, deprecations, and planned removal releases;
4. include defaults, result fields, record keys, units, and periodic conventions
   when they carry scientific meaning;
5. leave uncertain new surfaces **provisional** rather than omitting them;
6. do not mark a surface **stable** until its tests and documentation define the
   contract clearly.

The release review must verify that `__all__`, docstrings, guides, reference
pages, migration notes, and this inventory agree.

## Accepted v0.7 contract decisions

The following boundaries are already accepted:

- `pyvoro2.inverse` is the canonical inverse namespace;
- separator implementation is owned by `pyvoro2.inverse.separator`;
- `pyvoro2.powerfit` and broad top-level separator exports are
  compatibility-only for v0.7, with planned removal in v0.8;
- both forward `compute(...)` functions return
  `pyvoro2.TessellationResult` by default;
- `output='cells'` is the explicit raw compatibility route;
- `PlanarComputeResult` is a compatibility alias during v0.7;
- deep immutability of nested raw records is not part of the v0.7 contract.

See [ADR 0004](decisions/0004-canonical-inverse-namespace.md) and
[ADR 0005](decisions/0005-tessellation-result-contract.md).

## Lifecycle summary for the preferred v0.7 API

| Surface | Intended status | Notes |
|---|---|---|
| Domain classes and domain geometry semantics | Stable | Mature bounded and periodic behavior; capability differences remain explicit by dimension. |
| `pyvoro2.compute` and `pyvoro2.planar.compute` | Stable | Function role is stable; the new structured default and exact keyword contract must be finalized and tested. |
| `weights=` and `radii=` mathematical meaning | Stable | Mutual exclusivity, one global representation shift, finite input, and empty-cell behavior are part of the contract. |
| `pyvoro2.TessellationResult` core contract | Stable candidate | Core identity, ID alignment, measures, empty mask, and representation metadata should be stable at release. |
| Detailed optional result conveniences and raw geometry views | Provisional | Refine through implementation and chemvoro-shaped validation. |
| `pyvoro2.inverse` preferred high-level separator workflow | Stable candidate | Main observations/fit entry point should be suitable for chemvoro at release. |
| `pyvoro2.inverse.separator` advanced problem and operator views | Provisional | Public for research use, but likely to evolve before prescribed measures and mixed problems. |
| Realization-aware active-set API | Experimental | Practical outer algorithm; no universal convergence claim. |
| Optional sparse quadratic backend | Experimental if shipped | Backend selection and performance policy remain conditional. |
| `pyvoro2.powerfit` | Compatibility-only and deprecated | One-way shim during v0.7; planned removal in v0.8. |
| Broad separator-specific exports from top-level `pyvoro2` | Compatibility-only and deprecated | New code imports from `pyvoro2.inverse`; planned removal in v0.8. |
| Native extension and solver-internal modules | Internal | No compatibility guarantee. |

“Stable candidate” means that the release intends to make the surface stable,
but final approval occurs only after implementation, tests, documentation, and
downstream validation are complete.

## Spatial forward namespace: `pyvoro2`

### Stable baseline and v0.7 candidates

| Group | Names / behavior | v0.7 intent |
|---|---|---|
| Domains | `Box`, `OrthorhombicCell`, `PeriodicCell` | Stable |
| Operations | `compute`, `locate`, `ghost_cells` | Stable |
| Structured result | `TessellationResult` | Stable candidate |
| Weight transforms | `weights_to_radii`, `radii_to_weights` | Stable; implementation moves out of separator-specific code |
| Tessellation diagnostics | `TessellationDiagnostics`, `TessellationIssue`, `TessellationError`, `analyze_tessellation`, `validate_tessellation` | Stable unless the baseline audit identifies undocumented schema details |
| Duplicate handling | `DuplicatePair`, `DuplicateError`, `duplicate_check` | Stable |
| Geometry annotations | `annotate_face_properties` | Stable candidate |
| Normalization | `NormalizedVertices`, `NormalizedTopology`, normalization and validation helpers | Stable or provisional per detailed baseline audit |
| Package metadata | `__version__`, `planar` | Stable |

### Compatibility-only top-level inverse names

The following v0.6.3 exports remain available during v0.7 but are not preferred
for new code:

```text
PairBisectorConstraints
resolve_pair_bisector_constraints
SquaredLoss
HuberLoss
Interval
FixedValue
SoftIntervalPenalty
ExponentialBoundaryPenalty
ReciprocalBoundaryPenalty
L2Regularization
FitModel
AlgebraicEdgeDiagnostics
ConstraintGraphDiagnostics
ConnectivityDiagnostics
ConnectivityDiagnosticsError
HardConstraintConflictTerm
HardConstraintConflict
PowerWeightFitResult
RealizedPairDiagnostics
UnaccountedRealizedPair
UnaccountedRealizedPairError
build_fit_report
build_realized_report
build_active_set_report
dumps_report_json
write_report_json
ActiveSetOptions
ActiveSetIteration
ActiveSetPathSummary
PairConstraintDiagnostics
SelfConsistentPowerFitResult
fit_power_weights
match_realized_pairs
solve_self_consistent_power_weights
```

Planned removal: v0.8, unless a later accepted release decision extends the
transition.

## Planar namespace: `pyvoro2.planar`

| Group | Names / behavior | v0.7 intent |
|---|---|---|
| Domains | `Box`, `RectangularCell` | Stable |
| Operations | `compute`, `locate`, `ghost_cells` | Stable |
| Structured result | `TessellationResult` re-export | Stable candidate |
| Historical result name | `PlanarComputeResult` | Compatibility-only alias; planned removal or reconsideration in v0.8 |
| Diagnostics and validation | Planar tessellation and normalization diagnostics | Stable unless the baseline audit identifies schema details needing provisional status |
| Duplicate handling and annotations | `duplicate_check`, `annotate_edge_properties` | Stable candidates |
| Normalization | Planar normalization helpers and result objects | Stable or provisional per detailed baseline audit |
| Visualization | `plot_tessellation` | Provisional optional convenience |

## Canonical inverse namespace: `pyvoro2.inverse`

### Preferred high-level separator API

The exact implemented signatures are finalized during the namespace issue. The
accepted preferred names are:

| Name | Intended status | Meaning |
|---|---|---|
| `SeparatorObservations` | Stable candidate | Resolved pairwise separator observations, including periodic image labels and confidence. |
| `resolve_separator_observations` | Stable candidate | Validate and resolve raw separator observations against sites and domain. |
| `SeparatorFitResult` | Stable candidate | Fitted state, observation residuals, identification metadata, and solver status. |
| `fit_weights_from_separators` | Stable candidate | Preferred fixed-observation fit entry point. |
| `weights_to_radii`, `radii_to_weights` | Stable re-export where useful | Same neutral transforms as top-level pyvoro2. |

### Advanced separator API

Expected public but initially provisional surfaces include:

- `SeparatorFitProblem` and problem-building/evaluation helpers;
- objective model pieces such as squared/Huber losses, hard intervals,
  penalties, and regularization;
- graph, connectivity, incidence, Laplacian, and objective-breakdown views;
- result packaging for externally computed weights;
- realization matching and record/report builders.

The active-set outer workflow and its path/result types remain **experimental**.

The implementation issue must produce an exact old-to-new name map. Historical
names under `pyvoro2.powerfit` may be aliases to canonical classes or wrappers,
but numerical implementations must not be duplicated.

## Forward return contract

### Preferred route

```python
result = pyvoro2.compute(..., output='result')
result = pyvoro2.planar.compute(..., output='result')
```

`output='result'` is the default.

The stable-candidate core of `TessellationResult` is:

| Concept | Required semantics |
|---|---|
| `dimension` | `2` or `3`; never inferred from record shape by the caller |
| `domain` | The validated domain associated with the computation |
| `mode` | Standard or power computation |
| site coordinates | Aligned with original input order |
| external IDs | Aligned with original input order and preserved through cell lookup |
| `cells` | Raw backend-shaped cell records; contained mutability is documented |
| cell measures | Area in 2D or volume in 3D, aligned with input-site order |
| empty-cell state | Explicit mask aligned with input-site order, independent of whether raw empty records were requested |
| input weights | Mathematical weights when the caller supplied them |
| backend radii | Actual non-negative backend representation when power mode used it |
| representation shift | One common additive weight shift used to construct backend radii |
| diagnostics | Present only when requested or required by a check |
| normalized output | Present only where requested and supported |
| boundary access | Available only when required face/edge data were requested or computed |

Exact field names, helper names, and which optional conveniences are stable
versus provisional must be filled in by the result-contract implementation
issue. The result should be structurally immutable when straightforward, but
nested raw records are not promised to be deeply immutable.

### Raw compatibility route

```python
cells = pyvoro2.compute(..., output='cells')
cells = pyvoro2.planar.compute(..., output='cells')
```

The raw route preserves established list/tuple behavior during v0.7. Existing
record keys and meanings are inventoried by characterization tests before the
default return changes.

## Scientifically meaningful semantics to inventory explicitly

The following are API even when no dedicated Python class represents them:

- coordinate units are caller-defined but consistent within one computation;
- power weights have squared-coordinate units;
- backend radii have coordinate units;
- one global additive weight shift leaves the complete power diagram unchanged;
- disconnected separator-observation components have additional unidentified
  offsets that may change global realization;
- external IDs remain attached to original sites;
- periodic neighbor shifts identify the realized image and are not silently
  replaced by a nearest image;
- zero-confidence separator rows do not identify weight differences;
- algebraic fit does not imply realized-boundary support;
- empty/hidden cells are represented deterministically according to the chosen
  output route;
- error/status behavior for infeasibility and wrong-image realization is part of
  the public scientific contract.

## Deprecation and removal schedule

| Surface | v0.7 | Planned v0.8 action |
|---|---|---|
| `pyvoro2.powerfit` | Compatibility-only; optional `DeprecationWarning`; migration docs | Remove unless an explicit release decision extends it |
| Broad top-level separator exports | Compatibility-only; migration docs | Remove from top-level |
| `PairBisectorConstraints` and other historical names | Compatibility aliases | Remove or retain only if justified by real usage and documented explicitly |
| `PlanarComputeResult` | Compatibility alias to `TessellationResult` | Remove or reconsider after v0.7 downstream feedback |
| Raw default return | Available through `output='cells'` | Continue as explicit route unless a later decision removes it |
| Planar `return_result=` | Compatibility-only | Remove after migration to `output=` |

## Final release review checklist

- [ ] Every preferred public import is listed with a lifecycle category.
- [ ] Every compatibility alias has a replacement and removal horizon.
- [ ] `__all__` matches the intended namespace policy.
- [ ] Forward output modes and diagnostic combinations are characterized.
- [ ] Stable `TessellationResult` fields and mutable contained values are
      documented.
- [ ] Raw record keys used by compatibility tests are listed or referenced.
- [ ] Preferred separator names and exact historical aliases are complete.
- [ ] Active-set and optional sparse behavior are labelled experimental.
- [ ] Default changes and scientific semantics appear in migration notes and
      release notes.
- [ ] The chemvoro-shaped integration workflow uses only stable or deliberately
      provisional public surfaces.
- [ ] The maintainer approves the final inventory before the release candidate.
