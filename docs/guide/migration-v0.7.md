# Migrating from v0.6.3 to v0.7

v0.7 changes the preferred result and inverse namespaces while preserving one
explicit transition release. This guide covers ordinary source changes; the
archived manuscript environment remains pinned to v0.6.3.

## Forward `compute(...)` now returns `TessellationResult`

### 3D

v0.6.3 code commonly expected a raw list:

```python
cells = pyvoro2.compute(points, domain=box)
```

The v0.7 default is a structured result:

```python
result = pyvoro2.compute(points, domain=box)
cells = result.cells
volumes = result.cell_measures
empty = result.empty_mask
```

To preserve the old raw return exactly, add one selector:

```python
cells = pyvoro2.compute(points, domain=box, output='cells')
```

When `return_diagnostics=True`, the structured path always returns one result
and stores diagnostics inside it:

```python
result = pyvoro2.compute(
    points,
    domain=box,
    return_diagnostics=True,
)
diagnostics = result.require_tessellation_diagnostics()
```

The explicit raw route retains the historical tuple:

```python
cells, diagnostics = pyvoro2.compute(
    points,
    domain=box,
    output='cells',
    return_diagnostics=True,
)
```

### Planar 2D

Use the same `output=` selector in `pyvoro2.planar`:

```python
result = pyvoro2.planar.compute(points2d, domain=box2d)
cells = pyvoro2.planar.compute(
    points2d,
    domain=box2d,
    output='cells',
)
```

`return_result=` is deprecated and removed in v0.8. Passing either boolean in
v0.7 emits `DeprecationWarning`. `PlanarComputeResult` is an identity alias to
`TessellationResult` during v0.7 and is also removed in v0.8.

Normalization requires structured output. Replace combinations involving
`return_result=True` with the default result or `output='result'`.

## Use mathematical `weights=` directly

v0.6.3 required callers to choose non-negative backend radii before forward
power computation. v0.7 accepts the mathematical weights:

```python
result = pyvoro2.compute(
    points,
    domain=box,
    mode='power',
    weights=weights,
)
```

The result records:

- `input_weights`: the supplied mathematical weights;
- `backend_radii`: the actual non-negative radii passed to Voro++;
- `representation_shift`: the one common additive weight shift used for that
  representation.

Existing valid `radii=` calls remain supported. In power mode, supply exactly
one of `weights=` or `radii=`. Standard mode rejects both.

Do not interpret the representation shift or shifted radii as independently
fitted scientific quantities. Adding one common constant to every weight leaves
the complete power diagram unchanged.

## Common result fields and optional data

The stable core fields are:

```text
dimension
domain
mode
sites
ids
cells
cell_measures
empty_mask
input_weights
backend_radii
representation_shift
tessellation_diagnostics
normalized_vertices
normalized_topology
```

Use capability properties and `require_*` methods rather than assuming optional
geometry was computed:

```python
if result.has_boundaries:
    boundaries = result.require_boundaries()

if result.has_tessellation_diagnostics:
    diagnostics = result.require_tessellation_diagnostics()
```

`sites`, `ids`, measures, masks, weights, and radii are owned read-only arrays.
The outer dataclass is frozen. The raw `cells` list and its nested dictionaries
remain shared and mutable; they are not deep-copied or deep-frozen. The aligned
measure and empty-mask arrays are construction-time snapshots and do not update
if raw records are later mutated.

Application code should normally receive results from `compute(...)` rather
than construct `TessellationResult` directly. Direct-construction and some
convenience accessors remain provisional.

## Canonical inverse imports

Replace historical package imports:

```python
from pyvoro2.powerfit import (
    PairBisectorConstraints,
    PowerWeightFitResult,
    fit_power_weights,
    resolve_pair_bisector_constraints,
)
```

with the high-level canonical API:

```python
from pyvoro2.inverse import (
    SeparatorObservations,
    SeparatorFitResult,
    fit_weights_from_separators,
    resolve_separator_observations,
)
```

The core mapping is:

| v0.6.3 name | v0.7 preferred name |
|---|---|
| `PairBisectorConstraints` | `SeparatorObservations` |
| `resolve_pair_bisector_constraints` | `resolve_separator_observations` |
| `PowerFitProblem` | `SeparatorFitProblem` |
| `PowerWeightFitResult` | `SeparatorFitResult` |
| `fit_power_weights` | `fit_weights_from_separators` |

Use `pyvoro2.inverse.separator` for advanced models, problem/operator views,
realization matching, report builders, and the experimental active-set workflow:

```python
import pyvoro2.inverse as inverse
import pyvoro2.inverse.separator as separator

model = separator.FitModel(mismatch=separator.SquaredLoss())
fit = inverse.fit_weights_from_separators(
    points,
    observations,
    model=model,
)
```

## Top-level separator exports

v0.6.3 allowed broad imports such as:

```python
from pyvoro2 import FitModel, fit_power_weights
```

These names remain compatibility-only in v0.7 but are not part of the preferred
top-level package. Move them to `pyvoro2.inverse` or
`pyvoro2.inverse.separator` now. They are removed from top-level `pyvoro2` in
v0.8.

## `pyvoro2.powerfit` transition

Importing `pyvoro2.powerfit` in v0.7 loads a thin one-way shim and emits a
hidden-by-default `DeprecationWarning`. It contains no independent solver
implementation.

The entire package and its direct submodules are removed in v0.8. The removal
horizon is fixed; the project does not plan to retain the old namespace based on
hypothetical downstream use.

Historical core aliases also remain visible from the advanced canonical
separator package during v0.7 so that objects retain one implementation and
compatible identity. New code should not use those aliases; they are removed in
v0.8 together with the shim.

## Layered separator results

The flat v0.6.3 fields remain available during v0.7, while the canonical result
also groups them by meaning:

```python
state = fit.state
identification = fit.identification
observations = fit.observation_view(resolved_observations)
objective = fit.objective
algebraic = fit.algebraic
termination = fit.solver_termination
```

Realization is still a separate operation and result. Active-set path data are
experimental outer-loop diagnostics. See [Choosing an API](choosing-api.md) and
the [separator-fitting guide](powerfit.md).

## Removal summary for v0.8

| v0.7 transition surface | v0.8 action |
|---|---|
| `pyvoro2.powerfit` and historical submodules | Remove |
| broad top-level separator exports | Remove |
| historical separator core aliases in `pyvoro2.inverse.separator` | Remove |
| lazy top-level `pyvoro2.powerfit` attribute | Remove |
| `PlanarComputeResult` | Remove |
| planar `return_result=` | Remove |
| `output='cells'` | Retain as an explicit useful raw-output mode |

v0.8 is a cleanup-only release. Prescribed cell measures begin in v0.9, and
mixed separator-plus-measure fitting begins in v0.10.
