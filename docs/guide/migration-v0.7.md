# Migrating from v0.6.3 through v0.8

v0.7 changed the preferred result and inverse namespaces while preserving one
explicit transition release. v0.8 completes that transition by removing the
announced compatibility-only surfaces. This guide covers ordinary source
changes; the archived manuscript environment remains pinned to v0.6.3.

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

The v0.7 `return_result=` selector and `PlanarComputeResult` alias are absent in
v0.8. Use the default `TessellationResult`, `output='result'`, or the explicit
`output='cells'` raw route. Normalization requires structured output.

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
convenience accessors remain provisional. Direct construction validates the
documented aligned metadata but does not normalize hand-written backend-style
records, recompute geometry, or verify that the records form a valid
tessellation.

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

| v0.6.3 name | Current canonical name |
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

These names were compatibility-only in v0.7 and are absent from top-level
`pyvoro2` in v0.8. Move them to `pyvoro2.inverse` or
`pyvoro2.inverse.separator`.

## `pyvoro2.powerfit` transition

`pyvoro2.powerfit` was a temporary v0.7 compatibility facade with no independent
solver implementation. The package, its direct submodules, the lazy top-level
attribute, broad top-level separator exports, and the five mapped historical
core aliases are absent in v0.8. Canonical separator APIs live under
`pyvoro2.inverse` and `pyvoro2.inverse.separator`.

## Layered separator results

The canonical v0.8 result retains the established flat result fields and also
groups the same data by meaning:

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

## Completed removal summary for v0.8

| v0.7 transition surface | v0.8 status |
|---|---|
| `pyvoro2.powerfit` and historical submodules | Removed |
| broad top-level separator exports | Removed |
| historical separator core aliases in `pyvoro2.inverse.separator` | Removed |
| lazy top-level `pyvoro2.powerfit` attribute | Removed |
| `PlanarComputeResult` | Removed |
| planar `return_result=` | Removed |
| `output='cells'` | Retained as an explicit useful raw-output mode |

v0.8 is a cleanup-only release. v0.9 performs functional/API stabilization and
downstream readiness, 1.0 stabilizes the existing core, prescribed cell
measures begin in v1.1, and mixed separator-plus-measure fitting begins in v1.2.
Canonical numerical algorithms, defaults, result fields, record keys, and gauge
behavior are unchanged by these removals.
