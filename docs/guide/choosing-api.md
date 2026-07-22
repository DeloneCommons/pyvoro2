# Choosing an API

pyvoro2 has one normal forward path and one normal separator-inverse path.
Advanced and compatibility namespaces exist for narrower purposes; they are not
parallel APIs of equal status.

## Forward computation

Choose the namespace by dimension:

```python
import pyvoro2 as pv          # 3D
import pyvoro2.planar as pv2  # 2D
```

Use `compute(...)` and keep its default structured result:

```python
result3d = pv.compute(points3d, domain=domain3d)
result2d = pv2.compute(points2d, domain=domain2d)
```

Both calls return the same stable `TessellationResult` class. Use its aligned
arrays and capability checks for normal downstream work.

Select `output='cells'` only when a low-level workflow intentionally wants the
raw backend-shaped cell dictionaries:

```python
cells = pv.compute(points3d, domain=domain3d, output='cells')
```

This is an explicit supported output mode, not the recommended general result
container.

For power diagrams, prefer mathematical weights:

```python
result = pv.compute(
    points3d,
    domain=domain3d,
    mode='power',
    weights=weights,
)
```

Use `radii=` only when backend-compatible radii are already the data you have.
Weights are the mathematical power parameters; radii are a non-unique shifted
representation used by Voro++.

## Separator inverse fitting

### Normal high-level path

Use `pyvoro2.inverse` for fixed-observation separator fitting:

```python
import pyvoro2.inverse as inverse

observations = inverse.resolve_separator_observations(
    points,
    rows,
    domain=domain,
)
fit = inverse.fit_weights_from_separators(
    points,
    observations,
)
```

This stable surface intentionally contains only:

- `SeparatorObservations`;
- `resolve_separator_observations`;
- `SeparatorFitResult`;
- `fit_weights_from_separators`;
- `weights_to_radii` and `radii_to_weights`.

It is the preferred path for applications and downstream packages.

### Advanced separator research path

Use `pyvoro2.inverse.separator` when you need explicit model pieces, exported
problem/operator views, realization matching, reports, or the experimental
active-set outer loop:

```python
import pyvoro2.inverse.separator as separator

model = separator.FitModel(
    mismatch=separator.HuberLoss(delta=0.02),
)
fit = separator.fit_weights_from_separators(
    points,
    observations,
    model=model,
)
```

Most of this larger surface is provisional. The active-set workflow is
experimental and has no universal convergence guarantee.

### Historical compatibility path

`pyvoro2.powerfit`, broad top-level separator exports, and historical separator
names are deprecated compatibility-only routes in v0.7. They exist only to make
the v0.6.3-to-v0.7 migration explicit and will be removed in v0.8.
Do not use them in new code.

See the [v0.7 migration guide](migration-v0.7.md) for exact replacements.

## Feature and lifecycle status

| Capability | v0.7 status | Recommended use |
|---|---|---|
| 2D/3D domains and `compute`, `locate`, `ghost_cells` | Stable | Normal public API |
| `weights=` and `radii=` power input semantics | Stable | Prefer `weights=` for mathematical workflows |
| `TessellationResult` core fields and `output='cells'` | Stable | Default structured result; raw output when intentionally needed |
| Diagnostics, validation, normalization, duplicate checks, face/edge annotations | Stable | Normal scientific workflow |
| Visualization helpers | Provisional | Optional inspection convenience |
| `pyvoro2.inverse` high-level separator workflow | Stable | Normal inverse API |
| Advanced objective, problem, operator, realization, and report objects in `pyvoro2.inverse.separator` | Provisional | Research and specialized downstream use |
| Explicit SciPy sparse quadratic separator backend | Provisional | Large static sparse quadratic observation graphs |
| Realization-aware active-set refinement | Experimental | Opt-in diagnostic outer algorithm |
| `pyvoro2.powerfit`, broad top-level inverse exports, historical separator aliases, planar `return_result=`, `PlanarComputeResult` | Compatibility-only and deprecated | Migration only; removed in v0.8 |
| v0.8 cleanup and compatibility removal | Planned | No new numerical features |
| Prescribed cell areas/volumes | Planned for v0.9 | Not implemented |
| Mixed separator and cell-measure fitting | Planned for v0.10 | Not implemented |

The complete name-by-name contract is in the
[v0.7 API inventory](../development/api-inventory.md).

## Static scalability contract

v0.7 supports large **static sparse** separator-observation graphs through the
explicit SciPy-backed quadratic path when its model restrictions are satisfied.
The supported scalable regime is a locality graph with observation count that
remains sparse relative to all possible site pairs.

The following are not implied:

- scalable dense or arbitrary all-pairs observation graphs;
- automatic sparse-backend selection;
- sparse execution for Huber, hard-constrained, penalty, or active-set branches;
- prepared solvers or warm starts across changing frames;
- molecular-dynamics trajectory throughput;
- parallel forward tessellation, multiprocessing ownership semantics, GPU, or
  distributed execution.

Trajectory-scale and parallel workflow design is explicitly deferred until
post-1.0 evidence identifies where that responsibility belongs.

## Result-layer map

A separator workflow can produce several related but distinct kinds of data:

```text
resolved observations
        |
        v
fixed-observation fit
        +--> fitted state and backend representation
        +--> observation predictions and objective values
        +--> identification, graph, incidence, and operator diagnostics
        +--> fixed-solver termination
        |
        v  optional forward realization
realized geometry
        +--> requested-image / other-image / unrealized matching
        +--> cells, boundaries, measures, and empty-cell diagnostics
        |
        v  optional experimental outer loop
active-set path and outer termination
```

A small algebraic residual does not prove that the requested pair is a realized
face. The realized geometry is not part of the exact fixed-observation solve,
and active-set refinement is a separate experimental outer algorithm.

The [separator-fitting guide](powerfit.md) explains the layers in detail. The
[glossary](glossary.md) defines the recurring terms.
