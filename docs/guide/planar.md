# Planar 2D (`pyvoro2.planar`)

pyvoro2 now ships a dedicated **planar 2D namespace**:

```python
import pyvoro2.planar as pv2
```

This is intentionally separate from the 3D top-level API. The goal is to keep
both surfaces explicit and mathematically honest:

- `pyvoro2` is the 3D package,
- `pyvoro2.planar` is the 2D package.

The current 2D release scope is deliberately limited to the domains that the
vendored legacy backend supports well:

- `pv2.Box`
- `pv2.RectangularCell`

There is **no** planar `PeriodicCell` yet. Rectangular periodic domains can be
periodic in either or both planar axes.

## Basic compute

```python
import numpy as np
import pyvoro2.planar as pv2

pts = np.array([
    [0.2, 0.2],
    [0.8, 0.2],
    [0.5, 0.8],
], dtype=float)

result = pv2.compute(
    pts,
    domain=pv2.Box(((0.0, 1.0), (0.0, 1.0))),
    return_vertices=True,
    return_edges=True,
)
```

The default return is the same `TessellationResult` class used in 3D. Its raw
planar records are available as `result.cells` and remain dimension-specific by
design:

- `area` instead of `volume`,
- `edges` instead of `faces`,
- `adjacent_shift` is a length-2 periodic image shift when requested.

## Power weights

Planar power diagrams accept mathematical weights through the same explicit
contract as the 3D `compute(...)` function:

```python
weights = np.array([-0.1, 0.0, 0.2])
result = pv2.compute(
    pts,
    domain=pv2.Box(((0.0, 1.0), (0.0, 1.0))),
    mode='power',
    weights=weights,
    include_empty=True,
)
```

The convention is \(\pi_i(x)=\lVert x-p_i\rVert^2-w_i\). Weights have
squared-length units and may be negative. Internally, one common global shift
converts them to non-negative length-unit backend radii, so a common additive
change to all weights leaves areas, adjacency, realized edges, and periodic
image shifts unchanged within numerical tolerance. Existing `radii=` calls
remain valid and numerically unchanged in power mode, but supplying both
representations is an error. Standard mode rejects either representation.
`weights=` is currently a `compute(...)` argument only. The input and converted
representation must remain finite; non-finite input or overflow during
conversion raises `ValueError` before native computation.
Finite representability is necessary for conversion but does not guarantee a
numerically resolvable native tessellation. Voro++ evaluates radical geometry
with binary64 squared-radius arithmetic, so very large absolute `radii**2`
values or genuine weight ranges relative to squared coordinate/domain scales
can lose geometric resolution. There is no universal safe cutoff: the onset
depends on scale, geometry, platform, and compiler, and periodic power
tessellations are a particularly sensitive regime. pyvoro2 does not silently
weaken validation or alter the requested power geometry in this unsupported
regime.

## Rectangular periodic cells and edge shifts

For periodic rectangular domains, request `return_edge_shifts=True` when you
need the explicit periodic image of the neighboring site:

```python
cell = pv2.RectangularCell(
    ((0.0, 1.0), (0.0, 1.0)),
    periodic=(True, True),
)

result = pv2.compute(
    pts,
    domain=cell,
    return_vertices=True,
    return_edges=True,
    return_edge_shifts=True,
)
```

The planar wrapper reconstructs these edge shifts in Python and also repairs a
legacy backend quirk where some fully periodic adjacencies can otherwise appear
with negative neighbor ids.

## `locate(...)` and `ghost_cells(...)`

The planar namespace mirrors the 3D operation names:

```python
owners = pv2.locate(pts, [[0.1, 0.2], [0.9, 0.2]], domain=cell)

ghost = pv2.ghost_cells(
    pts,
    [[0.5, 0.5]],
    domain=cell,
    return_vertices=True,
    return_edges=True,
)
```

So the same three high-level questions exist in both dimensions:

1. compute every cell,
2. locate the owner of a query point,
3. compute the hypothetical cell of a query point without inserting it.

## Diagnostics and wrapper-level convenience

Planar `compute(...)` supports the same kind of post-compute convenience that
3D users already expect, but specialized for 2D semantics:

```python
result = pv2.compute(
    pts,
    domain=cell,
    return_diagnostics=True,
)
diag = result.require_tessellation_diagnostics()
```

For periodic domains, the wrapper automatically computes the temporary geometry
needed for reciprocity checks and then strips it back out of `result.cells`
unless you explicitly requested it. The result's boundary and periodic-shift
capability flags describe only that final user-visible geometry.

The same holds for normalization convenience:

```python
result = pv2.compute(
    pts,
    domain=cell,
    return_diagnostics=True,
    normalize='topology',
)
```

This returns the common `pv2.TessellationResult` bundling:

- raw `cells`,
- optional tessellation diagnostics,
- optional normalized vertices,
- optional normalized topology.

`pv2.PlanarComputeResult` remains a compatibility-only identity alias to
`TessellationResult` during v0.7. The deprecated
`return_result: bool | None = None` selector uses `None` to mean “not supplied”;
passing either boolean emits a deprecation warning. Both transition surfaces are
removed in v0.8. New code uses `TessellationResult` and `output=`.

Code that deliberately needs raw cell dictionaries can select the explicit
low-level output mode:

```python
cells = pv2.compute(pts, domain=cell, output='cells')
cells, diag = pv2.compute(
    pts,
    domain=cell,
    output='cells',
    return_diagnostics=True,
)
```

`output='cells'` cannot be combined with normalization, because normalization
is structured output rather than a raw-cell side effect.

## Planar normalization

The dedicated planar normalization helpers are:

- `pv2.normalize_vertices(...)`
- `pv2.normalize_edges(...)`
- `pv2.normalize_topology(...)`
- `pv2.validate_normalized_topology(...)`

In planar topology work, the globally deduplicated boundary objects are
**edges**, not faces.

## Planar plotting

For quick inspection, use the optional matplotlib helper:

```python
from pyvoro2.planar import plot_tessellation

fig, ax = plot_tessellation(result.cells, annotate_ids=True)
```

Install it with:

```bash
pip install "pyvoro2[viz2d]"
```

or install both 2D and 3D visualization helpers with:

```bash
pip install "pyvoro2[viz]"
```

## Planar separator fitting

The canonical separator inverse API supports planar domains as well as 3D
domains. Use `pyvoro2.inverse` for the normal fixed-observation workflow and
`pyvoro2.inverse.separator` for advanced models, realization diagnostics, and
the experimental active-set outer loop.
The solver vocabulary is shared between 2D and 3D; what changes is the meaning
of the realized boundary measure:

- face area in 3D,
- edge length in 2D.

The current planar domain restriction still applies here: rectangular periodic
cells are supported, but there is no planar oblique-periodic `PeriodicCell`
yet.


See [Choosing an API](choosing-api.md) and [Separator fitting](powerfit.md) for the canonical imports and lifecycle status.
