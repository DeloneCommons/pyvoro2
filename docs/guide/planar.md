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

Planar `compute`, `locate`, and `ghost_cells` use the same strict
[native-construction contract](operations.md#native-construction-controls-and-rejection)
as their spatial counterparts. Planar points and queries use two columns, and
explicit `blocks` has length 2. `init_mem` and block counts are positive exact
integers, `block_size` is a positive finite real scalar, and the same exact
1-GiB known-eager-allocation cap applies before native construction.

Planar bounds are copied into nested built-in-float tuples and rectangular
periodicity into built-in-Boolean tuples. Public flags require Python or NumPy
Boolean scalars; IDs and counts require exact non-Boolean integers; and
explicit tolerances must be finite in their documented positive or
non-negative range. `Box.from_points` rejects empty/non-finite inputs before
reduction. Public rectangular remapping validates finite points and `eps` and
checks signed-int64 shift range before conversion. Ordinary `compute` keeps
its private preparation and transport shifts as Python integers; only a
requested final public edge shift must fit signed int64. Large common
translations can therefore cancel without an earlier preparation-range
failure. Public remapping and ghost/locate integer policies are unchanged.

Every inserted planar generator must lie in `[lo, hi)` on non-periodic axes;
periodic axes are remapped first. Generator pairs at squared distance at most
`1e-10` always raise before native insertion. The public duplicate mode,
threshold, and wrap options control only optional diagnostics above that floor.
`ghost_cells` queries are temporary generators and follow the same rule, while
`locate` queries are not inserted.

Ordinary `compute` also verifies that every persistent input reached native
storage. An insertion omission raises in both standard and power mode; it is
not a hidden power cell. A successfully inserted power cell may still become
hidden during cell computation.

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
converts them to non-negative length-unit backend radii. One common additive
weight shift preserves the exact diagram when the represented weight
differences are preserved; radius conversion and native arithmetic can still
lose those differences. Existing `radii=` calls remain supported in power
mode, but supplying both
representations is an error. Standard mode rejects either representation.
`locate` also accepts `weights=`; power `ghost_cells` accepts either
`weights=`/`ghost_weights=` or `radii=`/`ghost_radii=` as a complete family.
The input and converted
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
    return_vertices=False,
    return_adjacency=False,
    return_edges=True,
    return_edge_shifts=True,
)
```

Each ordinary edge's owner and image come from its native source occurrence.
They are not selected by a residual, finite search window or nearest image.
For lattice rows `A`, an edge from source `i` with neighbor `j` and shift `s`
identifies `points[j] + s @ A` relative to the original source `points[i]`.
This remains true when input representatives lie outside the primary periodic
cell. `cell['site']` is the original input site; returned vertices, when
requested, are numerical native geometry centered on that source chart.

A self-image edge names the same persistent owner and has a nonzero shift.
Real walls keep their negative side code and omit `adjacent_shift`; a zero
tuple is not a wall image. Ownership remains certified when
`return_edge_shifts=False`. Public vertices and vertex adjacency may both be
omitted while requesting edge shifts. `has_periodic_shifts` reports requested
shift availability, including available-but-empty output; it does not assert
that every native edge is a positive exact semantic boundary.

Ordinary planar computation currently admits the qualified Linux x86_64
GCC 13.3 baseline-SSE2 native profile, with binary64 evaluation, contraction and
fast-math disabled, and no LTO or AVX/FMA target. Unsupported source/build or
runtime arithmetic profiles fail explicitly. Package support on another
platform does not by itself qualify this planar certification path.

The ordinary `compute` keywords `edge_shift_search`, `validate_edge_shifts`,
`repair_edge_shifts` and `edge_shift_tol` are removed. See the
[v0.9 planar migration guide](migration-v0.9.md).

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

Ghost edge reconstruction remains a separate legacy path. Its reconstruction
controls and requirement for public vertices when requesting ghost shifts are
unchanged; ordinary compute's source certification does not certify ghost
boundary identity.

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

Requested compute diagnostics also reconstruct two complete exact ideals:
E from actual stored native sites, periods and exact backend-radius squares;
S from original sites, public periods and mathematical weights (or exact
supplied-radius squares). They audit contact status, positive boundary coverage
and required reciprocity without changing native ownership or shifts.

After a complete exact audit, `tessellation_line_offset_tol` and
`tessellation_line_angle_tol` control a separate numerical comparison of
reciprocal native segment unions in classes positive in both ideals. This
check uses internal local coordinates and needs no public vertices. These
tolerances do not choose images or define exact positivity.

Raw collapsed occurrences are retained. An internally collapsed extra record
with consistently nonpositive E/S contact is nonfatal when positive coverage
is complete. Noncollapsed nonpositive records, E/S status conflicts, missing
positive boundaries and positive boundaries represented only by collapsed
records are errors. Collapse inside the native computation differs from
distinct internal endpoints rounding to the same public coordinate.

Use `tessellation_check='raise'` to require an okay completed diagnostic, or
`'warn'` to warn when it is not okay. `'diagnose'` attaches findings without
raising; `'none'` takes no diagnostic action, though `return_diagnostics=True`
still requests the audit. Unavailable provenance, insertion failure,
unsupported profile or an unrepresentable required public view always fails
atomically. Exact-audit resource exhaustion is reported as an incomplete audit
and can preserve already attributed shifts under non-raising actions.
Numerical reciprocal inspection has its own work and representation refusals;
these attach error findings without changing the completed exact audit or
discarding attributed shifts under an action that permits return.

Standalone `analyze_tessellation` and validation of public dictionaries check
the supplied numerical records. Those dictionaries alone do not contain the
stored native population, original weight operands or internal occurrence
witness needed for compute's exact E/S audit.

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

Normalization keeps wall/generator kind and owner/image provenance separate
before geometric pooling and preserves each local occurrence association,
including repeated occurrences. `normalize_edges` and `normalize_topology`
refuse a mapping that collapses distinct public edge endpoints, whether caused
by tolerance pooling or periodic seam remapping. Equal public endpoints can be
retained but do not by themselves prove native internal collapse.
`normalize_vertices` alone remains a numerical vertex pool with raw vertices
and local mappings. Normalization tolerance never determines exact semantic
positivity.

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

Planar realization requires a complete successful exact consistency audit and
uses the complete positive S boundary-class set. It derives length once per
exact segment/class rather than counting every raw edge occurrence. A requested
numerical length that cannot be represented as a finite nonzero binary64 value
fails as a representation problem; it does not turn a positive exact segment
into an absent boundary. Numerical edge annotations remain descriptors of
native output.

The current planar domain restriction still applies here: rectangular periodic
cells are supported, but there is no planar oblique-periodic `PeriodicCell`
yet.


See [Choosing an API](choosing-api.md) and [Separator fitting](powerfit.md) for the canonical imports and lifecycle status.
