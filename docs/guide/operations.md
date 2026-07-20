# Operations

pyvoro2 exposes three high-level operations. They correspond to three common
questions you may ask about a set of sites. The same three verbs also exist in
`pyvoro2.planar` for 2D workflows.

1. **What does the full tessellation look like?**  
   (Compute every Voronoi/power cell.)
2. **Which site owns this location in space?**  
   (Assign arbitrary query points to sites.)
3. **What would the cell of a hypothetical point be?**  
   (Compute a “probe” cell without inserting the point.)

All operations are **stateless**: pyvoro2 creates a Voro++ container in C++, inserts the sites, performs the computation, and returns Python data structures. There is no persistent container object that you need to manage.

The same three operation names also exist in the dedicated 2D namespace
`pyvoro2.planar`. See [Planar 2D](planar.md) for the planar-specific domains,
result schema, and wrapper conveniences.

## Coordinate scale and numerical safety

Voro++ uses a few **fixed absolute tolerances** internally (notably a hard
near-duplicate check around `~1e-5` in the coordinate units of the container).
This is fast and robust for “order-1” coordinate systems, but it means that
very small unit systems can be problematic.

If your coordinates are in SI meters for atomistic systems (typical distances
around `1e-10`), Voro++ may treat distinct sites as “too close” and terminate
the process.

pyvoro2 intentionally does **not** rescale inputs automatically.
If you work in very small or very large units, **rescale explicitly** before
calling `compute`, `locate`, or `ghost_cells` (for example, multiply all
coordinates and domain vectors by a constant).

As an additional safety net, you can ask pyvoro2 to run a fast **Python-side**
near-duplicate pre-check before entering the C++ layer:

```python
result = pyvoro2.compute(
    points,
    domain=cell,
    duplicate_check='raise',  # recommended ("warn" is diagnostic only)
)
```

This checks for point pairs closer than ~`1e-5` (in your coordinate units).
Using `duplicate_check='raise'` prevents Voro++ from terminating the process.

Note: `duplicate_check='warn'` only reports the issue and still enters the C++ layer.
If your points truly violate Voro++'s hard threshold, the process may still terminate.

## 1) `compute(...)`: tessellate all sites

`compute` computes the Voronoi (standard) or power/Laguerre (weighted) cell for
each site. It returns a `TessellationResult` by default in both dimensions.

### Standard Voronoi

```python
result = pyvoro2.compute(points, domain=box, mode='standard')
```

This is the classic “midplane” Voronoi construction. Raw cell dictionaries are
available as `result.cells`; input-aligned measures and empty state are
available as `result.cell_measures` and `result.empty_mask`.

### Power/Laguerre (weighted)

```python
result = pyvoro2.compute(
    points,
    domain=box,
    mode='power',
    weights=weights,
    include_empty=True,
)
```

Here `weights[i]` is the \(w_i\) in
\(\lVert x-p_i\rVert^2-w_i\). Weights have squared-length units and may be
negative. pyvoro2 uses one common global shift to convert them to non-negative
backend radii before entering Voro++; this representation shift does not change
the diagram. Adding a common constant to all weights is therefore geometrically
invariant. The input and converted representation must remain finite;
non-finite input or overflow during conversion raises `ValueError` before the
native call.
Finite representability is necessary for conversion but does not guarantee a
numerically resolvable native tessellation. Voro++ evaluates radical geometry
with binary64 squared-radius arithmetic, so very large absolute `radii**2`
values or genuine weight ranges relative to squared coordinate/domain scales
can lose geometric resolution. There is no universal safe cutoff: the onset
depends on scale, geometry, platform, and compiler, and periodic power
tessellations are a particularly sensitive regime. pyvoro2 does not silently
weaken validation or alter the requested power geometry in this unsupported
regime.

Power-mode `compute(...)` requires exactly one of `weights=` or the existing
`radii=` representation. Radii have length units and should not be interpreted
as unique physical radii; valid radius-based power computations remain
unchanged. Standard-mode `compute(...)` rejects both representations. Direct
weights are available on `compute(...)`, not on `locate(...)` or
`ghost_cells(...)`.

Power diagrams can produce **empty cells** (volume 0). Voro++ omits those in its iteration;
pyvoro2 can reinsert explicit empty-cell records when `include_empty=True`.

### Periodic neighbor image shifts

In periodic domains, a face between $i$ and $j$ corresponds to a specific periodic image of $j$.
If your goal is a periodic neighbor graph, this image information is essential.

Request it with:

```python
result = pyvoro2.compute(
    points,
    domain=cell,
    return_faces=True,
    return_vertices=True,
    return_face_shifts=True,
)
```

`result.require_boundaries()` returns the faces aligned with original input
order. Each face can include:

- `adjacent_cell`: neighbor id
- `adjacent_shift`: integer shift `(na, nb, nc)` describing which neighbor image produced the face

### Raw-cell compatibility and migration

Existing code that needs the historical raw return can keep it with one added
keyword:

```python
cells = pyvoro2.compute(points, domain=box, output='cells')
```

With `output='cells'`, `return_diagnostics=True` retains the historical
`(cells, diagnostics)` tuple. With the preferred structured output,
diagnostics are stored in `result.tessellation_diagnostics` and the return is
always one `TessellationResult`:

```python
result = pyvoro2.compute(
    points,
    domain=box,
    return_diagnostics=True,
)
diagnostics = result.require_tessellation_diagnostics()
```

## 2) `locate(...)`: assign query points to sites

`locate` answers a simpler question than full tessellation:

> Given a query point $q$, which site owns it?

This wraps the Voro++ routine `find_voronoi_cell`.

```python
out = pyvoro2.locate(points, queries, domain=cell, return_owner_position=True)
owner_ids = out['owner_id']
```

If `return_owner_position=True`, the output also contains `owner_pos`.
In periodic domains this position may be a **periodic image** of the generator, chosen
consistently with the query point.

## 3) `ghost_cells(...)`: compute probe (ghost) cells

`ghost_cells` asks a slightly different question:

> What would the cell of $q$ look like if $q$ were inserted as an additional site?

This wraps the Voro++ routine `compute_ghost_cell`.

```python
ghost = pyvoro2.ghost_cells(points, queries, domain=cell)
```

Each returned record describes the polyhedron of the ghost cell.

### `query` vs `site` in periodic domains

For periodic domains, pyvoro2 wraps each query into the primary cell before calling Voro++.
Therefore each record contains:

- `query`: the original coordinate you supplied
- `site`: the coordinate actually used for the computation (a wrapped periodic representative)

They are in the same coordinate system and differ only by an integer lattice translation.
This wrapping makes results easier to compare and visualize.
