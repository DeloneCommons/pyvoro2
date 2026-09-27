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

Voro++ uses **fixed absolute tolerances** internally. pyvoro2 reserves the
distance through `1e-5` (squared distance at most `1e-10`) as a mandatory
backend-safety regime and rejects any generator pair there before insertion.
This prevents public duplicate options from admitting a known fatal/invalid
native pair, but very small unit systems can still be unsuitable for meaningful
geometry.

pyvoro2 intentionally does **not** rescale inputs automatically.
If you work in very small or very large units, **rescale explicitly** before
calling `compute`, `locate`, or `ghost_cells` (for example, multiply all
coordinates and domain vectors by a constant).

You can also request an optional **Python-side** policy above the mandatory
floor:

```python
result = pyvoro2.compute(
    points,
    domain=cell,
    duplicate_check='raise',
    duplicate_threshold=1e-3,
)
```

`duplicate_check='off'` disables only this additional policy. `warn` warns for
a safe pair strictly below the user threshold and then proceeds; `raise`
raises `DuplicateError`. A threshold at or below `1e-5` adds no optional range.
`duplicate_wrap=False` selects unwrapped Cartesian distance only for the
optional policy. Mandatory periodic safety always uses certified minimum-image
geometry, including seams, corners, partial periodicity, and triclinic cells.

All inserted generators also obey native containment. On every non-periodic
axis the valid interval is half-open: `lo <= x < hi`. Periodic axes are remapped
to their primary interval. This applies to the persistent sites used by all
three operations and to temporary `ghost_cells` generators. Locate queries are
not inserted and retain their existing outside-query behavior.

## Native construction controls and rejection

Every `compute`, `locate`, and `ghost_cells` call validates its inputs before
constructing the stateless native container. Site coordinates must have shape
`(n, 3)`, query coordinates must have shape `(m, 3)`, and all coordinates must
be finite. Power radii must have shape `(n,)`, while ghost radii are scalar or
query-aligned as documented; every radius must be finite and non-negative.
Box bounds must be finite and strictly ordered, and periodic parameters must
produce finite, safe native constructor arithmetic. Invalid values raise
`ValueError` before native construction.

The three native grid controls have strict meanings:

- `init_mem` is the positive initial per-block particle capacity. It must be
  a positive exact non-Boolean index-protocol scalar within the C++ `int`
  range. Python integers and in-range NumPy signed or unsigned integer scalars
  are common examples; other genuine index-protocol scalars are accepted too.
  Python/NumPy Booleans, floats, strings, complex values, and scalar arrays are
  rejected.
- `blocks` is an explicit length-3 sequence of positive exact non-Boolean
  index-protocol scalars, one per axis, with the same examples, rejections, and
  C++ `int` range. When supplied, these are the selected counts instead of
  counts derived from `block_size`.
- `block_size` is an optional positive finite real scalar used to derive block
  counts. A supplied value is validated even when explicit `blocks` select the
  counts.

Before allocation, pyvoro2 also checks native integer products and a
source-derived estimate of allocations known to occur eagerly during container
construction. The aggregate estimate may be at most exactly 1 GiB
(1,073,741,824 bytes); a larger estimate raises `ValueError`. This is a safety
limit, not a promise about total peak memory, and there is no unsafe override.
Use fewer blocks, a larger derived `block_size`, or a smaller `init_mem` when a
valid configuration exceeds it. These guards change rejection behavior only;
valid in-cap tessellations follow the existing numerical path.

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

Power-mode `compute(...)` and `locate(...)` require exactly one of `weights=`
or the existing `radii=` representation. Power-mode `ghost_cells(...)`
requires one complete `weights=`/`ghost_weights=` or
`radii=`/`ghost_radii=` family and converts persistent and ghost weights with
one common gauge. Radii have length units and should not be interpreted as
unique physical radii; valid radius-based power computations remain unchanged.
Standard mode rejects both representations.

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

### Structured and raw cell output

The structured result is the normal path. Code that deliberately needs the raw
cell dictionaries can select the explicit low-level output mode:

```python
cells = pyvoro2.compute(points, domain=box, output='cells')
```

`output='cells'` is a supported current output mode and is retained in v0.8; it
is not part of the compatibility-removal list. With this mode,
`return_diagnostics=True` returns `(cells, diagnostics)`. With the preferred
structured output,
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

Periodic locate always includes `query`, `query_wrapped` and `query_shift`,
with `(m,d)` shapes. Wrapping solves the affine problem exactly over the
validated binary64 operands; `query_wrapped` is its nearest-even float view.
Rectangular spans use the domain's binary64 span. A rounded upper endpoint or
backend seam snap does not change the exact user shift.

With `return_owner_position=True`, periodic calls also include original
`owner_site`, native `owner_pos`, and exact user-basis `owner_shift`. The exact
image is `owner_site + owner_shift @ A` in the original query chart. The native
`owner_pos` preserves storage and frame rounding and may differ from the
rounded exact image. For example, unit-period `P_x=nextafter(1,0)` and `q_x=0`
return native position 0 with shift -1; the exact image is `-2**-53`.
Do not add `query_shift` to `owner_shift`.

Not-found rows have owner ID -1, NaN owner coordinates and zero owner shifts.
The owner selector omits all three fields when false; query views remain.
Nonperiodic calls retain their existing keys. Zero-query batches retain empty
array shapes and construct no native container. Returned arrays own their data.

Actual native insertion omission and unsafe integer execution raise explicit
ValueError-compatible failures. Requested image certification may also refuse
ambiguous, inconsistent, resource-limited or unrepresentable results. The
Provisional protocol exposes `code`, `stage`, `query_index` and bounded
`details`; see the [reference](../reference/api.md). Native selection remains
the owner answer, without an additional exact ownership theorem.

## 3) `ghost_cells(...)`: compute probe (ghost) cells

`ghost_cells` asks a slightly different question:

> What would the cell of $q$ look like if $q$ were inserted as an additional site?

For each query, pyvoro2 computes only the selected cell in a fresh native
population containing the persistent generators and one initialized temporary
generator. The query does not remain in a persistent container.

```python
ghost = pyvoro2.ghost_cells(points, queries, domain=cell)
```

Each returned record describes the polyhedron of the ghost cell.

A ghost query is temporarily inserted. It must therefore lie in every
non-periodic half-open interval and be safely distinct from the persistent
generators. An outside non-periodic ghost now raises `ValueError`; a valid
inserted ghost may still produce an empty cell geometrically.

`ghost_cells` returns a list of raw dictionaries, rather than a
`TessellationResult`. Records retain `id=-1` and the input `query_index`.
Every requested 3D face (or 2D edge) has a `boundary_reference` describing
the source-certified native boundary when it is positive in the complete
exact public-semantic ghost cell:

```python
{
    'kind': 'generator',  # or 'ghost_self' or 'wall'
    'generator_id': 7,    # external persistent ID, otherwise None
    'shift': (0, -1, 0),  # user-basis image; None without periodicity
    'wall_id': None,      # existing physical side ID for a wall
}
```

A `generator` reference uses the original persistent site and has a user-basis
integer shift in every periodic domain, including `(0, 0, 0)`. A
`ghost_self` reference names another image of this query, with a nonzero
shift and no generator ID. A real nonperiodic `wall` has its source-qualified
side ID and no shift. The entire value is `None` only when the retained raw
native occurrence is proved internally collapsed; public rounding alone does
not remove a positive reference. `adjacent_cell` is a compatibility view for
qualified persistent generators and walls; ghost self has no
`adjacent_cell`. Requested public vertices are not needed for certification.

Certificate-bearing calls are initially qualified on Linux x86_64 GCC 13.3
under the reviewed strict binary64 native profile. Unsupported source/build
profiles and incomplete or inconsistent certificates raise a
`ValueError`-compatible error with `code`, `stage`, `query_index` and bounded
`details`; no partial batch is returned. Geometry-only calls use the safe
initialized native route without claiming this boundary certificate. WP7 has been independently accepted; WP8 metadata implementation remains
subject to its separate exact-head review.

### `query` vs `site` in periodic domains

For periodic domains, preparation remaps each query before native insertion.
Both dimensions' records contain:

- `query`: the original coordinate you supplied, in every domain;
- `query_wrapped` and `query_shift`: exact user-wrap float/int64 views in periodic domains;
- `site`: the actual stored ghost's materialized Cartesian representative
  anchoring returned native geometry.

The original query and stored site need not be exactly separated by an integer
lattice translation as binary64 values: preparation, insertion and frame
conversion can round separately. Generator boundary images reconstruct as
`original_generator_site + shift @ A`; ghost self images reconstruct as
`site + shift @ A`. Boundary shifts stay in the stored-ghost chart; never add
`query_shift` to them. No `site_shift` is returned.

All required ghost certification runs before `include_empty=False` removes
empty records. Only retained records need the new public query fields to fit
int64. Thus a filtered, already-qualified empty ghost with private shift
`2**70` may yield `[]`; retaining that row raises
`GHOST_SHIFT_UNREPRESENTABLE` at materialization with `field='query_shift'`.
Filtering preserves original `query_index` values and cannot hide certification
failure. Geometry-only calls do not acquire a new semantic cell certificate.
