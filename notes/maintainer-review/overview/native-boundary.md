# pyvoro2 native boundary: a source-reading map

**Status:** factual architecture notes for the v0.8-based maintainer reassessment, not an approved redesign. This note describes the current source, not the historical abandoned v0.9 implementation.

## The layers and one-call lifecycle

`src/pyvoro2/api.py` (3D) and `src/pyvoro2/planar/api.py` (2D) expose `compute`, `locate`, and `ghost_cells`. These are Python entry points, distinct from the compiled `pyvoro2._core` and `pyvoro2._core2d` extensions registered through pybind11 in `cpp/bindings.cpp` and `cpp/bindings2d.cpp`. pybind11 registers callable C++ functions; it is not a separate Python tessellation engine.

The typical flow is:

1. Validate the public contract and coerce/own input data in Python ([`inputs.py`](../../../src/pyvoro2/_internal/inputs.py), [`validation.py`](../../../src/pyvoro2/_internal/validation.py), [`power_input.py`](../../../src/pyvoro2/_internal/power_input.py)).
2. Prepare persistent generators: periodic remapping, insertion containment, mandatory duplicate safety, optional duplicate policy, internal IDs ([`generator_preparation.py`](../../../src/pyvoro2/_internal/generator_preparation.py)); build an operation-local periodic snapshot when appropriate ([`domain_geometry.py`](../../../src/pyvoro2/_internal/spatial/domain_geometry.py)).
3. Dispatch to a compiled binding. Each native route invokes [`native_preconditions.hpp`](../../../cpp/native_preconditions.hpp) before constructing a Voro++ container.
4. Construct the container and insert permanent generators with `put`; perform the requested operation; return native results to Python.
5. Map internal IDs back to public IDs, convert internal coordinates when needed, and package/annotate/diagnose results in Python.

**Lifetime:** each independent public call creates a new container. No persistent container is retained across calls. Within one call, there can still be useful batching and container reuse.

## Native route matrix

| Operation | 3D backend | 2D backend |
| --- | --- | --- |
| `compute` | `box` / fully periodic triclinic × `standard` / `power` = four routes | rectangular `box` × `standard` / `power` = two routes |
| `locate` | Same four combinations | Same two combinations |
| `ghost_cells` | Same four combinations | Same two combinations |

This is **12 spatial plus 6 planar construction routes**. In 2D, `RectangularCell` periodicity is supplied as flags to a rectangular container; there is no separate native triclinic 2D backend in this baseline. In 3D, `OrthorhombicCell` uses the `box` route with periodic-axis flags; `PeriodicCell` uses the dedicated fully periodic Voro++ container.

## What the operations actually do

| Operation | Native calculation | Native result | Does it build all cell geometries? |
| --- | --- | --- | --- |
| `compute` | Iterate generators with `c_loop_all`; call `compute_cell` for each nonempty cell | List of cell dictionaries via `build_cell_dict` | Computes the requested tessellation's nonempty cells independently |
| `locate` | For each query, call `find_voronoi_cell`, a block/worklist-based owner search | Tuple of NumPy arrays: `found`, `owner_id`, `owner_pos` | **No**; does not construct/query polygon/polyhedron membership |
| `ghost_cells` | Calculate each temporarily introduced query generator's cell | List of ghost cell dictionaries, including empty status | **No**; computes only queried ghost cells, not all persistent cells |

Sources: [3D computation](../../../cpp/bindings.cpp#L167-L182), [3D locate bindings](../../../cpp/bindings.cpp#L515-L780), [3D ghost bindings](../../../cpp/bindings.cpp#L782-L1048), [2D locate/ghost bindings](../../../cpp/bindings2d.cpp#L259-L505), [Voro++ 3D search](../../../vendor/voro++/src/v_compute.cc#L76-L226).

**An important asymmetry:** in 3D, a batch of ghost queries uses one loaded container and calls Voro++ `compute_ghost_cell` for each query. In 2D, the binding builds a **fresh container and reinserts all permanent generators per ghost query**, then inserts a synthetic-ID generator and computes **only its cell**. This is not rebuilding all N existing cell polyhedra, but container construction and insertion are repeated within the batch. Both implementations are stateless across separate public calls.

For `locate`, both dimensions construct/insert once for the entire query batch; they do not construct any Voronoi cells. The Voro++ search uses spatial blocks, predefined worklists and distance/radius bounds to prune candidate blocks, rather than always scanning all generators.

`locate(..., mode='power')` in **v0.8** accepts `radii`, not mathematical `weights`. Native weighted lookup minimizes squared distance minus squared radius; it can choose a generator other than the closest Euclidean one. The Python result contains `found` and `owner_id`, plus `owner_pos` only when requested. Native computes/returns the position array even if the public Python flag omits the field. For periodic searches, `owner_pos` may be a periodic image of the owner; `locate` does not directly return an integer lattice shift.

## Local-cell output is not a global mesh

[`build_cell_dict()`](../../../cpp/bindings.cpp#L31-L138) exports `id`, `volume` and `site`, with optional cell-local `vertices`, `adjacency`, and `faces` (2D: `area` and `edges`). Here `adjacency` connects vertices **inside one cell**, not generator-to-generator adjacency. Face neighbor IDs alone are not periodic image identities; Python adds other derived metadata when requested. Individual Voro++ cell vertex indices are **not** global tessellation vertex IDs.

The `return_*` options skip **export and Python object construction**, not Voro++ cell computation. See [use cases and exploratory benchmarks](../documentation-notes/geometry-export.md). Distinguish local cells, derived face contacts, and a separately derived global complex: [geometry layers](../potential-work/geometry-layers.md).

## Boundaries of this map

The current native interface is comparatively narrow. Python owns substantial preparation, periodic conversion, ID provenance, image-shift reconstruction, diagnostics, and result presentation. Native preflight reduces known process-safety risks, **not** all possible geometry errors; see [boundary contracts](../contracts/native-boundary.md) and [Voro++ contracts](../contracts/voro.md).

No API refactor, persistent context, 2D ghost optimization, or performance claim is authorized by this overview. Those belong in [potential native performance work](../potential-work/native-performance.md).
