# Selective geometry export: purpose and cost

**Status:** documentation working note (observed v0.8 behavior, not a proposed API change).

## Why would a user skip geometry fields?

Voro++ computes a complete cell even when pyvoro2 does not return its vertices, vertex adjacency, or boundary records. The `return_vertices`, `return_adjacency`, and `return_faces` (3D) / `return_edges` (2D) flags avoid **extracting and materializing** unneeded fields as nested Python lists/dicts; they do **not** skip the underlying `compute_cell` operation.

See [3D `build_cell_dict()`](../../../cpp/bindings.cpp#L31-L138), [2D `build_cell_dict()`](../../../cpp/bindings2d.cpp#L29-L94), and [public 3D `compute()`](../../../src/pyvoro2/api.py#L235-L264).

The three outputs represent different things:

- **Vertices**: cell-local vertex coordinates.
- **Adjacency**: edges connecting vertices *within the same cell*, **not** the graph of neighboring generators.
- **Faces / edges**: cell-local boundary vertex indices together with neighboring generator IDs (or boundary markers).

The `volume` (3D) / `area` (2D), `id`, and `site` fields remain available when all optional geometry fields are disabled. `output='result'` versus `output='cells'` chooses the Python return *representation*, not the set of geometric fields.

## Scientific/use-case examples

### Cell measures only

For density/volume statistics, periodic sum checks, size-distribution histograms, or objectives depending solely on cell measures, there is no need to export the cell's vertex and face topology:

```python
result = pv.compute(
    points,
    domain=box,
    return_vertices=False,
    return_adjacency=False,
    return_faces=False,
)
volumes = result.cell_measures
```

A similar example exists in [the basic-compute notebook](../../../docs/notebooks/01_basic_compute.md).

### Realized contacts in separator inverse fitting

To identify face contacts and analyze realized separators, one needs boundary records and their vertex positions, but typically not the additional cell-local *vertex-edge graph*:

```python
cells = pv.compute(
    points,
    domain=domain,
    mode='power',
    weights=weights,
    return_vertices=True,
    return_faces=True,
    return_adjacency=False,
    output='cells',
)
```

This pattern is used in [`inverse/separator/realize.py`](../../../src/pyvoro2/inverse/separator/realize.py#L551-L568). Periodic workflows can additionally request image shifts, which impose their own prerequisites.

### Full geometry

Keep all three flags enabled (current defaults) for inspection of individual cell polyhedra, vertex-edge graph algorithms, or consumers needing all cell-local geometry. Requesting faces without vertices gives boundary **indices** without the corresponding coordinate list; this may be useful for some low-level consumers but is not a self-contained boundary geometry.

## Exploratory performance snapshot

Measured on **2026-10-10** using the source-review kit pinned to source commit `8866348a3d23ad0b6d9b8db9772f72e64c743055` (native/build files match v0.8.0), plus the official **0.8.0 CPython 3.13** Linux x86-64 binary extensions. Runtime: Python 3.13.5, NumPy 2.3.5, Linux x86-64, glibc 2.41.

| Returned data | 3D 3,000 sites: median ms | 3D output MiB | 2D 10,000 sites: median ms | 2D output MiB |
| --- | ---: | ---: | ---: | ---: |
| All (vertices, adjacency, boundaries) | 129 | 39.4 | 111 | 46.1 |
| Vertices + boundaries; no adjacency | 120 | 31.9 | 90 | 38.7 |
| Vertices only | 74 | 13.8 | 46 | 14.2 |
| Boundaries only | 97 | 19.6 | 71 | 29.3 |
| Essential fields only (id, measure, site) | 59 | 1.5 | 27 | 4.9 |

**How measured:** direct calls to `_core.compute_box_standard` and `_core2d.compute_box_standard` (not public Python validation, diagnostics, or result packaging); NumPy RNG seeds 20261010 (3D) and 20261011 (2D), independent uniform points in `[0,1)^d`, nonperiodic unit box, blocks `(10,10,10)` in 3D and `(100,100)` in 2D, `init_mem=8`. Each configuration was warmed up and timed over **seven** calls, taking the median of `time.perf_counter()` wall times with `gc.collect()` before each timing. All variants produced the full set of cells and total measure ~1.0.

Output MiB is a recursive `sys.getsizeof` estimate of the returned list/dict/tuple structure and contained Python objects (deduplicating shared object identities). It is **not** peak RSS, native allocation, or an estimate of persistent NumPy inputs. Other environments, site distributions, domains, and builds can yield very different timings.

**Interpretation:** in this experiment, 3D full export was ~2.2× the time and ~26× the Python output footprint of essential-fields-only export. Even though Voro++ computed each full cell in both cases, the additional Python data construction was measurable. Results are **illustrative exploratory measurements**, not a qualified library benchmark or a universal speed guarantee. Re-run with a checked-in reproducible benchmark and representative real-world data before citing performance as a published claim.

## User-facing documentation idea

Introduce the feature with the **measure-only** example before discussing the native implementation. Explain that “skip returned geometry” differs from “skip computing cell geometry,” and give a simple guide for common workflows (measures only, realized boundaries, full local geometry). Do not suggest that cell-local `adjacency` is a tessellation-wide adjacency graph.

Possible ergonomics follow-up: [geometry output presets](../potential-work/geometry-output-presets.md). This is **not** part of the current v0.8 API.
