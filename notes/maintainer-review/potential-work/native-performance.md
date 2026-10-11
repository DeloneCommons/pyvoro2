# Native query performance and reusable contexts

**Status:** exploratory, **unapproved** performance/usability candidates. Do not add to v0.9 scope. Reconsider only after benchmarks and demonstrated use; a read-only reusable native context is considered **low priority, likely beyond v1.0**, not a committed roadmap item.

## Existing behavior before proposing optimizations

| Workflow | What is already reused? | Possible cost |
| --- | --- | --- |
| One `locate(points, queries, ...)` call | **One** container and insertion of the permanent sites for the full query batch; Voro++'s block/worklist nearest-site search prunes blocks | Repeating separate public calls repeats Python preparation and container construction |
| One 3D `ghost_cells(points, queries, ...)` call | **One** container loaded with permanent sites, with `compute_ghost_cell` per requested ghost | Ghost-cell clipping per query; no container reuse across separate calls |
| One 2D `ghost_cells(points, queries, ...)` call | No container reuse **between ghost queries**: each query creates a fresh container, reinserts permanent sites, adds a synthetic ghost ID and computes only that ghost cell | Repeated construction/insertion within one batch |
| One `compute(...)` call | One loaded container, cells computed independently | Full geometry extraction costs extra memory/time; existing selective `return_*` options address this ([measurements](../documentation-notes/geometry-export.md)) |

Sources: [3D locate](../../../cpp/bindings.cpp#L515-L780), [3D ghost](../../../cpp/bindings.cpp#L782-L1048), [2D locate and ghost](../../../cpp/bindings2d.cpp#L259-L505), [Voro++ worklist search](../../../vendor/voro++/src/v_compute.cc#L76-L226).

**Do not confuse the native spatial block search with the separate sparse safety-bucket scans** in [`native_preconditions.hpp`](../../../cpp/native_preconditions.hpp). Nor does `locate` test preconstructed convex cell membership: it minimizes ordinary/weighted distance to site candidates and prunes spatial blocks.

## Candidate A: reusable read-only native context

A possible class/context could own a prepared Voro++ container and serve repeated `locate`/`ghost`/possibly `compute` calls for **unchanged generators**. Motivation: interactive/adaptive numerical integration, adaptive query grids, geometric optimization experiments, and other workloads whose next queries are not known in advance.

- Benefits should be compared to **existing batched calls**, which already amortize construction for known queries. An isolated timing of a single `locate` query is insufficient.
- Python/native lifetime, ownership of generators and buffers, thread safety, periodic-image caches, and consistency with validation must be specified before any public API proposal.
- Keep the current stateless API available if any context is later introduced.
- **Not** a proposal for a generally mutable container or incremental tessellation: Voro++'s insertion/clear and periodic-image caching do not provide a straightforward safe arbitrary remove/update interface. Reweighting a fixed site set may require rebuilding, erasing the expected benefit.

This is a **future research candidate only**, not approved for v0.9 and not expected to be prioritized for v1.0 without new evidence.

## Candidate B: 2D ghost batch overhead

Quantify the cost of reconstructing the 2D native container for every ghost query. This does **not** mean recomputing the N persistent Voronoi cells. Investigate whether a safer reusable-setup or temporary insertion strategy is possible under the 2D vendor's actual API and periodic state. Do not assume 3D `compute_ghost_cell` exists in 2D, or that deleting/updating arbitrary generators is supported.

## Candidate C: locate on regular/adaptive query grids

A suggestion worth measuring: axis-ordered scanning, sweep-line style locality, reuse of previous winning site or different spatial indices might help some structured grids. **No superiority has been established**. Voro++ already scans the query's block, uses a precomputed worklist and geometric lower bounds to avoid examining distant blocks. Convexity of the final Voronoi cells does not by itself make a simple x/y/z sort a proven faster point-location algorithm.

Distinguish **few queries** (construction may dominate), **many known queries** (current native batching already helps), **dense regular query grids** (spatial coherence might be exploitable), and **adaptive sequential queries** (a reusable context might help). Test both `standard` and `power`, nonperiodic and periodic layouts, 2D and 3D, and a range of block counts and site distributions.

## Evidence needed before design work

1. Measure Python validation/preparation, native preflight, container creation/`put`, per-query search or clipping, and Python result packaging separately where feasible.
2. Compare total wall time, memory/peak RSS and scaling in site count `N` and query count `M`; keep build/compiler and geometry fixed for fair comparisons.
3. Compare naive exact/small-oracle ownership to Voro++ results (ordinary and power distances, periodic images, ties/boundaries); do not accept benchmark speed at the cost of wrong owners.
4. Benchmark current batched `locate` as a strong baseline, and report the cost of many separate calls explicitly.
5. Only then decide whether an internal optimization, a read-only context, a 2D ghost enhancement, or just better usage examples are warranted.

A separate [geometry-output preset idea](geometry-output-presets.md) addresses API ergonomics, not native point-location search.

**No benchmark result, algorithmic win, implementation or milestone is claimed by this note.**
