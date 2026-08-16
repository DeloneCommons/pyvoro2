# Topology and neighbor graphs

A very common reason to compute a tessellation in a periodic system is not the
polyhedra themselves, but the **neighbor graph**:

- Which sites are adjacent?
- Which periodic image produces that adjacency?
- Can we build a reproducible graph representation for downstream analysis?

This page explains why periodic graphs are subtle, and how pyvoro2 supports this workflow.

## Why periodic adjacency needs an image shift

In a periodic cell, each site has infinitely many periodic images.
A face between site $i$ and site $j$ therefore corresponds to **one specific image** of $j$.

If you record only “$i$ neighbors $j$”, you lose information. The edge is ambiguous.

pyvoro2 can annotate each face with:

- `adjacent_cell`: the neighbor site id
- `adjacent_shift`: an integer lattice shift `(na, nb, nc)` describing which image produced the face

You enable this with `return_face_shifts=True`:

```python
result = pyvoro2.compute(
    points,
    domain=cell,
    return_faces=True,
    return_face_shifts=True,
)
cells = result.cells
```

For `PeriodicCell`, the shift is expressed in the $(a,b,c)$ lattice basis.
For `OrthorhombicCell`, the shift is expressed in axis-aligned lattice units.

## Building a periodic graph in practice

A minimal workflow looks like this:

1) Compute a tessellation with face shifts
2) Extract edges `(i, j, shift)` from faces

```python
edges = []
for c in cells:
    i = c['id']
    for f in c.get('faces', []):
        j = f['adjacent_cell']
        if j < 0:
            # boundary face (in a non-periodic domain)
            continue
        shift = f.get('adjacent_shift', (0, 0, 0))
        edges.append((i, j, shift))
```

In many scientific applications you will then:

- merge duplicate edges,
- choose an orientation convention (e.g., keep only `i < j`), and
- use `shift` to translate neighbor positions consistently.

## Normalization utilities

When you want to build a **reproducible** periodic graph, it is often helpful to normalize
geometric entities (vertices, edges, faces) so that they have a global indexing.
This makes it easier to compare results across different runs or different point orders.

pyvoro2 provides:

- `normalize_vertices(...)`
- `normalize_edges_faces(...)`
- `normalize_topology(...)`

These utilities are most useful for periodic settings.

## Diagnostics: catching subtle issues early

When building graphs, you typically want a few simple consistency checks.
pyvoro2 provides `analyze_tessellation(...)`, and
`compute(..., return_diagnostics=True)`. The latter stores diagnostics in the
returned `TessellationResult`.

Diagnostics can check, for example:

- whether every non-empty cell has a finite non-negative area or volume,
- whether cell measures sum to the domain measure (within tolerance),
- whether expected IDs have the meaning declared by `mode`, and
- whether periodic face or edge reciprocity holds.

The overall `diagnostics.ok` value has one meaning in both dimensions: it is
false when an error issue exists, and warning/info-only findings are nonfatal.
Strict validation and `compute(..., tessellation_check='warn'|'raise')` use
that final value directly.

Expected IDs are mode-sensitive. A missing standard ID is an error. A missing
power ID is an informational hidden/empty site and appears in both
`missing_ids` and `empty_ids`. With `mode=None`, a missing expected ID is a
warning because the analyzer cannot infer whether hidden cells are valid.

On periodic data, public `analyze_tessellation(..., check_reciprocity=True)`
treats the requested reciprocity check as required. The validation and compute
wrappers can inspect it optionally through their existing
`require_reciprocity` options. When face/edge marking is enabled, a new pass
clears the analyzer-owned `orphan`, `reciprocal_missing`, and
`reciprocal_mismatch` flags before recording current failures.

This is not “proving correctness”, but it is extremely effective at catching mistakes
in downstream graph code.
