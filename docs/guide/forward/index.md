# Tessellations & periodic geometry

Compute cells from fixed sites to obtain volumes, boundaries and periodic
neighbors. This 3D quickstart uses a fully periodic **triclinic** `PeriodicCell`;
its skew lattice vectors are rows of `vectors`. Fractional positions become
Cartesian coordinates by multiplication with that matrix.

```python
import numpy as np
import pyvoro2 as pv

vectors = np.array([[2.0, 0.0, 0.0],
                    [0.5, 1.5, 0.0],
                    [0.25, 0.375, 1.0]])
cell = pv.PeriodicCell(vectors=vectors)
points = np.array([[0.15, 0.20, 0.30],
                   [0.65, 0.25, 0.35],
                   [0.35, 0.75, 0.70],
                   [0.80, 0.70, 0.85]]) @ vectors
result = pv.compute(
    points, domain=cell, ids=[40, 10, 70, 20],
    return_vertices=True, return_faces=True, return_face_shifts=True,
)

# The cells partition a lattice cell of volume det(vectors) = 3.
np.testing.assert_allclose(result.cell_measures.sum(),
                           np.linalg.det(vectors), rtol=1e-10, atol=1e-12)
assert not result.empty_mask.any()
boundaries = result.require_boundaries()  # aligned with result.ids
assert all(boundaries)
assert all(c['vertices'] for c in result.cells)
contacts = [
    (int(site_id), face['adjacent_cell'], tuple(face['adjacent_shift']))
    for site_id, faces in zip(result.ids, boundaries)
    for face in faces
]
assert all(len(shift) == 3 for _, _, shift in contacts)
assert any(any(shift) for _, _, shift in contacts)
print(round(result.cell_measures.sum(), 10))  # 3.0
```

An oriented contact `(i, j, s)` names the image of neighbor `j` translated by
`s @ vectors`, not just a pair of site IDs. Shifts can be nonzero even for a
contact with another image of the same site. Preserve them when building a
[periodic contact graph](../periodic-contacts.md); reciprocal faces carry the
opposite shift. The volume check above is a useful independent check, not a
proof of every face's geometry.

## Geometric output and normalization

The shared [TessellationResult](../../reference/result.md) has input-aligned
`ids`, `cell_measures` (3D volumes, 2D areas), and `empty_mask`. Power cells may
be hidden: their measures are zero and their empty state is recorded even when
raw empty-cell dictionaries are omitted. `require_boundaries()` returns
input-aligned face collections in 3D or edge collections in 2D, and raises if
boundary data were not requested. Raw `result.cells` is not an input-order
contract; requested vertices and face vertex indices are cell-local.

Spatial normalization is an explicit additional operation:

```python
topology = pv.normalize_topology(result.cells, domain=cell)
assert isinstance(topology, pv.NormalizedTopology)
assert len(topology.global_vertices) > 0
assert topology.global_edges and topology.global_faces
```

[Spatial `normalize_topology`](../../reference/normalize.md) constructs global
vertices, edges and faces, with per-cell mappings and lattice shifts. Those
objects belong to `NormalizedTopology`, not a `result.global_faces` field;
3D `compute` does not automatically attach normalized output. Deduplication
uses numerical tolerances and does not promise a universal certified incidence
complex.

[Planar normalization](../planar.md#planar-normalization) has its own
`pyvoro2.planar.normalize_topology` and `NormalizedTopology` for vertices and
edges. Planar `compute(normalize='vertices'|'topology')` can attach these
structures to the result; `normalize='none'` is the default. Do not assume the
3D and 2D convenience options are interchangeable.

## Choose the next operation

- [3D domains](../domains.md): finite `Box`, fully or partially periodic
  `OrthorhombicCell`, and fully periodic triclinic `PeriodicCell`.
- [Operations](../operations.md): standard and power `compute`, `locate` and
  `ghost_cells`. For power `compute`, supply `mode='power', weights=...`;
  power queries and ghost cells retain their radii-based interface.
- [Planar 2D](../planar.md): finite boxes and rectangular periodic cells;
  oblique periodic 2D cells are not supported.
- [Visualization](../visualization.md), [worked notebooks](../../examples/index.md),
  and [capabilities and limitations](../capabilities.md).
- [Inverse fitting](../inverse/index.md): estimate weights from separator data.
