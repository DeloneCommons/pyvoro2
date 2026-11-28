# Periodic contacts and image shifts

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
import pyvoro2 as pv

# Two sites in a unit cube: they meet through two different x images.
cell = pv.PeriodicCell(vectors=((1.0, 0.0, 0.0),
                                (0.0, 1.0, 0.0),
                                (0.0, 0.0, 1.0)))
points = [[0.25, 0.5, 0.5], [0.75, 0.5, 0.5]]
result = pv.compute(
    points, domain=cell,
    return_vertices=True, return_faces=True, return_face_shifts=True,
)
cells = result.cells
```

For `PeriodicCell`, the shift is expressed in the $(a,b,c)$ lattice basis.
For `OrthorhombicCell`, it is expressed in axis-aligned lattice units. In both
cases the raw tuple `(i, j, s)` refers to the native cell anchors used for that
computation. Its reciprocal in the same reference is `(j, i, -s)`.

The coefficient order follows the supplied basis, but the raw shift is not
generally a shift to apply directly to the **original caller coordinates**.
Preparation can translate and numerically change those representatives. Keep
raw contact labels with the result and coordinate reference that produced them.
Separator realization diagnostics perform a separate checked, same-operation
transport into an effective source-indexed convention; they do not rewrite raw
cell labels or certify source/native geometry equivalence. See
[separator fitting](inverse/separator-fitting.md) and
[ADR 0033](../development/decisions/0033-source-connectors-and-realization-reference.md).

Current face/edge image attribution uses bounded numerical reconstruction.
Successful residual and reciprocity checks do not establish complete source
contacts, and certified nearest-image selection for separators does not change
that limitation. A nearest image is not necessarily the only image that has a
face. Use the current output within its documented reconstruction limits;
stronger observed-contact contracts remain separate development work.

## Building a periodic graph in practice

Extract the requested image data directly. A missing `adjacent_shift` is an
error in an image-aware workflow; substituting zero would invent a contact
with the unshifted image. Genuine non-periodic wall labels are excluded from
the site-contact graph. An unexplained negative label in a periodic direction
must not be silently interpreted as a wall or a zero-shift neighbor. The
following recipe uses the fully periodic example above and refuses negative
labels, since that example has no domain walls.

```python
def contact_key(i, j, shift):
    """Canonical key for one reciprocal pair of image-qualified contacts."""
    shift = tuple(shift)
    return min((i, j, shift), (j, i, tuple(-s for s in shift)))


contacts = set()
for c in cells:
    i = c['id']
    for face in c['faces']:
        j = face['adjacent_cell']
        if j < 0:
            raise ValueError('Unresolved neighbor in fully periodic output')
        contacts.add(contact_key(i, j, face['adjacent_shift']))
```

Taking the lexicographic minimum identifies only reciprocal orientations.
It preserves both self-image contacts `(i, i, s)` with nonzero `s` and distinct
shifts for a given site pair. Keeping only `i < j` would discard every
self-image contact.

In the unit-cube example, the two x contacts have keys `(0, 1, (0, 0, 0))`
and `(0, 1, (-1, 0, 0))`. Each site also meets its own images along y and z:
for example `(0, 0, (0, 1, 0))` and `(0, 0, (0, -1, 0))` form one
reciprocal pair. The complete set has **six** keys: two between-site contacts
and four self-image contacts.

A site-ID-only simple graph can deliberately discard shifts, merge all
contacts for each site pair, and omit loops. That is a lossy projection of the
image-qualified graph and cannot reconstruct its periodic geometry.

## Normalization utilities

To index shared geometry, use the explicit spatial helpers
`pyvoro2.normalize_vertices`, `pyvoro2.normalize_edges_faces`, or
`pyvoro2.normalize_topology`. The last returns a `NormalizedTopology` containing
global vertices, edges, faces, and mappings from cell-local geometry, including
lattice shifts. It does not attach these objects automatically to a spatial
`TessellationResult`, and its tolerance-based deduplication is not a universal
certified incidence-complex contract or a guarantee of stable numbering after
input permutation.

See [geometric output and normalization](forward/index.md#geometric-output-and-normalization)
and the [spatial reference](../reference/normalize.md). The
[planar helpers](planar.md#planar-normalization) construct global vertices and
edges with their own dimension-specific output.

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

These checks detect inconsistencies in the observed output. Even mutually
consistent contacts and a correct total volume do not prove that every positive
source facet was returned by Voro++. Source-semantic diagnostics and optional
global incidence assembly are separate from these checks; see
[capabilities and limitations](capabilities.md).
