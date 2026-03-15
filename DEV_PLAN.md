# Development plan (0.6.x track)

This file is an internal working plan for the next refactoring/feature cycle.
It is intentionally more concrete than the public roadmap and may evolve during
implementation.

## Current backend decision

We should **not** switch pyvoro2 to `voro-dev` just to obtain 2D support.

From the currently vendored upstream snapshots:

- the dedicated legacy 2D code already provides a realistic first 2D surface
  for pyvoro2 (standard + power tessellations in planar rectangular domains,
  with optional x/y periodicity), while
- `voro-dev` still does **not** appear to add a 2D analogue of pyvoro2's 3D
  `PeriodicCell` / triclinic non-orthogonal periodic domain. It reorganizes the
  backend (separate `_2d` / `_3d` code, iterators, threading-related changes),
  but the non-orthogonal periodic machinery remains 3D-oriented.

That means a future backend migration remains optional rather than required for
2D feature parity. The public Python API should therefore be designed now so
that a later engine swap is mostly internal.

## 0.6.x goals

1. finish the 3D refactoring needed to avoid duplicating geometry and input
   logic when 2D lands;
2. add a first-class planar namespace and bindings on top of the existing 2D
   backend;
3. keep the inverse-fitting / mathematical API reusable across dimensions;
4. prepare, but do not prematurely promise, a future path for planar oblique
   periodic domains.

## Public 2D API shape

### Namespace

Expose 2D from a dedicated namespace:

```python
import pyvoro2.planar as pv2
```

Do **not** silently overload the current 3D top-level API based on
`points.shape[1] == 2`.

### Domains

First release should expose:

- `pyvoro2.planar.Box`
- `pyvoro2.planar.RectangularCell`

The first 2D release should **not** expose `pyvoro2.planar.PeriodicCell` yet,
because the current backend scope is honest only for rectangular planar domains
with optional x/y periodicity.

### Main operations

Plan for symmetry with the 3D API:

- `pyvoro2.planar.compute(...)`
- `pyvoro2.planar.locate(...)`
- `pyvoro2.planar.ghost_cells(...)`
- `pyvoro2.planar.analyze_tessellation(...)`
- `pyvoro2.planar.validate_tessellation(...)`
- `pyvoro2.planar.normalize_vertices(...)`
- `pyvoro2.planar.normalize_topology(...)`
- `pyvoro2.planar.validate_normalized_topology(...)`
- `pyvoro2.planar.annotate_edge_properties(...)`

### Raw 2D cell schema

Keep raw 2D output natural and dimension-specific:

- `area` instead of `volume`
- `edges` instead of `faces`
- `adjacent_shift` remains acceptable as the periodic-image key, but it is now
  a length-2 tuple

We should not force a fake dimension-neutral raw schema. Cross-dimensional
consistency belongs one layer above, not in the lowest-level cell dictionary.

### Visualization

Add a dedicated `viz2d.py` using `matplotlib` and return `(fig, ax)`.
This should be a small optional dependency surface, separate from `viz3d.py`.

## Power-fit / inverse-fitting plan

The inverse-fitting layer is the main candidate for real cross-dimensional reuse.

### Keep the stable math boundary

Retain `PairBisectorConstraints` as the stable boundary between:

- pair-generation / geometric normalization, and
- weight solving / active-set refinement / reporting.

### Dimension-neutral refactor targets

Refactor the internals so that:

- `PairBisectorConstraints.shifts` is shape `(m, d)`
- `PairBisectorConstraints.delta` is shape `(m, d)`
- shift parsing is parameterized by dimension
- shift-to-Cartesian conversion is delegated to a domain adapter
- `boundary_measure` remains generic (`face area` in 3D, `edge length` in 2D)

The public solver names can stay shared:

- `resolve_pair_bisector_constraints(...)`
- `fit_power_weights(...)`
- `match_realized_pairs(...)`
- `solve_self_consistent_power_weights(...)`

## Internal refactoring plan

### 1. Shared validation helpers

Keep extracting validation/coercion from the top-level wrappers into internal
helpers so 2D can reuse them without copy/paste.

### 2. Internal domain-geometry adapters

Continue building small internal adapters that answer:

- dimension
- periodic axes
- lattice vectors / shift-to-Cartesian mapping
- remapping into the primary domain
- nearest-image resolution
- default block-grid heuristics

Current 3D work lives in `_domain_geometry.py`; 2D should get a matching
adapter rather than duplicating geometry logic inside the planar API wrapper.

### 3. Thin public wrappers

`src/pyvoro2/api.py` should remain a thin public surface.
More of the work should move into internal modules so that:

- 3D and 2D wrappers stay parallel, and
- package-wide validation / geometry behavior is easier to test in isolation.

## Binding plan

### Separate extension modules

Build 2D as a separate compiled module:

- `_core` for 3D (existing)
- `_core2d` for 2D (new)

Do not try to merge 2D and 3D into one extension.

### First 2D binding scope

Bind the following operations first:

- `compute_box_standard` / `compute_box_power`
- `locate_box_standard` / `locate_box_power`
- `ghost_box_standard` / `ghost_box_power`

with 2D-specific payloads and reconstruction of edge records from the ordered
polygon vertices and neighbor data returned by the backend.

## Testing plan

Mirror the current 3D coverage categories for 2D:

- standard tessellations
- power tessellations
- bounded boxes
- rectangular periodic domains
- empty-cell behavior in power mode
- locate
- ghost cells
- periodic edge shifts
- topology normalization and validation
- edge-property annotation
- power-fit resolution / solving / realization / active set
- visualization smoke tests
- fuzz/property tests

## Planar `PeriodicCell` note

The current C++ situation does not obviously provide a native 2D oblique
periodic container. That should **not** block the first 2D release.

However, we should keep one exploratory fallback in mind:

- implement a future planar `PeriodicCell` via a **pseudo-3D run** with
  zero `z` coordinates, a carefully controlled out-of-plane setup, and
  post-processing that projects the resulting 3D cell data back to 2D.

This is **not** the preferred first implementation path, and it may turn out to
be too fragile or too expensive for general use. It should be treated as a
research option for later, not as the baseline 0.6.x plan.

## Remaining 0.6.x preparatory steps before public 2D

- finish extracting shared 3D API validation / geometry logic;
- complete the first dimension-neutral refactor of the power-fit boundary;
- add `_core2d` build scaffolding and minimal planar Python namespace;
- add `viz2d.py`;
- document the exact first-release scope for 2D rectangular domains;
- decide whether any exploratory pseudo-3D `PeriodicCell` prototype is worth
  testing behind an internal or experimental flag.
