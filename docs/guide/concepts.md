# Concepts

This guide introduces the geometric objects needed to use pyvoro2. Precise
mathematical definitions and invariants are collected in
[Power diagrams](../theory/power-diagrams.md).

## Sites, domains, and cells

Let

\[
p_1,\ldots,p_n \in \mathbb{R}^d, \qquad d\in\{2,3\},
\]

be **sites** or **generators**. A tessellation assigns points in a bounded or
periodic domain to these sites, producing convex polygonal cells in 2D or
polyhedral cells in 3D.

Cells can be used as:

- a spatial partition;
- a neighbor definition;
- a periodic graph source;
- a geometric model of regions of influence.

pyvoro2 computes two related tessellations using Voro++.

## Standard Voronoi tessellation

The standard Voronoi cell of site \(i\) contains points no farther from \(p_i\)
than from any other site:

\[
V_i=\left\{x:\lVert x-p_i\rVert^2
\le\lVert x-p_j\rVert^2\;\forall j\right\}.
\]

A boundary between two neighboring sites lies on their perpendicular bisector.
In pyvoro2, use:

```python
cells = compute(points, domain=domain, mode='standard')
```

## Power/Laguerre tessellation

A power diagram assigns a scalar **power weight** \(w_i\) to each site and
compares

\[
\pi_i(x)=\lVert x-p_i\rVert^2-w_i.
\]

The cell is

\[
P_i=\left\{x:\pi_i(x)\le\pi_j(x)\;\forall j\right\}.
\]

The separating hyperplane between sites \(i\) and \(j\) depends on the weight
difference \(w_i-w_j\). Larger weight tends to enlarge a cell, but the result is
a collective geometric construction rather than an independent radius around
each point.

### Current radius-based API

Voro++ represents a weight as the square of a non-negative radius. The current
forward pyvoro2 API therefore uses:

```python
cells = compute(
    points,
    domain=domain,
    mode='power',
    radii=radii,
    include_empty=True,
)
```

Mathematically, the weights are \(w_i=r_i^2\). More generally, one common
constant can be added to every weight and the diagram is unchanged. Fitted
weights can therefore be shifted before conversion to non-negative backend
radii.

Radii in this context are a computational representation unless an application
supplies an independent physical interpretation.

### Global gauge

Replacing every weight by \(w_i+c\) leaves all pairwise power-distance
comparisons unchanged. This is the global additive **gauge** of a power diagram.

The absolute weight level is therefore not unique; weight differences and the
realized geometry are the gauge-invariant quantities.

A disconnected inverse observation graph creates additional unidentified
relative offsets between its components. Those are not necessarily harmless
gauge shifts of the complete diagram, because changing them can alter how
sites from different components compete. See the
[separator theory page](../theory/separator-inverse.md).

## Empty cells

A weighted site can be dominated everywhere and have an empty cell. This is a
normal power-diagram outcome, not automatically a numerical error.

Use `include_empty=True` when downstream code needs an explicit record for every
input site. In inverse workflows, empty endpoint cells are also reported by
realization diagnostics.

## Periodicity and neighbor images

In a periodic domain, every site has infinitely many translated images. An
adjacency therefore needs both:

- the neighbor site ID;
- the integer lattice shift of the image that produced the boundary.

In 3D cell records these are commonly:

- `adjacent_cell`;
- `adjacent_shift=(n_a,n_b,n_c)`.

Enable image labels with `return_face_shifts=True`. The planar namespace exposes
analogous edge-shift information for rectangular periodic domains.

The shift is essential for crystal graphs, transport networks, and periodic
separator observations. Different images of the same site are not separate
weight unknowns.

## Cell and boundary measures

The documentation uses **cell measure** for:

- area in 2D;
- volume in 3D.

It uses **boundary measure** for:

- edge length in 2D;
- face area in 3D.

This shared vocabulary is useful for code that supports both dimensions and for
future inverse fitting from prescribed cell measures.

## Algebraic separators and realized boundaries

Every pair of weighted sites defines a two-site separator hyperplane. A pair is
a realized neighbor only where that separator survives competition from all
other sites and intersects the domain with positive boundary measure.

Consequently:

- fitting a requested separator position is an algebraic inverse problem;
- checking whether the pair is an actual face/edge is a forward geometric
  realization problem.

pyvoro2 reports these as separate layers.

## Stateless design

The current public API is stateless. Each call to `compute(...)`, `locate(...)`,
or `ghost_cells(...)` creates a Voro++ container, inserts the sites, performs the
operation, and returns Python objects.

This avoids hidden mutable state and keeps computations reproducible.
