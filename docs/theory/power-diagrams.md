# Power diagrams

Power diagrams are the weighted analogue of ordinary Voronoi diagrams. This
page fixes the notation and terminology used throughout pyvoro2. The definitions
apply in both two and three dimensions unless a domain or backend limitation is
stated explicitly.

The purpose is not to reproduce a full computational-geometry treatment. It is
to make the mathematical objects behind the forward API, separator fitting, and
future prescribed-measure methods precise enough that the same words keep the
same meaning across the project.

## Geometric setting

Let

\[
p_1,\ldots,p_n \in \mathbb{R}^d, \qquad d\in\{2,3\},
\]

be fixed **sites**. A tessellation is computed in a domain \(D\), which may be
bounded or periodic. Each site may also carry an external ID. The ID is metadata,
but its association with the corresponding site and cell must be preserved by
forward computation, normalization, inverse fitting, and serialization.

## Voronoi cells

The ordinary Voronoi cell of site \(i\) is

\[
V_i = \left\{x\in D : \lVert x-p_i\rVert^2
      \le \lVert x-p_j\rVert^2 \text{ for every }j\right\}.
\]

A boundary between two sites lies on their Euclidean perpendicular bisector.
The domain clips bounded cells and identifies periodically equivalent geometry
when periodic boundary conditions are used.

## Weighted cells and power distance

A power diagram assigns one scalar **power weight** \(w_i\) to each site and
compares the power distances

\[
\pi_i(x)=\lVert x-p_i\rVert^2-w_i.
\]

The power, or Laguerre, cell of site \(i\) is

\[
P_i = \left\{x\in D : \pi_i(x)\le\pi_j(x)
      \text{ for every }j\right\}.
\]

Increasing \(w_i\) lowers \(\pi_i\) everywhere and therefore favors site \(i\)
in the competition for space. This often enlarges its cell, but a weight is not
itself a radius, an area, or a volume. The final cell is determined collectively
by all sites, all weights, and the domain.

The ordinary Voronoi diagram is recovered when all weights are equal. Because a
common additive shift has no geometric effect, those equal weights may be taken
to be zero.

## Pairwise separators and realized boundaries

For two distinct sites \(i\) and \(j\), their pairwise power separator is the
hyperplane on which their power distances agree:

\[
\pi_i(x)=\pi_j(x).
\]

Expanding this equality gives

\[
2(p_j-p_i)\cdot x
= \lVert p_j\rVert^2-\lVert p_i\rVert^2+w_i-w_j.
\]

The sites therefore fix the separator orientation, while the weight difference
\(w_i-w_j\) fixes its offset. This affine dependence on a weight difference is
the basis of separator-based inverse fitting.

The pairwise separator is an algebraic object: it can be written for every pair
of distinct sites. It becomes a **realized boundary** only where the tied pair
also beats every competing site and intersects the domain with positive
\((d-1)\)-dimensional measure. In 2D that boundary is an edge; in 3D it is a
face. A fitted pairwise equation and a realized cell boundary are therefore
related, but not equivalent, statements.

## Weights, radii, and backend representation

Voro++ represents a weighted site by a non-negative radius \(r_i\) and uses
\(r_i^2\) as its effective power weight. The forward pyvoro2 `compute(...)`
APIs accept mathematical `weights=` directly and retain `radii=` as an
alternative representation.

Inverse methods naturally produce general real-valued weights, including
negative values. Choose a common constant \(c\) such that \(w_i+c\ge 0\) for all
sites, and define

\[
r_i=\sqrt{w_i+c}.
\]

The shifted weights produce the same diagram because every pairwise comparison
is unchanged:

\[
\bigl(\lVert x-p_i\rVert^2-(w_i+c)\bigr)
-
\bigl(\lVert x-p_j\rVert^2-(w_j+c)\bigr)
=
\pi_i(x)-\pi_j(x).
\]

The radii passed to Voro++ are therefore a computational representation of the
weights. The selected shift is representation metadata, not a uniquely
recovered physical quantity.

### Numerical dynamic range of the backend

Finite weights or radii are not by themselves a guarantee that the native
radical tessellation is numerically resolvable. Voro++ performs parts of the
power computation with binary64 expressions involving absolute squared radii.
If those values, or the genuine range of the power weights, become very large
relative to squared coordinate and domain scales, small geometric terms can be
lost to floating-point rounding.

There is no universal radius limit that pyvoro2 can state independently of the
problem scale. The onset depends on the geometry, the requested accuracy, the
platform, and compiler floating-point behavior; periodic power tessellations
are a particularly sensitive regime. For intuition only, in a unit-scale
problem an absolute radius near \(10^6\) makes the spacing of numbers near
\(r^2\) roughly \(10^{-4}\), while around \(10^8\) an order-one squared-distance
term can be lost entirely. These values are illustrations, not supported-range
thresholds.

The weight-first route removes an irrelevant common additive weight offset
before conversion, but it cannot remove a genuinely large weight range. Direct
`radii=` input may additionally contain a large common squared-radius offset.
pyvoro2 currently preserves the caller's radius representation rather than
silently re-gauging it, and it does not weaken topology or periodic-shift
validation to make an unresolved native result appear valid. A future upstream
Voro++ fix can be adopted through the normal vendored-backend update without
changing the mathematical pyvoro2 API.

## Global additive gauge

For any scalar \(c\), replacing every weight by \(w_i+c\) leaves the complete
power diagram unchanged. This one-dimensional invariance is the **global
additive gauge**.

A solver or result object may choose a convenient representative, such as
mean-zero weights or the minimal shift required for non-negative backend radii.
The geometry and all weight differences are independent of that choice.

A different ambiguity appears when inverse observations connect only separate
subsets of sites. Each connected observation component may then admit its own
unidentified offset. Such component offsets leave the observed equations
unchanged, but they are not generally symmetries of the complete diagram:
changing the relative offsets can alter competition between components. This
distinction is developed in
[Inverse fitting from separator observations](separator-inverse.md).

## Empty and hidden cells

A weighted site may be dominated everywhere in the domain and therefore have an
empty cell. This is a normal geometric possibility in a power diagram, not by
itself evidence of numerical failure.

Empty-cell policy is nevertheless part of the scientific result. Forward code
may need an explicit record for every input site; realization diagnostics must
report when a requested boundary cannot exist because an endpoint cell is
empty; and a future prescribed-measure solver must distinguish an intended zero
target from collapse of a cell with a positive target.

## Periodic images and image-labelled adjacency

In a periodic domain, a site represents the translated images

\[
p_j + A s, \qquad s\in\mathbb{Z}^d,
\]

where the columns of \(A\) are the lattice vectors. A realized periodic boundary
is therefore identified by both the neighbor site \(j\) and the integer image
shift \(s\).

pyvoro2 exposes these integer shifts on periodic boundaries. The convention must
remain consistent across forward output, graph construction, separator
observations, and realization matching. Different images of the same unordered
site pair can give distinct observation records, but the inverse problem still
has one weight per site rather than one weight per image.

## Cell measures, boundary measures, and topology

The project uses **cell measure** as a dimension-independent term:

- area in 2D;
- volume in 3D.

Likewise, **boundary measure** means edge length in 2D and face area in 3D. This
shared vocabulary lets forward results and future prescribed-measure inverse
methods use a common conceptual interface without hiding dimension-specific
geometry.

For a complete bounded or periodic tessellation, the non-overlapping cell
measures should sum to the domain measure within numerical tolerance. Any
inverse method based on target measures must state how it handles mass-balance
mismatch, partial targets, and empty cells.

Two non-empty cells are adjacent when their common boundary has positive
\((d-1)\)-dimensional measure. Degenerate contacts of zero measure, duplicated
numerical vertices, and backend record ordering should not be mistaken for
stable topology. pyvoro2 therefore provides validation and normalization in
addition to raw cell records.

## Relation to the current API

The current development API computes power diagrams through `mode='power'` and
exactly one of `weights=...` or `radii=...`. Direct weights are available on
the spatial and planar `compute(...)` functions, not automatically on every
forward operation. The radius-based path remains available for compatibility.

See the [concepts guide](../guide/concepts.md) for current calls, the
[architecture](../development/architecture.md) for the target result contract,
and the [separator theory](separator-inverse.md) for the implemented inverse
problem.
