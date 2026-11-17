---
title: Home
hide:
  - navigation
  - toc
---

<div class="landing" markdown="1">
<div class="landing-hero" markdown="1">
<div class="landing-intro" markdown="1">

# Voronoi and power tessellations *in Python*

pyvoro2 computes cells around points in 2D and 3D. It can also infer the
weights of those points from measured cell boundaries.

<div class="landing-links" markdown="1">

[Installation](#installation)
[API Reference](reference/index.md)

</div>

</div>
<figure class="landing-figure">
  <img src="assets/landing/triclinic-voronoi.png" width="1600" height="900"
       alt="Four translucent Voronoi polyhedra surround dark site markers. Blue and green foreground cells extend across the dashed outline of a skew lattice cell.">
  <figcaption>3D Voronoi tessellation in a fully periodic triclinic cell.</figcaption>
</figure>
</div>

<div class="landing-paths" markdown="1">
<section markdown="1">

## [Compute tessellations](guide/forward/index.md)

Find the region nearest each site, or assign weights to compute a power
(Laguerre) diagram. pyvoro2 provides Python bindings to Voro++ with structured
results and diagnostics. Inspect cell shapes, measure areas or volumes, and see
[which neighbor image](guide/periodic-contacts.md) shares a boundary.

- **[2D](guide/planar.md)** — Finite rectangles, with optional periodicity along either or both axes.
- **[3D](guide/domains.md)** — Finite boxes, orthorhombic cells with per-axis periodicity, or fully periodic triclinic cells.

[Capabilities and limitations](guide/capabilities.md)

</section>
<section markdown="1">

## [Fit power weights](guide/inverse/index.md)

With sites fixed, measure where a boundary crosses the line between two sites.
Repeat for other pairs. Inverse fitting finds one set of weights that best
matches the observations, even when they disagree.

Compute the fitted diagram to see which cells touch. A third cell can separate
a pair even when its boundary position is fitted exactly.

<figure class="landing-cycle">
  <img src="assets/landing/separator-cycle.png" width="820" height="444" loading="lazy"
       alt="Left, ticks on the three sides of a dashed triangle mark measurements between fixed sites. Right, the same ticks sit just off the computed edges, and all three cells share pairwise contacts.">
  <figcaption>The three ticks mark observed positions. The fitted edges miss all three marks, yet each pair of cells meets.</figcaption>
</figure>

</section>
</div>

<section class="landing-start" markdown="1">
<div markdown="1">

## Installation

```bash
python -m pip install pyvoro2
```

For standard GIL-enabled CPython 3.10–3.14. Only NumPy is required at runtime.

[Wheels and source builds](https://github.com/DeloneCommons/pyvoro2/blob/dev/CONTRIBUTING.md)

</div>
<div markdown="1">

## Divide a unit cube

Place two sites symmetrically about the midplane. Each Voronoi cell occupies half the cube.

```python
import numpy as np
import pyvoro2 as pv

points = np.array([
    [0.25, 0.5, 0.5],
    [0.75, 0.5, 0.5],
])
box = pv.Box(((0.0, 1.0),) * 3)
result = pv.compute(points, domain=box)
print(result.cell_measures)  # [0.5 0.5]
```

</div>
</section>

<nav class="landing-explore" aria-labelledby="explore-the-documentation" markdown="1">

## Explore the documentation

- [Examples](examples/index.md)<br>
  Build a periodic neighbor graph or follow a worked separator fit.
- [Theory](theory/index.md)<br>
  Learn how distance and weight determine a cell boundary.
- [API Reference](reference/index.md)<br>
  Look up Python signatures, return values and output conventions.

</nav>

<div class="landing-footer" markdown="1">

pyvoro2 is beta. Project code is licensed under LGPLv3+. Vendored Voro++ retains its upstream license.

[Project](project/index.md) · [Development](development/index.md) · [License and attribution](project/license.md)

</div>
</div>
