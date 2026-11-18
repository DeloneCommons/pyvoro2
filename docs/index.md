---
title: Home
template: landing.html
hide:
  - navigation
  - toc
---

<div class="landing" markdown="1">
<div class="landing-hero" markdown="1">
<div class="landing-intro" markdown="1">

# Voronoi and power tessellations *in Python*

Python tools built on Voro++ for periodic geometry, cell analysis, and
inverse weight fitting.

</div>
<figure class="landing-figure" markdown="1">
  <img src="assets/landing/triclinic-voronoi.png#only-light" width="1600" height="900"
       alt="Four complete Voronoi polyhedra surround four points. A dashed skew frame outlines one fundamental domain; the polyhedra extend across that frame.">
  <img src="assets/landing/exploration/triclinic-voronoi-dark.png#only-dark" width="1600" height="900"
       alt="Four complete Voronoi polyhedra surround four points. A dashed skew frame outlines one fundamental domain; the polyhedra extend across that frame.">
  <figcaption markdown="span">Fully periodic triclinic 3D Voronoi · [Example](guide/forward/index.md)</figcaption>
</figure>
</div>

<div class="landing-paths" markdown="1">
<section markdown="1">

## [Compute tessellations](guide/forward/index.md)

Standard Voronoi and power/Laguerre diagrams, with dedicated 2D and 3D interfaces.

<dl class="landing-facts">
  <dt>Domains &amp; contacts</dt>
  <dd>Bounded domains. Rectangular periodic 2D; orthorhombic or fully periodic triclinic 3D. Contacts retain the neighbor’s lattice image.</dd>
  <dt>Geometry &amp; measures</dt>
  <dd>Cell shapes, areas and volumes. Global vertices and edges in 2D; separate normalization helpers for vertices, edges and faces in 3D.</dd>
  <dt>Diagnostics &amp; validation</dt>
  <dd>Check area or volume totals and periodic-contact consistency.</dd>
</dl>

[Capabilities and limitations](guide/capabilities.md)

</section>
<section markdown="1">

## [Fit power weights](guide/inverse/index.md)

Fit power weights to observed pairwise separator positions at fixed points,
reconciling the observations in one weighted diagram.

<figure class="landing-cycle">
  <img src="assets/landing/separator-cycle.png#only-light" width="820" height="444" loading="lazy"
       alt="Two panels show the same three fixed points. Left: short marks on dashed lines joining the points locate three observed pairwise separators. Right: the power diagram computed from fitted weights has three shared edges. The original marks remain offset from those edges; none of the observations is matched exactly.">
  <img src="assets/landing/exploration/separator-cycle-dark.png#only-dark" width="820" height="444" loading="lazy"
       alt="Two panels show the same three fixed points. Left: short marks on dashed lines joining the points locate three observed pairwise separators. Right: the power diagram computed from fitted weights has three shared edges. The original marks remain offset from those edges; none of the observations is matched exactly.">
  <figcaption>Observed positions (ticks) are approximately matched by one fitted diagram. All three cell pairs meet in this example.</figcaption>
</figure>

</section>
</div>

<section class="landing-quickstart" markdown="1">

## Quick start

<div class="landing-start" markdown="1">
<div markdown="1">

<h3 id="installation">Install</h3>

```bash
python -m pip install pyvoro2
```

CPython 3.10–3.14 (GIL enabled). NumPy is the only required runtime dependency.

[Wheels and source builds](https://github.com/DeloneCommons/pyvoro2/blob/dev/CONTRIBUTING.md)

</div>
<div markdown="1">

### Compute

```python
import pyvoro2 as pv

points = [[0.25, 0.5, 0.5],
          [0.75, 0.5, 0.5]]
box = pv.Box(((0.0, 1.0),) * 3)
result = pv.compute(points, domain=box)
print(result.cell_measures)
# [0.5 0.5]
```

</div>
</div>
</section>

<nav class="landing-explore" aria-label="Explore documentation" markdown="1">

[Examples](examples/index.md)
[Theory](theory/index.md)
[API Reference](reference/index.md)

</nav>

<div class="landing-footer" markdown="1">

Beta · Project code: LGPLv3+ · Voro++: upstream license.

[Project](project/index.md) · [Development](development/index.md) · [License and attribution](project/license.md)

</div>
</div>
