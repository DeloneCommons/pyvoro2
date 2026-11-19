<div class="pv-overview" markdown="1">

# Theory

<div class="pv-overview-lead" markdown="1">

How do sites and weights divide space, and what can observed separators tell
us about those weights? These two topics develop the mathematical ideas
independently of Python API names.

</div>

<div class="pv-overview-row" markdown="1">
<div markdown="1">

## Power diagrams

Assign a site a scalar weight and compare its power distance with those of
all other sites. Each cell contains the points where its site has the least
power distance. Equal weights recover the ordinary Voronoi diagram.

Two sites have equal power along a pairwise separator, whose position depends
on their weight difference. Only the portions where they also beat competing
sites can form a shared cell boundary.

**[Explore power diagrams](power-diagrams.md)**

</div>
<figure markdown="1">

<div class="pv-overview-equation" markdown="1">

\[
\begin{gathered}
\pi_i(x)=\\
\lVert x-p_i\rVert^2-w_i
\end{gathered}
\]

</div>

<figcaption markdown="1">

Power distance is the squared distance from a point to a site, minus that
site’s weight. A weight has units of squared length; it is not a cell area or
volume.

</figcaption>
</figure>
</div>

<div class="pv-overview-row pv-overview-row--reverse" markdown="1">
<div markdown="1">

## Inverse fitting from separator observations

An observed separator position between two fixed sites constrains their
weight difference. Across a cycle of observations, those signed differences
must sum to zero for exact compatibility. Inconsistent observations require
an approximate fit.

Fitting and geometric realization answer different questions: even an
exactly fitted pairwise equation may be hidden by another site's cell.
The chapter develops observation graphs, fitting and checks on actual
boundaries, including periodic images.

**[Explore separator fitting](separator-inverse.md)**

</div>
<figure>
  <img src="../assets/landing/separator-cycle.png#only-light" width="820" height="444" alt="Three fixed sites with observed separator ticks on the left; fitted power cells on the right. The original ticks do not coincide exactly with the fitted edges.">
  <img src="../assets/landing/separator-cycle-dark.png#only-dark" width="820" height="444" alt="Three fixed sites with observed separator ticks on the left; fitted power cells on the right. The original ticks do not coincide exactly with the fitted edges.">
  <figcaption>The three illustrated observations are not exactly compatible. Ticks show the observed positions; computed edges belong to the fitted power diagram. All three pairs meet here, without exactly matching the ticks.</figcaption>
</figure>
</div>

For current calls, see the [User Guide](../guide/concepts.md). The separator
chapter links full proofs and numerical evidence. Prescribed-measure and mixed
inverse methods remain research candidates in the [roadmap](../project/roadmap.md),
without approved implementation or release targets.

</div>
