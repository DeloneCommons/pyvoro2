# Theory

The theory section records the mathematical definitions and distinctions that
should remain valid even when Python names, modules, or result containers
change. It states the central equations and interpretations in a compact form;
full proofs and numerical evidence remain in the manuscript and archived
reproducibility materials.

A useful reading order is:

1. [Power diagrams](power-diagrams.md) introduces sites, weights, cells,
   pairwise separators, backend radii, global gauge, periodic images, and
   dimension-independent measure terminology.
2. [Inverse fitting from separator observations](separator-inverse.md) shows how
   local separator data become a graph of weight differences, why cycles control
   compatibility, how the Laplacian estimator arises, and why algebraic fit must
   be separated from geometric realization.

These pages are deliberately independent of provisional API names. Current
calls are documented in the [user guide](../guide/concepts.md), while future
prescribed-measure and mixed inverse methods remain in the
[roadmap](../project/roadmap.md) until their mathematical and software contracts
are mature enough for dedicated theory pages.
