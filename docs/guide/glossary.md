# Glossary

This glossary fixes the vocabulary used by the forward and separator-inverse
documentation. The theory pages give the mathematical details.

## Power weight

The scalar \(w_i\) in the power function

\[
\pi_i(x)=\lVert x-p_i\rVert^2-w_i.
\]

Power weights have squared-coordinate units. Only differences between weights
affect pairwise separators, and adding one common constant to all weights leaves
the complete power diagram unchanged.

## Backend radius

The non-negative length-unit value passed to Voro++ for power/Laguerre
computation. pyvoro2 converts weights to radii through a common additive shift
before taking square roots. Backend radii are a representation of weights, not
unique physical radii.

## Representation shift

The one common constant added to every mathematical weight so that the shifted
values can be represented as non-negative squared backend radii. It chooses one
backend representation within the global gauge and does not add scientific
information.

## Global gauge

The freedom to add the same constant to every power weight without changing the
complete diagram. A reported weight vector therefore needs a documented gauge
or representation convention.

## Disconnected-component offset

An independent additive offset between components of a disconnected informative
separator-observation graph. Separator observations inside each component do
not determine these relative offsets. Unlike one global gauge shift, changing
component offsets can change the complete realized diagram by changing
competition between components.

## Separator observation

One measured or hypothesized position of the power separator for an ordered
site pair and, in a periodic problem, one requested image of the second site.
The observation may be represented as a fraction along the connector or as a
position. It implies one desired weight difference.

## Informative observation

A separator observation with positive confidence. Informative rows contribute
to data identification and the observation Laplacian. Zero-confidence rows may
remain in records and may carry hard restrictions or penalties, but they do not
identify weight differences from observed data.

## Observation graph

The row-indexed multigraph whose vertices are sites and whose edges are
separator observations. Repeated rows and different periodic images remain
separate edges. Connectivity controls which weight differences are identified.

## Realized face

A positive-measure boundary that actually exists between two cells in the
computed power diagram. In 2D this boundary is an edge. In a periodic problem,
realization includes the specific neighbor image shift.

A fitted separator equation can have a small residual even when the requested
pair is not a realized face.

## Realization diagnostics

Post-fit geometric information that compares requested pairs and image shifts
with the boundaries of a forward-computed diagram. It is separate from the
fixed-observation algebraic fit.

## Active set

The subset of candidate separator observations currently used by the
realization-aware outer algorithm. The active-set workflow alternates fitting
and geometric checks, then adds or removes rows according to hysteretic rules.
It is experimental and is not part of the exact fixed-observation theory.

## Cell measure

The dimension-neutral term for cell area in 2D or cell volume in 3D.
`TessellationResult.cell_measures` is an input-aligned forward result. Fitting
weights from prescribed cell measures is planned for v0.9 and is not implemented
in v0.7.

## Static sparse observation graph

A fixed separator-observation graph whose number of rows is sparse relative to
all possible site pairs. v0.7 provides an optional explicit SciPy-backed
quadratic solve for this regime. The term does not imply trajectory processing,
all-pairs scalability, automatic backend selection, or parallel execution.

## Further reading

- [Power diagrams](../theory/power-diagrams.md)
- [Inverse fitting from separator observations](../theory/separator-inverse.md)
- [Choosing an API](choosing-api.md)
- [Separator fitting](powerfit.md)
