# Illustrations to develop

**Status:** ideas only; no illustrations prepared.

- **Cell-by-cell vs. global complex:** locally numbered cell vertices versus shared global identities.
- **Clipping:** one generator, one neighboring generator, and the retained half-space.
- **Numerical mismatch:** two independently computed neighboring cells with slightly different face contours.
- **Periodic neighbors:** same site ID, multiple lattice-image shifts, including self-images.
- **Observation vs. realization:** pairwise separator observations can be algebraically consistent without the requested face being realized.
- **Global complex for ChemVoro:** cells/faces/edges/vertices as geometric candidates for comparison with QTAIM structures, explicitly *not* asserted to be density critical points.
- **Near-degenerate topology:** why a tiny genuine face or channel cannot simply be collapsed according to a universal length/area threshold.

Prefer one concrete geometric question per figure; link each figure to its corresponding scientific limitation or contract.

## Documentation landing image and visualization

- **First Zensical pass:** use a polished static hero image, preferably rendered from real pyvoro2 `compute()` geometry by a standalone optional PyVista script. The first documentation pass must not wait for py3Dmol/K3D refactoring. Keep the input/scene-generation recipe reproducible where practical; do not make PyVista a required package dependency.
- **Final documentation pass:** after the highest-priority v0.9 functionality, reassess the hero image and worked examples against the actual package visualizers. Replace the static asset only if a better native-generated figure is available; the initial image need not be discarded merely because it used an external renderer.
- **Candidate renderers, not promises:** preserve py3Dmol; prototype K3D as an optional 3D addition with minimal shared geometry preparation if justified. For 2D, retain Matplotlib and improve cell fills/colors; consider Plotly as a later, demand-driven interactive option. See [P-04a/P-04b in the review register](../review-register.md).
- **Scientific fidelity:** a rendered collection of cell-local surfaces is not, by itself, a certified globally stitched periodic complex. Keep periodic-image and geometric-identity claims out of decorative captions unless supported by the underlying computation.
