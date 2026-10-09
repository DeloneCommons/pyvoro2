# Global topology and periodic images

**Status:** research candidate. **Motivation:** a globally consistent Voronoi/Laguerre complex is a *must-have long-term downstream capability* for ChemVoro's geometric study of atomic basins and potential void/channel analysis. This does **not** approve a construction algorithm or bind it to v0.9.

Voro++ computes cells independently. Current pyvoro2 offers numerical vertex/topology normalization and periodic neighbor-image reconstruction, but they do not certify a universally complete exact global complex.

## Candidate investigations

- Audit current numerical globalization and its assumptions about vertex and edge identity.
- Compare **incidence-first** reconstruction (correspondence before coordinate pooling) with **direct semantic construction** from Laguerre half-spaces.
- Reconstruct/verify periodic translation assignments through boundary correspondence, consistent image transport, winding cycles, and self-neighbor relations.
- Distinguish native occurrences, geometric coincidence, numerical matches, exact identity, and *complete* contact coverage.
- Classify boundary mismatches: coordinate noise, extra collinear vertices, different face contours, missing reciprocal faces, and genuinely omitted positive-area faces.
- Build representative cases and independent mathematical oracles before choosing an algorithm or tolerance policy. Measure correctness *and* computational cost.

**Do not assume “small face ⇒ removable artifact.”** Missing reciprocal faces can indicate wrong geometry or pruning; tiny true features can change topology. Native face matching cannot recover a positive face missing from the native result. Any proposed correction should preserve the raw result and state what was inferred versus proved.

## Scientific role

A Laguerre complex supplies cell/face/edge/vertex structures that can be compared with QTAIM atomic basins and candidate BCP/RCP/CCP locations. This is a **geometric analogy**, not a proof that Laguerre features are electron-density critical points.

Global networks may also support void/channel analysis. Accessible-surface distance and power distance are different metrics for unequal atom radii; scientific definitions and topology-preserving tolerances require their own validation.

## Initial code and evidence

- [`normalize.py`](../../../src/pyvoro2/normalize.py) and [`_internal/normalization.py`](../../../src/pyvoro2/_internal/normalization.py)
- [3D face shifts](../../../src/pyvoro2/_internal/spatial/face_shifts.py) and [2D edge shifts](../../../src/pyvoro2/_internal/planar/edge_shifts.py)
- [3D bindings](../../../cpp/bindings.cpp) and [2D bindings](../../../cpp/bindings2d.cpp)
- [Requirements R-T01–R-T04 and R-O01](../../../docs/development/reboot/requirements-inventory.md#r-o01) · [independent evidence E09–E10](../../../docs/development/reboot/evidence-index.md#e09)

**Open question:** what levels of proven identity, completeness, and image correctness can be offered at an acceptable cost? No representation, correction policy, certification rule, or public API has been selected.
