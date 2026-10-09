# Voro++: mathematical and implementation contracts

**Scope:** vendored 2D/3D Voro++ in the v0.8-based reboot. “Required” means a property of the ideal diagram, **not** proof that binary64 execution always preserves it.

| ID | Mathematical expectation / behavior | Implementation boundary |
| --- | --- | --- |
| G1 | Each cell is an intersection of half-spaces and is convex. | Incremental clipping; degeneracies are handled numerically. |
| G2 | Cuts cannot enlarge a cell; exact results do not depend on cut order. | Floating-point outcomes can depend on order and tolerances. |
| G3 | Every generator that could change a cell must be considered. | Block pruning is intended to be conservative; an earlier radical-pruning failure was fixed. |
| G4 | Nonempty cells partition the domain; measures sum to its measure. | Cells are built independently; no global partition certificate is produced. |
| G5 | Shared boundaries are mutually consistent in the ideal diagram. | Native cell records can disagree; a coherent global mesh is not returned. |
| P1 | Power cuts depend on differences `w_i-w_j`, not a common weight offset. | Squared-radius arithmetic can break numerical gauge invariance. |
| P2 | A power cell may be empty or not contain its generator. | Empty native cells are reported through unsuccessful `compute_cell`. |
| P3 | Equivalent periodic site images define equivalent physical tessellations. | Images are handled internally; returned neighbor site IDs omit the image shift. |
| P4 | The total measure of a fully periodic tessellation equals the lattice fundamental measure. | Not globally certified; strongly skewed lattices have specialized bounded search/initialization paths. |
| N1 | Representable finite inputs should not be mistaken for an accuracy guarantee. | Large-radius cases produce wrong geometry; see [investigation](../potential-work/large-weight-numerics.md). |
| N2 | Distinct geometric identities must not be merged solely because coordinates round to the same value. | Numerical globalization alone cannot prove exact vertex identity. |
| N3 | A missing positive face is a correctness defect, not automatically a removable tiny artifact. | Matching existing native faces cannot reconstruct a face that the backend omitted. |
| S1 | An API ideally reports errors without terminating its caller. | Some native error paths call `exit()`; input preflight cannot certify all cases. |

## Distinguish three claims

- **Ideal:** follows from the mathematical Voronoi/power construction.
- **Implemented:** a corresponding numerical algorithm exists in the current native source.
- **Verified:** a specified test or independent oracle passes for a specified input and build. This does **not** imply universal validity.

Evidence: [historical requirements R-T01–R-T04 and R-O01/R-O03](../../../docs/development/reboot/requirements-inventory.md) · [evidence index E06–E10](../../../docs/development/reboot/evidence-index.md). See also [Voro++ overview](../overview/voro.md).
