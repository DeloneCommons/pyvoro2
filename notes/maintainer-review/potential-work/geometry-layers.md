# Geometry and graph layers

**Status:** proposed responsibility split; not an accepted public API design.

Keep distinct objects distinct, and avoid making expensive global reconstruction a prerequisite for ordinary forward tessellation.

| Layer | Represents | Primary uses |
| --- | --- | --- |
| **Local cells (core)** | Independent cell-local vertices, edges/faces, measures, neighbor site IDs | Forward computation, simple pyvoro-like workflows |
| **Face adjacency (derived)** | Realized site-pair contacts, face geometry, periodic image shift for each directed contact | Contact graphs, scientific neighbor analysis, realization diagnostics |
| **Global geometric complex (derived)** | Unique geometric vertices/edges/faces, incidence, periodic identifications and winding | Topology of pores/channels, QTAIM-inspired geometric analogues, geometric networks |

The **separator-observation graph** is separate: its nodes are sites and its edges are measured pairwise constraints for inverse fitting. It is not a tessellation-derived adjacency graph and needs no global cell geometry.

## Intended separation

- Raw/local cells must remain usable if adjacency construction or globalization fails.
- Globalization is a **downstream-required capability**, but need not run on every `compute` call.
- Periodic adjacency must distinguish multiple images of the same site; an unordered site pair alone is insufficient.
- Global vertex identity, coordinate closeness, and face incidence are different assertions. Preserve raw geometry and label any repair/approximation.
- First assess how much the current `TessellationResult` and normalization functions already provide; do not invent a second core API.

## Candidate: field integration and fixed-boundary analysis

**Status:** ChemVoro scientific research direction, not an accepted pyvoro2 API or release
commitment. Prioritize gas-phase molecules; crystal IAS integration is out of initial scope.

- **Atomic-basin integrals:** compare \(\int_{\Omega_A^L}\rho(\mathbf r)\,dV\),
  populations, charges and other properties of Laguerre cells with QTAIM basins.
  A power cell need not contain its generating site.
- **Interatomic-surface integrals:** use realized Laguerre faces \(F_{AB}^L\)
  as planar approximations to QTAIM interatomic surfaces (IAS). Study face area,
  \(\int_{F_{AB}^L}\rho\,dS\), other surface integrals, and averages per area.
  A normal-gradient residual \(\nabla\rho\cdot\mathbf n\) tests the *zero-flux*
  condition independently of integral correlations; a Laguerre face is not a
  QTAIM zero-flux surface by definition.
- **Quadrature:** consider face triangulation and adaptive surface integration;
  convex-cell tetrahedralization and volume integration from a proven internal
  point (not automatically the generator); or a shared grid with batched
  `locate()` for precomputed volumetric fields.
- **Frozen versus refitted boundaries:** fixed sites and weights give fixed
  Laguerre boundaries even as \(\rho\) redistributes. Refit against separators
  in a new state to expose the *boundary-motion* contribution separately from
  a density-only change in the original frozen partition.
- **Comparison design:** use the same wavefunction, compatible finite
  integration regions/cutoffs, converged quadrature, and independent QTAIM
  reference integrations. Include ordinary Voronoi and fixed-radius power
  baselines. A good correlation alone does not prove IAS geometric agreement.

For aligned states, writing \(N_A(\rho,\Omega)=\int_\Omega\rho\,dV\) gives
\[
\Delta N_A =
[N_A(\rho_1,\Omega_0)-N_A(\rho_0,\Omega_0)]
+[N_A(\rho_1,\Omega_1)-N_A(\rho_1,\Omega_0)].
\]
This separates density change on a frozen region from subsequent boundary
change at the final density. The split depends on the **chosen order**;
changing nuclear geometries also requires coordinate/atom correspondence.

**Possible ownership:** pyvoro2 could eventually expose wavefunction-agnostic
geometric quadrature/sampling primitives; ChemVoro would own field evaluation,
QTAIM comparisons, observables, and benchmarks. No release assignment; reconsider only after evidence, design review, and prioritization.

Related: [numerical input considerations](../documentation-notes/numerical-inputs.md)
· [native query performance](native-performance.md)
· [historical global-topology questions](../../../docs/development/reboot/requirements-inventory.md#r-o01).

Follow-up: [reassessment register](../review-register.md) (R-O01 / P-01) · [current result contract](../../../docs/development/decisions/0005-tessellation-result-contract.md).
