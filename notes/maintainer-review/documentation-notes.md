# Documentation notes

Status: **Working ideas — not approved documentation structure.**

## Overall direction

Make pyvoro2 understandable at four levels:

1. **Introduction:** What is Voronoi/Laguerre tessellation? What does Voro++ do? What does pyvoro2 add, and why would someone use it?
2. **Scientific and user documentation:** Mathematical concepts, supported functionality, limitations, practical workflows, and examples.
3. **Implementation architecture:** Important files and functions, native/Python boundaries, data flow, numerical algorithms, validation, and guarantees.
4. **Development and maintenance:** ADRs, tests, project policies, release procedures, historical evidence, and contributions.

The first two levels should be accessible without understanding the codebase.

## First-level introduction

Explain Voro++ visually and in simple language.

Its fundamental approach is cell-based: individual Voronoi cells are constructed through successive geometric cuts. A shared global vertex/edge/face complex is not the primary backend representation.

Explain why this matters: independently computed cell boundaries may have numerical inconsistencies, and local geometry is not automatically a certified global topology.

Then introduce pyvoro2:

- What it already provides beyond native Voro++.
- What remains approximate or restricted.
- What makes periodic neighbor identities and lattice shifts difficult.
- Why global topology reconstruction is useful, but not automatically exact.

Clearly distinguish current implemented features from future candidates.

## Illustrations to develop

- Cell-based construction versus a shared global complex.
- A numerical mismatch between independently computed neighboring cells.
- Periodic neighbors: same site identity, different lattice images.
- Why arbitrary pairwise separator observations may not define one realizable power diagram.

## Presentation principle

Start with intuitive examples and visual explanations. Introduce equations and implementation details only when they answer a question the reader already understands.
