# Documentation levels and style

**Status:** proposed presentation structure.

Make pyvoro2 understandable at four levels:

1. **Introduction:** Voronoi/Laguerre intuition; Voro++ versus pyvoro2; why use the package.
2. **Scientific and user workflows:** mathematical definitions, limitations, runnable examples, common operations.
3. **Implementation architecture:** key files/functions, native–Python boundary, data flow, numerical algorithms, validation and guarantees.
4. **Development and maintenance:** ADRs, tests, policies, historical evidence, releases and contributions.

The first two levels should not require reading C++ or the package source.

**Presentation principles:** lead with intuitive two-site examples; state the practical question before an equation; introduce implementation details only when they answer it. Explicitly separate ideal mathematics, implemented behavior, confirmed numerical failures, and research proposals.

Keep future API candidates out of present-tense user documentation. Follow the repository's [documentation conventions](../../../docs/development/documentation-conventions.md).
