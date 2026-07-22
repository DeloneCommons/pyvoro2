# API reference

The API reference is generated from the docstrings in the current source tree.
Use it when you need exact signatures, parameters, return fields, or exception
behavior. For task-oriented examples, begin with the [user guide](../guide/concepts.md);
for mathematical interpretation, see the [theory section](../theory/index.md).
Lifecycle classifications are authoritative in the
[v0.7 API inventory](../development/api-inventory.md); the concise user-facing
summary is in [Choosing an API](../guide/choosing-api.md).

## Spatial API (3D)

The top-level `pyvoro2` namespace contains 3D domains, forward operations,
diagnostics, validation, normalization, face properties, and visualization.

- [Domains](domains.md)
- [Forward API](api.md)
- [Common tessellation result](result.md)
- [Diagnostics](diagnostics.md)
- [Validation](validation.md)
- [Normalization](normalize.md)

## Planar API (2D)

The explicit `pyvoro2.planar` namespace contains planar domains, forward
operations, structured results, diagnostics, normalization, edge properties,
and visualization.

- [Planar overview](planar/index.md)
- [Planar API](planar/api.md)
- [Common tessellation result](result.md)
- [Historical planar result alias](planar/result.md)

## Separator fitting

The canonical fixed-observation workflow is deliberately small at
`pyvoro2.inverse`; advanced separator-specific functionality lives at
`pyvoro2.inverse.separator`.

- [High-level inverse API](inverse/index.md)
- [Advanced separator API](inverse/separator.md)

The historical `pyvoro2.powerfit` package and broad top-level separator exports
are deprecated compatibility routes for v0.7 and will be removed in v0.8.

- [Compatibility overview](powerfit/index.md)
- [Constraints and observations](powerfit/constraints.md)
- [Objective models](powerfit/model.md)
- [Solver and results](powerfit/solver.md)
- [Realization](powerfit/realize.md)
- [Active-set refinement](powerfit/active.md)
- [Reports](powerfit/report.md)
