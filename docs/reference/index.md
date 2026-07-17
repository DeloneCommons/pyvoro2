# API reference

The API reference is generated from the docstrings in the current source tree.
Use it when you need exact signatures, parameters, return fields, or exception
behavior. For task-oriented examples, begin with the [user guide](../guide/concepts.md);
for mathematical interpretation, see the [theory section](../theory/index.md).

## Spatial API (3D)

The top-level `pyvoro2` namespace contains 3D domains, forward operations,
diagnostics, validation, normalization, face properties, and visualization.

- [Domains](domains.md)
- [Forward API](api.md)
- [Diagnostics](diagnostics.md)
- [Validation](validation.md)
- [Normalization](normalize.md)

## Planar API (2D)

The explicit `pyvoro2.planar` namespace contains planar domains, forward
operations, structured results, diagnostics, normalization, edge properties,
and visualization.

- [Planar overview](planar/index.md)
- [Planar API](planar/api.md)
- [Planar result](planar/result.md)

## Separator fitting

The current separator-based inverse API lives in `pyvoro2.powerfit` and is also
partly re-exported from the top-level package for compatibility.

- [Power-fitting overview](powerfit/index.md)
- [Constraints and observations](powerfit/constraints.md)
- [Objective models](powerfit/model.md)
- [Solver and results](powerfit/solver.md)
- [Realization](powerfit/realize.md)
- [Active-set refinement](powerfit/active.md)
- [Reports](powerfit/report.md)
