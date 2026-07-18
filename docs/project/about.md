# About pyvoro2

## Summary

pyvoro2 is a scientific Python package for **forward and inverse weighted
tessellations** in two and three dimensions.

Its forward core wraps vendored Voro++ backends for:

- standard Voronoi tessellations;
- power/Laguerre tessellations;
- bounded and periodic domains;
- explicit periodic neighbor-image labels;
- diagnostics, normalization, and graph-ready geometric output.

Its implemented inverse layer fits power weights from selected pairwise
separator observations, reports compatibility, identifiability, and
hard-constraint feasibility, and separates algebraic fitting from
realized-boundary checks.
Prescribed cell measures and mixed inverse problems are roadmap items rather
than current release claims.

## What is Voro++?

Voro++ is an established C++ library for efficient Voronoi-cell computation.
pyvoro2 vendors a 3D snapshot and the legacy upstream 2D sources, then builds two
separate pybind11 extensions.

The vendored 3D snapshot includes the upstream numeric robustness fix for
power/Laguerre radical pruning, avoiding rare cross-platform failures in fully
periodic weighted tessellations.

pyvoro2 stays close to the backend where that is useful, but it is not only a
binding. Much of its scientific value is in the Python-side domain model,
periodic image reconstruction, validation, normalized topology, inverse
problem, and structured diagnostics.

## Explicit dimensional namespaces

The package keeps dimensions visible:

- `pyvoro2` is the 3D namespace, with `Box`, partially periodic
  `OrthorhombicCell`, and triclinic `PeriodicCell`;
- `pyvoro2.planar` is the 2D namespace, with planar `Box` and rectangular
  periodic `RectangularCell`.

Common concepts should use aligned terminology, but the package does not claim
that both backends support every identical domain or output feature.

## What pyvoro2 adds

Compared with a minimal wrapper, pyvoro2 provides:

- Python-side validation and duplicate safety checks;
- 3D triclinic and partially periodic domain handling;
- rectangular periodic planar workflows;
- image-labelled adjacency for periodic graph construction;
- tessellation diagnostics and strict validation;
- vertex/topology normalization;
- face and edge property annotation;
- `locate` and `ghost_cells` operations;
- optional 2D/3D visualization helpers;
- separator-based inverse fitting with confidence, robust losses, hard
  restrictions, graph diagnostics, realization matching, and active-set path
  diagnostics;
- record/JSON-friendly reporting for downstream research packages.

## Forward and inverse roles

The forward core remains independently useful. A user does not need the inverse
layer to compute cells, measures, boundaries, or periodic neighbor graphs.

The inverse layer answers a different question: given fixed sites and partial
geometric observations, which power weights reconcile those observations? The
current method uses pairwise separator positions. Future methods are intended to
reuse the same geometry and result contract for prescribed cell measures and
mixed data.

The mathematical distinction between weights, backend radii, global gauge,
disconnected observation offsets, and realized boundaries is described in the
[theory section](../theory/index.md).

## Relationship to downstream applications

pyvoro2 is intentionally domain-agnostic. For example, chemvoro can supply
atomic reference data and proposed interatomic separator positions while
pyvoro2 performs the geometric and inverse calculations.

Chemistry-specific interpretation, atomic models, and application policy belong
in chemvoro rather than the pyvoro2 core. This separation also lets pyvoro2 serve
materials, image reconstruction, and other weighted-tessellation workflows.

## Stateless computation

The current public API is stateless: each call creates a backend container,
inserts the sites, performs the operation, and returns Python objects.

This avoids hidden mutable state and keeps calls reproducible. A persistent
container/index could be considered for a demonstrated performance need, but it
is not a near-term architectural requirement.

## Testing and validation

Numerical geometry needs layered validation. pyvoro2 uses:

- deterministic unit and regression tests in the default `pytest` run;
- opt-in fuzz/property tests for random geometries;
- optional cross-checks against the older `pyvoro` wrapper;
- notebook execution and export checks;
- strict documentation and README synchronization checks;
- source/wheel distribution checks and smoke tests.

Typical local validation is:

```bash
python -m pip install -e ".[all]"
pytest -q
python tools/release_check.py
```

See [`CONTRIBUTING.md`](https://github.com/DeloneCommons/pyvoro2/blob/main/CONTRIBUTING.md)
for the development workflow.

## Design and roadmap

- [Development workflow](../development/development-workflow.md) defines how
  plans, decisions, issues, implementation, changelog entries, and releases fit
  together.
- [v0.7 development plan](../development/plans/v0.7.md) records the active
  release scope, accepted decisions, work packages, validation, and release
  gates.
- [Architecture](../development/architecture.md) describes the current v0.6.3
  implementation and target v0.7 responsibilities.
- [API lifecycle](../development/api-lifecycle.md) defines stability and
  compatibility.
- [Decision records](../development/decisions/index.md) explain durable choices.
- [Roadmap](roadmap.md) describes version-level direction toward prescribed
  measures, mixed problems, 1.0, and future research.
