# API Reference

Use this reference for exact signatures, parameters, return fields and
exceptions from the current source docstrings. Start with the
[User Guide](../guide/index.md) for a task or [Theory](../theory/index.md) for
mathematical interpretation. Lifecycle classifications are recorded in the
[v0.8 API inventory](../development/api-inventory.md) and summarized in
[Choosing an API](../guide/choosing-api.md).

## Find a task or symbol

The navigation groups documentation by dimension and task; it does not change
Python import paths. Both 2D and 3D `compute` return the shared
[TessellationResult (2D/3D)](result.md) by default. Other operations retain their
own return contracts.

| Task or symbol | Public import or call | Exact reference |
|---|---|---|
| Compute 3D standard/power cells | `pyvoro2.compute` | [3D operations](api.md) |
| Compute 2D standard/power cells | `pyvoro2.planar.compute` | [Planar operations](planar/api.md) |
| Fully periodic triclinic cell | `from pyvoro2 import PeriodicCell` | [3D domains](domains.md) |
| Input-aligned IDs, measures and empty state | `from pyvoro2 import TessellationResult` | [Shared result](result.md) |
| Global 3D vertices, edges and faces | `pyvoro2.normalize_topology`, `pyvoro2.NormalizedTopology` | [Spatial normalization](normalize.md) |
| Global 2D vertices and edges | `pyvoro2.planar.normalize_topology`, `pyvoro2.planar.NormalizedTopology`; `pyvoro2.planar.compute(normalize='topology')` | [Planar normalization](planar/normalize.md), [compute](planar/api.md) |
| Image-qualified periodic contacts | `pyvoro2.compute(return_face_shifts=True)`; `pyvoro2.planar.compute(return_edge_shifts=True)` | [3D](api.md), [2D](planar/api.md), and [contact recipe](../guide/periodic-contacts.md) |
| Stable fixed-observation inverse fit | `from pyvoro2.inverse import SeparatorObservations, resolve_separator_observations, fit_weights_from_separators` | [High-level inverse API](inverse/index.md) |
| Provisional models, realization and reports; experimental active-set loop | `import pyvoro2.inverse.separator as separator` | [Advanced separator API](inverse/separator.md) |
| Planar edge properties and visualization | `pyvoro2.edge_properties`, `pyvoro2.viz2d` | [Edge properties](edge_properties.md), [visualization](viz2d.md) |

## Reference groups

- **3D API — `pyvoro2`**: [domains](domains.md), [operations](api.md),
  [diagnostics](diagnostics.md), [validation](validation.md),
  [duplicate checks](duplicates.md), [normalization](normalize.md),
  [face properties](face_properties.md), and [visualization](viz3d.md).
- **2D API — `pyvoro2.planar`**: [namespace overview](planar/index.md),
  [domains](planar/domains.md), [operations](planar/api.md),
  [diagnostics](planar/diagnostics.md), [normalized-geometry validation](planar/validation.md),
  and [normalization](planar/normalize.md). Edge properties and visualization
  are grouped here while retaining their top-level module import paths above.
- **Inverse fitting — `pyvoro2.inverse`**: [high-level API](inverse/index.md)
  and [advanced separator API](inverse/separator.md).
