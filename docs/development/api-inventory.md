# v0.7 public API inventory

- **Status:** v0.6.3 baseline characterized; v0.7 target inventory remains living
- **Baseline:** v0.6.3
- **Target:** v0.7.0
- **Baseline audit:** [issue #6](https://github.com/DeloneCommons/pyvoro2/issues/6), completed 2026-07-18
- **Policy:** [API lifecycle and compatibility](api-lifecycle.md)
- **Plan:** [v0.7 development plan](plans/v0.7.md)

This inventory records the intended lifecycle status of public imports, return
routes, record schemas, defaults, and scientific semantics for v0.7. It is a
release artifact, not a retrospective document to be written after the release.

The initial categories below guide implementation. They must be checked against
actual code, tests, documentation, and downstream integration before v0.7.0 is
published.

## How to maintain this inventory

For every v0.7 issue that changes public behavior:

1. update the relevant row or section in the same change;
2. distinguish the v0.6.3 baseline from the intended v0.7 state;
3. record aliases, deprecations, and planned removal releases;
4. include defaults, result fields, record keys, units, and periodic conventions
   when they carry scientific meaning;
5. leave uncertain new surfaces **provisional** rather than omitting them;
6. do not mark a surface **stable** until its tests and documentation define the
   contract clearly.

The release review must verify that `__all__`, docstrings, guides, reference
pages, migration notes, and this inventory agree.

## Factual v0.6.3 baseline

This section records behavior observed in the v0.6.3 source tree before any
v0.7 public implementation. It was checked against package `__all__` values,
call signatures, generated API reference pages, user guides, source notebooks,
and the existing tests. Focused executable checks live in
`tests/test_v063_forward_baseline.py` and
`tests/test_v063_inverse_baseline.py`.

“Baseline” does not make every current convenience stable forever. It identifies
what the v0.7 compatibility routes must preserve deliberately and prevents a
module move or result-default change from silently changing established
behavior.

### Current public namespaces, documented module routes, and `__all__`

The exact membership and absence of duplicates in the three package-root lists
and the documented `pyvoro2.viz3d` list are pinned by the baseline tests; export
ordering is not treated as a compatibility promise. The grouped contents below
are exhaustive for those lists. Documented direct module routes are recorded
separately so a later module move does not preserve only package-root imports.

#### `pyvoro2` — 62 exports

| Group | Exact current exports |
|---|---|
| Domains and operations | `Box`, `OrthorhombicCell`, `PeriodicCell`, `compute`, `locate`, `ghost_cells` |
| Tessellation diagnostics | `TessellationDiagnostics`, `TessellationIssue`, `TessellationError`, `analyze_tessellation`, `validate_tessellation` |
| Normalization diagnostics | `NormalizationDiagnostics`, `NormalizationIssue`, `NormalizationError`, `validate_normalized_topology` |
| Duplicate handling | `DuplicatePair`, `DuplicateError`, `duplicate_check` |
| Geometry annotation | `annotate_face_properties` |
| Normalization | `NormalizedVertices`, `NormalizedTopology`, `normalize_vertices`, `normalize_edges_faces`, `normalize_topology` |
| Weight/radius transforms | `radii_to_weights`, `weights_to_radii` |
| Package metadata/namespaces | `__version__`, `planar` |
| Historical inverse surface | the exact 34-name compatibility list under [Compatibility-only top-level inverse names](#compatibility-only-top-level-inverse-names) |

The weight/radius transforms are implemented in `pyvoro2.powerfit.transforms`
in v0.6.3 but are listed separately because ADR 0004 assigns them neutral
ownership in v0.7.

#### `pyvoro2.planar` — 25 exports

```text
Box
RectangularCell
PlanarComputeResult
compute
locate
ghost_cells
DuplicatePair
DuplicateError
duplicate_check
annotate_edge_properties
plot_tessellation
TessellationIssue
TessellationDiagnostics
TessellationError
analyze_tessellation
validate_tessellation
NormalizedVertices
NormalizedTopology
normalize_vertices
normalize_edges
normalize_topology
NormalizationIssue
NormalizationDiagnostics
NormalizationError
validate_normalized_topology
```

#### `pyvoro2.powerfit` — 42 exports

```text
PairBisectorConstraints
resolve_pair_bisector_constraints
SquaredLoss
HuberLoss
Interval
FixedValue
SoftIntervalPenalty
ExponentialBoundaryPenalty
ReciprocalBoundaryPenalty
L2Regularization
FitModel
AlgebraicEdgeDiagnostics
ConstraintGraphDiagnostics
ConnectivityDiagnostics
ConnectivityDiagnosticsError
HardConstraintConflictTerm
HardConstraintConflict
PowerFitBounds
PowerFitPredictions
PowerFitObjectiveBreakdown
PowerFitProblem
PowerWeightFitResult
build_power_fit_problem
build_power_fit_result
RealizedPairDiagnostics
UnaccountedRealizedPair
UnaccountedRealizedPairError
build_fit_report
build_realized_report
build_active_set_report
dumps_report_json
write_report_json
ActiveSetOptions
ActiveSetIteration
ActiveSetPathSummary
PairConstraintDiagnostics
SelfConsistentPowerFitResult
fit_power_weights
match_realized_pairs
solve_self_consistent_power_weights
radii_to_weights
weights_to_radii
```

`PowerFitBounds`, `PowerFitPredictions`, `PowerFitObjectiveBreakdown`,
`PowerFitProblem`, `build_power_fit_problem`, and `build_power_fit_result` are
public from `pyvoro2.powerfit` but are **not** top-level `pyvoro2` exports. v0.7
must not broaden the historical top-level surface while preserving the old
package.

#### `pyvoro2.viz3d` — 9 exports

```text
VizStyle
make_view
add_axes
add_sites
add_vertices
add_domain_wireframe
add_cell_wireframe
add_tessellation_wireframe
view_tessellation
```

`pyvoro2.viz2d` has no explicit `__all__`; its documented public function is
`plot_tessellation`, which is the same object re-exported by
`pyvoro2.planar`. Importing either visualization module does not require its
optional rendering dependency until a rendering function is called.

#### Documented direct module routes

The generated v0.6.3 reference documents these direct module routes in addition
to the package-root exports:

- `pyvoro2.api`, `domains`, `diagnostics`, `duplicates`, `edge_properties`,
  `face_properties`, `normalize`, `validation`, `viz2d`, and `viz3d`;
- `pyvoro2.planar.api`, `domains`, `diagnostics`, `normalize`, `result`, and
  `validation`; and
- `pyvoro2.powerfit.active`, `constraints`, `model`, `realize`, `report`, and
  `solver`.

Where a direct-module object is also exported from its package root, v0.6.3
uses the same object rather than a wrapper. The `powerfit.model` reference also
documents `ScalarMismatch`, `HardConstraint`, and `ScalarPenalty` directly;
these three base classes are not `pyvoro2.powerfit` package exports. The
documented submodule `__all__` values are:

| Module | Exact `__all__` |
|---|---|
| `pyvoro2.powerfit.report` | `build_fit_report`, `build_realized_report`, `build_active_set_report`, `dumps_report_json`, `write_report_json` |
| `pyvoro2.powerfit.solver` | `fit_power_weights`, `ConnectivityDiagnosticsError` |

Private underscore-prefixed helpers rendered nowhere in the public reference
are not part of this baseline.

### Current forward signatures and defaults

The domain constructors are:

```text
pyvoro2.Box(bounds)
pyvoro2.OrthorhombicCell(bounds, periodic=(True, True, True))
pyvoro2.PeriodicCell(vectors, origin=(0.0, 0.0, 0.0))
pyvoro2.planar.Box(bounds)
pyvoro2.planar.RectangularCell(bounds, periodic=(True, True))
```

Documented domain conveniences, with annotations omitted, are:

```text
pyvoro2.Box.from_points(points, padding=2.0)
pyvoro2.OrthorhombicCell.lattice_vectors
pyvoro2.OrthorhombicCell.remap_cart(
    points, *, return_shifts=False, eps=None,
)
pyvoro2.PeriodicCell.from_params(
    bx, bxy, by, bxz, byz, bz, *, origin=(0.0, 0.0, 0.0),
)
pyvoro2.PeriodicCell.to_internal_params()
pyvoro2.PeriodicCell.cart_to_internal(points)
pyvoro2.PeriodicCell.internal_to_cart(points_internal)
pyvoro2.PeriodicCell.remap_internal(
    points_internal, *, return_shifts=False, eps=None,
)
pyvoro2.PeriodicCell.wrap_internal(points_internal)
pyvoro2.PeriodicCell.remap_cart(
    points, *, return_shifts=False, eps=None,
)
pyvoro2.planar.Box.from_points(points, padding=2.0)
pyvoro2.planar.RectangularCell.lattice_vectors
pyvoro2.planar.RectangularCell.remap_cart(
    points, *, return_shifts=False, eps=None,
)
```

The exact spatial operation signatures, with annotations omitted here for
readability, are:

```text
compute(
    points, *, domain, ids=None,
    duplicate_check='off', duplicate_threshold=1e-5,
    duplicate_wrap=True, duplicate_max_pairs=10,
    block_size=None, blocks=None, init_mem=8,
    mode='standard', radii=None,
    return_vertices=True, return_adjacency=True, return_faces=True,
    return_face_shifts=False, face_shift_search=2, include_empty=False,
    validate_face_shifts=True, repair_face_shifts=False, face_shift_tol=None,
    return_diagnostics=False, tessellation_check='none',
    tessellation_require_reciprocity=None,
    tessellation_volume_tol_rel=1e-8,
    tessellation_volume_tol_abs=1e-12,
    tessellation_plane_offset_tol=None,
    tessellation_plane_angle_tol=None,
)

locate(
    points, queries, *, domain, ids=None,
    duplicate_check='off', duplicate_threshold=1e-5,
    duplicate_wrap=True, duplicate_max_pairs=10,
    block_size=None, blocks=None, init_mem=8,
    mode='standard', radii=None, return_owner_position=False,
)

ghost_cells(
    points, queries, *, domain, ids=None,
    duplicate_check='off', duplicate_threshold=1e-5,
    duplicate_wrap=True, duplicate_max_pairs=10,
    block_size=None, blocks=None, init_mem=8,
    mode='standard', radii=None, ghost_radius=None,
    return_vertices=True, return_adjacency=True, return_faces=True,
    include_empty=True,
)
```

The exact planar operation signatures are:

```text
compute(
    points, *, domain, ids=None,
    duplicate_check='off', duplicate_threshold=1e-5,
    duplicate_wrap=True, duplicate_max_pairs=10,
    block_size=None, blocks=None, init_mem=8,
    mode='standard', radii=None,
    return_vertices=True, return_adjacency=True, return_edges=True,
    return_edge_shifts=False, edge_shift_search=2, include_empty=False,
    validate_edge_shifts=True, repair_edge_shifts=False, edge_shift_tol=None,
    return_diagnostics=False, return_result=False, normalize='none',
    normalization_tol=None, tessellation_check='none',
    tessellation_require_reciprocity=None,
    tessellation_area_tol_rel=1e-8,
    tessellation_area_tol_abs=1e-12,
    tessellation_line_offset_tol=None,
    tessellation_line_angle_tol=None,
)

locate(
    points, queries, *, domain, ids=None,
    duplicate_check='off', duplicate_threshold=1e-5,
    duplicate_wrap=True, duplicate_max_pairs=10,
    block_size=None, blocks=None, init_mem=8,
    mode='standard', radii=None, return_owner_position=False,
)

ghost_cells(
    points, queries, *, domain, ids=None,
    duplicate_check='off', duplicate_threshold=1e-5,
    duplicate_wrap=True, duplicate_max_pairs=10,
    block_size=None, blocks=None, init_mem=8,
    mode='standard', radii=None, ghost_radius=None,
    return_vertices=True, return_adjacency=True, return_edges=True,
    return_edge_shifts=False, edge_shift_search=2, include_empty=True,
    validate_edge_shifts=True, repair_edge_shifts=False, edge_shift_tol=None,
)
```

`radii` are required when `mode='power'`; planar and spatial power ghost calls
also require `ghost_radius`. There is no v0.6.3 forward `weights=` argument.

Supporting forward call defaults are also part of the observed surface:

| Call | Current optional parameters and defaults |
|---|---|
| spatial `analyze_tessellation(cells, domain, ...)` | `expected_ids=None`, `mode=None`, `volume_tol_rel=1e-8`, `volume_tol_abs=1e-12`, `check_reciprocity=True`, `check_plane_mismatch=True`, `plane_offset_tol=None`, `plane_angle_tol=None`, `mark_faces=True` |
| spatial `validate_tessellation(cells, domain, ...)` | `expected_ids=None`, `mode=None`, `level='basic'`, `require_reciprocity=None`, volume tolerances `1e-8`/`1e-12`, plane tolerances `None`, `mark_faces=None` |
| planar `analyze_tessellation(cells, domain, ...)` | `expected_ids=None`, `mode=None`, `area_tol_rel=1e-8`, `area_tol_abs=1e-12`, `check_reciprocity=True`, `check_line_mismatch=True`, `line_offset_tol=None`, `line_angle_tol=None`, `mark_edges=True` |
| planar `validate_tessellation(cells, domain, ...)` | `expected_ids=None`, `mode=None`, `level='basic'`, `require_reciprocity=None`, area tolerances `1e-8`/`1e-12`, line tolerances `None`, `mark_edges=None` |
| spatial/planar `duplicate_check(points, ...)` | `threshold=1e-5`, `domain=None`, `wrap=True`, `mode='raise'`, `max_pairs=10` |
| spatial `normalize_vertices(cells, ...)` | required `domain`; `tol=None`, `require_face_shifts=True`, `copy_cells=True` |
| spatial `normalize_edges_faces(normalized_vertices, ...)` | required `domain`; `tol=None`, `copy_cells=True` |
| spatial `normalize_topology(cells, ...)` | required `domain`; `tol=None`, `require_face_shifts=True`, `copy_cells=True` |
| planar `normalize_vertices(cells, ...)` | required `domain`; `tol=None`, `require_edge_shifts=True`, `copy_cells=True` |
| planar `normalize_edges(normalized_vertices, ...)` | required `domain`; `tol=None`, `copy_cells=True` |
| planar `normalize_topology(cells, ...)` | required `domain`; `tol=None`, `require_edge_shifts=True`, `copy_cells=True` |
| spatial `validate_normalized_topology(normalized, domain, ...)` | `level='basic'`, all four checks enabled, `max_examples=10` |
| planar `validate_normalized_topology(normalized, domain, ...)` | `level='basic'`, all four checks enabled, `max_examples=10` |
| `weights_to_radii(weights, ...)` | `r_min=0.0`, `weight_shift=None`; returns `(radii, applied_shift)` |
| `radii_to_weights(radii)` | no optional parameters; returns squared radii |

### Current visualization signatures and defaults

The documented planar visualization entry point is:

```text
pyvoro2.viz2d.plot_tessellation(
    cells, *, ax=None, domain=None, show_sites=False, annotate_ids=False,
)
```

The optional spatial visualization surface is:

```text
VizStyle(
    background='0xffffff', site_color='0x777777', site_radius=0.093,
    site_label_color='0x000000', site_label_background='0xffffff',
    site_label_font_size=8, edge_color='0x1f77b4', edge_line_width=2.5,
    domain_color='0x000000', domain_line_width=2.5,
    vertex_color='0xff7f0e', vertex_radius=0.04,
    vertex_label_color='0x000000', vertex_label_background='0xffffff',
    vertex_label_font_size=7, axes_line_width=2.0,
    axes_label_font_size=12, axes_color_x='0xff0000',
    axes_color_y='0x00aa00', axes_color_z='0x0000ff',
)
make_view(*, width=640, height=480, background='0xffffff')
add_axes(
    view, *, origin=(0.0, 0.0, 0.0), length=1.0, line_width=2.0,
    label_font_size=12, color_x='0xff0000', color_y='0x00aa00',
    color_z='0x0000ff',
)
add_sites(
    view, points, *, labels=None, color='0x777777', radius=0.093,
    label_color='0x000000', label_background='0xffffff',
    label_font_size=8,
)
add_vertices(
    view, vertices, *, labels=None, color='0xff7f0e', radius=0.04,
    label_color='0x000000', label_background='0xffffff',
    label_font_size=7,
)
add_domain_wireframe(view, domain, *, color='0x000000', line_width=2.5)
add_cell_wireframe(view, cell, *, color='0x1f77b4', line_width=2.5)
add_tessellation_wireframe(
    view, cells, *, color='0x1f77b4', line_width=2.5, cell_ids=None,
)
view_tessellation(
    cells, *, domain=None, show_sites=True, show_site_labels=True,
    max_site_labels=200, show_domain=True, show_axes=True, axes_length=None,
    wrap_cells=False, cell_ids=None, show_vertices=False,
    show_vertex_labels='auto', max_vertex_labels=200, style=None,
    width=640, height=480, zoom=True,
)
```

### Characterized v0.6.3 forward return matrix

| Namespace/request | v0.6.3 return |
|---|---|
| spatial default | raw `list[dict]` |
| spatial `return_diagnostics=True` | `(cells, TessellationDiagnostics)` tuple |
| spatial `tessellation_check='diagnose'|'warn'|'raise'` without `return_diagnostics` | raw list after the requested check; diagnostics are not returned |
| planar default | raw `list[dict]` |
| planar `return_diagnostics=True`, with no result/normalization request | `(cells, TessellationDiagnostics)` tuple |
| planar `return_result=True` | `PlanarComputeResult` |
| planar `normalize='vertices'|'topology'` | `PlanarComputeResult`, even when `return_result=False` |
| planar result/normalization plus `return_diagnostics=True` | one `PlanarComputeResult` carrying diagnostics, never a tuple |
| planar `tessellation_check='diagnose'|'warn'|'raise'` without a result or `return_diagnostics` request | raw list after the requested check; diagnostics are not returned |

`PlanarComputeResult` is a frozen, slotted outer dataclass with fields
`cells`, `tessellation_diagnostics=None`, `normalized_vertices=None`, and
`normalized_topology=None`. Its exact documented conveniences are the
`has_tessellation_diagnostics`, `has_normalized_vertices`,
`has_normalized_topology`, `global_vertices`, and `global_edges` properties and
the `require_tessellation_diagnostics()`, `require_normalized_vertices()`, and
`require_normalized_topology()` methods. Its nested raw cell records remain
mutable. There is no spatial counterpart in v0.6.3.

### Current raw record schemas and ordering

| Operation | Required/base keys | Optional keys |
|---|---|---|
| spatial `compute` cell | `id`, `volume`, `site` | `vertices`, `adjacency`, `faces`; inserted hidden cells also have `empty=True` |
| spatial face | `adjacent_cell`, `vertices` | `adjacent_shift`; diagnostic flags `orphan`, `reciprocal_missing`, `reciprocal_mismatch`; annotation fields `centroid`, `normal`, `area`, `other_site`, `intersection`, `intersection_inside`, `intersection_centroid_dist`, `intersection_edge_min_dist` |
| planar `compute` cell | `id`, `area`, `site` | `vertices`, `adjacency`, `edges`; inserted hidden cells also have `empty=True` |
| planar edge | `adjacent_cell`, `vertices` | `adjacent_shift`; diagnostic flags `orphan`, `reciprocal_missing`, `reciprocal_mismatch`; annotation fields `midpoint`, `tangent`, `normal`, `length`, `other_site` |
| spatial `locate` | `found`, `owner_id` arrays | `owner_pos` when requested |
| planar `locate` | `found`, `owner_id` arrays | `owner_pos` when requested |
| spatial ghost cell | normal spatial cell keys plus `id=-1`, `empty`, `query_index`, `query` | requested geometry keys |
| planar ghost cell | normal planar cell keys plus `id=-1`, `empty`, `query_index` | requested geometry keys; unlike spatial ghost records, no `query` key |

Geometry keys are omitted, not set to `None`, when the corresponding
`return_*` switch is false. Wall neighbors retain negative backend IDs.
Nonnegative cell and neighbor IDs are remapped to user `ids` after computation;
`locate.owner_id` is remapped in the same way. `site` remains the generator
coordinate used by the computation and is not replaced by the external ID.

Raw compute output follows backend cell iteration. The characterized ordinary
case returns cells in input/internal-ID order, but consumers should not treat a
raw list position as the public site lookup mechanism. If hidden cells are
actually inserted by `include_empty=True`, the wrapper sorts by internal site
index before remapping external IDs, yielding a full input-aligned list. With
`include_empty=False`, hidden power cells are absent and the list is shorter.

For a hidden power cell, `include_empty=True` inserts `id`, `empty=True`, zero
`volume`/`area`, `site`, and empty lists only for requested geometry. Nonempty
records do not gain `empty=False`. Spatial `compute` applies its reinsertion
helper whenever `include_empty=True`; planar `compute` invokes it only in power
mode. Standard diagrams ordinarily have no hidden cells, so this asymmetry has
no normal observable effect but is part of the implementation baseline.

Spatial normalized result fields are `NormalizedVertices(global_vertices,
cells)` and `NormalizedTopology(global_vertices, global_edges, global_faces,
cells)`. Planar fields are `NormalizedVertices(global_vertices, cells)` and
`NormalizedTopology(global_vertices, global_edges, cells)`.

### Current forward diagnostic fields

| Type | Exact dataclass fields |
|---|---|
| spatial `TessellationIssue` | `code`, `severity`, `message`, `examples` |
| spatial `TessellationDiagnostics` | `domain_volume`, `sum_cell_volume`, `volume_ratio`, `volume_gap`, `volume_overlap`, `n_sites_expected`, `n_cells_returned`, `missing_ids`, `empty_ids`, `face_shift_available`, `reciprocity_checked`, `n_faces_total`, `n_faces_orphan`, `n_faces_mismatched`, `issues`, `ok_volume`, `ok_reciprocity`, `ok` |
| planar `TessellationIssue` | `code`, `severity`, `message`, `examples` |
| planar `TessellationDiagnostics` | `domain_area`, `sum_cell_area`, `area_ratio`, `area_gap`, `area_overlap`, `n_sites_expected`, `n_cells_returned`, `missing_ids`, `empty_ids`, `edge_shift_available`, `reciprocity_checked`, `n_edges_total`, `n_edges_orphan`, `n_edges_mismatched`, `issues`, `ok_area`, `ok_reciprocity`, `ok` |
| spatial `NormalizationDiagnostics` | `n_cells`, `n_global_vertices`, `n_global_edges`, `n_global_faces`, `is_periodic_domain`, `fully_periodic_domain`, `has_wall_faces`, `n_vertex_face_shift_mismatch`, `n_face_vertex_set_mismatch`, `n_vertices_low_incidence`, `n_edges_low_incidence`, `n_cells_bad_euler`, `issues`, `ok_vertex_face_shift`, `ok_face_vertex_sets`, `ok_incidence`, `ok_euler`, `ok` |
| planar `NormalizationDiagnostics` | `n_cells`, `n_global_vertices`, `n_global_edges`, `is_periodic_domain`, `fully_periodic_domain`, `has_wall_edges`, `n_vertex_edge_shift_mismatch`, `n_edge_vertex_set_mismatch`, `n_vertices_low_incidence`, `n_cells_bad_polygon`, `issues`, `ok_vertex_edge_shift`, `ok_edge_vertex_sets`, `ok_incidence`, `ok_polygon`, `ok` |

### Current inverse signatures and constructor defaults

The high-level and advanced call defaults are:

| Call | Required inputs; current keyword defaults |
|---|---|
| `resolve_pair_bisector_constraints(points, constraints, ...)` | `measurement='fraction'`, `domain=None`, `ids=None`, `index_mode='index'`, `image='nearest'`, `image_search=1`, `confidence=None`, `allow_empty=False` |
| `fit_power_weights(points, constraints, ...)` | resolver defaults above except `allow_empty` is internal; `model=None`, `r_min=0.0`, `weight_shift=None`, `solver='auto'`, `max_iter=2000`, `rho=1.0`, `tol_abs=1e-6`, `tol_rel=1e-5`, `connectivity_check='warn'` |
| `build_power_fit_problem(constraints, ...)` | `model=None` |
| `build_power_fit_result(problem, weights, ...)` | `solver='external'`, `status='optimal'`, `status_detail=None`, `converged=True`, `n_iter=0`, `warnings=()`, `canonicalize_gauge=True`, `r_min=0.0`, `weight_shift=None` |
| `match_realized_pairs(points, ...)` | required keyword-only `domain`, `radii`, `constraints`; `return_boundary_measure=False`, `return_cells=False`, `return_tessellation_diagnostics=False`, `tessellation_check='diagnose'`, `unaccounted_pair_check='diagnose'` |
| `solve_self_consistent_power_weights(points, constraints, ...)` | required keyword-only `domain`; resolver defaults; `model=None`, `active0=None`, `options=None`, `r_min=0.0`, `weight_shift=None`, `fit_solver='auto'`, `fit_max_iter=2000`, `fit_rho=1.0`, `fit_tol_abs=1e-6`, `fit_tol_rel=1e-5`, all four return switches `False`, `tessellation_check='diagnose'`, `connectivity_check='warn'`, `unaccounted_pair_check='warn'` |
| `build_fit_report(result, constraints, ...)` | `use_ids=False` |
| `build_realized_report(diagnostics, constraints, ...)` | `use_ids=False` |
| `build_active_set_report(result, ...)` | `use_ids=False` |
| `dumps_report_json(report, ...)` | `indent=2`, `sort_keys=False` |
| `write_report_json(report, path, ...)` | `indent=2`, `sort_keys=False` |

The objective/model constructors are:

```text
SquaredLoss()
HuberLoss(delta=1.0)
Interval(lower, upper)
FixedValue(value)
SoftIntervalPenalty(lower, upper, strength)
ExponentialBoundaryPenalty(
    lower=0.0, upper=1.0, margin=0.02,
    strength=1.0, tau=0.01,
)
ReciprocalBoundaryPenalty(
    lower=0.0, upper=1.0, margin=0.05,
    strength=1.0, epsilon=1e-6,
)
L2Regularization(strength=0.0, reference=None)
FitModel(
    mismatch=SquaredLoss(), feasible=None, penalties=(),
    regularization=L2Regularization(),
)
ActiveSetOptions(
    add_after=1, drop_after=2, relax=1.0, max_iter=25,
    cycle_window=8, weight_step_tol=1e-8,
)
```

### Current inverse result fields

The primary containers have the following exact dataclass fields:

| Type | Fields |
|---|---|
| `PairBisectorConstraints` | `n_points`, `i`, `j`, `shifts`, `target`, `confidence`, `measurement`, `distance`, `distance2`, `delta`, `target_fraction`, `target_position`, `input_index`, `explicit_shift`, `ids`, `warnings` |
| `PowerFitProblem` | `constraints`, `model`, `alpha`, `beta`, `z_obs`, `edge_weight`, `regularization_strength`, `regularization_reference`, `offset_identifying_constraint_mask`, `bounds`, `connectivity`, `hard_feasible`, `hard_conflict` |
| `PowerWeightFitResult` | `status`, `hard_feasible`, `weights`, `radii`, `weight_shift`, `measurement`, `target`, `predicted`, `predicted_fraction`, `predicted_position`, `residuals`, `rms_residual`, `max_residual`, `used_shifts`, `solver`, `n_iter`, `converged`, `conflict`, `warnings`, `status_detail`, `connectivity`, `edge_diagnostics`, `objective_breakdown` |
| `RealizedPairDiagnostics` | `realized`, `unrealized`, `realized_same_shift`, `realized_other_shift`, `realized_shifts`, `endpoint_i_empty`, `endpoint_j_empty`, `boundary_measure`, `cells`, `tessellation_diagnostics`, `unaccounted_pairs`, `warnings` |
| `PairConstraintDiagnostics` | `site_i`, `site_j`, `shift`, `target`, `confidence`, `predicted`, `predicted_fraction`, `predicted_position`, `residuals`, `active`, `realized`, `realized_same_shift`, `realized_other_shift`, `realized_shifts`, `endpoint_i_empty`, `endpoint_j_empty`, `boundary_measure`, `toggle_count`, `realized_toggle_count`, `first_realized_iter`, `last_realized_iter`, `marginal`, `status` |
| `SelfConsistentPowerFitResult` | `constraints`, `fit`, `realized`, `diagnostics`, `active_mask`, `n_outer_iter`, `converged`, `termination`, `cycle_length`, `marginal_constraints`, `rms_residual_all`, `max_residual_all`, `tessellation_diagnostics`, `history`, `path_summary`, `warnings`, `connectivity` |

Supporting fields are exact as follows:

| Type | Fields |
|---|---|
| `PowerFitBounds` | `measurement_lower`, `measurement_upper`, `difference_lower`, `difference_upper` |
| `PowerFitPredictions` | `difference`, `fraction`, `position`, `measurement` |
| `PowerFitObjectiveBreakdown` | `total`, `mismatch`, `penalties_total`, `penalty_terms`, `regularization`, `hard_constraints_satisfied`, `hard_max_violation` |
| `AlgebraicEdgeDiagnostics` | `alpha`, `beta`, `z_obs`, `z_fit`, `residual`, `edge_weight`, `weighted_l2`, `weighted_rmse`, `rmse`, `mae` |
| `ConstraintGraphDiagnostics` | `n_points`, `n_constraints`, `n_edges`, `isolated_points`, `connected_components`, `fully_connected`; property `n_components` |
| `ConnectivityDiagnostics` | `unconstrained_points`, `candidate_graph`, `effective_graph`, `active_graph=None`, `active_effective_graph=None`, `candidate_offsets_identified_by_data=False`, `active_offsets_identified_by_data=None`, `offsets_identified_in_objective=False`, `gauge_policy=''`, `messages=()` |
| `HardConstraintConflictTerm` | `constraint_index`, `site_i`, `site_j`, `relation`, `bound_value` |
| `HardConstraintConflict` | `component_nodes`, `cycle_nodes`, `terms`, `message`; property `constraint_indices` |
| `UnaccountedRealizedPair` | `site_i`, `site_j`, `realized_shifts`, `boundary_measure=None` |
| `ActiveSetIteration` | `iteration`, `n_active`, `n_realized`, `n_added`, `n_removed`, `rms_residual_all`, `max_residual_all`, `weight_step_norm`, `n_active_fit`, `fit_active_graph_n_components`, `fit_active_effective_graph_n_components`, `fit_active_offsets_identified_by_data`, `n_unaccounted_pairs` |
| `ActiveSetPathSummary` | `n_iterations`, `ever_fit_active_graph_disconnected`, `ever_fit_active_effective_graph_disconnected`, `ever_fit_active_offsets_unidentified_by_data`, `ever_unaccounted_pairs`, `max_fit_active_graph_components`, `max_fit_active_effective_graph_components`, `max_n_unaccounted_pairs`, `first_fit_active_graph_disconnected_iter`, `first_fit_active_effective_graph_disconnected_iter`, `first_unaccounted_pairs_iter` |

The public inverse dataclasses are generally frozen and slotted.
`PairBisectorConstraints`, `PowerFitProblem`, `PowerFitBounds`,
`PowerFitPredictions`, and `AlgebraicEdgeDiagnostics` copy their owned arrays
into read-only arrays. `PowerWeightFitResult`, realization diagnostics, and
active-set result containers do not deep-freeze every contained array; callers
must not infer deep immutability from the frozen outer dataclass.

The generated reference also documents these result/problem conveniences:

```text
PairBisectorConstraints.pair_labels(*, use_ids=False)
PairBisectorConstraints.to_records(*, use_ids=False)
PairBisectorConstraints.subset(mask)
PowerFitProblem.canonicalize_gauge(weights)
RealizedPairDiagnostics.to_records(constraints, *, use_ids=False)
RealizedPairDiagnostics.unaccounted_records(*, ids=None)
RealizedPairDiagnostics.to_report(constraints, *, use_ids=False)
PairConstraintDiagnostics.to_records(*, ids=None)
SelfConsistentPowerFitResult.to_records(*, use_ids=False)
SelfConsistentPowerFitResult.to_report(*, use_ids=False)
UnaccountedRealizedPair.to_record(*, ids=None)
```

### Current inverse record schemas

Record order follows constraint order. `use_ids=True` substitutes external site
labels where the relevant container has `ids`.

| Producer | Exact keys |
|---|---|
| `PairBisectorConstraints.to_records()` | `constraint_index`, `site_i`, `site_j`, `shift`, `target`, `confidence`, `measurement`, `distance`, `target_fraction`, `target_position`, `input_index`, `explicit_shift` |
| `PowerWeightFitResult.to_records(...)` | `constraint_index`, `site_i`, `site_j`, `shift`, `measurement`, `target`, `predicted`, `predicted_fraction`, `predicted_position`, `residual`, `alpha`, `beta`, `z_obs`, `z_fit`, `algebraic_residual`, `edge_weight` |
| `RealizedPairDiagnostics.to_records(...)` | `constraint_index`, `site_i`, `site_j`, `shift`, `realized`, `realized_same_shift`, `realized_other_shift`, `realized_shifts`, `endpoint_i_empty`, `endpoint_j_empty`, `boundary_measure` |
| `PairConstraintDiagnostics.to_records(...)` / active result | `constraint_index`, `site_i`, `site_j`, `shift`, `target`, `confidence`, `predicted`, `predicted_fraction`, `predicted_position`, `residual`, `active`, `realized`, `realized_same_shift`, `realized_other_shift`, `realized_shifts`, `endpoint_i_empty`, `endpoint_j_empty`, `boundary_measure`, `toggle_count`, `realized_toggle_count`, `first_realized_iter`, `last_realized_iter`, `marginal`, `status` |
| `HardConstraintConflictTerm.to_record()` | `constraint_index`, `site_i`, `site_j`, `relation`, `bound_value` |
| `UnaccountedRealizedPair.to_record()` | `site_i`, `site_j`, `realized_shifts`, `boundary_measure` |

Measurement-space `residual` and algebraic `z_obs - z_fit` are distinct.
Periodic shifts are integer tuples of the resolved dimension. A confidence-zero
row remains in candidate records but does not identify an effective graph edge.

### Current inverse report schemas

| Report | Exact top-level keys | Exact summary keys |
|---|---|---|
| fit | `kind`, `summary`, `constraints`, `fit_records`, `edge_diagnostics`, `objective_breakdown`, `weights`, `radii`, `weight_shift`, `used_shifts`, `warnings`, `conflict`, `connectivity` | `status`, `is_optimal`, `is_infeasible`, `hard_feasible`, `solver`, `measurement`, `n_constraints`, `n_points`, `converged`, `status_detail`, `n_iter`, `rms_residual`, `max_residual`, `conflicting_constraint_indices` |
| realized | `kind`, `summary`, `records`, `unrealized`, `unaccounted_pairs`, `warnings`, `tessellation_diagnostics` | `n_constraints`, `n_realized`, `n_same_shift`, `n_other_shift`, `n_unrealized`, `n_unaccounted_pairs` |
| active set | `kind`, `summary`, `constraints`, `fit`, `realized`, `diagnostics`, `marginal_records`, `history`, `path_summary`, `tessellation_diagnostics`, `warnings`, `connectivity` | `termination`, `converged`, `n_outer_iter`, `cycle_length`, `n_constraints`, `n_active_final`, `n_realized_final`, `rms_residual_all`, `max_residual_all`, `marginal_constraint_indices` |

Nested fit `edge_diagnostics` uses the fields of
`AlgebraicEdgeDiagnostics`; `objective_breakdown` uses the fields of
`PowerFitObjectiveBreakdown`. Connectivity records contain
`unconstrained_points`, candidate/effective/active graph records, both
data-identification flags, `offsets_identified_in_objective`, `gauge_policy`,
and `messages`. Graph records contain `n_points`, `n_constraints`, `n_edges`,
`isolated_points`, `connected_components`, `n_components`, and
`fully_connected`.

Active history rows contain `iteration`, `n_active`, `n_realized`, `n_added`,
`n_removed`, `rms_residual_all`, `max_residual_all`, `weight_step_norm`,
`n_active_fit`, the two fit-active graph component counts,
`fit_active_offsets_identified_by_data`, and `n_unaccounted_pairs`. Path summary
records contain all `ActiveSetPathSummary` fields. Tessellation report records
provide dimension-neutral measure/boundary keys plus the corresponding 2D
area/edge or 3D volume/face aliases.

### Calls exercised by repository examples

The source notebooks use the raw spatial forward calls and the historical
top-level inverse imports. In particular, the compatibility set needed to keep
the current notebooks reproducible is:

```text
Box, OrthorhombicCell, PeriodicCell, compute, locate, ghost_cells,
normalize_vertices, normalize_topology, annotate_face_properties,
resolve_pair_bisector_constraints, FitModel, SquaredLoss, Interval, FixedValue,
ExponentialBoundaryPenalty, fit_power_weights, match_realized_pairs,
ActiveSetOptions, solve_self_consistent_power_weights, dumps_report_json,
pyvoro2.viz3d.VizStyle, pyvoro2.viz3d.view_tessellation
```

They also call `to_records(...)`, `to_report(...)`, and conflict record helpers.
Notebook execution/output publication itself belongs to issue #20; this
baseline only records the callable paths that issue #12 must keep working while
the notebooks later migrate to preferred v0.7 imports.

No manuscript program or paper environment is stored in this repository. The
paper-style numerical regression subset is therefore deferred to issue #15,
while the documented v0.6.3 algebraic formulas, periodic-image semantics, and
top-level calls above remain the compatibility baseline. The unmilestoned
manuscript figure/reference work remains issue #19.

### Chemistry-neutral downstream requirements captured by the baseline

A chemvoro-shaped caller needs to be able to:

1. keep arbitrary downstream metadata outside pyvoro2 while passing stable
   integer external IDs;
2. resolve separator observations by those IDs and preserve explicit periodic
   image shifts;
3. fit weights, distinguish measurement and algebraic residuals, and inspect
   component-offset identification;
4. realize the fitted radii, detect empty endpoints and wrong/unaccounted
   realized pairs, and request boundary measure;
5. export records/reports without importing private modules; and
6. avoid relying on raw cell-list position as the mapping back to downstream
   objects.

These are requirements for the preferred v0.7 surfaces, not permission to add
chemistry-specific models or metadata containers to pyvoro2.

### Baseline reconciliation with accepted ADRs

The observed v0.6.3 baseline does not contradict ADR 0003, ADR 0004, or ADR
0005. It exposes transition conditions that those decisions already account
for:

- current power computation is radius-first and the conversion implementation
  is owned by `pyvoro2.powerfit`, while ADR 0002 and ADR 0004 require
  mathematical weight semantics and neutral transform ownership in v0.7;
- current separator implementation and broad imports live under
  `pyvoro2.powerfit` and top-level `pyvoro2`, while ADR 0004 deliberately keeps
  those paths as v0.7 compatibility shims during the ownership move; and
- current spatial and planar compute calls return different raw/structured
  variants, while ADR 0005 deliberately changes the preferred default and
  preserves the characterized variants through explicit compatibility routes.

These are planned migrations rather than incompatible scientific meanings. No
WP-01 stop condition was triggered, and dependent implementation may preserve
the recorded baseline through the compatibility policy without reopening an
accepted decision.

## Current v0.7 implementation status

The current development tree owns the two weight/radius conversion
implementations in the private neutral module `pyvoro2._weight_transforms`.
Top-level `pyvoro2`, `pyvoro2.powerfit`, and
`pyvoro2.powerfit.transforms` expose the same function objects, with the latter
two retained as historical compatibility routes. Separator implementation code
imports the neutral module directly. Issue #7 changed ownership without
changing behavior. Issue #21 then hardened the shared numerical contract:
`r_min` must be finite and non-negative, and any non-finite intermediate or
result from squaring or applying the representation shift raises `ValueError`.
Signatures, defaults, global-shift behavior, and valid finite representable
results for those helpers remain the characterized v0.6.3 behavior above.

Issue #8 adds the keyword-only `weights=None` argument immediately before the
existing `radii=None` argument on `pyvoro2.compute(...)` and
`pyvoro2.planar.compute(...)`. In power mode exactly one representation is
required. Weights must have shape `(n,)`, must be finite, and may be positive,
zero, or negative; the required global shift and converted representation must
also remain finite and representable. They are converted with the default
`weights_to_radii(weights)` policy, which applies one common global
representation shift. Non-finite input, conversion overflow, supplying both
representations, supplying neither, or supplying weights in standard mode
raises `ValueError` before native tessellation. Standard mode also rejects every
non-`None` `radii=` argument before native tessellation rather than preserving
the v0.6.3 behavior of silently ignoring it. Valid radius-based power
computation remains numerically unchanged. Finite representability is
necessary for conversion but does not guarantee a numerically resolvable native
tessellation. Voro++ uses binary64 squared-radius arithmetic, so very large
absolute backend ``radii**2`` values or genuine weight ranges relative to
squared coordinate/domain scales can lose geometric resolution. No universal
safe cutoff is promised; sensitivity depends on scale, geometry, platform, and
compiler, especially for periodic power tessellations.

The private dimension-neutral `pyvoro2._power_input` resolution path keeps the
validated input weights, resolved backend radii, and representation shift
together for later `TessellationResult` construction. It imports only the
neutral transform and input-validation helpers. The resolved radii feed native
2D, 3D box/orthorhombic, and 3D triclinic power calls, as well as periodic
edge/face shift inference. `locate(...)` and `ghost_cells(...)` signatures are
unchanged.

Issues #9 and #10 implement and wire the public structured result. Both
`compute(...)` functions expose keyword-only `output='result'|'cells'` and
return `TessellationResult` by default. The shared builder receives the final
user-visible raw cells and the exact resolved power input without repeating
native computation, diagnostics, normalization, or annotation. The explicit
raw route preserves the factual v0.6.3 list/tuple behavior above.
`PlanarComputeResult` is the identical class object as `TessellationResult`.
Explicit planar `return_result=` use emits `DeprecationWarning` and is resolved
through the compatibility matrix below. Canonical inverse ownership remains
assigned to later issues; this status note does not alter the factual v0.6.3
baseline.

## Accepted v0.7 contract decisions

The following boundaries are already accepted:

- `pyvoro2.inverse` is the canonical inverse namespace;
- separator implementation is owned by `pyvoro2.inverse.separator`;
- `pyvoro2.powerfit` and broad top-level separator exports are
  compatibility-only for v0.7, with planned removal in v0.8;
- both forward `compute(...)` functions return
  `pyvoro2.TessellationResult` by default;
- `output='cells'` is the explicit raw compatibility route;
- `PlanarComputeResult` is a compatibility alias during v0.7;
- deep immutability of nested raw records is not part of the v0.7 contract.

See [ADR 0004](decisions/0004-canonical-inverse-namespace.md) and
[ADR 0005](decisions/0005-tessellation-result-contract.md).

## Lifecycle summary for the preferred v0.7 API

| Surface | Intended status | Notes |
|---|---|---|
| Domain classes and domain geometry semantics | Stable | Mature bounded and periodic behavior; capability differences remain explicit by dimension. |
| `pyvoro2.compute` and `pyvoro2.planar.compute` | Stable | Direct weight/radius behavior, the common structured default, and explicit raw compatibility output are implemented and tested. |
| `weights=` and `radii=` mathematical meaning | Stable | Mode-specific rejection/exclusivity, one global representation shift, finite and representable conversion, and empty-cell behavior are part of the contract. |
| `pyvoro2.TessellationResult` core contract | Stable candidate | The shared class, private construction path, and both public compute integrations are implemented by issues #9 and #10. |
| Detailed optional result conveniences and raw geometry views | Provisional | Refine through implementation and chemvoro-shaped validation. |
| `pyvoro2.inverse` preferred high-level separator workflow | Stable candidate | Main observations/fit entry point should be suitable for chemvoro at release. |
| `pyvoro2.inverse.separator` advanced problem and operator views | Provisional | Public for research use, but likely to evolve before prescribed measures and mixed problems. |
| Realization-aware active-set API | Experimental | Practical outer algorithm; no universal convergence claim. |
| Optional sparse quadratic backend | Experimental if shipped | Backend selection and performance policy remain conditional. |
| `pyvoro2.powerfit` | Compatibility-only and deprecated | One-way shim during v0.7; planned removal in v0.8. |
| Broad separator-specific exports from top-level `pyvoro2` | Compatibility-only and deprecated | New code imports from `pyvoro2.inverse`; planned removal in v0.8. |
| Native extension and solver-internal modules | Internal | No compatibility guarantee. |

“Stable candidate” means that the release intends to make the surface stable,
but final approval occurs only after implementation, tests, documentation, and
downstream validation are complete.

## Spatial forward namespace: `pyvoro2`

### Stable baseline and v0.7 candidates

| Group | Names / behavior | v0.7 intent |
|---|---|---|
| Domains | `Box`, `OrthorhombicCell`, `PeriodicCell` | Stable |
| Operations | `compute`, `locate`, `ghost_cells` | Stable |
| Structured result | `TessellationResult` | Stable candidate |
| Weight transforms | `weights_to_radii`, `radii_to_weights` | Stable; neutral implementation requires finite inputs and finite representable results |
| Tessellation diagnostics | `TessellationDiagnostics`, `TessellationIssue`, `TessellationError`, `analyze_tessellation`, `validate_tessellation` | Stable unless the baseline audit identifies undocumented schema details |
| Duplicate handling | `DuplicatePair`, `DuplicateError`, `duplicate_check` | Stable |
| Geometry annotations | `annotate_face_properties` | Stable candidate |
| Normalization | `NormalizedVertices`, `NormalizedTopology`, normalization and validation helpers | Stable or provisional per detailed baseline audit |
| Package metadata | `__version__`, `planar` | Stable |

### Compatibility-only top-level inverse names

The following v0.6.3 exports remain available during v0.7 but are not preferred
for new code:

```text
PairBisectorConstraints
resolve_pair_bisector_constraints
SquaredLoss
HuberLoss
Interval
FixedValue
SoftIntervalPenalty
ExponentialBoundaryPenalty
ReciprocalBoundaryPenalty
L2Regularization
FitModel
AlgebraicEdgeDiagnostics
ConstraintGraphDiagnostics
ConnectivityDiagnostics
ConnectivityDiagnosticsError
HardConstraintConflictTerm
HardConstraintConflict
PowerWeightFitResult
RealizedPairDiagnostics
UnaccountedRealizedPair
UnaccountedRealizedPairError
build_fit_report
build_realized_report
build_active_set_report
dumps_report_json
write_report_json
ActiveSetOptions
ActiveSetIteration
ActiveSetPathSummary
PairConstraintDiagnostics
SelfConsistentPowerFitResult
fit_power_weights
match_realized_pairs
solve_self_consistent_power_weights
```

Planned removal: v0.8, unless a later accepted release decision extends the
transition.

## Planar namespace: `pyvoro2.planar`

| Group | Names / behavior | v0.7 intent |
|---|---|---|
| Domains | `Box`, `RectangularCell` | Stable |
| Operations | `compute`, `locate`, `ghost_cells` | Stable |
| Structured result | `TessellationResult` re-export | Stable candidate |
| Historical result name | `PlanarComputeResult` | Compatibility-only identity alias to `TessellationResult`, with removal or reconsideration planned for v0.8 |
| Diagnostics and validation | Planar tessellation and normalization diagnostics | Stable unless the baseline audit identifies schema details needing provisional status |
| Duplicate handling and annotations | `duplicate_check`, `annotate_edge_properties` | Stable candidates |
| Normalization | Planar normalization helpers and result objects | Stable or provisional per detailed baseline audit |
| Visualization | `plot_tessellation` | Provisional optional convenience |

## Canonical inverse namespace: `pyvoro2.inverse`

### Preferred high-level separator API

The exact implemented signatures are finalized during the namespace issue. The
accepted preferred names are:

| Name | Intended status | Meaning |
|---|---|---|
| `SeparatorObservations` | Stable candidate | Resolved pairwise separator observations, including periodic image labels and confidence. |
| `resolve_separator_observations` | Stable candidate | Validate and resolve raw separator observations against sites and domain. |
| `SeparatorFitResult` | Stable candidate | Fitted state, observation residuals, identification metadata, and solver status. |
| `fit_weights_from_separators` | Stable candidate | Preferred fixed-observation fit entry point. |
| `weights_to_radii`, `radii_to_weights` | Stable re-export where useful | Same neutral transforms as top-level pyvoro2. |

### Advanced separator API

Expected public but initially provisional surfaces include:

- `SeparatorFitProblem` and problem-building/evaluation helpers;
- objective model pieces such as squared/Huber losses, hard intervals,
  penalties, and regularization;
- graph, connectivity, incidence, Laplacian, and objective-breakdown views;
- result packaging for externally computed weights;
- realization matching and record/report builders.

The active-set outer workflow and its path/result types remain **experimental**.

The implementation issue must produce an exact old-to-new name map. Historical
names under `pyvoro2.powerfit` may be aliases to canonical classes or wrappers,
but numerical implementations must not be duplicated.

## Forward return contract

### Implemented common data contract

Issues #9 and #10 implement one frozen, slotted `TessellationResult` class and export
the identical class object as both `pyvoro2.TessellationResult` and
`pyvoro2.planar.TessellationResult`. Its private shared builder aligns cells by
final external ID, represents omitted empty cells explicitly in aligned
arrays, and does not invoke native computation, diagnostics, normalization, or
boundary annotation.

The stable-candidate fields are exact:

| Field | Lifecycle | Semantics |
|---|---|---|
| `dimension` | Stable candidate | Explicit `2` or `3`. |
| `domain` | Stable candidate | Validated domain used by the computation. |
| `mode` | Stable candidate | `"standard"` or `"power"`. |
| `sites` | Stable candidate | Read-only owned `(n, dimension)` copy of validated input coordinates in original input order. |
| `ids` | Stable candidate | Read-only owned `(n,)` external-ID array in original input order; omitted IDs become `np.arange(n, dtype=np.int64)`. |
| `cells` | Stable candidate | Exact supplied raw-cell list after ID remapping; the list, dictionaries, and nested records are not copied or frozen. |
| `cell_measures` | Stable candidate | Read-only owned `(n,)` construction-time snapshot of areas or volumes aligned with input order; hidden cells are zero. |
| `empty_mask` | Stable candidate | Read-only owned boolean `(n,)` construction-time snapshot aligned with input order, including raw records omitted by `include_empty=False`. |
| `input_weights` | Stable candidate | Read-only owned mathematical input weights for weight-first power input; otherwise `None`. |
| `backend_radii` | Stable candidate | Read-only owned exact native power radii; `None` in standard mode. |
| `representation_shift` | Stable candidate | Finite common additive shift for weight-first conversion; `None` for standard or direct-radius input. |
| `tessellation_diagnostics` | Stable candidate | Existing dimension-specific diagnostics when computed; otherwise `None`. |
| `normalized_vertices` | Stable candidate | Existing dimension-specific vertex normalization when computed; otherwise `None`. |
| `normalized_topology` | Stable candidate | Existing dimension-specific topology normalization when computed; otherwise `None`. |

The following small convenience surface is **provisional** while downstream
integration validates the exact access vocabulary:

| Convenience | Lifecycle | Semantics |
|---|---|---|
| `measure_kind` | Provisional | `"area"` in 2D or `"volume"` in 3D. |
| `boundary_kind` | Provisional | `"edges"` in 2D or `"faces"` in 3D. |
| `has_tessellation_diagnostics`, `has_normalized_vertices`, `has_normalized_topology` | Provisional | Distinguish absent optional objects from present objects. |
| `has_boundaries`, `has_periodic_shifts` | Provisional | Report explicit builder capabilities, including available-but-empty geometry. |
| `require_tessellation_diagnostics()`, `require_normalized_vertices()`, `require_normalized_topology()` | Provisional | Return optional objects or raise a clear `ValueError`. |
| `require_boundaries()` | Provisional | Return input-order-aligned edge/face collections, using an empty collection for hidden sites, or raise when boundaries were unavailable. |
| `global_vertices`, `global_edges` | Provisional compatibility conveniences | Forward to available planar normalized objects, preserving the historical `PlanarComputeResult` access pattern; otherwise `None`. |

The outer object prevents field replacement. Its aligned arrays are copies and
are non-writeable, so construction never marks caller-owned arrays read-only.
The raw `cells` list and its nested dictionaries remain shared and mutable by
design. Later raw-record mutation does not update the `cell_measures` or
`empty_mask` snapshots. Boundary access revalidates mutable boundary record
types, current empty flags, required non-empty records, and periodic-shift
fields and raises if mutation made them inconsistent with the recorded
snapshots or capabilities. An empty cell cannot contain realized edge or face
records; both omitted and explicitly empty boundary collections remain valid.

Direct dataclass construction is supported and validates raw IDs, measures,
empty state, representation metadata, and capability metadata against the
aligned fields. Weight-first metadata must satisfy the shared exact
weight/shift-to-radius transform. Boundary and periodic-shift availability are
private keyword-only construction state supplied by the shared builder;
keeping them as normal dataclass initialization fields preserves them through
`dataclasses.replace()` without adding stable public result fields. Deep copies
and pickle round trips preserve the exact existing snapshot state rather than
revalidating it against later permitted raw-record mutation; reconstructed
arrays remain owned and read-only, and capability state is preserved.

### Preferred compute route

```python
result = pyvoro2.compute(..., output='result')
result = pyvoro2.planar.compute(..., output='result')
```

Omitting `output` is equivalent to `output='result'`. Structured output is
always one `TessellationResult`, never a tuple. Diagnostics computed because of
`return_diagnostics=True` or `tessellation_check='diagnose'|'warn'|'raise'`
are stored in `result.tessellation_diagnostics`.

### Raw compatibility route

```python
cells = pyvoro2.compute(..., output='cells')
cells = pyvoro2.planar.compute(..., output='cells')
```

This route preserves the established list/tuple behavior. Without
`return_diagnostics=True` it returns only the raw list, including when a
tessellation check computed diagnostics internally. With
`return_diagnostics=True` it returns `(cells, diagnostics)`. Raw record schemas,
ordering, external IDs, requested geometry, and numerical behavior remain the
characterized baseline.

### Planar compatibility selector matrix

The public compatibility parameter is `return_result: bool | None = None`.
`None` means that the selector was omitted and follows the `output=` contract.
Passing either boolean emits `DeprecationWarning`; `output=` is the replacement.

| Planar selection | Result |
|---|---|
| both selectors omitted | `TessellationResult` |
| `return_result=None` | same as omitted; no deprecation warning |
| `output='result'` | `TessellationResult` |
| `output='cells'` | historical raw list/tuple route |
| `return_result=True`, `output` omitted | `TessellationResult` |
| explicit `return_result=False`, `output` omitted, no normalization | historical raw list/tuple route |
| explicit `return_result=False`, `output` omitted, normalization requested | `TessellationResult`, preserving the historical normalization override |
| equivalent explicit `output` and `return_result` | requested route, plus warning |
| conflicting explicit `output` and `return_result` | `ValueError`, plus warning |
| explicit `output='cells'` with normalization | `ValueError` |

`PlanarComputeResult` from both `pyvoro2.planar` and
`pyvoro2.planar.result` is an identity alias to `pyvoro2.TessellationResult`.

## Scientifically meaningful semantics to inventory explicitly

The following are API even when no dedicated Python class represents them:

- coordinate units are caller-defined but consistent within one computation;
- power weights have squared-coordinate units;
- positive, zero, and negative finite power weights are valid when the global
  shift and converted representation remain finite and representable;
- non-finite weight input or conversion overflow raises `ValueError` before
  native computation;
- finite representability does not guarantee a numerically resolvable native
  tessellation; Voro++ uses binary64 squared-radius arithmetic, so very large
  absolute backend ``radii**2`` values or genuine weight ranges relative to
  squared coordinate/domain scales can lose geometric resolution. No universal
  safe cutoff is promised, and periodic power tessellations are particularly
  sensitive;
- backend radii have coordinate units;
- one global additive weight shift leaves the complete power diagram unchanged;
- power-mode `compute(...)` requires exactly one of `weights=` or `radii=`,
  while standard mode rejects both arguments;
- valid radius-based power computation remains numerically unchanged;
- direct `weights=` input currently belongs to the two `compute(...)`
  functions, not to every forward operation;
- backend radii are a shifted representation and are not unique physical
  radii;
- disconnected separator-observation components have additional unidentified
  offsets that may change global realization;
- external IDs remain attached to original sites;
- periodic neighbor shifts identify the realized image and are not silently
  replaced by a nearest image;
- zero-confidence separator rows do not identify weight differences;
- algebraic fit does not imply realized-boundary support;
- empty/hidden cells are represented deterministically according to the chosen
  output route;
- error/status behavior for infeasibility and wrong-image realization is part of
  the public scientific contract.

## Deprecation and removal schedule

| Surface | v0.7 | Planned v0.8 action |
|---|---|---|
| `pyvoro2.powerfit` | Compatibility-only; optional `DeprecationWarning`; migration docs | Remove unless an explicit release decision extends it |
| Broad top-level separator exports | Compatibility-only; migration docs | Remove from top-level |
| `PairBisectorConstraints` and other historical names | Compatibility aliases | Remove or retain only if justified by real usage and documented explicitly |
| `PlanarComputeResult` | Compatibility alias to `TessellationResult` | Remove or reconsider after v0.7 downstream feedback |
| Raw cell return | Available through `output='cells'` | Continue as explicit route unless a later decision removes it |
| Planar `return_result=` | Compatibility-only | Remove after migration to `output=` |

## Final release review checklist

- [ ] Every preferred public import is listed with a lifecycle category.
- [ ] Every compatibility alias has a replacement and removal horizon.
- [ ] `__all__` matches the intended namespace policy.
- [x] Forward output modes and diagnostic combinations are characterized.
- [ ] Stable `TessellationResult` fields and mutable contained values are
      documented.
- [ ] Raw record keys used by compatibility tests are listed or referenced.
- [ ] Preferred separator names and exact historical aliases are complete.
- [ ] Active-set and optional sparse behavior are labelled experimental.
- [ ] Default changes and scientific semantics appear in migration notes and
      release notes.
- [ ] The chemvoro-shaped integration workflow uses only stable or deliberately
      provisional public surfaces.
- [ ] The maintainer approves the final inventory before the release candidate.
