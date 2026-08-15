# v0.8 public API inventory

- **Status:** Finalized against the v0.8 tree on 2026-07-24; release
  qualification remains in issue #33
- **Historical baseline:** v0.6.3
- **Previous contract:** v0.7.0
- **Target:** v0.8.0
- **v0.8 audit:** [issue #32](https://github.com/DeloneCommons/pyvoro2/issues/32)
- **Policy:** [API lifecycle and compatibility](api-lifecycle.md)
- **Plan:** [active v0.8 development plan](plans/v0.8.md)
- **Decisions:** [ADR 0004](decisions/0004-canonical-inverse-namespace.md),
  [ADR 0005](decisions/0005-tessellation-result-contract.md),
  [ADR 0006](decisions/0006-v0.8-cleanup-release.md),
  [ADR 0007](decisions/0007-separator-objective-contract.md),
  [ADR 0008](decisions/0008-separator-solver-and-linear-backend.md),
  [ADR 0009](decisions/0009-certified-scalar-proximal-solver.md),
  [ADR 0010](decisions/0010-native-construction-preconditions.md),
  [ADR 0011](decisions/0011-strict-input-and-ownership-contract.md),
  [ADR 0012](decisions/0012-certified-periodic-image-geometry.md),
  [ADR 0013](decisions/0013-central-generator-preparation-and-backend-safety.md), and
  [ADR 0014](decisions/0014-separator-observation-and-source-identity.md)

This inventory is the authoritative v0.8 lifecycle contract for public imports,
return routes, record schemas, defaults, and scientific semantics. It has been
checked against the current source, tests, documentation, executed notebooks,
distribution configuration, GitHub Actions workflows, and downstream-shaped
regression assets. Release qualification must verify that packaged artifacts
preserve this contract.

The historical v0.6.3 baseline is retained below because it explains the v0.7
migration. It is not a list of current imports. Current v0.8 exports,
signatures, lifecycle classifications, internal boundaries, and removals begin
at [Current v0.8 contract](#current-v08-contract).

## How to maintain this inventory

For every issue that changes public behavior:

1. update the relevant row or section in the same change;
2. distinguish historical behavior from the current implemented state;
3. record aliases, deprecations, and their removal releases;
4. include defaults, result fields, record keys, units, and periodic conventions
   when they carry scientific meaning;
5. leave uncertain new surfaces **provisional** rather than omitting them;
6. do not mark a surface **stable** until its tests and documentation define the
   contract clearly.

Issue #32 finalizes the v0.8 classifications below. Release review must verify that
`__all__`, docstrings, guides, reference pages, migration notes, and this
inventory remain synchronized.

## Factual v0.6.3 baseline

This section records behavior observed in the v0.6.3 source tree before any
v0.7 public implementation. It was checked against package `__all__` values,
call signatures, generated API reference pages, user guides, source notebooks,
and the existing tests. The surviving canonical contracts derived from those
characterizations live in
`tests/forward/common/test_forward_api_contract.py` and
`tests/inverse/separator/test_api_contract.py`. Assertions that existed only
for the compatibility routes removed by issue #28 are not retained.

“Baseline” does not make every current convenience stable forever. It identifies
what the v0.7 compatibility routes must preserve deliberately and prevents a
module move or result-default change from silently changing established
behavior.

### v0.6.3 public namespaces, documented module routes, and `__all__`

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
| Historical inverse surface | the exact 34-name list under [Historical v0.7 top-level inverse compatibility set](#historical-v07-top-level-inverse-compatibility-set) |

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
did not broaden the historical top-level surface while preserving the old
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

### v0.6.3 forward signatures and defaults

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

All six current forward operations prepare inserted generators centrally.
Non-periodic coordinates use the half-open interval `[lo, hi)` and periodic
coordinates are remapped to their primary representation before native
dispatch. This includes each temporary `ghost_cells` query; `locate` queries
themselves are not inserted and retain their existing query semantics.

The frozen backend-safety floor is squared distance `1e-10` (distance `1e-5`),
inclusive. It is always active, uses certified minimum-image geometry for
periodic pairs, and is not controlled by the public duplicate mode, threshold,
wrap flag, or pair-report limit. The existing `off`/`warn`/`raise` policy
applies only to safe pairs strictly below a user threshold above `1e-5`;
`duplicate_wrap=False` changes only that optional metric.

`DuplicatePair(i, j, distance)` is unchanged. `DuplicateError` remains a
`ValueError` with compatible positional `args`, `.pairs`, `.threshold`, and
string behavior. It additionally exposes `kind`, `safety_distance_squared`,
`safety_distance`, `user_threshold`, `minimum_image_used`,
`optional_wrap_used`, `truncated`, `operation`, and `external_ids`. Mandatory
errors use `.threshold == 1e-5`; optional errors retain the configured user
threshold.

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

### v0.6.3 visualization signatures and defaults

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

### Raw record schemas and ordering retained in v0.8

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

### Forward diagnostic fields retained in v0.8

| Type | Exact dataclass fields |
|---|---|
| spatial `TessellationIssue` | `code`, `severity`, `message`, `examples` |
| spatial `TessellationDiagnostics` | `domain_volume`, `sum_cell_volume`, `volume_ratio`, `volume_gap`, `volume_overlap`, `n_sites_expected`, `n_cells_returned`, `missing_ids`, `empty_ids`, `face_shift_available`, `reciprocity_checked`, `n_faces_total`, `n_faces_orphan`, `n_faces_mismatched`, `issues`, `ok_volume`, `ok_reciprocity`, `ok` |
| planar `TessellationIssue` | `code`, `severity`, `message`, `examples` |
| planar `TessellationDiagnostics` | `domain_area`, `sum_cell_area`, `area_ratio`, `area_gap`, `area_overlap`, `n_sites_expected`, `n_cells_returned`, `missing_ids`, `empty_ids`, `edge_shift_available`, `reciprocity_checked`, `n_edges_total`, `n_edges_orphan`, `n_edges_mismatched`, `issues`, `ok_area`, `ok_reciprocity`, `ok` |
| spatial `NormalizationDiagnostics` | `n_cells`, `n_global_vertices`, `n_global_edges`, `n_global_faces`, `is_periodic_domain`, `fully_periodic_domain`, `has_wall_faces`, `n_vertex_face_shift_mismatch`, `n_face_vertex_set_mismatch`, `n_vertices_low_incidence`, `n_edges_low_incidence`, `n_cells_bad_euler`, `issues`, `ok_vertex_face_shift`, `ok_face_vertex_sets`, `ok_incidence`, `ok_euler`, `ok` |
| planar `NormalizationDiagnostics` | `n_cells`, `n_global_vertices`, `n_global_edges`, `is_periodic_domain`, `fully_periodic_domain`, `has_wall_edges`, `n_vertex_edge_shift_mismatch`, `n_edge_vertex_set_mismatch`, `n_vertices_low_incidence`, `n_cells_bad_polygon`, `issues`, `ok_vertex_edge_shift`, `ok_edge_vertex_sets`, `ok_incidence`, `ok_polygon`, `ok` |

### v0.6.3 inverse signatures and constructor defaults

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

### Inverse result fields retained under canonical v0.8 names

The primary containers have the following exact dataclass fields. The
parenthetical names are historical v0.6.3/v0.7 identities and are absent from
the v0.8 namespace:

| Type | Fields |
|---|---|
| `SeparatorObservations` (historical `PairBisectorConstraints`) | `n_points`, `i`, `j`, `shifts`, `target`, `confidence`, `measurement`, `distance`, `distance2`, `delta`, `target_fraction`, `target_position`, `input_index`, `explicit_shift`, `ids`, `warnings` |
| `SeparatorFitProblem` (historical `PowerFitProblem`) | `constraints`, `model`, `alpha`, `beta`, `z_obs`, `edge_weight`, `regularization_strength`, `regularization_reference`, `offset_identifying_constraint_mask`, `bounds`, `connectivity`, `hard_feasible`, `hard_conflict` |
| `SeparatorFitResult` (historical `PowerWeightFitResult`) | `status`, `hard_feasible`, `weights`, `radii`, `weight_shift`, `measurement`, `target`, `predicted`, `predicted_fraction`, `predicted_position`, `residuals`, `rms_residual`, `max_residual`, `used_shifts`, `solver`, `n_iter`, `converged`, `conflict`, `warnings`, `linear_backend`, `status_detail`, `connectivity`, `edge_diagnostics`, `objective_breakdown` |
| `RealizedPairDiagnostics` | `realized`, `unrealized`, `realized_same_shift`, `realized_other_shift`, `realized_shifts`, `endpoint_i_empty`, `endpoint_j_empty`, `boundary_measure`, `cells`, `tessellation_diagnostics`, `unaccounted_pairs`, `warnings` |
| `PairConstraintDiagnostics` | `site_i`, `site_j`, `shift`, `target`, `confidence`, `predicted`, `predicted_fraction`, `predicted_position`, `residuals`, `active`, `realized`, `realized_same_shift`, `realized_other_shift`, `realized_shifts`, `endpoint_i_empty`, `endpoint_j_empty`, `boundary_measure`, `toggle_count`, `realized_toggle_count`, `first_realized_iter`, `last_realized_iter`, `marginal`, `status` |
| `SelfConsistentPowerFitResult` | `constraints`, `fit`, `realized`, `diagnostics`, `active_mask`, `n_outer_iter`, `converged`, `termination`, `cycle_length`, `marginal_constraints`, `rms_residual_all`, `max_residual_all`, `tessellation_diagnostics`, `history`, `path_summary`, `warnings`, `connectivity` |

For the experimental active-set result, `realized`, `diagnostics`,
`rms_residual_all`, and `max_residual_all` are optional. They are all present
for one coherent final weighted state and all `None` when the final accepted
fit has no usable weights. `tessellation_diagnostics` remains optional even for
an available state because its analysis is requested separately. The additive
computed properties `final_state_available`,
`final_state_unavailable_reason`, and `final_refit_converged` expose final-layer
availability and final inner-fit convergence without adding stored dataclass
fields. Outer `converged` remains true exactly for
`termination == 'self_consistent'`.

`SeparatorFitProblem.offset_identifying_constraint_mask` retains its historical
field name. It is the model-coupling mask used to decompose solver subproblems:
positive-confidence rows are
included, and hard restrictions or positive-strength penalties make their
affected rows part of the same numerical subproblem. Zero-strength penalties
are mathematically absent and do not affect this mask. It is not the
informative observation mask and does not claim data identification or unique
objective selection.

Supporting fields are exact as follows:

| Type | Fields |
|---|---|
| `PowerFitBounds` | `measurement_lower`, `measurement_upper`, `difference_lower`, `difference_upper` |
| `PowerFitPredictions` | `difference`, `fraction`, `position`, `measurement` |
| `PowerFitObjectiveBreakdown` | `total`, `mismatch`, `penalties_total`, `penalty_terms`, `regularization`, `hard_constraints_satisfied`, `hard_max_violation`, `hard_max_tolerance` |
| `SeparatorSolverTerminationView` | `status`, `status_detail`, `solver`, `linear_backend`, `n_iter`, `converged`, `hard_feasible`, `conflict`, `warnings` |
| `AlgebraicEdgeDiagnostics` | `alpha`, `beta`, `z_obs`, `z_fit`, `residual`, `edge_weight`, `weighted_l2`, `weighted_rmse`, `rmse`, `mae` |
| `ConstraintGraphDiagnostics` | `n_points`, `n_constraints`, `n_edges`, `isolated_points`, `connected_components`, `fully_connected`; property `n_components` |
| `ConnectivityDiagnostics` | `unconstrained_points`, `candidate_graph`, `effective_graph`, `active_graph=None`, `active_effective_graph=None`, `candidate_offsets_identified_by_data=False`, `active_offsets_identified_by_data=None`, `offsets_identified_in_objective=False`, `gauge_policy=''`, `messages=()` |
| `HardConstraintConflictTerm` | `constraint_index`, `site_i`, `site_j`, `relation`, `bound_value` |
| `HardConstraintConflict` | `component_nodes`, `cycle_nodes`, `terms`, `message`; property `constraint_indices` |
| `UnaccountedRealizedPair` | `site_i`, `site_j`, `realized_shifts`, `boundary_measure=None` |
| `ActiveSetIteration` | `iteration`, `n_active`, `n_realized`, `n_added`, `n_removed`, `rms_residual_all`, `max_residual_all`, `weight_step_norm`, `n_active_fit`, `fit_active_graph_n_components`, `fit_active_effective_graph_n_components`, `fit_active_offsets_identified_by_data`, `n_unaccounted_pairs` |
| `ActiveSetPathSummary` | `n_iterations`, `ever_fit_active_graph_disconnected`, `ever_fit_active_effective_graph_disconnected`, `ever_fit_active_offsets_unidentified_by_data`, `ever_unaccounted_pairs`, `max_fit_active_graph_components`, `max_fit_active_effective_graph_components`, `max_n_unaccounted_pairs`, `first_fit_active_graph_disconnected_iter`, `first_fit_active_effective_graph_disconnected_iter`, `first_unaccounted_pairs_iter` |

The public inverse dataclasses are generally frozen and slotted.
`SeparatorObservations`, `SeparatorFitProblem`, `PowerFitBounds`,
`PowerFitPredictions`, and `AlgebraicEdgeDiagnostics` copy their owned arrays
into read-only arrays. `SeparatorFitResult`, realization diagnostics, and
active-set result containers do not deep-freeze every contained array; callers
must not infer deep immutability from the frozen outer dataclass.

`SeparatorObservations` retains the exact public field list above and its
existing public constructor signature. Direct construction validates an exact
dimension of two or three, point count, integer endpoint/shift/input-index
categories and ranges, aligned row shapes, distinct endpoints, unique
non-negative input indices, finite non-negative confidence, finite nonzero
connector geometry, IDs, and warnings. It recomputes distance values from
`delta` and both measurement forms from the canonical target and distance.
Finite redundant values are accepted only under
`np.allclose(..., rtol=8*np.finfo(np.float64).eps, atol=0.0)` and are replaced
by the recomputed binary64 values. This tolerance checks internal constructor
consistency; it is not source equivalence.

Issue #13 preserves those exact dataclass fields and adds the following
provisional, non-copying access paths:

| Layer | Access path | Existing data exposed |
|---|---|---|
| Fitted state | `SeparatorFitResult.state` | `weights` as `mathematical_weights`, backend `radii`, and compatibility `weight_shift` as `global_representation_shift` |
| Identification | `SeparatorFitResult.identification` | informative positive-confidence components, global-gauge identification (`False` for separator data), observational component-offset identification, conservative objective selection, component-alignment policy, sites isolated in the informative graph as `unconstrained_sites`, and `connectivity` (whose compatibility `unconstrained_points` remains candidate-based) |
| Observations | `SeparatorFitResult.observation_view(observations)` | measurement targets, confidence from the supplied resolved observations, predictions in all existing forms, residuals/summaries, and requested shifts; the supplied set must satisfy the exact observation/source association policy below |
| Objective | `SeparatorFitResult.objective` | existing `objective_breakdown` object |
| Algebraic diagnostics | `SeparatorFitResult.algebraic` | existing `edge_diagnostics` and `connectivity` objects; no graph-operator representation |
| Fixed solver termination | `SeparatorFitResult.solver_termination` | status/detail, solver method, linear backend, iterations, convergence, hard feasibility, conflict, and warnings |
| Requested-image matching | `RealizedPairDiagnostics.requested_image_matching` | any/same-shift/other-shift realization, realized shifts, and unrealized indices |
| Realized geometry | `RealizedPairDiagnostics.geometry` | empty endpoints, optional boundary measure/cells/tessellation diagnostics, unaccounted pairs, and warnings |
| Active-set organization | `SelfConsistentPowerFitResult.inner_fit`, `.final_realization`, `.candidate_diagnostics`, `.outer_termination`, `.path` | final objects when available, outer termination, active mask, marginals, history, and path summary |

Issue #14 preserves the exact `SeparatorFitProblem` dataclass fields and adds
two provisional computed properties:

| Mathematical layer | Access path | Contract |
|---|---|---|
| Observation multigraph | `SeparatorFitProblem.observation_graph` | `SeparatorObservationGraphView` with site count, distinct observation rows, oriented endpoints, input indices, requested shifts, shared `alpha`, `beta`, `z_obs`, and `rho` arrays, a positive-confidence informative mask, existing connectivity, and dense/optional-SciPy incidence conversion |
| Quadratic normal operator | `SeparatorFitProblem.quadratic_operator` | `SeparatorQuadraticOperatorView` with matrix-free and dense/optional-SciPy observation Laplacian and L2-regularized normal operators, scale-safe direct `observation_rhs = B @ q` for `q_r = confidence_r * alpha_r * (target_r - beta_r)`, `regularized_normal_rhs`, regularization data, hard-bound metadata, and component/nullity interpretation |

The incidence matrix has shape `(n_sites, n_observations)` and column `r`
equal to `+1` at `site_i[r]` and `-1` at `site_j[r]`. Every resolved row is a
column, including repeats, periodic parallel observations, and zero-confidence
rows. The latter have `informative_mask[r] == False` and `rho[r] == 0`, so they
do not connect informative components or contribute to the observation
Laplacian and right-hand side. `z_obs` remains diagnostic and is not required
to reconstruct a finite normal RHS.

The quadratic view is available only for `SquaredLoss` with no
positive-strength scalar penalties. Zero-strength penalties are absent and do
not hide the view. Optional L2 regularization is included exactly. Hard
interval or equality restrictions may coexist but remain visible through
`problem.bounds`; the view reports that unconstrained normal equations do not
characterize a constrained fit in general. Huber mismatch and models with
positive-strength scalar penalties retain the graph view but reject
`quadratic_operator` rather than presenting a partial system as the full
objective. Sparse conversion imports SciPy lazily; SciPy is neither a runtime
dependency nor a solver backend in issue #14.

The canonical `component_alignment_policy` view value is the same stored string
as compatibility-facing `ConnectivityDiagnostics.gauge_policy`; only the access
name is clarified. `global_representation_shift` is the compatibility
`weight_shift` value used for backend-radius conversion: it selects a backend
representation within the global geometric gauge, is distinct from independent
component offsets, and is not information recovered from observations. All
array-valued views share the arrays already owned by their result or
resolved-observation source.

The connectivity `effective_graph` and `active_effective_graph` contain only
positive-confidence rows. The corresponding `*_identified_by_data` fields are
true exactly when their informative graph is connected. The compatibility
field `offsets_identified_in_objective` is conservative: it is true when the
relevant informative graph is connected or positive L2 regularization
guarantees selection of otherwise free offsets. Hard restrictions and scalar
penalties do not make these fields true. An exact hard equality can fix an
offset in a particular problem, but that separate constraint-identifiability
case is outside the current view rather than generalized prematurely.

Every valid observation set has source-independent row IDs and an ordered
observation-set fingerprint. Its namespace identity contains dimension, point
count, measurement, and IDs; each row identity contains endpoints, shift,
measurement, target, confidence, resolved connector and distance values, both
measurement forms, and the explicit-shift flag. Warnings do not affect
identity. Subsets preserve retained row IDs and input indices, while row
reordering changes the set fingerprint. These identities do not change if
exact source provenance is bound later.

Source binding is separate, optional, exact, and monotonic. Resolver-created
observations are bound; valid directly constructed observations remain
unbound until a source-aware operation independently recomputes and verifies
their connector geometry. The bound source contains caller-order points before
periodic remapping, exact domain representation, dimension/count, and exact ID
provenance. It survives subsets, shallow/deep copy, same-version pickle,
`dataclasses.replace(...)`, and `copy.replace(...)` where available; an
inconsistent replacement raises. Binding and origin associations are private,
not public dataclass fields or source arguments.

One association policy applies to views, records, reports, realization, and
active-set operations: two unbound objects are accepted only for the exact same
observation model; two bound objects are accepted only for the exact same
source; a bound/unbound pair or two different bound sources are rejected.
Fingerprint agreement never replaces exact comparison of canonical values.
The private originating-observation association on fit and diagnostic results
continues to survive copy, replacement, and pickle reconstruction. Reports use
that authoritative origin rather than borrowing provenance from an arbitrary
supplied object.

The generated reference also documents these result/problem conveniences:

```text
SeparatorObservations.pair_labels(*, use_ids=False)
SeparatorObservations.to_records(*, use_ids=False)
SeparatorObservations.subset(mask)
SeparatorFitProblem.observation_graph
SeparatorFitProblem.quadratic_operator
SeparatorFitProblem.canonicalize_gauge(weights)
RealizedPairDiagnostics.to_records(constraints, *, use_ids=False)
RealizedPairDiagnostics.unaccounted_records(*, ids=None)
RealizedPairDiagnostics.to_report(constraints, *, use_ids=False)
PairConstraintDiagnostics.to_records(*, ids=None)
SelfConsistentPowerFitResult.to_records(*, use_ids=False)
SelfConsistentPowerFitResult.to_report(*, use_ids=False)
UnaccountedRealizedPair.to_record(*, ids=None)
```

The active result's `to_records(...)`, `final_realization`, and
`candidate_diagnostics` return `None` when its final fit has no usable weights.

### Inverse record schemas retained under canonical v0.8 names

Record order follows constraint order. `use_ids=True` substitutes the stable
external site IDs where the relevant container has `ids`. Separator IDs are
input-order-aligned, unique, non-negative integers; Python integers and NumPy
integer scalars are accepted without lossy float or string conversion. Raw
observation endpoints are strict integers in both `index_mode='index'` and
`index_mode='id'`.

| Producer | Exact keys |
|---|---|
| `SeparatorObservations.to_records()` | `constraint_index`, `row_id`, `site_i`, `site_j`, `shift`, `target`, `confidence`, `measurement`, `distance`, `target_fraction`, `target_position`, `input_index`, `explicit_shift` |
| `SeparatorFitResult.to_records(...)` | `constraint_index`, `row_id`, `site_i`, `site_j`, `shift`, `measurement`, `target`, `predicted`, `predicted_fraction`, `predicted_position`, `residual`, `alpha`, `beta`, `z_obs`, `z_fit`, `algebraic_residual`, `edge_weight` |
| `RealizedPairDiagnostics.to_records(...)` | `constraint_index`, `row_id`, `site_i`, `site_j`, `shift`, `realized`, `realized_same_shift`, `realized_other_shift`, `realized_shifts`, `endpoint_i_empty`, `endpoint_j_empty`, `boundary_measure` |
| `PairConstraintDiagnostics.to_records(...)` / active result | `constraint_index`, `row_id`, `site_i`, `site_j`, `shift`, `target`, `confidence`, `predicted`, `predicted_fraction`, `predicted_position`, `residual`, `active`, `realized`, `realized_same_shift`, `realized_other_shift`, `realized_shifts`, `endpoint_i_empty`, `endpoint_j_empty`, `boundary_measure`, `toggle_count`, `realized_toggle_count`, `first_realized_iter`, `last_realized_iter`, `marginal`, `status` |
| `HardConstraintConflictTerm.to_record()` | `constraint_index`, `site_i`, `site_j`, `relation`, `bound_value` |
| `UnaccountedRealizedPair.to_record()` | `site_i`, `site_j`, `realized_shifts`, `boundary_measure` |

Measurement-space `residual` and algebraic `z_obs - z_fit` are distinct.
Periodic shifts are integer tuples of the resolved dimension. A confidence-zero
row remains in candidate records but never identifies an informative graph
edge. Configured hard restrictions or penalties may still constrain the row's
predicted separator value; that model coupling is separate from observational
identification.

### Inverse report schemas retained in v0.8

All three report families add the exact common top-level keys `schema`,
`producer`, `source`, and `observation_set` while retaining their existing kind
and family-specific keys. `schema` is exactly
`{"name": "pyvoro2.inverse.separator.report", "version": 1}` and `producer`
is exactly `{"name": "pyvoro2", "version": pyvoro2.__version__}`.
`observation_set` has exactly `fingerprint`, `measurement`, `n_rows`, and
`row_ids`.

| Report | Exact top-level keys | Exact summary keys |
|---|---|---|
| fit | `schema`, `producer`, `source`, `observation_set`, `kind`, `summary`, `constraints`, `fit_records`, `edge_diagnostics`, `objective_breakdown`, `weights`, `radii`, `weight_shift`, `used_shifts`, `warnings`, `conflict`, `connectivity` | `status`, `is_optimal`, `is_infeasible`, `hard_feasible`, `solver`, `linear_backend`, `measurement`, `n_constraints`, `n_points`, `converged`, `status_detail`, `n_iter`, `rms_residual`, `max_residual`, `conflicting_constraint_indices` |
| realized | `schema`, `producer`, `source`, `observation_set`, `kind`, `summary`, `records`, `unrealized`, `unaccounted_pairs`, `warnings`, `tessellation_diagnostics` | `n_constraints`, `n_realized`, `n_same_shift`, `n_other_shift`, `n_unrealized`, `n_unaccounted_pairs` |
| active set | `schema`, `producer`, `source`, `observation_set`, `kind`, `availability`, `summary`, `constraints`, `fit`, `realized`, `diagnostics`, `marginal_records`, `history`, `path_summary`, `tessellation_diagnostics`, `warnings`, `connectivity` | `termination`, `converged`, `n_outer_iter`, `cycle_length`, `n_constraints`, `n_active_final`, `n_realized_final`, `rms_residual_all`, `max_residual_all`, `marginal_constraint_indices` |

`source` has exactly `binding`, `fingerprint`, `dimension`, `n_points`,
`points`, `domain`, and `ids`. An unbound source reports
`binding="unbound"` and null fingerprint, points, domain, and IDs. A bound
source reports `binding="bound"`, its exact fingerprint and caller-order source
data. Bound domain records use only `none`, `planar_box`,
`planar_rectangular_cell`, `spatial_box`, `spatial_orthorhombic_cell`, or
`spatial_periodic_cell`, with the fields fixed by ADR 0014. A bound
`{"kind": "none"}` domain is distinct from unbound `domain=null`.

Nested fit `edge_diagnostics` uses the fields of
`AlgebraicEdgeDiagnostics`; `objective_breakdown` uses the fields of
`PowerFitObjectiveBreakdown`: `total`, `mismatch`, `penalties_total`,
`penalty_terms`, `regularization`, `hard_constraints_satisfied`,
`hard_max_violation`, and `hard_max_tolerance`. Connectivity records contain
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

Issue #13 did not add, remove, or rename report keys. Existing fit report
state keys (`weights`, `radii`, `weight_shift`), `connectivity`, observation
records/summaries, `objective_breakdown`, `edge_diagnostics`, and solver
summary/conflict/warnings correspond respectively to the state,
identification, observations, objective, algebraic, and solver layers.
Realized `records`/`unrealized` describe requested-image matching, while
unaccounted pairs, optional record geometry, and tessellation diagnostics
describe realized geometry. Active `fit`, `realized`, `diagnostics`, `summary`,
`history`, and `path_summary` describe the final inner fit, final realization,
per-candidate diagnostics, outer termination, and active-set path.
Issue #36 subsequently adds the approved nested
`objective_breakdown.hard_max_tolerance` key and separates the fit-summary
`solver` and `linear_backend` fields. Issue #42 adds the common versioned
envelope and `row_id` values above without renaming or removing those existing
fields. Report builders return JSON-native values. Finite fit, realized, and
active reports round-trip exactly; JSON serialization rejects NaN and infinity.
ADR 0015 makes active final-state assembly atomic. Its `availability` block has
exactly `weights`, `realization`, `records`, and `reason`. All flags are true
and `reason` is null for an available final state. All flags are false and
`reason` is the final fit status when weights are unavailable. In that mode,
`realized`, `diagnostics`, `marginal_records`, and
`tessellation_diagnostics` are null, as are `summary.n_realized_final`,
`summary.rms_residual_all`, and `summary.max_residual_all`. The nested fit
report remains present; the active summary retains outer termination while the
nested fit retains final inner status and convergence. Every active report
round-trips exactly through strict JSON.

### Historical calls exercised by repository examples

The v0.6.3 notebooks used raw spatial forward calls and historical top-level
inverse imports. The characterized historical set was:

```text
Box, OrthorhombicCell, PeriodicCell, compute, locate, ghost_cells,
normalize_vertices, normalize_topology, annotate_face_properties,
resolve_pair_bisector_constraints, FitModel, SquaredLoss, Interval, FixedValue,
ExponentialBoundaryPenalty, fit_power_weights, match_realized_pairs,
ActiveSetOptions, solve_self_consistent_power_weights, dumps_report_json,
pyvoro2.viz3d.VizStyle, pyvoro2.viz3d.view_tessellation
```

They also called `to_records(...)`, `to_report(...)`, and conflict record
helpers. This list is retained only as migration history. Current source
notebooks use `TessellationResult`, `pyvoro2.inverse`, and
`pyvoro2.inverse.separator`; the removed names are not required to execute or
export them.

No manuscript program or paper environment is stored in this repository. The
repository-owned paper-style regression subset and downstream-shaped examples
use the canonical current APIs. The documented v0.6.3 algebraic formulas,
periodic-image semantics, and calls above remain historical context for archived
research and migration.

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

These are requirements for the preferred current surfaces, not permission to add
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

## Current v0.8 contract

The current tree has one structured forward contract, one canonical high-level
separator route, and one advanced separator namespace. Runtime inspection and
the import/signature tests establish these exact package export counts:

| Namespace | Exact `__all__` size | Lifecycle boundary |
|---|---:|---|
| `pyvoro2` | 29 | Stable forward/result surface plus package metadata |
| `pyvoro2.planar` | 25 | Stable explicit 2D surface plus provisional plotting |
| `pyvoro2.inverse` | 6 | Stable normal fixed-observation separator workflow |
| `pyvoro2.inverse.separator` | 53 | Stable core names, provisional advanced objects, experimental active-set objects |
| `pyvoro2.viz3d` | 9 | Provisional optional visualization |

There is no current `pyvoro2.powerfit`, top-level separator export set,
historical separator alias, `PlanarComputeResult`, planar `return_result=`, or
`pyvoro2.planar.result`. Those names appear below only in migration/history
sections.

The sole weight/radius implementation lives in the internal neutral module
`pyvoro2._internal.weight_transforms`; the stable exports from `pyvoro2`,
`pyvoro2.inverse`, and `pyvoro2.inverse.separator` are identical function
objects. The other pure-Python helpers live under `pyvoro2._internal` with
explicit shared, spatial, or planar ownership. The native extensions retain
their root-owned internal names `_core` and `_core2d`.

Plain `import pyvoro2` imports the pure-Python forward/result surface and the
planar namespace, but not `pyvoro2.inverse`, `_core`, or `_core2d`. Importing
either canonical inverse namespace also leaves the native modules unloaded.
The 3D and 2D wrappers load `_core` and `_core2d`, respectively, on the first
forward geometry operation that requires them.

### Exact current forward signatures

Annotations are omitted here only for readability; parameter kind, order, and
defaults are exact:

```text
pyvoro2.compute(
    points, *, domain, ids=None,
    duplicate_check='off', duplicate_threshold=1e-5,
    duplicate_wrap=True, duplicate_max_pairs=10,
    block_size=None, blocks=None, init_mem=8,
    mode='standard', weights=None, radii=None,
    return_vertices=True, return_adjacency=True, return_faces=True,
    return_face_shifts=False, face_shift_search=2, include_empty=False,
    validate_face_shifts=True, repair_face_shifts=False, face_shift_tol=None,
    return_diagnostics=False, output='result',
    tessellation_check='none', tessellation_require_reciprocity=None,
    tessellation_volume_tol_rel=1e-8,
    tessellation_volume_tol_abs=1e-12,
    tessellation_plane_offset_tol=None,
    tessellation_plane_angle_tol=None,
)

pyvoro2.planar.compute(
    points, *, domain, ids=None,
    duplicate_check='off', duplicate_threshold=1e-5,
    duplicate_wrap=True, duplicate_max_pairs=10,
    block_size=None, blocks=None, init_mem=8,
    mode='standard', weights=None, radii=None,
    return_vertices=True, return_adjacency=True, return_edges=True,
    return_edge_shifts=False, edge_shift_search=2, include_empty=False,
    validate_edge_shifts=True, repair_edge_shifts=False, edge_shift_tol=None,
    return_diagnostics=False, output='result',
    normalize='none', normalization_tol=None,
    tessellation_check='none', tessellation_require_reciprocity=None,
    tessellation_area_tol_rel=1e-8,
    tessellation_area_tol_abs=1e-12,
    tessellation_line_offset_tol=None,
    tessellation_line_angle_tol=None,
)
```

The current `locate(...)`, `ghost_cells(...)`, domain, diagnostics,
validation, duplicate, normalization, annotation, transform, and visualization
signatures are exactly the retained signatures listed in the corresponding
sections above. In particular, `locate(...)` and `ghost_cells(...)` remain
radius-only in power mode; `weights=` belongs only to `compute(...)`.

Power-mode `compute(...)` requires exactly one of `weights=` and `radii=`.
Standard mode rejects both. Weight input must have shape `(n,)` and all
conversion arithmetic must remain finite. Valid direct-radius behavior is
unchanged. Finite representability does not promise geometric resolution when
squared backend radii or genuine weight ranges overwhelm squared geometry
scales.

The current native-construction controls have one shared contract in both
dimensions. `init_mem` and each explicit `blocks` entry must be a positive
exact non-Boolean index-protocol scalar within the C++ `int` range; `blocks`
has length 3 spatially and length 2 planarly. `block_size`, when supplied, must
be a positive finite real numeric scalar. Points and queries must have the
dimension-appropriate matrix shape and contain only finite values; radii must
have the required vector shape and be finite and non-negative. Bounds and
periodic constructor parameters must be finite, ordered where applicable, and
safe for the arithmetic evaluated by Voro++. Violations raise `ValueError`
before native construction.

The rest of the stable forward input surface uses the same strict source-type
policy. External IDs are exact non-Boolean integers, non-negative, unique,
length-aligned where site input is available, and signed-int64 representable;
direct diagnostic `expected_ids` use the same contract. Search counts and
normalization example caps are non-negative exact integers, with
`max_examples=0` retaining its no-examples boundary. Duplicate pair limits are
positive exact integers. Public flags accept only Python or NumPy Boolean
scalars. Duplicate thresholds are positive finite reals, and shift,
normalization, and diagnostic tolerances use their documented positive or
non-negative finite ranges. These stricter rejections do not change signatures,
defaults, result schemas, or valid forward values.

Normalization revalidates mutable raw cell metadata at each public
normalization boundary: cell and adjacent-cell IDs, local/global vertex
indices, and lattice shifts use exact signed-int64-compatible rules with their
field-specific non-negative constraints. Coordinate/tolerance quantization is
accepted only when the quotient and rounded key are finite and signed-int64
representable; unsupported relationships raise before topology construction
or in-place annotation.

All domain constructors canonicalize bounds, periodic flags, triclinic vectors,
and origins into owned nested built-in tuples. `Box.from_points` rejects empty
or non-finite data before reduction and accepts non-negative padding only when
the resulting bounds remain finite and strictly ordered. `PeriodicCell`
requires a right-handed basis while retaining its existing conditioning
warning and rejection thresholds. Remap helpers validate exact Boolean flags,
finite non-negative `eps`, and finite points, and reject a lattice shift that
cannot fit signed int64 before casting. ADR 0011 records this contract.

All 18 internal native construction routes repeat converted-value checks and
checked constructor arithmetic before allocation. An aggregate source-derived
estimate of known eager native construction allocations may be at most exactly
`1 << 30` bytes; estimates greater than 1 GiB raise `ValueError`, with no
unsafe override. This is a defensive change to invalid-input rejection, not a
new public name, signature, default, result schema, or valid numerical
behavior. The exact layering and estimate policy are fixed by ADR 0010.

Periodic nearest-image inference now certifies the exact Euclidean minimum for
the dyadic rational values represented by supplied binary64 coordinates and
lattice components. Rectangular and orthorhombic domains use an exact per-axis
fast path; fully periodic non-orthogonal 3D cells exact-enumerate a
proof-derived finite coefficient box. Separator tie orientation uses the
resolved internal site indices in the fixed point ordering, so lattice
translation and pair reversal retain their expected shift/displacement
invariants. External IDs remain metadata and do not participate in geometric
tie selection; no point-array permutation invariant or public tie mode is
introduced. Explicit observation shifts remain authoritative even when a
different image is nearer.

The public `image_search` parameter remains a non-negative exact integer with
default one in all three separator entry points below. It is now only a capped
incumbent-seeding hint: it may change private candidate counts and runtime but
cannot change a successful inferred shift, displacement, or distance. There is
no approximate mode or boundary-warning result. If exact certification exceeds
the frozen private resource or signed-int64 shift contract, inference raises
without returning an approximate image. When periodic wrapping is enabled
(`wrap=True`, or `duplicate_wrap=True` in forward operations), periodic
duplicate checks use the same exact pair-distance primitive for pairs evaluated
by the current scanner. With wrapping disabled, the established unwrapped
Cartesian check is preserved. R5 still owns complete seam scanning and
mandatory native-safety policy independent of `duplicate_wrap`. ADR 0012
records this contract.

### Exact current inverse signatures and defaults

The stable high-level calls are:

```text
resolve_separator_observations(
    points, constraints, *, measurement='fraction', domain=None, ids=None,
    index_mode='index', image='nearest', image_search=1, confidence=None,
    allow_empty=False,
)

fit_weights_from_separators(
    points, constraints, *, measurement='fraction', domain=None, ids=None,
    index_mode='index', image='nearest', image_search=1, confidence=None,
    model=None, r_min=0.0, weight_shift=None, solver='direct',
    linear_backend='dense', admm_max_iter=2000, admm_rho=1.0,
    admm_abs_tol=1e-6, admm_rel_tol=1e-5,
    connectivity_check='warn',
)

weights_to_radii(weights, *, r_min=0.0, weight_shift=None)
radii_to_weights(radii)
```

The principal advanced calls are:

```text
build_power_fit_problem(constraints, *, model=None)
build_power_fit_result(
    problem, weights, *, solver='external', linear_backend=None,
    status='optimal',
    status_detail=None, converged=True, n_iter=0, warnings=(),
    canonicalize_gauge=True, r_min=0.0, weight_shift=None,
)
match_realized_pairs(
    points, *, domain, constraints, weights=None, radii=None,
    return_boundary_measure=False, return_cells=False,
    return_tessellation_diagnostics=False,
    tessellation_check='diagnose', unaccounted_pair_check='diagnose',
)
solve_self_consistent_power_weights(
    points, constraints, *, measurement='fraction', domain, ids=None,
    index_mode='index', image='nearest', image_search=1, confidence=None,
    model=None, active0=None, options=None, r_min=0.0, weight_shift=None,
    fit_solver='direct', fit_linear_backend='dense',
    fit_admm_max_iter=2000, fit_admm_rho=1.0,
    fit_admm_abs_tol=1e-6, fit_admm_rel_tol=1e-5,
    return_history=False, return_cells=False,
    return_boundary_measure=False, return_tessellation_diagnostics=False,
    tessellation_check='diagnose', connectivity_check='warn',
    unaccounted_pair_check='warn',
)
```

`match_realized_pairs(...)` requires exactly one of `weights=` and `radii=`.
The active-set wrapper forwards the same method and linear-backend choices
through the `fit_*` parameters. Report helper defaults and the
objective/model/active-set constructor defaults are exactly those listed in
the retained constructor table above.

The public row-only chain

```text
SeparatorObservations
-> build_power_fit_problem
-> build_power_fit_result
-> build_fit_report
```

remains valid for directly constructed source-unbound observations; none of
these builders gains a public points, domain, or source argument. If an already
resolved observation object is fitted with points, those points establish or
verify exact source points. Omitted/default `domain=None` makes no additional
domain assertion against an already bound object and does not erase its bound
domain. For an unbound object it binds the explicit no-domain representation.
An explicit non-`None` domain must match or establish that exact domain.
Realization and active-set operations verify the complete source they use.

SciPy is optional. `linear_backend='dense'` uses NumPy and never imports SciPy.
Explicit `linear_backend='sparse'` and explicit sparse matrix conversion import
SciPy lazily and raise an actionable `ImportError` when it is absent. There is
no site-count backend selection. Direct solving accepts only purely quadratic
models; ADMM is required for Huber mismatch, hard restrictions, and active
scalar penalties, and explicit ADMM also executes for a quadratic model
whenever a component solve is required.
Degenerate fits that need no component solve, including empty observation sets
and models with only singleton components, report `solver='none'`,
`linear_backend=None`, and `n_iter=0`.

## Accepted v0.8 contract decisions

The following boundaries are already accepted:

- `pyvoro2.inverse` is the canonical inverse namespace;
- separator implementation is owned by `pyvoro2.inverse.separator`;
- `pyvoro2.powerfit`, broad top-level separator exports, and the other
  v0.7-only routes are removed;
- both forward `compute(...)` functions return
  `pyvoro2.TessellationResult` by default;
- `output='cells'` is the explicit supported raw-output route;
- `TessellationResult` is the only planar structured-result name;
- deep immutability of nested raw records is not part of the contract;
- squared mismatch, quadratic Huber mismatch, and L2 regularization use the
  accepted half-factor objective convention;
- zero-strength scalar penalties are absent, hard bounds use the shared
  float64 scale-aware tolerance, and successful solver results have finite
  reported soft objectives;
- positive-strength scalar-penalty proximal coordinates succeed only with
  private proved exact point signs or an adjacent numeric-binary64 sign bracket,
  while an uncertified attempt maps to the existing `numerical_failure` schema;
- inferred periodic nearest images are exact-certified, explicit image shifts
  remain authoritative, and `image_search` is a correctness-neutral seed hint;
- source-independent row and observation-set identity is always available,
  while exact geometry-source provenance is optional and monotonic;
- row-aligned records carry stable row IDs and report schema version 1 carries
  authoritative source and observation-set provenance.

See [ADR 0004](decisions/0004-canonical-inverse-namespace.md) and
[ADR 0005](decisions/0005-tessellation-result-contract.md), as refined by
[ADR 0006](decisions/0006-v0.8-cleanup-release.md), together with
[ADR 0007](decisions/0007-separator-objective-contract.md),
[ADR 0008](decisions/0008-separator-solver-and-linear-backend.md),
[ADR 0009](decisions/0009-certified-scalar-proximal-solver.md),
[ADR 0010](decisions/0010-native-construction-preconditions.md),
[ADR 0011](decisions/0011-strict-input-and-ownership-contract.md),
[ADR 0012](decisions/0012-certified-periodic-image-geometry.md),
[ADR 0013](decisions/0013-central-generator-preparation-and-backend-safety.md),
and [ADR 0014](decisions/0014-separator-observation-and-source-identity.md).

## Lifecycle summary for the v0.8 API

| Surface | v0.8 status | Notes |
|---|---|---|
| Domain classes and domain geometry semantics | Stable | Mature bounded and periodic behavior; capability differences remain explicit by dimension. |
| `pyvoro2.compute` and `pyvoro2.planar.compute` | Stable | Direct weight/radius behavior, the common structured default, and explicit raw output are implemented and tested. |
| `weights=` and `radii=` mathematical meaning | Stable | Mode-specific rejection/exclusivity, one global representation shift, finite and representable conversion, and empty-cell behavior are part of the contract. |
| `pyvoro2.TessellationResult` core contract | Stable | The shared class and both public compute integrations are stable; direct construction is classified separately as provisional. |
| Detailed optional result conveniences and raw geometry views | Provisional | Refine through implementation and chemvoro-shaped validation. |
| `pyvoro2.inverse` preferred high-level separator workflow | Stable | Validated normal observations/fit entry point for applications and chemvoro-shaped workflows. |
| `pyvoro2.inverse.separator` advanced problem and operator views | Provisional | Public for research use, but may evolve before v0.9 prescribed measures and v0.10 mixed problems. |
| Realization-aware active-set API | Experimental | Practical outer algorithm; no universal convergence claim. |
| Optional sparse linear backend | Provisional | Explicit `linear_backend='sparse'` supports direct quadratic solving and ADMM weight systems; it requires SciPy and is never selected by site count. |
| v0.7-only inverse and planar transition routes | Removed | Ordinary import, attribute, or argument failure; replacements are in the migration guide. |
| `pyvoro2._internal`, native extensions, and solver-internal modules | Internal | No compatibility guarantee; `_internal` has no package-level convenience exports. |

### Documented module-route status

Objects imported from these documented modules retain the lifecycle status
assigned above. The module route itself has the following status:

| Module route | v0.8 status |
|---|---|
| `pyvoro2.api`, `domains`, `diagnostics`, `duplicates`, `face_properties`, `normalize`, `validation` | Stable |
| `pyvoro2.edge_properties` | Stable for its documented annotation helper |
| `pyvoro2.result` | Stable module route for `TessellationResult`; direct construction remains provisional |
| `pyvoro2.viz2d`, `pyvoro2.viz3d` | Provisional optional conveniences |
| `pyvoro2.planar.api`, `domains`, `diagnostics`, `duplicates`, `normalize`, `validation` | Stable |
| `pyvoro2.inverse` | Stable high-level route |
| `pyvoro2.inverse.separator` and its non-active submodules | Mixed route: stable high-level core names and provisional advanced objects |
| `pyvoro2.inverse.separator.active` | Experimental |
| `pyvoro2.planar.result`, `pyvoro2.powerfit`, and its direct submodules | Removed; these are not current module routes |
| `pyvoro2._internal` helpers and native `_core`/`_core2d` extensions | Internal; the native modules remain outside `_internal` |

## Spatial forward namespace: `pyvoro2`

The exact current 29-name `pyvoro2.__all__` is:

```text
Box
OrthorhombicCell
PeriodicCell
TessellationResult
compute
locate
ghost_cells
TessellationDiagnostics
TessellationIssue
TessellationError
analyze_tessellation
validate_tessellation
NormalizationDiagnostics
NormalizationIssue
NormalizationError
validate_normalized_topology
DuplicatePair
DuplicateError
duplicate_check
annotate_face_properties
NormalizedVertices
NormalizedTopology
normalize_vertices
normalize_edges_faces
normalize_topology
radii_to_weights
weights_to_radii
__version__
planar
```

All are **stable** names or surfaces. The `__version__` value naturally tracks
the installed release, and `planar` is the explicit 2D namespace.
Visualization remains a direct provisional module route rather than a
top-level export.

### Historical v0.7 top-level inverse compatibility set

The following 34 names were top-level `pyvoro2` compatibility exports in v0.7.
They are listed only to make the removal inventory exact; none is a current
top-level export:

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

## Planar namespace: `pyvoro2.planar`

The exact current 25-name `pyvoro2.planar.__all__` is:

```text
Box
RectangularCell
TessellationResult
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

`plot_tessellation` is **provisional** and optional. The other 24 names are
**stable**. The re-exported `TessellationResult` is the identical class object
as `pyvoro2.TessellationResult`.

## Canonical inverse namespace: `pyvoro2.inverse`

### Preferred high-level separator API

The exact current `pyvoro2.inverse.__all__` is:

```text
SeparatorObservations
resolve_separator_observations
SeparatorFitResult
fit_weights_from_separators
weights_to_radii
radii_to_weights
```

The preferred names have these final lifecycle assignments:

| Name | v0.8 status | Meaning |
|---|---|---|
| `SeparatorObservations` | Stable | Canonical pairwise separator rows with periodic image labels, confidence, source-independent identity, and optional exact source binding. |
| `resolve_separator_observations` | Stable | Validate and resolve raw separator observations against sites and domain. |
| `SeparatorFitResult` | Stable | Existing flat fit contract plus layered state, observation, identification, objective, algebraic, and fixed-solver access. |
| `fit_weights_from_separators` | Stable | Preferred fixed-observation fit entry point. |
| `weights_to_radii`, `radii_to_weights` | Stable re-export where useful | Same neutral transforms as top-level pyvoro2. |

### Advanced separator API

After the v0.8 issue-#28 removal,
`pyvoro2.inverse.separator.__all__` contains exactly the following 53 names:

```text
SeparatorObservations
resolve_separator_observations
SeparatorFitProblem
SeparatorFitResult
fit_weights_from_separators
SeparatorFitStateView
SeparatorIdentificationView
SeparatorObservationView
SeparatorAlgebraicView
SeparatorSolverTerminationView
SeparatorObservationGraphView
SeparatorQuadraticOperatorView
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
build_power_fit_problem
build_power_fit_result
RequestedImageMatchView
RealizedGeometryView
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
ActiveSetTerminationView
ActiveSetPathView
PairConstraintDiagnostics
SelfConsistentPowerFitResult
match_realized_pairs
solve_self_consistent_power_weights
radii_to_weights
weights_to_radii
```

The canonical core and neutral transforms have the statuses assigned above.
The objective model, problem construction/evaluation, fixed-fit and realization
view types, realization, reporting, and diagnostic objects are initially
**provisional**. The active-set outer workflow and its options, termination/path
views, iteration, path, diagnostic, and result objects are **experimental** and
separator-specific.

The historical v0.7 identity map, removed in v0.8, was:

| Removed historical name | Current canonical name | Historical relationship |
|---|---|---|
| `PairBisectorConstraints` | `SeparatorObservations` | Identity alias; historical name compatibility-only through v0.7 |
| `resolve_pair_bisector_constraints` | `resolve_separator_observations` | Identity alias; historical name compatibility-only through v0.7 |
| `PowerFitProblem` | `SeparatorFitProblem` | Identity alias; historical name compatibility-only through v0.7 |
| `PowerWeightFitResult` | `SeparatorFitResult` | Identity alias; historical name compatibility-only through v0.7 |
| `fit_power_weights` | `fit_weights_from_separators` | Identity alias; historical name compatibility-only through v0.7 |

The accepted provisional advanced surfaces include:

- `SeparatorFitProblem` and problem-building/evaluation helpers;
- problem-owned `SeparatorObservationGraphView` and
  `SeparatorQuadraticOperatorView`, including dense NumPy and optional lazy
  SciPy conversions; the explicit sparse solver consumes this operator through
  the separate fixed-fit entry point;
- objective model pieces such as squared/Huber losses, hard intervals,
  penalties, and regularization;
- graph, connectivity, incidence, Laplacian, and objective-breakdown views;
- result packaging for externally computed weights;
- layered fixed-fit and realization views that reference existing result data;
- realization matching and record/report builders.

Issue #36 freezes the separator objective semantics. For
`e = beta + alpha * (w_i - w_j) - target`, squared loss is
`0.5 * e**2`; Huber loss is `0.5 * e**2` for `abs(e) <= delta` and
`delta * (abs(e) - 0.5 * delta)` otherwise. Confidence multiplies only
mismatch. L2 is
`0.5 * strength * ||weights - reference||**2`, so the normal system remains
`A = L_obs + strength * I` and
`b = b_obs + strength * reference`.

Soft-interval and exponential strengths retain their existing meanings.
For reciprocal inward distance `d`, the contribution is zero for
`d >= margin`, `strength * (1 / d - 1 / margin)` for
`epsilon < d < margin`, and
`strength * ((1 / epsilon - 1 / margin)
- (d - epsilon) / epsilon**2)` for `d <= epsilon`; lower and upper
contributions are summed. Zero-strength penalties are exact no-ops. Hard rows
use
`1e-12 + 64 * finfo(float64).eps * max(abs(lower), abs(prediction),
abs(upper))`, and successful solver results require finite reported
soft-objective components and totals. ADR 0007 records the derivative,
continuation, and compatibility rationale.

Issue #37 replaces the scalar penalty proximal loop without changing that
objective or any public name/default/schema. The private solver compiles one
positive-strength term kernel once, evaluates complete source expressions,
represents exact structural breakpoints and one-sided reciprocal-margin
derivatives, and brackets rigorous derivative signs. It accepts only proved
exact point signs or adjacent numeric-binary64 localization. Scaled binary64
accumulation preserves determinable exponential signs through raw overflow;
bounded 80/160-digit work resolves only rare ambiguities. Direct termwise
objective differences select adjacent endpoints. Exhaustion or unresolved
evaluation becomes the existing structured `numerical_failure`; a failed
proximal attempt does not increment completed ADMM iterations. ADR 0009 records
the private numerical contract.

The realization matcher accepts weight-first and radius-representation inputs
as mutually exclusive current routes. New workflows use fitted mathematical
weights; direct radii remain a supported advanced representation input.

Separator integer, Boolean, finite-value, and ownership policy is shared with
the forward layer. Observation endpoints, shifts, provenance indices, search
counts, iteration counts, and active-set hysteresis counts are exact
non-Boolean integers with field-specific ranges. Flags and masks are exact
Booleans. Model parameters, confidence, solver tolerances, radii floors, and
optional representation shifts reject non-real or non-finite input before
solver work. A nonzero `r_min` and explicit `weight_shift` are rejected as a
deterministic input conflict before solving, active-set iteration, prediction,
objective evaluation, or radius construction. Finite observation source
coordinates must also yield finite representable connector differences,
squared distances, distances, and measurement conversions. Model and option
scalars are stored as built-in Python values.
`SeparatorObservations`, `L2Regularization.reference`, and directly retained
problem arrays own C-contiguous read-only copies; `FitModel.penalties` owns a
tuple. The documented wider non-negative separator external-ID range remains
unchanged. These adoption rules change invalid-input rejection only, not the
R1/R2 objective, backend selection, solver defaults, active-set mathematics,
or valid numerical fit. R6 subsequently adds row/set identity, optional exact
source binding, and report-schema metadata without changing public fit or
builder signatures. In particular, finite extreme-scale source inputs retain
R1/R2's stabilized handling of derived scaled-row infinities inside the
canonical problem builder; direct problem construction itself is finite-strict.

The active-set outer workflow and its path/result types remain **experimental**.
The explicit SciPy sparse linear backend is **provisional**. It supports direct
quadratic solving and ADMM weight systems, including active-set forwarding,
without changing the solver method.

During v0.7, `pyvoro2.powerfit.__all__` remained the exact 42-name historical
list recorded in the v0.6.3 baseline section. It did not export the canonical
names or contain implementation logic. Issue #28 removed that package, the
broad top-level historical separator set, and the five historical identity
aliases in v0.8.

## Forward return contract

### Implemented common data contract

Issues #9 and #10 implement one frozen, slotted `TessellationResult` class and export
the identical class object as both `pyvoro2.TessellationResult` and
`pyvoro2.planar.TessellationResult`. Its private shared builder aligns cells by
final external ID, represents omitted empty cells explicitly in aligned
arrays, and does not invoke native computation, diagnostics, normalization, or
boundary annotation.

The stable fields are exact:

| Field | Lifecycle | Semantics |
|---|---|---|
| `dimension` | Stable | Explicit `2` or `3`. |
| `domain` | Stable | Validated domain used by the computation. |
| `mode` | Stable | `"standard"` or `"power"`. |
| `sites` | Stable | Read-only owned `(n, dimension)` copy of validated input coordinates in original input order. |
| `ids` | Stable | Read-only owned `(n,)` external-ID array in original input order; omitted IDs become `np.arange(n, dtype=np.int64)`. |
| `cells` | Stable | Exact supplied raw-cell list after ID remapping; the list, dictionaries, and nested records are not copied or frozen. |
| `cell_measures` | Stable | Read-only owned `(n,)` construction-time snapshot of areas or volumes aligned with input order; hidden cells are zero. |
| `empty_mask` | Stable | Read-only owned boolean `(n,)` construction-time snapshot aligned with input order, including raw records omitted by `include_empty=False`. |
| `input_weights` | Stable | Read-only owned mathematical input weights for weight-first power input; otherwise `None`. |
| `backend_radii` | Stable | Read-only owned exact native power radii; `None` in standard mode. |
| `representation_shift` | Stable | Finite common additive shift for weight-first conversion; `None` for standard or direct-radius input. |
| `tessellation_diagnostics` | Stable | Existing dimension-specific diagnostics when computed; otherwise `None`. |
| `normalized_vertices` | Stable | Existing dimension-specific vertex normalization when computed; otherwise `None`. |
| `normalized_topology` | Stable | Existing dimension-specific topology normalization when computed; otherwise `None`. |

The following convenience surface remains **provisional**:

| Convenience | Lifecycle | Semantics |
|---|---|---|
| `measure_kind` | Provisional | `"area"` in 2D or `"volume"` in 3D. |
| `boundary_kind` | Provisional | `"edges"` in 2D or `"faces"` in 3D. |
| `has_tessellation_diagnostics`, `has_normalized_vertices`, `has_normalized_topology` | Provisional | Distinguish absent optional objects from present objects. |
| `has_boundaries`, `has_periodic_shifts` | Provisional | Report explicit builder capabilities, including available-but-empty geometry. |
| `require_tessellation_diagnostics()`, `require_normalized_vertices()`, `require_normalized_topology()` | Provisional | Return optional objects or raise a clear `ValueError`. |
| `require_boundaries()` | Provisional | Return input-order-aligned edge/face collections, using an empty collection for hidden sites, or raise when boundaries were unavailable. |
| `global_vertices`, `global_edges` | Provisional conveniences | Forward to available planar normalized objects; otherwise `None`. |

The outer object prevents field replacement. Its aligned arrays are copies and
are non-writeable, so construction never marks caller-owned arrays read-only.
The raw `cells` list and its nested dictionaries remain shared and mutable by
design. Later raw-record mutation does not update the `cell_measures` or
`empty_mask` snapshots. Boundary access revalidates mutable boundary record
types, current empty flags, required non-empty records, and periodic-shift
fields and raises if mutation made them inconsistent with the recorded
snapshots or capabilities. An empty cell cannot contain realized edge or face
records; both omitted and explicitly empty boundary collections remain valid.

Direct dataclass construction is **provisional** and validates documented raw
IDs, measures, empty state, representation metadata, and capability metadata
against the aligned fields. It does not normalize arbitrary hand-written
backend-style dictionaries, recompute derived geometry, or geometrically verify
the records. Weight-first metadata must satisfy the shared exact
weight/shift-to-radius transform. Boundary and periodic-shift availability are
private keyword-only construction state supplied by the shared builder;
keeping them as normal dataclass initialization fields preserves them through
`dataclasses.replace()` without adding stable public result fields. Deep copies
and same-version pickle round trips preserve the exact existing snapshot state
rather than revalidating it against later permitted raw-record mutation;
reconstructed arrays remain owned and read-only, and capability state is
preserved. No cross-version pickle compatibility is promised.

### Preferred compute route

```python
result = pyvoro2.compute(..., output='result')
result = pyvoro2.planar.compute(..., output='result')
```

Omitting `output` is equivalent to `output='result'`. Structured output is
always one `TessellationResult`, never a tuple. Diagnostics computed because of
`return_diagnostics=True` or `tessellation_check='diagnose'|'warn'|'raise'`
are stored in `result.tessellation_diagnostics`.

### Raw output route

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

### Historical v0.7 planar compatibility selector matrix

In v0.7, the public compatibility parameter was
`return_result: bool | None = None`.
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

In v0.7, `PlanarComputeResult` from both `pyvoro2.planar` and
`pyvoro2.planar.result` was an identity alias to
`pyvoro2.TessellationResult`; issue #28 removed both routes in v0.8.

## Supported Python and distribution contract

The support claim is derived from `pyproject.toml`, the CI and wheel workflows,
and `tools/check_wheel_matrix.py`:

| Layer | Exact v0.8 contract |
|---|---|
| Package metadata | `Requires-Python: >=3.10`; classifiers list Python 3.10, 3.11, 3.12, 3.13, and 3.14 |
| Supported source builds | Standard GIL-enabled CPython 3.10–3.14 |
| Source-install CI | All five supported versions on Linux, macOS, and Windows |
| Wheel interpreters | CPython tags `cp310`, `cp311`, `cp312`, `cp313`, `cp314` |
| Wheel platforms | manylinux x86_64, Windows AMD64, macOS arm64, macOS x86_64 |
| Release artifact count | Exactly 20 wheels and one matching source distribution |
| Optional SciPy | Not a runtime dependency; installed for wheel tests and imported only by explicit sparse paths |

The open-ended metadata lower bound allows installation tooling to evaluate a
future Python version, but Python versions newer than 3.14 are not part of the
tested v0.8 support contract. Free-threaded CPython, alternative interpreters,
musllinux, non-x86_64 Linux, 32-bit and arm64 Windows, and macOS universal2 are
explicitly excluded from the v0.8 wheel matrix. Source installation on an
unlisted environment may succeed but is not a prebuilt-wheel or tested-support
promise.

Every release wheel must contain both native modules, `_core` and `_core2d`,
and is installed and exercised on a compatible runner. The source distribution
is validated separately, rebuilt into one wheel under build isolation, and
installed in a fresh no-SciPy environment. These distribution checks do not
turn the internal native module names into public API.

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
- every valid separator observation row and ordered observation set has a
  deterministic source-independent identity; warnings are not row identity;
- exact separator source provenance, when known, preserves caller-order points,
  exact domain representation, dimension/count, and ID provenance, and cannot
  be erased or rebound;
- unbound and bound observations never associate with one another, and two
  bound origins associate only under exact canonical source equality;
- zero-confidence separator rows do not identify weight differences or enter
  the informative observation graph; hard restrictions and penalties may
  constrain their predicted values but remain separate from data
  identification;
- row confidence multiplies only separator mismatch; squared mismatch and the
  Huber quadratic branch are `0.5 * residual**2`, while L2 regularization is
  `0.5 * strength * ||weights - reference||**2`;
- zero-strength scalar penalties do not affect objective values, coupling,
  backend selection, or quadratic-operator availability;
- `hard_max_violation` is a raw violation and `hard_max_tolerance` records the
  maximum shared absolute-plus-relative float64 tolerance used;
- optimal or converged solver results have finite reported soft-objective
  components and totals;
- algebraic fit does not imply realized-boundary support;
- empty/hidden cells are represented deterministically according to the chosen
  output route;
- error/status behavior for infeasibility and wrong-image realization is part of
  the public scientific contract.

## Deprecation and fixed removal schedule

| Surface | v0.7 | v0.8 action |
|---|---|---|
| `pyvoro2.powerfit` | Compatibility-only and deprecated; loading it emits one hidden-by-default `DeprecationWarning` naming the canonical namespaces and v0.8 horizon | Remove |
| Broad top-level separator exports | Compatibility-only; no invasive attribute wrappers during v0.7; documented migration path | Remove from top-level |
| Five mapped historical core names | Compatibility-only and deprecated identity aliases; no per-use warnings | Remove from canonical separator exports |
| `PlanarComputeResult` | Compatibility-only and deprecated alias to `TessellationResult` | Remove |
| Raw cell return | Available through `output='cells'` | Continue as explicit route unless a later decision removes it |
| Planar `return_result=` | Compatibility-only and deprecated | Remove |

### v0.8 compatibility removal status

Issue #28 completed this schedule without changing canonical numerical
behavior. The v0.8 tree:

- has no `pyvoro2.powerfit` package, direct submodules, or lazy top-level
  package attribute;
- exports no separator-specific objects from top-level `pyvoro2`;
- contains none of the five mapped historical core aliases in either
  `pyvoro2.inverse.separator` or their former direct canonical submodules;
- exports only `TessellationResult` from the planar namespace and does not
  provide `pyvoro2.planar.result`;
- has no planar `return_result=` parameter; and
- retains `output='cells'` as the explicit supported raw-output route.

The high-level `pyvoro2.inverse` export set, canonical class and function names,
solver defaults, numerical values, result fields, record keys, and gauge
policies are unchanged.

### v0.8 private-helper organization status

Issue #30 moved all private pure-Python implementation helpers into
`pyvoro2._internal`:

```text
pyvoro2._internal.cell_output
pyvoro2._internal.inputs
pyvoro2._internal.power_input
pyvoro2._internal.validation
pyvoro2._internal.weight_transforms
pyvoro2._internal.spatial.domain_geometry
pyvoro2._internal.spatial.domain_utils
pyvoro2._internal.spatial.face_shifts
pyvoro2._internal.planar.domain_geometry
pyvoro2._internal.planar.edge_shifts
```

These module routes and every object available only from them are
**internal**. The `_internal`, `_internal.spatial`, and `_internal.planar`
package initializers provide no convenience re-exports. The former root helper
modules and former `pyvoro2.planar` helper modules are absent, with no
compatibility shims, because they were never documented or exported as public
API.

The stable public `weights_to_radii` and `radii_to_weights` exports remain
identical function objects across `pyvoro2`, `pyvoro2.inverse`, and
`pyvoro2.inverse.separator`; only their internal implementation-module metadata
now names `pyvoro2._internal.weight_transforms`. Public signatures, defaults,
transform semantics, forward and inverse numerical results, record schemas,
and lazy native-extension loading are unchanged.

`pyvoro2.__about__` remains root-owned build metadata rather than a helper
module. Native `pyvoro2._core` and `pyvoro2._core2d` remain root-owned internal
extensions with their established names and loading behavior.

## Final release review checklist

- [x] Every preferred public import is listed with a lifecycle category.
- [x] Every removed compatibility alias has a canonical replacement and
      completed removal record.
- [x] `__all__` matches the intended namespace policy.
- [x] Forward output modes and diagnostic combinations are characterized.
- [x] Stable `TessellationResult` fields and mutable contained values are
      documented.
- [x] Exact current raw and inverse record keys are listed.
- [x] Preferred separator names and exact historical aliases are complete.
- [x] Active-set behavior is labelled experimental; the included sparse quadratic backend is labelled provisional and narrowly scoped.
- [x] Default changes and scientific semantics appear in migration notes and
      release notes.
- [x] The chemvoro-shaped integration workflow uses only stable or deliberately
      provisional public surfaces.
- [x] The inventory was re-audited against the v0.8 tree on 2026-07-24 for
      maintainer and release review.
