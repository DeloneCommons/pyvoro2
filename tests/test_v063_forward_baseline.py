"""Characterization of the v0.6.3 forward compatibility surface.

These tests intentionally describe the baseline before v0.7 changes the
preferred result contract.  They should remain attached to the explicit raw
compatibility route introduced by the v0.7 result work.
"""

from __future__ import annotations

from dataclasses import fields
import inspect

import numpy as np

import pyvoro2 as pv
import pyvoro2.api as spatial_api
import pyvoro2.diagnostics as spatial_diagnostics
import pyvoro2.domains as spatial_domains
import pyvoro2.duplicates as spatial_duplicates
import pyvoro2.edge_properties as planar_edge_properties
import pyvoro2.face_properties as spatial_face_properties
import pyvoro2.normalize as spatial_normalize
import pyvoro2.planar as pv2
import pyvoro2.planar.api as planar_api
import pyvoro2.planar.diagnostics as planar_diagnostics
import pyvoro2.planar.domains as planar_domains
import pyvoro2.planar.normalize as planar_normalize
import pyvoro2.planar.result as planar_result
import pyvoro2.planar.validation as planar_validation
import pyvoro2.validation as spatial_validation
import pyvoro2.viz2d as viz2d
import pyvoro2.viz3d as viz3d


REQUIRED = inspect.Parameter.empty


def _parameter_defaults(callable_) -> tuple[tuple[str, object], ...]:
    return tuple(
        (name, parameter.default)
        for name, parameter in inspect.signature(callable_).parameters.items()
    )


def _field_names(dataclass_type) -> tuple[str, ...]:
    return tuple(field.name for field in fields(dataclass_type))


def _assert_positional_parameters(callable_, expected: tuple[str, ...]) -> None:
    parameters = inspect.signature(callable_).parameters.values()
    assert all(
        parameter.kind is not inspect.Parameter.POSITIONAL_ONLY
        for parameter in parameters
    )
    assert tuple(
        parameter.name
        for parameter in parameters
        if parameter.kind is inspect.Parameter.POSITIONAL_OR_KEYWORD
    ) == expected
    assert all(
        parameter.kind in {
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
            inspect.Parameter.KEYWORD_ONLY,
        }
        for parameter in parameters
    )


def test_documented_forward_module_routes_are_characterized() -> None:
    routes = (
        (spatial_api.compute, pv.compute),
        (spatial_api.locate, pv.locate),
        (spatial_api.ghost_cells, pv.ghost_cells),
        (spatial_domains.Box, pv.Box),
        (spatial_domains.OrthorhombicCell, pv.OrthorhombicCell),
        (spatial_domains.PeriodicCell, pv.PeriodicCell),
        (spatial_diagnostics.analyze_tessellation,
         pv.analyze_tessellation),
        (spatial_diagnostics.validate_tessellation,
         pv.validate_tessellation),
        (spatial_diagnostics.TessellationIssue,
         pv.TessellationIssue),
        (spatial_diagnostics.TessellationDiagnostics,
         pv.TessellationDiagnostics),
        (spatial_diagnostics.TessellationError,
         pv.TessellationError),
        (spatial_duplicates.DuplicatePair, pv.DuplicatePair),
        (spatial_duplicates.DuplicateError, pv.DuplicateError),
        (spatial_duplicates.duplicate_check, pv.duplicate_check),
        (spatial_face_properties.annotate_face_properties,
         pv.annotate_face_properties),
        (spatial_normalize.normalize_vertices, pv.normalize_vertices),
        (spatial_normalize.normalize_edges_faces,
         pv.normalize_edges_faces),
        (spatial_normalize.normalize_topology, pv.normalize_topology),
        (spatial_normalize.NormalizedVertices, pv.NormalizedVertices),
        (spatial_normalize.NormalizedTopology, pv.NormalizedTopology),
        (spatial_validation.NormalizationIssue,
         pv.NormalizationIssue),
        (spatial_validation.NormalizationDiagnostics,
         pv.NormalizationDiagnostics),
        (spatial_validation.NormalizationError,
         pv.NormalizationError),
        (spatial_validation.validate_normalized_topology,
         pv.validate_normalized_topology),
        (planar_api.compute, pv2.compute),
        (planar_api.locate, pv2.locate),
        (planar_api.ghost_cells, pv2.ghost_cells),
        (planar_domains.Box, pv2.Box),
        (planar_domains.RectangularCell, pv2.RectangularCell),
        (planar_diagnostics.analyze_tessellation,
         pv2.analyze_tessellation),
        (planar_diagnostics.validate_tessellation,
         pv2.validate_tessellation),
        (planar_diagnostics.TessellationIssue,
         pv2.TessellationIssue),
        (planar_diagnostics.TessellationDiagnostics,
         pv2.TessellationDiagnostics),
        (planar_diagnostics.TessellationError,
         pv2.TessellationError),
        (planar_edge_properties.annotate_edge_properties,
         pv2.annotate_edge_properties),
        (planar_normalize.normalize_vertices, pv2.normalize_vertices),
        (planar_normalize.normalize_edges, pv2.normalize_edges),
        (planar_normalize.normalize_topology, pv2.normalize_topology),
        (planar_normalize.NormalizedVertices, pv2.NormalizedVertices),
        (planar_normalize.NormalizedTopology, pv2.NormalizedTopology),
        (planar_result.PlanarComputeResult, pv2.PlanarComputeResult),
        (planar_validation.NormalizationIssue,
         pv2.NormalizationIssue),
        (planar_validation.NormalizationDiagnostics,
         pv2.NormalizationDiagnostics),
        (planar_validation.NormalizationError,
         pv2.NormalizationError),
        (planar_validation.validate_normalized_topology,
         pv2.validate_normalized_topology),
    )
    assert all(direct is packaged for direct, packaged in routes)


def test_spatial_compute_signature_and_defaults_are_characterized() -> None:
    defaults = _parameter_defaults(pv.compute)
    assert defaults[11] == ('weights', None)
    assert defaults[:11] + defaults[12:] == (
        ('points', REQUIRED),
        ('domain', REQUIRED),
        ('ids', None),
        ('duplicate_check', 'off'),
        ('duplicate_threshold', 1e-5),
        ('duplicate_wrap', True),
        ('duplicate_max_pairs', 10),
        ('block_size', None),
        ('blocks', None),
        ('init_mem', 8),
        ('mode', 'standard'),
        ('radii', None),
        ('return_vertices', True),
        ('return_adjacency', True),
        ('return_faces', True),
        ('return_face_shifts', False),
        ('face_shift_search', 2),
        ('include_empty', False),
        ('validate_face_shifts', True),
        ('repair_face_shifts', False),
        ('face_shift_tol', None),
        ('return_diagnostics', False),
        ('tessellation_check', 'none'),
        ('tessellation_require_reciprocity', None),
        ('tessellation_volume_tol_rel', 1e-8),
        ('tessellation_volume_tol_abs', 1e-12),
        ('tessellation_plane_offset_tol', None),
        ('tessellation_plane_angle_tol', None),
    )


def test_planar_compute_signature_and_defaults_are_characterized() -> None:
    defaults = _parameter_defaults(pv2.compute)
    assert defaults[11] == ('weights', None)
    assert defaults[:11] + defaults[12:] == (
        ('points', REQUIRED),
        ('domain', REQUIRED),
        ('ids', None),
        ('duplicate_check', 'off'),
        ('duplicate_threshold', 1e-5),
        ('duplicate_wrap', True),
        ('duplicate_max_pairs', 10),
        ('block_size', None),
        ('blocks', None),
        ('init_mem', 8),
        ('mode', 'standard'),
        ('radii', None),
        ('return_vertices', True),
        ('return_adjacency', True),
        ('return_edges', True),
        ('return_edge_shifts', False),
        ('edge_shift_search', 2),
        ('include_empty', False),
        ('validate_edge_shifts', True),
        ('repair_edge_shifts', False),
        ('edge_shift_tol', None),
        ('return_diagnostics', False),
        ('return_result', False),
        ('normalize', 'none'),
        ('normalization_tol', None),
        ('tessellation_check', 'none'),
        ('tessellation_require_reciprocity', None),
        ('tessellation_area_tol_rel', 1e-8),
        ('tessellation_area_tol_abs', 1e-12),
        ('tessellation_line_offset_tol', None),
        ('tessellation_line_angle_tol', None),
    )


def test_locate_and_ghost_signatures_are_characterized() -> None:
    common = (
        ('points', REQUIRED),
        ('queries', REQUIRED),
        ('domain', REQUIRED),
        ('ids', None),
        ('duplicate_check', 'off'),
        ('duplicate_threshold', 1e-5),
        ('duplicate_wrap', True),
        ('duplicate_max_pairs', 10),
    )
    spatial_dispatch = (
        ('block_size', None),
        ('blocks', None),
        ('init_mem', 8),
        ('mode', 'standard'),
        ('radii', None),
    )
    planar_dispatch = spatial_dispatch

    locate_tail = (('return_owner_position', False),)
    assert _parameter_defaults(pv.locate) == (
        common + spatial_dispatch + locate_tail
    )
    assert _parameter_defaults(pv2.locate) == (
        common + planar_dispatch + locate_tail
    )

    assert _parameter_defaults(pv.ghost_cells) == (
        common
        + spatial_dispatch
        + (
            ('ghost_radius', None),
            ('return_vertices', True),
            ('return_adjacency', True),
            ('return_faces', True),
            ('include_empty', True),
        )
    )
    assert _parameter_defaults(pv2.ghost_cells) == (
        common
        + planar_dispatch
        + (
            ('ghost_radius', None),
            ('return_vertices', True),
            ('return_adjacency', True),
            ('return_edges', True),
            ('return_edge_shifts', False),
            ('edge_shift_search', 2),
            ('include_empty', True),
            ('validate_edge_shifts', True),
            ('repair_edge_shifts', False),
            ('edge_shift_tol', None),
        )
    )


def test_documented_domain_signatures_are_characterized() -> None:
    signatures = (
        (pv.Box, (('bounds', REQUIRED),)),
        (
            pv.Box.from_points,
            (('points', REQUIRED), ('padding', 2.0)),
        ),
        (
            pv.OrthorhombicCell,
            (('bounds', REQUIRED), ('periodic', (True, True, True))),
        ),
        (
            pv.OrthorhombicCell.remap_cart,
            (
                ('self', REQUIRED),
                ('points', REQUIRED),
                ('return_shifts', False),
                ('eps', None),
            ),
        ),
        (
            pv.PeriodicCell,
            (('vectors', REQUIRED), ('origin', (0.0, 0.0, 0.0))),
        ),
        (
            pv.PeriodicCell.from_params,
            (
                ('bx', REQUIRED),
                ('bxy', REQUIRED),
                ('by', REQUIRED),
                ('bxz', REQUIRED),
                ('byz', REQUIRED),
                ('bz', REQUIRED),
                ('origin', (0.0, 0.0, 0.0)),
            ),
        ),
        (pv.PeriodicCell.to_internal_params, (('self', REQUIRED),)),
        (
            pv.PeriodicCell.cart_to_internal,
            (('self', REQUIRED), ('points', REQUIRED)),
        ),
        (
            pv.PeriodicCell.internal_to_cart,
            (('self', REQUIRED), ('points_internal', REQUIRED)),
        ),
        (
            pv.PeriodicCell.remap_internal,
            (
                ('self', REQUIRED),
                ('points_internal', REQUIRED),
                ('return_shifts', False),
                ('eps', None),
            ),
        ),
        (
            pv.PeriodicCell.wrap_internal,
            (('self', REQUIRED), ('points_internal', REQUIRED)),
        ),
        (
            pv.PeriodicCell.remap_cart,
            (
                ('self', REQUIRED),
                ('points', REQUIRED),
                ('return_shifts', False),
                ('eps', None),
            ),
        ),
        (pv2.Box, (('bounds', REQUIRED),)),
        (
            pv2.Box.from_points,
            (('points', REQUIRED), ('padding', 2.0)),
        ),
        (
            pv2.RectangularCell,
            (('bounds', REQUIRED), ('periodic', (True, True))),
        ),
        (
            pv2.RectangularCell.remap_cart,
            (
                ('self', REQUIRED),
                ('points', REQUIRED),
                ('return_shifts', False),
                ('eps', None),
            ),
        ),
    )
    for callable_, expected in signatures:
        assert _parameter_defaults(callable_) == expected


def test_supporting_forward_signatures_are_characterized() -> None:
    signatures = (
        (
            pv.analyze_tessellation,
            (
                ('cells', REQUIRED),
                ('domain', REQUIRED),
                ('expected_ids', None),
                ('mode', None),
                ('volume_tol_rel', 1e-8),
                ('volume_tol_abs', 1e-12),
                ('check_reciprocity', True),
                ('check_plane_mismatch', True),
                ('plane_offset_tol', None),
                ('plane_angle_tol', None),
                ('mark_faces', True),
            ),
        ),
        (
            pv.validate_tessellation,
            (
                ('cells', REQUIRED),
                ('domain', REQUIRED),
                ('expected_ids', None),
                ('mode', None),
                ('level', 'basic'),
                ('require_reciprocity', None),
                ('volume_tol_rel', 1e-8),
                ('volume_tol_abs', 1e-12),
                ('plane_offset_tol', None),
                ('plane_angle_tol', None),
                ('mark_faces', None),
            ),
        ),
        (
            pv2.analyze_tessellation,
            (
                ('cells', REQUIRED),
                ('domain', REQUIRED),
                ('expected_ids', None),
                ('mode', None),
                ('area_tol_rel', 1e-8),
                ('area_tol_abs', 1e-12),
                ('check_reciprocity', True),
                ('check_line_mismatch', True),
                ('line_offset_tol', None),
                ('line_angle_tol', None),
                ('mark_edges', True),
            ),
        ),
        (
            pv2.validate_tessellation,
            (
                ('cells', REQUIRED),
                ('domain', REQUIRED),
                ('expected_ids', None),
                ('mode', None),
                ('level', 'basic'),
                ('require_reciprocity', None),
                ('area_tol_rel', 1e-8),
                ('area_tol_abs', 1e-12),
                ('line_offset_tol', None),
                ('line_angle_tol', None),
                ('mark_edges', None),
            ),
        ),
        (
            pv.duplicate_check,
            (
                ('points', REQUIRED),
                ('threshold', 1e-5),
                ('domain', None),
                ('wrap', True),
                ('mode', 'raise'),
                ('max_pairs', 10),
            ),
        ),
        (
            pv.normalize_vertices,
            (
                ('cells', REQUIRED),
                ('domain', REQUIRED),
                ('tol', None),
                ('require_face_shifts', True),
                ('copy_cells', True),
            ),
        ),
        (
            pv.normalize_edges_faces,
            (
                ('nv', REQUIRED),
                ('domain', REQUIRED),
                ('tol', None),
                ('copy_cells', True),
            ),
        ),
        (
            pv.normalize_topology,
            (
                ('cells', REQUIRED),
                ('domain', REQUIRED),
                ('tol', None),
                ('require_face_shifts', True),
                ('copy_cells', True),
            ),
        ),
        (
            pv2.normalize_vertices,
            (
                ('cells', REQUIRED),
                ('domain', REQUIRED),
                ('tol', None),
                ('require_edge_shifts', True),
                ('copy_cells', True),
            ),
        ),
        (
            pv2.normalize_edges,
            (
                ('nv', REQUIRED),
                ('domain', REQUIRED),
                ('tol', None),
                ('copy_cells', True),
            ),
        ),
        (
            pv2.normalize_topology,
            (
                ('cells', REQUIRED),
                ('domain', REQUIRED),
                ('tol', None),
                ('require_edge_shifts', True),
                ('copy_cells', True),
            ),
        ),
        (
            pv.validate_normalized_topology,
            (
                ('normalized', REQUIRED),
                ('domain', REQUIRED),
                ('level', 'basic'),
                ('check_vertex_face_shift', True),
                ('check_face_vertex_sets', True),
                ('check_incidence', True),
                ('check_euler', True),
                ('max_examples', 10),
            ),
        ),
        (
            pv2.validate_normalized_topology,
            (
                ('normalized', REQUIRED),
                ('domain', REQUIRED),
                ('level', 'basic'),
                ('check_vertex_edge_shift', True),
                ('check_edge_vertex_sets', True),
                ('check_incidence', True),
                ('check_polygon', True),
                ('max_examples', 10),
            ),
        ),
        (
            planar_result.PlanarComputeResult.require_tessellation_diagnostics,
            (('self', REQUIRED),),
        ),
        (
            planar_result.PlanarComputeResult.require_normalized_vertices,
            (('self', REQUIRED),),
        ),
        (
            planar_result.PlanarComputeResult.require_normalized_topology,
            (('self', REQUIRED),),
        ),
        (
            pv.annotate_face_properties,
            (
                ('cells', REQUIRED),
                ('domain', REQUIRED),
                ('diagnostics', None),
                ('tol', 1e-10),
            ),
        ),
        (
            pv2.annotate_edge_properties,
            (
                ('cells', REQUIRED),
                ('domain', REQUIRED),
                ('tol', 1e-12),
            ),
        ),
        (
            pv.weights_to_radii,
            (
                ('weights', REQUIRED),
                ('r_min', 0.0),
                ('weight_shift', None),
            ),
        ),
        (pv.radii_to_weights, (('radii', REQUIRED),)),
    )
    for callable_, expected in signatures:
        assert _parameter_defaults(callable_) == expected


def test_visualization_signatures_are_characterized() -> None:
    signatures = (
        (
            viz2d.plot_tessellation,
            (
                ('cells', REQUIRED),
                ('ax', None),
                ('domain', None),
                ('show_sites', False),
                ('annotate_ids', False),
            ),
        ),
        (
            viz3d.VizStyle,
            (
                ('background', '0xffffff'),
                ('site_color', '0x777777'),
                ('site_radius', 0.093),
                ('site_label_color', '0x000000'),
                ('site_label_background', '0xffffff'),
                ('site_label_font_size', 8),
                ('edge_color', '0x1f77b4'),
                ('edge_line_width', 2.5),
                ('domain_color', '0x000000'),
                ('domain_line_width', 2.5),
                ('vertex_color', '0xff7f0e'),
                ('vertex_radius', 0.04),
                ('vertex_label_color', '0x000000'),
                ('vertex_label_background', '0xffffff'),
                ('vertex_label_font_size', 7),
                ('axes_line_width', 2.0),
                ('axes_label_font_size', 12),
                ('axes_color_x', '0xff0000'),
                ('axes_color_y', '0x00aa00'),
                ('axes_color_z', '0x0000ff'),
            ),
        ),
        (
            viz3d.make_view,
            (
                ('width', 640),
                ('height', 480),
                ('background', '0xffffff'),
            ),
        ),
        (
            viz3d.add_axes,
            (
                ('view', REQUIRED),
                ('origin', (0.0, 0.0, 0.0)),
                ('length', 1.0),
                ('line_width', 2.0),
                ('label_font_size', 12),
                ('color_x', '0xff0000'),
                ('color_y', '0x00aa00'),
                ('color_z', '0x0000ff'),
            ),
        ),
        (
            viz3d.add_sites,
            (
                ('view', REQUIRED),
                ('points', REQUIRED),
                ('labels', None),
                ('color', '0x777777'),
                ('radius', 0.093),
                ('label_color', '0x000000'),
                ('label_background', '0xffffff'),
                ('label_font_size', 8),
            ),
        ),
        (
            viz3d.add_vertices,
            (
                ('view', REQUIRED),
                ('vertices', REQUIRED),
                ('labels', None),
                ('color', '0xff7f0e'),
                ('radius', 0.04),
                ('label_color', '0x000000'),
                ('label_background', '0xffffff'),
                ('label_font_size', 7),
            ),
        ),
        (
            viz3d.add_domain_wireframe,
            (
                ('view', REQUIRED),
                ('domain', REQUIRED),
                ('color', '0x000000'),
                ('line_width', 2.5),
            ),
        ),
        (
            viz3d.add_cell_wireframe,
            (
                ('view', REQUIRED),
                ('cell', REQUIRED),
                ('color', '0x1f77b4'),
                ('line_width', 2.5),
            ),
        ),
        (
            viz3d.add_tessellation_wireframe,
            (
                ('view', REQUIRED),
                ('cells', REQUIRED),
                ('color', '0x1f77b4'),
                ('line_width', 2.5),
                ('cell_ids', None),
            ),
        ),
        (
            viz3d.view_tessellation,
            (
                ('cells', REQUIRED),
                ('domain', None),
                ('show_sites', True),
                ('show_site_labels', True),
                ('max_site_labels', 200),
                ('show_domain', True),
                ('show_axes', True),
                ('axes_length', None),
                ('wrap_cells', False),
                ('cell_ids', None),
                ('show_vertices', False),
                ('show_vertex_labels', 'auto'),
                ('max_vertex_labels', 200),
                ('style', None),
                ('width', 640),
                ('height', 480),
                ('zoom', True),
            ),
        ),
    )
    for callable_, expected in signatures:
        assert _parameter_defaults(callable_) == expected


def test_forward_positional_and_keyword_only_parameters_are_characterized() -> None:
    positional_parameters = (
        (pv.compute, ('points',)),
        (pv.locate, ('points', 'queries')),
        (pv.ghost_cells, ('points', 'queries')),
        (pv2.compute, ('points',)),
        (pv2.locate, ('points', 'queries')),
        (pv2.ghost_cells, ('points', 'queries')),
        (pv.Box, ('bounds',)),
        (pv.Box.from_points, ('points', 'padding')),
        (pv.OrthorhombicCell, ('bounds', 'periodic')),
        (pv.OrthorhombicCell.remap_cart, ('self', 'points')),
        (pv.PeriodicCell, ('vectors', 'origin')),
        (
            pv.PeriodicCell.from_params,
            ('bx', 'bxy', 'by', 'bxz', 'byz', 'bz'),
        ),
        (pv.PeriodicCell.to_internal_params, ('self',)),
        (pv.PeriodicCell.cart_to_internal, ('self', 'points')),
        (pv.PeriodicCell.internal_to_cart, ('self', 'points_internal')),
        (pv.PeriodicCell.remap_internal, ('self', 'points_internal')),
        (pv.PeriodicCell.wrap_internal, ('self', 'points_internal')),
        (pv.PeriodicCell.remap_cart, ('self', 'points')),
        (pv2.Box, ('bounds',)),
        (pv2.Box.from_points, ('points', 'padding')),
        (pv2.RectangularCell, ('bounds', 'periodic')),
        (pv2.RectangularCell.remap_cart, ('self', 'points')),
        (pv.analyze_tessellation, ('cells', 'domain')),
        (pv.validate_tessellation, ('cells', 'domain')),
        (pv2.analyze_tessellation, ('cells', 'domain')),
        (pv2.validate_tessellation, ('cells', 'domain')),
        (pv.duplicate_check, ('points',)),
        (pv.normalize_vertices, ('cells',)),
        (pv.normalize_edges_faces, ('nv',)),
        (pv.normalize_topology, ('cells',)),
        (pv2.normalize_vertices, ('cells',)),
        (pv2.normalize_edges, ('nv',)),
        (pv2.normalize_topology, ('cells',)),
        (pv.validate_normalized_topology, ('normalized', 'domain')),
        (pv2.validate_normalized_topology, ('normalized', 'domain')),
        (
            planar_result.PlanarComputeResult.require_tessellation_diagnostics,
            ('self',),
        ),
        (
            planar_result.PlanarComputeResult.require_normalized_vertices,
            ('self',),
        ),
        (
            planar_result.PlanarComputeResult.require_normalized_topology,
            ('self',),
        ),
        (pv.annotate_face_properties, ('cells', 'domain')),
        (pv2.annotate_edge_properties, ('cells', 'domain')),
        (pv.weights_to_radii, ('weights',)),
        (pv.radii_to_weights, ('radii',)),
        (viz2d.plot_tessellation, ('cells',)),
        (
            viz3d.VizStyle,
            tuple(name for name, _ in _parameter_defaults(viz3d.VizStyle)),
        ),
        (viz3d.make_view, ()),
        (viz3d.add_axes, ('view',)),
        (viz3d.add_sites, ('view', 'points')),
        (viz3d.add_vertices, ('view', 'vertices')),
        (viz3d.add_domain_wireframe, ('view', 'domain')),
        (viz3d.add_cell_wireframe, ('view', 'cell')),
        (viz3d.add_tessellation_wireframe, ('view', 'cells')),
        (viz3d.view_tessellation, ('cells',)),
    )
    for callable_, expected in positional_parameters:
        _assert_positional_parameters(callable_, expected)


def test_forward_result_and_diagnostic_fields_are_characterized() -> None:
    expected_fields = (
        (
            spatial_diagnostics.TessellationIssue,
            ('code', 'severity', 'message', 'examples'),
        ),
        (
            spatial_diagnostics.TessellationDiagnostics,
            (
                'domain_volume',
                'sum_cell_volume',
                'volume_ratio',
                'volume_gap',
                'volume_overlap',
                'n_sites_expected',
                'n_cells_returned',
                'missing_ids',
                'empty_ids',
                'face_shift_available',
                'reciprocity_checked',
                'n_faces_total',
                'n_faces_orphan',
                'n_faces_mismatched',
                'issues',
                'ok_volume',
                'ok_reciprocity',
                'ok',
            ),
        ),
        (
            spatial_validation.NormalizationIssue,
            ('code', 'severity', 'message', 'examples'),
        ),
        (spatial_duplicates.DuplicatePair, ('i', 'j', 'distance')),
        (
            spatial_validation.NormalizationDiagnostics,
            (
                'n_cells',
                'n_global_vertices',
                'n_global_edges',
                'n_global_faces',
                'is_periodic_domain',
                'fully_periodic_domain',
                'has_wall_faces',
                'n_vertex_face_shift_mismatch',
                'n_face_vertex_set_mismatch',
                'n_vertices_low_incidence',
                'n_edges_low_incidence',
                'n_cells_bad_euler',
                'issues',
                'ok_vertex_face_shift',
                'ok_face_vertex_sets',
                'ok_incidence',
                'ok_euler',
                'ok',
            ),
        ),
        (spatial_normalize.NormalizedVertices, ('global_vertices', 'cells')),
        (
            spatial_normalize.NormalizedTopology,
            ('global_vertices', 'global_edges', 'global_faces', 'cells'),
        ),
        (
            planar_diagnostics.TessellationIssue,
            ('code', 'severity', 'message', 'examples'),
        ),
        (
            planar_diagnostics.TessellationDiagnostics,
            (
                'domain_area',
                'sum_cell_area',
                'area_ratio',
                'area_gap',
                'area_overlap',
                'n_sites_expected',
                'n_cells_returned',
                'missing_ids',
                'empty_ids',
                'edge_shift_available',
                'reciprocity_checked',
                'n_edges_total',
                'n_edges_orphan',
                'n_edges_mismatched',
                'issues',
                'ok_area',
                'ok_reciprocity',
                'ok',
            ),
        ),
        (
            planar_validation.NormalizationIssue,
            ('code', 'severity', 'message', 'examples'),
        ),
        (
            planar_validation.NormalizationDiagnostics,
            (
                'n_cells',
                'n_global_vertices',
                'n_global_edges',
                'is_periodic_domain',
                'fully_periodic_domain',
                'has_wall_edges',
                'n_vertex_edge_shift_mismatch',
                'n_edge_vertex_set_mismatch',
                'n_vertices_low_incidence',
                'n_cells_bad_polygon',
                'issues',
                'ok_vertex_edge_shift',
                'ok_edge_vertex_sets',
                'ok_incidence',
                'ok_polygon',
                'ok',
            ),
        ),
        (planar_normalize.NormalizedVertices, ('global_vertices', 'cells')),
        (
            planar_normalize.NormalizedTopology,
            ('global_vertices', 'global_edges', 'cells'),
        ),
        (
            planar_result.PlanarComputeResult,
            (
                'cells',
                'tessellation_diagnostics',
                'normalized_vertices',
                'normalized_topology',
            ),
        ),
    )
    for dataclass_type, expected in expected_fields:
        assert _field_names(dataclass_type) == expected


def test_spatial_raw_return_variants_are_characterized() -> None:
    points = np.array([[0.25, 0.5, 0.5], [0.75, 0.5, 0.5]])
    domain = pv.Box(((0.0, 1.0), (0.0, 1.0), (0.0, 1.0)))

    cells = pv.compute(points, domain=domain)
    diagnosed_cells = pv.compute(
        points,
        domain=domain,
        tessellation_check='diagnose',
    )
    cells_with_diagnostics = pv.compute(
        points,
        domain=domain,
        return_diagnostics=True,
    )

    assert isinstance(cells, list)
    assert isinstance(diagnosed_cells, list)
    assert isinstance(cells_with_diagnostics, tuple)
    assert len(cells_with_diagnostics) == 2
    assert isinstance(cells_with_diagnostics[0], list)
    assert isinstance(cells_with_diagnostics[1], pv.TessellationDiagnostics)


def test_planar_raw_and_structured_return_variants_are_characterized() -> None:
    points = np.array([[0.25, 0.5], [0.75, 0.5]])
    domain = pv2.Box(((0.0, 1.0), (0.0, 1.0)))

    cells = pv2.compute(points, domain=domain)
    diagnosed_cells = pv2.compute(
        points,
        domain=domain,
        tessellation_check='diagnose',
    )
    cells_with_diagnostics = pv2.compute(
        points,
        domain=domain,
        return_diagnostics=True,
    )
    result = pv2.compute(
        points,
        domain=domain,
        return_result=True,
        return_diagnostics=True,
    )
    normalized_result = pv2.compute(
        points,
        domain=domain,
        return_vertices=False,
        return_adjacency=False,
        return_edges=False,
        normalize='vertices',
    )

    assert isinstance(cells, list)
    assert isinstance(diagnosed_cells, list)
    assert isinstance(cells_with_diagnostics, tuple)
    assert len(cells_with_diagnostics) == 2
    assert isinstance(cells_with_diagnostics[1], pv2.TessellationDiagnostics)

    assert isinstance(result, pv2.PlanarComputeResult)
    assert result.has_tessellation_diagnostics is True
    assert isinstance(normalized_result, pv2.PlanarComputeResult)
    assert normalized_result.has_normalized_vertices is True
    assert normalized_result.has_tessellation_diagnostics is False


def test_raw_cell_keys_external_ids_and_order_are_characterized() -> None:
    points3 = np.array([[0.25, 0.5, 0.5], [0.75, 0.5, 0.5]])
    cells3 = pv.compute(
        points3,
        domain=pv.Box(((0.0, 1.0), (0.0, 1.0), (0.0, 1.0))),
        ids=[20, 10],
    )
    assert [cell['id'] for cell in cells3] == [20, 10]
    assert set(cells3[0]) == {
        'id',
        'volume',
        'site',
        'vertices',
        'adjacency',
        'faces',
    }
    assert set(cells3[0]['faces'][0]) == {'adjacent_cell', 'vertices'}

    points2 = np.array([[0.25, 0.5], [0.75, 0.5]])
    cells2 = pv2.compute(
        points2,
        domain=pv2.Box(((0.0, 1.0), (0.0, 1.0))),
        ids=[20, 10],
    )
    assert [cell['id'] for cell in cells2] == [20, 10]
    assert set(cells2[0]) == {
        'id',
        'area',
        'site',
        'vertices',
        'adjacency',
        'edges',
    }
    assert set(cells2[0]['edges'][0]) == {'adjacent_cell', 'vertices'}


def test_raw_optional_geometry_keys_are_omitted_when_disabled() -> None:
    points3 = np.array([[0.25, 0.5, 0.5], [0.75, 0.5, 0.5]])
    cells3 = pv.compute(
        points3,
        domain=pv.Box(((0.0, 1.0), (0.0, 1.0), (0.0, 1.0))),
        return_vertices=False,
        return_adjacency=False,
        return_faces=False,
    )
    assert set(cells3[0]) == {'id', 'volume', 'site'}

    points2 = np.array([[0.25, 0.5], [0.75, 0.5]])
    cells2 = pv2.compute(
        points2,
        domain=pv2.Box(((0.0, 1.0), (0.0, 1.0))),
        return_vertices=False,
        return_adjacency=False,
        return_edges=False,
    )
    assert set(cells2[0]) == {'id', 'area', 'site'}


def test_standard_include_empty_is_a_noop_in_both_dimensions() -> None:
    points3 = np.array([[0.25, 0.5, 0.5], [0.75, 0.5, 0.5]])
    common3 = {
        'domain': pv.Box(((0.0, 1.0), (0.0, 1.0), (0.0, 1.0))),
        'ids': [20, 10],
        'return_vertices': False,
        'return_adjacency': False,
        'return_faces': False,
    }
    omitted3 = pv.compute(points3, include_empty=False, **common3)
    included3 = pv.compute(points3, include_empty=True, **common3)
    assert [cell['id'] for cell in omitted3] == [20, 10]
    assert [cell['id'] for cell in included3] == [20, 10]
    assert all('empty' not in cell for cell in included3)
    np.testing.assert_allclose(
        [cell['volume'] for cell in included3],
        [cell['volume'] for cell in omitted3],
    )

    points2 = np.array([[0.25, 0.5], [0.75, 0.5]])
    common2 = {
        'domain': pv2.Box(((0.0, 1.0), (0.0, 1.0))),
        'ids': [20, 10],
        'return_vertices': False,
        'return_adjacency': False,
        'return_edges': False,
    }
    omitted2 = pv2.compute(points2, include_empty=False, **common2)
    included2 = pv2.compute(points2, include_empty=True, **common2)
    assert [cell['id'] for cell in omitted2] == [20, 10]
    assert [cell['id'] for cell in included2] == [20, 10]
    assert all('empty' not in cell for cell in included2)
    np.testing.assert_allclose(
        [cell['area'] for cell in included2],
        [cell['area'] for cell in omitted2],
    )


def test_hidden_power_cells_are_omitted_or_reinserted_in_both_dimensions() -> None:
    points3 = np.array([[0.25, 0.5, 0.5], [0.75, 0.5, 0.5]])
    common3 = {
        'domain': pv.Box(((0.0, 1.0), (0.0, 1.0), (0.0, 1.0))),
        'mode': 'power',
        'radii': [1.0, 2.0],
        'ids': [20, 10],
        'return_vertices': False,
        'return_adjacency': False,
        'return_faces': False,
    }
    omitted3 = pv.compute(points3, include_empty=False, **common3)
    included3 = pv.compute(points3, include_empty=True, **common3)
    assert [cell['id'] for cell in omitted3] == [10]
    assert [cell['id'] for cell in included3] == [20, 10]
    assert included3[0] == {
        'id': 20,
        'empty': True,
        'volume': 0.0,
        'site': [0.25, 0.5, 0.5],
    }
    assert 'empty' not in included3[1]

    points2 = np.array([[0.25, 0.5], [0.75, 0.5]])
    common2 = {
        'domain': pv2.Box(((0.0, 1.0), (0.0, 1.0))),
        'mode': 'power',
        'radii': [1.0, 2.0],
        'ids': [20, 10],
        'return_vertices': False,
        'return_adjacency': False,
        'return_edges': False,
    }
    omitted2 = pv2.compute(points2, include_empty=False, **common2)
    included2 = pv2.compute(points2, include_empty=True, **common2)
    assert [cell['id'] for cell in omitted2] == [10]
    assert [cell['id'] for cell in included2] == [20, 10]
    assert included2[0] == {
        'id': 20,
        'empty': True,
        'area': 0.0,
        'site': [0.25, 0.5],
    }
    assert 'empty' not in included2[1]
