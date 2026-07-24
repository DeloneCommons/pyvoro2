"""Canonical inverse imports, signatures, and exported-schema regressions."""

from __future__ import annotations

from dataclasses import fields
import inspect

import numpy as np

import pyvoro2 as pv
import pyvoro2.planar as pv2
import pyvoro2.inverse.separator as separator
import pyvoro2.inverse.separator.active as separator_active
import pyvoro2.inverse.separator.constraints as separator_constraints
import pyvoro2.inverse.separator.model as separator_model
import pyvoro2.inverse.separator.realize as separator_realize
import pyvoro2.inverse.separator.report as separator_report
import pyvoro2.inverse.separator.solver as separator_solver
import pyvoro2.viz2d as viz2d
import pyvoro2.viz3d as viz3d


REQUIRED = inspect.Parameter.empty


TOP_LEVEL_ALL = (
    'Box',
    'OrthorhombicCell',
    'PeriodicCell',
    'TessellationResult',
    'compute',
    'locate',
    'ghost_cells',
    'TessellationDiagnostics',
    'TessellationIssue',
    'TessellationError',
    'analyze_tessellation',
    'validate_tessellation',
    'NormalizationDiagnostics',
    'NormalizationIssue',
    'NormalizationError',
    'validate_normalized_topology',
    'DuplicatePair',
    'DuplicateError',
    'duplicate_check',
    'annotate_face_properties',
    'NormalizedVertices',
    'NormalizedTopology',
    'normalize_vertices',
    'normalize_edges_faces',
    'normalize_topology',
    'radii_to_weights',
    'weights_to_radii',
    '__version__',
    'planar',
)

PLANAR_ALL = (
    'Box',
    'RectangularCell',
    'TessellationResult',
    'compute',
    'locate',
    'ghost_cells',
    'DuplicatePair',
    'DuplicateError',
    'duplicate_check',
    'annotate_edge_properties',
    'plot_tessellation',
    'TessellationIssue',
    'TessellationDiagnostics',
    'TessellationError',
    'analyze_tessellation',
    'validate_tessellation',
    'NormalizedVertices',
    'NormalizedTopology',
    'normalize_vertices',
    'normalize_edges',
    'normalize_topology',
    'NormalizationIssue',
    'NormalizationDiagnostics',
    'NormalizationError',
    'validate_normalized_topology',
)

SEPARATOR_ALL = (
    'SeparatorObservations',
    'resolve_separator_observations',
    'SeparatorFitProblem',
    'SeparatorFitResult',
    'fit_weights_from_separators',
    'SeparatorFitStateView',
    'SeparatorIdentificationView',
    'SeparatorObservationView',
    'SeparatorAlgebraicView',
    'SeparatorSolverTerminationView',
    'SeparatorObservationGraphView',
    'SeparatorQuadraticOperatorView',
    'SquaredLoss',
    'HuberLoss',
    'Interval',
    'FixedValue',
    'SoftIntervalPenalty',
    'ExponentialBoundaryPenalty',
    'ReciprocalBoundaryPenalty',
    'L2Regularization',
    'FitModel',
    'AlgebraicEdgeDiagnostics',
    'ConstraintGraphDiagnostics',
    'ConnectivityDiagnostics',
    'ConnectivityDiagnosticsError',
    'HardConstraintConflictTerm',
    'HardConstraintConflict',
    'PowerFitBounds',
    'PowerFitPredictions',
    'PowerFitObjectiveBreakdown',
    'build_power_fit_problem',
    'build_power_fit_result',
    'RequestedImageMatchView',
    'RealizedGeometryView',
    'RealizedPairDiagnostics',
    'UnaccountedRealizedPair',
    'UnaccountedRealizedPairError',
    'build_fit_report',
    'build_realized_report',
    'build_active_set_report',
    'dumps_report_json',
    'write_report_json',
    'ActiveSetOptions',
    'ActiveSetIteration',
    'ActiveSetPathSummary',
    'ActiveSetTerminationView',
    'ActiveSetPathView',
    'PairConstraintDiagnostics',
    'SelfConsistentPowerFitResult',
    'match_realized_pairs',
    'solve_self_consistent_power_weights',
    'radii_to_weights',
    'weights_to_radii',
)

SEPARATOR_REPORT_ALL = (
    'build_fit_report',
    'build_realized_report',
    'build_active_set_report',
    'dumps_report_json',
    'write_report_json',
)

SEPARATOR_SOLVER_ALL = (
    'fit_weights_from_separators',
    'ConnectivityDiagnosticsError',
)

VIZ3D_ALL = (
    'VizStyle',
    'make_view',
    'add_axes',
    'add_sites',
    'add_vertices',
    'add_domain_wireframe',
    'add_cell_wireframe',
    'add_tessellation_wireframe',
    'view_tessellation',
)


def _field_names(dataclass_type) -> tuple[str, ...]:
    return tuple(field.name for field in fields(dataclass_type))


def _parameter_defaults(callable_) -> tuple[tuple[str, object], ...]:
    return tuple(
        (name, parameter.default)
        for name, parameter in inspect.signature(callable_).parameters.items()
    )


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


def test_public_all_exports_are_characterized_exactly() -> None:
    assert len(pv.__all__) == len(TOP_LEVEL_ALL)
    assert set(pv.__all__) == set(TOP_LEVEL_ALL)
    assert len(pv2.__all__) == len(PLANAR_ALL)
    assert set(pv2.__all__) == set(PLANAR_ALL)
    assert len(separator.__all__) == len(SEPARATOR_ALL)
    assert set(separator.__all__) == set(SEPARATOR_ALL)
    assert len(separator_report.__all__) == len(SEPARATOR_REPORT_ALL)
    assert set(separator_report.__all__) == set(SEPARATOR_REPORT_ALL)
    assert len(separator_solver.__all__) == len(SEPARATOR_SOLVER_ALL)
    assert set(separator_solver.__all__) == set(SEPARATOR_SOLVER_ALL)
    assert len(viz3d.__all__) == len(VIZ3D_ALL)
    assert set(viz3d.__all__) == set(VIZ3D_ALL)


def test_documented_submodule_import_routes_are_characterized() -> None:
    assert viz2d.plot_tessellation is pv2.plot_tessellation

    package_routes = (
        (
            separator_constraints.SeparatorObservations,
            separator.SeparatorObservations,
        ),
        (
            separator_constraints.resolve_separator_observations,
            separator.resolve_separator_observations,
        ),
        (
            separator_active.solve_self_consistent_power_weights,
            separator.solve_self_consistent_power_weights,
        ),
        (separator_active.ActiveSetOptions, separator.ActiveSetOptions),
        (separator_active.ActiveSetIteration, separator.ActiveSetIteration),
        (separator_active.ActiveSetPathSummary, separator.ActiveSetPathSummary),
        (
            separator_active.PairConstraintDiagnostics,
            separator.PairConstraintDiagnostics,
        ),
        (
            separator_active.SelfConsistentPowerFitResult,
            separator.SelfConsistentPowerFitResult,
        ),
        (separator_model.SquaredLoss, separator.SquaredLoss),
        (separator_model.HuberLoss, separator.HuberLoss),
        (separator_model.Interval, separator.Interval),
        (separator_model.FixedValue, separator.FixedValue),
        (separator_model.SoftIntervalPenalty, separator.SoftIntervalPenalty),
        (
            separator_model.ExponentialBoundaryPenalty,
            separator.ExponentialBoundaryPenalty,
        ),
        (
            separator_model.ReciprocalBoundaryPenalty,
            separator.ReciprocalBoundaryPenalty,
        ),
        (separator_model.L2Regularization, separator.L2Regularization),
        (separator_model.FitModel, separator.FitModel),
        (
            separator_realize.RealizedPairDiagnostics,
            separator.RealizedPairDiagnostics,
        ),
        (
            separator_realize.UnaccountedRealizedPair,
            separator.UnaccountedRealizedPair,
        ),
        (
            separator_realize.UnaccountedRealizedPairError,
            separator.UnaccountedRealizedPairError,
        ),
        (
            separator_realize.match_realized_pairs,
            separator.match_realized_pairs,
        ),
        (separator_report.build_fit_report, separator.build_fit_report),
        (
            separator_report.build_realized_report,
            separator.build_realized_report,
        ),
        (
            separator_report.build_active_set_report,
            separator.build_active_set_report,
        ),
        (separator_report.dumps_report_json, separator.dumps_report_json),
        (separator_report.write_report_json, separator.write_report_json),
        (
            separator_solver.fit_weights_from_separators,
            separator.fit_weights_from_separators,
        ),
        (
            separator_solver.ConnectivityDiagnosticsError,
            separator.ConnectivityDiagnosticsError,
        ),
    )
    assert all(direct is packaged for direct, packaged in package_routes)

    assert issubclass(separator.SquaredLoss, separator_model.ScalarMismatch)
    assert issubclass(separator.Interval, separator_model.HardConstraint)
    assert issubclass(
        separator.SoftIntervalPenalty,
        separator_model.ScalarPenalty,
    )


def test_notebook_only_visualization_imports_are_characterized() -> None:
    assert viz3d.VizStyle.__name__ == 'VizStyle'
    assert callable(viz3d.view_tessellation)


def test_advanced_inverse_exports_stay_in_separator_namespace() -> None:
    advanced_problem_names = {
        'PowerFitBounds',
        'PowerFitPredictions',
        'PowerFitObjectiveBreakdown',
        'SeparatorFitProblem',
        'build_power_fit_problem',
        'build_power_fit_result',
    }
    assert advanced_problem_names <= set(separator.__all__)
    assert advanced_problem_names.isdisjoint(pv.__all__)
    assert set(SEPARATOR_ALL).isdisjoint(
        set(pv.__all__) - {'radii_to_weights', 'weights_to_radii'}
    )


def test_inverse_entrypoint_signatures_and_defaults_are_characterized() -> None:
    assert _parameter_defaults(separator.resolve_separator_observations) == (
        ('points', REQUIRED),
        ('constraints', REQUIRED),
        ('measurement', 'fraction'),
        ('domain', None),
        ('ids', None),
        ('index_mode', 'index'),
        ('image', 'nearest'),
        ('image_search', 1),
        ('confidence', None),
        ('allow_empty', False),
    )
    assert _parameter_defaults(separator.fit_weights_from_separators) == (
        ('points', REQUIRED),
        ('constraints', REQUIRED),
        ('measurement', 'fraction'),
        ('domain', None),
        ('ids', None),
        ('index_mode', 'index'),
        ('image', 'nearest'),
        ('image_search', 1),
        ('confidence', None),
        ('model', None),
        ('r_min', 0.0),
        ('weight_shift', None),
        ('solver', 'auto'),
        ('max_iter', 2000),
        ('rho', 1.0),
        ('tol_abs', 1e-6),
        ('tol_rel', 1e-5),
        ('connectivity_check', 'warn'),
    )
    assert _parameter_defaults(separator.build_power_fit_result) == (
        ('problem', REQUIRED),
        ('weights', REQUIRED),
        ('solver', 'external'),
        ('status', 'optimal'),
        ('status_detail', None),
        ('converged', True),
        ('n_iter', 0),
        ('warnings', ()),
        ('canonicalize_gauge', True),
        ('r_min', 0.0),
        ('weight_shift', None),
    )
    assert _parameter_defaults(separator.match_realized_pairs) == (
        ('points', REQUIRED),
        ('domain', REQUIRED),
        ('constraints', REQUIRED),
        ('weights', None),
        ('radii', None),
        ('return_boundary_measure', False),
        ('return_cells', False),
        ('return_tessellation_diagnostics', False),
        ('tessellation_check', 'diagnose'),
        ('unaccounted_pair_check', 'diagnose'),
    )
    assert _parameter_defaults(
        separator.solve_self_consistent_power_weights
    ) == (
        ('points', REQUIRED),
        ('constraints', REQUIRED),
        ('measurement', 'fraction'),
        ('domain', REQUIRED),
        ('ids', None),
        ('index_mode', 'index'),
        ('image', 'nearest'),
        ('image_search', 1),
        ('confidence', None),
        ('model', None),
        ('active0', None),
        ('options', None),
        ('r_min', 0.0),
        ('weight_shift', None),
        ('fit_solver', 'auto'),
        ('fit_max_iter', 2000),
        ('fit_rho', 1.0),
        ('fit_tol_abs', 1e-6),
        ('fit_tol_rel', 1e-5),
        ('return_history', False),
        ('return_cells', False),
        ('return_boundary_measure', False),
        ('return_tessellation_diagnostics', False),
        ('tessellation_check', 'diagnose'),
        ('connectivity_check', 'warn'),
        ('unaccounted_pair_check', 'warn'),
    )


def test_inverse_supporting_signatures_and_defaults_are_characterized() -> None:
    signatures = (
        (
            separator.build_power_fit_problem,
            (('constraints', REQUIRED), ('model', None)),
        ),
        (
            separator.build_fit_report,
            (
                ('result', REQUIRED),
                ('constraints', REQUIRED),
                ('use_ids', False),
            ),
        ),
        (
            separator.build_realized_report,
            (
                ('diagnostics', REQUIRED),
                ('constraints', REQUIRED),
                ('use_ids', False),
            ),
        ),
        (
            separator.build_active_set_report,
            (('result', REQUIRED), ('use_ids', False)),
        ),
        (
            separator.dumps_report_json,
            (
                ('report', REQUIRED),
                ('indent', 2),
                ('sort_keys', False),
            ),
        ),
        (
            separator.write_report_json,
            (
                ('report', REQUIRED),
                ('path', REQUIRED),
                ('indent', 2),
                ('sort_keys', False),
            ),
        ),
        (separator.SquaredLoss, ()),
        (separator.HuberLoss, (('delta', 1.0),)),
        (
            separator.Interval,
            (('lower', REQUIRED), ('upper', REQUIRED)),
        ),
        (separator.FixedValue, (('value', REQUIRED),)),
        (
            separator.SoftIntervalPenalty,
            (
                ('lower', REQUIRED),
                ('upper', REQUIRED),
                ('strength', REQUIRED),
            ),
        ),
        (
            separator.ExponentialBoundaryPenalty,
            (
                ('lower', 0.0),
                ('upper', 1.0),
                ('margin', 0.02),
                ('strength', 1.0),
                ('tau', 0.01),
            ),
        ),
        (
            separator.ReciprocalBoundaryPenalty,
            (
                ('lower', 0.0),
                ('upper', 1.0),
                ('margin', 0.05),
                ('strength', 1.0),
                ('epsilon', 1e-6),
            ),
        ),
        (
            separator.L2Regularization,
            (('strength', 0.0), ('reference', None)),
        ),
        (
            separator.ActiveSetOptions,
            (
                ('add_after', 1),
                ('drop_after', 2),
                ('relax', 1.0),
                ('max_iter', 25),
                ('cycle_window', 8),
                ('weight_step_tol', 1e-8),
            ),
        ),
        (
            separator.ConnectivityDiagnostics,
            (
                ('unconstrained_points', REQUIRED),
                ('candidate_graph', REQUIRED),
                ('effective_graph', REQUIRED),
                ('active_graph', None),
                ('active_effective_graph', None),
                ('candidate_offsets_identified_by_data', False),
                ('active_offsets_identified_by_data', None),
                ('offsets_identified_in_objective', False),
                ('gauge_policy', ''),
                ('messages', ()),
            ),
        ),
        (
            separator.UnaccountedRealizedPair,
            (
                ('site_i', REQUIRED),
                ('site_j', REQUIRED),
                ('realized_shifts', REQUIRED),
                ('boundary_measure', None),
            ),
        ),
        (
            separator.ActiveSetIteration,
            (
                ('iteration', REQUIRED),
                ('n_active', REQUIRED),
                ('n_realized', REQUIRED),
                ('n_added', REQUIRED),
                ('n_removed', REQUIRED),
                ('rms_residual_all', REQUIRED),
                ('max_residual_all', REQUIRED),
                ('weight_step_norm', REQUIRED),
                ('n_active_fit', None),
                ('fit_active_graph_n_components', None),
                ('fit_active_effective_graph_n_components', None),
                ('fit_active_offsets_identified_by_data', None),
                ('n_unaccounted_pairs', None),
            ),
        ),
        (
            separator.ActiveSetPathSummary,
            (
                ('n_iterations', REQUIRED),
                ('ever_fit_active_graph_disconnected', REQUIRED),
                ('ever_fit_active_effective_graph_disconnected', REQUIRED),
                ('ever_fit_active_offsets_unidentified_by_data', REQUIRED),
                ('ever_unaccounted_pairs', REQUIRED),
                ('max_fit_active_graph_components', REQUIRED),
                ('max_fit_active_effective_graph_components', REQUIRED),
                ('max_n_unaccounted_pairs', REQUIRED),
                ('first_fit_active_graph_disconnected_iter', None),
                ('first_fit_active_effective_graph_disconnected_iter', None),
                ('first_unaccounted_pairs_iter', None),
            ),
        ),
    )
    for callable_, expected in signatures:
        assert _parameter_defaults(callable_) == expected

    model_signature = inspect.signature(separator.FitModel)
    assert tuple(model_signature.parameters) == (
        'mismatch',
        'feasible',
        'penalties',
        'regularization',
    )
    model = separator.FitModel()
    assert isinstance(model.mismatch, separator.SquaredLoss)
    assert model.feasible is None
    assert model.penalties == ()
    assert isinstance(model.regularization, separator.L2Regularization)


def test_documented_inverse_method_signatures_are_characterized() -> None:
    signatures = (
        (
            separator.SeparatorObservations.pair_labels,
            (('self', REQUIRED), ('use_ids', False)),
        ),
        (
            separator.SeparatorObservations.to_records,
            (('self', REQUIRED), ('use_ids', False)),
        ),
        (
            separator.SeparatorObservations.subset,
            (('self', REQUIRED), ('mask', REQUIRED)),
        ),
        (
            separator.SeparatorFitProblem.canonicalize_gauge,
            (('self', REQUIRED), ('weights', REQUIRED)),
        ),
        (
            separator.SeparatorFitResult.to_records,
            (
                ('self', REQUIRED),
                ('constraints', REQUIRED),
                ('use_ids', False),
            ),
        ),
        (
            separator.SeparatorFitResult.to_report,
            (
                ('self', REQUIRED),
                ('constraints', REQUIRED),
                ('use_ids', False),
            ),
        ),
        (
            separator.RealizedPairDiagnostics.to_records,
            (
                ('self', REQUIRED),
                ('constraints', REQUIRED),
                ('use_ids', False),
            ),
        ),
        (
            separator.RealizedPairDiagnostics.unaccounted_records,
            (('self', REQUIRED), ('ids', None)),
        ),
        (
            separator.RealizedPairDiagnostics.to_report,
            (
                ('self', REQUIRED),
                ('constraints', REQUIRED),
                ('use_ids', False),
            ),
        ),
        (
            separator.PairConstraintDiagnostics.to_records,
            (('self', REQUIRED), ('ids', None)),
        ),
        (
            separator.SelfConsistentPowerFitResult.to_records,
            (('self', REQUIRED), ('use_ids', False)),
        ),
        (
            separator.SelfConsistentPowerFitResult.to_report,
            (('self', REQUIRED), ('use_ids', False)),
        ),
        (
            separator.HardConstraintConflictTerm.to_record,
            (('self', REQUIRED), ('ids', None)),
        ),
        (
            separator.HardConstraintConflict.to_records,
            (('self', REQUIRED), ('ids', None)),
        ),
        (
            separator.UnaccountedRealizedPair.to_record,
            (('self', REQUIRED), ('ids', None)),
        ),
    )
    for callable_, expected in signatures:
        assert _parameter_defaults(callable_) == expected


def test_inverse_positional_and_keyword_only_parameters_are_characterized() -> None:
    positional_parameters = (
        (
            separator.resolve_separator_observations,
            ('points', 'constraints'),
        ),
        (separator.fit_weights_from_separators, ('points', 'constraints')),
        (separator.build_power_fit_problem, ('constraints',)),
        (separator.build_power_fit_result, ('problem', 'weights')),
        (separator.match_realized_pairs, ('points',)),
        (
            separator.solve_self_consistent_power_weights,
            ('points', 'constraints'),
        ),
        (separator.build_fit_report, ('result', 'constraints')),
        (separator.build_realized_report, ('diagnostics', 'constraints')),
        (separator.build_active_set_report, ('result',)),
        (separator.dumps_report_json, ('report',)),
        (separator.write_report_json, ('report', 'path')),
        (separator.SquaredLoss, ()),
        (separator.HuberLoss, ('delta',)),
        (separator.Interval, ('lower', 'upper')),
        (separator.FixedValue, ('value',)),
        (
            separator.SoftIntervalPenalty,
            ('lower', 'upper', 'strength'),
        ),
        (
            separator.ExponentialBoundaryPenalty,
            ('lower', 'upper', 'margin', 'strength', 'tau'),
        ),
        (
            separator.ReciprocalBoundaryPenalty,
            ('lower', 'upper', 'margin', 'strength', 'epsilon'),
        ),
        (separator.L2Regularization, ('strength', 'reference')),
        (
            separator.FitModel,
            ('mismatch', 'feasible', 'penalties', 'regularization'),
        ),
        (
            separator.ActiveSetOptions,
            (
                'add_after',
                'drop_after',
                'relax',
                'max_iter',
                'cycle_window',
                'weight_step_tol',
            ),
        ),
        (
            separator.ConnectivityDiagnostics,
            (
                'unconstrained_points',
                'candidate_graph',
                'effective_graph',
                'active_graph',
                'active_effective_graph',
                'candidate_offsets_identified_by_data',
                'active_offsets_identified_by_data',
                'offsets_identified_in_objective',
                'gauge_policy',
                'messages',
            ),
        ),
        (
            separator.UnaccountedRealizedPair,
            ('site_i', 'site_j', 'realized_shifts', 'boundary_measure'),
        ),
        (
            separator.ActiveSetIteration,
            tuple(
                name
                for name, _ in _parameter_defaults(
                    separator.ActiveSetIteration
                )
            ),
        ),
        (
            separator.ActiveSetPathSummary,
            tuple(
                name
                for name, _ in _parameter_defaults(
                    separator.ActiveSetPathSummary
                )
            ),
        ),
        (separator.SeparatorObservations.pair_labels, ('self',)),
        (separator.SeparatorObservations.to_records, ('self',)),
        (separator.SeparatorObservations.subset, ('self', 'mask')),
        (
            separator.SeparatorFitProblem.canonicalize_gauge,
            ('self', 'weights'),
        ),
        (
            separator.SeparatorFitResult.to_records,
            ('self', 'constraints'),
        ),
        (
            separator.SeparatorFitResult.to_report,
            ('self', 'constraints'),
        ),
        (
            separator.RealizedPairDiagnostics.to_records,
            ('self', 'constraints'),
        ),
        (separator.RealizedPairDiagnostics.unaccounted_records, ('self',)),
        (
            separator.RealizedPairDiagnostics.to_report,
            ('self', 'constraints'),
        ),
        (separator.PairConstraintDiagnostics.to_records, ('self',)),
        (separator.SelfConsistentPowerFitResult.to_records, ('self',)),
        (separator.SelfConsistentPowerFitResult.to_report, ('self',)),
        (separator.HardConstraintConflictTerm.to_record, ('self',)),
        (separator.HardConstraintConflict.to_records, ('self',)),
        (separator.UnaccountedRealizedPair.to_record, ('self',)),
    )
    for callable_, expected in positional_parameters:
        _assert_positional_parameters(callable_, expected)


def test_public_inverse_result_fields_are_characterized() -> None:
    assert _field_names(separator.SeparatorObservations) == (
        'n_points',
        'i',
        'j',
        'shifts',
        'target',
        'confidence',
        'measurement',
        'distance',
        'distance2',
        'delta',
        'target_fraction',
        'target_position',
        'input_index',
        'explicit_shift',
        'ids',
        'warnings',
    )
    assert _field_names(separator.SeparatorFitProblem) == (
        'constraints',
        'model',
        'alpha',
        'beta',
        'z_obs',
        'edge_weight',
        'regularization_strength',
        'regularization_reference',
        'offset_identifying_constraint_mask',
        'bounds',
        'connectivity',
        'hard_feasible',
        'hard_conflict',
    )
    assert _field_names(separator.SeparatorFitResult) == (
        'status',
        'hard_feasible',
        'weights',
        'radii',
        'weight_shift',
        'measurement',
        'target',
        'predicted',
        'predicted_fraction',
        'predicted_position',
        'residuals',
        'rms_residual',
        'max_residual',
        'used_shifts',
        'solver',
        'n_iter',
        'converged',
        'conflict',
        'warnings',
        'status_detail',
        'connectivity',
        'edge_diagnostics',
        'objective_breakdown',
    )
    assert _field_names(separator.RealizedPairDiagnostics) == (
        'realized',
        'unrealized',
        'realized_same_shift',
        'realized_other_shift',
        'realized_shifts',
        'endpoint_i_empty',
        'endpoint_j_empty',
        'boundary_measure',
        'cells',
        'tessellation_diagnostics',
        'unaccounted_pairs',
        'warnings',
    )
    assert _field_names(separator.PairConstraintDiagnostics) == (
        'site_i',
        'site_j',
        'shift',
        'target',
        'confidence',
        'predicted',
        'predicted_fraction',
        'predicted_position',
        'residuals',
        'active',
        'realized',
        'realized_same_shift',
        'realized_other_shift',
        'realized_shifts',
        'endpoint_i_empty',
        'endpoint_j_empty',
        'boundary_measure',
        'toggle_count',
        'realized_toggle_count',
        'first_realized_iter',
        'last_realized_iter',
        'marginal',
        'status',
    )
    assert _field_names(separator.SelfConsistentPowerFitResult) == (
        'constraints',
        'fit',
        'realized',
        'diagnostics',
        'active_mask',
        'n_outer_iter',
        'converged',
        'termination',
        'cycle_length',
        'marginal_constraints',
        'rms_residual_all',
        'max_residual_all',
        'tessellation_diagnostics',
        'history',
        'path_summary',
        'warnings',
        'connectivity',
    )


def test_supporting_inverse_result_fields_are_characterized() -> None:
    expected_fields = (
        (
            separator.PowerFitBounds,
            (
                'measurement_lower',
                'measurement_upper',
                'difference_lower',
                'difference_upper',
            ),
        ),
        (
            separator.PowerFitPredictions,
            ('difference', 'fraction', 'position', 'measurement'),
        ),
        (
            separator.PowerFitObjectiveBreakdown,
            (
                'total',
                'mismatch',
                'penalties_total',
                'penalty_terms',
                'regularization',
                'hard_constraints_satisfied',
                'hard_max_violation',
            ),
        ),
        (
            separator.AlgebraicEdgeDiagnostics,
            (
                'alpha',
                'beta',
                'z_obs',
                'z_fit',
                'residual',
                'edge_weight',
                'weighted_l2',
                'weighted_rmse',
                'rmse',
                'mae',
            ),
        ),
        (
            separator.ConstraintGraphDiagnostics,
            (
                'n_points',
                'n_constraints',
                'n_edges',
                'isolated_points',
                'connected_components',
                'fully_connected',
            ),
        ),
        (
            separator.ConnectivityDiagnostics,
            (
                'unconstrained_points',
                'candidate_graph',
                'effective_graph',
                'active_graph',
                'active_effective_graph',
                'candidate_offsets_identified_by_data',
                'active_offsets_identified_by_data',
                'offsets_identified_in_objective',
                'gauge_policy',
                'messages',
            ),
        ),
        (
            separator.HardConstraintConflictTerm,
            (
                'constraint_index',
                'site_i',
                'site_j',
                'relation',
                'bound_value',
            ),
        ),
        (
            separator.HardConstraintConflict,
            ('component_nodes', 'cycle_nodes', 'terms', 'message'),
        ),
        (
            separator.UnaccountedRealizedPair,
            ('site_i', 'site_j', 'realized_shifts', 'boundary_measure'),
        ),
        (
            separator.ActiveSetIteration,
            (
                'iteration',
                'n_active',
                'n_realized',
                'n_added',
                'n_removed',
                'rms_residual_all',
                'max_residual_all',
                'weight_step_norm',
                'n_active_fit',
                'fit_active_graph_n_components',
                'fit_active_effective_graph_n_components',
                'fit_active_offsets_identified_by_data',
                'n_unaccounted_pairs',
            ),
        ),
        (
            separator.ActiveSetPathSummary,
            (
                'n_iterations',
                'ever_fit_active_graph_disconnected',
                'ever_fit_active_effective_graph_disconnected',
                'ever_fit_active_offsets_unidentified_by_data',
                'ever_unaccounted_pairs',
                'max_fit_active_graph_components',
                'max_fit_active_effective_graph_components',
                'max_n_unaccounted_pairs',
                'first_fit_active_graph_disconnected_iter',
                'first_fit_active_effective_graph_disconnected_iter',
                'first_unaccounted_pairs_iter',
            ),
        ),
    )
    for dataclass_type, expected in expected_fields:
        assert _field_names(dataclass_type) == expected


def test_inverse_record_and_report_schemas_are_characterized() -> None:
    points = np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
    domain = pv.Box(((-5.0, 5.0), (-5.0, 5.0), (-5.0, 5.0)))
    constraints = separator.resolve_separator_observations(
        points,
        [(0, 1, 0.25)],
        domain=domain,
    )
    fit = separator.fit_weights_from_separators(
        points,
        constraints,
        connectivity_check='diagnose',
    )
    realized = separator.match_realized_pairs(
        points,
        domain=domain,
        radii=fit.radii,
        constraints=constraints,
        return_boundary_measure=True,
        return_tessellation_diagnostics=True,
        unaccounted_pair_check='diagnose',
    )
    active = separator.solve_self_consistent_power_weights(
        points,
        constraints,
        domain=domain,
        options=separator.ActiveSetOptions(max_iter=4),
        return_history=True,
        return_boundary_measure=True,
        return_tessellation_diagnostics=True,
        connectivity_check='diagnose',
        unaccounted_pair_check='diagnose',
    )

    assert set(constraints.to_records()[0]) == {
        'constraint_index',
        'site_i',
        'site_j',
        'shift',
        'target',
        'confidence',
        'measurement',
        'distance',
        'target_fraction',
        'target_position',
        'input_index',
        'explicit_shift',
    }
    assert set(fit.to_records(constraints)[0]) == {
        'constraint_index',
        'site_i',
        'site_j',
        'shift',
        'measurement',
        'target',
        'predicted',
        'predicted_fraction',
        'predicted_position',
        'residual',
        'alpha',
        'beta',
        'z_obs',
        'z_fit',
        'algebraic_residual',
        'edge_weight',
    }
    assert set(realized.to_records(constraints)[0]) == {
        'constraint_index',
        'site_i',
        'site_j',
        'shift',
        'realized',
        'realized_same_shift',
        'realized_other_shift',
        'realized_shifts',
        'endpoint_i_empty',
        'endpoint_j_empty',
        'boundary_measure',
    }
    assert set(active.to_records()[0]) == {
        'constraint_index',
        'site_i',
        'site_j',
        'shift',
        'target',
        'confidence',
        'predicted',
        'predicted_fraction',
        'predicted_position',
        'residual',
        'active',
        'realized',
        'realized_same_shift',
        'realized_other_shift',
        'realized_shifts',
        'endpoint_i_empty',
        'endpoint_j_empty',
        'boundary_measure',
        'toggle_count',
        'realized_toggle_count',
        'first_realized_iter',
        'last_realized_iter',
        'marginal',
        'status',
    }

    fit_report = fit.to_report(constraints)
    assert set(fit_report) == {
        'kind',
        'summary',
        'constraints',
        'fit_records',
        'edge_diagnostics',
        'objective_breakdown',
        'weights',
        'radii',
        'weight_shift',
        'used_shifts',
        'warnings',
        'conflict',
        'connectivity',
    }
    assert set(fit_report['summary']) == {
        'status',
        'is_optimal',
        'is_infeasible',
        'hard_feasible',
        'solver',
        'measurement',
        'n_constraints',
        'n_points',
        'converged',
        'status_detail',
        'n_iter',
        'rms_residual',
        'max_residual',
        'conflicting_constraint_indices',
    }

    realized_report = realized.to_report(constraints)
    assert set(realized_report) == {
        'kind',
        'summary',
        'records',
        'unrealized',
        'unaccounted_pairs',
        'warnings',
        'tessellation_diagnostics',
    }
    assert set(realized_report['summary']) == {
        'n_constraints',
        'n_realized',
        'n_same_shift',
        'n_other_shift',
        'n_unrealized',
        'n_unaccounted_pairs',
    }

    active_report = active.to_report()
    assert set(active_report) == {
        'kind',
        'summary',
        'constraints',
        'fit',
        'realized',
        'diagnostics',
        'marginal_records',
        'history',
        'path_summary',
        'tessellation_diagnostics',
        'warnings',
        'connectivity',
    }
    assert set(active_report['summary']) == {
        'termination',
        'converged',
        'n_outer_iter',
        'cycle_length',
        'n_constraints',
        'n_active_final',
        'n_realized_final',
        'rms_residual_all',
        'max_residual_all',
        'marginal_constraint_indices',
    }
