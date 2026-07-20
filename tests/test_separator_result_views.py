"""Layered separator result views introduced for v0.7 issue #13."""

from __future__ import annotations

import copy
from dataclasses import fields, replace
import importlib
import inspect
import json
from pathlib import Path
import pickle
import warnings

import numpy as np
import pytest


FIT_REPORT_KEYS = {
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
REALIZED_REPORT_KEYS = {
    'kind',
    'summary',
    'records',
    'unrealized',
    'unaccounted_pairs',
    'warnings',
    'tessellation_diagnostics',
}
ACTIVE_REPORT_KEYS = {
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


def test_successful_fit_views_share_existing_arrays_and_values() -> None:
    import pyvoro2.inverse as inverse
    import pyvoro2.inverse.separator as separator

    points = np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]], dtype=float)
    observations = inverse.resolve_separator_observations(
        points,
        [(0, 1, 0.25)],
        confidence=[0.75],
    )
    model = separator.FitModel(
        penalties=(separator.SoftIntervalPenalty(0.0, 1.0, 2.0),),
        regularization=separator.L2Regularization(strength=0.25),
    )
    fit = inverse.fit_weights_from_separators(
        points,
        observations,
        model=model,
        solver='admm',
        weight_shift=3.0,
        connectivity_check='diagnose',
    )
    assert '_originating_observations' not in {
        field.name for field in fields(type(fit))
    }
    binding_parameter = inspect.signature(type(fit)).parameters[
        '_originating_observations_init'
    ]
    assert binding_parameter.kind is inspect.Parameter.KEYWORD_ONLY
    assert binding_parameter.default is None

    state = fit.state
    assert state.mathematical_weights is fit.weights
    assert state.backend_radii is fit.radii
    assert state.global_representation_shift == fit.weight_shift == 3.0
    assert np.shares_memory(state.mathematical_weights, fit.weights)
    assert np.shares_memory(state.backend_radii, fit.radii)

    observation_view = fit.observation_view(observations)
    assert observation_view.targets is fit.target
    assert observation_view.target_fraction is observations.target_fraction
    assert observation_view.target_position is observations.target_position
    assert observation_view.confidence is observations.confidence
    assert observation_view.predictions is fit.predicted
    assert observation_view.predicted_fraction is fit.predicted_fraction
    assert observation_view.predicted_position is fit.predicted_position
    assert observation_view.residuals is fit.residuals
    assert observation_view.requested_shifts is fit.used_shifts
    np.testing.assert_array_equal(observation_view.targets, observations.target)
    np.testing.assert_array_equal(observation_view.confidence, [0.75])

    assert fit.objective is fit.objective_breakdown
    objective = fit.objective
    assert objective is not None
    assert objective.total == pytest.approx(
        objective.mismatch
        + objective.penalties_total
        + objective.regularization
    )
    assert objective.penalty_terms[0][0] == 'SoftIntervalPenalty'
    assert objective.penalties_total == pytest.approx(
        sum(value for _, value in objective.penalty_terms)
    )

    algebraic = fit.algebraic
    assert algebraic.edge_diagnostics is fit.edge_diagnostics
    assert algebraic.connectivity is fit.connectivity
    termination = fit.solver_termination
    assert termination.status == fit.status
    assert termination.status_detail == fit.status_detail
    assert termination.backend == fit.solver
    assert termination.n_iter == fit.n_iter
    assert termination.converged is fit.converged
    assert termination.hard_feasible is fit.hard_feasible
    assert termination.conflict is fit.conflict
    assert termination.warnings is fit.warnings

    report = fit.to_report(observations)
    assert set(report) == FIT_REPORT_KEYS
    assert json.loads(separator.dumps_report_json(report))['kind'] == (
        'power_weight_fit'
    )


def test_observation_view_rejects_a_different_resolved_observation_set() -> None:
    import pyvoro2.inverse as inverse

    points = np.array([[0.0, 0.0], [2.0, 0.0], [4.0, 0.0]], dtype=float)
    observations = inverse.resolve_separator_observations(
        points,
        [(0, 1, 0.25)],
        confidence=[0.75],
    )
    fit = inverse.fit_weights_from_separators(points, observations)

    other_targets = inverse.resolve_separator_observations(
        points,
        [(0, 1, 0.30)],
    )
    with pytest.raises(ValueError, match=r'\(target\)'):
        fit.observation_view(other_targets)

    other_measurement = replace(observations, measurement='position')
    with pytest.raises(ValueError, match=r'\(measurement\)'):
        fit.observation_view(other_measurement)

    # These have the same length, measurement, selected target, and shift as
    # the fitted rows.  Each was accepted by the former partial validation.
    other_pairs = inverse.resolve_separator_observations(
        points,
        [(1, 2, 0.25)],
        confidence=[0.75],
    )
    with pytest.raises(ValueError, match=r'\(i\)'):
        fit.observation_view(other_pairs)

    other_confidence = replace(observations, confidence=np.array([0.25]))
    with pytest.raises(ValueError, match=r'\(confidence\)'):
        fit.observation_view(other_confidence)

    other_geometry = inverse.resolve_separator_observations(
        np.array([[0.0, 0.0], [3.0, 0.0], [4.0, 0.0]], dtype=float),
        [(0, 1, 0.25)],
        confidence=[0.75],
    )
    with pytest.raises(ValueError, match=r'\(distance\)'):
        fit.observation_view(other_geometry)

    equivalent = inverse.resolve_separator_observations(
        points,
        [(0, 1, 0.25)],
        confidence=[0.75],
    )
    view = fit.observation_view(equivalent)
    assert view.confidence is equivalent.confidence

    without_connectivity = inverse.fit_weights_from_separators(
        points,
        observations,
        connectivity_check='none',
    )
    assert without_connectivity.observation_view(observations).confidence is (
        observations.confidence
    )


def test_observation_binding_survives_standard_reconstruction() -> None:
    import pyvoro2.inverse as inverse

    points = np.array([[0.0, 0.0], [2.0, 0.0]], dtype=float)
    observations = inverse.resolve_separator_observations(
        points,
        [(0, 1, 0.25)],
        confidence=[0.75],
    )
    fit = inverse.fit_weights_from_separators(points, observations)
    assert type(fit).__getstate__.__module__ == (
        'pyvoro2.inverse.separator.types'
    )
    assert type(fit).__setstate__.__module__ == (
        'pyvoro2.inverse.separator.types'
    )

    reconstructed = (
        copy.copy(fit),
        copy.deepcopy(fit),
        pickle.loads(pickle.dumps(fit)),
        replace(fit),
    )
    for rebuilt in reconstructed:
        view = rebuilt.observation_view(observations)
        np.testing.assert_array_equal(view.confidence, observations.confidence)
        np.testing.assert_array_equal(view.target_position, [0.5])

    inconsistent = replace(fit, target=np.array([0.5]))
    with pytest.raises(ValueError, match='fit result target'):
        inconsistent.observation_view(observations)


def test_identification_view_distinguishes_gauge_offsets_policy_and_shift() -> None:
    import pyvoro2.inverse as inverse
    import pyvoro2.inverse.separator as separator

    points = np.array(
        [[0.0, 0.0], [2.0, 0.0], [4.0, 0.0]],
        dtype=float,
    )
    observations = inverse.resolve_separator_observations(
        points,
        [(0, 1, 0.25), (1, 2, 0.75)],
        confidence=[1.0, 0.0],
    )
    fit = inverse.fit_weights_from_separators(
        points,
        observations,
        weight_shift=2.0,
        connectivity_check='diagnose',
    )

    identification = fit.identification
    assert identification.global_geometric_gauge_identified_by_data is False
    assert identification.effective_observation_components == ((0, 1), (2,))
    assert identification.relative_component_offsets_identified_by_data is False
    assert identification.component_offsets_selected_by_objective is False
    assert identification.component_alignment_policy == (
        fit.connectivity.gauge_policy
    )
    assert identification.unconstrained_sites == (2,)
    assert fit.connectivity.unconstrained_points == ()
    assert identification.connectivity is fit.connectivity
    assert fit.state.global_representation_shift == 2.0
    assert fit.state.global_representation_shift != (
        identification.component_alignment_policy
    )
    assert fit.algebraic.edge_diagnostics.edge_weight[1] == 0.0

    regularized = inverse.fit_weights_from_separators(
        points,
        observations,
        model=separator.FitModel(
            regularization=separator.L2Regularization(strength=0.5)
        ),
        connectivity_check='diagnose',
    )
    assert (
        regularized.identification.relative_component_offsets_identified_by_data
        is False
    )
    assert regularized.identification.component_offsets_selected_by_objective is True
    assert regularized.connectivity.offsets_identified_in_objective is True


def test_model_terms_do_not_claim_observational_offset_identification() -> None:
    import pyvoro2.inverse as inverse
    import pyvoro2.inverse.separator as separator

    points = np.array(
        [[0.0, 0.0], [2.0, 0.0], [4.0, 0.0]],
        dtype=float,
    )
    observations = inverse.resolve_separator_observations(
        points,
        [(0, 1, 0.25), (1, 2, 0.75)],
        confidence=[1.0, 0.0],
    )
    models = (
        separator.FitModel(feasible=separator.Interval(0.0, 1.0)),
        separator.FitModel(
            penalties=(separator.SoftIntervalPenalty(0.0, 1.0, 1.0),)
        ),
        separator.FitModel(
            penalties=(separator.SoftIntervalPenalty(0.0, 1.0, 0.0),)
        ),
    )
    for model in models:
        problem = separator.build_power_fit_problem(observations, model=model)
        # The historical problem mask still includes model-coupled rows for
        # numerical decomposition, but it no longer defines identification.
        np.testing.assert_array_equal(
            problem.offset_identifying_constraint_mask,
            [True, True],
        )
        assert problem.suggested_anchor_indices == ()
        candidate_weights = np.array([0.0, 2.0, 2.0])
        np.testing.assert_array_equal(
            problem.canonicalize_gauge(candidate_weights),
            candidate_weights,
        )
        constrained = inverse.fit_weights_from_separators(
            points,
            observations,
            model=model,
            solver='admm',
            connectivity_check='diagnose',
        )
        assert constrained.identification.effective_observation_components == (
            (0, 1),
            (2,),
        )
        assert constrained.identification.unconstrained_sites == (2,)
        assert (
            constrained.identification.relative_component_offsets_identified_by_data
            is False
        )
        assert (
            constrained.identification.component_offsets_selected_by_objective
            is False
        )
        assert constrained.connectivity.candidate_offsets_identified_by_data is False
        assert constrained.connectivity.offsets_identified_in_objective is False
        assert any(
            'informative observation graph' in message
            for message in constrained.connectivity.messages
        )
        assert all(
            'current objective' not in message
            for message in constrained.connectivity.messages
        )


def test_connected_informative_graph_identifies_relative_offsets() -> None:
    import pyvoro2.inverse as inverse

    points = np.array(
        [[0.0, 0.0], [2.0, 0.0], [4.0, 0.0]],
        dtype=float,
    )

    connected_observations = inverse.resolve_separator_observations(
        points,
        [(0, 1, 0.25), (1, 2, 0.75)],
    )
    connected = inverse.fit_weights_from_separators(
        points,
        connected_observations,
        connectivity_check='diagnose',
    )
    assert connected.identification.effective_observation_components == (
        (0, 1, 2),
    )
    assert connected.identification.relative_component_offsets_identified_by_data
    assert connected.identification.component_offsets_selected_by_objective is False


def test_active_set_identification_uses_informative_rows_only() -> None:
    from pyvoro2 import Box
    import pyvoro2.inverse.separator as separator

    points = np.array(
        [[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [4.0, 0.0, 0.0]],
        dtype=float,
    )
    box = Box(((-5.0, 9.0), (-5.0, 5.0), (-5.0, 5.0)))
    models = (
        separator.FitModel(feasible=separator.Interval(0.0, 1.0)),
        separator.FitModel(
            penalties=(separator.SoftIntervalPenalty(0.0, 1.0, 1.0),)
        ),
        separator.FitModel(
            penalties=(separator.SoftIntervalPenalty(0.0, 1.0, 0.0),)
        ),
    )
    for model in models:
        result = separator.solve_self_consistent_power_weights(
            points,
            [(0, 1, 0.5), (1, 2, 0.5)],
            confidence=[1.0, 0.0],
            domain=box,
            model=model,
            options=separator.ActiveSetOptions(max_iter=5),
            return_history=True,
            connectivity_check='diagnose',
            unaccounted_pair_check='diagnose',
        )
        assert np.array_equal(result.active_mask, [True, True])
        assert result.connectivity is not None
        assert result.connectivity.effective_graph.connected_components == (
            (0, 1),
            (2,),
        )
        assert result.connectivity.active_effective_graph is not None
        assert (
            result.connectivity.active_effective_graph.connected_components
            == ((0, 1), (2,))
        )
        assert result.connectivity.candidate_offsets_identified_by_data is False
        assert result.connectivity.active_offsets_identified_by_data is False
        assert result.connectivity.offsets_identified_in_objective is False
        assert result.inner_fit.identification.unconstrained_sites == (2,)
        assert result.history is not None
        assert all(
            row.fit_active_offsets_identified_by_data is False
            for row in result.history
        )


def test_infeasible_fit_views_preserve_conflict_and_none_arrays() -> None:
    import pyvoro2.inverse as inverse
    import pyvoro2.inverse.separator as separator

    points = np.array(
        [[0.0, 0.0], [2.0, 0.0], [4.0, 0.0]],
        dtype=float,
    )
    observations = inverse.resolve_separator_observations(
        points,
        [(0, 1, 0.0), (1, 2, 0.0), (0, 2, 0.0)],
        measurement='position',
    )
    fit = inverse.fit_weights_from_separators(
        points,
        observations,
        model=separator.FitModel(feasible=separator.FixedValue(0.0)),
        solver='admm',
        connectivity_check='diagnose',
    )

    assert fit.state.mathematical_weights is None
    assert fit.state.backend_radii is None
    observation_view = fit.observation_view(observations)
    assert observation_view.predictions is None
    assert observation_view.residuals is None
    assert fit.objective is None
    assert fit.algebraic.edge_diagnostics is fit.edge_diagnostics
    assert fit.algebraic.edge_diagnostics.residual is None
    assert fit.solver_termination.conflict is fit.conflict
    assert fit.solver_termination.hard_feasible is False
    assert fit.conflicting_constraint_indices == (0, 1, 2)

    report = fit.to_report(observations)
    assert set(report) == FIT_REPORT_KEYS
    loaded = json.loads(separator.dumps_report_json(report))
    assert loaded['summary']['status'] == 'infeasible_hard_constraints'
    assert loaded['conflict']['constraint_indices'] == [0, 1, 2]


def test_canonical_and_historical_result_paths_share_view_types() -> None:
    import pyvoro2
    import pyvoro2.inverse as inverse
    import pyvoro2.inverse.separator as separator

    with warnings.catch_warnings():
        warnings.simplefilter('ignore', DeprecationWarning)
        powerfit = importlib.import_module('pyvoro2.powerfit')

    assert powerfit.PowerWeightFitResult is separator.SeparatorFitResult
    assert pyvoro2.PowerWeightFitResult is separator.SeparatorFitResult
    points = np.array(
        [[0.0, 0.0], [2.0, 0.0], [4.0, 0.0]],
        dtype=float,
    )
    observations = inverse.resolve_separator_observations(
        points,
        [(0, 1, 0.25), (1, 2, 0.75)],
        confidence=[1.0, 0.0],
    )
    model = separator.FitModel(
        penalties=(separator.SoftIntervalPenalty(0.0, 1.0, 0.0),)
    )
    canonical = inverse.fit_weights_from_separators(
        points,
        observations,
        model=model,
        solver='admm',
    )
    historical = powerfit.fit_power_weights(
        points,
        observations,
        model=model,
        solver='admm',
    )

    assert type(canonical.state) is type(historical.state)
    assert type(canonical.identification) is type(historical.identification)
    assert type(canonical.observation_view(observations)) is type(
        historical.observation_view(observations)
    )
    assert type(canonical.algebraic) is type(historical.algebraic)
    assert type(canonical.solver_termination) is type(
        historical.solver_termination
    )
    np.testing.assert_array_equal(canonical.weights, historical.weights)
    assert canonical.identification == historical.identification
    assert (
        canonical.identification.relative_component_offsets_identified_by_data
        is False
    )
    assert canonical.identification.component_offsets_selected_by_objective is False


def test_realization_views_separate_matching_from_optional_geometry() -> None:
    from pyvoro2 import Box, PeriodicCell
    import pyvoro2.inverse as inverse
    import pyvoro2.inverse.separator as separator

    points = np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]], dtype=float)
    box = Box(((-5.0, 5.0), (-5.0, 5.0), (-5.0, 5.0)))
    observations = inverse.resolve_separator_observations(
        points,
        [(0, 1, 0.5)],
        domain=box,
    )
    fit = inverse.fit_weights_from_separators(points, observations)
    no_geometry = separator.match_realized_pairs(
        points,
        domain=box,
        radii=fit.radii,
        constraints=observations,
    )

    matching = no_geometry.requested_image_matching
    geometry = no_geometry.geometry
    assert matching.any_realization is no_geometry.realized
    assert matching.same_requested_shift is no_geometry.realized_same_shift
    assert matching.another_periodic_shift is no_geometry.realized_other_shift
    assert matching.realized_shifts is no_geometry.realized_shifts
    assert matching.unrealized_observation_indices is no_geometry.unrealized
    assert bool(matching.same_requested_shift[0]) is True
    assert geometry.endpoint_i_empty is no_geometry.endpoint_i_empty
    assert geometry.endpoint_j_empty is no_geometry.endpoint_j_empty
    assert geometry.boundary_measure is None
    assert geometry.cells is None
    assert geometry.tessellation_diagnostics is None

    with_geometry = separator.match_realized_pairs(
        points,
        domain=box,
        radii=fit.radii,
        constraints=observations,
        return_boundary_measure=True,
        return_cells=True,
        return_tessellation_diagnostics=True,
    )
    assert with_geometry.geometry.boundary_measure is with_geometry.boundary_measure
    assert with_geometry.geometry.cells is with_geometry.cells
    assert with_geometry.geometry.tessellation_diagnostics is (
        with_geometry.tessellation_diagnostics
    )
    assert with_geometry.geometry.boundary_measure[0] > 0.0
    report = with_geometry.to_report(observations)
    assert set(report) == REALIZED_REPORT_KEYS
    json.loads(separator.dumps_report_json(report))

    cell = PeriodicCell(
        vectors=((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))
    )
    periodic_points = np.array(
        [[0.1, 0.5, 0.5], [0.9, 0.5, 0.5]],
        dtype=float,
    )
    periodic_observations = inverse.resolve_separator_observations(
        periodic_points,
        [(0, 1, 0.5, (1, 0, 0))],
        domain=cell,
        image='given_only',
    )
    periodic_fit = inverse.fit_weights_from_separators(
        periodic_points,
        periodic_observations,
    )
    wrong_image = separator.match_realized_pairs(
        periodic_points,
        domain=cell,
        radii=periodic_fit.radii,
        constraints=periodic_observations,
    ).requested_image_matching
    assert bool(wrong_image.any_realization[0]) is True
    assert bool(wrong_image.same_requested_shift[0]) is False
    assert bool(wrong_image.another_periodic_shift[0]) is True
    assert (-1, 0, 0) in wrong_image.realized_shifts[0]


def test_active_set_views_keep_inner_final_outer_and_path_layers_separate() -> None:
    from pyvoro2 import Box
    import pyvoro2.inverse.separator as separator

    points = np.array(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]],
        dtype=float,
    )
    box = Box(((-5.0, 5.0), (-5.0, 5.0), (-5.0, 5.0)))
    result = separator.solve_self_consistent_power_weights(
        points,
        [(0, 1, 0.5), (1, 2, 0.5), (0, 2, 0.5)],
        domain=box,
        options=separator.ActiveSetOptions(max_iter=6),
        return_history=True,
        return_boundary_measure=True,
        connectivity_check='diagnose',
        unaccounted_pair_check='diagnose',
    )

    assert result.inner_fit is result.fit
    assert result.final_realization is result.realized
    assert result.candidate_diagnostics is result.diagnostics
    termination = result.outer_termination
    assert termination.status == result.termination
    assert termination.converged is result.converged
    assert termination.n_outer_iter == result.n_outer_iter
    assert termination.cycle_length == result.cycle_length
    assert termination.warnings is result.warnings
    path = result.path
    assert path.active_mask is result.active_mask
    assert np.shares_memory(path.active_mask, result.active_mask)
    assert path.marginal_constraint_indices is result.marginal_constraints
    assert path.history is result.history
    assert path.summary is result.path_summary
    assert path.history is not None
    assert path.summary is not None

    report = result.to_report()
    assert set(report) == ACTIVE_REPORT_KEYS
    assert set(report['fit']) == FIT_REPORT_KEYS
    assert set(report['realized']) == REALIZED_REPORT_KEYS
    loaded = json.loads(separator.dumps_report_json(report))
    assert loaded['summary']['termination'] == result.termination
    assert loaded['path_summary']['n_iterations'] == result.path_summary.n_iterations


def test_view_exports_are_canonical_only_and_high_level_surface_stays_small() -> None:
    import pyvoro2.inverse as inverse
    import pyvoro2.inverse.separator as separator

    view_names = {
        'SeparatorFitStateView',
        'SeparatorIdentificationView',
        'SeparatorObservationView',
        'SeparatorAlgebraicView',
        'SeparatorSolverTerminationView',
        'RequestedImageMatchView',
        'RealizedGeometryView',
        'ActiveSetTerminationView',
        'ActiveSetPathView',
    }
    assert view_names <= set(separator.__all__)
    assert tuple(inverse.__all__) == (
        'SeparatorObservations',
        'resolve_separator_observations',
        'SeparatorFitResult',
        'fit_weights_from_separators',
        'weights_to_radii',
        'radii_to_weights',
    )

    with warnings.catch_warnings():
        warnings.simplefilter('ignore', DeprecationWarning)
        powerfit = importlib.import_module('pyvoro2.powerfit')
    assert view_names.isdisjoint(powerfit.__all__)
    assert all(not hasattr(powerfit, name) for name in view_names)


def test_separator_all_matches_the_api_inventory() -> None:
    import pyvoro2.inverse.separator as separator

    inventory_path = (
        Path(__file__).resolve().parents[1]
        / 'docs'
        / 'development'
        / 'api-inventory.md'
    )
    inventory = inventory_path.read_text(encoding='utf-8')
    marker = (
        '`pyvoro2.inverse.separator.__all__` contains exactly the following '
        '56 names:\n\n```text\n'
    )
    exported_block = inventory.split(marker, maxsplit=1)[1].split(
        '\n```',
        maxsplit=1,
    )[0]
    assert tuple(exported_block.splitlines()) == tuple(separator.__all__)
