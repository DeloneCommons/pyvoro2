import json
from dataclasses import replace

import numpy as np
import pytest


def _fit_result(constraints, *, status, weights, converged):
    from pyvoro2.inverse.separator import weights_to_radii
    from pyvoro2.inverse.separator.types import SeparatorFitResult

    if weights is None:
        radii = None
        weight_shift = None
        predicted = None
        residuals = None
        rms_residual = None
        max_residual = None
    else:
        weights = np.asarray(weights, dtype=np.float64)
        radii, weight_shift = weights_to_radii(weights)
        predicted = np.zeros(constraints.n_constraints, dtype=np.float64)
        residuals = np.zeros(constraints.n_constraints, dtype=np.float64)
        rms_residual = 0.0
        max_residual = 0.0

    return SeparatorFitResult(
        status=status,
        hard_feasible=status != 'infeasible_hard_constraints',
        weights=weights,
        radii=radii,
        weight_shift=weight_shift,
        measurement=constraints.measurement,
        target=constraints.target.copy(),
        predicted=predicted,
        predicted_fraction=predicted,
        predicted_position=predicted,
        residuals=residuals,
        rms_residual=rms_residual,
        max_residual=max_residual,
        used_shifts=constraints.shifts.copy(),
        solver='direct',
        n_iter=1,
        converged=converged,
        conflict=None,
        warnings=(f'synthetic {status}',),
    )


def _realization(same):
    from pyvoro2.inverse.separator.realize import RealizedPairDiagnostics

    same = np.asarray(same, dtype=bool)
    dim = 3
    return RealizedPairDiagnostics(
        realized=same.copy(),
        unrealized=tuple(np.flatnonzero(~same).tolist()),
        realized_same_shift=same.copy(),
        realized_other_shift=np.zeros(same.size, dtype=bool),
        realized_shifts=tuple(
            ((0,) * dim,) if bool(value) else tuple() for value in same
        ),
        endpoint_i_empty=np.zeros(same.size, dtype=bool),
        endpoint_j_empty=np.zeros(same.size, dtype=bool),
        boundary_measure=None,
        cells=None,
        tessellation_diagnostics=None,
    )


@pytest.mark.parametrize(
    ('final_status', 'final_weights', 'final_converged', 'available'),
    (
        ('optimal', np.array([1.0, -1.0]), True, True),
        ('max_iter', np.array([2.0, -2.0]), False, True),
        ('infeasible_hard_constraints', None, False, False),
        ('numerical_failure', None, False, False),
    ),
)
def test_post_loop_final_refit_state_matrix_is_atomic(
    monkeypatch,
    final_status,
    final_weights,
    final_converged,
    available,
):
    import pyvoro2.inverse.separator.active as active_mod
    from pyvoro2 import Box
    from pyvoro2.inverse.separator import (
        ActiveSetOptions,
        build_power_fit_problem,
        dumps_report_json,
    )

    points = np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
    domain = Box(((-5.0, 5.0), (-5.0, 5.0), (-5.0, 5.0)))
    fit_calls = {'count': 0}
    realized_radii = []

    def fake_fit(points_arg, constraints, **kwargs):
        fit_calls['count'] += 1
        if fit_calls['count'] == 1:
            return _fit_result(
                constraints,
                status='optimal',
                weights=np.zeros(2),
                converged=True,
            )
        return _fit_result(
            constraints,
            status=final_status,
            weights=final_weights,
            converged=final_converged,
        )

    def fake_realize(*args, **kwargs):
        realized_radii.append(np.asarray(kwargs['radii']).copy())
        return _realization([True])

    monkeypatch.setattr(active_mod, 'fit_weights_from_separators', fake_fit)
    monkeypatch.setattr(active_mod, 'match_realized_pairs', fake_realize)

    result = active_mod.solve_self_consistent_power_weights(
        points,
        [(0, 1, 0.5)],
        domain=domain,
        options=ActiveSetOptions(max_iter=3),
        return_history=True,
    )

    assert result.termination == 'self_consistent'
    assert result.converged is True
    assert result.fit.status == final_status
    assert result.final_refit_converged is final_converged
    assert result.final_state_available is available
    assert result.final_state_unavailable_reason == (
        None if available else final_status
    )

    if available:
        assert len(realized_radii) == 2
        np.testing.assert_array_equal(realized_radii[-1], result.fit.radii)
        prediction = build_power_fit_problem(result.constraints).predict(
            result.fit.weights
        )
        np.testing.assert_allclose(
            result.diagnostics.predicted,
            prediction.measurement,
        )
        assert result.realized is not None
        assert result.diagnostics is not None
        assert result.to_records() is not None
    else:
        assert len(realized_radii) == 1
        assert result.fit.weights is None
        assert result.realized is None
        assert result.final_realization is None
        assert result.diagnostics is None
        assert result.candidate_diagnostics is None
        assert result.rms_residual_all is None
        assert result.max_residual_all is None
        assert result.tessellation_diagnostics is None
        assert result.to_records() is None

    report = result.to_report()
    assert json.loads(dumps_report_json(report)) == report
    assert report['summary']['termination'] == 'self_consistent'
    assert report['fit']['summary']['status'] == final_status
    assert report['availability'] == {
        'weights': available,
        'realization': available,
        'records': available,
        'reason': None if available else final_status,
    }
    if not available:
        assert report['realized'] is None
        assert report['diagnostics'] is None
        assert report['marginal_records'] is None
        assert report['summary']['n_realized_final'] is None
        assert report['summary']['rms_residual_all'] is None
        assert report['summary']['max_residual_all'] is None
        assert report['tessellation_diagnostics'] is None
    state = active_mod._accepted_state_from_result(result)
    assert state.origin.generation == 'final_refit'
    assert state.origin.accepted_outer_iteration == result.n_outer_iter


@pytest.mark.parametrize(
    ('outer_case', 'expected_termination', 'outer_realizations'),
    (
        ('cycle', 'cycle_detected', ([True, False], [False, True], [True, False])),
        ('max_outer', 'max_outer_iter', ([False],)),
    ),
)
def test_failed_final_refit_preserves_prior_outer_stop(
    monkeypatch,
    outer_case,
    expected_termination,
    outer_realizations,
):
    import pyvoro2.inverse.separator.active as active_mod
    from pyvoro2 import Box
    from pyvoro2.inverse.separator import ActiveSetOptions

    n_rows = len(outer_realizations[0])
    points = np.column_stack(
        (np.arange(n_rows + 1, dtype=float), np.zeros((n_rows + 1, 2)))
    )
    rows = [(idx, idx + 1, 0.5) for idx in range(n_rows)]
    domain = Box(((-5.0, 10.0), (-5.0, 5.0), (-5.0, 5.0)))
    fit_calls = {'count': 0}
    realize_calls = {'count': 0}

    def fake_fit(points_arg, constraints, **kwargs):
        fit_calls['count'] += 1
        if fit_calls['count'] <= len(outer_realizations):
            return _fit_result(
                constraints,
                status='optimal',
                weights=np.zeros(n_rows + 1),
                converged=True,
            )
        return _fit_result(
            constraints,
            status='numerical_failure',
            weights=None,
            converged=False,
        )

    def fake_realize(*args, **kwargs):
        index = realize_calls['count']
        realize_calls['count'] += 1
        return _realization(outer_realizations[index])

    monkeypatch.setattr(active_mod, 'fit_weights_from_separators', fake_fit)
    monkeypatch.setattr(active_mod, 'match_realized_pairs', fake_realize)
    options = (
        ActiveSetOptions(add_after=1, drop_after=1, cycle_window=4, max_iter=8)
        if outer_case == 'cycle'
        else ActiveSetOptions(drop_after=2, max_iter=1)
    )

    result = active_mod.solve_self_consistent_power_weights(
        points,
        rows,
        domain=domain,
        options=options,
        return_history=True,
    )

    assert result.termination == expected_termination
    assert result.converged is False
    assert result.fit.status == 'numerical_failure'
    assert result.final_state_available is False
    assert result.realized is None
    assert result.diagnostics is None
    assert realize_calls['count'] == len(outer_realizations)
    assert active_mod._accepted_state_from_result(
        result
    ).origin.generation == 'final_refit'


@pytest.mark.parametrize(
    ('fit_status', 'outer_termination'),
    (
        ('infeasible_hard_constraints', 'infeasible_active_set'),
        ('numerical_failure', 'numerical_failure'),
    ),
)
def test_first_inner_failure_has_explicitly_unavailable_final_layers(
    monkeypatch,
    fit_status,
    outer_termination,
):
    import pyvoro2.inverse.separator.active as active_mod
    from pyvoro2 import Box
    from pyvoro2.inverse.separator import dumps_report_json

    points = np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
    domain = Box(((-5.0, 5.0), (-5.0, 5.0), (-5.0, 5.0)))

    def fake_fit(points_arg, constraints, **kwargs):
        return _fit_result(
            constraints,
            status=fit_status,
            weights=None,
            converged=False,
        )

    def unexpected_realize(*args, **kwargs):
        raise AssertionError('no realization is valid without final weights')

    monkeypatch.setattr(active_mod, 'fit_weights_from_separators', fake_fit)
    monkeypatch.setattr(active_mod, 'match_realized_pairs', unexpected_realize)

    result = active_mod.solve_self_consistent_power_weights(
        points,
        [(0, 1, 0.5)],
        domain=domain,
    )

    assert result.termination == outer_termination
    assert result.fit.status == fit_status
    assert result.fit.weights is None
    assert result.final_state_available is False
    assert result.final_state_unavailable_reason == fit_status
    assert result.realized is None
    assert result.diagnostics is None
    assert result.rms_residual_all is None
    assert result.max_residual_all is None
    assert result.tessellation_diagnostics is None
    assert result.to_records() is None
    report = result.to_report()
    assert json.loads(dumps_report_json(report)) == report
    state = active_mod._accepted_state_from_result(result)
    assert state.origin.generation == 'outer_failure'
    assert state.origin.active_row_ids == tuple(
        row['row_id']
        for row in result.constraints.subset(result.active_mask).to_records()
    )


@pytest.mark.parametrize('malformation', ('missing', 'inconsistent_radii'))
def test_malformed_claimed_weighted_fit_normalizes_to_numerical_failure(
    monkeypatch,
    malformation,
):
    import pyvoro2.inverse.separator.active as active_mod
    from pyvoro2 import Box

    points = np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
    domain = Box(((-5.0, 5.0), (-5.0, 5.0), (-5.0, 5.0)))

    def fake_fit(points_arg, constraints, **kwargs):
        fit = _fit_result(
            constraints,
            status='optimal',
            weights=(None if malformation == 'missing' else np.zeros(2)),
            converged=True,
        )
        if malformation == 'inconsistent_radii':
            fit = replace(fit, radii=np.ones(2))
        return fit

    monkeypatch.setattr(active_mod, 'fit_weights_from_separators', fake_fit)

    result = active_mod.solve_self_consistent_power_weights(
        points,
        [(0, 1, 0.5)],
        domain=domain,
    )

    assert result.termination == 'numerical_failure'
    assert result.fit.status == 'numerical_failure'
    assert result.fit.status_detail is not None
    assert 'complete finite weight/radius vector' in result.fit.status_detail
    assert result.final_state_available is False
    assert result.final_state_unavailable_reason == 'numerical_failure'


def test_available_final_state_matches_independent_prediction_realization_and_origin():
    from pyvoro2 import Box
    from pyvoro2.inverse.separator import (
        ActiveSetOptions,
        build_power_fit_problem,
        match_realized_pairs,
        solve_self_consistent_power_weights,
    )
    from pyvoro2.inverse.separator._identity import (
        _originating_observations,
        _require_observation_association,
        _row_ids,
    )

    points = np.array(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]]
    )
    domain = Box(((-5.0, 5.0), (-5.0, 5.0), (-5.0, 5.0)))
    result = solve_self_consistent_power_weights(
        points,
        [(0, 1, 0.5), (1, 2, 0.5), (0, 2, 0.5)],
        domain=domain,
        options=ActiveSetOptions(max_iter=5),
        return_boundary_measure=True,
    )

    assert result.final_state_available is True
    assert result.fit.weights is not None
    problem = build_power_fit_problem(result.constraints)
    prediction = problem.predict(result.fit.weights)
    target = result.constraints.target_fraction
    residuals = prediction.measurement - target
    independently_realized = match_realized_pairs(
        points,
        domain=domain,
        radii=result.fit.radii,
        constraints=result.constraints,
        return_boundary_measure=True,
    )

    np.testing.assert_array_equal(
        result.diagnostics.predicted,
        prediction.measurement,
    )
    np.testing.assert_array_equal(result.diagnostics.residuals, residuals)
    np.testing.assert_array_equal(
        result.realized.realized,
        independently_realized.realized,
    )
    np.testing.assert_array_equal(
        result.realized.realized_same_shift,
        independently_realized.realized_same_shift,
    )
    np.testing.assert_allclose(
        result.rms_residual_all,
        np.sqrt(np.mean(residuals * residuals)),
    )
    np.testing.assert_allclose(
        result.max_residual_all,
        np.max(np.abs(residuals)),
    )

    active_origin = result.constraints.subset(result.active_mask)
    fit_origin = _originating_observations(result.fit, context='test fit')
    realized_origin = _originating_observations(
        result.realized,
        context='test realization',
    )
    diagnostics_origin = _originating_observations(
        result.diagnostics,
        context='test diagnostics',
    )
    _require_observation_association(active_origin, fit_origin, context='test fit')
    _require_observation_association(
        result.constraints,
        realized_origin,
        context='test realization',
    )
    _require_observation_association(
        result.constraints,
        diagnostics_origin,
        context='test diagnostics',
    )
    full_row_ids = _row_ids(result.constraints)
    expected_active_row_ids = tuple(
        row_id
        for row_id, active in zip(full_row_ids, result.active_mask)
        if bool(active)
    )
    assert _row_ids(fit_origin) == expected_active_row_ids
    assert _row_ids(diagnostics_origin) == full_row_ids

    with pytest.raises(ValueError, match='requires realization'):
        replace(result, diagnostics=None)


def test_extreme_finite_max_iter_state_has_scale_safe_residual_summaries(
    monkeypatch,
):
    import pyvoro2.inverse.separator.active as active_mod
    from pyvoro2 import Box
    from pyvoro2.inverse.separator import ActiveSetOptions, dumps_report_json

    points = np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
    domain = Box(((-5.0, 5.0), (-5.0, 5.0), (-5.0, 5.0)))
    weights = np.array([0.0, 1.0e200])

    def fake_fit(points_arg, constraints, **kwargs):
        return _fit_result(
            constraints,
            status='max_iter',
            weights=weights,
            converged=False,
        )

    def fake_realize(*args, **kwargs):
        return _realization([True])

    monkeypatch.setattr(active_mod, 'fit_weights_from_separators', fake_fit)
    monkeypatch.setattr(active_mod, 'match_realized_pairs', fake_realize)

    result = active_mod.solve_self_consistent_power_weights(
        points,
        [(0, 1, 0.5)],
        confidence=[0.0],
        domain=domain,
        options=ActiveSetOptions(max_iter=2),
        return_history=True,
        connectivity_check='diagnose',
    )

    expected_prediction = 0.5 + (
        result.fit.weights[0] - result.fit.weights[1]
    ) / (2.0 * 2.0**2)
    expected_residual = expected_prediction - 0.5
    expected_rms = 1.25e199

    assert result.termination == 'self_consistent'
    assert result.converged is True
    assert result.fit.status == 'max_iter'
    assert result.fit.converged is False
    assert result.final_refit_converged is False
    assert result.final_state_available is True
    assert np.isfinite(result.rms_residual_all)
    assert result.rms_residual_all == pytest.approx(expected_rms)
    assert len(result.history) == 1
    assert np.isfinite(result.history[0].rms_residual_all)
    assert result.history[0].rms_residual_all == pytest.approx(expected_rms)
    np.testing.assert_allclose(
        result.diagnostics.predicted,
        [expected_prediction],
    )
    np.testing.assert_allclose(
        result.diagnostics.residuals,
        [expected_residual],
    )
    report = result.to_report()
    assert json.loads(dumps_report_json(report)) == report


def test_unavailable_final_refit_preserves_scale_safe_extreme_history(
    monkeypatch,
):
    import pyvoro2.inverse.separator.active as active_mod
    from pyvoro2 import Box
    from pyvoro2.inverse.separator import ActiveSetOptions, dumps_report_json

    points = np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
    domain = Box(((-5.0, 5.0), (-5.0, 5.0), (-5.0, 5.0)))
    weights = np.array([0.0, 1.0e200])
    fit_calls = {'count': 0}

    def fake_fit(points_arg, constraints, **kwargs):
        fit_calls['count'] += 1
        if fit_calls['count'] == 1:
            return _fit_result(
                constraints,
                status='max_iter',
                weights=weights,
                converged=False,
            )
        return _fit_result(
            constraints,
            status='numerical_failure',
            weights=None,
            converged=False,
        )

    def fake_realize(*args, **kwargs):
        return _realization([False])

    monkeypatch.setattr(active_mod, 'fit_weights_from_separators', fake_fit)
    monkeypatch.setattr(active_mod, 'match_realized_pairs', fake_realize)

    result = active_mod.solve_self_consistent_power_weights(
        points,
        [(0, 1, 0.5)],
        confidence=[0.0],
        domain=domain,
        options=ActiveSetOptions(drop_after=2, max_iter=1),
        return_history=True,
        connectivity_check='diagnose',
    )

    assert result.termination == 'max_outer_iter'
    assert result.fit.status == 'numerical_failure'
    assert result.final_state_available is False
    assert result.realized is None
    assert result.diagnostics is None
    assert len(result.history) == 1
    assert np.isfinite(result.history[0].rms_residual_all)
    assert result.history[0].rms_residual_all == pytest.approx(1.25e199)
    report = result.to_report()
    assert json.loads(dumps_report_json(report)) == report


def test_extreme_finite_weight_step_norm_is_scale_safe(monkeypatch):
    import pyvoro2.inverse.separator.active as active_mod
    from pyvoro2 import Box
    from pyvoro2.inverse.separator import (
        ActiveSetOptions,
        dumps_report_json,
        FitModel,
        L2Regularization,
    )

    points = np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
    domain = Box(((-5.0, 5.0), (-5.0, 5.0), (-5.0, 5.0)))
    first_weights = np.zeros(2)
    stepped_weights = np.array([1.0e200, -1.0e200])
    model = FitModel(
        regularization=L2Regularization(
            strength=1.0e-200,
            reference=np.zeros(2),
        )
    )
    fit_calls = {'count': 0}

    def fake_fit(points_arg, constraints, **kwargs):
        fit_calls['count'] += 1
        weights = first_weights if fit_calls['count'] == 1 else stepped_weights
        return _fit_result(
            constraints,
            status='max_iter',
            weights=weights,
            converged=False,
        )

    def fake_realize(*args, **kwargs):
        return _realization([False])

    monkeypatch.setattr(active_mod, 'fit_weights_from_separators', fake_fit)
    monkeypatch.setattr(active_mod, 'match_realized_pairs', fake_realize)

    result = active_mod.solve_self_consistent_power_weights(
        points,
        [(0, 1, 0.5)],
        confidence=[0.0],
        domain=domain,
        model=model,
        options=ActiveSetOptions(drop_after=3, max_iter=2),
        return_history=True,
        connectivity_check='diagnose',
    )

    expected_step_norm = np.sqrt(2.0) * 1.0e200
    assert result.termination == 'max_outer_iter'
    assert len(result.history) == 2
    assert np.isfinite(result.history[1].weight_step_norm)
    assert result.history[1].weight_step_norm == pytest.approx(
        expected_step_norm
    )
    report = result.to_report()
    assert json.loads(dumps_report_json(report)) == report
