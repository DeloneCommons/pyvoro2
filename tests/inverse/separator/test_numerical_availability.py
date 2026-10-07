"""Numerical availability is bounded, producer-owned and strict in JSON."""

import copy
from dataclasses import fields, replace
from decimal import Decimal, localcontext
from fractions import Fraction
import json
import pickle

import numpy as np
import pytest

from pyvoro2.planar import Box
from pyvoro2.inverse import fit_self_consistent_weights_from_separators
from pyvoro2.inverse.separator import (
    FitModel,
    FixedValue,
    L2Regularization,
    SquaredLoss,
    build_power_fit_problem,
    build_power_fit_result,
    fit_weights_from_separators,
    resolve_separator_observations,
    solve_self_consistent_power_weights,
)

RANGE = 'out_of_binary64_range'
DEPENDENCY = 'unavailable_dependency'


def _strict(report):
    assert json.loads(json.dumps(report, allow_nan=False)) == report
    assert report['schema'] == {
        'name': 'pyvoro2.inverse.separator.report',
        'version': 3,
    }
    assert isinstance(report['unavailable_diagnostics'], dict)


def _decimal(value):
    return Decimal.from_float(float(value))


def test_dyadic_source_encodings_keep_model_and_source_owners_separate():
    row_ids = []
    for space, target, alpha, rho, residual in (
        ('fraction', 0.75, 2.0, 32.0, -0.25),
        ('position', 0.375, 1.0, 8.0, -0.125),
    ):
        obs = resolve_separator_observations(
            [[0.0, 0.0], [0.5, 0.0]],
            [(0, 1, target)],
            measurement=space,
            confidence=[8.0],
        )
        problem = build_power_fit_problem(
            obs, model=FitModel(mismatch=SquaredLoss(space='position'))
        )
        np.testing.assert_array_equal(problem.alpha, [1.0])
        np.testing.assert_array_equal(problem.beta, [0.25])
        np.testing.assert_array_equal(problem.mismatch_target, [0.375])
        np.testing.assert_array_equal(problem.z_obs, [0.125])
        np.testing.assert_array_equal(problem.edge_weight, [8.0])
        np.testing.assert_array_equal(
            problem.quadratic_operator.observation_rhs, [1.0, -1.0]
        )
        np.testing.assert_array_equal(
            problem.quadratic_operator.regularized_normal_matrix_dense(),
            [[8.0, -8.0], [-8.0, 8.0]],
        )
        assert problem.evaluate_objective(np.zeros(2)) == 0.0625
        fit = build_power_fit_result(
            problem, np.zeros(2), status='external', converged=False
        )
        assert fit.residuals[0] == residual
        assert fit.edge_diagnostics.alpha[0] == alpha
        assert fit.edge_diagnostics.edge_weight[0] == rho
        assert fit.edge_diagnostics.weighted_l2 == pytest.approx(
            np.sqrt(8.0) * abs(residual)
        )
        row = fit.to_records(obs)[0]
        assert row['unavailable_diagnostics'] == {}
        row_ids.append(row['row_id'])
        _strict(fit.to_report(obs))
    assert row_ids[0] != row_ids[1]


@pytest.mark.parametrize('conflict', [False, True])
def test_native_active_source_curvature_overflow_keeps_result_and_reports(conflict):
    count = 2 if conflict else 1
    result = fit_self_consistent_weights_from_separators(
        [[0.0, 0.0], [0.5, 0.0]],
        [(0, 1, 0.5)] * count,
        confidence=[1e308] * count,
        domain=Box([[-1.0, 1.0], [-1.0, 1.0]]),
        model=FitModel(
            mismatch=SquaredLoss(space='position'),
            feasible=FixedValue([0.1, 0.9]) if conflict else None,
        ),
        fit_solver='admm' if conflict else 'direct',
    )
    assert result.termination == (
        'infeasible_active_set' if conflict else 'self_consistent'
    )
    assert result.fit.status == (
        'infeasible_hard_constraints' if conflict else 'optimal'
    )
    assert result.final_state_available is (not conflict)
    model_problem = build_power_fit_problem(
        result.fit._require_policy().observations,
        model=result.fit._require_policy().model,
    )
    np.testing.assert_array_equal(model_problem.edge_weight, [1e308] * count)
    report = result.to_report()
    _strict(report)
    _strict(report['fit'])
    expected = {}
    for index, row in enumerate(report['fit']['fit_records']):
        assert row['edge_weight'] is None
        assert row['unavailable_diagnostics'] == {'/edge_weight': RANGE}
        assert row['predicted'] is None if conflict else row['predicted'] == 0.5
        expected[f'/fit/fit_records/{index}/edge_weight'] = RANGE
        expected[f'/fit/edge_diagnostics/edge_weight/{index}'] = RANGE
    assert report['unavailable_diagnostics'] == expected
    if not conflict:
        _strict(report['realized'])
        assert report['realized']['unavailable_diagnostics'] == {}
        assert report['diagnostics'][0]['unavailable_diagnostics'] == {}
    else:
        assert report['diagnostics'] is None and report['fit']['weights'] is None


def test_genuine_model_curvature_overflow_does_not_refuse_zero_optimum():
    points = [[0.0, 0.0], [0.25, 0.0]]
    obs = resolve_separator_observations(
        points, [(0, 1, 0.125)], measurement='position', confidence=[1e308]
    )
    problem = build_power_fit_problem(obs)
    assert np.isposinf(problem.edge_weight[0])
    fit = fit_weights_from_separators(points, obs)
    assert fit.status == 'optimal' and fit.objective.total == 0.0
    np.testing.assert_array_equal(fit.weights, [0.0, 0.0])
    report = fit.to_report(obs)
    _strict(report)
    assert report['unavailable_diagnostics'] == {
        '/fit_records/0/edge_weight': RANGE,
        '/edge_diagnostics/edge_weight/0': RANGE,
    }


def test_complete_weighted_residual_survives_unscaled_overflow():
    maximum = np.finfo(float).max
    reference = np.array([maximum / 4.0, -maximum / 4.0])
    points = [[0.0, 0.0], [0.5, 0.0]]
    obs = resolve_separator_observations(
        points, [(0, 1, -maximum)], confidence=[1e-320]
    )
    fit = fit_weights_from_separators(
        points,
        obs,
        model=FitModel(regularization=L2Regularization(1.0, reference)),
    )
    assert fit.status == 'optimal' and np.isfinite(fit.objective.total)
    np.testing.assert_array_equal(fit.weights, reference)
    assert np.isposinf(fit.residuals[0])
    with localcontext() as context:
        context.prec = 1000
        exact = (
            Decimal('.5')
            + 2 * _decimal(reference[0])
            - 2 * _decimal(reference[1])
            + _decimal(maximum)
        )
        expected = float((_decimal(1e-320) * exact * exact).sqrt())
    assert fit.edge_diagnostics.weighted_l2 == pytest.approx(expected, rel=8e-15)
    assert fit.edge_diagnostics.weighted_rmse == pytest.approx(expected, rel=8e-15)
    report = fit.to_report(obs)
    _strict(report)
    assert report['summary']['rms_residual'] is None
    assert report['unavailable_diagnostics']['/summary/rms_residual'] == RANGE
    assert '/edge_diagnostics/weighted_l2' not in report['unavailable_diagnostics']
    assert report['fit_records'][0]['residual'] is None


def test_rms_recovers_from_complete_rows_without_dropping_overflowed_row():
    maximum = np.finfo(float).max
    reference = np.array([maximum / 8.0, -maximum / 8.0])
    points = [[0.0, 0.0], [0.5, 0.0]]
    obs = resolve_separator_observations(
        points,
        [(0, 1, -0.75 * maximum), (0, 1, 0.5 * maximum)],
        confidence=[0.0, 0.0],
    )
    fit = fit_weights_from_separators(
        points,
        obs,
        model=FitModel(regularization=L2Regularization(1.0, reference)),
    )
    assert fit.status == 'optimal' and np.isposinf(fit.residuals[0])
    residuals = [
        Fraction(1, 2)
        + 2 * Fraction(float(reference[0]))
        - 2 * Fraction(float(reference[1]))
        - Fraction(float(t))
        for t in obs.target
    ]
    assert residuals[0] > Fraction(float(maximum))
    with localcontext() as context:
        context.prec = 1000
        squared = (
            sum(
                Decimal(r.numerator)
                / Decimal(r.denominator)
                * (Decimal(r.numerator) / Decimal(r.denominator))
                for r in residuals
            )
            / 2
        )
        expected = float(squared.sqrt())
    assert np.isfinite(expected)
    assert fit.rms_residual == pytest.approx(expected, rel=8e-15)
    assert np.isposinf(fit.max_residual)
    report = fit.to_report(obs)
    _strict(report)
    assert '/summary/rms_residual' not in report['unavailable_diagnostics']
    assert report['unavailable_diagnostics']['/summary/max_residual'] == RANGE


def test_unavailable_alpha_never_creates_false_observed_difference():
    d = 2.0**-536
    points = [[0.0, 0.0], [d, 0.0]]
    obs = resolve_separator_observations(
        points, [(0, 1, 2.0**1000)], confidence=[2.0**-1000]
    )
    model = FitModel(mismatch=SquaredLoss(space='position'))
    problem = build_power_fit_problem(obs, model=model)
    assert np.isfinite(problem.alpha[0]) and problem.edge_weight[0] == 2.0**70
    native_fit = fit_weights_from_separators(points, obs, model=model)
    assert native_fit.status in ('optimal', 'numerical_failure')
    if native_fit.weights is None:
        assert native_fit.status == 'numerical_failure' and native_fit.hard_feasible
    fit = build_power_fit_result(
        problem,
        np.array([2.0**-72, -(2.0**-72)]),
        status='external',
        converged=False,
    )
    assert np.isfinite(fit.objective.total)
    assert np.isposinf(fit.edge_diagnostics.alpha[0])
    assert np.isnan(fit.edge_diagnostics.z_obs[0])
    report = fit.to_report(obs)
    _strict(report)
    assert report['unavailable_diagnostics']['/edge_diagnostics/alpha/0'] == RANGE
    assert report['unavailable_diagnostics']['/edge_diagnostics/z_obs/0'] == DEPENDENCY
    zero_obs = resolve_separator_observations(
        points, [(0, 1, 2.0**1000)], confidence=[0.0]
    )
    zero_problem = build_power_fit_problem(zero_obs)
    assert np.isposinf(zero_problem.alpha[0])
    assert np.isnan(zero_problem.z_obs[0])
    assert np.isnan(zero_problem.observation_graph.z_obs[0])
    np.testing.assert_array_equal(zero_problem.edge_weight, [0.0])
    np.testing.assert_array_equal(
        zero_problem.quadratic_operator.observation_rhs, [0.0, 0.0]
    )


def _unavailable_active(mismatch='position', *, facade=False):
    maximum = np.finfo(float).max
    solve = (
        fit_self_consistent_weights_from_separators
        if facade
        else solve_self_consistent_power_weights
    )
    extra = {} if facade else {'return_history': True}
    return solve(
        [[0.0, 0.0], [0.25, 0.0]],
        [(0, 1, 0.5)],
        confidence=[0.0],
        domain=Box([[-1.0, 1.0], [-1.0, 1.0]]),
        model=FitModel(
            mismatch=SquaredLoss(space=mismatch),
            regularization=L2Regularization(1.0, [maximum / 8.0, -maximum / 8.0]),
        ),
        return_cells=True,
        connectivity_check='diagnose',
        unaccounted_pair_check='diagnose',
        **extra,
    )


@pytest.mark.parametrize('mismatch', ['position', 'fraction'])
def test_native_active_unavailable_cells_keep_atomic_final_state_and_provenance(
    mismatch,
):
    result = _unavailable_active(mismatch)
    assert result.final_state_available and result.fit.status == 'optimal'
    assert result.realized is not None and result.realized.cells is not None
    assert np.isposinf(result.diagnostics.predicted[0])
    assert np.isposinf(result.diagnostics.residuals[0])
    assert np.isposinf(result.rms_residual_all)
    records = result.diagnostics.to_records()
    assert records[0]['unavailable_diagnostics'] == {
        '/predicted': RANGE,
        '/predicted_fraction': RANGE,
        '/residual': RANGE,
    }
    report = result.to_report()
    _strict(report)
    local = dict(records[0]['unavailable_diagnostics'])
    if mismatch == 'fraction':
        local.update({'/mismatch_predicted': RANGE, '/mismatch_residual': RANGE})
    assert report['diagnostics'][0]['unavailable_diagnostics'] == local
    assert report['marginal_records'][0]['unavailable_diagnostics'] == local
    expected_map = {
        prefix + path: reason
        for prefix in ('/diagnostics/0', '/marginal_records/0')
        for path, reason in local.items()
    }
    expected_map.update(
        {'/summary/rms_residual_all': RANGE, '/summary/max_residual_all': RANGE}
    )
    for index, record in enumerate(result.history):
        assert np.isposinf(record.rms_residual_all)
        assert (
            report['unavailable_diagnostics'][f'/history/{index}/rms_residual_all']
            == RANGE
        )
        assert 'unavailable_diagnostics' not in report['history'][index]
        expected_map.update(
            {
                f'/history/{index}/rms_residual_all': RANGE,
                f'/history/{index}/max_residual_all': RANGE,
            }
        )
    assert report['unavailable_diagnostics'] == expected_map
    assert all('unavailable_diagnostics' not in row for row in report['constraints'])
    assert all(
        'unavailable_diagnostics' not in row for row in report['realized']['records']
    )
    for rebuilt in (
        copy.copy(result),
        copy.deepcopy(result),
        pickle.loads(pickle.dumps(result)),
        replace(result),
    ):
        assert rebuilt.to_report() == report
    for rebuilt in (
        copy.copy(result.diagnostics),
        copy.deepcopy(result.diagnostics),
        pickle.loads(pickle.dumps(result.diagnostics)),
        replace(result.diagnostics),
    ):
        assert rebuilt.to_records() == records
    with pytest.raises(ValueError, match='diagnostic|provenance|evaluation'):
        replace(result.diagnostics, predicted=np.array([-np.inf])).to_records()
    unbound = type(result.diagnostics)(
        **{
            field.name: getattr(result.diagnostics, field.name)
            for field in fields(result.diagnostics)
        },
        _originating_observations_init=result.constraints,
    )
    with pytest.raises(ValueError, match='diagnostic|provenance|non-finite'):
        unbound.to_records()
    wrong_history = (
        replace(result.history[0], rms_residual_all=np.nan),
    ) + result.history[1:]
    with pytest.raises(ValueError, match='history|provenance|evaluation'):
        replace(result, history=wrong_history).to_report()
    manual_history = tuple(
        type(row)(**{field.name: getattr(row, field.name) for field in fields(row)})
        for row in result.history
    )
    with pytest.raises(ValueError, match='history|provenance|non-finite'):
        replace(result, history=manual_history).to_report()


def test_native_supported_facade_keeps_unavailable_source_cells():
    result = _unavailable_active(facade=True)
    assert result.final_state_available and result.realized.cells is not None
    _strict(result.to_report())
    assert result.to_records()[0]['unavailable_diagnostics'] == {
        '/predicted': RANGE,
        '/predicted_fraction': RANGE,
        '/residual': RANGE,
    }


def test_dependency_producer_records_survive_history_reconstruction():
    from pyvoro2.inverse.separator import ActiveSetIteration
    from pyvoro2.inverse.separator._diagnostics import (
        _HistorySnapshot,
        _history_values,
    )
    from pyvoro2.inverse.separator.problem import _source_diagnostic_values

    obs = resolve_separator_observations(
        [[0.0, 0.0], [2.0**-536, 0.0]],
        [(0, 1, 2.0**1000)],
        confidence=[0.0],
    )
    produced = _source_diagnostic_values(obs, np.zeros(2))
    row = ActiveSetIteration(
        1,
        1,
        0,
        0,
        0,
        produced['rms_residual'].value,
        produced['max_residual'].value,
        0.0,
        _history_snapshot_init=_HistorySnapshot.produced(
            obs,
            1,
            produced['rms_residual'],
            produced['max_residual'],
        ),
    )
    for rebuilt in (
        copy.copy(row),
        copy.deepcopy(row),
        pickle.loads(pickle.dumps(row)),
        replace(row),
    ):
        unavailable = {}
        values = _history_values(rebuilt, obs)
        assert (
            values['rms_residual_all'].json_value(
                '/history/0/rms_residual_all', unavailable
            )
            is None
        )
        assert unavailable == {'/history/0/rms_residual_all': DEPENDENCY}
    with pytest.raises(ValueError, match='stale iteration'):
        _history_values(replace(row, iteration=2), obs)


@pytest.mark.parametrize('value', [np.inf, -np.inf, np.nan])
def test_generic_json_normalizer_never_uses_metadata_to_admit_raw_nonfinite(value):
    from pyvoro2.inverse.separator.report import _jsonable_report_value

    with pytest.raises(ValueError, match='NaN|infinite'):
        _jsonable_report_value(
            {'predicted': value, 'unavailable_diagnostics': {'/predicted': RANGE}}
        )


def test_fit_export_rejects_injected_nonfinite_derived_values():
    obs = resolve_separator_observations([[0.0, 0.0], [1.0, 0.0]], [(0, 1, 0.5)])
    fit = build_power_fit_result(build_power_fit_problem(obs), np.zeros(2))
    with pytest.raises(ValueError, match='diagnostic|final weights|evaluation'):
        replace(fit, residuals=np.array([np.inf])).to_records(obs)
    corrupted = replace(fit.edge_diagnostics, edge_weight=np.array([np.inf]))
    with pytest.raises(ValueError, match='diagnostic|evaluation'):
        replace(fit, edge_diagnostics=corrupted).to_report(obs)
    with pytest.raises(ValueError, match='finite'):
        replace(fit, weights=np.array([np.inf, 0.0])).to_report(obs)
    with pytest.raises(ValueError, match='NaN|infinite|non-finite'):
        replace(
            fit, objective_breakdown=replace(fit.objective, total=np.inf)
        ).to_report(obs)


@pytest.mark.parametrize(
    'status,converged',
    [
        ('optimal', True),
        ('optimal', False),
        ('max_iter', True),
    ],
)
def test_native_reconstruction_cannot_relabel_hard_violating_max_iter_as_success(
    status,
    converged,
):
    from pyvoro2.inverse.separator import ActiveSetOptions
    from pyvoro2.planar import Box

    points = [[0.0, 0.0], [1.0, 0.0]]
    model = FitModel(
        feasible=FixedValue(0.3),
        regularization=L2Regularization(1.0),
    )
    active = solve_self_consistent_power_weights(
        points,
        [(0, 1, 0.8)],
        domain=Box([[-1.0, 2.0], [-1.0, 1.0]]),
        model=model,
        fit_solver='admm',
        fit_admm_max_iter=1,
        options=ActiveSetOptions(max_iter=1),
    )
    fit = active.fit
    assert fit.status == 'max_iter' and not fit.converged
    assert active.final_state_available
    assert not fit.objective.hard_constraints_satisfied
    _strict(active.to_report())
    _strict(fit.to_report(active.constraints))

    # Matching or stale objective metadata cannot replace a fresh hard-row
    # predicate on the exact returned vector and selected policy.
    relabeled = replace(
        fit,
        status=status,
        converged=converged,
        objective_breakdown=replace(
            fit.objective,
            hard_constraints_satisfied=True,
            hard_max_violation=0.0,
        ),
    )
    with pytest.raises(ValueError, match='hard'):
        relabeled.to_records(active.constraints)
    with pytest.raises(ValueError, match='hard'):
        relabeled.to_report(active.constraints)
    with pytest.raises(ValueError):
        replace(active, fit=relabeled)
