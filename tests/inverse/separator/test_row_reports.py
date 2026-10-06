"""Schema v2 preserves the complete model and independent row coordinates."""

import json
from dataclasses import replace

import pytest

from pyvoro2.inverse.separator import (
    ExponentialBoundaryPenalty, FitModel, FixedValue, HuberLoss, Interval,
    L2Regularization, ReciprocalBoundaryPenalty, SoftIntervalPenalty,
    build_power_fit_problem, build_power_fit_result, dumps_report_json,
    fit_weights_from_separators, resolve_separator_observations,
)


def test_v2_complete_policy_wrappers_and_source_identity_are_exact():
    observations = resolve_separator_observations(
        [[0., 0.], [2., 0.]], [(0, 1, .25), (0, 1, .75)],
    )
    model = FitModel(
        mismatch=HuberLoss(.2, space='position'),
        feasible=Interval([0., .1], 1., applicable=[True, False]),
        penalties=(SoftIntervalPenalty(0., 1., [0., 2.]),
                   ExponentialBoundaryPenalty(strength=0., space='position'),
                   ReciprocalBoundaryPenalty(strength=[0., 0.])),
        regularization=L2Regularization(0., [7., 9.]),
    )
    problem = build_power_fit_problem(observations, model=model)
    result = build_power_fit_result(problem, [0., 0.])
    report = result.to_report(observations)
    assert report['schema'] == {
        'name': 'pyvoro2.inverse.separator.report', 'version': 3}
    assert report['model_spaces'] == {
        'mismatch': 'position', 'hard_constraint': 'fraction',
        'penalties': ['fraction', 'position', 'fraction'],
    }
    policy = report['model_policy']
    assert set(policy) == {'mismatch', 'hard_constraint', 'penalties', 'regularization'}
    assert policy['mismatch'] == {
        'family': 'HuberLoss', 'space': 'position',
        'parameters': {'delta': {'kind': 'uniform', 'value': .2}},
    }
    assert policy['hard_constraint'] == {
        'family': 'Interval', 'space': 'fraction',
        'parameters': {'lower': {'kind': 'rows', 'values': [0., .1]},
                       'upper': {'kind': 'uniform', 'value': 1.}},
        'applicable': {'kind': 'rows', 'values': [True, False]},
    }
    assert policy['penalties'][1]['parameters']['strength'] == {
        'kind': 'uniform', 'value': 0.,
    }
    assert policy['penalties'][2]['parameters']['epsilon'] == {
        'kind': 'uniform', 'value': 1e-6,
    }
    assert policy['regularization'] == {
        'family': 'L2Regularization', 'strength': 0.,
        'reference': {'kind': 'sites', 'values': [7., 9.]},
    }
    row = report['fit_records'][0]
    assert (row['measurement'], row['target'],
            row['predicted']) == ('fraction', .25, .5)
    assert (row['mismatch_space'], row['mismatch_target'], row['mismatch_predicted'],
            row['mismatch_residual']) == ('position', .5, 1., .5)
    assert report['summary']['mismatch_space'] == 'position'
    assert json.loads(dumps_report_json(report)) == report
    # A test-side reader reconstructs only the literal public v2 grammar.
    import pyvoro2.inverse.separator as public

    def reconstruct(term):
        if term is None:
            return None
        values = {name: field['value'] if field['kind'] == 'uniform' else
                  field['values'] for name, field in term['parameters'].items()}
        if 'applicable' in term:
            mask = term['applicable']
            values['applicable'] = (
                mask['value'] if mask['kind'] == 'uniform' else mask['values']
            )
        return getattr(public, term['family'])(space=term['space'], **values)

    regularization = policy['regularization']
    reference = regularization['reference']
    reconstructed = FitModel(
        mismatch=reconstruct(policy['mismatch']),
        feasible=reconstruct(policy['hard_constraint']),
        penalties=tuple(reconstruct(term) for term in policy['penalties']),
        regularization=L2Regularization(
            regularization['strength'], None if reference['kind'] == 'implicit_zero'
            else reference['values'],
        ),
    )
    reconstructed_problem = build_power_fit_problem(observations, model=reconstructed)
    assert reconstructed_problem.resolved_policy == problem.resolved_policy
    # Both source residuals are +/-1/4, both position residuals +/-1/2.
    # Original Huber units give 2 * .2 * (.5 - .2/2); every prior is absent/inactive.
    assert result.objective.total == pytest.approx(2*.2*(.5-.2/2), abs=1e-16)
    changed = build_power_fit_problem(observations, model=FitModel())
    assert problem.resolved_policy['row_ids'] == changed.resolved_policy['row_ids']


@pytest.mark.parametrize('empty_rows', [False, True])
def test_empty_report_preserves_uniform_versus_rows_and_explicit_reference(empty_rows):
    observations = resolve_separator_observations([[0., 0.]], [], allow_empty=True)
    value = [] if empty_rows else .5
    model = FitModel(feasible=FixedValue(value, applicable=[] if empty_rows else False))
    result = fit_weights_from_separators([[0., 0.]], observations, model=model)
    report = result.to_report(observations)
    assert report['model_policy']['hard_constraint']['parameters']['value'] == (
        {'kind': 'rows', 'values': []} if empty_rows else
        {'kind': 'uniform', 'value': .5}
    )
    assert report['fit_records'] == []


def test_no_weight_report_retains_policy_and_target_with_null_predictions():
    observations = resolve_separator_observations(
        [[0., 0.], [1., 0.], [2., 0.]],
        [(0, 1, .5), (1, 2, .5), (0, 2, .5)],
    )
    result = fit_weights_from_separators(
        [[0., 0.], [1., 0.], [2., 0.]], observations,
        model=FitModel(feasible=FixedValue([0., 0., 0.], space='position')),
        solver='admm',
    )
    assert result.weights is None
    report = result.to_report(observations)
    assert report['model_policy']['hard_constraint']['space'] == 'position'
    assert report['fit_records'][0]['mismatch_target'] == .5
    assert report['fit_records'][0]['mismatch_predicted'] is None
    assert report['fit_records'][0]['mismatch_residual'] is None
    legacy = replace(result, _bound_policy_init=None)
    with pytest.raises(ValueError, match='policy'):
        legacy.to_report(observations)
