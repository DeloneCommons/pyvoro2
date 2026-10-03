"""WP10 row policy, independent units, and projection regressions."""

import copy
from dataclasses import replace
import pickle

import numpy as np
import pytest

from pyvoro2.inverse.separator import (
    ExponentialBoundaryPenalty, FitModel, FixedValue, HuberLoss, Interval,
    L2Regularization, ReciprocalBoundaryPenalty, SoftIntervalPenalty,
    SquaredLoss, resolve_separator_observations,
)


def _observations():
    points = np.array([[0., 0.], [2., 0.], [0., 3.], [2., 3.]])
    return resolve_separator_observations(
        points, [(0, 1, .25), (0, 2, .4), (0, 1, .75)],
    )


def test_row_term_owns_inputs_and_preserves_scalar_and_vector_categories():
    lower = np.array([0., .1, .2])
    applicable = np.array([True, False, True])
    term = Interval(lower, [.8, .8, .2], applicable=applicable, space='position')
    lower[:] = -99
    applicable[:] = False
    np.testing.assert_array_equal(term.lower, [0., .1, .2])
    np.testing.assert_array_equal(term.applicable, [True, False, True])
    assert not term.lower.flags.writeable
    assert not term.applicable.flags.writeable
    assert isinstance(Interval(0, 1).lower, float)
    assert Interval(1, 1).upper == 1
    for restored in (copy.deepcopy(term), pickle.loads(pickle.dumps(term))):
        np.testing.assert_array_equal(restored.lower, term.lower)
        assert not restored.lower.flags.writeable


@pytest.mark.parametrize('bad', [True, 1j, '1', [True, .2], [[0.]], [np.nan]])
def test_row_terms_reject_invalid_configured_values_even_when_absent(bad):
    with pytest.raises(ValueError):
        Interval(bad, 1., applicable=False)
    with pytest.raises(ValueError):
        SoftIntervalPenalty(bad, 1., 0.)


@pytest.mark.parametrize('bad', [0, 1, 'yes', None, [True, 1], [[True]]])
def test_hard_applicability_requires_actual_boolean_scalar_or_vector(bad):
    with pytest.raises(ValueError):
        FixedValue(0., applicable=bad)


@pytest.mark.parametrize('constructor', [SquaredLoss, HuberLoss])
@pytest.mark.parametrize('bad', ['length', ['fraction'], 1, False])
def test_spaces_are_term_global_strict_choices(constructor, bad):
    with pytest.raises(ValueError):
        constructor(space=bad)


def test_boundary_and_soft_shape_validation_checks_all_configured_rows():
    with pytest.raises(ValueError):
        SoftIntervalPenalty([0., 1.], [1., 1.], [1., 0.])
    with pytest.raises(ValueError):
        ExponentialBoundaryPenalty([0., 0.], [1., .01], strength=[1., 0.])
    with pytest.raises(ValueError):
        ReciprocalBoundaryPenalty([0., 0.], [1., .01], strength=[1., 0.])
    for field, cls in [('margin', ExponentialBoundaryPenalty),
                       ('tau', ExponentialBoundaryPenalty),
                       ('margin', ReciprocalBoundaryPenalty),
                       ('epsilon', ReciprocalBoundaryPenalty),
                       ('delta', HuberLoss)]:
        with pytest.raises(ValueError):
            cls(**{field: [.01]})


def test_binding_projects_by_order_and_never_projects_site_reference():
    from pyvoro2.inverse.separator._policy import _bind_policy

    observations = _observations()
    model = FitModel(
        mismatch=SquaredLoss(space='position'),
        feasible=Interval([0., .1, .2], 1., applicable=[True, False, True]),
        penalties=(SoftIntervalPenalty(0., 1., [1., 2., 3.]),),
        regularization=L2Regularization(0., [10., 20., 30., 40.]),
    )
    policy = _bind_policy(observations, model)
    selected = policy.project([2, 0])
    np.testing.assert_array_equal(selected.model.feasible.lower, [.2, 0.])
    np.testing.assert_array_equal(selected.model.penalties[0].strength, [3., 1.])
    np.testing.assert_array_equal(selected.model.regularization.reference,
                                  [10., 20., 30., 40.])
    assert selected.view['model_spaces'] == {
        'mismatch': 'position', 'hard_constraint': 'fraction',
        'penalties': ('fraction',),
    }
    assert selected.view['model_policy']['hard_constraint']['parameters']['lower'] == {
        'kind': 'rows', 'values': (.2, 0.),
    }
    empty_term = policy.project([]).view['model_policy']['penalties'][0]
    empty_parameters = empty_term['parameters']
    assert empty_parameters['strength'] == {
        'kind': 'rows', 'values': (),
    }
    assert selected.view['row_ids'] == tuple(
        observations.to_records()[k]['row_id'] for k in [2, 0]
    )
    with pytest.raises(TypeError):
        selected.view['model_spaces']['mismatch'] = 'fraction'
    with pytest.raises(ValueError):
        _bind_policy(observations, replace(model, feasible=Interval([0.], 1.)))
    with pytest.raises(ValueError):
        Interval([0., .1], [1., 1., 1.])
