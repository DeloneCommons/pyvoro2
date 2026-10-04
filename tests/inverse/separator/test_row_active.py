"""Candidate policy survives active selection, reentry, and atomic validation."""

from dataclasses import replace

import numpy as np
import pytest

from pyvoro2 import Box
from pyvoro2.inverse.separator import (
    ActiveSetOptions, FitModel, Interval, SoftIntervalPenalty, SquaredLoss,
    solve_self_consistent_power_weights,
)
from pyvoro2.inverse.separator._policy import _bind_policy
from pyvoro2.inverse.separator.realize import RealizedPairDiagnostics


def _realization(same):
    same = np.asarray(same, dtype=bool)
    return RealizedPairDiagnostics(
        realized=same.copy(), unrealized=tuple(np.flatnonzero(~same)),
        realized_same_shift=same.copy(),
        realized_other_shift=np.zeros(len(same), dtype=bool),
        realized_shifts=tuple(((0, 0, 0),) if flag else () for flag in same),
        endpoint_i_empty=np.zeros(len(same), dtype=bool),
        endpoint_j_empty=np.zeros(len(same), dtype=bool),
        boundary_measure=None, cells=None, tessellation_diagnostics=None,
    )


def test_active_drop_reentry_and_final_refit_use_original_candidate_policy(monkeypatch):
    import pyvoro2.inverse.separator.active as active

    original_fit = active.fit_weights_from_separators
    calls = []
    masks = iter(([False, True, True], [True, False, True], [True, False, True]))

    def fit(points, observations, **kwargs):
        expected = np.array([1., 2., 3.])[observations.input_index]
        np.testing.assert_array_equal(kwargs['model'].penalties[0].strength, expected)
        calls.append(tuple(observations.input_index))
        return original_fit(points, observations, **kwargs)

    def realize(*args, **kwargs):
        return _realization(next(masks, [True, False, True]))

    monkeypatch.setattr(active, 'fit_weights_from_separators', fit)
    monkeypatch.setattr(active, '_match_realized_pairs', realize)
    model = FitModel(
        mismatch=SquaredLoss(space='position'),
        feasible=Interval(0., 1., applicable=[False, False, False]),
        penalties=(SoftIntervalPenalty(.4, .6, [1., 2., 3.]),),
    )
    result = solve_self_consistent_power_weights(
        [[0., 0., 0.], [2., 0., 0.], [0., 2., 0.]],
        [(0, 1, .3), (0, 1, .5), (0, 2, .7)],
        domain=Box(((-5., 5.),) * 3), model=model, fit_solver='admm',
        options=ActiveSetOptions(drop_after=1, max_iter=4), return_history=True,
    )
    assert calls[:3] == [(0, 1, 2), (1, 2), (0, 2)]
    assert calls[-1] == (0, 2)
    assert result.resolved_policy['model_policy']['penalties'][0]['parameters'][
        'strength'] == {'kind': 'rows', 'values': (1., 2., 3.)}
    assert result.fit.resolved_policy['model_policy']['penalties'][0]['parameters'][
        'strength'] == {'kind': 'rows', 'values': (1., 3.)}
    swapped = _bind_policy(result.constraints.subset(result.active_mask), replace(
        result.fit._require_policy().model,
        penalties=(SoftIntervalPenalty(.4, .6, [3., 1.]),),
    ))
    bad_fit = replace(result.fit, _bound_policy_init=swapped)
    with pytest.raises(ValueError, match='policy'):
        replace(result, fit=bad_fit)


def test_unselected_candidate_hard_rows_are_not_compiled_for_prediction(monkeypatch):
    import pyvoro2.inverse.separator.active as active

    monkeypatch.setattr(active, '_match_realized_pairs',
                        lambda *args, **kwargs: _realization([True, False]))
    result = solve_self_consistent_power_weights(
        [[0., 0., 0.], [2., 0., 0.]], [(0, 1, .5), (0, 1, .5)],
        domain=Box(((-5., 5.),) * 3), active0=[True, False],
        model=FitModel(feasible=Interval([0., 1e308], [1., 1e308])),
        fit_solver='admm', options=ActiveSetOptions(max_iter=2),
    )
    assert result.fit.status == 'optimal'
    assert result.resolved_policy['model_policy']['hard_constraint']['parameters'][
        'lower']['values'] == (0., 1e308)


@pytest.mark.parametrize('dim', [2, 3])
def test_native_periodic_active_policy_keeps_distinct_images(dim):
    """Requires an admitted native build; mocks cannot qualify this contract."""
    import pyvoro2 as spatial
    import pyvoro2.planar as planar

    domain = (planar.RectangularCell(((0., 1.),) * 2) if dim == 2 else
              spatial.PeriodicCell(vectors=np.eye(3)))
    points = np.full((2, dim), .5)
    points[:, 0] = [.125, .875]
    shifts = [(-1,) + (0,) * (dim-1), (1,) + (0,) * (dim-1)]
    model = FitModel(
        mismatch=SquaredLoss(space='position'),
        feasible=Interval([0., .25], [1., .75], applicable=[True, False]),
        penalties=(SoftIntervalPenalty(.4, .6, [1., 4.]),),
    )
    result = solve_self_consistent_power_weights(
        points, [(0, 1, .5, shift) for shift in shifts], domain=domain,
        image='given_only', model=model, fit_solver='admm',
        options=ActiveSetOptions(drop_after=1, max_iter=5),
    )
    assert result.termination == 'self_consistent'
    np.testing.assert_array_equal(result.active_mask, [True, False])
    assert result.realized.realized_other_shift[1]
    assert result.resolved_policy['model_policy']['penalties'][0]['parameters'][
        'strength']['values'] == (1., 4.)
    assert result.fit.resolved_policy['model_policy']['penalties'][0]['parameters'][
        'strength']['values'] == (1.,)
    assert result.fit.mismatch_space == 'position'
