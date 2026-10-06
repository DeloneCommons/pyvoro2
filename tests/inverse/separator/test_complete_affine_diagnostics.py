"""Source diagnostics evaluate complete affine rows at their own states."""

import copy
from dataclasses import fields, replace
from fractions import Fraction
import pickle

import numpy as np
import pytest

from pyvoro2.planar import Box
from pyvoro2.inverse.separator import (
    ActiveSetOptions,
    FitModel,
    FixedValue,
    L2Regularization,
    SquaredLoss,
    solve_self_consistent_power_weights,
)


def _cancellation_fit():
    a = 2.0**-47
    return solve_self_consistent_power_weights(
        [[0.0, 0.0], [16.0, 0.0]],
        [(0, 1, 8.0)],
        measurement='position',
        confidence=[2.0**112],
        domain=Box([[-16.0, 32.0], [-16.0, 16.0]]),
        model=FitModel(
            mismatch=SquaredLoss(space='fraction'),
            regularization=L2Regularization(2.0**95, [a, -a]),
        ),
        connectivity_check='diagnose',
        unaccounted_pair_check='diagnose',
        return_history=True,
    )


def test_native_active_cancellation_uses_complete_affine_residual_everywhere():
    result = _cancellation_fit()
    fit = result.fit
    assert fit.status == 'optimal' and result.final_state_available
    np.testing.assert_allclose(fit.weights, [2.0**-48, -(2.0**-48)], rtol=1e-14, atol=0)
    difference = Fraction(float(fit.weights[0])) - Fraction(float(fit.weights[1]))
    expected_source = float(difference / 32)
    expected_model = float(difference / 512)
    assert expected_source != 0.0 and fit.predicted[0] == fit.target[0]
    assert fit.residuals[0] == expected_source
    assert result.diagnostics.residuals[0] == expected_source
    assert result.mismatch_residuals[0] == expected_model
    assert result.rms_residual_all == abs(expected_source)
    assert result.max_residual_all == abs(expected_source)
    assert fit.objective.total == pytest.approx(1.0, rel=1e-14)
    report = result.to_report()
    assert report['diagnostics'][0]['residual'] == expected_source
    assert report['fit']['fit_records'][0]['residual'] == expected_source
    assert report['history'][0]['rms_residual_all'] == abs(expected_source)
    for rebuilt in (
        copy.copy(result),
        copy.deepcopy(result),
        pickle.loads(pickle.dumps(result)),
        replace(result),
    ):
        assert rebuilt.to_report() == report
    finite_manual_history = tuple(
        type(row)(**{item.name: getattr(row, item.name) for item in fields(row)})
        for row in result.history
    )
    assert replace(result, history=finite_manual_history).to_report() == report


@pytest.mark.parametrize('corruption', ['outer', 'inner', 'both'])
def test_reconstruction_rejects_individually_or_jointly_wrong_residuals(corruption):
    result = _cancellation_fit()
    fit = result.fit
    diagnostics = result.diagnostics
    if corruption in ('inner', 'both'):
        fit = replace(fit, residuals=np.zeros(1), rms_residual=0.0, max_residual=0.0)
    if corruption in ('outer', 'both'):
        diagnostics = replace(diagnostics, residuals=np.zeros(1))
    with pytest.raises(ValueError, match='diagnostic|final weights|residual'):
        replace(
            result,
            fit=fit,
            diagnostics=diagnostics,
            rms_residual_all=(
                0.0 if corruption in ('outer', 'both') else result.rms_residual_all
            ),
            max_residual_all=(
                0.0 if corruption in ('outer', 'both') else result.max_residual_all
            ),
        )


def test_native_relaxed_history_summaries_use_each_evaluated_vector(monkeypatch):
    import pyvoro2.inverse.separator.active as active

    original = active._match_realized_pairs
    evaluated = []

    def sample_realization(*args, **kwargs):
        evaluated.append(np.asarray(kwargs['semantic_weights']).copy())
        return original(*args, **kwargs)

    monkeypatch.setattr(active, '_match_realized_pairs', sample_realization)
    result = solve_self_consistent_power_weights(
        [[0.0, 0.0], [2.0, 0.0], [0.0, 2.0]],
        [(0, 1, 0.25), (0, 2, 0.75), (1, 2, 0.5), (0, 1, 0.9)],
        confidence=[1.0, 1.0, 1.0, 0.0],
        domain=Box([[-8.0, 8.0], [-8.0, 8.0]]),
        active0=[True, True, False, False],
        options=ActiveSetOptions(relax=0.5, max_iter=4),
        return_history=True,
    )
    assert len(result.history) >= 2
    assert len(evaluated) == len(result.history) + 1
    rows = [
        (0, 1, Fraction(1, 8), Fraction(0.25)),
        (0, 2, Fraction(1, 8), Fraction(0.75)),
        (1, 2, Fraction(1, 16), Fraction(0.5)),
        (0, 1, Fraction(1, 8), Fraction(0.9)),
    ]
    for record, weights in zip(result.history, evaluated):
        residuals = [
            Fraction(1, 2)
            + alpha * (Fraction(float(weights[i])) - Fraction(float(weights[j])))
            - target
            for i, j, alpha, target in rows
        ]
        expected_rms = np.sqrt(float(sum(r * r for r in residuals) / len(rows)))
        assert record.rms_residual_all == pytest.approx(expected_rms, rel=4e-15)
        assert record.max_residual_all == float(max(abs(r) for r in residuals))
    np.testing.assert_array_equal(
        result.diagnostics.residuals[result.active_mask],
        result.fit.residuals,
    )


def test_native_full_candidate_evaluation_does_not_certify_excluded_hard_rows():
    result = solve_self_consistent_power_weights(
        [[0.0, 0.0], [0.25, 0.0]],
        [(0, 1, 0.5), (0, 1, 0.5)],
        domain=Box([[-1.0, 1.0], [-1.0, 1.0]]),
        confidence=[0.0, 0.0],
        active0=[False, True],
        options=ActiveSetOptions(add_after=2, max_iter=1),
        fit_solver='admm',
        model=FitModel(
            feasible=FixedValue([0.3, 0.5]),
            regularization=L2Regularization(1.0),
        ),
    )
    assert result.final_state_available and result.fit.status == 'optimal'
    assert result.fit.objective.hard_constraints_satisfied
    assert result.active_mask.tolist() == [False, True]
    np.testing.assert_array_equal(result.diagnostics.residuals, [0.0, 0.0])
    assert result.to_report()['fit']['fit_records'][0]['constraint_index'] == 0
