"""#118: classify original diagnostic expressions before binary64 rounding."""

import copy
from dataclasses import replace
from decimal import Context, Decimal, localcontext
from fractions import Fraction
import json
import pickle
import sys
import warnings

import numpy as np
import pytest

from pyvoro2.inverse.separator import (
    ActiveSetOptions, FitModel, FixedValue, HuberLoss, L2Regularization,
    build_power_fit_problem, build_power_fit_result,
    fit_weights_from_separators, resolve_separator_observations,
    solve_self_consistent_power_weights,
)
from pyvoro2.inverse.separator._diagnostics import (
    _affine_diagnostic, _weighted_affine_norms,
)
from pyvoro2.planar import Box


MAX = sys.float_info.max
TINY = float.fromhex('0x0.0000000000001p-1022')
RANGE = 'out_of_binary64_range'


@pytest.fixture(params=['warn', 'raise'])
def fp_policy(request):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always', RuntimeWarning)
        with np.errstate(all=request.param):
            yield
    assert not [w for w in caught if issubclass(w.category, RuntimeWarning)]


def _oracle(rows, confidence, denominator):
    total = sum((Fraction(float(c)) * r**2 for r, c in zip(rows, confidence)),
                Fraction())
    if total > denominator * Fraction(MAX)**2:
        return None
    with localcontext(Context(prec=1000)):
        mean = Decimal(total.numerator) / Decimal(total.denominator) / denominator
        return float(mean.sqrt())


def _strict(report):
    assert report['schema'] == {
        'name': 'pyvoro2.inverse.separator.report', 'version': 3,
    }
    assert json.loads(json.dumps(report, allow_nan=False)) == report
    for path, reason in report['unavailable_diagnostics'].items():
        assert reason in (RANGE, 'unavailable_dependency')
        value = report
        for part in path[1:].split('/'):
            key = part.replace('~1', '/').replace('~0', '~')
            value = value[int(key)] if isinstance(value, list) else value[key]
        assert value is None
    return report


def _assert_scalar(report, key, expected):
    path = '/edge_diagnostics/' + key
    value = report['edge_diagnostics'][key]
    if expected is None:
        assert value is None
        assert report['unavailable_diagnostics'][path] == RANGE
    else:
        assert value == pytest.approx(expected, rel=2e-15, abs=0.)
        assert path not in report['unavailable_diagnostics']


@pytest.mark.parametrize('residuals,confidence', [
    ([MAX, 0.], [2., 0.]),
    ([MAX], [float.fromhex('0x1.0000000000001p+0')]),
    ([MAX, 1.], [1., 1.]),
    ([MAX, 0.], [1., 1.]),
    ([MAX], [0.]),
    ([MAX], [TINY]),
    ([MAX], [.7]),
    ([MAX], [1.3]),
    ([MAX], [MAX]),
    ([-MAX, -1.], [1., 1.]),
    ([3., -4., 0.], [.7, 1.3, 0.]),
], ids=['rmse-at-max', 'confidence-above-one', 'norm-above-max',
        'norm-at-max', 'absent', 'subnormal-confidence', 'below-one',
        'above-one', 'maximum-confidence', 'negative-norm', 'ordinary'])
def test_public_weighted_range_uses_original_confidence(
    residuals, confidence, fp_policy,
):
    points = [[0., 0.], [1., 0.]]
    observations = resolve_separator_observations(
        points, [(0, 1, -r) for r in residuals], measurement='position',
        confidence=confidence,
    )
    # beta + alpha*(-1) = 0 exactly; no rounded target changes the oracle.
    exact = [Fraction(float(r)) for r in residuals]
    delta = TINY if max(confidence) == MAX else 2.**-80
    problem = build_power_fit_problem(
        observations, model=FitModel(mismatch=HuberLoss(delta=delta)),
    )
    fit = build_power_fit_result(
        problem, np.array([-1., 0.]), status='external', converged=False,
        canonicalize_gauge=False,
    )
    assert np.isfinite(fit.objective.total)
    report = _strict(fit.to_report(observations))
    _assert_scalar(report, 'weighted_l2', _oracle(exact, confidence, 1))
    _assert_scalar(report, 'weighted_rmse',
                   _oracle(exact, confidence, len(exact)))
    if residuals == [MAX, 0.] and confidence == [2., 0.]:
        assert fit.edge_diagnostics.weighted_rmse == MAX


@pytest.mark.parametrize('count', [1, 4])
def test_public_row_and_max_cannot_round_away_range_exit(count, fp_policy):
    weight = 2.**970
    points = [[0., 0.], [1., 0.], [3., 0.], [4., 0.]]
    observations = resolve_separator_observations(
        points, [(0, 1, -MAX)] + [(2, 3, .5)] * (count - 1),
        measurement='position', confidence=np.zeros(count),
    )
    exact = [Fraction(MAX) + Fraction(weight) / 2 + Fraction(1, 2)]
    exact += [Fraction()] * (count - 1)
    assert exact[0] > Fraction(MAX) and float(exact[0]) == MAX
    fit = fit_weights_from_separators(
        points, observations, model=FitModel(
            regularization=L2Regularization(0., [weight, 0., 0., 0.]),
        ),
    )
    assert fit.status == 'optimal' and fit.converged
    assert fit.objective.total == 0.
    np.testing.assert_array_equal(fit.weights, [weight, 0., 0., 0.])
    assert np.isposinf(fit.residuals[0]) and np.isposinf(fit.max_residual)
    report = _strict(fit.to_report(observations))
    row = report['fit_records'][0]
    assert row['residual'] is None and row['mismatch_residual'] is None
    assert row['unavailable_diagnostics']['/residual'] == RANGE
    assert row['unavailable_diagnostics']['/mismatch_residual'] == RANGE
    assert report['summary']['max_residual'] is None
    assert report['unavailable_diagnostics']['/summary/max_residual'] == RANGE
    expected_rms = _oracle(exact, [1.] * count, count)
    assert report['summary']['rms_residual'] == expected_rms
    assert report['unavailable_diagnostics'].get('/summary/rms_residual') == (
        RANGE if expected_rms is None else None
    )
    assert fit.edge_diagnostics.weighted_l2 == 0.
    assert fit.edge_diagnostics.weighted_rmse == 0.
    for changed in (replace(fit, residuals=np.full(count, MAX)),
                    replace(fit, max_residual=MAX)):
        with pytest.raises(ValueError, match='diagnostic|evaluation'):
            changed.to_report(observations)


@pytest.mark.parametrize('sign', [-1., 1.])
@pytest.mark.parametrize('offset', [-TINY, 0., TINY, 2.**969])
@pytest.mark.parametrize('scaled', [False, True])
def test_affine_row_range_precedes_materialization(sign, offset, scaled, fp_policy):
    # Scaled and unscaled paths must share the same exact range semantics.
    scales = (-2., .5) if scaled else ()
    exact = Fraction(sign) * (Fraction(MAX) + Fraction(offset))
    if scaled:
        exact = -exact
    value = _affine_diagnostic(0., 1., sign * MAX, 0., -sign * offset, *scales)
    if abs(exact) > Fraction(MAX):
        assert value.reasons == RANGE and np.isinf(value.value)
        assert np.signbit(value.value) == (exact < 0)
    else:
        assert value.reasons is None and value.value == float(exact)


def test_affine_large_cancellation_is_still_available(fp_policy):
    value = _affine_diagnostic(.5, MAX, MAX, MAX, -.5)
    assert value.reasons is None and value.value == 1.


@pytest.mark.parametrize('field', range(6))
def test_weighted_dependencies_precede_range_exit(field, fp_policy):
    operands = [[0., 0.], [1., 1.], [MAX, 0.], [0., 0.], [0., 0.], [2., 1.]]
    operands[field][1] = np.inf
    for value in _weighted_affine_norms(*operands):
        assert value.reasons == 'unavailable_dependency' and np.isnan(value.value)


def test_weighted_zero_confidence_removes_unavailable_work(fp_policy):
    for value in _weighted_affine_norms(0., np.inf, 0., 0., 0., 0.):
        assert value.reasons is None and value.value == 0.


def test_ordinary_diagnostics_do_not_need_exact_rows(fp_policy, monkeypatch):
    import pyvoro2.inverse.separator._diagnostics as diagnostics

    def unexpected(*args):
        pytest.fail('ordinary-range diagnostics invoked exact row arithmetic')

    monkeypatch.setattr(diagnostics, '_exact_affine_row', unexpected)
    operands = (.5, .5, [3., -7., 0.], 0., .5)
    rows = diagnostics._affine_diagnostic(*operands)
    np.testing.assert_array_equal(rows.value, [1.5, -3.5, 0.])
    l2, rms = diagnostics._weighted_affine_norms(*operands, [1., 1., 0.])
    assert l2.value == pytest.approx(14.5**.5)
    assert rms.value == pytest.approx((14.5 / 3)**.5)


@pytest.mark.parametrize('status,converged', [('optimal', True), ('external', False)])
def test_unavailable_prediction_preserves_hard_certificate(
    status, converged, fp_policy,
):
    observations = resolve_separator_observations(
        [[0., 0.], [.5, 0.]], [(0, 1, MAX)], confidence=[0.],
    )
    problem = build_power_fit_problem(
        observations, model=FitModel(feasible=FixedValue(MAX)),
    )
    weights = np.array([MAX / 2, 0.])
    exact_prediction = Fraction(1, 2) + 2 * Fraction(float(weights[0]))
    assert exact_prediction > Fraction(MAX) and float(exact_prediction) == MAX
    # Existing hard semantics consume the conditioned binary64 prediction.
    # A diagnostic range flag must not alter that predicate or its tolerance.
    breakdown = problem.objective_breakdown(weights)
    assert breakdown.hard_constraints_satisfied
    assert breakdown.hard_max_violation == 0.
    assert np.isfinite(breakdown.hard_max_tolerance)
    fit = build_power_fit_result(
        problem, weights, status=status, converged=converged,
        canonicalize_gauge=False,
    )
    report = _strict(fit.to_report(observations))
    assert fit.status == status and fit.converged is converged
    assert fit.objective.total == 0.
    assert report['fit_records'][0]['predicted'] is None
    assert report['fit_records'][0]['unavailable_diagnostics']['/predicted'] == RANGE
    assert fit.residuals[0] == .5


@pytest.mark.parametrize('drop_after', [1, 10], ids=['marginal', 'selected'])
def test_native_active_range_propagation_and_provenance(
    drop_after, fp_policy, monkeypatch,
):
    import pyvoro2.inverse.separator.active as active

    original = active._match_realized_pairs
    evaluated = []

    def sample(*args, **kwargs):
        evaluated.append(np.asarray(kwargs['semantic_weights']).copy())
        return original(*args, **kwargs)

    # Read-only instrumentation; native topology and the solver remain real.
    monkeypatch.setattr(active, '_match_realized_pairs', sample)
    result = solve_self_consistent_power_weights(
        [[0., 0.], [1., 0.]], [(0, 1, -MAX)], measurement='position',
        confidence=[0.], domain=Box([[-1., 2.], [-1., 1.]]),
        model=FitModel(regularization=L2Regularization(0., [2.**970, 0.])),
        options=ActiveSetOptions(drop_after=drop_after, max_iter=2),
        connectivity_check='diagnose', unaccounted_pair_check='diagnose',
        return_history=True, return_cells=True,
    )
    assert result.final_state_available and result.fit.status == 'optimal'
    assert result.realized is not None and result.realized.cells is not None
    assert len(evaluated) == len(result.history) + 1
    assert result.active_mask.tolist() == [drop_after != 1]
    report = _strict(result.to_report())
    _strict(report['fit'])
    records = result.diagnostics.to_records()
    assert records[0]['residual'] is None
    assert records[0]['unavailable_diagnostics'] == {'/residual': RANGE}
    local = {'/residual': RANGE, '/mismatch_residual': RANGE}
    assert report['diagnostics'][0]['unavailable_diagnostics'] == local
    expected_map = {'/diagnostics/0' + p: r for p, r in local.items()}
    if drop_after == 1:
        assert report['marginal_records'][0]['unavailable_diagnostics'] == local
        expected_map.update({'/marginal_records/0' + p: r for p, r in local.items()})
    expected_map.update({
        '/summary/rms_residual_all': RANGE, '/summary/max_residual_all': RANGE,
    })
    for index, (history, weights) in enumerate(zip(result.history, evaluated)):
        exact = (Fraction(1, 2) + (Fraction(float(weights[0]))
                 - Fraction(float(weights[1]))) / 2 + Fraction(MAX))
        assert exact > Fraction(MAX)
        assert np.isposinf(history.rms_residual_all)
        assert np.isposinf(history.max_residual_all)
        assert np.isfinite(history.weight_step_norm)
        for name in ('rms_residual_all', 'max_residual_all'):
            expected_map[f'/history/{index}/{name}'] = RANGE
    expected_map.update({'/fit' + p: r for p, r in
                         report['fit']['unavailable_diagnostics'].items()})
    assert report['unavailable_diagnostics'] == expected_map
    assert result.diagnostics.to_records() == records  # Rebasing owns copies.
    for rebuilt in (copy.copy(result), copy.deepcopy(result), replace(result),
                    pickle.loads(pickle.dumps(result))):
        assert rebuilt.to_report() == report
    for rebuilt in (copy.copy(result.diagnostics), copy.deepcopy(result.diagnostics),
                    replace(result.diagnostics),
                    pickle.loads(pickle.dumps(result.diagnostics))):
        assert rebuilt.to_records() == records
    with pytest.raises(ValueError, match='diagnostic|evaluation|residual'):
        replace(result, diagnostics=replace(result.diagnostics, residuals=[MAX]))
