"""Complete algebraic aggregates survive unavailable individual rows (#116)."""

from decimal import Decimal, localcontext
from fractions import Fraction
import json

import numpy as np
import pytest

from pyvoro2.inverse.separator import (
    FitModel,
    L2Regularization,
    dumps_report_json,
    fit_weights_from_separators,
    resolve_separator_observations,
)
from pyvoro2.inverse.separator._diagnostics import _derived
from pyvoro2.inverse.separator.problem import (
    _MeasurementGeometry,
    _algebraic_reduction,
)


MAXIMUM = float(np.finfo(float).max)
TINY = float.fromhex('0x0.0000000000001p-1022')
RANGE = 'out_of_binary64_range'
DEPENDENCY = 'unavailable_dependency'


def _materialize(exact):
    try:
        return float(exact)
    except OverflowError:
        return -np.inf if exact < 0 else np.inf


def _oracle(rows):
    """Round only complete exact reductions of the actual binary64 operands."""
    mean_square = sum((value * value for value in rows), Fraction()) / len(rows)
    with localcontext() as context:
        context.prec = 1000
        rmse = float((Decimal(mean_square.numerator)
                      / Decimal(mean_square.denominator)).sqrt())
    mae = _materialize(sum(map(abs, rows), Fraction()) / len(rows))
    return rmse, mae


def _assert_finite_report_aggregates(fit, observations, expected):
    report = fit.to_report(observations)
    assert report['schema'] == {
        'name': 'pyvoro2.inverse.separator.report', 'version': 3,
    }
    assert json.loads(dumps_report_json(report)) == report
    for name, value in zip(('rmse', 'mae'), expected):
        assert np.isfinite(value)
        assert getattr(fit.edge_diagnostics, name) == pytest.approx(value, rel=2e-15)
        assert report['edge_diagnostics'][name] == pytest.approx(value, rel=2e-15)
        assert f'/edge_diagnostics/{name}' not in report['unavailable_diagnostics']
    return report


def test_finite_algebraic_aggregates_do_not_abort_valid_fit():
    reference = np.array([MAXIMUM / 16 * 5, -MAXIMUM / 16 * 5])
    points = [[0., 0.], [.5, 0.]]
    observations = resolve_separator_observations(
        points, [(0, 1, -MAXIMUM), (0, 1, MAXIMUM)], confidence=[0., 0.],
    )
    fit = fit_weights_from_separators(
        points, observations,
        model=FitModel(regularization=L2Regularization(1., reference)),
    )
    assert fit.status == 'optimal' and fit.converged
    np.testing.assert_array_equal(fit.weights, reference)
    assert np.all(np.isfinite(fit.radii))
    assert fit.objective.total == 0.
    difference = Fraction(float(reference[0])) - Fraction(float(reference[1]))
    rows = [(Fraction(float(target)) - Fraction(1, 2)) / 2 - difference
            for target in observations.target]
    assert abs(rows[0]) > Fraction(MAXIMUM)
    expected = _oracle(rows)
    assert expected == (1.4388565604129415e308, 1.1235582092889472e308)
    report = _assert_finite_report_aggregates(fit, observations, expected)
    assert report['edge_diagnostics']['residual'][0] is None
    assert report['unavailable_diagnostics']['/edge_diagnostics/residual/0'] == RANGE
    assert report['fit_records'][0]['algebraic_residual'] is None
    assert report['fit_records'][0]['unavailable_diagnostics'][
        '/algebraic_residual'
    ] == RANGE


def test_small_alpha_aggregate_recovery_from_complete_finite_operands():
    points = [[0., 0.], [2., 0.]]
    observations = resolve_separator_observations(
        points, [(0, 1, MAXIMUM)] + [(0, 1, 1.)] * 63,
        measurement='position', confidence=np.zeros(64),
    )
    fit = fit_weights_from_separators(
        points, observations,
        model=FitModel(regularization=L2Regularization(1., [0., 0.])),
    )
    assert fit.status == 'optimal' and fit.converged
    np.testing.assert_array_equal(fit.weights, [0., 0.])
    assert fit.objective.total == 0.
    rows = [(Fraction(float(target)) - 1) * 4
            for target in observations.target]
    assert rows[0] > Fraction(MAXIMUM) and rows[1:] == [Fraction()] * 63
    expected = _oracle(rows)
    assert expected == (8.988465674311579e307, 1.1235582092889473e307)
    report = _assert_finite_report_aggregates(fit, observations, expected)
    for name in ('z_obs', 'residual'):
        assert report['edge_diagnostics'][name][0] is None
        assert report['unavailable_diagnostics'][f'/edge_diagnostics/{name}/0'] == RANGE
    assert report['fit_records'][0]['unavailable_diagnostics'] == {
        '/z_obs': RANGE, '/algebraic_residual': RANGE,
    }


@pytest.mark.parametrize('alpha,beta,target,left,right', [
    pytest.param(2., .5, [-MAXIMUM, MAXIMUM],
                 MAXIMUM / 16 * 5, -MAXIMUM / 16 * 5, id='alpha-above-one'),
    pytest.param(1., 0., [-MAXIMUM] + [MAXIMUM] * 63,
                 MAXIMUM / 2, -MAXIMUM / 2, id='alpha-one'),
    pytest.param(.25, 1., [MAXIMUM] + [1.] * 63,
                 0., 0., id='small-alpha'),
    pytest.param(-.25, 1., [MAXIMUM] + [1.] * 63,
                 0., 0., id='negative-alpha'),
    pytest.param(TINY, 0., [2.**-49] + [0.] * 63,
                 0., 0., id='reciprocal-overflow'),
    pytest.param(-TINY, 0., [2.**-49] + [0.] * 63,
                 0., 0., id='negative-reciprocal-overflow'),
    pytest.param(2.**-1022, 0., [8.] + [0.] * 63,
                 0., 0., id='smallest-normal-alpha'),
    pytest.param(TINY, 0., [0.] + [np.ldexp(MAXIMUM, -1073)] * 63,
                 MAXIMUM, -MAXIMUM, id='mixed-range-and-exact-cancellation'),
    pytest.param(MAXIMUM, 0., [0.] * 64,
                 MAXIMUM, -MAXIMUM, id='complete-products-overflow'),
    pytest.param(.25, 0., [MAXIMUM] * 64,
                 0., 0., id='true-aggregate-out-of-range'),
])
def test_algebraic_aggregate_recovery_boundaries(alpha, beta, target, left, right):
    # Synthetic signed coefficients exercise the private arithmetic boundary;
    # source geometry remains positive and public solver admission is unchanged.
    count = len(target)
    alpha_array = np.full(count, alpha)
    geom = _MeasurementGeometry(
        alpha_array, np.full(count, beta), np.array(target),
        np.array(target), np.array(target),
    )
    difference = Fraction(float(left)) - Fraction(float(right))
    observed = [(Fraction(float(t)) - Fraction(beta)) / Fraction(alpha)
                for t in target]
    rows = [value - difference for value in observed]
    expected = _oracle(rows)
    z_obs = _derived(np.array([_materialize(value) for value in observed]))
    z_fit = _derived(np.full(count, _materialize(difference)))
    residual = _derived(np.array([_materialize(value) for value in rows]))
    with np.errstate(all='raise'):
        for mean_abs, value in zip((False, True), expected):
            actual = _algebraic_reduction(
                geom, np.full(count, left), np.full(count, right),
                z_obs, z_fit, residual, mean_abs=mean_abs,
            )
            if np.isfinite(value):
                assert actual.reasons is None
                assert actual.value == pytest.approx(value, rel=2e-15)
            else:
                assert actual.reasons == RANGE and np.isposinf(actual.value)


@pytest.mark.parametrize('alpha', [0., np.inf])
def test_algebraic_aggregate_keeps_genuinely_unavailable_dependency(alpha):
    geom = _MeasurementGeometry(
        np.array([alpha]), np.array([0.]), np.array([1.]),
        np.array([1.]), np.array([1.]),
    )
    # Undefined division or an unavailable coefficient is not a range proof.
    unavailable = _derived(np.array([np.nan]), operands_available=False)
    for mean_abs in (False, True):
        actual = _algebraic_reduction(
            geom, np.zeros(1), np.zeros(1), unavailable, _derived(np.zeros(1)),
            unavailable, mean_abs=mean_abs,
        )
        assert actual.reasons == DEPENDENCY and np.isnan(actual.value)
