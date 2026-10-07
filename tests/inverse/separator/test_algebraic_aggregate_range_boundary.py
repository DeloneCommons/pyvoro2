"""RR1: classify the exact aggregate at MAX, before rounding to binary64."""

from decimal import Decimal, DefaultContext, Inexact, Overflow, localcontext
from fractions import Fraction
import json
import warnings

import numpy as np
import pytest

from pyvoro2.inverse.separator import (
    FitModel, L2Regularization, fit_weights_from_separators,
    resolve_separator_observations,
)
from pyvoro2.inverse.separator._diagnostics import _derived
from pyvoro2.inverse.separator.problem import (
    _MeasurementGeometry, _algebraic_reduction,
)


MAXIMUM = float(np.finfo(float).max)
TINY = float.fromhex('0x0.0000000000001p-1022')
RANGE = 'out_of_binary64_range'
DEPENDENCY = 'unavailable_dependency'


def _oracle(rows, *, mean_abs):
    """Exact range predicate; decimal rounding is only a finite-value oracle."""
    power = 1 if mean_abs else 2
    total = sum((abs(row)**power for row in rows), Fraction())
    limit = len(rows) * Fraction(MAXIMUM)**power
    if total > limit:
        return np.inf, RANGE
    mean = total / len(rows)
    with localcontext() as context:
        context.prec = 1000
        value = Decimal(mean.numerator) / Decimal(mean.denominator)
        if not mean_abs:
            value = value.sqrt()
        return float(value), None


def _rounded(value):
    try:
        return float(value)
    except OverflowError:
        return -np.inf if value < 0 else np.inf


def _private_case(alpha, beta, target, left=0., right=0.):
    count = len(target)
    operands = np.broadcast_arrays(alpha, beta, target, left, right)
    a, b, t, l, r = (np.asarray(v, dtype=float) for v in operands)
    observed = [(Fraction(float(tk)) - Fraction(float(bk))) / Fraction(float(ak))
                for ak, bk, tk in zip(a, b, t)]
    fitted = [Fraction(float(lk)) - Fraction(float(rk)) for lk, rk in zip(l, r)]
    exact = [zk - wk for zk, wk in zip(observed, fitted)]
    assert len(exact) == count
    geom = _MeasurementGeometry(a, b, t, t, t)
    diagnostics = tuple(_derived(np.array([_rounded(v) for v in values]))
                        for values in (observed, fitted, exact))
    return (geom, l, r, *diagnostics), exact


@pytest.mark.parametrize('fp_policy', ['warn', 'raise'])
@pytest.mark.parametrize('step', [-1, 0, 1], ids=['below', 'witness', 'above'])
@pytest.mark.parametrize('count,distance,beta,target_hex,metric', [
    (59, 16., 8., '0x1.eb97e455b9edap+1021', 'rmse'),
    (105, 64., 32., '0x1.a3fffffffffffp+1023', 'mae'),
])
def test_public_aggregate_boundary_keeps_valid_fit(
    count, distance, beta, target_hex, metric, step, fp_policy,
):
    target = float.fromhex(target_hex)
    if step:
        target = float(np.nextafter(target, np.inf if step > 0 else 0.))
    points = [[0., 0.], [distance, 0.]]
    alpha = 1. / (2 * distance)
    exact = [(Fraction(target) - Fraction(beta)) / Fraction(alpha)]
    exact += [Fraction()] * (count - 1)
    power = 1 if metric == 'mae' else 2
    assert (abs(exact[0])**power < count * Fraction(MAXIMUM)**power) == (step <= 0)
    with warnings.catch_warnings(record=True) as caught, np.errstate(all=fp_policy):
        warnings.simplefilter('always', RuntimeWarning)
        observations = resolve_separator_observations(
            points, [(0, 1, target)] + [(0, 1, beta)] * (count - 1),
            measurement='position', confidence=np.zeros(count),
        )
        fit = fit_weights_from_separators(
            points, observations,
            model=FitModel(regularization=L2Regularization(1., [0., 0.])),
        )
        assert fit.status == 'optimal' and fit.converged
        np.testing.assert_array_equal(fit.weights, [0., 0.])
        assert np.all(np.isfinite(fit.radii)) and fit.objective.total == 0.
        report = fit.to_report(observations)
    assert not [w for w in caught if issubclass(w.category, RuntimeWarning)]
    assert report['schema'] == {
        'name': 'pyvoro2.inverse.separator.report', 'version': 3,
    }
    assert json.loads(json.dumps(report, allow_nan=False)) == report
    for name in ('rmse', 'mae'):
        expected, reason = _oracle(exact, mean_abs=(name == 'mae'))
        path = f'/edge_diagnostics/{name}'
        if reason is None:
            assert report['unavailable_diagnostics'].get(path) is None
            assert report['edge_diagnostics'][name] == pytest.approx(
                expected, rel=2e-15,
            )
            assert getattr(fit.edge_diagnostics, name) == pytest.approx(
                expected, rel=2e-15,
            )
            if name == metric and step == 0:
                assert expected == MAXIMUM
                assert getattr(fit.edge_diagnostics, name) == MAXIMUM
        else:
            assert report['edge_diagnostics'][name] is None
            assert report['unavailable_diagnostics'][path] == RANGE
    assert report['edge_diagnostics']['residual'][0] is None
    assert report['unavailable_diagnostics']['/edge_diagnostics/residual/0'] == RANGE
    assert report['fit_records'][0]['unavailable_diagnostics'][
        '/algebraic_residual'
    ] == RANGE


@pytest.mark.parametrize('fp_policy', ['warn', 'raise'])
@pytest.mark.parametrize('sign', [1., -1.])
@pytest.mark.parametrize('step', [-1, 0, 1])
def test_reciprocal_mantissa_boundary_from_original_operands(sign, step, fp_policy):
    alpha = sign * float.fromhex('0x1.cb1855f25cbc0p-2')
    target = float.fromhex('0x1.cb1855f25cbbfp+1023')
    if step:
        target = float(np.nextafter(target, np.inf if step > 0 else 0.))
    args, exact = _private_case(alpha, 0., [target, 0., 0., 0.])
    assert (exact[0]**2 < 4 * Fraction(MAXIMUM)**2) == (step <= 0)
    with np.errstate(all=fp_policy):
        actual = _algebraic_reduction(*args, mean_abs=False)
    expected, reason = _oracle(exact, mean_abs=False)
    assert actual.reasons == reason
    assert actual.value == pytest.approx(expected, rel=2e-15)


@pytest.mark.parametrize('fp_policy', ['warn', 'raise'])
@pytest.mark.parametrize('sign', [1., -1.])
@pytest.mark.parametrize('beta', [TINY, 0., -TINY], ids=['below', 'exact', 'above'])
@pytest.mark.parametrize('sparse', [False, True], ids=['finite-rows', 'range-row'])
@pytest.mark.parametrize('mean_abs', [False, True], ids=['rmse', 'mae'])
def test_semantic_boundary_precedes_binary64_overflow_rounding(
    mean_abs, sparse, beta, sign, fp_policy,
):
    # A subnormal change straddles MAX but all three final values round to
    # MAX. Testing float overflow alone would accept the out-of-range side.
    count = (2 if mean_abs else 4) if sparse else 3
    alpha = sign * (.5 if sparse else 1.)
    targets = [MAXIMUM] + [beta] * (count - 1) if sparse else [MAXIMUM] * count
    args, exact = _private_case(alpha, beta, targets)
    expected, reason = _oracle(exact, mean_abs=mean_abs)
    assert reason == (RANGE if beta < 0 else None)
    if reason is None:
        assert expected == MAXIMUM
    with np.errstate(all=fp_policy):
        actual = _algebraic_reduction(*args, mean_abs=mean_abs)
    assert actual.reasons == reason
    assert actual.value == expected


@pytest.mark.parametrize('fp_policy', ['warn', 'raise'])
@pytest.mark.parametrize('target', [0., TINY, 1.])
def test_large_cancelled_operands_keep_small_complete_aggregate(target, fp_policy):
    args, exact = _private_case(1., 0., [target] * 59, MAXIMUM, MAXIMUM)
    with np.errstate(all=fp_policy):
        for mean_abs in (False, True):
            actual = _algebraic_reduction(*args, mean_abs=mean_abs)
            assert actual.reasons is None
            assert actual.value == _oracle(exact, mean_abs=mean_abs)[0] == target


@pytest.mark.parametrize('policy', ['inexact-trap', 'small-exponent-range'])
def test_rmse_does_not_inherit_mutable_decimal_defaults(policy):
    original = DefaultContext.copy()
    try:
        if policy == 'inexact-trap':
            DefaultContext.traps[Inexact] = True
        else:
            DefaultContext.Emax = 100
            DefaultContext.traps[Overflow] = False
        with np.errstate(all='raise'):
            test_public_aggregate_boundary_keeps_valid_fit(
                59, 16., 8., '0x1.eb97e455b9edap+1021', 'rmse', 0, 'raise',
            )
    finally:
        DefaultContext.Emax = original.Emax
        DefaultContext.traps = original.traps


@pytest.mark.parametrize('field,value', [
    ('alpha', 0.), ('alpha', np.inf), ('beta', np.inf),
    ('target', np.nan), ('left', np.inf), ('right', np.nan),
])
def test_genuine_dependency_dominates_an_earlier_range_row(field, value):
    # The first contribution alone exceeds the limit. The later dependency
    # must still be checked; neither early range exit nor cancellation can
    # license reconstructing unavailable accepted operands.
    operands = dict(alpha=[.25, 1.], beta=[0., 0.], target=[MAXIMUM, 0.],
                    left=[0., 0.], right=[0., 0.])
    operands[field][1] = value
    a, b, t, l, r = (
        np.asarray(operands[name])
        for name in ('alpha', 'beta', 'target', 'left', 'right')
    )
    geom = _MeasurementGeometry(a, b, t, t, t)
    rows = _derived(np.array([np.inf, np.nan]), operands_available=[True, False])
    for mean_abs in (False, True):
        with np.errstate(all='raise'):
            actual = _algebraic_reduction(
                geom, l, r, rows, _derived(np.zeros(2)), rows, mean_abs=mean_abs,
            )
        assert actual.reasons == DEPENDENCY and np.isnan(actual.value)
