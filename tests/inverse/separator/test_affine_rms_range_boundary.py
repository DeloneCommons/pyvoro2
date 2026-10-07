"""RR2: affine RMS availability belongs to the exact complete aggregate."""

from decimal import Context, Decimal, DefaultContext, Inexact, localcontext
from fractions import Fraction
import json
import warnings

import numpy as np
import pytest

from pyvoro2.inverse.separator import (
    FitModel, HuberLoss, L2Regularization, build_power_fit_problem,
    build_power_fit_result, fit_weights_from_separators,
    resolve_separator_observations,
)
from pyvoro2.inverse.separator._diagnostics import (
    _affine_diagnostic, _affine_rms, _derived,
)


MAXIMUM = float(np.finfo(float).max)
TINY = float.fromhex('0x0.0000000000001p-1022')
RANGE = 'out_of_binary64_range'
DEPENDENCY = 'unavailable_dependency'


def _exact_rows(beta, alpha, left, right, target, *scales):
    arrays = np.broadcast_arrays(beta, alpha, left, right, target, *scales)
    result = []
    for operands in zip(*(v.ravel() for v in arrays)):
        b, a, l, r, t, *factors = (Fraction(float(v)) for v in operands)
        value = b + a * l - a * r - t
        for factor in factors:
            value *= factor
        result.append(value)
    return result


def _oracle(rows):
    if not rows:
        return 0., None
    total = sum((row**2 for row in rows), Fraction())
    if total > len(rows) * Fraction(MAXIMUM)**2:
        return np.inf, RANGE
    mean = total / len(rows)
    with localcontext(Context(prec=1000)):
        value = (Decimal(mean.numerator) / Decimal(mean.denominator)).sqrt()
    return float(value), None


def _strict_report(fit, observations):
    report = fit.to_report(observations)
    assert report['schema'] == {
        'name': 'pyvoro2.inverse.separator.report', 'version': 3,
    }
    assert json.loads(json.dumps(report, allow_nan=False)) == report
    return report


@pytest.mark.parametrize('fp_policy', ['warn', 'raise'])
@pytest.mark.parametrize('step', [-1, 0, 1], ids=['below', 'witness', 'above'])
def test_public_source_rms_boundary(step, fp_policy):
    weight = float.fromhex('0x1.eb97e455b9edap+1023')
    if step:
        weight = float(np.nextafter(weight, np.inf if step > 0 else 0.))
    points = [[0., 0.], [.0625, 0.], [10., 0.], [11., 0.]]
    exact = _exact_rows(
        [.03125] + [.5] * 58, [8.] + [.5] * 58,
        [weight] + [0.] * 58, 0., [.03125] + [.5] * 58,
    )
    assert exact == [8 * Fraction(weight)] + [Fraction()] * 58
    expected, reason = _oracle(exact)
    assert (reason is None) == (step <= 0)
    if step == 0:
        assert exact[0]**2 < 59 * Fraction(MAXIMUM)**2
        assert expected == MAXIMUM
    with warnings.catch_warnings(record=True) as caught, np.errstate(all=fp_policy):
        warnings.simplefilter('always', RuntimeWarning)
        observations = resolve_separator_observations(
            points, [(0, 1, .03125)] + [(2, 3, .5)] * 58,
            measurement='position', confidence=np.zeros(59),
        )
        fit = fit_weights_from_separators(
            points, observations, model=FitModel(
                regularization=L2Regularization(0., [weight, 0., 0., 0.]),
            ),
        )
        assert fit.status == 'optimal' and fit.converged
        np.testing.assert_array_equal(fit.weights, [weight, 0., 0., 0.])
        assert fit.objective.total == 0. and np.all(np.isfinite(fit.radii))
        report = _strict_report(fit, observations)
    assert not [w for w in caught if issubclass(w.category, RuntimeWarning)]
    assert fit.rms_residual == expected
    assert report['unavailable_diagnostics'].get('/summary/rms_residual') == reason
    assert report['summary']['rms_residual'] == (None if reason else expected)
    assert report['fit_records'][0]['unavailable_diagnostics']['/residual'] == RANGE


@pytest.mark.parametrize('fp_policy', ['warn', 'raise'])
@pytest.mark.parametrize('step', [-1, 0, 1], ids=['below', 'witness', 'above'])
def test_public_weighted_rms_boundary_with_finite_huber_objective(step, fp_policy):
    target = float.fromhex('0x1.eb97e455b9edap+1021')
    if step:
        target = float(np.nextafter(target, np.inf if step > 0 else 0.))
    targets = [target] + [8.] * 58
    confidence = np.array([1024.] + [0.] * 58)
    exact = _exact_rows(8., 1. / 32, 0., 0., targets, np.sqrt(confidence))
    expected, reason = _oracle(exact)
    assert (reason is None) == (step <= 0)
    assert exact[0]**2 > Fraction(MAXIMUM)**2  # L2 really is out of range.
    if step == 0:
        assert exact[0]**2 < 59 * Fraction(MAXIMUM)**2
        assert expected == MAXIMUM
    delta = 2.**-20
    objective = 1024 * Fraction(delta) * (
        Fraction(target) - 8 - Fraction(delta) / 2
    )
    with warnings.catch_warnings(record=True) as caught, np.errstate(all=fp_policy):
        warnings.simplefilter('always', RuntimeWarning)
        observations = resolve_separator_observations(
            [[0., 0.], [16., 0.]], [(0, 1, t) for t in targets],
            measurement='position', confidence=confidence,
        )
        problem = build_power_fit_problem(
            observations, model=FitModel(mismatch=HuberLoss(delta=delta)),
        )
        # A valid public external candidate need not claim solver optimality.
        fit = build_power_fit_result(
            problem, np.zeros(2), status='external', converged=False,
        )
        assert np.isfinite(fit.objective.total)
        assert fit.objective.total == float(objective)
        assert np.all(np.isfinite(fit.radii))
        report = _strict_report(fit, observations)
    assert not [w for w in caught if issubclass(w.category, RuntimeWarning)]
    assert fit.edge_diagnostics.weighted_rmse == expected
    assert report['unavailable_diagnostics'].get(
        '/edge_diagnostics/weighted_rmse',
    ) == reason
    assert report['edge_diagnostics']['weighted_rmse'] == (
        None if reason else expected
    )
    assert report['edge_diagnostics']['weighted_l2'] is None
    assert report['unavailable_diagnostics']['/edge_diagnostics/weighted_l2'] == RANGE


@pytest.mark.parametrize('fp_policy', ['warn', 'raise'])
@pytest.mark.parametrize('sign', [-1., 1.])
@pytest.mark.parametrize('offset', [-TINY, 0., TINY], ids=['below', 'at', 'above'])
@pytest.mark.parametrize('sparse', [False, True], ids=['finite-rows', 'range-row'])
def test_exact_boundary_even_when_rows_round_to_finite_max(
    sparse, offset, sign, fp_policy,
):
    # All final values round to MAX, but MAX + TINY is semantically unavailable.
    targets = [-sign * offset] + [0.] * 3 if sparse else [-sign * offset] * 3
    left = [sign * MAXIMUM] + [0.] * 3 if sparse else [sign * MAXIMUM] * 3
    operands = (0., 1., left, 0., targets)
    scales = (2.,) if sparse else ()
    expected, reason = _oracle(_exact_rows(*operands, *scales))
    assert reason == (RANGE if offset > 0 else None)
    if reason is None:
        assert expected == MAXIMUM
    with np.errstate(all=fp_policy):
        rows = _affine_diagnostic(*operands, *scales)
        actual = _affine_rms(*operands, rows, *scales)
    assert actual.reasons == reason and actual.value == expected


@pytest.mark.parametrize('fp_policy', ['warn', 'raise'])
@pytest.mark.parametrize('operands,scales', [
    ((0., 1., [0., 0.], 0., 0.), ()),
    ((.5, 2., [1., -3., 0.], 4., [0., 3., 1.]), ([.5, 2., 0.], -2.)),
    ((0., 8., [MAXIMUM, 0., 0., 0.], 0., 0.), (.25,)),
    ((0., MAXIMUM, [MAXIMUM] * 3, MAXIMUM, TINY), ()),
    ((0., MAXIMUM, [MAXIMUM] * 3, MAXIMUM, 1.), (2.**-600, 2.**600)),
    ((0., 8., [MAXIMUM, MAXIMUM], 0., 0.), ([0., 2.**-500], 2.**-500)),
], ids=['zero', 'moderate', 'recover-range-row', 'cancel-to-tiny',
        'cancel-scaled', 'include-every-scale'])
def test_complete_affine_rms_controls(operands, scales, fp_policy):
    expected, reason = _oracle(_exact_rows(*operands, *scales))
    with np.errstate(all=fp_policy):
        rows = _affine_diagnostic(*operands, *scales)
        actual = _affine_rms(*operands, rows, *scales)
    assert actual.reasons == reason
    assert actual.value == pytest.approx(expected, rel=2e-15, abs=0.)


@pytest.mark.parametrize('fp_policy', ['warn', 'raise'])
@pytest.mark.parametrize('field', range(6), ids=[
    'beta', 'alpha', 'left', 'right', 'target', 'scale',
])
def test_dependency_checked_before_range_exit_or_cancellation(field, fp_policy):
    operands = [[0., 0.], [8., 1.], [MAXIMUM, 0.], [0., 0.], [0., 0.], [1., 1.]]
    operands[field][1] = np.inf
    with np.errstate(all=fp_policy):
        rows = _affine_diagnostic(*operands)
        actual = _affine_rms(*operands[:5], rows, operands[5])
    assert actual.reasons == DEPENDENCY and np.isnan(actual.value)


@pytest.mark.parametrize('fp_policy', ['warn', 'raise'])
def test_zero_scale_absence_and_typed_dependency_remain_distinct(fp_policy):
    operands = (0., np.inf, [0., 0.], 0., 0.)
    with np.errstate(all=fp_policy):
        rows = _affine_diagnostic(*operands, 0.)
        absent = _affine_rms(*operands, rows, 0.)
        unavailable = _derived(np.array([np.nan, 0.]), operands_available=[False, True])
        dependency = _affine_rms(0., 1., [0., 0.], 0., 0., unavailable, 0.)
    assert absent.value == 0. and absent.reasons is None
    assert np.isnan(dependency.value) and dependency.reasons == DEPENDENCY


@pytest.mark.parametrize('fp_policy', ['warn', 'raise'])
def test_empty_affine_and_public_summaries_remain_available_zero(fp_policy):
    with np.errstate(all=fp_policy):
        for scales in ((), (np.array([]),)):
            rows = _affine_diagnostic([], [], [], [], [], *scales)
            actual = _affine_rms([], [], [], [], [], rows, *scales)
            assert actual.value == 0. and actual.reasons is None
        points = [[0., 0.], [1., 0.]]
        observations = resolve_separator_observations(points, [], allow_empty=True)
        fit = fit_weights_from_separators(points, observations)
        report = _strict_report(fit, observations)
    assert fit.rms_residual == 0.
    assert all(getattr(fit.edge_diagnostics, name) == 0. for name in (
        'weighted_l2', 'weighted_rmse', 'rmse', 'mae',
    ))
    assert report['unavailable_diagnostics'] == {}


def test_affine_rms_does_not_inherit_decimal_defaults():
    original = DefaultContext.copy()
    try:
        DefaultContext.traps[Inexact] = True
        DefaultContext.Emax = 100
        operands = (0., 8., [float.fromhex('0x1.eb97e455b9edap+1023')]
                    + [0.] * 58, 0., 0.)
        with np.errstate(all='raise'):
            rows = _affine_diagnostic(*operands)
            actual = _affine_rms(*operands, rows)
        assert actual.value == MAXIMUM and actual.reasons is None
    finally:
        DefaultContext.Emax = original.Emax
        DefaultContext.traps = original.traps
