"""#118 R1/R2: original source operands authorize diagnostic availability."""

from contextlib import contextmanager
import copy
from dataclasses import replace
from fractions import Fraction as Q
import json
import math
import pickle
import sys
import warnings

import numpy as np
import pytest

from pyvoro2.inverse.separator import (
    FitModel, FixedValue, HuberLoss, L2Regularization, SoftIntervalPenalty, SquaredLoss,
    build_power_fit_problem, build_power_fit_result, fit_weights_from_separators,
    resolve_separator_observations,
)
from pyvoro2.inverse.separator.problem import _edge_diagnostic_values


MAX = sys.float_info.max
TINY = float.fromhex('0x0.0000000000001p-1022')
LIMIT = Q(MAX)
RANGE = 'out_of_binary64_range'


@contextmanager
def _evaluation(policy):
    # Source resolution happens before this context: the strict witness tests
    # already accepted operands, independently of subnormal allclose checks.
    with warnings.catch_warnings(record=True) as caught, np.errstate(all=policy):
        warnings.simplefilter('always', RuntimeWarning)
        yield
    assert not [w for w in caught if issubclass(w.category, RuntimeWarning)]


def _report(fit, observations):
    report = fit.to_report(observations)
    assert report['schema'] == {
        'name': 'pyvoro2.inverse.separator.report', 'version': 3,
    }
    assert json.loads(json.dumps(report, allow_nan=False)) == report
    for path, reason in report['unavailable_diagnostics'].items():
        assert reason in (RANGE, 'unavailable_dependency')
        leaf = report
        for part in path[1:].split('/'):
            key = part.replace('~1', '/').replace('~0', '~')
            leaf = leaf[int(key)] if isinstance(leaf, list) else leaf[key]
        assert leaf is None
    return report


def _edge_leaf(report, name, exact):
    key = 'algebraic_residual' if name == 'residual' else name
    row = report['fit_records'][0]
    unavailable = abs(exact) > LIMIT
    for value, path, reasons in (
        (report['edge_diagnostics'][name][0], '/edge_diagnostics/' + name + '/0',
         report['unavailable_diagnostics']),
        (row[key], '/fit_records/0/' + key, report['unavailable_diagnostics']),
        (row[key], '/' + key, row['unavailable_diagnostics']),
    ):
        assert (value is None) == unavailable
        assert reasons.get(path) == (RANGE if unavailable else None)
        if not unavailable:
            assert value == pytest.approx(float(exact), rel=2e-15, abs=0.)


def _zero_fit(points, target, measurement, weights):
    observations = resolve_separator_observations(
        points, [(0, 1, target)], measurement=measurement, confidence=[0.],
    )
    model = FitModel(regularization=L2Regularization(0., weights))
    return observations, model


@pytest.mark.parametrize('policy', ['warn', 'raise'])
@pytest.mark.parametrize('sign', [-1., 1.])
@pytest.mark.parametrize('offset', [-1., 0., 1., -TINY, TINY])
@pytest.mark.parametrize('distance', [1., 2.])
def test_source_fitted_difference_range_precedes_rounding(
    policy, sign, offset, distance,
):
    weights = [sign * MAX, sign * offset]
    points = [[0., 0.], [distance, 0.]]
    observations, model = _zero_fit(points, distance / 2, 'position', weights)
    difference = Q(weights[0]) - Q(weights[1])
    with _evaluation(policy):
        fit = fit_weights_from_separators(points, observations, model=model)
        assert fit.status == 'optimal' and fit.converged
        assert fit.objective.total == 0.
        np.testing.assert_array_equal(fit.weights, weights)
        report = _report(fit, observations)
        _edge_leaf(report, 'z_fit', difference)
        _edge_leaf(report, 'z_obs', Q())
        _edge_leaf(report, 'residual', -difference)
        assert report['fit_records'][0]['residual'] == float(
            difference / (2 * Q(distance)),
        )
        assert '/residual' not in report['fit_records'][0]['unavailable_diagnostics']
        # The public source prediction's difference has the same source owner.
        prediction = build_power_fit_problem(observations, model=model).predict(weights)
        assert np.isinf(prediction.difference[0]) == (abs(difference) > LIMIT)
        if abs(difference) > LIMIT:
            rounded = sign * MAX
            bad_edge = replace(fit.edge_diagnostics,
                               z_fit=[rounded], residual=[-rounded])
            with pytest.raises(ValueError, match='diagnostic|evaluation'):
                replace(fit, edge_diagnostics=bad_edge).to_report(observations)


@pytest.mark.parametrize('policy', ['warn', 'raise'])
@pytest.mark.parametrize('target', [-MAX, math.nextafter(-MAX, 0.), MAX])
def test_source_observation_difference_range_precedes_rounding(policy, target):
    points = [[0., 0.], [.5, .5]]
    observations, model = _zero_fit(points, target, 'fraction', [0., 0.])
    exact = Q(target) - Q(1, 2)  # Accepted d²=.5, alpha=1, beta=.5.
    with _evaluation(policy):
        fit = fit_weights_from_separators(points, observations, model=model)
        assert fit.status == 'optimal' and fit.converged and fit.objective.total == 0.
        report = _report(fit, observations)
        _edge_leaf(report, 'z_obs', exact)
        _edge_leaf(report, 'residual', exact)
        if abs(exact) > LIMIT:
            bad = replace(fit.edge_diagnostics, z_obs=[-MAX], residual=[-MAX])
            with pytest.raises(ValueError, match='diagnostic|evaluation'):
                replace(fit, edge_diagnostics=bad).to_report(observations)


CURVATURE_CONFIDENCE = float.fromhex('0x1.76cf41f212d78p+226')
CURVATURE_ALPHA = float.fromhex('0x1.a723f789854a0p+398')
CURVATURE_DISTANCE = float.fromhex('0x1.199999999999ap-200')


@pytest.mark.parametrize('policy', ['warn', 'raise'])
@pytest.mark.parametrize('confidence', [
    math.nextafter(CURVATURE_CONFIDENCE, 0.), CURVATURE_CONFIDENCE,
    math.nextafter(CURVATURE_CONFIDENCE, math.inf), 0., TINY, .7, 1.3,
])
def test_source_curvature_range_uses_original_confidence(policy, confidence):
    points = [[0., 0.], [CURVATURE_DISTANCE, 0.]]
    observations = resolve_separator_observations(
        points, [(0, 1, .5)], measurement='fraction', confidence=[confidence],
    )
    assert .5 / observations.distance2[0] == CURVATURE_ALPHA
    model = FitModel(mismatch=SquaredLoss(space='position'))
    problem = build_power_fit_problem(observations, model=model)
    exact = Q(confidence) * Q(CURVATURE_ALPHA)**2
    with _evaluation(policy):
        fit = fit_weights_from_separators(points, observations, model=model)
        assert fit.status == 'optimal' and fit.converged and fit.objective.total == 0.
        np.testing.assert_array_equal(fit.weights, [0., 0.])
        # Model curvature has separate units.
        assert np.isfinite(problem.edge_weight[0])
        report = _report(fit, observations)
        _edge_leaf(report, 'edge_weight', exact)
        if abs(exact) > LIMIT:
            with pytest.raises(ValueError, match='diagnostic|evaluation'):
                replace(fit, edge_diagnostics=replace(
                    fit.edge_diagnostics, edge_weight=[MAX],
                )).to_report(observations)


@pytest.mark.parametrize('policy', ['warn', 'raise'])
@pytest.mark.parametrize('distance,target,weights', [
    (1., MAX, [MAX, -1.]),
    (1., -MAX, [-MAX, 1.]),
    (.7, MAX, [MAX, -1.]),
])
def test_unavailable_edge_leaves_do_not_poison_complete_residual(
    policy, distance, target, weights,
):
    points = [[0., 0.], [distance, 0.]]
    observations, model = _zero_fit(points, target, 'fraction', weights)
    alpha = Q(float(.5 / observations.distance2[0]))
    z_obs = (Q(target) - Q(1, 2)) / alpha
    z_fit = Q(weights[0]) - Q(weights[1])
    exact = z_obs - z_fit
    assert abs(z_obs) > LIMIT or abs(z_fit) > LIMIT
    assert abs(exact) <= LIMIT
    with _evaluation(policy):
        fit = fit_weights_from_separators(points, observations, model=model)
        report = _report(fit, observations)
        _edge_leaf(report, 'z_obs', z_obs)
        _edge_leaf(report, 'z_fit', z_fit)
        _edge_leaf(report, 'residual', exact)


@pytest.mark.parametrize('policy', ['warn', 'raise'])
def test_both_unavailable_edge_leaves_can_have_finite_residual(policy):
    points = [[0., 0.], [.5, .5]]
    observations, model = _zero_fit(points, -MAX, 'fraction', [-MAX, 1.])
    with _evaluation(policy):
        fit = fit_weights_from_separators(points, observations, model=model)
        report = _report(fit, observations)
        _edge_leaf(report, 'z_obs', -LIMIT - Q(1, 2))
        _edge_leaf(report, 'z_fit', -LIMIT - 1)
        _edge_leaf(report, 'residual', Q(1, 2))
        assert report['edge_diagnostics']['rmse'] == .5
        assert report['edge_diagnostics']['mae'] == .5


@pytest.mark.parametrize('policy', ['warn', 'raise'])
def test_strict_diagnostic_cancellation_reaches_exact_dispatch_first(policy):
    distance = float.fromhex('0x1.6a09e667f3bd2p-513')
    weight = float.fromhex('0x1.0000000000008p+0')
    alpha = float.fromhex('0x1.ffffffffffff0p+1023')
    points = [[0., 0.], [distance, 0.]]
    observations = resolve_separator_observations(
        points, [(0, 1, .5)], measurement='fraction', confidence=[0.],
    )
    assert observations.distance2[0].hex() == '0x0.2000000000001p-1022'
    assert Q(alpha) * Q(weight) > LIMIT
    assert Q(1, 2) + Q(alpha) * Q(weight) - Q(alpha) * Q(weight) == Q(1, 2)
    model = FitModel(mismatch=SquaredLoss(space='position'),
                     regularization=L2Regularization(0., [weight, weight]))
    problem = build_power_fit_problem(observations, model=model)
    assert problem.alpha[0] != alpha  # Accepted source/model separation.
    with _evaluation(policy):
        assert problem.objective_breakdown([weight, weight]).total == 0.
        built = build_power_fit_result(
            problem, [weight, weight], status='optimal', converged=True,
            canonicalize_gauge=False,
        )
        fitted = fit_weights_from_separators(points, observations, model=model)
        reports = []
        for fit in (built, fitted):
            assert fit.status == 'optimal' and fit.converged
            np.testing.assert_array_equal(fit.weights, [weight, weight])
            assert fit.objective.total == 0.
            report = _report(fit, observations)
            assert report['fit_records'][0]['predicted'] == .5
            assert report['fit_records'][0]['residual'] == 0.
            assert report['unavailable_diagnostics'] == {}
            reports.append(report)
    for fit, report in zip((built, fitted), reports):
        # Deepcopy/pickle validate restored observations under ordinary input
        # policy; strict evaluation of those accepted copies remains required.
        copies = (copy.copy(fit), copy.deepcopy(fit), replace(fit),
                  pickle.loads(pickle.dumps(fit)))
        with _evaluation(policy):
            for copied in copies:
                assert copied.to_report(observations) == report


@pytest.mark.parametrize('policy', ['warn', 'raise'])
def test_exact_diagnostic_range_keeps_conditioned_hard_and_penalty_values(policy):
    observations = resolve_separator_observations(
        [[0., 0.], [.5, 0.]], [(0, 1, .25)], measurement='position', confidence=[0.],
    )
    problem = build_power_fit_problem(observations, model=FitModel(
        feasible=FixedValue(MAX), penalties=[SoftIntervalPenalty(-1., MAX, 1.)],
    ))
    assert Q(1, 4) + Q(MAX) == LIMIT + Q(1, 4)
    with _evaluation(policy):
        breakdown = problem.objective_breakdown([MAX, 0.])
        assert breakdown.hard_constraints_satisfied
        assert breakdown.penalties_total == 1. / 16
        fit = build_power_fit_result(
            problem, [MAX, 0.], status='optimal', converged=True,
            canonicalize_gauge=False,
        )
        report = _report(fit, observations)
        assert report['fit_records'][0]['predicted'] is None
        row = report['fit_records'][0]
        assert row['unavailable_diagnostics'] == {
            '/' + name: RANGE for name in (
                'predicted', 'predicted_fraction', 'predicted_position',
                'mismatch_predicted',
            )
        }
        assert report['fit_records'][0]['residual'] == MAX


@pytest.mark.parametrize('policy', ['warn', 'raise'])
def test_stratified_source_edge_sweep_from_original_operands(policy):
    distances = [2.**-200, CURVATURE_DISTANCE, .5, .7, 1., 1.3, 2., 2.**510]
    confidences = [0., TINY, math.nextafter(1., 0.), 1., math.nextafter(1., math.inf),
                   .7, 1.3, CURVATURE_CONFIDENCE, MAX]
    pairs = [(MAX, -TINY), (-MAX, TINY), (MAX, MAX), (1.3, -.7)]
    count = 0
    for measurement in ('fraction', 'position'):
        for d in distances:
            for c in confidences:
                for left, right in pairs:
                    # Three/five/seven rows exercise aggregate denominators
                    # while each edge leaf is classified independently.
                    accepted_limit = min(
                        LIMIT,
                        LIMIT / Q(d) if measurement == 'fraction' else LIMIT * Q(d),
                    )
                    boundary = math.nextafter(float(accepted_limit), 0.)
                    targets = [.5, -boundary, boundary] + [.7] * (2 * (count % 3))
                    observations = resolve_separator_observations(
                        [[0., 0.], [d, 0.]], [(0, 1, t) for t in targets],
                        measurement=measurement, confidence=[c] * len(targets),
                    )
                    source_alpha = float(.5 / (observations.distance2[0]
                                               if measurement == 'fraction'
                                               else observations.distance[0]))
                    source_beta = .5 if measurement == 'fraction' else d / 2
                    with _evaluation(policy):
                        values = _edge_diagnostic_values(
                            observations, np.array([left, right]),
                        )
                    for i, target in enumerate(targets):
                        obs = (Q(target) - Q(source_beta)) / Q(source_alpha)
                        fitted = Q(left) - Q(right)
                        expected = {
                            'z_obs': obs, 'z_fit': fitted, 'residual': obs - fitted,
                            'edge_weight': Q(c) * Q(source_alpha)**2,
                        }
                        for name, exact in expected.items():
                            diagnostic = values[name].item(i)
                            outside = abs(exact) > LIMIT
                            assert diagnostic.reasons == (RANGE if outside else None)
                            assert np.isinf(diagnostic.value) == outside
                            if outside:
                                assert np.signbit(diagnostic.value) == (exact < 0)
                    count += 1
    assert count == 576


@pytest.mark.parametrize('policy', ['warn', 'raise'])
@pytest.mark.parametrize('confidence', [0., TINY])
def test_source_edge_dependencies_and_zero_confidence_absence(policy, confidence):
    distance = 2.**-530
    points = [[0., 0.], [distance, 0.]]
    observations = resolve_separator_observations(
        points, [(0, 1, .5)], measurement='fraction', confidence=[confidence],
    )
    model = FitModel(mismatch=SquaredLoss(space='position'))
    with _evaluation(policy):
        fit = fit_weights_from_separators(points, observations, model=model)
        assert fit.status == 'optimal' and fit.converged and fit.objective.total == 0.
        report = _report(fit, observations)
        row = report['fit_records'][0]
        for name in ('z_obs', 'algebraic_residual'):
            assert row[name] is None
            assert row['unavailable_diagnostics']['/' + name] == (
                'unavailable_dependency'
            )
        assert row['z_fit'] == 0. and '/z_fit' not in row['unavailable_diagnostics']
        if confidence == 0.:
            assert row['edge_weight'] == 0.
            assert '/edge_weight' not in row['unavailable_diagnostics']
        else:
            assert row['edge_weight'] is None
            assert row['unavailable_diagnostics']['/edge_weight'] == (
                'unavailable_dependency'
            )


@pytest.mark.parametrize('policy', ['warn', 'raise'])
def test_ordinary_source_edge_reporting_retains_vectorized_path(policy, monkeypatch):
    import pyvoro2.inverse.separator._diagnostics as diagnostics

    def unexpected(*args, **kwargs):
        pytest.fail('ordinary reporting entered exceptional exact arithmetic')

    observations = resolve_separator_observations(
        [[0., 0.], [2., 0.]], [(0, 1, 1.)] * 1000,
        measurement='position', confidence=[.7] * 1000,
    )
    monkeypatch.setattr(diagnostics, '_exact_range_value', unexpected)
    monkeypatch.setattr(diagnostics, '_exact_affine_row', unexpected)
    monkeypatch.setattr(diagnostics, '_exact_mean_diagnostic', unexpected)
    with _evaluation(policy):
        values = _edge_diagnostic_values(observations, np.array([3., -1.]))
        assert values['z_obs'].value.tolist() == [0.] * 1000
        assert values['z_fit'].value.tolist() == [4.] * 1000
        assert values['residual'].value.tolist() == [-4.] * 1000
        assert values['edge_weight'].value.tolist() == [.7 / 16] * 1000
        assert all(reason is None for value in values.values()
                   for reason in (value.reasons if isinstance(value.reasons, tuple)
                                  else (value.reasons,)))


@pytest.mark.parametrize('policy', ['warn', 'raise'])
@pytest.mark.parametrize('scaled', [False, True])
def test_mixed_affine_dispatch_preserves_absence_dependencies_and_signs(policy, scaled):
    from pyvoro2.inverse.separator._diagnostics import _affine_diagnostic

    alpha = float.fromhex('0x1.ffffffffffff0p+1023')
    weight = float.fromhex('0x1.0000000000008p+0')
    operands = (
        [.5, .5, 0., 0., .5], [alpha, .5, 1., 1., np.inf],
        [weight, 3., -MAX, 0., 0.], [weight, -1., 0., 0., 0.],
        [.5, .5, TINY, 0., .5],
    )
    if scaled:
        # Structural absence wins over the last row's unavailable coefficient.
        operands = (*operands, [1., -2., 1., 0., 0.])
    with _evaluation(policy):
        value = _affine_diagnostic(*operands)
    assert value.value[:2].tolist() == [0., -4. if scaled else 2.]
    assert np.isneginf(value.value[2]) and value.reasons[2] == RANGE
    assert value.value[3] == 0. and value.reasons[3] is None
    if scaled:
        assert value.value[4] == 0. and value.reasons[4] is None
    else:
        assert np.isnan(value.value[4]) and value.reasons[4] == 'unavailable_dependency'


@pytest.mark.parametrize('policy', ['warn', 'raise'])
@pytest.mark.parametrize('sign', [-1., 1.])
@pytest.mark.parametrize('cancel', [False, True])
def test_scaled_affine_dispatch_bounds_product_prefixes(policy, sign, cancel):
    from pyvoro2.inverse.separator._diagnostics import _affine_diagnostic

    alpha = float.fromhex('0x1.0000000000008p+0')
    scale = sign * float.fromhex('0x1.ffffffffffff0p+1023')
    left = 2.**-20
    right = left if cancel else math.nextafter(left, 0.)
    exact = Q(scale) * Q(alpha) * (Q(left) - Q(right))
    assert abs(Q(scale) * Q(alpha)) > LIMIT and abs(exact) <= LIMIT
    with _evaluation(policy):
        value = _affine_diagnostic(0., alpha, left, right, 0., scale)
        assert value.reasons is None and value.value == float(exact)


@pytest.mark.parametrize('policy', ['warn', 'raise'])
@pytest.mark.parametrize('status,converged', [('external', False), ('optimal', True)])
def test_public_weighted_zero_residual_avoids_overflowing_product_prefixes(
    policy, status, converged,
):
    distance = float.fromhex('0x1.ffffffffffff8p-263')
    confidence = float.fromhex('0x1.fffffffffffe0p+1001')
    alpha = float.fromhex('0x1.0000000000008p+523')
    weight = 2.**-20
    observations = resolve_separator_observations(
        [[0., 0.], [distance, 0.]], [(0, 1, .5)], confidence=[confidence],
    )
    problem = build_power_fit_problem(
        observations, model=FitModel(mismatch=HuberLoss(delta=1.)),
    )
    assert problem.alpha[0] == alpha
    assert Q(float(np.sqrt(confidence))) * Q(alpha) > LIMIT
    assert Q(1, 2) + Q(alpha) * Q(weight) - Q(alpha) * Q(weight) - Q(1, 2) == 0
    with _evaluation(policy):
        fit = build_power_fit_result(
            problem, [weight, weight], status=status, converged=converged,
            canonicalize_gauge=False,
        )
        report = _report(fit, observations)
        assert fit.objective.total == 0. and fit.residuals[0] == 0.
        assert report['edge_diagnostics']['weighted_l2'] == 0.
        assert report['edge_diagnostics']['weighted_rmse'] == 0.
        assert report['fit_records'][0]['unavailable_diagnostics'] == {
            '/edge_weight': RANGE,
        }
