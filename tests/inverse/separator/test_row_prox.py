"""Independent WP10 proximal anchors."""

from fractions import Fraction

import numpy as np
import pytest

from pyvoro2.inverse.separator import FitModel, SoftIntervalPenalty


@pytest.mark.parametrize('family', ['soft', 'exponential', 'reciprocal'])
def test_nondyadic_affine_compilation_matches_original_parameter_derivative(family):
    from pyvoro2.inverse.separator import (
        ExponentialBoundaryPenalty, ReciprocalBoundaryPenalty,
    )
    from pyvoro2.inverse.separator._scalar_prox import _compile_scalar_prox_spec
    from pyvoro2.inverse.separator._objective import (
        _scalar_derivative_exact_expression, _scalar_derivative_terms,
    )

    constructors = {
        'soft': SoftIntervalPenalty,
        'exponential': ExponentialBoundaryPenalty,
        'reciprocal': ReciprocalBoundaryPenalty,
    }
    penalty = constructors[family](lower=0., upper=1., strength=1.3)
    scale, offset = Fraction(3, 7), Fraction(-1, 11)
    spec = _compile_scalar_prox_spec(
        FitModel(penalties=(penalty,)), penalty_affines=((scale, offset),),
    )
    for y in (-.1, .01, .13, .75, 2.3, 3.):
        point = Fraction.from_float(y)
        q = scale * point + offset
        strength = Fraction.from_float(penalty.strength)
        expected = point  # rho=1, v=0, confidence=0
        exponentials = {}
        if family == 'soft':
            expected += 2 * strength * scale * (min(q, 0) + max(q-1, 0))
        elif family == 'reciprocal':
            margin = Fraction.from_float(penalty.margin)
            epsilon = Fraction.from_float(penalty.epsilon)
            for sign, distance in [(-1, q), (1, 1-q)]:
                if distance < margin:
                    expected += sign * scale * strength / max(distance, epsilon)**2
        else:
            margin = Fraction.from_float(penalty.margin)
            tau = Fraction.from_float(penalty.tau)
            exponentials[(margin-q)/tau] = -scale*strength/tau
            exponentials[(q-1+margin)/tau] = scale*strength/tau
        expression = _scalar_derivative_exact_expression(
            spec.objective, y=y, target=0., confidence=0., v=0., rho=1., side='plus',
        )
        assert expression.rational == expected
        assert {exponent: coefficient for coefficient, exponent in
                expression.exponentials} == exponentials
        if family != 'exponential':
            balls = _scalar_derivative_terms(
                spec.objective, y=y, target=0., confidence=0., v=0., rho=1.,
            )
            lower, upper = balls.plus.physical_bounds()
            assert lower <= float(expected) <= upper


def test_policy_complete_prox_key_separates_the_analytic_strength_anchor():
    from pyvoro2.inverse.separator.solver import _prox_measurement_objective

    result = _prox_measurement_objective(
        np.array([.5, .5]), np.array([.5, .5]), np.ones(2),
        model=FitModel(penalties=(SoftIntervalPenalty(0., .25, [1., 4.]),)),
        rho=1., y_lo=np.zeros(2), y_hi=np.ones(2),
    )
    np.testing.assert_array_equal(result, [float(Fraction(3, 8)),
                                           float(Fraction(3, 10))])


def test_transformed_soft_prox_preserves_exact_original_units():
    from pyvoro2.inverse.separator._scalar_prox import (
        _compile_scalar_prox_spec, _solve_scalar_prox_coordinate,
    )

    # p is the prox coordinate; the penalty is applied to f=p/2.
    spec = _compile_scalar_prox_spec(
        FitModel(penalties=(SoftIntervalPenalty(.375, .625, 8.),)),
        penalty_affines=((Fraction(1, 2), Fraction(0)),),
    )
    result = _solve_scalar_prox_coordinate(
        spec=spec, target=.5, confidence=1., v=.5, rho=1., lower=0., upper=2.,
    )
    assert result.value == float(Fraction(2, 3))
    assert result.certificate_kind in ('exact_point', 'adjacent_bracket', 'point')


def test_nondyadic_transformed_batch_matches_independent_rational_minima():
    from pyvoro2.inverse.separator._scalar_prox import (
        _compile_scalar_prox_spec, _solve_scalar_prox_batch_ordinary,
    )

    scale, offset = Fraction(3, 7), Fraction(-1, 11)
    model = FitModel(penalties=(SoftIntervalPenalty(.3, .6, 1.3),))
    spec = _compile_scalar_prox_spec(model, penalty_affines=((scale, offset),))
    target = np.linspace(.3, .5, 16)
    v = np.linspace(.2, .4, 16)
    strength = Fraction.from_float(1.3)
    expected = [float((Fraction.from_float(t)+Fraction.from_float(x) +
                       2*strength*scale*(Fraction.from_float(.3)-offset)) /
                      (2+2*strength*scale**2)) for t, x in zip(target, v)]
    outcome = _solve_scalar_prox_batch_ordinary(
        spec=spec, initial=(v+target)/2, target=target,
        confidence=np.ones(16), v=v, rho=1.,
        lower=np.zeros(16), upper=np.full(16, 2.),
    )
    assert np.all(outcome.certified)
    np.testing.assert_array_equal([r.value for r in outcome.results], expected)


def test_heterogeneous_policies_retain_genuine_homogeneous_batches(monkeypatch):
    from pyvoro2.inverse.separator import solver

    real_batch = solver._solve_scalar_prox_batch_ordinary
    certified = []

    def observed_batch(**kwargs):
        outcome = real_batch(**kwargs)
        certified.append((np.count_nonzero(outcome.eligible),
                          np.count_nonzero(outcome.certified)))
        return outcome

    monkeypatch.setattr(solver, '_solve_scalar_prox_batch_ordinary', observed_batch)
    strengths = np.repeat([1., 4.], 16)
    target = np.tile(np.linspace(.45, .55, 16), 2)
    result = solver._prox_measurement_objective(
        target, target, np.ones(32),
        model=FitModel(penalties=(SoftIntervalPenalty(0., .25, strengths),)),
        rho=1., y_lo=np.zeros(32), y_hi=np.ones(32),
    )
    expected = [float((2*Fraction.from_float(t)+Fraction.from_float(s)/2) /
                      (2+2*Fraction.from_float(s))) for t, s in zip(target, strengths)]
    np.testing.assert_array_equal(result, expected)
    assert [eligible for eligible, _ in certified] == [16, 16]
    assert sum(count for _, count in certified) > 0
