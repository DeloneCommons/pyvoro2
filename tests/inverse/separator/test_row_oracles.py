"""Original-unit Decimal oracles independent of effective penalty compilation."""

from decimal import Decimal, localcontext
from fractions import Fraction

import numpy as np
import pytest

from pyvoro2.inverse.separator import (
    ExponentialBoundaryPenalty, FitModel, HuberLoss, ReciprocalBoundaryPenalty,
    SoftIntervalPenalty, SquaredLoss, resolve_separator_observations,
)
from pyvoro2.inverse.separator._policy import _bind_policy
from pyvoro2.inverse.separator.problem import _penalty_affines
from pyvoro2.inverse.separator.solver import (
    _compile_row_prox_specs, _prox_measurement_objective,
)


def _decimal(value):
    value = value if isinstance(value, Fraction) else Fraction.from_float(float(value))
    return Decimal(value.numerator) / Decimal(value.denominator)


def _original_oracle(y, *, target, v, alpha_m, beta_m, alpha_p, beta_p, penalty,
                     huber, derivative):
    q = beta_p + alpha_p * (y-beta_m) / alpha_m
    scale = alpha_p / alpha_m
    lower, upper, strength = map(
        _decimal, (penalty.lower, penalty.upper, penalty.strength))
    residual = y-target
    delta = _decimal(.1)
    if derivative:
        mismatch = max(-delta, min(delta, residual)) if huber else residual
        value = mismatch + y-v
    else:
        mismatch = (delta*(abs(residual)-delta/2) if huber and abs(residual) > delta
                    else residual**2/2)
        value = mismatch + (y-v)**2/2
    if isinstance(penalty, SoftIntervalPenalty):
        lo, hi = min(q-lower, Decimal(0)), max(q-upper, Decimal(0))
        return value + (2*strength*scale*(lo+hi) if derivative else
                        strength*(lo**2+hi**2))
    margin = _decimal(penalty.margin)
    if isinstance(penalty, ExponentialBoundaryPenalty):
        tau = _decimal(penalty.tau)
        lo, hi = ((lower+margin-q)/tau).exp(), ((q-upper+margin)/tau).exp()
        return value + (strength*scale*(hi-lo)/tau if derivative else strength*(lo+hi))
    epsilon = _decimal(penalty.epsilon)
    for sign, distance in ((-1, q-lower), (1, upper-q)):
        if distance >= margin:
            continue
        if derivative:
            value += sign*scale*strength / max(distance, epsilon)**2
        elif distance > epsilon:
            value += strength*(1/distance-1/margin)
        else:
            value += strength*(1/epsilon-1/margin-(distance-epsilon)/epsilon**2)
    return value


@pytest.mark.parametrize('source', ['fraction', 'position'])
@pytest.mark.parametrize('mismatch', ['fraction', 'position'])
@pytest.mark.parametrize('family', ['soft', 'exponential', 'reciprocal'])
@pytest.mark.parametrize('huber', [False, True])
def test_unequal_distance_mixed_prox_matches_original_unit_decimal_minimum(
    source, mismatch, family, huber,
):
    # sqrt(2) and distance2=2 have independent binary64 affine operands.
    points = [[0., 0.], [1., 1.], [2., 0.]]
    rows = [(0, 1, .2), (0, 2, .8)]
    observations = resolve_separator_observations(points, rows, measurement=source)
    constructors = {'soft': lambda: SoftIntervalPenalty(.3, .7, .2),
                    'exponential': lambda: ExponentialBoundaryPenalty(
                        margin=.05, tau=.2, strength=.03),
                    'reciprocal': lambda: ReciprocalBoundaryPenalty(
                        margin=.1, epsilon=.01, strength=.0005)}
    space = 'position' if mismatch == 'fraction' else 'fraction'
    penalty = constructors[family]()
    from dataclasses import replace
    penalty = replace(penalty, space=space)
    loss = HuberLoss(.1, space=mismatch) if huber else SquaredLoss(space=mismatch)
    model = FitModel(mismatch=loss, penalties=(penalty,))
    policy = _bind_policy(observations, model)
    specs = _compile_row_prox_specs(model, 2, penalty_affines=_penalty_affines(policy))
    target = (observations.target_fraction if mismatch == 'fraction'
              else observations.target_position)
    inputs = np.array([-.15, 1.15])
    actual = _prox_measurement_objective(
        inputs, target, np.ones(2), model=model, rho=1.,
        y_lo=None, y_hi=None, spec=specs,
    )
    with localcontext() as context:
        context.prec = 80
        distances = zip(observations.distance, observations.distance2)
        for index, (d, d2) in enumerate(distances):
            # These are original-unit binary64 row laws, independently assembled.
            laws = {'fraction': (_decimal(.5/d2), _decimal(.5)),
                    'position': (_decimal(.5/d), _decimal(.5*d))}
            operands = dict(target=_decimal(target[index]), v=_decimal(inputs[index]),
                            alpha_m=laws[mismatch][0], beta_m=laws[mismatch][1],
                            alpha_p=laws[space][0], beta_p=laws[space][1],
                            penalty=penalty, huber=huber)
            lower, upper = Decimal(-10), Decimal(10)
            for _ in range(200):
                middle = (lower+upper)/2
                gradient = _original_oracle(middle, derivative=True, **operands)
                if gradient < 0:
                    lower = middle
                else:
                    upper = middle
            proposal = float((lower+upper)/2)
            candidates = [np.nextafter(proposal, -np.inf), proposal,
                          np.nextafter(proposal, np.inf)]
            expected = min(candidates, key=lambda candidate: _original_oracle(
                _decimal(candidate), derivative=False, **operands))
            assert actual[index] == expected
