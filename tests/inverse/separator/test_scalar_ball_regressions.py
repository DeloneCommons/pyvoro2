"""Independent regressions for the R2 ball-certificate replacement."""

from __future__ import annotations

from decimal import Decimal, localcontext
from fractions import Fraction
import math
import random
import warnings

import numpy as np
import pytest

from pyvoro2.inverse import separator
import pyvoro2.inverse.separator._objective as scalar_objective
import pyvoro2.inverse.separator._scalar_prox as scalar_prox
from pyvoro2.inverse.separator import solver as separator_solver
from pyvoro2.inverse.separator._scalar_prox import (
    _compile_scalar_prox_spec,
    _ScalarProxError,
    _ScalarProxFailure,
    _solve_scalar_prox_batch_ordinary,
    _solve_scalar_prox_coordinate,
)


def _d(value: float) -> Decimal:
    return Decimal.from_float(float(value))


def _oracle_scalar_terms(
    model: separator.FitModel,
    *,
    y: float,
    target: float,
    confidence: float,
    v: float,
    rho: float,
) -> tuple[Decimal, Decimal]:
    """Independent smooth derivative and curvature at a binary64 point."""

    with localcontext() as context:
        context.prec = 200
        point = _d(y)
        residual = point - _d(target)
        if isinstance(model.mismatch, separator.HuberLoss):
            delta = _d(model.mismatch.delta)
            derivative = _d(confidence) * max(-delta, min(residual, delta))
            curvature = (
                _d(confidence) if -delta < residual < delta else Decimal(0)
            )
        else:
            derivative = _d(confidence) * residual
            curvature = _d(confidence)
        derivative += _d(rho) * (point - _d(v))
        curvature += _d(rho)
        for penalty in model.penalties:
            strength = _d(penalty.strength)
            if isinstance(penalty, separator.SoftIntervalPenalty):
                if point < _d(penalty.lower):
                    derivative += 2 * strength * (point - _d(penalty.lower))
                    curvature += 2 * strength
                elif point > _d(penalty.upper):
                    derivative += 2 * strength * (point - _d(penalty.upper))
                    curvature += 2 * strength
                continue
            if isinstance(penalty, separator.ExponentialBoundaryPenalty):
                tau = _d(penalty.tau)
                left = (
                    (_d(penalty.lower) + _d(penalty.margin) - point) / tau
                ).exp()
                right = (
                    (point - _d(penalty.upper) + _d(penalty.margin)) / tau
                ).exp()
                derivative += strength / tau * (right - left)
                curvature += strength / (tau * tau) * (right + left)
                continue
            margin = _d(penalty.margin)
            epsilon = _d(penalty.epsilon)
            lower_distance = point - _d(penalty.lower)
            upper_distance = _d(penalty.upper) - point
            if lower_distance < margin:
                distance = max(lower_distance, epsilon)
                derivative -= strength / (distance * distance)
                if lower_distance > epsilon:
                    curvature += 2 * strength / (distance**3)
            if upper_distance < margin:
                distance = max(upper_distance, epsilon)
                derivative += strength / (distance * distance)
                if upper_distance > epsilon:
                    curvature += 2 * strength / (distance**3)
        return +derivative, +curvature


def _oracle_scalar_objective(
    model: separator.FitModel,
    *,
    y: float,
    target: float,
    confidence: float,
    v: float,
    rho: float,
) -> Decimal:
    with localcontext() as context:
        context.prec = 200
        point = _d(y)
        residual = point - _d(target)
        if isinstance(model.mismatch, separator.HuberLoss):
            delta = _d(model.mismatch.delta)
            mismatch = (
                residual * residual / 2
                if abs(residual) <= delta
                else delta * (abs(residual) - delta / 2)
            )
        else:
            mismatch = residual * residual / 2
        value = _d(confidence) * mismatch
        value += _d(rho) * (point - _d(v)) ** 2 / 2
        for penalty in model.penalties:
            strength = _d(penalty.strength)
            if isinstance(penalty, separator.SoftIntervalPenalty):
                value += strength * max(
                    _d(penalty.lower) - point,
                    Decimal(0),
                ) ** 2
                value += strength * max(
                    point - _d(penalty.upper),
                    Decimal(0),
                ) ** 2
                continue
            if isinstance(penalty, separator.ExponentialBoundaryPenalty):
                tau = _d(penalty.tau)
                value += strength * (
                    ((_d(penalty.lower) + _d(penalty.margin) - point) / tau).exp()
                    + ((point - _d(penalty.upper) + _d(penalty.margin)) / tau).exp()
                )
                continue
            margin = _d(penalty.margin)
            epsilon = _d(penalty.epsilon)
            for distance in (
                point - _d(penalty.lower),
                _d(penalty.upper) - point,
            ):
                if distance >= margin:
                    continue
                at_epsilon = strength * (1 / epsilon - 1 / margin)
                value += (
                    strength * (1 / distance - 1 / margin)
                    if distance >= epsilon
                    else at_epsilon
                    + strength * (epsilon - distance) / (epsilon * epsilon)
                )
        return +value


def _fraction_neighbors(value: Fraction) -> tuple[float, float]:
    nearest = float(value)
    nearest_exact = Fraction.from_float(nearest)
    if nearest_exact == value:
        return (
            math.nextafter(nearest, -math.inf),
            math.nextafter(nearest, math.inf),
        )
    if nearest_exact < value:
        return nearest, math.nextafter(nearest, math.inf)
    return math.nextafter(nearest, -math.inf), nearest


def _finding_d_derivative(value: float, *, scale: float = 1.0) -> Decimal:
    target = -377.50566923018596 * scale
    delta = 3.342765432428535 * scale
    confidence = 113507256.09123965
    v = -219.7275672967312 * scale
    rho = 212751954959.19275
    lower = 150.53416318031995 * scale
    upper = 157.82864753114254 * scale
    margin = 2.794426547999039 * scale
    strength = 240.8079420544141 * scale * scale
    tau = 5.339859001557991 * scale
    with localcontext() as context:
        context.prec = 160
        y = _d(value)
        residual = y - _d(target)
        threshold = _d(delta)
        mismatch = (
            -_d(confidence) * threshold
            if residual < -threshold
            else _d(confidence) * threshold
            if residual > threshold
            else _d(confidence) * residual
        )
        left = ((_d(lower) + _d(margin) - y) / _d(tau)).exp()
        right = ((y - (_d(upper) - _d(margin))) / _d(tau)).exp()
        return +(
            mismatch
            + _d(rho) * (y - _d(v))
            + (_d(strength) / _d(tau)) * (right - left)
        )


def _finding_d_model(*, scale: float = 1.0) -> separator.FitModel:
    return separator.FitModel(
        mismatch=separator.HuberLoss(delta=3.342765432428535 * scale),
        penalties=(
            separator.ExponentialBoundaryPenalty(
                lower=150.53416318031995 * scale,
                upper=157.82864753114254 * scale,
                margin=2.794426547999039 * scale,
                strength=240.8079420544141 * scale * scale,
                tau=5.339859001557991 * scale,
            ),
        ),
    )


def test_finding_d_old_bracket_is_rejected_and_new_ball_contains_oracle() -> None:
    old_lower = 5.459709759468414
    old_upper = 5.4597097594684145
    assert _finding_d_derivative(old_lower) < 0
    assert _finding_d_derivative(old_upper) < 0

    spec = _compile_scalar_prox_spec(_finding_d_model())
    kwargs = dict(
        target=-377.50566923018596,
        confidence=113507256.09123965,
        v=-219.7275672967312,
        rho=212751954959.19275,
    )
    result = _solve_scalar_prox_coordinate(
        spec=spec,
        upper=427.4186885272726,
        **kwargs,
    )
    assert result.certificate_kind == 'adjacent_bracket'
    lower, upper = result.final_bracket
    assert _finding_d_derivative(lower) < 0 < _finding_d_derivative(upper)
    assert result.fallback_count == 0
    for value in result.final_bracket:
        enclosure = scalar_objective._scalar_derivative_terms(
            spec.objective,
            y=value,
            **kwargs,
        ).plus.physical_bounds()
        oracle = _finding_d_derivative(value)
        assert _d(enclosure[0]) <= oracle <= _d(enclosure[1])


def test_finding_e_direct_huber_difference_selects_upper_endpoint() -> None:
    model = separator.FitModel(
        mismatch=separator.HuberLoss(delta=0.1),
        penalties=(separator.SoftIntervalPenalty(-2.0, 2.0, 1e-12),),
    )
    spec = _compile_scalar_prox_spec(model)
    kwargs = dict(target=10.0, confidence=100.0, v=0.0, rho=100000000.0)
    result = _solve_scalar_prox_coordinate(spec=spec, **kwargs)
    lower, upper = result.final_bracket
    exact = (Fraction.from_float(upper) - Fraction.from_float(lower)) * (
        Fraction.from_float(kwargs['rho'])
        * Fraction(1, 2)
        * (
            Fraction.from_float(upper)
            + Fraction.from_float(lower)
            - 2 * Fraction.from_float(kwargs['v'])
        )
        - Fraction.from_float(kwargs['confidence'])
        * Fraction.from_float(0.1)
    )
    assert exact == Fraction(
        -204175,
        44601490397061246283071436545296723011960832,
    )
    assert result.value == 1.0000000000000001e-07
    difference = scalar_objective._scalar_objective_difference(
        spec.objective,
        lower=lower,
        upper=upper,
        **kwargs,
    )
    ordinary_lower, ordinary_upper = difference.physical_bounds()
    assert Fraction.from_float(ordinary_lower) <= exact
    assert exact <= Fraction.from_float(ordinary_upper)
    assert difference.strictly_negative


@pytest.mark.parametrize(
    ('model', 'point'),
    (
        (separator.FitModel(), 0.25),
        (
            separator.FitModel(mismatch=separator.HuberLoss(delta=0.3)),
            -1.0,
        ),
        (
            separator.FitModel(mismatch=separator.HuberLoss(delta=0.3)),
            0.2,
        ),
        (
            separator.FitModel(mismatch=separator.HuberLoss(delta=0.3)),
            1.0,
        ),
        (
            separator.FitModel(
                penalties=(separator.SoftIntervalPenalty(-0.2, 0.7, 0.4),)
            ),
            -0.5,
        ),
        (
            separator.FitModel(
                penalties=(separator.SoftIntervalPenalty(-0.2, 0.7, 0.4),)
            ),
            0.2,
        ),
        (
            separator.FitModel(
                penalties=(separator.SoftIntervalPenalty(-0.2, 0.7, 0.4),)
            ),
            1.0,
        ),
        (
            separator.FitModel(
                penalties=(
                    separator.ExponentialBoundaryPenalty(
                        lower=-1.0,
                        upper=1.2,
                        margin=0.1,
                        strength=0.03,
                        tau=0.2,
                    ),
                )
            ),
            0.1,
        ),
        *tuple(
            (
                separator.FitModel(
                    penalties=(
                        separator.ReciprocalBoundaryPenalty(
                            lower=0.0,
                            upper=1.0,
                            margin=0.2,
                            strength=0.4,
                            epsilon=0.1,
                        ),
                    )
                ),
                point,
            )
            for point in (0.05, 0.15, 0.5, 0.85, 0.95)
        ),
    ),
)
def test_derivative_curvature_and_direct_difference_branch_matrix(
    model: separator.FitModel,
    point: float,
) -> None:
    kwargs = dict(target=0.1, confidence=1.3, v=-0.2, rho=0.8)
    spec = _compile_scalar_prox_spec(model)
    derivative, curvature = _oracle_scalar_terms(model, y=point, **kwargs)
    terms = scalar_objective._scalar_derivative_terms(
        spec.objective,
        y=point,
        **kwargs,
    )
    for enclosure in (terms.minus, terms.plus):
        lower, upper = enclosure.physical_bounds()
        assert Decimal.from_float(lower) <= derivative
        assert derivative <= Decimal.from_float(upper)
    curvature_lower, curvature_upper = terms.curvature.physical_bounds()
    assert Decimal.from_float(curvature_lower) <= curvature
    assert curvature <= Decimal.from_float(curvature_upper)

    adjacent = math.nextafter(point, math.inf)
    exact_difference = (
        _oracle_scalar_objective(model, y=adjacent, **kwargs)
        - _oracle_scalar_objective(model, y=point, **kwargs)
    )
    difference = scalar_objective._scalar_objective_difference(
        spec.objective,
        lower=point,
        upper=adjacent,
        **kwargs,
    )
    difference_lower, difference_upper = difference.physical_bounds()
    assert Decimal.from_float(difference_lower) <= exact_difference
    assert exact_difference <= Decimal.from_float(difference_upper)


def test_complete_branch_boundaries_have_containing_evidence() -> None:
    kwargs = dict(target=0.1, confidence=1.3, v=-0.2, rho=0.8)
    huber = separator.FitModel(mismatch=separator.HuberLoss(delta=0.3))
    soft = separator.FitModel(
        penalties=(separator.SoftIntervalPenalty(-0.2, 0.7, 0.4),)
    )
    reciprocal_penalty = separator.ReciprocalBoundaryPenalty(
        lower=0.0,
        upper=1.0,
        margin=0.2,
        strength=0.4,
        epsilon=0.1,
    )
    reciprocal = separator.FitModel(penalties=(reciprocal_penalty,))
    boundaries = (
        (
            huber,
            Fraction.from_float(kwargs['target'])
            - Fraction.from_float(huber.mismatch.delta),
        ),
        (
            huber,
            Fraction.from_float(kwargs['target'])
            + Fraction.from_float(huber.mismatch.delta),
        ),
        (soft, Fraction.from_float(-0.2)),
        (soft, Fraction.from_float(0.7)),
        (
            reciprocal,
            Fraction.from_float(reciprocal_penalty.lower)
            + Fraction.from_float(reciprocal_penalty.epsilon),
        ),
        (
            reciprocal,
            Fraction.from_float(reciprocal_penalty.lower)
            + Fraction.from_float(reciprocal_penalty.margin),
        ),
        (
            reciprocal,
            Fraction.from_float(reciprocal_penalty.upper)
            - Fraction.from_float(reciprocal_penalty.margin),
        ),
        (
            reciprocal,
            Fraction.from_float(reciprocal_penalty.upper)
            - Fraction.from_float(reciprocal_penalty.epsilon),
        ),
    )
    for model, boundary in boundaries:
        lower, upper = _fraction_neighbors(boundary)
        spec = _compile_scalar_prox_spec(model)
        for point in (lower, upper):
            derivative, curvature = _oracle_scalar_terms(
                model,
                y=point,
                **kwargs,
            )
            terms = scalar_objective._scalar_derivative_terms(
                spec.objective,
                y=point,
                **kwargs,
            )
            for enclosure in (terms.minus, terms.plus):
                enclosure_lower, enclosure_upper = enclosure.physical_bounds()
                assert Decimal.from_float(enclosure_lower) <= derivative
                assert derivative <= Decimal.from_float(enclosure_upper)
            curvature_lower, curvature_upper = (
                terms.curvature.physical_bounds()
            )
            assert Decimal.from_float(curvature_lower) <= curvature
            assert curvature <= Decimal.from_float(curvature_upper)
        exact_difference = (
            _oracle_scalar_objective(model, y=upper, **kwargs)
            - _oracle_scalar_objective(model, y=lower, **kwargs)
        )
        difference = scalar_objective._scalar_objective_difference(
            spec.objective,
            lower=lower,
            upper=upper,
            **kwargs,
        )
        difference_lower, difference_upper = difference.physical_bounds()
        assert Decimal.from_float(difference_lower) <= exact_difference
        assert exact_difference <= Decimal.from_float(difference_upper)


def test_generated_lower_huber_adjacent_family_has_no_false_success() -> None:
    rng = random.Random(20260803)
    for _ in range(30):
        exponent = rng.randint(-40, 20)
        anchor = math.ldexp(rng.uniform(0.6, 1.4), exponent)
        delta = math.ldexp(rng.uniform(0.05, 0.3), rng.randint(-5, 5))
        confidence = math.ldexp(rng.uniform(0.5, 2.0), rng.randint(-5, 15))
        rho = math.ldexp(rng.uniform(0.5, 2.0), rng.randint(5, 40))
        shift = (
            Fraction.from_float(confidence)
            * Fraction.from_float(delta)
            / Fraction.from_float(rho)
        )
        v = float(Fraction.from_float(anchor) - shift)
        root = Fraction.from_float(v) + shift
        target = float(root + 8 * Fraction.from_float(delta) + 1)
        width = 4 * max(1.0, abs(float(root)), abs(target), abs(delta))
        model = separator.FitModel(
            mismatch=separator.HuberLoss(delta=delta),
            penalties=(
                separator.SoftIntervalPenalty(-width, width, 1e-12),
            ),
        )
        result = _solve_scalar_prox_coordinate(
            spec=_compile_scalar_prox_spec(model),
            target=target,
            confidence=confidence,
            v=v,
            rho=rho,
        )
        lower, upper = map(Fraction.from_float, result.final_bracket)
        derivative_lower = (
            -Fraction.from_float(confidence) * Fraction.from_float(delta)
            + Fraction.from_float(rho) * (lower - Fraction.from_float(v))
        )
        derivative_upper = (
            -Fraction.from_float(confidence) * Fraction.from_float(delta)
            + Fraction.from_float(rho) * (upper - Fraction.from_float(v))
        )
        assert derivative_lower < 0 < derivative_upper


def test_finding_d_power_of_two_ci_subset_has_no_false_bracket() -> None:
    for exponent in range(-30, 31, 6):
        scale = math.ldexp(1.0, exponent)
        result = _solve_scalar_prox_coordinate(
            spec=_compile_scalar_prox_spec(_finding_d_model(scale=scale)),
            target=-377.50566923018596 * scale,
            confidence=113507256.09123965,
            v=-219.7275672967312 * scale,
            rho=212751954959.19275,
            upper=427.4186885272726 * scale,
        )
        lower, upper = result.final_bracket
        assert _finding_d_derivative(
            lower,
            scale=scale,
        ) < 0 < _finding_d_derivative(upper, scale=scale)


def test_heterogeneous_batch_matches_scalar_reference_exactly() -> None:
    model = separator.FitModel(
        mismatch=separator.HuberLoss(delta=0.3),
        penalties=(
            separator.SoftIntervalPenalty(-0.2, 0.7, 0.2),
            separator.ExponentialBoundaryPenalty(
                lower=-1.0,
                upper=1.2,
                margin=0.1,
                strength=0.03,
                tau=0.2,
            ),
        ),
    )
    spec = _compile_scalar_prox_spec(model)
    count = 64
    fraction = np.linspace(0.0, 1.0, count)
    target = -0.15 + 0.4 * fraction
    confidence = 0.7 + 0.6 * fraction
    v = np.where(
        fraction < 0.3,
        -0.7 + 0.2 * fraction / 0.3,
        0.1 + 0.4 * (fraction - 0.3) / 0.7,
    )
    lower = -1.1 + 0.2 * fraction
    upper = 0.9 + 0.2 * fraction
    batched = separator_solver._prox_measurement_objective(
        v,
        target,
        confidence,
        model=model,
        rho=1.0,
        y_lo=lower,
        y_hi=upper,
        spec=spec,
    )
    scalar = np.array([
        _solve_scalar_prox_coordinate(
            spec=spec,
            target=float(target[index]),
            confidence=float(confidence[index]),
            v=float(v[index]),
            rho=1.0,
            lower=float(lower[index]),
            upper=float(upper[index]),
        ).value
        for index in range(count)
    ])
    np.testing.assert_array_equal(batched, scalar)


def test_heterogeneous_batch_never_brackets_outside_hard_domain() -> None:
    model = separator.FitModel(
        penalties=(
            separator.ExponentialBoundaryPenalty(
                lower=-1.0,
                upper=1.0,
                strength=0.01,
                tau=1.0,
            ),
        ),
    )
    spec = _compile_scalar_prox_spec(model)
    count = 16
    target = np.linspace(-10.0, -8.0, count)
    confidence = np.linspace(0.7, 1.3, count)
    v = np.linspace(-9.5, -7.5, count)
    lower = np.zeros(count)
    upper = np.ones(count)
    batched = separator_solver._prox_measurement_objective(
        v,
        target,
        confidence,
        model=model,
        rho=1.0,
        y_lo=lower,
        y_hi=upper,
        spec=spec,
    )
    scalar_results = [
        _solve_scalar_prox_coordinate(
            spec=spec,
            target=float(target[index]),
            confidence=float(confidence[index]),
            v=float(v[index]),
            rho=1.0,
            lower=0.0,
            upper=1.0,
        )
        for index in range(count)
    ]
    scalar = np.array([result.value for result in scalar_results])
    initial = _batch_initial(
        model,
        v,
        target,
        confidence,
        lower,
        upper,
    )
    outcome = _solve_scalar_prox_batch_ordinary(
        spec=spec,
        initial=initial,
        target=target,
        confidence=confidence,
        v=v,
        rho=1.0,
        lower=lower,
        upper=upper,
    )
    np.testing.assert_array_equal(batched, scalar)
    np.testing.assert_array_equal(batched, lower)
    assert np.all(outcome.eligible)
    assert not np.any(outcome.certified)
    assert all(result is None for result in outcome.results)
    assert all(
        result.certificate_kind == 'point_kkt'
        for result in scalar_results
    )


def _heterogeneous_batch_case(
    count: int = 10,
) -> tuple[
    separator.FitModel,
    object,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    model = separator.FitModel(
        mismatch=separator.HuberLoss(delta=0.3),
        penalties=(
            separator.SoftIntervalPenalty(-0.2, 0.7, 0.2),
            separator.ExponentialBoundaryPenalty(
                lower=-1.0,
                upper=1.2,
                margin=0.1,
                strength=0.03,
                tau=0.2,
            ),
        ),
    )
    fraction = np.linspace(0.0, 1.0, count)
    target = -0.15 + 0.4 * fraction
    confidence = 0.7 + 0.6 * fraction
    v = np.where(
        fraction < 0.3,
        -0.65 + 0.15 * fraction / 0.3,
        0.1 + 0.4 * (fraction - 0.3) / 0.7,
    )
    lower = -1.1 + 0.2 * fraction
    upper = 0.9 + 0.2 * fraction
    return (
        model,
        _compile_scalar_prox_spec(model),
        target,
        confidence,
        v,
        lower,
        upper,
    )


def _batch_initial(
    model: separator.FitModel,
    v: np.ndarray,
    target: np.ndarray,
    confidence: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
) -> np.ndarray:
    initial = separator_solver._prox_measurement_mismatch_only(
        v,
        target,
        confidence,
        model.mismatch,
        1.0,
    )
    return np.minimum(np.maximum(initial, lower), upper)


def _scalar_cache_case() -> tuple[
    separator.FitModel,
    object,
    dict[str, np.ndarray],
]:
    model = separator.FitModel(
        penalties=(
            separator.ExponentialBoundaryPenalty(
                lower=-1.0,
                upper=1.0,
                margin=0.1,
                strength=0.03,
                tau=0.2,
            ),
        ),
    )
    inputs = {
        'target': np.full(2, 0.15),
        'confidence': np.full(2, 1.1),
        'v': np.full(2, -0.25),
        'lower': np.full(2, -0.8),
        'upper': np.full(2, 0.95),
    }
    return model, _compile_scalar_prox_spec(model), inputs


def _scalar_cache_references(
    solve,
    *,
    spec: object,
    inputs: dict[str, np.ndarray],
    rho: float,
) -> np.ndarray:
    return np.array([
        solve(
            spec=spec,
            target=float(inputs['target'][index]),
            confidence=float(inputs['confidence'][index]),
            v=float(inputs['v'][index]),
            rho=rho,
            lower=float(inputs['lower'][index]),
            upper=float(inputs['upper'][index]),
        ).value
        for index in range(inputs['target'].size)
    ])


def _run_scalar_cache_case(
    *,
    model: separator.FitModel,
    spec: object,
    inputs: dict[str, np.ndarray],
    rho: float,
) -> np.ndarray:
    return separator_solver._prox_measurement_objective(
        inputs['v'],
        inputs['target'],
        inputs['confidence'],
        model=model,
        rho=rho,
        y_lo=inputs['lower'],
        y_hi=inputs['upper'],
        spec=spec,
    )


def test_scalar_coordinate_cache_reuses_equal_complete_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model, spec, inputs = _scalar_cache_case()
    original = separator_solver._solve_scalar_prox_coordinate
    dispatches = []

    def counted(*args, **kwargs):
        dispatches.append(kwargs)
        return original(*args, **kwargs)

    monkeypatch.setattr(
        separator_solver,
        '_solve_scalar_prox_coordinate',
        counted,
    )
    actual = _run_scalar_cache_case(
        model=model,
        spec=spec,
        inputs=inputs,
        rho=1.0,
    )
    expected = _scalar_cache_references(
        original,
        spec=spec,
        inputs=inputs,
        rho=1.0,
    )
    assert len(dispatches) == 1
    np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize(
    ('field', 'replacement'),
    (
        ('target', 0.45),
        ('confidence', 1.6),
        ('v', 0.2),
        ('lower', 0.3),
        ('upper', -0.3),
    ),
)
def test_scalar_coordinate_cache_key_includes_each_field(
    monkeypatch: pytest.MonkeyPatch,
    field: str,
    replacement: float,
) -> None:
    model, spec, inputs = _scalar_cache_case()
    inputs[field][1] = replacement
    original = separator_solver._solve_scalar_prox_coordinate
    dispatches = []

    def counted(*args, **kwargs):
        dispatches.append(kwargs)
        return original(*args, **kwargs)

    monkeypatch.setattr(
        separator_solver,
        '_solve_scalar_prox_coordinate',
        counted,
    )
    actual = _run_scalar_cache_case(
        model=model,
        spec=spec,
        inputs=inputs,
        rho=1.0,
    )
    expected = _scalar_cache_references(
        original,
        spec=spec,
        inputs=inputs,
        rho=1.0,
    )
    assert len(dispatches) == 2
    np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize(
    'field',
    ('target', 'confidence', 'v', 'lower', 'upper'),
)
def test_scalar_coordinate_cache_uses_numeric_signed_zero(
    monkeypatch: pytest.MonkeyPatch,
    field: str,
) -> None:
    model, spec, inputs = _scalar_cache_case()
    inputs[field] = np.array([-0.0, 0.0])
    assert np.signbit(inputs[field][0])
    assert not np.signbit(inputs[field][1])
    original = separator_solver._solve_scalar_prox_coordinate
    dispatches = []

    def counted(*args, **kwargs):
        dispatches.append(kwargs)
        return original(*args, **kwargs)

    monkeypatch.setattr(
        separator_solver,
        '_solve_scalar_prox_coordinate',
        counted,
    )
    actual = _run_scalar_cache_case(
        model=model,
        spec=spec,
        inputs=inputs,
        rho=1.0,
    )
    expected = _scalar_cache_references(
        original,
        spec=spec,
        inputs=inputs,
        rho=1.0,
    )
    assert len(dispatches) == 1
    np.testing.assert_array_equal(actual, expected)


def test_scalar_coordinate_cache_is_local_to_each_call_and_spec(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first_model, first_spec, first_inputs = _scalar_cache_case()
    second_model = separator.FitModel(
        penalties=(
            separator.ExponentialBoundaryPenalty(
                lower=-0.4,
                upper=0.5,
                margin=0.05,
                strength=0.2,
                tau=0.1,
            ),
        ),
    )
    second_spec = _compile_scalar_prox_spec(second_model)
    second_inputs = {
        name: values[:1].copy()
        for name, values in first_inputs.items()
    }
    first_inputs = {
        name: values[:1].copy()
        for name, values in first_inputs.items()
    }
    original = separator_solver._solve_scalar_prox_coordinate
    dispatches = []

    def counted(*args, **kwargs):
        dispatches.append(kwargs)
        return original(*args, **kwargs)

    monkeypatch.setattr(
        separator_solver,
        '_solve_scalar_prox_coordinate',
        counted,
    )
    first = _run_scalar_cache_case(
        model=first_model,
        spec=first_spec,
        inputs=first_inputs,
        rho=1.0,
    )
    second = _run_scalar_cache_case(
        model=second_model,
        spec=second_spec,
        inputs=second_inputs,
        rho=2.0,
    )
    first_reference = _scalar_cache_references(
        original,
        spec=first_spec,
        inputs=first_inputs,
        rho=1.0,
    )
    second_reference = _scalar_cache_references(
        original,
        spec=second_spec,
        inputs=second_inputs,
        rho=2.0,
    )
    assert len(dispatches) == 2
    np.testing.assert_array_equal(first, first_reference)
    np.testing.assert_array_equal(second, second_reference)
    assert first[0] != second[0]


def test_strict_numpy_policy_is_invariant_across_batch_threshold() -> None:
    model = separator.FitModel(
        penalties=(separator.ReciprocalBoundaryPenalty(),)
    )
    spec = _compile_scalar_prox_spec(model)
    target = np.linspace(-2.0, -1.0, 8)
    confidence = np.linspace(0.8, 1.2, 8)
    v = np.linspace(-2.2, -1.2, 8)
    expected = np.array([
        _solve_scalar_prox_coordinate(
            spec=spec,
            target=float(target[index]),
            confidence=float(confidence[index]),
            v=float(v[index]),
            rho=1.0,
        ).value
        for index in range(8)
    ])
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        for count in (7, 8):
            with np.errstate(all='raise'):
                actual = separator_solver._prox_measurement_objective(
                    v[:count],
                    target[:count],
                    confidence[:count],
                    model=model,
                    rho=1.0,
                    y_lo=None,
                    y_hi=None,
                    spec=spec,
                )
            np.testing.assert_array_equal(actual, expected[:count])
    assert not [item for item in caught if issubclass(item.category, RuntimeWarning)]


def test_batch_results_retain_scalar_equivalent_certificate_evidence() -> None:
    model, spec, target, confidence, v, lower, upper = (
        _heterogeneous_batch_case()
    )
    initial = _batch_initial(
        model,
        v,
        target,
        confidence,
        lower,
        upper,
    )
    with np.errstate(all='raise'):
        outcome = _solve_scalar_prox_batch_ordinary(
            spec=spec,
            initial=initial,
            target=target,
            confidence=confidence,
            v=v,
            rho=1.0,
            lower=lower,
            upper=upper,
        )
    assert np.all(outcome.eligible)
    assert np.all(outcome.certified)
    assert not outcome.arithmetic_exception
    for index, batch in enumerate(outcome.results):
        assert batch is not None
        scalar = _solve_scalar_prox_coordinate(
            spec=spec,
            target=float(target[index]),
            confidence=float(confidence[index]),
            v=float(v[index]),
            rho=1.0,
            lower=float(lower[index]),
            upper=float(upper[index]),
        )
        assert batch.value == scalar.value
        assert batch.certificate_kind == scalar.certificate_kind
        assert batch.final_bracket == scalar.final_bracket
        assert batch.localization_bound == scalar.localization_bound
        assert batch.execution_path == 'batch_ordinary'
        assert batch.fallback_count == 0
        assert scalar.fallback_count in (0, 1)
        assert batch.adjacent_bracket_evidence is not None
        left, right = batch.adjacent_bracket_evidence
        assert left.plus_resolved_sign == -1
        assert left.plus_enclosure[1] < 0.0
        assert right.minus_resolved_sign == 1
        assert right.minus_enclosure[0] > 0.0
        terminal = batch.terminal_difference_evidence
        assert terminal is not None
        assert terminal.fallback_resolved_sign is None
        assert terminal.ordinary_resolved_sign in (-1, 1)
        assert terminal.selected_endpoint == batch.value
        expected = (
            terminal.lower_endpoint
            if terminal.ordinary_resolved_sign > 0
            else terminal.upper_endpoint
        )
        assert batch.value == expected
        if scalar.fallback_count:
            scalar_terminal = scalar.terminal_difference_evidence
            assert scalar_terminal is not None
            assert scalar_terminal.ordinary_resolved_sign is None
            assert scalar_terminal.fallback_resolved_sign in (-1, 1)
            assert scalar_terminal.fallback_enclosure is not None
            fallback_lower, fallback_upper = (
                scalar_terminal.fallback_enclosure
            )
            assert fallback_upper < 0 or fallback_lower > 0


def test_batch_programming_error_is_not_converted_to_scalar_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model, spec, target, confidence, v, lower, upper = (
        _heterogeneous_batch_case()
    )
    initial = _batch_initial(
        model,
        v,
        target,
        confidence,
        lower,
        upper,
    )

    def broken_objective(*args: object, **kwargs: object) -> float:
        raise TypeError('injected programming error')

    monkeypatch.setattr(scalar_prox, '_objective_at', broken_objective)
    with pytest.raises(TypeError, match='injected programming error'):
        _solve_scalar_prox_batch_ordinary(
            spec=spec,
            initial=initial,
            target=target,
            confidence=confidence,
            v=v,
            rho=1.0,
            lower=lower,
            upper=upper,
        )


def test_exceptional_batch_lane_routes_alone_without_warning() -> None:
    model, spec, target, confidence, v, lower, upper = (
        _heterogeneous_batch_case()
    )
    target[-1] = 100.0
    v[-1] = 100.0
    lower[-1] = -np.inf
    upper[-1] = np.inf
    initial = _batch_initial(
        model,
        v,
        target,
        confidence,
        lower,
        upper,
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        with np.errstate(all='raise'):
            outcome = _solve_scalar_prox_batch_ordinary(
                spec=spec,
                initial=initial,
                target=target,
                confidence=confidence,
                v=v,
                rho=1.0,
                lower=lower,
                upper=upper,
            )
            actual = separator_solver._prox_measurement_objective(
                v,
                target,
                confidence,
                model=model,
                rho=1.0,
                y_lo=lower,
                y_hi=upper,
                spec=spec,
            )
    assert np.all(outcome.certified[:-1])
    assert not outcome.certified[-1]
    assert outcome.results[-1] is None
    scalar = np.array([
        _solve_scalar_prox_coordinate(
            spec=spec,
            target=float(target[index]),
            confidence=float(confidence[index]),
            v=float(v[index]),
            rho=1.0,
            lower=float(lower[index]),
            upper=float(upper[index]),
        ).value
        for index in range(target.size)
    ])
    np.testing.assert_array_equal(actual, scalar)
    assert not [item for item in caught if issubclass(item.category, RuntimeWarning)]


def test_exceptional_lane_failure_keeps_original_and_local_row_identity(
    monkeypatch,
) -> None:
    model, spec, target, confidence, v, lower, upper = (
        _heterogeneous_batch_case()
    )
    target[-1] = 100.0
    v[-1] = 100.0
    lower[-1] = -np.inf
    upper[-1] = np.inf
    original = separator_solver._solve_scalar_prox_coordinate
    scalar_targets: list[float] = []

    def fail_exceptional(*args, **kwargs):
        scalar_targets.append(float(kwargs['target']))
        if kwargs['target'] == 100.0:
            raise _ScalarProxError(_ScalarProxFailure(
                reason='review_exceptional_lane',
                scalar_iterations=3,
                expansion_count=2,
                last_candidate=100.0,
                last_finite_bracket=None,
                fallback_count=1,
            ))
        return original(*args, **kwargs)

    monkeypatch.setattr(
        separator_solver,
        '_solve_scalar_prox_coordinate',
        fail_exceptional,
    )
    with pytest.raises(separator_solver._NumericalFailure) as error:
        separator_solver._prox_measurement_objective(
            v,
            target,
            confidence,
            model=model,
            rho=1.0,
            y_lo=lower,
            y_hi=upper,
            spec=spec,
            row_indices=np.arange(700, 710),
        )
    message = str(error.value)
    assert 'observation row 709' in message
    assert 'component-local row 9' in message
    assert 'review_exceptional_lane' in message
    assert scalar_targets == [100.0]


def test_signed_zero_batch_and_scalar_failure_classification_agree() -> None:
    model = separator.FitModel(
        penalties=(
            separator.ExponentialBoundaryPenalty(
                lower=-1.0,
                upper=1.0,
                margin=0.0,
                strength=0.01,
                tau=1.0,
            ),
        ),
    )
    spec = _compile_scalar_prox_spec(model)
    target = np.ldexp(np.arange(1.0, 9.0), -10)
    confidence = np.ones(8)
    v = -target
    lower = np.full(8, -np.inf)
    upper = np.full(8, np.inf)
    initial = _batch_initial(
        model,
        v,
        target,
        confidence,
        lower,
        upper,
    )
    np.testing.assert_array_equal(initial, np.zeros(8))
    outcome = _solve_scalar_prox_batch_ordinary(
        spec=spec,
        initial=initial,
        target=target,
        confidence=confidence,
        v=v,
        rho=1.0,
        lower=lower,
        upper=upper,
    )
    assert not np.any(outcome.certified)
    assert all(result is None for result in outcome.results)
    for index in range(8):
        with pytest.raises(_ScalarProxError) as error:
            _solve_scalar_prox_coordinate(
                spec=spec,
                target=float(target[index]),
                confidence=1.0,
                v=float(v[index]),
                rho=1.0,
            )
        assert error.value.failure.reason == (
            'high_precision_derivative_sign_unresolved'
        )
