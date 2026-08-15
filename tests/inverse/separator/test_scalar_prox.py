"""Independent certificates for the private scalar proximal solver."""

from __future__ import annotations

import json
import math
from decimal import Decimal, localcontext
from fractions import Fraction

import numpy as np
import pytest

from pyvoro2 import Box
import pyvoro2.inverse.separator as separator
import pyvoro2.inverse.separator._objective as scalar_objective
import pyvoro2.inverse.separator._scalar_prox as scalar_prox
import pyvoro2.inverse.separator.problem as separator_problem
import pyvoro2.inverse.separator.solver as separator_solver
from pyvoro2.inverse.separator._scalar_prox import (
    _compile_scalar_prox_spec,
    _ScalarProxError,
    _solve_scalar_prox_coordinate,
)


def _exact_decimal(value: float | Fraction) -> Decimal:
    fraction = value if isinstance(value, Fraction) else Fraction.from_float(value)
    return Decimal(fraction.numerator) / Decimal(fraction.denominator)


def _d(value: float) -> Decimal:
    return Decimal.from_float(float(value))


def _independent_q(
    distance: Decimal,
    penalty: separator.ReciprocalBoundaryPenalty,
) -> Decimal:
    strength = _d(penalty.strength)
    margin = _d(penalty.margin)
    epsilon = _d(penalty.epsilon)
    if distance >= margin:
        return Decimal(0)
    if distance > epsilon:
        return strength * (1 / distance - 1 / margin)
    return strength * (
        1 / epsilon
        - 1 / margin
        - (distance - epsilon) / (epsilon * epsilon)
    )


def _independent_objective(
    y: float | Decimal,
    *,
    model: separator.FitModel,
    target: float,
    confidence: float,
    v: float,
    rho: float,
) -> Decimal:
    """High-precision oracle written without production objective helpers."""

    with localcontext() as context:
        context.prec = 100
        value = y if isinstance(y, Decimal) else _d(y)
        residual = value - _d(target)
        confidence_value = _d(confidence)
        if isinstance(model.mismatch, separator.SquaredLoss):
            total = confidence_value * residual * residual / 2
        else:
            delta = _d(model.mismatch.delta)
            if abs(residual) <= delta:
                mismatch = residual * residual / 2
            else:
                mismatch = delta * (abs(residual) - delta / 2)
            total = confidence_value * mismatch
        displacement = value - _d(v)
        total += _d(rho) * displacement * displacement / 2
        for penalty in model.penalties:
            if float(penalty.strength) == 0.0:
                continue
            strength = _d(penalty.strength)
            lower = _d(penalty.lower)
            upper = _d(penalty.upper)
            if isinstance(penalty, separator.SoftIntervalPenalty):
                total += strength * (
                    max(lower - value, Decimal(0)) ** 2
                    + max(value - upper, Decimal(0)) ** 2
                )
            elif isinstance(
                penalty,
                separator.ExponentialBoundaryPenalty,
            ):
                margin = _d(penalty.margin)
                tau = _d(penalty.tau)
                total += strength * (
                    ((lower + margin - value) / tau).exp()
                    + ((value - (upper - margin)) / tau).exp()
                )
            else:
                total += _independent_q(value - lower, penalty)
                total += _independent_q(upper - value, penalty)
        return total


def _independent_derivative(
    y: Decimal,
    *,
    model: separator.FitModel,
    target: float,
    confidence: float,
    v: float,
    rho: float,
) -> Decimal:
    """Smooth-piece derivative oracle used only for deterministic bisection."""

    with localcontext() as context:
        context.prec = 100
        residual = y - _d(target)
        confidence_value = _d(confidence)
        if isinstance(model.mismatch, separator.SquaredLoss):
            result = confidence_value * residual
        else:
            delta = _d(model.mismatch.delta)
            result = confidence_value * max(-delta, min(residual, delta))
        result += _d(rho) * (y - _d(v))
        for penalty in model.penalties:
            if float(penalty.strength) == 0.0:
                continue
            strength = _d(penalty.strength)
            lower = _d(penalty.lower)
            upper = _d(penalty.upper)
            if isinstance(penalty, separator.SoftIntervalPenalty):
                if y < lower:
                    result += 2 * strength * (y - lower)
                elif y > upper:
                    result += 2 * strength * (y - upper)
            elif isinstance(
                penalty,
                separator.ExponentialBoundaryPenalty,
            ):
                margin = _d(penalty.margin)
                tau = _d(penalty.tau)
                result += strength / tau * (
                    -((lower + margin - y) / tau).exp()
                    + ((y - (upper - margin)) / tau).exp()
                )
            else:
                margin = _d(penalty.margin)
                epsilon = _d(penalty.epsilon)
                lower_distance = y - lower
                if lower_distance < margin:
                    denominator = max(lower_distance, epsilon)
                    result -= strength / (denominator * denominator)
                upper_distance = upper - y
                if upper_distance < margin:
                    denominator = max(upper_distance, epsilon)
                    result += strength / (denominator * denominator)
        return result


def _independent_minimum(
    *,
    model: separator.FitModel,
    target: float,
    confidence: float,
    v: float,
    rho: float,
    lower: float = -math.inf,
    upper: float = math.inf,
) -> Decimal:
    with localcontext() as context:
        context.prec = 100
        lo = _d(lower) if math.isfinite(lower) else Decimal(-1)
        hi = _d(upper) if math.isfinite(upper) else Decimal(1)
        while _independent_derivative(
            lo,
            model=model,
            target=target,
            confidence=confidence,
            v=v,
            rho=rho,
        ) > 0 and not math.isfinite(lower):
            lo *= 2
        while _independent_derivative(
            hi,
            model=model,
            target=target,
            confidence=confidence,
            v=v,
            rho=rho,
        ) < 0 and not math.isfinite(upper):
            hi *= 2
        if _independent_derivative(
            lo,
            model=model,
            target=target,
            confidence=confidence,
            v=v,
            rho=rho,
        ) >= 0:
            return lo
        if _independent_derivative(
            hi,
            model=model,
            target=target,
            confidence=confidence,
            v=v,
            rho=rho,
        ) <= 0:
            return hi
        for _ in range(420):
            middle = (lo + hi) / 2
            derivative = _independent_derivative(
                middle,
                model=model,
                target=target,
                confidence=confidence,
                v=v,
                rho=rho,
            )
            if derivative < 0:
                lo = middle
            else:
                hi = middle
        return (lo + hi) / 2


def _solve(model: separator.FitModel, **kwargs) -> scalar_prox._ScalarProxResult:
    return _solve_scalar_prox_coordinate(
        spec=_compile_scalar_prox_spec(model),
        **kwargs,
    )


def _assert_local_minimum(
    result: scalar_prox._ScalarProxResult,
    *,
    model: separator.FitModel,
    target: float,
    confidence: float,
    v: float,
    rho: float,
    lower: float = -math.inf,
    upper: float = math.inf,
) -> None:
    value = result.value
    objective = _independent_objective(
        value,
        model=model,
        target=target,
        confidence=confidence,
        v=v,
        rho=rho,
    )
    for direction in (-math.inf, math.inf):
        neighbor = math.nextafter(value, direction)
        if lower <= neighbor <= upper and math.isfinite(neighbor):
            neighbor_objective = _independent_objective(
                neighbor,
                model=model,
                target=target,
                confidence=confidence,
                v=v,
                rho=rho,
            )
            allowance = Decimal(128) * _d(np.finfo(np.float64).eps) * max(
                abs(objective),
                abs(neighbor_objective),
            )
            assert objective <= neighbor_objective + allowance
    assert math.isfinite(result.objective)
    assert result.certificate_kind in {
        'equality',
        'point_kkt',
        'adjacent_bracket',
    }


def test_complete_expression_objective_authority_reaches_reports_and_json() -> None:
    lower = 1.0
    upper = math.nextafter(lower, math.inf)
    margin = 2.0**-53
    tau = 1.0e-17
    rho = 1.0e30
    penalty = separator.ExponentialBoundaryPenalty(
        lower=lower,
        upper=upper,
        margin=margin,
        strength=1.0,
        tau=tau,
    )
    model = separator.FitModel(
        feasible=separator.Interval(lower, upper),
        penalties=(penalty,),
    )
    spec = _compile_scalar_prox_spec(model)
    result = _solve_scalar_prox_coordinate(
        spec=spec,
        target=0.0,
        confidence=0.0,
        v=upper,
        rho=rho,
        lower=lower,
        upper=upper,
    )

    def oracle(value: float, *, include_proximal: bool) -> Decimal:
        with localcontext() as context:
            context.prec = 100
            y = _exact_decimal(value)
            lo = _exact_decimal(lower)
            hi = _exact_decimal(upper)
            m = _exact_decimal(margin)
            scale = _exact_decimal(tau)
            total = ((lo + m - y) / scale).exp()
            total += ((y - (hi - m)) / scale).exp()
            if include_proximal:
                total += _exact_decimal(rho) * (
                    y - _exact_decimal(upper)
                ) ** 2 / 2
            return total

    expected = min((lower, upper), key=lambda value: oracle(
        value,
        include_proximal=True,
    ))
    assert expected == upper
    assert result.value == expected
    assert result.certificate_kind == 'adjacent_bracket'
    assert result.final_bracket == (lower, upper)
    direct = scalar_objective._scalar_objective_difference(
        spec.objective,
        lower=lower,
        upper=upper,
        target=0.0,
        confidence=0.0,
        v=upper,
        rho=rho,
    )
    assert direct.strictly_negative

    public_penalty = float(
        scalar_objective._penalty_value(np.array([expected]), penalty)[0]
    )
    private_total = scalar_objective._scalar_objective_value(
        spec.objective,
        y=expected,
        target=0.0,
        confidence=0.0,
        v=upper,
        rho=rho,
    )
    assert private_total == result.objective
    assert public_penalty == pytest.approx(float(oracle(
        expected,
        include_proximal=False,
    )), rel=2.0e-15)

    points = np.array([[0.0, 0.0], [1.0, 0.0]])
    observations = separator.resolve_separator_observations(
        points,
        [(0, 1, expected)],
        confidence=np.array([0.0]),
    )
    problem = separator.build_power_fit_problem(observations, model=model)
    difference = 2.0 * (expected - 0.5)
    weights = np.array([0.5 * difference, -0.5 * difference])
    fit = separator.build_power_fit_result(
        problem,
        weights,
        solver='external',
        status='external_candidate',
        converged=False,
        canonicalize_gauge=False,
    )
    assert fit.predicted is not None
    assert fit.predicted[0] == expected
    assert fit.objective_breakdown is not None
    assert fit.objective_breakdown.penalties_total == public_penalty
    assert problem.evaluate_objective(weights) == public_penalty
    report = separator.build_fit_report(fit, observations)
    payload = json.loads(separator.dumps_report_json(report))
    assert report['objective_breakdown']['penalties_total'] == public_penalty
    assert payload['objective_breakdown']['penalties_total'] == public_penalty


def test_affine_midpoint_exponential_value_is_exact_across_public_paths() -> None:
    lower = 0.5
    upper = math.nextafter(lower, math.inf)
    margin = 2.0**-54
    penalty = separator.ExponentialBoundaryPenalty(
        lower=lower,
        upper=upper,
        margin=margin,
        strength=1.0,
        tau=1.0e-18,
    )
    points = np.array([[0.0, 0.0], [1.0, 0.0]])
    observations = separator.resolve_separator_observations(
        points,
        [(0, 1, lower)],
    )
    problem = separator.build_power_fit_problem(
        observations,
        model=separator.FitModel(penalties=(penalty,)),
    )
    difference = 2.0**-53
    weights = np.array([difference / 2, -difference / 2])
    result = separator.build_power_fit_result(
        problem,
        weights,
        solver='external',
        status='external_candidate',
        converged=False,
        canonicalize_gauge=False,
    )
    report = separator.build_fit_report(result, observations)
    payload = json.loads(separator.dumps_report_json(report))

    # Independent exact-expression oracle: both exponent numerators vanish at
    # 0.5 + 2**-54, so the two exponential terms are exactly one.
    exact_measurement = Fraction(1, 2) + Fraction(1, 2**54)
    assert Fraction.from_float(lower) + Fraction.from_float(margin) == (
        exact_measurement
    )
    assert Fraction.from_float(upper) - Fraction.from_float(margin) == (
        exact_measurement
    )
    assert problem.evaluate_objective(weights) == 2.0
    assert result.objective_breakdown is not None
    assert result.objective_breakdown.penalties_total == 2.0
    assert report['objective_breakdown']['penalties_total'] == 2.0
    assert payload['objective_breakdown']['penalties_total'] == 2.0


def test_exponential_constructor_uses_exact_complete_overlap() -> None:
    lower = 1.0
    upper = math.nextafter(math.nextafter(lower, math.inf), math.inf)
    margin = math.nextafter((upper - lower) / 2.0, math.inf)
    assert lower + margin == upper - margin
    assert (
        Fraction.from_float(lower) + Fraction.from_float(margin)
        > Fraction.from_float(upper) - Fraction.from_float(margin)
    )
    with pytest.raises(ValueError, match='margin is too large'):
        separator.ExponentialBoundaryPenalty(
            lower=lower,
            upper=upper,
            margin=margin,
        )


def test_cancellation_has_sign_bracket_not_tolerance_point_success() -> None:
    strength_1 = 1.0e12
    strength_2 = math.nextafter(strength_1, math.inf)
    model = separator.FitModel(
        penalties=(
            separator.ReciprocalBoundaryPenalty(
                lower=0.0,
                upper=1.0,
                margin=0.2,
                epsilon=0.1,
                strength=strength_1,
            ),
            separator.ReciprocalBoundaryPenalty(
                lower=-1.0,
                upper=0.0,
                margin=0.2,
                epsilon=0.1,
                strength=strength_2,
            ),
        )
    )
    spec = _compile_scalar_prox_spec(model)
    bad = scalar_prox._evaluate_derivative(
        spec,
        y=-0.1,
        target=0.0,
        confidence=0.0,
        v=0.0,
        rho=1.0,
    )
    assert bad.certified_negative
    assert not scalar_prox._point_condition(
        bad,
        at_lower=False,
        at_upper=False,
    )

    result = _solve_scalar_prox_coordinate(
        spec=spec,
        target=0.0,
        confidence=0.0,
        v=0.0,
        rho=1.0,
    )
    epsilon = Fraction.from_float(0.1)
    minimum = -(
        Fraction.from_float(strength_2) - Fraction.from_float(strength_1)
    ) / (epsilon * epsilon)
    bracket = tuple(Fraction.from_float(value) for value in result.final_bracket)
    assert result.certificate_kind == 'adjacent_bracket'
    assert bracket[0] < minimum < bracket[1]
    assert result.value == result.final_bracket[1]
    assert result.point_derivative_evidence is None
    assert result.adjacent_bracket_evidence is not None
    left_evidence, right_evidence = result.adjacent_bracket_evidence
    assert left_evidence.y == result.final_bracket[0]
    assert right_evidence.y == result.final_bracket[1]
    assert (
        left_evidence.plus_enclosure[1] < 0
        or left_evidence.plus_enclosure[0] <= 0 <= (
            left_evidence.plus_enclosure[1]
        )
    )
    assert (
        right_evidence.minus_enclosure[0] > 0
        or right_evidence.minus_enclosure[0] <= 0 <= (
            right_evidence.minus_enclosure[1]
        )
    )
    assert left_evidence.plus_resolved_sign == -1
    assert right_evidence.minus_resolved_sign == 1


def test_point_certificate_is_reconstructable_from_private_metadata() -> None:
    result = _solve(
        separator.FitModel(
            penalties=(separator.ReciprocalBoundaryPenalty(),)
        ),
        target=0.0,
        confidence=1.0,
        v=0.0,
        rho=1.0,
        lower=0.0,
        upper=1.0,
    )
    assert result.certificate_kind == 'point_kkt'
    assert result.adjacent_bracket_evidence is None
    evidence = result.point_derivative_evidence
    assert evidence is not None
    minus_ok = (
        evidence.minus_resolved_sign <= 0
        if evidence.minus_resolved_sign is not None
        else evidence.minus_enclosure[1] <= 0
    )
    plus_ok = (
        evidence.plus_resolved_sign >= 0
        if evidence.plus_resolved_sign is not None
        else evidence.plus_enclosure[0] >= 0
    )
    assert minus_ok and plus_ok


def test_irrelevant_positive_penalties_and_order_preserve_terminal_result() -> None:
    strength_1 = 1.0e12
    strength_2 = math.nextafter(strength_1, math.inf)
    active = (
        separator.ReciprocalBoundaryPenalty(0.0, 1.0, 0.2, strength_1, 0.1),
        separator.ReciprocalBoundaryPenalty(-1.0, 0.0, 0.2, strength_2, 0.1),
    )
    irrelevant = (
        separator.SoftIntervalPenalty(-0.1, 0.5, 3.0),
        separator.SoftIntervalPenalty(
            math.nextafter(-0.1, -math.inf),
            0.6,
            7.0,
        ),
    )
    orders = (
        active,
        active + irrelevant,
        tuple(reversed(active + irrelevant)),
        (irrelevant[1], active[0], irrelevant[0], active[1]),
    )
    results = [
        _solve(
            separator.FitModel(penalties=penalties),
            target=0.0,
            confidence=0.0,
            v=0.0,
            rho=1.0,
        )
        for penalties in orders
    ]
    assert {result.value for result in results} == {results[0].value}
    assert {result.final_bracket for result in results} == {
        results[0].final_bracket
    }


def test_direct_difference_ignores_huge_common_tangent_constant() -> None:
    tiny = math.ulp(0.0)
    penalty = separator.ReciprocalBoundaryPenalty(
        lower=0.0,
        upper=1.0,
        margin=0.2,
        epsilon=0.1,
        strength=1.0e300,
    )
    spec = _compile_scalar_prox_spec(
        separator.FitModel(penalties=(penalty,))
    )
    difference = scalar_objective._scalar_objective_difference(
        spec.objective,
        lower=-tiny,
        upper=0.0,
        target=0.0,
        confidence=0.0,
        v=0.0,
        rho=1.0,
    )
    with localcontext() as context:
        context.prec = 100
        epsilon = _exact_decimal(penalty.epsilon)
        strength = _exact_decimal(penalty.strength)
        expected = -strength * _exact_decimal(tiny) / (epsilon * epsilon)
    assert expected < 0
    assert difference.strictly_negative
    lower_bound, upper_bound = difference.physical_bounds()
    assert lower_bound <= float(expected) <= upper_bound


def test_minimum_subnormal_soft_square_preserves_finite_source_value() -> None:
    maximum = np.finfo(np.float64).max
    tiny = math.ulp(0.0)
    model = separator.FitModel(
        penalties=(separator.SoftIntervalPenalty(-1.0, 1.0, tiny),)
    )
    spec = _compile_scalar_prox_spec(model)
    value = scalar_objective._scalar_objective_value(
        spec.objective,
        y=maximum,
        target=0.0,
        confidence=0.0,
        v=maximum,
        rho=1.0,
    )
    expected = float(
        Fraction.from_float(tiny)
        * (Fraction.from_float(maximum) - 1) ** 2
    )
    public = scalar_objective._penalty_value(
        np.array([maximum]),
        model.penalties[0],
    )[0]
    assert math.isfinite(value)
    assert value == public == expected == 1.5966722476277755e293


def test_range_safe_adjacent_direct_soft_difference() -> None:
    maximum = np.finfo(np.float64).max
    lower = math.nextafter(maximum, -math.inf)
    tiny = math.ulp(0.0)
    model = separator.FitModel(
        penalties=(separator.SoftIntervalPenalty(-1.0, 1.0, tiny),)
    )
    spec = _compile_scalar_prox_spec(model)
    enclosure = scalar_objective._scalar_objective_difference(
        spec.objective,
        lower=lower,
        upper=maximum,
        target=0.0,
        confidence=0.0,
        v=maximum,
        rho=0.0,
    )
    exact = Fraction.from_float(tiny) * (
        (Fraction.from_float(maximum) - 1) ** 2
        - (Fraction.from_float(lower) - 1) ** 2
    )
    lower_bound, upper_bound = enclosure.physical_bounds()
    assert math.isfinite(float(exact))
    assert lower_bound <= float(exact) <= upper_bound
    assert enclosure.strictly_positive


def test_maximum_finite_proximal_row_returns_without_arithmetic_exception() -> None:
    maximum = np.finfo(np.float64).max
    model = separator.FitModel(
        penalties=(
            separator.SoftIntervalPenalty(
                -1.0,
                1.0,
                math.ulp(0.0),
            ),
        )
    )
    actual = separator_solver._prox_measurement_objective(
        np.array([maximum]),
        np.array([0.0]),
        np.array([0.0]),
        model=model,
        rho=1.0,
        y_lo=None,
        y_hi=None,
    )
    assert actual[0] == maximum
    certified = _solve(
        model,
        target=0.0,
        confidence=0.0,
        v=maximum,
        rho=1.0,
    )
    assert certified.value == maximum
    assert certified.certificate_kind in {'point_kkt', 'adjacent_bracket'}
    assert certified.objective == 1.5966722476277755e293


@pytest.mark.parametrize(
    ('displacement', 'confidence'),
    [
        (math.ldexp(1.5, -537), math.ldexp(1.0, 600)),
        (math.ldexp(1.0, -600), math.ldexp(1.0, 600)),
        (np.finfo(np.float64).max, math.ulp(0.0)),
    ],
)
def test_scaled_squared_mismatch_keeps_complete_source_factors(
    displacement: float,
    confidence: float,
) -> None:
    model = separator.FitModel()
    spec = _compile_scalar_prox_spec(model)
    exact = (
        Fraction(1, 2)
        * Fraction.from_float(confidence)
        * Fraction.from_float(displacement) ** 2
    )
    expected = float(exact)
    private = scalar_objective._scalar_objective_value(
        spec.objective,
        y=displacement,
        target=0.0,
        confidence=confidence,
        v=displacement,
        rho=1.0,
    )
    public = scalar_objective._mismatch_values(
        np.array([displacement]),
        np.array([0.0]),
        np.array([confidence]),
        model.mismatch,
    )[0]
    assert private == public == expected


def test_huber_and_soft_values_keep_separate_compound_factors() -> None:
    y = math.ldexp(1.0, 500)
    delta = math.ldexp(1.0, -400)
    confidence = math.ldexp(1.0, -800)
    huber = separator.HuberLoss(delta)
    model = separator.FitModel(mismatch=huber)
    spec = _compile_scalar_prox_spec(model)
    exact_y = Fraction.from_float(y)
    exact_delta = Fraction.from_float(delta)
    exact_confidence = Fraction.from_float(confidence)
    expected_huber = float(
        exact_confidence
        * exact_delta
        * (exact_y - exact_delta / 2)
    )
    private_huber = scalar_objective._scalar_objective_value(
        spec.objective,
        y=y,
        target=0.0,
        confidence=confidence,
        v=y,
        rho=1.0,
    )
    public_huber = scalar_objective._mismatch_values(
        np.array([y]),
        np.array([0.0]),
        np.array([confidence]),
        huber,
    )[0]

    displacement = math.ldexp(1.5, -537)
    strength = math.ldexp(1.0, 600)
    penalty = separator.SoftIntervalPenalty(0.0, 1.0, strength)
    expected_soft = float(
        Fraction.from_float(strength)
        * Fraction.from_float(displacement) ** 2
    )
    public_soft = scalar_objective._penalty_value(
        np.array([-displacement]),
        penalty,
    )[0]
    soft_spec = _compile_scalar_prox_spec(
        separator.FitModel(penalties=(penalty,))
    )
    private_soft = scalar_objective._scalar_objective_value(
        soft_spec.objective,
        y=-displacement,
        target=0.0,
        confidence=0.0,
        v=-displacement,
        rho=1.0,
    )
    assert private_huber == public_huber == expected_huber
    assert private_soft == public_soft == expected_soft


def test_huber_quadratic_and_proximal_square_keep_complete_factors() -> None:
    displacement = math.ldexp(1.5, -537)
    scale = math.ldexp(1.0, 600)
    expected = float(
        Fraction(1, 2)
        * Fraction.from_float(scale)
        * Fraction.from_float(displacement) ** 2
    )
    huber = separator.HuberLoss(math.ldexp(1.0, -500))
    huber_spec = _compile_scalar_prox_spec(
        separator.FitModel(mismatch=huber)
    )
    private_huber = scalar_objective._scalar_objective_value(
        huber_spec.objective,
        y=displacement,
        target=0.0,
        confidence=scale,
        v=displacement,
        rho=1.0,
    )
    public_huber = scalar_objective._mismatch_values(
        np.array([displacement]),
        np.array([0.0]),
        np.array([scale]),
        huber,
    )[0]
    proximal = scalar_objective._scalar_objective_value(
        _compile_scalar_prox_spec(separator.FitModel()).objective,
        y=displacement,
        target=displacement,
        confidence=1.0,
        v=0.0,
        rho=scale,
    )
    assert private_huber == public_huber == expected
    assert proximal == expected


def test_l2_value_keeps_complete_half_strength_and_square_factors() -> None:
    displacement = math.ldexp(1.0, -600)
    strength = math.ldexp(1.0, 600)
    expected = float(
        Fraction(1, 2)
        * Fraction.from_float(strength)
        * Fraction.from_float(displacement) ** 2
    )
    assert scalar_objective._l2_value(
        np.array([displacement]),
        np.array([0.0]),
        strength,
    ) == expected


def test_ordinary_squared_mismatch_avoids_scalar_normalization(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def forbidden(*args: object, **kwargs: object) -> object:
        raise AssertionError('ordinary squared rows reached scalar normalization')

    monkeypatch.setattr(
        scalar_objective,
        '_scaled_square_difference_dd',
        forbidden,
    )
    shape = (16, 32)
    measurement = np.linspace(-2.0, 3.0, np.prod(shape)).reshape(shape)
    target = np.full(shape, 0.25)
    confidence = np.full(shape, 1.5)
    actual = scalar_objective._mismatch_values(
        measurement,
        target,
        confidence,
        separator.SquaredLoss(),
    )
    expected = 0.75 * (measurement - target) ** 2
    np.testing.assert_allclose(actual, expected, rtol=4e-16, atol=0.0)


def test_ordinary_affine_mismatch_avoids_scalar_normalization(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def forbidden(*args: object, **kwargs: object) -> object:
        raise AssertionError('ordinary affine rows reached scalar normalization')

    monkeypatch.setattr(
        scalar_objective,
        '_normalized_affine_parts',
        forbidden,
    )
    shape = (16, 32)
    left = np.linspace(0.25, 1.25, np.prod(shape)).reshape(shape)
    zeros = np.zeros(shape)
    ones = np.ones(shape)
    actual = scalar_objective._weighted_squared_affine_value(
        zeros,
        ones,
        left,
        zeros,
        zeros,
        np.full(shape, 2.0),
    )
    np.testing.assert_allclose(actual, left**2, rtol=4e-16, atol=0.0)


def test_ordinary_huber_linear_rows_avoid_scalar_normalization(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def forbidden(*args: object, **kwargs: object) -> object:
        raise AssertionError('ordinary Huber rows reached scalar normalization')

    monkeypatch.setattr(
        scalar_objective,
        '_huber_linear_value_dd',
        forbidden,
    )
    shape = (16, 32)
    measurement = np.linspace(2.0, 3.0, np.prod(shape)).reshape(shape)
    target = np.zeros(shape)
    confidence = np.full(shape, 1.25)
    delta = 0.5
    actual = scalar_objective._mismatch_values(
        measurement,
        target,
        confidence,
        separator.HuberLoss(delta),
    )
    expected = confidence * delta * (measurement - delta / 2)
    np.testing.assert_allclose(actual, expected, rtol=4e-16, atol=0.0)


def test_ordinary_l2_terms_avoid_scalar_normalization(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def forbidden(*args: object, **kwargs: object) -> object:
        raise AssertionError('ordinary L2 rows reached scalar normalization')

    monkeypatch.setattr(
        scalar_objective,
        '_scaled_square_difference_dd',
        forbidden,
    )
    weights = np.linspace(-2.0, 3.0, 2048)
    reference = np.full(weights.shape, 0.25)
    expected = math.fsum(
        0.75 * float(value - 0.25) ** 2
        for value in weights
    )
    actual = scalar_objective._l2_value(weights, reference, 1.5)
    assert actual == pytest.approx(expected, rel=4e-16)


def test_mixed_squared_rows_repair_only_exceptional_positions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = scalar_objective._scaled_square_difference_dd
    calls: list[tuple[float, float, tuple[float, ...]]] = []

    def recorded(
        left: float,
        right: float,
        *factors: float,
    ) -> object:
        calls.append((left, right, factors))
        return original(left, right, *factors)

    monkeypatch.setattr(
        scalar_objective,
        '_scaled_square_difference_dd',
        recorded,
    )
    shape = (3, 7)
    measurement = np.ones(shape)
    target = np.zeros(shape)
    confidence = np.full(shape, 2.0)
    exceptional_index = (1, 4)
    maximum = np.finfo(np.float64).max
    confidence[exceptional_index] = math.ulp(0.0)
    measurement[exceptional_index] = maximum
    expected_exceptional = float(
        Fraction(1, 2)
        * Fraction.from_float(math.ulp(0.0))
        * Fraction.from_float(maximum) ** 2
    )
    actual = scalar_objective._mismatch_values(
        measurement,
        target,
        confidence,
        separator.SquaredLoss(),
    )
    expected = np.ones(shape)
    expected[exceptional_index] = expected_exceptional
    np.testing.assert_array_equal(actual, expected)
    assert calls == [(maximum, 0.0, (0.5, math.ulp(0.0)))]


def test_mixed_affine_rows_repair_only_exceptional_positions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = scalar_objective._normalized_affine_parts
    calls: list[tuple[float, float, float, float, float]] = []

    def recorded(
        beta: float,
        alpha: float,
        left: float,
        right: float,
        target: float = 0.0,
    ) -> object:
        calls.append((beta, alpha, left, right, target))
        return original(beta, alpha, left, right, target)

    monkeypatch.setattr(
        scalar_objective,
        '_normalized_affine_parts',
        recorded,
    )
    shape = (2, 5)
    beta = np.zeros(shape)
    alpha = np.ones(shape)
    left = np.ones(shape)
    right = np.zeros(shape)
    target = np.zeros(shape)
    confidence = np.full(shape, 2.0)
    exceptional_index = (1, 3)
    maximum = np.finfo(np.float64).max
    exceptional_right = math.nextafter(2.0, 0.0)
    alpha[exceptional_index] = maximum
    left[exceptional_index] = 2.0
    right[exceptional_index] = exceptional_right
    confidence[exceptional_index] = math.ldexp(1.0, -1000)
    exact_residual = Fraction.from_float(maximum) * (
        Fraction.from_float(2.0) - Fraction.from_float(exceptional_right)
    )
    expected_exceptional = float(
        Fraction(1, 2)
        * Fraction.from_float(confidence[exceptional_index])
        * exact_residual**2
    )
    actual = scalar_objective._weighted_squared_affine_value(
        beta,
        alpha,
        left,
        right,
        target,
        confidence,
    )
    expected = np.ones(shape)
    expected[exceptional_index] = expected_exceptional
    np.testing.assert_array_equal(actual, expected)
    assert calls == [(0.0, maximum, 2.0, exceptional_right, 0.0)]


def test_exceptional_huber_and_l2_rows_use_scalar_normalization(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_huber = scalar_objective._huber_linear_value_dd
    original_square = scalar_objective._scaled_square_difference_dd
    huber_calls = 0
    square_calls = 0

    def recorded_huber(*args: object, **kwargs: object) -> object:
        nonlocal huber_calls
        huber_calls += 1
        return original_huber(*args, **kwargs)

    def recorded_square(*args: object, **kwargs: object) -> object:
        nonlocal square_calls
        square_calls += 1
        return original_square(*args, **kwargs)

    monkeypatch.setattr(
        scalar_objective,
        '_huber_linear_value_dd',
        recorded_huber,
    )
    monkeypatch.setattr(
        scalar_objective,
        '_scaled_square_difference_dd',
        recorded_square,
    )
    y = math.ldexp(1.0, 500)
    delta = math.ldexp(1.0, -400)
    confidence = math.ldexp(1.0, -800)
    expected_huber = float(
        Fraction.from_float(confidence)
        * Fraction.from_float(delta)
        * (
            Fraction.from_float(y)
            - Fraction.from_float(delta) / 2
        )
    )
    actual_huber = scalar_objective._mismatch_values(
        np.array([2.0, y, 3.0]),
        np.zeros(3),
        np.array([1.0, confidence, 1.0]),
        separator.HuberLoss(delta),
    )[1]

    maximum = np.finfo(np.float64).max
    weights = np.array([0.0, maximum, 0.0])
    expected_l2 = float(
        Fraction(1, 2)
        * Fraction.from_float(math.ulp(0.0))
        * Fraction.from_float(maximum) ** 2
    )
    actual_l2 = scalar_objective._l2_value(
        weights,
        np.zeros(3),
        math.ulp(0.0),
    )
    assert actual_huber == expected_huber
    assert actual_l2 == expected_l2
    assert huber_calls == 1
    assert square_calls == 1


def test_derivative_consumers_disable_discarded_value_work(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    mismatch_flags: list[bool] = []
    penalty_flags: list[bool] = []
    original_mismatch = separator_problem._objective_mismatch_terms
    original_penalty = separator_problem._penalty_terms

    def recorded_mismatch(*args: object, **kwargs: object) -> object:
        mismatch_flags.append(bool(kwargs.get('evaluate_value', True)))
        return original_mismatch(*args, **kwargs)

    def recorded_penalty(*args: object, **kwargs: object) -> object:
        penalty_flags.append(bool(kwargs.get('evaluate_value', True)))
        return original_penalty(*args, **kwargs)

    monkeypatch.setattr(
        separator_problem,
        '_objective_mismatch_terms',
        recorded_mismatch,
    )
    monkeypatch.setattr(
        separator_problem,
        '_penalty_terms',
        recorded_penalty,
    )
    y = np.array([0.25, 0.75])
    separator_problem._mismatch_derivatives(
        y,
        np.zeros(2),
        np.ones(2),
        separator.SquaredLoss(),
    )
    separator_problem._penalty_derivatives(
        y,
        separator.SoftIntervalPenalty(0.0, 1.0, 1.0),
    )
    assert mismatch_flags == [False]
    assert penalty_flags == [False]


def test_scaled_square_final_range_rounding_is_from_complete_product() -> None:
    displacement = math.ldexp(1.0, -600)
    cases = (
        (math.ldexp(1.0, 151), math.ldexp(1.0, -1050)),
        (math.ldexp(1.0, 100), 0.0),
        (1.0, math.inf),
    )
    for confidence, expected in cases:
        y = displacement if math.isfinite(expected) else np.finfo(np.float64).max
        exact = (
            Fraction(1, 2)
            * Fraction.from_float(confidence)
            * Fraction.from_float(y) ** 2
        )
        try:
            oracle = float(exact)
        except OverflowError:
            oracle = math.inf
        assert oracle == expected
        private = scalar_objective._scalar_objective_value(
            _compile_scalar_prox_spec(separator.FitModel()).objective,
            y=y,
            target=0.0,
            confidence=confidence,
            v=y,
            rho=1.0,
        )
        public = scalar_objective._mismatch_values(
            np.array([y]),
            np.array([0.0]),
            np.array([confidence]),
            separator.SquaredLoss(),
        )[0]
        assert private == public == oracle


@pytest.mark.parametrize('sign', [-1.0, 1.0])
def test_scaled_factored_direct_difference_preserves_both_signs(
    sign: float,
) -> None:
    magnitude = math.ldexp(1.5, -537)
    lower = sign * magnitude
    upper = math.nextafter(lower, math.inf)
    confidence = math.ldexp(1.0, 600)
    exact = Fraction(1, 2) * Fraction.from_float(confidence) * (
        Fraction.from_float(upper) ** 2
        - Fraction.from_float(lower) ** 2
    )
    direct = scalar_objective._factored_square_difference(
        lower,
        upper,
        0.0,
        0.5,
        confidence,
    )
    enclosure = scalar_objective._scalar_objective_difference(
        _compile_scalar_prox_spec(separator.FitModel()).objective,
        lower=lower,
        upper=upper,
        target=0.0,
        confidence=confidence,
        v=lower,
        rho=0.0,
    )
    lower_bound, upper_bound = enclosure.physical_bounds()
    assert direct.value == float(exact)
    assert lower_bound <= float(exact) <= upper_bound
    assert enclosure.strictly_positive is (exact > 0)
    assert enclosure.strictly_negative is (exact < 0)


def test_scaled_affine_objective_reaches_result_report_and_json() -> None:
    displacement = math.ldexp(1.0, -600)
    confidence = math.ldexp(1.0, 600)
    expected = float(
        Fraction(1, 2)
        * Fraction.from_float(confidence)
        * Fraction.from_float(displacement) ** 2
    )
    points = np.array([[0.0, 0.0], [1.0, 0.0]])
    observations = separator.resolve_separator_observations(
        points,
        [(0, 1, 0.5)],
        confidence=np.array([confidence]),
    )
    problem = separator.build_power_fit_problem(observations)
    weights = np.array([displacement, -displacement])
    result = separator.build_power_fit_result(
        problem,
        weights,
        solver='external',
        status='external_candidate',
        converged=False,
        canonicalize_gauge=False,
    )
    report = separator.build_fit_report(result, observations)
    payload = json.loads(separator.dumps_report_json(report))
    assert result.objective_breakdown is not None
    assert problem.evaluate_objective(weights) == expected
    assert result.objective_breakdown.total == expected
    assert result.objective_breakdown.mismatch == expected
    assert report['objective_breakdown']['total'] == expected
    assert payload['objective_breakdown']['total'] == expected


def test_affine_value_normalizes_products_before_cancellation() -> None:
    alpha = np.finfo(np.float64).max
    left = 2.0
    right = math.nextafter(left, 0.0)
    confidence = math.ldexp(1.0, -1000)
    exact_residual = Fraction.from_float(alpha) * (
        Fraction.from_float(left) - Fraction.from_float(right)
    )
    expected = float(
        Fraction(1, 2)
        * Fraction.from_float(confidence)
        * exact_residual**2
    )
    actual = scalar_objective._weighted_squared_affine_value(
        np.array([0.0]),
        np.array([alpha]),
        np.array([left]),
        np.array([right]),
        np.array([0.0]),
        np.array([confidence]),
    )[0]
    assert actual == expected


def test_scaled_objectives_remain_certified_without_fallback() -> None:
    displacement = math.ldexp(1.0, -600)
    confidence = math.ldexp(1.0, 600)
    expected = float(
        Fraction(1, 2)
        * Fraction.from_float(confidence)
        * Fraction.from_float(displacement) ** 2
    )
    equality = _solve(
        separator.FitModel(),
        target=0.0,
        confidence=confidence,
        v=displacement,
        rho=1.0,
        lower=displacement,
        upper=displacement,
    )
    point = _solve(
        separator.FitModel(),
        target=0.0,
        confidence=confidence,
        v=displacement,
        rho=1.0,
        lower=displacement,
        upper=math.inf,
    )

    lower = math.ldexp(1.0, -500)
    upper = math.nextafter(lower, math.inf)
    adjacent_scale = math.ldexp(1.0, 600)
    adjacent_rho = math.ldexp(1.0, 599)
    adjacent = _solve(
        separator.FitModel(),
        target=lower,
        confidence=adjacent_scale,
        v=upper,
        rho=adjacent_rho,
        lower=lower,
        upper=upper,
    )
    adjacent_expected = float(
        Fraction(1, 2)
        * Fraction.from_float(adjacent_rho)
        * (Fraction.from_float(upper) - Fraction.from_float(lower)) ** 2
    )
    assert equality.certificate_kind == 'equality'
    assert equality.objective == expected
    assert point.certificate_kind == 'point_kkt'
    assert point.objective == expected
    assert adjacent.certificate_kind == 'adjacent_bracket'
    assert adjacent.objective == adjacent_expected
    assert equality.fallback_count == point.fallback_count == 0
    assert adjacent.fallback_count == 0


def test_direct_difference_enclosures_contain_independent_adjacent_oracle() -> None:
    model = separator.FitModel(
        mismatch=separator.HuberLoss(delta=0.3),
        penalties=(
            separator.SoftIntervalPenalty(-0.2, 0.7, 0.8),
            separator.ExponentialBoundaryPenalty(
                lower=-1.0,
                upper=1.2,
                margin=0.1,
                strength=0.03,
                tau=0.2,
            ),
            separator.ReciprocalBoundaryPenalty(
                lower=-1.0,
                upper=1.2,
                margin=0.2,
                strength=0.004,
                epsilon=0.02,
            ),
        ),
    )
    spec = _compile_scalar_prox_spec(model)
    rng = np.random.default_rng(937)
    kwargs = dict(target=-0.11, confidence=1.4, v=0.23, rho=0.9)
    for lower in rng.uniform(-0.9, 1.1, 64):
        lower = float(lower)
        upper = math.nextafter(lower, math.inf)
        enclosure = scalar_objective._scalar_objective_difference(
            spec.objective,
            lower=lower,
            upper=upper,
            **kwargs,
        )
        exact = _independent_objective(upper, model=model, **kwargs)
        exact -= _independent_objective(lower, model=model, **kwargs)
        lower_bound, upper_bound = enclosure.physical_bounds()
        assert _d(lower_bound) <= exact <= _d(upper_bound)


def test_derivative_enclosures_contain_independent_smooth_oracle() -> None:
    model = separator.FitModel(
        mismatch=separator.HuberLoss(delta=0.3),
        penalties=(
            separator.SoftIntervalPenalty(-0.2, 0.7, 0.8),
            separator.ExponentialBoundaryPenalty(
                lower=-1.0,
                upper=1.2,
                margin=0.1,
                strength=0.03,
                tau=0.2,
            ),
            separator.ReciprocalBoundaryPenalty(
                lower=-1.0,
                upper=1.2,
                margin=0.2,
                strength=0.004,
                epsilon=0.02,
            ),
        ),
    )
    spec = _compile_scalar_prox_spec(model)
    rng = np.random.default_rng(411)
    kwargs = dict(target=-0.11, confidence=1.4, v=0.23, rho=0.9)
    for value in rng.uniform(-0.9, 1.1, 64):
        value = float(value)
        evaluation = scalar_prox._evaluate_derivative(
            spec,
            y=value,
            **kwargs,
        )
        exact = _independent_derivative(
            _d(value),
            model=model,
            **kwargs,
        )
        minus, plus = evaluation.physical_enclosures()
        assert _d(minus[0]) <= exact <= _d(minus[1])
        assert _d(plus[0]) <= exact <= _d(plus[1])


def test_reciprocal_cancellation_fallback_uses_exact_algebraic_sign(
    monkeypatch,
) -> None:
    first = 1.0e12
    second = math.nextafter(first, math.inf)
    model = separator.FitModel(
        penalties=(
            separator.ReciprocalBoundaryPenalty(
                0.0, 1.0, 0.2, first, 0.1
            ),
            separator.ReciprocalBoundaryPenalty(
                -1.0, 0.0, 0.2, second, 0.1
            ),
        )
    )
    spec = _compile_scalar_prox_spec(model)
    evaluation = scalar_prox._evaluate_derivative(
        spec,
        y=-0.1,
        target=0.0,
        confidence=0.0,
        v=0.0,
        rho=1.0,
    )
    assert evaluation.minus.strictly_negative
    assert evaluation.plus.strictly_negative
    ambiguous = scalar_prox._ScaledEnclosure(-1.0, 1.0, 0.0)
    evaluation = scalar_prox._DerivativeEvaluation(
        y=evaluation.y,
        minus=ambiguous,
        plus=ambiguous,
        curvature_enclosure=evaluation.curvature_enclosure,
        smooth=evaluation.smooth,
    )

    def forbidden_interval(*args, **kwargs):
        raise AssertionError('purely algebraic sign requested Decimal work')

    monkeypatch.setattr(
        scalar_prox,
        '_exact_expression_interval',
        forbidden_interval,
    )
    resolved = scalar_prox._fallback_resolve_derivative(
        evaluation,
        spec=spec,
        target=0.0,
        confidence=0.0,
        v=0.0,
        rho=1.0,
    )
    assert resolved is not None
    assert resolved.fallback_minus_sign == -1
    assert resolved.fallback_plus_sign == -1
    assert resolved.fallback_minus_exact
    assert resolved.fallback_plus_exact
    assert resolved.fallback_minus_interval is None
    assert resolved.fallback_plus_interval is None


def test_exponential_fallback_interval_contains_independent_oracle() -> None:
    model = separator.FitModel(
        penalties=(
            separator.ExponentialBoundaryPenalty(
                lower=-0.4,
                upper=1.1,
                margin=0.07,
                strength=0.13,
                tau=0.09,
            ),
        )
    )
    spec = _compile_scalar_prox_spec(model)
    kwargs = dict(
        y=0.213,
        target=-0.17,
        confidence=1.3,
        v=0.42,
        rho=0.7,
    )
    expression = scalar_objective._scalar_derivative_exact_expression(
        spec.objective,
        side='plus',
        **kwargs,
    )
    interval = scalar_objective._exact_expression_interval(
        expression,
        precision=80,
    )
    with localcontext() as context:
        context.prec = 220
        oracle = _exact_decimal(expression.rational)
        for coefficient, exponent in expression.exponentials:
            oracle += _exact_decimal(coefficient) * (
                _exact_decimal(exponent).exp()
            )
    assert interval.lower <= oracle <= interval.upper


def test_exponential_interval_excluding_zero_authorizes_correct_sign() -> None:
    model = separator.FitModel(
        penalties=(separator.ExponentialBoundaryPenalty(),)
    )
    spec = _compile_scalar_prox_spec(model)
    expression = scalar_objective._scalar_derivative_exact_expression(
        spec.objective,
        y=0.1,
        target=0.0,
        confidence=1.0,
        v=0.0,
        rho=1.0,
        side='plus',
    )
    interval = scalar_objective._exact_expression_interval(
        expression,
        precision=80,
    )
    assert interval.lower > 0
    assert interval.sign == 1


def test_terminal_exponential_fallback_interval_contains_independent_oracle() -> None:
    model = separator.FitModel(
        mismatch=separator.HuberLoss(0.3),
        penalties=(
            separator.SoftIntervalPenalty(-0.2, 0.7, 0.8),
            separator.ExponentialBoundaryPenalty(
                lower=-1.0,
                upper=1.2,
                margin=0.1,
                strength=0.03,
                tau=0.2,
            ),
            separator.ReciprocalBoundaryPenalty(
                lower=-1.0,
                upper=1.2,
                margin=0.2,
                strength=0.004,
                epsilon=0.02,
            ),
        ),
    )
    spec = _compile_scalar_prox_spec(model)
    lower = 0.213
    upper = math.nextafter(lower, math.inf)
    kwargs = dict(target=-0.11, confidence=1.4, v=0.23, rho=0.9)
    interval = (
        scalar_objective._scalar_objective_difference_fallback_interval(
            spec.objective,
            lower=lower,
            upper=upper,
            precision=80,
            **kwargs,
        )
    )
    with localcontext() as context:
        context.prec = 100
        oracle = _independent_objective(
            upper,
            model=model,
            **kwargs,
        ) - _independent_objective(
            lower,
            model=model,
            **kwargs,
        )
    assert interval.lower <= oracle <= interval.upper
    assert interval.sign == (1 if oracle > 0 else -1)


def test_zero_containing_intervals_cannot_authorize_precision_agreement(
    monkeypatch,
) -> None:
    model = separator.FitModel(
        penalties=(separator.ExponentialBoundaryPenalty(),)
    )
    spec = _compile_scalar_prox_spec(model)
    ambiguous = scalar_prox._ScaledEnclosure(-1.0, 1.0, 0.0)
    curvature = scalar_prox._ScaledEnclosure(1.0, 1.0, 0.0)
    precisions: list[int] = []

    def ambiguous_derivative(spec, *, y, **kwargs):
        return scalar_prox._DerivativeEvaluation(
            y=y,
            minus=ambiguous,
            plus=ambiguous,
            curvature_enclosure=curvature,
            smooth=True,
        )

    def unresolved_interval(expression, *, precision):
        precisions.append(precision)
        # A nearest computation could report the same tiny nonzero sign at
        # both precisions; the rigorous enclosure still contains zero.
        return scalar_objective._DecimalInterval(
            Decimal('-1e-200'),
            Decimal('1e-200'),
        )

    monkeypatch.setattr(
        scalar_prox,
        '_evaluate_derivative',
        ambiguous_derivative,
    )
    monkeypatch.setattr(
        scalar_prox,
        '_exact_expression_interval',
        unresolved_interval,
    )
    with pytest.raises(_ScalarProxError) as error:
        _solve_scalar_prox_coordinate(
            spec=spec,
            target=0.0,
            confidence=1.0,
            v=0.0,
            rho=1.0,
        )
    assert error.value.failure.reason == 'high_precision_derivative_sign_unresolved'
    assert precisions == [80, 80, 160, 160]


def test_unresolved_terminal_objective_interval_is_structured_failure(
    monkeypatch,
) -> None:
    model = separator.FitModel(
        penalties=(separator.ExponentialBoundaryPenalty(),)
    )
    spec = _compile_scalar_prox_spec(model)
    precisions: list[int] = []

    monkeypatch.setattr(
        scalar_prox,
        '_scalar_objective_difference',
        lambda *args, **kwargs: scalar_prox._ScaledEnclosure(
            -1.0,
            1.0,
            0.0,
        ),
    )

    def unresolved_interval(expression, *, precision):
        precisions.append(precision)
        return scalar_objective._DecimalInterval(Decimal(-1), Decimal(1))

    monkeypatch.setattr(
        scalar_prox,
        '_exact_expression_interval',
        unresolved_interval,
    )
    with pytest.raises(_ScalarProxError) as error:
        _solve_scalar_prox_coordinate(
            spec=spec,
            target=0.0,
            confidence=1.0,
            v=0.0,
            rho=1.0,
        )
    assert error.value.failure.reason == (
        'high_precision_objective_difference_unresolved'
    )
    assert precisions == [80, 160]


def test_ordinary_exponential_path_does_not_use_decimal(monkeypatch) -> None:
    model = separator.FitModel(
        penalties=(separator.ExponentialBoundaryPenalty(),)
    )
    spec = _compile_scalar_prox_spec(model)

    class ForbiddenDecimal:
        def __new__(cls, *args, **kwargs):
            raise AssertionError('ordinary scalar path constructed Decimal')

        @classmethod
        def from_float(cls, value):
            raise AssertionError('ordinary scalar path constructed Decimal')

    monkeypatch.setattr(scalar_objective, 'Decimal', ForbiddenDecimal)
    monkeypatch.setattr(scalar_prox, 'Decimal', ForbiddenDecimal)
    result = _solve_scalar_prox_coordinate(
        spec=spec,
        target=0.0,
        confidence=1.0,
        v=0.0,
        rho=1.0,
    )
    assert result.fallback_count == 0
    assert result.certificate_kind == 'adjacent_bracket'


def test_reciprocal_regression_matches_independent_kink_minimum() -> None:
    model = separator.FitModel(
        penalties=(separator.ReciprocalBoundaryPenalty(),)
    )
    result = _solve(
        model,
        target=0.0,
        confidence=1.0,
        v=0.0,
        rho=1.0,
        lower=0.0,
        upper=1.0,
    )

    assert result.value == 0.05
    assert result.objective == pytest.approx(0.0025)
    assert result.certificate_kind == 'point_kkt'
    assert _independent_objective(
        result.value,
        model=model,
        target=0.0,
        confidence=1.0,
        v=0.0,
        rho=1.0,
    ) == pytest.approx(Decimal('0.0025'))


def test_exponential_regression_matches_decimal_derivative_bisection() -> None:
    model = separator.FitModel(
        penalties=(separator.ExponentialBoundaryPenalty(),)
    )
    result = _solve(
        model,
        target=-2.0,
        confidence=1.0,
        v=-2.0,
        rho=1.0,
    )
    oracle = _independent_minimum(
        model=model,
        target=-2.0,
        confidence=1.0,
        v=-2.0,
        rho=1.0,
    )

    assert result.value == pytest.approx(float(oracle), rel=0.0, abs=2e-16)
    assert result.objective == pytest.approx(4.251465264116678, rel=2e-15)
    _assert_local_minimum(
        result,
        model=model,
        target=-2.0,
        confidence=1.0,
        v=-2.0,
        rho=1.0,
    )


@pytest.mark.parametrize(
    (
        'model',
        'target',
        'confidence',
        'v',
        'rho',
        'expected',
        'expected_kind',
    ),
    [
        (
            separator.FitModel(),
            2.0,
            3.0,
            -1.0,
            2.0,
            0.8,
            'adjacent_bracket',
        ),
        (
            separator.FitModel(mismatch=separator.HuberLoss(delta=0.5)),
            2.0,
            3.0,
            -1.0,
            2.0,
            -0.25,
            'point_kkt',
        ),
    ],
)
def test_squared_and_huber_closed_form_prox(
    model,
    target,
    confidence,
    v,
    rho,
    expected,
    expected_kind,
) -> None:
    result = _solve(
        model,
        target=target,
        confidence=confidence,
        v=v,
        rho=rho,
    )
    assert result.value == pytest.approx(expected, rel=0.0, abs=2e-16)
    assert result.certificate_kind == expected_kind


@pytest.mark.parametrize(
    'model',
    [
        separator.FitModel(
            penalties=(separator.SoftIntervalPenalty(-0.2, 0.4, 3.0),)
        ),
        separator.FitModel(
            penalties=(
                separator.ExponentialBoundaryPenalty(
                    lower=-0.5,
                    upper=1.5,
                    margin=0.1,
                    strength=0.2,
                    tau=0.2,
                ),
            )
        ),
        separator.FitModel(
            penalties=(
                separator.ReciprocalBoundaryPenalty(
                    lower=-0.5,
                    upper=1.5,
                    margin=0.2,
                    strength=0.01,
                    epsilon=0.02,
                ),
            )
        ),
        separator.FitModel(
            mismatch=separator.HuberLoss(delta=0.3),
            penalties=(
                separator.SoftIntervalPenalty(-0.1, 0.8, 0.7),
                separator.ExponentialBoundaryPenalty(
                    lower=-0.5,
                    upper=1.5,
                    margin=0.1,
                    strength=0.01,
                    tau=0.2,
                ),
                separator.ReciprocalBoundaryPenalty(
                    lower=-0.5,
                    upper=1.5,
                    margin=0.15,
                    strength=0.005,
                    epsilon=0.01,
                ),
            ),
        ),
    ],
)
def test_individual_penalties_and_allowed_sum_match_decimal_oracle(model) -> None:
    kwargs = dict(target=-0.7, confidence=1.3, v=0.9, rho=0.8)
    result = _solve(model, **kwargs)
    oracle = _independent_minimum(model=model, **kwargs)
    assert result.value == pytest.approx(float(oracle), rel=0.0, abs=2e-14)
    _assert_local_minimum(result, model=model, **kwargs)


@pytest.mark.parametrize(
    ('lower', 'upper', 'target', 'expected_kind'),
    [
        (-math.inf, math.inf, 0.25, 'point_kkt'),
        (0.5, math.inf, -2.0, 'point_kkt'),
        (-math.inf, -0.5, 2.0, 'point_kkt'),
        (0.5, 0.5, -2.0, 'equality'),
        (0.0, 1.0, -2.0, 'point_kkt'),
        (0.0, 1.0, 2.0, 'point_kkt'),
    ],
)
def test_finite_one_sided_unbounded_and_equality_domains(
    lower,
    upper,
    target,
    expected_kind,
) -> None:
    model = separator.FitModel()
    result = _solve(
        model,
        target=target,
        confidence=1.0,
        v=target,
        rho=1.0,
        lower=lower,
        upper=upper,
    )
    expected = min(max(target, lower), upper)
    assert result.value == expected
    assert result.certificate_kind == expected_kind


def test_breakpoint_neighbors_and_coincident_breakpoints_are_compiled_exactly() -> None:
    model = separator.FitModel(
        mismatch=separator.HuberLoss(delta=0.1),
        penalties=(
            separator.SoftIntervalPenalty(0.1, 0.9, 2.0),
            separator.ReciprocalBoundaryPenalty(
                lower=0.0,
                upper=1.0,
                margin=0.1,
                strength=0.5,
                epsilon=0.01,
            ),
        ),
    )
    spec = _compile_scalar_prox_spec(model)
    breakpoints = scalar_prox._coordinate_breakpoints(
        spec,
        target=0.0,
        lower=0.0,
        upper=1.0,
    )
    candidates = scalar_prox._candidate_values(
        breakpoints,
        lower=0.0,
        upper=1.0,
    )

    assert len(breakpoints) < 10
    for breakpoint in breakpoints:
        if not 0 <= breakpoint <= 1:
            continue
        rounded = float(breakpoint)
        if breakpoint != Fraction.from_float(rounded):
            below = math.nextafter(rounded, -math.inf)
            above = math.nextafter(rounded, math.inf)
            assert rounded in candidates
            assert below in candidates or above in candidates


def test_every_structural_breakpoint_and_both_binary64_neighborhoods_evaluate() -> None:
    model = separator.FitModel(
        mismatch=separator.HuberLoss(delta=0.07),
        penalties=(
            separator.SoftIntervalPenalty(0.13, 0.83, 0.4),
            separator.ReciprocalBoundaryPenalty(
                lower=0.03,
                upper=0.97,
                margin=0.11,
                strength=0.002,
                epsilon=0.009,
            ),
        ),
    )
    spec = _compile_scalar_prox_spec(model)
    lower = 0.04
    upper = 0.93
    breakpoints = scalar_prox._coordinate_breakpoints(
        spec,
        target=0.41,
        lower=lower,
        upper=upper,
    )
    for breakpoint in breakpoints:
        below, above = scalar_prox._fraction_float_neighbors(breakpoint)
        probes = {
            below,
            above,
            math.nextafter(below, -math.inf),
            math.nextafter(above, math.inf),
        }
        for probe in probes:
            if not math.isfinite(probe) or not lower <= probe <= upper:
                continue
            evaluation = scalar_prox._evaluate_derivative(
                spec,
                y=probe,
                target=0.41,
                confidence=1.2,
                v=0.55,
                rho=0.8,
            )
            assert evaluation.g_minus <= evaluation.g_plus
            assert math.isfinite(
                float(
                    _independent_objective(
                        probe,
                        model=model,
                        target=0.41,
                        confidence=1.2,
                        v=0.55,
                        rho=0.8,
                    )
                )
            )

    kink_model = separator.FitModel(
        penalties=(separator.ReciprocalBoundaryPenalty(),)
    )
    kink_spec = _compile_scalar_prox_spec(kink_model)
    kink = scalar_prox._evaluate_derivative(
        kink_spec,
        y=0.05,
        target=0.0,
        confidence=1.0,
        v=0.0,
        rho=1.0,
    )
    assert kink.g_minus < kink.g_plus
    assert kink.smooth is False


def test_zero_strength_terms_are_exactly_absent_and_not_compiled() -> None:
    plain = separator.FitModel()
    zero = separator.FitModel(
        penalties=(
            separator.SoftIntervalPenalty(-1.0, 1.0, 0.0),
            separator.ExponentialBoundaryPenalty(strength=0.0),
            separator.ReciprocalBoundaryPenalty(strength=0.0),
        )
    )
    kwargs = dict(target=0.3, confidence=1.7, v=-0.4, rho=2.2)
    plain_result = _solve(plain, **kwargs)
    zero_result = _solve(zero, **kwargs)
    assert zero_result == plain_result
    assert _compile_scalar_prox_spec(zero).objective.penalties == ()


def test_vector_fast_rows_do_not_call_general_scalar_solver(monkeypatch) -> None:
    def forbidden(**kwargs):
        raise AssertionError('general scalar solver was called')

    monkeypatch.setattr(
        separator_solver,
        '_solve_scalar_prox_coordinate',
        forbidden,
    )
    values = np.array([0.2, 0.5])
    target = np.array([0.2, 0.5])
    confidence = np.ones(2)
    for model in (
        separator.FitModel(),
        separator.FitModel(
            penalties=(separator.SoftIntervalPenalty(0.0, 1.0, 3.0),)
        ),
        separator.FitModel(
            penalties=(separator.ReciprocalBoundaryPenalty(),)
        ),
        separator.FitModel(
            penalties=(separator.ExponentialBoundaryPenalty(strength=0.0),)
        ),
    ):
        actual = separator_solver._prox_measurement_objective(
            values,
            target,
            confidence,
            model=model,
            rho=1.0,
            y_lo=None,
            y_hi=None,
        )
        np.testing.assert_array_equal(actual, values)


@pytest.mark.parametrize(
    'model',
    [
        separator.FitModel(
            penalties=(
                separator.ExponentialBoundaryPenalty(
                    lower=-1.0,
                    upper=1.0,
                    margin=0.01,
                    strength=1e-250,
                    tau=1e-3,
                ),
            )
        ),
        separator.FitModel(
            penalties=(
                separator.ReciprocalBoundaryPenalty(
                    lower=-1.0,
                    upper=1.0,
                    margin=1e-100,
                    strength=1e-200,
                    epsilon=1e-200,
                ),
            )
        ),
    ],
)
def test_extreme_finite_penalty_scales_remain_certifiable(model) -> None:
    result = _solve(
        model,
        target=-0.2,
        confidence=1.0,
        v=0.3,
        rho=2.0,
    )
    assert math.isfinite(result.value)
    assert math.isfinite(result.objective)


def test_expansion_and_iteration_limits_return_structured_failure(
    monkeypatch,
) -> None:
    model = separator.FitModel(
        penalties=(separator.ExponentialBoundaryPenalty(),)
    )
    spec = _compile_scalar_prox_spec(model)
    monkeypatch.setattr(scalar_prox, '_MAX_EXPANSIONS', 0)
    with pytest.raises(_ScalarProxError) as expansion_error:
        _solve_scalar_prox_coordinate(
            spec=spec,
            target=-2.0,
            confidence=1.0,
            v=-2.0,
            rho=1.0,
        )
    assert expansion_error.value.failure.reason == 'bracketing_expansion_limit'
    assert expansion_error.value.failure.scalar_iterations == 0

    monkeypatch.setattr(scalar_prox, '_MAX_EXPANSIONS', 128)
    monkeypatch.setattr(scalar_prox, '_MAX_SCALAR_ITERATIONS', 0)
    with pytest.raises(_ScalarProxError) as iteration_error:
        _solve_scalar_prox_coordinate(
            spec=spec,
            target=-2.0,
            confidence=1.0,
            v=-2.0,
            rho=1.0,
        )
    assert iteration_error.value.failure.reason == 'scalar_iteration_limit'
    assert iteration_error.value.failure.last_finite_bracket is not None


def test_nonfinite_authoritative_objective_is_failure(monkeypatch) -> None:
    model = separator.FitModel()
    monkeypatch.setattr(
        scalar_prox,
        '_scalar_objective_value',
        lambda *args, **kwargs: math.inf,
    )
    with pytest.raises(_ScalarProxError) as error:
        _solve(
            model,
            target=0.0,
            confidence=1.0,
            v=0.0,
            rho=1.0,
        )
    assert error.value.failure.reason == 'nonfinite_authoritative_objective'


def test_adjacent_binary64_bracket_certificate(monkeypatch) -> None:
    lower_root = 1.0
    upper_root = math.nextafter(lower_root, math.inf)
    root = (_d(lower_root) + _d(upper_root)) / 2
    model = separator.FitModel()

    def exact_derivative(spec, *, y, **kwargs):
        derivative = float(_d(y) - root)
        enclosure = scalar_prox._ScaledEnclosure(
            derivative,
            derivative,
            0.0,
        )
        return scalar_prox._DerivativeEvaluation(
            y=y,
            minus=enclosure,
            plus=enclosure,
            curvature_enclosure=scalar_prox._ScaledEnclosure(
                1.0,
                1.0,
                0.0,
            ),
            smooth=True,
        )

    monkeypatch.setattr(scalar_prox, '_evaluate_derivative', exact_derivative)
    result = _solve(
        model,
        target=lower_root,
        confidence=1.0,
        v=upper_root,
        rho=1.0,
    )
    assert result.certificate_kind == 'adjacent_bracket'
    assert result.final_bracket == (lower_root, upper_root)
    assert result.value == scalar_prox._ties_to_even_endpoint(
        lower_root,
        upper_root,
    )
    assert result.fallback_count == 1


def test_deterministic_randomized_smooth_cases_match_decimal_oracle() -> None:
    rng = np.random.default_rng(3702)
    for _ in range(24):
        target = float(rng.uniform(-1.0, 1.0))
        v = float(rng.uniform(-1.0, 1.0))
        confidence = float(10 ** rng.uniform(-2.0, 2.0))
        rho = float(10 ** rng.uniform(-2.0, 2.0))
        if rng.integers(0, 2):
            mismatch = separator.SquaredLoss()
        else:
            mismatch = separator.HuberLoss(float(rng.uniform(0.05, 0.8)))
        penalties = (
            separator.SoftIntervalPenalty(-0.4, 0.7, float(rng.uniform(0.0, 3.0))),
            separator.ExponentialBoundaryPenalty(
                lower=-1.0,
                upper=1.2,
                margin=0.1,
                strength=float(10 ** rng.uniform(-4.0, -1.0)),
                tau=float(rng.uniform(0.1, 0.4)),
            ),
        )
        model = separator.FitModel(mismatch=mismatch, penalties=penalties)
        kwargs = dict(
            target=target,
            confidence=confidence,
            v=v,
            rho=rho,
        )
        result = _solve(model, **kwargs)
        oracle = _independent_minimum(model=model, **kwargs)
        assert result.value == pytest.approx(float(oracle), rel=0.0, abs=5e-13)
        _assert_local_minimum(result, model=model, **kwargs)


@pytest.mark.parametrize('linear_backend', ['dense', 'sparse'])
def test_end_to_end_admm_with_scalar_penalty(linear_backend) -> None:
    if linear_backend == 'sparse':
        pytest.importorskip('scipy.sparse.linalg')
    points = np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
    model = separator.FitModel(
        penalties=(separator.ExponentialBoundaryPenalty(),)
    )
    result = separator.fit_weights_from_separators(
        points,
        [(0, 1, -2.0)],
        model=model,
        solver='admm',
        linear_backend=linear_backend,
        admm_max_iter=5000,
        connectivity_check='diagnose',
    )
    assert result.status == 'optimal'
    assert result.converged is True
    assert result.objective_breakdown is not None
    assert math.isfinite(result.objective_breakdown.total)


def test_scalar_failure_forwards_to_result_report_json_and_active_set(
    monkeypatch,
) -> None:
    failure = scalar_prox._ScalarProxFailure(
        reason='forced_test_failure',
        scalar_iterations=7,
        expansion_count=3,
        last_candidate=0.25,
        last_finite_bracket=(0.0, 1.0),
        last_derivative_enclosure=((-2.0, -1.0), (1.0, 2.0)),
        localization_bound=1.0,
        fallback_count=2,
    )

    def forced_failure(**kwargs):
        raise _ScalarProxError(failure)

    monkeypatch.setattr(
        separator_solver,
        '_solve_scalar_prox_coordinate',
        forced_failure,
    )
    points = np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
    domain = Box(((-2.0, 4.0), (-2.0, 2.0), (-2.0, 2.0)))
    observations = separator.resolve_separator_observations(
        points,
        [(0, 1, 0.25)],
        domain=domain,
    )
    model = separator.FitModel(
        penalties=(separator.ExponentialBoundaryPenalty(),)
    )
    result = separator.fit_weights_from_separators(
        points,
        observations,
        model=model,
        solver='admm',
        connectivity_check='diagnose',
    )
    report = separator.build_fit_report(result, observations)
    encoded = json.loads(json.dumps(report))

    assert result.status == 'numerical_failure'
    assert result.converged is False
    assert result.weights is None
    assert result.objective_breakdown is None
    assert result.n_iter == 0
    assert result.solver_termination.status == 'numerical_failure'
    assert result.solver_termination.converged is False
    assert result.solver_termination.n_iter == 0
    assert 'observation row 0' in result.status_detail
    assert 'component-local row 0' in result.status_detail
    assert 'forced_test_failure' in result.status_detail
    assert 'last_candidate=0.25' in result.status_detail
    assert 'last_finite_bracket=(0.0, 1.0)' in result.status_detail
    assert 'last_derivative_enclosure=((-2.0, -1.0), (1.0, 2.0))' in (
        result.status_detail
    )
    assert 'localization_bound=1.0' in result.status_detail
    assert 'fallback_count=2' in result.status_detail
    assert encoded['summary']['status'] == 'numerical_failure'
    assert encoded['objective_breakdown'] is None

    active = separator.solve_self_consistent_power_weights(
        points,
        observations,
        model=model,
        domain=domain,
        fit_solver='admm',
        connectivity_check='diagnose',
        unaccounted_pair_check='diagnose',
    )
    assert active.termination == 'numerical_failure'
    assert active.converged is False
    assert active.fit.status == 'numerical_failure'
    assert 'last_candidate=0.25' in active.fit.status_detail
    assert 'fallback_count=2' in active.fit.status_detail
    assert active.final_state_available is False
    assert active.final_state_unavailable_reason == 'numerical_failure'
    assert active.realized is None
    assert active.diagnostics is None


def test_later_scalar_failure_counts_only_completed_admm_iterations(
    monkeypatch,
) -> None:
    original = separator_solver._solve_scalar_prox_coordinate
    calls = 0

    def fail_second_call(**kwargs):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise _ScalarProxError(
                scalar_prox._ScalarProxFailure(
                    reason='forced_later_failure',
                    scalar_iterations=5,
                    expansion_count=1,
                    last_candidate=-0.25,
                    last_finite_bracket=(-0.5, 0.0),
                    fallback_count=1,
                )
            )
        return original(**kwargs)

    monkeypatch.setattr(
        separator_solver,
        '_solve_scalar_prox_coordinate',
        fail_second_call,
    )
    points = np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
    result = separator.fit_weights_from_separators(
        points,
        [(0, 1, -2.0)],
        model=separator.FitModel(
            penalties=(separator.ExponentialBoundaryPenalty(),)
        ),
        solver='admm',
        admm_max_iter=5000,
        connectivity_check='diagnose',
    )
    assert calls == 2
    assert result.status == 'numerical_failure'
    assert result.converged is False
    assert result.n_iter == 1
    assert result.weights is None
    assert result.objective_breakdown is None
    assert 'last_candidate=-0.25' in result.status_detail
