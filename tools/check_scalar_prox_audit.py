#!/usr/bin/env python3
"""Run the complete deterministic R2 algebraic and exponential audits."""

from __future__ import annotations

from collections import Counter
from decimal import Decimal, ROUND_CEILING, ROUND_FLOOR, localcontext
from fractions import Fraction
import math
import random
import struct

from pyvoro2.inverse import separator
from pyvoro2.inverse.separator._scalar_prox import (
    _compile_scalar_prox_spec,
    _ScalarProxError,
    _solve_scalar_prox_coordinate,
)


def _finding_d_derivative(value: float, parameters: dict[str, float]) -> Decimal:
    with localcontext() as context:
        context.prec = 220
        d = Decimal.from_float
        y = d(value)
        residual = y - d(parameters['target'])
        delta = d(parameters['delta'])
        mismatch = (
            -d(parameters['confidence']) * delta
            if residual < -delta
            else d(parameters['confidence']) * delta
            if residual > delta
            else d(parameters['confidence']) * residual
        )
        left = (
            (
                d(parameters['lower'])
                + d(parameters['margin'])
                - y
            ) / d(parameters['tau'])
        ).exp()
        right = (
            (
                y
                - d(parameters['upper'])
                + d(parameters['margin'])
            ) / d(parameters['tau'])
        ).exp()
        return +(
            mismatch
            + d(parameters['rho']) * (y - d(parameters['v']))
            + d(parameters['strength']) / d(parameters['tau'])
            * (right - left)
        )


def _outward_decimal(value: Decimal) -> tuple[Decimal, Decimal]:
    """Round a high-precision independent value outward at 160 digits."""

    with localcontext() as context:
        context.prec = 160
        context.rounding = ROUND_FLOOR
        lower = (+value).next_minus()
    with localcontext() as context:
        context.prec = 160
        context.rounding = ROUND_CEILING
        upper = (+value).next_plus()
    return lower, upper


def _finding_d_derivative_interval(
    value: float,
    parameters: dict[str, float],
) -> tuple[Decimal, Decimal]:
    with localcontext() as context:
        context.prec = 220
        central = _finding_d_derivative(value, parameters)
    return _outward_decimal(central)


def _finding_d_objective_difference_interval(
    lower: float,
    upper: float,
    parameters: dict[str, float],
) -> tuple[Decimal, Decimal]:
    """Independently evaluate ``F(upper)-F(lower)`` without total subtraction."""

    with localcontext() as context:
        context.prec = 220
        d = Decimal.from_float
        gap = d(upper) - d(lower)
        mismatch = d(parameters['confidence']) * d(parameters['delta']) * gap
        proximal = (
            d(parameters['rho'])
            * gap
            * (d(upper) + d(lower) - 2 * d(parameters['v']))
            / 2
        )
        lower_argument_at_upper = (
            d(parameters['lower'])
            + d(parameters['margin'])
            - d(upper)
        ) / d(parameters['tau'])
        lower_argument_at_lower = (
            d(parameters['lower'])
            + d(parameters['margin'])
            - d(lower)
        ) / d(parameters['tau'])
        upper_argument_at_upper = (
            d(upper)
            - d(parameters['upper'])
            + d(parameters['margin'])
        ) / d(parameters['tau'])
        upper_argument_at_lower = (
            d(lower)
            - d(parameters['upper'])
            + d(parameters['margin'])
        ) / d(parameters['tau'])
        exponential = d(parameters['strength']) * (
            lower_argument_at_upper.exp()
            - lower_argument_at_lower.exp()
            + upper_argument_at_upper.exp()
            - upper_argument_at_lower.exp()
        )
        central = mismatch + proximal + exponential
    return _outward_decimal(central)


def _float_interval_contains_fraction(
    enclosure: tuple[float, float],
    value: Fraction,
) -> bool:
    return (
        Fraction.from_float(enclosure[0])
        <= value
        <= Fraction.from_float(enclosure[1])
    )


def _float_interval_contains_decimal(
    enclosure: tuple[float, float],
    value: tuple[Decimal, Decimal],
) -> bool:
    return (
        Decimal.from_float(enclosure[0]) <= value[0]
        and value[1] <= Decimal.from_float(enclosure[1])
    )


def _ties_to_even_endpoint(lower: float, upper: float) -> float:
    for value in (lower, upper):
        bits = struct.unpack('>Q', struct.pack('>d', value))[0]
        if bits & 1 == 0:
            return value
    raise AssertionError('adjacent binary64 endpoints did not contain an even LSB')


def _algebraic_audit() -> tuple[int, int, list[tuple[object, ...]]]:
    rng = random.Random(20260803)
    successes = 0
    failures = 0
    violations: list[tuple[object, ...]] = []
    for index in range(300):
        exponent = rng.randint(-40, 20)
        anchor = math.ldexp(rng.uniform(0.6, 1.4), exponent)
        delta = math.ldexp(rng.uniform(0.05, 0.3), rng.randint(-5, 5))
        confidence = math.ldexp(
            rng.uniform(0.5, 2.0),
            rng.randint(-5, 15),
        )
        rho = math.ldexp(rng.uniform(0.5, 2.0), rng.randint(5, 40))
        shift = (
            Fraction.from_float(confidence)
            * Fraction.from_float(delta)
            / Fraction.from_float(rho)
        )
        v = float(Fraction.from_float(anchor) - shift)
        root = Fraction.from_float(v) + shift
        target = float(
            root
            + 8 * Fraction.from_float(delta)
            + abs(Fraction.from_float(anchor)) / 10
            + 1
        )
        width = 4 * max(1.0, abs(float(root)), abs(target), abs(delta))
        model = separator.FitModel(
            mismatch=separator.HuberLoss(delta=delta),
            penalties=(
                separator.SoftIntervalPenalty(-width, width, 1e-12),
            ),
        )
        try:
            result = _solve_scalar_prox_coordinate(
                spec=_compile_scalar_prox_spec(model),
                target=target,
                confidence=confidence,
                v=v,
                rho=rho,
            )
        except _ScalarProxError:
            failures += 1
            continue
        successes += 1
        if result.certificate_kind != 'adjacent_bracket':
            violations.append(('unexpected_kind', index, result.certificate_kind))
            continue
        lower_float, upper_float = result.final_bracket
        lower = Fraction.from_float(lower_float)
        upper = Fraction.from_float(upper_float)
        derivative_lower = (
            -Fraction.from_float(confidence) * Fraction.from_float(delta)
            + Fraction.from_float(rho) * (lower - Fraction.from_float(v))
        )
        derivative_upper = (
            -Fraction.from_float(confidence) * Fraction.from_float(delta)
            + Fraction.from_float(rho) * (upper - Fraction.from_float(v))
        )
        if not derivative_lower < 0 < derivative_upper:
            violations.append(('false_bracket', index, result.final_bracket))
            continue
        derivative_evidence = result.adjacent_bracket_evidence
        if derivative_evidence is None:
            violations.append(('missing_derivative_evidence', index))
            continue
        left_evidence, right_evidence = derivative_evidence
        if not _float_interval_contains_fraction(
            left_evidence.plus_enclosure,
            derivative_lower,
        ):
            violations.append(('left_enclosure_miss', index))
        if not _float_interval_contains_fraction(
            right_evidence.minus_enclosure,
            derivative_upper,
        ):
            violations.append(('right_enclosure_miss', index))
        if (
            left_evidence.plus_resolved_sign != -1
            or right_evidence.minus_resolved_sign != 1
        ):
            violations.append(('derivative_evidence_sign', index))
        difference = (upper - lower) * (
            Fraction.from_float(rho)
            * Fraction(1, 2)
            * (upper + lower - 2 * Fraction.from_float(v))
            - Fraction.from_float(confidence) * Fraction.from_float(delta)
        )
        terminal = result.terminal_difference_evidence
        if terminal is None:
            violations.append(('missing_terminal_evidence', index))
            continue
        if not _float_interval_contains_fraction(
            terminal.ordinary_enclosure,
            difference,
        ):
            violations.append(('difference_enclosure_miss', index))
        exact_sign = int(difference > 0) - int(difference < 0)
        recorded_sign = (
            terminal.fallback_resolved_sign
            if terminal.fallback_resolved_sign is not None
            else terminal.ordinary_resolved_sign
        )
        if recorded_sign != exact_sign:
            violations.append(('difference_evidence_sign', index))
        if terminal.fallback_resolved_sign is not None:
            ordinary_lower, ordinary_upper = terminal.ordinary_enclosure
            if not ordinary_lower <= 0.0 <= ordinary_upper:
                violations.append(('unnecessary_difference_fallback', index))
            if not terminal.fallback_exact:
                violations.append(('algebraic_fallback_not_exact', index))
        expected = (
            lower_float
            if difference > 0
            else upper_float
            if difference < 0
            else _ties_to_even_endpoint(lower_float, upper_float)
        )
        if result.value != expected or terminal.selected_endpoint != expected:
            violations.append((
                'wrong_endpoint',
                index,
                result.final_bracket,
                result.value,
                expected,
            ))
        if result.localization_bound != upper_float - lower_float:
            violations.append(('wrong_localization', index))
    return successes, failures, violations


def _exponential_audit() -> tuple[int, int, list[tuple[object, ...]]]:
    base = {
        'delta': 3.342765432428535,
        'lower': 150.53416318031995,
        'upper': 157.82864753114254,
        'margin': 2.794426547999039,
        'strength': 240.8079420544141,
        'tau': 5.339859001557991,
        'target': -377.50566923018596,
        'confidence': 113507256.09123965,
        'v': -219.7275672967312,
        'rho': 212751954959.19275,
        'hard_upper': 427.4186885272726,
    }
    successes = 0
    failures = 0
    violations: list[tuple[object, ...]] = []
    for exponent in range(-30, 31):
        scale = math.ldexp(1.0, exponent)
        parameters = dict(base)
        for key in (
            'delta',
            'lower',
            'upper',
            'margin',
            'tau',
            'target',
            'v',
            'hard_upper',
        ):
            parameters[key] *= scale
        parameters['strength'] *= scale * scale
        model = separator.FitModel(
            mismatch=separator.HuberLoss(delta=parameters['delta']),
            penalties=(
                separator.ExponentialBoundaryPenalty(
                    lower=parameters['lower'],
                    upper=parameters['upper'],
                    margin=parameters['margin'],
                    strength=parameters['strength'],
                    tau=parameters['tau'],
                ),
            ),
        )
        try:
            result = _solve_scalar_prox_coordinate(
                spec=_compile_scalar_prox_spec(model),
                target=parameters['target'],
                confidence=parameters['confidence'],
                v=parameters['v'],
                rho=parameters['rho'],
                upper=parameters['hard_upper'],
            )
        except _ScalarProxError:
            failures += 1
            continue
        successes += 1
        if result.certificate_kind != 'adjacent_bracket':
            violations.append((
                'unexpected_kind',
                exponent,
                result.certificate_kind,
            ))
            continue
        lower, upper = result.final_bracket
        derivative_lower = _finding_d_derivative_interval(lower, parameters)
        derivative_upper = _finding_d_derivative_interval(upper, parameters)
        if not derivative_lower[1] < 0 < derivative_upper[0]:
            violations.append((
                'false_bracket',
                exponent,
                result.final_bracket,
            ))
            continue
        derivative_evidence = result.adjacent_bracket_evidence
        if derivative_evidence is None:
            violations.append(('missing_derivative_evidence', exponent))
            continue
        left_evidence, right_evidence = derivative_evidence
        if not _float_interval_contains_decimal(
            left_evidence.plus_enclosure,
            derivative_lower,
        ):
            violations.append(('left_enclosure_miss', exponent))
        if not _float_interval_contains_decimal(
            right_evidence.minus_enclosure,
            derivative_upper,
        ):
            violations.append(('right_enclosure_miss', exponent))
        for side, evidence, oracle in (
            ('left', left_evidence, derivative_lower),
            ('right', right_evidence, derivative_upper),
        ):
            fallback_sign = evidence.plus_fallback_resolved_sign
            fallback_interval = evidence.plus_fallback_enclosure
            ordinary_sign = evidence.plus_ordinary_resolved_sign
            if side == 'right':
                fallback_sign = evidence.minus_fallback_resolved_sign
                fallback_interval = evidence.minus_fallback_enclosure
                ordinary_sign = evidence.minus_ordinary_resolved_sign
            if fallback_sign is not None:
                if ordinary_sign is not None:
                    violations.append((
                        'unnecessary_derivative_fallback',
                        exponent,
                    ))
                if fallback_interval is None:
                    violations.append((
                        'missing_derivative_fallback_interval',
                        exponent,
                    ))
                elif not (
                    fallback_interval[0] <= oracle[0]
                    and oracle[1] <= fallback_interval[1]
                ):
                    violations.append(('derivative_fallback_miss', exponent))
        difference = _finding_d_objective_difference_interval(
            lower,
            upper,
            parameters,
        )
        terminal = result.terminal_difference_evidence
        if terminal is None:
            violations.append(('missing_terminal_evidence', exponent))
            continue
        if not _float_interval_contains_decimal(
            terminal.ordinary_enclosure,
            difference,
        ):
            violations.append(('difference_enclosure_miss', exponent))
        exact_sign = -1 if difference[1] < 0 else 1 if difference[0] > 0 else 0
        recorded_sign = (
            terminal.fallback_resolved_sign
            if terminal.fallback_resolved_sign is not None
            else terminal.ordinary_resolved_sign
        )
        if recorded_sign != exact_sign:
            violations.append(('difference_evidence_sign', exponent))
        if terminal.fallback_resolved_sign is not None:
            ordinary_lower, ordinary_upper = terminal.ordinary_enclosure
            if not ordinary_lower <= 0.0 <= ordinary_upper:
                violations.append(('unnecessary_difference_fallback', exponent))
            if terminal.fallback_enclosure is None:
                violations.append(('missing_difference_fallback_interval', exponent))
            elif not (
                terminal.fallback_enclosure[0] <= difference[0]
                and difference[1] <= terminal.fallback_enclosure[1]
            ):
                violations.append(('difference_fallback_miss', exponent))
        expected = lower if exact_sign > 0 else upper
        if exact_sign == 0:
            expected = _ties_to_even_endpoint(lower, upper)
        if result.value != expected or terminal.selected_endpoint != expected:
            violations.append(('wrong_endpoint', exponent, result.value, expected))
        if result.localization_bound != upper - lower:
            violations.append(('wrong_localization', exponent))
    return successes, failures, violations


def main() -> int:
    algebraic = _algebraic_audit()
    exponential = _exponential_audit()
    for name, count, result in (
        ('algebraic', 300, algebraic),
        ('exponential', 61, exponential),
    ):
        successes, failures, violations = result
        kinds = Counter(row[0] for row in violations)
        print(
            f'{name}_generated={count} successes={successes} '
            f'structured_failures={failures} violations={len(violations)} '
            f'derivative_enclosures={2 * successes} '
            f'objective_differences={successes} '
            f'endpoint_selections={successes} evidence_records={successes} '
            f'localization_checks={successes} types={dict(kinds)}'
        )
        for row in violations[:8]:
            print(f'{name.upper()}_VIOLATION', row)
    return 1 if algebraic[1:] != (0, []) or exponential[1:] != (0, []) else 0


if __name__ == '__main__':
    raise SystemExit(main())
