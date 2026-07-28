"""Authoritative scalar formulas for separator inverse objectives."""

from __future__ import annotations

import math
from dataclasses import dataclass
from fractions import Fraction

import numpy as np

from ._numerics import (
    _exact_sum_products_scalar,
    _exact_sum_ratios_scalar,
    _fraction,
    _stable_affine_residual,
    _stable_product,
    _stable_product_scalar,
    _stable_ratio_difference,
    _stable_ratio_product,
    _stable_scaled_affine_residual,
    _stable_scaled_difference,
    _stable_scaled_difference_scalar,
    _stable_sum,
    _stable_sum_products,
    _stable_sum_products_sign,
    _stable_sum_scalar,
    _stable_sum_squares,
)
from .model import (
    ExponentialBoundaryPenalty,
    HuberLoss,
    ReciprocalBoundaryPenalty,
    ScalarPenalty,
    SoftIntervalPenalty,
    SquaredLoss,
)


HARD_ATOL = 1e-12
HARD_RTOL = 64.0 * np.finfo(np.float64).eps
_FLOAT_EPSILON = np.finfo(np.float64).eps
_FLOAT_MAX_FRACTION = _fraction(np.finfo(np.float64).max)
_SQRT_HALF = math.sqrt(0.5)


@dataclass(frozen=True, slots=True)
class _QuadraticRowData:
    """Scale-safe row curvature, normal RHS, and diagnostic target."""

    rho: np.ndarray
    rhs: np.ndarray
    z_obs: np.ndarray


def _huber_difference_branches(
    measurement: np.ndarray,
    target: np.ndarray,
    delta: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Classify ``measurement - target`` against exact ``+/- delta``."""

    lower_sign = _stable_sum_products_sign(
        (
            (measurement,),
            (-1.0, target),
            (delta,),
        )
    )
    upper_sign = _stable_sum_products_sign(
        (
            (measurement,),
            (-1.0, target),
            (-delta,),
        )
    )
    quadratic = (lower_sign >= 0) & (upper_sign <= 0)
    direction = np.where(upper_sign > 0, 1.0, -1.0)
    return quadratic, direction


def _huber_affine_branches(
    beta: np.ndarray,
    alpha: np.ndarray,
    left: np.ndarray,
    right: np.ndarray,
    target: np.ndarray,
    delta: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Classify a complete affine residual against exact ``+/- delta``."""

    common = (
        (beta,),
        (alpha, left),
        (-1.0, alpha, right),
        (-1.0, target),
    )
    lower_sign = _stable_sum_products_sign(common + ((delta,),))
    upper_sign = _stable_sum_products_sign(common + ((-delta,),))
    quadratic = (lower_sign >= 0) & (upper_sign <= 0)
    direction = np.where(upper_sign > 0, 1.0, -1.0)
    return quadratic, direction


def _active_scalar_penalties(
    penalties: tuple[ScalarPenalty, ...],
) -> tuple[ScalarPenalty, ...]:
    """Return scalar penalties that are not exact zero-strength no-ops."""

    return tuple(
        penalty
        for penalty in penalties
        if float(penalty.strength) != 0.0
    )


def _quadratic_row_data(
    alpha: np.ndarray,
    beta: np.ndarray,
    target: np.ndarray,
    confidence: np.ndarray,
) -> _QuadraticRowData:
    """Construct authoritative quadratic row curvature and RHS values.

    For row ``r``, the returned values are
    ``rho = confidence * alpha**2`` and
    ``rhs = confidence * alpha * (target - beta)``.  The diagnostic
    ``z_obs`` is computed independently and is never used to reconstruct the
    normal RHS.
    """

    alpha_array, beta_array, target_array, confidence_array = np.broadcast_arrays(
        np.asarray(alpha, dtype=np.float64),
        np.asarray(beta, dtype=np.float64),
        np.asarray(target, dtype=np.float64),
        np.asarray(confidence, dtype=np.float64),
    )
    rho = np.zeros(alpha_array.shape, dtype=np.float64)
    rhs = np.zeros(alpha_array.shape, dtype=np.float64)
    active = confidence_array != 0.0
    if np.any(active):
        rho[active] = _stable_product(
            confidence_array[active],
            alpha_array[active],
            alpha_array[active],
        )
        rhs[active] = _stable_scaled_difference(
            target_array[active],
            beta_array[active],
            confidence_array[active],
            alpha_array[active],
        )
    z_obs = _stable_ratio_difference(
        target_array,
        beta_array,
        alpha_array,
    )
    return _QuadraticRowData(rho=rho, rhs=rhs, z_obs=z_obs)


def _weighted_squared_difference_value(
    left: np.ndarray,
    right: np.ndarray,
    confidence: np.ndarray,
) -> np.ndarray:
    """Return ``0.5 * confidence * (left - right)**2`` scale-safely."""

    scaled = _stable_scaled_difference(
        left,
        right,
        np.sqrt(confidence),
        _SQRT_HALF,
    )
    return _stable_product(scaled, scaled)


def _weighted_squared_affine_value(
    beta: np.ndarray,
    alpha: np.ndarray,
    left: np.ndarray,
    right: np.ndarray,
    target: np.ndarray,
    confidence: np.ndarray,
) -> np.ndarray:
    """Return one weighted affine square without storing its residual."""

    residual, exceptional = _stable_affine_residual(
        beta,
        alpha,
        left,
        right,
        target,
        return_exceptional=True,
    )
    value = np.zeros(residual.shape, dtype=np.float64)
    ordinary = np.isfinite(residual) & ~exceptional
    if np.any(ordinary):
        value[ordinary] = _stable_product(
            0.5,
            confidence[ordinary],
            residual[ordinary],
            residual[ordinary],
        )
    finite_operands = np.logical_and.reduce(
        (
            np.isfinite(beta),
            np.isfinite(alpha),
            np.isfinite(left),
            np.isfinite(right),
            np.isfinite(target),
            np.isfinite(confidence),
        )
    )
    exact = finite_operands & (
        exceptional | ~np.isfinite(residual)
    )
    for flat_index in np.flatnonzero(exact):
        exact_residual = _exact_affine_fraction(
            float(beta.flat[flat_index]),
            float(alpha.flat[flat_index]),
            float(left.flat[flat_index]),
            float(right.flat[flat_index]),
            float(target.flat[flat_index]),
        )
        exact_value = (
            Fraction(1, 2)
            * _fraction(float(confidence.flat[flat_index]))
            * exact_residual
            * exact_residual
        )
        value.flat[flat_index] = _fraction_to_extended_float(exact_value)
    unresolved = ~ordinary & ~exact
    if np.any(unresolved):
        scaled = _stable_scaled_affine_residual(
            beta,
            alpha,
            left,
            right,
            target,
            np.sqrt(confidence),
            _SQRT_HALF,
            active=unresolved,
        )
        value[unresolved] = _stable_product(
            scaled[unresolved],
            scaled[unresolved],
        )
    return value


def _mismatch_terms(
    measurement: np.ndarray,
    target: np.ndarray,
    confidence: np.ndarray,
    mismatch: SquaredLoss | HuberLoss,
    *,
    evaluate_value: bool = True,
    evaluate_first: bool = True,
    evaluate_second: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return rowwise mismatch value and derivatives in one masked pass."""

    y, target_array, confidence_array = np.broadcast_arrays(
        np.asarray(measurement, dtype=np.float64),
        np.asarray(target, dtype=np.float64),
        np.asarray(confidence, dtype=np.float64),
    )
    value = np.zeros(y.shape, dtype=np.float64)
    first = np.zeros(y.shape, dtype=np.float64)
    second = np.zeros(y.shape, dtype=np.float64)
    active = confidence_array != 0.0
    if not np.any(active):
        return value, first, second

    active_y = y[active]
    active_target = target_array[active]
    active_confidence = confidence_array[active]
    if isinstance(mismatch, SquaredLoss):
        if evaluate_value:
            value[active] = _weighted_squared_difference_value(
                active_y,
                active_target,
                active_confidence,
            )
        if evaluate_first:
            first[active] = _stable_scaled_difference(
                active_y,
                active_target,
                active_confidence,
            )
        if evaluate_second:
            second[active] = active_confidence
        return value, first, second

    if not isinstance(mismatch, HuberLoss):
        raise TypeError(f'unsupported mismatch: {type(mismatch)!r}')

    delta = float(mismatch.delta)
    residual = _stable_scaled_difference(
        active_y,
        active_target,
        1.0,
    )
    quadratic, direction = _huber_difference_branches(
        active_y,
        active_target,
        delta,
    )
    linear = ~quadratic
    active_value = np.zeros_like(residual)
    active_first = np.zeros_like(residual)
    active_second = np.zeros_like(residual)
    if np.any(quadratic):
        if evaluate_value:
            active_value[quadratic] = _weighted_squared_difference_value(
                active_y[quadratic],
                active_target[quadratic],
                active_confidence[quadratic],
            )
        if evaluate_first:
            active_first[quadratic] = _stable_scaled_difference(
                active_y[quadratic],
                active_target[quadratic],
                active_confidence[quadratic],
            )
        if evaluate_second:
            active_second[quadratic] = active_confidence[quadratic]
    if np.any(linear):
        linear_residual = residual[linear]
        linear_y = active_y[linear]
        linear_target = active_target[linear]
        linear_confidence = active_confidence[linear]
        if evaluate_value:
            finite_residual = np.isfinite(linear_residual)
            linear_value = np.empty_like(linear_residual)
            if np.any(finite_residual):
                linear_value[finite_residual] = _stable_scaled_difference(
                    np.abs(linear_residual[finite_residual]),
                    0.5 * delta,
                    linear_confidence[finite_residual],
                    delta,
                )
            exceptional = ~finite_residual
            for flat_index in np.flatnonzero(exceptional):
                measurement_value = float(linear_y.flat[flat_index])
                target_value = float(linear_target.flat[flat_index])
                confidence_value = float(linear_confidence.flat[flat_index])
                if not (
                    math.isfinite(measurement_value)
                    and math.isfinite(target_value)
                ):
                    linear_value.flat[flat_index] = float('inf')
                    continue
                if measurement_value >= target_value:
                    positive, negative = measurement_value, target_value
                else:
                    positive, negative = target_value, measurement_value
                linear_value.flat[flat_index] = _exact_sum_products_scalar(
                    (
                        (confidence_value, delta, positive),
                        (-1.0, confidence_value, delta, negative),
                        (-0.5, confidence_value, delta, delta),
                    )
                )
            active_value[linear] = linear_value
        if evaluate_first:
            active_first[linear] = _stable_product(
                direction[linear],
                linear_confidence,
                delta,
            )
    value[active] = active_value
    first[active] = active_first
    second[active] = active_second
    return value, first, second


def _mismatch_values_from_affine(
    beta: np.ndarray,
    alpha: np.ndarray,
    left: np.ndarray,
    right: np.ndarray,
    target: np.ndarray,
    confidence: np.ndarray,
    mismatch: SquaredLoss | HuberLoss,
) -> np.ndarray:
    """Evaluate mismatch values without requiring representable residuals."""

    arrays = np.broadcast_arrays(
        np.asarray(beta, dtype=np.float64),
        np.asarray(alpha, dtype=np.float64),
        np.asarray(left, dtype=np.float64),
        np.asarray(right, dtype=np.float64),
        np.asarray(target, dtype=np.float64),
        np.asarray(confidence, dtype=np.float64),
    )
    (
        beta_array,
        alpha_array,
        left_array,
        right_array,
        target_array,
        confidence_array,
    ) = arrays
    value = np.zeros(beta_array.shape, dtype=np.float64)
    active = confidence_array != 0.0
    if not np.any(active):
        return value

    active_beta = beta_array[active]
    active_alpha = alpha_array[active]
    active_left = left_array[active]
    active_right = right_array[active]
    active_target = target_array[active]
    active_confidence = confidence_array[active]
    if isinstance(mismatch, SquaredLoss):
        value[active] = _weighted_squared_affine_value(
            active_beta,
            active_alpha,
            active_left,
            active_right,
            active_target,
            active_confidence,
        )
        return value
    if not isinstance(mismatch, HuberLoss):
        raise TypeError(f'unsupported mismatch: {type(mismatch)!r}')

    delta = float(mismatch.delta)
    residual = _stable_affine_residual(
        active_beta,
        active_alpha,
        active_left,
        active_right,
        active_target,
    )
    quadratic, direction = _huber_affine_branches(
        active_beta,
        active_alpha,
        active_left,
        active_right,
        active_target,
        delta,
    )
    active_value = np.empty(residual.shape, dtype=np.float64)
    if np.any(quadratic):
        active_value[quadratic] = _weighted_squared_affine_value(
            active_beta[quadratic],
            active_alpha[quadratic],
            active_left[quadratic],
            active_right[quadratic],
            active_target[quadratic],
            active_confidence[quadratic],
        )
    linear = ~quadratic
    if np.any(linear):
        linear_direction = direction[linear]
        linear_confidence = active_confidence[linear]
        active_value[linear] = _stable_sum_products(
            (
                (
                    linear_direction,
                    linear_confidence,
                    delta,
                    active_beta[linear],
                ),
                (
                    linear_direction,
                    linear_confidence,
                    delta,
                    active_alpha[linear],
                    active_left[linear],
                ),
                (
                    -linear_direction,
                    linear_confidence,
                    delta,
                    active_alpha[linear],
                    active_right[linear],
                ),
                (
                    -linear_direction,
                    linear_confidence,
                    delta,
                    active_target[linear],
                ),
                (-0.5, linear_confidence, delta, delta),
            )
        )
    value[active] = active_value
    return value


def _mismatch_values(
    measurement: np.ndarray,
    target: np.ndarray,
    confidence: np.ndarray,
    mismatch: SquaredLoss | HuberLoss,
) -> np.ndarray:
    return _mismatch_terms(
        measurement,
        target,
        confidence,
        mismatch,
        evaluate_first=False,
        evaluate_second=False,
    )[0]


def _mismatch_first_derivative(
    measurement: np.ndarray,
    target: np.ndarray,
    confidence: np.ndarray,
    mismatch: SquaredLoss | HuberLoss,
) -> np.ndarray:
    return _mismatch_terms(
        measurement,
        target,
        confidence,
        mismatch,
        evaluate_value=False,
        evaluate_second=False,
    )[1]


def _mismatch_second_derivative(
    measurement: np.ndarray,
    target: np.ndarray,
    confidence: np.ndarray,
    mismatch: SquaredLoss | HuberLoss,
) -> np.ndarray:
    return _mismatch_terms(
        measurement,
        target,
        confidence,
        mismatch,
        evaluate_value=False,
        evaluate_first=False,
    )[2]


def _reciprocal_masks(
    distance: np.ndarray,
    *,
    epsilon: float,
    margin: float,
) -> tuple[np.ndarray, np.ndarray]:
    continuation = distance <= epsilon
    reciprocal = (distance > epsilon) & (distance < margin)
    return continuation, reciprocal


def _reciprocal_boundary_value(
    distance: np.ndarray,
    *,
    margin: float,
    epsilon: float,
    strength: float,
) -> np.ndarray:
    result = np.zeros_like(distance)
    continuation, reciprocal = _reciprocal_masks(
        distance,
        epsilon=epsilon,
        margin=margin,
    )
    if np.any(continuation):
        continuation_distance = distance[continuation]
        base = _stable_ratio_product(
            (
                strength,
                _stable_scaled_difference(margin, epsilon, 1.0),
            ),
            (epsilon, margin),
        )
        correction = _stable_ratio_product(
            (
                strength,
                _stable_scaled_difference(
                    continuation_distance,
                    epsilon,
                    1.0,
                ),
            ),
            (epsilon, epsilon),
        )
        combined = _stable_sum(base, -correction)
        exceptional = ~np.isfinite(combined)
        for flat_index in np.flatnonzero(exceptional):
            d = float(continuation_distance.flat[flat_index])
            if not math.isfinite(d):
                combined.flat[flat_index] = float('inf')
                continue
            combined.flat[flat_index] = _exact_sum_ratios_scalar(
                (
                    ((strength, margin), (epsilon, margin)),
                    ((-strength, epsilon), (epsilon, margin)),
                    ((-strength, d), (epsilon, epsilon)),
                    ((strength, epsilon), (epsilon, epsilon)),
                )
            )
        result[continuation] = combined
    if np.any(reciprocal):
        reciprocal_distance = distance[reciprocal]
        result[reciprocal] = _stable_ratio_product(
            (
                strength,
                _stable_scaled_difference(
                    margin,
                    reciprocal_distance,
                    1.0,
                ),
            ),
            (reciprocal_distance, margin),
        )
    return result


def _reciprocal_boundary_value_from_operands(
    left: np.ndarray | float,
    right: np.ndarray | float,
    *,
    margin: float,
    epsilon: float,
    strength: float,
) -> np.ndarray:
    """Evaluate ``q(left - right)`` without requiring a finite difference."""

    return _reciprocal_boundary_terms_from_operands(
        left,
        right,
        margin=margin,
        epsilon=epsilon,
        strength=strength,
        evaluate_value=True,
        evaluate_first=False,
        evaluate_second=False,
    )[0]


def _reciprocal_boundary_first(
    distance: np.ndarray,
    *,
    margin: float,
    epsilon: float,
    strength: float,
) -> np.ndarray:
    result = np.zeros_like(distance)
    continuation, reciprocal = _reciprocal_masks(
        distance,
        epsilon=epsilon,
        margin=margin,
    )
    if np.any(continuation):
        result[continuation] = -_stable_ratio_product(
            (strength,),
            (epsilon, epsilon),
        )
    if np.any(reciprocal):
        d = distance[reciprocal]
        result[reciprocal] = -_stable_ratio_product(
            (strength,),
            (d, d),
        )
    return result


def _reciprocal_boundary_second(
    distance: np.ndarray,
    *,
    margin: float,
    epsilon: float,
    strength: float,
) -> np.ndarray:
    result = np.zeros_like(distance)
    _, reciprocal = _reciprocal_masks(
        distance,
        epsilon=epsilon,
        margin=margin,
    )
    if np.any(reciprocal):
        d = distance[reciprocal]
        result[reciprocal] = _stable_ratio_product(
            (2.0, strength),
            (d, d, d),
        )
    return result


def _reciprocal_branch_uncertain(
    distance: np.ndarray,
    expression_scale: np.ndarray,
    *,
    epsilon: float,
    margin: float,
    operation_count: int,
) -> np.ndarray:
    """Identify distances whose rounded value may cross a branch boundary."""

    scale = np.maximum(
        np.asarray(expression_scale, dtype=np.float64),
        max(abs(float(epsilon)), abs(float(margin))),
    )
    error_bound = _stable_product(
        scale,
        max(1, int(operation_count)) * _FLOAT_EPSILON,
    )
    uncertain = np.zeros(distance.shape, dtype=bool)
    for boundary in (epsilon, margin):
        gap = _stable_scaled_difference(distance, boundary, 1.0)
        uncertain |= np.abs(gap) <= error_bound
    return uncertain


def _exact_reciprocal_boundary_terms(
    distance: Fraction,
    *,
    margin: float,
    epsilon: float,
    strength: float,
    evaluate_value: bool,
    evaluate_first: bool,
    evaluate_second: bool,
) -> tuple[float, float, float]:
    """Evaluate one reciprocal branch exactly from binary64 operands."""

    margin_fraction = _fraction(margin)
    epsilon_fraction = _fraction(epsilon)
    strength_fraction = _fraction(strength)
    value = Fraction(0)
    first = Fraction(0)
    second = Fraction(0)
    if distance < margin_fraction:
        if distance > epsilon_fraction:
            if evaluate_value:
                value = strength_fraction * (
                    Fraction(1, 1) / distance
                    - Fraction(1, 1) / margin_fraction
                )
            if evaluate_first:
                first = -strength_fraction / (distance * distance)
            if evaluate_second:
                second = (
                    2 * strength_fraction
                    / (distance * distance * distance)
                )
        else:
            if evaluate_value:
                value = strength_fraction * (
                    Fraction(1, 1) / epsilon_fraction
                    - Fraction(1, 1) / margin_fraction
                    - (distance - epsilon_fraction)
                    / (epsilon_fraction * epsilon_fraction)
                )
            if evaluate_first:
                first = -strength_fraction / (
                    epsilon_fraction * epsilon_fraction
                )
    return (
        _fraction_to_extended_float(value) if evaluate_value else 0.0,
        _fraction_to_extended_float(first) if evaluate_first else 0.0,
        _fraction_to_extended_float(second) if evaluate_second else 0.0,
    )


def _reciprocal_boundary_terms_from_operands(
    left: np.ndarray | float,
    right: np.ndarray | float,
    *,
    margin: float,
    epsilon: float,
    strength: float,
    evaluate_value: bool,
    evaluate_first: bool,
    evaluate_second: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Evaluate reciprocal terms with branch-safe operand comparisons."""

    left_array, right_array = np.broadcast_arrays(
        np.asarray(left, dtype=np.float64),
        np.asarray(right, dtype=np.float64),
    )
    distance = _stable_scaled_difference(left_array, right_array, 1.0)
    value = (
        _reciprocal_boundary_value(
            distance,
            margin=margin,
            epsilon=epsilon,
            strength=strength,
        )
        if evaluate_value
        else np.zeros_like(distance)
    )
    first = (
        _reciprocal_boundary_first(
            distance,
            margin=margin,
            epsilon=epsilon,
            strength=strength,
        )
        if evaluate_first
        else np.zeros_like(distance)
    )
    second = (
        _reciprocal_boundary_second(
            distance,
            margin=margin,
            epsilon=epsilon,
            strength=strength,
        )
        if evaluate_second
        else np.zeros_like(distance)
    )
    finite_operands = np.isfinite(left_array) & np.isfinite(right_array)
    expression_scale = np.maximum(
        np.maximum(np.abs(left_array), np.abs(right_array)),
        np.abs(distance),
    )
    exceptional = finite_operands & (
        ~np.isfinite(distance)
        | _reciprocal_branch_uncertain(
            distance,
            expression_scale,
            epsilon=epsilon,
            margin=margin,
            operation_count=3,
        )
    )
    for flat_index in np.flatnonzero(exceptional):
        exact_distance = (
            _fraction(float(left_array.flat[flat_index]))
            - _fraction(float(right_array.flat[flat_index]))
        )
        exact_value, exact_first, exact_second = (
            _exact_reciprocal_boundary_terms(
                exact_distance,
                margin=margin,
                epsilon=epsilon,
                strength=strength,
                evaluate_value=evaluate_value,
                evaluate_first=evaluate_first,
                evaluate_second=evaluate_second,
            )
        )
        if evaluate_value:
            value.flat[flat_index] = exact_value
        if evaluate_first:
            first.flat[flat_index] = exact_first
        if evaluate_second:
            second.flat[flat_index] = exact_second
    return value, first, second


def _penalty_value(
    measurement: np.ndarray,
    penalty: ScalarPenalty,
) -> np.ndarray:
    """Return the authoritative rowwise value for one scalar penalty."""

    y = np.asarray(measurement, dtype=np.float64)
    strength = float(penalty.strength)
    if strength == 0.0:
        return np.zeros_like(y)

    if isinstance(penalty, SoftIntervalPenalty):
        result = np.zeros_like(y)
        lower_mask = y < float(penalty.lower)
        upper_mask = y > float(penalty.upper)
        for mask, boundary in (
            (lower_mask, float(penalty.lower)),
            (upper_mask, float(penalty.upper)),
        ):
            if np.any(mask):
                scaled_displacement = _stable_scaled_difference(
                    y[mask],
                    boundary,
                    math.sqrt(strength),
                )
                result[mask] = _stable_product(
                    scaled_displacement,
                    scaled_displacement,
                )
        return result

    if isinstance(penalty, ExponentialBoundaryPenalty):
        left = float(penalty.lower) + float(penalty.margin)
        right = float(penalty.upper) - float(penalty.margin)
        tau = float(penalty.tau)
        with np.errstate(over='ignore', invalid='ignore'):
            return strength * (
                np.exp((left - y) / tau)
                + np.exp((y - right) / tau)
            )

    if isinstance(penalty, ReciprocalBoundaryPenalty):
        lower_value = _reciprocal_boundary_value_from_operands(
            y,
            float(penalty.lower),
            margin=float(penalty.margin),
            epsilon=float(penalty.epsilon),
            strength=strength,
        )
        upper_value = _reciprocal_boundary_value_from_operands(
            float(penalty.upper),
            y,
            margin=float(penalty.margin),
            epsilon=float(penalty.epsilon),
            strength=strength,
        )
        return _stable_sum(lower_value, upper_value)

    raise TypeError(f'unsupported penalty: {type(penalty)!r}')


def _fraction_to_hard_endpoint_float(value: Fraction) -> float:
    """Round a hard-bound endpoint, saturating only to finite float64."""

    if value > _FLOAT_MAX_FRACTION:
        return np.finfo(np.float64).max
    if value < -_FLOAT_MAX_FRACTION:
        return -np.finfo(np.float64).max
    return float(value)


def _fraction_to_extended_float(value: Fraction) -> float:
    """Round an objective value, preserving true range overflow as infinity."""

    try:
        return float(value)
    except OverflowError:
        return float('-inf') if value < 0 else float('inf')


def _exact_affine_fraction(
    beta: float,
    alpha: float,
    left: float,
    right: float,
    target: float,
) -> Fraction:
    """Return an exact affine residual for finite binary64 operands."""

    return (
        _fraction(beta)
        + _fraction(alpha) * _fraction(left)
        - _fraction(alpha) * _fraction(right)
        - _fraction(target)
    )


def _reciprocal_boundary_value_from_affine(
    beta: np.ndarray,
    alpha: np.ndarray,
    left: np.ndarray,
    right: np.ndarray,
    boundary: float,
    *,
    direction: float,
    margin: float,
    epsilon: float,
    strength: float,
) -> np.ndarray:
    """Evaluate reciprocal ``q`` from a complete affine boundary distance."""

    arrays = np.broadcast_arrays(
        np.asarray(beta, dtype=np.float64),
        np.asarray(alpha, dtype=np.float64),
        np.asarray(left, dtype=np.float64),
        np.asarray(right, dtype=np.float64),
    )
    beta_array, alpha_array, left_array, right_array = arrays
    distance = _stable_affine_residual(
        beta_array,
        alpha_array,
        left_array,
        right_array,
        boundary,
    )
    if direction < 0.0:
        distance = _stable_product(-1.0, distance)
    result = _reciprocal_boundary_value(
        distance,
        margin=margin,
        epsilon=epsilon,
        strength=strength,
    )
    finite_operands = np.logical_and.reduce(
        tuple(np.isfinite(array) for array in arrays)
    )
    alpha_left = _stable_product(alpha_array, left_array)
    alpha_right = _stable_product(alpha_array, right_array)
    expression_scale = np.maximum.reduce(
        (
            np.abs(beta_array),
            np.abs(alpha_left),
            np.abs(alpha_right),
            np.full(distance.shape, abs(float(boundary))),
            np.abs(distance),
        )
    )
    exceptional = finite_operands & (
        ~np.isfinite(distance)
        | _reciprocal_branch_uncertain(
            distance,
            expression_scale,
            epsilon=epsilon,
            margin=margin,
            operation_count=8,
        )
    )
    for flat_index in np.flatnonzero(exceptional):
        exact_distance = _exact_affine_fraction(
            float(beta_array.flat[flat_index]),
            float(alpha_array.flat[flat_index]),
            float(left_array.flat[flat_index]),
            float(right_array.flat[flat_index]),
            boundary,
        )
        if direction < 0.0:
            exact_distance = -exact_distance
        result.flat[flat_index] = _exact_reciprocal_boundary_terms(
            exact_distance,
            margin=margin,
            epsilon=epsilon,
            strength=strength,
            evaluate_value=True,
            evaluate_first=False,
            evaluate_second=False,
        )[0]
    return result


def _penalty_value_from_affine(
    beta: np.ndarray,
    alpha: np.ndarray,
    left: np.ndarray,
    right: np.ndarray,
    measurement: np.ndarray,
    penalty: ScalarPenalty,
) -> np.ndarray:
    """Evaluate a penalty without requiring a representable prediction."""

    y = np.asarray(measurement, dtype=np.float64)
    strength = float(penalty.strength)
    if strength == 0.0:
        return np.zeros_like(y)

    if isinstance(penalty, SoftIntervalPenalty):
        lower = float(penalty.lower)
        upper = float(penalty.upper)
        lower_distance = _stable_affine_residual(
            beta,
            alpha,
            left,
            right,
            lower,
        )
        upper_distance = _stable_affine_residual(
            beta,
            alpha,
            left,
            right,
            upper,
        )
        lower_mask = lower_distance < 0.0
        upper_mask = upper_distance > 0.0
        result = np.zeros_like(y)
        scale = math.sqrt(strength)
        if np.any(lower_mask):
            scaled = _stable_scaled_affine_residual(
                beta,
                alpha,
                left,
                right,
                lower,
                scale,
                active=lower_mask,
            )
            result[lower_mask] = _stable_product(
                scaled[lower_mask],
                scaled[lower_mask],
            )
        if np.any(upper_mask):
            scaled = _stable_scaled_affine_residual(
                beta,
                alpha,
                left,
                right,
                upper,
                scale,
                active=upper_mask,
            )
            result[upper_mask] = _stable_product(
                scaled[upper_mask],
                scaled[upper_mask],
            )
        return result

    if isinstance(penalty, ReciprocalBoundaryPenalty):
        lower_value = _reciprocal_boundary_value_from_affine(
            beta,
            alpha,
            left,
            right,
            float(penalty.lower),
            direction=1.0,
            margin=float(penalty.margin),
            epsilon=float(penalty.epsilon),
            strength=strength,
        )
        upper_value = _reciprocal_boundary_value_from_affine(
            beta,
            alpha,
            left,
            right,
            float(penalty.upper),
            direction=-1.0,
            margin=float(penalty.margin),
            epsilon=float(penalty.epsilon),
            strength=strength,
        )
        return _stable_sum(lower_value, upper_value)

    return _penalty_value(y, penalty)


def _penalty_first_derivative(
    measurement: np.ndarray,
    penalty: ScalarPenalty,
) -> np.ndarray:
    """Return the authoritative rowwise first derivative for one penalty."""

    y = np.asarray(measurement, dtype=np.float64)
    strength = float(penalty.strength)
    if strength == 0.0:
        return np.zeros_like(y)

    if isinstance(penalty, SoftIntervalPenalty):
        result = np.zeros_like(y)
        lower_mask = y < float(penalty.lower)
        upper_mask = y > float(penalty.upper)
        if np.any(lower_mask):
            result[lower_mask] = _stable_scaled_difference(
                y[lower_mask],
                float(penalty.lower),
                2.0,
                strength,
            )
        if np.any(upper_mask):
            result[upper_mask] = _stable_scaled_difference(
                y[upper_mask],
                float(penalty.upper),
                2.0,
                strength,
            )
        return result

    if isinstance(penalty, ExponentialBoundaryPenalty):
        left = float(penalty.lower) + float(penalty.margin)
        right = float(penalty.upper) - float(penalty.margin)
        tau = float(penalty.tau)
        with np.errstate(over='ignore', invalid='ignore'):
            lower_term = np.exp((left - y) / tau)
            upper_term = np.exp((y - right) / tau)
            return strength * (-lower_term + upper_term) / tau

    if isinstance(penalty, ReciprocalBoundaryPenalty):
        lower_first = _reciprocal_boundary_terms_from_operands(
            y,
            float(penalty.lower),
            margin=float(penalty.margin),
            epsilon=float(penalty.epsilon),
            strength=strength,
            evaluate_value=False,
            evaluate_first=True,
            evaluate_second=False,
        )[1]
        upper_first = _reciprocal_boundary_terms_from_operands(
            float(penalty.upper),
            y,
            margin=float(penalty.margin),
            epsilon=float(penalty.epsilon),
            strength=strength,
            evaluate_value=False,
            evaluate_first=True,
            evaluate_second=False,
        )[1]
        return _stable_sum(lower_first, -upper_first)

    raise TypeError(f'unsupported penalty: {type(penalty)!r}')


def _penalty_second_derivative(
    measurement: np.ndarray,
    penalty: ScalarPenalty,
) -> np.ndarray:
    """Return the authoritative rowwise second derivative for one penalty."""

    y = np.asarray(measurement, dtype=np.float64)
    strength = float(penalty.strength)
    if strength == 0.0:
        return np.zeros_like(y)

    if isinstance(penalty, SoftIntervalPenalty):
        result = np.zeros_like(y)
        result[(y < float(penalty.lower)) | (y > float(penalty.upper))] = (
            2.0 * strength
        )
        return result

    if isinstance(penalty, ExponentialBoundaryPenalty):
        left = float(penalty.lower) + float(penalty.margin)
        right = float(penalty.upper) - float(penalty.margin)
        tau = float(penalty.tau)
        with np.errstate(over='ignore', invalid='ignore'):
            lower_term = np.exp((left - y) / tau)
            upper_term = np.exp((y - right) / tau)
            return strength * (lower_term + upper_term) / (tau * tau)

    if isinstance(penalty, ReciprocalBoundaryPenalty):
        lower_second = _reciprocal_boundary_terms_from_operands(
            y,
            float(penalty.lower),
            margin=float(penalty.margin),
            epsilon=float(penalty.epsilon),
            strength=strength,
            evaluate_value=False,
            evaluate_first=False,
            evaluate_second=True,
        )[2]
        upper_second = _reciprocal_boundary_terms_from_operands(
            float(penalty.upper),
            y,
            margin=float(penalty.margin),
            epsilon=float(penalty.epsilon),
            strength=strength,
            evaluate_value=False,
            evaluate_first=False,
            evaluate_second=True,
        )[2]
        return _stable_sum(lower_second, upper_second)

    raise TypeError(f'unsupported penalty: {type(penalty)!r}')


def _penalty_terms(
    measurement: np.ndarray,
    penalty: ScalarPenalty,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return authoritative rowwise value and derivatives for one penalty."""

    return (
        _penalty_value(measurement, penalty),
        _penalty_first_derivative(measurement, penalty),
        _penalty_second_derivative(measurement, penalty),
    )


def _l2_value(
    weights: np.ndarray,
    reference: np.ndarray,
    strength: float,
) -> float:
    """Return ``0.5 * strength * ||weights - reference||**2`` stably."""

    strength_value = float(strength)
    if strength_value == 0.0:
        return 0.0
    scale = math.sqrt(strength_value) * math.sqrt(0.5)
    scaled_difference = _stable_scaled_difference(
        weights,
        reference,
        scale,
    )
    return _stable_sum_squares(scaled_difference)


def _hard_tolerance(*values: np.ndarray | float) -> np.ndarray:
    """Return the shared float64 absolute-plus-relative hard tolerance."""

    arrays = np.broadcast_arrays(
        *(np.asarray(value, dtype=np.float64) for value in values)
    )
    scale = np.zeros_like(arrays[0], dtype=np.float64)
    for value in arrays:
        scale = np.maximum(scale, np.abs(value))
    relative = _stable_product(HARD_RTOL, scale)
    return _stable_sum(HARD_ATOL, relative)


def _hard_row_status(
    lower: np.ndarray,
    measurement: np.ndarray,
    upper: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return authoritative row satisfaction, raw violation, and tolerance."""

    lower_array, measurement_array, upper_array = np.broadcast_arrays(
        np.asarray(lower, dtype=np.float64),
        np.asarray(measurement, dtype=np.float64),
        np.asarray(upper, dtype=np.float64),
    )
    lower_gap = _stable_scaled_difference(
        lower_array,
        measurement_array,
        1.0,
    )
    upper_gap = _stable_scaled_difference(
        measurement_array,
        upper_array,
        1.0,
    )
    violation = np.maximum(np.maximum(lower_gap, upper_gap), 0.0)
    tolerance = _hard_tolerance(
        lower_array,
        measurement_array,
        upper_array,
    )
    finite = (
        np.isfinite(lower_array)
        & np.isfinite(measurement_array)
        & np.isfinite(upper_array)
    )
    satisfied = finite & (violation <= tolerance)
    return satisfied, violation, tolerance


def _hard_row_satisfied_scalar(
    lower: float,
    measurement: float,
    upper: float,
) -> bool:
    if not (
        math.isfinite(lower)
        and math.isfinite(measurement)
        and math.isfinite(upper)
    ):
        return False
    violation = max(
        _stable_scaled_difference_scalar(lower, measurement, 1.0),
        _stable_scaled_difference_scalar(measurement, upper, 1.0),
        0.0,
    )
    relative = _stable_product_scalar(
        HARD_RTOL,
        max(
            abs(lower),
            abs(measurement),
            abs(upper),
        ),
    )
    tolerance = _stable_sum_scalar(HARD_ATOL, relative)
    return violation <= tolerance


def _tight_lower_accepted_bound(
    lower: float,
    upper: float,
    candidate: float,
) -> float:
    value = (
        -np.finfo(np.float64).max
        if candidate == float('-inf')
        else candidate
    )
    while not _hard_row_satisfied_scalar(lower, value, upper):
        value = math.nextafter(value, float('inf'))
    while True:
        previous = math.nextafter(value, float('-inf'))
        if not _hard_row_satisfied_scalar(lower, previous, upper):
            return value
        value = previous


def _tight_upper_accepted_bound(
    lower: float,
    upper: float,
    candidate: float,
) -> float:
    value = (
        np.finfo(np.float64).max
        if candidate == float('inf')
        else candidate
    )
    while not _hard_row_satisfied_scalar(lower, value, upper):
        value = math.nextafter(value, float('-inf'))
    while True:
        following = math.nextafter(value, float('inf'))
        if not _hard_row_satisfied_scalar(lower, following, upper):
            return value
        value = following


def _hard_accepted_measurement_bounds(
    lower: np.ndarray,
    upper: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return the exact finite interval accepted by the public row predicate."""

    lower_array, upper_array = np.broadcast_arrays(
        np.asarray(lower, dtype=np.float64),
        np.asarray(upper, dtype=np.float64),
    )
    accepted_lower = np.empty(lower_array.shape, dtype=np.float64)
    accepted_upper = np.empty(upper_array.shape, dtype=np.float64)
    atol_fraction = _fraction(HARD_ATOL)
    rtol_fraction = _fraction(HARD_RTOL)
    one_minus_rtol = Fraction(1) - rtol_fraction
    for index in np.ndindex(lower_array.shape):
        lower_value = float(lower_array[index])
        upper_value = float(upper_array[index])
        if not (math.isfinite(lower_value) and math.isfinite(upper_value)):
            accepted_lower[index] = float('nan')
            accepted_upper[index] = float('nan')
            continue
        scale = max(abs(lower_value), abs(upper_value))
        scale_fraction = _fraction(scale)
        lower_fraction = _fraction(lower_value)
        upper_fraction = _fraction(upper_value)
        fixed_tolerance = atol_fraction + rtol_fraction * scale_fraction
        lower_candidate = lower_fraction - fixed_tolerance
        if abs(lower_candidate) <= scale_fraction:
            approximate_lower_fraction = lower_candidate
        else:
            approximate_lower_fraction = (
                lower_fraction - atol_fraction
            ) / one_minus_rtol
        upper_candidate = upper_fraction + fixed_tolerance
        if abs(upper_candidate) <= scale_fraction:
            approximate_upper_fraction = upper_candidate
        else:
            approximate_upper_fraction = (
                upper_fraction + atol_fraction
            ) / one_minus_rtol
        approximate_lower = _fraction_to_hard_endpoint_float(
            approximate_lower_fraction
        )
        approximate_upper = _fraction_to_hard_endpoint_float(
            approximate_upper_fraction
        )
        accepted_lower[index] = _tight_lower_accepted_bound(
            lower_value,
            upper_value,
            approximate_lower,
        )
        accepted_upper[index] = _tight_upper_accepted_bound(
            lower_value,
            upper_value,
            approximate_upper,
        )
    return accepted_lower, accepted_upper
