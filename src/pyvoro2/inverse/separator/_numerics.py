"""Scale-safe private arithmetic for separator inverse computations."""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from fractions import Fraction

import numpy as np


_FLOAT_MAX = np.finfo(np.float64).max
_FLOAT_EPSILON = np.finfo(np.float64).eps
_CONDITIONING_RELATIVE_LIMIT = math.sqrt(_FLOAT_EPSILON)
_AFFINE_CONDITIONING_RELATIVE_LIMIT = 64.0 * _FLOAT_EPSILON
_NORMAL_MIN_EXPONENT = -1021
_NORMAL_MAX_EXPONENT = 1023
_SPLITTER = 134217729.0


def _stable_product_scalar(*factors: float) -> float:
    """Multiply finite scalars without avoidable intermediate range loss."""

    values = tuple(float(value) for value in factors)
    if any(math.isnan(value) for value in values):
        return float('nan')
    if any(value == 0.0 for value in values):
        return 0.0

    negative = sum(value < 0.0 for value in values) % 2
    if any(math.isinf(value) for value in values):
        return float('-inf') if negative else float('inf')

    mantissa = -1.0 if negative else 1.0
    exponent = 0
    for value in values:
        part, part_exponent = math.frexp(abs(value))
        mantissa *= part
        exponent += part_exponent
        mantissa, normalization = math.frexp(mantissa)
        exponent += normalization
    try:
        return math.ldexp(mantissa, exponent)
    except OverflowError:
        return float('-inf') if negative else float('inf')


def _stable_ratio_product_scalar(
    numerators: Iterable[float],
    denominators: Iterable[float],
) -> float:
    """Evaluate a product ratio without first forming reciprocal powers."""

    numerator_values = tuple(float(value) for value in numerators)
    denominator_values = tuple(float(value) for value in denominators)
    values = numerator_values + denominator_values
    if any(math.isnan(value) for value in values):
        return float('nan')
    if any(value == 0.0 for value in denominator_values):
        if any(value == 0.0 for value in numerator_values):
            return float('nan')
        negative = (
            sum(value < 0.0 for value in numerator_values)
            + sum(value < 0.0 for value in denominator_values)
        ) % 2
        return float('-inf') if negative else float('inf')
    if any(value == 0.0 for value in numerator_values):
        return 0.0

    negative = (
        sum(value < 0.0 for value in numerator_values)
        + sum(value < 0.0 for value in denominator_values)
    ) % 2
    mantissa = -1.0 if negative else 1.0
    exponent = 0
    for value in numerator_values:
        part, part_exponent = math.frexp(abs(value))
        mantissa *= part
        exponent += part_exponent
        mantissa, normalization = math.frexp(mantissa)
        exponent += normalization
    for value in denominator_values:
        part, part_exponent = math.frexp(abs(value))
        mantissa /= part
        exponent -= part_exponent
        mantissa, normalization = math.frexp(mantissa)
        exponent += normalization
    try:
        return math.ldexp(mantissa, exponent)
    except OverflowError:
        return float('-inf') if negative else float('inf')


def _stable_sum_scalar(*values: float) -> float:
    """Sum scalars accurately, returning infinity only on true range overflow."""

    floats = tuple(float(value) for value in values)
    if any(math.isnan(value) for value in floats):
        return float('nan')
    try:
        return float(math.fsum(floats))
    except (OverflowError, ValueError):
        if all(math.isfinite(value) for value in floats):
            total = sum((_fraction(value) for value in floats), Fraction(0))
            try:
                return float(total)
            except OverflowError:
                return float('-inf') if total < 0 else float('inf')
        positive = any(value == float('inf') for value in floats)
        negative = any(value == float('-inf') for value in floats)
        if positive and negative:
            return float('nan')
        if positive:
            return float('inf')
        if negative:
            return float('-inf')
        return float('-inf') if all(value <= 0.0 for value in floats) else float('inf')


def _fraction(value: float) -> Fraction:
    numerator, denominator = float(value).as_integer_ratio()
    return Fraction(numerator, denominator)


def _exact_sum_products_scalar(
    products: Sequence[Sequence[float]],
) -> float:
    """Evaluate a finite sum of finite products exactly before float rounding."""

    total = Fraction(0)
    for factors in products:
        term = Fraction(1)
        for factor in factors:
            value = float(factor)
            if not math.isfinite(value):
                return _stable_sum_scalar(
                    *(
                        _stable_product_scalar(*product)
                        for product in products
                    )
                )
            term *= _fraction(value)
        total += term
    try:
        return float(total)
    except OverflowError:
        return float('-inf') if total < 0 else float('inf')


def _exact_sum_products_sign_scalar(
    products: Sequence[Sequence[float]],
) -> int:
    """Return the exact sign of a finite sum of binary64 products."""

    total = Fraction(0)
    for factors in products:
        term = Fraction(1)
        for factor in factors:
            value = float(factor)
            if not math.isfinite(value):
                approximate = _exact_sum_products_scalar(products)
                return int(approximate > 0.0) - int(approximate < 0.0)
            term *= _fraction(value)
        total += term
    return int(total > 0) - int(total < 0)


def _exact_sum_ratios_scalar(
    terms: Sequence[tuple[Sequence[float], Sequence[float]]],
) -> float:
    """Evaluate a finite sum of finite product ratios exactly."""

    total = Fraction(0)
    for numerators, denominators in terms:
        term = Fraction(1)
        for numerator in numerators:
            term *= _fraction(float(numerator))
        for denominator in denominators:
            term /= _fraction(float(denominator))
        total += term
    try:
        return float(total)
    except OverflowError:
        return float('-inf') if total < 0 else float('inf')


def _exact_weighted_average_scalar(
    left: float,
    left_weight: float,
    right: float,
    right_weight: float,
) -> float:
    """Evaluate a finite two-term weighted average exactly before rounding."""

    denominator = _fraction(left_weight) + _fraction(right_weight)
    if denominator == 0:
        return float('nan')
    value = (
        _fraction(left_weight) * _fraction(left)
        + _fraction(right_weight) * _fraction(right)
    ) / denominator
    try:
        return float(value)
    except OverflowError:
        return float('-inf') if value < 0 else float('inf')


def _same_sign(left: float, right: float) -> bool:
    return (left >= 0.0 and right >= 0.0) or (
        left <= 0.0 and right <= 0.0
    )


def _stable_scaled_difference_scalar(
    left: float,
    right: float,
    *scale_factors: float,
) -> float:
    """Evaluate ``prod(scale_factors) * (left - right)`` stably."""

    scales = tuple(float(value) for value in scale_factors)
    if any(value == 0.0 for value in scales):
        return 0.0
    left_value = float(left)
    right_value = float(right)
    if (
        math.isfinite(left_value)
        and math.isfinite(right_value)
        and all(math.isfinite(value) for value in scales)
    ):
        return _exact_sum_products_scalar(
            (
                scales + (left_value,),
                (-1.0,) + scales + (right_value,),
            )
        )
    if _same_sign(left_value, right_value):
        return _stable_product_scalar(
            *scales,
            left_value - right_value,
        )
    return _stable_sum_scalar(
        _stable_product_scalar(*scales, left_value),
        -_stable_product_scalar(*scales, right_value),
    )


def _stable_ratio_difference_scalar(
    left: float,
    right: float,
    denominator: float,
) -> float:
    """Evaluate ``(left - right) / denominator`` stably."""

    left_value = float(left)
    right_value = float(right)
    if _same_sign(left_value, right_value):
        return _stable_ratio_product_scalar(
            (left_value - right_value,),
            (denominator,),
        )
    return _stable_sum_scalar(
        _stable_ratio_product_scalar((left_value,), (denominator,)),
        -_stable_ratio_product_scalar((right_value,), (denominator,)),
    )


def _can_add(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    finite = np.isfinite(left) & np.isfinite(right)
    opposite = np.signbit(left) != np.signbit(right)
    return finite & (
        opposite
        | (np.abs(left) <= _FLOAT_MAX - np.abs(right))
    )


def _can_subtract(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    finite = np.isfinite(left) & np.isfinite(right)
    same = np.signbit(left) == np.signbit(right)
    return finite & (
        same
        | (np.abs(left) <= _FLOAT_MAX - np.abs(right))
    )


def _can_multiply(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    finite = np.isfinite(left) & np.isfinite(right)
    left_absolute = np.abs(left)
    right_absolute = np.abs(right)
    large_left = left_absolute > 1.0
    threshold = np.full(left.shape, _FLOAT_MAX, dtype=np.float64)
    threshold[large_left] = _FLOAT_MAX / left_absolute[large_left]
    range_safe = finite & (
        (left_absolute == 0.0)
        | (right_absolute == 0.0)
        | ~large_left
        | (right_absolute <= threshold)
    )
    nonzero = (
        finite
        & (left_absolute != 0.0)
        & (right_absolute != 0.0)
    )
    underflow_risk = np.zeros(left.shape, dtype=bool)
    if np.any(nonzero):
        left_part, left_exponent = np.frexp(left_absolute[nonzero])
        right_part, right_exponent = np.frexp(right_absolute[nonzero])
        _, normalization = np.frexp(left_part * right_part)
        product_exponent = (
            left_exponent
            + right_exponent
            + normalization
        )
        underflow_risk[nonzero] = (
            product_exponent < _NORMAL_MIN_EXPONENT
        )
    return range_safe & ~underflow_risk


def _direct_product(
    arrays: Sequence[np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    result = np.ones(arrays[0].shape, dtype=np.float64)
    safe = np.ones(result.shape, dtype=bool)
    for array in arrays:
        step_safe = _can_multiply(result, array)
        active = safe & step_safe
        result[active] *= array[active]
        safe &= step_safe
    return result, safe


def _normal_product_parts(
    arrays: Sequence[np.ndarray],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    shape = arrays[0].shape
    mantissa = np.ones(shape, dtype=np.float64)
    exponent = np.zeros(shape, dtype=np.int64)
    finite_nonzero = np.ones(shape, dtype=bool)
    for array in arrays:
        finite_nonzero &= np.isfinite(array) & (array != 0.0)
        part, part_exponent = np.frexp(array)
        mantissa *= part
        exponent += part_exponent.astype(np.int64)
        mantissa, normalization = np.frexp(mantissa)
        exponent += normalization.astype(np.int64)
    normal = (
        finite_nonzero
        & (exponent >= _NORMAL_MIN_EXPONENT)
        & (exponent <= _NORMAL_MAX_EXPONENT)
    )
    return mantissa, exponent, normal


def _stable_product(*factors: object) -> np.ndarray:
    """Vectorized product with scalar fallbacks only for exceptional rows."""

    arrays = np.broadcast_arrays(
        *(np.asarray(value, dtype=np.float64) for value in factors)
    )
    if not arrays:
        return np.asarray(1.0, dtype=np.float64)
    result, direct = _direct_product(arrays)
    if np.all(direct):
        return result
    zero = np.logical_or.reduce(
        tuple(array == 0.0 for array in arrays)
    )
    result[zero] = 0.0
    exceptional = ~direct & ~zero
    exceptional_arrays = tuple(array[exceptional] for array in arrays)
    mantissa, exponent, normal = _normal_product_parts(exceptional_arrays)
    exceptional_result = np.empty(mantissa.shape, dtype=np.float64)
    if np.any(normal):
        exceptional_result[normal] = np.ldexp(
            mantissa[normal],
            exponent[normal],
        )
    for flat_index in np.flatnonzero(~normal):
        exceptional_result.flat[flat_index] = _stable_product_scalar(
            *(
                float(array.flat[flat_index])
                for array in exceptional_arrays
            )
        )
    result[exceptional] = exceptional_result
    for flat_index in np.flatnonzero(~direct & zero & np.logical_or.reduce(
        tuple(np.isnan(array) for array in arrays)
    )):
        result.flat[flat_index] = _stable_product_scalar(
            *(float(array.flat[flat_index]) for array in arrays)
        )
    return result


def _stable_ratio_product(
    numerators: Sequence[object],
    denominators: Sequence[object],
) -> np.ndarray:
    """Vectorized product ratio with exceptional scalar fallbacks."""

    arrays = np.broadcast_arrays(
        *(
            np.asarray(value, dtype=np.float64)
            for value in (*numerators, *denominators)
        )
    )
    numerator_arrays = arrays[:len(numerators)]
    denominator_arrays = arrays[len(numerators):]
    shape = arrays[0].shape
    mantissa = np.ones(shape, dtype=np.float64)
    exponent = np.zeros(shape, dtype=np.int64)
    regular = np.ones(shape, dtype=bool)
    numerator_zero = np.zeros(shape, dtype=bool)
    for array in numerator_arrays:
        regular &= np.isfinite(array) & (array != 0.0)
        numerator_zero |= array == 0.0
        part, part_exponent = np.frexp(array)
        mantissa *= part
        exponent += part_exponent.astype(np.int64)
        mantissa, normalization = np.frexp(mantissa)
        exponent += normalization.astype(np.int64)
    for array in denominator_arrays:
        regular &= np.isfinite(array) & (array != 0.0)
        part, part_exponent = np.frexp(array)
        mantissa /= np.where(array == 0.0, 1.0, part)
        exponent -= part_exponent.astype(np.int64)
        mantissa, normalization = np.frexp(mantissa)
        exponent += normalization.astype(np.int64)
    normal = (
        regular
        & (exponent >= _NORMAL_MIN_EXPONENT)
        & (exponent <= _NORMAL_MAX_EXPONENT)
    )
    result = np.zeros(shape, dtype=np.float64)
    if np.any(normal):
        result[normal] = np.ldexp(mantissa[normal], exponent[normal])
    exceptional = ~normal & ~(
        numerator_zero
        & np.logical_and.reduce(
            tuple(array != 0.0 for array in denominator_arrays)
        )
    )
    for flat_index in np.flatnonzero(exceptional):
        result.flat[flat_index] = _stable_ratio_product_scalar(
            tuple(float(array.flat[flat_index]) for array in numerator_arrays),
            tuple(float(array.flat[flat_index]) for array in denominator_arrays),
        )
    return result


def _stable_normalized_ratio(
    numerator: np.ndarray,
    denominator: np.ndarray,
    *,
    active: np.ndarray,
) -> np.ndarray:
    """Return nonnegative normalized ratios without unsafe subnormal division."""

    numerator_array, denominator_array, active_array = np.broadcast_arrays(
        np.asarray(numerator, dtype=np.float64),
        np.asarray(denominator, dtype=np.float64),
        np.asarray(active, dtype=bool),
    )
    result = np.zeros(numerator_array.shape, dtype=np.float64)
    material = (
        active_array
        & (numerator_array != 0.0)
        & (denominator_array != 0.0)
    )
    if not np.any(material):
        return result

    numerator_part, numerator_exponent = np.frexp(
        numerator_array[material]
    )
    denominator_part, denominator_exponent = np.frexp(
        denominator_array[material]
    )
    ratio_part, normalization = np.frexp(
        numerator_part / denominator_part
    )
    ratio_exponent = (
        numerator_exponent
        - denominator_exponent
        + normalization
    )
    normal = (
        (ratio_exponent >= _NORMAL_MIN_EXPONENT)
        & (ratio_exponent <= _NORMAL_MAX_EXPONENT)
    )
    material_result = np.empty(ratio_part.shape, dtype=np.float64)
    if np.any(normal):
        material_result[normal] = np.ldexp(
            ratio_part[normal],
            ratio_exponent[normal],
        )
    for flat_index in np.flatnonzero(~normal):
        material_result.flat[flat_index] = _stable_ratio_product_scalar(
            (float(numerator_array[material].flat[flat_index]),),
            (float(denominator_array[material].flat[flat_index]),),
        )
    result[material] = material_result
    return result


def _cancellation_error_bound(
    result: np.ndarray,
    terms: Sequence[np.ndarray],
    *,
    operation_count: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return mixed-sign rows and their first-order forward-error bound."""

    finite_terms = np.logical_and.reduce(
        tuple(np.isfinite(term) for term in terms)
    )
    positive = np.logical_or.reduce(tuple(term > 0.0 for term in terms))
    negative = np.logical_or.reduce(tuple(term < 0.0 for term in terms))
    mixed = finite_terms & positive & negative
    if not np.any(mixed):
        return mixed, np.zeros(result.shape, dtype=np.float64)

    absolute_terms = tuple(np.abs(term) for term in terms)
    scale = np.maximum.reduce(absolute_terms)
    scaled_l1 = np.zeros(result.shape, dtype=np.float64)
    for absolute in absolute_terms:
        material = (
            mixed
            & (absolute != 0.0)
            & (scale != 0.0)
        )
        contribution = _stable_normalized_ratio(
            absolute,
            scale,
            active=material,
        )
        scaled_l1 += contribution
    error_bound = _stable_product(
        scale,
        scaled_l1,
        max(1, int(operation_count)) * _FLOAT_EPSILON,
    )
    return mixed, error_bound


def _cancellation_risk(
    result: np.ndarray,
    terms: Sequence[np.ndarray],
    *,
    operation_count: int,
    relative_limit: float = _CONDITIONING_RELATIVE_LIMIT,
) -> np.ndarray:
    """Identify mixed-sign rows with materially uncertain relative accuracy."""

    mixed, error_bound = _cancellation_error_bound(
        result,
        terms,
        operation_count=operation_count,
    )
    relative_bound = _stable_product(
        np.abs(result),
        float(relative_limit),
    )
    return mixed & (
        ~np.isfinite(result)
        | (error_bound >= relative_bound)
    )


def _two_sum(
    left: np.ndarray,
    right: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return a rounded sum and its exact finite-input rounding error."""

    summed = left + right
    right_virtual = summed - left
    error = (
        (left - (summed - right_virtual))
        + (right - right_virtual)
    )
    return summed, error


def _split_product_operand(
    values: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Split ordinary finite values into nonoverlapping high/low parts."""

    mantissa, exponent = np.frexp(values)
    scaled = _SPLITTER * mantissa
    high_mantissa = scaled - (scaled - mantissa)
    high = np.ldexp(high_mantissa, exponent)
    return high, values - high


def _two_product_error(
    left: np.ndarray,
    right: np.ndarray,
    product: np.ndarray,
) -> np.ndarray:
    """Return the exact ordinary-product rounding error via Dekker splitting."""

    left_high, left_low = _split_product_operand(left)
    right_high, right_low = _split_product_operand(right)
    return (
        (
            (left_high * right_high - product)
            + left_high * right_low
        )
        + left_low * right_high
        + left_low * right_low
    )


def _compensated_affine_residual(
    beta: np.ndarray,
    alpha: np.ndarray,
    left: np.ndarray,
    right: np.ndarray,
    target: np.ndarray,
    left_product: np.ndarray,
    right_product: np.ndarray,
) -> np.ndarray:
    """Evaluate ordinary affine rows with product and sum compensation."""

    left_error = _two_product_error(alpha, left, left_product)
    right_error = _two_product_error(-alpha, right, right_product)
    total = np.zeros(beta.shape, dtype=np.float64)
    correction = np.zeros(beta.shape, dtype=np.float64)
    for term in (
        beta,
        left_product,
        right_product,
        -target,
        left_error,
        right_error,
    ):
        total, error = _two_sum(total, term)
        correction += error
    total, error = _two_sum(total, correction)
    return total + error


def _has_compensation_exponent_margin(
    values: np.ndarray,
    *,
    lower_margin: int,
    upper_margin: int,
) -> np.ndarray:
    """Return rows safe for raw compensated arithmetic at every exponent."""

    array = np.asarray(values, dtype=np.float64)
    _, exponent = np.frexp(np.abs(array))
    return (
        (array == 0.0)
        | (
            np.isfinite(array)
            & (exponent >= _NORMAL_MIN_EXPONENT + int(lower_margin))
            & (exponent <= _NORMAL_MAX_EXPONENT - int(upper_margin))
        )
    )


def _stable_sum(*values: object) -> np.ndarray:
    """Vectorized sum with exact range and cancellation fallbacks."""

    arrays = np.broadcast_arrays(
        *(np.asarray(value, dtype=np.float64) for value in values)
    )
    result = np.zeros(arrays[0].shape, dtype=np.float64)
    safe = np.ones(result.shape, dtype=bool)
    for array in arrays:
        add_safe = _can_add(result, array)
        active = safe & add_safe
        result[active] += array[active]
        safe &= add_safe
    exceptional = ~safe
    exceptional |= _cancellation_risk(
        result,
        arrays,
        operation_count=len(arrays),
    )
    for flat_index in np.flatnonzero(exceptional):
        result.flat[flat_index] = _exact_sum_products_scalar(
            tuple((float(array.flat[flat_index]),) for array in arrays)
        )
    return result


def _stable_sum_products(
    products: Sequence[Sequence[object]],
    *,
    return_exceptional: bool = False,
) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    """Evaluate a sum of products with complete-expression fallbacks."""

    flat_values = tuple(
        value
        for product in products
        for value in product
    )
    broadcast = np.broadcast_arrays(
        *(np.asarray(value, dtype=np.float64) for value in flat_values)
    )
    broadcast_products: list[tuple[np.ndarray, ...]] = []
    offset = 0
    for product in products:
        width = len(product)
        broadcast_products.append(tuple(broadcast[offset:offset + width]))
        offset += width
    product_values = tuple(
        _stable_product(*product)
        for product in broadcast_products
    )
    result = _stable_sum(*product_values)
    vanished_product = np.zeros(result.shape, dtype=bool)
    for product, product_value in zip(
        broadcast_products,
        product_values,
    ):
        finite_nonzero = np.logical_and.reduce(
            tuple(np.isfinite(array) & (array != 0.0) for array in product)
        )
        vanished_product |= finite_nonzero & (product_value == 0.0)
    exceptional = _cancellation_risk(
        result,
        product_values,
        operation_count=(
            len(product_values)
            + sum(max(0, len(product) - 1) for product in products)
        ),
    )
    exceptional |= vanished_product
    exceptional |= (
        ~np.isfinite(result)
        & np.logical_and.reduce(
            tuple(np.isfinite(array) for array in broadcast)
        )
    )
    for flat_index in np.flatnonzero(exceptional):
        result.flat[flat_index] = _exact_sum_products_scalar(
            tuple(
                tuple(
                    float(array.flat[flat_index])
                    for array in product
                )
                for product in broadcast_products
            )
        )
    if return_exceptional:
        return result, exceptional
    return result


def _stable_sum_products_sign(
    products: Sequence[Sequence[object]],
) -> np.ndarray:
    """Return exact signs for exceptional sums of binary64 products."""

    result, exceptional = _stable_sum_products(
        products,
        return_exceptional=True,
    )
    sign = np.zeros(result.shape, dtype=np.int8)
    sign[result > 0.0] = 1
    sign[result < 0.0] = -1
    if not np.any(exceptional):
        return sign

    flat_values = tuple(
        value
        for product in products
        for value in product
    )
    broadcast = np.broadcast_arrays(
        *(np.asarray(value, dtype=np.float64) for value in flat_values)
    )
    broadcast_products: list[tuple[np.ndarray, ...]] = []
    offset = 0
    for product in products:
        width = len(product)
        broadcast_products.append(tuple(broadcast[offset:offset + width]))
        offset += width
    for flat_index in np.flatnonzero(exceptional):
        sign.flat[flat_index] = _exact_sum_products_sign_scalar(
            tuple(
                tuple(
                    float(array.flat[flat_index])
                    for array in product
                )
                for product in broadcast_products
            )
        )
    return sign


def _power_scaled_product_scalar(
    power: int,
    *factors: float,
) -> float:
    """Return ``2**power * prod(factors)`` without storing the power."""

    values = tuple(float(value) for value in factors)
    if any(math.isnan(value) for value in values):
        return float('nan')
    if any(value == 0.0 for value in values):
        return 0.0
    negative = sum(value < 0.0 for value in values) % 2
    if any(math.isinf(value) for value in values):
        return float('-inf') if negative else float('inf')

    mantissa = -1.0 if negative else 1.0
    exponent = int(power)
    for value in values:
        part, part_exponent = math.frexp(abs(value))
        mantissa *= part
        exponent += part_exponent
        mantissa, normalization = math.frexp(mantissa)
        exponent += normalization
    try:
        return math.ldexp(mantissa, exponent)
    except OverflowError:
        return float('-inf') if negative else float('inf')


def _power_scaled_product(
    power: int,
    *factors: object,
) -> np.ndarray:
    """Vectorized ``2**power * prod(factors)`` with range-safe fallbacks."""

    arrays = np.broadcast_arrays(
        *(np.asarray(factor, dtype=np.float64) for factor in factors)
    )
    shape = arrays[0].shape
    mantissa = np.ones(shape, dtype=np.float64)
    exponent = np.full(shape, int(power), dtype=np.int64)
    regular = np.ones(shape, dtype=bool)
    zero = np.zeros(shape, dtype=bool)
    for array in arrays:
        regular &= np.isfinite(array) & (array != 0.0)
        zero |= array == 0.0
        part, part_exponent = np.frexp(array)
        mantissa *= part
        exponent += part_exponent.astype(np.int64)
        mantissa, normalization = np.frexp(mantissa)
        exponent += normalization.astype(np.int64)

    normal = (
        regular
        & (exponent >= _NORMAL_MIN_EXPONENT)
        & (exponent <= _NORMAL_MAX_EXPONENT)
    )
    result = np.zeros(shape, dtype=np.float64)
    if np.any(normal):
        result[normal] = np.ldexp(mantissa[normal], exponent[normal])
    exceptional = ~normal & ~zero
    for flat_index in np.flatnonzero(exceptional):
        result.flat[flat_index] = _power_scaled_product_scalar(
            int(power),
            *(float(array.flat[flat_index]) for array in arrays),
        )
    return result


def _finite_product_exponents(*factors: object) -> tuple[np.ndarray, np.ndarray]:
    """Return validity and normalized binary exponents for finite products."""

    arrays = np.broadcast_arrays(
        *(np.asarray(factor, dtype=np.float64) for factor in factors)
    )
    valid = np.ones(arrays[0].shape, dtype=bool)
    mantissa = np.ones(arrays[0].shape, dtype=np.float64)
    exponent = np.zeros(arrays[0].shape, dtype=np.int64)
    for array in arrays:
        valid &= np.isfinite(array) & (array != 0.0)
        part, part_exponent = np.frexp(np.abs(array))
        mantissa *= part
        exponent += part_exponent.astype(np.int64)
        mantissa, normalization = np.frexp(mantissa)
        exponent += normalization.astype(np.int64)
    return valid, exponent


def _power_scaled_sum_products(
    power: int,
    products: Sequence[Sequence[object]],
) -> np.ndarray:
    """Evaluate ``2**power * sum(products)`` as one expression."""

    scaled_products = tuple(
        (1.0,) + tuple(product)
        for product in products
    )
    flat_values = tuple(
        value
        for product in scaled_products
        for value in product
    )
    broadcast = np.broadcast_arrays(
        *(np.asarray(value, dtype=np.float64) for value in flat_values)
    )
    broadcast_products: list[tuple[np.ndarray, ...]] = []
    offset = 0
    for product in scaled_products:
        width = len(product)
        broadcast_products.append(tuple(broadcast[offset:offset + width]))
        offset += width
    values = tuple(
        _power_scaled_product(power, *product)
        for product in broadcast_products
    )
    result = _stable_sum(*values)
    finite_inputs = np.logical_and.reduce(
        tuple(np.isfinite(array) for array in broadcast)
    )
    exceptional = finite_inputs & (
        ~np.isfinite(result)
        | _cancellation_risk(
            result,
            values,
            operation_count=(
                len(values)
                + sum(max(0, len(product) - 1) for product in products)
            ),
            relative_limit=_AFFINE_CONDITIONING_RELATIVE_LIMIT,
        )
    )
    for flat_index in np.flatnonzero(exceptional):
        exact = Fraction(0)
        for product in broadcast_products:
            term = Fraction(1)
            for array in product:
                term *= _fraction(float(array.flat[flat_index]))
            exact += term
        if power >= 0:
            exact *= 1 << int(power)
        else:
            exact /= 1 << int(-power)
        try:
            result.flat[flat_index] = float(exact)
        except OverflowError:
            result.flat[flat_index] = (
                float('-inf') if exact < 0 else float('inf')
            )
    return result


def _stable_scaled_difference(
    left: object,
    right: object,
    *scale_factors: object,
) -> np.ndarray:
    """Vectorized ``prod(scale_factors) * (left - right)``."""

    arrays = np.broadcast_arrays(
        *(
            np.asarray(value, dtype=np.float64)
            for value in (left, right, *scale_factors)
        )
    )
    left_array, right_array = arrays[:2]
    scale_arrays = arrays[2:]
    result = np.zeros(left_array.shape, dtype=np.float64)
    zero_scale = np.logical_or.reduce(
        tuple(array == 0.0 for array in scale_arrays)
    )
    difference_safe = _can_subtract(left_array, right_array)
    ordinary = difference_safe & ~zero_scale
    if np.any(ordinary):
        difference = np.zeros_like(result)
        difference[ordinary] = (
            left_array[ordinary] - right_array[ordinary]
        )
        product = _stable_product(*scale_arrays, difference)
        result[ordinary] = product[ordinary]
    exceptional = ~difference_safe & ~zero_scale
    for flat_index in np.flatnonzero(exceptional):
        result.flat[flat_index] = _stable_scaled_difference_scalar(
            float(left_array.flat[flat_index]),
            float(right_array.flat[flat_index]),
            *(float(array.flat[flat_index]) for array in scale_arrays),
        )
    return result


def _stable_scaled_sum(
    values: Sequence[object],
    *scale_factors: object,
) -> np.ndarray:
    """Evaluate ``prod(scale_factors) * sum(values)`` compositionally."""

    arrays = np.broadcast_arrays(
        *(
            np.asarray(value, dtype=np.float64)
            for value in (*values, *scale_factors)
        )
    )
    value_arrays = arrays[:len(values)]
    scale_arrays = arrays[len(values):]
    summed = _stable_sum(*value_arrays)
    result = _stable_product(*scale_arrays, summed)
    exceptional = (
        ~np.isfinite(summed)
        & np.logical_and.reduce(
            tuple(np.isfinite(array) for array in arrays)
        )
        & ~np.logical_or.reduce(
            tuple(array == 0.0 for array in scale_arrays)
        )
    )
    for flat_index in np.flatnonzero(exceptional):
        scales = tuple(
            float(array.flat[flat_index])
            for array in scale_arrays
        )
        result.flat[flat_index] = _exact_sum_products_scalar(
            tuple(
                scales + (float(array.flat[flat_index]),)
                for array in value_arrays
            )
        )
    return result


def _stable_ratio_difference(
    left: object,
    right: object,
    denominator: object,
) -> np.ndarray:
    """Vectorized scale-safe quotient of a difference."""

    left_array, right_array, denominator_array = np.broadcast_arrays(
        np.asarray(left, dtype=np.float64),
        np.asarray(right, dtype=np.float64),
        np.asarray(denominator, dtype=np.float64),
    )
    result = np.empty(left_array.shape, dtype=np.float64)
    difference_safe = _can_subtract(left_array, right_array)
    ordinary = difference_safe & np.isfinite(denominator_array)
    if np.any(ordinary):
        difference = left_array[ordinary] - right_array[ordinary]
        result[ordinary] = _stable_ratio_product(
            (difference,),
            (denominator_array[ordinary],),
        )
    for flat_index in np.flatnonzero(~ordinary):
        result.flat[flat_index] = _stable_ratio_difference_scalar(
            float(left_array.flat[flat_index]),
            float(right_array.flat[flat_index]),
            float(denominator_array.flat[flat_index]),
        )
    return result


def _stable_affine_residual(
    beta: object,
    alpha: object,
    left: object,
    right: object,
    target: object,
    *,
    active: object | None = None,
    return_exceptional: bool = False,
) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    """Evaluate one complete affine residual with cancellation fallbacks."""

    arrays = np.broadcast_arrays(
        *(
            np.asarray(value, dtype=np.float64)
            for value in (beta, alpha, left, right, target)
        )
    )
    beta_array, alpha_array, left_array, right_array, target_array = arrays
    if active is None:
        active_array = np.ones(beta_array.shape, dtype=bool)
    else:
        active_array = np.broadcast_to(
            np.asarray(active, dtype=bool),
            beta_array.shape,
        )
    result = np.zeros(beta_array.shape, dtype=np.float64)
    exceptional = np.zeros(beta_array.shape, dtype=bool)
    if not np.any(active_array):
        if return_exceptional:
            return result, exceptional
        return result

    active_beta = beta_array[active_array]
    active_alpha = alpha_array[active_array]
    active_left = left_array[active_array]
    active_right = right_array[active_array]
    active_target = target_array[active_array]
    left_product = _stable_product(active_alpha, active_left)
    right_product = _stable_product(
        -1.0,
        active_alpha,
        active_right,
    )
    terms = (
        active_beta,
        left_product,
        right_product,
        -active_target,
    )
    active_result = np.zeros(active_beta.shape, dtype=np.float64)
    range_safe = np.ones(active_beta.shape, dtype=bool)
    for term in terms:
        step_safe = _can_add(active_result, term)
        ordinary = range_safe & step_safe
        active_result[ordinary] += term[ordinary]
        range_safe &= step_safe

    finite_operands = np.logical_and.reduce(
        (
            np.isfinite(active_beta),
            np.isfinite(active_alpha),
            np.isfinite(active_left),
            np.isfinite(active_right),
            np.isfinite(active_target),
        )
    )
    vanished_product = (
        (
            np.isfinite(active_alpha)
            & (active_alpha != 0.0)
            & np.isfinite(active_left)
            & (active_left != 0.0)
            & (left_product == 0.0)
        )
        | (
            np.isfinite(active_alpha)
            & (active_alpha != 0.0)
            & np.isfinite(active_right)
            & (active_right != 0.0)
            & (right_product == 0.0)
        )
    )
    mixed, error_bound = _cancellation_error_bound(
        active_result,
        terms,
        operation_count=7,
    )
    exact_bound = _stable_product(
        np.abs(active_result),
        _CONDITIONING_RELATIVE_LIMIT,
    )
    exact = (
        ~range_safe
        | vanished_product
        | (
            finite_operands
            & ~np.isfinite(active_result)
        )
        | (
            mixed
            & (
                ~np.isfinite(active_result)
                | (error_bound >= exact_bound)
            )
        )
    )
    compensated_bound = _stable_product(
        np.abs(active_result),
        _AFFINE_CONDITIONING_RELATIVE_LIMIT,
    )
    compensate = (
        ~exact
        & mixed
        & (error_bound >= compensated_bound)
    )
    if np.any(compensate):
        compensation_safe = np.logical_and.reduce(
            (
                finite_operands,
                _has_compensation_exponent_margin(
                    active_alpha,
                    lower_margin=32,
                    upper_margin=0,
                ),
                _has_compensation_exponent_margin(
                    active_left,
                    lower_margin=32,
                    upper_margin=0,
                ),
                _has_compensation_exponent_margin(
                    active_right,
                    lower_margin=32,
                    upper_margin=0,
                ),
                _has_compensation_exponent_margin(
                    active_beta,
                    lower_margin=64,
                    upper_margin=4,
                ),
                _has_compensation_exponent_margin(
                    active_target,
                    lower_margin=64,
                    upper_margin=4,
                ),
                _has_compensation_exponent_margin(
                    left_product,
                    lower_margin=64,
                    upper_margin=4,
                ),
                _has_compensation_exponent_margin(
                    right_product,
                    lower_margin=64,
                    upper_margin=4,
                ),
                _has_compensation_exponent_margin(
                    active_result,
                    lower_margin=64,
                    upper_margin=4,
                ),
            )
        )
        unsafe_compensation = compensate & ~compensation_safe
        exact |= unsafe_compensation
        selected = np.flatnonzero(compensate & compensation_safe)
        if selected.size:
            active_result[selected] = _compensated_affine_residual(
                active_beta[selected],
                active_alpha[selected],
                active_left[selected],
                active_right[selected],
                active_target[selected],
                left_product[selected],
                right_product[selected],
            )

    for flat_index in np.flatnonzero(exact):
        active_result.flat[flat_index] = _exact_sum_products_scalar(
            (
                (float(active_beta.flat[flat_index]),),
                (
                    float(active_alpha.flat[flat_index]),
                    float(active_left.flat[flat_index]),
                ),
                (
                    -1.0,
                    float(active_alpha.flat[flat_index]),
                    float(active_right.flat[flat_index]),
                ),
                (-float(active_target.flat[flat_index]),),
            )
        )
    result[active_array] = active_result
    exceptional[active_array] = exact
    if return_exceptional:
        return result, exceptional
    return result


def _stable_scaled_affine_residual(
    beta: object,
    alpha: object,
    left: object,
    right: object,
    target: object,
    *scale_factors: object,
    active: object | None = None,
    return_exceptional: bool = False,
) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    """Evaluate a scaled complete affine residual without storing it first."""

    arrays = np.broadcast_arrays(
        *(
            np.asarray(value, dtype=np.float64)
            for value in (
                beta,
                alpha,
                left,
                right,
                target,
                *scale_factors,
            )
        )
    )
    beta_array, alpha_array, left_array, right_array, target_array = arrays[:5]
    scale_arrays = arrays[5:]
    if active is None:
        active_array = np.ones(beta_array.shape, dtype=bool)
    else:
        active_array = np.broadcast_to(
            np.asarray(active, dtype=bool),
            beta_array.shape,
        )
    result = np.zeros(beta_array.shape, dtype=np.float64)
    exceptional = np.zeros(beta_array.shape, dtype=bool)
    if not np.any(active_array):
        if return_exceptional:
            return result, exceptional
        return result
    active_scales = tuple(array[active_array] for array in scale_arrays)
    active_result = _stable_sum_products(
        (
            active_scales + (beta_array[active_array],),
            active_scales + (
                alpha_array[active_array],
                left_array[active_array],
            ),
            (-1.0,) + active_scales + (
                alpha_array[active_array],
                right_array[active_array],
            ),
            (-1.0,) + active_scales + (target_array[active_array],),
        ),
        return_exceptional=return_exceptional,
    )
    if return_exceptional:
        active_values, active_exceptional = active_result
        result[active_array] = active_values
        exceptional[active_array] = active_exceptional
        return result, exceptional
    result[active_array] = active_result
    return result


def _stable_affine_difference(
    beta: object,
    alpha: object,
    left: object,
    right: object,
) -> np.ndarray:
    """Evaluate ``beta + alpha * left - alpha * right`` directly."""

    return _stable_affine_residual(
        beta,
        alpha,
        left,
        right,
        0.0,
    )


def _stable_weighted_average(
    left: object,
    left_weight: object,
    right: object,
    right_weight: object,
) -> np.ndarray:
    """Return a stable convex weighted average of two arrays."""

    left_array, left_weight_array, right_array, right_weight_array = (
        np.broadcast_arrays(
            np.asarray(left, dtype=np.float64),
            np.asarray(left_weight, dtype=np.float64),
            np.asarray(right, dtype=np.float64),
            np.asarray(right_weight, dtype=np.float64),
        )
    )
    scale = np.maximum(left_weight_array, right_weight_array)
    valid_scale = np.isfinite(scale) & (scale != 0.0)
    left_scaled = _stable_normalized_ratio(
        left_weight_array,
        scale,
        active=valid_scale,
    )
    right_scaled = _stable_normalized_ratio(
        right_weight_array,
        scale,
        active=valid_scale,
    )
    denominator = _stable_sum(left_scaled, right_scaled)
    right_fraction = _stable_normalized_ratio(
        right_scaled,
        denominator,
        active=valid_scale & (denominator != 0.0),
    )
    adjustment = _stable_scaled_difference(
        right_array,
        left_array,
        right_fraction,
    )
    result = _stable_sum(left_array, adjustment)
    left_term = _stable_product(left_scaled, left_array)
    right_term = _stable_product(right_scaled, right_array)
    scaled_numerator = _stable_sum(left_term, right_term)
    exceptional = _cancellation_risk(
        scaled_numerator,
        (left_term, right_term),
        operation_count=5,
        relative_limit=_AFFINE_CONDITIONING_RELATIVE_LIMIT,
    )
    exceptional |= _cancellation_risk(
        result,
        (left_array, adjustment),
        operation_count=5,
        relative_limit=_AFFINE_CONDITIONING_RELATIVE_LIMIT,
    )
    exceptional |= (
        ((left_weight_array != 0.0) & (left_scaled == 0.0))
        | ((right_weight_array != 0.0) & (right_scaled == 0.0))
    )
    exceptional |= (
        (left_weight_array > 0.0)
        & (right_weight_array > 0.0)
        & (
            np.minimum(left_scaled, right_scaled)
            < _AFFINE_CONDITIONING_RELATIVE_LIMIT
        )
    )
    finite_inputs = (
        np.isfinite(left_array)
        & np.isfinite(left_weight_array)
        & np.isfinite(right_array)
        & np.isfinite(right_weight_array)
    )
    exceptional |= finite_inputs & ~np.isfinite(result)
    exceptional &= finite_inputs & valid_scale
    for flat_index in np.flatnonzero(exceptional):
        result.flat[flat_index] = _exact_weighted_average_scalar(
            float(left_array.flat[flat_index]),
            float(left_weight_array.flat[flat_index]),
            float(right_array.flat[flat_index]),
            float(right_weight_array.flat[flat_index]),
        )
    result[~valid_scale] = float('nan')
    return result


def _stable_incidence_accumulate(
    n_sites: int,
    site_i: np.ndarray,
    site_j: np.ndarray,
    row_values: np.ndarray,
) -> np.ndarray:
    """Return ``B @ row_values`` with masked exact site fallbacks."""

    n = int(n_sites)
    i = np.asarray(site_i, dtype=np.int64)
    j = np.asarray(site_j, dtype=np.int64)
    values = np.asarray(row_values, dtype=np.float64)
    if values.size == 0:
        return np.zeros(n, dtype=np.float64)
    degree = (
        np.bincount(i, minlength=n)
        + np.bincount(j, minlength=n)
    )
    if not np.all(np.isfinite(values)):
        terms: list[list[float]] = [[] for _ in range(n)]
        for site_i_value, site_j_value, value in zip(
            i.tolist(),
            j.tolist(),
            values.tolist(),
        ):
            terms[int(site_i_value)].append(float(value))
            terms[int(site_j_value)].append(-float(value))
        return np.asarray(
            [_stable_sum_scalar(*site_terms) for site_terms in terms],
            dtype=np.float64,
        )

    absolute = np.abs(values)
    site_scale = np.zeros(n, dtype=np.float64)
    np.maximum.at(site_scale, i, absolute)
    np.maximum.at(site_scale, j, absolute)
    safe_limit = np.full(n, _FLOAT_MAX, dtype=np.float64)
    occupied = degree != 0
    safe_limit[occupied] /= degree[occupied]
    range_safe = ~occupied | (site_scale <= safe_limit)

    i_values = np.zeros_like(values)
    j_values = np.zeros_like(values)
    i_safe = range_safe[i]
    j_safe = range_safe[j]
    i_values[i_safe] = values[i_safe]
    j_values[j_safe] = values[j_safe]
    result = (
        np.bincount(i, weights=i_values, minlength=n)
        - np.bincount(j, weights=j_values, minlength=n)
    )

    positive = (
        np.bincount(i, weights=values > 0.0, minlength=n)
        + np.bincount(j, weights=values < 0.0, minlength=n)
    ) > 0.0
    negative = (
        np.bincount(i, weights=values < 0.0, minlength=n)
        + np.bincount(j, weights=values > 0.0, minlength=n)
    ) > 0.0
    mixed = positive & negative

    i_material = (
        (absolute != 0.0)
        & (site_scale[i] != 0.0)
    )
    j_material = (
        (absolute != 0.0)
        & (site_scale[j] != 0.0)
    )
    i_normalized = _stable_normalized_ratio(
        absolute,
        site_scale[i],
        active=i_material,
    )
    j_normalized = _stable_normalized_ratio(
        absolute,
        site_scale[j],
        active=j_material,
    )
    scaled_l1 = (
        np.bincount(i, weights=i_normalized, minlength=n)
        + np.bincount(j, weights=j_normalized, minlength=n)
    )
    error_bound = _stable_product(
        site_scale,
        scaled_l1,
        (degree + 2) * _FLOAT_EPSILON,
    )
    relative_limit = _stable_product(
        np.abs(result),
        _AFFINE_CONDITIONING_RELATIVE_LIMIT,
    )
    exceptional = ~range_safe
    exceptional |= mixed & (
        ~np.isfinite(result)
        | (error_bound >= relative_limit)
    )
    exceptional_sites = np.flatnonzero(exceptional)
    if exceptional_sites.size:
        exceptional_index = np.full(n, -1, dtype=np.int64)
        exceptional_index[exceptional_sites] = np.arange(
            exceptional_sites.size,
            dtype=np.int64,
        )
        grouped_terms: list[list[float]] = [
            [] for _ in range(exceptional_sites.size)
        ]
        exceptional_i = exceptional[i]
        for group, value in zip(
            exceptional_index[i[exceptional_i]].tolist(),
            values[exceptional_i].tolist(),
        ):
            grouped_terms[group].append(float(value))
        exceptional_j = exceptional[j]
        for group, value in zip(
            exceptional_index[j[exceptional_j]].tolist(),
            values[exceptional_j].tolist(),
        ):
            grouped_terms[group].append(-float(value))
        result[exceptional_sites] = np.asarray(
            [
                _stable_sum_scalar(*site_terms)
                for site_terms in grouped_terms
            ],
            dtype=np.float64,
        )
    return np.asarray(result, dtype=np.float64)


def _scaled_sum_squares_state(values: object) -> tuple[float, float, int]:
    array = np.asarray(values, dtype=np.float64)
    count = int(array.size)
    if count == 0:
        return 0.0, 1.0, 0
    absolute = np.abs(array.ravel())
    if np.any(np.isnan(absolute)):
        return float('nan'), float('nan'), count
    scale = float(np.max(absolute))
    if scale == 0.0 or math.isinf(scale):
        return scale, 1.0, count
    scaled = np.zeros_like(absolute)
    _, exponents = np.frexp(absolute)
    _, scale_exponent = math.frexp(scale)
    # Ratios below 2**-500 cannot affect a binary64 norm for any realizable
    # array, and omitting them avoids an underflowing square.
    material = (absolute != 0.0) & (
        exponents >= scale_exponent - 500
    )
    scaled[material] = absolute[material] / scale
    return scale, float(np.dot(scaled, scaled)), count


def _stable_norm(values: object) -> float:
    """Return a scale-safe Euclidean norm."""

    scale, sum_squares, _ = _scaled_sum_squares_state(values)
    if scale == 0.0 or not math.isfinite(scale):
        return scale
    return _stable_product_scalar(scale, math.sqrt(sum_squares))


def _stable_sum_squares(values: object) -> float:
    """Return a scale-safe sum of squares."""

    scale, sum_squares, _ = _scaled_sum_squares_state(values)
    if scale == 0.0:
        return 0.0
    if not math.isfinite(scale):
        return scale
    return _stable_product_scalar(scale, scale, sum_squares)


def _stable_rms(values: object) -> float:
    """Return a scale-safe root-mean-square value."""

    scale, sum_squares, count = _scaled_sum_squares_state(values)
    if count == 0 or scale == 0.0:
        return 0.0
    if not math.isfinite(scale):
        return scale
    return _stable_product_scalar(scale, math.sqrt(sum_squares / count))


def _stable_mean_abs(values: object) -> float:
    """Return a scale-safe mean absolute value."""

    array = np.asarray(values, dtype=np.float64)
    if array.size == 0:
        return 0.0
    absolute = np.abs(array.ravel())
    if np.any(np.isnan(absolute)):
        return float('nan')
    scale = float(np.max(absolute))
    if scale == 0.0 or math.isinf(scale):
        return scale
    scaled = np.zeros_like(absolute)
    _, exponents = np.frexp(absolute)
    _, scale_exponent = math.frexp(scale)
    # A ratio below the normal range cannot affect the rounded mean for any
    # realizable NumPy array.  Omitting it avoids executing an inexact
    # subnormal division under a strict floating-point error policy.
    material = (absolute != 0.0) & (
        exponents >= scale_exponent + _NORMAL_MIN_EXPONENT
    )
    scaled[material] = absolute[material] / scale
    return _stable_product_scalar(scale, float(np.mean(scaled)))
