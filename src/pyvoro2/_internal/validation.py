"""Small strict validators shared by private Python entry-point helpers."""

from __future__ import annotations

from numbers import Real
import operator
from typing import Any, Sequence

import numpy as np


CPP_INT_MAX = int(np.iinfo(np.intc).max)


def _index_value(value: object, *, name: str) -> int:
    if isinstance(value, (bool, np.bool_)) or isinstance(value, np.ndarray):
        raise ValueError(
            f'{name} must be an exact integer; Boolean and array values '
            'are not accepted'
        )
    try:
        return int(operator.index(value))
    except (OverflowError, TypeError, ValueError):
        raise ValueError(f'{name} must be an exact integer') from None


def _check_index_maximum(value: int, *, name: str, maximum: int | None) -> None:
    if maximum is not None and value > maximum:
        raise ValueError(
            f'{name} must fit the destination range (maximum {maximum})'
        )


def require_index(
    value: object,
    *,
    name: str,
    minimum: int | None = None,
    maximum: int | None = None,
) -> int:
    """Return an exact non-Boolean index-protocol scalar within a range."""

    result = _index_value(value, name=name)
    if minimum is not None and result < minimum:
        raise ValueError(
            f'{name} must fit the destination range (minimum {minimum})'
        )
    _check_index_maximum(result, name=name, maximum=maximum)
    return result


def require_positive_index(
    value: object,
    *,
    name: str,
    maximum: int | None = None,
) -> int:
    """Return a positive exact non-Boolean index-protocol scalar."""

    result = _index_value(value, name=name)
    if result <= 0:
        raise ValueError(f'{name} must be a positive exact integer')
    _check_index_maximum(result, name=name, maximum=maximum)
    return result


def require_nonnegative_index(
    value: object,
    *,
    name: str,
    maximum: int | None = None,
) -> int:
    """Return a non-negative exact non-Boolean index-protocol scalar."""

    result = _index_value(value, name=name)
    if result < 0:
        raise ValueError(f'{name} must be a non-negative exact integer')
    _check_index_maximum(result, name=name, maximum=maximum)
    return result


def require_bool(value: object, *, name: str) -> bool:
    """Return an exact Python or NumPy Boolean scalar."""

    if not isinstance(value, (bool, np.bool_)):
        raise ValueError(f'{name} must be a Boolean scalar')
    return bool(value)


def require_optional_bool(value: object | None, *, name: str) -> bool | None:
    """Return ``None`` or an exact Python/NumPy Boolean scalar."""

    if value is None:
        return None
    return require_bool(value, name=name)


def require_bool_tuple(
    values: Sequence[object],
    *,
    name: str,
    length: int,
) -> tuple[bool, ...]:
    """Return a fixed-length tuple of exact Python/NumPy Booleans."""

    try:
        items = tuple(values)
    except TypeError:
        raise ValueError(
            f'{name} must be a length-{length} sequence of Boolean values'
        ) from None
    if len(items) != length:
        raise ValueError(
            f'{name} must be a length-{length} sequence of Boolean values'
        )
    return tuple(
        require_bool(item, name=f'{name}[{index}]')
        for index, item in enumerate(items)
    )


def require_bool_mask(
    values: Sequence[object] | np.ndarray,
    *,
    name: str,
    length: int | None = None,
) -> np.ndarray:
    """Return an owned read-only one-dimensional exact-Boolean mask."""

    try:
        arr = np.asarray(values)
    except (TypeError, ValueError):
        raise ValueError(f'{name} must be a one-dimensional Boolean mask') from None
    if arr.ndim != 1 or (length is not None and arr.shape != (length,)):
        if length is None:
            expected = 'one-dimensional'
        else:
            expected = f'have shape ({length},)'
        raise ValueError(f'{name} must {expected} and contain Boolean values')
    if arr.dtype.kind != 'b':
        if arr.dtype.kind != 'O' or not all(
            isinstance(item, (bool, np.bool_)) for item in arr.tolist()
        ):
            raise ValueError(f'{name} must contain only Boolean values')
    result = np.array(arr, dtype=np.bool_, copy=True, order='C')
    result.setflags(write=False)
    return result


def _is_real_numeric(value: object) -> bool:
    return (
        isinstance(value, (Real, np.integer, np.floating))
        and not isinstance(value, (bool, np.bool_))
    )


def _require_real_scalar_kind(value: object, *, name: str) -> object:
    if (
        isinstance(value, (bool, np.bool_, complex, np.complexfloating, str, bytes))
        or isinstance(value, np.ndarray)
        or not _is_real_numeric(value)
    ):
        raise ValueError(
            f'{name} must be a real numeric scalar; Boolean, complex, string, '
            'and array values are not accepted'
        )
    return value


def require_finite_real(value: object, *, name: str) -> float:
    """Return a finite float from a non-Boolean real numeric scalar."""

    real_value = _require_real_scalar_kind(value, name=name)
    try:
        result = float(real_value)
    except (OverflowError, TypeError, ValueError):
        raise ValueError(f'{name} must be a finite real numeric scalar') from None
    if not np.isfinite(result):
        raise ValueError(f'{name} must be a finite real numeric scalar')
    return result


def require_positive_finite_real(value: object, *, name: str) -> float:
    """Return a strictly positive finite real scalar."""

    result = require_finite_real(value, name=name)
    if result <= 0.0:
        raise ValueError(f'{name} must be a positive finite real scalar')
    return result


def require_nonnegative_finite_real(value: object, *, name: str) -> float:
    """Return a non-negative finite real scalar."""

    result = require_finite_real(value, name=name)
    if result < 0.0:
        raise ValueError(f'{name} must be a non-negative finite real scalar')
    return result


def require_real_in_interval(
    value: object,
    *,
    name: str,
    lower: float,
    upper: float,
) -> float:
    """Return a finite real scalar in the closed interval ``[lower, upper]``."""

    if not np.isfinite(lower) or not np.isfinite(upper) or lower > upper:
        raise ValueError('validator interval must be finite and ordered')
    result = require_finite_real(value, name=name)
    if result < lower or result > upper:
        raise ValueError(
            f'{name} must be a finite real scalar in [{lower}, {upper}]'
        )
    return result


def _as_original_array(values: Any, *, name: str) -> np.ndarray:
    try:
        # Object conversion preserves the element categories of mixed Python
        # sequences. A normal NumPy coercion could otherwise turn ``[False,
        # 1.0]`` into floats before Boolean rejection.
        return np.asarray(values, dtype=object)
    except (TypeError, ValueError):
        raise ValueError(f'{name} must be an array of real numeric values') from None


def _require_real_array_kind(arr: np.ndarray, *, name: str) -> None:
    if arr.dtype.kind in 'iuf':
        return
    if arr.dtype.kind == 'O' and all(
        _is_real_numeric(item)
        for item in arr.flat
    ):
        return
    raise ValueError(
        f'{name} must contain only real numeric values; Boolean, complex, '
        'and string values are not accepted'
    )


def _finite_float64_array(arr: np.ndarray, *, name: str) -> np.ndarray:
    _require_real_array_kind(arr, name=name)
    try:
        result = np.asarray(arr, dtype=np.float64)
    except (OverflowError, TypeError, ValueError):
        raise ValueError(
            f'{name} must contain real numeric values representable as float64'
        ) from None
    if not np.all(np.isfinite(result)):
        raise ValueError(f'{name} must contain only finite values')
    return result


def require_ordered_bounds(
    values: Sequence[Sequence[object]] | np.ndarray,
    *,
    name: str,
    dim: int,
) -> tuple[tuple[float, float], ...]:
    """Return finite strictly ordered bounds with finite interval lengths."""

    arr = _as_original_array(values, name=name)
    if arr.shape != (dim, 2):
        raise ValueError(f'{name} must have shape ({dim}, 2)')
    converted = _finite_float64_array(arr, name=name)
    result: list[tuple[float, float]] = []
    for axis, (lo, hi) in enumerate(converted):
        with np.errstate(over='ignore', invalid='ignore'):
            length = float(hi - lo)
        if not hi > lo or not np.isfinite(length):
            raise ValueError(
                f'{name}[{axis}] must be strictly ordered with a finite '
                'positive length'
            )
        result.append((float(lo), float(hi)))
    return tuple(result)
