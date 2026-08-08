"""Shared transforms between mathematical power weights and backend radii."""

from __future__ import annotations

from numbers import Real

import numpy as np


def _finite_1d_real_array(values: np.ndarray, *, name: str) -> np.ndarray:
    """Validate without importing the surrounding package.

    This module is intentionally executable in isolation; the local checks
    mirror the shared public real-array contract while preserving that
    architectural boundary.
    """

    try:
        original = np.asarray(values, dtype=object)
    except (TypeError, ValueError):
        raise ValueError(f'{name} must be a 1D real numeric array') from None
    if original.ndim != 1:
        raise ValueError(f'{name} must be 1D')
    if not all(
        isinstance(value, (Real, np.integer, np.floating))
        and not isinstance(value, (bool, np.bool_))
        for value in original
    ):
        raise ValueError(
            f'{name} must contain only real numeric values; Boolean, complex, '
            'and string values are not accepted'
        )
    try:
        result = np.asarray(original, dtype=np.float64)
    except (OverflowError, TypeError, ValueError):
        raise ValueError(
            f'{name} must contain real numeric values representable as float64'
        ) from None
    if not np.all(np.isfinite(result)):
        raise ValueError(f'{name} must contain only finite values')
    return result


def _finite_real_scalar(value: object, *, name: str) -> float:
    if (
        not isinstance(value, (Real, np.integer, np.floating))
        or isinstance(value, (bool, np.bool_))
    ):
        raise ValueError(f'{name} must be finite and real numeric')
    try:
        result = float(value)
    except (OverflowError, TypeError, ValueError):
        raise ValueError(f'{name} must be finite and real numeric') from None
    if not np.isfinite(result):
        raise ValueError(f'{name} must be finite and real numeric')
    return result


def radii_to_weights(radii: np.ndarray) -> np.ndarray:
    """Convert radii to finite power weights (``w = r^2``).

    Raises ``ValueError`` when the input or its squared result is non-finite.
    """

    r = _finite_1d_real_array(radii, name='radii')
    if np.any(r < 0):
        raise ValueError('radii must be non-negative')
    with np.errstate(over='ignore', invalid='ignore'):
        weights = r * r
    if not np.all(np.isfinite(weights)):
        raise ValueError('radii produced non-finite weights')
    return weights


def validate_weight_representation_options(
    r_min: object,
    weight_shift: object | None,
) -> tuple[float, float | None]:
    """Validate the shared backend-radius representation controls."""

    r_min_value = _finite_real_scalar(r_min, name='r_min')
    if r_min_value < 0.0:
        raise ValueError('r_min must be >= 0')
    weight_shift_value = (
        None
        if weight_shift is None
        else _finite_real_scalar(weight_shift, name='weight_shift')
    )
    if weight_shift_value is not None and r_min_value != 0.0:
        raise ValueError('specify at most one of r_min and weight_shift')
    return r_min_value, weight_shift_value


def weights_to_radii(
    weights: np.ndarray,
    *,
    r_min: float = 0.0,
    weight_shift: float | None = None,
) -> tuple[np.ndarray, float]:
    """Convert power weights to finite radii using one global shift.

    Raises ``ValueError`` when an input, intermediate value, or result is
    non-finite.
    """

    w = _finite_1d_real_array(weights, name='weights')

    r_min, weight_shift = validate_weight_representation_options(
        r_min,
        weight_shift,
    )

    if weight_shift is not None:
        C = weight_shift
    else:
        with np.errstate(over='ignore', invalid='ignore'):
            r_min_squared = r_min * r_min
        if not np.isfinite(r_min_squared):
            raise ValueError('r_min squared must be finite')
        w_min = float(np.min(w)) if w.size else 0.0
        with np.errstate(over='ignore', invalid='ignore'):
            C = r_min_squared - w_min
        if not np.isfinite(C):
            raise ValueError('derived weight shift must be finite')

    with np.errstate(over='ignore', invalid='ignore'):
        w_shifted = w + C
    if not np.all(np.isfinite(w_shifted)):
        raise ValueError('weight shift produced non-finite values')
    if np.any(w_shifted < -1e-14):
        raise ValueError('weight shift produced negative values (numerical issue)')
    w_shifted = np.maximum(w_shifted, 0.0)
    with np.errstate(over='ignore', invalid='ignore'):
        radii = np.sqrt(w_shifted)
    if not np.all(np.isfinite(radii)) or not np.isfinite(C):
        raise ValueError('weight-to-radius transform produced non-finite results')
    return radii, float(C)
