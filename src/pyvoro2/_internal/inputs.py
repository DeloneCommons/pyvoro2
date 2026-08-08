"""Internal coercion helpers for public Python entry points.

These helpers intentionally keep error messages stable so the public API can be
refactored without changing its validation surface.
"""

from __future__ import annotations

import sys
from typing import Sequence

import numpy as np

from .validation import (
    CPP_INT_MAX,
    INT64_MAX,
    INT64_MIN,
    _as_original_array,
    _finite_float64_array,
    _real_float64_array,
    require_bool,
    require_nonnegative_index,
    require_positive_finite_real,
    require_positive_index,
    require_string_choice,
)


def coerce_point_array(
    values: Sequence[Sequence[float]] | np.ndarray,
    *,
    name: str,
    dim: int,
) -> np.ndarray:
    """Return a finite ``(n, dim)`` float64 array."""

    arr = _as_original_array(values, name=name)
    if arr.ndim != 2 or arr.shape[1] != dim:
        raise ValueError(f'{name} must have shape (n, {dim})')
    return _finite_float64_array(arr, name=name)


def coerce_finite_vector(
    values: Sequence[float] | np.ndarray,
    *,
    name: str,
    n: int,
) -> np.ndarray:
    """Return a finite real float64 vector with shape ``(n,)``."""

    arr = _as_original_array(values, name=name)
    if arr.shape != (n,):
        raise ValueError(f'{name} must have shape (n,)')
    return _finite_float64_array(arr, name=name)


def coerce_real_vector(
    values: Sequence[float] | np.ndarray,
    *,
    name: str,
    n: int,
) -> np.ndarray:
    """Return a real-numeric float64 vector, retaining derived infinities.

    This narrower helper is for established derived numerical intermediates,
    not public source data. Public source vectors use
    :func:`coerce_finite_vector`.
    """

    arr = _as_original_array(values, name=name)
    if arr.shape != (n,):
        raise ValueError(f'{name} must have shape ({n},)')
    return _real_float64_array(arr, name=name)


def coerce_finite_1d_array(
    values: Sequence[float] | np.ndarray,
    *,
    name: str,
) -> np.ndarray:
    """Return a finite real float64 array with shape ``(n,)``."""

    arr = _as_original_array(values, name=name)
    if arr.ndim != 1:
        raise ValueError(f'{name} must be 1D')
    return _finite_float64_array(arr, name=name)


def coerce_finite_matrix(
    values: Sequence[Sequence[float]] | np.ndarray,
    *,
    name: str,
    shape: tuple[int, int],
) -> np.ndarray:
    """Return a finite real float64 matrix with an exact shape."""

    arr = _as_original_array(values, name=name)
    if arr.shape != shape:
        raise ValueError(f'{name} must have shape {shape}')
    return _finite_float64_array(arr, name=name)


def coerce_external_id_array(
    ids: Sequence[int] | np.ndarray,
    *,
    name: str = 'ids',
    n: int | None = None,
) -> np.ndarray:
    """Return exact, non-negative, unique signed-int64 external IDs."""

    try:
        ids_arr = np.asarray(ids, dtype=object)
    except (TypeError, ValueError):
        expected = 'a 1D sequence' if n is None else 'a 1D sequence of length n'
        raise ValueError(f'{name} must be {expected}') from None
    if ids_arr.ndim != 1 or (n is not None and ids_arr.shape != (n,)):
        expected = 'a 1D sequence' if n is None else 'a 1D sequence of length n'
        raise ValueError(f'{name} must be {expected}')
    values = [
        require_nonnegative_index(
            item,
            name=f'{name}[{position}]',
            maximum=INT64_MAX,
        )
        for position, item in enumerate(ids_arr)
    ]
    result = np.array(values, dtype=np.int64, copy=True, order='C')
    if np.unique(result).size != result.size:
        raise ValueError(f'{name} must be unique')
    return result


def coerce_id_array(
    ids: Sequence[int] | np.ndarray | None,
    *,
    n: int,
) -> np.ndarray | None:
    """Return validated non-negative unique IDs or ``None``."""

    if ids is None:
        return None
    return coerce_external_id_array(ids, name='ids', n=n)


def coerce_nonnegative_vector(
    values: Sequence[float] | np.ndarray,
    *,
    name: str,
    n: int,
) -> np.ndarray:
    """Return a finite non-negative float64 vector with shape ``(n,)``."""

    arr = coerce_finite_vector(values, name=name, n=n)
    if np.any(arr < 0):
        raise ValueError(f'{name} must be non-negative')
    return arr


def coerce_nonnegative_scalar_or_vector(
    values: float | Sequence[float] | np.ndarray,
    *,
    name: str,
    n: int,
    length_name: str,
) -> np.ndarray:
    """Return a finite non-negative float64 vector.

    Scalars are broadcast to shape ``(n,)``. Vector inputs must already have
    shape ``(n,)``.
    """

    arr = _as_original_array(values, name=name)
    if arr.ndim == 0:
        scalar = _finite_float64_array(arr, name=name)
        arr = np.full((n,), float(scalar), dtype=np.float64)
    elif arr.shape == (n,):
        arr = _finite_float64_array(arr, name=name)
    else:
        raise ValueError(
            f'{name} must be a scalar or have shape ({length_name},)'
        )
    if np.any(arr < 0):
        raise ValueError(f'{name} must be non-negative')
    return arr


def coerce_native_block_parameters(
    *,
    blocks: Sequence[object] | None,
    block_size: object | None,
    dim: int,
) -> tuple[tuple[int, ...] | None, float | None]:
    """Validate explicit native block counts and optional block size."""

    block_size_value = (
        None
        if block_size is None
        else require_positive_finite_real(block_size, name='block_size')
    )
    if blocks is None:
        return None, block_size_value

    try:
        items = tuple(blocks)
    except TypeError:
        raise ValueError(
            f'blocks must be a length-{dim} sequence of positive exact integers'
        ) from None
    if len(items) != dim:
        raise ValueError(f'blocks must have length {dim}')
    counts = tuple(
        require_positive_index(
            item,
            name=f'blocks[{axis}]',
            maximum=CPP_INT_MAX,
        )
        for axis, item in enumerate(items)
    )
    return counts, block_size_value


def require_internal_id_range(n: int, *, name: str = 'points') -> None:
    """Require generated internal IDs ``0..n-1`` to fit C++ ``int``."""

    if n < 0 or n - 1 > CPP_INT_MAX:
        raise ValueError(
            f'{name} length is too large; internal IDs must fit C++ int range'
        )


def require_query_index_range(m: int, *, name: str = 'queries') -> None:
    """Require generated ghost query indices ``0..m-1`` to fit C++ ``int``."""

    if m < 0 or m - 1 > CPP_INT_MAX:
        raise ValueError(
            f'{name} length is too large; generated query indices must fit '
            'the C++ int destination range'
        )


def require_planar_ghost_site_id_range(
    n: int,
    *,
    name: str = 'points',
) -> None:
    """Keep planar generated site IDs below the reserved native ghost ID."""

    if n < 0 or n > CPP_INT_MAX:
        raise ValueError(
            f'{name}/site length is too large for planar ghost construction; '
            'generated site IDs must fit the C++ int destination range with '
            'CPP_INT_MAX reserved as the native ghost ID'
        )


def owned_readonly_array(
    values: Sequence[object] | np.ndarray,
    *,
    dtype: np.dtype | type | None = None,
) -> np.ndarray:
    """Return a C-contiguous owned array whose write flag is disabled."""

    result = np.array(values, dtype=dtype, copy=True, order='C')
    result.setflags(write=False)
    return result


def floor_to_int64(values: np.ndarray, *, name: str) -> np.ndarray:
    """Floor finite float64 values after proving signed-int64 representability."""

    arr = np.asarray(values, dtype=np.float64)
    if not np.all(np.isfinite(arr)):
        raise ValueError(f'{name} must contain only finite values before flooring')
    with np.errstate(over='ignore', invalid='ignore'):
        floored = np.floor(arr)
    upper_exclusive = float(2**63)
    if (
        not np.all(np.isfinite(floored))
        or np.any(floored < float(INT64_MIN))
        or np.any(floored >= upper_exclusive)
    ):
        raise ValueError(
            f'{name} floor result must be representable as signed int64'
        )
    return floored.astype(np.int64)


def round_to_int64(values: np.ndarray, *, name: str) -> np.ndarray:
    """Round finite float64 values after proving signed-int64 representability."""

    arr = np.asarray(values, dtype=np.float64)
    if not np.all(np.isfinite(arr)):
        raise ValueError(f'{name} must contain only finite values before rounding')
    with np.errstate(over='ignore', invalid='ignore'):
        rounded = np.round(arr)
    upper_exclusive = float(2**63)
    if (
        not np.all(np.isfinite(rounded))
        or np.any(rounded < float(INT64_MIN))
        or np.any(rounded >= upper_exclusive)
    ):
        raise ValueError(
            f'{name} rounded result must be representable as signed int64'
        )
    return rounded.astype(np.int64)


def checked_int64_add(
    left: np.ndarray,
    right: np.ndarray,
    *,
    name: str,
) -> np.ndarray:
    """Add aligned int64 arrays after checking for destination overflow."""

    lhs = np.asarray(left, dtype=np.int64)
    rhs = np.asarray(right, dtype=np.int64)
    if lhs.shape != rhs.shape:
        raise ValueError(f'{name} operands must have the same shape')
    positive = rhs > 0
    negative = rhs < 0
    if np.any(positive):
        limit = INT64_MAX - rhs[positive]
        if np.any(lhs[positive] > limit):
            raise ValueError(f'{name} must be representable as signed int64')
    if np.any(negative):
        limit = INT64_MIN - rhs[negative]
        if np.any(lhs[negative] < limit):
            raise ValueError(f'{name} must be representable as signed int64')
    return lhs + rhs


def validate_forward_mode(mode: object) -> str:
    """Return a canonical native forward construction mode."""

    return require_string_choice(
        mode,
        name='mode',
        choices=('standard', 'power'),
    )


def validate_duplicate_check_mode(mode: object) -> str:
    """Return a canonical public duplicate-check mode string."""

    return require_string_choice(
        mode,
        name='duplicate_check',
        choices=('off', 'warn', 'raise'),
    )


def validate_duplicate_options(
    *,
    threshold: object,
    wrap: object,
    max_pairs: object,
    threshold_name: str = 'duplicate_threshold',
    wrap_name: str = 'duplicate_wrap',
    max_pairs_name: str = 'duplicate_max_pairs',
) -> tuple[float, bool, int]:
    """Validate the common duplicate threshold, wrap flag, and pair limit."""

    return (
        require_positive_finite_real(threshold, name=threshold_name),
        require_bool(wrap, name=wrap_name),
        require_positive_index(
            max_pairs,
            name=max_pairs_name,
            maximum=sys.maxsize,
        ),
    )
