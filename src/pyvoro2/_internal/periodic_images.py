"""Certified minimum-image geometry for binary64 lattice data.

This module is private.  It solves the exact closest-image problem defined by
the dyadic rational values represented by the supplied binary64 coordinates
and lattice vectors.  Float arrays in the result are rounded views of that
exact calculation; candidate selection never depends on floating linear
algebra or a bounded heuristic search window.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from functools import lru_cache
from itertools import product
import math
import struct
from typing import Iterator, Sequence

import numpy as np

from .inputs import coerce_finite_matrix, coerce_point_array
from .validation import (
    INT64_MAX,
    INT64_MIN,
    require_bool_tuple,
    require_index,
    require_nonnegative_index,
)


_MAX_SEED_CANDIDATES = 4_096
_MAX_EXACT_CANDIDATES_PER_PAIR = 1_000_000
_MAX_EXACT_CANDIDATES_PER_BATCH = 5_000_000
_BASIS_CACHE_SIZE = 128


@dataclass(frozen=True, slots=True)
class ExactDistanceKey:
    """Private exact squared-distance representation ``N / 2**exponent``."""

    numerator: int
    denominator_exponent: int


@dataclass(frozen=True, slots=True)
class MinimumImageBatch:
    """Private successful batch result for certified minimum-image geometry."""

    shift: np.ndarray
    displacement: np.ndarray
    distance_squared: np.ndarray
    exact_distance_key: tuple[ExactDistanceKey, ...]
    method: str
    candidate_count: np.ndarray
    tie_count: np.ndarray
    seed_count: np.ndarray
    seed_truncated: np.ndarray
    certified: bool = True


class MinimumImageCertificationError(RuntimeError):
    """Raised when exact minimum-image certification cannot be completed."""

    def __init__(
        self,
        message: str,
        *,
        stage: str,
        method: str,
        pair_index: int | None,
        basis_summary: dict[str, object],
        interval_widths: tuple[int, ...] | None = None,
        candidate_bound: int | None = None,
        configured_limit: int | None = None,
    ) -> None:
        super().__init__(message)
        self.stage = stage
        self.method = method
        self.pair_index = pair_index
        self.basis_summary = dict(basis_summary)
        self.interval_widths = interval_widths
        self.candidate_bound = candidate_bound
        self.configured_limit = configured_limit


class _BasisPreparationError(ValueError):
    pass


def _diagnostic_condition_number(matrix: np.ndarray) -> float:
    """Return a best-effort float condition estimate for diagnostics only."""

    try:
        with np.errstate(over='ignore', invalid='ignore', divide='ignore'):
            return float(np.linalg.cond(matrix))
    except np.linalg.LinAlgError:
        return float('inf')


@dataclass(frozen=True, slots=True)
class _BasisData:
    dimension: int
    periodic_axes: tuple[bool, ...]
    fractions: tuple[tuple[Fraction, ...], ...]
    integer_rows: tuple[tuple[int, ...], ...]
    denominator_exponent: int
    inverse: tuple[tuple[Fraction, ...], ...] | None
    inverse_column_l1: tuple[Fraction, ...] | None
    inverse_column_numerators: tuple[tuple[int, ...], ...] | None
    inverse_column_denominators: tuple[int, ...] | None
    orthogonal: bool
    bit_patterns: tuple[int, ...]
    condition_number: float

    @property
    def method(self) -> str:
        if self.orthogonal:
            return 'orthogonal-exact'
        return 'triclinic-finite-box'

    @property
    def summary(self) -> dict[str, object]:
        return {
            'dimension': self.dimension,
            'periodic_axes': self.periodic_axes,
            'lattice_bits': tuple(f'{value:016x}' for value in self.bit_patterns),
            'condition_number': self.condition_number,
        }


@dataclass(frozen=True, slots=True)
class _RowSolution:
    shift: tuple[int, ...]
    displacement_numerators: tuple[int, ...]
    denominator_exponent: int
    distance_numerator: int
    candidate_count: int
    tie_count: int
    seed_count: int
    seed_truncated: bool


@dataclass(frozen=True, slots=True)
class _TriclinicPlan:
    pair_index: int
    displacement_numerators: tuple[int, ...]
    lattice_numerators: tuple[tuple[int, ...], ...]
    denominator_exponent: int
    lower: tuple[int, ...]
    upper: tuple[int, ...]
    candidate_count: int
    seed_count: int
    seed_truncated: bool
    tie_orientation: int


@dataclass(frozen=True, slots=True)
class _ExactTriclinicBucketLayout:
    """Exact source-binary64 data for sparse triclinic bucket scans."""

    keys: tuple[tuple[int, ...], ...]
    bins: tuple[int, ...]
    coefficient_bounds: tuple[Fraction, ...]


def _readonly_array(values: object, *, dtype: np.dtype | type) -> np.ndarray:
    result = np.array(values, dtype=dtype, copy=True, order='C')
    result.setflags(write=False)
    return result


def _float_from_bits(bits: int) -> float:
    return struct.unpack('>d', struct.pack('>Q', bits))[0]


def _dyadic_parts(value: float) -> tuple[int, int]:
    numerator, denominator = float(value).as_integer_ratio()
    exponent = denominator.bit_length() - 1
    if denominator != 1 << exponent:
        raise _BasisPreparationError('binary64 denominator is not a power of two')
    return numerator, exponent


def _determinant_3x3(matrix: Sequence[Sequence[Fraction]]) -> Fraction:
    a, b, c = matrix
    return (
        a[0] * (b[1] * c[2] - b[2] * c[1])
        - a[1] * (b[0] * c[2] - b[2] * c[0])
        + a[2] * (b[0] * c[1] - b[1] * c[0])
    )


def _inverse_fraction_matrix(
    matrix: Sequence[Sequence[Fraction]],
) -> tuple[tuple[Fraction, ...], ...]:
    dimension = len(matrix)
    work = [
        list(row) + [Fraction(int(i == j)) for j in range(dimension)]
        for i, row in enumerate(matrix)
    ]
    for column in range(dimension):
        pivot = next(
            (row for row in range(column, dimension) if work[row][column]),
            None,
        )
        if pivot is None:
            raise _BasisPreparationError('lattice is exactly singular')
        if pivot != column:
            work[column], work[pivot] = work[pivot], work[column]
        scale = work[column][column]
        work[column] = [value / scale for value in work[column]]
        for row in range(dimension):
            if row == column:
                continue
            factor = work[row][column]
            if factor:
                work[row] = [
                    left - factor * right
                    for left, right in zip(work[row], work[column])
                ]
    return tuple(
        tuple(work[row][dimension:]) for row in range(dimension)
    )


@lru_cache(maxsize=_BASIS_CACHE_SIZE)
def _prepare_basis(
    dimension: int,
    periodic_axes: tuple[bool, ...],
    bit_patterns: tuple[int, ...],
) -> _BasisData:
    floats = tuple(_float_from_bits(value) for value in bit_patterns)
    rows_float = tuple(
        tuple(floats[row * dimension + column] for column in range(dimension))
        for row in range(dimension)
    )
    parts = tuple(_dyadic_parts(value) for value in floats)
    exponent = max((item[1] for item in parts), default=0)
    integer_values = tuple(
        numerator << (exponent - value_exponent)
        for numerator, value_exponent in parts
    )
    integer_rows = tuple(
        tuple(
            integer_values[row * dimension + column]
            for column in range(dimension)
        )
        for row in range(dimension)
    )
    fractions = tuple(
        tuple(Fraction(integer_rows[row][column], 1 << exponent)
              for column in range(dimension))
        for row in range(dimension)
    )
    orthogonal = all(
        fractions[row][column] == 0
        for row in range(dimension)
        for column in range(dimension)
        if row != column
    )
    if orthogonal and any(fractions[axis][axis] <= 0 for axis in range(dimension)):
        raise _BasisPreparationError(
            'canonical orthogonal lattice lengths must be strictly positive'
        )

    inverse = None
    inverse_column_l1 = None
    inverse_column_numerators = None
    inverse_column_denominators = None
    if not orthogonal:
        if dimension != 3 or not all(periodic_axes):
            raise _BasisPreparationError(
                'non-orthogonal certification is supported only for fully '
                'periodic three-dimensional lattices'
            )
        if _determinant_3x3(fractions) <= 0:
            raise _BasisPreparationError(
                'canonical triclinic lattice must be exactly right-handed'
            )

    # Fully periodic 3D bucket layouts need the same exact inverse whether the
    # supplied PeriodicCell happens to be skew or diagonal.  Keeping its
    # integer form on the existing bounded basis cache avoids Fraction work in
    # the per-point R5-SC-001 key loop.
    if dimension == 3 and all(periodic_axes):
        if orthogonal:
            inverse = tuple(
                tuple(
                    Fraction(1, 1) / fractions[row][row]
                    if row == column
                    else Fraction()
                    for column in range(dimension)
                )
                for row in range(dimension)
            )
        else:
            inverse = _inverse_fraction_matrix(fractions)
        inverse_column_l1 = tuple(
            sum((abs(inverse[row][column]) for row in range(dimension)), Fraction())
            for column in range(dimension)
        )
        inverse_column_denominators = tuple(
            math.lcm(
                *(inverse[row][column].denominator for row in range(dimension))
            )
            for column in range(dimension)
        )
        inverse_column_numerators = tuple(
            tuple(
                inverse[row][column].numerator
                * (
                    inverse_column_denominators[column]
                    // inverse[row][column].denominator
                )
                for row in range(dimension)
            )
            for column in range(dimension)
        )

    condition = _diagnostic_condition_number(
        np.asarray(rows_float, dtype=np.float64)
    )
    return _BasisData(
        dimension=dimension,
        periodic_axes=periodic_axes,
        fractions=fractions,
        integer_rows=integer_rows,
        denominator_exponent=exponent,
        inverse=inverse,
        inverse_column_l1=inverse_column_l1,
        inverse_column_numerators=inverse_column_numerators,
        inverse_column_denominators=inverse_column_denominators,
        orthogonal=orthogonal,
        bit_patterns=bit_patterns,
        condition_number=condition,
    )


def _basis_key(
    lattice_vectors: np.ndarray,
    periodic_axes: tuple[bool, ...],
) -> tuple[int, tuple[bool, ...], tuple[int, ...]]:
    contiguous = np.ascontiguousarray(lattice_vectors, dtype=np.float64)
    bits = tuple(int(value) for value in contiguous.view(np.uint64).ravel())
    return contiguous.shape[0], periodic_axes, bits


def _basis_cache_info():
    """Return private cache statistics for deterministic tests/benchmarks."""

    return _prepare_basis.cache_info()


def _basis_cache_clear() -> None:
    """Clear the private bounded basis cache for deterministic tests."""

    _prepare_basis.cache_clear()


def _exact_triclinic_bucket_layout(
    points: np.ndarray,
    *,
    origin: np.ndarray,
    lattice_vectors: np.ndarray,
    radius: float,
) -> _ExactTriclinicBucketLayout:
    """Return exact keys and proof-sized bins for a triclinic scan.

    All values are interpreted as their exact source-binary64 dyadics.  The
    cached inverse is rational, point/origin subtraction and multiplication
    use integers, and modulo/key assignment uses Euclidean integer division.
    Thus a point is never rounded across a bucket boundary or the 0/1 seam.
    """

    pts = np.asarray(points, dtype=np.float64)
    origin_array = np.asarray(origin, dtype=np.float64)
    lattice = np.asarray(lattice_vectors, dtype=np.float64)
    radius_value = float(radius)
    if pts.ndim != 2 or pts.shape[1] != 3:
        raise ValueError('triclinic bucket points must have shape (n, 3)')
    if origin_array.shape != (3,):
        raise ValueError('triclinic bucket origin must have shape (3,)')
    if lattice.shape != (3, 3):
        raise ValueError('triclinic bucket lattice must have shape (3, 3)')
    if not (
        np.all(np.isfinite(pts))
        and np.all(np.isfinite(origin_array))
        and np.all(np.isfinite(lattice))
    ):
        raise ValueError('triclinic bucket inputs must be finite')
    if not math.isfinite(radius_value) or radius_value <= 0.0:
        raise ValueError('triclinic bucket radius must be positive and finite')

    basis = _prepare_basis(
        *_basis_key(lattice, (True, True, True)),
    )
    inverse_l1 = basis.inverse_column_l1
    inverse_numerators = basis.inverse_column_numerators
    inverse_denominators = basis.inverse_column_denominators
    if (
        inverse_l1 is None
        or inverse_numerators is None
        or inverse_denominators is None
    ):
        raise _BasisPreparationError(
            'triclinic bucket layout requires an exact three-dimensional inverse'
        )

    radius_fraction = Fraction.from_float(radius_value)
    coefficient_bounds = tuple(
        radius_fraction * value for value in inverse_l1
    )
    if any(bound <= 0 for bound in coefficient_bounds):
        raise _BasisPreparationError(
            'triclinic coefficient bounds must be exactly positive'
        )
    bins = tuple(
        1 if bound >= 1 else bound.denominator // bound.numerator
        for bound in coefficient_bounds
    )

    origin_parts = tuple(_dyadic_parts(float(value)) for value in origin_array)
    keys: list[tuple[int, ...]] = []
    for point in pts:
        point_parts = tuple(_dyadic_parts(float(value)) for value in point)
        exponent = max(
            *(item[1] for item in origin_parts),
            *(item[1] for item in point_parts),
        )
        delta = tuple(
            (
                point_parts[axis][0]
                << (exponent - point_parts[axis][1])
            ) - (
                origin_parts[axis][0]
                << (exponent - origin_parts[axis][1])
            )
            for axis in range(3)
        )
        point_key: list[int] = []
        for column in range(3):
            coordinate_numerator = sum(
                delta[row] * inverse_numerators[column][row]
                for row in range(3)
            )
            coordinate_denominator = (
                (1 << exponent) * inverse_denominators[column]
            )
            modulo_numerator = coordinate_numerator % coordinate_denominator
            point_key.append(
                (modulo_numerator * bins[column])
                // coordinate_denominator
            )
        keys.append(tuple(point_key))

    return _ExactTriclinicBucketLayout(
        keys=tuple(keys),
        bins=bins,
        coefficient_bounds=coefficient_bounds,
    )


def _aligned_integer_geometry(
    pi: np.ndarray,
    pj: np.ndarray,
    basis: _BasisData,
) -> tuple[tuple[int, ...], tuple[tuple[int, ...], ...], int]:
    pi_parts = tuple(_dyadic_parts(float(value)) for value in pi)
    pj_parts = tuple(_dyadic_parts(float(value)) for value in pj)
    exponent = max(
        basis.denominator_exponent,
        *(item[1] for item in pi_parts),
        *(item[1] for item in pj_parts),
    )
    displacement = tuple(
        (
            pj_parts[axis][0] << (exponent - pj_parts[axis][1])
        ) - (
            pi_parts[axis][0] << (exponent - pi_parts[axis][1])
        )
        for axis in range(basis.dimension)
    )
    lattice = tuple(
        tuple(
            value << (exponent - basis.denominator_exponent)
            for value in row
        )
        for row in basis.integer_rows
    )
    return displacement, lattice, exponent


def _candidate_displacement(
    displacement: tuple[int, ...],
    lattice: tuple[tuple[int, ...], ...],
    shift: tuple[int, ...],
) -> tuple[int, ...]:
    dimension = len(displacement)
    return tuple(
        displacement[column]
        + sum(shift[row] * lattice[row][column] for row in range(dimension))
        for column in range(dimension)
    )


def _distance_numerator(values: tuple[int, ...]) -> int:
    return sum(value * value for value in values)


def _prefer_shift(
    candidate: tuple[int, ...],
    incumbent: tuple[int, ...],
    orientation: int,
) -> bool:
    if orientation == 1:
        return candidate < incumbent
    return candidate > incumbent


def _check_selected_shift(
    shift: tuple[int, ...],
    *,
    pair_index: int,
    basis: _BasisData,
) -> None:
    if any(value < INT64_MIN or value > INT64_MAX for value in shift):
        raise MinimumImageCertificationError(
            'certified minimum-image shift is outside the signed-int64 contract',
            stage='shift_range',
            method=basis.method,
            pair_index=pair_index,
            basis_summary=basis.summary,
        )


def _solve_orthogonal_row(
    displacement: tuple[int, ...],
    lattice: tuple[tuple[int, ...], ...],
    exponent: int,
    *,
    pair_index: int,
    basis: _BasisData,
    orientation: int,
) -> _RowSolution:
    choices: list[tuple[int, ...]] = []
    for axis, periodic in enumerate(basis.periodic_axes):
        if not periodic:
            choices.append((0,))
            continue
        length = lattice[axis][axis]
        numerator = -displacement[axis]
        lower = numerator // length
        upper = -((-numerator) // length)
        choices.append((lower,) if lower == upper else (lower, upper))

    best_shift: tuple[int, ...] | None = None
    best_values: tuple[int, ...] | None = None
    best_distance: int | None = None
    tie_count = 0
    for shift in product(*choices):
        values = _candidate_displacement(displacement, lattice, shift)
        distance = _distance_numerator(values)
        if best_distance is None or distance < best_distance:
            best_shift = shift
            best_values = values
            best_distance = distance
            tie_count = 1
        elif distance == best_distance:
            tie_count += 1
            assert best_shift is not None
            if _prefer_shift(shift, best_shift, orientation):
                best_shift = shift
                best_values = values

    assert best_shift is not None and best_values is not None
    assert best_distance is not None
    _check_selected_shift(best_shift, pair_index=pair_index, basis=basis)
    return _RowSolution(
        shift=best_shift,
        displacement_numerators=best_values,
        denominator_exponent=exponent,
        distance_numerator=best_distance,
        candidate_count=math.prod(len(item) for item in choices),
        tie_count=tie_count,
        seed_count=0,
        seed_truncated=False,
    )


def _nearest_integer(value: Fraction, orientation: int) -> int:
    lower = math.floor(value)
    upper = math.ceil(value)
    lower_distance = value - lower
    upper_distance = upper - value
    if lower_distance < upper_distance:
        return lower
    if upper_distance < lower_distance:
        return upper
    return lower if orientation == 1 else upper


def _seed_offsets(
    dimension: int,
    radius: int,
) -> tuple[Iterator[tuple[int, ...]], int, bool]:
    largest_complete_radius = 0
    while (
        (2 * (largest_complete_radius + 1) + 1) ** dimension
        <= _MAX_SEED_CANDIDATES
    ):
        largest_complete_radius += 1
    truncated = radius > largest_complete_radius
    seed_count = (
        _MAX_SEED_CANDIDATES
        if truncated
        else (2 * radius + 1) ** dimension
    )

    def generate() -> Iterator[tuple[int, ...]]:
        yielded = 0
        zero = (0,) * dimension
        yield zero
        yielded += 1
        shell = 1
        while shell <= radius and yielded < _MAX_SEED_CANDIDATES:
            values = range(-shell, shell + 1)
            for offset in product(values, repeat=dimension):
                if max(abs(value) for value in offset) != shell:
                    continue
                yield offset
                yielded += 1
                if yielded >= _MAX_SEED_CANDIDATES:
                    return
            shell += 1

    return generate(), seed_count, truncated


def _fractional_coordinates(
    displacement: tuple[int, ...],
    exponent: int,
    inverse: tuple[tuple[Fraction, ...], ...],
) -> tuple[Fraction, ...]:
    dimension = len(displacement)
    denominator = 1 << exponent
    return tuple(
        sum(
            (
                Fraction(displacement[row], denominator)
                * inverse[row][column]
            )
            for row in range(dimension)
        )
        for column in range(dimension)
    )


def _ceil_sqrt(value: int) -> int:
    root = math.isqrt(value)
    return root if root * root == value else root + 1


def _prepare_triclinic_row(
    displacement: tuple[int, ...],
    lattice: tuple[tuple[int, ...], ...],
    exponent: int,
    *,
    pair_index: int,
    basis: _BasisData,
    orientation: int,
    image_search: int,
) -> _TriclinicPlan:
    assert basis.inverse is not None
    assert basis.inverse_column_l1 is not None
    q = _fractional_coordinates(displacement, exponent, basis.inverse)
    center = tuple(_nearest_integer(-value, orientation) for value in q)

    offsets, seed_count, seed_truncated = _seed_offsets(3, image_search)
    incumbent_shift: tuple[int, ...] | None = None
    incumbent_distance: int | None = None
    for offset in offsets:
        shift = tuple(center[axis] + offset[axis] for axis in range(3))
        values = _candidate_displacement(displacement, lattice, shift)
        distance = _distance_numerator(values)
        if incumbent_distance is None or distance < incumbent_distance:
            incumbent_shift = shift
            incumbent_distance = distance
        elif distance == incumbent_distance:
            assert incumbent_shift is not None
            if _prefer_shift(shift, incumbent_shift, orientation):
                incumbent_shift = shift
    assert incumbent_shift is not None and incumbent_distance is not None

    upper_norm = Fraction(_ceil_sqrt(incumbent_distance), 1 << exponent)
    lower = tuple(
        math.ceil(-q[axis] - upper_norm * basis.inverse_column_l1[axis])
        for axis in range(3)
    )
    upper = tuple(
        math.floor(-q[axis] + upper_norm * basis.inverse_column_l1[axis])
        for axis in range(3)
    )
    widths = tuple(upper[axis] - lower[axis] + 1 for axis in range(3))
    if any(width <= 0 for width in widths):
        raise MinimumImageCertificationError(
            'proof-derived triclinic candidate box is unexpectedly empty',
            stage='finite_box',
            method=basis.method,
            pair_index=pair_index,
            basis_summary=basis.summary,
            interval_widths=widths,
        )
    candidate_count = math.prod(widths)
    if candidate_count > _MAX_EXACT_CANDIDATES_PER_PAIR:
        raise MinimumImageCertificationError(
            'proof-derived triclinic candidate box exceeds the per-pair budget',
            stage='pair_candidate_budget',
            method=basis.method,
            pair_index=pair_index,
            basis_summary=basis.summary,
            interval_widths=widths,
            candidate_bound=candidate_count,
            configured_limit=_MAX_EXACT_CANDIDATES_PER_PAIR,
        )
    return _TriclinicPlan(
        pair_index=pair_index,
        displacement_numerators=displacement,
        lattice_numerators=lattice,
        denominator_exponent=exponent,
        lower=lower,
        upper=upper,
        candidate_count=candidate_count,
        seed_count=seed_count,
        seed_truncated=seed_truncated,
        tie_orientation=orientation,
    )


def _solve_triclinic_plan(
    plan: _TriclinicPlan,
    *,
    basis: _BasisData,
) -> _RowSolution:
    best_shift: tuple[int, ...] | None = None
    best_values: tuple[int, ...] | None = None
    best_distance: int | None = None
    tie_count = 0
    ranges = tuple(
        range(plan.lower[axis], plan.upper[axis] + 1) for axis in range(3)
    )
    for shift in product(*ranges):
        values = _candidate_displacement(
            plan.displacement_numerators,
            plan.lattice_numerators,
            shift,
        )
        distance = _distance_numerator(values)
        if best_distance is None or distance < best_distance:
            best_shift = shift
            best_values = values
            best_distance = distance
            tie_count = 1
        elif distance == best_distance:
            tie_count += 1
            assert best_shift is not None
            if _prefer_shift(shift, best_shift, plan.tie_orientation):
                best_shift = shift
                best_values = values

    assert best_shift is not None and best_values is not None
    assert best_distance is not None
    _check_selected_shift(best_shift, pair_index=plan.pair_index, basis=basis)
    return _RowSolution(
        shift=best_shift,
        displacement_numerators=best_values,
        denominator_exponent=plan.denominator_exponent,
        distance_numerator=best_distance,
        candidate_count=plan.candidate_count,
        tie_count=tie_count,
        seed_count=plan.seed_count,
        seed_truncated=plan.seed_truncated,
    )


def _dyadic_float(numerator: int, exponent: int) -> float:
    return float(Fraction(numerator, 1 << exponent))


def _validate_tie_orientation(
    values: Sequence[int] | np.ndarray,
    *,
    length: int,
) -> np.ndarray:
    raw = np.asarray(values, dtype=object)
    if raw.shape != (length,):
        raise ValueError(f'tie_orientation must have shape ({length},)')
    result = []
    for index, value in enumerate(raw):
        orientation = require_index(
            value,
            name=f'tie_orientation[{index}]',
            minimum=-1,
            maximum=1,
        )
        if orientation == 0:
            raise ValueError('tie_orientation values must be +1 or -1')
        result.append(orientation)
    return np.asarray(result, dtype=np.int8)


def minimum_image_displacements(
    pi: Sequence[Sequence[float]] | np.ndarray,
    pj: Sequence[Sequence[float]] | np.ndarray,
    *,
    lattice_vectors: Sequence[Sequence[float]] | np.ndarray,
    periodic_axes: Sequence[bool],
    tie_orientation: Sequence[int] | np.ndarray,
    image_search: int,
) -> MinimumImageBatch:
    """Return exact-certified minimum-image geometry for a row batch.

    ``image_search`` seeds an incumbent neighborhood only.  The successful
    result is independent of that neighborhood because the triclinic path
    exhaustively enumerates the subsequently proved finite box.
    """

    raw_pi = np.asarray(pi, dtype=object)
    if raw_pi.ndim != 2 or raw_pi.shape[1] not in (2, 3):
        raise ValueError('pi must have shape (m, d) with d in {2, 3}')
    dimension = int(raw_pi.shape[1])
    pi_array = coerce_point_array(raw_pi, name='pi', dim=dimension)
    pj_array = coerce_point_array(pj, name='pj', dim=dimension)
    if pi_array.shape != pj_array.shape:
        raise ValueError('pi and pj must have the same shape')
    lattice = coerce_finite_matrix(
        lattice_vectors,
        name='lattice_vectors',
        shape=(dimension, dimension),
    )
    axes = require_bool_tuple(
        periodic_axes,
        name='periodic_axes',
        length=dimension,
    )
    orientation = _validate_tie_orientation(
        tie_orientation,
        length=pi_array.shape[0],
    )
    search = require_nonnegative_index(
        image_search,
        name='image_search',
    )

    key = _basis_key(lattice, axes)
    try:
        basis = _prepare_basis(*key)
    except _BasisPreparationError as exc:
        summary = {
            'dimension': dimension,
            'periodic_axes': axes,
            'lattice_bits': tuple(f'{value:016x}' for value in key[2]),
            'condition_number': _diagnostic_condition_number(lattice),
        }
        raise MinimumImageCertificationError(
            str(exc),
            stage='basis_preparation',
            method='unresolved',
            pair_index=None,
            basis_summary=summary,
        ) from None

    prepared: list[_RowSolution | _TriclinicPlan] = []
    cumulative_candidates = 0
    for pair_index in range(pi_array.shape[0]):
        displacement, lattice_integers, exponent = _aligned_integer_geometry(
            pi_array[pair_index],
            pj_array[pair_index],
            basis,
        )
        if basis.orthogonal:
            row: _RowSolution | _TriclinicPlan = _solve_orthogonal_row(
                displacement,
                lattice_integers,
                exponent,
                pair_index=pair_index,
                basis=basis,
                orientation=int(orientation[pair_index]),
            )
        else:
            row = _prepare_triclinic_row(
                displacement,
                lattice_integers,
                exponent,
                pair_index=pair_index,
                basis=basis,
                orientation=int(orientation[pair_index]),
                image_search=search,
            )
        cumulative_candidates += row.candidate_count
        if cumulative_candidates > _MAX_EXACT_CANDIDATES_PER_BATCH:
            widths = None
            if isinstance(row, _TriclinicPlan):
                widths = tuple(
                    row.upper[axis] - row.lower[axis] + 1
                    for axis in range(dimension)
                )
            raise MinimumImageCertificationError(
                'exact candidate count exceeds the cumulative batch budget',
                stage='batch_candidate_budget',
                method=basis.method,
                pair_index=pair_index,
                basis_summary=basis.summary,
                interval_widths=widths,
                candidate_bound=cumulative_candidates,
                configured_limit=_MAX_EXACT_CANDIDATES_PER_BATCH,
            )
        prepared.append(row)

    solutions = tuple(
        row
        if isinstance(row, _RowSolution)
        else _solve_triclinic_plan(row, basis=basis)
        for row in prepared
    )
    shift_values: object = [solution.shift for solution in solutions]
    if not solutions:
        shift_values = np.empty((0, dimension), dtype=np.int64)
    shifts = _readonly_array(shift_values, dtype=np.int64)
    displacement_float = []
    distance_float = []
    exact_keys = []
    for solution in solutions:
        try:
            displacement_row = tuple(
                _dyadic_float(value, solution.denominator_exponent)
                for value in solution.displacement_numerators
            )
            distance_value = _dyadic_float(
                solution.distance_numerator,
                2 * solution.denominator_exponent,
            )
        except OverflowError:
            raise MinimumImageCertificationError(
                'certified exact geometry has no finite binary64 result view',
                stage='binary64_result_view',
                method=basis.method,
                pair_index=len(displacement_float),
                basis_summary=basis.summary,
            ) from None
        if not all(math.isfinite(value) for value in displacement_row):
            raise MinimumImageCertificationError(
                'certified displacement has no finite binary64 result view',
                stage='binary64_result_view',
                method=basis.method,
                pair_index=len(displacement_float),
                basis_summary=basis.summary,
            )
        if not math.isfinite(distance_value):
            raise MinimumImageCertificationError(
                'certified distance has no finite binary64 result view',
                stage='binary64_result_view',
                method=basis.method,
                pair_index=len(displacement_float),
                basis_summary=basis.summary,
            )
        displacement_float.append(displacement_row)
        distance_float.append(distance_value)
        exact_keys.append(
            ExactDistanceKey(
                numerator=solution.distance_numerator,
                denominator_exponent=2 * solution.denominator_exponent,
            )
        )

    displacement_values: object = displacement_float
    if not solutions:
        displacement_values = np.empty((0, dimension), dtype=np.float64)
    return MinimumImageBatch(
        shift=shifts,
        displacement=_readonly_array(
            displacement_values,
            dtype=np.float64,
        ),
        distance_squared=_readonly_array(distance_float, dtype=np.float64),
        exact_distance_key=tuple(exact_keys),
        method=basis.method,
        candidate_count=_readonly_array(
            [solution.candidate_count for solution in solutions],
            dtype=np.int64,
        ),
        tie_count=_readonly_array(
            [solution.tie_count for solution in solutions],
            dtype=np.int64,
        ),
        seed_count=_readonly_array(
            [solution.seed_count for solution in solutions],
            dtype=np.int64,
        ),
        seed_truncated=_readonly_array(
            [solution.seed_truncated for solution in solutions],
            dtype=np.bool_,
        ),
    )


def exact_distance_less_than(
    key: ExactDistanceKey,
    threshold: float,
) -> bool:
    """Compare one exact squared-distance key with an exact binary64 radius."""

    threshold_value = float(threshold)
    if not math.isfinite(threshold_value) or threshold_value < 0.0:
        raise ValueError('threshold must be a non-negative finite binary64 value')
    numerator, denominator = threshold_value.as_integer_ratio()
    return (
        key.numerator * denominator * denominator
        < numerator * numerator * (1 << key.denominator_exponent)
    )


def exact_distance_squared_less_equal(
    key: ExactDistanceKey,
    threshold_squared: float,
) -> bool:
    """Compare an exact squared-distance key to a binary64 squared limit.

    Unlike :func:`exact_distance_less_than`, this helper is inclusive and its
    argument is already a squared distance.  It exists for the fixed backend
    safety boundary; user radius comparisons keep the established strict
    helper above.
    """

    threshold_value = float(threshold_squared)
    if not math.isfinite(threshold_value) or threshold_value < 0.0:
        raise ValueError(
            'threshold_squared must be a non-negative finite binary64 value'
        )
    numerator, denominator = threshold_value.as_integer_ratio()
    return (
        key.numerator * denominator
        <= numerator * (1 << key.denominator_exponent)
    )
