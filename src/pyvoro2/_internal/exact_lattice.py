"""Small exact-arithmetic primitives for binary64 3D lattices.

This private module deliberately has no native or domain imports.  It treats
each finite binary64 value as its represented dyadic rational number.

The rank-3 reducer uses exact LLL with ``delta = 3/4``.  On aligned integer
rows, size reduction preserves the positive integer product of prefix Gram
determinants, while every strict Lovasz swap decreases that potential.
Together with finite descending size-reduction passes and bounded index motion,
this proves termination; the private step/work limits below are resource guards,
not the mathematical termination argument.

Charged work is a deterministic formula-level score: fixed scalar arithmetic,
comparison, and row-update blocks receive documented-sized charges.  It is not
a count of processor instructions, allocations, or ``Fraction`` internals.
Operand-growth limits observe every explicitly materialized semantic integer or
normalized rational result, including compound subexpressions before a later
cancellation; they likewise do not model temporary integers internal to
Python's rational normalization.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from functools import lru_cache
import math
import operator
import struct
from typing import Callable, Iterable, Sequence

import numpy as np


_CACHE_SIZE = 128
_REDUCTION_METHOD = 'exact-lll-rank3'
_REDUCTION_POLICY = (
    'delta=3/4;source-row-order;descending-full-size-reduction;'
    'nearest-half-ties-toward-zero;strict-lovasz-swap;'
    'first-nonzero-cartesian-positive;no-final-sort;v2'
)
_DELTA = Fraction(3, 4)


IntRow3 = tuple[int, int, int]
IntMatrix3 = tuple[IntRow3, IntRow3, IntRow3]
ExactRow3 = tuple[Fraction, Fraction, Fraction]
ExactMatrix3 = tuple[ExactRow3, ExactRow3, ExactRow3]


@dataclass(frozen=True, slots=True)
class ExactLatticeReductionLimits:
    """Finite private resource policy for exact rank-3 reduction."""

    max_work: int = 1_000_000
    max_steps: int = 100_000
    max_transform_bits: int = 4_096
    max_inverse_transform_bits: int = 4_096
    max_integer_bits: int = 32_768
    max_rational_bits: int = 65_536

    def __post_init__(self) -> None:
        for name in (
            'max_work',
            'max_steps',
            'max_transform_bits',
            'max_inverse_transform_bits',
            'max_integer_bits',
            'max_rational_bits',
        ):
            value = getattr(self, name)
            if isinstance(value, bool):
                raise ValueError(f'{name} must be a positive exact integer')
            try:
                integer = operator.index(value)
            except TypeError:
                raise ValueError(
                    f'{name} must be a positive exact integer'
                ) from None
            if integer <= 0:
                raise ValueError(f'{name} must be a positive exact integer')
            object.__setattr__(self, name, int(integer))


DEFAULT_REDUCTION_LIMITS = ExactLatticeReductionLimits()


class ExactLatticeReductionResourceError(RuntimeError):
    """Raised when a private exact-reduction resource limit is exhausted."""

    def __init__(
        self,
        message: str,
        *,
        stage: str,
        resource: str,
        observed: int,
        configured_limit: int,
        source_summary: dict[str, object],
    ) -> None:
        super().__init__(message)
        self.method = _REDUCTION_METHOD
        self.policy = _REDUCTION_POLICY
        self.stage = stage
        self.resource = resource
        self.observed = int(observed)
        self.configured_limit = int(configured_limit)
        self.source_summary = dict(source_summary)


class ExactLatticeReductionInvariantError(RuntimeError):
    """Raised when independently recomputed exact certification fails."""


@dataclass(frozen=True, slots=True)
class ExactLatticeReductionDiagnostics:
    """Deterministic exact work and operand-growth evidence."""

    steps: int
    swaps: int
    size_reductions: int
    work: int
    certification_work: int
    max_integer_bits: int
    max_rational_bits: int
    max_transform_bits: int
    max_inverse_transform_bits: int


@dataclass(frozen=True, slots=True)
class _ReductionTraceEvent:
    """One immutable test-only view of an actual reducer row operation."""

    kind: str
    row_index: int
    previous_index: int | None
    before: IntMatrix3
    after: IntMatrix3


@dataclass(frozen=True, slots=True)
class ExactReducedBasis3D:
    """Immutable certified exact reduction of one binary64 row basis."""

    reduced_rows: ExactMatrix3
    transform: IntMatrix3
    inverse_transform: IntMatrix3
    method: str
    policy: str
    delta: Fraction
    limits: ExactLatticeReductionLimits
    diagnostics: ExactLatticeReductionDiagnostics
    certified: bool = True

    def map_reduced_to_user(
        self,
        coefficients: Sequence[int],
    ) -> IntRow3:
        """Map ``s_reduced`` to ``s_user = s_reduced @ transform``."""

        row = _coerce_integer_row3(coefficients, name='reduced coefficients')
        return _coefficient_matrix_product(row, self.transform)

    def map_user_to_reduced(
        self,
        coefficients: Sequence[int],
    ) -> IntRow3:
        """Map ``s_user`` to ``s_reduced = s_user @ inverse_transform``."""

        row = _coerce_integer_row3(coefficients, name='user coefficients')
        return _coefficient_matrix_product(row, self.inverse_transform)


def dyadic_parts(value: float) -> tuple[int, int]:
    """Return the integer numerator and power-of-two denominator exponent."""

    numerator, denominator = float(value).as_integer_ratio()
    exponent = denominator.bit_length() - 1
    if denominator != 1 << exponent:
        raise ValueError('binary64 denominator is not a power of two')
    return numerator, exponent


def determinant_3x3(matrix: Sequence[Sequence[Fraction]]) -> Fraction:
    """Return the exact determinant of one 3-by-3 rational matrix."""

    a, b, c = matrix
    return (
        a[0] * (b[1] * c[2] - b[2] * c[1])
        - a[1] * (b[0] * c[2] - b[2] * c[0])
        + a[2] * (b[0] * c[1] - b[1] * c[0])
    )


def inverse_fraction_matrix(
    matrix: Sequence[Sequence[Fraction]],
) -> tuple[tuple[Fraction, ...], ...]:
    """Return an exact rational inverse using deterministic pivoting."""

    dimension = len(matrix)
    work = [
        [Fraction(value) for value in row]
        + [Fraction(int(i == j)) for j in range(dimension)]
        for i, row in enumerate(matrix)
    ]
    for column in range(dimension):
        pivot = next(
            (row for row in range(column, dimension) if work[row][column]),
            None,
        )
        if pivot is None:
            raise ValueError('lattice is exactly singular')
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


@dataclass(frozen=True, slots=True)
class ExactBasis3D:
    """Immutable determinant and inverse data for one binary64 basis."""

    rows: tuple[tuple[Fraction, Fraction, Fraction], ...]
    determinant: Fraction
    adjugate: tuple[tuple[Fraction, Fraction, Fraction], ...]

    @property
    def determinant_sign(self) -> int:
        return 1 if self.determinant > 0 else -1

    def solve_row(
        self,
        values: tuple[Fraction, Fraction, Fraction],
    ) -> tuple[Fraction, Fraction, Fraction]:
        det = self.determinant
        return tuple(
            sum(values[index] * self.adjugate[index][column]
                for index in range(3)) / det
            for column in range(3)
        )  # type: ignore[return-value]

    def subtract_lattice_shift(
        self,
        point: tuple[Fraction, Fraction, Fraction],
        shifts: tuple[int, int, int],
    ) -> tuple[Fraction, Fraction, Fraction]:
        """Return ``point - shifts @ rows`` in exact source arithmetic."""

        return tuple(
            point[column]
            - sum(shifts[index] * self.rows[index][column]
                  for index in range(3))
            for column in range(3)
        )  # type: ignore[return-value]


def _bits(value: float) -> int:
    return struct.unpack('>Q', struct.pack('>d', float(value)))[0]


def _from_bits(bits: int) -> float:
    return struct.unpack('>d', struct.pack('>Q', bits))[0]


@lru_cache(maxsize=_CACHE_SIZE)
def _basis_from_bits(bit_patterns: tuple[int, ...]) -> ExactBasis3D:
    values = tuple(Fraction.from_float(_from_bits(bits))
                   for bits in bit_patterns)
    rows = tuple(
        tuple(values[row * 3 + column] for column in range(3))
        for row in range(3)
    )
    determinant = determinant_3x3(rows)
    if determinant == 0:
        raise ValueError('cell vectors are exactly singular')
    a, b, c = rows
    adjugate = (
        (
            b[1] * c[2] - b[2] * c[1],
            a[2] * c[1] - a[1] * c[2],
            a[1] * b[2] - a[2] * b[1],
        ),
        (
            b[2] * c[0] - b[0] * c[2],
            a[0] * c[2] - a[2] * c[0],
            a[2] * b[0] - a[0] * b[2],
        ),
        (
            b[0] * c[1] - b[1] * c[0],
            a[1] * c[0] - a[0] * c[1],
            a[0] * b[1] - a[1] * b[0],
        ),
    )
    return ExactBasis3D(rows=rows, determinant=determinant,
                        adjugate=adjugate)


def exact_basis_3d(matrix: np.ndarray) -> ExactBasis3D:
    """Return cached exact data for a finite float64 ``(3, 3)`` matrix."""

    array = np.asarray(matrix, dtype=np.float64)
    return _basis_from_bits(tuple(_bits(value) for value in array.flat))


def exact_point(values: np.ndarray) -> tuple[Fraction, Fraction, Fraction]:
    """Convert one finite binary64 point into represented exact rationals."""

    return tuple(
        Fraction.from_float(float(value)) for value in values
    )  # type: ignore[return-value]


def finite_float_view(value: Fraction, *, operation: str) -> float:
    """Return the nearest-even binary64 view or fail explicitly."""

    try:
        result = float(value)
    except OverflowError as exc:
        raise ValueError(
            f'{operation} result has no finite binary64 view'
        ) from exc
    if not np.isfinite(result):
        raise ValueError(f'{operation} result has no finite binary64 view')
    return result


@dataclass(slots=True)
class _ReductionMonitor:
    limits: ExactLatticeReductionLimits
    source_summary: dict[str, object]
    steps: int = 0
    swaps: int = 0
    size_reductions: int = 0
    work: int = 0
    certification_work: int = 0
    max_integer_bits: int = 0
    max_rational_bits: int = 0
    max_transform_bits: int = 0
    max_inverse_transform_bits: int = 0

    def _fail(
        self,
        *,
        stage: str,
        resource: str,
        observed: int,
        limit: int,
    ) -> None:
        raise ExactLatticeReductionResourceError(
            f'exact lattice reduction exceeded {resource} during {stage}: '
            f'observed={observed}, limit={limit}',
            stage=stage,
            resource=resource,
            observed=observed,
            configured_limit=limit,
            source_summary=self.source_summary,
        )

    def charge(
        self,
        amount: int,
        *,
        stage: str,
        certification: bool = False,
    ) -> None:
        self.work += amount
        if certification:
            self.certification_work += amount
        if self.work > self.limits.max_work:
            self._fail(
                stage=stage,
                resource='work',
                observed=self.work,
                limit=self.limits.max_work,
            )

    def step(self, *, stage: str) -> None:
        self.steps += 1
        if self.steps > self.limits.max_steps:
            self._fail(
                stage=stage,
                resource='steps',
                observed=self.steps,
                limit=self.limits.max_steps,
            )

    def observe_integer(
        self,
        value: int,
        *,
        stage: str,
        role: str = 'integer',
    ) -> None:
        bits = abs(int(value)).bit_length()
        self.max_integer_bits = max(self.max_integer_bits, bits)
        if bits > self.limits.max_integer_bits:
            self._fail(
                stage=stage,
                resource='integer_bits',
                observed=bits,
                limit=self.limits.max_integer_bits,
            )
        if role == 'transform':
            self.max_transform_bits = max(self.max_transform_bits, bits)
            if bits > self.limits.max_transform_bits:
                self._fail(
                    stage=stage,
                    resource='transform_bits',
                    observed=bits,
                    limit=self.limits.max_transform_bits,
                )
        elif role == 'inverse_transform':
            self.max_inverse_transform_bits = max(
                self.max_inverse_transform_bits, bits
            )
            if bits > self.limits.max_inverse_transform_bits:
                self._fail(
                    stage=stage,
                    resource='inverse_transform_bits',
                    observed=bits,
                    limit=self.limits.max_inverse_transform_bits,
                )

    def observe_fraction(self, value: Fraction, *, stage: str) -> None:
        bits = max(
            abs(value.numerator).bit_length(),
            value.denominator.bit_length(),
        )
        self.max_rational_bits = max(self.max_rational_bits, bits)
        if bits > self.limits.max_rational_bits:
            self._fail(
                stage=stage,
                resource='rational_bits',
                observed=bits,
                limit=self.limits.max_rational_bits,
            )

    def diagnostics(self) -> ExactLatticeReductionDiagnostics:
        return ExactLatticeReductionDiagnostics(
            steps=self.steps,
            swaps=self.swaps,
            size_reductions=self.size_reductions,
            work=self.work,
            certification_work=self.certification_work,
            max_integer_bits=self.max_integer_bits,
            max_rational_bits=self.max_rational_bits,
            max_transform_bits=self.max_transform_bits,
            max_inverse_transform_bits=self.max_inverse_transform_bits,
        )


def _coerce_integer_row3(
    values: Sequence[int],
    *,
    name: str,
) -> IntRow3:
    if len(values) != 3:
        raise ValueError(f'{name} must contain exactly three integers')
    result = []
    for index, value in enumerate(values):
        if isinstance(value, (bool, np.bool_)):
            raise ValueError(f'{name}[{index}] must be an exact integer')
        try:
            result.append(int(operator.index(value)))
        except TypeError:
            raise ValueError(
                f'{name}[{index}] must be an exact integer'
            ) from None
    return tuple(result)  # type: ignore[return-value]


def _coefficient_matrix_product(
    coefficients: IntRow3,
    matrix: IntMatrix3,
) -> IntRow3:
    return tuple(
        sum(coefficients[index] * matrix[index][column]
            for index in range(3))
        for column in range(3)
    )  # type: ignore[return-value]


def _integer_determinant_3x3(
    matrix: Sequence[Sequence[int]],
    *,
    monitor: _ReductionMonitor,
    stage: str,
) -> int:
    a, b, c = matrix
    products = (
        b[1] * c[2],
        b[2] * c[1],
        b[0] * c[2],
        b[2] * c[0],
        b[0] * c[1],
        b[1] * c[0],
    )
    for value in products:
        monitor.observe_integer(value, stage=stage)
    minors = (
        products[0] - products[1],
        products[2] - products[3],
        products[4] - products[5],
    )
    for value in minors:
        monitor.observe_integer(value, stage=stage)
    terms = tuple(a[index] * minors[index] for index in range(3))
    for value in terms:
        monitor.observe_integer(value, stage=stage)
    partial = terms[0] - terms[1]
    monitor.observe_integer(partial, stage=stage)
    determinant = partial + terms[2]
    monitor.observe_integer(determinant, stage=stage)
    return determinant


def _integer_inverse_3x3(
    matrix: IntMatrix3,
    *,
    monitor: _ReductionMonitor,
) -> IntMatrix3:
    determinant = _integer_determinant_3x3(
        matrix,
        monitor=monitor,
        stage='inverse_transform',
    )
    monitor.charge(18, stage='inverse_transform')
    if determinant not in (-1, 1):
        raise ExactLatticeReductionInvariantError(
            'exact LLL transform is not unimodular'
        )
    a, b, c = matrix

    def minor(
        left_a: int,
        left_b: int,
        right_a: int,
        right_b: int,
    ) -> int:
        left = left_a * left_b
        right = right_a * right_b
        monitor.observe_integer(left, stage='inverse_transform')
        monitor.observe_integer(right, stage='inverse_transform')
        value = left - right
        monitor.observe_integer(value, stage='inverse_transform')
        return value

    adjugate = (
        (
            minor(b[1], c[2], b[2], c[1]),
            minor(a[2], c[1], a[1], c[2]),
            minor(a[1], b[2], a[2], b[1]),
        ),
        (
            minor(b[2], c[0], b[0], c[2]),
            minor(a[0], c[2], a[2], c[0]),
            minor(a[2], b[0], a[0], b[2]),
        ),
        (
            minor(b[0], c[1], b[1], c[0]),
            minor(a[1], c[0], a[0], c[1]),
            minor(a[0], b[1], a[1], b[0]),
        ),
    )
    result = tuple(
        tuple(value // determinant for value in row) for row in adjugate
    )
    for row in result:
        for value in row:
            monitor.observe_integer(
                value,
                stage='inverse_transform',
                role='inverse_transform',
            )
    return result  # type: ignore[return-value]


def _integer_matrix_product(
    left: IntMatrix3,
    right: IntMatrix3,
    *,
    monitor: _ReductionMonitor,
    stage: str,
) -> IntMatrix3:
    result_rows = []
    for row in range(3):
        result_row = []
        for column in range(3):
            total = 0
            for index in range(3):
                product = left[row][index] * right[index][column]
                monitor.observe_integer(product, stage=stage)
                total += product
                monitor.observe_integer(total, stage=stage)
            result_row.append(total)
        result_rows.append(tuple(result_row))
    result = tuple(result_rows)
    monitor.charge(45, stage=stage, certification=True)
    return result  # type: ignore[return-value]


def _observed_fraction_sum(
    values: Iterable[Fraction],
    *,
    monitor: _ReductionMonitor,
    stage: str,
) -> Fraction:
    total = Fraction()
    for value in values:
        monitor.observe_fraction(value, stage=stage)
        total += value
        monitor.observe_fraction(total, stage=stage)
    return total


def _transform_exact_rows(
    transform: IntMatrix3,
    source: ExactMatrix3,
    *,
    monitor: _ReductionMonitor,
    stage: str,
    certification: bool,
) -> ExactMatrix3:
    result = tuple(
        tuple(
            _observed_fraction_sum(
                (transform[row][index] * source[index][column]
                 for index in range(3)),
                monitor=monitor,
                stage=stage,
            )
            for column in range(3)
        )
        for row in range(3)
    )
    monitor.charge(45, stage=stage, certification=certification)
    for row in result:
        for value in row:
            monitor.observe_fraction(value, stage=stage)
    return result  # type: ignore[return-value]


def _vector_gram_schmidt(
    rows: Sequence[Sequence[int]],
    *,
    monitor: _ReductionMonitor,
) -> tuple[
    tuple[tuple[Fraction, ...], ...],
    tuple[tuple[Fraction, ...], ...],
    tuple[Fraction, ...],
]:
    """Reducer-side exact Gram--Schmidt on the current integer rows."""

    stars: list[tuple[Fraction, ...]] = []
    coefficients: list[list[Fraction]] = [[], [], []]
    squared: list[Fraction] = []
    for row_index, row in enumerate(rows):
        star = [Fraction(value) for value in row]
        for previous in range(row_index):
            numerator = _observed_fraction_sum(
                (
                    Fraction(row[column]) * stars[previous][column]
                    for column in range(3)
                ),
                monitor=monitor,
                stage='gram_schmidt',
            )
            coefficient = numerator / squared[previous]
            monitor.charge(8, stage='gram_schmidt')
            monitor.observe_fraction(numerator, stage='gram_schmidt')
            monitor.observe_fraction(coefficient, stage='gram_schmidt')
            coefficients[row_index].append(coefficient)
            updated_star = []
            for column in range(3):
                product = coefficient * stars[previous][column]
                monitor.observe_fraction(product, stage='gram_schmidt')
                value = star[column] - product
                monitor.observe_fraction(value, stage='gram_schmidt')
                updated_star.append(value)
            star = updated_star
            for value in star:
                monitor.observe_fraction(value, stage='gram_schmidt')
        star_tuple = tuple(star)
        norm = _observed_fraction_sum(
            (value * value for value in star_tuple),
            monitor=monitor,
            stage='gram_schmidt',
        )
        monitor.charge(8, stage='gram_schmidt')
        monitor.observe_fraction(norm, stage='gram_schmidt')
        if norm <= 0:
            raise ExactLatticeReductionInvariantError(
                'exact Gram--Schmidt produced a non-positive norm'
            )
        stars.append(star_tuple)
        squared.append(norm)
    return (
        tuple(stars),
        tuple(tuple(row) for row in coefficients),
        tuple(squared),
    )


def _nearest_integer_ties_toward_zero(
    value: Fraction,
    *,
    monitor: _ReductionMonitor,
) -> int:
    lower = math.floor(value)
    remainder = value - lower
    monitor.observe_integer(lower, stage='size_reduction_rounding')
    monitor.observe_fraction(remainder, stage='size_reduction_rounding')
    half = Fraction(1, 2)
    if remainder < half:
        nearest = lower
    elif remainder > half:
        nearest = lower + 1
    else:
        nearest = lower if value > 0 else lower + 1
    monitor.observe_integer(nearest, stage='size_reduction_rounding')
    return nearest


def _first_nonzero_positive(row: Sequence[Fraction | int]) -> bool:
    for value in row:
        if value:
            return value > 0
    raise ExactLatticeReductionInvariantError('reduced basis contains a zero row')


def _certificate_gram_schmidt(
    rows: ExactMatrix3,
    *,
    monitor: _ReductionMonitor,
) -> tuple[tuple[tuple[Fraction, ...], ...], tuple[Fraction, ...]]:
    """Independently reconstruct GS data from the exact Gram matrix."""

    gram = tuple(
        tuple(
            _observed_fraction_sum(
                (left * right for left, right in zip(row, other)),
                monitor=monitor,
                stage='certificate_gram',
            )
            for other in rows
        )
        for row in rows
    )
    monitor.charge(45, stage='certificate_gram', certification=True)
    for row in gram:
        for value in row:
            monitor.observe_fraction(value, stage='certificate_gram')

    coefficients = [[Fraction() for _ in range(3)] for _ in range(3)]
    squared = [Fraction() for _ in range(3)]
    for row in range(3):
        for column in range(row):
            terms = []
            for previous in range(column):
                coefficient_product = (
                    coefficients[row][previous]
                    * coefficients[column][previous]
                )
                monitor.observe_fraction(
                    coefficient_product, stage='certificate_gram'
                )
                term = coefficient_product * squared[previous]
                monitor.observe_fraction(term, stage='certificate_gram')
                terms.append(term)
            correction = _observed_fraction_sum(
                terms,
                monitor=monitor,
                stage='certificate_gram',
            )
            numerator = gram[row][column] - correction
            monitor.observe_fraction(numerator, stage='certificate_gram')
            coefficients[row][column] = numerator / squared[column]
            monitor.charge(6 + 5 * column,
                           stage='certificate_gram', certification=True)
            monitor.observe_fraction(
                coefficients[row][column], stage='certificate_gram'
            )
        terms = []
        for previous in range(row):
            coefficient_squared = coefficients[row][previous] ** 2
            monitor.observe_fraction(
                coefficient_squared, stage='certificate_gram'
            )
            term = coefficient_squared * squared[previous]
            monitor.observe_fraction(term, stage='certificate_gram')
            terms.append(term)
        correction = _observed_fraction_sum(
            terms,
            monitor=monitor,
            stage='certificate_gram',
        )
        squared[row] = gram[row][row] - correction
        monitor.charge(4 + 4 * row,
                       stage='certificate_gram', certification=True)
        monitor.observe_fraction(squared[row], stage='certificate_gram')
    return tuple(tuple(row) for row in coefficients), tuple(squared)


def _certify_reduction(
    *,
    source: ExactMatrix3,
    reduced: ExactMatrix3,
    transform: IntMatrix3,
    inverse_transform: IntMatrix3,
    monitor: _ReductionMonitor,
) -> None:
    """Recompute every mathematical postcondition in normal execution."""

    if any(
        isinstance(value, bool) or not isinstance(value, int)
        for matrix in (transform, inverse_transform)
        for row in matrix
        for value in row
    ):
        raise ExactLatticeReductionInvariantError(
            'reduction transforms are not exact integer matrices'
        )
    for row in transform:
        for value in row:
            monitor.observe_integer(
                value,
                stage='certificate_transform_limits',
                role='transform',
            )
    for row in inverse_transform:
        for value in row:
            monitor.observe_integer(
                value,
                stage='certificate_inverse_transform_limits',
                role='inverse_transform',
            )
    expected = _transform_exact_rows(
        transform,
        source,
        monitor=monitor,
        stage='certificate_reconstruction',
        certification=True,
    )
    if expected != reduced:
        raise ExactLatticeReductionInvariantError(
            'reduced basis does not equal transform @ source basis'
        )
    determinant = _integer_determinant_3x3(
        transform,
        monitor=monitor,
        stage='certificate_unimodularity',
    )
    monitor.charge(18, stage='certificate_unimodularity', certification=True)
    if determinant not in (-1, 1):
        raise ExactLatticeReductionInvariantError(
            'reduction transform determinant is not +/-1'
        )
    identity: IntMatrix3 = ((1, 0, 0), (0, 1, 0), (0, 0, 1))
    if _integer_matrix_product(
        transform,
        inverse_transform,
        monitor=monitor,
        stage='certificate_left_inverse',
    ) != identity:
        raise ExactLatticeReductionInvariantError(
            'transform @ inverse_transform is not identity'
        )
    if _integer_matrix_product(
        inverse_transform,
        transform,
        monitor=monitor,
        stage='certificate_right_inverse',
    ) != identity:
        raise ExactLatticeReductionInvariantError(
            'inverse_transform @ transform is not identity'
        )

    coefficients, squared = _certificate_gram_schmidt(
        reduced,
        monitor=monitor,
    )
    if any(value <= 0 for value in squared):
        raise ExactLatticeReductionInvariantError(
            'reduced basis has a non-positive Gram--Schmidt squared norm'
        )
    for row in range(1, 3):
        for column in range(row):
            if abs(coefficients[row][column]) > Fraction(1, 2):
                raise ExactLatticeReductionInvariantError(
                    'reduced basis is not fully size-reduced'
                )
    for row in range(1, 3):
        coefficient_squared = coefficients[row][row - 1] ** 2
        monitor.charge(
            1,
            stage='certificate_lovasz',
            certification=True,
        )
        monitor.observe_fraction(
            coefficient_squared, stage='certificate_lovasz'
        )
        factor = _DELTA - coefficient_squared
        monitor.charge(
            1,
            stage='certificate_lovasz',
            certification=True,
        )
        monitor.observe_fraction(factor, stage='certificate_lovasz')
        right = factor * squared[row - 1]
        monitor.charge(
            1,
            stage='certificate_lovasz',
            certification=True,
        )
        monitor.observe_fraction(right, stage='certificate_lovasz')
        if squared[row] < right:
            raise ExactLatticeReductionInvariantError(
                'reduced basis violates an exact Lovasz condition'
            )
    if not all(_first_nonzero_positive(row) for row in reduced):
        raise ExactLatticeReductionInvariantError(
            'reduced basis violates the final row-sign convention'
        )
    if monitor.work < monitor.certification_work:
        raise ExactLatticeReductionInvariantError(
            'reduction resource accounting is internally inconsistent'
        )


def _source_from_bits(bit_patterns: tuple[int, ...]) -> ExactMatrix3:
    values = tuple(
        Fraction.from_float(_from_bits(bits)) for bits in bit_patterns
    )
    return tuple(
        tuple(values[row * 3 + column] for column in range(3))
        for row in range(3)
    )  # type: ignore[return-value]


def _aligned_integer_rows(
    source: ExactMatrix3,
    *,
    monitor: _ReductionMonitor,
) -> tuple[list[list[int]], int]:
    exponent = max(
        value.denominator.bit_length() - 1
        for row in source
        for value in row
    )
    rows: list[list[int]] = []
    for row in source:
        integer_row = []
        for value in row:
            value_exponent = value.denominator.bit_length() - 1
            if value.denominator != 1 << value_exponent:
                raise ExactLatticeReductionInvariantError(
                    'binary64 source did not produce a dyadic rational'
                )
            integer = value.numerator << (exponent - value_exponent)
            monitor.observe_integer(integer, stage='source_alignment')
            integer_row.append(integer)
        rows.append(integer_row)
    monitor.charge(18, stage='source_alignment')
    return rows, exponent


def _as_int_matrix(rows: Sequence[Sequence[int]]) -> IntMatrix3:
    return tuple(  # type: ignore[return-value]
        tuple(int(value) for value in row)
        for row in rows
    )


def _reduce_exact_lll(
    source: ExactMatrix3,
    *,
    monitor: _ReductionMonitor,
    operation_observer: Callable[[_ReductionTraceEvent], None] | None = None,
) -> tuple[ExactMatrix3, IntMatrix3, IntMatrix3]:
    rows, _exponent = _aligned_integer_rows(source, monitor=monitor)
    source_determinant = _integer_determinant_3x3(
        rows,
        monitor=monitor,
        stage='source_determinant',
    )
    monitor.charge(18, stage='source_determinant')
    if source_determinant == 0:
        raise ValueError('cell vectors are exactly singular')
    transform = [[1 if row == column else 0 for column in range(3)]
                 for row in range(3)]
    for row in transform:
        for value in row:
            monitor.observe_integer(
                value, stage='transform_initialization', role='transform'
            )

    row_index = 1
    while row_index < 3:
        monitor.step(stage='lll_iteration')
        for previous in range(row_index - 1, -1, -1):
            _stars, coefficients, _squared = _vector_gram_schmidt(
                rows,
                monitor=monitor,
            )
            nearest = _nearest_integer_ties_toward_zero(
                coefficients[row_index][previous],
                monitor=monitor,
            )
            monitor.charge(2, stage='size_reduction_decision')
            if nearest == 0:
                continue
            if operation_observer is not None:
                before = _as_int_matrix(rows)
            updated_row = []
            updated_transform = []
            for column in range(3):
                product = nearest * rows[previous][column]
                monitor.observe_integer(product, stage='size_reduction_update')
                updated_row.append(rows[row_index][column] - product)
                transform_product = nearest * transform[previous][column]
                monitor.observe_integer(
                    transform_product, stage='size_reduction_update'
                )
                updated_transform.append(
                    transform[row_index][column] - transform_product
                )
            rows[row_index] = updated_row
            transform[row_index] = updated_transform
            monitor.size_reductions += 1
            monitor.charge(18, stage='size_reduction_update')
            for value in rows[row_index]:
                monitor.observe_integer(value, stage='size_reduction_update')
            for value in transform[row_index]:
                monitor.observe_integer(
                    value,
                    stage='size_reduction_update',
                    role='transform',
                )
            if operation_observer is not None:
                operation_observer(
                    _ReductionTraceEvent(
                        kind='size_reduction',
                        row_index=row_index,
                        previous_index=previous,
                        before=before,
                        after=_as_int_matrix(rows),
                    )
                )

        _stars, coefficients, squared = _vector_gram_schmidt(
            rows,
            monitor=monitor,
        )
        coefficient_squared = coefficients[row_index][row_index - 1] ** 2
        monitor.observe_fraction(
            coefficient_squared, stage='lovasz_decision'
        )
        factor = _DELTA - coefficient_squared
        monitor.observe_fraction(factor, stage='lovasz_decision')
        right = factor * squared[row_index - 1]
        monitor.observe_fraction(right, stage='lovasz_decision')
        monitor.charge(4, stage='lovasz_decision')
        if squared[row_index] < right:
            if operation_observer is not None:
                before = _as_int_matrix(rows)
            rows[row_index], rows[row_index - 1] = (
                rows[row_index - 1], rows[row_index]
            )
            transform[row_index], transform[row_index - 1] = (
                transform[row_index - 1], transform[row_index]
            )
            monitor.swaps += 1
            monitor.charge(1, stage='lovasz_swap')
            if operation_observer is not None:
                operation_observer(
                    _ReductionTraceEvent(
                        kind='swap',
                        row_index=row_index,
                        previous_index=row_index - 1,
                        before=before,
                        after=_as_int_matrix(rows),
                    )
                )
            row_index = max(row_index - 1, 1)
        else:
            row_index += 1

    for row_index, row in enumerate(rows):
        if not _first_nonzero_positive(row):
            if operation_observer is not None:
                before = _as_int_matrix(rows)
            rows[row_index] = [-value for value in row]
            transform[row_index] = [
                -value for value in transform[row_index]
            ]
            monitor.charge(6, stage='sign_normalization')
            for value in transform[row_index]:
                monitor.observe_integer(
                    value,
                    stage='sign_normalization',
                    role='transform',
                )
            if operation_observer is not None:
                operation_observer(
                    _ReductionTraceEvent(
                        kind='sign_normalization',
                        row_index=row_index,
                        previous_index=None,
                        before=before,
                        after=_as_int_matrix(rows),
                    )
                )

    transform_result = _as_int_matrix(transform)
    reduced = _transform_exact_rows(
        transform_result,
        source,
        monitor=monitor,
        stage='reduced_basis_materialization',
        certification=False,
    )
    inverse_transform = _integer_inverse_3x3(
        transform_result,
        monitor=monitor,
    )
    _certify_reduction(
        source=source,
        reduced=reduced,
        transform=transform_result,
        inverse_transform=inverse_transform,
        monitor=monitor,
    )
    return reduced, transform_result, inverse_transform


def _uncached_reduction_from_bits(
    bit_patterns: tuple[int, ...],
    policy: str,
    limits: ExactLatticeReductionLimits,
    *,
    operation_observer: Callable[[_ReductionTraceEvent], None] | None = None,
) -> ExactReducedBasis3D:
    if policy != _REDUCTION_POLICY:
        raise ExactLatticeReductionInvariantError(
            'unknown exact lattice reduction policy identity'
        )
    source_summary = {
        'lattice_bits': tuple(f'{value:016x}' for value in bit_patterns),
        'policy': policy,
    }
    monitor = _ReductionMonitor(
        limits=limits,
        source_summary=source_summary,
    )
    source = _source_from_bits(bit_patterns)
    reduced, transform, inverse_transform = _reduce_exact_lll(
        source,
        monitor=monitor,
        operation_observer=operation_observer,
    )
    return ExactReducedBasis3D(
        reduced_rows=reduced,
        transform=transform,
        inverse_transform=inverse_transform,
        method=_REDUCTION_METHOD,
        policy=policy,
        delta=_DELTA,
        limits=limits,
        diagnostics=monitor.diagnostics(),
    )


@lru_cache(maxsize=_CACHE_SIZE)
def _reduction_from_bits(
    bit_patterns: tuple[int, ...],
    policy: str,
    limits: ExactLatticeReductionLimits,
) -> ExactReducedBasis3D:
    return _uncached_reduction_from_bits(bit_patterns, policy, limits)


def _validated_reduction_bits(
    matrix: Sequence[Sequence[float]] | np.ndarray,
) -> tuple[int, ...]:
    try:
        array = np.asarray(matrix, dtype=np.float64)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(
            'matrix must be a finite binary64 (3, 3) array'
        ) from exc
    if array.shape != (3, 3):
        raise ValueError('matrix must have shape (3, 3)')
    if not np.all(np.isfinite(array)):
        raise ValueError('matrix must contain only finite binary64 values')
    contiguous = np.ascontiguousarray(array, dtype=np.float64)
    return tuple(int(value) for value in contiguous.view(np.uint64).ravel())


def exact_lll_reduce_3d(
    matrix: Sequence[Sequence[float]] | np.ndarray,
    *,
    limits: ExactLatticeReductionLimits = DEFAULT_REDUCTION_LIMITS,
) -> ExactReducedBasis3D:
    """Return the certified deterministic exact rank-3 LLL reduction.

    The input is an ordered binary64 row basis.  The exact result remains a
    rational object and is never materialized through a derived float matrix.
    """

    if not isinstance(limits, ExactLatticeReductionLimits):
        raise ValueError('limits must be an ExactLatticeReductionLimits value')
    bit_patterns = _validated_reduction_bits(matrix)
    return _reduction_from_bits(bit_patterns, _REDUCTION_POLICY, limits)


def _trace_exact_lll_reduce_3d(
    matrix: Sequence[Sequence[float]] | np.ndarray,
    *,
    limits: ExactLatticeReductionLimits = DEFAULT_REDUCTION_LIMITS,
) -> tuple[ExactReducedBasis3D, tuple[_ReductionTraceEvent, ...]]:
    """Run uncached reduction with a test-only actual-row-operation trace."""

    if not isinstance(limits, ExactLatticeReductionLimits):
        raise ValueError('limits must be an ExactLatticeReductionLimits value')
    events: list[_ReductionTraceEvent] = []
    result = _uncached_reduction_from_bits(
        _validated_reduction_bits(matrix),
        _REDUCTION_POLICY,
        limits,
        operation_observer=events.append,
    )
    return result, tuple(events)


def _reduction_cache_info():
    """Return private reducer cache statistics for deterministic tests."""

    return _reduction_from_bits.cache_info()


def _reduction_cache_clear() -> None:
    """Clear the bounded private successful-reduction cache."""

    _reduction_from_bits.cache_clear()
