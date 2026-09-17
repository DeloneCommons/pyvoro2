"""Small exact-arithmetic primitives for binary64 3D lattices.

This private module deliberately has no native or domain imports.  It treats
each finite binary64 value as its represented dyadic rational number.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from functools import lru_cache
import struct
from typing import Sequence

import numpy as np


_CACHE_SIZE = 128


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
        list(row) + [Fraction(int(i == j)) for j in range(dimension)]
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
