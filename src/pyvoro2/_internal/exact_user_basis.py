"""Exact binary64 user-coordinate arithmetic, independent of native admission.

Only rank-three linear/affine operations live here, not lattice search. Bounded
caches own immutable tuples. Constructor validation needs integer determinant
work only; rational inverse construction is lazy and shared across batches.
"""
from __future__ import annotations

from fractions import Fraction
from functools import lru_cache
from typing import Sequence, TypeVar

import numpy as np

from .validation import INT64_MAX, INT64_MIN


_Scalar = TypeVar('_Scalar', int, Fraction)


def _determinant_3x3(matrix: Sequence[Sequence[_Scalar]]) -> _Scalar:
    # Shared unchanged expansion formerly local to periodic_images.
    a, b, c = matrix
    return (
        a[0] * (b[1] * c[2] - b[2] * c[1])
        - a[1] * (b[0] * c[2] - b[2] * c[0])
        + a[2] * (b[0] * c[1] - b[1] * c[0])
    )


@lru_cache(maxsize=128)
def _integer_basis(vectors):
    """Return M, denominator and det(M), with A = M / denominator exactly."""
    ratios = [[value.as_integer_ratio() for value in row] for row in vectors]
    denominator = max(d for row in ratios for _, d in row)
    matrix = tuple(tuple(n * (denominator // d) for n, d in row) for row in ratios)
    return matrix, denominator, _determinant_3x3(matrix)


def determinant_sign(vectors) -> int:
    """Sign of the determinant of canonical finite binary64 row tuples."""
    determinant = _integer_basis(vectors)[2]
    return (determinant > 0) - (determinant < 0)


@lru_cache(maxsize=128)
def _adjugate_basis(vectors):
    """Immutable integer cofactors shared by source solves and local lifting."""
    matrix, denominator, determinant = _integer_basis(vectors)
    if determinant == 0:
        raise ValueError('cell vectors are exactly singular')
    adjugate = []
    for i in range(3):
        row = []
        for j in range(3):
            # Inverse[i,j] is cofactor[j,i] / det(A).
            minor = [[matrix[r][c] for c in range(3) if c != i]
                     for r in range(3) if r != j]
            cofactor = (-1)**(i + j) * (
                minor[0][0] * minor[1][1] - minor[0][1] * minor[1][0])
            row.append(cofactor)
        adjugate.append(tuple(row))
    return tuple(adjugate)


@lru_cache(maxsize=128)
def _inverse_basis(vectors):
    """Exact A inverse by integer cofactors; prepared only for inverse maps."""
    _, denominator, determinant = _integer_basis(vectors)
    return tuple(tuple(Fraction(denominator * c, determinant) for c in row)
                 for row in _adjugate_basis(vectors))


def local_residuals(points, vectors, origin):
    """Yield exact local Cartesian numerators, denominator and wide source lift.

    For each canonical point, k=floor((p-o) A^-1), r=p-o-k A. Retain r as
    integers over one dyadic denominator: rounding it before projection loses
    the once-rounded local-dot contract. No native frame or int64 limit here.
    """
    key = tuple(tuple(float(v) for v in row) for row in vectors)
    matrix, basis_denominator, determinant = _integer_basis(key)
    adjugate = _adjugate_basis(key)
    offsets = tuple(float(v).as_integer_ratio() for v in origin)
    constant_denominator = max(basis_denominator, *(d for _, d in offsets))
    for point in points:
        ratios = tuple(float(v).as_integer_ratio() for v in point)
        denominator = max(constant_denominator, *(d for _, d in ratios))
        delta = [n * (denominator // d) - o * (denominator // od)
                 for (n, d), (o, od) in zip(ratios, offsets)]
        # Python // handles either determinant sign, with no floating inverse.
        shift = tuple(sum(delta[i] * adjugate[i][j] for i in range(3))
                      * basis_denominator // (denominator * determinant)
                      for j in range(3))
        scale = denominator // basis_denominator
        residual = tuple(delta[j] - scale * sum(shift[i] * matrix[i][j]
                                                for i in range(3))
                         for j in range(3))
        yield residual, denominator, shift


def _fraction_rows(rows):
    return tuple(tuple(Fraction(float(value)) for value in row) for row in rows)


def _solve_rows(points, vectors, origin):
    inverse = _inverse_basis(vectors)
    exact_origin = tuple(Fraction(value) for value in origin)
    for point in points:
        delta = [Fraction(float(value)) - offset
                 for value, offset in zip(point, exact_origin)]
        yield tuple(sum(delta[i] * inverse[i][j] for i in range(3))
                    for j in range(3))


def _float_view(rows, count, operation):
    # Allocate independently even for empty input. No partly filled array is
    # exposed if a later coordinate cannot be represented as a finite float.
    result = np.empty((count, 3), dtype=np.float64)
    try:
        for i, row in enumerate(rows):
            result[i] = [float(value) for value in row]
    except OverflowError:
        raise ValueError(f'{operation} requires a finite binary64 view') from None
    if not np.all(np.isfinite(result)):
        raise ValueError(f'{operation} requires a finite binary64 view')
    return result


def cart_to_fractional(points, vectors, origin):
    return _float_view(_solve_rows(points, vectors, origin), len(points),
                       'cart_to_fractional')


def fractional_to_cart(fractional, vectors, origin):
    basis = _fraction_rows(vectors)
    offset = tuple(Fraction(value) for value in origin)
    rows = (tuple(offset[j] + sum(f[i] * basis[i][j] for i in range(3))
                  for j in range(3)) for f in _fraction_rows(fractional))
    return _float_view(rows, len(fractional), 'fractional_to_cart')


def _wrap(rows, count, operation, return_shifts, *, vectors=None, points=None):
    wrapped = []
    shifts = []
    basis = None if vectors is None else _fraction_rows(vectors)
    for index, f in enumerate(rows):
        shift = tuple(value.numerator // value.denominator for value in f)
        if return_shifts:
            if any(value < INT64_MIN or value > INT64_MAX for value in shift):
                raise ValueError(f'{operation} shifts must fit signed int64')
            shifts.append(shift)
        if basis is None:
            wrapped.append(tuple(value - s for value, s in zip(f, shift)))
        else:
            # Use original x exactly, without rounding f or x-o first.
            wrapped.append(tuple(Fraction(float(points[index, j]))
                                 - sum(shift[i] * basis[i][j] for i in range(3))
                                 for j in range(3)))
    view = _float_view(wrapped, count, operation)
    if not return_shifts:
        return view
    exposed = np.empty((count, 3), dtype=np.int64)
    if count:
        exposed[:] = shifts  # All arbitrary-size integers checked before casting.
    return view, exposed


def wrap_fractional(fractional, return_shifts):
    return _wrap(_fraction_rows(fractional), len(fractional), 'wrap_fractional',
                 return_shifts)


def wrap_cart(points, vectors, origin, return_shifts):
    return _wrap(_solve_rows(points, vectors, origin), len(points), 'wrap_cart',
                 return_shifts, vectors=vectors, points=points)
