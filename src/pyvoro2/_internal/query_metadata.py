"""Exact user query views, independent of backend insertion/remapping."""

from __future__ import annotations

from fractions import Fraction
from numbers import Integral

import numpy as np

from .exact_lattice import exact_basis_3d
from .ghost import GhostFailure


def user_frame(geometry, snapshot=None):
    """Return the operation's user rows/origin; spans are binary64 operands."""
    if snapshot is not None:
        return snapshot.vectors, snapshot.origin
    return (np.asarray(geometry.lattice_vectors_cart, dtype=np.float64),
            np.asarray([axis[0] for axis in geometry.native_bounds]))


def exact_query_row(query, lattice, origin, periodic):
    """Return exact q - n A and unbounded n from the exact affine solve."""
    point = tuple(Fraction(float(v)) for v in query)
    delta = tuple(v - Fraction(float(o)) for v, o in zip(point, origin))
    dim = len(point)
    if dim == 3:
        basis = exact_basis_3d(lattice)
        fractional = basis.solve_row(delta)
        rows = basis.rows
    else:
        # The only admitted planar periodic domain is Cartesian rectangular.
        rows = tuple(tuple(Fraction(float(v)) for v in row) for row in lattice)
        fractional = tuple(delta[i] / rows[i][i] for i in range(dim))
    shifts = tuple(v.numerator // v.denominator if periodic[i] else 0
                   for i, v in enumerate(fractional))
    wrapped = tuple(point[j] - sum(shifts[i] * rows[i][j] for i in range(dim))
                    for j in range(dim))
    return wrapped, shifts


def materialize_query_row(query, lattice, origin, periodic, failure, index):
    wrapped, shifts = exact_query_row(query, lattice, origin, periodic)
    if any(not -(2**63) <= s < 2**63 for s in shifts):
        raise failure('Cannot materialize signed int64 query shift',
                      query_index=index, field='query_shift')
    try:
        floats = [float(v) for v in wrapped]
        if not np.isfinite(floats).all():
            raise OverflowError('nonfinite wrapped coordinate')
    except OverflowError as exc:
        raise failure('Cannot materialize finite query view',
                      query_index=index, field='query_wrapped') from exc
    return floats, shifts


def locate_query_views(queries, geometry, snapshot, failure):
    lattice, origin = user_frame(geometry, snapshot)
    wrapped = np.empty_like(queries)
    shifts = np.empty(queries.shape, dtype=np.int64)
    for i, query in enumerate(queries):
        wrapped[i], shifts[i] = materialize_query_row(
            query, lattice, origin, geometry.periodic_axes, failure, i,
        )
    return {'query': queries.copy(), 'query_wrapped': wrapped,
            'query_shift': shifts}


def ghost_query_views(cells, queries, geometry, snapshot, include_empty):
    """Validate all associations, then materialize views only on retained rows."""
    dim = geometry.dim
    if len(cells) != len(queries):
        raise GhostFailure('GHOST_PROVENANCE_INCONSISTENT',
                           'Incomplete native query association',
                           stage='provenance', dimension=dim)
    for i, cell in enumerate(cells):
        index = cell.get('query_index')
        if isinstance(index, (bool, np.bool_)) or not isinstance(index, Integral) \
                or int(index) != i:
            raise GhostFailure('GHOST_PROVENANCE_INCONSISTENT',
                               'Invalid native query association',
                               stage='provenance', query_index=i, dimension=dim)
    retained = cells if include_empty else [c for c in cells if not c['empty']]
    periodic = geometry.has_any_periodic_axis
    if periodic:
        lattice, origin = user_frame(geometry, snapshot)

    def failure(message, **details):
        return GhostFailure('GHOST_SHIFT_UNREPRESENTABLE', message,
                            stage='materialization', dimension=dim, **details)

    for cell in retained:
        i = int(cell['query_index'])
        cell['query'] = queries[i].tolist()
        if periodic:
            cell['query_wrapped'], cell['query_shift'] = materialize_query_row(
                queries[i], lattice, origin, geometry.periodic_axes, failure, i,
            )
    return retained
