"""Locate source integrity, producer enclosure and public result packaging."""

from __future__ import annotations

from dataclasses import replace
from fractions import Fraction as F

import numpy as np

from .ghost import _bounded
from .native_translation import (
    CartesianCompatibilityBox, DEFAULT_NATIVE_TRANSLATION_LIMITS,
    NativeTranslationAmbiguityError, NativeTranslationInconsistencyError,
    NativeTranslationInvariantError, NativeTranslationResourceError,
    certify_native_translation,
)
from .query_metadata import locate_query_views, user_frame


_LIMITS = DEFAULT_NATIVE_TRANSLATION_LIMITS
_MAX_OWNER_QUERIES = 4096
_MAX_BATCH_CANDIDATES = 16_000_000
_CODES = frozenset({
    'LOCATE_BACKEND_INSERTION', 'LOCATE_NATIVE_UNSUPPORTED',
    'LOCATE_PROVENANCE_AMBIGUOUS', 'LOCATE_PROVENANCE_INCONSISTENT',
    'LOCATE_CERTIFICATION_RESOURCE', 'LOCATE_METADATA_UNREPRESENTABLE',
})


class LocateFailure(ValueError):
    """Private class implementing the provisional inspectable failure protocol."""

    def __init__(self, code, message, *, stage, query_index=None, **details):
        if code not in _CODES:
            raise ValueError('unknown locate failure code')
        self.code, self.stage, self.query_index = code, stage, query_index
        self.details = _bounded(details)
        super().__init__(f'{code} [{stage}]: {str(message)[:512]}')


def _materialization(message, **details):
    return LocateFailure('LOCATE_METADATA_UNREPRESENTABLE', message,
                         stage='materialization', **details)


def _inconsistent(message, index=None, **details):
    return LocateFailure('LOCATE_PROVENANCE_INCONSISTENT', message,
                         stage='provenance', query_index=index, **details)


def _native_failure(exc):
    parts = str(exc).split(':', 4)
    if len(parts) == 5 and parts[0] == 'locate_native' and parts[3] in _CODES:
        index = None if parts[2] == 'None' else int(parts[2])
        return LocateFailure(parts[3], parts[4], stage=parts[1],
                             query_index=index, reason=parts[4])
    return None


def _validated_native(result, n, m, dim, certificate, requested):
    if not isinstance(result, tuple) or len(result) != (4 if certificate else 3):
        raise _inconsistent('Malformed native result tuple')
    found, ids, positions = (np.asarray(v) for v in result[:3])
    if (found.shape != (m,) or found.dtype.kind != 'b' or ids.shape != (m,)
            or ids.dtype.kind not in 'iu' or positions.shape != (m, dim)
            or positions.dtype != np.float64):
        raise _inconsistent('Malformed native array shape/type')
    invalid = (found & ((ids < 0) | (ids >= n))) | (~found & (ids != -1))
    if invalid.any():
        raise _inconsistent('Native found/owner identity mismatch',
                            int(np.flatnonzero(invalid)[0]))
    invalid = found & ~np.isfinite(positions).all(axis=1) if requested else \
        np.zeros(m, dtype=bool)
    if invalid.any():
        raise _materialization('Nonfinite native owner position',
                               query_index=int(np.flatnonzero(invalid)[0]),
                               field='owner_pos')
    return found.copy(), ids.copy(), positions.copy()


def _source_packet(packet, n, m, dim):
    if not isinstance(packet, dict) or packet.get('schema') != 'locate-source-v1':
        raise _inconsistent('Missing source packet')
    if packet.get('fp_policy') != 'binary64-noncontracting-v1':
        raise LocateFailure('LOCATE_NATIVE_UNSUPPORTED', 'Unqualified producer',
                            stage='profile')
    arrays = {}
    for name, shape, kind in (
        ('stored', (n, dim), 'f'), ('insertion_shifts', (n, dim), 'iu'),
        ('query_removals', (m, dim), 'iu'), ('lattice', (dim, dim), 'f'),
        ('copy_bounds', (dim,), 'iu'),
    ):
        value = np.asarray(packet.get(name))
        if (value.shape != shape or value.dtype.kind not in kind
                or not np.isfinite(value).all()
                or (kind == 'f' and value.dtype != np.float64)):
            raise _inconsistent('Invalid source packet field', field=name)
        arrays[name] = value
    if (arrays['copy_bounds'] < 0).any():
        raise _inconsistent('Negative producer coefficient bound')
    return arrays


def _exact_rows(values):
    return tuple(tuple(F(float(v)) for v in row) for row in values)


def owner_enclosure(*, original, preparation, stored, insertion, query_removal,
                    copy_bounds, lattice, native_lattice, snapshot, periodic):
    """Complete non-circular Cartesian translation box for WP4.

    Native image construction and final position assembly contain fewer than
    32 rounded scalar operations on linear coordinate expressions. The sum of
    absolute leaves, including upper-y recomputation and x wraps, is bounded by
    8 M. gamma_32 * 8 M plus a subnormal allowance encloses their total error.
    M and coefficient bounds come from the source grid/query-remap operands,
    never from a recovered shift or the requested public owner position.
    The actual preparation/storage defect and L-to-A frame defect remain exact.
    See docs/development/wp8-implementation.md for the source derivation.
    """
    dim = len(original)
    a, native = _exact_rows(lattice), _exact_rows(native_lattice)
    rotation = (_exact_rows(snapshot.rotation_to_internal) if snapshot is not None
                else tuple(tuple(F(int(i == j)) for j in range(dim))
                           for i in range(dim)))
    origin = tuple(F(float(v)) for v in snapshot.origin) if snapshot is not None \
        else (F(0),) * dim
    b = tuple(F(float(v)) for v in stored)
    k = tuple(int(x) + int(y) for x, y in zip(preparation, insertion))
    # At most one extra x/y/z period from the final region displacement; the
    # triclinic y/z displacement is already represented by copied storage.
    bounds = tuple(int(copy_bounds[i]) + abs(int(query_removal[i])) + 1
                   if periodic[i] else 0 for i in range(dim))
    magnitude = tuple(
        abs(b[j]) + sum(bounds[i] * abs(native[i][j]) for i in range(dim))
        for j in range(dim)
    )
    u, tiny = F(1, 2**53), F(1, 2**1074)
    gamma = 32 * u / (1 - 32 * u)
    assembly = tuple(gamma * 8 * v + 64 * tiny for v in magnitude)
    native_cart = tuple(
        origin[j] + sum(b[i] * rotation[i][j] for i in range(dim))
        for j in range(dim)
    )
    mapped_lattice = tuple(
        tuple(sum(native[i][k] * rotation[k][j] for k in range(dim))
              for j in range(dim))
        for i in range(dim)
    )
    defect = tuple(native_cart[j] - (F(float(original[j])) - sum(
        k[i] * a[i][j] for i in range(dim))) for j in range(dim))
    radius = []
    for j in range(dim):
        error = sum(assembly[i] * abs(rotation[i][j]) for i in range(dim))
        error += sum(bounds[i] * abs(mapped_lattice[i][j] - a[i][j])
                     for i in range(dim))
        if snapshot is not None:
            # Three-product BLAS dot (any ordinary association/FMA), then origin.
            scale = abs(origin[j]) + sum(
                (magnitude[i] + assembly[i]) * abs(rotation[i][j])
                for i in range(dim)
            )
            error += gamma * scale + 64 * tiny
        radius.append(error)
    return defect, tuple(radius), k, bounds


def _owner_shifts(found, owners, positions, packet, prepared, geometry, snapshot):
    dim, m, n = geometry.dim, len(found), len(prepared.internal_ids)
    packet = _source_packet(packet, n, m, dim)
    lattice, _origin = user_frame(geometry, snapshot)
    out = np.zeros((m, dim), dtype=np.int64)
    remaining = _MAX_BATCH_CANDIDATES
    for i in np.flatnonzero(found):
        i, j = int(i), int(owners[i])
        original = prepared.input_points_cart[j]
        defect, radius, removal, bounds = owner_enclosure(
            original=original, preparation=prepared.remap_shifts[j],
            stored=packet['stored'][j], insertion=packet['insertion_shifts'][j],
            query_removal=packet['query_removals'][i],
            copy_bounds=packet['copy_bounds'], lattice=lattice,
            native_lattice=packet['lattice'], snapshot=snapshot,
            periodic=geometry.periodic_axes,
        )
        center = tuple(F(float(positions[i, k])) - F(float(original[k]))
                       - defect[k] for k in range(dim))
        box = CartesianCompatibilityBox(
            tuple(c - r for c, r in zip(center, radius)),
            tuple(c + r for c, r in zip(center, radius)),
        )
        if remaining <= 0:
            raise LocateFailure('LOCATE_CERTIFICATION_RESOURCE',
                                'Complete owner batch candidate budget exhausted',
                                stage='provenance', query_index=i,
                                resource='batch_candidates')
        try:
            recovered = certify_native_translation(
                lattice, box, periodic_axes=geometry.periodic_axes,
                limits=replace(_LIMITS, max_candidates=min(
                    _LIMITS.max_candidates, remaining)),
            )
            shift = recovered.shift
            remaining -= max(1, recovered.diagnostics.candidate_bound or 0)
        except (NativeTranslationAmbiguityError, NativeTranslationInconsistencyError,
                NativeTranslationResourceError, NativeTranslationInvariantError) as exc:
            code = ('LOCATE_PROVENANCE_AMBIGUOUS' if isinstance(
                exc, NativeTranslationAmbiguityError) else
                'LOCATE_CERTIFICATION_RESOURCE' if isinstance(
                    exc, NativeTranslationResourceError) else
                'LOCATE_PROVENANCE_INCONSISTENT')
            reason = ('proof_invariant' if isinstance(
                exc, NativeTranslationInvariantError) else exc.reason)
            raise LocateFailure(code, str(exc), stage=exc.stage, query_index=i,
                                reason=reason, source_reason=exc.reason) from exc
        if any(abs(s + k) > b for s, k, b in zip(shift, removal, bounds)):
            raise _inconsistent('Recovered image violates producer bound', i,
                                reason='proof_invariant')
        if any(not -(2**63) <= s < 2**63 for s in shift):
            raise _materialization('Cannot materialize signed int64 owner shift',
                                   query_index=i, field='owner_shift')
        out[i] = shift
    return out


def locate_prepared(prepared, queries, *, geometry, snapshot, blocks, init_mem,
                    mode, return_owner_position, core_loader, external_ids):
    """One shared packaging boundary for six native locate routes."""
    dim, m, n = geometry.dim, len(queries), len(prepared.internal_ids)
    periodic = geometry.has_any_periodic_axis
    out = locate_query_views(queries, geometry, snapshot, _materialization) \
        if periodic else {}
    certificate = periodic and return_owner_position
    if m == 0:
        out.update(found=np.empty(0, dtype=bool), owner_id=np.empty(0, dtype=np.int64))
        if return_owner_position:
            out['owner_pos'] = np.empty((0, dim), dtype=np.float64)
            if periodic:
                out['owner_site'] = np.empty((0, dim), dtype=np.float64)
                out['owner_shift'] = np.empty((0, dim), dtype=np.int64)
        return out
    if certificate and m > _MAX_OWNER_QUERIES:
        raise LocateFailure('LOCATE_CERTIFICATION_RESOURCE',
                            'Complete owner batch exceeds private query budget',
                            stage='provenance', resource='owner_queries',
                            observed=m, limit=_MAX_OWNER_QUERIES)
    q_native = queries
    if snapshot is not None:
        with np.errstate(over='ignore', invalid='ignore'):
            q_native = snapshot.cart_to_internal(queries)
        if not np.isfinite(q_native).all():
            index = int(np.flatnonzero(~np.isfinite(q_native).all(axis=1))[0])
            raise LocateFailure('LOCATE_NATIVE_UNSUPPORTED', 'Nonfinite query frame',
                                stage='native', query_index=index)
    args = [prepared.native_points, prepared.internal_ids]
    if mode == 'power':
        args.append(prepared.backend_radii)
    args.extend([snapshot.params if snapshot is not None else geometry.native_bounds,
                 blocks])
    if snapshot is None:
        args.append(geometry.periodic_axes)
    args.extend([init_mem, q_native])
    name = f'locate_{"periodic" if snapshot is not None else "box"}_{mode}'
    try:
        result = getattr(core_loader(), name)(*args, return_source=certificate)
    except ValueError as exc:
        translated = _native_failure(exc)
        if translated is None:
            raise
        raise translated from exc
    except MemoryError as exc:
        raise LocateFailure('LOCATE_CERTIFICATION_RESOURCE', 'Native allocation',
                            stage='native') from exc
    found, owners, positions = _validated_native(
        result, n, m, dim, certificate, return_owner_position,
    )
    if snapshot is not None and return_owner_position:
        with np.errstate(over='ignore', invalid='ignore'):
            positions = snapshot.internal_to_cart(positions)
    if return_owner_position:
        if not np.isfinite(positions[found]).all():
            index = int(np.flatnonzero(found & ~np.isfinite(positions).all(axis=1))[0])
            raise _materialization('Nonfinite owner Cartesian view',
                                   query_index=index, field='owner_pos')
        positions[~found] = np.nan
        out['owner_pos'] = positions
        if periodic:
            sites = np.full((m, dim), np.nan)
            sites[found] = prepared.input_points_cart[owners[found]]
            out['owner_site'] = sites
            out['owner_shift'] = _owner_shifts(
                found, owners, positions, result[3], prepared, geometry, snapshot,
            )
    if external_ids:
        owners = owners.astype(np.int64)
        owners[found] = prepared.external_ids[owners[found]]
    out.update(found=found, owner_id=owners)
    return out
