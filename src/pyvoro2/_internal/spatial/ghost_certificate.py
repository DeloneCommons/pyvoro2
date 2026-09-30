"""Selected 3D ghost occurrence attribution and stored-chart packaging.

The native packet contains the complete augmented insertion but only the
selected ghost cell. Producer replay decides each native occurrence before
the independent exact public-semantic gate sees its class.
"""

from __future__ import annotations

import math

import numpy as np

from ..ghost import (GhostFailure, GhostOccurrence, certify_semantics,
                     from_native_failure, semantic_weights)
from ..inputs import coerce_finite_matrix
from ..native_admission import require_component, require_environment
from ..native_qualification import NativeQualificationError
from ..native_runtime import checked_call, checked_tuple
from ..validation import require_ordered_bounds
from .wp5_certificate import _check_packet
from .wp5_common import WP5Budget, WP5Failure
from .wp5_cycle import audit_cycle
from .wp5_producer import Producer


_SCHEMA = 'wp7-selected-ghost-3d-v1'
_MAX_OCCURRENCES = 262_144
_MAX_SOURCE_TOKENS = 1_000_000


def _admit(query_index=None, *, artifact=False):
    try:
        if artifact:
            return require_component('wp7-spatial')
        else:
            require_environment()
    except NativeQualificationError as exc:
        raise GhostFailure('GHOST_NATIVE_UNSUPPORTED', str(exc), stage='native',
                           query_index=query_index, dimension=3,
                           reason=exc.reason, detail=exc.detail) from exc


def _inconsistent(message, query_index, **details):
    raise GhostFailure('GHOST_PROVENANCE_INCONSISTENT', message,
                       stage='provenance', query_index=query_index,
                       dimension=3, **details)


def _finite(value, query_index, *, field):
    try:
        number = float(value)
    except (ValueError, TypeError, OverflowError) as exc:
        raise GhostFailure(
            'GHOST_SHIFT_UNREPRESENTABLE',
            f'Cannot materialize finite ghost {field}', stage='materialization',
            query_index=query_index, dimension=3, field=field,
        ) from exc
    if not math.isfinite(number):
        raise GhostFailure(
            'GHOST_SHIFT_UNREPRESENTABLE',
            f'Cannot materialize finite ghost {field}', stage='materialization',
            query_index=query_index, dimension=3, field=field,
        )
    return number


def _finite_point(row, query_index, *, field):
    if len(row) != 3:
        _inconsistent(f'Native {field} is not a 3D point', query_index)
    return [_finite(value, query_index, field=field) for value in row]


def _stored_cart(site, snapshot, query_index):
    _admit(query_index)
    values = _finite_point(site, query_index, field='stored site')
    if snapshot is not None:
        with np.errstate(over='ignore', invalid='ignore'):
            try:
                values = snapshot.internal_to_cart(
                    np.asarray(values, dtype=np.float64).reshape(1, 3)
                )
            finally:
                _admit(query_index)
            values = values.reshape(3)
    return _finite_point(values, query_index, field='stored Cartesian site')


def _public_vertices(cell, snapshot, query_index):
    _admit(query_index)
    # Voro++'s ordinary vertex view adds half the doubled local coordinate
    # to the stored native site, before transforming the whole point once.
    points = [
        [_finite(float(cell['site'][k]) + .5 * float(row[k]), query_index,
                 field='native vertex') for k in range(3)]
        for row in cell['vertices_doubled']
    ]
    if points and snapshot is not None:
        with np.errstate(over='ignore', invalid='ignore'):
            try:
                points = snapshot.internal_to_cart(np.asarray(points))
            finally:
                _admit(query_index)
            points = points.tolist()
    return [_finite_point(point, query_index, field='Cartesian vertex')
            for point in points]


def _cohort(packet, query_index):
    # This is packet consistency after external component admission, not a
    # compiler-version or enabled-ISA allowlist.
    module = _admit(query_index, artifact=True)
    try:
        expected = module._spatial_witness_profile()
    finally:
        _admit(query_index)
    build = packet.get('build', {})
    if (not isinstance(build, dict)
            or any(build.get(key) != expected.get(key) for key in (
                'source_sha256', 'ghost_source_sha256', 'compiler_id', 'compiler',
                'x86_64', 'sse2', 'avx', 'fma'))
            or build.get('int_bits') != 32
            or build.get('int_min') != -(1 << 31)
            or build.get('int_max') != (1 << 31) - 1
            or build.get('binary64') is not True
            or build.get('float_eval_method') != 0
            or build.get('round_to_nearest') is not True
            or build.get('gradual_underflow') is not True
            or build.get('runtime_compatible') is not True
            or build.get('fast_math') is not False
            or build.get('fp_contract') != 'off'
            or build.get('ipo') is not False
            or build.get('native_fp_policy') != 'binary64-noncontracting-v1'
            or build.get('ghost_selected_route') != 'wp7-initialized-selected-v1'):
        raise GhostFailure(
            'GHOST_NATIVE_UNSUPPORTED',
            'Selected ghost source/FP packet metadata is inconsistent', stage='native',
            query_index=query_index, dimension=3,
        )


def _packet_identity(packet, query_index, n):
    if (not isinstance(packet, dict) or packet.get('schema') != _SCHEMA
            or type(packet.get('query_index')) is not int
            or packet['query_index'] != query_index
            or type(packet.get('ghost_internal_id')) is not int
            or packet['ghost_internal_id'] != n):
        _inconsistent('Selected ghost packet identity differs', query_index)
    _cohort(packet, query_index)
    if n + 1 > _MAX_SOURCE_TOKENS:
        raise GhostFailure(
            'GHOST_CERTIFICATION_RESOURCE',
            'Augmented ghost source population exceeds certified limit',
            stage='provenance', query_index=query_index, dimension=3,
            required=n + 1, limit=_MAX_SOURCE_TOKENS,
        )
    if len(packet.get('sites', ())) != n + 1:
        raise GhostFailure(
            'GHOST_BACKEND_INSERTION',
            'Actual native ghost insertion population is incomplete',
            stage='insertion', query_index=query_index, dimension=3,
            expected=n + 1, observed=len(packet.get('sites', ())),
        )
    try:
        selected = packet['cells'][0]
        if type(selected.get('computed')) is not bool:
            _inconsistent('Selected ghost disposition is malformed', query_index)
        origins = selected['origins']
        faces = selected['faces']
        if len(origins) > _MAX_SOURCE_TOKENS or len(faces) > _MAX_OCCURRENCES:
            raise GhostFailure(
                'GHOST_CERTIFICATION_RESOURCE',
                'Selected ghost witness exceeds bounded native records',
                stage='provenance', query_index=query_index, dimension=3,
                source_tokens=len(origins), faces=len(faces),
                source_limit=_MAX_SOURCE_TOKENS,
                face_limit=_MAX_OCCURRENCES,
            )
    except (KeyError, IndexError, TypeError):
        _inconsistent('Malformed selected ghost witness', query_index)
    try:
        _check_packet(packet, n + 1, selected_sources={n})
    except WP5Failure as exc:
        raise from_native_failure(exc, dimension=3, query_index=query_index,
                                  stage='provenance') from exc


def _occurrences(packet, producer, prepared, query_index, n, budget):
    cell = packet['cells'][0]
    origins = {origin['token']: origin for origin in cell['origins']}
    if len(origins) > _MAX_SOURCE_TOKENS or any(
            token > _MAX_SOURCE_TOKENS for token in origins):
        raise GhostFailure(
            'GHOST_CERTIFICATION_RESOURCE', 'Native ghost tokens exceed limit',
            stage='provenance', query_index=query_index, dimension=3,
            required=max((len(origins), *origins), default=0),
            limit=_MAX_SOURCE_TOKENS,
        )
    if len(cell['faces']) > _MAX_OCCURRENCES:
        raise GhostFailure(
            'GHOST_CERTIFICATION_RESOURCE', 'Too many native ghost faces',
            stage='provenance', query_index=query_index, dimension=3,
            required=len(cell['faces']), limit=_MAX_OCCURRENCES,
        )
    attributed = []
    for face in cell['faces']:
        origin = origins[face['token']]
        label = producer.attribute(cell, origin)
        owner, sigma = label.owner, label.shift
        if sigma is None:
            if not -6 <= owner <= -1 or origin['kind'] != 'orthogonal_seed':
                _inconsistent('Source-qualified wall has invalid owner',
                              query_index, token=face['token'])
            occurrence = GhostOccurrence(None, None, wall_id=owner)
        else:
            if len(sigma) != 3 or not 0 <= owner <= n:
                _inconsistent('Native image has invalid augmented owner',
                              query_index, token=face['token'])
            if owner == n:
                if not any(sigma):
                    _inconsistent('Primary-self occurrence has no image',
                                  query_index, token=face['token'])
                shift = tuple(int(v) for v in sigma)
            else:
                shift = tuple(int(sigma[k])
                              - int(prepared.remap_shifts[owner, k])
                              - producer.removals[owner][k] for k in range(3))
            occurrence = GhostOccurrence(owner, shift)
        # _check_packet validated incidence, edge-slot association, token and
        # support provenance. Rank-deficient projected cycles are raw artifacts.
        try:
            audit_cycle([cell['vertices_doubled'][v] for v in face['vertices']],
                        origin['normal'], origin['offset'], budget=budget)
        except WP5Failure as exc:
            if exc.code != 'WP5_NATIVE_CYCLE_COLLAPSED':
                raise
            occurrence = GhostOccurrence(occurrence.owner, occurrence.shift,
                                         wall_id=occurrence.wall_id,
                                         collapsed=True)
        attributed.append(occurrence)
    return attributed


def _map_wp5(exc, query_index, stage):
    if exc.context.get('stage') == 'profile':
        return GhostFailure('GHOST_NATIVE_UNSUPPORTED', str(exc), stage='native',
                            query_index=query_index, dimension=3,
                            reason=exc.context.get('reason'),
                            detail=exc.context.get('detail'))
    message = str(exc)
    if exc.code == 'WP5_SOURCE_PROFILE_MISMATCH' and any(
            phrase in message for phrase in (
                'insertion row count differs', 'persistent site identity differs',
                'actual native storage differs', 'actual native radius differs',
                'nonperiodic insertion outside blocks',
            )):
        return GhostFailure(
            'GHOST_BACKEND_INSERTION',
            'Actual selected ghost insertion differs from source replay',
            stage='insertion', query_index=query_index, dimension=3,
            invariant='augmented_insertion',
        )
    return from_native_failure(exc, dimension=3, query_index=query_index,
                               stage=stage)


def certify_ghost_packets(packets, *, prepared, temporary, power_input,
                          domain, snapshot, return_vertices,
                          return_adjacency):
    """Certify a complete selected-ghost batch atomically, preserving native views."""
    n = len(prepared.internal_ids)
    m = len(temporary.native_points)
    if len(packets) != m:
        _inconsistent('Ghost packet count differs from query count', None,
                      packets=len(packets), queries=m)
    if not m:
        return []
    _admit(artifact=True)
    budget = WP5Budget()
    try:
        periodic = tuple(checked_call(bool, value) for value in checked_tuple(
            checked_call(getattr, domain, 'periodic')
        )) if checked_call(hasattr, domain, 'periodic') else (
            (False, False, False) if snapshot is None else (True, True, True))
        if snapshot is not None:
            lattice = snapshot.vectors
        elif checked_call(hasattr, domain, 'lattice_vectors'):
            lattice = coerce_finite_matrix(
                checked_call(getattr, domain, 'lattice_vectors'),
                name='lattice vectors', shape=(3, 3),
            )
        else:
            # A nonperiodic Box has no declared lattice; these rows are only
            # positive spans used by the exact ideal's bounded outer polytope.
            spans = [hi - lo for lo, hi in require_ordered_bounds(
                checked_call(getattr, domain, 'bounds'),
                name='domain bounds', dim=3,
            )]
            lattice = np.diag(spans)
        bounds = None if snapshot is not None else require_ordered_bounds(
            checked_call(getattr, domain, 'bounds'),
            name='domain bounds', dim=3,
        )
    finally:
        _admit()
    result = []
    for query_index, packet in enumerate(packets):
        try:
            _packet_identity(packet, query_index, n)
            native_points = np.concatenate((
                prepared.native_points,
                temporary.native_points[query_index:query_index + 1],
            ))
            ids = tuple(int(v) for v in prepared.internal_ids) + (n,)
            radii = None
            if prepared.backend_radii is not None:
                radii = np.concatenate((
                    prepared.backend_radii,
                    temporary.backend_radii[query_index:query_index + 1],
                ))
            producer = Producer(packet, native_points, ids, radii, budget=budget,
                                selected_sources={n})
            cell = packet['cells'][0]
            if not cell['computed'] and (cell['faces'] or cell['vertices_doubled']
                                         or cell['adjacency'] or cell['volume'] != 0):
                _inconsistent('Deleted native ghost has nonempty geometry',
                              query_index)
            site = _stored_cart(packet['sites'][n]['site'], snapshot, query_index)
            if cell['computed'] and not cell['faces']:
                _inconsistent('Computed ghost cell has no boundary occurrences',
                              query_index)
            occurrences = _occurrences(packet, producer, prepared,
                                       query_index, n, budget)
            weights, ghost_weight = semantic_weights(power_input, query_index, n)
            semantic = certify_semantics(
                points=prepared.input_points_cart, ghost_site=site,
                lattice=lattice, bounds=bounds, periodic=periodic,
                weights=weights, ghost_weight=ghost_weight,
                occurrences=occurrences, present=bool(cell['computed']),
                query_index=query_index, external_ids=prepared.external_ids,
                budget=budget,
            )
            _admit(query_index)
            references = semantic.references
            if len(references) != len(cell['faces']):
                _inconsistent('Semantic reference count differs from native faces',
                              query_index)
            public = {
                'id': -1, 'query_index': query_index,
                'query': temporary.input_points_cart[query_index].tolist(),
                'site': site, 'empty': not bool(cell['computed']),
                'volume': _finite(cell['volume'], query_index, field='native volume'),
                'faces': [],
            }
            reflected = snapshot is not None and snapshot.parity < 0
            for face, occurrence, reference in zip(cell['faces'], occurrences,
                                                   references):
                cycle = list(face['vertices'])
                if reflected:
                    cycle.reverse()
                item = {'vertices': cycle, 'boundary_reference': reference}
                if occurrence.wall_id is not None:
                    item['adjacent_cell'] = occurrence.wall_id
                elif occurrence.owner is not None and occurrence.owner < n:
                    item['adjacent_cell'] = int(prepared.external_ids[occurrence.owner])
                public['faces'].append(item)
            if return_vertices:
                public['vertices'] = _public_vertices(cell, snapshot, query_index)
            if return_adjacency:
                public['adjacency'] = [list(reversed(row)) if reflected else list(row)
                                       for row in cell['adjacency']]
            result.append(public)
        except WP5Failure as exc:
            raise _map_wp5(exc, query_index, 'provenance') from exc
    return result
