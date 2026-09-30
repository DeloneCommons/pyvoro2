"""Selected planar ghost attribution, stored chart and positive references."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from ..ghost import (GhostFailure, GhostOccurrence, certify_semantics,
                     from_native_failure, semantic_weights)
from ..native_admission import require_component, require_environment
from ..native_qualification import NativeQualificationError
from .domain_geometry import geometry2d
from .wp6_certificate import (AttributionLimits, WP6Failure, _bits,
                              _checked_packet, _finite)
from .wp6_ideal import ExactAuditBudget


def _admit(query_index=None, *, artifact=False):
    try:
        if artifact:
            require_component('wp7-planar')
        else:
            require_environment()
    except NativeQualificationError as exc:
        raise GhostFailure('GHOST_NATIVE_UNSUPPORTED', str(exc), stage='native',
                           query_index=query_index, dimension=2,
                           reason=exc.reason, detail=exc.detail) from exc


def certify_ghost_packets(cells, packets, *, prepared, temporary, power_input,
                          domain, return_edge_shifts):
    """Return a complete batch or raise before any partial list escapes."""
    n = len(prepared.internal_ids)
    m = len(temporary.internal_ids)
    if len(cells) != m or len(packets) != m:
        raise GhostFailure('GHOST_PROVENANCE_INCONSISTENT',
                           'Selected native packet batch is incomplete',
                           stage='provenance', dimension=2)
    if not m:
        return []
    _admit(artifact=True)
    try:
        geom = geometry2d(domain)
    finally:
        _admit()
    budget = ExactAuditBudget()
    result = []
    count = 0
    for qi, (native, packet) in enumerate(zip(cells, packets)):
        try:
            if (packet['query_index'] != qi or packet['ghost_internal_id'] != n
                    or native['query_index'] != qi or native['id'] != -1
                    or type(native['empty']) is not bool):
                raise ValueError('Selected query/source identity mismatch')
            source, = packet['sources']
            if source['id'] != n or source['present'] != (not native['empty']):
                raise ValueError('Selected source/native disposition mismatch')
            original = np.concatenate((prepared.input_points_cart,
                                       temporary.input_points_cart[qi:qi + 1]))
            removals = np.concatenate((prepared.remap_shifts,
                                       temporary.remap_shifts[qi:qi + 1]))
            radii = (None if prepared.backend_radii is None else
                     np.concatenate((prepared.backend_radii,
                                     temporary.backend_radii[qi:qi + 1])))
            augmented = SimpleNamespace(
                input_points_cart=original, remap_shifts=removals,
                backend_radii=radii, internal_ids=tuple(range(n + 1)),
            )
            cert = _checked_packet(
                [] if native['empty'] else [dict(native, id=n)], packet,
                augmented, domain, 'standard' if radii is None else 'power',
                AttributionLimits(), selected_sources=(n,),
            )
            stored = cert.storage[n]['point']
            if len(native['site']) != 2 or any(
                    _bits(native['site'][k]) != _bits(stored[k]) for k in range(2)):
                raise ValueError('Selected native site differs from actual storage')
            site = [_finite(v) for v in stored]
            if native['empty'] and (
                    native['area'] != 0.0 or native.get('edges')
                    or native.get('vertices') or native.get('adjacency')):
                raise ValueError('Deleted source carries nonempty native geometry')
            count += len(cert.occurrences)
            if count > AttributionLimits().max_occurrences:
                raise GhostFailure('GHOST_CERTIFICATION_RESOURCE',
                                   'Complete ghost batch exceeds occurrence budget',
                                   stage='provenance', query_index=qi, dimension=2,
                                   resource='occurrences', required=count,
                                   limit=AttributionLimits().max_occurrences)
            occurrences = []
            for o in cert.occurrences:
                if o.sigma is None:
                    occurrences.append(GhostOccurrence(None, None, o.side,
                                                       o.collapsed))
                else:
                    shift = (o.sigma if o.owner == n else
                             tuple(int(o.sigma[k]) - cert.transport[o.owner][k]
                                   for k in range(2)))
                    occurrences.append(GhostOccurrence(o.owner, shift,
                                                       collapsed=o.collapsed))
            weights, ghost_weight = semantic_weights(power_input, qi, n)
            semantic = certify_semantics(
                points=prepared.input_points_cart, ghost_site=site,
                lattice=geom.lattice_vectors_cart, bounds=geom.native_bounds,
                periodic=geom.periodic_axes, weights=weights,
                ghost_weight=ghost_weight, occurrences=occurrences,
                present=not native['empty'], query_index=qi,
                external_ids=prepared.external_ids, budget=budget,
            )
            _admit(qi)
            out = dict(native, site=site, edges=[])
            for raw, o, ref in zip(native['edges'], occurrences, semantic.references):
                edge = dict(raw)
                edge.pop('adjacent_cell', None)
                edge.pop('adjacent_shift', None)
                edge['boundary_reference'] = ref
                if o.wall_id is not None:
                    edge['adjacent_cell'] = o.wall_id
                elif o.owner != n:
                    edge['adjacent_cell'] = int(prepared.external_ids[o.owner])
                if return_edge_shifts and ref is not None and ref['shift'] is not None:
                    edge['adjacent_shift'] = ref['shift']
                out['edges'].append(edge)
            result.append(out)
        except GhostFailure:
            raise
        except WP6Failure as exc:
            raise from_native_failure(exc, 2, qi) from exc
        except (KeyError, IndexError, TypeError, ValueError, OverflowError) as exc:
            raise GhostFailure('GHOST_PROVENANCE_INCONSISTENT', str(exc),
                               stage='provenance', query_index=qi,
                               dimension=2) from exc
    return result
