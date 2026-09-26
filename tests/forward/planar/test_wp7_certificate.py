"""Stored-ghost transport and malformed-packet failure constructions."""

import copy

import numpy as np
import pytest

from pyvoro2 import _core2d
from pyvoro2._internal.generator_preparation import (
    prepare_generators, prepare_temporary_generators,
)
from pyvoro2._internal.planar.domain_geometry import geometry2d
from pyvoro2._internal.power_input import resolve_ghost_power_input
from pyvoro2.planar import RectangularCell


def _fixture():
    domain = RectangularCell(((0., 1.), (0., 1.)))
    geom = geometry2d(domain)
    kwargs = dict(geometry=geom, backend_radii=None, duplicate_check='off',
                  duplicate_threshold=1e-5, duplicate_wrap=True,
                  duplicate_max_pairs=10)
    prepared = prepare_generators([[2.25, .5]], operation='ghost_cells',
                                  external_ids=[31], **kwargs)
    temporary = prepare_temporary_generators([[3.5, .5]], persistent=prepared,
                                             **kwargs)
    power = resolve_ghost_power_input(
        mode='standard', weights=None, radii=None, ghost_weights=None,
        ghost_radii=None, n=1, m=1,
    )
    cells, packets = _core2d._ghost_box_standard_witness(
        prepared.native_points, np.arange(1, dtype=np.int32),
        domain.bounds, (1, 1), (True, True), 1, (False, False, True),
        temporary.native_points,
    )
    return cells, packets, dict(
        prepared=prepared, temporary=temporary, power_input=power,
        domain=domain, return_edge_shifts=True,
    )


def test_stored_ghost_generator_shift_excludes_query_translation():
    from pyvoro2._internal.planar.ghost_certificate import certify_ghost_packets

    cells, packets, kwargs = _fixture()
    result = certify_ghost_packets(cells, packets, **kwargs)
    assert result[0]['site'] == [.5, .5]
    refs = [e['boundary_reference'] for e in result[0]['edges']]
    assert any(r['kind'] == 'generator' and r['generator_id'] == 31
               and r['shift'] == (-2, 0) for r in refs)
    assert all('adjacent_cell' not in e for e in result[0]['edges']
               if e['boundary_reference']['kind'] == 'ghost_self')


@pytest.mark.parametrize('malformation', [
    'missing_insertion', 'wrong_query', 'unknown_owner', 'missing_occurrence',
])
def test_constructed_malformed_packet_fails_atomically(malformation):
    from pyvoro2._internal.planar.ghost_certificate import certify_ghost_packets

    cells, packets, kwargs = _fixture()
    packets = copy.deepcopy(packets)
    packet = packets[0]
    expected = 'GHOST_PROVENANCE_INCONSISTENT'
    if malformation == 'missing_insertion':
        packet['inserted'].pop()
        expected = 'GHOST_BACKEND_INSERTION'
    elif malformation == 'wrong_query':
        packet['query_index'] = 9
    elif malformation == 'unknown_owner':
        origin = next(o for o in packet['sources'][0]['origins']
                      if o['kind'] == 'particle')
        origin['owner'] = 77
    else:
        packet['sources'][0]['origins'].pop()
    with pytest.raises(ValueError) as caught:
        certify_ghost_packets(cells, packets, **kwargs)
    assert caught.value.code == expected
    assert caught.value.query_index == 0


def test_constructed_native_fragments_keep_occurrence_order_and_shared_class():
    """A split outgoing segment is a construction, not an observed native case."""
    from pyvoro2._internal.planar.ghost_certificate import certify_ghost_packets

    cells, packets, kwargs = _fixture()
    source = packets[0]['sources'][0]
    slot = next(k for k, q in enumerate(source['next'])
                if source['local2'][k] != source['local2'][q])
    successor = source['next'][slot]
    new_slot = len(source['local2'])
    first, second = source['local2'][slot], source['local2'][successor]
    source['local2'].append([(a + b) / 2 for a, b in zip(first, second)])
    source['next'][slot] = new_slot
    source['next'].append(successor)
    origin = dict(source['origins'][slot], slot=new_slot, next=successor)
    source['origins'][slot]['next'] = new_slot
    source['origins'].append(origin)
    raw = cells[0]['edges']
    fragment = dict(raw[slot], vertices=[new_slot, successor])
    raw[slot]['vertices'] = [slot, new_slot]
    raw.append(fragment)
    result = certify_ghost_packets(cells, packets, **kwargs)[0]['edges']
    assert len(result) == new_slot + 1
    assert [e['vertices'] for e in result] == [e['vertices'] for e in raw]
    assert result[slot]['boundary_reference'] is not None
    assert result[slot]['boundary_reference'] == result[-1]['boundary_reference']
