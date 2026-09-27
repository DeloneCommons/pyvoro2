"""WP7 selected ghost-cell evidence, independent of the Python certifier."""
from __future__ import annotations

import math
import struct
import sys

import numpy as np
import pytest

from pyvoro2 import _core2d


BOUNDS = ((0.0, 1.0), (0.0, 1.0))
MASKS = ((False, False), (True, False), (False, True), (True, True))


def selected(points, queries, *, radii=None, ghost_radii=None, periodic=(True, True),
             bounds=BOUNDS, blocks=(1, 1), opts=(True, True, True), init_mem=1):
    points = np.asarray(points, dtype=float).reshape(-1, 2)
    queries = np.asarray(queries, dtype=float).reshape(-1, 2)
    args = [points, np.arange(len(points), dtype=np.int32)]
    mode = 'standard' if radii is None else 'power'
    if radii is not None:
        args.append(np.asarray(radii, dtype=float))
    args.extend([bounds, blocks, periodic, init_mem, opts, queries])
    if radii is not None:
        args.append(np.asarray(ghost_radii, dtype=float))
    return getattr(_core2d, f'_ghost_box_{mode}_witness')(*args)


@pytest.mark.parametrize('periodic', MASKS)
@pytest.mark.parametrize('power', (False, True))
def test_selected_ghost_is_inserted_and_initialization_is_source_attributed(
        periodic, power):
    cells, packets = selected([], [[0.25, 0.5]], radii=[] if power else None,
                              ghost_radii=[0.25] if power else None,
                              periodic=periodic, opts=(False, False, True))
    assert len(cells) == len(packets) == 1
    cell, packet = cells[0], packets[0]
    assert cell['id'] == -1 and cell['query_index'] == 0
    assert cell['site'] == [0.25, 0.5] and not cell['empty']
    assert 'vertices' not in cell and 'adjacency' not in cell
    assert packet['ghost_internal_id'] == 0 and packet['query_index'] == 0
    assert len(packet['inserted']) == 1
    assert packet['inserted'][0]['id'] == 0
    assert tuple(packet['inserted'][0]['point']) == tuple(cell['site'])
    (source,) = packet['sources']
    assert source['id'] == 0 and source['present']
    assert len(source['local2']) == len(source['next']) == len(source['origins']) == 4
    assert {row['side'] for row in source['origins']} == {-1, -2, -3, -4}
    for slot, origin in enumerate(source['origins']):
        assert origin['kind'] == 'initialization'
        assert origin['source'] == 0 and origin['slot'] == slot
        assert origin['next'] == source['next'][slot]
        assert cell['edges'][slot]['vertices'] == [slot, origin['next']]


@pytest.mark.parametrize('periodic', MASKS)
@pytest.mark.parametrize('power', (False, True))
def test_selected_ghost_labels_actual_persistent_and_self_images(periodic, power):
    cells, packets = selected([[0.25, 0.25], [0.75, 0.75]], [[0.5, 0.5]],
                              radii=[0.1, 0.2] if power else None,
                              ghost_radii=[0.15] if power else None,
                              periodic=periodic)
    assert len(cells) == len(packets) == 1
    assert {row['id'] for row in packets[0]['inserted']} == {0, 1, 2}
    source, = packets[0]['sources']
    assert source['id'] == 2 and source['present']
    for origin in source['origins']:
        if origin['kind'] == 'particle':
            assert origin['owner'] in {0, 1, 2}
            assert len(origin['sigma']) == 2
            assert all(-1 <= value <= 1 for value in origin['sigma'])
            assert all(periodic[axis] or value == 0
                       for axis, value in enumerate(origin['sigma']))
            assert origin['owner'] != 2 or tuple(origin['sigma']) != (0, 0)
        else:
            assert origin['kind'] == 'initialization'
            assert origin['side'] in {-1, -2, -3, -4}


def test_independent_query_containers_and_native_internal_geometry_without_vertices():
    args = dict(periodic=(True, True), opts=(False, False, True))
    points = [[0.125, 0.25], [0.75, 0.875]]
    queries = [[0.25, 0.75], [0.5, 0.5]]
    cells, packets = selected(points, queries, **args)
    for qi, query in enumerate(queries):
        one_cells, one_packets = selected(points, [query], **args)
        assert cells[qi]['edges'] == one_cells[0]['edges']
        assert cells[qi]['area'] == one_cells[0]['area']
        assert packets[qi]['sources'] == one_packets[0]['sources']
        assert packets[qi]['inserted'] == one_packets[0]['inserted']
        assert packets[qi]['query_index'] == qi
        assert 'vertices' not in cells[qi]
        assert packets[qi]['sources'][0]['local2']


@pytest.mark.parametrize('power', (False, True))
def test_insertion_failure_is_not_a_hidden_or_empty_cell(power):
    bounds = ((-1.0, 1.0), (0.0, 1.0))
    query = [[math.nextafter(1.0, 0.0), 0.5]]
    with pytest.raises(RuntimeError, match=r'planar_certification:insertion:omitted'):
        selected([], query, radii=[] if power else None,
                 ghost_radii=[0.0] if power else None,
                 periodic=(False, True), bounds=bounds,
                 opts=(False, False, False))


def test_actual_ghost_storage_and_source_step_replay():
    query = [[math.nextafter(0.1, 0.0), 0.5]]
    cells, packets = selected([], query, bounds=((0., .1), (0., 1.)),
                              blocks=(5, 1), periodic=(True, False))
    row, = packets[0]['inserted']
    assert row['id'] == packets[0]['ghost_internal_id'] == 0
    assert tuple(row['h']) == (1, 0)
    assert struct.pack('>d', row['point'][0]) == struct.pack('>d', -2.**-56)
    assert tuple(cells[0]['site']) == tuple(row['point'])


def test_power_deleted_ghost_keeps_verified_population_and_empty_record():
    cells, packets = selected([[0.25, 0.5]], [[0.75, 0.5]],
                              radii=[4.0], ghost_radii=[0.0],
                              opts=(False, False, True))
    assert len(cells) == len(packets) == 1
    assert cells[0]['empty'] and cells[0]['area'] == 0
    assert cells[0]['edges'] == []
    assert {row['id'] for row in packets[0]['inserted']} == {0, 1}
    source, = packets[0]['sources']
    assert source['id'] == 1 and not source['present']
    assert source['local2'] == source['next'] == source['origins'] == []


@pytest.mark.parametrize('power', (False, True))
def test_geometry_only_legacy_native_entry_also_rejects_missing_ghost(power):
    points = np.empty((0, 2), dtype=float)
    ids = np.empty((0,), dtype=np.int32)
    queries = np.asarray([[math.nextafter(1.0, 0.0), .5]])
    args = [points, ids]
    if power:
        args.append(np.empty((0,), dtype=float))
    args.extend([((-1., 1.), (0., 1.)), (1, 1), (False, True),
                 1, (False, False, False), queries])
    if power:
        args.append(np.asarray([0.]))
    fn = _core2d.ghost_box_power if power else _core2d.ghost_box_standard
    with pytest.raises(RuntimeError, match=r'planar_certification:insertion:omitted'):
        fn(*args)


@pytest.mark.parametrize('power', (False, True))
def test_zero_queries_validate_inputs_and_return_without_profile_gate(power):
    cells, packets = selected([[.25, .25]], [],
                              radii=[0.] if power else None,
                              ghost_radii=[] if power else None)
    assert cells == packets == []


def test_whole_batch_witness_budget_refuses_before_serializing_prefix():
    queries = np.repeat(np.array([[.5, .5]]), 120_000, axis=0)
    with pytest.raises(RuntimeError,
                       match=r'planar_certification:resource:batch_observer'):
        selected([], queries, opts=(False, False, True))


def _recursive_size(value, seen=None):
    if seen is None:
        seen = set()
    if id(value) in seen:
        return 0
    seen.add(id(value))
    total = sys.getsizeof(value)
    if isinstance(value, dict):
        return total + sum(_recursive_size(k, seen) + _recursive_size(v, seen)
                           for k, v in value.items())
    if isinstance(value, (list, tuple)):
        return total + sum(_recursive_size(item, seen) for item in value)
    return total


@pytest.mark.parametrize('points, queries', (
    ([], [[.5, .5]]),
    ([[.05 + .1 * i, .05 + .1 * j] for i in range(10) for j in range(10)],
     [[.51, .51]]),
))
def test_retained_packet_charge_bounds_representative_python_payload(points, queries):
    _, packets = selected(points, queries, opts=(False, False, True))
    profile = _core2d._planar_witness_profile()
    row_count = sum(len(packet['inserted']) for packet in packets)
    edge_count = sum(len(packet['sources'][0]['origins']) for packet in packets)
    charged = (len(packets) * profile['ghost_packet_charge_bytes'] +
               row_count * profile['ghost_inserted_row_charge_bytes'] +
               edge_count * profile['ghost_occurrence_charge_bytes'])
    assert _recursive_size(packets) <= charged
    assert charged <= profile['ghost_batch_limit_bytes']
