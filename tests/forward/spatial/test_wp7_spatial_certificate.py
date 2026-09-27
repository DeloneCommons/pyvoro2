"""Constructed selected-source packets; these are not observed ghost evidence."""

from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest

from pyvoro2 import _core
from pyvoro2._internal.spatial.wp5_common import WP5Failure
from pyvoro2._internal.spatial.wp5_certificate import _check_packet
from pyvoro2._internal.spatial.wp5_producer import Producer


def _selected():
    points = np.array([[.25, .25, .25], [.75, .75, .75]])
    ids = np.array([0, 1], dtype=np.int32)
    packet = _core._observe_box(
        points, ids, ((0., 1.),) * 3, (1, 1, 1),
        (True, True, True), 1,
    )
    packet['cells'] = [packet['cells'][1]]
    return packet, points, ids


def test_selected_scope_replays_full_augmented_insertion():
    packet, points, ids = _selected()
    _check_packet(packet, 2, selected_sources={1})
    producer = Producer(packet, points, ids, selected_sources={1})
    assert producer.removals == ((0, 0, 0), (0, 0, 0))
    for origin in packet['cells'][0]['origins']:
        if origin['kind'] != 'construction_bound':
            producer.attribute(packet['cells'][0], origin)
    with pytest.raises(WP5Failure, match='insertion'):
        Producer(packet, points[:1], ids[:1], selected_sources={1})


def test_selected_scope_rejects_missing_or_wrong_cell():
    packet, points, ids = _selected()
    wrong = deepcopy(packet)
    wrong['cells'][0]['id'] = 0
    with pytest.raises(WP5Failure):
        _check_packet(wrong, 2, selected_sources={1})
    with pytest.raises(WP5Failure):
        Producer(wrong, points, ids, selected_sources={1})
    missing = deepcopy(packet)
    missing['cells'] = []
    with pytest.raises(WP5Failure):
        _check_packet(missing, 2, selected_sources={1})


def _ghost_case():
    from pyvoro2.domains import OrthorhombicCell

    # Construct an augmented stock packet with a persistent original at 2.25.
    points = np.array([[.25, .5, .5], [.5, .5, .5]])
    packet = _core._observe_box(
        points, np.array([0, 1], dtype=np.int32), ((0., 1.),) * 3,
        (1, 1, 1), (True, True, True), 1,
    )
    packet['cells'] = [packet['cells'][1]]
    packet.update(schema='wp7-selected-ghost-3d-v1', query_index=0,
                  ghost_internal_id=1)
    packet['build']['ghost_selected_route'] = 'wp7-initialized-selected-v1'
    packet['build']['ghost_source_sha256'] = (
        'a58520644947526a312eda64c947fa56b5eeef77e2da942eb945ad43a911dbdc'
    )
    packet['build'].update(compiler_id='GNU', x86_64=True, sse2=True,
                           avx=False, fma=False)
    prepared = SimpleNamespace(
        internal_ids=np.array([0]), native_points=points[:1],
        input_points_cart=np.array([[2.25, .5, .5]]),
        remap_shifts=np.array([[2, 0, 0]]), external_ids=np.array([44]),
        backend_radii=None,
    )
    temporary = SimpleNamespace(
        native_points=points[1:], input_points_cart=np.array([[3.5, .5, .5]]),
        backend_radii=None,
    )
    power = SimpleNamespace(input_weights=None, input_ghost_weights=None,
                            backend_radii=None, backend_ghost_radii=None)
    options = dict(prepared=prepared, temporary=temporary, power_input=power,
                   domain=OrthorhombicCell(((0., 1.),) * 3), snapshot=None,
                   return_vertices=False, return_adjacency=False)
    return packet, options


def test_constructed_ghost_packet_uses_stored_chart_and_external_identity():
    from pyvoro2._internal.spatial.ghost_certificate import certify_ghost_packets

    packet, options = _ghost_case()
    cells = certify_ghost_packets(
        [packet], **options,
    )
    assert len(cells) == 1
    assert cells[0]['query'] == [3.5, .5, .5]
    assert cells[0]['site'] == [.5, .5, .5]
    assert 'vertices' not in cells[0]
    generator = [face for face in cells[0]['faces']
                 if face['boundary_reference'] is not None
                 and face['boundary_reference']['kind'] == 'generator']
    assert any(face['boundary_reference'] == {
        'kind': 'generator', 'generator_id': 44,
        'shift': (-2, 0, 0), 'wall_id': None,
    } for face in generator)
    assert all(face['adjacent_cell'] == 44 for face in generator)
    assert all(face.get('adjacent_cell') != 1 for face in cells[0]['faces'])


@pytest.mark.parametrize('field,value', (
    ('ghost_selected_route', None),
    ('ghost_source_sha256', '0' * 64),
    ('compiler_id', 'Clang'), ('compiler', '13.2.0'),
    ('x86_64', False), ('sse2', False), ('avx', True), ('fma', True),
    ('int_bits', 64), ('binary64', False), ('fp_contract', 'fast'),
))
def test_constructed_unsupported_selected_route_refuses_atomically(field, value):
    from pyvoro2._internal.ghost import GhostFailure
    from pyvoro2._internal.spatial.ghost_certificate import certify_ghost_packets

    packet, options = _ghost_case()
    packet['build'][field] = value
    with pytest.raises(GhostFailure) as caught:
        certify_ghost_packets([packet], **options)
    assert caught.value.code == 'GHOST_NATIVE_UNSUPPORTED'
    assert caught.value.stage == 'native'
    assert caught.value.query_index == 0


def test_constructed_invalid_selected_cell_incidence_is_not_semantic_collapse():
    from pyvoro2._internal.ghost import GhostFailure
    from pyvoro2._internal.spatial.ghost_certificate import certify_ghost_packets

    packet, options = _ghost_case()
    packet['cells'][0]['faces'][0]['edge_tokens'][0] = -1
    with pytest.raises(GhostFailure) as caught:
        certify_ghost_packets([packet], **options)
    assert caught.value.code == 'GHOST_PROVENANCE_INCONSISTENT'
    assert caught.value.stage == 'provenance'


def test_constructed_native_collapsed_cycle_keeps_source_class():
    from pyvoro2._internal.spatial.ghost_certificate import _occurrences
    from pyvoro2._internal.spatial.wp5_common import WP5Budget

    packet, options = _ghost_case()
    packet['cells'][0]['vertices_doubled'] = [[0., 0., 0.]] * len(
        packet['cells'][0]['vertices_doubled'])
    prep = options['prepared']
    points = np.concatenate((prep.native_points, options['temporary'].native_points))
    producer = Producer(packet, points, (0, 1), selected_sources={1})
    found = _occurrences(packet, producer, prep, 0, 1, WP5Budget())
    assert found and all(o.collapsed for o in found)
    assert any(o.owner == 0 and o.shift == (-2, 0, 0) for o in found)


def test_constructed_collapsed_source_does_not_materialize_its_shift():
    from pyvoro2._internal.spatial.ghost_certificate import _occurrences
    from pyvoro2._internal.spatial.wp5_common import WP5Budget

    packet, options = _ghost_case()
    packet['cells'][0]['vertices_doubled'] = [[0., 0., 0.]] * len(
        packet['cells'][0]['vertices_doubled'])
    prep = options['prepared']
    prep.remap_shifts = np.array([[2**70, 0, 0]], dtype=object)
    points = np.concatenate((prep.native_points, options['temporary'].native_points))
    producer = Producer(packet, points, (0, 1), selected_sources={1})
    found = _occurrences(packet, producer, prep, 0, 1, WP5Budget())
    assert all(o.collapsed for o in found)
    assert any(o.owner == 0 and o.shift[0] <= -(2**70) for o in found)


def test_constructed_storage_mismatch_is_backend_insertion_failure():
    from pyvoro2._internal.ghost import GhostFailure
    from pyvoro2._internal.spatial.ghost_certificate import certify_ghost_packets

    packet, options = _ghost_case()
    packet['sites'][1]['site'][0] += .125
    with pytest.raises(GhostFailure) as caught:
        certify_ghost_packets([packet], **options)
    assert caught.value.code == 'GHOST_BACKEND_INSERTION'
    assert caught.value.stage == 'insertion'


def test_constructed_missing_ghost_site_is_backend_insertion_failure():
    from pyvoro2._internal.ghost import GhostFailure
    from pyvoro2._internal.spatial.ghost_certificate import certify_ghost_packets

    packet, options = _ghost_case()
    packet['sites'].pop()
    with pytest.raises(GhostFailure) as caught:
        certify_ghost_packets([packet], **options)
    assert caught.value.code == 'GHOST_BACKEND_INSERTION'
    assert caught.value.stage == 'insertion'


def test_constructed_packet_batch_identity_cannot_shift_query_index():
    from pyvoro2._internal.ghost import GhostFailure
    from pyvoro2._internal.spatial.ghost_certificate import certify_ghost_packets

    packet, options = _ghost_case()
    packet['query_index'] = 1
    with pytest.raises(GhostFailure) as caught:
        certify_ghost_packets([packet], **options)
    assert caught.value.code == 'GHOST_PROVENANCE_INCONSISTENT'
    assert caught.value.query_index == 0


def test_constructed_empty_population_box_has_six_wall_references():
    from pyvoro2.domains import Box
    from pyvoro2._internal.spatial.ghost_certificate import certify_ghost_packets

    ghost = np.array([[.5, .5, .5]])
    packet = _core._observe_box(
        ghost, np.array([0], dtype=np.int32), ((0., 1.),) * 3,
        (1, 1, 1), (False, False, False), 1,
    )
    packet.update(schema='wp7-selected-ghost-3d-v1', query_index=0,
                  ghost_internal_id=0)
    packet['build']['ghost_selected_route'] = 'wp7-initialized-selected-v1'
    packet['build']['ghost_source_sha256'] = (
        'a58520644947526a312eda64c947fa56b5eeef77e2da942eb945ad43a911dbdc'
    )
    packet['build'].update(compiler_id='GNU', x86_64=True, sse2=True,
                           avx=False, fma=False)
    prepared = SimpleNamespace(
        internal_ids=np.array([], dtype=int), native_points=np.empty((0, 3)),
        input_points_cart=np.empty((0, 3)),
        remap_shifts=np.empty((0, 3), dtype=int),
        external_ids=np.array([], dtype=int), backend_radii=None,
    )
    temporary = SimpleNamespace(native_points=ghost, input_points_cart=ghost,
                                backend_radii=None)
    power = SimpleNamespace(input_weights=None, input_ghost_weights=None,
                            backend_radii=None, backend_ghost_radii=None)
    cells = certify_ghost_packets(
        [packet], prepared=prepared, temporary=temporary, power_input=power,
        domain=Box(((0., 1.),) * 3), snapshot=None,
        return_vertices=True, return_adjacency=True,
    )
    assert cells[0]['volume'] == 1.0
    assert len(cells[0]['vertices']) == 8
    assert len(cells[0]['faces']) == 6
    assert {f['boundary_reference']['wall_id'] for f in cells[0]['faces']} == set(
        range(-6, 0))
    assert all(f['boundary_reference']['kind'] == 'wall' for f in cells[0]['faces'])
