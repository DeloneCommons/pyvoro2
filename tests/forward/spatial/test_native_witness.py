"""Independent expectations for the private native construction observer.

These witnesses describe the backend operation and its surviving raw faces;
they do not certify positive face measure or public periodic image labels.
"""

from __future__ import annotations

from collections import Counter
from fractions import Fraction
from itertools import product
import struct

import numpy as np
import pytest

from pyvoro2 import _core


BOUNDS = ((0.0, 4.0), (0.0, 4.0), (0.0, 4.0))
BLOCKS = (1, 1, 1)
MASKS = tuple(product((False, True), repeat=3))
CELL_PARAMS = (4.0, 1.0, 4.0, 0.5, 1.5, 4.0)


def _bits(value):
    return struct.pack('=d', value)


def _has_noncollinear_vertices(cell, face):
    """Exact affine dimension of the recorded binary64 vertex coordinates."""
    vertices = [tuple(map(Fraction, cell['vertices_doubled'][index]))
                for index in face['vertices']]
    differences = [tuple(v[k] - vertices[0][k] for k in range(3))
                   for v in vertices[1:]]
    return any(
        left[k] * right[(k + 1) % 3] != left[(k + 1) % 3] * right[k]
        for left in differences for right in differences for k in range(3)
    )


def _box(points, ids, periodic=(False, False, False), radii=None,
         bounds=BOUNDS, blocks=BLOCKS):
    return _core._observe_box(
        np.asarray(points, dtype=np.float64),
        np.asarray(ids, dtype=np.int32),
        bounds, blocks, periodic, 8, radii=radii,
    )


def _periodic(points, ids, radii=None, params=CELL_PARAMS, blocks=BLOCKS):
    return _core._observe_periodic(
        np.asarray(points, dtype=np.float64),
        np.asarray(ids, dtype=np.int32),
        params, blocks, 8, radii=radii,
    )


def _legacy_box(points, ids, periodic, radii, bounds=BOUNDS, blocks=BLOCKS):
    args = [np.asarray(points, dtype=np.float64),
            np.asarray(ids, dtype=np.int32)]
    if radii is not None:
        args.append(np.asarray(radii, dtype=np.float64))
    args.extend([bounds, blocks, periodic, 8, (True, True, True)])
    fn = (_core.compute_box_standard if radii is None
          else _core.compute_box_power)
    return fn(*args)


def _legacy_periodic(points, ids, radii, params=CELL_PARAMS, blocks=BLOCKS):
    args = [np.asarray(points, dtype=np.float64),
            np.asarray(ids, dtype=np.int32)]
    if radii is not None:
        args.append(np.asarray(radii, dtype=np.float64))
    args.extend([params, blocks, 8, (True, True, True)])
    fn = (_core.compute_periodic_standard if radii is None
          else _core.compute_periodic_power)
    return fn(*args)


def _assert_cycles(cell):
    """Check the full graph without deriving face semantics from geometry."""
    origins = {origin['token']: origin for origin in cell['origins']}
    assert len(origins) == len(cell['origins'])
    assert all(isinstance(token, int) and token > 0 for token in origins)
    adjacency = cell['adjacency']
    assert cell['vertex_orders'] == [len(row) for row in adjacency]
    assert len(adjacency) == len(cell['vertices_doubled'])
    actual_edges = Counter()
    for face in cell['faces']:
        cycle = face['vertices']
        assert len(cycle) >= 3
        assert len(set(cycle)) == len(cycle)
        assert face['token'] in origins
        assert origins[face['token']]['kind'] != 'construction_bound'
        assert face['edge_tokens'] == [face['token']] * len(cycle)
        assert face['legacy_owner'] == origins[face['token']]['legacy_owner']
        for start, end in zip(cycle, cycle[1:] + cycle[:1]):
            assert end in adjacency[start]
            actual_edges[start, end] += 1
    expected_edges = Counter(
        (start, end)
        for start, row in enumerate(adjacency)
        for end in row
    )
    assert actual_edges == expected_edges


def _assert_legacy_equivalent(packet, legacy):
    observed = {cell['id']: cell for cell in packet['cells']}
    expected = {cell['id']: cell for cell in legacy}
    assert {pid for pid, cell in observed.items() if cell['computed']} == set(
        expected
    )
    for pid, cell in observed.items():
        applicable = True if cell['computed'] else None
        assert cell['noninterference'] == {
            'computed': True, 'geometry': applicable, 'topology': applicable,
            'owners': applicable, 'volume': applicable,
        }
        if not cell['computed']:
            assert cell['volume'] == 0.0
            assert cell['vertices_doubled'] == []
            assert cell['faces'] == []
            continue
        raw = expected[pid]
        assert _bits(cell['volume']) == _bits(raw['volume'])
        assert cell['site'] == raw['site']
        vertices = np.asarray(cell['vertices_doubled']) * 0.5
        vertices += np.asarray(cell['site'])
        np.testing.assert_array_equal(vertices, raw['vertices'])
        assert cell['adjacency'] == raw['adjacency']
        assert [face['vertices'] for face in cell['faces']] == [
            face['vertices'] for face in raw['faces']
        ]
        assert [face['legacy_owner'] for face in cell['faces']] == [
            face['adjacent_cell'] for face in raw['faces']
        ]
        _assert_cycles(cell)


@pytest.mark.parametrize('periodic', MASKS)
@pytest.mark.parametrize('power', (False, True))
@pytest.mark.parametrize('count', (1, 3))
def test_box_all_periodic_masks_preserve_native_geometry_and_seed_meaning(
    periodic, power, count,
):
    points = np.array([[1.0, 1.0, 1.0], [3.0, 1.0, 1.0],
                       [2.0, 3.0, 2.0]])[:count]
    ids = np.array([0] if count == 1 else [2, 0, 1], dtype=np.int32)
    radii = np.array([0.125, 0.25, 0.5])[:count] if power else None
    packet = _box(points, ids, periodic, radii)

    assert packet['context']['kind'] == 'box'
    assert tuple(packet['context']['periodic']) == periodic
    assert packet['context']['power'] is power
    assert {cell['id'] for cell in packet['cells']} == set(ids)
    _assert_legacy_equivalent(
        packet, _legacy_box(points, ids, periodic, radii),
    )
    for cell in packet['cells']:
        seeds = [o for o in cell['origins'] if o['kind'] == 'orthogonal_seed']
        assert len(seeds) == 6
        assert {o['side'] for o in seeds} == {-1, -2, -3, -4, -5, -6}
        for origin in seeds:
            axis = origin['axis']
            sense = origin['sense']
            assert sense in (-1, 1)
            assert origin['side'] == -(2 * axis + (1 if sense == -1 else 2))
            assert origin['periodic'] is periodic[axis]
            assert origin['legacy_owner'] == origin['side']
            assert origin['owner'] == (
                cell['id'] if periodic[axis] else origin['side']
            )
            normal = np.zeros(3)
            normal[axis] = sense
            np.testing.assert_array_equal(origin['normal'], normal)
            expected_offset = (
                4.0 if periodic[axis]
                else 2.0 * sense * (
                    BOUNDS[axis][sense == 1] - cell['site'][axis]
                )
            )
            assert origin['offset'] == expected_offset
        if count == 1:
            origins = {o['token']: o for o in cell['origins']}
            assert len(cell['faces']) == 6
            final = [origins[f['token']] for f in cell['faces']]
            # Native marginal-facet handling replaces coincident seed tags
            # with the actual periodic self-particle cut tags.
            assert {o['side'] for o in final
                    if o['kind'] == 'orthogonal_seed'} == {
                -(2 * axis + side)
                for axis in range(3) if not periodic[axis]
                for side in (1, 2)
            }
            self_cuts = [o for o in final if o['kind'] == 'particle']
            expected_normals = set()
            for axis in range(3):
                if periodic[axis]:
                    for sense in (-1.0, 1.0):
                        normal = [0.0, 0.0, 0.0]
                        normal[axis] = 4.0 * sense
                        expected_normals.add(tuple(normal))
            assert {tuple(o['normal']) for o in self_cuts} == expected_normals
            assert all(o['owner'] == 0 for o in self_cuts)


@pytest.mark.parametrize('power', (False, True))
@pytest.mark.parametrize('ids', ([0, 1], [1, 0]))
def test_periodic_owner_permutation_and_genuine_owner_zero(power, ids):
    points = np.array([[1.0, 1.0, 1.0], [3.0, 2.0, 2.0]])
    radii = np.array([0.125, 0.25]) if power else None
    packet = _periodic(points, ids, radii)
    _assert_legacy_equivalent(packet, _legacy_periodic(points, ids, radii))

    by_id = {cell['id']: cell for cell in packet['cells']}
    for pid, cell in by_id.items():
        origins = {o['token']: o for o in cell['origins']}
        assert all(o['owner'] == pid for o in origins.values()
                   if o['kind'] == 'triclinic_seed')
        other_faces = [f for f in cell['faces']
                       if origins[f['token']]['kind'] == 'particle'
                       and origins[f['token']]['owner'] == 1 - pid]
        assert other_faces
        assert all(f['legacy_owner'] == 1 - pid for f in other_faces)
    # Zero is an actual other particle here, not evidence of a seed face.
    assert any(o['kind'] == 'particle' and o['owner'] == 0
               for o in by_id[1]['origins'])


@pytest.mark.parametrize('power', (False, True))
def test_single_owner_skew_periodic_faces_retain_actual_self_cut_provenance(
    power,
):
    points = [[1.0, 1.0, 1.0]]
    radii = np.array([0.125]) if power else None
    packet = _periodic(points, [0], radii)
    _assert_legacy_equivalent(packet, _legacy_periodic(points, [0], radii))
    assert tuple(packet['context']['cell_params']) == CELL_PARAMS
    cell = packet['cells'][0]
    assert cell['computed']
    assert cell['volume'] == pytest.approx(64.0)
    origins = {o['token']: o for o in cell['origins']}
    assert cell['faces']
    for face in cell['faces']:
        origin = origins[face['token']]
        assert origin['kind'] == 'particle'
        assert origin['owner'] == 0
        assert face['legacy_owner'] == 0


@pytest.mark.parametrize('power', (False, True))
def test_triclinic_route_distinguishes_seed_and_particle_faces_with_legacy_zero(
    power,
):
    params = (4096.0, 1024.0, 4096.0, 512.0, 1536.0, 4096.0)
    points = [[1024.0, 1024.0, 1024.0], [3072.0, 2048.0, 2048.0]]
    radii = np.array([40960.0, 40960.0]) if power else None
    packet = _periodic(points, [0, 1], radii, params=params)
    _assert_legacy_equivalent(
        packet, _legacy_periodic(points, [0, 1], radii, params=params),
    )
    cell = next(cell for cell in packet['cells'] if cell['id'] == 1)
    origins = {o['token']: o for o in cell['origins']}
    kinds = Counter()
    for face in cell['faces']:
        origin = origins[face['token']]
        if face['legacy_owner'] != 0:
            continue
        kinds[origin['kind']] += 1
        if origin['kind'] == 'particle':
            assert origin['owner'] == 0
        else:
            assert origin['kind'] == 'triclinic_seed'
            assert origin['owner'] == 1
        assert _has_noncollinear_vertices(cell, face)
    assert kinds['particle'] > 0
    assert kinds['triclinic_seed'] > 0
    _assert_cycles(cell)


@pytest.mark.parametrize('pid', (0, 1))
def test_triclinic_surviving_seed_legacy_zero_has_current_owner(pid):
    params = (4096.0, 1024.0, 4096.0, 512.0, 1536.0, 4096.0)
    points = np.array([[1024.0, 1024.0, 1024.0],
                       [3072.0, 2048.0, 2048.0]])
    ids = [1 - pid, pid]
    radii = np.array([0.0, 40960.0])
    packet = _periodic(points, ids, radii, params=params)
    _assert_legacy_equivalent(
        packet, _legacy_periodic(points, ids, radii, params=params),
    )
    by_id = {cell['id']: cell for cell in packet['cells']}
    assert not by_id[1 - pid]['computed']
    assert [site['id'] for site in packet['sites']] == [0, 1]
    assert packet['sites'][1 - pid]['site'] == list(points[0])
    assert packet['sites'][1 - pid]['radius'] == 0.0
    assert packet['sites'][pid]['radius'] == 40960.0
    current = by_id[pid]
    assert current['computed']
    origins = {o['token']: o for o in current['origins']}
    seed_faces = [f for f in current['faces']
                  if origins[f['token']]['kind'] == 'triclinic_seed']
    seeds = [origins[f['token']] for f in seed_faces]
    assert len(seed_faces) == 10
    assert all(_has_noncollinear_vertices(current, f) for f in seed_faces)
    assert all(o['legacy_owner'] == 0 and o['owner'] == pid for o in seeds)
    assert all(o['kind'] != 'particle' or o['owner'] == pid
               for o in (origins[f['token']] for f in current['faces']))


@pytest.mark.parametrize('power', (False, True))
def test_periodic_persistent_self_cut_occurrences_are_retained(power):
    radii = np.array([0.5]) if power else None
    packet = _box([[1.0, 1.0, 1.0]], [0], (True,) * 3, radii)
    cell = packet['cells'][0]
    self_cuts = [o for o in cell['origins'] if o['kind'] == 'particle']
    assert self_cuts
    surviving = {face['token'] for face in cell['faces']}
    for origin in self_cuts:
        assert origin['owner'] == cell['id']
        assert origin['legacy_owner'] == cell['id']
        normal = np.asarray(origin['normal'])
        assert np.any(normal != 0.0)
        np.testing.assert_array_equal(normal / 4.0, np.round(normal / 4.0))
        assert origin['offset'] == normal @ normal
    assert any(origin['token'] not in surviving for origin in self_cuts)
    assert surviving <= {origin['token'] for origin in self_cuts}
    # Coincident native cuts can replace seed tags without changing geometry;
    # other actual self-cut occurrences leave no surviving face tag.
    assert len(surviving) == 6
    _assert_cycles(cell)


@pytest.mark.parametrize('observe', (_box, _periodic))
def test_native_site_table_preserves_id_association_and_radius_bits(observe):
    points = np.array([[1.0, 1.0, 1.0], [3.0, 2.0, 2.0],
                       [2.0, 3.0, 1.0]])
    ids = [2, 0, 1]
    radii = np.array([np.nextafter(0.5, 1.0), -0.0,
                      np.nextafter(0.25, 0.0)])
    packet = observe(points, ids, radii=radii)
    assert [site['id'] for site in packet['sites']] == [0, 1, 2]
    for row, pid in enumerate(ids):
        site = packet['sites'][pid]
        np.testing.assert_array_equal(site['site'], points[row])
        assert _bits(site['radius']) == _bits(radii[row])
        cell = next(c for c in packet['cells'] if c['id'] == pid)
        assert _bits(cell['radius']) == _bits(radii[row])
    assert packet['build']['binary64'] is True
    assert packet['build']['round_to_nearest'] is True
    assert packet['build']['gradual_underflow'] is True


@pytest.mark.parametrize('power', (False, True))
def test_particle_plane_preserves_doubled_local_native_coefficients(power):
    points = [[1.0, 2.0, 2.0], [3.0, 2.0, 2.0]]
    radii = np.array([0.5, 1.0]) if power else None
    packet = _box(points, [0, 1], radii=radii)
    for cell in packet['cells']:
        pid = cell['id']
        origins = {o['token']: o for o in cell['origins']}
        particle_faces = [f for f in cell['faces']
                          if origins[f['token']]['kind'] == 'particle']
        assert len(particle_faces) == 1
        face = particle_faces[0]
        origin = origins[face['token']]
        assert origin['owner'] == 1 - pid
        normal = [2.0 if pid == 0 else -2.0, 0.0, 0.0]
        np.testing.assert_array_equal(origin['normal'], normal)
        expected = ((3.25, 4.75)[pid] if power else 4.0)
        assert origin['offset'] == expected
        # The exact dyadic fixture has no tolerance ambiguity.
        vertices = np.asarray(cell['vertices_doubled'])[face['vertices']]
        np.testing.assert_array_equal(vertices @ normal, expected)


def test_power_plane_offset_operation_order_is_observable():
    radius = float(2**27)
    offsets = _core._test_power_offset_order(1.0, radius, radius)
    # r_scale evaluates (1 + 2**54) - 2**54, losing the initial one.
    # r_scale_check evaluates 1 + (2**54 - 2**54), retaining it.
    assert _bits(offsets['r_scale']) == _bits(0.0)
    assert _bits(offsets['r_scale_check_offset']) == _bits(1.0)
    assert offsets['r_scale_check_passes'] is True


@pytest.mark.parametrize('observe', (_box, _periodic))
@pytest.mark.parametrize('power', (False, True))
def test_empty_input_keeps_native_packet_context(observe, power):
    radii = np.empty(0, dtype=np.float64) if power else None
    packet = observe(np.empty((0, 3)), [], radii=radii)
    assert packet['cells'] == []
    assert packet['sites'] == []
    assert packet['context']['power'] is power
    assert tuple(packet['context']['blocks']) == BLOCKS
    assert packet['context']['init_mem'] == 8


@pytest.mark.parametrize('observe', (_box, _periodic))
def test_observer_data_remains_owned_after_inputs_and_container_change(observe):
    points = np.array([[1.0, 1.0, 1.0]])
    radii = np.array([np.nextafter(0.5, 1.0)])
    packet = observe(points, [0], radii=radii)
    points[:] = 3.0
    radii[:] = 2.0
    observe(points, [0], radii=radii)
    assert packet['sites'][0]['site'] == [1.0, 1.0, 1.0]
    assert _bits(packet['sites'][0]['radius']) == _bits(
        np.nextafter(0.5, 1.0)
    )
    _assert_cycles(packet['cells'][0])


@pytest.mark.parametrize('observe', (_box, _periodic))
@pytest.mark.parametrize('power', (False, True))
@pytest.mark.parametrize('invalid', ('blocks', 'ids', 'duplicates'))
def test_observer_uses_existing_native_construction_preflight(
    observe, power, invalid,
):
    points = np.array([[1.0, 1.0, 1.0], [3.0, 2.0, 2.0]])
    ids = [0, 1]
    radii = np.array([0.25, 0.5]) if power else None
    kwargs = {}
    if invalid == 'blocks':
        kwargs['blocks'] = (0, 1, 1)
    elif invalid == 'ids':
        ids = [0, 0]
    else:
        points[1] = points[0]
    with pytest.raises(ValueError):
        observe(points, ids, radii=radii, **kwargs)


def test_zero_volume_owner_can_supply_a_genuine_surviving_face():
    points = np.array([[0.5, 1.0, 1.0], [1.5, 1.0, 1.0],
                       [2.5, 1.0, 1.0]])
    radii = np.array([1.0, 0.0, 1.0])
    bounds = ((0.0, 4.0), (0.0, 2.0), (0.0, 2.0))
    packet = _box(points, [0, 1, 2], radii=radii, bounds=bounds)
    _assert_legacy_equivalent(
        packet, _legacy_box(points, [0, 1, 2], (False,) * 3, radii,
                            bounds=bounds),
    )
    middle = next(cell for cell in packet['cells'] if cell['id'] == 1)
    assert middle['volume'] == 0.0
    assert packet['sites'][1]['id'] == 1
    witnessed = []
    for cell in packet['cells']:
        if cell['id'] == 1:
            continue
        origins = {o['token']: o for o in cell['origins']}
        for face in cell['faces']:
            origin = origins[face['token']]
            if origin['kind'] == 'particle' and origin['owner'] == 1:
                vertices = np.asarray(cell['vertices_doubled'])[face['vertices']]
                vertices = vertices * 0.5 + np.asarray(cell['site'])
                # Exact corners independently establish this square face.
                assert {tuple(v) for v in vertices} == {
                    (1.5, 0.0, 0.0), (1.5, 0.0, 2.0),
                    (1.5, 2.0, 0.0), (1.5, 2.0, 2.0),
                }
                witnessed.append((cell['id'], face['token']))
    assert witnessed
