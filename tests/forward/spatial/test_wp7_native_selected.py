"""Selected ghost execution has initialized storage and one source cell."""

import numpy as np
import pytest
import ctypes
import os
import sys

from pyvoro2 import _core


@pytest.mark.parametrize('power', [False, True])
@pytest.mark.parametrize('periodic', [(False, False, False), (True, False, True),
                                      (True, True, True)])
def test_selected_box_packet(power, periodic):
    points = np.array([[.2, .3, .4]], dtype=float)
    ids = np.array([0], dtype=np.int32)
    queries = np.array([[.8, .7, .6], [.8, .7, .6]])
    radii = np.array([.1]) if power else None
    ghost_radii = np.array([.2, .3]) if power else None
    packets = _core._observe_ghost_box(
        points, ids, ((0., 1.),) * 3, (1, 1, 1), periodic, 1,
        queries, radii, ghost_radii,
    )
    assert len(packets) == 2
    for index, packet in enumerate(packets):
        assert packet['query_index'] == index
        assert packet['ghost_internal_id'] == 1
        assert len(packet['sites']) == 2
        assert [site['id'] for site in packet['sites']] == [0, 1]
        assert len(packet['cells']) == 1
        assert packet['cells'][0]['id'] == 1
        assert packet['cells'][0]['noninterference']['computed'] is True


@pytest.mark.parametrize('power', [False, True])
def test_selected_triclinic_packet(power):
    points = np.array([[.2, .3, .4]], dtype=float)
    ids = np.array([0], dtype=np.int32)
    queries = np.array([[.8, .7, .6]], dtype=float)
    radii = np.array([.1]) if power else None
    ghost_radii = np.array([.2]) if power else None
    packets = _core._observe_ghost_periodic(
        points, ids, (1., .1, 1., .2, .1, 1.), (1, 1, 1), 1,
        queries, radii, ghost_radii,
    )
    assert len(packets) == 1
    packet = packets[0]
    assert [site['id'] for site in packet['sites']] == [0, 1]
    assert len(packet['cells']) == 1
    assert packet['cells'][0]['id'] == 1
    assert packet['cells'][0]['noninterference']['computed'] is True


def test_selected_ghost_growth_and_empty_population():
    points = np.empty((0, 3), dtype=float)
    ids = np.empty((0,), dtype=np.int32)
    queries = np.array([[.5, .5, .5]])
    packets = _core._observe_ghost_box(
        points, ids, ((0., 1.),) * 3, (1, 1, 1),
        (True, True, True), 1, queries,
    )
    assert [site['id'] for site in packets[0]['sites']] == [0]
    assert packets[0]['cells'][0]['computed'] is True
    assert any(origin['owner'] == 0 and origin['periodic']
               for origin in packets[0]['cells'][0]['origins'])

    points = np.array([[.15, .2, .3], [.7, .8, .9]])
    ids = np.array([1, 0], dtype=np.int32)
    packets = _core._observe_ghost_periodic(
        points, ids, (1., .1, 1., .2, .1, 1.), (1, 1, 1), 1, queries,
    )
    assert [site['id'] for site in packets[0]['sites']] == [0, 1, 2]
    assert packets[0]['sites'][2]['block_slot'] == 2
    assert packets[0]['cells'][0]['noninterference']['volume'] is True


def test_geometry_only_never_exposes_temporary_owner():
    points = np.empty((0, 3), dtype=float)
    ids = np.empty((0,), dtype=np.int32)
    queries = np.array([[.5, .5, .5]])
    cells = _core.ghost_box_standard(
        points, ids, ((0., 1.),) * 3, (1, 1, 1),
        (True, True, True), 1, (False, False, True), queries,
    )
    assert len(cells) == 1
    assert cells[0]['site'] == [.5, .5, .5]
    assert all('adjacent_cell' not in face for face in cells[0]['faces'])


@pytest.mark.parametrize('periodic_domain', [False, True])
@pytest.mark.parametrize('power', [False, True])
def test_geometry_only_all_native_routes_replay_augmented_storage(
        periodic_domain, power):
    points = np.array([[.2, .3, .4], [.7, .6, .5]])
    ids = np.array([0, 1], dtype=np.int32)
    queries = np.array([[.8, .7, .6]])
    radii = np.array([.1, .2])
    opts = (True, True, True)
    if periodic_domain:
        common = (points, ids, (1., .1, 1., .2, .1, 1.),
                  (1, 1, 1), 1, opts, queries)
        if power:
            cells = _core.ghost_periodic_power(
                points, ids, radii, *common[2:], np.array([.3]))
        else:
            cells = _core.ghost_periodic_standard(*common)
    else:
        common = (points, ids, ((0., 1.),) * 3, (1, 1, 1),
                  (True, True, True), 1, opts, queries)
        if power:
            cells = _core.ghost_box_power(
                points, ids, radii, *common[2:], np.array([.3]))
        else:
            cells = _core.ghost_box_standard(*common)
    assert len(cells) == 1
    assert cells[0]['site'] == [.8, .7, .6]
    assert all(face.get('adjacent_cell') != len(points)
               for face in cells[0]['faces'])


@pytest.mark.skipif(os.name != 'posix', reason='fenv probe uses libc')
def test_selected_environment_refusal_and_zero_query_validation():
    libc = ctypes.CDLL(None)
    if not hasattr(libc, 'fegetround') or not hasattr(libc, 'fesetround'):
        pytest.skip('C floating environment unavailable')
    nearest = libc.fegetround()
    upward = 0x800  # FE_UPWARD on the qualified Linux x86_64 cohort.
    points = np.empty((0, 3), dtype=float)
    ids = np.empty((0,), dtype=np.int32)
    bounds = ((0., 1.),) * 3
    try:
        assert libc.fesetround(upward) == 0
        with pytest.raises(ValueError, match=(
            r'^ghost_native:native:None:GHOST_NATIVE_UNSUPPORTED:'
        )):
            _core._observe_ghost_box(
                points, ids, bounds, (1, 1, 1),
                (True, True, True), 1, np.array([[.5, .5, .5]]),
            )
        assert _core._observe_ghost_box(
            points, ids, bounds, (1, 1, 1),
            (True, True, True), 1, np.empty((0, 3)),
        ) == []
    finally:
        libc.fesetround(nearest)


def test_selected_batch_rejects_retained_packet_budget_before_computation():
    # 105 * ((1000+1)*2048 + 16384) exceeds the private 64 MiB packet cap.
    grid = np.arange(10, dtype=float) / 10 + .04
    points = np.array(np.meshgrid(grid, grid, grid)).reshape(3, -1).T.copy()
    ids = np.arange(len(points), dtype=np.int32)
    queries = np.tile(np.array([[.505, .505, .505]]), (105, 1))
    with pytest.raises(ValueError, match=(
        r'^ghost_native:preparation:None:GHOST_CERTIFICATION_RESOURCE:'
    )):
        _core._observe_ghost_box(
            points, ids, ((0., 1.),) * 3, (1, 1, 1),
            (False, False, False), 1, queries,
        )


def test_selected_packet_budget_covers_recursive_python_payload():
    # A representative larger 3D packet checks the CPython charged envelope,
    # including topology lists/dicts. It does not assert exact allocator RSS.
    grid = np.arange(5, dtype=float) / 5 + .06
    points = np.array(np.meshgrid(grid, grid, grid)).reshape(3, -1).T.copy()
    ids = np.arange(len(points), dtype=np.int32)
    packet = _core._observe_ghost_box(
        points, ids, ((0., 1.),) * 3, (2, 2, 2),
        (True, True, True), 1, np.array([[.53, .48, .57]]),
    )[0]
    cell = packet['cells'][0]
    charged = ((len(points) + 1) * 2048 + 16384 +
               len(cell['origins']) * 2048 + len(cell['faces']) * 2048 +
               len(cell['vertices_doubled']) * 512 +
               sum(cell['vertex_orders']) * 512)
    seen = set()

    def recursive_size(value):
        if id(value) in seen:
            return 0
        seen.add(id(value))
        size = sys.getsizeof(value)
        if isinstance(value, dict):
            return size + sum(recursive_size(k) + recursive_size(v)
                              for k, v in value.items())
        if isinstance(value, (list, tuple)):
            return size + sum(recursive_size(item) for item in value)
        return size

    assert recursive_size(packet) <= charged
