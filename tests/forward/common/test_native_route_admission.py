"""Component composition, no-work exclusions, and replay entry guards."""
from __future__ import annotations

import importlib
import subprocess
import sys
import textwrap
from types import SimpleNamespace

import numpy as np
import pytest

from pyvoro2 import _core, _core2d
from pyvoro2._internal import native_qualification as qualification
from pyvoro2._internal.native_qualification import NativeQualificationError


@pytest.mark.parametrize('component,owner,entry,code', [
    ('wp5-spatial', 'spatial.wp5_common', 'require_wp5',
     'WP5_SOURCE_PROFILE_MISMATCH'),
    ('wp6-planar', 'planar.wp6_certificate', 'require_wp6',
     'WP6_PROFILE_UNSUPPORTED'),
    ('wp7-spatial', 'spatial.ghost_certificate', '_admit',
     'GHOST_NATIVE_UNSUPPORTED'),
    ('wp7-planar', 'planar.ghost_certificate', '_admit',
     'GHOST_NATIVE_UNSUPPORTED'),
    ('wp8-spatial', 'locate', '_admit', 'LOCATE_NATIVE_UNSUPPORTED'),
    ('wp8-planar', 'locate', '_admit', 'LOCATE_NATIVE_UNSUPPORTED'),
])
def test_external_refusal_keeps_route_family_and_component(
        monkeypatch, component, owner, entry, code):
    seen = []

    def refuse(module, requested):
        seen.append((module, requested))
        raise NativeQualificationError('payload_mismatch', 'test changed payload')

    monkeypatch.setattr(qualification, 'require_native', refuse)
    target = getattr(importlib.import_module('pyvoro2._internal.' + owner), entry)
    args = ((2 if component.endswith('-planar') else 3),) \
        if component.startswith('wp8') else ()
    kwargs = {'artifact': True} if component.startswith('wp7') else {}
    with pytest.raises(Exception) as caught:
        target(*args, **kwargs)
    assert caught.value.code == code
    details = getattr(caught.value, 'context', getattr(caught.value, 'details', {}))
    assert details['reason'] == 'payload_mismatch'
    assert seen == [(_core2d if component.endswith('-planar') else _core, component)]


def test_favorable_native_profile_cannot_grant_planar_admission(monkeypatch):
    from pyvoro2._internal.planar.wp6_profile import validate_profile

    def refuse(module, component):
        raise NativeQualificationError('missing_qualification', 'no external record')

    monkeypatch.setattr(qualification, 'require_native', refuse)
    profile = dict(_core2d._planar_witness_profile(), qualified=True,
                   source_supported=True, cohort_supported=True)
    with pytest.raises(RuntimeError, match='profile:missing_qualification:'):
        validate_profile(profile)


@pytest.mark.parametrize('dimension', [2, 3])
def test_empty_ghost_consumer_does_not_request_component(monkeypatch, dimension):
    name = 'planar' if dimension == 2 else 'spatial'
    module = importlib.import_module(f'pyvoro2._internal.{name}.ghost_certificate')

    def forbidden(component):
        pytest.fail('empty ghost batch requested native qualification')

    monkeypatch.setattr(module, 'require_component', forbidden)
    prepared = SimpleNamespace(internal_ids=())
    temporary = SimpleNamespace(internal_ids=(), native_points=())
    kwargs = dict(prepared=prepared, temporary=temporary, power_input=None,
                  domain=None)
    if dimension == 2:
        result = module.certify_ghost_packets([], [], return_edge_shifts=True,
                                              **kwargs)
    else:
        result = module.certify_ghost_packets([], snapshot=None,
                                              return_vertices=True,
                                              return_adjacency=True, **kwargs)
    assert result == []


@pytest.mark.parametrize('dimension', [2, 3])
@pytest.mark.parametrize('empty', [False, True])
def test_id_only_and_empty_locate_do_not_request_owner_component(
        monkeypatch, dimension, empty):
    from pyvoro2._internal import locate

    def forbidden(component):
        pytest.fail('locate requested unnecessary owner certificate')

    monkeypatch.setattr(locate, 'require_component', forbidden)
    count = 0 if empty else 1
    queries = np.full((count, dimension), .25)
    prepared = SimpleNamespace(
        internal_ids=np.array([0], dtype=np.int32),
        native_points=np.full((1, dimension), .5),
        external_ids=np.array([91], dtype=np.int64),
    )
    geometry = SimpleNamespace(
        dim=dimension, has_any_periodic_axis=True,
        periodic_axes=(True,) * dimension,
        native_bounds=((0., 1.),) * dimension,
        lattice_vectors_cart=np.eye(dimension),
    )

    def native(*args, return_source):
        assert not return_source
        return (np.ones(count, dtype=bool), np.zeros(count, dtype=np.int32),
                np.full((count, dimension), .5))

    def core_loader():
        assert not empty, 'empty locate invoked a native loader'
        return SimpleNamespace(locate_box_standard=native)

    result = locate.locate_prepared(
        prepared, queries, geometry=geometry, snapshot=None,
        blocks=(1,) * dimension, init_mem=1, mode='standard',
        return_owner_position=empty, core_loader=core_loader, external_ids=True,
    )
    assert result['owner_id'].tolist() == ([] if empty else [91])
    if empty:
        assert result['owner_shift'].shape == (0, dimension)


@pytest.mark.skipif(sys.platform != 'linux', reason='GNU fenv controls')
@pytest.mark.parametrize('route', ['producer', 'spatial_view', 'planar_view',
                                   'locate_enclosure', 'locate_transform'])
def test_retained_replay_and_transform_recheck_current_state(route):
    script = r'''
        import ctypes
        from types import SimpleNamespace
        import numpy as np
        from pyvoro2 import _core, _core2d
        from pyvoro2._internal.spatial.wp5_producer import Producer
        from pyvoro2._internal.spatial.wp5_certificate import FaceCertificate
        from pyvoro2._internal.planar.wp6_certificate import EdgeCertificate
        from pyvoro2._internal import locate
        lib = ctypes.CDLL(None)
        previous = lib.fegetround()
        queries = np.array([[.25, .5]])
        geometry = SimpleNamespace(
            dim=2, has_any_periodic_axis=True, periodic_axes=(True, True),
            native_bounds=((0., 1.),) * 2, lattice_vectors_cart=np.eye(2))
        prepared = SimpleNamespace(internal_ids=np.array([0]),
                                   native_points=np.array([[.5, .5]]))
        def transform(values):
            assert lib.fesetround(0x800) == 0
            return values
        snapshot = SimpleNamespace(cart_to_internal=transform,
                                   vectors=np.eye(2), origin=np.zeros(2))
        def forbidden():
            raise AssertionError('numeric continuation after poisoned transform')
        actions = {
            'producer': lambda: Producer.__new__(Producer).attribute(None, None),
            'spatial_view': lambda: FaceCertificate.__new__(
                FaceCertificate).public_cells(return_vertices=True,
                    return_adjacency=True, include_empty=True),
            'planar_view': lambda: EdgeCertificate.__new__(
                EdgeCertificate).public_cells(vertices=True, adjacency=True,
                    edges=True, shifts=True, include_empty=True),
            'locate_enclosure': lambda: locate.owner_enclosure(
                original=None, preparation=None, stored=None, insertion=None,
                query_removal=None, copy_bounds=None, lattice=None, native_lattice=None,
                snapshot=None, periodic=None),
            'locate_transform': lambda: locate.locate_prepared(
                prepared, queries, geometry=geometry, snapshot=snapshot,
                blocks=(1, 1), init_mem=1, mode='standard', return_owner_position=False,
                core_loader=forbidden, external_ids=False),
        }
        refused = None
        try:
            if ROUTE != 'locate_transform':
                assert lib.fesetround(0x800) == 0
            try:
                actions[ROUTE]()
            except Exception as exc:
                refused = getattr(exc, 'code', None)
            assert lib.fegetround() == 0x800, 'runtime guard normalized caller controls'
        finally:
            assert lib.fesetround(previous) == 0
        assert refused in ('WP5_UNSUPPORTED_FP_PROFILE', 'WP6_PROFILE_UNSUPPORTED',
                           'LOCATE_NATIVE_UNSUPPORTED'), refused
    '''
    result = subprocess.run([sys.executable, '-c', 'ROUTE = ' + repr(route) + '\n' +
                             textwrap.dedent(script)], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
