"""Unsafe native execution and selector-gated image materialization."""

import numpy as np
import pytest

from test_wp8_metadata import rectangular


@pytest.mark.parametrize('dim', [2, 3])
@pytest.mark.parametrize('periodic', [False, True])
def test_unsafe_native_query_cast_is_structured(dim, periodic):
    api, domain = rectangular(dim, periodic=periodic)
    with pytest.raises(ValueError) as caught:
        api.locate([[.5] * dim], [[float(2**40), *([.5] * (dim - 1))]],
                   domain=domain)
    assert caught.value.code == 'LOCATE_NATIVE_UNSUPPORTED'
    assert caught.value.query_index == 0


@pytest.mark.parametrize('dim', [2, 3])
def test_huge_original_owner_id_only_and_owner_shift_selector(dim):
    api, domain = rectangular(dim)
    points = [[float(2**70), *([.5] * (dim - 1))]]
    queries = [[.125, *([.5] * (dim - 1))]]
    out = api.locate(points, queries, domain=domain)
    assert out['owner_id'].tolist() == [0]
    with pytest.raises(ValueError) as caught:
        api.locate(points, queries, domain=domain, return_owner_position=True)
    assert caught.value.code == 'LOCATE_METADATA_UNREPRESENTABLE'
    assert caught.value.details['field'] == 'owner_shift'


@pytest.mark.parametrize('dim', [2, 3])
def test_zero_queries_skip_native_population_construction(dim, monkeypatch):
    api, domain = rectangular(dim)
    import importlib
    module = importlib.import_module(api.locate.__module__)

    def forbidden():
        pytest.fail('zero-query locate loaded the native module')

    monkeypatch.setattr(module, '_require_core2d' if dim == 2 else '_require_core',
                        forbidden)
    out = api.locate([[.5] * dim], np.empty((0, dim)), domain=domain,
                     return_owner_position=True)
    assert out['found'].shape == (0,)
    for name in ('query', 'query_wrapped', 'query_shift',
                 'owner_site', 'owner_pos', 'owner_shift'):
        assert out[name].shape == (0, dim)


@pytest.mark.skipif(
    __import__('sys').platform != 'linux' or
    __import__('platform').machine() not in ('x86_64', 'amd64'),
    reason='Linux x86 fenv constants',
)
@pytest.mark.parametrize('family', ['standard', 'weights', 'radii'])
def test_triclinic_profile_refusal_precedes_duplicate_preflight(family):
    import ctypes
    import pyvoro2

    libc = ctypes.CDLL(None)
    cell = pyvoro2.PeriodicCell(np.eye(3))
    np.finfo(np.float64)  # Cache NumPy limits before changing the process fenv.
    previous = libc.fegetround()
    opts = {} if family == 'standard' else {'mode': 'power', family: [0., 0.]}
    try:
        assert libc.fesetround(0x800) == 0  # FE_UPWARD on this qualified Linux cohort.
        with pytest.raises(ValueError) as caught:
            pyvoro2.locate([[.25, .5, .5], [.75, .5, .5]], [[.375, .5, .5]],
                           domain=cell, return_owner_position=True, **opts)
    finally:
        assert libc.fesetround(previous) == 0
    assert caught.value.code == 'LOCATE_NATIVE_UNSUPPORTED'
    assert caught.value.stage == 'profile'
