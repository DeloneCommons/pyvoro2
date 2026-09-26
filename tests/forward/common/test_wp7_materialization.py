"""Constructed native output failures must remain atomic and inspectable."""

import importlib
from types import SimpleNamespace

import numpy as np
import pytest


@pytest.mark.parametrize('dim', [2, 3])
@pytest.mark.parametrize('field', ['site', 'vertices', 'measure'])
def test_geometry_only_nonfinite_output_uses_materialization_protocol(
        monkeypatch, dim, field):
    prefix = 'pyvoro2.planar' if dim == 2 else 'pyvoro2'
    package = importlib.import_module(prefix)
    api = importlib.import_module(prefix + '.api')
    measure = 'area' if dim == 2 else 'volume'
    good = dict(id=-1, query_index=0, empty=False, site=[.5] * dim,
                vertices=[[.25] * dim], **{measure: 1.})
    bad = dict(good, query_index=1)
    if field == 'measure':
        bad[measure] = float('inf')
    elif field == 'vertices':
        bad[field] = [[float('inf')] * dim]
    else:
        bad[field] = [float('inf')] * dim
    core = SimpleNamespace(ghost_box_standard=lambda *args: [good, bad])
    monkeypatch.setattr(api, '_require_core2d' if dim == 2 else '_require_core',
                        lambda: core)
    with pytest.raises(ValueError) as caught:
        package.ghost_cells(
            np.empty((0, dim)), [[.5] * dim, [.75] * dim],
            domain=package.Box(((0., 1.),) * dim),
            **{'return_edges' if dim == 2 else 'return_faces': False},
        )
    assert caught.value.code == 'GHOST_SHIFT_UNREPRESENTABLE'
    assert caught.value.stage == 'materialization'
    assert caught.value.query_index == 1


def test_native_prequery_resource_failure_has_no_fabricated_query_index():
    from pyvoro2._internal.ghost import from_native_failure

    failure = from_native_failure(RuntimeError(
        'ghost_native:preparation:None:GHOST_CERTIFICATION_RESOURCE:memory cap'),
        3, None)
    assert failure.code == 'GHOST_CERTIFICATION_RESOURCE'
    assert failure.stage == 'preparation'
    assert failure.query_index is None


def test_actual_native_observer_memory_refusal_is_structured():
    import pyvoro2

    with pytest.raises(ValueError) as caught:
        pyvoro2.ghost_cells(np.empty((0, 3)), [[.5] * 3],
                            domain=pyvoro2.Box(((0., 1.),) * 3),
                            blocks=(100, 100, 100))
    assert caught.value.code == 'GHOST_CERTIFICATION_RESOURCE'
    assert caught.value.stage == 'preparation'
    assert caught.value.query_index is None


def test_empty_ordinary_spatial_source_scope_remains_valid():
    import pyvoro2

    result = pyvoro2.compute(
        np.empty((0, 3)), domain=pyvoro2.OrthorhombicCell(((0., 1.),) * 3),
        return_face_shifts=True,
    )
    assert result.cells == []
