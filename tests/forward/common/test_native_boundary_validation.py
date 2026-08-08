from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
import warnings

import numpy as np
import pytest

import pyvoro2
import pyvoro2.api as api3d
import pyvoro2.planar as planar
import pyvoro2.planar.api as api2d
import pyvoro2._internal.inputs as input_helpers
import pyvoro2._internal.spatial.domain_geometry as domain_geometry
from pyvoro2._internal.validation import CPP_INT_MAX


@dataclass
class RecordingCore:
    calls: list[tuple[str, tuple[Any, ...]]] = field(default_factory=list)

    def __getattr__(self, name: str):
        def call(*args: Any):
            self.calls.append((name, args))
            if name.startswith('locate_'):
                queries = np.asarray(args[-1])
                dim = int(queries.shape[1])
                count = int(queries.shape[0])
                return (
                    np.zeros(count, dtype=np.bool_),
                    np.full(count, -1, dtype=np.int32),
                    np.full((count, dim), np.nan, dtype=np.float64),
                )
            return []

        return call


@dataclass
class NoNativeCalls:
    count: int = 0

    def __getattr__(self, name: str):
        def call(*args: Any):
            self.count += 1
            raise AssertionError(f'native method {name} was called')

        return call


@dataclass(frozen=True)
class IndexValue:
    value: int

    def __index__(self) -> int:
        return self.value


BOUNDARIES = (
    (3, 'compute'),
    (3, 'locate'),
    (3, 'ghost_cells'),
    (2, 'compute'),
    (2, 'locate'),
    (2, 'ghost_cells'),
)


def _install_core(monkeypatch, dim: int, core: object) -> None:
    if dim == 3:
        monkeypatch.setattr(api3d, '_core', core)
        monkeypatch.setattr(api3d, '_CORE_IMPORT_ERROR', None)
    else:
        monkeypatch.setattr(api2d, '_core2d', core)
        monkeypatch.setattr(api2d, '_CORE2D_IMPORT_ERROR', None)


def _default_points(dim: int) -> np.ndarray:
    return np.array(
        [[0] * dim, [1] * dim],
        dtype=np.int64,
    )


def _default_domain(dim: int):
    bounds = tuple((0.0, 2.0) for _ in range(dim))
    if dim == 3:
        return pyvoro2.Box(bounds=bounds)
    return planar.Box(bounds=bounds)


def _mutable_periodic_cell():
    vectors: list[list[object]] = [
        [2.0, 0.0, 0.0],
        [0.25, 2.0, 0.0],
        [0.1, -0.2, 2.0],
    ]
    origin: list[object] = [0.1, -0.1, 0.2]
    cell = pyvoro2.PeriodicCell(vectors=vectors, origin=origin)
    return cell, vectors, origin


def _warning_level_periodic_cell() -> pyvoro2.PeriodicCell:
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        return pyvoro2.PeriodicCell(
            vectors=(
                (1.0, 0.0, 0.0),
                (0.0, 1.0, 0.0),
                (0.0, 0.0, 1e-11),
            )
        )


def _invoke(dim: int, operation: str, **overrides: Any):
    package = pyvoro2 if dim == 3 else planar
    points = overrides.pop('points', _default_points(dim))
    queries = overrides.pop(
        'queries',
        np.full((1, dim), 0.5, dtype=np.float64),
    )
    domain = overrides.pop('domain', _default_domain(dim))
    kwargs: dict[str, Any] = {'domain': domain}
    if operation == 'compute':
        kwargs['output'] = 'cells'
        kwargs['return_vertices'] = False
        kwargs['return_adjacency'] = False
        if dim == 3:
            kwargs['return_faces'] = False
        else:
            kwargs['return_edges'] = False
        kwargs.update(overrides)
        return package.compute(points, **kwargs)
    if operation == 'locate':
        kwargs.update(overrides)
        return package.locate(points, queries, **kwargs)

    kwargs['return_vertices'] = False
    kwargs['return_adjacency'] = False
    if dim == 3:
        kwargs['return_faces'] = False
    else:
        kwargs['return_edges'] = False
    kwargs.update(overrides)
    return package.ghost_cells(points, queries, **kwargs)


@pytest.mark.parametrize(('dim', 'operation'), BOUNDARIES)
@pytest.mark.parametrize(
    ('name', 'value', 'message'),
    [
        ('init_mem', 0, 'init_mem.*positive'),
        ('blocks', 'zero-block', r'blocks\[0\].*positive'),
        ('block_size', 0.0, 'block_size.*positive'),
    ],
)
def test_every_native_construction_rejects_invalid_controls_before_call(
    monkeypatch,
    dim: int,
    operation: str,
    name: str,
    value: object,
    message: str,
) -> None:
    core = NoNativeCalls()
    _install_core(monkeypatch, dim, core)
    if name == 'blocks':
        value = (0,) + (1,) * (dim - 1)

    with pytest.raises(ValueError, match=message):
        _invoke(dim, operation, **{name: value})
    assert core.count == 0


@pytest.mark.parametrize(
    ('value', 'message'),
    [
        (0, 'positive'),
        (-1, 'positive'),
        (True, 'exact integer'),
        (np.bool_(False), 'exact integer'),
        (1.0, 'exact integer'),
        (1.5, 'exact integer'),
        ('1', 'exact integer'),
        (1 + 0j, 'exact integer'),
        (np.array(1), 'exact integer'),
        (CPP_INT_MAX + 1, 'destination range'),
    ],
)
def test_init_mem_strict_scalar_matrix_rejects_before_native(
    monkeypatch,
    value: object,
    message: str,
) -> None:
    core = NoNativeCalls()
    _install_core(monkeypatch, 3, core)

    with pytest.raises(ValueError, match=f'init_mem.*{message}'):
        _invoke(3, 'compute', init_mem=value)
    assert core.count == 0


@pytest.mark.parametrize(
    'value',
    [1, np.int32(2), np.uint32(3), IndexValue(4)],
)
def test_init_mem_accepts_positive_exact_index_scalars(
    monkeypatch,
    value: object,
) -> None:
    core = RecordingCore()
    _install_core(monkeypatch, 3, core)

    _invoke(3, 'compute', init_mem=value)
    assert core.calls[0][1][5] == int(value.__index__())


@pytest.mark.parametrize(
    ('value', 'message'),
    [
        (0, 'positive'),
        (-1, 'positive'),
        (True, 'exact integer'),
        (np.bool_(False), 'exact integer'),
        (1.0, 'exact integer'),
        (1.5, 'exact integer'),
        ('1', 'exact integer'),
        (1 + 0j, 'exact integer'),
        (np.array(1), 'exact integer'),
        (CPP_INT_MAX + 1, 'destination range'),
    ],
)
def test_explicit_block_count_strict_scalar_matrix(
    monkeypatch,
    value: object,
    message: str,
) -> None:
    core = NoNativeCalls()
    _install_core(monkeypatch, 3, core)

    with pytest.raises(ValueError, match=rf'blocks\[0\].*{message}'):
        _invoke(3, 'compute', blocks=(value, 1, 1))
    assert core.count == 0


@pytest.mark.parametrize(
    'value',
    [1, np.int64(2), np.uint64(3), IndexValue(4)],
)
def test_explicit_blocks_accept_positive_exact_index_scalars(
    monkeypatch,
    value: object,
) -> None:
    core = RecordingCore()
    _install_core(monkeypatch, 3, core)

    _invoke(3, 'compute', blocks=(value, 1, 1))
    assert core.calls[0][1][3] == (int(value.__index__()), 1, 1)


@pytest.mark.parametrize(
    'value',
    [0.0, -1.0, np.nan, np.inf, -np.inf, True, 1 + 0j, '1.0', np.array(1.0)],
)
def test_block_size_rejects_invalid_real_scalars_before_native(
    monkeypatch,
    value: object,
) -> None:
    core = NoNativeCalls()
    _install_core(monkeypatch, 3, core)

    with pytest.raises(ValueError, match='block_size.*(real|finite|positive)'):
        _invoke(3, 'compute', block_size=value)
    assert core.count == 0


@pytest.mark.parametrize('value', [1, 0.5, np.float32(0.25)])
def test_block_size_accepts_positive_finite_real_scalars(
    monkeypatch,
    value: object,
) -> None:
    core = RecordingCore()
    _install_core(monkeypatch, 3, core)

    _invoke(3, 'compute', block_size=value)
    assert core.calls


@pytest.mark.parametrize('dim', [2, 3])
@pytest.mark.parametrize(
    ('kind', 'message'),
    [
        ('shape', 'shape'),
        ('bool', 'real numeric'),
        ('string', 'real numeric'),
        ('complex', 'real numeric'),
        ('nan', 'finite'),
        ('inf', 'finite'),
        ('negative_inf', 'finite'),
    ],
)
def test_points_reject_invalid_shape_kind_and_finiteness_before_native(
    monkeypatch,
    dim: int,
    kind: str,
    message: str,
) -> None:
    core = NoNativeCalls()
    _install_core(monkeypatch, dim, core)
    if kind == 'shape':
        points = np.zeros(dim)
    elif kind == 'bool':
        points = np.zeros((2, dim), dtype=np.bool_)
    elif kind == 'string':
        points = np.full((2, dim), '0')
    elif kind == 'complex':
        points = np.zeros((2, dim), dtype=np.complex128)
    else:
        points = np.zeros((2, dim), dtype=np.float64)
        if kind == 'nan':
            points[0, 0] = np.nan
        elif kind == 'inf':
            points[0, 0] = np.inf
        else:
            points[0, 0] = -np.inf

    with pytest.raises(ValueError, match=f'points.*{message}'):
        _invoke(dim, 'compute', points=points)
    assert core.count == 0


@pytest.mark.parametrize('dim', [2, 3])
@pytest.mark.parametrize('operation', ['locate', 'ghost_cells'])
@pytest.mark.parametrize(
    ('queries', 'message'),
    [
        (np.zeros(2), 'shape'),
        (np.array([[True, False, True]], dtype=np.bool_), 'real numeric'),
        (np.array([['0', '0', '0']]), 'real numeric'),
        (np.array([[0.0, np.nan, 0.0]]), 'finite'),
        (np.array([[0.0, np.inf, 0.0]]), 'finite'),
        (np.array([[0.0, -np.inf, 0.0]]), 'finite'),
    ],
)
def test_queries_reject_invalid_arrays_before_native(
    monkeypatch,
    dim: int,
    operation: str,
    queries: np.ndarray,
    message: str,
) -> None:
    core = NoNativeCalls()
    _install_core(monkeypatch, dim, core)
    query_values = queries[:, :dim] if queries.ndim == 2 else queries

    with pytest.raises(ValueError, match=f'queries.*{message}'):
        _invoke(dim, operation, queries=query_values)
    assert core.count == 0


@pytest.mark.parametrize('dim', [2, 3])
@pytest.mark.parametrize(
    ('radii', 'message'),
    [
        (np.zeros((2, 1)), 'shape'),
        (np.array([True, False]), 'real numeric'),
        (np.array(['0', '1']), 'real numeric'),
        (np.array([0 + 0j, 1 + 0j]), 'real numeric'),
        (np.array([0.0, np.nan]), 'finite'),
        (np.array([0.0, np.inf]), 'finite'),
        (np.array([0.0, -np.inf]), 'finite'),
        (np.array([0.0, -1.0]), 'non-negative'),
    ],
)
def test_radii_reject_invalid_arrays_before_native(
    monkeypatch,
    dim: int,
    radii: np.ndarray,
    message: str,
) -> None:
    core = NoNativeCalls()
    _install_core(monkeypatch, dim, core)

    with pytest.raises(ValueError, match=f'radii.*{message}'):
        _invoke(dim, 'compute', mode='power', radii=radii)
    assert core.count == 0


@pytest.mark.parametrize(('dim', 'operation'), BOUNDARIES)
def test_each_power_native_construction_rejects_boolean_radii_before_call(
    monkeypatch,
    dim: int,
    operation: str,
) -> None:
    core = NoNativeCalls()
    _install_core(monkeypatch, dim, core)
    kwargs: dict[str, object] = {
        'mode': 'power',
        'radii': np.array([True, False]),
    }
    if operation == 'ghost_cells':
        kwargs['ghost_radius'] = 0.0

    with pytest.raises(ValueError, match='radii.*real numeric'):
        _invoke(dim, operation, **kwargs)
    assert core.count == 0


@pytest.mark.parametrize('dim', [2, 3])
@pytest.mark.parametrize(
    ('weights', 'message'),
    [
        (np.array([True, False]), 'real numeric'),
        (np.array(['0', '1']), 'real numeric'),
        (np.array([0 + 0j, 1 + 0j]), 'real numeric'),
        (np.array([0.0, np.nan]), 'finite'),
    ],
)
def test_weights_that_form_native_radii_are_strictly_validated(
    monkeypatch,
    dim: int,
    weights: np.ndarray,
    message: str,
) -> None:
    core = NoNativeCalls()
    _install_core(monkeypatch, dim, core)

    with pytest.raises(ValueError, match=f'weights.*{message}'):
        _invoke(dim, 'compute', mode='power', weights=weights)
    assert core.count == 0


@pytest.mark.parametrize('dim', [2, 3])
def test_integer_coordinate_arrays_reach_standard_and_power_native_calls(
    monkeypatch,
    dim: int,
) -> None:
    core = RecordingCore()
    _install_core(monkeypatch, dim, core)
    points = _default_points(dim)

    _invoke(dim, 'compute', points=points)
    _invoke(dim, 'compute', points=points, mode='power', radii=np.array([0, 1]))

    assert len(core.calls) == 2
    for _name, args in core.calls:
        assert np.asarray(args[0]).dtype == np.float64
    assert np.asarray(core.calls[1][1][2]).dtype == np.float64


@pytest.mark.parametrize('dim', [2, 3])
@pytest.mark.parametrize('bad_value', [0.0, np.nan, True, '0'])
def test_domain_bounds_are_owned_after_construction(
    monkeypatch,
    dim: int,
    bad_value: object,
) -> None:
    core = RecordingCore()
    _install_core(monkeypatch, dim, core)
    bounds = [[0.0, 2.0] for _ in range(dim)]
    domain_type = pyvoro2.Box if dim == 3 else planar.Box
    domain = domain_type(bounds=bounds)
    bounds[0][1] = bad_value

    _invoke(dim, 'compute', domain=domain)

    assert domain.bounds == tuple((0.0, 2.0) for _ in range(dim))
    assert len(core.calls) == 1


@pytest.mark.parametrize(
    'vectors',
    [
        [['2', '0', '0'], ['0.25', '2', '0'], ['0.1', '-0.2', '2']],
        [[True, False, False], [False, True, False], [False, False, True]],
    ],
)
def test_periodic_vector_kinds_are_rejected_at_construction(
    vectors: list[list[object]],
) -> None:
    with pytest.raises(ValueError, match=r'vectors.*real numeric'):
        pyvoro2.PeriodicCell(vectors=vectors)


@pytest.mark.parametrize('operation', ['compute', 'locate', 'ghost_cells'])
@pytest.mark.parametrize('bad_value', [2 + 0j, True, '2.0', np.nan, np.inf])
def test_periodic_vectors_are_owned_without_numpy_warning(
    monkeypatch,
    operation: str,
    bad_value: object,
) -> None:
    core = RecordingCore()
    _install_core(monkeypatch, 3, core)
    domain, vectors, _origin = _mutable_periodic_cell()
    vectors[0][0] = bad_value

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        _invoke(3, operation, domain=domain)
    assert caught == []
    assert domain.vectors == (
        (2.0, 0.0, 0.0),
        (0.25, 2.0, 0.0),
        (0.1, -0.2, 2.0),
    )
    assert len(core.calls) == 1


@pytest.mark.parametrize('operation', ['compute', 'locate', 'ghost_cells'])
@pytest.mark.parametrize('bad_value', [0.1 + 0j, False, '0.1', np.nan, -np.inf])
def test_periodic_origin_is_owned_without_numpy_warning(
    monkeypatch,
    operation: str,
    bad_value: object,
) -> None:
    core = RecordingCore()
    _install_core(monkeypatch, 3, core)
    domain, _vectors, origin = _mutable_periodic_cell()
    origin[0] = bad_value

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        _invoke(3, operation, domain=domain)
    assert caught == []
    assert domain.origin == (0.1, -0.1, 0.2)
    assert len(core.calls) == 1


@pytest.mark.parametrize('bad_value', [True, '0.1'])
def test_periodic_origin_kinds_are_rejected_at_construction(
    bad_value: object,
) -> None:
    with pytest.raises(ValueError, match=r'origin.*real numeric'):
        pyvoro2.PeriodicCell(
            vectors=((2.0, 0.0, 0.0), (0.25, 2.0, 0.0), (0.1, -0.2, 2.0)),
            origin=(bad_value, 0.0, 0.0),
        )


@pytest.mark.parametrize('target', ['vectors', 'origin'])
def test_periodic_shapes_are_owned_after_construction(
    monkeypatch,
    target: str,
) -> None:
    core = RecordingCore()
    _install_core(monkeypatch, 3, core)
    domain, vectors, origin = _mutable_periodic_cell()
    if target == 'vectors':
        vectors.append([0.0, 0.0, 1.0])
    else:
        origin.append(0.0)

    _invoke(3, 'compute', domain=domain)

    assert np.asarray(domain.vectors).shape == (3, 3)
    assert np.asarray(domain.origin).shape == (3,)
    assert len(core.calls) == 1


def test_warning_level_periodic_cell_warns_only_at_explicit_construction(
    monkeypatch,
) -> None:
    with pytest.warns(
        RuntimeWarning,
        match='PeriodicCell lattice vectors are very ill-conditioned',
    ):
        domain = pyvoro2.PeriodicCell(
            vectors=(
                (1.0, 0.0, 0.0),
                (0.0, 1.0, 0.0),
                (0.0, 0.0, 1e-11),
            )
        )

    core = RecordingCore()
    _install_core(monkeypatch, 3, core)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        _invoke(3, 'compute', domain=domain, blocks=(1, 1, 1))

    replayed = [
        warning
        for warning in caught
        if 'PeriodicCell lattice vectors are very ill-conditioned'
        in str(warning.message)
    ]
    assert replayed == []
    assert core.calls[0][0] == 'compute_periodic_standard'


@pytest.mark.parametrize('blocks', [None, (1, 1, 1)])
def test_warning_level_compute_reuses_one_snapshot_under_warning_as_error(
    monkeypatch,
    blocks: tuple[int, int, int] | None,
) -> None:
    domain = _warning_level_periodic_cell()
    core = RecordingCore()
    _install_core(monkeypatch, 3, core)
    original = domain_geometry.DomainGeometry3D.native_periodic_snapshot
    snapshot_calls = 0

    def counted_snapshot(self):
        nonlocal snapshot_calls
        snapshot_calls += 1
        return original(self)

    monkeypatch.setattr(
        domain_geometry.DomainGeometry3D,
        'native_periodic_snapshot',
        counted_snapshot,
    )
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        _invoke(3, 'compute', domain=domain, blocks=blocks)

    assert snapshot_calls == 1
    assert core.calls[0][0] == 'compute_periodic_standard'


@pytest.mark.parametrize('operation', ['locate', 'ghost_cells'])
def test_warning_level_query_operations_reach_native_under_warning_as_error(
    monkeypatch,
    operation: str,
) -> None:
    domain = _warning_level_periodic_cell()
    core = RecordingCore()
    _install_core(monkeypatch, 3, core)

    with warnings.catch_warnings():
        warnings.simplefilter('error')
        _invoke(3, operation, domain=domain)

    prefix = 'locate' if operation == 'locate' else 'ghost'
    assert core.calls[0][0] == f'{prefix}_periodic_standard'


@pytest.mark.parametrize('dim', [2, 3])
def test_public_ghost_query_index_boundary_precedes_native_dispatch(
    monkeypatch,
    dim: int,
) -> None:
    monkeypatch.setattr(input_helpers, 'CPP_INT_MAX', 1)
    points = np.zeros((1, dim), dtype=np.float64)

    recording_core = RecordingCore()
    _install_core(monkeypatch, dim, recording_core)
    _invoke(dim, 'ghost_cells', points=points, queries=np.zeros((2, dim)))
    assert len(recording_core.calls) == 1

    rejecting_core = NoNativeCalls()
    _install_core(monkeypatch, dim, rejecting_core)
    with pytest.raises(
        ValueError,
        match=r'queries length.*query indices.*destination range',
    ):
        _invoke(dim, 'ghost_cells', points=points, queries=np.zeros((3, dim)))
    assert rejecting_core.count == 0


def test_public_planar_ghost_reserves_native_ghost_id_before_dispatch(
    monkeypatch,
) -> None:
    monkeypatch.setattr(input_helpers, 'CPP_INT_MAX', 2)

    recording_core = RecordingCore()
    _install_core(monkeypatch, 2, recording_core)
    _invoke(2, 'ghost_cells', points=np.zeros((2, 2)))
    assert len(recording_core.calls) == 1

    rejecting_core = NoNativeCalls()
    _install_core(monkeypatch, 2, rejecting_core)
    with pytest.raises(
        ValueError,
        match=r'points/site length.*destination range.*reserved.*ghost ID',
    ):
        _invoke(2, 'ghost_cells', points=np.zeros((3, 2)))
    assert rejecting_core.count == 0


@pytest.mark.parametrize('operation', ['compute', 'locate', 'ghost_cells'])
@pytest.mark.parametrize('mode', ['standard', 'power'])
def test_valid_periodic_native_parameters_reach_spatial_core(
    monkeypatch,
    operation: str,
    mode: str,
) -> None:
    core = RecordingCore()
    _install_core(monkeypatch, 3, core)
    domain = pyvoro2.PeriodicCell.from_params(2, 0.25, 2, 0.1, -0.2, 2)
    kwargs: dict[str, object] = {'mode': mode}
    if mode == 'power':
        kwargs['radii'] = np.array([0.0, 0.5])
        if operation == 'ghost_cells':
            kwargs['ghost_radius'] = 0.25

    _invoke(3, operation, domain=domain, **kwargs)

    prefix = 'ghost' if operation == 'ghost_cells' else operation
    assert core.calls[0][0] == f'{prefix}_periodic_{mode}'
    params = core.calls[0][1][2 if mode == 'standard' else 3]
    assert np.all(np.isfinite(params))
    assert params[0] > 0 and params[2] > 0 and params[5] > 0
    np.testing.assert_allclose(
        params,
        domain.to_internal_params(),
        rtol=1e-14,
        atol=1e-14,
    )
