from __future__ import annotations

from dataclasses import dataclass
import subprocess
import sys
import textwrap
from types import ModuleType
from typing import Any

import numpy as np
import pytest

from pyvoro2 import _core, _core2d


CPP_INT_MAX = int(np.iinfo(np.intc).max)


@dataclass(frozen=True)
class NativePath:
    module: ModuleType
    name: str
    dim: int
    geometry: str
    mode: str
    operation: str

    @property
    def label(self) -> str:
        return f'{self.module.__name__.rsplit(".", 1)[-1]}:{self.name}'


# Source-auditable constructor coverage table for issue #38 R3-A. Each row is
# exercised with an invalid precondition and with representative valid data.
NATIVE_CONSTRUCTOR_PATHS = (
    NativePath(_core, 'compute_box_standard', 3, 'box', 'standard', 'compute'),
    NativePath(_core, 'compute_box_power', 3, 'box', 'power', 'compute'),
    NativePath(
        _core,
        'compute_periodic_standard',
        3,
        'periodic',
        'standard',
        'compute',
    ),
    NativePath(
        _core,
        'compute_periodic_power',
        3,
        'periodic',
        'power',
        'compute',
    ),
    NativePath(_core, 'locate_box_standard', 3, 'box', 'standard', 'locate'),
    NativePath(_core, 'locate_box_power', 3, 'box', 'power', 'locate'),
    NativePath(
        _core,
        'locate_periodic_standard',
        3,
        'periodic',
        'standard',
        'locate',
    ),
    NativePath(
        _core,
        'locate_periodic_power',
        3,
        'periodic',
        'power',
        'locate',
    ),
    NativePath(_core, 'ghost_box_standard', 3, 'box', 'standard', 'ghost'),
    NativePath(_core, 'ghost_box_power', 3, 'box', 'power', 'ghost'),
    NativePath(
        _core,
        'ghost_periodic_standard',
        3,
        'periodic',
        'standard',
        'ghost',
    ),
    NativePath(
        _core,
        'ghost_periodic_power',
        3,
        'periodic',
        'power',
        'ghost',
    ),
    NativePath(_core2d, 'compute_box_standard', 2, 'box', 'standard', 'compute'),
    NativePath(_core2d, 'compute_box_power', 2, 'box', 'power', 'compute'),
    NativePath(_core2d, 'locate_box_standard', 2, 'box', 'standard', 'locate'),
    NativePath(_core2d, 'locate_box_power', 2, 'box', 'power', 'locate'),
    NativePath(_core2d, 'ghost_box_standard', 2, 'box', 'standard', 'ghost'),
    NativePath(_core2d, 'ghost_box_power', 2, 'box', 'power', 'ghost'),
)

BOX_CONSTRUCTOR_PATHS = tuple(
    path for path in NATIVE_CONSTRUCTOR_PATHS if path.geometry == 'box'
)
PERIODIC_CONSTRUCTOR_PATHS = tuple(
    path for path in NATIVE_CONSTRUCTOR_PATHS if path.geometry == 'periodic'
)
PLANAR_BOX_CONSTRUCTOR_PATHS = tuple(
    path for path in BOX_CONSTRUCTOR_PATHS if path.dim == 2
)


def _native_args(
    path: NativePath,
    *,
    points: np.ndarray | None = None,
    ids: np.ndarray | None = None,
    radii: np.ndarray | None = None,
    bounds: object | None = None,
    cell_params: object | None = None,
    blocks: object | None = None,
    periodic: object | None = None,
    init_mem: object = 1,
    queries: np.ndarray | None = None,
    ghost_radii: np.ndarray | None = None,
) -> list[Any]:
    dim = path.dim
    if points is None:
        points = np.array(
            [[0.25] * dim, [0.75] * dim],
            dtype=np.float64,
        )
    if ids is None:
        ids = np.arange(points.shape[0], dtype=np.int32)
    if radii is None:
        radii = np.linspace(0.0, 0.05, points.shape[0], dtype=np.float64)
    if queries is None:
        queries = np.full((1, dim), 0.5, dtype=np.float64)
    if ghost_radii is None:
        ghost_radii = np.full(queries.shape[0], 0.1, dtype=np.float64)
    if bounds is None:
        bounds = tuple((0.0, 1.0) for _ in range(dim))
    if cell_params is None:
        cell_params = (1.0, 0.0, 1.0, 0.0, 0.0, 1.0)
    if blocks is None:
        blocks = (1,) * dim
    if periodic is None:
        periodic = (False,) * dim

    args: list[Any] = [points, ids]
    if path.mode == 'power':
        args.append(radii)
    if path.geometry == 'periodic':
        args.extend((cell_params, blocks, init_mem))
    else:
        args.extend((bounds, blocks, periodic, init_mem))
    if path.operation in ('compute', 'ghost'):
        args.append((True, True, True))
    if path.operation in ('locate', 'ghost'):
        args.append(queries)
    if path.operation == 'ghost' and path.mode == 'power':
        args.append(ghost_radii)
    return args


def _empty_native_args(path: NativePath, **overrides: object) -> list[Any]:
    empty_points = np.empty((0, path.dim), dtype=np.float64)
    empty_queries = np.empty((0, path.dim), dtype=np.float64)
    values: dict[str, object] = {
        'points': empty_points,
        'ids': np.empty(0, dtype=np.int32),
        'radii': np.empty(0, dtype=np.float64),
        'queries': empty_queries,
        'ghost_radii': np.empty(0, dtype=np.float64),
    }
    values.update(overrides)
    return _native_args(path, **values)


@pytest.mark.parametrize('path', NATIVE_CONSTRUCTOR_PATHS, ids=lambda p: p.label)
def test_all_18_constructor_paths_reach_preflight(path: NativePath) -> None:
    with pytest.raises(ValueError, match='init_mem.*positive'):
        getattr(path.module, path.name)(*_native_args(path, init_mem=0))


@pytest.mark.parametrize('path', NATIVE_CONSTRUCTOR_PATHS, ids=lambda p: p.label)
def test_all_18_constructor_paths_accept_representative_valid_calls(
    path: NativePath,
) -> None:
    result = getattr(path.module, path.name)(*_native_args(path))
    if path.operation == 'locate':
        assert len(result) == 3
        assert result[0].shape == (1,)
        assert result[1].shape == (1,)
        assert result[2].shape == (1, path.dim)
    else:
        assert isinstance(result, list)
        assert len(result) == (1 if path.operation == 'ghost' else 2)


@pytest.mark.parametrize('path', NATIVE_CONSTRUCTOR_PATHS, ids=lambda p: p.label)
def test_all_18_constructor_paths_accept_smallest_empty_arrays(
    path: NativePath,
) -> None:
    points = np.empty((0, path.dim), dtype=np.float64)
    queries = np.empty((0, path.dim), dtype=np.float64)
    result = getattr(path.module, path.name)(
        *_native_args(
            path,
            points=points,
            ids=np.empty(0, dtype=np.int32),
            radii=np.empty(0, dtype=np.float64),
            queries=queries,
            ghost_radii=np.empty(0, dtype=np.float64),
        )
    )
    if path.operation == 'locate':
        assert all(value.shape[0] == 0 for value in result)
    else:
        assert result == []


@pytest.mark.parametrize(
    ('points', 'message'),
    [
        (np.zeros(3), 'points.*shape'),
        (np.array([[0.0, np.nan, 0.0]]), 'points.*finite'),
        (np.array([[0.0, np.inf, 0.0]]), 'points.*finite'),
    ],
)
def test_direct_core_rejects_point_shape_and_finiteness(
    points: np.ndarray,
    message: str,
) -> None:
    path = NATIVE_CONSTRUCTOR_PATHS[0]
    with pytest.raises(ValueError, match=message):
        getattr(path.module, path.name)(
            *_native_args(path, points=points, ids=np.array([0], dtype=np.int32))
        )


@pytest.mark.parametrize(
    ('points', 'message'),
    [
        (np.zeros(2), 'points.*shape'),
        (np.array([[0.0, np.nan]]), 'points.*finite'),
        (np.array([[0.0, -np.inf]]), 'points.*finite'),
    ],
)
def test_direct_planar_core_rejects_point_shape_and_finiteness(
    points: np.ndarray,
    message: str,
) -> None:
    path = NATIVE_CONSTRUCTOR_PATHS[12]
    with pytest.raises(ValueError, match=message):
        getattr(path.module, path.name)(
            *_native_args(path, points=points, ids=np.array([0], dtype=np.int32))
        )


@pytest.mark.parametrize(
    ('queries', 'message'),
    [
        (np.zeros(3), 'queries.*shape'),
        (np.array([[0.0, 0.0, np.nan]]), 'queries.*finite'),
        (np.array([[0.0, 0.0, -np.inf]]), 'queries.*finite'),
    ],
)
def test_direct_periodic_core_rejects_query_shape_and_finiteness(
    queries: np.ndarray,
    message: str,
) -> None:
    path = NATIVE_CONSTRUCTOR_PATHS[6]
    with pytest.raises(ValueError, match=message):
        getattr(path.module, path.name)(*_native_args(path, queries=queries))


@pytest.mark.parametrize(
    ('queries', 'message'),
    [
        (np.zeros(2), 'queries.*shape'),
        (np.array([[0.0, np.nan]]), 'queries.*finite'),
        (np.array([[0.0, np.inf]]), 'queries.*finite'),
    ],
)
def test_direct_planar_core_rejects_query_shape_and_finiteness(
    queries: np.ndarray,
    message: str,
) -> None:
    path = NATIVE_CONSTRUCTOR_PATHS[14]
    with pytest.raises(ValueError, match=message):
        getattr(path.module, path.name)(*_native_args(path, queries=queries))


@pytest.mark.parametrize(
    ('radii', 'message'),
    [
        (np.zeros((2, 1)), 'radii.*shape'),
        (np.array([0.0, np.nan]), 'radii.*finite'),
        (np.array([0.0, np.inf]), 'radii.*finite'),
        (np.array([0.0, -0.1]), 'radii.*non-negative'),
    ],
)
def test_direct_planar_core_rejects_radius_preconditions(
    radii: np.ndarray,
    message: str,
) -> None:
    path = NATIVE_CONSTRUCTOR_PATHS[13]
    with pytest.raises(ValueError, match=message):
        getattr(path.module, path.name)(*_native_args(path, radii=radii))


@pytest.mark.parametrize(
    ('radii', 'message'),
    [
        (np.zeros((2, 1)), 'radii.*shape'),
        (np.array([0.0, np.nan]), 'radii.*finite'),
        (np.array([0.0, -0.1]), 'radii.*non-negative'),
    ],
)
def test_direct_spatial_core_rejects_radius_preconditions(
    radii: np.ndarray,
    message: str,
) -> None:
    path = NATIVE_CONSTRUCTOR_PATHS[1]
    with pytest.raises(ValueError, match=message):
        getattr(path.module, path.name)(*_native_args(path, radii=radii))


@pytest.mark.parametrize(
    ('ghost_radii', 'message'),
    [
        (np.zeros((1, 1)), 'ghost_radii.*shape'),
        (np.array([np.nan]), 'ghost_radii.*finite'),
        (np.array([-0.1]), 'ghost_radii.*non-negative'),
    ],
)
def test_direct_ghost_core_rejects_ghost_radius_preconditions(
    ghost_radii: np.ndarray,
    message: str,
) -> None:
    path = NATIVE_CONSTRUCTOR_PATHS[9]
    with pytest.raises(ValueError, match=message):
        getattr(path.module, path.name)(
            *_native_args(path, ghost_radii=ghost_radii)
        )


@pytest.mark.parametrize(
    ('ghost_radii', 'message'),
    [
        (np.zeros((1, 1)), 'ghost_radii.*shape'),
        (np.array([np.inf]), 'ghost_radii.*finite'),
        (np.array([-0.1]), 'ghost_radii.*non-negative'),
    ],
)
def test_direct_planar_ghost_rejects_ghost_radius_preconditions(
    ghost_radii: np.ndarray,
    message: str,
) -> None:
    path = NATIVE_CONSTRUCTOR_PATHS[17]
    with pytest.raises(ValueError, match=message):
        getattr(path.module, path.name)(
            *_native_args(path, ghost_radii=ghost_radii)
        )


@pytest.mark.parametrize(
    'bounds',
    [
        ((0.0, 1.0), (0.0, np.nan), (0.0, 1.0)),
        ((0.0, 1.0), (1.0, 1.0), (0.0, 1.0)),
        ((0.0, 1.0), (2.0, 1.0), (0.0, 1.0)),
        ((-np.finfo(float).max, np.finfo(float).max),) * 3,
    ],
)
def test_direct_core_rejects_invalid_rectangular_bounds(bounds: object) -> None:
    path = NATIVE_CONSTRUCTOR_PATHS[0]
    with pytest.raises(ValueError, match='bounds'):
        getattr(path.module, path.name)(*_native_args(path, bounds=bounds))


@pytest.mark.parametrize(
    'bounds',
    [
        ((0.0, 1.0), (0.0, np.nan)),
        ((0.0, 1.0), (1.0, 1.0)),
        ((0.0, 1.0), (2.0, 1.0)),
        ((-np.finfo(float).max, np.finfo(float).max),) * 2,
    ],
)
def test_direct_planar_core_rejects_invalid_rectangular_bounds(
    bounds: object,
) -> None:
    path = NATIVE_CONSTRUCTOR_PATHS[12]
    with pytest.raises(ValueError, match='bounds'):
        getattr(path.module, path.name)(*_native_args(path, bounds=bounds))


@pytest.mark.parametrize('path', BOX_CONSTRUCTOR_PATHS, ids=lambda p: p.label)
def test_all_box_paths_reject_derived_binary64_overflow(
    path: NativePath,
) -> None:
    # container_base's periodic 0.25 max_len_sq weighting remains finite for
    # these inputs, but v_compute's unweighted bxsq and the voro_base worklist
    # arithmetic do not. Empty arrays isolate constructor preflight.
    bounds = ((0.0, 1e154),) * path.dim
    periodic = (True,) * path.dim
    with pytest.raises(ValueError, match=r'Voro\+\+ bxsq'):
        getattr(path.module, path.name)(
            *_empty_native_args(path, bounds=bounds, periodic=periodic)
        )


@pytest.mark.parametrize(
    'path', PLANAR_BOX_CONSTRUCTOR_PATHS, ids=lambda path: path.label
)
def test_all_planar_box_paths_construct_large_span_with_safe_block_widths(
    path: NativePath,
) -> None:
    # container_2d.cc:28-38 never squares full domain spans. Its constructor
    # arithmetic uses the 1e152 block widths in v_base_2d and v_compute_2d.
    bounds = ((0.0, 1e154),) * 2
    queries = np.empty((0, 2), dtype=np.float64)
    ghost_radii = np.empty(0, dtype=np.float64)
    if path.operation == 'ghost':
        # Ghost containers are loop-owned. An out-of-domain query proves that
        # construction succeeds without requesting downstream cell geometry
        # whose arithmetic is outside this constructor-preflight regression.
        queries = np.array([[-1.0, -1.0]], dtype=np.float64)
        ghost_radii = np.zeros(1, dtype=np.float64)

    result = getattr(path.module, path.name)(
        *_empty_native_args(
            path,
            bounds=bounds,
            blocks=(100, 100),
            periodic=(False, False),
            queries=queries,
            ghost_radii=ghost_radii,
        )
    )

    if path.operation == 'locate':
        assert isinstance(result, tuple)
        assert tuple(value.shape for value in result) == ((0,), (0,), (0, 2))
    else:
        assert isinstance(result, list)
        if path.operation == 'ghost':
            assert len(result) == 1
            record = result[0]
            assert isinstance(record, dict)
            assert record['empty'] is True
            assert record['area'] == 0.0
            assert np.isfinite(record['area'])
            assert record['site'] == [-1.0, -1.0]
            assert record['vertices'] == []
            assert record['adjacency'] == []
            assert record['edges'] == []
        else:
            assert result == []


def test_planar_compute_accepts_full_spans_whose_squares_overflow() -> None:
    # Each 2e154 full-span square would overflow, but 200 blocks retain the
    # same safe 1e152 block width used by the source's actual expressions.
    path = NATIVE_CONSTRUCTOR_PATHS[12]
    result = getattr(path.module, path.name)(
        *_empty_native_args(
            path,
            bounds=((0.0, 2e154),) * 2,
            blocks=(200, 200),
            periodic=(False, False),
        )
    )
    assert isinstance(result, list)
    assert result == []


def test_spatial_box_retains_full_domain_max_len_sq_guard() -> None:
    # 100 blocks make bxsq and the 65-block worklist bound finite. The actual
    # container.cc:34-35 non-periodic full-span accumulation still overflows.
    path = NATIVE_CONSTRUCTOR_PATHS[0]
    with pytest.raises(ValueError, match='maximum squared length'):
        getattr(path.module, path.name)(
            *_empty_native_args(
                path,
                bounds=((0.0, 1e154),) * 3,
                blocks=(100, 100, 100),
                periodic=(False, False, False),
            )
        )


@pytest.mark.parametrize(
    'path',
    [NATIVE_CONSTRUCTOR_PATHS[0], NATIVE_CONSTRUCTOR_PATHS[12]],
    ids=['spatial', 'planar'],
)
def test_box_paths_reject_worklist_distance_overflow(path: NativePath) -> None:
    # The unscaled bxsq is finite at 1e153, isolating the stronger source-based
    # 65*block_width worklist coordinate bound.
    bounds = ((0.0, 1e153),) * path.dim
    with pytest.raises(ValueError, match='worklist squared-distance'):
        getattr(path.module, path.name)(
            *_empty_native_args(path, bounds=bounds)
        )


@pytest.mark.parametrize(
    ('path', 'magnitude'),
    [
        (NATIVE_CONSTRUCTOR_PATHS[0], 1e150),
        (NATIVE_CONSTRUCTOR_PATHS[12], 1e150),
    ],
    ids=['spatial', 'planar'],
)
def test_large_safely_bounded_rectangular_constructor_is_accepted(
    path: NativePath,
    magnitude: float,
) -> None:
    bounds = ((0.0, magnitude),) * path.dim
    result = getattr(path.module, path.name)(
        *_empty_native_args(path, bounds=bounds)
    )
    assert result == []


@pytest.mark.parametrize(
    'cell_params',
    [
        (np.nan, 0.0, 1.0, 0.0, 0.0, 1.0),
        (1.0, np.inf, 1.0, 0.0, 0.0, 1.0),
        (0.0, 0.0, 1.0, 0.0, 0.0, 1.0),
        (1.0, 0.0, -1.0, 0.0, 0.0, 1.0),
        (1.0, 0.0, 1.0, 0.0, 0.0, 0.0),
    ],
)
def test_direct_core_rejects_invalid_periodic_parameters(
    cell_params: object,
) -> None:
    path = NATIVE_CONSTRUCTOR_PATHS[2]
    with pytest.raises(ValueError, match='cell_params'):
        getattr(path.module, path.name)(
            *_native_args(path, cell_params=cell_params)
        )


@pytest.mark.parametrize(
    'path', PERIODIC_CONSTRUCTOR_PATHS, ids=lambda path: path.label
)
def test_all_periodic_paths_reject_shell_vector_binary64_overflow(
    path: NativePath,
) -> None:
    params = (1e152, 1e154, 1e152, 0.0, 0.0, 1e152)
    with pytest.raises(ValueError, match='shell-vector squared norm'):
        getattr(path.module, path.name)(
            *_empty_native_args(path, cell_params=params)
        )


def test_periodic_constructor_rejects_cubic_tolerance_overflow() -> None:
    path = NATIVE_CONSTRUCTOR_PATHS[2]
    params = (1e110, 0.0, 1e110, 0.0, 0.0, 1e110)
    with pytest.raises(ValueError, match='unit-cell cubic tolerance'):
        getattr(path.module, path.name)(
            *_empty_native_args(path, cell_params=params)
        )


def test_large_safely_bounded_periodic_constructor_is_accepted() -> None:
    path = NATIVE_CONSTRUCTOR_PATHS[2]
    params = (1e100, 2e99, 1e100, -1e99, 1e99, 1e100)
    result = getattr(path.module, path.name)(
        *_empty_native_args(path, cell_params=params)
    )
    assert result == []


@pytest.mark.parametrize(
    ('ids', 'message'),
    [
        (np.array([0, 0], dtype=np.int32), 'duplicate'),
        (np.array([-1, 1], dtype=np.int32), 'non-negative'),
        (np.array([0, 2], dtype=np.int32), r'range \[0, n\)'),
    ],
)
def test_direct_core_rejects_invalid_internal_ids(
    ids: np.ndarray,
    message: str,
) -> None:
    path = NATIVE_CONSTRUCTOR_PATHS[0]
    with pytest.raises(ValueError, match=message):
        getattr(path.module, path.name)(*_native_args(path, ids=ids))


def test_direct_planar_ghost_rejects_reserved_ghost_id_conflict() -> None:
    path = NATIVE_CONSTRUCTOR_PATHS[16]
    ids = np.array([0, CPP_INT_MAX], dtype=np.int32)
    with pytest.raises(ValueError, match='reserved ghost ID'):
        getattr(path.module, path.name)(*_native_args(path, ids=ids))


@pytest.mark.parametrize(
    ('path', 'blocks'),
    [
        (NATIVE_CONSTRUCTOR_PATHS[0], (0, 1, 1)),
        (NATIVE_CONSTRUCTOR_PATHS[0], (-1, 1, 1)),
        (NATIVE_CONSTRUCTOR_PATHS[12], (0, 1)),
        (NATIVE_CONSTRUCTOR_PATHS[12], (-1, 1)),
    ],
    ids=['spatial-zero', 'spatial-negative', 'planar-zero', 'planar-negative'],
)
def test_direct_cores_reject_nonpositive_block_counts(
    path: NativePath,
    blocks: object,
) -> None:
    with pytest.raises(ValueError, match=r'blocks\[0\].*positive'):
        getattr(path.module, path.name)(*_native_args(path, blocks=blocks))


@pytest.mark.parametrize(
    'path',
    [NATIVE_CONSTRUCTOR_PATHS[0], NATIVE_CONSTRUCTOR_PATHS[12]],
    ids=['spatial', 'planar'],
)
def test_direct_cores_reject_negative_init_mem(path: NativePath) -> None:
    with pytest.raises(ValueError, match='init_mem.*positive'):
        getattr(path.module, path.name)(*_native_args(path, init_mem=-1))


def test_checked_count_add_multiply_and_byte_boundaries() -> None:
    cap = 1 << 30
    assert _core._test_checked_count(CPP_INT_MAX) == CPP_INT_MAX
    assert _core._test_checked_int_add(CPP_INT_MAX, 0) == CPP_INT_MAX
    assert _core._test_checked_int_multiply(CPP_INT_MAX, 1) == CPP_INT_MAX
    assert _core._test_allocation_estimate(cap) == cap
    assert _core._EAGER_ALLOCATION_LIMIT_BYTES == cap

    with pytest.raises(ValueError, match='destination range'):
        _core._test_checked_count(CPP_INT_MAX + 1)
    with pytest.raises(ValueError, match='destination range'):
        _core._test_checked_int_add(CPP_INT_MAX, 1)
    with pytest.raises(ValueError, match='destination range'):
        _core._test_checked_int_multiply(CPP_INT_MAX, 2)
    with pytest.raises(ValueError, match='1073741824-byte safety limit'):
        _core._test_allocation_estimate(cap + 1)


@pytest.mark.parametrize(
    ('blocks', 'periodic', 'init_mem', 'message'),
    [
        ((CPP_INT_MAX, 2, 1), (False, False, False), 1, 'block product'),
        ((CPP_INT_MAX, 1, 1), (True, False, False), 1, 'mask dimension'),
        ((1, 1, 1), (False, False, False), CPP_INT_MAX, r'ps \* init_mem'),
    ],
)
def test_direct_core_checks_vendor_int_expressions_before_construction(
    blocks: object,
    periodic: object,
    init_mem: int,
    message: str,
) -> None:
    path = NATIVE_CONSTRUCTOR_PATHS[0]
    with pytest.raises(ValueError, match=message):
        getattr(path.module, path.name)(
            *_native_args(
                path,
                blocks=blocks,
                periodic=periodic,
                init_mem=init_mem,
            )
        )


@pytest.mark.parametrize(
    ('path', 'blocks'),
    [
        (NATIVE_CONSTRUCTOR_PATHS[0], (300, 300, 300)),
        (NATIVE_CONSTRUCTOR_PATHS[12], (5000, 5000)),
    ],
    ids=['spatial', 'planar'],
)
def test_fixed_rectangular_configuration_exceeding_one_gib_is_rejected(
    path: NativePath,
    blocks: object,
) -> None:
    with pytest.raises(ValueError, match='1073741824-byte safety limit'):
        getattr(path.module, path.name)(*_native_args(path, blocks=blocks))


def test_fixed_periodic_configuration_uses_conservative_extension_bound() -> None:
    path = NATIVE_CONSTRUCTOR_PATHS[2]
    params = (1.0, 10000.0, 1.0, 0.0, 0.0, 1.0)
    with pytest.raises(ValueError, match='1073741824-byte safety limit'):
        getattr(path.module, path.name)(
            *_native_args(path, cell_params=params, blocks=(1, 1, 1))
        )


@pytest.mark.parametrize(
    'script',
    [
        '''
        import numpy as np
        from pyvoro2 import _core
        try:
            _core.compute_box_standard(
                np.array([[0.25, 0.25, 0.25], [0.75, 0.75, 0.75]]),
                np.array([0, 1], dtype=np.int32),
                ((0.0, 1.0),) * 3,
                (1, 1, 1),
                (False, False, False),
                0,
                (False, False, False),
            )
        except ValueError as exc:
            print(type(exc).__name__, exc)
        else:
            raise SystemExit('missing Python exception')
        ''',
        '''
        import numpy as np
        from pyvoro2 import _core2d
        try:
            _core2d.ghost_box_power(
                np.array([[0.25, 0.25], [0.75, 0.75]]),
                np.array([0, 1], dtype=np.int32),
                np.array([0.0, 0.1]),
                ((0.0, 1.0),) * 2,
                (1, 1),
                (False, False),
                0,
                (False, False, False),
                np.array([[0.5, 0.5]]),
                np.array([0.0]),
            )
        except ValueError as exc:
            print(type(exc).__name__, exc)
        else:
            raise SystemExit('missing Python exception')
        ''',
    ],
    ids=['spatial-init-mem-zero', 'planar-ghost-init-mem-zero'],
)
def test_historically_unsafe_fixed_inputs_raise_in_subprocess(script: str) -> None:
    completed = subprocess.run(
        [sys.executable, '-c', textwrap.dedent(script)],
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.startswith('ValueError init_mem')


@pytest.mark.parametrize(
    'script',
    [
        '''
        import numpy as np
        from pyvoro2 import _core2d
        try:
            _core2d.compute_box_standard(
                np.empty((0, 2)),
                np.empty(0, dtype=np.int32),
                ((0.0, 1e154),) * 2,
                (1, 1),
                (True, True),
                1,
                (False, False, False),
            )
        except ValueError as exc:
            print(type(exc).__name__, exc)
        else:
            raise SystemExit('missing Python exception')
        ''',
        '''
        import numpy as np
        from pyvoro2 import _core
        try:
            _core.compute_box_standard(
                np.empty((0, 3)),
                np.empty(0, dtype=np.int32),
                ((0.0, 1e154),) * 3,
                (1, 1, 1),
                (True, True, True),
                1,
                (False, False, False),
            )
        except ValueError as exc:
            print(type(exc).__name__, exc)
        else:
            raise SystemExit('missing Python exception')
        ''',
        '''
        import numpy as np
        from pyvoro2 import _core
        try:
            _core.compute_periodic_standard(
                np.empty((0, 3)),
                np.empty(0, dtype=np.int32),
                (1e152, 1e154, 1e152, 0.0, 0.0, 1e152),
                (1, 1, 1),
                1,
                (False, False, False),
            )
        except ValueError as exc:
            print(type(exc).__name__, exc)
        else:
            raise SystemExit('missing Python exception')
        ''',
    ],
    ids=['planar-box-bxsq', 'spatial-box-bxsq', 'periodic-shell-vector'],
)
def test_floating_constructor_overflow_raises_in_subprocess(script: str) -> None:
    completed = subprocess.run(
        [sys.executable, '-c', textwrap.dedent(script)],
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.startswith('ValueError')


@pytest.mark.parametrize(
    ('label', 'script'),
    [
        (
            'invalid-block-controls',
            '''
            import numpy as np
            from pyvoro2 import _core, _core2d

            cases = (
                lambda: _core.compute_box_standard(
                    np.empty((0, 3)), np.empty(0, dtype=np.int32),
                    ((0.0, 1.0),) * 3, (0, 1, 1),
                    (False, False, False), 1, (False, False, False),
                ),
                lambda: _core2d.compute_box_standard(
                    np.empty((0, 2)), np.empty(0, dtype=np.int32),
                    ((0.0, 1.0),) * 2, (-1, 1),
                    (False, False), 1, (False, False, False),
                ),
            )
            for case in cases:
                try:
                    case()
                except ValueError:
                    pass
                else:
                    raise SystemExit('missing Python exception')
            print('OK invalid-block-controls')
            ''',
        ),
        (
            'resource-cap-rejections',
            '''
            import numpy as np
            from pyvoro2 import _core, _core2d

            cases = (
                lambda: _core.compute_box_standard(
                    np.empty((0, 3)), np.empty(0, dtype=np.int32),
                    ((0.0, 1.0),) * 3, (300, 300, 300),
                    (False, False, False), 1, (False, False, False),
                ),
                lambda: _core2d.compute_box_standard(
                    np.empty((0, 2)), np.empty(0, dtype=np.int32),
                    ((0.0, 1.0),) * 2, (5000, 5000),
                    (False, False), 1, (False, False, False),
                ),
                lambda: _core.compute_periodic_standard(
                    np.empty((0, 3)), np.empty(0, dtype=np.int32),
                    (1.0, 10000.0, 1.0, 0.0, 0.0, 1.0),
                    (1, 1, 1), 1, (False, False, False),
                ),
            )
            for case in cases:
                try:
                    case()
                except ValueError as exc:
                    if '1073741824-byte safety limit' not in str(exc):
                        raise
                else:
                    raise SystemExit('missing Python exception')
            print('OK resource-cap-rejections')
            ''',
        ),
        (
            'malformed-spatial-native-inputs',
            '''
            import numpy as np
            from pyvoro2 import _core

            points = np.array([[0.25, 0.25, 0.25], [0.75, 0.75, 0.75]])
            ids = np.array([0, 1], dtype=np.int32)
            radii = np.array([0.0, 0.1])
            bounds = ((0.0, 1.0),) * 3
            params = (1.0, 0.0, 1.0, 0.0, 0.0, 1.0)
            blocks = (1, 1, 1)
            periodic = (False, False, False)
            opts = (False, False, False)
            queries = np.array([[0.5, 0.5, 0.5]])
            cases = (
                lambda: _core.compute_box_standard(
                    np.array([[0.0, np.nan, 0.0]]),
                    np.array([0], dtype=np.int32), bounds, blocks,
                    periodic, 1, opts,
                ),
                lambda: _core.locate_periodic_standard(
                    points, ids, params, blocks, 1,
                    np.array([[0.0, 0.0, np.inf]]),
                ),
                lambda: _core.compute_box_power(
                    points, ids, np.array([0.0, -0.1]), bounds,
                    blocks, periodic, 1, opts,
                ),
                lambda: _core.compute_periodic_standard(
                    points, ids, (np.nan, *params[1:]), blocks, 1, opts,
                ),
            )
            for case in cases:
                try:
                    case()
                except ValueError:
                    pass
                else:
                    raise SystemExit('missing Python exception')
            print('OK malformed-spatial-native-inputs')
            ''',
        ),
        (
            'malformed-planar-native-inputs',
            '''
            import numpy as np
            from pyvoro2 import _core2d

            points = np.array([[0.25, 0.25], [0.75, 0.75]])
            ids = np.array([0, 1], dtype=np.int32)
            bounds = ((0.0, 1.0),) * 2
            blocks = (1, 1)
            periodic = (False, False)
            opts = (False, False, False)
            cases = (
                lambda: _core2d.compute_box_standard(
                    np.array([[0.0, -np.inf]]),
                    np.array([0], dtype=np.int32), bounds, blocks,
                    periodic, 1, opts,
                ),
                lambda: _core2d.locate_box_standard(
                    points, ids, bounds, blocks, periodic, 1,
                    np.array([[0.0, np.nan]]),
                ),
                lambda: _core2d.ghost_box_power(
                    points, ids, np.array([0.0, 0.1]), bounds,
                    blocks, periodic, 1, opts, np.array([[0.5, 0.5]]),
                    np.array([-0.1]),
                ),
            )
            for case in cases:
                try:
                    case()
                except ValueError:
                    pass
                else:
                    raise SystemExit('missing Python exception')
            print('OK malformed-planar-native-inputs')
            ''',
        ),
        (
            'valid-spatial-families',
            '''
            import numpy as np
            from pyvoro2 import _core

            points = np.array([[0.25, 0.25, 0.25], [0.75, 0.75, 0.75]])
            ids = np.array([0, 1], dtype=np.int32)
            radii = np.array([0.0, 0.1])
            bounds = ((0.0, 1.0),) * 3
            params = (1.0, 0.0, 1.0, 0.0, 0.0, 1.0)
            blocks = (1, 1, 1)
            periodic = (False, False, False)
            opts = (False, False, False)
            queries = np.array([[0.5, 0.5, 0.5]])
            ghost_radii = np.array([0.05])
            _core.compute_box_standard(
                points, ids, bounds, blocks, periodic, 1, opts,
            )
            _core.compute_periodic_power(
                points, ids, radii, params, blocks, 1, opts,
            )
            _core.locate_periodic_standard(
                points, ids, params, blocks, 1, queries,
            )
            _core.locate_box_power(
                points, ids, radii, bounds, blocks, periodic, 1, queries,
            )
            _core.ghost_box_standard(
                points, ids, bounds, blocks, periodic, 1, opts, queries,
            )
            _core.ghost_periodic_power(
                points, ids, radii, params, blocks, 1, opts,
                queries, ghost_radii,
            )
            print('OK valid-spatial-families')
            ''',
        ),
        (
            'valid-planar-families',
            '''
            import numpy as np
            from pyvoro2 import _core2d

            points = np.array([[0.25, 0.25], [0.75, 0.75]])
            ids = np.array([0, 1], dtype=np.int32)
            radii = np.array([0.0, 0.1])
            bounds = ((0.0, 1.0),) * 2
            blocks = (1, 1)
            periodic = (False, False)
            opts = (False, False, False)
            queries = np.array([[0.5, 0.5]])
            ghost_radii = np.array([0.05])
            _core2d.compute_box_standard(
                points, ids, bounds, blocks, periodic, 1, opts,
            )
            _core2d.compute_box_power(
                points, ids, radii, bounds, blocks, periodic, 1, opts,
            )
            _core2d.locate_box_standard(
                points, ids, bounds, blocks, periodic, 1, queries,
            )
            _core2d.locate_box_power(
                points, ids, radii, bounds, blocks, periodic, 1, queries,
            )
            _core2d.ghost_box_standard(
                points, ids, bounds, blocks, periodic, 1, opts, queries,
            )
            _core2d.ghost_box_power(
                points, ids, radii, bounds, blocks, periodic, 1, opts,
                queries, ghost_radii,
            )
            print('OK valid-planar-families')
            ''',
        ),
    ],
    ids=[
        'invalid-block-controls',
        'resource-cap-rejections',
        'malformed-spatial-native-inputs',
        'malformed-planar-native-inputs',
        'valid-spatial-families',
        'valid-planar-families',
    ],
)
def test_fixed_issue_matrix_returns_normally_in_subprocess(
    label: str,
    script: str,
) -> None:
    completed = subprocess.run(
        [sys.executable, '-c', textwrap.dedent(script)],
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.strip() == f'OK {label}'
