from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from itertools import product
import math
import subprocess
import sys
import textwrap
from types import ModuleType
from typing import Any

import numpy as np
import pytest

from pyvoro2 import _core, _core2d


CPP_INT_MAX = int(np.iinfo(np.intc).max)
EAGER_ALLOCATION_LIMIT_BYTES = 1 << 30
NATIVE_SAFETY_DISTANCE = Fraction.from_float(1e-5)
NATIVE_SAFETY_DISTANCE_SQUARED = Fraction.from_float(1e-10)
NATIVE_BINARY64_WITNESS_PARAMS = (
    0.00017856460945140374,
    -5.725280006612657,
    1.65273947780994,
    -391.7349918299756,
    113.08370683864604,
    0.10045884903938922,
)
NATIVE_LARGE_SHIFT_WITNESS_LEFT = (
    2.7785562527544788e-05,
    0.6974223475701162,
    0.1004588490393892,
)


def _as_fraction(value: float) -> Fraction:
    """Interpret a binary64 test input exactly, without production helpers."""

    return Fraction.from_float(float(value))


def _fraction_periodic_lattice(
    params: tuple[float, ...],
) -> tuple[tuple[Fraction, ...], ...]:
    bx, bxy, by, bxz, byz, bz = map(_as_fraction, params)
    zero = Fraction()
    return (
        (bx, zero, zero),
        (bxy, by, zero),
        (bxz, byz, bz),
    )


def _fraction_periodic_inverse(
    params: tuple[float, ...],
) -> tuple[tuple[Fraction, ...], ...]:
    """Closed-form inverse of the exact source-binary64 triangular basis."""

    bx, bxy, by, bxz, byz, bz = map(_as_fraction, params)
    zero = Fraction()
    return (
        (1 / bx, zero, zero),
        (-bxy / (bx * by), 1 / by, zero),
        (
            (bxy * byz - bxz * by) / (bx * by * bz),
            -byz / (by * bz),
            1 / bz,
        ),
    )


def _fraction_coefficient_bounds(
    params: tuple[float, ...],
) -> tuple[Fraction, ...]:
    inverse = _fraction_periodic_inverse(params)
    return tuple(
        NATIVE_SAFETY_DISTANCE
        * sum((abs(inverse[row][column]) for row in range(3)), Fraction())
        for column in range(3)
    )


def _fraction_floor(value: Fraction) -> int:
    return value.numerator // value.denominator


def _fraction_ceil(value: Fraction) -> int:
    return -((-value.numerator) // value.denominator)


def _fraction_periodic_pair_is_unsafe(
    left: tuple[float, float, float] | np.ndarray,
    right: tuple[float, float, float] | np.ndarray,
    params: tuple[float, ...],
) -> bool:
    """Independent exact all-shift oracle for bounded-size test cells."""

    lattice = _fraction_periodic_lattice(params)
    inverse = _fraction_periodic_inverse(params)
    bounds = _fraction_coefficient_bounds(params)
    delta = tuple(
        _as_fraction(float(right[axis])) - _as_fraction(float(left[axis]))
        for axis in range(3)
    )
    coefficient = tuple(
        sum(
            (delta[row] * inverse[row][column] for row in range(3)),
            Fraction(),
        )
        for column in range(3)
    )
    ranges = tuple(
        range(
            _fraction_ceil(-coefficient[axis] - bounds[axis]),
            _fraction_floor(-coefficient[axis] + bounds[axis]) + 1,
        )
        for axis in range(3)
    )
    for shift in product(*ranges):
        displacement = tuple(
            delta[coordinate]
            + sum(
                (
                    shift[row] * lattice[row][coordinate]
                    for row in range(3)
                ),
                Fraction(),
            )
            for coordinate in range(3)
        )
        distance_squared = sum(
            (value * value for value in displacement), Fraction()
        )
        if distance_squared <= NATIVE_SAFETY_DISTANCE_SQUARED:
            return True
    return False


def _native_scanner_reports_unsafe(
    points: np.ndarray,
    inserted_queries: np.ndarray,
    params: tuple[float, ...],
    *,
    expected_message: str,
) -> bool:
    """Classify only the expected native duplicate rejection as unsafe."""

    try:
        _core._test_periodic_safety_candidate_count(
            points,
            params,
            inserted_queries,
        )
    except ValueError as exc:
        if expected_message not in str(exc):
            raise
        return True
    return False


def _fraction_periodic_key(
    point: tuple[float, float, float] | np.ndarray,
    params: tuple[float, ...],
    bins: tuple[int, int, int],
) -> tuple[int, int, int]:
    inverse = _fraction_periodic_inverse(params)
    exact_point = tuple(_as_fraction(float(value)) for value in point)
    coefficient = tuple(
        sum(
            (
                exact_point[row] * inverse[row][column]
                for row in range(3)
            ),
            Fraction(),
        )
        for column in range(3)
    )
    return tuple(
        _fraction_floor(coefficient[axis] * bins[axis]) % bins[axis]
        for axis in range(3)
    )


def _periodic_known_eager_oracle(
    ey: int,
    ez: int,
    *,
    blocks: tuple[int, int, int],
    init_mem: int,
    particle_stride: int,
) -> tuple[dict[str, int], int]:
    """Reproduce the vendored eager allocations without production helpers."""
    pointer_bytes = 8
    int_bytes = 4
    double_bytes = 8
    unsigned_int_bytes = 4
    char_bytes = 1
    assert np.dtype(np.uintp).itemsize == pointer_bytes
    assert np.dtype(np.intc).itemsize == int_bytes
    assert np.dtype(np.uintc).itemsize == unsigned_int_bytes
    assert np.dtype(np.float64).itemsize == double_bytes

    nx, ny, nz = blocks
    primary_blocks = nx * ny * nz
    oy = ny + 2 * ey
    oz = nz + 2 * ez
    extended_blocks = nx * oy * oz
    hx = 2 * nx + 1
    hy = 2 * ey + 1
    hz = 2 * ez + 1
    hxy = hx * hy
    mask_size = hxy * hz
    queue_size = 3 * (3 + hxy + hz * (hx + hy))

    unit_cell_bytes = (
        256 * pointer_bytes
        + 256 * int_bytes
        + 256 * unsigned_int_bytes
        + 4 * 256 * double_bytes
        + 2 * 64 * int_bytes
        + 64 * pointer_bytes
        + (2 * 256 + 32) * int_bytes
    )
    for order in range(64):
        count = 256 * 7 if order == 3 else 8 * (2 * order + 1)
        unit_cell_bytes += count * int_bytes

    breakdown = {
        'extended_id_pointers': extended_blocks * pointer_bytes,
        'extended_particle_pointers': extended_blocks * pointer_bytes,
        'extended_counters': 2 * extended_blocks * int_bytes,
        'image_flags': extended_blocks * char_bytes,
        'primary_particle_storage': (
            primary_blocks
            * init_mem
            * (int_bytes + particle_stride * double_bytes)
        ),
        'compute_mask': mask_size * unsigned_int_bytes,
        'compute_queue': queue_size * int_bytes,
        'worklist': 64 * 64 * double_bytes,
        'unit_cell': unit_cell_bytes,
    }
    return breakdown, sum(breakdown.values())


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
@pytest.mark.parametrize('value', [-np.nextafter(0.0, 1.0), 1.0])
def test_direct_inserted_points_require_primary_half_open_coordinates(
    path: NativePath,
    value: float,
) -> None:
    points = np.full((1, path.dim), 0.25, dtype=np.float64)
    points[0, 0] = value
    with pytest.raises(ValueError, match='primary half-open interval'):
        getattr(path.module, path.name)(
            *_native_args(
                path,
                points=points,
                ids=np.array([0], dtype=np.int32),
                radii=np.array([0.0]),
            )
        )


@pytest.mark.parametrize('path', NATIVE_CONSTRUCTOR_PATHS, ids=lambda p: p.label)
def test_direct_persistent_duplicates_rejected_before_insertion(
    path: NativePath,
) -> None:
    points = np.full((2, path.dim), 0.25, dtype=np.float64)
    with pytest.raises(ValueError, match='backend-unsafe'):
        getattr(path.module, path.name)(*_native_args(path, points=points))


@pytest.mark.parametrize(
    'path',
    PERIODIC_CONSTRUCTOR_PATHS + BOX_CONSTRUCTOR_PATHS,
    ids=lambda p: p.label,
)
def test_direct_periodic_seam_pair_rejected(path: NativePath) -> None:
    points = np.full((2, path.dim), 0.25, dtype=np.float64)
    points[0, 0] = 0.0
    points[1, 0] = np.nextafter(1.0, 0.0)
    overrides: dict[str, object] = {'points': points}
    if path.geometry == 'box':
        overrides['periodic'] = (True,) + (False,) * (path.dim - 1)
    with pytest.raises(ValueError, match='backend-unsafe'):
        getattr(path.module, path.name)(*_native_args(path, **overrides))


@pytest.mark.parametrize(
    'path',
    tuple(
        value for value in NATIVE_CONSTRUCTOR_PATHS
        if value.operation == 'ghost'
    ),
    ids=lambda p: p.label,
)
def test_direct_ghost_duplicate_and_outside_query_rejected(
    path: NativePath,
) -> None:
    points = np.full((1, path.dim), 0.25, dtype=np.float64)
    duplicate_query = points.copy()
    with pytest.raises(ValueError, match='backend-unsafe'):
        getattr(path.module, path.name)(
            *_native_args(
                path,
                points=points,
                ids=np.array([0], dtype=np.int32),
                radii=np.array([0.0]),
                queries=duplicate_query,
                ghost_radii=np.array([0.0]),
            )
        )

    outside_query = np.full((1, path.dim), 0.5, dtype=np.float64)
    outside_query[0, 0] = 1.0
    with pytest.raises(ValueError, match='primary half-open interval'):
        getattr(path.module, path.name)(
            *_native_args(
                path,
                points=points,
                ids=np.array([0], dtype=np.int32),
                radii=np.array([0.0]),
                queries=outside_query,
                ghost_radii=np.array([0.0]),
            )
        )


@pytest.mark.parametrize(
    'path',
    tuple(
        value for value in NATIVE_CONSTRUCTOR_PATHS
        if value.operation == 'locate'
    ),
    ids=lambda p: p.label,
)
def test_direct_locate_queries_are_not_subject_to_insertion_containment(
    path: NativePath,
) -> None:
    queries = np.full((1, path.dim), 2.0, dtype=np.float64)
    result = getattr(path.module, path.name)(
        *_native_args(path, queries=queries)
    )
    assert len(result) == 3


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
        # Ghost containers are loop-owned. Keep the temporary generator in the
        # primary half-open box while avoiding downstream extreme geometry.
        queries = np.array([[0.5, 0.5]], dtype=np.float64)
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
            assert record['empty'] is False
            assert record['area'] == np.inf
            assert record['site'] == [0.5, 0.5]
            assert len(record['vertices']) == 4
            assert len(record['adjacency']) == 4
            assert len(record['edges']) == 4
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


def test_spatial_ghost_does_not_claim_planar_synthetic_id_reservation() -> None:
    path = NATIVE_CONSTRUCTOR_PATHS[8]
    ids = np.array([0, CPP_INT_MAX], dtype=np.int32)
    with pytest.raises(ValueError, match=r'range \[0, n\)') as caught:
        getattr(path.module, path.name)(*_native_args(path, ids=ids))
    assert 'reserved ghost ID' not in str(caught.value)


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
    cap = EAGER_ALLOCATION_LIMIT_BYTES
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


def test_periodic_extent_bound_covers_orthogonal_unitcell_source_extent() -> None:
    a, b, c = 1.0, 10.0, 1.0
    radius = 0.5 * math.sqrt(a * a + b * b + c * c)
    max_uv_y = b / 2.0 + radius
    max_uv_z = c / 2.0 + radius
    source_ey = math.floor(max_uv_y / b * 100) + 1
    source_ez = math.floor(max_uv_z / c) + 1

    old_radius_bound = 0.5 * (a + b + c)
    old_ey = math.floor(old_radius_bound / b * 100) + 1

    assert max_uv_y == 10.049752469181039
    assert old_ey == 61
    assert source_ey == 101

    estimate = _core._test_periodic_resource_estimate(
        (a, 0.0, b, 0.0, 0.0, c),
        (1, 100, 1),
        1,
        3,
    )
    assert estimate['ey_bound'] >= source_ey
    assert estimate['ez_bound'] >= source_ez


def test_fixed_periodic_cap_false_negative_oracles_and_estimator() -> None:
    a, b, c = 0.01, 10.0, 10.0
    blocks = (1, 1529, 1529)
    radius = 0.5 * math.sqrt(a * a + b * b + c * c)
    source_ey = math.floor((b / 2.0 + radius) / b * blocks[1]) + 1
    source_ez = math.floor((c / 2.0 + radius) / c * blocks[2]) + 1
    assert source_ey == source_ez == 1846

    old_radius_bound = 0.5 * (a + b + c)
    old_ey = math.floor(old_radius_bound / b * blocks[1]) + 1
    old_ez = math.floor(old_radius_bound / c * blocks[2]) + 1
    assert old_ey == old_ez == 1530

    old_breakdown, old_total = _periodic_known_eager_oracle(
        old_ey,
        old_ez,
        blocks=blocks,
        init_mem=1,
        particle_stride=3,
    )
    source_breakdown, source_total = _periodic_known_eager_oracle(
        source_ey,
        source_ez,
        blocks=blocks,
        init_mem=1,
        particle_stride=3,
    )
    assert sum(old_breakdown.values()) == old_total == 817_212_577
    assert (
        sum(source_breakdown.values())
        == source_total
        == 1_074_700_753
    )
    assert old_total < EAGER_ALLOCATION_LIMIT_BYTES
    assert source_total > EAGER_ALLOCATION_LIMIT_BYTES

    estimate = _core._test_periodic_resource_estimate(
        (a, 0.0, b, 0.0, 0.0, c),
        blocks,
        1,
        3,
    )
    conservative_extent = sum(abs(value) for value in (a, 0.0, b, 0.0, 0.0, c))
    conservative_ey = math.floor(
        conservative_extent / b * blocks[1]
    ) + 1
    conservative_ez = math.floor(
        conservative_extent / c * blocks[2]
    ) + 1
    conservative_breakdown, conservative_total = _periodic_known_eager_oracle(
        conservative_ey,
        conservative_ez,
        blocks=blocks,
        init_mem=1,
        particle_stride=3,
    )
    assert conservative_ey == conservative_ez == 3060
    assert (
        sum(conservative_breakdown.values())
        == conservative_total
        == 2_427_965_977
    )
    oy = blocks[1] + 2 * conservative_ey
    oz = blocks[2] + 2 * conservative_ez
    hx = 2 * blocks[0] + 1
    hy = 2 * conservative_ey + 1
    hz = 2 * conservative_ez + 1
    assert estimate == {
        'primary_blocks': math.prod(blocks),
        'ey_bound': conservative_ey,
        'ez_bound': conservative_ez,
        'oy': oy,
        'oz': oz,
        'extended_blocks': blocks[0] * oy * oz,
        'hx': hx,
        'hy': hy,
        'hz': hz,
        'mask_size': hx * hy * hz,
        'queue_size': 3 * (3 + hx * hy + hz * (hx + hy)),
        'known_eager_bytes': conservative_total,
    }
    assert estimate['ey_bound'] >= source_ey
    assert estimate['ez_bound'] >= source_ez
    assert estimate['known_eager_bytes'] > EAGER_ALLOCATION_LIMIT_BYTES


@pytest.mark.parametrize(
    'path', PERIODIC_CONSTRUCTOR_PATHS, ids=lambda path: path.label
)
def test_all_six_periodic_paths_accept_sheared_in_cap_calls(
    path: NativePath,
) -> None:
    result = getattr(path.module, path.name)(
        *_native_args(
            path,
            cell_params=(2.0, 0.25, 2.0, 0.1, -0.2, 2.0),
        )
    )
    if path.operation == 'locate':
        assert len(result) == 3
        assert result[0].shape == (1,)
    else:
        assert isinstance(result, list)
        assert len(result) == (1 if path.operation == 'ghost' else 2)


def test_representative_anisotropic_periodic_case_remains_in_cap() -> None:
    params = (0.1, 0.0, 10.0, 0.0, 0.0, 10.0)
    blocks = (1, 100, 100)
    estimate = _core._test_periodic_resource_estimate(params, blocks, 1, 3)
    assert estimate['known_eager_bytes'] < EAGER_ALLOCATION_LIMIT_BYTES

    path = NATIVE_CONSTRUCTOR_PATHS[2]
    result = getattr(path.module, path.name)(
        *_empty_native_args(path, cell_params=params, blocks=blocks)
    )
    assert result == []


@pytest.mark.skipif(
    not sys.platform.startswith('linux'),
    reason='the dangerous-constructor regression requires Linux RLIMIT_AS',
)
def test_all_six_periodic_paths_reject_corrected_cap_case_bounded() -> None:
    script = '''
    import os
    import resource
    import numpy as np
    from pyvoro2 import _core

    points = np.empty((0, 3))
    ids = np.empty(0, dtype=np.int32)
    radii = np.empty(0)
    queries = np.empty((0, 3))
    ghost_radii = np.empty(0)
    params = (0.01, 0.0, 10.0, 0.0, 0.0, 10.0)
    blocks = (1, 1529, 1529)
    opts = (False, False, False)
    cases = (
        ('compute-standard', lambda: _core.compute_periodic_standard(
            points, ids, params, blocks, 1, opts,
        )),
        ('compute-power', lambda: _core.compute_periodic_power(
            points, ids, radii, params, blocks, 1, opts,
        )),
        ('locate-standard', lambda: _core.locate_periodic_standard(
            points, ids, params, blocks, 1, queries,
        )),
        ('locate-power', lambda: _core.locate_periodic_power(
            points, ids, radii, params, blocks, 1, queries,
        )),
        ('ghost-standard', lambda: _core.ghost_periodic_standard(
            points, ids, params, blocks, 1, opts, queries,
        )),
        ('ghost-power', lambda: _core.ghost_periodic_power(
            points, ids, radii, params, blocks, 1, opts, queries,
            ghost_radii,
        )),
    )

    page_size = os.sysconf('SC_PAGE_SIZE')
    with open('/proc/self/statm', encoding='ascii') as statm:
        current_address_space = int(statm.read().split()[0]) * page_size
    limit = current_address_space + 256 * 1024 * 1024
    resource.setrlimit(resource.RLIMIT_AS, (limit, limit))

    for label, case in cases:
        try:
            case()
        except ValueError as exc:
            if '1073741824-byte safety limit' not in str(exc):
                raise
        else:
            raise SystemExit(f'{label}: missing 1-GiB ValueError')
        print(f'OK {label}')
    '''
    completed = subprocess.run(
        [sys.executable, '-c', textwrap.dedent(script)],
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.splitlines() == [
        'OK compute-standard',
        'OK compute-power',
        'OK locate-standard',
        'OK locate-power',
        'OK ghost-standard',
        'OK ghost-power',
    ]


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


def test_r5_direct_native_safety_matrix_returns_exceptions_in_subprocess() -> None:
    script = r'''
        import numpy as np
        from pyvoro2 import _core, _core2d

        opts = (False, False, False)
        bounds3 = ((0.0, 1.0),) * 3
        bounds2 = ((0.0, 1.0),) * 2
        blocks3 = (1, 1, 1)
        blocks2 = (1, 1)
        periodic3 = (True, False, False)
        periodic2 = (True, False)
        params = (1.0, 0.2, 1.0, 0.1, -0.1, 1.0)

        cases = (
            lambda: _core.compute_box_standard(
                np.array([[1.0, 0.25, 0.25]]), np.array([0], np.int32),
                bounds3, blocks3, periodic3, 1, opts,
            ),
            lambda: _core.compute_box_power(
                np.full((2, 3), 0.25), np.array([0, 1], np.int32),
                np.zeros(2), bounds3, blocks3, periodic3, 1, opts,
            ),
            lambda: _core.compute_periodic_standard(
                np.array([[0.0, 0.25, 0.25],
                          [np.nextafter(1.0, 0.0), 0.25, 0.25]]),
                np.array([0, 1], np.int32), params, blocks3, 1, opts,
            ),
            lambda: _core.ghost_periodic_power(
                np.array([[0.25, 0.25, 0.25]]), np.array([0], np.int32),
                np.zeros(1), params, blocks3, 1, opts,
                np.array([[0.25, 0.25, 0.25]]), np.zeros(1),
            ),
            lambda: _core2d.compute_box_standard(
                np.full((2, 2), 0.25), np.array([0, 1], np.int32),
                bounds2, blocks2, periodic2, 1, opts,
            ),
            lambda: _core2d.ghost_box_power(
                np.array([[0.25, 0.25]]), np.array([0], np.int32),
                np.zeros(1), bounds2, blocks2, periodic2, 1, opts,
                np.array([[1.0, 0.5]]), np.zeros(1),
            ),
        )
        for case in cases:
            try:
                case()
            except ValueError:
                pass
            else:
                raise SystemExit('missing R5 native safety exception')

        _core.compute_box_standard(
            np.array([[0.0, 0.25, 0.25], [0.75, 0.75, 0.75]]),
            np.array([0, 1], np.int32), bounds3, blocks3,
            (False, False, False), 1, opts,
        )
        _core2d.ghost_box_standard(
            np.array([[0.25, 0.25]]), np.array([0], np.int32),
            bounds2, blocks2, (False, False), 1, opts,
            np.array([[0.75, 0.75]]),
        )
        print('OK R5 direct native safety')
    '''
    completed = subprocess.run(
        [sys.executable, '-c', textwrap.dedent(script)],
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert completed.returncode == 0, (
        f'R5 native safety subprocess returned {completed.returncode}\n'
        f'stdout:\n{completed.stdout}\nstderr:\n{completed.stderr}'
    )
    assert completed.stdout.strip() == 'OK R5 direct native safety'


def test_native_portable_safety_arithmetic_in_subprocess() -> None:
    script = r'''
        import math
        import numpy as np
        from pyvoro2 import _core, _core2d

        opts = (False, False, False)
        unsafe_x = 6.287573474728336e-06
        safe_x = 6.287573474728343e-06
        delta_y = 7.776015676417624e-06
        points2_unsafe = np.array([[0.0, delta_y], [unsafe_x, 0.0]])
        points2_safe = np.array([[0.0, delta_y], [safe_x, 0.0]])
        points3_unsafe = np.column_stack(
            (points2_unsafe, np.full(2, 0.5))
        )
        points3_safe = np.column_stack((points2_safe, np.full(2, 0.5)))
        u = 9.999999998039799e-06
        tiny = 1.2705494208814505e-21
        dy = 1.9800000000000001e-10
        split2_unsafe = np.array([[u, dy], [-tiny, 0.0]])
        split3_unsafe = np.column_stack(
            (split2_unsafe, np.full(2, 0.5))
        )

        def require_value_error(label, case, match=None):
            try:
                case()
            except ValueError as exc:
                if match is not None and match not in str(exc):
                    raise
            else:
                raise SystemExit(f'{label}: missing Python ValueError')

        require_value_error(
            'multidimensional-2d',
            lambda: _core2d.compute_box_standard(
                points2_unsafe, np.array([0, 1], np.int32),
                ((0.0, 1.0),) * 2, (1, 1), (False, False), 1, opts,
            ),
        )
        require_value_error(
            'multidimensional-3d',
            lambda: _core.compute_box_standard(
                points3_unsafe, np.array([0, 1], np.int32),
                ((0.0, 1.0),) * 3, (1, 1, 1),
                (False, False, False), 1, opts,
            ),
        )
        require_value_error(
            'split-coordinate-2d',
            lambda: _core2d.compute_box_standard(
                split2_unsafe, np.array([0, 1], np.int32),
                ((-1.0, 1.0),) * 2, (1, 1), (False, False), 1, opts,
            ),
            'backend-unsafe',
        )
        require_value_error(
            'split-coordinate-3d',
            lambda: _core.compute_box_standard(
                split3_unsafe, np.array([0, 1], np.int32),
                ((-1.0, 1.0),) * 3, (1, 1, 1),
                (False, False, False), 1, opts,
            ),
            'backend-unsafe',
        )

        _core2d.compute_box_standard(
            points2_safe, np.array([0, 1], np.int32),
            ((0.0, 1.0),) * 2, (1, 1), (False, False), 1, opts,
        )
        _core.compute_box_standard(
            points3_safe, np.array([0, 1], np.int32),
            ((0.0, 1.0),) * 3, (1, 1, 1),
            (False, False, False), 1, opts,
        )

        span = math.ldexp(1e-5, 63)
        require_value_error(
            'signed-int64-bin-boundary',
            lambda: _core._test_rectangular_safety_candidate_count(
                np.array([[1.0, 1.0, 1.0], [2.0, 2.0, 2.0]]),
                ((0.0, span),) * 3,
                (False, False, False),
                np.empty((0, 3)),
            ),
            'more sparse bins',
        )

        tiny_params = (1e-110, 0.0, 1e-110, 0.0, 0.0, 1e-110)
        require_value_error(
            'nonfinite-periodic-safety-geometry',
            lambda: _core.compute_periodic_standard(
                np.zeros((2, 3)), np.array([0, 1], np.int32),
                tiny_params, (1, 1, 1), 1, opts,
            ),
        )
        print('OK portable native safety arithmetic')
    '''
    completed = subprocess.run(
        [sys.executable, '-c', textwrap.dedent(script)],
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert completed.returncode == 0, (
        f'portable safety subprocess returned {completed.returncode}\n'
        f'stdout:\n{completed.stdout}\nstderr:\n{completed.stderr}'
    )
    assert completed.stdout.strip() == 'OK portable native safety arithmetic'


def test_native_sparse_rectangular_scanner_keeps_large_span_locality() -> None:
    bounds = ((0.0, 200.0),) * 3
    periodic = (True, False, False)
    axis = 1.000001 + 1.1e-5 * np.arange(17)
    points = np.array(list(product(axis, repeat=3)), dtype=np.float64)
    empty = np.empty((0, 3), dtype=np.float64)

    within_count = _core._test_rectangular_safety_candidate_count(
        points,
        bounds,
        periodic,
        empty,
    )
    cross_count = _core._test_rectangular_safety_candidate_count(
        points[::2],
        bounds,
        periodic,
        points[1::2],
    )
    assert within_count < 1_000_000
    assert cross_count < 1_000_000

    located = _core.locate_box_standard(
        points,
        np.arange(len(points), dtype=np.int32),
        bounds,
        (1, 1, 1),
        periodic,
        1,
        empty,
    )
    assert all(value.shape[0] == 0 for value in located)


def test_native_sparse_key_overflow_fails_structurally_before_insertion() -> None:
    points = np.array([[1.0, 1.0, 1.0], [2.0, 2.0, 2.0]])
    with pytest.raises(ValueError, match='more sparse bins'):
        _core.locate_box_standard(
            points,
            np.arange(len(points), dtype=np.int32),
            ((0.0, 1e100),) * 3,
            (1, 1, 1),
            (True, False, False),
            1,
            np.empty((0, 3), dtype=np.float64),
        )


def test_native_binary64_certificate_encloses_fixed_fraction_witness() -> None:
    bounds_raw, bins_raw = _core._test_periodic_safety_certificate(
        NATIVE_BINARY64_WITNESS_PARAMS
    )
    bounds = tuple(map(float, bounds_raw))
    bins = tuple(map(int, bins_raw))
    exact = _fraction_coefficient_bounds(NATIVE_BINARY64_WITNESS_PARAMS)

    assert all(
        _as_fraction(bound) >= expected
        for bound, expected in zip(bounds, exact)
    )
    assert exact[0] > Fraction(1, 4)
    assert _as_fraction(float.fromhex('0x1.fffffffffffeep-3')) < exact[0]
    assert bins == (3, 146, 10045)


@pytest.mark.parametrize('expected_bins', [1, 2, 3, 4])
def test_native_certified_layout_covers_small_bin_counts(
    expected_bins: int,
) -> None:
    diagonal = (expected_bins + 0.5) * 1e-5
    params = (diagonal, 0.0, diagonal, 0.0, 0.0, diagonal)
    bounds_raw, bins_raw = _core._test_periodic_safety_certificate(params)
    exact = _fraction_coefficient_bounds(params)

    assert tuple(map(int, bins_raw)) == (expected_bins,) * 3
    assert all(
        _as_fraction(float(bound)) >= expected
        for bound, expected in zip(bounds_raw, exact)
    )


@pytest.mark.parametrize('denominator', [2, 3, 4])
def test_native_certified_bounds_cover_near_rational_transitions(
    denominator: int,
) -> None:
    diagonal = denominator * 1e-5
    diagonals = []
    for direction in (0.0, None, math.inf):
        value = diagonal
        if direction is not None:
            for _ in range(16):
                value = np.nextafter(value, direction)
        diagonals.append(float(value))

    observed_bins = []
    for value in diagonals:
        params = (value, 0.0, value, 0.0, 0.0, value)
        bounds_raw, bins_raw = _core._test_periodic_safety_certificate(params)
        exact = _fraction_coefficient_bounds(params)
        bins = tuple(map(int, bins_raw))
        observed_bins.append(bins[0])
        assert all(
            _as_fraction(float(bound)) >= expected
            for bound, expected in zip(bounds_raw, exact)
        )
        assert all(
            count <= _fraction_floor(1 / expected)
            for count, expected in zip(bins, exact)
        )

    assert observed_bins == [denominator - 1, denominator - 1, denominator]


def test_native_certified_layout_retains_subnormal_positive_bounds() -> None:
    params = (1e308, 0.0, 1e308, 0.0, 0.0, 1e308)
    bounds_raw, bins_raw = _core._test_periodic_safety_certificate(params)
    bounds = tuple(map(float, bounds_raw))

    assert all(0.0 < bound < np.finfo(np.float64).tiny for bound in bounds)
    assert tuple(map(int, bins_raw)) == (1 << 53,) * 3
    assert all(
        _as_fraction(bound) >= expected
        for bound, expected in zip(bounds, _fraction_coefficient_bounds(params))
    )


def test_native_uncertain_key_alias_budget_is_cumulative() -> None:
    assert _core._test_periodic_key_alias_budget(
        [1, 500_001, 500_001]
    ) == 1_000_000
    with pytest.raises(ValueError, match='key aliases.*budget'):
        _core._test_periodic_key_alias_budget([1, 500_001, 500_002])
    with pytest.raises(ValueError, match='at least one certified key'):
        _core._test_periodic_key_alias_budget([0])


def test_native_interval_keys_cover_boundaries_and_periodic_seams() -> None:
    diagonal = math.ldexp(1.0, -14)
    boundary_params = (diagonal, 0.0, diagonal, 0.0, 0.0, diagonal)
    _bounds, bins_raw = _core._test_periodic_safety_certificate(
        boundary_params
    )
    bins = tuple(map(int, bins_raw))
    boundary_point = (diagonal / 2.0,) * 3
    boundary_keys = {
        tuple(map(int, key))
        for key in _core._test_periodic_safety_keys(
            boundary_point, boundary_params
        )
    }
    assert bins == (6, 6, 6)
    assert (
        _fraction_periodic_key(boundary_point, boundary_params, bins)
        in boundary_keys
    )
    assert {key[0] for key in boundary_keys} == {2, 3}
    assert {key[1] for key in boundary_keys} == {2, 3}
    assert {key[2] for key in boundary_keys} == {2, 3}

    seam_params = (1.0, 0.5, 1.0, 0.25, -0.5, 1.0)
    _bounds, seam_bins_raw = _core._test_periodic_safety_certificate(
        seam_params
    )
    seam_bins = tuple(map(int, seam_bins_raw))
    # This nonzero Cartesian point has exact fractional coordinate (0, 1, 0).
    seam_point = (0.5, 1.0, 0.0)
    seam_keys = {
        tuple(map(int, key))
        for key in _core._test_periodic_safety_keys(seam_point, seam_params)
    }
    assert _fraction_periodic_key(seam_point, seam_params, seam_bins) in seam_keys
    assert {key[0] for key in seam_keys} == {0, seam_bins[0] - 1}
    assert {key[1] for key in seam_keys} == {0, seam_bins[1] - 1}


def test_native_inclusive_threshold_and_false_positive_halo() -> None:
    params = (1.0, 0.0, 1.0, 0.0, 0.0, 1.0)
    denominator = 1 << 43
    boundary = np.array(
        [79091395 / denominator, 38492551 / denominator, 1 / denominator],
        dtype=np.float64,
    )
    exact_boundary_squared = sum(
        (_as_fraction(value) ** 2 for value in boundary), Fraction()
    )
    assert exact_boundary_squared == NATIVE_SAFETY_DISTANCE_SQUARED

    inside = boundary.copy()
    inside[0] = np.nextafter(inside[0], 0.0)
    outside_one_ulp = boundary.copy()
    outside_one_ulp[0] = np.nextafter(outside_one_ulp[0], math.inf)
    outside_two_ulps = outside_one_ulp.copy()
    outside_two_ulps[0] = np.nextafter(outside_two_ulps[0], math.inf)

    origin = (0.0, 0.0, 0.0)
    assert _fraction_periodic_pair_is_unsafe(origin, inside, params)
    assert _fraction_periodic_pair_is_unsafe(origin, boundary, params)
    assert not _fraction_periodic_pair_is_unsafe(origin, outside_one_ulp, params)
    assert not _fraction_periodic_pair_is_unsafe(origin, outside_two_ulps, params)
    assert _core._test_periodic_pair_is_unsafe(origin, inside, params)
    assert _core._test_periodic_pair_is_unsafe(origin, boundary, params)
    # The outward interval deliberately rejects the first safe binary64 point.
    assert _core._test_periodic_pair_is_unsafe(origin, outside_one_ulp, params)
    assert not _core._test_periodic_pair_is_unsafe(origin, outside_two_ulps, params)


@pytest.mark.parametrize(
    ('left', 'right', 'params'),
    [
        (
            NATIVE_LARGE_SHIFT_WITNESS_LEFT,
            (0.0, 0.0, 0.0),
            NATIVE_BINARY64_WITNESS_PARAMS,
        ),
        (
            (0.75, 0.0, 0.0),
            (0.25, np.nextafter(1.0, 0.0), np.nextafter(1.0, 0.0)),
            (1.0, 1.5, 1.0, 0.0, 0.0, 1.0),
        ),
    ],
    ids=['large-shift-13530--68-1', 'shift-2--1--1'],
)
def test_native_outward_shift_boxes_reach_unsafe_images(
    left: tuple[float, float, float],
    right: tuple[float, float, float],
    params: tuple[float, ...],
) -> None:
    assert _fraction_periodic_pair_is_unsafe(left, right, params)
    assert _core._test_periodic_pair_is_unsafe(left, right, params)


@pytest.mark.parametrize(
    'path',
    tuple(
        path
        for path in PERIODIC_CONSTRUCTOR_PATHS
        if path.operation != 'ghost'
    ),
    ids=lambda p: p.label,
)
def test_native_large_shift_persistent_pair_fails_before_insertion(
    path: NativePath,
) -> None:
    points = np.array(
        [NATIVE_LARGE_SHIFT_WITNESS_LEFT, (0.0, 0.0, 0.0)],
        dtype=np.float64,
    )
    with pytest.raises(ValueError, match='backend-unsafe periodic pair'):
        getattr(path.module, path.name)(
            *_native_args(
                path,
                points=points,
                cell_params=NATIVE_BINARY64_WITNESS_PARAMS,
            )
        )


@pytest.mark.parametrize(
    'path',
    tuple(
        path
        for path in PERIODIC_CONSTRUCTOR_PATHS
        if path.operation == 'ghost'
    ),
    ids=lambda p: p.label,
)
def test_native_large_shift_ghost_pair_fails_before_insertion(
    path: NativePath,
) -> None:
    points = np.zeros((1, 3), dtype=np.float64)
    query = np.array([NATIVE_LARGE_SHIFT_WITNESS_LEFT], dtype=np.float64)
    with pytest.raises(ValueError, match='backend-unsafe periodic ghost'):
        getattr(path.module, path.name)(
            *_native_args(
                path,
                points=points,
                ids=np.array([0], dtype=np.int32),
                radii=np.zeros(1),
                queries=query,
                ghost_radii=np.zeros(1),
                cell_params=NATIVE_BINARY64_WITNESS_PARAMS,
            )
        )


def test_native_interval_classifier_matches_fixed_seed_fraction_oracle() -> None:
    rng = np.random.default_rng(20260813)
    exact_unsafe_count = 0
    false_negative_count = 0
    false_positive_count = 0

    for case in range(256):
        diagonal = rng.uniform(0.5, 2.0, size=3)
        if case % 4 < 2:
            off_diagonal = (0.0, 0.0, 0.0)
        else:
            off_diagonal = tuple(rng.uniform(-2.0, 2.0, size=3))
        params = (
            float(diagonal[0]),
            float(off_diagonal[0]),
            float(diagonal[1]),
            float(off_diagonal[1]),
            float(off_diagonal[2]),
            float(diagonal[2]),
        )
        left = rng.uniform(-1.0, 1.0, size=3)
        shift = rng.integers(-8, 9, size=3)
        lattice = np.array(
            [
                [params[0], 0.0, 0.0],
                [params[1], params[2], 0.0],
                [params[3], params[4], params[5]],
            ],
            dtype=np.float64,
        )
        residual = rng.normal(size=3)
        boundary_scale = 1.0 - 1e-8 if case % 2 == 0 else 1.0 + 1e-8
        residual *= (
            NATIVE_SAFETY_DISTANCE
            * boundary_scale
            / np.linalg.norm(residual)
        )
        right = left - shift @ lattice + residual

        expected = _fraction_periodic_pair_is_unsafe(left, right, params)
        actual = bool(
            _core._test_periodic_pair_is_unsafe(left, right, params)
        )
        exact_unsafe_count += int(expected)
        false_negative_count += int(expected and not actual)
        false_positive_count += int(actual and not expected)

    assert exact_unsafe_count == 128
    assert false_negative_count == 0
    assert false_positive_count == 0


def test_native_candidate_scanner_matches_fixed_seed_fraction_oracle() -> None:
    rng = np.random.default_rng(20260814)
    empty = np.empty((0, 3), dtype=np.float64)
    statistics = {
        'persistent': {
            'total': 0,
            'exact_unsafe': 0,
            'false_negative': 0,
            'false_positive': 0,
        },
        'ghost': {
            'total': 0,
            'exact_unsafe': 0,
            'false_negative': 0,
            'false_positive': 0,
        },
    }
    cell_kinds = {'diagonal': 0, 'sheared': 0}
    pair_kinds = {'interior': 0, 'seam': 0}

    for case in range(256):
        diagonal = rng.uniform(0.75, 2.0, size=3)
        is_sheared = case % 4 >= 2
        if is_sheared:
            off_diagonal = rng.uniform(-0.35, 0.35, size=3)
            cell_kinds['sheared'] += 1
        else:
            off_diagonal = np.zeros(3, dtype=np.float64)
            cell_kinds['diagonal'] += 1
        params = (
            float(diagonal[0]),
            float(off_diagonal[0]),
            float(diagonal[1]),
            float(off_diagonal[1]),
            float(off_diagonal[2]),
            float(diagonal[2]),
        )

        direction = rng.normal(size=3)
        is_seam = (case // 4) % 2 == 1
        if is_seam:
            # Make the x image cross the periodic seam.  Keeping the left
            # endpoint one binary64 step below bx also exercises uncertain
            # seam-key aliases without violating half-open containment.
            direction[0] = abs(direction[0]) + 0.5
        direction /= np.linalg.norm(direction)
        boundary_scale = 1.0 - 1e-8 if case % 2 == 0 else 1.0 + 1e-8
        residual = (
            float(NATIVE_SAFETY_DISTANCE)
            * boundary_scale
            * direction
        )

        if is_seam:
            pair_kinds['seam'] += 1
            left = np.array(
                [
                    np.nextafter(diagonal[0], 0.0),
                    rng.uniform(0.25, 0.75) * diagonal[1],
                    rng.uniform(0.25, 0.75) * diagonal[2],
                ],
                dtype=np.float64,
            )
            right = np.array(
                [residual[0], left[1] + residual[1], left[2] + residual[2]],
                dtype=np.float64,
            )
        else:
            pair_kinds['interior'] += 1
            left = rng.uniform(0.25, 0.75, size=3) * diagonal
            right = left + residual
            left = np.asarray(left, dtype=np.float64)
            right = np.asarray(right, dtype=np.float64)

        assert np.all(left >= 0.0)
        assert np.all(left < diagonal)
        assert np.all(right >= 0.0)
        assert np.all(right < diagonal)
        expected = _fraction_periodic_pair_is_unsafe(left, right, params)

        dispatches = (
            (
                'persistent',
                np.vstack((left, right)),
                empty,
                'backend-unsafe periodic pair',
            ),
            (
                'ghost',
                left[None, :],
                right[None, :],
                'backend-unsafe periodic ghost',
            ),
        )
        for label, points, inserted_queries, message in dispatches:
            actual = _native_scanner_reports_unsafe(
                points,
                inserted_queries,
                params,
                expected_message=message,
            )
            statistics[label]['total'] += 1
            statistics[label]['exact_unsafe'] += int(expected)
            statistics[label]['false_negative'] += int(
                expected and not actual
            )
            statistics[label]['false_positive'] += int(
                actual and not expected
            )

    expected_statistics = {
        'total': 256,
        'exact_unsafe': 128,
        'false_negative': 0,
        'false_positive': 0,
    }
    assert cell_kinds == {'diagonal': 128, 'sheared': 128}
    assert pair_kinds == {'interior': 128, 'seam': 128}
    assert statistics['persistent'] == expected_statistics
    assert statistics['ghost'] == expected_statistics


def test_native_triclinic_sparse_scanner_has_exact_local_work_counts() -> None:
    params = (2.0, 0.25, 2.0, 0.1, -0.2, 2.0)
    axis = 0.2 + 2.1e-5 * np.arange(17)
    points = np.array(list(product(axis, repeat=3)), dtype=np.float64)
    empty = np.empty((0, 3), dtype=np.float64)

    within_count = _core._test_periodic_safety_candidate_count(
        points, params, empty
    )
    cross_count = _core._test_periodic_safety_candidate_count(
        points[::2], params, points[1::2]
    )

    assert len(points) == 4913
    assert within_count == 1660
    assert cross_count == 1567
    assert within_count < len(points) * (len(points) - 1) // 2


def test_native_seam_aliases_are_deduplicated_for_persistent_and_ghost_scans(
) -> None:
    params = (2.0, 0.0, 2.0, 0.0, 0.0, 2.0)
    left = []
    right = []
    for index in range(64):
        y = 0.1 + index * 0.025
        left.append((0.0, y, 0.2))
        right.append(
            (
                np.nextafter(2.0, 0.0),
                y + 8e-6,
                0.2 + 8e-6,
            )
        )
    persistent = np.asarray(left, dtype=np.float64)
    ghosts = np.asarray(right, dtype=np.float64)
    empty = np.empty((0, 3), dtype=np.float64)

    within_count = _core._test_periodic_safety_candidate_count(
        np.vstack((persistent, ghosts)), params, empty
    )
    cross_count = _core._test_periodic_safety_candidate_count(
        persistent, params, ghosts
    )

    assert within_count == 64
    assert cross_count == 64


def test_native_binary64_triclinic_certificate_in_subprocess() -> None:
    script = r'''
        import numpy as np
        from pyvoro2 import _core

        params = (
            0.00017856460945140374,
            -5.725280006612657,
            1.65273947780994,
            -391.7349918299756,
            113.08370683864604,
            0.10045884903938922,
        )
        left = (
            2.7785562527544788e-05,
            0.6974223475701162,
            0.1004588490393892,
        )
        right = (0.0, 0.0, 0.0)
        bounds, bins = _core._test_periodic_safety_certificate(params)
        if not (bounds[0] > 0.25 and tuple(bins) == (3, 146, 10045)):
            raise SystemExit('binary64 certificate did not enclose witness')
        if not _core._test_periodic_pair_is_unsafe(left, right, params):
            raise SystemExit('large-shift witness was missed')

        points = np.array([left, right], dtype=np.float64)
        try:
            _core.compute_periodic_standard(
                points,
                np.array([0, 1], dtype=np.int32),
                params,
                (1, 1, 1),
                1,
                (False, False, False),
            )
        except ValueError as exc:
            if 'backend-unsafe periodic pair' not in str(exc):
                raise
        else:
            raise SystemExit('persistent witness reached insertion')

        try:
            _core.ghost_periodic_standard(
                np.zeros((1, 3)),
                np.array([0], dtype=np.int32),
                params,
                (1, 1, 1),
                1,
                (False, False, False),
                np.array([left], dtype=np.float64),
            )
        except ValueError as exc:
            if 'backend-unsafe periodic ghost' not in str(exc):
                raise
        else:
            raise SystemExit('ghost witness reached insertion')
        print('OK R5 binary64 triclinic certificate')
    '''
    completed = subprocess.run(
        [sys.executable, '-c', textwrap.dedent(script)],
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert completed.returncode == 0, (
        f'binary64 certificate subprocess returned {completed.returncode}\n'
        f'stdout:\n{completed.stdout}\nstderr:\n{completed.stderr}'
    )
    assert completed.stdout.strip() == 'OK R5 binary64 triclinic certificate'


def test_native_shift_enumeration_budget_fails_structurally() -> None:
    params = (1e-8, 0.0, 1e-8, 0.0, 0.0, 1e-8)
    with pytest.raises(ValueError, match='shift enumeration.*budget'):
        _core._test_periodic_pair_is_unsafe(
            (0.0, 0.0, 0.0),
            (0.0, 0.0, 0.0),
            params,
        )


def test_native_uncertifiable_keys_and_shifts_fail_structurally() -> None:
    identity = (1.0, 0.0, 1.0, 0.0, 0.0, 1.0)
    with pytest.raises(ValueError, match='key.*cannot certify'):
        _core._test_periodic_safety_keys((1e308, 0.0, 0.0), identity)
    with pytest.raises(ValueError, match='shift enumeration.*exact binary64'):
        _core._test_periodic_pair_is_unsafe(
            (1e308, 0.0, 0.0),
            (0.0, 0.0, 0.0),
            identity,
        )
    with pytest.raises(ValueError, match='cannot certify.*inverse basis'):
        _core._test_periodic_safety_certificate(
            (1e-310, 0.0, 1e-310, 0.0, 0.0, 1e-310)
        )
