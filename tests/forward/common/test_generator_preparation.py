from __future__ import annotations

from dataclasses import dataclass, field
from fractions import Fraction
from itertools import combinations, product
import math
from typing import Any
import warnings

import numpy as np
import pytest

import pyvoro2
import pyvoro2.api as api3d
from pyvoro2._internal.duplicate_scanning import (
    candidate_pair_count,
    candidate_pairs,
    cross_candidate_pairs,
    scan_close_pairs,
    scan_cross_close_pairs,
)
from pyvoro2._internal.generator_preparation import (
    BACKEND_SAFETY_DISTANCE,
    BACKEND_SAFETY_DISTANCE_SQUARED,
    prepare_generators,
)
from pyvoro2._internal.periodic_images import (
    _basis_cache_clear,
    _basis_cache_info,
    _exact_triclinic_bucket_layout,
)
from pyvoro2._internal.planar.domain_geometry import geometry2d
from pyvoro2._internal.spatial.domain_geometry import geometry3d
from pyvoro2.duplicates import DuplicateError
import pyvoro2.planar as planar
import pyvoro2.planar.api as api2d


@dataclass
class RecordingSafeCore:
    calls: list[tuple[str, tuple[Any, ...]]] = field(default_factory=list)
    compute_rows: list[dict[str, object]] | None = None

    def __getattr__(self, name: str):
        def call(*args: Any):
            self.calls.append((name, args))
            if name.startswith('compute_'):
                if self.compute_rows is not None:
                    return list(self.compute_rows)
                return [{'id': int(value)} for value in np.asarray(args[1])]
            if name.startswith('locate_'):
                queries = np.asarray(args[-1], dtype=np.float64)
                return (
                    np.zeros(len(queries), dtype=np.bool_),
                    np.full(len(queries), -1, dtype=np.int32),
                    np.full(queries.shape, np.nan, dtype=np.float64),
                )
            return []

        return call


def _install_core(monkeypatch, dim: int, core: object) -> None:
    if dim == 3:
        monkeypatch.setattr(api3d, '_core', core)
        monkeypatch.setattr(api3d, '_CORE_IMPORT_ERROR', None)
    else:
        monkeypatch.setattr(api2d, '_core2d', core)
        monkeypatch.setattr(api2d, '_CORE2D_IMPORT_ERROR', None)


def _package(dim: int):
    return pyvoro2 if dim == 3 else planar


def _box(dim: int):
    bounds = tuple((0.0, 1.0) for _ in range(dim))
    return pyvoro2.Box(bounds) if dim == 3 else planar.Box(bounds)


def _partial_cell(dim: int):
    bounds = tuple((0.0, 1.0) for _ in range(dim))
    periodic = (True,) + (False,) * (dim - 1)
    if dim == 3:
        return pyvoro2.OrthorhombicCell(bounds, periodic=periodic)
    return planar.RectangularCell(bounds, periodic=periodic)


def _invoke(
    dim: int,
    operation: str,
    points: np.ndarray,
    *,
    domain,
    queries: np.ndarray | None = None,
    mode: str = 'standard',
    **kwargs: object,
):
    package = _package(dim)
    if queries is None:
        queries = np.full((1, dim), 0.75, dtype=np.float64)
    common: dict[str, object] = {'domain': domain, 'mode': mode, **kwargs}
    if mode == 'power':
        common['radii'] = np.zeros(len(points), dtype=np.float64)
    if operation == 'compute':
        return package.compute(
            points,
            output='cells',
            return_vertices=False,
            return_adjacency=False,
            **(
                {**common, 'return_faces': False}
                if dim == 3
                else {**common, 'return_edges': False}
            ),
        )
    if operation == 'locate':
        return package.locate(points, queries, **common)
    if mode == 'power':
        common['ghost_radius'] = 0.0
    return package.ghost_cells(
        points,
        queries,
        return_vertices=False,
        return_adjacency=False,
        **(
            {**common, 'return_faces': False}
            if dim == 3
            else {**common, 'return_edges': False}
        ),
    )


def _boundary_values() -> tuple[tuple[float, bool], ...]:
    return (
        (0.0, True),
        (float(np.nextafter(0.0, -np.inf)), False),
        (float(np.nextafter(0.0, np.inf)), True),
        (float(np.nextafter(1.0, 0.0)), True),
        (1.0, False),
        (float(np.nextafter(1.0, np.inf)), False),
    )


@pytest.mark.parametrize('dim', [2, 3])
@pytest.mark.parametrize('operation', ['compute', 'locate', 'ghost_cells'])
@pytest.mark.parametrize('domain_factory', [_box, _partial_cell])
def test_persistent_generator_half_open_boundaries(
    monkeypatch,
    dim: int,
    operation: str,
    domain_factory,
) -> None:
    domain = domain_factory(dim)
    periodic = tuple(
        bool(value) for value in getattr(domain, 'periodic', (False,) * dim)
    )
    for axis in range(dim):
        if periodic[axis]:
            continue
        for value, accepted in _boundary_values():
            core = RecordingSafeCore()
            _install_core(monkeypatch, dim, core)
            points = np.full((1, dim), 0.25, dtype=np.float64)
            points[0, axis] = value
            if accepted:
                _invoke(dim, operation, points, domain=domain)
                assert len(core.calls) == 1
            else:
                with pytest.raises(ValueError, match=f'axis {axis}'):
                    _invoke(dim, operation, points, domain=domain)
                assert core.calls == []


@pytest.mark.parametrize('dim', [2, 3])
@pytest.mark.parametrize('domain_factory', [_box, _partial_cell])
@pytest.mark.parametrize('mode', ['standard', 'power'])
def test_temporary_ghost_generator_half_open_boundaries(
    monkeypatch,
    dim: int,
    domain_factory,
    mode: str,
) -> None:
    domain = domain_factory(dim)
    periodic = tuple(
        bool(value) for value in getattr(domain, 'periodic', (False,) * dim)
    )
    persistent = np.full((1, dim), 0.25, dtype=np.float64)
    for axis in range(dim):
        if periodic[axis]:
            continue
        for value, accepted in _boundary_values():
            core = RecordingSafeCore()
            _install_core(monkeypatch, dim, core)
            query = np.full((1, dim), 0.75, dtype=np.float64)
            query[0, axis] = value
            if accepted:
                _invoke(
                    dim,
                    'ghost_cells',
                    persistent,
                    domain=domain,
                    queries=query,
                    mode=mode,
                )
                assert len(core.calls) == 1
            else:
                with pytest.raises(ValueError, match=f'axis {axis}'):
                    _invoke(
                        dim,
                        'ghost_cells',
                        persistent,
                        domain=domain,
                        queries=query,
                        mode=mode,
                    )
                assert core.calls == []


@pytest.mark.parametrize('dim', [2, 3])
@pytest.mark.parametrize('operation', ['compute', 'locate', 'ghost_cells'])
@pytest.mark.parametrize('periodic_value', [1.0, 4.25, -3.75])
def test_periodic_axes_remap_persistent_generators_before_dispatch(
    monkeypatch,
    dim: int,
    operation: str,
    periodic_value: float,
) -> None:
    core = RecordingSafeCore()
    _install_core(monkeypatch, dim, core)
    point = np.full((1, dim), 0.25, dtype=np.float64)
    point[0, 0] = periodic_value
    _invoke(dim, operation, point, domain=_partial_cell(dim))

    native_points = np.asarray(core.calls[0][1][0])
    assert 0.0 <= native_points[0, 0] < 1.0
    assert native_points[0, 0] == pytest.approx(periodic_value % 1.0)


@pytest.mark.parametrize('dim', [2, 3])
@pytest.mark.parametrize('periodic_value', [1.0, 4.25, -3.75])
def test_periodic_axes_remap_temporary_ghost_before_dispatch(
    monkeypatch,
    dim: int,
    periodic_value: float,
) -> None:
    core = RecordingSafeCore()
    _install_core(monkeypatch, dim, core)
    point = np.full((1, dim), 0.6, dtype=np.float64)
    query = np.full((1, dim), 0.25, dtype=np.float64)
    query[0, 0] = periodic_value
    _invoke(
        dim,
        'ghost_cells',
        point,
        domain=_partial_cell(dim),
        queries=query,
    )

    native_query = np.asarray(core.calls[0][1][-1])
    assert 0.0 <= native_query[0, 0] < 1.0
    assert native_query[0, 0] == pytest.approx(periodic_value % 1.0)


@pytest.mark.parametrize('dim', [2, 3])
def test_outside_locate_query_keeps_existing_query_semantics(
    monkeypatch,
    dim: int,
) -> None:
    core = RecordingSafeCore()
    _install_core(monkeypatch, dim, core)
    query = np.full((1, dim), 2.0, dtype=np.float64)
    _invoke(
        dim,
        'locate',
        np.full((1, dim), 0.25),
        domain=_box(dim),
        queries=query,
    )
    np.testing.assert_array_equal(np.asarray(core.calls[0][1][-1]), query)


@pytest.mark.parametrize('dim', [2, 3])
@pytest.mark.parametrize('mode', ['standard', 'power'])
@pytest.mark.parametrize('duplicate_check', ['off', 'warn', 'raise'])
@pytest.mark.parametrize('threshold', [5e-6, 1e-5, 1e-3])
@pytest.mark.parametrize('wrap', [False, True])
def test_mandatory_duplicate_floor_cannot_be_disabled(
    monkeypatch,
    dim: int,
    mode: str,
    duplicate_check: str,
    threshold: float,
    wrap: bool,
) -> None:
    core = RecordingSafeCore()
    _install_core(monkeypatch, dim, core)
    points = np.full((2, dim), 0.25, dtype=np.float64)
    with pytest.raises(DuplicateError) as caught:
        _invoke(
            dim,
            'compute',
            points,
            domain=_box(dim),
            mode=mode,
            duplicate_check=duplicate_check,
            duplicate_threshold=threshold,
            duplicate_wrap=wrap,
        )

    error = caught.value
    assert error.kind == 'backend_safety'
    assert error.threshold == BACKEND_SAFETY_DISTANCE
    assert error.safety_distance_squared == BACKEND_SAFETY_DISTANCE_SQUARED
    assert error.user_threshold == threshold
    assert error.operation == 'compute'
    assert error.external_ids == ((0, 1),)
    assert core.calls == []


@pytest.mark.parametrize(
    ('separation', 'unsafe'),
    [
        (5e-6, True),
        (float(np.nextafter(1e-5, 0.0)), True),
        (1e-5, False),
        (float(np.nextafter(1e-5, np.inf)), False),
    ],
)
@pytest.mark.parametrize('dim', [2, 3])
@pytest.mark.parametrize('mode', ['standard', 'power'])
def test_nonperiodic_squared_floor_boundary(
    monkeypatch,
    separation: float,
    unsafe: bool,
    dim: int,
    mode: str,
) -> None:
    core = RecordingSafeCore()
    _install_core(monkeypatch, dim, core)
    points = np.full((2, dim), 0.25)
    points[0, 0] = 0.0
    points[1, 0] = separation
    if unsafe:
        with pytest.raises(DuplicateError, match='backend-unsafe'):
            _invoke(dim, 'compute', points, domain=_box(dim), mode=mode)
        assert core.calls == []
    else:
        _invoke(dim, 'compute', points, domain=_box(dim), mode=mode)
        assert len(core.calls) == 1


@pytest.mark.parametrize(
    ('delta_x', 'unsafe'),
    [
        (6.287573474728336e-06, True),
        (6.287573474728343e-06, False),
    ],
)
@pytest.mark.parametrize('dim', [2, 3])
@pytest.mark.parametrize('mode', ['standard', 'power'])
def test_nonperiodic_multidimensional_squared_floor_is_exact(
    monkeypatch,
    delta_x: float,
    unsafe: bool,
    dim: int,
    mode: str,
) -> None:
    delta_y = -7.776015676417624e-06
    delta = (delta_x, delta_y) + ((0.0,) if dim == 3 else ())
    exact_squared = sum(
        (Fraction.from_float(value) ** 2 for value in delta),
        Fraction(),
    )
    threshold = Fraction.from_float(BACKEND_SAFETY_DISTANCE_SQUARED)
    assert (exact_squared <= threshold) is unsafe

    core = RecordingSafeCore()
    _install_core(monkeypatch, dim, core)
    points = np.zeros((2, dim), dtype=np.float64)
    points[0, 1] = -delta_y
    points[1, 0] = delta_x

    if unsafe:
        with pytest.raises(DuplicateError) as caught:
            _invoke(dim, 'compute', points, domain=_box(dim), mode=mode)
        assert caught.value.kind == 'backend_safety'
        assert core.calls == []
    else:
        _invoke(dim, 'compute', points, domain=_box(dim), mode=mode)
        assert len(core.calls) == 1


def _split_coordinate_floor_points(
    dim: int,
    *,
    second_x_multiplier: float,
) -> np.ndarray:
    u = 9.999999998039799e-06
    tiny = 1.2705494208814505e-21
    dy = 1.9800000000000001e-10
    points = np.zeros((2, dim), dtype=np.float64)
    points[0, :2] = (u, dy)
    points[1, 0] = -second_x_multiplier * tiny
    if dim == 3:
        points[:, 2] = 0.25
    return points


def _exact_cartesian_squared(
    left: np.ndarray,
    right: np.ndarray,
) -> Fraction:
    return sum(
        (
            Fraction.from_float(float(left_value))
            - Fraction.from_float(float(right_value))
        ) ** 2
        for left_value, right_value in zip(left, right)
    )


def test_nonperiodic_scanners_subtract_source_coordinates_exactly() -> None:
    points = _split_coordinate_floor_points(2, second_x_multiplier=1.0)
    threshold = Fraction.from_float(BACKEND_SAFETY_DISTANCE_SQUARED)
    exact_squared = _exact_cartesian_squared(points[0], points[1])
    rounded_delta = points[0] - points[1]
    rounded_squared = sum(
        (
            Fraction.from_float(float(value)) ** 2
            for value in rounded_delta
        ),
        Fraction(),
    )
    assert exact_squared <= threshold
    assert rounded_squared > threshold

    within = scan_close_pairs(
        points,
        radius=BACKEND_SAFETY_DISTANCE,
        inclusive_squared=BACKEND_SAFETY_DISTANCE_SQUARED,
        max_pairs=10,
    )
    cross = scan_cross_close_pairs(
        points[:1],
        points[1:],
        radius=BACKEND_SAFETY_DISTANCE,
        inclusive_squared=BACKEND_SAFETY_DISTANCE_SQUARED,
        max_pairs=10,
    )
    assert [(i, j) for i, j, _distance in within.pairs] == [(0, 1)]
    assert [(i, j) for i, j, _distance in cross.pairs] == [(0, 0)]


@pytest.mark.parametrize(
    ('second_x_multiplier', 'unsafe'),
    [(1.0, True), (2.0, False)],
)
@pytest.mark.parametrize('operation', ['compute', 'ghost_cells'])
@pytest.mark.parametrize('dim', [2, 3])
@pytest.mark.parametrize('mode', ['standard', 'power'])
def test_central_preparation_uses_exact_source_coordinate_distance(
    monkeypatch,
    second_x_multiplier: float,
    unsafe: bool,
    operation: str,
    dim: int,
    mode: str,
) -> None:
    points = _split_coordinate_floor_points(
        dim,
        second_x_multiplier=second_x_multiplier,
    )
    threshold = Fraction.from_float(BACKEND_SAFETY_DISTANCE_SQUARED)
    exact_squared = _exact_cartesian_squared(points[0], points[1])
    assert (exact_squared <= threshold) is unsafe

    core = RecordingSafeCore()
    _install_core(monkeypatch, dim, core)
    bounds = tuple((-1.0, 1.0) for _ in range(dim))
    domain = pyvoro2.Box(bounds) if dim == 3 else planar.Box(bounds)
    if operation == 'compute':
        persistent = points
        queries = None
        ids = np.array([41, 83], dtype=np.int64)
        expected_external_ids = ((41, 83),)
    else:
        persistent = points[:1]
        queries = points[1:]
        ids = np.array([41], dtype=np.int64)
        expected_external_ids = ((41, 0),)

    if unsafe:
        with pytest.raises(DuplicateError) as caught:
            _invoke(
                dim,
                operation,
                persistent,
                domain=domain,
                queries=queries,
                mode=mode,
                ids=ids,
            )
        assert caught.value.kind == 'backend_safety'
        assert caught.value.operation == operation
        assert caught.value.external_ids == expected_external_ids
        assert core.calls == []
    else:
        _invoke(
            dim,
            operation,
            persistent,
            domain=domain,
            queries=queries,
            mode=mode,
            ids=ids,
        )
        assert len(core.calls) == 1


@pytest.mark.parametrize(
    ('separation', 'unsafe'),
    [
        (float(np.nextafter(1e-5, 0.0)), True),
        (1e-5, False),
        (float(np.nextafter(1e-5, np.inf)), False),
    ],
)
@pytest.mark.parametrize('dim', [2, 3])
@pytest.mark.parametrize('mode', ['standard', 'power'])
def test_periodic_squared_floor_boundary_uses_exact_minimum_image(
    monkeypatch,
    separation: float,
    unsafe: bool,
    dim: int,
    mode: str,
) -> None:
    core = RecordingSafeCore()
    _install_core(monkeypatch, dim, core)
    bounds = tuple((0.0, 1.0) for _ in range(dim))
    domain = (
        pyvoro2.OrthorhombicCell(bounds, periodic=(True,) * dim)
        if dim == 3
        else planar.RectangularCell(bounds, periodic=(True,) * dim)
    )
    points = np.full((2, dim), 0.25)
    points[0, 0] = 0.0
    points[1, 0] = separation
    if unsafe:
        with pytest.raises(DuplicateError) as caught:
            _invoke(dim, 'compute', points, domain=domain, mode=mode)
        assert caught.value.kind == 'backend_safety'
        assert caught.value.minimum_image_used is True
        assert core.calls == []
    else:
        _invoke(dim, 'compute', points, domain=domain, mode=mode)
        assert len(core.calls) == 1


@pytest.mark.parametrize('dim', [2, 3])
@pytest.mark.parametrize('mode', ['standard', 'power'])
def test_partial_periodic_equivalence_is_always_mandatory(
    monkeypatch,
    dim: int,
    mode: str,
) -> None:
    core = RecordingSafeCore()
    _install_core(monkeypatch, dim, core)
    points = np.full((2, dim), 0.4)
    points[1, 0] += 1.0
    with pytest.raises(DuplicateError) as caught:
        _invoke(
            dim,
            'compute',
            points,
            domain=_partial_cell(dim),
            mode=mode,
            duplicate_check='off',
            duplicate_wrap=False,
        )
    assert caught.value.minimum_image_used is True
    assert caught.value.optional_wrap_used is False
    assert core.calls == []


@pytest.mark.parametrize('dim', [2, 3])
@pytest.mark.parametrize('mode', ['standard', 'power'])
def test_fully_periodic_corner_equivalence_is_mandatory(
    monkeypatch,
    dim: int,
    mode: str,
) -> None:
    core = RecordingSafeCore()
    _install_core(monkeypatch, dim, core)
    bounds = tuple((0.0, 1.0) for _ in range(dim))
    domain = (
        pyvoro2.OrthorhombicCell(bounds, periodic=(True,) * dim)
        if dim == 3
        else planar.RectangularCell(bounds, periodic=(True,) * dim)
    )
    points = np.vstack((np.full(dim, 0.1), np.full(dim, 1.1)))
    with pytest.raises(DuplicateError) as caught:
        _invoke(
            dim,
            'compute',
            points,
            domain=domain,
            mode=mode,
            duplicate_check='off',
            duplicate_wrap=False,
        )
    assert caught.value.kind == 'backend_safety'
    assert caught.value.minimum_image_used is True
    assert core.calls == []


@pytest.mark.parametrize('mode', ['standard', 'power'])
def test_triclinic_lattice_equivalence_is_always_mandatory(
    monkeypatch,
    mode: str,
) -> None:
    core = RecordingSafeCore()
    _install_core(monkeypatch, 3, core)
    cell = pyvoro2.PeriodicCell(
        vectors=((1.0, 0.0, 0.0), (0.35, 1.0, 0.0), (0.2, -0.1, 1.0))
    )
    point = np.array([0.2, 0.3, 0.4])
    points = np.vstack((point, point + np.asarray(cell.vectors[1])))
    with pytest.raises(DuplicateError) as caught:
        _invoke(
            3,
            'compute',
            points,
            domain=cell,
            mode=mode,
            duplicate_check='raise',
            duplicate_threshold=5e-6,
            duplicate_wrap=False,
        )
    assert caught.value.kind == 'backend_safety'
    assert caught.value.minimum_image_used is True
    assert core.calls == []


@pytest.mark.parametrize('dim', [2, 3])
@pytest.mark.parametrize('mode', ['off', 'warn', 'raise'])
def test_optional_safe_above_floor_policy(
    monkeypatch,
    dim: int,
    mode: str,
) -> None:
    core = RecordingSafeCore()
    _install_core(monkeypatch, dim, core)
    left = np.full(dim, 0.25)
    right = left.copy()
    right[0] += 1e-4
    points = np.vstack((left, right))
    kwargs = dict(
        duplicate_check=mode,
        duplicate_threshold=1e-3,
        duplicate_wrap=True,
    )
    if mode == 'warn':
        with pytest.warns(RuntimeWarning, match='user threshold'):
            _invoke(3 if dim == 3 else 2, 'compute', points, domain=_box(dim), **kwargs)
    elif mode == 'raise':
        with pytest.raises(DuplicateError) as caught:
            _invoke(dim, 'compute', points, domain=_box(dim), **kwargs)
        error = caught.value
        assert error.kind == 'user_threshold'
        assert error.threshold == 1e-3
        assert error.user_threshold == 1e-3
        assert error.optional_wrap_used is False
    else:
        with warnings.catch_warnings(record=True) as caught_warnings:
            warnings.simplefilter('always')
            _invoke(dim, 'compute', points, domain=_box(dim), **kwargs)
        assert caught_warnings == []
    assert len(core.calls) == (1 if mode != 'raise' else 0)


def test_optional_threshold_comparison_remains_strict(monkeypatch) -> None:
    core = RecordingSafeCore()
    _install_core(monkeypatch, 2, core)
    points = np.array([[0.25, 0.25], [0.251, 0.25]])
    _invoke(
        2,
        'compute',
        points,
        domain=_box(2),
        duplicate_check='raise',
        duplicate_threshold=1e-3,
    )
    assert len(core.calls) == 1


@pytest.mark.parametrize('dim', [2, 3])
@pytest.mark.parametrize('mode', ['standard', 'power'])
def test_ghost_temporary_generator_mandatory_safety(
    monkeypatch,
    dim: int,
    mode: str,
) -> None:
    core = RecordingSafeCore()
    _install_core(monkeypatch, dim, core)
    point = np.full((1, dim), 0.25)
    with pytest.raises(DuplicateError) as caught:
        _invoke(
            dim,
            'ghost_cells',
            point,
            domain=_box(dim),
            queries=point.copy(),
            mode=mode,
            duplicate_check='off',
        )
    assert caught.value.kind == 'backend_safety'
    assert caught.value.operation == 'ghost_cells'
    assert core.calls == []


@pytest.mark.parametrize('dim', [2, 3])
@pytest.mark.parametrize('mode', ['standard', 'power'])
def test_valid_distinct_ghost_reaches_native(
    monkeypatch,
    dim: int,
    mode: str,
) -> None:
    core = RecordingSafeCore()
    _install_core(monkeypatch, dim, core)
    point = np.full((1, dim), 0.25)
    query = np.full((1, dim), 0.75)
    _invoke(
        dim,
        'ghost_cells',
        point,
        domain=_box(dim),
        queries=query,
        mode=mode,
    )
    assert len(core.calls) == 1


def test_periodically_equivalent_ghost_ignores_optional_wrap(monkeypatch) -> None:
    core = RecordingSafeCore()
    _install_core(monkeypatch, 3, core)
    point = np.array([[0.25, 0.25, 0.25]])
    query = np.array([[1.25, 0.25, 0.25]])
    with pytest.raises(DuplicateError) as caught:
        _invoke(
            3,
            'ghost_cells',
            point,
            domain=_partial_cell(3),
            queries=query,
            duplicate_check='warn',
            duplicate_threshold=5e-6,
            duplicate_wrap=False,
        )
    assert caught.value.minimum_image_used is True
    assert caught.value.optional_wrap_used is False
    assert core.calls == []


def test_mandatory_scan_reports_truthful_truncation(monkeypatch) -> None:
    core = RecordingSafeCore()
    _install_core(monkeypatch, 2, core)
    points = np.full((4, 2), 0.25)
    with pytest.raises(DuplicateError) as caught:
        _invoke(
            2,
            'compute',
            points,
            domain=_box(2),
            duplicate_max_pairs=2,
        )
    assert caught.value.truncated is True
    assert len(caught.value.pairs) == 2
    assert 'at least 2' in str(caught.value)


@pytest.mark.parametrize(
    ('mode', 'rows', 'message'),
    [
        ('standard', [{'id': 0}], 'every internal ID'),
        ('standard', [{'id': 0}, {'id': 0}], 'duplicate internal ID'),
        ('standard', [{'id': 0}, {'id': 2}], 'out-of-range'),
        ('standard', [{'id': 0}, {'id': True}], 'malformed'),
        ('power', [{'id': 1}, {'id': 1}], 'duplicate internal ID'),
        ('power', [{'id': -1}], 'out-of-range'),
    ],
)
def test_compute_rejects_malformed_raw_native_ids_before_packaging(
    monkeypatch,
    mode: str,
    rows: list[dict[str, object]],
    message: str,
) -> None:
    core = RecordingSafeCore(compute_rows=rows)
    _install_core(monkeypatch, 3, core)
    points = np.array([[0.25, 0.25, 0.25], [0.75, 0.75, 0.75]])
    with pytest.raises(RuntimeError, match=message):
        _invoke(3, 'compute', points, domain=_box(3), mode=mode)


def test_power_compute_accepts_unique_hidden_cell_subset(monkeypatch) -> None:
    core = RecordingSafeCore(compute_rows=[{'id': 1}])
    _install_core(monkeypatch, 3, core)
    points = np.array([[0.25, 0.25, 0.25], [0.75, 0.75, 0.75]])
    rows = _invoke(3, 'compute', points, domain=_box(3), mode='power')
    assert [row['id'] for row in rows] == [1]


def _brute_rectangular_pairs(
    points: np.ndarray,
    *,
    radius: float,
    bounds: tuple[tuple[float, float], ...],
    periodic: tuple[bool, ...],
) -> set[tuple[int, int]]:
    found: set[tuple[int, int]] = set()
    for i, j in combinations(range(len(points)), 2):
        delta = np.abs(points[i] - points[j])
        for axis, wraps in enumerate(periodic):
            if wraps:
                span = bounds[axis][1] - bounds[axis][0]
                delta[axis] = min(delta[axis], span - delta[axis])
        if math.sqrt(math.fsum(float(value) ** 2 for value in delta)) < radius:
            found.add((i, j))
    return found


@pytest.mark.parametrize('dim', [2, 3])
@pytest.mark.parametrize('radius', [0.04, 0.18, 1.25])
def test_rectangular_candidate_scanner_matches_independent_brute_force(
    dim: int,
    radius: float,
) -> None:
    bounds = tuple((0.0, 1.0) for _ in range(dim))
    periodic = (True,) + (False,) * (dim - 1)
    domain = _partial_cell(dim)
    geometry = geometry3d(domain) if dim == 3 else geometry2d(domain)
    points = np.array(
        [
            [0.01, 0.02, 0.03],
            [0.99, 0.02, 0.03],
            [0.03, 0.17, 0.03],
            [0.62, 0.61, 0.61],
            [0.97, 0.95, 0.95],
        ],
        dtype=np.float64,
    )[:, :dim]
    expected = _brute_rectangular_pairs(
        points,
        radius=radius,
        bounds=bounds,
        periodic=periodic,
    )
    scan = scan_close_pairs(
        points,
        radius=radius,
        geometry=geometry,
        max_pairs=100,
    )
    actual = {(i, j) for i, j, _distance in scan.pairs}
    assert actual == expected
    candidates = list(candidate_pairs(points, radius=radius, geometry=geometry))
    assert len(candidates) == len(set(candidates))
    assert expected <= set(candidates)


def _brute_triclinic_pairs(
    points: np.ndarray,
    lattice: np.ndarray,
    radius: float,
) -> set[tuple[int, int]]:
    found: set[tuple[int, int]] = set()
    for i, j in combinations(range(len(points)), 2):
        best_distance = math.inf
        best_shift = None
        for shift in product(range(-2, 3), repeat=3):
            displacement = points[i] - points[j] - np.asarray(shift) @ lattice
            distance = float(np.linalg.norm(displacement))
            if distance < best_distance:
                best_distance = distance
                best_shift = shift
        assert best_shift is not None
        assert all(abs(value) < 2 for value in best_shift)
        if best_distance < radius:
            found.add((i, j))
    return found


def _fraction_inverse_3x3(
    lattice: np.ndarray,
) -> tuple[tuple[Fraction, ...], ...]:
    """Independent exact source-binary64 inverse for R5-SC-001 tests."""

    matrix = tuple(
        tuple(Fraction.from_float(float(value)) for value in row)
        for row in lattice
    )
    a, b, c = matrix
    determinant = (
        a[0] * (b[1] * c[2] - b[2] * c[1])
        - a[1] * (b[0] * c[2] - b[2] * c[0])
        + a[2] * (b[0] * c[1] - b[1] * c[0])
    )
    assert determinant > 0

    def cofactor(row: int, column: int) -> Fraction:
        rows = [index for index in range(3) if index != row]
        columns = [index for index in range(3) if index != column]
        minor = (
            matrix[rows[0]][columns[0]]
            * matrix[rows[1]][columns[1]]
            - matrix[rows[0]][columns[1]]
            * matrix[rows[1]][columns[0]]
        )
        return minor if (row + column) % 2 == 0 else -minor

    return tuple(
        tuple(cofactor(column, row) / determinant for column in range(3))
        for row in range(3)
    )


def _fraction_bucket_oracle(
    points: np.ndarray,
    *,
    origin: np.ndarray,
    lattice: np.ndarray,
    radius: float,
) -> tuple[
    tuple[tuple[int, ...], ...],
    tuple[int, ...],
    tuple[Fraction, ...],
]:
    inverse = _fraction_inverse_3x3(lattice)
    radius_fraction = Fraction.from_float(float(radius))
    bounds = tuple(
        radius_fraction
        * sum((abs(inverse[row][axis]) for row in range(3)), Fraction())
        for axis in range(3)
    )
    bins = tuple(
        1 if bound >= 1 else bound.denominator // bound.numerator
        for bound in bounds
    )
    exact_origin = tuple(
        Fraction.from_float(float(value)) for value in origin
    )
    keys = []
    for point in points:
        delta = tuple(
            Fraction.from_float(float(point[axis])) - exact_origin[axis]
            for axis in range(3)
        )
        coordinates = tuple(
            sum(
                (delta[row] * inverse[row][axis] for row in range(3)),
                Fraction(),
            )
            for axis in range(3)
        )
        wrapped = tuple(value - math.floor(value) for value in coordinates)
        keys.append(
            tuple(math.floor(wrapped[axis] * bins[axis]) for axis in range(3))
        )
    return tuple(keys), bins, bounds


def _fraction_minimum_geometry(
    left: np.ndarray,
    right: np.ndarray,
    lattice: np.ndarray,
    *,
    shift_radius: int = 3,
) -> tuple[Fraction, tuple[tuple[int, ...], ...]]:
    """Fixed-cube Fraction oracle independent of R4 and candidate code."""

    delta = tuple(
        Fraction.from_float(float(right[axis]))
        - Fraction.from_float(float(left[axis]))
        for axis in range(3)
    )
    exact_lattice = tuple(
        tuple(Fraction.from_float(float(value)) for value in row)
        for row in lattice
    )
    best: Fraction | None = None
    minimizers: list[tuple[int, ...]] = []
    for shift in product(
        range(-shift_radius, shift_radius + 1),
        repeat=3,
    ):
        displacement = tuple(
            delta[column]
            + sum(
                (
                    shift[row] * exact_lattice[row][column]
                    for row in range(3)
                ),
                Fraction(),
            )
            for column in range(3)
        )
        distance_squared = sum(
            (value * value for value in displacement),
            Fraction(),
        )
        if best is None or distance_squared < best:
            best = distance_squared
            minimizers = [shift]
        elif distance_squared == best:
            minimizers.append(shift)
    assert best is not None
    assert all(
        all(abs(value) < shift_radius for value in shift)
        for shift in minimizers
    )
    return best, tuple(minimizers)


def _fraction_close_pairs(
    left: np.ndarray,
    right: np.ndarray,
    lattice: np.ndarray,
    radius: float,
    *,
    within: bool,
) -> set[tuple[int, int]]:
    threshold_squared = Fraction.from_float(float(radius)) ** 2
    pairs = combinations(range(len(left)), 2) if within else product(
        range(len(left)), range(len(right))
    )
    return {
        (i, j)
        for i, j in pairs
        if _fraction_minimum_geometry(
            left[i],
            right[j] if not within else left[j],
            lattice,
        )[0] < threshold_squared
    }


def test_r5_sc_001_review_witness_uses_exact_bound_bins_and_keys() -> None:
    lattice = np.array(
        [
            [0.05185516747129633, 0.0, 0.0],
            [1721.8706815591584, 2.1100027038765874, 0.0],
            [4310.293148429306, -3510.8020392421427, 5986.1025926131315],
        ],
        dtype=np.float64,
    )
    origin = np.array([0.125, -0.25, 0.5], dtype=np.float64)
    points = np.array(
        [
            origin,
            [0.1, 0.2, 0.3],
            [-10.0, 20.0, -30.0],
            np.nextafter(origin, np.inf),
        ],
        dtype=np.float64,
    )
    expected_keys, expected_bins, expected_bounds = _fraction_bucket_oracle(
        points,
        origin=origin,
        lattice=lattice,
        radius=BACKEND_SAFETY_DISTANCE,
    )

    layout = _exact_triclinic_bucket_layout(
        points,
        origin=origin,
        lattice_vectors=lattice,
        radius=BACKEND_SAFETY_DISTANCE,
    )

    assert layout.coefficient_bounds == expected_bounds
    assert layout.bins == expected_bins
    assert layout.keys == expected_keys
    assert expected_bounds[0] > Fraction(1, 4)
    assert expected_bins[0] == 3
    assert (
        Fraction.from_float(BACKEND_SAFETY_DISTANCE) ** 2
        > Fraction.from_float(BACKEND_SAFETY_DISTANCE_SQUARED)
    )


@pytest.mark.parametrize(
    ('center', 'expected_bins'),
    [
        (0.5, (2, 2, 1)),
        (1.0 / 3.0, (3, 3, 2)),
        (0.25, (4, 4, 3)),
    ],
)
def test_r5_sc_001_exact_bins_straddle_reciprocal_boundaries(
    center: float,
    expected_bins: tuple[int, int, int],
) -> None:
    radii = (
        float(np.nextafter(center, 0.0)),
        center,
        float(np.nextafter(center, np.inf)),
    )
    actual = []
    for radius in radii:
        layout = _exact_triclinic_bucket_layout(
            np.zeros((0, 3)),
            origin=np.zeros(3),
            lattice_vectors=np.eye(3),
            radius=radius,
        )
        actual.append(layout.bins[0])
        exact_radius = Fraction.from_float(radius)
        assert layout.coefficient_bounds[0] == exact_radius
        assert layout.bins[0] == max(
            1,
            exact_radius.denominator // exact_radius.numerator,
        )
    assert tuple(actual) == expected_bins


def test_r5_sc_001_exact_keys_cover_boundaries_seams_and_multiple_axes() -> None:
    below = np.nextafter(np.array([0.25, 0.5, 0.75]), 0.0)
    exact = np.array([0.25, 0.5, 0.75])
    above = np.nextafter(exact, np.inf)
    subnormal = np.nextafter(0.0, np.inf)
    points = np.vstack(
        (
            below,
            exact,
            above,
            np.zeros(3),
            np.full(3, subnormal),
            np.full(3, np.nextafter(1.0, 0.0)),
            np.full(3, -subnormal),
            np.ones(3),
        )
    )
    expected_keys, expected_bins, _bounds = _fraction_bucket_oracle(
        points,
        origin=np.zeros(3),
        lattice=np.eye(3),
        radius=0.25,
    )
    layout = _exact_triclinic_bucket_layout(
        points,
        origin=np.zeros(3),
        lattice_vectors=np.eye(3),
        radius=0.25,
    )

    assert expected_bins == (4, 4, 4)
    assert layout.keys == expected_keys
    assert layout.keys == (
        (0, 1, 2),
        (1, 2, 3),
        (1, 2, 3),
        (0, 0, 0),
        (0, 0, 0),
        (3, 3, 3),
        (3, 3, 3),
        (0, 0, 0),
    )


def test_r5_sc_001_smallest_subnormal_radius_finds_exact_periodic_duplicates(
) -> None:
    radius = float(np.nextafter(0.0, np.inf))
    cell = pyvoro2.PeriodicCell(
        ((1.0, 0.0, 0.0), (0.7, 1.0, 0.0), (0.2, -0.3, 1.0))
    )
    geometry = geometry3d(cell)
    lattice = np.asarray(cell.vectors, dtype=np.float64)
    points = np.vstack((np.zeros(3), lattice[0]))

    assert list(candidate_pairs(points, radius=radius, geometry=geometry)) == [
        (0, 1)
    ]
    within = scan_close_pairs(
        points,
        radius=radius,
        geometry=geometry,
        max_pairs=10,
    )
    cross = scan_cross_close_pairs(
        points[:1],
        points[1:],
        radius=radius,
        geometry=geometry,
        max_pairs=10,
    )
    assert [(i, j) for i, j, _distance in within.pairs] == [(0, 1)]
    assert [(i, j) for i, j, _distance in cross.pairs] == [(0, 0)]


def test_r5_sc_001_within_and_cross_scans_match_fraction_oracle() -> None:
    cell = pyvoro2.PeriodicCell(
        ((1.0, 0.0, 0.0), (0.72, 0.8, 0.0), (0.41, -0.28, 0.9))
    )
    geometry = geometry3d(cell)
    lattice = np.asarray(cell.vectors, dtype=np.float64)
    fractional = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.999, 0.001, 0.0],
            [0.25, 0.25, 0.25],
            [0.26, 0.24, 0.25],
            [0.55, 0.61, 0.42],
            [0.85, 0.15, 0.91],
        ]
    )
    points = fractional @ lattice
    radius = 0.03
    expected_within = _fraction_close_pairs(
        points,
        points,
        lattice,
        radius,
        within=True,
    )
    within_candidates = set(
        candidate_pairs(points, radius=radius, geometry=geometry)
    )
    within = scan_close_pairs(
        points,
        radius=radius,
        geometry=geometry,
        max_pairs=100,
    )
    assert expected_within <= within_candidates
    assert {(i, j) for i, j, _distance in within.pairs} == expected_within

    reference = points[::2]
    queries = points[1::2]
    expected_cross = _fraction_close_pairs(
        reference,
        queries,
        lattice,
        radius,
        within=False,
    )
    cross_candidates = set(
        cross_candidate_pairs(
            reference,
            queries,
            radius=radius,
            geometry=geometry,
        )
    )
    cross = scan_cross_close_pairs(
        reference,
        queries,
        radius=radius,
        geometry=geometry,
        max_pairs=100,
    )
    assert expected_cross <= cross_candidates
    assert {(i, j) for i, j, _distance in cross.pairs} == expected_cross


def test_r5_sc_001_nontrivial_image_candidate_reaches_r4_classifier() -> None:
    cell = pyvoro2.PeriodicCell.from_params(1, 1.5, 1, 0, 0, 1)
    geometry = geometry3d(cell)
    lattice = np.asarray(cell.vectors, dtype=np.float64)
    points = np.array(
        [
            [0.63696169, 0.26978671, 0.04097352],
            [0.01652764, 0.81327024, 0.91275558],
        ]
    )
    exact_distance, minimizers = _fraction_minimum_geometry(
        points[0],
        points[1],
        lattice,
    )
    assert minimizers == ((2, -1, -1),)
    assert exact_distance < Fraction(1, 4)
    assert list(
        candidate_pairs(points, radius=0.5, geometry=geometry)
    ) == [(0, 1)]
    scan = scan_close_pairs(
        points,
        radius=0.5,
        geometry=geometry,
        max_pairs=10,
    )
    assert [(i, j) for i, j, _distance in scan.pairs] == [(0, 1)]


def test_r5_sc_001_fixed_seed_clouds_match_fraction_oracle() -> None:
    rng = np.random.default_rng(20260813)
    radius = 0.04
    checked_pairs = 0
    checked_cross_pairs = 0
    exact_close_pairs = 0
    exact_cross_close_pairs = 0
    for _case in range(6):
        lattice = np.array(
            [
                [rng.uniform(0.8, 1.2), 0.0, 0.0],
                [rng.uniform(-0.8, 0.8), rng.uniform(0.8, 1.2), 0.0],
                [
                    rng.uniform(-0.8, 0.8),
                    rng.uniform(-0.8, 0.8),
                    rng.uniform(0.8, 1.2),
                ],
            ],
            dtype=np.float64,
        )
        cell = pyvoro2.PeriodicCell(lattice)
        geometry = geometry3d(cell)
        lattice = np.asarray(cell.vectors, dtype=np.float64)
        fractional = rng.uniform(0.1, 0.9, size=(5, 3))
        points = fractional @ lattice
        points[1] = points[0] + rng.uniform(-0.01, 0.01, size=3)
        points[3] = points[2] + lattice[0] + rng.uniform(
            -0.01,
            0.01,
            size=3,
        )

        expected = _fraction_close_pairs(
            points,
            points,
            lattice,
            radius,
            within=True,
        )
        candidates = set(
            candidate_pairs(points, radius=radius, geometry=geometry)
        )
        scan = scan_close_pairs(
            points,
            radius=radius,
            geometry=geometry,
            max_pairs=100,
        )
        actual = {(i, j) for i, j, _distance in scan.pairs}
        assert expected <= candidates
        assert actual == expected

        reference = points[::2]
        queries = points[1::2]
        expected_cross = _fraction_close_pairs(
            reference,
            queries,
            lattice,
            radius,
            within=False,
        )
        cross_candidates = set(
            cross_candidate_pairs(
                reference,
                queries,
                radius=radius,
                geometry=geometry,
            )
        )
        cross = scan_cross_close_pairs(
            reference,
            queries,
            radius=radius,
            geometry=geometry,
            max_pairs=100,
        )
        assert expected_cross <= cross_candidates
        assert {
            (i, j) for i, j, _distance in cross.pairs
        } == expected_cross
        checked_pairs += len(points) * (len(points) - 1) // 2
        checked_cross_pairs += len(reference) * len(queries)
        exact_close_pairs += len(expected)
        exact_cross_close_pairs += len(expected_cross)

    assert checked_pairs == 60
    assert checked_cross_pairs == 36
    assert exact_close_pairs >= 12
    assert exact_cross_close_pairs >= 12


def test_r5_sc_001_warning_conditioned_cell_candidates_remain_complete() -> None:
    lattice = np.array(
        [[1.0, 0.0, 0.0], [1.0, 1e-11, 0.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    with pytest.warns(RuntimeWarning, match='very ill-conditioned'):
        cell = pyvoro2.PeriodicCell(lattice)
    geometry = geometry3d(cell)
    lattice = np.asarray(cell.vectors, dtype=np.float64)
    points = np.array([[0.0, 0.0, 0.0], [0.0, 2.5e-12, 0.0]])
    expected = _fraction_close_pairs(
        points,
        points,
        lattice,
        BACKEND_SAFETY_DISTANCE,
        within=True,
    )
    candidates = set(
        candidate_pairs(
            points,
            radius=BACKEND_SAFETY_DISTANCE,
            geometry=geometry,
        )
    )
    scan = scan_close_pairs(
        points,
        radius=BACKEND_SAFETY_DISTANCE,
        geometry=geometry,
        inclusive_squared=BACKEND_SAFETY_DISTANCE_SQUARED,
        max_pairs=10,
    )
    assert expected == {(0, 1)}
    assert expected <= candidates
    assert {(i, j) for i, j, _distance in scan.pairs} == expected


def test_r5_sc_001_triclinic_sparse_locality_and_basis_cache_reuse() -> None:
    cell = pyvoro2.PeriodicCell(
        ((1.0, 0.0, 0.0), (0.35, 1.0, 0.0), (0.2, -0.1, 1.0))
    )
    geometry = geometry3d(cell)
    lattice = np.asarray(cell.vectors, dtype=np.float64)
    axis = np.linspace(0.01, 0.99, 8)
    points = np.array(list(product(axis, repeat=3))) @ lattice

    _basis_cache_clear()
    within_count = candidate_pair_count(
        points,
        radius=1e-4,
        geometry=geometry,
        limit=10_000,
    )
    after_within = _basis_cache_info()
    cross_count = len(
        list(
            cross_candidate_pairs(
                points[::2],
                points[1::2],
                radius=1e-4,
                geometry=geometry,
            )
        )
    )
    after_cross = _basis_cache_info()

    assert len(points) == 512
    assert within_count < 8 * len(points)
    assert cross_count < 8 * len(points)
    assert after_within.misses == 1
    assert after_cross.misses == 1
    assert after_cross.hits == after_within.hits + 1


@pytest.mark.parametrize('radius', [0.08, 0.32, 1.4])
def test_triclinic_candidate_scanner_matches_exhaustive_lattice_oracle(
    radius: float,
) -> None:
    cell = pyvoro2.PeriodicCell(
        vectors=((1.0, 0.0, 0.0), (0.72, 0.8, 0.0), (0.41, -0.28, 0.9))
    )
    geometry = geometry3d(cell)
    lattice = np.asarray(cell.vectors, dtype=np.float64)
    fractional = np.array(
        [
            [0.01, 0.02, 0.03],
            [0.98, 0.02, 0.03],
            [0.04, 0.91, 0.05],
            [0.51, 0.47, 0.44],
            [0.88, 0.84, 0.93],
        ]
    )
    points = fractional @ lattice
    expected = _brute_triclinic_pairs(points, lattice, radius)
    scan = scan_close_pairs(
        points,
        radius=radius,
        geometry=geometry,
        max_pairs=100,
    )
    actual = {(i, j) for i, j, _distance in scan.pairs}
    assert actual == expected
    candidates = list(candidate_pairs(points, radius=radius, geometry=geometry))
    assert len(candidates) == len(set(candidates))
    assert expected <= set(candidates)


def test_well_separated_large_cloud_has_local_candidate_work() -> None:
    axes = np.linspace(0.001, 0.999, 20)
    points = np.array(list(product(axes, repeat=3)), dtype=np.float64)
    count = candidate_pair_count(points, radius=1e-5)
    assert len(points) == 8000
    assert count == 0


def _large_span_local_cloud() -> tuple[np.ndarray, object]:
    cell = pyvoro2.OrthorhombicCell(
        ((0.0, 200.0),) * 3,
        periodic=(True, False, False),
    )
    axis = 1.000001 + 1.1e-5 * np.arange(17)
    points = np.array(list(product(axis, repeat=3)), dtype=np.float64)
    return points, geometry3d(cell)


def test_sparse_rectangular_scanner_keeps_radius_local_large_span_bins() -> None:
    points, geometry = _large_span_local_cloud()
    count = candidate_pair_count(
        points,
        radius=1e-5,
        geometry=geometry,
        limit=1_000_000,
    )
    assert len(points) == 17**3
    assert count < 1_000_000


def test_large_span_mandatory_scan_completes_without_unsafe_pairs() -> None:
    points, geometry = _large_span_local_cloud()
    scan = scan_close_pairs(
        points,
        radius=1e-5,
        geometry=geometry,
        inclusive_squared=1e-10,
        max_pairs=10,
    )
    assert scan.pairs == ()
    assert scan.candidate_count < 1_000_000


def test_sparse_rectangular_cross_scanner_keeps_large_span_locality() -> None:
    points, geometry = _large_span_local_cloud()
    reference = points[::2]
    queries = points[1::2]
    candidates = list(
        cross_candidate_pairs(
            reference,
            queries,
            radius=1e-5,
            geometry=geometry,
        )
    )
    assert len(candidates) < 1_000_000
    scan = scan_cross_close_pairs(
        reference,
        queries,
        radius=1e-5,
        geometry=geometry,
        inclusive_squared=1e-10,
        max_pairs=10,
    )
    assert scan.pairs == ()
    assert scan.candidate_count == len(candidates)


def test_python_sparse_keys_handle_finite_bin_quotient_overflow() -> None:
    cell = pyvoro2.OrthorhombicCell(
        ((0.0, 1e100),) * 3,
        periodic=(True, False, False),
    )
    points = np.array([[1.0, 1.0, 1.0], [2.0, 2.0, 2.0]])
    assert candidate_pair_count(
        points,
        radius=1e-5,
        geometry=geometry3d(cell),
        limit=10,
    ) == 0


def test_dense_scanner_truncates_without_materializing_all_pairs() -> None:
    points = np.zeros((2000, 2), dtype=np.float64)
    scan = scan_close_pairs(points, radius=1e-3, max_pairs=3)
    assert len(scan.pairs) == 3
    assert scan.truncated is True
    assert scan.candidate_count < len(points) * (len(points) - 1) // 2


def test_prepared_arrays_are_owned_read_only_and_keep_provenance() -> None:
    domain = planar.RectangularCell(
        ((0.0, 1.0), (0.0, 1.0)),
        periodic=(True, False),
    )
    points = np.array([[2.25, 0.5]])
    prepared = prepare_generators(
        points,
        geometry=geometry2d(domain),
        operation='compute',
        external_ids=np.array([42]),
        backend_radii=None,
        duplicate_check='off',
        duplicate_threshold=1e-5,
        duplicate_wrap=False,
        duplicate_max_pairs=10,
    )
    points[:] = 0.0
    assert prepared.input_points_cart[0, 0] == 2.25
    assert prepared.primary_points_cart[0, 0] == 0.25
    assert prepared.remap_shifts[0, 0] == 2
    assert prepared.internal_ids.tolist() == [0]
    assert prepared.external_ids.tolist() == [42]
    for value in (
        prepared.input_points_cart,
        prepared.primary_points_cart,
        prepared.native_points,
        prepared.remap_shifts,
        prepared.internal_ids,
        prepared.external_ids,
    ):
        assert value.flags.owndata
        assert value.flags.writeable is False
