from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace
import warnings

import numpy as np
import pytest

import pyvoro2
from pyvoro2 import planar


class _Truthy:
    def __bool__(self) -> bool:
        return True


class _ExplodingCells:
    def __iter__(self):
        raise AssertionError('cell analysis must not run for invalid expected_ids')


class _ExplodingArray:
    def __array__(self, dtype=None, copy=None):
        raise AssertionError('array coercion must not run for an invalid string mode')


class EqualAny:
    def __init__(self) -> None:
        self.comparisons = 0

    def __eq__(self, other: object) -> bool:
        self.comparisons += 1
        return True

    def __ne__(self, other: object) -> bool:
        self.comparisons += 1
        return False


_STRING_IMPOSTOR_FACTORIES = (
    pytest.param(lambda value: np.array(value), id='zero_dim_array'),
    pytest.param(lambda value: np.array([value]), id='one_element_array'),
    pytest.param(
        lambda value: np.array([value, value]),
        id='multi_element_array',
    ),
    pytest.param(
        lambda value: np.array(value, dtype=object),
        id='object_array',
    ),
    pytest.param(lambda value: value.encode(), id='bytes'),
    pytest.param(lambda value: bytearray(value.encode()), id='bytearray'),
    pytest.param(lambda value: 1, id='integer'),
    pytest.param(lambda value: True, id='boolean'),
    pytest.param(lambda value: None, id='none'),
    pytest.param(lambda value: EqualAny(), id='equal_any'),
)


_NON_NONE_STRING_IMPOSTOR_FACTORIES = _STRING_IMPOSTOR_FACTORIES[:-2] + (
    _STRING_IMPOSTOR_FACTORIES[-1],
)


def _normalization_case(
    dim: int,
    *,
    periodic: bool,
) -> tuple[object, list[dict], object, str]:
    if dim == 3:
        api = pyvoro2
        points = np.array([[0.5, 0.5, 0.5], [1.5, 0.5, 0.5]])
        domain = (
            pyvoro2.OrthorhombicCell(
                ((0, 2), (0, 1), (0, 1)),
                periodic=(True, True, True),
            )
            if periodic
            else pyvoro2.Box(((0, 2), (0, 1), (0, 1)))
        )
        cells = api.compute(
            points,
            domain=domain,
            output='cells',
            return_vertices=True,
            return_faces=True,
            return_face_shifts=periodic,
        )
        return api, cells, domain, 'faces'

    api = planar
    points = np.array([[0.5, 0.5], [1.5, 0.5]])
    domain = (
        planar.RectangularCell(
            ((0, 2), (0, 1)),
            periodic=(True, True),
        )
        if periodic
        else planar.Box(((0, 2), (0, 1)))
    )
    cells = api.compute(
        points,
        domain=domain,
        output='cells',
        return_vertices=True,
        return_edges=True,
        return_edge_shifts=periodic,
    )
    return api, cells, domain, 'edges'


def _first_boundary_with_vertices(
    cells: list[dict],
    boundary_key: str,
) -> dict:
    return next(
        boundary
        for cell in cells
        for boundary in cell[boundary_key]
        if boundary.get('vertices')
    )


@pytest.mark.parametrize(
    'ids',
    [
        [0, 1],
        [np.int32(10), np.int64(20)],
        [np.uint32(10), np.uint64(20)],
    ],
)
def test_forward_ids_accept_exact_integer_scalars(ids: list[object]) -> None:
    cells = pyvoro2.compute(
        np.array([[0.25, 0.5, 0.5], [1.75, 0.5, 0.5]]),
        domain=pyvoro2.Box(((0, 2), (0, 1), (0, 1))),
        ids=ids,
        output='cells',
    )

    assert [cell['id'] for cell in cells] == [int(value) for value in ids]


@pytest.mark.parametrize(
    'invalid',
    [True, np.bool_(False), 1.0, 1.5, '1', 1 + 0j, np.array(1), -1, 2**63],
)
def test_forward_ids_reject_lossy_or_out_of_range_values(invalid: object) -> None:
    with pytest.raises(ValueError, match=r'ids\[0\]'):
        pyvoro2.compute(
            np.array([[0.25, 0.5, 0.5], [1.75, 0.5, 0.5]]),
            domain=pyvoro2.Box(((0, 2), (0, 1), (0, 1))),
            ids=[invalid, 1],
        )


@pytest.mark.parametrize(
    ('analyze', 'domain'),
    [
        (
            pyvoro2.analyze_tessellation,
            pyvoro2.Box(((0, 1), (0, 1), (0, 1))),
        ),
        (planar.analyze_tessellation, planar.Box(((0, 1), (0, 1)))),
    ],
)
def test_diagnostic_expected_ids_accept_exact_external_ids(
    analyze: Callable[..., object],
    domain: object,
) -> None:
    diagnostics = analyze(
        [],
        domain,
        expected_ids=[0, np.int32(2), np.uint64(4)],
    )

    assert diagnostics.n_sites_expected == 3
    assert diagnostics.missing_ids == (0, 2, 4)


@pytest.mark.parametrize(
    'expected_ids',
    [
        [True],
        [np.bool_(False)],
        [1.0],
        [1.5],
        ['1'],
        [1 + 0j],
        [np.array(1)],
        np.array(1),
        [-1],
        [2**63],
        [1, 1],
    ],
)
@pytest.mark.parametrize(
    ('analyze', 'domain'),
    [
        (
            pyvoro2.analyze_tessellation,
            pyvoro2.Box(((0, 1), (0, 1), (0, 1))),
        ),
        (planar.analyze_tessellation, planar.Box(((0, 1), (0, 1)))),
    ],
)
def test_diagnostic_expected_ids_reject_before_cell_analysis(
    analyze: Callable[..., object],
    domain: object,
    expected_ids: object,
) -> None:
    with pytest.raises(ValueError, match='expected_ids'):
        analyze(_ExplodingCells(), domain, expected_ids=expected_ids)


@pytest.mark.parametrize('value', [0, np.int32(1), np.uint64(2)])
def test_nonnegative_search_counts_accept_exact_integers(value: object) -> None:
    result = planar.compute(
        np.array([[0.25, 0.5], [1.75, 0.5]]),
        domain=planar.Box(((0, 2), (0, 1))),
        edge_shift_search=value,
    )
    assert result.sites.shape[0] == 2


@pytest.mark.parametrize(
    'invalid',
    [True, np.bool_(False), 1.0, 1.5, '1', 1 + 0j, np.array(1), -1],
)
def test_nonnegative_search_counts_reject_non_indices(invalid: object) -> None:
    with pytest.raises(ValueError, match='edge_shift_search'):
        planar.compute(
            np.array([[0.25, 0.5], [1.75, 0.5]]),
            domain=planar.Box(((0, 2), (0, 1))),
            edge_shift_search=invalid,
        )


@pytest.mark.parametrize('value', [1, np.int32(2), np.uint64(3)])
def test_duplicate_pair_limit_accepts_positive_exact_integers(
    value: object,
) -> None:
    assert pyvoro2.duplicate_check(
        np.array([[0.0, 0.0, 0.0]]),
        mode='return',
        max_pairs=value,
    ) == ()


@pytest.mark.parametrize(
    'invalid',
    [
        True,
        np.bool_(False),
        1.0,
        1.5,
        '1',
        1 + 0j,
        np.array(1),
        0,
        -1,
    ],
)
def test_duplicate_pair_limit_rejects_non_positive_indices(
    invalid: object,
) -> None:
    with pytest.raises(ValueError, match='max_pairs'):
        pyvoro2.duplicate_check(
            np.array([[0.0, 0.0, 0.0]]),
            mode='return',
            max_pairs=invalid,
        )


@pytest.mark.parametrize('value', [True, False, np.bool_(True), np.bool_(False)])
def test_public_booleans_accept_only_boolean_scalars(value: object) -> None:
    result = planar.compute(
        np.array([[0.25, 0.5], [1.75, 0.5]]),
        domain=planar.Box(((0, 2), (0, 1))),
        return_vertices=value,
    )
    assert result.sites.shape[0] == 2


@pytest.mark.parametrize('invalid', [0, 1, 'true', '', _Truthy()])
def test_public_booleans_reject_truthy_and_falsy_non_booleans(
    invalid: object,
) -> None:
    with pytest.raises(ValueError, match='return_vertices.*Boolean'):
        planar.compute(
            np.array([[0.25, 0.5], [1.75, 0.5]]),
            domain=planar.Box(((0, 2), (0, 1))),
            return_vertices=invalid,
        )


@pytest.mark.parametrize(
    ('domain_type', 'bounds', 'periodic'),
    [
        (
            pyvoro2.OrthorhombicCell,
            [[0, 2], [-1, 1], [0, 3]],
            [True, False, np.bool_(True)],
        ),
        (
            planar.RectangularCell,
            [[0, 2], [-1, 1]],
            [True, np.bool_(False)],
        ),
    ],
)
def test_rectangular_domains_own_bounds_and_periodicity(
    domain_type: Callable[..., object],
    bounds: list[list[object]],
    periodic: list[object],
) -> None:
    domain = domain_type(bounds=bounds, periodic=periodic)
    expected_bounds = tuple(tuple(float(value) for value in row) for row in bounds)
    expected_periodic = tuple(bool(value) for value in periodic)

    bounds[0][0] = -100
    periodic[0] = False

    assert domain.bounds == expected_bounds
    assert domain.periodic == expected_periodic
    assert all(type(value) is float for row in domain.bounds for value in row)
    assert all(type(value) is bool for value in domain.periodic)


@pytest.mark.parametrize(
    ('box_type', 'bounds'),
    [
        (pyvoro2.Box, [[0, 2], [-1, 1], [0, 3]]),
        (planar.Box, [[0, 2], [-1, 1]]),
    ],
)
def test_boxes_own_canonical_float_bounds(
    box_type: Callable[..., object],
    bounds: list[list[object]],
) -> None:
    box = box_type(bounds)
    expected = tuple(tuple(float(value) for value in row) for row in bounds)

    bounds[0][1] = 100

    assert box.bounds == expected
    assert all(type(value) is float for row in box.bounds for value in row)


def test_periodic_cell_owns_canonical_vectors_and_origin() -> None:
    vectors = np.array(
        [[2, 0, 0], [0.25, 2, 0], [0.1, -0.2, 2]],
        dtype=np.float64,
    )
    origin = [0, 0.25, -0.5]
    cell = pyvoro2.PeriodicCell(vectors=vectors, origin=origin)

    vectors[0, 0] = 99
    origin[1] = 99

    assert cell.vectors[0] == (2.0, 0.0, 0.0)
    assert cell.origin == (0.0, 0.25, -0.5)
    assert all(type(value) is float for row in cell.vectors for value in row)
    assert all(type(value) is float for value in cell.origin)


@pytest.mark.parametrize(
    ('box_type', 'dim'),
    [(pyvoro2.Box, 3), (planar.Box, 2)],
)
def test_box_from_points_validates_before_reduction(
    box_type: Callable[..., object],
    dim: int,
) -> None:
    with pytest.raises(ValueError, match='at least one point'):
        box_type.from_points(np.empty((0, dim)))
    for invalid in (np.nan, np.inf, -np.inf):
        points = np.zeros((2, dim))
        points[1, 0] = invalid
        with pytest.raises(ValueError, match='finite'):
            box_type.from_points(points)
    with pytest.raises(ValueError, match='real numeric'):
        box_type.from_points([[False] * dim, [1.0] * dim])
    with pytest.raises(ValueError, match='real numeric'):
        box_type.from_points([['0'] * dim, ['1'] * dim])


@pytest.mark.parametrize(
    ('box_type', 'points'),
    [
        (pyvoro2.Box, [[0, 0, 0], [1, 2, 3]]),
        (planar.Box, [[0, 0], [1, 2]]),
    ],
)
@pytest.mark.parametrize('invalid', [-1, np.nan, np.inf, True, '1'])
def test_box_from_points_requires_finite_nonnegative_padding(
    box_type: Callable[..., object],
    points: list[list[int]],
    invalid: object,
) -> None:
    with pytest.raises(ValueError, match='padding'):
        box_type.from_points(points, padding=invalid)


@pytest.mark.parametrize(
    ('box_type', 'points'),
    [
        (pyvoro2.Box, [[0, 0, 0], [1, 2, 3]]),
        (planar.Box, [[0, 0], [1, 2]]),
    ],
)
def test_zero_padding_is_allowed_only_for_ordered_resulting_bounds(
    box_type: Callable[..., object],
    points: list[list[int]],
) -> None:
    box = box_type.from_points(points, padding=0)
    assert all(upper > lower for lower, upper in box.bounds)

    with pytest.raises(ValueError, match='strictly ordered'):
        box_type.from_points([points[0], points[0]], padding=0)


@pytest.mark.parametrize('nonfinite', [np.nan, np.inf, -np.inf])
def test_domain_scalar_nan_inf_matrix(nonfinite: float) -> None:
    with pytest.raises(ValueError, match='bounds.*finite'):
        pyvoro2.Box(((0, 1), (0, nonfinite), (0, 1)))
    with pytest.raises(ValueError, match='vectors.*finite'):
        pyvoro2.PeriodicCell(
            ((1, 0, 0), (0, 1, 0), (0, 0, nonfinite))
        )
    with pytest.raises(ValueError, match='origin.*finite'):
        pyvoro2.PeriodicCell(
            ((1, 0, 0), (0, 1, 0), (0, 0, 1)),
            origin=(0, nonfinite, 0),
        )


def test_periodic_cell_rejects_left_handed_and_accepts_right_handed() -> None:
    with pytest.raises(ValueError, match='right-handed.*determinant > 0'):
        pyvoro2.PeriodicCell(((1, 0, 0), (0, 0, 1), (0, 1, 0)))

    cell = pyvoro2.PeriodicCell(((1, 0, 0), (0, 1, 0), (0, 0, 1)))
    assert np.linalg.det(np.asarray(cell.vectors)) > 0


@pytest.mark.parametrize(
    ('field', 'value'),
    [
        ('bx', 0.0),
        ('by', -1.0),
        ('bz', np.inf),
        ('bxy', np.nan),
        ('bxz', True),
        ('byz', '0'),
    ],
)
def test_periodic_cell_from_params_validates_all_parameters_first(
    field: str,
    value: object,
) -> None:
    kwargs = {
        'bx': 1.0,
        'bxy': 0.0,
        'by': 1.0,
        'bxz': 0.0,
        'byz': 0.0,
        'bz': 1.0,
    }
    kwargs[field] = value
    with pytest.raises(ValueError, match=field):
        pyvoro2.PeriodicCell.from_params(**kwargs)


@pytest.mark.parametrize('invalid', [0, 1, 'true', _Truthy()])
def test_rectangular_periodicity_requires_exact_booleans(invalid: object) -> None:
    with pytest.raises(ValueError, match=r'periodic\[0\].*Boolean'):
        planar.RectangularCell(((0, 1), (0, 1)), periodic=(invalid, True))


@pytest.mark.parametrize(
    'cell',
    [
        pyvoro2.OrthorhombicCell(
            ((0, 1), (0, 1), (0, 1)),
            periodic=(True, True, True),
        ),
        planar.RectangularCell(
            ((0, 1), (0, 1)),
            periodic=(True, True),
        ),
        pyvoro2.PeriodicCell(((1, 0, 0), (0.2, 1, 0), (0.1, 0.3, 1))),
    ],
)
def test_remap_rejects_nonfinite_inputs_eps_and_non_boolean_flag(
    cell: object,
) -> None:
    dim = 2 if isinstance(cell, planar.RectangularCell) else 3
    bad_points = np.zeros((1, dim))
    bad_points[0, 0] = np.nan
    with pytest.raises(ValueError, match='points.*finite'):
        cell.remap_cart(bad_points)
    with pytest.raises(ValueError, match='eps.*finite'):
        cell.remap_cart(np.zeros((1, dim)), eps=np.inf)
    with pytest.raises(ValueError, match='return_shifts.*Boolean'):
        cell.remap_cart(np.zeros((1, dim)), return_shifts=1)


@pytest.mark.parametrize(
    ('cell', 'points'),
    [
        (
            pyvoro2.OrthorhombicCell(
                ((0, 1), (0, 1), (0, 1)),
                periodic=(True, True, True),
            ),
            np.array([[1e20, 0.0, 0.0]]),
        ),
        (
            planar.RectangularCell(
                ((0, 1), (0, 1)),
                periodic=(True, True),
            ),
            np.array([[1e20, 0.0]]),
        ),
        (
            pyvoro2.PeriodicCell(
                ((1, 0, 0), (0.2, 1, 0), (0.1, 0.3, 1))
            ),
            np.array([[1e20, 0.0, 0.0]]),
        ),
    ],
)
def test_remap_rejects_unrepresentable_shift_quotients(
    cell: object,
    points: np.ndarray,
) -> None:
    with pytest.raises(ValueError, match='signed int64'):
        cell.remap_cart(points, return_shifts=True)


def test_valid_remap_values_and_shift_sign_are_unchanged() -> None:
    cell = pyvoro2.PeriodicCell(
        ((2, 0, 0), (0.25, 2, 0), (0.1, -0.2, 2)),
        origin=(0.1, -0.1, 0.2),
    )
    points = np.array([[4.75, -2.2, 2.35], [-1.4, 3.9, -2.1]])

    remapped, shifts = cell.remap_cart(points, return_shifts=True)
    lattice = np.asarray(cell.vectors)

    np.testing.assert_allclose(remapped + shifts @ lattice, points, atol=1e-12)
    assert shifts.dtype == np.int64


@pytest.mark.parametrize('nonfinite', [np.nan, np.inf, -np.inf])
def test_duplicate_normalization_and_diagnostic_tolerances_are_finite(
    nonfinite: float,
) -> None:
    domain3d = pyvoro2.Box(((0, 1), (0, 1), (0, 1)))
    domain2d = planar.Box(((0, 1), (0, 1)))
    with pytest.raises(ValueError, match='threshold'):
        pyvoro2.duplicate_check(
            np.zeros((1, 3)),
            threshold=nonfinite,
            mode='return',
        )
    with pytest.raises(ValueError, match='tol'):
        pyvoro2.normalize_vertices([], domain=domain3d, tol=nonfinite)
    with pytest.raises(ValueError, match='volume_tol_rel'):
        pyvoro2.analyze_tessellation(
            [],
            domain3d,
            volume_tol_rel=nonfinite,
        )
    with pytest.raises(ValueError, match='line_angle_tol'):
        planar.analyze_tessellation(
            [],
            domain2d,
            line_angle_tol=nonfinite,
        )


@pytest.mark.parametrize('invalid', [0, 1, 'false', _Truthy()])
def test_duplicate_normalization_and_diagnostic_flags_are_exact_booleans(
    invalid: object,
) -> None:
    domain3d = pyvoro2.Box(((0, 1), (0, 1), (0, 1)))
    with pytest.raises(ValueError, match='wrap.*Boolean'):
        pyvoro2.duplicate_check(
            np.zeros((1, 3)),
            wrap=invalid,
            mode='return',
        )
    with pytest.raises(ValueError, match='copy_cells.*Boolean'):
        pyvoro2.normalize_vertices([], domain=domain3d, copy_cells=invalid)
    with pytest.raises(ValueError, match='check_reciprocity.*Boolean'):
        pyvoro2.analyze_tessellation(
            [],
            domain3d,
            check_reciprocity=invalid,
        )


@pytest.mark.parametrize('value', [0, np.int32(1), np.uint64(2)])
@pytest.mark.parametrize(
    ('api', 'domain'),
    [
        (pyvoro2, pyvoro2.Box(((0, 1), (0, 1), (0, 1)))),
        (planar, planar.Box(((0, 1), (0, 1)))),
    ],
)
def test_normalization_example_limit_accepts_nonnegative_exact_integers(
    api: object,
    domain: object,
    value: object,
) -> None:
    normalized = api.normalize_vertices([], domain=domain)
    diagnostics = api.validate_normalized_topology(
        normalized,
        domain,
        max_examples=value,
    )

    assert diagnostics.issues == ()


@pytest.mark.parametrize(
    ('api', 'domain', 'dim', 'boundary_key', 'issue_code'),
    [
        (
            pyvoro2,
            pyvoro2.OrthorhombicCell(
                ((0, 1), (0, 1), (0, 1)),
                periodic=(True, True, True),
            ),
            3,
            'faces',
            'FACE_MISSING_ADJACENT_SHIFT',
        ),
        (
            planar,
            planar.RectangularCell(
                ((0, 1), (0, 1)),
                periodic=(True, True),
            ),
            2,
            'edges',
            'EDGE_MISSING_ADJACENT_SHIFT',
        ),
    ],
)
def test_zero_normalization_example_limit_retains_issues_without_examples(
    api: object,
    domain: object,
    dim: int,
    boundary_key: str,
    issue_code: str,
) -> None:
    cell = {
        'id': 0,
        'vertex_global_id': [0],
        'vertex_shift': [(0,) * dim],
        boundary_key: [{'adjacent_cell': 1, 'vertices': [0]}],
    }
    normalized = api.NormalizedVertices(
        global_vertices=np.zeros((1, dim)),
        cells=[cell],
    )
    diagnostics = api.validate_normalized_topology(
        normalized,
        domain,
        max_examples=0,
    )

    issue = next(item for item in diagnostics.issues if item.code == issue_code)
    assert issue.examples == ()


@pytest.mark.parametrize(
    'invalid',
    [True, np.bool_(False), 0.0, 1.0, '1', 1 + 0j, np.array(1), -1],
)
@pytest.mark.parametrize(
    ('api', 'domain'),
    [
        (pyvoro2, pyvoro2.Box(((0, 1), (0, 1), (0, 1)))),
        (planar, planar.Box(((0, 1), (0, 1)))),
    ],
)
def test_normalization_example_limit_rejects_lossy_or_negative_values(
    api: object,
    domain: object,
    invalid: object,
) -> None:
    normalized = api.normalize_vertices([], domain=domain)
    with pytest.raises(ValueError, match='max_examples'):
        api.validate_normalized_topology(
            normalized,
            domain,
            max_examples=invalid,
        )


@pytest.mark.parametrize('dim', [2, 3])
@pytest.mark.parametrize('periodic', [False, True])
@pytest.mark.parametrize('tol', [1e-20, np.nextafter(0.0, 1.0)])
def test_normalization_rejects_unrepresentable_quantization_before_mutation(
    dim: int,
    periodic: bool,
    tol: float,
) -> None:
    api, cells, domain, _boundary_key = _normalization_case(
        dim,
        periodic=periodic,
    )
    snapshot = repr(cells)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        with np.errstate(all='raise'):
            with pytest.raises(
                ValueError,
                match='quantization values representable as signed int64',
            ):
                api.normalize_topology(
                    cells,
                    domain=domain,
                    tol=tol,
                    copy_cells=False,
                )

    assert caught == []
    assert repr(cells) == snapshot


@pytest.mark.parametrize(
    ('dim', 'expected_vertices'),
    [(2, 6), (3, 12)],
)
def test_normalization_valid_compute_control_keeps_distinct_vertices(
    dim: int,
    expected_vertices: int,
) -> None:
    api, cells, domain, _boundary_key = _normalization_case(
        dim,
        periodic=False,
    )
    normalized = api.normalize_topology(cells, domain=domain)

    assert normalized.global_vertices.shape == (expected_vertices, dim)


@pytest.mark.parametrize('dim', [2, 3])
def test_periodic_power_normalization_preserves_empty_cells(dim: int) -> None:
    radii = np.array([1.0, 2.0])
    if dim == 2:
        points = np.array([[0.1, 0.5], [0.9, 0.5]])
        domain = planar.RectangularCell(
            ((0, 1), (0, 1)),
            periodic=(True, True),
        )
        result = planar.compute(
            points,
            domain=domain,
            mode='power',
            radii=radii,
            include_empty=True,
            normalize='topology',
        )
        normalized = result.require_normalized_topology()
    else:
        points = np.array([[0.1, 0.5, 0.5], [0.9, 0.5, 0.5]])
        domain = pyvoro2.PeriodicCell(
            ((1, 0, 0), (0, 1, 0), (0, 0, 1)),
        )
        cells = pyvoro2.compute(
            points,
            domain=domain,
            mode='power',
            radii=radii,
            include_empty=True,
            output='cells',
            return_vertices=True,
            return_faces=True,
            return_face_shifts=True,
        )
        normalized = pyvoro2.normalize_topology(cells, domain=domain)

    assert any(cell.get('empty') is True for cell in normalized.cells)
    assert normalized.global_vertices.shape[1] == dim


_LOSSY_NORMALIZATION_INTEGERS = [
    True,
    np.bool_(False),
    0.0,
    0.5,
    '0',
    0 + 0j,
    np.array(0),
    2**63,
]


@pytest.mark.parametrize('dim', [2, 3])
@pytest.mark.parametrize(
    ('field', 'error_name'),
    [
        ('cell_id', r'cells\[0\]\.id'),
        ('local_vertex', r'\.vertices'),
        ('adjacent_cell', r'\.adjacent_cell'),
        ('adjacent_shift', r'\.adjacent_shift'),
    ],
)
@pytest.mark.parametrize('invalid', _LOSSY_NORMALIZATION_INTEGERS)
def test_periodic_normalization_rejects_lossy_raw_metadata_before_mutation(
    dim: int,
    field: str,
    error_name: str,
    invalid: object,
) -> None:
    api, cells, domain, boundary_key = _normalization_case(dim, periodic=True)
    boundary = _first_boundary_with_vertices(cells, boundary_key)
    if field == 'cell_id':
        cells[0]['id'] = invalid
    elif field == 'local_vertex':
        vertices = list(boundary['vertices'])
        vertices[0] = invalid
        boundary['vertices'] = vertices
    elif field == 'adjacent_cell':
        boundary['adjacent_cell'] = invalid
    else:
        shift = list(boundary['adjacent_shift'])
        shift[0] = invalid
        boundary['adjacent_shift'] = shift
    snapshot = repr(cells)

    with pytest.raises(ValueError, match=error_name):
        api.normalize_topology(cells, domain=domain, copy_cells=False)

    assert repr(cells) == snapshot


@pytest.mark.parametrize('dim', [2, 3])
@pytest.mark.parametrize(
    ('field', 'error_name'),
    [
        ('cell_id', r'cells\[0\]\.id'),
        ('local_vertex', r'\.vertices'),
        ('adjacent_cell', r'\.adjacent_cell'),
    ],
)
@pytest.mark.parametrize('invalid', _LOSSY_NORMALIZATION_INTEGERS)
def test_nonperiodic_normalization_rejects_lossy_raw_metadata_before_mutation(
    dim: int,
    field: str,
    error_name: str,
    invalid: object,
) -> None:
    api, cells, domain, boundary_key = _normalization_case(
        dim,
        periodic=False,
    )
    boundary = _first_boundary_with_vertices(cells, boundary_key)
    if field == 'cell_id':
        cells[0]['id'] = invalid
    elif field == 'local_vertex':
        vertices = list(boundary['vertices'])
        vertices[0] = invalid
        boundary['vertices'] = vertices
    else:
        boundary['adjacent_cell'] = invalid
    snapshot = repr(cells)

    with pytest.raises(ValueError, match=error_name):
        api.normalize_topology(cells, domain=domain, copy_cells=False)

    assert repr(cells) == snapshot


@pytest.mark.parametrize('dim', [2, 3])
@pytest.mark.parametrize(
    ('field', 'invalid'),
    [
        ('cell_id', -1),
        ('local_vertex', -1),
        ('adjacent_cell', -(2**63) - 1),
        ('adjacent_shift', -(2**63) - 1),
    ],
)
def test_periodic_normalization_enforces_metadata_destination_ranges(
    dim: int,
    field: str,
    invalid: int,
) -> None:
    api, cells, domain, boundary_key = _normalization_case(dim, periodic=True)
    boundary = _first_boundary_with_vertices(cells, boundary_key)
    if field == 'cell_id':
        cells[0]['id'] = invalid
    elif field == 'local_vertex':
        vertices = list(boundary['vertices'])
        vertices[0] = invalid
        boundary['vertices'] = vertices
    elif field == 'adjacent_cell':
        boundary['adjacent_cell'] = invalid
    else:
        shift = list(boundary['adjacent_shift'])
        shift[0] = invalid
        boundary['adjacent_shift'] = shift

    with pytest.raises(ValueError):
        api.normalize_topology(cells, domain=domain)


@pytest.mark.parametrize('dim', [2, 3])
def test_periodic_normalization_accepts_exact_numpy_record_metadata(
    dim: int,
) -> None:
    api, cells, domain, boundary_key = _normalization_case(dim, periodic=True)
    cells[0]['id'] = np.uint64(cells[0]['id'])
    boundary = _first_boundary_with_vertices(cells, boundary_key)
    boundary['adjacent_cell'] = np.int64(boundary['adjacent_cell'])
    boundary['vertices'] = [
        np.uint64(value) for value in boundary['vertices']
    ]
    boundary['adjacent_shift'] = tuple(
        np.int32(value) for value in boundary['adjacent_shift']
    )

    normalized = api.normalize_topology(cells, domain=domain)

    assert normalized.global_vertices.shape[1] == dim


@pytest.mark.parametrize('dim', [2, 3])
@pytest.mark.parametrize('invalid', [True, '0', 0 + 0j, np.nan, np.inf])
def test_normalization_rejects_invalid_vertex_coordinates_before_mutation(
    dim: int,
    invalid: object,
) -> None:
    api, cells, domain, _boundary_key = _normalization_case(
        dim,
        periodic=False,
    )
    cells[0]['vertices'][0][0] = invalid
    snapshot = repr(cells)

    with pytest.raises(ValueError, match=r'cells\[0\]\.vertices'):
        api.normalize_vertices(cells, domain=domain, copy_cells=False)

    assert repr(cells) == snapshot


@pytest.mark.parametrize('dim', [2, 3])
@pytest.mark.parametrize('field', ['vertex_global_id', 'vertex_shift'])
@pytest.mark.parametrize('invalid', _LOSSY_NORMALIZATION_INTEGERS)
def test_topology_normalization_revalidates_retained_integer_metadata(
    dim: int,
    field: str,
    invalid: object,
) -> None:
    api, cells, domain, _boundary_key = _normalization_case(dim, periodic=True)
    normalized = api.normalize_vertices(cells, domain=domain)
    if field == 'vertex_global_id':
        normalized.cells[0][field][0] = invalid
    else:
        shift = list(normalized.cells[0][field][0])
        shift[0] = invalid
        normalized.cells[0][field][0] = tuple(shift)
    snapshot = repr(normalized.cells)
    normalize_topology_records = (
        api.normalize_edges_faces if dim == 3 else api.normalize_edges
    )

    with pytest.raises(ValueError, match=field):
        normalize_topology_records(
            normalized,
            domain=domain,
            copy_cells=False,
        )

    assert repr(normalized.cells) == snapshot


@pytest.mark.parametrize('dim', [2, 3])
def test_topology_normalization_rejects_derived_shift_overflow_before_mutation(
    dim: int,
) -> None:
    api, cells, domain, boundary_key = _normalization_case(dim, periodic=True)
    normalized = api.normalize_vertices(cells, domain=domain)
    cell_index, boundary = next(
        (cell_index, boundary)
        for cell_index, cell in enumerate(normalized.cells)
        for boundary in cell[boundary_key]
        if len(boundary.get('vertices', ())) >= 2
    )
    u, v = boundary['vertices'][:2]
    low = (-(2**63),) + (0,) * (dim - 1)
    high = (2**63 - 1,) + (0,) * (dim - 1)
    normalized.cells[cell_index]['vertex_shift'][u] = low
    normalized.cells[cell_index]['vertex_shift'][v] = high
    snapshot = repr(normalized.cells)
    normalize_topology_records = (
        api.normalize_edges_faces if dim == 3 else api.normalize_edges
    )

    with pytest.raises(ValueError, match='difference.*signed int64'):
        normalize_topology_records(
            normalized,
            domain=domain,
            copy_cells=False,
        )

    assert repr(normalized.cells) == snapshot


@pytest.mark.parametrize('make_invalid', _STRING_IMPOSTOR_FACTORIES)
@pytest.mark.parametrize(
    ('api', 'dim'),
    [(pyvoro2, 3), (planar, 2)],
)
def test_duplicate_modes_reject_non_string_categories_before_point_coercion(
    make_invalid: Callable[[str], object],
    api: object,
    dim: int,
) -> None:
    invalid = make_invalid('return')

    with pytest.raises(ValueError, match='mode'):
        api.duplicate_check(_ExplodingArray(), mode=invalid)

    if isinstance(invalid, EqualAny):
        assert invalid.comparisons == 0


_FORWARD_STRING_BOUNDARIES = (
    (pyvoro2, pyvoro2.Box(((0, 1), (0, 1), (0, 1))), 'mode', 'standard'),
    (pyvoro2, pyvoro2.Box(((0, 1), (0, 1), (0, 1))), 'output', 'result'),
    (
        pyvoro2,
        pyvoro2.Box(((0, 1), (0, 1), (0, 1))),
        'duplicate_check',
        'off',
    ),
    (
        pyvoro2,
        pyvoro2.Box(((0, 1), (0, 1), (0, 1))),
        'tessellation_check',
        'none',
    ),
    (planar, planar.Box(((0, 1), (0, 1))), 'mode', 'standard'),
    (planar, planar.Box(((0, 1), (0, 1))), 'output', 'result'),
    (planar, planar.Box(((0, 1), (0, 1))), 'normalize', 'none'),
    (planar, planar.Box(((0, 1), (0, 1))), 'duplicate_check', 'off'),
    (
        planar,
        planar.Box(((0, 1), (0, 1))),
        'tessellation_check',
        'none',
    ),
)


@pytest.mark.parametrize('make_invalid', _STRING_IMPOSTOR_FACTORIES)
@pytest.mark.parametrize(
    ('api', 'domain', 'field', 'allowed'),
    _FORWARD_STRING_BOUNDARIES,
)
def test_compute_string_options_reject_categories_before_point_coercion(
    make_invalid: Callable[[str], object],
    api: object,
    domain: object,
    field: str,
    allowed: str,
) -> None:
    invalid = make_invalid(allowed)

    with pytest.raises(ValueError, match=field):
        api.compute(
            _ExplodingArray(),
            domain=domain,
            **{field: invalid},
        )

    if isinstance(invalid, EqualAny):
        assert invalid.comparisons == 0


@pytest.mark.parametrize(
    ('api', 'points', 'domain', 'extra'),
    [
        (
            pyvoro2,
            np.array([[0.25, 0.5, 0.5], [0.75, 0.5, 0.5]]),
            pyvoro2.Box(((0, 1), (0, 1), (0, 1))),
            {},
        ),
        (
            planar,
            np.array([[0.25, 0.5], [0.75, 0.5]]),
            planar.Box(((0, 1), (0, 1))),
            {'normalize': np.str_('none')},
        ),
    ],
)
def test_forward_numpy_string_scalars_canonicalize_to_builtin_strings(
    api: object,
    points: np.ndarray,
    domain: object,
    extra: dict[str, object],
) -> None:
    result = api.compute(
        points,
        domain=domain,
        mode=np.str_('standard'),
        output=np.str_('result'),
        duplicate_check=np.str_('off'),
        tessellation_check=np.str_('none'),
        **extra,
    )

    assert type(result.mode) is str
    assert result.mode == 'standard'


def test_planar_numpy_string_normalization_choice_is_unchanged() -> None:
    result = planar.compute(
        np.array([[0.25, 0.5], [0.75, 0.5]]),
        domain=planar.Box(((0, 1), (0, 1))),
        normalize=np.str_('vertices'),
    )

    assert result.normalized_vertices is not None


@pytest.mark.parametrize(
    'make_invalid',
    _NON_NONE_STRING_IMPOSTOR_FACTORIES,
)
@pytest.mark.parametrize(
    ('analyze', 'domain'),
    [
        (
            pyvoro2.analyze_tessellation,
            pyvoro2.Box(((0, 1), (0, 1), (0, 1))),
        ),
        (planar.analyze_tessellation, planar.Box(((0, 1), (0, 1)))),
    ],
)
def test_diagnostic_modes_reject_categories_before_cell_analysis(
    make_invalid: Callable[[str], object],
    analyze: Callable[..., object],
    domain: object,
) -> None:
    invalid = make_invalid('standard')

    with pytest.raises(ValueError, match='mode'):
        analyze(_ExplodingCells(), domain, mode=invalid)

    if isinstance(invalid, EqualAny):
        assert invalid.comparisons == 0


@pytest.mark.parametrize(
    ('analyze', 'domain'),
    [
        (
            pyvoro2.analyze_tessellation,
            pyvoro2.Box(((0, 1), (0, 1), (0, 1))),
        ),
        (planar.analyze_tessellation, planar.Box(((0, 1), (0, 1)))),
    ],
)
def test_optional_diagnostic_mode_still_accepts_none(
    analyze: Callable[..., object],
    domain: object,
) -> None:
    diagnostics = analyze([], domain, mode=None)

    assert diagnostics is not None


@pytest.mark.parametrize('make_invalid', _STRING_IMPOSTOR_FACTORIES)
@pytest.mark.parametrize(
    ('validate', 'domain'),
    [
        (
            pyvoro2.validate_tessellation,
            pyvoro2.Box(((0, 1), (0, 1), (0, 1))),
        ),
        (planar.validate_tessellation, planar.Box(((0, 1), (0, 1)))),
    ],
)
def test_diagnostic_levels_reject_categories_before_cell_analysis(
    make_invalid: Callable[[str], object],
    validate: Callable[..., object],
    domain: object,
) -> None:
    invalid = make_invalid('basic')

    with pytest.raises(ValueError, match='level'):
        validate(_ExplodingCells(), domain, level=invalid)

    if isinstance(invalid, EqualAny):
        assert invalid.comparisons == 0


@pytest.mark.parametrize('make_invalid', _STRING_IMPOSTOR_FACTORIES)
@pytest.mark.parametrize(
    ('api', 'domain'),
    [
        (pyvoro2, pyvoro2.Box(((0, 1), (0, 1), (0, 1)))),
        (planar, planar.Box(((0, 1), (0, 1)))),
    ],
)
def test_normalized_validation_levels_reject_before_topology_access(
    make_invalid: Callable[[str], object],
    api: object,
    domain: object,
) -> None:
    invalid = make_invalid('basic')

    with pytest.raises(ValueError, match='level'):
        api.validate_normalized_topology(object(), domain, level=invalid)

    if isinstance(invalid, EqualAny):
        assert invalid.comparisons == 0


def test_string_choice_sets_remain_exact_and_case_sensitive() -> None:
    with pytest.raises(ValueError, match='mode'):
        pyvoro2.duplicate_check(np.zeros((1, 3)), mode='RETURN')
    with pytest.raises(ValueError, match='mode'):
        pyvoro2.compute(
            np.zeros((1, 3)),
            domain=pyvoro2.Box(((0, 1), (0, 1), (0, 1))),
            mode='Standard',
        )
    with pytest.raises(ValueError, match='normalize'):
        planar.compute(
            np.zeros((1, 2)),
            domain=planar.Box(((0, 1), (0, 1))),
            normalize='None',
        )


@pytest.mark.parametrize('make_invalid', _STRING_IMPOSTOR_FACTORIES)
def test_visualization_string_mode_rejects_before_view_construction(
    make_invalid: Callable[[str], object],
) -> None:
    from pyvoro2.viz3d import view_tessellation

    invalid = make_invalid('auto')

    with pytest.raises(ValueError, match='show_vertex_labels'):
        view_tessellation([], show_vertex_labels=invalid)

    if isinstance(invalid, EqualAny):
        assert invalid.comparisons == 0


@pytest.mark.parametrize('make_invalid', _STRING_IMPOSTOR_FACTORIES)
def test_direct_tessellation_result_mode_rejects_non_string_categories(
    make_invalid: Callable[[str], object],
) -> None:
    result = pyvoro2.compute(
        np.array([[0.25, 0.5, 0.5], [0.75, 0.5, 0.5]]),
        domain=pyvoro2.Box(((0, 1), (0, 1), (0, 1))),
    )
    invalid = make_invalid('standard')

    with pytest.raises(ValueError, match='mode'):
        replace(result, mode=invalid)

    if isinstance(invalid, EqualAny):
        assert invalid.comparisons == 0


def test_direct_string_value_objects_store_builtin_strings() -> None:
    from pyvoro2.diagnostics import TessellationIssue

    result = pyvoro2.compute(
        np.array([[0.25, 0.5, 0.5], [0.75, 0.5, 0.5]]),
        domain=pyvoro2.Box(((0, 1), (0, 1), (0, 1))),
    )
    replaced = replace(result, mode=np.str_('standard'))
    issue = TessellationIssue(
        code=np.str_('CODE'),
        severity=np.str_('warning'),
        message=np.str_('message'),
    )

    assert type(replaced.mode) is str
    assert type(issue.code) is str
    assert type(issue.severity) is str
    assert type(issue.message) is str
