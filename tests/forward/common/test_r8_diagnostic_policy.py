from __future__ import annotations

import copy
import math
import sys
import warnings
from dataclasses import dataclass
from types import ModuleType
from typing import Any

import numpy as np
import pytest

import pyvoro2
import pyvoro2.api as api3d
import pyvoro2.planar as planar
import pyvoro2.planar.api as api2d


@dataclass(frozen=True)
class DimensionCase:
    api: ModuleType
    domain: object
    periodic_domain: object
    points: np.ndarray
    measure: str
    boundary: str
    shift_dim: int
    no_shift_code: str
    compute_geometry: dict[str, bool]
    mismatch_option: str
    mark_option: str


def _case(dim: int) -> DimensionCase:
    if dim == 3:
        return DimensionCase(
            api=pyvoro2,
            domain=pyvoro2.Box(((0, 1), (0, 1), (0, 1))),
            periodic_domain=pyvoro2.OrthorhombicCell(
                ((0, 1), (0, 1), (0, 1)),
                periodic=(True, True, True),
            ),
            points=np.array(
                [[0.2, 0.2, 0.2], [0.7, 0.25, 0.3], [0.4, 0.75, 0.8]],
            ),
            measure='volume',
            boundary='faces',
            shift_dim=3,
            no_shift_code='NO_FACE_SHIFTS',
            compute_geometry={
                'return_vertices': True,
                'return_faces': True,
                'return_face_shifts': True,
            },
            mismatch_option='check_plane_mismatch',
            mark_option='mark_faces',
        )
    return DimensionCase(
        api=planar,
        domain=planar.Box(((0, 1), (0, 1))),
        periodic_domain=planar.RectangularCell(
            ((0, 1), (0, 1)),
            periodic=(True, True),
        ),
        points=np.array([[0.2, 0.2], [0.8, 0.25], [0.4, 0.8]]),
        measure='area',
        boundary='edges',
        shift_dim=2,
        no_shift_code='NO_EDGE_SHIFTS',
        compute_geometry={
            'return_vertices': True,
            'return_edges': True,
            'return_edge_shifts': True,
        },
        mismatch_option='check_line_mismatch',
        mark_option='mark_edges',
    )


def _closed_cell(case: DimensionCase) -> dict[str, Any]:
    return {'id': 0, case.measure: 1.0, 'empty': False}


def _measure_codes(diag: object) -> list[str]:
    measure_codes = {
        'MISSING_CELL_MEASURE',
        'INVALID_CELL_MEASURE',
        'NONFINITE_CELL_MEASURE',
        'NEGATIVE_CELL_MEASURE',
        'EMPTY_CELL_NONZERO_MEASURE',
    }
    return [
        issue.code
        for issue in diag.issues
        if issue.code in measure_codes
    ]


@pytest.mark.parametrize('dim', [2, 3])
@pytest.mark.parametrize(
    ('mode', 'code', 'severity', 'ok', 'hidden'),
    [
        ('standard', 'MISSING_IDS', 'error', False, False),
        ('power', 'HIDDEN_IDS', 'info', True, True),
        (None, 'MISSING_IDS', 'warning', True, False),
    ],
)
def test_expected_id_policy_matrix(
    dim: int,
    mode: str | None,
    code: str,
    severity: str,
    ok: bool,
    hidden: bool,
) -> None:
    case = _case(dim)
    cells = [_closed_cell(case)]

    diag = case.api.analyze_tessellation(
        cells,
        case.domain,
        expected_ids=[0, 1],
        mode=mode,
    )

    issue = next(issue for issue in diag.issues if issue.code == code)
    assert issue.severity == severity
    assert diag.missing_ids == (1,)
    assert diag.empty_ids == ((1,) if hidden else ())
    assert diag.ok is ok
    basic = case.api.validate_tessellation(
        cells,
        case.domain,
        expected_ids=[0, 1],
        mode=mode,
        level='basic',
    )
    assert basic == diag
    if ok:
        strict = case.api.validate_tessellation(
            cells,
            case.domain,
            expected_ids=[0, 1],
            mode=mode,
            level='strict',
        )
        assert strict.ok is True
    else:
        with pytest.raises(case.api.TessellationError) as exc_info:
            case.api.validate_tessellation(
                cells,
                case.domain,
                expected_ids=[0, 1],
                mode=mode,
                level='strict',
            )
        assert exc_info.value.diagnostics.ok is False
        assert code in str(exc_info.value)


_MISSING = object()


@pytest.mark.parametrize('dim', [2, 3])
@pytest.mark.parametrize(
    ('value', 'expected_code'),
    [
        (_MISSING, 'MISSING_CELL_MEASURE'),
        ('oops', 'INVALID_CELL_MEASURE'),
        (None, 'INVALID_CELL_MEASURE'),
        (True, 'INVALID_CELL_MEASURE'),
        (np.bool_(False), 'INVALID_CELL_MEASURE'),
        (1 + 2j, 'INVALID_CELL_MEASURE'),
        (np.array([1.0, 2.0]), 'INVALID_CELL_MEASURE'),
        (np.nan, 'NONFINITE_CELL_MEASURE'),
        (np.inf, 'NONFINITE_CELL_MEASURE'),
        (-np.inf, 'NONFINITE_CELL_MEASURE'),
        (-1.0, 'NEGATIVE_CELL_MEASURE'),
    ],
)
def test_nonempty_measure_category_matrix(
    dim: int,
    value: object,
    expected_code: str,
) -> None:
    case = _case(dim)
    cell = _closed_cell(case)
    if value is _MISSING:
        del cell[case.measure]
    else:
        cell[case.measure] = value

    diag = case.api.analyze_tessellation([cell], case.domain)

    assert _measure_codes(diag) == [expected_code]
    issue = next(issue for issue in diag.issues if issue.code == expected_code)
    assert issue.severity == 'error'
    assert not any(issue.code in ('GAP', 'OVERLAP') for issue in diag.issues)
    assert getattr(diag, f'ok_{case.measure}') is False
    assert diag.ok is False


@pytest.mark.parametrize('dim', [2, 3])
@pytest.mark.parametrize(
    ('value', 'expected_codes'),
    [
        (_MISSING, []),
        (0.0, []),
        (0, []),
        (0.25, ['EMPTY_CELL_NONZERO_MEASURE']),
        (-0.25, ['NEGATIVE_CELL_MEASURE', 'EMPTY_CELL_NONZERO_MEASURE']),
        ('oops', ['INVALID_CELL_MEASURE']),
        (None, ['INVALID_CELL_MEASURE']),
        (np.nan, ['NONFINITE_CELL_MEASURE']),
        (np.inf, ['NONFINITE_CELL_MEASURE']),
        (-np.inf, ['NONFINITE_CELL_MEASURE']),
    ],
)
def test_empty_measure_category_matrix(
    dim: int,
    value: object,
    expected_codes: list[str],
) -> None:
    case = _case(dim)
    empty = {'id': 1, 'empty': True}
    if value is not _MISSING:
        empty[case.measure] = value
    cells = [_closed_cell(case), empty]

    diag = case.api.analyze_tessellation(cells, case.domain)

    assert _measure_codes(diag) == expected_codes
    assert all(
        issue.severity == 'error'
        for issue in diag.issues
        if issue.code in expected_codes
    )
    assert diag.ok is (not expected_codes)


@pytest.mark.parametrize('dim', [2, 3])
def test_valid_nonempty_zero_is_measure_valid_but_not_closed(dim: int) -> None:
    case = _case(dim)
    cell = _closed_cell(case)
    cell[case.measure] = np.float64(0.0)

    diag = case.api.analyze_tessellation([cell], case.domain)

    assert _measure_codes(diag) == []
    assert [(issue.code, issue.severity) for issue in diag.issues] == [
        ('GAP', 'error')
    ]
    assert diag.ok is False


@pytest.mark.parametrize('dim', [2, 3])
def test_stable_measure_sum_uses_test_local_fsum_oracle(dim: int) -> None:
    case = _case(dim)
    tiny = math.ldexp(1.0, -54)
    values = [1.0, tiny, tiny, tiny, tiny]
    expected = math.fsum(values)
    if dim == 3:
        domain = pyvoro2.Box(((0, expected), (0, 1), (0, 1)))
        tolerance = {'volume_tol_rel': 0.0, 'volume_tol_abs': 0.0}
    else:
        domain = planar.Box(((0, expected), (0, 1)))
        tolerance = {'area_tol_rel': 0.0, 'area_tol_abs': 0.0}
    cells = [
        {'id': index, case.measure: value, 'empty': False}
        for index, value in enumerate(values)
    ]

    diag = case.api.analyze_tessellation(cells, domain, **tolerance)

    assert getattr(diag, f'sum_cell_{case.measure}') == expected
    assert diag.ok is True


@pytest.mark.parametrize('dim', [2, 3])
def test_finite_measure_sum_range_overflow_is_closure_overlap(dim: int) -> None:
    case = _case(dim)
    values = [1e308, 1e308]
    assert all(math.isfinite(value) for value in values)
    cells = [
        {'id': index, case.measure: value, 'empty': False}
        for index, value in enumerate(values)
    ]

    diag = case.api.analyze_tessellation(cells, case.domain)

    assert _measure_codes(diag) == []
    total = getattr(diag, f'sum_cell_{case.measure}')
    assert math.isinf(total) and total > 0.0
    overlap = next(issue for issue in diag.issues if issue.code == 'OVERLAP')
    assert overlap.severity == 'error'
    assert getattr(diag, f'ok_{case.measure}') is False
    assert diag.ok is False

    with pytest.raises(case.api.TessellationError) as exc_info:
        case.api.validate_tessellation(
            cells,
            case.domain,
            level='strict',
        )
    strict_diag = exc_info.value.diagnostics
    strict_overlap = next(
        issue for issue in strict_diag.issues if issue.code == 'OVERLAP'
    )
    assert strict_overlap.severity == 'error'
    assert _measure_codes(strict_diag) == []
    assert math.isinf(getattr(strict_diag, f'sum_cell_{case.measure}'))
    assert strict_diag.ok is False


@pytest.mark.parametrize('dim', [2, 3])
def test_aggregate_overflow_precedes_infinite_derived_tolerance(dim: int) -> None:
    case = _case(dim)
    values = [1e308, 1e308]
    assert all(math.isfinite(value) for value in values)
    if dim == 3:
        domain = pyvoro2.Box(((0, 2), (0, 1), (0, 1)))
        tolerance = {'volume_tol_rel': sys.float_info.max}
    else:
        domain = planar.Box(((0, 2), (0, 1)))
        tolerance = {'area_tol_rel': sys.float_info.max}
    cells = [
        {'id': index, case.measure: value, 'empty': False}
        for index, value in enumerate(values)
    ]

    diag = case.api.analyze_tessellation(cells, domain, **tolerance)

    total = getattr(diag, f'sum_cell_{case.measure}')
    overlap_value = getattr(diag, f'{case.measure}_overlap')
    assert math.isinf(total) and total > 0.0
    assert math.isinf(overlap_value) and overlap_value > 0.0
    assert _measure_codes(diag) == []
    overlap = next(issue for issue in diag.issues if issue.code == 'OVERLAP')
    assert overlap.severity == 'error'
    assert getattr(diag, f'ok_{case.measure}') is False
    assert diag.ok is False

    with pytest.raises(case.api.TessellationError) as exc_info:
        case.api.validate_tessellation(
            cells,
            domain,
            level='strict',
            **tolerance,
        )
    strict_diag = exc_info.value.diagnostics
    strict_overlap = next(
        issue for issue in strict_diag.issues if issue.code == 'OVERLAP'
    )
    assert strict_overlap.severity == 'error'
    assert _measure_codes(strict_diag) == []
    assert strict_diag.ok is False


@pytest.mark.parametrize('dim', [2, 3])
def test_valid_measure_closure_within_tolerance_is_ok(dim: int) -> None:
    case = _case(dim)
    cell = _closed_cell(case)
    cell[case.measure] = 1.0 - 5e-9

    diag = case.api.analyze_tessellation([cell], case.domain)

    assert not any(issue.code in ('GAP', 'OVERLAP') for issue in diag.issues)
    assert getattr(diag, f'ok_{case.measure}') is True
    assert diag.ok is True


@pytest.mark.parametrize('dim', [2, 3])
@pytest.mark.parametrize(('measure', 'code'), [(0.5, 'GAP'), (1.5, 'OVERLAP')])
def test_valid_closure_failures_are_errors(
    dim: int,
    measure: float,
    code: str,
) -> None:
    case = _case(dim)
    cell = _closed_cell(case)
    cell[case.measure] = measure

    diag = case.api.analyze_tessellation([cell], case.domain)

    assert (code, 'error') in [
        (issue.code, issue.severity) for issue in diag.issues
    ]
    assert diag.ok is False


def _periodic_cells(case: DimensionCase) -> list[dict[str, Any]]:
    return case.api.compute(
        case.points,
        domain=case.periodic_domain,
        output='cells',
        **case.compute_geometry,
    )


def _direct_boundary_map(
    cells: list[dict[str, Any]],
    case: DimensionCase,
) -> dict[tuple[int, int, tuple[int, ...]], tuple[int, int]]:
    result: dict[tuple[int, int, tuple[int, ...]], tuple[int, int]] = {}
    for cell_index, cell in enumerate(cells):
        for boundary_index, boundary in enumerate(cell.get(case.boundary) or []):
            neighbor = int(boundary.get('adjacent_cell', -1))
            if neighbor < 0 or 'adjacent_shift' not in boundary:
                continue
            shift = tuple(int(value) for value in boundary['adjacent_shift'])
            result[(int(cell['id']), neighbor, shift)] = (
                cell_index,
                boundary_index,
            )
    return result


def _reciprocal_key(
    key: tuple[int, int, tuple[int, ...]],
) -> tuple[int, int, tuple[int, ...]]:
    return key[1], key[0], tuple(-value for value in key[2])


def _select_reciprocal_pair(
    cells: list[dict[str, Any]],
    case: DimensionCase,
    *,
    zero_shift: bool = False,
) -> tuple[
    tuple[int, int, tuple[int, ...]],
    tuple[int, int, tuple[int, ...]],
    dict[tuple[int, int, tuple[int, ...]], tuple[int, int]],
]:
    mapping = _direct_boundary_map(cells, case)
    candidates = [
        key
        for key in mapping
        if key[0] != key[1]
        and (not zero_shift or all(value == 0 for value in key[2]))
        and _reciprocal_key(key) in mapping
    ]
    assert candidates

    def boundary_size(key: tuple[int, int, tuple[int, ...]]) -> float:
        cell_index, boundary_index = mapping[key]
        cell = cells[cell_index]
        boundary = cell[case.boundary][boundary_index]
        indices = np.asarray(boundary['vertices'], dtype=np.int64)
        vertices = np.asarray(cell['vertices'], dtype=np.float64)[indices]
        if case.shift_dim == 2:
            return float(np.linalg.norm(vertices[1] - vertices[0]))
        area_vector = 0.5 * np.sum(
            np.cross(vertices, np.roll(vertices, -1, axis=0)),
            axis=0,
        )
        return float(np.linalg.norm(area_vector))

    key = max(candidates, key=boundary_size)
    assert boundary_size(key) > 0.0
    return key, _reciprocal_key(key), mapping


@pytest.mark.parametrize('dim', [2, 3])
def test_required_and_optional_missing_shift_policy(dim: int) -> None:
    case = _case(dim)
    cells = _periodic_cells(case)
    for cell in cells:
        for boundary in cell.get(case.boundary) or []:
            boundary.pop('adjacent_shift', None)

    required = case.api.analyze_tessellation(cells, case.periodic_domain)
    optional = case.api.validate_tessellation(
        cells,
        case.periodic_domain,
        require_reciprocity=False,
    )

    required_issue = next(
        issue for issue in required.issues if issue.code == case.no_shift_code
    )
    optional_issue = next(
        issue for issue in optional.issues if issue.code == case.no_shift_code
    )
    assert required_issue.severity == 'error'
    assert required.ok_reciprocity is False
    assert required.ok is False
    assert optional_issue.severity == 'info'
    assert optional.ok_reciprocity is False
    assert optional.ok is True


@pytest.mark.parametrize('dim', [2, 3])
def test_partial_missing_shift_metadata_is_unavailable(dim: int) -> None:
    case = _case(dim)
    cells = _periodic_cells(case)
    key, reciprocal, mapping = _select_reciprocal_pair(
        cells,
        case,
        zero_shift=True,
    )
    assert key[0] != key[1]
    assert all(value == 0 for value in key[2])
    assert reciprocal in mapping

    for directed in (key, reciprocal):
        cell_index, boundary_index = mapping[directed]
        cells[cell_index][case.boundary][boundary_index].pop('adjacent_shift')

    relevant = [
        boundary
        for cell in cells
        for boundary in (cell.get(case.boundary) or [])
        if int(boundary.get('adjacent_cell', -1)) >= 0
    ]
    assert sum('adjacent_shift' not in boundary for boundary in relevant) == 2
    assert any('adjacent_shift' in boundary for boundary in relevant)

    required = case.api.analyze_tessellation(cells, case.periodic_domain)
    optional = case.api.validate_tessellation(
        cells,
        case.periodic_domain,
        require_reciprocity=False,
    )

    required_issue = next(
        issue for issue in required.issues if issue.code == case.no_shift_code
    )
    optional_issue = next(
        issue for issue in optional.issues if issue.code == case.no_shift_code
    )
    availability_field = (
        'face_shift_available' if dim == 3 else 'edge_shift_available'
    )
    assert required_issue.severity == 'error'
    assert getattr(required, availability_field) is False
    assert required.reciprocity_checked is False
    assert required.ok_reciprocity is False
    assert required.ok is False
    assert optional_issue.severity == 'info'
    assert getattr(optional, availability_field) is False
    assert optional.reciprocity_checked is False
    assert optional.ok_reciprocity is False
    assert optional.ok is True


@pytest.mark.parametrize('dim', [2, 3])
def test_wall_shift_metadata_is_not_required(dim: int) -> None:
    case = _case(dim)
    if dim == 3:
        domain = pyvoro2.OrthorhombicCell(
            ((0, 1), (0, 1), (0, 1)),
            periodic=(True, True, False),
        )
    else:
        domain = planar.RectangularCell(
            ((0, 1), (0, 1)),
            periodic=(True, False),
        )
    cells = case.api.compute(
        case.points,
        domain=domain,
        output='cells',
        **case.compute_geometry,
    )
    wall = next(
        boundary
        for cell in cells
        for boundary in (cell.get(case.boundary) or [])
        if int(boundary.get('adjacent_cell', -1)) < 0
    )
    wall.pop('adjacent_shift')

    diag = case.api.analyze_tessellation(cells, domain)

    availability_field = (
        'face_shift_available' if dim == 3 else 'edge_shift_available'
    )
    assert getattr(diag, availability_field) is True
    assert diag.reciprocity_checked is True
    assert diag.ok_reciprocity is True
    assert not any(issue.code == case.no_shift_code for issue in diag.issues)
    assert diag.ok is True


@pytest.mark.parametrize('dim', [2, 3])
def test_required_and_optional_orphan_policy_uses_direct_key_oracle(dim: int) -> None:
    case = _case(dim)
    cells = _periodic_cells(case)
    key, reciprocal, mapping = _select_reciprocal_pair(cells, case)
    cell_index, boundary_index = mapping[reciprocal]
    cells[cell_index][case.boundary].pop(boundary_index)
    direct_map = _direct_boundary_map(cells, case)
    assert key in direct_map
    assert reciprocal not in direct_map
    mismatch_options = {case.mismatch_option: False}

    required = case.api.analyze_tessellation(
        cells,
        case.periodic_domain,
        **mismatch_options,
    )
    optional = case.api.validate_tessellation(
        cells,
        case.periodic_domain,
        require_reciprocity=False,
        **{
            ('plane_angle_tol' if dim == 3 else 'line_angle_tol'): 0.0,
        },
    )

    required_issue = next(
        issue for issue in required.issues if issue.code == 'MISSING_RECIPROCAL'
    )
    optional_issue = next(
        issue for issue in optional.issues if issue.code == 'MISSING_RECIPROCAL'
    )
    assert required_issue.severity == 'error'
    assert required.ok is False
    assert optional_issue.severity == 'warning'
    assert optional.ok is True


def _move_boundary_off_geometry(
    cells: list[dict[str, Any]],
    case: DimensionCase,
    location: tuple[int, int],
) -> None:
    cell_index, boundary_index = location
    cell = cells[cell_index]
    boundary = cell[case.boundary][boundary_index]
    indices = np.asarray(boundary['vertices'], dtype=np.int64)
    vertices = np.asarray(cell['vertices'], dtype=np.float64)
    selected = vertices[indices]
    if case.shift_dim == 2:
        tangent = selected[1] - selected[0]
        normal = np.array([-tangent[1], tangent[0]])
    else:
        normal = np.zeros(3, dtype=np.float64)
        for index in range(selected.shape[0]):
            normal += np.cross(selected[index], selected[(index + 1) % len(selected)])
    normal /= np.linalg.norm(normal)
    vertices[np.unique(indices)] += 0.05 * normal
    cell['vertices'] = vertices.tolist()


@pytest.mark.parametrize('dim', [2, 3])
def test_required_and_optional_reciprocal_mismatch_policy(dim: int) -> None:
    case = _case(dim)
    cells = _periodic_cells(case)
    key, reciprocal, mapping = _select_reciprocal_pair(cells, case)
    assert key in mapping and reciprocal in mapping
    _move_boundary_off_geometry(cells, case, mapping[key])

    required = case.api.analyze_tessellation(cells, case.periodic_domain)
    optional = case.api.validate_tessellation(
        cells,
        case.periodic_domain,
        require_reciprocity=False,
    )

    required_issue = next(
        issue for issue in required.issues if issue.code == 'RECIPROCAL_MISMATCH'
    )
    optional_issue = next(
        issue for issue in optional.issues if issue.code == 'RECIPROCAL_MISMATCH'
    )
    assert required_issue.severity == 'error'
    assert required.ok is False
    assert optional_issue.severity == 'warning'
    assert optional.ok is True


@pytest.mark.parametrize('dim', [2, 3])
def test_repeated_marked_analysis_clears_owned_annotations(dim: int) -> None:
    case = _case(dim)
    cells = _periodic_cells(case)
    key, reciprocal, mapping = _select_reciprocal_pair(cells, case)
    reciprocal_cell, reciprocal_boundary = mapping[reciprocal]
    restored = cells[reciprocal_cell][case.boundary].pop(reciprocal_boundary)
    mismatch_options = {case.mismatch_option: False}

    first = case.api.analyze_tessellation(
        cells,
        case.periodic_domain,
        **mismatch_options,
    )
    current = _direct_boundary_map(cells, case)
    survivor_cell, survivor_boundary = current[key]
    survivor = cells[survivor_cell][case.boundary][survivor_boundary]
    assert survivor['orphan'] is True
    assert survivor['reciprocal_missing'] is True
    assert (
        first.n_faces_orphan if dim == 3 else first.n_edges_orphan
    ) >= 1

    for cell in cells:
        for boundary in cell.get(case.boundary) or []:
            boundary['reciprocal_mismatch'] = True
    cells[reciprocal_cell][case.boundary].insert(reciprocal_boundary, restored)
    second = case.api.analyze_tessellation(
        cells,
        case.periodic_domain,
        **mismatch_options,
    )

    assert (second.n_faces_orphan if dim == 3 else second.n_edges_orphan) == 0
    assert not any(
        boundary[field]
        for cell in cells
        for boundary in (cell.get(case.boundary) or [])
        for field in ('orphan', 'reciprocal_missing', 'reciprocal_mismatch')
    )


@pytest.mark.parametrize('dim', [2, 3])
def test_disabled_marking_preserves_caller_annotations(dim: int) -> None:
    case = _case(dim)
    cells = _periodic_cells(case)
    boundary = next(
        boundary
        for cell in cells
        for boundary in (cell.get(case.boundary) or [])
    )
    boundary.update(
        orphan=True,
        reciprocal_missing=True,
        reciprocal_mismatch=True,
    )

    case.api.analyze_tessellation(
        cells,
        case.periodic_domain,
        **{case.mark_option: False},
    )

    assert boundary['orphan'] is True
    assert boundary['reciprocal_missing'] is True
    assert boundary['reciprocal_mismatch'] is True


def test_planar_warning_only_normalized_topology_is_nonfatal() -> None:
    case = _case(2)
    topology = planar.normalize_topology(
        _periodic_cells(case),
        domain=case.periodic_domain,
    )
    bad_polygon = copy.deepcopy(topology)
    bad_polygon.cells[0]['edges'].pop()

    polygon_diag = planar.validate_normalized_topology(
        bad_polygon,
        case.periodic_domain,
        level='strict',
        check_vertex_edge_shift=False,
        check_edge_vertex_sets=False,
        check_incidence=False,
    )

    polygon_issue = next(
        issue for issue in polygon_diag.issues if issue.code == 'BAD_POLYGON_COUNT'
    )
    assert polygon_issue.severity == 'warning'
    assert polygon_diag.ok_polygon is False
    assert polygon_diag.ok is True

    low_incidence = copy.deepcopy(topology)
    low_incidence.global_edges.clear()
    incidence_diag = planar.validate_normalized_topology(
        low_incidence,
        case.periodic_domain,
        level='strict',
        check_vertex_edge_shift=False,
        check_edge_vertex_sets=False,
        check_polygon=False,
    )
    incidence_issue = next(
        issue
        for issue in incidence_diag.issues
        if issue.code == 'LOW_VERTEX_INCIDENCE'
    )
    assert incidence_issue.severity == 'warning'
    assert incidence_diag.ok_incidence is False
    assert incidence_diag.ok is True


def test_planar_normalized_error_findings_remain_fatal() -> None:
    case = _case(2)
    topology = planar.normalize_topology(
        _periodic_cells(case),
        domain=case.periodic_domain,
    )

    shift_mismatch = copy.deepcopy(topology)
    shift_location = next(
        (cell_index, vertex_index)
        for cell_index, cell in enumerate(shift_mismatch.cells)
        for vertex_index, shift in enumerate(cell['vertex_shift'])
        if tuple(shift) != (0, 0)
    )
    cell_index, vertex_index = shift_location
    shift_mismatch.cells[cell_index]['vertex_shift'][vertex_index] = (0, 0)
    shift_diag = planar.validate_normalized_topology(
        shift_mismatch,
        case.periodic_domain,
        level='basic',
    )
    shift_issue = next(
        issue
        for issue in shift_diag.issues
        if issue.code == 'VERTEX_EDGE_SHIFT_MISMATCH'
    )
    assert shift_issue.severity == 'error'
    assert shift_diag.ok is False
    with pytest.raises(planar.NormalizationError):
        planar.validate_normalized_topology(
            shift_mismatch,
            case.periodic_domain,
            level='strict',
        )

    set_mismatch = copy.deepcopy(topology)
    original_gid = int(set_mismatch.cells[0]['vertex_global_id'][0])
    replacement_gid = (original_gid + 1) % len(set_mismatch.global_vertices)
    set_mismatch.cells[0]['vertex_global_id'][0] = replacement_gid
    set_diag = planar.validate_normalized_topology(
        set_mismatch,
        case.periodic_domain,
        level='basic',
        check_vertex_edge_shift=False,
    )
    set_issue = next(
        issue
        for issue in set_diag.issues
        if issue.code == 'EDGE_VERTEX_SET_MISMATCH'
    )
    assert set_issue.severity == 'error'
    assert set_diag.ok is False


def _spatial_periodic_topology() -> tuple[object, object]:
    domain = pyvoro2.OrthorhombicCell(
        ((0, 9), (0, 9), (0, 9)),
        periodic=(True, True, True),
    )
    points = np.random.default_rng(2408).uniform(0.0, 9.0, size=(30, 3))
    topology = pyvoro2.normalize_topology(
        pyvoro2.compute(
            points,
            domain=domain,
            output='cells',
            return_vertices=True,
            return_faces=True,
            return_face_shifts=True,
        ),
        domain=domain,
    )
    return domain, topology


def test_spatial_warning_only_euler_mismatch_remains_nonfatal() -> None:
    domain, topology = _spatial_periodic_topology()
    topology.cells[0]['faces'].pop()

    diag = pyvoro2.validate_normalized_topology(
        topology,
        domain,
        level='strict',
        check_vertex_face_shift=False,
        check_face_vertex_sets=False,
        check_incidence=False,
    )

    issue = next(
        issue
        for issue in diag.issues
        if issue.code == 'EULER_CHARACTERISTIC_MISMATCH'
    )
    assert issue.severity == 'warning'
    assert diag.ok_euler is False
    assert diag.ok is True


def test_spatial_normalized_error_finding_remains_fatal() -> None:
    domain, topology = _spatial_periodic_topology()
    shift_location = next(
        (cell_index, vertex_index)
        for cell_index, cell in enumerate(topology.cells)
        for vertex_index, shift in enumerate(cell['vertex_shift'])
        if tuple(shift) != (0, 0, 0)
    )
    cell_index, vertex_index = shift_location
    topology.cells[cell_index]['vertex_shift'][vertex_index] = (0, 0, 0)

    diag = pyvoro2.validate_normalized_topology(
        topology,
        domain,
        level='basic',
        check_face_vertex_sets=False,
        check_incidence=False,
        check_euler=False,
    )

    issue = next(
        issue
        for issue in diag.issues
        if issue.code == 'VERTEX_FACE_SHIFT_MISMATCH'
    )
    assert issue.severity == 'error'
    assert diag.ok is False
    with pytest.raises(pyvoro2.NormalizationError):
        pyvoro2.validate_normalized_topology(
            topology,
            domain,
            level='strict',
            check_face_vertex_sets=False,
            check_incidence=False,
            check_euler=False,
        )


@pytest.mark.parametrize('dim', [2, 3])
@pytest.mark.parametrize('invalid', [-1.0, np.nan, np.inf, -np.inf])
def test_all_diagnostic_tolerances_reject_negative_or_nonfinite(
    dim: int,
    invalid: float,
) -> None:
    case = _case(dim)
    if dim == 3:
        options = (
            'volume_tol_rel',
            'volume_tol_abs',
            'plane_offset_tol',
            'plane_angle_tol',
        )
    else:
        options = (
            'area_tol_rel',
            'area_tol_abs',
            'line_offset_tol',
            'line_angle_tol',
        )
    for option in options:
        with pytest.raises(ValueError, match=option):
            case.api.analyze_tessellation(
                [_closed_cell(case)],
                case.domain,
                **{option: invalid},
            )


@pytest.mark.parametrize('dim', [2, 3])
@pytest.mark.parametrize('invalid', [0.0, -1.0, np.nan, np.inf, -np.inf])
def test_normalization_tolerance_remains_positive_and_finite(
    dim: int,
    invalid: float,
) -> None:
    case = _case(dim)
    with pytest.raises(ValueError, match='tol'):
        case.api.normalize_vertices([], domain=case.domain, tol=invalid)


def _diagnostics_with_error_and_passing_subchecks(dim: int) -> object:
    if dim == 3:
        issue = pyvoro2.TessellationIssue('SENTINEL_ERROR', 'error', 'sentinel')
        return pyvoro2.TessellationDiagnostics(
            1.0, 1.0, 1.0, 0.0, 0.0, 2, 2, (), (), False, False,
            0, 0, 0, (issue,), True, True, False,
        )
    issue = planar.TessellationIssue('SENTINEL_ERROR', 'error', 'sentinel')
    return planar.TessellationDiagnostics(
        1.0, 1.0, 1.0, 0.0, 0.0, 2, 2, (), (), False, False,
        0, 0, 0, (issue,), True, True, False,
    )


@pytest.mark.parametrize('dim', [2, 3])
def test_compute_warn_and_raise_consume_final_diag_ok(
    dim: int,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    case = _case(dim)
    diagnostic = _diagnostics_with_error_and_passing_subchecks(dim)
    module = api3d if dim == 3 else api2d
    monkeypatch.setattr(module, '_analyze_tessellation', lambda *a, **k: diagnostic)

    with warnings.catch_warnings(record=True) as diagnosed_warnings:
        diagnosed = case.api.compute(
            case.points[:2],
            domain=case.domain,
            tessellation_check='diagnose',
        )
    assert diagnosed_warnings == []
    assert diagnosed.tessellation_diagnostics is diagnostic

    with pytest.warns(UserWarning, match='tessellation_check failed') as records:
        case.api.compute(
            case.points[:2],
            domain=case.domain,
            tessellation_check='warn',
        )
    assert len(records) == 1

    with pytest.raises(case.api.TessellationError) as exc_info:
        case.api.compute(
            case.points[:2],
            domain=case.domain,
            tessellation_check='raise',
        )
    assert exc_info.value.diagnostics is diagnostic
