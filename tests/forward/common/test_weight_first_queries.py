"""Public contracts for weight-first locate and ghost-cell queries."""

from __future__ import annotations

from dataclasses import dataclass
import inspect
from typing import Any

import numpy as np
import pytest

import pyvoro2 as pv
import pyvoro2.api as api3d
import pyvoro2.planar as pv2
import pyvoro2.planar.api as api2d


@dataclass(frozen=True)
class QueryCase:
    name: str
    module: Any
    api_module: Any
    points: np.ndarray
    queries: np.ndarray
    domain: object
    measure_key: str
    boundary_flag: str


CASES = (
    QueryCase(
        'planar',
        pv2,
        api2d,
        np.array([[0.25, 0.5], [0.75, 0.5]], dtype=float),
        np.array([[0.2, 0.5], [0.8, 0.5]], dtype=float),
        pv2.Box(((0.0, 1.0), (0.0, 1.0))),
        'area',
        'return_edges',
    ),
    QueryCase(
        'planar-periodic',
        pv2,
        api2d,
        np.array([[0.25, 0.5], [0.75, 0.5]], dtype=float),
        np.array([[0.2, 0.5], [0.8, 0.5]], dtype=float),
        pv2.RectangularCell(
            ((0.0, 1.0), (0.0, 1.0)),
            periodic=(True, True),
        ),
        'area',
        'return_edges',
    ),
    QueryCase(
        'spatial-box',
        pv,
        api3d,
        np.array([[0.25, 0.5, 0.5], [0.75, 0.5, 0.5]], dtype=float),
        np.array([[0.2, 0.5, 0.5], [0.8, 0.5, 0.5]], dtype=float),
        pv.Box(((0.0, 1.0), (0.0, 1.0), (0.0, 1.0))),
        'volume',
        'return_faces',
    ),
    QueryCase(
        'spatial-periodic',
        pv,
        api3d,
        np.array([[0.25, 0.5, 0.5], [0.75, 0.5, 0.5]], dtype=float),
        np.array([[0.2, 0.5, 0.5], [0.8, 0.5, 0.5]], dtype=float),
        pv.OrthorhombicCell(
            ((0.0, 1.0), (0.0, 1.0), (0.0, 1.0)),
            periodic=(True, True, True),
        ),
        'volume',
        'return_faces',
    ),
    QueryCase(
        'spatial-periodic-cell',
        pv,
        api3d,
        np.array([[0.25, 0.5, 0.5], [0.75, 0.5, 0.5]], dtype=float),
        np.array([[0.2, 0.5, 0.5], [0.8, 0.5, 0.5]], dtype=float),
        pv.PeriodicCell(
            vectors=((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)),
        ),
        'volume',
        'return_faces',
    ),
)


def _case_id(case: QueryCase) -> str:
    return case.name


def _manual_combined_radii(
    weights: np.ndarray,
    ghost_weights: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Independent one-gauge oracle for the backend-radius representation."""

    combined = np.concatenate((weights, ghost_weights))
    shifted = combined - float(np.min(combined))
    radii = np.sqrt(shifted)
    return radii[: len(weights)], radii[len(weights) :]


def _ghost_call(
    case: QueryCase,
    *,
    points: np.ndarray | None = None,
    queries: np.ndarray | None = None,
    **kwargs: object,
) -> list[dict[str, Any]]:
    options: dict[str, object] = {
        'return_vertices': True,
        'return_adjacency': False,
        case.boundary_flag: False,
    }
    options.update(kwargs)
    return case.module.ghost_cells(
        case.points if points is None else points,
        case.queries if queries is None else queries,
        domain=case.domain,
        **options,
    )


def _sorted_vertices(cell: dict[str, Any]) -> np.ndarray:
    vertices = np.asarray(cell['vertices'], dtype=float)
    if not len(vertices):
        return vertices
    keys = tuple(vertices[:, axis] for axis in reversed(range(vertices.shape[1])))
    return vertices[np.lexsort(keys)]


def _assert_same_unit_periodic_vertices(
    left: np.ndarray,
    right: np.ndarray,
) -> None:
    assert left.shape == right.shape
    if not len(left):
        return

    delta = left[:, None, :] - right[None, :, :]
    residual = delta - np.rint(delta)
    equivalent = np.all(
        np.isclose(residual, 0.0, rtol=0.0, atol=1e-12),
        axis=2,
    )
    matched_left = np.full(len(right), -1, dtype=int)

    def match(left_index: int, seen: np.ndarray) -> bool:
        for right_index in np.flatnonzero(equivalent[left_index]):
            if seen[right_index]:
                continue
            seen[right_index] = True
            previous = matched_left[right_index]
            if previous < 0 or match(int(previous), seen):
                matched_left[right_index] = left_index
                return True
        return False

    assert all(
        match(index, np.zeros(len(right), dtype=bool))
        for index in range(len(left))
    )


def test_unit_periodic_vertices_accept_lattice_image_representatives() -> None:
    left = np.array([[0.25, 0.0, 0.5], [0.75, 0.5, 0.0]])
    right = np.array([[0.75, 0.5, 1.0], [0.25, 1.0, 0.5]])

    _assert_same_unit_periodic_vertices(left, right)


def test_unit_periodic_vertices_reject_noninteger_displacement() -> None:
    left = np.array([[0.25, 0.0, 0.5], [0.25, 1.0, 0.5]])
    right = np.array([[0.25, 0.0, 0.5], [0.25, 0.75, 0.5]])

    with pytest.raises(AssertionError):
        _assert_same_unit_periodic_vertices(left, right)


def _assert_same_ghost_geometry(
    left: list[dict[str, Any]],
    right: list[dict[str, Any]],
    case: QueryCase,
) -> None:
    assert len(left) == len(right)
    for left_cell, right_cell in zip(left, right, strict=True):
        assert left_cell['query_index'] == right_cell['query_index']
        assert bool(left_cell['empty']) is bool(right_cell['empty'])
        assert float(left_cell[case.measure_key]) == pytest.approx(
            float(right_cell[case.measure_key]),
            rel=1e-12,
            abs=1e-12,
        )
        left_vertices = _sorted_vertices(left_cell)
        right_vertices = _sorted_vertices(right_cell)
        if case.name == 'spatial-periodic-cell':
            _assert_same_unit_periodic_vertices(left_vertices, right_vertices)
        else:
            np.testing.assert_allclose(
                left_vertices,
                right_vertices,
                rtol=1e-12,
                atol=1e-12,
            )


@pytest.mark.parametrize('case', CASES, ids=_case_id)
def test_locate_weights_match_explicit_radius_oracle(case: QueryCase) -> None:
    weights = np.array([-0.04, 0.08])
    shifted = weights - float(np.min(weights))
    radii = np.sqrt(shifted)

    weighted = case.module.locate(
        case.points,
        case.queries,
        domain=case.domain,
        mode='power',
        weights=weights,
        return_owner_position=True,
    )
    radius_based = case.module.locate(
        case.points,
        case.queries,
        domain=case.domain,
        mode='power',
        radii=radii,
        return_owner_position=True,
    )

    np.testing.assert_array_equal(weighted['found'], radius_based['found'])
    np.testing.assert_array_equal(weighted['owner_id'], radius_based['owner_id'])
    np.testing.assert_allclose(weighted['owner_pos'], radius_based['owner_pos'])


@pytest.mark.parametrize('case', CASES, ids=_case_id)
def test_locate_common_weight_shift_preserves_owners(case: QueryCase) -> None:
    weights = np.array([-0.04, 0.08])
    base = case.module.locate(
        case.points,
        case.queries,
        domain=case.domain,
        mode='power',
        weights=weights,
    )
    shifted = case.module.locate(
        case.points,
        case.queries,
        domain=case.domain,
        mode='power',
        weights=weights + 7.25,
    )

    np.testing.assert_array_equal(base['found'], shifted['found'])
    np.testing.assert_array_equal(base['owner_id'], shifted['owner_id'])


@pytest.mark.parametrize('case', CASES, ids=_case_id)
def test_locate_rejects_missing_or_ambiguous_power_input(case: QueryCase) -> None:
    with pytest.raises(ValueError, match='exactly one'):
        case.module.locate(
            case.points,
            case.queries,
            domain=case.domain,
            mode='power',
        )
    with pytest.raises(ValueError, match='mutually exclusive'):
        case.module.locate(
            case.points,
            case.queries,
            domain=case.domain,
            mode='power',
            weights=np.zeros(2),
            radii=np.zeros(2),
        )


@pytest.mark.parametrize('case', CASES, ids=_case_id)
@pytest.mark.parametrize('argument', ['weights', 'radii'])
def test_locate_standard_mode_rejects_power_input(
    case: QueryCase,
    argument: str,
) -> None:
    with pytest.raises(ValueError, match=argument):
        case.module.locate(
            case.points,
            case.queries,
            domain=case.domain,
            **{argument: np.zeros(2)},
        )


@pytest.mark.parametrize('case', CASES, ids=_case_id)
@pytest.mark.parametrize(
    ('weights', 'message'),
    (
        (np.zeros((2, 1)), 'shape'),
        (np.array([0.0, np.nan]), 'finite'),
    ),
)
def test_locate_rejects_malformed_weights_before_native_dispatch(
    monkeypatch: pytest.MonkeyPatch,
    case: QueryCase,
    weights: np.ndarray,
    message: str,
) -> None:
    def reject_native() -> None:
        raise AssertionError('native dispatch was attempted')

    native_loader = (
        '_require_core2d' if case.module is pv2 else '_require_core'
    )
    monkeypatch.setattr(case.api_module, native_loader, reject_native)
    with pytest.raises(ValueError, match=f'weights.*{message}'):
        case.module.locate(
            case.points,
            case.queries,
            domain=case.domain,
            mode='power',
            weights=weights,
        )


@pytest.mark.parametrize('case', CASES, ids=_case_id)
def test_ghost_weights_match_one_combined_gauge_radius_oracle(
    case: QueryCase,
) -> None:
    weights = np.array([-0.06, 0.09])
    ghost_weights = np.array([0.03, -0.02])
    radii, ghost_radii = _manual_combined_radii(weights, ghost_weights)

    weighted = _ghost_call(
        case,
        mode='power',
        weights=weights,
        ghost_weights=ghost_weights,
    )
    radius_based = _ghost_call(
        case,
        mode='power',
        radii=radii,
        ghost_radii=ghost_radii,
    )

    _assert_same_ghost_geometry(weighted, radius_based, case)


@pytest.mark.parametrize('case', CASES, ids=_case_id)
def test_ghost_common_weight_shift_preserves_geometry(case: QueryCase) -> None:
    weights = np.array([-0.06, 0.09])
    ghost_weights = np.array([0.03, -0.02])

    base = _ghost_call(
        case,
        mode='power',
        weights=weights,
        ghost_weights=ghost_weights,
    )
    shifted = _ghost_call(
        case,
        mode='power',
        weights=weights + 4.5,
        ghost_weights=ghost_weights + 4.5,
    )

    _assert_same_ghost_geometry(base, shifted, case)


@pytest.mark.parametrize('case', CASES, ids=_case_id)
@pytest.mark.parametrize('shifted_family', ['persistent', 'ghost'])
def test_ghost_independent_weight_shift_changes_geometry(
    case: QueryCase,
    shifted_family: str,
) -> None:
    dimension = case.points.shape[1]
    points = np.full((1, dimension), 0.5)
    queries = np.full((1, dimension), 0.5)
    points[0, 0] = 0.25
    queries[0, 0] = 0.75
    weights = np.array([0.0])
    ghost_weights = np.array([0.0])

    base = _ghost_call(
        case,
        points=points,
        queries=queries,
        mode='power',
        weights=weights,
        ghost_weights=ghost_weights,
    )
    if shifted_family == 'persistent':
        weights = weights + 0.1
    else:
        ghost_weights = ghost_weights + 0.1
    independently_shifted = _ghost_call(
        case,
        points=points,
        queries=queries,
        mode='power',
        weights=weights,
        ghost_weights=ghost_weights,
    )

    assert float(base[0][case.measure_key]) != pytest.approx(
        float(independently_shifted[0][case.measure_key]),
        rel=1e-10,
        abs=1e-10,
    )


INVALID_GHOST_FAMILIES = (
    {},
    {'weights': np.zeros(2)},
    {'ghost_weights': np.zeros(2)},
    {'radii': np.zeros(2)},
    {'ghost_radii': np.zeros(2)},
    {'weights': np.zeros(2), 'radii': np.zeros(2)},
    {'ghost_weights': np.zeros(2), 'ghost_radii': np.zeros(2)},
    {'weights': np.zeros(2), 'ghost_radii': np.zeros(2)},
    {'radii': np.zeros(2), 'ghost_weights': np.zeros(2)},
    {
        'weights': np.zeros(2),
        'radii': np.zeros(2),
        'ghost_weights': np.zeros(2),
    },
    {
        'weights': np.zeros(2),
        'ghost_weights': np.zeros(2),
        'ghost_radii': np.zeros(2),
    },
    {
        'weights': np.zeros(2),
        'radii': np.zeros(2),
        'ghost_weights': np.zeros(2),
        'ghost_radii': np.zeros(2),
    },
)


@pytest.mark.parametrize('case', CASES, ids=_case_id)
@pytest.mark.parametrize('power_input', INVALID_GHOST_FAMILIES)
def test_ghost_rejects_incomplete_or_mixed_power_families(
    case: QueryCase,
    power_input: dict[str, np.ndarray],
) -> None:
    with pytest.raises(ValueError, match='complete.*family'):
        _ghost_call(case, mode='power', **power_input)


@pytest.mark.parametrize('case', CASES, ids=_case_id)
@pytest.mark.parametrize(
    'argument',
    ['weights', 'radii', 'ghost_weights', 'ghost_radii'],
)
def test_ghost_standard_mode_rejects_every_power_input(
    case: QueryCase,
    argument: str,
) -> None:
    with pytest.raises(ValueError, match=argument):
        _ghost_call(case, **{argument: np.zeros(2)})


@pytest.mark.parametrize('case', CASES, ids=_case_id)
@pytest.mark.parametrize('family', ['weights', 'radii'])
@pytest.mark.parametrize('temporary_shape', ['scalar', 'per-query'])
def test_ghost_temporary_power_input_accepts_scalar_or_per_query_values(
    case: QueryCase,
    family: str,
    temporary_shape: str,
) -> None:
    temporary: float | np.ndarray = (
        0.02 if temporary_shape == 'scalar' else np.array([0.02, 0.04])
    )
    kwargs = (
        {'weights': np.array([0.0, 0.05]), 'ghost_weights': temporary}
        if family == 'weights'
        else {'radii': np.array([0.0, 0.05]), 'ghost_radii': temporary}
    )

    cells = _ghost_call(case, mode='power', **kwargs)

    assert [cell['query_index'] for cell in cells] == [0, 1]


@pytest.mark.parametrize('case', CASES, ids=_case_id)
@pytest.mark.parametrize(
    ('power_input', 'message'),
    (
        (
            {'weights': np.zeros((2, 1)), 'ghost_weights': 0.0},
            'weights.*shape',
        ),
        (
            {'weights': np.zeros(2), 'ghost_weights': np.nan},
            'ghost_weights.*finite',
        ),
        (
            {'radii': np.array([0.0, -1.0]), 'ghost_radii': 0.0},
            'radii.*non-negative',
        ),
        (
            {'radii': np.zeros(2), 'ghost_radii': -1.0},
            'ghost_radii.*non-negative',
        ),
    ),
)
def test_ghost_rejects_invalid_power_input_before_native_dispatch(
    monkeypatch: pytest.MonkeyPatch,
    case: QueryCase,
    power_input: dict[str, object],
    message: str,
) -> None:
    def reject_native() -> None:
        raise AssertionError('native dispatch was attempted')

    native_loader = (
        '_require_core2d' if case.module is pv2 else '_require_core'
    )
    monkeypatch.setattr(case.api_module, native_loader, reject_native)
    with pytest.raises(ValueError, match=message):
        _ghost_call(case, mode='power', **power_input)


@pytest.mark.parametrize('case', CASES, ids=_case_id)
def test_removed_ghost_radius_is_absent_and_unsupported(case: QueryCase) -> None:
    assert 'ghost_radius' not in inspect.signature(
        case.module.ghost_cells
    ).parameters
    with pytest.raises(TypeError, match='ghost_radius'):
        _ghost_call(
            case,
            mode='power',
            radii=np.zeros(2),
            ghost_radius=0.0,
        )
