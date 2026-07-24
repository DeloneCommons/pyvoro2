"""Contract and integration coverage for direct forward power weights."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
import inspect
from typing import Any, Callable

import numpy as np
import pytest

import pyvoro2 as pv
import pyvoro2.api as api3d
import pyvoro2.planar as pv2
import pyvoro2.planar.api as api2d
from pyvoro2._power_input import resolve_power_input


Compute = Callable[..., Any]


@dataclass(frozen=True)
class ForwardCase:
    name: str
    compute: Compute
    points: np.ndarray
    domain: object
    measure_key: str
    boundary_key: str
    shift_size: int | None = None
    kwargs: dict[str, Any] = field(default_factory=dict)


CONTRACT_CASES = (
    ForwardCase(
        'planar',
        pv2.compute,
        np.array([[0.2, 0.3], [0.8, 0.7]], dtype=float),
        pv2.Box(((0.0, 1.0), (0.0, 1.0))),
        'area',
        'edges',
    ),
    ForwardCase(
        'spatial',
        pv.compute,
        np.array([[0.2, 0.3, 0.4], [0.8, 0.7, 0.6]], dtype=float),
        pv.Box(((0.0, 1.0), (0.0, 1.0), (0.0, 1.0))),
        'volume',
        'faces',
    ),
)


INTEGRATION_CASES = (
    ForwardCase(
        'planar-box',
        pv2.compute,
        np.array(
            [[0.2, 0.2], [0.75, 0.2], [0.35, 0.75], [0.8, 0.8]],
            dtype=float,
        ),
        pv2.Box(((0.0, 1.0), (0.0, 1.0))),
        'area',
        'edges',
    ),
    ForwardCase(
        'planar-rectangular-periodic',
        pv2.compute,
        np.array(
            [[0.12, 0.2], [0.75, 0.25], [0.35, 0.72], [0.88, 0.82]],
            dtype=float,
        ),
        pv2.RectangularCell(
            ((0.0, 1.0), (0.0, 1.0)),
            periodic=(True, True),
        ),
        'area',
        'edges',
        shift_size=2,
        kwargs={
            'return_edge_shifts': True,
            'edge_shift_search': 2,
            'return_diagnostics': True,
            'tessellation_check': 'raise',
        },
    ),
    ForwardCase(
        'spatial-box',
        pv.compute,
        np.array(
            [
                [0.2, 0.2, 0.2],
                [0.75, 0.2, 0.25],
                [0.3, 0.75, 0.3],
                [0.75, 0.75, 0.75],
                [0.2, 0.4, 0.8],
            ],
            dtype=float,
        ),
        pv.Box(((0.0, 1.0), (0.0, 1.0), (0.0, 1.0))),
        'volume',
        'faces',
    ),
    ForwardCase(
        'spatial-orthorhombic-periodic',
        pv.compute,
        np.array(
            [
                [0.12, 0.2, 0.2],
                [0.75, 0.2, 0.25],
                [0.3, 0.72, 0.3],
                [0.78, 0.78, 0.75],
                [0.2, 0.4, 0.82],
            ],
            dtype=float,
        ),
        pv.OrthorhombicCell(
            ((0.0, 1.0), (0.0, 1.0), (0.0, 1.0)),
            periodic=(True, True, False),
        ),
        'volume',
        'faces',
        shift_size=3,
        kwargs={
            'return_face_shifts': True,
            'face_shift_search': 2,
            'return_diagnostics': True,
            'tessellation_check': 'raise',
        },
    ),
    ForwardCase(
        'spatial-triclinic-periodic',
        pv.compute,
        np.array(
            [
                [0.12, 0.2, 0.2],
                [0.75, 0.2, 0.25],
                [0.3, 0.72, 0.3],
                [0.78, 0.78, 0.75],
                [0.2, 0.4, 0.82],
            ],
            dtype=float,
        ),
        pv.PeriodicCell(
            ((1.0, 0.0, 0.0), (0.2, 1.0, 0.0), (0.1, 0.15, 1.0))
        ),
        'volume',
        'faces',
        shift_size=3,
        kwargs={
            'return_face_shifts': True,
            'face_shift_search': 2,
            'return_diagnostics': True,
            'tessellation_check': 'raise',
        },
    ),
)


PORTABLE_PERIODIC_SHIFT_CASES = (
    ForwardCase(
        'planar-portable-periodic-power',
        pv2.compute,
        np.array([[0.1, 0.1], [0.1, 0.2]], dtype=float),
        pv2.RectangularCell(
            ((0.0, 1.0), (0.0, 1.0)),
            periodic=(True, True),
        ),
        'area',
        'edges',
        shift_size=2,
        kwargs={'return_edge_shifts': True},
    ),
    ForwardCase(
        'spatial-portable-periodic-power',
        pv.compute,
        np.array([[0.1, 0.1, 0.1], [0.1, 0.1, 0.2]], dtype=float),
        pv.PeriodicCell(
            ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))
        ),
        'volume',
        'faces',
        shift_size=3,
        kwargs={'return_face_shifts': True},
    ),
)


PORTABLE_PERIODIC_WEIGHTS = np.array([0.0, 1e6])


def _case_id(case: ForwardCase) -> str:
    return case.name


def _cells_and_diagnostics(result: Any) -> tuple[list[dict[str, Any]], Any | None]:
    if isinstance(result, pv.TessellationResult):
        return result.cells, result.tessellation_diagnostics
    if isinstance(result, tuple):
        cells, diagnostics = result
        return cells, diagnostics
    return result, None


def _cell_map(cells: list[dict[str, Any]]) -> dict[int, dict[str, Any]]:
    return {int(cell['id']): cell for cell in cells}


def _assert_native_periodic_topology_is_valid(
    cells: list[dict[str, Any]],
    case: ForwardCase,
) -> None:
    assert sum(float(cell[case.measure_key]) for cell in cells) == pytest.approx(
        1.0,
        rel=1e-9,
        abs=1e-12,
    )

    cells_by_id = _cell_map(cells)
    for cell in cells:
        if bool(cell.get('empty', False)):
            continue
        for boundary in cell.get(case.boundary_key, ()):
            adjacent_id = int(boundary['adjacent_cell'])
            if adjacent_id < 0:
                continue
            adjacent_cell = cells_by_id.get(adjacent_id)
            assert adjacent_cell is not None
            assert not bool(adjacent_cell.get('empty', False))


def _boundary_signature(
    cells: list[dict[str, Any]],
    *,
    boundary_key: str,
    shift_size: int | None,
) -> Counter[tuple[int, int, tuple[int, ...] | None]]:
    signature: Counter[tuple[int, int, tuple[int, ...] | None]] = Counter()
    for cell in cells:
        cell_id = int(cell['id'])
        for boundary in cell.get(boundary_key, ()):
            shift = None
            if shift_size is not None:
                shift = tuple(int(value) for value in boundary['adjacent_shift'])
                assert len(shift) == shift_size
            signature[(cell_id, int(boundary['adjacent_cell']), shift)] += 1
    return signature


def _boundary_geometry_signature(
    cells: list[dict[str, Any]],
    case: ForwardCase,
) -> Counter[
    tuple[
        int,
        int,
        tuple[int, ...] | None,
        tuple[tuple[float, ...], ...],
    ]
]:
    dimension = int(case.points.shape[1])
    signature: Counter[
        tuple[
            int,
            int,
            tuple[int, ...] | None,
            tuple[tuple[float, ...], ...],
        ]
    ] = Counter()
    for cell in cells:
        cell_id = int(cell['id'])
        vertices = np.round(
            np.asarray(cell.get('vertices', ()), dtype=float).reshape(
                (-1, dimension)
            ),
            decimals=9,
        )
        for boundary in cell.get(case.boundary_key, ()):
            shift = None
            if case.shift_size is not None:
                shift = tuple(
                    int(value) for value in boundary['adjacent_shift']
                )
            boundary_vertices = tuple(
                sorted(
                    tuple(float(value) for value in vertices[int(index)])
                    for index in boundary.get('vertices', ())
                )
            )
            signature[
                (
                    cell_id,
                    int(boundary['adjacent_cell']),
                    shift,
                    boundary_vertices,
                )
            ] += 1
    return signature


def _vertex_cloud(cell: dict[str, Any], dimension: int) -> np.ndarray:
    vertices = np.asarray(cell.get('vertices', ()), dtype=float).reshape(
        (-1, dimension)
    )
    if vertices.size == 0:
        return vertices
    # Quantize before sorting so roundoff in a nominally equal leading
    # coordinate cannot swap two otherwise distinct vertices.
    vertices = np.round(vertices, decimals=9)
    order = np.lexsort(
        tuple(vertices[:, axis] for axis in range(dimension - 1, -1, -1))
    )
    return vertices[order]


def _vertex_adjacency_signature(
    cell: dict[str, Any],
    dimension: int,
) -> set[tuple[tuple[float, ...], tuple[float, ...]]]:
    vertices = np.round(
        np.asarray(cell.get('vertices', ()), dtype=float).reshape(
            (-1, dimension)
        ),
        decimals=9,
    )
    signature: set[tuple[tuple[float, ...], tuple[float, ...]]] = set()
    for index, neighbors in enumerate(cell.get('adjacency', ())):
        source = tuple(float(value) for value in vertices[index])
        for neighbor in neighbors:
            target = tuple(float(value) for value in vertices[int(neighbor)])
            signature.add(tuple(sorted((source, target))))
    return signature


def _boundary_vertex_signature(
    cell: dict[str, Any],
    case: ForwardCase,
) -> Counter[tuple[tuple[float, ...], ...]]:
    dimension = int(case.points.shape[1])
    vertices = np.round(
        np.asarray(cell.get('vertices', ()), dtype=float).reshape(
            (-1, dimension)
        ),
        decimals=9,
    )
    return Counter(
        tuple(
            sorted(
                tuple(float(value) for value in vertices[int(index)])
                for index in boundary.get('vertices', ())
            )
        )
        for boundary in cell.get(case.boundary_key, ())
    )


def _assert_same_diagram(
    left: list[dict[str, Any]],
    right: list[dict[str, Any]],
    case: ForwardCase,
) -> None:
    """Compare order-independent geometric outputs at wrapper precision."""

    left_by_id = _cell_map(left)
    right_by_id = _cell_map(right)
    assert left_by_id.keys() == right_by_id.keys()
    assert _boundary_signature(
        left,
        boundary_key=case.boundary_key,
        shift_size=case.shift_size,
    ) == _boundary_signature(
        right,
        boundary_key=case.boundary_key,
        shift_size=case.shift_size,
    )
    assert _boundary_geometry_signature(
        left,
        case,
    ) == _boundary_geometry_signature(right, case)

    dimension = int(case.points.shape[1])
    for cell_id in sorted(left_by_id):
        cell_left = left_by_id[cell_id]
        cell_right = right_by_id[cell_id]
        assert bool(cell_left.get('empty', False)) is bool(
            cell_right.get('empty', False)
        )
        assert float(cell_left[case.measure_key]) == pytest.approx(
            float(cell_right[case.measure_key]),
            rel=1e-9,
            abs=1e-11,
        )
        np.testing.assert_allclose(
            _vertex_cloud(cell_left, dimension),
            _vertex_cloud(cell_right, dimension),
            rtol=1e-9,
            atol=1e-10,
        )
        assert _vertex_adjacency_signature(
            cell_left,
            dimension,
        ) == _vertex_adjacency_signature(cell_right, dimension)


def _assert_shift_annotation_preserves_diagram(
    unannotated: list[dict[str, Any]],
    annotated: list[dict[str, Any]],
    case: ForwardCase,
) -> None:
    unannotated_by_id = _cell_map(unannotated)
    annotated_by_id = _cell_map(annotated)
    assert unannotated_by_id.keys() == annotated_by_id.keys()

    dimension = int(case.points.shape[1])
    for cell_id in sorted(unannotated_by_id):
        cell_unannotated = unannotated_by_id[cell_id]
        cell_annotated = annotated_by_id[cell_id]
        assert bool(cell_unannotated.get('empty', False)) is bool(
            cell_annotated.get('empty', False)
        )
        assert float(cell_unannotated[case.measure_key]) == pytest.approx(
            float(cell_annotated[case.measure_key]),
            rel=1e-12,
            abs=1e-12,
        )
        np.testing.assert_allclose(
            cell_unannotated['site'],
            cell_annotated['site'],
            rtol=0.0,
            atol=0.0,
        )
        np.testing.assert_allclose(
            _vertex_cloud(cell_unannotated, dimension),
            _vertex_cloud(cell_annotated, dimension),
            rtol=0.0,
            atol=0.0,
        )
        assert _vertex_adjacency_signature(
            cell_unannotated,
            dimension,
        ) == _vertex_adjacency_signature(cell_annotated, dimension)
        assert _boundary_vertex_signature(
            cell_unannotated,
            case,
        ) == _boundary_vertex_signature(cell_annotated, case)


@pytest.mark.parametrize('case', CONTRACT_CASES, ids=_case_id)
def test_power_compute_requires_exactly_one_representation(
    case: ForwardCase,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = api2d if case.compute is pv2.compute else api3d

    def unexpected_core() -> Any:
        raise AssertionError('native core resolution must not be reached')

    require_name = '_require_core2d' if module is api2d else '_require_core'
    monkeypatch.setattr(module, require_name, unexpected_core)

    with pytest.raises(ValueError, match='exactly one'):
        case.compute(case.points, domain=case.domain, mode='power')
    with pytest.raises(ValueError, match='mutually exclusive'):
        case.compute(
            case.points,
            domain=case.domain,
            mode='power',
            weights=np.zeros(len(case.points)),
            radii=np.ones(len(case.points)),
        )


@pytest.mark.parametrize('case', CONTRACT_CASES, ids=_case_id)
@pytest.mark.parametrize(
    'weights',
    (
        np.zeros(1),
        np.zeros((2, 1)),
    ),
    ids=('wrong-length', 'wrong-rank'),
)
def test_power_compute_rejects_wrong_weight_shape(
    case: ForwardCase,
    weights: np.ndarray,
) -> None:
    with pytest.raises(ValueError, match=r'weights must have shape \(n,\)'):
        case.compute(
            case.points,
            domain=case.domain,
            mode='power',
            weights=weights,
        )


@pytest.mark.parametrize('case', CONTRACT_CASES, ids=_case_id)
@pytest.mark.parametrize('bad_value', (np.nan, np.inf, -np.inf))
def test_power_compute_rejects_nonfinite_weights(
    case: ForwardCase,
    bad_value: float,
) -> None:
    weights = np.zeros(len(case.points), dtype=float)
    weights[-1] = bad_value
    with pytest.raises(ValueError, match='weights must contain only finite values'):
        case.compute(
            case.points,
            domain=case.domain,
            mode='power',
            weights=weights,
        )


@pytest.mark.parametrize('case', CONTRACT_CASES, ids=_case_id)
def test_standard_compute_rejects_power_inputs_before_core(
    case: ForwardCase,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = api2d if case.compute is pv2.compute else api3d

    def unexpected_core() -> Any:
        raise AssertionError('native core resolution must not be reached')

    require_name = '_require_core2d' if module is api2d else '_require_core'
    monkeypatch.setattr(module, require_name, unexpected_core)

    with pytest.raises(ValueError, match='not supported.*standard'):
        case.compute(
            case.points,
            domain=case.domain,
            mode='standard',
            weights=np.zeros(len(case.points)),
        )

    with pytest.raises(ValueError, match='radii.*standard'):
        case.compute(
            case.points,
            domain=case.domain,
            mode='standard',
            radii=np.array([np.nan]),
        )


@pytest.mark.parametrize('case', CONTRACT_CASES, ids=_case_id)
def test_power_compute_rejects_unrepresentable_finite_weights(
    case: ForwardCase,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = api2d if case.compute is pv2.compute else api3d

    def unexpected_core() -> Any:
        raise AssertionError('native core resolution must not be reached')

    require_name = '_require_core2d' if module is api2d else '_require_core'
    monkeypatch.setattr(module, require_name, unexpected_core)
    limit = np.finfo(np.float64).max
    with pytest.raises(ValueError, match='non-finite'):
        case.compute(
            case.points,
            domain=case.domain,
            mode='power',
            weights=np.array([-limit, limit]),
        )


@pytest.mark.parametrize('case', PORTABLE_PERIODIC_SHIFT_CASES, ids=_case_id)
@pytest.mark.parametrize('representation', ('weights', 'radii'))
def test_portable_periodic_power_input_supports_image_shifts(
    case: ForwardCase,
    representation: str,
) -> None:
    base_kwargs: dict[str, Any] = {
        'return_vertices': True,
        'return_adjacency': True,
        'include_empty': True,
    }
    if case.boundary_key == 'edges':
        base_kwargs['return_edges'] = True
    else:
        base_kwargs['return_faces'] = True

    weights = PORTABLE_PERIODIC_WEIGHTS
    values = weights if representation == 'weights' else np.sqrt(weights)
    power_input = {representation: values}
    unannotated, _ = _cells_and_diagnostics(
        case.compute(
            case.points,
            domain=case.domain,
            mode='power',
            **power_input,
            **base_kwargs,
        )
    )
    _assert_native_periodic_topology_is_valid(unannotated, case)
    annotated, _ = _cells_and_diagnostics(
        case.compute(
            case.points,
            domain=case.domain,
            mode='power',
            **power_input,
            **base_kwargs,
            **case.kwargs,
        )
    )
    assert sum(
        float(cell[case.measure_key]) for cell in annotated
    ) == pytest.approx(1.0, rel=1e-9, abs=1e-12)

    _assert_shift_annotation_preserves_diagram(unannotated, annotated, case)

    resolved_shifts: list[tuple[int, int, tuple[int, ...]]] = []
    for cell in annotated:
        cell_id = int(cell['id'])
        for boundary in cell.get(case.boundary_key, ()):
            adjacent_id = int(boundary['adjacent_cell'])
            if adjacent_id < 0:
                continue
            shift = boundary['adjacent_shift']
            assert isinstance(shift, tuple)
            assert len(shift) == case.shift_size
            assert all(type(value) is int for value in shift)
            resolved_shifts.append((cell_id, adjacent_id, shift))

    assert resolved_shifts
    assert any(
        cell_id == adjacent_id and any(value != 0 for value in shift)
        for cell_id, adjacent_id, shift in resolved_shifts
    )


@pytest.mark.parametrize('case', CONTRACT_CASES, ids=_case_id)
def test_power_compute_accepts_empty_points_and_weights(case: ForwardCase) -> None:
    dimension = int(case.points.shape[1])
    cells, diagnostics = _cells_and_diagnostics(
        case.compute(
            np.empty((0, dimension), dtype=float),
            domain=case.domain,
            mode='power',
            weights=np.empty((0,), dtype=float),
        )
    )
    assert cells == []
    assert diagnostics is None


def test_shared_resolution_keeps_result_metadata_together() -> None:
    weights = np.array([-0.25, 0.0, 0.75])
    resolved = resolve_power_input(
        mode='power',
        weights=weights,
        radii=None,
        n=3,
    )
    np.testing.assert_array_equal(resolved.input_weights, weights)
    np.testing.assert_allclose(resolved.backend_radii, [0.0, 0.5, 1.0])
    assert resolved.representation_shift == pytest.approx(0.25)

    radii = np.array([0.1, 0.2, 0.3])
    resolved_radii = resolve_power_input(
        mode='power',
        weights=None,
        radii=radii,
        n=3,
    )
    assert resolved_radii.input_weights is None
    np.testing.assert_array_equal(resolved_radii.backend_radii, radii)
    assert resolved_radii.representation_shift is None


@pytest.mark.parametrize('case', INTEGRATION_CASES, ids=_case_id)
def test_weights_match_manual_global_radius_conversion(case: ForwardCase) -> None:
    weights = np.array([-0.08, 0.0, 0.05, 0.1, 0.02])[: len(case.points)]
    radii, _shift = pv.weights_to_radii(weights)
    ids = np.arange(101, 101 + len(case.points), dtype=int)

    from_weights, weights_diag = _cells_and_diagnostics(
        case.compute(
            case.points,
            domain=case.domain,
            ids=ids,
            mode='power',
            weights=weights,
            include_empty=True,
            **case.kwargs,
        )
    )
    from_radii, radii_diag = _cells_and_diagnostics(
        case.compute(
            case.points,
            domain=case.domain,
            ids=ids,
            mode='power',
            radii=radii,
            include_empty=True,
            **case.kwargs,
        )
    )

    assert set(_cell_map(from_weights)) == set(ids)
    _assert_same_diagram(from_weights, from_radii, case)
    if weights_diag is not None:
        assert weights_diag.ok is True
        assert radii_diag.ok is True


@pytest.mark.parametrize('case', INTEGRATION_CASES, ids=_case_id)
def test_global_weight_gauge_preserves_complete_diagram(case: ForwardCase) -> None:
    weights = np.array([-0.08, 0.0, 0.05, 0.1, 0.02])[: len(case.points)]
    shifted = weights + 0.75

    base, base_diag = _cells_and_diagnostics(
        case.compute(
            case.points,
            domain=case.domain,
            mode='power',
            weights=weights,
            include_empty=True,
            **case.kwargs,
        )
    )
    gauged, gauged_diag = _cells_and_diagnostics(
        case.compute(
            case.points,
            domain=case.domain,
            mode='power',
            weights=shifted,
            include_empty=True,
            **case.kwargs,
        )
    )

    _assert_same_diagram(base, gauged, case)
    if base_diag is not None:
        assert base_diag.ok is True
        assert gauged_diag.ok is True


@pytest.mark.parametrize(
    ('case', 'weights', 'radii'),
    (
        (
            ForwardCase(
                'planar-hidden',
                pv2.compute,
                np.array([[0.1, 0.5], [0.9, 0.5]], dtype=float),
                pv2.RectangularCell(
                    ((0.0, 1.0), (0.0, 1.0)), periodic=(True, True)
                ),
                'area',
                'edges',
                shift_size=2,
                kwargs={'return_edge_shifts': True},
            ),
            np.array([1.0, 4.0]),
            np.array([1.0, 2.0]),
        ),
        (
            ForwardCase(
                'spatial-hidden',
                pv.compute,
                np.array([[0.1, 0.5, 0.5], [0.9, 0.5, 0.5]], dtype=float),
                pv.PeriodicCell(
                    ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))
                ),
                'volume',
                'faces',
                shift_size=3,
                kwargs={'return_face_shifts': True, 'face_shift_search': 1},
            ),
            np.array([1.0, 4.0]),
            np.array([1.0, 2.0]),
        ),
    ),
    ids=lambda value: value.name if isinstance(value, ForwardCase) else None,
)
def test_weight_input_preserves_hidden_cells_and_periodic_shifts(
    case: ForwardCase,
    weights: np.ndarray,
    radii: np.ndarray,
) -> None:
    weighted, _ = _cells_and_diagnostics(
        case.compute(
            case.points,
            domain=case.domain,
            ids=[17, 29],
            mode='power',
            weights=weights,
            include_empty=True,
            **case.kwargs,
        )
    )
    radius_based, _ = _cells_and_diagnostics(
        case.compute(
            case.points,
            domain=case.domain,
            ids=[17, 29],
            mode='power',
            radii=radii,
            include_empty=True,
            **case.kwargs,
        )
    )
    gauged, _ = _cells_and_diagnostics(
        case.compute(
            case.points,
            domain=case.domain,
            ids=[17, 29],
            mode='power',
            weights=weights - 9.0,
            include_empty=True,
            **case.kwargs,
        )
    )

    assert _cell_map(weighted)[17].get('empty') is True
    assert float(_cell_map(weighted)[17][case.measure_key]) == 0.0
    _assert_same_diagram(weighted, radius_based, case)
    _assert_same_diagram(weighted, gauged, case)


def test_weights_are_added_only_to_compute_signatures() -> None:
    for compute in (pv.compute, pv2.compute):
        parameters = inspect.signature(compute).parameters
        names = tuple(parameters)
        assert names.index('weights') + 1 == names.index('radii')
        assert parameters['weights'].kind is inspect.Parameter.KEYWORD_ONLY
        assert parameters['weights'].default is None
    assert 'weights' not in inspect.signature(pv.locate).parameters
    assert 'weights' not in inspect.signature(pv2.locate).parameters
    assert 'weights' not in inspect.signature(pv.ghost_cells).parameters
    assert 'weights' not in inspect.signature(pv2.ghost_cells).parameters
