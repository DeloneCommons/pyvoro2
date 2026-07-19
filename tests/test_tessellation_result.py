"""Focused contract tests for the common tessellation result."""

from __future__ import annotations

import copy
from dataclasses import FrozenInstanceError, fields, replace
import pickle
from typing import Any, Callable

import numpy as np
import pytest

import pyvoro2 as pv
import pyvoro2.planar as pv2
from pyvoro2._power_input import ResolvedPowerInput, resolve_power_input
from pyvoro2.result import _build_tessellation_result


def _standard_input(n: int) -> ResolvedPowerInput:
    return resolve_power_input(
        mode='standard',
        weights=None,
        radii=None,
        n=n,
    )


def _build(
    *,
    dimension: int,
    domain: object,
    mode: str,
    sites: np.ndarray,
    cells: list[dict[str, Any]],
    ids: Any = None,
    power_input: ResolvedPowerInput | None = None,
    **kwargs: Any,
) -> pv.TessellationResult:
    if power_input is None:
        power_input = _standard_input(len(sites))
    return _build_tessellation_result(
        dimension=dimension,  # type: ignore[arg-type]
        domain=domain,
        mode=mode,  # type: ignore[arg-type]
        sites=sites,
        ids=ids,
        cells=cells,
        power_input=power_input,
        boundaries_available=kwargs.pop('boundaries_available', False),
        periodic_shifts_available=kwargs.pop(
            'periodic_shifts_available', False
        ),
        **kwargs,
    )


def test_public_imports_are_the_identical_class() -> None:
    assert pv.TessellationResult is pv2.TessellationResult
    assert 'TessellationResult' in pv.__all__
    assert 'TessellationResult' in pv2.__all__
    assert pv2.PlanarComputeResult is not pv.TessellationResult

    public_fields = tuple(
        item.name for item in fields(pv.TessellationResult)
        if not item.name.startswith('_')
    )
    assert public_fields == (
        'dimension',
        'domain',
        'mode',
        'sites',
        'ids',
        'cells',
        'cell_measures',
        'empty_mask',
        'input_weights',
        'backend_radii',
        'representation_shift',
        'tessellation_diagnostics',
        'normalized_vertices',
        'normalized_topology',
    )


def test_capability_state_is_preserved_by_replace() -> None:
    domain = pv2.RectangularCell(
        ((0.0, 1.0), (0.0, 1.0)), periodic=(True, True)
    )
    cells = [
        {
            'id': 7,
            'area': 1.0,
            'edges': [
                {
                    'adjacent_cell': 7,
                    'vertices': [],
                    'adjacent_shift': (1, 0),
                }
            ],
        }
    ]
    result = pv.TessellationResult(
        dimension=2,
        domain=domain,
        mode='standard',
        sites=np.array([[0.5, 0.5]]),
        ids=np.array([7]),
        cells=cells,
        cell_measures=np.array([1.0]),
        empty_mask=np.array([False]),
        _boundaries_available=True,
        _periodic_shifts_available=True,
    )

    assert result.has_boundaries is True
    assert result.has_periodic_shifts is True
    copied = replace(result)
    assert copied.has_boundaries is True
    assert copied.has_periodic_shifts is True
    assert copied.require_boundaries()[0] is cells[0]['edges']


@pytest.mark.parametrize(
    ('dimension', 'compute', 'domain', 'points', 'measure_key'),
    (
        (
            2,
            pv2.compute,
            pv2.Box(((0.0, 1.0), (0.0, 1.0))),
            np.array([[0.2, 0.5], [0.5, 0.5], [0.8, 0.5]]),
            'area',
        ),
        (
            3,
            pv.compute,
            pv.Box(((0.0, 1.0), (0.0, 1.0), (0.0, 1.0))),
            np.array(
                [
                    [0.2, 0.5, 0.5],
                    [0.5, 0.5, 0.5],
                    [0.8, 0.5, 0.5],
                ]
            ),
            'volume',
        ),
    ),
    ids=('planar', 'spatial'),
)
def test_real_raw_computations_align_by_external_id_not_cell_order(
    dimension: int,
    compute: Callable[..., Any],
    domain: object,
    points: np.ndarray,
    measure_key: str,
) -> None:
    ids = np.array([30, 10, 20], dtype=np.int64)
    cells = compute(points, domain=domain, ids=ids)
    expected_measures = {
        int(cell['id']): float(cell[measure_key]) for cell in cells
    }
    cells.reverse()

    result = _build(
        dimension=dimension,
        domain=domain,
        mode='standard',
        sites=points,
        ids=ids,
        cells=cells,
        boundaries_available=True,
    )

    assert result.cells is cells
    np.testing.assert_array_equal(result.sites, points)
    np.testing.assert_array_equal(result.ids, ids)
    np.testing.assert_allclose(
        result.cell_measures,
        [expected_measures[int(cell_id)] for cell_id in ids],
    )
    np.testing.assert_array_equal(result.empty_mask, [False, False, False])
    assert result.measure_kind == measure_key
    assert result.boundary_kind == ('edges' if dimension == 2 else 'faces')
    assert len(result.require_boundaries()) == len(points)


MATRIX_POINTS_2D = np.array(
    [[0.15, 0.2], [0.72, 0.22], [0.35, 0.75], [0.82, 0.8]]
)
MATRIX_POINTS_3D = np.array(
    [
        [0.15, 0.2, 0.25],
        [0.72, 0.22, 0.3],
        [0.35, 0.75, 0.4],
        [0.82, 0.8, 0.75],
    ]
)
MATRIX_WEIGHTS = np.array([-0.04, 0.0, 0.03, 0.07])


@pytest.mark.parametrize(
    (
        'dimension',
        'compute',
        'domain',
        'points',
        'mode',
        'weights',
        'shift_option',
    ),
    (
        (
            2,
            pv2.compute,
            pv2.Box(((0.0, 1.0), (0.0, 1.0))),
            MATRIX_POINTS_2D,
            'standard',
            None,
            None,
        ),
        (
            2,
            pv2.compute,
            pv2.RectangularCell(
                ((0.0, 1.0), (0.0, 1.0)), periodic=(True, False)
            ),
            MATRIX_POINTS_2D,
            'power',
            MATRIX_WEIGHTS,
            'return_edge_shifts',
        ),
        (
            3,
            pv.compute,
            pv.Box(((0.0, 1.0), (0.0, 1.0), (0.0, 1.0))),
            MATRIX_POINTS_3D,
            'standard',
            None,
            None,
        ),
        (
            3,
            pv.compute,
            pv.OrthorhombicCell(
                ((0.0, 1.0), (0.0, 1.0), (0.0, 1.0)),
                periodic=(True, False, True),
            ),
            MATRIX_POINTS_3D,
            'power',
            MATRIX_WEIGHTS,
            'return_face_shifts',
        ),
        (
            3,
            pv.compute,
            pv.PeriodicCell(
                ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))
            ),
            MATRIX_POINTS_3D,
            'standard',
            None,
            'return_face_shifts',
        ),
        (
            3,
            pv.compute,
            pv.PeriodicCell(
                ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))
            ),
            MATRIX_POINTS_3D,
            'power',
            MATRIX_WEIGHTS,
            'return_face_shifts',
        ),
    ),
    ids=(
        'planar-bounded-standard',
        'planar-partial-periodic-power',
        'spatial-bounded-standard',
        'spatial-partial-periodic-power',
        'spatial-periodic-standard',
        'spatial-periodic-power',
    ),
)
def test_forward_result_matrix_covers_domains_modes_and_diagnostics(
    dimension: int,
    compute: Callable[..., Any],
    domain: object,
    points: np.ndarray,
    mode: str,
    weights: np.ndarray | None,
    shift_option: str | None,
) -> None:
    ids = np.array([44, 11, 33, 22])
    compute_kwargs: dict[str, Any] = {'return_diagnostics': True}
    if shift_option is not None:
        compute_kwargs[shift_option] = True
    if weights is not None:
        compute_kwargs['weights'] = weights

    cells, diagnostics = compute(
        points,
        domain=domain,
        ids=ids,
        mode=mode,
        **compute_kwargs,
    )
    cells.reverse()
    power_input = resolve_power_input(
        mode=mode,
        weights=weights,
        radii=None,
        n=len(points),
    )
    result = _build(
        dimension=dimension,
        domain=domain,
        mode=mode,
        sites=points,
        ids=ids,
        cells=cells,
        power_input=power_input,
        tessellation_diagnostics=diagnostics,
        boundaries_available=True,
        periodic_shifts_available=shift_option is not None,
    )

    np.testing.assert_array_equal(result.ids, ids)
    np.testing.assert_array_equal(
        result.empty_mask, np.zeros(len(points), dtype=bool)
    )
    assert np.all(result.cell_measures > 0.0)
    assert result.require_tessellation_diagnostics() is diagnostics
    assert result.has_boundaries is True
    assert result.has_periodic_shifts is (shift_option is not None)
    boundaries = result.require_boundaries()
    assert len(boundaries) == len(points)
    if shift_option is not None:
        assert all(
            'adjacent_shift' in boundary
            for collection in boundaries
            for boundary in collection
        )

    assert result.has_normalized_vertices is False
    assert result.has_normalized_topology is False
    if dimension == 3:
        with pytest.raises(ValueError, match='normalized vertices'):
            result.require_normalized_vertices()
        with pytest.raises(ValueError, match='normalized topology'):
            result.require_normalized_topology()


@pytest.mark.parametrize(
    ('dimension', 'compute', 'domain', 'points', 'measure_key', 'geometry'),
    (
        (
            2,
            pv2.compute,
            pv2.Box(((0.0, 1.0), (0.0, 1.0))),
            np.array([[0.25, 0.5], [0.75, 0.5]]),
            'area',
            {
                'return_vertices': False,
                'return_adjacency': False,
                'return_edges': False,
            },
        ),
        (
            3,
            pv.compute,
            pv.Box(((0.0, 1.0), (0.0, 1.0), (0.0, 1.0))),
            np.array([[0.25, 0.5, 0.5], [0.75, 0.5, 0.5]]),
            'volume',
            {
                'return_vertices': False,
                'return_adjacency': False,
                'return_faces': False,
            },
        ),
    ),
    ids=('planar', 'spatial'),
)
def test_omitted_and_explicit_hidden_power_cells_align_identically(
    dimension: int,
    compute: Callable[..., Any],
    domain: object,
    points: np.ndarray,
    measure_key: str,
    geometry: dict[str, bool],
) -> None:
    ids = np.array([20, 10])
    radii = np.array([1.0, 2.0])
    power_input = resolve_power_input(
        mode='power',
        weights=None,
        radii=radii,
        n=2,
    )
    common = {
        'domain': domain,
        'ids': ids,
        'mode': 'power',
        'radii': radii,
        **geometry,
    }
    omitted_cells = compute(points, include_empty=False, **common)
    explicit_cells = compute(points, include_empty=True, **common)

    omitted = _build(
        dimension=dimension,
        domain=domain,
        mode='power',
        sites=points,
        ids=ids,
        cells=omitted_cells,
        power_input=power_input,
    )
    explicit = _build(
        dimension=dimension,
        domain=domain,
        mode='power',
        sites=points,
        ids=ids,
        cells=explicit_cells,
        power_input=power_input,
    )

    assert [cell['id'] for cell in omitted_cells] == [10]
    assert any(cell.get('empty') is True for cell in explicit_cells)
    np.testing.assert_array_equal(omitted.empty_mask, [True, False])
    np.testing.assert_array_equal(explicit.empty_mask, omitted.empty_mask)
    np.testing.assert_allclose(explicit.cell_measures, omitted.cell_measures)
    assert omitted.cell_measures[0] == 0.0
    assert omitted.cell_measures[1] == pytest.approx(
        float(omitted_cells[0][measure_key])
    )


def test_builder_rejects_duplicate_and_unknown_raw_cells() -> None:
    points = np.array([[0.25, 0.5], [0.75, 0.5]])
    domain = pv2.Box(((0.0, 1.0), (0.0, 1.0)))
    cells = pv2.compute(points, domain=domain, ids=[20, 10])

    with pytest.raises(ValueError, match='duplicate raw cell ID'):
        _build(
            dimension=2,
            domain=domain,
            mode='standard',
            sites=points,
            ids=[20, 10],
            cells=[cells[0], cells[0]],
            boundaries_available=True,
        )

    unknown = dict(cells[1])
    unknown['id'] = 99
    with pytest.raises(ValueError, match='not present in input ids'):
        _build(
            dimension=2,
            domain=domain,
            mode='standard',
            sites=points,
            ids=[20, 10],
            cells=[cells[0], unknown],
            boundaries_available=True,
        )


def test_omitted_and_explicit_standard_empty_cells_align_identically() -> None:
    points = np.array([[0.5, 0.5, 0.5], [0.5, 0.5, 0.5]])
    ids = np.array([20, 10])
    domain = pv.Box(((0.0, 1.0), (0.0, 1.0), (0.0, 1.0)))
    geometry = {
        'return_vertices': False,
        'return_adjacency': False,
        'return_faces': False,
    }
    omitted_cells = pv.compute(
        points,
        domain=domain,
        ids=ids,
        include_empty=False,
        **geometry,
    )
    explicit_cells = pv.compute(
        points,
        domain=domain,
        ids=ids,
        include_empty=True,
        **geometry,
    )

    omitted = _build(
        dimension=3,
        domain=domain,
        mode='standard',
        sites=points,
        ids=ids,
        cells=omitted_cells,
    )
    explicit = _build(
        dimension=3,
        domain=domain,
        mode='standard',
        sites=points,
        ids=ids,
        cells=explicit_cells,
    )

    assert omitted_cells == []
    assert all(cell.get('empty') is True for cell in explicit_cells)
    np.testing.assert_array_equal(omitted.empty_mask, [True, True])
    np.testing.assert_array_equal(explicit.empty_mask, omitted.empty_mask)
    np.testing.assert_array_equal(omitted.cell_measures, [0.0, 0.0])
    np.testing.assert_array_equal(
        explicit.cell_measures, omitted.cell_measures
    )


def test_available_boundaries_use_empty_collection_for_omitted_hidden_site() -> None:
    points = np.array([[0.25, 0.5], [0.75, 0.5]])
    ids = np.array([20, 10])
    radii = np.array([1.0, 2.0])
    domain = pv2.Box(((0.0, 1.0), (0.0, 1.0)))
    cells = pv2.compute(
        points,
        domain=domain,
        ids=ids,
        mode='power',
        radii=radii,
        include_empty=False,
    )
    result = _build(
        dimension=2,
        domain=domain,
        mode='power',
        sites=points,
        ids=ids,
        cells=cells,
        power_input=resolve_power_input(
            mode='power', weights=None, radii=radii, n=2
        ),
        boundaries_available=True,
    )

    boundaries = result.require_boundaries()
    assert boundaries[0] == []
    assert boundaries[1] is cells[0]['edges']


@pytest.mark.parametrize(
    ('dimension', 'domain', 'measure_key', 'boundary_key'),
    (
        (
            2,
            pv2.Box(((0.0, 1.0), (0.0, 1.0))),
            'area',
            'edges',
        ),
        (
            3,
            pv.Box(((0.0, 1.0), (0.0, 1.0), (0.0, 1.0))),
            'volume',
            'faces',
        ),
    ),
    ids=('planar', 'spatial'),
)
def test_empty_cells_cannot_expose_boundary_geometry(
    dimension: int,
    domain: object,
    measure_key: str,
    boundary_key: str,
) -> None:
    def construct(cell: dict[str, Any]) -> pv.TessellationResult:
        return pv.TessellationResult(
            dimension=dimension,  # type: ignore[arg-type]
            domain=domain,
            mode='power',
            sites=np.full((1, dimension), 0.5),
            ids=np.array([0]),
            cells=[cell],
            cell_measures=np.array([0.0]),
            empty_mask=np.array([True]),
            backend_radii=np.array([0.0]),
            _boundaries_available=True,
        )

    omitted = construct({'id': 0, measure_key: 0.0, 'empty': True})
    assert omitted.require_boundaries() == [[]]

    explicit_cell = {
        'id': 0,
        measure_key: 0.0,
        'empty': True,
        boundary_key: [],
    }
    explicit = construct(explicit_cell)
    assert explicit.require_boundaries() == [[]]

    boundary = {'adjacent_cell': -1, 'vertices': [0, 1]}
    explicit_cell[boundary_key].append(boundary)
    with pytest.raises(ValueError, match=f'empty.*{boundary_key}.*must be empty'):
        explicit.require_boundaries()

    explicit_cell[boundary_key] = None
    with pytest.raises(ValueError, match=f'{boundary_key} must be a list'):
        explicit.require_boundaries()

    inconsistent_cell = {
        'id': 0,
        measure_key: 0.0,
        'empty': True,
        boundary_key: [boundary],
    }
    with pytest.raises(ValueError, match=f'empty.*{boundary_key}.*must be empty'):
        construct(inconsistent_cell)


@pytest.mark.parametrize('mode', ('standard', 'power'))
def test_boundary_access_rejects_removed_or_reclassified_nonempty_cell(
    mode: str,
) -> None:
    points = np.array([[0.25, 0.5], [0.75, 0.5]])
    domain = pv2.Box(((0.0, 1.0), (0.0, 1.0)))
    weights = np.array([-0.01, 0.02]) if mode == 'power' else None
    compute_kwargs = {'weights': weights} if weights is not None else {}
    cells = pv2.compute(
        points,
        domain=domain,
        mode=mode,
        **compute_kwargs,
    )
    result = _build(
        dimension=2,
        domain=domain,
        mode=mode,
        sites=points,
        cells=cells,
        power_input=resolve_power_input(
            mode=mode,
            weights=weights,
            radii=None,
            n=2,
        ),
        boundaries_available=True,
    )

    removed = cells.pop()
    with pytest.raises(ValueError, match='missing for non-empty'):
        result.require_boundaries()
    cells.append(removed)

    cells[0]['empty'] = True
    with pytest.raises(ValueError, match='no longer matches'):
        result.require_boundaries()
    del cells[0]['empty']


def test_power_representation_metadata_is_preserved() -> None:
    points = np.array([[0.25, 0.5], [0.75, 0.5]])
    domain = pv2.Box(((0.0, 1.0), (0.0, 1.0)))
    weights = np.array([-0.25, 0.75])
    resolved_weights = resolve_power_input(
        mode='power', weights=weights, radii=None, n=2
    )
    weighted_cells = pv2.compute(
        points, domain=domain, mode='power', weights=weights
    )
    weighted = _build(
        dimension=2,
        domain=domain,
        mode='power',
        sites=points,
        cells=weighted_cells,
        power_input=resolved_weights,
        boundaries_available=True,
    )
    np.testing.assert_array_equal(weighted.input_weights, weights)
    np.testing.assert_array_equal(
        weighted.backend_radii, resolved_weights.backend_radii
    )
    assert weighted.representation_shift == pytest.approx(0.25)

    radii = np.array([0.25, 0.75])
    resolved_radii = resolve_power_input(
        mode='power', weights=None, radii=radii, n=2
    )
    radius_cells = pv2.compute(
        points, domain=domain, mode='power', radii=radii
    )
    radius_based = _build(
        dimension=2,
        domain=domain,
        mode='power',
        sites=points,
        cells=radius_cells,
        power_input=resolved_radii,
        boundaries_available=True,
    )
    assert radius_based.input_weights is None
    np.testing.assert_array_equal(radius_based.backend_radii, radii)
    assert radius_based.representation_shift is None

    standard_cells = pv2.compute(points, domain=domain)
    standard = _build(
        dimension=2,
        domain=domain,
        mode='standard',
        sites=points,
        cells=standard_cells,
        boundaries_available=True,
    )
    assert standard.input_weights is None
    assert standard.backend_radii is None
    assert standard.representation_shift is None

    inconsistent = ResolvedPowerInput(
        input_weights=resolved_weights.input_weights,
        backend_radii=np.array([9.0, 9.0]),
        representation_shift=resolved_weights.representation_shift,
    )
    with pytest.raises(ValueError, match='backend_radii do not match'):
        _build(
            dimension=2,
            domain=domain,
            mode='power',
            sites=points,
            cells=weighted_cells,
            power_input=inconsistent,
            boundaries_available=True,
        )


def test_optional_objects_and_capability_helpers_are_explicit() -> None:
    points = np.array([[0.25, 0.5], [0.75, 0.5]])
    domain = pv2.Box(((0.0, 1.0), (0.0, 1.0)))
    old_result = pv2.compute(
        points,
        domain=domain,
        return_result=True,
        return_diagnostics=True,
        normalize='topology',
    )
    result = _build(
        dimension=2,
        domain=domain,
        mode='standard',
        sites=points,
        cells=old_result.cells,
        tessellation_diagnostics=old_result.tessellation_diagnostics,
        normalized_vertices=old_result.normalized_vertices,
        normalized_topology=old_result.normalized_topology,
        boundaries_available=True,
    )

    assert result.has_tessellation_diagnostics is True
    assert result.has_normalized_vertices is True
    assert result.has_normalized_topology is True
    assert (
        result.require_tessellation_diagnostics()
        is old_result.tessellation_diagnostics
    )
    assert result.require_normalized_vertices() is old_result.normalized_vertices
    assert result.require_normalized_topology() is old_result.normalized_topology

    absent = _build(
        dimension=2,
        domain=domain,
        mode='standard',
        sites=points,
        cells=pv2.compute(points, domain=domain, return_edges=False),
    )
    assert absent.has_tessellation_diagnostics is False
    assert absent.has_normalized_vertices is False
    assert absent.has_normalized_topology is False
    assert absent.has_boundaries is False
    with pytest.raises(ValueError, match='diagnostics are not available'):
        absent.require_tessellation_diagnostics()
    with pytest.raises(ValueError, match='normalized vertices are not available'):
        absent.require_normalized_vertices()
    with pytest.raises(ValueError, match='normalized topology is not available'):
        absent.require_normalized_topology()
    with pytest.raises(ValueError, match='edges are not available'):
        absent.require_boundaries()


def test_periodic_shift_capability_is_explicit_and_boundaries_are_aligned() -> None:
    points = np.array([[0.2, 0.5], [0.8, 0.5]])
    ids = np.array([20, 10])
    domain = pv2.RectangularCell(
        ((0.0, 1.0), (0.0, 1.0)), periodic=(True, True)
    )
    cells = pv2.compute(
        points,
        domain=domain,
        ids=ids,
        return_edge_shifts=True,
    )
    cells.reverse()
    result = _build(
        dimension=2,
        domain=domain,
        mode='standard',
        sites=points,
        ids=ids,
        cells=cells,
        boundaries_available=True,
        periodic_shifts_available=True,
    )
    assert result.has_boundaries is True
    assert result.has_periodic_shifts is True
    boundaries = result.require_boundaries()
    assert len(boundaries) == 2
    assert all(boundaries)
    assert all(
        'adjacent_shift' in edge
        for edge_list in boundaries
        for edge in edge_list
    )

    first_edge = boundaries[0][0]
    saved_shift = first_edge.pop('adjacent_shift')
    with pytest.raises(ValueError, match='without adjacent_shift'):
        result.require_boundaries()
    first_edge['adjacent_shift'] = saved_shift

    boundaries[0].append([])
    with pytest.raises(ValueError, match='records must be dictionaries'):
        result.require_boundaries()
    boundaries[0].pop()


def test_owned_arrays_are_read_only_and_raw_cells_remain_shared_mutable() -> None:
    sites = np.array([[0.25, 0.5], [0.75, 0.5]])
    ids = np.array([20, 10])
    weights = np.array([-0.25, 0.75])
    domain = pv2.Box(((0.0, 1.0), (0.0, 1.0)))
    cells = pv2.compute(
        sites, domain=domain, ids=ids, mode='power', weights=weights
    )
    power_input = resolve_power_input(
        mode='power', weights=weights, radii=None, n=2
    )
    original_radii = power_input.backend_radii.copy()
    result = _build(
        dimension=2,
        domain=domain,
        mode='power',
        sites=sites,
        ids=ids,
        cells=cells,
        power_input=power_input,
        boundaries_available=True,
    )

    sites[:] = -1.0
    ids[:] = 99
    weights[:] = 123.0
    assert power_input.backend_radii is not None
    power_input.backend_radii[:] = 456.0
    np.testing.assert_array_equal(
        result.sites, [[0.25, 0.5], [0.75, 0.5]]
    )
    np.testing.assert_array_equal(result.ids, [20, 10])
    np.testing.assert_array_equal(result.input_weights, [-0.25, 0.75])
    np.testing.assert_array_equal(result.backend_radii, original_radii)

    for array in (
        result.sites,
        result.ids,
        result.cell_measures,
        result.empty_mask,
        result.input_weights,
        result.backend_radii,
    ):
        assert array is not None
        assert array.flags.writeable is False
        with pytest.raises(ValueError):
            array.flat[0] = 0

    with pytest.raises(FrozenInstanceError):
        result.mode = 'standard'  # type: ignore[misc]

    assert result.cells is cells
    measure_snapshot = result.cell_measures.copy()
    cells[0]['area'] = 999.0
    np.testing.assert_array_equal(result.cell_measures, measure_snapshot)
    cells[0]['custom'] = {'mutable': True}
    assert result.cells[0]['custom']['mutable'] is True
    result.cells[0]['custom']['mutable'] = False
    assert cells[0]['custom']['mutable'] is False


def test_deepcopy_and_pickle_restore_read_only_arrays_and_capabilities() -> None:
    sites = np.array([[0.2, 0.5], [0.8, 0.5]])
    ids = np.array([20, 10])
    weights = np.array([0.0, 0.01])
    domain = pv2.RectangularCell(
        ((0.0, 1.0), (0.0, 1.0)), periodic=(True, True)
    )
    cells = pv2.compute(
        sites,
        domain=domain,
        ids=ids,
        mode='power',
        weights=weights,
        return_edge_shifts=True,
    )
    result = _build(
        dimension=2,
        domain=domain,
        mode='power',
        sites=sites,
        ids=ids,
        cells=cells,
        power_input=resolve_power_input(
            mode='power', weights=weights, radii=None, n=2
        ),
        boundaries_available=True,
        periodic_shifts_available=True,
    )
    measure_snapshot = result.cell_measures.copy()
    result.cells[0]['area'] = 999.0

    for restored in (
        copy.deepcopy(result),
        pickle.loads(pickle.dumps(result)),
    ):
        assert restored.has_boundaries is True
        assert restored.has_periodic_shifts is True
        assert len(restored.require_boundaries()) == 2
        for name in (
            'sites',
            'ids',
            'cell_measures',
            'empty_mask',
            'input_weights',
            'backend_radii',
        ):
            original_array = getattr(result, name)
            restored_array = getattr(restored, name)
            assert original_array is not None
            assert restored_array is not None
            np.testing.assert_array_equal(restored_array, original_array)
            assert restored_array.flags.writeable is False

        assert restored.cells is not result.cells
        assert restored.cells[0]['area'] == 999.0
        np.testing.assert_array_equal(
            restored.cell_measures, measure_snapshot
        )


def test_empty_input_shapes_and_available_but_empty_capabilities() -> None:
    for dimension, compute, domain in (
        (2, pv2.compute, pv2.Box(((0.0, 1.0), (0.0, 1.0)))),
        (
            3,
            pv.compute,
            pv.Box(((0.0, 1.0), (0.0, 1.0), (0.0, 1.0))),
        ),
    ):
        sites = np.empty((0, dimension))
        cells = compute(sites, domain=domain, ids=[])
        result = _build(
            dimension=dimension,
            domain=domain,
            mode='standard',
            sites=sites,
            ids=[],
            cells=cells,
            boundaries_available=True,
        )
        assert result.sites.shape == (0, dimension)
        assert result.ids.shape == (0,)
        assert result.ids.dtype == np.dtype(np.int64)
        assert result.cell_measures.shape == (0,)
        assert result.empty_mask.shape == (0,)
        assert result.has_boundaries is True
        assert result.has_periodic_shifts is False
        assert result.require_boundaries() == []


def test_result_is_not_list_like() -> None:
    result = _build(
        dimension=2,
        domain=pv2.Box(((0.0, 1.0), (0.0, 1.0))),
        mode='standard',
        sites=np.empty((0, 2)),
        cells=[],
    )
    assert not isinstance(result, list)
    with pytest.raises(TypeError):
        result[0]  # type: ignore[index]
    with pytest.raises(AttributeError):
        result.append({})  # type: ignore[attr-defined]


def test_constructor_rejects_invalid_core_and_representation_metadata() -> None:
    domain = pv2.Box(((0.0, 1.0), (0.0, 1.0)))
    base = {
        'dimension': 2,
        'domain': domain,
        'mode': 'standard',
        'sites': np.array([[0.25, 0.5]]),
        'ids': np.array([0]),
        'cells': [{'id': 0, 'area': 1.0}],
        'cell_measures': np.array([1.0]),
        'empty_mask': np.array([False]),
    }

    invalid_cases = (
        ({'dimension': 4}, 'dimension'),
        ({'mode': 'other'}, 'mode'),
        ({'sites': np.array([[0.25, 0.5, 0.75]])}, 'sites'),
        ({'ids': np.array([-1])}, 'non-negative'),
        ({'cell_measures': np.array([np.nan])}, 'finite'),
        ({'cell_measures': np.array([-1.0])}, 'non-negative'),
        ({'cell_measures': np.array([99.0])}, 'must match raw cells'),
        ({'empty_mask': np.array([0])}, 'booleans'),
        ({'empty_mask': np.array([True])}, 'must match raw cells'),
        ({'backend_radii': np.array([1.0])}, 'standard mode'),
        (
            {'_periodic_shifts_available': True},
            'without boundaries',
        ),
        (
            {
                'mode': 'power',
                'backend_radii': None,
            },
            'requires backend_radii',
        ),
        (
            {
                'mode': 'power',
                'backend_radii': np.array([1.0]),
                'representation_shift': 1.0,
            },
            'direct-radius',
        ),
        (
            {
                'mode': 'power',
                'input_weights': np.array([0.0]),
                'backend_radii': np.array([0.0]),
            },
            'requires a representation shift',
        ),
        (
            {
                'mode': 'power',
                'input_weights': np.array([0.0]),
                'backend_radii': np.array([9.0]),
                'representation_shift': 0.0,
            },
            'backend_radii do not match',
        ),
    )
    for updates, message in invalid_cases:
        kwargs = dict(base)
        kwargs.update(updates)
        with pytest.raises(ValueError, match=message):
            pv.TessellationResult(**kwargs)  # type: ignore[arg-type]

    duplicate = dict(base)
    duplicate.update(
        sites=np.array([[0.25, 0.5], [0.75, 0.5]]),
        ids=np.array([1, 1]),
        cell_measures=np.array([0.5, 0.5]),
        empty_mask=np.array([False, False]),
    )
    with pytest.raises(ValueError, match='unique'):
        pv.TessellationResult(**duplicate)  # type: ignore[arg-type]
