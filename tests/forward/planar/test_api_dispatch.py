from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest

import pyvoro2.planar as pv2
import pyvoro2.planar.api as api2d


@dataclass
class FakeCore2D:
    """Dispatch double with explicit test-constructed occurrence packets.

    The rectangles and labels below are hand-specified test records, not
    native evidence. Only build/profile facts come from the installed backend.
    The real Python attribution and exact audit still consume these packets.
    """

    last_call: tuple[str, tuple] | None = None

    @staticmethod
    def _constructed_witness(points, ids, radii, bounds, blocks, periodic,
                             opts, polygons):
        from pyvoro2 import _core2d

        assert ids.tolist() == list(range(len(points)))
        periods = tuple(float(hi - lo) for lo, hi in bounds)
        packet = {
            'profile': _core2d._planar_witness_profile(),
            'bounds': bounds,
            'periods': periods,
            'periodic': tuple(bool(value) for value in periodic),
            'inserted': [],
            'sources': [],
        }
        occupied = {}
        cells = []
        for cid, point in enumerate(points):
            block_axes = [
                min(blocks[axis] - 1,
                    int((point[axis] - bounds[axis][0])
                        * blocks[axis] / periods[axis]))
                for axis in range(2)
            ]
            block = block_axes[0] + blocks[0] * block_axes[1]
            slot = occupied.get(block, 0)
            occupied[block] = slot + 1
            packet['inserted'].append({
                'id': cid, 'point': point.tolist(), 'radius': float(radii[cid]),
                'h': (0, 0), 'block': block, 'slot': slot,
            })
            if cid not in polygons:
                packet['sources'].append({
                    'id': cid, 'present': False, 'local2': [],
                    'next': [], 'origins': [],
                })
                continue
            area, vertices, labels = polygons[cid]
            outgoing = [1, 2, 3, 0]
            origins = []
            for edge_slot, (owner, sigma) in enumerate(labels):
                origin = {'source': cid, 'slot': edge_slot,
                          'next': outgoing[edge_slot]}
                if sigma is None:
                    origin.update(kind='initialization', side=owner)
                else:
                    origin.update(kind='particle', owner=owner, sigma=sigma)
                origins.append(origin)
            packet['sources'].append({
                'id': cid, 'present': True,
                'local2': (2.0 * (np.asarray(vertices) - point)).tolist(),
                'next': outgoing, 'origins': origins,
            })
            cell = {'id': cid, 'area': area, 'site': point.tolist()}
            if opts[0]:
                cell['vertices'] = vertices
            if opts[1]:
                cell['adjacency'] = [[1, 3], [2, 0], [3, 1], [0, 2]]
            if opts[2]:
                cell['edges'] = [
                    {'adjacent_cell': owner,
                     'vertices': [edge_slot, outgoing[edge_slot]]}
                    for edge_slot, (owner, _sigma) in enumerate(labels)
                ]
            cells.append(cell)
        return cells, packet

    def _compute_box_standard_witness(
        self,
        points,
        ids,
        bounds,
        blocks,
        periodic,
        init_mem,
        opts,
    ):
        self.last_call = (
            '_compute_box_standard_witness',
            (bounds, blocks, periodic, init_mem, opts),
        )
        assert np.array_equal(points, [[0.1, 0.5], [0.9, 0.5]])
        polygons = {
            0: (0.5, [[0., 0.], [0.5, 0.], [0.5, 1.], [0., 1.]],
                [(-3, None), (1, (0, 0)), (-4, None),
                 (1, (-1, 0)) if periodic[0] else (-1, None)]),
            1: (0.5, [[0.5, 0.], [1., 0.], [1., 1.], [0.5, 1.]],
                [(-3, None), (0, (1, 0)) if periodic[0] else (-2, None),
                 (-4, None), (0, (0, 0))]),
        }
        return self._constructed_witness(
            points, ids, np.zeros(2), bounds, blocks, periodic, opts, polygons,
        )

    def _compute_box_power_witness(
        self,
        points,
        ids,
        radii,
        bounds,
        blocks,
        periodic,
        init_mem,
        opts,
    ):
        self.last_call = (
            '_compute_box_power_witness', (radii.copy(), bounds, blocks, periodic),
        )
        assert np.array_equal(points, [[0., 0.], [1., 0.]])
        assert np.array_equal(radii, [1., 2.])
        polygons = {
            1: (2.0, [[0., 0.], [2., 0.], [2., 1.], [0., 1.]],
                [(-3, None), (-2, None), (-4, None), (-1, None)]),
        }
        return self._constructed_witness(
            points, ids, radii, bounds, blocks, periodic, opts, polygons,
        )

    def locate_box_standard(
        self,
        points,
        ids,
        bounds,
        blocks,
        periodic,
        init_mem,
        queries,
    ):
        self.last_call = ('locate_box_standard', (bounds, blocks, periodic, init_mem))
        return (
            np.array([True, False]),
            np.array([1, -1]),
            np.array([[1.0, 0.0], [np.nan, np.nan]]),
        )

    def locate_box_power(
        self,
        points,
        ids,
        radii,
        bounds,
        blocks,
        periodic,
        init_mem,
        queries,
    ):
        self.last_call = ('locate_box_power', (radii.copy(), bounds, blocks, periodic))
        return np.array([True]), np.array([0]), np.array([[0.0, 0.0]])

    def ghost_box_standard(
        self,
        points,
        ids,
        bounds,
        blocks,
        periodic,
        init_mem,
        opts,
        queries,
    ):
        self.last_call = (
            'ghost_box_standard',
            (bounds, blocks, periodic, init_mem, opts),
        )
        return [
            {
                'id': -1,
                'empty': False,
                'area': 0.25,
                'site': [0.25, 0.25],
                'vertices': [[0.0, 0.0], [0.5, 0.0], [0.5, 0.5], [0.0, 0.5]],
                'adjacency': [[1, 3], [2, 0], [3, 1], [0, 2]],
                'edges': [
                    {'adjacent_cell': 0, 'vertices': [0, 1]},
                    {'adjacent_cell': 1, 'vertices': [1, 2]},
                    {'adjacent_cell': -1, 'vertices': [2, 3]},
                    {'adjacent_cell': -2, 'vertices': [3, 0]},
                ],
                'query_index': 0,
            },
            {
                'id': -1,
                'empty': True,
                'area': 0.0,
                'site': [0.0, 0.0],
                'vertices': [],
                'adjacency': [],
                'edges': [],
                'query_index': 1,
            },
        ]

    def ghost_box_power(
        self,
        points,
        ids,
        radii,
        bounds,
        blocks,
        periodic,
        init_mem,
        opts,
        queries,
        ghost_radii,
    ):
        self.last_call = (
            'ghost_box_power',
            (
                radii.copy(),
                ghost_radii.copy(),
                bounds,
                blocks,
                periodic,
                init_mem,
                opts,
            ),
        )
        return []


@pytest.fixture()
def fake_core(monkeypatch) -> FakeCore2D:
    fake = FakeCore2D()
    monkeypatch.setattr(api2d, '_core2d', fake, raising=False)
    monkeypatch.setattr(api2d, '_CORE2D_IMPORT_ERROR', None, raising=False)
    return fake


def test_planar_compute_remaps_ids_and_adds_edge_shifts(fake_core) -> None:
    pts = np.array([[0.1, 0.5], [0.9, 0.5]], dtype=float)
    out = pv2.compute(
        pts,
        domain=pv2.RectangularCell(((0.0, 1.0), (0.0, 1.0)), periodic=(True, False)),
        ids=[10, 20],
        return_edge_shifts=True,
        output='cells',
    )

    assert fake_core.last_call is not None
    assert fake_core.last_call[0] == '_compute_box_standard_witness'
    assert [cell['id'] for cell in out] == [10, 20]

    c0 = out[0]
    c1 = out[1]
    shifts01 = {
        tuple(int(v) for v in edge['adjacent_shift'])
        for edge in c0['edges']
        if edge['adjacent_cell'] == 20
    }
    shifts10 = {
        tuple(int(v) for v in edge['adjacent_shift'])
        for edge in c1['edges']
        if edge['adjacent_cell'] == 10
    }
    assert shifts01 == {(-1, 0), (0, 0)}
    assert shifts10 == {(0, 0), (1, 0)}


def test_planar_compute_power_inserts_empty_cells(fake_core) -> None:
    pts = np.array([[0.0, 0.0], [1.0, 0.0]], dtype=float)
    out = pv2.compute(
        pts,
        domain=pv2.RectangularCell(((0.0, 2.0), (0.0, 1.0)), periodic=(True, False)),
        mode='power',
        radii=np.array([1.0, 2.0]),
        include_empty=True,
        output='cells',
    )

    assert fake_core.last_call is not None
    assert fake_core.last_call[0] == '_compute_box_power_witness'
    assert len(out) == 2
    assert out[0]['id'] == 0
    assert out[0]['empty'] is True
    assert out[0]['area'] == 0.0
    assert out[1]['id'] == 1


def test_planar_locate_remaps_owner_ids(fake_core) -> None:
    pts = np.array([[0.0, 0.0], [1.0, 0.0]], dtype=float)
    queries = np.array([[0.9, 0.0], [5.0, 5.0]], dtype=float)
    out = pv2.locate(
        pts,
        queries,
        domain=pv2.Box(((0.0, 2.0), (-1.0, 1.0))),
        ids=[100, 200],
        return_owner_position=True,
    )

    assert fake_core.last_call is not None
    assert fake_core.last_call[0] == 'locate_box_standard'
    assert out['found'].tolist() == [True, False]
    assert out['owner_id'].tolist() == [200, -1]
    assert out['owner_pos'].shape == (2, 2)


def test_planar_ghost_cells_remap_neighbor_ids(fake_core) -> None:
    pts = np.array([[0.0, 0.0], [1.0, 0.0]], dtype=float)
    queries = np.array([[0.5, 0.5], [1.5, 0.5]], dtype=float)
    out = pv2.ghost_cells(
        pts,
        queries,
        domain=pv2.Box(((0.0, 2.0), (0.0, 1.0))),
        ids=[10, 20],
        include_empty=False,
    )

    assert fake_core.last_call is not None
    assert fake_core.last_call[0] == 'ghost_box_standard'
    assert len(out) == 1
    assert out[0]['edges'][0]['adjacent_cell'] == 10
    assert out[0]['edges'][1]['adjacent_cell'] == 20


def test_planar_return_edge_shifts_requires_periodicity(fake_core) -> None:
    pts = np.array([[0.0, 0.0], [1.0, 0.0]], dtype=float)
    with pytest.raises(ValueError, match='periodic domains'):
        pv2.compute(
            pts,
            domain=pv2.Box(((0.0, 2.0), (0.0, 1.0))),
            return_edge_shifts=True,
        )


def test_planar_compute_return_diagnostics(fake_core) -> None:
    pts = np.array([[0.1, 0.5], [0.9, 0.5]], dtype=float)
    cells, diag = pv2.compute(
        pts,
        domain=pv2.Box(((0.0, 1.0), (0.0, 1.0))),
        return_diagnostics=True,
        output='cells',
        tessellation_check='diagnose',
    )

    assert fake_core.last_call is not None
    assert fake_core.last_call[0] == '_compute_box_standard_witness'
    assert isinstance(cells, list)
    assert diag.ok is True
    assert diag.ok_area is True
    assert diag.area_ratio == pytest.approx(1.0)


def test_planar_compute_result_carries_diagnostics(fake_core) -> None:
    pts = np.array([[0.1, 0.5], [0.9, 0.5]], dtype=float)
    result = pv2.compute(
        pts,
        domain=pv2.Box(((0.0, 1.0), (0.0, 1.0))),
        tessellation_check='diagnose',
    )

    assert isinstance(result, pv2.TessellationResult)
    assert result.has_tessellation_diagnostics is True
    assert result.require_tessellation_diagnostics().ok is True
    assert result.normalized_vertices is None
    assert result.normalized_topology is None


def test_planar_compute_normalize_vertices_returns_result(fake_core) -> None:
    pts = np.array([[0.1, 0.5], [0.9, 0.5]], dtype=float)
    result = pv2.compute(
        pts,
        domain=pv2.Box(((0.0, 1.0), (0.0, 1.0))),
        return_vertices=False,
        return_adjacency=False,
        return_edges=False,
        normalize='vertices',
    )

    assert isinstance(result, pv2.TessellationResult)
    assert fake_core.last_call is not None
    assert fake_core.last_call[0] == '_compute_box_standard_witness'
    assert fake_core.last_call[1][-1] == (False, False, True)
    assert set(result.cells[0].keys()) == {'id', 'area', 'site'}
    assert result.has_normalized_vertices is True
    assert result.global_vertices is not None
    assert result.global_vertices.shape == (6, 2)
    assert result.global_edges is None
    with pytest.raises(ValueError, match='normalized topology'):
        result.require_normalized_topology()


def test_planar_compute_normalize_topology_periodic_returns_result(
    fake_core,
) -> None:
    pts = np.array([[0.1, 0.5], [0.9, 0.5]], dtype=float)
    domain = pv2.RectangularCell(((0.0, 1.0), (0.0, 1.0)), periodic=(True, False))
    result = pv2.compute(
        pts,
        domain=domain,
        return_vertices=False,
        return_adjacency=False,
        return_edges=False,
        normalize='topology',
    )

    assert isinstance(result, pv2.TessellationResult)
    assert fake_core.last_call is not None
    assert fake_core.last_call[0] == '_compute_box_standard_witness'
    assert fake_core.last_call[1][-1] == (False, False, True)
    assert set(result.cells[0].keys()) == {'id', 'area', 'site'}
    assert result.has_normalized_vertices is True
    assert result.has_normalized_topology is True
    assert result.has_boundaries is False
    assert result.has_periodic_shifts is False
    assert result.global_edges is not None
    diag = pv2.validate_normalized_topology(
        result.require_normalized_topology(),
        domain,
        level='basic',
    )
    assert diag.is_periodic_domain is True
    assert diag.n_global_edges == len(result.global_edges)


def test_planar_compute_periodic_diagnostics_strip_internal_geometry(
    fake_core,
) -> None:
    pts = np.array([[0.1, 0.5], [0.9, 0.5]], dtype=float)
    result = pv2.compute(
        pts,
        domain=pv2.RectangularCell(
            ((0.0, 1.0), (0.0, 1.0)),
            periodic=(True, False),
        ),
        return_vertices=False,
        return_adjacency=False,
        return_edges=False,
        return_diagnostics=True,
        tessellation_check='diagnose',
    )

    assert fake_core.last_call is not None
    assert fake_core.last_call[0] == '_compute_box_standard_witness'
    assert fake_core.last_call[1][-1] == (False, False, False)

    assert isinstance(result, pv2.TessellationResult)
    assert 'vertices' not in result.cells[0]
    assert 'adjacency' not in result.cells[0]
    assert 'edges' not in result.cells[0]
    assert result.has_boundaries is False
    assert result.has_periodic_shifts is False
    diag = result.require_tessellation_diagnostics()
    assert diag.reciprocity_checked is True
    assert diag.ok_reciprocity is True


def test_planar_compute_tessellation_check_raise(fake_core) -> None:
    complete_witness = fake_core._compute_box_standard_witness

    def broken_area(*args, **kwargs):
        cells, packet = complete_witness(*args, **kwargs)
        cells[0]['area'] = 0.25
        return cells, packet

    fake_core._compute_box_standard_witness = broken_area

    pts = np.array([[0.1, 0.5], [0.9, 0.5]], dtype=float)
    with pytest.raises(pv2.TessellationError, match='tessellation_check failed'):
        pv2.compute(
            pts,
            domain=pv2.Box(((0.0, 1.0), (0.0, 1.0))),
            tessellation_check='raise',
        )


@pytest.mark.parametrize('action', ['none', 'diagnose', 'warn', 'raise'])
def test_missing_source_witness_refuses_before_publishing_provenance(
    fake_core, action,
) -> None:
    fake_core._compute_box_standard_witness = None

    with pytest.raises(pv2.TessellationError) as caught:
        pv2.compute(
            [[0.1, 0.5], [0.9, 0.5]],
            domain=pv2.RectangularCell(((0., 1.), (0., 1.))),
            tessellation_check=action,
        )

    assert any(issue.code == 'WP6_PROFILE_UNSUPPORTED'
               for issue in caught.value.diagnostics.issues)
    assert fake_core.last_call is None


def test_planar_compute_invalid_tessellation_check(fake_core) -> None:
    pts = np.array([[0.1, 0.5], [0.9, 0.5]], dtype=float)
    with pytest.raises(ValueError, match='tessellation_check'):
        pv2.compute(
            pts,
            domain=pv2.Box(((0.0, 1.0), (0.0, 1.0))),
            tessellation_check='nope',  # type: ignore[arg-type]
        )


def test_planar_compute_invalid_normalize(fake_core) -> None:
    pts = np.array([[0.1, 0.5], [0.9, 0.5]], dtype=float)
    with pytest.raises(ValueError, match='normalize'):
        pv2.compute(
            pts,
            domain=pv2.Box(((0.0, 1.0), (0.0, 1.0))),
            normalize='nope',  # type: ignore[arg-type]
        )


def test_planar_compute_invalid_output_precedes_native_compute(fake_core) -> None:
    pts = np.array([[0.1, 0.5], [0.9, 0.5]], dtype=float)
    with pytest.raises(ValueError, match='output.*result.*cells'):
        pv2.compute(
            pts,
            domain=pv2.Box(((0.0, 1.0), (0.0, 1.0))),
            output='nope',  # type: ignore[arg-type]
        )
    assert fake_core.last_call is None


@pytest.mark.parametrize(
    ('output', 'expected_type'),
    (('result', pv2.TessellationResult), ('cells', list)),
)
def test_planar_output_selects_canonical_result_shape(
    fake_core,
    output: str,
    expected_type: type,
) -> None:
    pts = np.array([[0.1, 0.5], [0.9, 0.5]], dtype=float)
    value = pv2.compute(
        pts,
        domain=pv2.Box(((0.0, 1.0), (0.0, 1.0))),
        output=output,  # type: ignore[arg-type]
    )
    assert isinstance(value, expected_type)


def test_planar_removed_return_result_argument_fails_before_compute(
    fake_core,
) -> None:
    pts = np.array([[0.1, 0.5], [0.9, 0.5]], dtype=float)
    with pytest.raises(TypeError, match='return_result'):
        pv2.compute(
            pts,
            domain=pv2.Box(((0.0, 1.0), (0.0, 1.0))),
            return_result=True,  # type: ignore[call-arg]
        )
    assert fake_core.last_call is None


def test_planar_explicit_raw_output_rejects_normalization(fake_core) -> None:
    pts = np.array([[0.1, 0.5], [0.9, 0.5]], dtype=float)
    with pytest.raises(ValueError, match='output="cells".*normalization'):
        pv2.compute(
            pts,
            domain=pv2.Box(((0.0, 1.0), (0.0, 1.0))),
            output='cells',
            normalize='vertices',
        )
    assert fake_core.last_call is None
