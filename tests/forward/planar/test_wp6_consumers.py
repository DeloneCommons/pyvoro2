"""Independent WP6 consumer cases built from mutable public test records.

These constructions are not native evidence and do not assert N/E/S truth.
Their literal labels and coordinates exercise the reduced public-record scope.
"""

from __future__ import annotations

import copy

import numpy as np
import pytest

import pyvoro2.planar as planar
from pyvoro2._internal.power_input import resolve_power_input
from pyvoro2.result import _build_tessellation_result


UNIT_BOUNDS = ((0.0, 1.0), (0.0, 1.0))


def _result(cells: list[dict], domain: object):
    """Construct a result without representing the records as native output."""

    return _build_tessellation_result(
        dimension=2,
        domain=domain,
        mode='standard',
        sites=np.array([[0.5, 0.5]]),
        ids=[0],
        cells=cells,
        power_input=resolve_power_input(
            mode='standard', weights=None, radii=None, n=1,
        ),
        boundaries_available=True,
        periodic_shifts_available=True,
    )


@pytest.mark.parametrize(
    ('periodic', 'walls'),
    [
        ((False, False), (-1, -2, -3, -4)),
        ((False, True), (-1, -2)),
        ((True, False), (-3, -4)),
        ((True, True), ()),
    ],
)
def test_result_capability_allows_known_planar_walls_without_images(
    periodic: tuple[bool, bool], walls: tuple[int, ...],
) -> None:
    domain = planar.RectangularCell(UNIT_BOUNDS, periodic=periodic)
    cells = [{'id': 0, 'area': 1.0, 'edges': [
        {'adjacent_cell': wall, 'vertices': []} for wall in walls
    ]}]

    result = _result(cells, domain)

    assert result.has_periodic_shifts is True
    assert result.require_boundaries() == [cells[0]['edges']]
    assert all('adjacent_shift' not in edge for edge in cells[0]['edges'])


@pytest.mark.parametrize('adjacent', [-1, -2, -17])
def test_result_does_not_accept_periodic_side_or_unknown_negative_as_wall(
    adjacent: int,
) -> None:
    domain = planar.RectangularCell(UNIT_BOUNDS, periodic=(True, False))
    cells = [{'id': 0, 'area': 1.0, 'edges': [
        {'adjacent_cell': adjacent, 'adjacent_shift': (0, 0)},
    ]}]

    with pytest.raises(ValueError, match='wall|adjacent_cell'):
        _result(cells, domain)


def test_mutated_planar_wall_is_revalidated_against_domain() -> None:
    domain = planar.RectangularCell(UNIT_BOUNDS, periodic=(True, False))
    cells = [{'id': 0, 'area': 1.0, 'edges': [{'adjacent_cell': -3}]}]
    result = _result(cells, domain)
    cells[0]['edges'][0]['adjacent_cell'] = -1

    with pytest.raises(ValueError, match='wall|adjacent_cell'):
        result.require_boundaries()


def test_normalization_accepts_shiftless_walls_in_periodic_vertex_incidence():
    domain = planar.RectangularCell(UNIT_BOUNDS, periodic=(True, False))
    cells = [{
        'id': 0,
        'vertices': [[0., 0.], [1., 0.], [1., 1.], [0., 1.]],
        'edges': [
            {'adjacent_cell': -3, 'vertices': [0, 1]},
            {'adjacent_cell': 0, 'adjacent_shift': (1, 0), 'vertices': [1, 2]},
            {'adjacent_cell': -4, 'vertices': [2, 3]},
            {'adjacent_cell': 0, 'adjacent_shift': (-1, 0), 'vertices': [3, 0]},
        ],
    }]

    normalized = planar.normalize_topology(cells, domain=domain)

    assert normalized.global_vertices.shape == (2, 2)
    assert len(normalized.global_edges) == 3
    edge_ids = normalized.cells[0]['edge_global_id']
    assert len(edge_ids) == 4
    assert edge_ids[1] == edge_ids[3]
    assert edge_ids[0] != edge_ids[2]
    assert all('adjacent_shift' not in cells[0]['edges'][i] for i in (0, 2))


def test_coincident_edge_pool_keeps_owner_wall_and_occurrence_associations():
    cells = [{
        'id': 0,
        'vertices': [[0., 0.], [0., 1.]],
        'edges': [
            {'adjacent_cell': 1, 'vertices': [0, 1]},
            {'adjacent_cell': 2, 'vertices': [0, 1]},
            {'adjacent_cell': -1, 'vertices': [0, 1]},
            {'adjacent_cell': 1, 'vertices': [0, 1]},
        ],
    }]
    normalized = planar.normalize_topology(cells, domain=planar.Box(UNIT_BOUNDS))

    assert len(normalized.global_edges) == 3
    assert {edge['cells'] for edge in normalized.global_edges} == {
        (0, 1), (0, 2), (-1, 0),
    }
    edge_ids = normalized.cells[0]['edge_global_id']
    assert edge_ids == [0, 1, 2, 0]
    assert len(normalized.cells[0]['edges']) == 4


def test_coincident_edge_pool_keeps_distinct_images_and_reciprocal_class():
    domain = planar.RectangularCell(UNIT_BOUNDS)
    # An explicitly constructed reduced numerical view: geometry is shared,
    # while the two independently supplied image classes must remain distinct.
    cells = [{
        'id': cid,
        'vertices': [[0.5, 0.25], [0.5, 0.75]],
        'vertex_global_id': [0, 1],
        'vertex_shift': [(0, 0), (0, 0)],
        'edges': [
            {'adjacent_cell': 1 - cid, 'adjacent_shift': shift,
             'vertices': [0, 1]}
            for shift in ((0, 0), (1 if cid == 0 else -1, 0))
        ],
    } for cid in (0, 1)]
    vertices = planar.NormalizedVertices(
        np.array([[0.5, 0.25], [0.5, 0.75]]), cells,
    )

    normalized = planar.normalize_edges(vertices, domain=domain)

    assert len(normalized.global_edges) == 2
    assert normalized.cells[0]['edge_global_id'] == [0, 1]
    assert normalized.cells[1]['edge_global_id'] == [0, 1]
    assert [edge['cell_shifts'] for edge in normalized.global_edges] == [
        ((0, 0), (0, 0)), ((0, 0), (1, 0)),
    ]


@pytest.mark.parametrize('adjacent', [-1, -17])
def test_normalization_refuses_invalid_negative_wall_labels(adjacent: int):
    domain = planar.RectangularCell(UNIT_BOUNDS, periodic=(True, False))
    cells = [{
        'id': 0, 'vertices': [[0., 0.], [0., 1.]],
        'edges': [{'adjacent_cell': adjacent, 'vertices': [0, 1]}],
    }]
    snapshot = copy.deepcopy(cells)

    with pytest.raises(ValueError, match='wall|adjacent_cell'):
        planar.normalize_topology(cells, domain=domain, copy_cells=False)

    assert cells == snapshot


@pytest.mark.parametrize(
    ('domain', 'adjacent', 'shift'),
    [
        (planar.Box(UNIT_BOUNDS), 1, (1, 0)),
        (planar.RectangularCell(UNIT_BOUNDS, periodic=(True, False)), 1, (0, 1)),
        (planar.RectangularCell(UNIT_BOUNDS, periodic=(True, False)), -3, (1, 0)),
    ],
)
def test_normalization_cannot_discard_nonzero_inapplicable_image_metadata(
    domain, adjacent, shift,
):
    cells = [{
        'id': 0, 'vertices': [[0., 0.], [0., 1.]],
        'edges': [{'adjacent_cell': adjacent, 'adjacent_shift': shift,
                   'vertices': [0, 1]}],
    }]

    with pytest.raises(ValueError, match='shift|image|wall'):
        planar.normalize_topology(cells, domain=domain)


def test_normalization_refuses_tolerance_collapse_of_distinct_public_endpoints():
    cells = [{
        'id': 0, 'vertices': [[0., 0.], [0., 2. ** -40]],
        'edges': [{'adjacent_cell': -1, 'vertices': [0, 1]}],
    }]
    snapshot = copy.deepcopy(cells)

    with pytest.raises(ValueError, match='collaps|distinct.*endpoints'):
        planar.normalize_topology(
            cells, domain=planar.Box(UNIT_BOUNDS), copy_cells=False,
        )

    assert cells == snapshot


def test_normalization_refuses_coordinate_collapse_across_distinct_vertex_ids():
    # Periodic remapping snaps the first two coordinates together; their
    # different incident owner sets still give them distinct global IDs.
    cells = [{
        'id': 0, 'vertices': [[0., 0.], [2. ** -52, 0.], [0.5, 0.5]],
        'edges': [
            {'adjacent_cell': 1, 'adjacent_shift': (0, 0), 'vertices': [0, 1]},
            {'adjacent_cell': 2, 'adjacent_shift': (0, 0), 'vertices': [1, 2]},
            {'adjacent_cell': 3, 'adjacent_shift': (0, 0), 'vertices': [2, 0]},
        ],
    }]

    with pytest.raises(ValueError, match='collaps|distinct.*endpoints'):
        planar.normalize_topology(cells, domain=planar.RectangularCell(UNIT_BOUNDS))


def test_public_coincident_endpoints_retain_each_provenance_and_occurrence():
    cells = [{
        'id': 0, 'vertices': [[-0., 0.], [0., -0.]],
        'edges': [
            {'adjacent_cell': -1, 'vertices': [0, 1]},
            {'adjacent_cell': -3, 'vertices': [0, 1]},
            {'adjacent_cell': -1, 'vertices': [0, 1]},
        ],
    }]

    normalized = planar.normalize_topology(cells, domain=planar.Box(UNIT_BOUNDS))

    assert normalized.global_vertices.shape == (1, 2)
    assert len(normalized.global_edges) == 2
    assert normalized.cells[0]['edge_global_id'] == [0, 1, 0]


def _fragment_records(*, gap: bool = False) -> list[dict]:
    """Test-only split public class and its complete reciprocal segment."""

    return [
        {'id': 0, 'area': 0.5,
         'vertices': [[0.5, 0.0], [0.5, 0.4 if gap else 0.5],
                      [0.5, 0.6 if gap else 0.5], [0.5, 1.0]],
         'edges': [
             {'adjacent_cell': 1, 'adjacent_shift': (0, 0), 'vertices': [0, 1]},
             {'adjacent_cell': 1, 'adjacent_shift': (0, 0), 'vertices': [2, 3]},
         ]},
        {'id': 1, 'area': 0.5, 'vertices': [[0.5, 0.0], [0.5, 1.0]],
         'edges': [
             {'adjacent_cell': 0, 'adjacent_shift': (0, 0), 'vertices': [0, 1]},
         ]},
    ]


def test_raw_diagnostics_compare_class_union_without_fragment_pairing():
    cells = _fragment_records()

    diagnostics = planar.analyze_tessellation(
        cells, planar.RectangularCell(UNIT_BOUNDS),
    )

    assert diagnostics.n_edges_total == 3
    assert diagnostics.n_edges_orphan == 0
    assert diagnostics.n_edges_mismatched == 0
    assert diagnostics.ok is True
    assert not any(issue.code == 'DUPLICATE_DIRECTED_EDGE'
                   for issue in diagnostics.issues)


def test_raw_diagnostics_find_class_union_gap_without_nearest_pairing():
    cells = _fragment_records(gap=True)

    diagnostics = planar.analyze_tessellation(
        cells, planar.RectangularCell(UNIT_BOUNDS),
    )

    assert diagnostics.n_edges_total == 3
    assert diagnostics.n_edges_mismatched == 1
    assert [(issue.code, issue.severity) for issue in diagnostics.issues] == [
        ('RECIPROCAL_MISMATCH', 'error'),
    ]
    assert all(edge['reciprocal_mismatch']
               for cell in cells for edge in cell['edges'])


def test_raw_tiny_reciprocal_segments_cannot_match_across_a_large_gap():
    cells = [
        {'id': cid, 'area': 0.5,
         'vertices': [[0.5, start], [0.5, start + 2. ** -40]],
         'edges': [
             {'adjacent_cell': 1 - cid, 'adjacent_shift': (0, 0),
              'vertices': [0, 1]},
         ]}
        for cid, start in ((0, 0.0), (1, 0.5))
    ]

    diagnostics = planar.analyze_tessellation(
        cells, planar.RectangularCell(UNIT_BOUNDS),
    )

    assert diagnostics.n_edges_total == 2
    assert diagnostics.n_edges_mismatched == 1
    assert diagnostics.ok is False


@pytest.mark.parametrize('endpoint', [[0.0, 1e-200], [0.0, -0.0]])
def test_raw_diagnostics_count_and_mark_all_tiny_or_coincident_orphans(endpoint):
    cells = [{
        'id': 0, 'area': 1.0, 'vertices': [[0.0, 0.0], endpoint],
        'edges': [
            {'adjacent_cell': 1, 'adjacent_shift': (0, 0), 'vertices': [0, 1]},
            {'adjacent_cell': 1, 'adjacent_shift': (0, 0), 'vertices': [0, 1]},
        ],
    }]
    domain = planar.RectangularCell(UNIT_BOUNDS)

    required = planar.analyze_tessellation(cells, domain)
    optional = planar.validate_tessellation(
        cells, domain, require_reciprocity=False, level='strict',
    )

    assert required.n_edges_total == 2
    assert required.n_edges_orphan == 2
    assert required.ok is False
    assert optional.ok is True
    assert [(issue.code, issue.severity) for issue in optional.issues] == [
        ('MISSING_RECIPROCAL', 'warning'),
    ]
    assert all(edge['orphan'] and edge['reciprocal_missing']
               for edge in cells[0]['edges'])


def test_raw_diagnostics_check_class_reciprocity_without_public_vertices():
    cells = [{'id': 0, 'area': 1.0, 'edges': [
        {'adjacent_cell': 0, 'adjacent_shift': (1, 0)},
    ]}]

    diagnostics = planar.analyze_tessellation(
        cells, planar.RectangularCell(UNIT_BOUNDS),
    )

    assert diagnostics.n_edges_total == 1
    assert diagnostics.n_edges_orphan == 1
    assert diagnostics.ok is False


@pytest.mark.parametrize('walls', [[], [-3, -4]])
def test_raw_diagnostics_accept_available_but_empty_generator_shift_set(walls):
    cells = [{'id': 0, 'area': 1.0, 'edges': [
        {'adjacent_cell': wall} for wall in walls
    ]}]

    diagnostics = planar.analyze_tessellation(
        cells, planar.RectangularCell(UNIT_BOUNDS, periodic=(True, False)),
    )

    assert diagnostics.edge_shift_available is True
    assert diagnostics.reciprocity_checked is True
    assert diagnostics.n_edges_total == 0
    assert diagnostics.ok is True


@pytest.mark.parametrize('adjacent', [-1, -17])
def test_raw_diagnostics_do_not_silently_exempt_invalid_negative_labels(adjacent):
    cells = [{'id': 0, 'area': 1.0, 'edges': [{'adjacent_cell': adjacent}]}]

    diagnostics = planar.analyze_tessellation(
        cells, planar.RectangularCell(UNIT_BOUNDS, periodic=(True, False)),
    )

    assert diagnostics.ok is False
    assert ('INVALID_EDGE_ADJACENCY', 'error') in [
        (issue.code, issue.severity) for issue in diagnostics.issues
    ]
