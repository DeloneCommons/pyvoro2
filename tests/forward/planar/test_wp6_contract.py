"""Independent public WP6 anchors, fixed before the production certifier.

Expected labels follow initialization source sides and elementary half-planes.
No expected result is produced by the implementation under test.
"""

import inspect
import itertools

import numpy as np
import pytest

from pyvoro2.planar import Box, RectangularCell, TessellationError, compute
from pyvoro2.planar import api

MASKS = tuple(itertools.product((False, True), repeat=2))
SIDES = {-1: (0, -1), -2: (0, 1), -3: (1, -1), -4: (1, 1)}


def representation(mode):
    if mode == 'standard':
        return {}
    return dict(mode='power', **{mode: [0.25]})


@pytest.mark.parametrize('mask', MASKS)
@pytest.mark.parametrize('mode', ['standard', 'radii', 'weights'])
def test_initialization_side_identity_is_independent_of_shift_output(mask, mode):
    domain = RectangularCell(((0.0, 1.0), (0.0, 1.0)), periodic=mask)
    result = compute([[0.25, 0.375]], domain=domain, **representation(mode))
    edges = result.cells[0]['edges']
    expected = sorted(0 if mask[axis] else side for side, (axis, _) in SIDES.items())
    assert sorted(e['adjacent_cell'] for e in edges) == expected
    assert all('adjacent_shift' not in e for e in edges)
    assert not result.has_periodic_shifts


@pytest.mark.parametrize('mask', MASKS[1:])
@pytest.mark.parametrize('adjacency', [False, True])
@pytest.mark.parametrize('mode', ['standard', 'radii', 'weights'])
def test_shifts_without_public_vertices_and_walls_without_fake_shift(
    mask,
    adjacency,
    mode,
):
    domain = RectangularCell(((0.0, 1.0), (0.0, 1.0)), periodic=mask)
    result = compute(
        [[0.25, 0.375]],
        domain=domain,
        return_vertices=False,
        return_adjacency=adjacency,
        return_edge_shifts=True,
        **representation(mode),
    )
    cell = result.cells[0]
    assert 'vertices' not in cell
    assert ('adjacency' in cell) == adjacency
    assert result.has_periodic_shifts
    actual = {
        (e['adjacent_cell'], tuple(e['adjacent_shift']))
        for e in cell['edges']
        if e['adjacent_cell'] >= 0
    }
    expected = set()
    for axis, active in enumerate(mask):
        if active:
            for sign in (-1, 1):
                s = [0, 0]
                s[axis] = sign
                expected.add((0, tuple(s)))
    assert actual == expected
    for edge in cell['edges']:
        if edge['adjacent_cell'] < 0:
            assert 'adjacent_shift' not in edge


@pytest.mark.parametrize('mode', ['standard', 'radii', 'weights'])
def test_native_insertion_omission_is_never_a_hidden_power_cell(mode):
    with pytest.raises(TessellationError) as caught:
        compute(
            [[np.nextafter(1.0, 0.0), 0.0]],
            domain=Box(((-1.0, 1.0), (-1.0, 1.0))),
            blocks=(1, 1),
            **representation(mode),
        )
    assert any('INSERTION' in i.code for i in caught.value.diagnostics.issues)


@pytest.mark.parametrize(
    'name',
    [
        'edge_shift_search',
        'validate_edge_shifts',
        'repair_edge_shifts',
        'edge_shift_tol',
    ],
)
def test_obsolete_ordinary_controls_are_removed_but_ghost_controls_remain(name):
    assert name not in inspect.signature(compute).parameters
    assert name in inspect.signature(api.ghost_cells).parameters
    with pytest.raises(TypeError, match=name):
        compute([[0.5, 0.5]], domain=Box(((0.0, 1.0), (0.0, 1.0))), **{name: 1})


def test_diamond_collapsed_self_records_are_artifacts_not_semantic_edges():
    # Exact cell is |zx|+|zy| <= 1/2, with four other-owner segments.
    domain = RectangularCell(((0.0, 1.0), (0.0, 1.0)))
    result, certificate = api._compute_with_certificate(
        [[0.25, 0.25], [0.75, 0.75]],
        domain=domain,
        return_vertices=False,
        return_edge_shifts=True,
        return_diagnostics=True,
        tessellation_check='raise',
    )
    assert result.tessellation_diagnostics.ok
    assert certificate.audit_complete
    # Four collapsed self occurrences lack opposite native occurrences. Their
    # nonpositive exact contacts are informational; the raw counter stays true.
    assert result.tessellation_diagnostics.n_edges_orphan == 4
    expected = {(0, 1, s) for s in ((0, 0), (-1, 0), (0, -1), (-1, -1))}
    expected |= {(1, 0, tuple(-x for x in s)) for _, _, s in expected.copy()}
    assert set(certificate.positive_boundaries()) == expected
    assert any(o.collapsed for o in certificate.occurrences)
    assert any('ARTIFACT' in i.code for i in result.tessellation_diagnostics.issues)


def test_original_source_chart_and_external_ids_preserve_image_transport():
    domain = RectangularCell(((0.0, 1.0), (0.0, 1.0)))
    base = np.array([[0.25, 0.25], [0.75, 0.75]])
    offsets = np.array([[3, -2], [-4, 5]])
    ids = np.array([2**63 - 1, 0], dtype=np.int64)
    plain = compute(base, domain=domain, return_edge_shifts=True)
    moved = compute(base + offsets, ids=ids, domain=domain, return_edge_shifts=True)
    by_id = {c['id']: c for c in moved.cells}
    for cell in plain.cells:
        i = cell['id']
        transported = by_id[int(ids[i])]
        assert transported['site'] == (base + offsets)[i].tolist()
        expected = {
            (
                int(ids[e['adjacent_cell']]),
                tuple(
                    int(s) + int(offsets[i, k]) - int(offsets[e['adjacent_cell'], k])
                    for k, s in enumerate(e['adjacent_shift'])
                ),
            )
            for e in cell['edges']
        }
        assert {
            (e['adjacent_cell'], tuple(e['adjacent_shift']))
            for e in transported['edges']
        } == expected


def test_empty_periodic_capability_is_available_but_empty():
    result = compute(
        np.empty((0, 2)),
        domain=RectangularCell(((0.0, 1.0), (0.0, 1.0))),
        return_edge_shifts=True,
        return_vertices=False,
    )
    assert result.cells == []
    assert result.has_periodic_shifts


@pytest.mark.parametrize('normalize', ['vertices', 'topology'])
@pytest.mark.parametrize('mask', MASKS)
def test_diagnostics_and_normalization_keep_geometry_private(normalize, mask):
    domain = RectangularCell(((0.0, 1.0), (0.0, 1.0)), periodic=mask)
    result = compute(
        [[0.25, 0.375]],
        domain=domain,
        normalize=normalize,
        return_vertices=False,
        return_edges=False,
        return_adjacency=False,
        tessellation_check='raise',
    )
    assert result.tessellation_diagnostics.ok
    assert all(not {'vertices', 'edges', 'adjacency'} & set(c) for c in result.cells)
    assert result.normalized_vertices is not None
    if normalize == 'topology':
        assert result.normalized_topology is not None


def test_noop_wall_generator_coincidence_does_not_overwrite_source_origin():
    domain = RectangularCell(((0.0, 1.0), (0.0, 1.0)), periodic=(False, True))
    result = compute(
        [[0.25, 0.5], [0.75, 0.5]],
        domain=domain,
        mode='power',
        radii=[0.75, 0.25],
        return_edge_shifts=True,
        return_diagnostics=True,
    )
    first = next(c for c in result.cells if c['id'] == 0)
    assert any(
        e['adjacent_cell'] == -2 and 'adjacent_shift' not in e for e in first['edges']
    )
    # The exact coincident generator provenance is positive but was a no-op
    # native cut. Auditing reports missing coverage; it cannot relabel the wall.
    assert any(
        i.code == 'WP6_MISSING_POSITIVE_COVERAGE'
        for i in result.tessellation_diagnostics.issues
    )


def test_input_permutation_and_equivalent_origin_preserve_semantic_classes():
    points = np.array([[0.125, 0.25], [0.625, 0.375], [0.5, 0.875]])
    permutation = [2, 0, 1]
    domain = RectangularCell(((0.0, 1.0), (0.0, 1.0)))
    _, original = api._compute_with_certificate(points, domain=domain)
    _, permuted = api._compute_with_certificate(points[permutation], domain=domain)
    offset = np.array([4.0, -8.0])
    moved_domain = RectangularCell(((4.0, 5.0), (-8.0, -7.0)))
    _, moved = api._compute_with_certificate(points + offset, domain=moved_domain)
    expected = set(original.positive_boundaries())
    assert {
        (permutation[i], permutation[j], s)
        for i, j, s in permuted.positive_boundaries()
    } == expected
    assert set(moved.positive_boundaries()) == expected
