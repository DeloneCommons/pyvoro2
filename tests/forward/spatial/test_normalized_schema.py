"""#99: enabled numerical checks cannot succeed on missing/corrupt operands.

These are public representation mutations, not native producer observations.
"""
from copy import deepcopy

import numpy as np
import pytest

import pyvoro2 as spatial


def _view(partial=False):
    domain = spatial.OrthorhombicCell(
        ((0., 1.),) * 3, periodic=(True, True, not partial))
    raw = spatial.compute([[.5] * 3], domain=domain, return_face_shifts=True)
    return spatial.normalize_topology(raw.cells, domain=domain), domain


@pytest.mark.parametrize('damage', [
    'vertex_global_id', 'vertex_shift', 'faces', 'edge_global_id', 'face_global_id',
    'short_mapping', 'bad_gid', 'bad_local_index', 'duplicate_source',
    'bad_edge_mapping', 'bad_face_mapping', 'wrong_edge_class', 'wrong_face_class',
])
@pytest.mark.parametrize('level', ['basic', 'strict'])
def test_missing_or_corrupt_normalized_operands_are_not_checked_success(damage, level):
    view, domain = _view()
    cell = view.cells[0]
    if damage in ('vertex_global_id', 'vertex_shift', 'faces',
                  'edge_global_id', 'face_global_id'):
        del cell[damage]
    elif damage == 'short_mapping':
        cell['vertex_global_id'].pop()
    elif damage == 'bad_gid':
        cell['vertex_global_id'][0] = len(view.global_vertices)
    elif damage == 'bad_local_index':
        cell['faces'][0]['vertices'][0] = len(cell['vertices'])
    elif damage == 'duplicate_source':
        view.cells.append(deepcopy(cell))
    elif damage == 'bad_edge_mapping':
        cell['edge_global_id'][0] = len(view.global_edges)
    elif damage == 'bad_face_mapping':
        cell['face_global_id'][0] = len(view.global_faces)
    elif damage == 'wrong_edge_class':
        cell['edge_global_id'][0] = (
            (cell['edge_global_id'][0] + 1) % len(view.global_edges))
    elif damage == 'wrong_face_class':
        cell['face_global_id'][0] = (
            (cell['face_global_id'][0] + 1) % len(view.global_faces))
    if level == 'strict':
        with pytest.raises(spatial.NormalizationError) as raised:
            spatial.validate_normalized_topology(view, domain, level=level)
        diag = raised.value.diagnostics
    else:
        diag = spatial.validate_normalized_topology(view, domain, level=level)
    assert not diag.ok
    assert any(issue.severity == 'error' for issue in diag.issues)


def test_nonperiodic_vertex_shift_cannot_be_accepted_as_an_image():
    view, domain = _view(partial=True)
    view.cells[0]['vertex_shift'][0] = (0, 0, 1)
    with pytest.raises(spatial.NormalizationError):
        spatial.validate_normalized_topology(view, domain, level='strict')


def test_nonperiodic_vertex_only_view_needs_no_unused_faces():
    # No face, incidence or Euler check applies here. Vertex mappings still do.
    domain = spatial.Box(((0., 1.),) * 3)
    view = spatial.normalize_vertices(
        [dict(id=17, vertices=[[.25, .5, .75]])], domain=domain)
    diag = spatial.validate_normalized_topology(
        view, domain, level='strict', check_euler=False)
    assert diag.ok
    np.testing.assert_array_equal(view.global_vertices, [[.25, .5, .75]])


def _checks(consumer):
    return dict(
        check_vertex_face_shift=consumer in ('all', 'vertex_face_shift'),
        check_face_vertex_sets=consumer in ('all', 'face_vertex_sets'),
        check_incidence=consumer == 'all',
        check_euler=consumer in ('all', 'euler'),
    )


@pytest.mark.parametrize('damage', ['missing', 'null'])
@pytest.mark.parametrize('weak', [False, True], ids=['topology', 'vertices'])
@pytest.mark.parametrize('consumer', [
    'all', 'euler', 'vertex_face_shift', 'face_vertex_sets',
])
@pytest.mark.parametrize('level', ['basic', 'strict'])
def test_consumed_face_cycles_cannot_disappear(damage, weak, consumer, level):
    view, domain = _view()
    if weak:
        view = spatial.NormalizedVertices(view.global_vertices, view.cells)
    checks = _checks(consumer)
    assert spatial.validate_normalized_topology(
        view, domain, level='strict', **checks).ok
    for face in view.cells[0]['faces']:
        if damage == 'missing':
            del face['vertices']
        else:
            face['vertices'] = None
    if level == 'strict':
        with pytest.raises(spatial.NormalizationError) as raised:
            spatial.validate_normalized_topology(view, domain, level=level, **checks)
        diag = raised.value.diagnostics
    else:
        diag = spatial.validate_normalized_topology(view, domain, level=level, **checks)
    assert not diag.ok
    assert any(issue.code == 'INVALID_NORMALIZED_MAPPING'
               and issue.severity == 'error' for issue in diag.issues)


def test_euler_only_weak_view_needs_cycles_but_no_face_shifts():
    view, domain = _view()
    view = spatial.NormalizedVertices(view.global_vertices, view.cells)
    for face in view.cells[0]['faces']:
        del face['adjacent_shift']
    diag = spatial.validate_normalized_topology(
        view, domain, level='strict', **_checks('euler'))
    assert diag.ok and diag.ok_euler
    assert diag.n_cells_bad_euler == 0
    assert diag.issues == ()
    assert all('adjacent_shift' not in face for face in view.cells[0]['faces'])


@pytest.mark.parametrize('damage', ['missing', 'null', 'fractional'])
@pytest.mark.parametrize('level', ['basic', 'strict'])
def test_euler_only_weak_view_keeps_consumed_neighbor_id_checks(damage, level):
    view, domain = _view()
    view = spatial.NormalizedVertices(view.global_vertices, view.cells)
    face = view.cells[0]['faces'][0]
    if damage == 'missing':
        del face['adjacent_cell']
    else:
        face['adjacent_cell'] = None if damage == 'null' else .5
    if level == 'strict':
        with pytest.raises(spatial.NormalizationError) as raised:
            spatial.validate_normalized_topology(
                view, domain, level=level, **_checks('euler'))
        diag = raised.value.diagnostics
    else:
        diag = spatial.validate_normalized_topology(
            view, domain, level=level, **_checks('euler'))
    assert not diag.ok
    assert any(i.code == 'INVALID_NORMALIZED_MAPPING' and i.severity == 'error'
               for i in diag.issues)


@pytest.mark.parametrize('consumer', ['vertex_face_shift', 'face_vertex_sets'])
@pytest.mark.parametrize('level', ['basic', 'strict'])
@pytest.mark.parametrize('weak', [False, True], ids=['topology', 'vertices'])
def test_enabled_periodic_consumers_still_require_face_shifts(consumer, level, weak):
    view, domain = _view()
    if weak:
        view = spatial.NormalizedVertices(view.global_vertices, view.cells)
    for face in view.cells[0]['faces']:
        del face['adjacent_shift']
    checks = _checks(consumer)
    if level == 'strict':
        with pytest.raises(spatial.NormalizationError) as raised:
            spatial.validate_normalized_topology(view, domain, level=level, **checks)
        diag = raised.value.diagnostics
    else:
        diag = spatial.validate_normalized_topology(view, domain, level=level, **checks)
    assert not diag.ok
    assert any(i.code == 'FACE_MISSING_ADJACENT_SHIFT' and i.severity == 'error'
               for i in diag.issues)


@pytest.mark.parametrize('damage', ['faces', 'missing_cycles', 'null_cycles'])
def test_disabled_face_consumers_allow_unused_operands(damage):
    view, domain = _view()
    view = spatial.NormalizedVertices(view.global_vertices, view.cells)
    if damage == 'faces':
        del view.cells[0]['faces']
    else:
        for face in view.cells[0]['faces']:
            if damage == 'missing_cycles':
                del face['vertices']
            else:
                face['vertices'] = None
            del face['adjacent_shift']
    assert spatial.validate_normalized_topology(
        view, domain, level='strict', **_checks('none')).ok


def test_periodic_consumers_do_not_require_unused_wall_cycles():
    view, domain = _view(partial=True)
    view = spatial.NormalizedVertices(view.global_vertices, view.cells)
    walls = [f for f in view.cells[0]['faces'] if f['adjacent_cell'] < 0]
    assert len(walls) == 2
    for face in walls:
        del face['vertices']
    assert spatial.validate_normalized_topology(
        view, domain, level='strict', check_euler=False).ok


@pytest.mark.parametrize('weak', [False, True], ids=['topology', 'vertices'])
def test_available_empty_cycles_are_not_missing_operands(weak):
    # Explicitly supplied schema control, not a new native producer claim.
    view, domain = _view()
    if weak:
        view = spatial.NormalizedVertices(view.global_vertices, view.cells)
    for face in view.cells[0]['faces']:
        face['vertices'] = []
    assert spatial.validate_normalized_topology(view, domain, level='strict').ok


def test_euler_warning_without_unused_shifts_keeps_counts_and_severity():
    view, domain = _view()
    view = spatial.NormalizedVertices(view.global_vertices, view.cells)
    view.cells[0]['faces'].pop()
    for face in view.cells[0]['faces']:
        del face['adjacent_shift']
    for limit in (0, 1, 10):
        for level in ('basic', 'strict'):
            diag = spatial.validate_normalized_topology(
                view, domain, level=level, max_examples=limit, **_checks('euler'))
            assert diag.ok and not diag.ok_euler
            assert diag.n_cells_bad_euler == 1
            assert len(diag.issues) == 1
            issue = diag.issues[0]
            assert issue.code == 'EULER_CHARACTERISTIC_MISMATCH'
            assert issue.severity == 'warning'
            assert len(issue.examples) == min(limit, 1)


def test_positive_fragment_reciprocity_uses_image_qualified_class_unions():
    # Synthetic/schema partition of an observed positive rectangle, not a
    # claimed retained lower-dimensional native N face. The reverse remains
    # unsplit: arbitrary one-to-one triangle/rectangle pairing is invalid.
    domain = spatial.OrthorhombicCell(((0., 1.),) * 3, periodic=(True, True, False))
    cells = spatial.compute([[.25, .5, .5], [.75, .5, .5]], domain=domain,
                            return_face_shifts=True).cells
    face = next(f for f in cells[0]['faces']
                if f['adjacent_cell'] == 1 and tuple(f['adjacent_shift']) == (0, 0, 0))
    a, b, c, d = face['vertices']
    second = dict(face, vertices=[a, c, d])
    face['vertices'] = [a, b, c]
    cells[0]['faces'].append(second)
    view = spatial.normalize_topology(cells, domain=domain)
    assert spatial.validate_normalized_topology(view, domain, level='strict').ok
    # Drop the unique corner from the class union: this must fail even though
    # another fragment and its repeated directed label remain available.
    second['vertices'] = [a, c]
    damaged = spatial.normalize_topology(cells, domain=domain)
    with pytest.raises(spatial.NormalizationError) as raised:
        spatial.validate_normalized_topology(damaged, domain, level='strict')
    assert any(i.code == 'RECIPROCAL_FACE_VERTEX_SET_MISMATCH'
               for i in raised.value.diagnostics.issues)
