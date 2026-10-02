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
