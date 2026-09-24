"""Public WP5 contracts with independently known cubic face labels."""

from itertools import product

import numpy as np
import pytest

from pyvoro2 import Box, OrthorhombicCell, PeriodicCell, compute
from pyvoro2.diagnostics import TessellationError


def _domain(periodic=(True, True, True)):
    return OrthorhombicCell(((0, 4), (0, 4), (0, 4)), periodic=periodic)


@pytest.mark.parametrize('vertices,adjacency,faces',
                         tuple(product((False, True), repeat=3)))
@pytest.mark.parametrize('power', [False, True])
def test_unrequested_shifts_preserve_all_native_output_flags(
    monkeypatch, vertices, adjacency, faces, power,
):
    from pyvoro2 import api

    def forbidden(*args, **kwargs):
        pytest.fail('ordinary output must not run WP5 certification')

    monkeypatch.setattr(api, '_certify_wp5', forbidden)
    options = {'mode': 'power', 'radii': [2.]} if power else {}
    result = compute([[1., 1., 1.]], domain=_domain(),
                     return_face_shifts=False, return_faces=faces,
                     return_vertices=vertices, return_adjacency=adjacency,
                     **options)
    assert not result.has_periodic_shifts
    assert result.has_boundaries == faces
    assert ('vertices' in result.cells[0]) == vertices
    assert ('adjacency' in result.cells[0]) == adjacency
    assert ('faces' in result.cells[0]) == faces


@pytest.mark.parametrize('vertices,adjacency', tuple(product((False, True), repeat=2)))
@pytest.mark.parametrize('output', ['cells', 'result'])
@pytest.mark.parametrize('power', [False, True])
def test_certified_flags_need_no_public_proof_geometry(vertices, adjacency,
                                                       output, power):
    options = {'mode': 'power', 'radii': [2**27 + 1]} if power else {}
    result = compute([[1, 1, 1]], domain=_domain(), output=output,
                     return_face_shifts=True, return_vertices=vertices,
                     return_adjacency=adjacency, **options)
    cells = result if output == 'cells' else result.cells
    cell = cells[0]
    assert ('vertices' in cell) == vertices
    assert ('adjacency' in cell) == adjacency
    assert {tuple(f['adjacent_shift']) for f in cell['faces']} == {
        (1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0),
        (0, 0, 1), (0, 0, -1),
    }
    assert {f['adjacent_cell'] for f in cell['faces']} == {0}
    if output == 'result':
        assert result.has_periodic_shifts


def test_wall_identity_has_no_image_and_source_sites_are_retained():
    result = compute([[9, 1, 1]], ids=[17], domain=_domain((True, False, False)),
                     return_face_shifts=True, return_vertices=False,
                     return_adjacency=False)
    cell = result.cells[0]
    assert cell['id'] == 17
    assert cell['site'] == [9, 1, 1]
    np.testing.assert_array_equal(result.sites, [[9, 1, 1]])
    walls = [f for f in cell['faces'] if f['adjacent_cell'] < 0]
    assert {f['adjacent_cell'] for f in walls} == {-3, -4, -5, -6}
    assert all('adjacent_shift' not in f for f in walls)


def test_legacy_controls_cannot_restrict_or_repair_the_certificate():
    labels = []
    for search, tol, validate, repair in [
        (0, 0, False, False), (9, 100, True, True),
    ]:
        result = compute([[1, 1, 1]], domain=_domain(), output='cells',
                         return_face_shifts=True, face_shift_search=search,
                         face_shift_tol=tol, validate_face_shifts=validate,
                         repair_face_shifts=repair)
        labels.append([(f['adjacent_cell'], f['adjacent_shift'])
                       for f in result[0]['faces']])
    assert labels[0] == labels[1]


@pytest.mark.parametrize('vectors', [
    np.diag([4., 4., 4.]), np.diag([-4., 4., 4.]),
])
def test_triclinic_frame_uses_original_source_chart(vectors):
    domain = PeriodicCell(vectors)
    result = compute([[9., 1., 1.]], domain=domain, return_face_shifts=True)
    cell = result.cells[0]
    assert cell['site'] == [9., 1., 1.]
    np.testing.assert_allclose(np.mean(cell['vertices'], axis=0), [9., 1., 1.])
    assert len(cell['faces']) == 6


def test_certificate_failure_is_structured_even_with_checks_disabled(monkeypatch):
    from pyvoro2 import api
    from pyvoro2._internal.spatial.wp5_common import WP5Failure

    def refuse(*args, **kwargs):
        raise WP5Failure('WP5_RESOURCE_LIMIT', 'deliberate finite-region refusal')

    monkeypatch.setattr(api, '_certify_wp5', refuse)
    with pytest.raises(TessellationError) as raised:
        compute([[1, 1, 1]], domain=_domain(), return_face_shifts=True,
                tessellation_check='none')
    assert not raised.value.diagnostics.ok
    assert any(issue.code == 'WP5_RESOURCE_LIMIT' and issue.severity == 'error'
               for issue in raised.value.diagnostics.issues)


@pytest.mark.parametrize('domain,faces', [
    (Box(((0, 4),) * 3), True), (_domain(), False),
])
def test_invalid_certified_options_rejected_before_native(monkeypatch, domain,
                                                          faces):
    from pyvoro2 import api

    def forbidden():
        pytest.fail('native entry requested before validating flags')

    monkeypatch.setattr(api, '_require_core', forbidden)
    with pytest.raises(ValueError, match='return_face_shifts'):
        compute([[1, 1, 1]], domain=domain, return_faces=faces,
                return_face_shifts=True)


@pytest.mark.parametrize('policy', ['none', 'diagnose', 'warn', 'raise'])
def test_known_shifts_survive_incomplete_exact_audit_by_action_policy(
    monkeypatch, policy,
):
    from pyvoro2._internal.spatial.wp5_common import WP5Failure
    from pyvoro2._internal.spatial.wp5_ideal import ExactIdeal

    def refuse(*args, **kwargs):
        raise WP5Failure('WP5_RESOURCE_LIMIT', 'exact polytope limit',
                         stage='ideal')

    monkeypatch.setattr(ExactIdeal, 'cell', refuse)
    options = dict(domain=_domain(), return_face_shifts=True,
                   return_vertices=False, return_adjacency=False,
                   return_diagnostics=True,
                   tessellation_check=policy)
    if policy == 'raise':
        with pytest.raises(TessellationError) as raised:
            compute([[1, 1, 1]], **options)
        assert any(i.code == 'WP5_RESOURCE_LIMIT'
                   for i in raised.value.diagnostics.issues)
        return
    if policy == 'warn':
        with pytest.warns(UserWarning, match='tessellation_check'):
            result = compute([[1, 1, 1]], **options)
    else:
        result = compute([[1, 1, 1]], **options)
    assert result.has_periodic_shifts
    assert len(result.cells[0]['faces']) == 6
    assert not result.tessellation_diagnostics.ok
    assert any(issue.code == 'WP5_RESOURCE_LIMIT'
               and issue.examples[0]['audit_scope'] == 'semantic'
               for issue in result.tessellation_diagnostics.issues)


def test_default_shifts_only_path_does_not_run_the_independent_ideal(monkeypatch):
    from pyvoro2._internal.spatial.wp5_ideal import ExactIdeal

    def forbidden(*args, **kwargs):
        pytest.fail('no semantic audit was requested')

    monkeypatch.setattr(ExactIdeal, '__init__', forbidden)
    result = compute([[1., 1., 1.]], domain=_domain(), return_face_shifts=True)
    assert result.has_periodic_shifts
    assert result.tessellation_diagnostics is None


@pytest.mark.parametrize('domain', [_domain(), PeriodicCell(np.diag([4., 4., 4.]))])
def test_requested_diagnostics_audit_even_without_requested_shifts(monkeypatch, domain):
    from pyvoro2._internal.spatial.wp5_ideal import ExactIdeal
    from pyvoro2._internal.spatial.wp5_common import WP5Failure

    def refuse(*args, **kwargs):
        raise WP5Failure('WP5_RESOURCE_LIMIT', 'independent ideal audit reached')

    monkeypatch.setattr(ExactIdeal, 'cell', refuse)
    result = compute([[1., 1., 1.]], domain=domain, return_face_shifts=False,
                     return_faces=False, return_vertices=False,
                     return_adjacency=False, return_diagnostics=True)
    assert not result.has_periodic_shifts
    assert not result.has_boundaries
    assert all('faces' not in cell and 'vertices' not in cell for cell in result.cells)
    assert any(i.code == 'WP5_RESOURCE_LIMIT'
               for i in result.tessellation_diagnostics.issues)


def test_optional_reciprocity_preserves_information_without_error_severity(monkeypatch):
    from dataclasses import replace
    from pyvoro2 import api
    from pyvoro2._internal.spatial.wp5_common import WP5Failure

    certify = api._certify_wp5

    def missing_reverse(*args, **kwargs):
        return replace(certify(*args, **kwargs), issues=(
            WP5Failure('WP5_RECIPROCAL_MISSING', 'deliberate reverse mismatch'),
        ))

    monkeypatch.setattr(api, '_certify_wp5', missing_reverse)
    result = compute([[1., 1., 1.]], domain=_domain(), return_face_shifts=True,
                     tessellation_check='raise', tessellation_require_reciprocity=False)
    issue = next(i for i in result.tessellation_diagnostics.issues
                 if i.code == 'WP5_RECIPROCAL_MISSING')
    assert issue.severity == 'warning'
    assert result.tessellation_diagnostics.ok
