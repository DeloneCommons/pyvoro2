"""WP5 scientific measures and atomic separator realization failures."""

from dataclasses import replace

import numpy as np
import pytest

import pyvoro2.api as spatial_api
import pyvoro2.inverse.separator.active as active_module
import pyvoro2.inverse.separator.realize as realize_module
from pyvoro2 import Box, OrthorhombicCell
from pyvoro2._internal.spatial.wp5_common import WP5Failure
from pyvoro2.diagnostics import (
    TessellationError,
    TessellationIssue,
    analyze_tessellation,
)
from pyvoro2.inverse.separator import (
    ActiveSetOptions,
    match_realized_pairs,
    resolve_separator_observations,
    solve_self_consistent_power_weights,
)


def _slab_problem(*, periodic=True, points=None):
    if points is None:
        points = np.array([[0.25, 1.0, 1.5], [0.75, 1.0, 1.5]])
    bounds = ((0.0, 1.0), (0.0, 2.0), (0.0, 3.0))
    domain = (
        OrthorhombicCell(bounds, periodic=(True, False, False))
        if periodic else Box(bounds)
    )
    constraints = resolve_separator_observations(
        points,
        [(0, 1, 0.5, (0, 0, 0))],
        domain=domain,
        image='given_only',
    )
    return points, domain, constraints


@pytest.mark.parametrize('return_boundary_measure', (False, True))
@pytest.mark.parametrize('hidden_first', (False, True))
def test_periodic_realization_without_cells_omits_public_vertex_view(
    monkeypatch, return_boundary_measure, hidden_first,
):
    from pyvoro2._internal.spatial.wp5_certificate import FaceCertificate

    points, domain, constraints = _slab_problem()
    public_cells = FaceCertificate.public_cells
    require_semantic = FaceCertificate.require_semantic_consistency
    serialized_cells = []
    audited = []

    def record_serialization(self, **kwargs):
        cells = public_cells(self, **kwargs)
        serialized_cells.extend(cells)
        return cells

    def record_semantic_audit(self):
        require_semantic(self)
        audited.append(self.semantic_consistent)

    monkeypatch.setattr(FaceCertificate, 'public_cells', record_serialization)
    monkeypatch.setattr(
        FaceCertificate, 'require_semantic_consistency', record_semantic_audit,
    )
    result = match_realized_pairs(
        points,
        domain=domain,
        constraints=constraints,
        # A weight advantage of 2 dominates the squared-distance advantage
        # anywhere along this unit-period x axis, hiding the first site.
        weights=np.array([0.0, 2.0 if hidden_first else 0.0]),
        return_cells=False,
        return_boundary_measure=return_boundary_measure,
    )

    assert audited and all(audited)
    assert serialized_cells
    assert all('vertices' not in cell for cell in serialized_cells)
    assert result.cells is None
    np.testing.assert_array_equal(result.endpoint_i_empty, [hidden_first])
    np.testing.assert_array_equal(result.endpoint_j_empty, [False])
    np.testing.assert_array_equal(result.realized_same_shift, [not hidden_first])
    if return_boundary_measure:
        if hidden_first:
            assert np.isnan(result.boundary_measure[0])
        else:
            np.testing.assert_array_equal(result.boundary_measure, [6.0])
    else:
        assert result.boundary_measure is None


@pytest.mark.parametrize('return_cells', (False, True))
def test_periodic_measure_uses_semantic_facet_not_native_area(
    monkeypatch, return_cells,
):
    points, domain, constraints = _slab_problem()
    annotate = realize_module.annotate_face_properties
    native_annotations = []

    def differing_native_area(cells, domain_arg):
        annotate(cells, domain_arg)
        native_annotations.append(cells)
        for cell in cells:
            for face in cell['faces']:
                face['area'] = 123.0

    monkeypatch.setattr(
        realize_module, 'annotate_face_properties', differing_native_area,
    )
    result = match_realized_pairs(
        points,
        domain=domain,
        constraints=constraints,
        weights=np.zeros(2),
        return_boundary_measure=True,
        return_cells=return_cells,
    )

    # The ideal cut is the 2 by 3 rectangle perpendicular to the x axis.
    np.testing.assert_array_equal(result.boundary_measure, [6.0])
    np.testing.assert_array_equal(result.realized_same_shift, [True])
    if return_cells:
        assert len(native_annotations) == 1
        assert result.cells is native_annotations[0]
        assert all(
            face['area'] == 123.0
            for cell in result.cells for face in cell['faces']
        )
    else:
        assert not native_annotations
        assert result.cells is None


def test_unaccounted_periodic_measures_use_semantic_facets(monkeypatch):
    points = np.array([
        [0.125, 1.0, 1.5], [0.375, 1.0, 1.5], [0.75, 1.0, 1.5],
    ])
    points, domain, constraints = _slab_problem(points=points)
    annotate = realize_module.annotate_face_properties

    def differing_native_area(cells, domain_arg):
        annotate(cells, domain_arg)
        for cell in cells:
            for face in cell['faces']:
                face['area'] = 123.0

    monkeypatch.setattr(
        realize_module, 'annotate_face_properties', differing_native_area,
    )
    result = match_realized_pairs(
        points,
        domain=domain,
        constraints=constraints,
        weights=np.zeros(3),
        return_boundary_measure=True,
        return_cells=True,
    )

    assert {
        (pair.site_i, pair.site_j): pair.boundary_measure
        for pair in result.unaccounted_pairs
    } == {(0, 2): 6.0, (1, 2): 6.0}


@pytest.mark.parametrize(
    ('field', 'value'),
    (
        ('area', float('inf')),
        ('centroid', [0.5, float('nan'), 1.5]),
        ('other_site', [float('inf'), 1.0, 1.5]),
        ('intersection_edge_min_dist', float('inf')),
    ),
)
def test_requested_nonfinite_native_descriptor_fails_atomically(
    monkeypatch, field, value,
):
    points, domain, constraints = _slab_problem()
    annotate = realize_module.annotate_face_properties

    def nonfinite_native_descriptor(cells, domain_arg):
        annotate(cells, domain_arg)
        cells[0]['faces'][0][field] = value

    monkeypatch.setattr(
        realize_module, 'annotate_face_properties', nonfinite_native_descriptor,
    )
    with pytest.raises(TessellationError) as raised:
        match_realized_pairs(
            points,
            domain=domain,
            constraints=constraints,
            weights=np.zeros(2),
            return_boundary_measure=True,
            return_cells=True,
            tessellation_check='none',
        )

    assert any(
        issue.code == 'WP5_NONFINITE_OUTPUT_VIEW' and issue.severity == 'error'
        for issue in raised.value.diagnostics.issues
    )


@pytest.mark.parametrize('representation', ('weights', 'radii'))
def test_periodic_realization_preserves_public_weight_representation(
    monkeypatch, representation,
):
    points, domain, constraints = _slab_problem()
    values = (
        np.array([float.fromhex('0x1.0000000000001p-7'), -0.015625])
        if representation == 'weights' else np.array([1.25, 1.75])
    )
    recorded = []

    class InputsCaptured(Exception):
        pass

    def capture_inputs(points_arg, *, semantic_weights=None, **kwargs):
        recorded.append((points_arg, semantic_weights, kwargs))
        raise InputsCaptured

    monkeypatch.setattr(
        spatial_api, '_compute_with_certificate', capture_inputs, raising=False,
    )
    with pytest.raises(InputsCaptured):
        match_realized_pairs(
            points,
            domain=domain,
            constraints=constraints,
            **{representation: values},
        )

    forwarded_points, semantic_weights, options = recorded[0]
    np.testing.assert_array_equal(forwarded_points, points)
    assert semantic_weights is None
    assert options[representation] is values
    assert options['radii' if representation == 'weights' else 'weights'] is None
    assert options['return_face_shifts'] is True
    assert options['output'] == 'cells'


def test_active_realization_keeps_mathematical_weights_and_actual_radius_gauge(
    monkeypatch,
):
    points, domain, _ = _slab_problem()
    compute = spatial_api._compute_with_certificate
    representations = []

    def capture_representation(points_arg, *, semantic_weights=None, **kwargs):
        representations.append((
            None if semantic_weights is None else semantic_weights.copy(),
            kwargs['radii'].copy(),
            kwargs['weights'],
        ))
        return compute(
            points_arg, semantic_weights=semantic_weights, **kwargs,
        )

    monkeypatch.setattr(
        spatial_api, '_compute_with_certificate', capture_representation,
    )
    result = solve_self_consistent_power_weights(
        points,
        [(0, 1, 0.5625, (0, 0, 0))],
        domain=domain,
        image='given_only',
        weight_shift=16.0,
        options=ActiveSetOptions(max_iter=2),
        return_boundary_measure=True,
    )

    assert result.converged
    assert len(representations) == 2  # Outer realization and accepted refit.
    for semantic_weights, radii, public_weights in representations:
        # t = 1/2 + (W_0-W_1)/(2*(1/2)**2) gives this exact contrast,
        # independently of the fit's additive mathematical gauge.
        assert semantic_weights[0] - semantic_weights[1] == 0.03125
        np.testing.assert_array_equal(semantic_weights, result.fit.weights)
        np.testing.assert_array_equal(radii, result.fit.radii)
        assert np.min(radii) > 3.9
        assert public_weights is None
    assert result.fit.weight_shift == 16.0
    np.testing.assert_array_equal(result.realized.boundary_measure, [6.0])


def _certificate_error(domain, code='WP5_REPRESENTATION_CONFLICT'):
    diagnostics = analyze_tessellation(
        [], domain, expected_ids=[0, 1], mode='power',
    )
    diagnostics = replace(
        diagnostics,
        issues=(TessellationIssue(
            code=code,
            severity='error',
            message='independent semantic and native-effective statuses differ',
        ),),
        ok=False,
    )
    return TessellationError('WP5 certificate failed', diagnostics)


@pytest.mark.parametrize('tessellation_check', ('none', 'diagnose', 'warn', 'raise'))
@pytest.mark.parametrize(
    'code', ('WP5_REPRESENTATION_CONFLICT', 'WP5_RESOURCE_LIMIT'),
)
def test_native_shift_capability_cannot_bypass_inverse_semantic_audit(
    monkeypatch, tessellation_check, code,
):
    from pyvoro2._internal.spatial.wp5_certificate import FaceCertificate

    points, domain, constraints = _slab_problem()
    error = _certificate_error(domain, code)

    def require_failed_audit(self):
        raise error

    monkeypatch.setattr(
        FaceCertificate, 'require_semantic_consistency', require_failed_audit,
        raising=False,
    )
    native_result = spatial_api.compute(
        points,
        domain=domain,
        mode='power',
        weights=np.zeros(2),
        return_face_shifts=True,
        tessellation_check='none',
    )
    assert native_result.has_periodic_shifts

    with pytest.raises(TessellationError) as raised:
        match_realized_pairs(
            points,
            domain=domain,
            constraints=constraints,
            weights=np.zeros(2),
            return_boundary_measure=False,
            tessellation_check=tessellation_check,
        )
    assert raised.value is error


@pytest.mark.parametrize(
    ('code', 'audit_complete'),
    (('WP5_REPRESENTATION_CONFLICT', True), ('WP5_RESOURCE_LIMIT', False)),
)
def test_native_attribution_with_failed_semantic_audit_is_not_realization(
    monkeypatch, code, audit_complete,
):
    points, domain, constraints = _slab_problem()
    certify = spatial_api._certify_wp5

    def fail_semantic_audit(*args, **kwargs):
        certificate = certify(*args, **kwargs)
        return replace(
            certificate,
            issues=(WP5Failure(code, 'semantic audit did not succeed'),),
            audit_complete=audit_complete,
        )

    monkeypatch.setattr(spatial_api, '_certify_wp5', fail_semantic_audit)
    native_result = spatial_api.compute(
        points,
        domain=domain,
        mode='power',
        weights=np.zeros(2),
        return_face_shifts=True,
        return_diagnostics=True,
        tessellation_check='none',
    )
    assert native_result.has_periodic_shifts
    diagnostics = native_result.tessellation_diagnostics
    assert not diagnostics.ok
    assert code in {issue.code for issue in diagnostics.issues}

    with pytest.raises(TessellationError) as raised:
        match_realized_pairs(
            points,
            domain=domain,
            constraints=constraints,
            weights=np.zeros(2),
            return_boundary_measure=False,
            tessellation_check='none',
        )
    assert code in {issue.code for issue in raised.value.diagnostics.issues}


@pytest.mark.parametrize('tessellation_check', ('none', 'diagnose', 'raise'))
def test_certificate_failure_is_never_empty_realization(
    monkeypatch, tessellation_check,
):
    points, domain, constraints = _slab_problem()
    error = _certificate_error(domain)

    def fail_certificate(*args, **kwargs):
        raise error

    monkeypatch.setattr(
        spatial_api, '_compute_with_certificate', fail_certificate, raising=False,
    )
    with pytest.raises(TessellationError) as raised:
        match_realized_pairs(
            points,
            domain=domain,
            constraints=constraints,
            weights=np.zeros(2),
            tessellation_check=tessellation_check,
        )
    assert raised.value is error
    assert raised.value.diagnostics.issues[0].severity == 'error'


@pytest.mark.parametrize('failure_call', (1, 2))
@pytest.mark.parametrize(
    ('semantic_issue', 'audit_complete'),
    (
        (None, True),
        ('WP5_REPRESENTATION_CONFLICT', True),
        ('WP5_RESOURCE_LIMIT', False),
    ),
)
def test_active_certificate_failure_never_publishes_accepted_state(
    monkeypatch, failure_call, semantic_issue, audit_complete,
):
    points, domain, constraints = _slab_problem()
    compute = spatial_api._compute_with_certificate
    error = _certificate_error(domain)
    accepted_states = []
    calls = 0
    accepted_state = active_module._AcceptedActiveSetState

    def fail_selected_certificate(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == failure_call and semantic_issue is None:
            raise error
        result, certificate = compute(*args, **kwargs)
        if calls == failure_call:
            certificate = replace(
                certificate,
                issues=(WP5Failure(
                    semantic_issue, 'semantic audit did not succeed',
                ),),
                audit_complete=audit_complete,
            )
        return result, certificate

    def record_accepted_state(*args, **kwargs):
        state = accepted_state(*args, **kwargs)
        accepted_states.append(state)
        return state

    monkeypatch.setattr(
        spatial_api, '_compute_with_certificate', fail_selected_certificate,
    )
    monkeypatch.setattr(
        active_module, '_AcceptedActiveSetState', record_accepted_state,
    )
    with pytest.raises(TessellationError) as raised:
        solve_self_consistent_power_weights(
            points,
            constraints,
            domain=domain,
            tessellation_check='none',
            options=ActiveSetOptions(max_iter=2),
            return_history=True,
        )

    if semantic_issue is None:
        assert raised.value is error
    else:
        assert semantic_issue in {
            issue.code for issue in raised.value.diagnostics.issues
        }
    assert calls == failure_call
    assert accepted_states == []


def test_nonperiodic_realization_stays_on_ordinary_compute(monkeypatch):
    points, domain, constraints = _slab_problem(periodic=False)

    def unexpected_certificate(*args, **kwargs):
        raise AssertionError('nonperiodic realization requested WP5')

    monkeypatch.setattr(
        spatial_api, '_compute_with_certificate', unexpected_certificate,
        raising=False,
    )
    result = match_realized_pairs(
        points,
        domain=domain,
        constraints=constraints,
        weights=np.zeros(2),
        return_boundary_measure=True,
    )
    np.testing.assert_array_equal(result.realized_same_shift, [True])
    np.testing.assert_array_equal(result.boundary_measure, [6.0])
