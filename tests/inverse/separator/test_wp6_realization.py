"""Independent planar semantic realization anchors (not raw edge counting)."""

import numpy as np
import pytest

from pyvoro2.planar import Box, RectangularCell, TessellationError
from pyvoro2.planar import api
from pyvoro2._internal.planar.wp6_certificate import EdgeCertificate
from pyvoro2.inverse.separator import (
    match_realized_pairs,
    resolve_separator_observations,
)
from pyvoro2.inverse.separator import realize


def test_required_native_midpoint_view_cannot_publish_infinity():
    points = np.array([[1e308, .25], [1e308, .75]])
    domain = RectangularCell(((0., 1.), (0., 1.)))
    constraints = resolve_separator_observations(points, [(0, 1, .5)], domain=domain)
    options = dict(domain=domain, constraints=constraints, weights=np.zeros(2),
                   return_boundary_measure=True)
    result = match_realized_pairs(points, **options, return_cells=False)
    np.testing.assert_array_equal(result.boundary_measure, [1.])
    with np.errstate(over='ignore', invalid='ignore'):
        with pytest.raises(TessellationError) as caught:
            match_realized_pairs(points, **options, return_cells=True)
    assert caught.value.diagnostics.issues[0].code == 'WP6_NONFINITE_OUTPUT_VIEW'


@pytest.mark.parametrize('periodic', [False, True])
@pytest.mark.parametrize('return_cells', [False, True])
def test_planar_length_is_exact_semantic_class_measure(
    monkeypatch,
    periodic,
    return_cells,
):
    points = np.array([[0.25, 1.0], [0.75, 1.0]])
    domain = (
        RectangularCell(((0.0, 1.0), (0.0, 2.0)), periodic=(True, False))
        if periodic
        else Box(((0.0, 1.0), (0.0, 2.0)))
    )
    constraints = resolve_separator_observations(points, [(0, 1, 0.5)], domain=domain)
    original = realize.annotate_edge_properties
    serializations = []
    public_cells = EdgeCertificate.public_cells

    def altered_descriptor(cells, domain):
        original(cells, domain)
        for cell in cells:
            for edge in cell['edges']:
                edge['length'] = 123.0

    def record(self, **kwargs):
        result = public_cells(self, **kwargs)
        serializations.extend(result)
        return result

    monkeypatch.setattr(realize, 'annotate_edge_properties', altered_descriptor)
    monkeypatch.setattr(EdgeCertificate, 'public_cells', record)
    result = match_realized_pairs(
        points,
        domain=domain,
        constraints=constraints,
        weights=np.zeros(2),
        return_boundary_measure=True,
        return_cells=return_cells,
    )
    np.testing.assert_array_equal(result.boundary_measure, [2.0])
    assert serializations
    assert all(('vertices' in cell) == return_cells for cell in serializations)


def test_planar_consumer_uses_complete_classes_without_raw_artifacts():
    # Constructed public records deliberately contain a repeated artifact.
    # This tests consumer plumbing, not the existence of a native producer case.
    cells = [
        {
            'id': 0,
            'edges': [
                {'adjacent_cell': 0, 'adjacent_shift': [1, 0]},
                {'adjacent_cell': 1, 'adjacent_shift': [0, 0]},
                {'adjacent_cell': 1, 'adjacent_shift': [0, 0]},
            ],
        },
        {'id': 1, 'edges': [{'adjacent_cell': 0, 'adjacent_shift': [0, 0]}]},
    ]
    classes = {(0, 1, (0, 0)), (1, 0, (0, 0)), (0, 1, (-1, 0))}
    measures = {key: 2.0 for key in classes}
    _, pairs, lengths = realize._collect_boundary_maps(
        cells,
        boundary_key='edges',
        shift_dim=2,
        return_boundary_measure=True,
        measure_field='length',
        certified_boundary_classes=classes,
        certified_boundary_measures=measures,
    )
    assert pairs == {(0, 1): {(0, 0), (-1, 0)}, (1, 0): {(0, 0)}}
    assert lengths == measures


def test_private_mathematical_weights_survive_backend_radius_selection(monkeypatch):
    points = np.array([[0.25, 0.5], [0.75, 0.5]])
    domain = RectangularCell(((0.0, 1.0), (0.0, 1.0)))
    constraints = resolve_separator_observations(points, [(0, 1, 0.5)], domain=domain)
    original = api._compute_with_certificate
    observed = []

    def capture(*args, **kwargs):
        observed.append(kwargs.get('semantic_weights'))
        return original(*args, **kwargs)

    monkeypatch.setattr(api, '_compute_with_certificate', capture)
    weights = np.array([0.0, 2.0])
    # Equal backend radii give two positive cells; S hides the first one.
    # An exact audit must refuse the mismatch even under diagnostics='none'.
    with pytest.raises(TessellationError):
        realize._match_realized_pairs(
            points,
            domain=domain,
            constraints=constraints,
            radii=np.zeros(2),
            semantic_weights=weights,
            tessellation_check='none',
        )
    assert len(observed) == 1
    np.testing.assert_array_equal(observed[0], weights)
