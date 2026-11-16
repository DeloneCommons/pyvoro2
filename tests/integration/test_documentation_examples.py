"""Execute the small DOC-W3 recipes, with independent geometric oracles."""

from pathlib import Path
import re

import numpy as np
import pytest

import pyvoro2 as pv


DOCS = Path(__file__).resolve().parents[2] / 'docs'


def _run_recipe(path):
    source = DOCS / path
    assert source.exists(), f'missing documented recipe: {path}'
    blocks = re.findall(r'^```python\n(.*?)^```', source.read_text(), re.M | re.S)
    assert blocks, f'no Python recipe in {path}'
    namespace = {}
    for block in blocks:
        exec(compile(block, str(source), 'exec'), namespace)
    return namespace


def test_triclinic_quickstart_measure_and_image_geometry():
    recipe = _run_recipe('guide/forward/index.md')
    result, cell = recipe['result'], recipe['cell']
    np.testing.assert_array_equal(result.ids, [40, 10, 70, 20])
    assert not result.empty_mask.any()
    # Triangular lattice: 2 * 1.5 * 1 = 3, independently of tessellation.
    assert np.linalg.det(cell.vectors) == pytest.approx(3.0)
    assert result.cell_measures.sum() == pytest.approx(3.0, abs=1e-10)
    contacts = set()
    for site_id, faces in zip(result.ids, result.require_boundaries()):
        assert faces
        for face in faces:
            shift = tuple(face['adjacent_shift'])
            assert len(shift) == 3
            contacts.add((int(site_id), face['adjacent_cell'], shift))
    assert any(any(shift) for _, _, shift in contacts)
    for i, j, shift in contacts:
        assert (j, i, tuple(-s for s in shift)) in contacts
    topology = recipe['topology']
    assert isinstance(topology, pv.NormalizedTopology)
    assert len(topology.global_vertices) > 0
    assert topology.global_edges and topology.global_faces
    assert not result.has_normalized_topology


def test_inverse_quickstart_analytic_separator_and_weight_gauge():
    recipe = _run_recipe('guide/inverse/index.md')
    fit = recipe['fit']
    assert fit.status == 'optimal'
    # Equal powers at x=0.5 for sites x=0,2: 0.25-w0 = 2.25-w1.
    assert fit.weights[0] - fit.weights[1] == pytest.approx(-2.0)
    np.testing.assert_allclose(fit.predicted_fraction, [0.25], atol=1e-12)
    np.testing.assert_allclose(fit.predicted_position, [0.5], atol=1e-12)
    np.testing.assert_allclose(fit.residuals, 0.0, atol=1e-12)
    points = recipe['points']
    for weights in (fit.weights, fit.weights + 7.0):
        power = np.sum((points - [0.5, 0.0, 0.0]) ** 2, axis=1) - weights
        assert power[0] == pytest.approx(power[1])


def test_periodic_contact_recipe_preserves_self_images_and_multiple_shifts():
    recipe = _run_recipe('guide/periodic-contacts.md')
    key = recipe['contact_key']
    assert key(7, 7, (1, 0, 0)) == (7, 7, (-1, 0, 0))
    assert key(7, 7, (-1, 0, 0)) == (7, 7, (-1, 0, 0))
    assert key(3, 8, (1, 0, 0)) == key(8, 3, (-1, 0, 0))
    assert key(3, 8, (0, 0, 0)) != key(3, 8, (1, 0, 0))
    # Two sites in the unit cube have two distinct x contacts and four
    # self-image pairs (two y/z pairs per site), hence six undirected edges.
    assert recipe['contacts'] == {
        (0, 1, (-1, 0, 0)), (0, 1, (0, 0, 0)),
        (0, 0, (0, -1, 0)), (0, 0, (0, 0, -1)),
        (1, 1, (0, -1, 0)), (1, 1, (0, 0, -1)),
    }


def test_periodic_contact_recipe_requires_shift_data():
    source = (DOCS / 'guide/periodic-contacts.md').read_text()
    blocks = re.findall(r'^```python\n(.*?)^```', source, re.M | re.S)
    # The extraction block must fail on absent shifts, not invent a zero image.
    namespace = {'cells': [{'id': 0, 'faces': [{'adjacent_cell': 1}]}]}
    with pytest.raises(KeyError, match='adjacent_shift'):
        exec(compile(blocks[1], 'contact extraction', 'exec'), namespace)
