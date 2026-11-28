"""Compiled composition of stable source lifting and the accepted S reference."""
from fractions import Fraction as F

import numpy as np
import pytest

import pyvoro2.api as api
from pyvoro2 import PeriodicCell
from pyvoro2.inverse.separator import (
    ActiveSetOptions, match_realized_pairs, resolve_separator_observations,
    solve_self_consistent_power_weights,
)


BASIS = np.array([[0., -1., -1.], [1., 0., 0.], [-1., -1., 0.]])
ORIGIN = np.array([1.25, -.5, .125])
POINTS = ORIGIN + np.array([[.25, .5, .5], [.5, .5, .5], [.75, .5, .5]])


def raw_labels(cells):
    return {(c['id'], face['adjacent_cell'], tuple(face['adjacent_shift']))
            for c in cells for face in c['faces'] if face['adjacent_cell'] >= 0}


@pytest.mark.parametrize('lift', [-2**48, 7, 2**50])
def test_matching_captures_stable_submission_and_transports_source_images(
    monkeypatch, lift,
):
    domain = PeriodicCell(BASIS, origin=ORIGIN)
    translated = POINTS.copy()
    translated[1] += lift * BASIS[0]
    t = np.array([[0, 0, 0], [lift, 0, 0], [0, 0, 0]])
    compute = api._require_core().compute_periodic_power
    prepare, capture = api.prepare_generators, api.capture_reference
    submissions, preparations, references = [], [], []

    def native(*args):
        cells = compute(*args)
        submissions.append((args[0].copy(), args[3],
                            {c['id']: list(c['site']) for c in cells}))
        return cells

    def preparation(*args, **kwargs):
        result = prepare(*args, **kwargs)
        preparations.append(result)
        return result

    def reference(*args, **kwargs):
        result = capture(*args, **kwargs)
        references.append(result)
        return result

    monkeypatch.setattr(api._require_core(), 'compute_periodic_power', native)
    monkeypatch.setattr(api, 'prepare_generators', preparation)
    monkeypatch.setattr(api, 'capture_reference', reference)
    results = []
    for points, translations in [(POINTS, t * 0), (translated, t)]:
        correction = translations[0] - translations[1]
        rows = [(0, 1, .5), (1, 0, .5), (1, 2, .5),
                (0, 1, .5, tuple(correction + [0, 1, 0]))]
        obs = resolve_separator_observations(points, rows, domain=domain)
        expected = [translations[0] - translations[1],
                    translations[1] - translations[0],
                    translations[1] - translations[2]]
        np.testing.assert_array_equal(obs.shifts[:3], expected)
        # Independent source connectors: quarter-unit slabs, both directions.
        np.testing.assert_array_equal(
            obs.delta[:3], [[.25, 0, 0], [-.25, 0, 0], [.25, 0, 0]])
        result = match_realized_pairs(
            points, domain=domain, constraints=obs, weights=np.zeros(3),
            return_cells=True, return_boundary_measure=True)
        assert result.realized_same_shift.tolist() == [True, True, True, False]
        assert result.realized_other_shift.tolist() == [False, False, False, True]
        for i, shift in enumerate(expected):
            assert result.realized_shifts[i] == (tuple(shift),)
        np.testing.assert_allclose(result.boundary_measure[:3], 1., rtol=0, atol=2e-12)
        results.append(result)
    assert len(references) == len(preparations) == len(submissions) == 2
    for ref, prepared, (submitted, params, anchors) in zip(
            references, preparations, submissions):
        np.testing.assert_array_equal(ref.submitted, submitted)
        np.testing.assert_array_equal(ref.preparation_shifts, prepared.remap_shifts)
        np.testing.assert_array_equal(ref.anchors, [anchors[i] for i in range(3)])
        bx, bxy, by, bxz, byz, bz = params
        np.testing.assert_array_equal(ref.lattice,
                                      [[bx, 0, 0], [bxy, by, 0], [bxz, byz, bz]])
        assert ref.anchor_present.all()
        assert not ref.submitted.flags.writeable
        # In this admitted fixture actual readback is identical. Thus the
        # independent exact relative-membership witness is h=0 for every pair.
        for i, j in [(0, 1), (1, 0), (1, 2)]:
            difference = [F(ref.submitted[j, a]) - F(ref.submitted[i, a])
                          - F(ref.anchors[j, a]) + F(ref.anchors[i, a])
                          for a in range(3)]
            assert difference == [F(0)] * 3
    np.testing.assert_array_equal(references[0].submitted, references[1].submitted)
    np.testing.assert_array_equal(references[0].anchors, references[1].anchors)
    np.testing.assert_array_equal(
        references[1].preparation_shifts - references[0].preparation_shifts, t)
    np.testing.assert_array_equal(preparations[1].input_points_cart, translated)
    assert raw_labels(results[0].cells) == raw_labels(results[1].cells)


@pytest.mark.parametrize('lift', [0, 7, 2**50])
def test_translated_oblique_active_state_keeps_requested_contacts(lift):
    points = POINTS.copy()
    points[1] += lift * BASIS[0]
    domain = PeriodicCell(BASIS, origin=ORIGIN)
    result = solve_self_consistent_power_weights(
        points, [(0, 1, .5), (1, 2, .5)], domain=domain,
        options=ActiveSetOptions(max_iter=3), return_history=True,
        return_boundary_measure=True, return_cells=True)
    assert result.termination == 'self_consistent'
    assert result.final_state_available
    assert result.active_mask.tolist() == [True, True]
    assert result.realized.realized_same_shift.tolist() == [True, True]
    np.testing.assert_array_equal(result.constraints.shifts,
                                  [[-lift, 0, 0], [lift, 0, 0]])
    np.testing.assert_allclose(result.realized.boundary_measure, 1., rtol=0, atol=2e-12)
    assert {c['id']: c['volume'] for c in result.realized.cells} == pytest.approx(
        {0: 3/8, 1: 1/4, 2: 3/8}, rel=0, abs=2e-12)
