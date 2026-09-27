"""Constructed malformed packets and proof failures, distinct from native cases."""
from dataclasses import replace
from fractions import Fraction as F
from types import SimpleNamespace

import numpy as np
import pytest

from pyvoro2 import planar
from pyvoro2._internal import locate as impl
from pyvoro2._internal import native_translation as translation
from pyvoro2._internal import query_metadata
from test_wp8_metadata import rectangular


def call():
    return planar.locate([[.25, .5]], [[.375, .5]],
                         domain=planar.RectangularCell(((0., 1.),) * 2),
                         ids=[901], return_owner_position=True)


def alter_native(monkeypatch, mutation):
    import pyvoro2.planar.api as api
    original = api._require_core2d().locate_box_standard

    def native(*args, **kwargs):
        result = original(*args, **kwargs)
        return mutation(result)

    monkeypatch.setattr(api, '_require_core2d',
                        lambda: SimpleNamespace(locate_box_standard=native))


@pytest.mark.parametrize('bad', ['shape', 'dtype', 'owner', 'sentinel', 'source'])
def test_invalid_native_association_fails_before_external_lookup(monkeypatch, bad):
    def mutation(result):
        found, ids, pos, packet = result
        if bad == 'shape':
            found = found[:, None]
        elif bad == 'dtype':
            found = found.astype(float)
        elif bad == 'owner':
            ids[:] = 12345
        elif bad == 'sentinel':
            found[:] = False
        elif bad == 'source':
            packet['stored'] = []
        return found, ids, pos, packet

    alter_native(monkeypatch, mutation)
    with pytest.raises(ValueError) as caught:
        call()
    assert caught.value.code == 'LOCATE_PROVENANCE_INCONSISTENT'
    assert len(str(caught.value)) < 1024


def test_nonfinite_unrequested_owner_view_does_not_gate_id_only(monkeypatch):
    import pyvoro2.planar.api as api
    original = api._require_core2d().locate_box_standard

    def native(*args, **kwargs):
        result = original(*args, **kwargs)
        result[2][:] = np.inf
        return result

    monkeypatch.setattr(api, '_require_core2d',
                        lambda: SimpleNamespace(locate_box_standard=native))
    out = planar.locate([[.25, .5]], [[.375, .5]],
                        domain=planar.RectangularCell(((0., 1.),) * 2))
    assert out['owner_id'].tolist() == [0]


def test_actual_wp4_zero_candidate_propagates_with_cause(monkeypatch):
    def mutation(result):
        result[2][:, 0] += .125
        return result

    alter_native(monkeypatch, mutation)
    with pytest.raises(ValueError) as caught:
        call()
    assert caught.value.code == 'LOCATE_PROVENANCE_INCONSISTENT'
    assert isinstance(caught.value.__cause__,
                      translation.NativeTranslationInconsistencyError)


@pytest.mark.parametrize('resource', [False, True])
def test_complete_region_ambiguity_and_resource_are_distinct(monkeypatch, resource):
    original = impl.owner_enclosure

    def wide(**kwargs):
        defect, radius, removal, bounds = original(**kwargs)
        return defect, (F(2),) * len(radius), removal, bounds

    monkeypatch.setattr(impl, 'owner_enclosure', wide)
    if resource:
        monkeypatch.setattr(impl, '_LIMITS', replace(impl._LIMITS, max_candidates=1))
    with pytest.raises(ValueError) as caught:
        call()
    assert caught.value.code == ('LOCATE_CERTIFICATION_RESOURCE' if resource
                                 else 'LOCATE_PROVENANCE_AMBIGUOUS')
    assert caught.value.query_index == 0 and caught.value.__cause__ is not None


def test_proof_invariant_retains_reason_and_cause(monkeypatch):
    box = translation.CartesianCompatibilityBox((F(0),) * 2, (F(0),) * 2)
    diagnostics = translation.certify_native_translation(np.eye(2), box).diagnostics
    failure = translation.NativeTranslationInvariantError(
        'constructed invariant defect', replace(diagnostics, reason='proof_invariant'))

    def invariant(*args, **kwargs):
        raise failure

    monkeypatch.setattr(impl, 'certify_native_translation', invariant)
    with pytest.raises(ValueError) as caught:
        call()
    assert caught.value.code == 'LOCATE_PROVENANCE_INCONSISTENT'
    assert caught.value.details['reason'] == 'proof_invariant'
    assert caught.value.__cause__ is failure


@pytest.mark.parametrize('index', [None, -1, 1, True])
def test_ghost_malformed_query_association_cannot_be_filtered(monkeypatch, index):
    import pyvoro2.planar.api as api
    original = api._require_core2d().ghost_box_standard

    def native(*args):
        cells = original(*args)
        cells[0]['query_index'] = index
        cells[0]['empty'] = True
        return cells

    monkeypatch.setattr(api, '_require_core2d',
                        lambda: SimpleNamespace(ghost_box_standard=native))
    with pytest.raises(ValueError) as caught:
        planar.ghost_cells(np.empty((0, 2)), [[.25, .5]],
                           domain=planar.Box(((0., 1.),) * 2),
                           return_edges=False, include_empty=False)
    assert caught.value.code == 'GHOST_PROVENANCE_INCONSISTENT'


def test_query_view_endpoints_and_overflow_are_checked_exactly():
    factory = impl._materialization
    for n in (-2**63, 2**63 - 1):
        # Tiny span makes the exact odd endpoint representable as a rational
        # coefficient even though float(n) itself would round upward.
        query = [float(n) if n < 0 else .5, .25]
        rows = [[1., 0.], [0., 1.]] if n < 0 else [[2.**-64, 0.], [0., 1.]]
        origin = [0., 0.] if n < 0 else [2.**-64, 0.]
        _, shift = query_metadata.materialize_query_row(
            query, rows, origin, (True, False), factory, 2)
        assert shift == (n, 0)
    with pytest.raises(ValueError) as caught:
        query_metadata.materialize_query_row(
            [float(2**63), .25], np.eye(2), [0., 0.], (True, False), factory, 2)
    assert caught.value.code == 'LOCATE_METADATA_UNREPRESENTABLE'
    assert caught.value.query_index == 2


@pytest.mark.parametrize('dim', [2, 3])
def test_complete_owner_batch_budget_does_not_gate_id_only(dim, monkeypatch):
    api, domain = rectangular(dim)
    monkeypatch.setattr(impl, '_MAX_OWNER_QUERIES', 1)
    queries = [[.25] * dim, [.375] * dim]
    assert api.locate([[.5] * dim], queries, domain=domain)['found'].all()
    with pytest.raises(ValueError) as caught:
        api.locate([[.5] * dim], queries, domain=domain, return_owner_position=True)
    assert caught.value.code == 'LOCATE_CERTIFICATION_RESOURCE'
