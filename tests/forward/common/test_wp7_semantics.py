"""Analytic ghost policy checks independent of native producer selection."""

from fractions import Fraction

import pytest


def _call(*, extra=(), present=True, points=(), weights=(), ghost_weight=0,
          ghost_site=(0.5, 0.5), omit=(), budget=None):
    from pyvoro2._internal.ghost import GhostOccurrence, certify_semantics

    n = len(points)
    axial = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    occurrences = [GhostOccurrence(n, s) for s in axial if s not in omit]
    occurrences += [GhostOccurrence(n, s, collapsed=collapsed)
                    for s, collapsed in extra]
    if not present:
        occurrences = []
    return certify_semantics(
        points=points, ghost_site=ghost_site,
        lattice=((1.0, 0.0), (0.0, 1.0)), bounds=((0.0, 1.0),) * 2,
        periodic=(True, True), weights=weights, ghost_weight=ghost_weight,
        occurrences=occurrences, present=present, query_index=7,
        external_ids=tuple(range(100, 100 + n)), budget=budget,
    ).references


def test_positive_reference_and_collapsed_diagonal_have_distinct_meaning():
    refs = _call(extra=(((1, 1), True),))
    assert refs[-1] is None
    assert refs[0] == dict(kind='ghost_self', generator_id=None,
                           shift=(-1, 0), wall_id=None)
    assert len(refs) == 5


def test_noncollapsed_diagonal_point_contact_is_hard_semantic_failure():
    with pytest.raises(ValueError) as caught:
        _call(extra=(((1, 1), False),))
    assert caught.value.code == 'GHOST_SEMANTIC_INCONSISTENT'
    assert caught.value.stage == 'semantic'
    assert caught.value.query_index == 7
    assert caught.value.details['contact_status'] == 'point'


def test_collapsed_occurrence_cannot_supply_positive_facet_coverage():
    with pytest.raises(ValueError) as caught:
        _call(omit=((1, 0),), extra=(((1, 0), True),))
    assert caught.value.code == 'GHOST_SEMANTIC_INCONSISTENT'
    assert caught.value.details['invariant'] == 'positive_facet_coverage'


def test_coincident_positive_constraints_need_only_retained_origin_coverage():
    refs = _call(points=((0.75, 0.5),), weights=(Fraction(-1, 4),),
                 ghost_site=(0.25, 0.5))
    assert len(refs) == 4
    assert all(ref['kind'] == 'ghost_self' for ref in refs)


def test_native_deletion_cannot_hide_full_dimensional_semantic_cell():
    with pytest.raises(ValueError) as caught:
        _call(present=False)
    assert caught.value.code == 'GHOST_SEMANTIC_INCONSISTENT'
    assert caught.value.details['semantic_dimension'] == 2


def test_complete_semantic_resource_limit_refuses_atomically():
    from pyvoro2._internal.planar.wp6_ideal import ExactAuditBudget

    with pytest.raises(ValueError) as caught:
        _call(budget=ExactAuditBudget(max_candidates=0))
    assert caught.value.code == 'GHOST_CERTIFICATION_RESOURCE'
    assert caught.value.stage == 'semantic'


def test_reference_shift_range_is_checked_only_at_public_materialization():
    from pyvoro2._internal.ghost import GhostOccurrence, boundary_reference

    huge = 2**70
    occurrence = GhostOccurrence(0, (huge, 0), collapsed=True)
    assert boundary_reference(occurrence, external_ids=(31,),
                              periodic=(True, False), query_index=0) is None
    with pytest.raises(ValueError) as caught:
        boundary_reference(GhostOccurrence(0, (huge, 0)), external_ids=(31,),
                           periodic=(True, False), query_index=0)
    assert caught.value.code == 'GHOST_SHIFT_UNREPRESENTABLE'


def test_ghost_failure_protocol_bounds_details_without_exposing_exception_type():
    from pyvoro2._internal.ghost import GhostFailure

    failure = GhostFailure('GHOST_CERTIFICATION_RESOURCE', 'exhausted',
                           stage='semantic', query_index=None,
                           examples=list(range(1000)), enormous=2**100000)
    assert isinstance(failure, ValueError)
    assert failure.query_index is None
    assert len(repr(failure.details)) < 4096


@pytest.mark.parametrize('weight, dimension', [(Fraction(5, 16), 1), (1, -1)])
def test_private_certificate_retains_deleted_semantic_dimension(weight, dimension):
    from pyvoro2._internal.ghost import certify_semantics

    certificate = certify_semantics(
        points=((.25, .5),), ghost_site=(.5, .5),
        lattice=((1., 0.), (0., 1.)), bounds=((0., 1.),) * 2,
        periodic=(False, False), weights=(weight,), ghost_weight=0,
        occurrences=(), present=False, query_index=2, external_ids=(31,),
    )
    assert certificate.references == ()
    assert certificate.semantic_dimension == dimension
    assert certificate.native_present is False
    assert certificate.query_index == 2
