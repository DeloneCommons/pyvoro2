"""Graph algebra uses independent integer equations, with no geometry inference."""
from dataclasses import replace
from fractions import Fraction as F

import pytest

from pyvoro2._internal.normalization_proof import (
    ProofFailure, VertexIdentity, identity_closure, value_digest,
)


DOMAIN = b'independent synthetic proof domain'
A, B, C = (0, 0), (1, 0), (2, 0)


def edge(first, second, shift):
    return VertexIdentity(DOMAIN, 2, first, second, shift,
                          'reciprocal-endpoint', (first,))


def test_arbitrary_integer_paths_and_consistent_closed_cycle():
    huge = 2 ** 100
    edges = [edge(A, B, (huge, 0)), edge(B, C, (0, -huge)),
             edge(A, C, (huge, -huge))]
    result = identity_closure((A, B, C), edges, domain=DOMAIN, dimension=2)
    assert result == {A: (A, (0, 0)), B: (A, (-huge, 0)), C: (A, (-huge, huge))}


@pytest.mark.parametrize('bad', [
    edge(A, C, (1, 1)),
    replace(edge(A, C, (1, 0)), domain=b'another computation'),
    replace(edge(A, C, (1, 0)), kind='boundary-incidence'),
    replace(edge(A, C, (1, 0)), shift=(True, 0)),
    replace(edge(A, C, (1, 0)), dimension=3),
])
def test_conflicting_or_non_identity_relations_refuse(bad):
    with pytest.raises(ProofFailure) as caught:
        identity_closure((A, B, C), [edge(A, B, (1, 0)), edge(B, C, (0, 0)), bad],
                         domain=DOMAIN, dimension=2)
    assert caught.value.code == 'NORMALIZATION_IDENTITY_CONFLICT'


def test_known_distinct_anchors_and_inconsistent_lifts_cannot_merge():
    for anchors in ({A: ((F(0), F(0)), (0, 0)), B: ((F(1, 4), F(0)), (0, 0))},
                    {A: ((F(0), F(0)), (0, 0)), B: ((F(0), F(0)), (1, 0))}):
        with pytest.raises(ProofFailure):
            identity_closure((A, B), [edge(A, B, (0, 0))], domain=DOMAIN,
                             dimension=2, anchors=anchors.items())


def test_typed_value_binding_preserves_bits_and_ignores_dictionary_order():
    assert value_digest({'a': [1], 2: F(1, 2)}) == (
        value_digest({2: F(1, 2), 'a': [1]}))
    values = (1, True, 1., '1', b'1', F(1), [1], (1,))
    assert len({value_digest(v) for v in values}) == 8
    assert value_digest(0.) != value_digest(-0.)
