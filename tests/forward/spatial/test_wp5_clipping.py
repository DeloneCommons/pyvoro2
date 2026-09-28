"""Clipping order, eager proof work and exact boundary retention regressions."""

from fractions import Fraction as F
from itertools import product

import pytest

from pyvoro2._internal.spatial.wp5_common import WP5Budget, WP5Failure, WP5Limits
from pyvoro2._internal.spatial.wp5_ideal import _Arithmetic, _Cut, _Polytope


def _cube(budget=None):
    cuts = [
        _Cut(('wall', -(2 * axis + side + 1)),
             tuple(F(sign if k == axis else 0) for k in range(3)), F(1))
        for axis in range(3) for side, sign in enumerate((-1, 1))
    ]
    vertices = {
        tuple(map(F, point)): frozenset(
            2 * k + int(x > 0) for k, x in enumerate(point)
        )
        for point in product((-1, 1), repeat=3)
    }
    return _Polytope(cuts, vertices, _Arithmetic(
        WP5Budget() if budget is None else budget))


def _cut(normal, offset, owner=9):
    return _Cut((owner, (0, 0, 0)), tuple(map(F, normal)), F(offset))


def test_noop_cut_does_not_hash_vertices_for_gap_retrieval(monkeypatch):
    polytope = _cube()
    original = polytope.vertices
    hash_fraction = F.__hash__
    calls = 0

    def counted(value):
        nonlocal calls
        calls += 1
        return hash_fraction(value)

    monkeypatch.setattr(F, '__hash__', counted)
    polytope.clip(_cut((1, 0, 0), 2))
    assert calls == 0
    assert polytope.vertices is original
    assert len(polytope.cuts) == 6
    # One multiply, one sum addition and one subtraction for each vertex.
    assert polytope.arithmetic.budget.work == 24


@pytest.mark.parametrize('offset', [0, 2])
def test_gap_evaluation_is_eager_before_cut_classification(offset):
    budget = WP5Budget(WP5Limits(work_limit=23))
    polytope = _cube(budget)
    original = polytope.vertices
    with pytest.raises(WP5Failure) as raised:
        polytope.clip(_cut((1, 0, 0), offset))
    assert raised.value.code == 'WP5_RESOURCE_LIMIT'
    assert raised.value.context == {
        'resource': 'work', 'required': 24, 'limit': 23,
        'stage': 'ideal_arithmetic',
    }
    assert budget.work == 23
    assert polytope.vertices is original
    assert len(polytope.cuts) == 6


def test_crossing_cut_keeps_vertex_order_and_coincident_active_constraints():
    polytope = _cube()
    cut = _cut((1, 0, 0), 0)
    polytope.clip(cut)
    expected = list(product((-1, 0), (-1, 1), (-1, 1)))
    assert list(polytope.vertices) == expected
    for x, y, z in expected:
        assert polytope.vertices[x, y, z] == {
            0 if x == -1 else 6, 2 if y == -1 else 3, 4 if z == -1 else 5,
        }

    coincident = _cut((1, 0, 0), 0, owner=10)
    polytope.clip(coincident)
    assert polytope.cuts[-2:] == [cut, coincident]
    assert list(polytope.vertices) == expected
    for x, y, z in expected:
        active = {0 if x == -1 else 6, 2 if y == -1 else 3,
                  4 if z == -1 else 5}
        if x == 0:
            active.add(7)
        assert polytope.vertices[x, y, z] == active


def test_on_plane_vertices_survive_dimension_collapse_then_empty_cut():
    polytope = _cube()
    polytope.clip(_cut((1, 0, 0), -1))
    assert list(polytope.vertices) == list(product((-1,), (-1, 1), (-1, 1)))
    polytope.clip(_cut((0, 1, 0), -1))
    assert list(polytope.vertices) == [(-1, -1, -1), (-1, -1, 1)]
    polytope.clip(_cut((0, 0, 1), -1))
    assert list(polytope.vertices.items()) == [
        ((-1, -1, -1), frozenset({0, 2, 4, 6, 7, 8})),
    ]
    polytope.clip(_cut((0, 0, 1), -2))
    assert not polytope.vertices
    work = polytope.arithmetic.budget.work
    polytope.clip(_cut((1, 1, 1), 0))
    assert len(polytope.cuts) == 10
    assert polytope.arithmetic.budget.work == work
