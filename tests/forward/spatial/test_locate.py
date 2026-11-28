from fractions import Fraction
from itertools import product

import numpy as np
import pytest

import pyvoro2 as pv
import pyvoro2.planar as pv2

from pyvoro2 import Box, OrthorhombicCell, locate


def test_locate_box_standard_two_points():
    pts = np.array([[-1.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=float)
    box = Box(bounds=((-2.0, 2.0), (-2.0, 2.0), (-2.0, 2.0)))

    q = np.array([[-0.5, 0.0, 0.0], [0.5, 0.0, 0.0]], dtype=float)

    out = locate(pts, q, domain=box, mode='standard')
    assert out['found'].shape == (2,)
    assert out['owner_id'].shape == (2,)
    assert out['found'].tolist() == [True, True]
    assert out['owner_id'].tolist() == [0, 1]

    out2 = locate(pts, q, domain=box, mode='standard', return_owner_position=True)
    assert 'owner_pos' in out2
    pos = np.asarray(out2['owner_pos'], dtype=float)
    assert pos.shape == (2, 3)
    assert np.allclose(pos[0], pts[0])
    assert np.allclose(pos[1], pts[1])


def test_locate_box_standard_remaps_ids():
    pts = np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]], dtype=float)
    ids = [10, 20]
    box = Box(bounds=((-5.0, 5.0), (-5.0, 5.0), (-5.0, 5.0)))
    q = np.array([[0.1, 0.0, 0.0], [1.9, 0.0, 0.0]], dtype=float)

    out = locate(pts, q, domain=box, ids=ids, mode='standard')
    assert out['found'].tolist() == [True, True]
    assert out['owner_id'].tolist() == [10, 20]


def test_locate_orthorhombic_periodic_x_returns_image_positions():
    # 1D periodic wire along x: [0,1) with periodic x, nonperiodic y/z.
    dom = OrthorhombicCell(
        bounds=((0.0, 1.0), (0.0, 1.0), (0.0, 1.0)), periodic=(True, False, False)
    )
    pts = np.array([[0.1, 0.5, 0.5], [0.9, 0.5, 0.5]], dtype=float)

    q = np.array(
        [
            [1.05, 0.5, 0.5],  # wraps to 0.05 -> closer to 0.1 (id 0)
            [-0.05, 0.5, 0.5],
        ],  # wraps to 0.95 -> closer to 0.9 (id 1)
        dtype=float,
    )

    out = locate(pts, q, domain=dom, mode='standard', return_owner_position=True)
    assert out['found'].tolist() == [True, True]
    assert out['owner_id'].tolist() == [0, 1]

    # Voro++ returns owner positions in the periodic image consistent with the query.
    pos = np.asarray(out['owner_pos'], dtype=float)
    Lx = dom.bounds[0][1] - dom.bounds[0][0]
    assert np.isclose(pos[0, 0], pts[0, 0] + 1 * Lx)
    assert np.isclose(pos[1, 0], pts[1, 0] - 1 * Lx)
    assert np.allclose(pos[:, 1:], pts[:, 1:])


# These exercise the public projection/ID/reporting route as well as the native
# backstop. Native precondition tests independently cover all six raw bindings.
@pytest.mark.parametrize('family', ['spatial_box', 'triclinic', 'planar'])
@pytest.mark.parametrize('mode', ['standard', 'power'])
@pytest.mark.parametrize('positions', [False, True])
@pytest.mark.parametrize('witness', ['product', 'owner', 'worklist'])
def test_public_locate_admission(family, mode, positions, witness):
    if family == 'triclinic' and witness == 'worklist':
        pytest.skip('offset-box cancellation witness is rectangular')
    dim = 2 if family == 'planar' else 3
    api = pv2 if dim == 2 else pv
    p = np.full((1, dim), .5)
    p[0, 0] = .125
    q = p.copy()
    bounds = [(0., 1.)] * dim
    blocks = [1] * dim
    if witness == 'product':
        q[0, 0] = -float(np.iinfo(np.intc).max)
        bounds[0] = (0., 3.)
        blocks[0] = 3
    elif witness == 'owner':
        q[0, 0] = int(np.iinfo(np.intc).max) + .875
    else:
        p[0, 0] = 2.**52
        q[0, 0] = 2.**52 - .5
        bounds[0] = (2.**52, 2.**52 + 1)
        blocks[0] = 8
    if family == 'triclinic':
        domain = pv.PeriodicCell(np.diag([b[1] for b in bounds]))
    else:
        cls = pv2.RectangularCell if dim == 2 else pv.OrthorhombicCell
        domain = cls(bounds, periodic=(True,) * dim)
    kwargs = {'radii': [.125]} if mode == 'power' else {}
    with np.errstate(all='raise'), pytest.raises(ValueError, match=r'queries\[0\]'):
        api.locate(p, q, ids=[123], domain=domain, mode=mode, blocks=blocks,
                   return_owner_position=positions, **kwargs)


@pytest.mark.parametrize('mode', ['standard', 'power'])
@pytest.mark.parametrize('positions', [False, True])
def test_locate_oblique_frame_preserves_native_images(mode, positions):
    basis = np.array([[1., 1., 0.], [0., 1., 1.], [1., 0., 1.]])
    origin = np.array([.125, -.25, .5])
    p = np.array([[.125, .25, .375], [.625, .25, .375]]) @ basis + origin
    shift = np.array([1, -2, 1])
    q = p + shift @ basis
    # Exact represented-source enumeration: each supplied query is precisely
    # one image of its owner, and every competing image is strictly farther.
    for i, query in enumerate(q):
        distances = []
        for j, point in enumerate(p):
            distances.append(min(sum(
                (Fraction(point[d]) + sum(int(t[k]) * Fraction(basis[k, d])
                 for k in range(3)) - Fraction(query[d]))**2 for d in range(3))
                for t in product(range(-3, 4), repeat=3)))
        assert distances[i] == 0 < distances[1 - i]
    kwargs = {'radii': [.125, .125]} if mode == 'power' else {}
    result = pv.locate(p, q, domain=pv.PeriodicCell(basis, origin=origin),
                       mode=mode, ids=[17, 11], blocks=(3, 2, 4),
                       return_owner_position=positions, **kwargs)
    assert result['found'].tolist() == [True, True]
    assert result['owner_id'].tolist() == [17, 11]
    assert ('owner_pos' in result) == positions
    if positions:
        np.testing.assert_allclose(result['owner_pos'], q, atol=2e-14, rtol=0)


@pytest.mark.parametrize('mode', ['standard', 'power'])
@pytest.mark.parametrize('positions', [False, True])
def test_locate_rejects_finite_queries_with_projected_overflow(mode, positions):
    domain = pv.PeriodicCell([[1., 1., 0.], [-1., 1., 0.], [0., 0., 1.]])
    kwargs = {'radii': [.125]} if mode == 'power' else {}
    q = np.full((1, 3), np.finfo(float).max)
    with np.errstate(all='raise'), pytest.raises(ValueError, match='queries.*finite'):
        pv.locate([[0., .5, .5]], q, domain=domain, mode=mode,
                  return_owner_position=positions, **kwargs)
