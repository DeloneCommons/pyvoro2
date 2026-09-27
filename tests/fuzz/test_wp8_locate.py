"""Separated analytic images in seeded dyadic triclinic input charts."""
from fractions import Fraction as F

import numpy as np
import pytest

import pyvoro2
from ._support import rng_for_run


@pytest.mark.fuzz
def test_fuzz_locate_exact_image_and_original_query_chart(fuzz_settings):
    # This triangular lattice has ||A^-1||_infinity < 2; every nonzero lattice
    # vector has infinity norm > 1/2. Query offsets below 1/32 cannot change
    # the unique nearest image of this single persistent source.
    basis = np.array([[1., 0., 0.], [.25, 1., 0.], [-.125, .25, 1.]])
    domain = pyvoro2.PeriodicCell(basis)
    for run in range(int(fuzz_settings['n'])):
        rng = rng_for_run(int(fuzz_settings['seed']), run)
        point = rng.integers(-32, 32, size=(1, 3)).astype(float) / 8
        shifts = rng.integers(-5, 6, size=(5, 3))
        images = point + shifts @ basis
        query = images + rng.integers(-7, 8, size=(5, 3)) / 256
        for family in ('standard', 'weights', 'radii'):
            opts = {} if family == 'standard' else {'mode': 'power', family: [0.]}
            out = pyvoro2.locate(point, query, domain=domain,
                                 return_owner_position=True, **opts)
            np.testing.assert_array_equal(out['owner_shift'], shifts)
            np.testing.assert_array_equal(
                out['owner_site'], np.repeat(point, 5, axis=0))
            for i, shift in enumerate(shifts):
                exact = [F(float(point[0, j])) + sum(
                    int(shift[k]) * F(float(basis[k, j])) for k in range(3))
                    for j in range(3)]
                assert out['owner_pos'][i].tolist() == [float(v) for v in exact]
