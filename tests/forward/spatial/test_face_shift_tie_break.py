"""The retired residual selector is replaced by exact source-chart identity."""

import pyvoro2 as pv


def test_public_source_translation_is_not_a_finite_search_window():
    domain = pv.OrthorhombicCell(((0, 4), (0, 4), (0, 4)))
    cells = pv.compute(
        [[9, 1, 1], [1, 3, 1]], domain=domain,
        return_face_shifts=True, face_shift_search=0,
        face_shift_tol=0, validate_face_shifts=False,
        output='cells',
    )
    assert {f['adjacent_shift'] for f in cells[0]['faces']
            if f['adjacent_cell'] == 1} == {(2, 0, 0), (2, -1, 0)}
    assert {f['adjacent_shift'] for f in cells[1]['faces']
            if f['adjacent_cell'] == 0} == {(-2, 0, 0), (-2, 1, 0)}


def test_large_equal_radius_is_not_a_residual_perturbation_model():
    from pyvoro2 import _core

    # The actual source expressions differ even when r*r is exactly represented.
    observed = _core._test_power_offset_order(1., float(2**27), float(2**27))
    assert observed['r_scale'] == 0.
    assert observed['r_scale_check_offset'] == 1.
