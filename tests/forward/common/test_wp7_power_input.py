"""Original ghost weights must survive their common native representation."""

import numpy as np

from pyvoro2._internal.power_input import resolve_ghost_power_input


def test_ghost_weight_resolution_retains_original_family_and_batch_gauge():
    resolved = resolve_ghost_power_input(
        mode='power', weights=[-3.0, 5.0], ghost_weights=[1.0, 9.0],
        radii=None, ghost_radii=None, n=2, m=2,
    )
    np.testing.assert_array_equal(resolved.input_weights, [-3.0, 5.0])
    np.testing.assert_array_equal(resolved.input_ghost_weights, [1.0, 9.0])
    assert resolved.representation_shift == 3.0
    np.testing.assert_array_equal(resolved.backend_radii, np.sqrt([0.0, 8.0]))
    np.testing.assert_array_equal(resolved.backend_ghost_radii, np.sqrt([4.0, 12.0]))


def test_ghost_weight_resolution_retains_broadcast_weights():
    resolved = resolve_ghost_power_input(
        mode='power', weights=[], ghost_weights=-0.125,
        radii=None, ghost_radii=None, n=0, m=3,
    )
    np.testing.assert_array_equal(resolved.input_weights, [])
    np.testing.assert_array_equal(resolved.input_ghost_weights, [-0.125] * 3)
    assert resolved.representation_shift == 0.125


def test_ghost_explicit_radii_do_not_fabricate_mathematical_weights():
    resolved = resolve_ghost_power_input(
        mode='power', weights=None, ghost_weights=None,
        radii=[0.1], ghost_radii=0.3, n=1, m=2,
    )
    assert resolved.input_weights is None
    assert resolved.input_ghost_weights is None
    assert resolved.representation_shift is None
    np.testing.assert_array_equal(resolved.backend_radii, [0.1])
    np.testing.assert_array_equal(resolved.backend_ghost_radii, [0.3, 0.3])
