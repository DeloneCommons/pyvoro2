import numpy as np
import pytest



def test_resolve_pair_bisector_constraints_preserves_explicit_periodic_shift():
    from pyvoro2 import PeriodicCell, resolve_pair_bisector_constraints

    cell = PeriodicCell(vectors=((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)))
    pts = np.array([[0.1, 0.5, 0.5], [0.9, 0.5, 0.5]], dtype=float)

    constraints = resolve_pair_bisector_constraints(
        pts,
        [(0, 1, 0.5, (-1, 0, 0))],
        measurement='fraction',
        domain=cell,
        image='given_only',
    )

    assert bool(constraints.explicit_shift[0]) is True
    assert tuple(int(v) for v in constraints.shifts[0]) == (-1, 0, 0)
    assert np.isclose(constraints.distance[0], 0.2)
    assert np.isclose(constraints.target_fraction[0], 0.5)
    assert np.isclose(constraints.target_position[0], 0.1)



def test_resolve_pair_bisector_constraints_rejects_shifts_on_nonperiodic_axes():
    from pyvoro2 import OrthorhombicCell, resolve_pair_bisector_constraints

    domain = OrthorhombicCell(
        bounds=((0.0, 1.0), (0.0, 1.0), (0.0, 1.0)), periodic=(True, False, True)
    )
    pts = np.array([[0.1, 0.2, 0.3], [0.9, 0.8, 0.7]], dtype=float)

    with pytest.raises(ValueError, match='non-periodic axes|non-periodic'):
        resolve_pair_bisector_constraints(
            pts,
            [(0, 1, 0.5, (0, 1, 0))],
            measurement='fraction',
            domain=domain,
            image='given_only',
        )
