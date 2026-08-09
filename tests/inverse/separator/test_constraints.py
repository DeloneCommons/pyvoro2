import numpy as np
import pytest


def test_resolve_separator_observations_preserves_explicit_periodic_shift():
    from pyvoro2 import PeriodicCell
    from pyvoro2.inverse.separator import resolve_separator_observations

    cell = PeriodicCell(vectors=((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)))
    pts = np.array([[0.1, 0.5, 0.5], [0.9, 0.5, 0.5]], dtype=float)

    constraints = resolve_separator_observations(
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


def test_resolve_separator_observations_rejects_shifts_on_nonperiodic_axes():
    from pyvoro2 import OrthorhombicCell
    from pyvoro2.inverse.separator import resolve_separator_observations

    domain = OrthorhombicCell(
        bounds=((0.0, 1.0), (0.0, 1.0), (0.0, 1.0)), periodic=(True, False, True)
    )
    pts = np.array([[0.1, 0.2, 0.3], [0.9, 0.8, 0.7]], dtype=float)

    with pytest.raises(ValueError, match='non-periodic axes|non-periodic'):
        resolve_separator_observations(
            pts,
            [(0, 1, 0.5, (0, 1, 0))],
            measurement='fraction',
            domain=domain,
            image='given_only',
        )


def test_resolved_constraints_export_records_and_ids():
    from pyvoro2 import Box
    from pyvoro2.inverse.separator import resolve_separator_observations

    pts = np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]], dtype=float)
    domain = Box(((-5.0, 5.0), (-5.0, 5.0), (-5.0, 5.0)))
    resolved = resolve_separator_observations(
        pts,
        [(10, 20, 0.25)],
        ids=[10, 20],
        index_mode='id',
        measurement='fraction',
        domain=domain,
    )

    rows_idx = resolved.to_records()
    rows_id = resolved.to_records(use_ids=True)
    assert rows_idx[0]['site_i'] == 0
    assert rows_idx[0]['site_j'] == 1
    assert rows_id[0]['site_i'] == 10
    assert rows_id[0]['site_j'] == 20
    assert rows_id[0]['measurement'] == 'fraction'


def test_external_ids_accept_python_and_numpy_integer_scalars() -> None:
    from pyvoro2.inverse.separator import resolve_separator_observations

    points = np.array(
        [[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]],
        dtype=float,
    )
    resolved = resolve_separator_observations(
        points,
        [(np.int32(20), np.uint64(10), 0.5)],
        ids=[np.int64(30), np.int32(20), 10],
        index_mode='id',
    )

    np.testing.assert_array_equal(resolved.ids, [30, 20, 10])
    np.testing.assert_array_equal(resolved.i, [1])
    np.testing.assert_array_equal(resolved.j, [2])
    assert resolved.ids is not None
    assert resolved.ids.flags.writeable is False
    assert resolved.to_records(use_ids=True)[0]['site_i'] == 20
    assert resolved.to_records(use_ids=True)[0]['site_j'] == 10


@pytest.mark.parametrize('large_id', (2**63 + 1, 2**64 + 1))
def test_external_ids_preserve_large_python_integers_exactly(
    large_id: int,
) -> None:
    from pyvoro2.inverse.separator import resolve_separator_observations

    points = np.array([[0.0, 0.0], [1.0, 0.0]], dtype=float)
    resolved = resolve_separator_observations(
        points,
        [(large_id, 0, 0.5)],
        ids=[0, large_id],
        index_mode='id',
    )

    assert resolved.ids is not None
    assert int(resolved.ids[1]) == large_id
    np.testing.assert_array_equal(resolved.i, [1])
    np.testing.assert_array_equal(resolved.j, [0])
    record = resolved.to_records(use_ids=True)[0]
    assert record['site_i'] == large_id
    assert record['site_j'] == 0


@pytest.mark.parametrize('invalid_id', (10.5, '10', True))
def test_external_ids_reject_non_integer_representations(
    invalid_id: object,
) -> None:
    from pyvoro2.inverse.separator import resolve_separator_observations

    points = np.array([[0.0, 0.0], [1.0, 0.0]], dtype=float)
    with pytest.raises(ValueError, match=r'ids\[0\] must be an integer'):
        resolve_separator_observations(
            points,
            [(0, 1, 0.5)],
            ids=[invalid_id, 20],  # type: ignore[list-item]
        )


def test_external_ids_must_be_non_negative() -> None:
    from pyvoro2.inverse.separator import resolve_separator_observations

    points = np.array([[0.0, 0.0], [1.0, 0.0]], dtype=float)
    with pytest.raises(ValueError, match='ids must be non-negative'):
        resolve_separator_observations(
            points,
            [(0, 1, 0.5)],
            ids=[-10, 20],
        )


@pytest.mark.parametrize('endpoint', (0.5, '0', False))
@pytest.mark.parametrize('index_mode', ('index', 'id'))
def test_observation_endpoints_reject_lossy_integer_conversions(
    endpoint: object,
    index_mode: str,
) -> None:
    from pyvoro2.inverse.separator import resolve_separator_observations

    points = np.array([[0.0, 0.0], [1.0, 0.0]], dtype=float)
    kwargs = (
        {'ids': [10, 20], 'index_mode': index_mode}
        if index_mode == 'id'
        else {}
    )
    with pytest.raises(
        ValueError,
        match=r'constraint 0 endpoint i must be an integer',
    ):
        resolve_separator_observations(
            points,
            [(endpoint, 20 if index_mode == 'id' else 1, 0.5)],
            **kwargs,  # type: ignore[arg-type]
        )


def test_resolve_separator_observations_has_no_search_boundary_warning():
    from pyvoro2 import PeriodicCell
    from pyvoro2.inverse.separator import resolve_separator_observations

    cell = PeriodicCell(vectors=((1.0, 0.0, 0.0), (0.2, 1.0, 0.0), (0.0, 0.0, 1.0)))
    pts = np.array([[0.1, 0.5, 0.5], [0.9, 0.5, 0.5]], dtype=float)

    constraints = resolve_separator_observations(
        pts,
        [(0, 1, 0.5)],
        measurement='fraction',
        domain=cell,
        image='nearest',
        image_search=1,
    )

    assert tuple(int(v) for v in constraints.shifts[0]) == (-1, 0, 0)
    assert not any('image_search boundary' in msg for msg in constraints.warnings)


def test_resolve_separator_observations_supports_planar_box() -> None:
    import pyvoro2.planar as pv2
    from pyvoro2.inverse.separator import resolve_separator_observations

    pts = np.array([[0.0, 0.0], [2.0, 0.0]], dtype=float)
    domain = pv2.Box(((-5.0, 5.0), (-5.0, 5.0)))
    constraints = resolve_separator_observations(
        pts,
        [(0, 1, 0.25)],
        measurement='fraction',
        domain=domain,
    )

    assert constraints.dim == 2
    assert tuple(int(v) for v in constraints.shifts[0]) == (0, 0)
    assert np.isclose(constraints.distance[0], 2.0)
    assert np.isclose(constraints.target_position[0], 0.5)


def test_resolve_separator_observations_supports_planar_periodic_shift() -> None:
    import pyvoro2.planar as pv2
    from pyvoro2.inverse.separator import resolve_separator_observations

    domain = pv2.RectangularCell(((0.0, 1.0), (0.0, 1.0)), periodic=(True, True))
    pts = np.array([[0.1, 0.5], [0.9, 0.5]], dtype=float)

    constraints = resolve_separator_observations(
        pts,
        [(0, 1, 0.5, (-1, 0))],
        measurement='fraction',
        domain=domain,
        image='given_only',
    )

    assert bool(constraints.explicit_shift[0]) is True
    assert tuple(int(v) for v in constraints.shifts[0]) == (-1, 0)
    assert np.isclose(constraints.distance[0], 0.2)
