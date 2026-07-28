import numpy as np


def test_unaccounted_pair_records_preserve_large_ids_and_validate_once(
    monkeypatch,
) -> None:
    import pyvoro2.inverse.separator.realize as realize_module
    from pyvoro2.inverse.separator import (
        RealizedPairDiagnostics,
        UnaccountedRealizedPair,
    )

    large_id = 2**63 + 1
    pair = UnaccountedRealizedPair(
        site_i=0,
        site_j=1,
        realized_shifts=((0, 0),),
    )
    assert pair.to_record(ids=[0, large_id])['site_j'] == large_id

    original_validator = realize_module._validated_ids_array
    validation_count = 0

    def counting_validator(ids, n_points=None):
        nonlocal validation_count
        validation_count += 1
        return original_validator(ids, n_points)

    monkeypatch.setattr(
        realize_module,
        '_validated_ids_array',
        counting_validator,
    )
    diagnostics = RealizedPairDiagnostics(
        realized=np.zeros(0, dtype=bool),
        unrealized=(),
        realized_same_shift=np.zeros(0, dtype=bool),
        realized_other_shift=np.zeros(0, dtype=bool),
        realized_shifts=(),
        endpoint_i_empty=np.zeros(0, dtype=bool),
        endpoint_j_empty=np.zeros(0, dtype=bool),
        boundary_measure=None,
        cells=None,
        tessellation_diagnostics=None,
        unaccounted_pairs=(pair, pair),
    )

    records = diagnostics.unaccounted_records(ids=[0, large_id])

    assert [record['site_j'] for record in records] == [large_id, large_id]
    assert validation_count == 1


def test_match_realized_pairs_flags_unrealized_constraints():
    from pyvoro2 import Box
    from pyvoro2.inverse.separator import (
        FitModel,
        Interval,
        fit_weights_from_separators,
        match_realized_pairs,
        resolve_separator_observations,
    )

    pts = np.array(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]],
        dtype=float,
    )
    domain = Box(((-5, 5), (-5, 5), (-5, 5)))
    constraints = resolve_separator_observations(
        pts,
        [(0, 2, 0.5)],
        measurement='fraction',
        domain=domain,
    )
    fit = fit_weights_from_separators(
        pts,
        constraints,
        model=FitModel(feasible=Interval(0.0, 1.0)),
        solver='admm',
        admm_max_iter=5000,
    )
    diag = match_realized_pairs(
        pts,
        domain=domain,
        radii=fit.radii,
        constraints=constraints,
    )

    assert diag.realized.shape == (1,)
    assert bool(diag.realized[0]) is False
    assert diag.unrealized == (0,)


def test_match_realized_pairs_reports_boundary_measure_when_requested():
    from pyvoro2 import Box
    from pyvoro2.inverse.separator import (
        fit_weights_from_separators,
        match_realized_pairs,
        resolve_separator_observations,
    )

    pts = np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]], dtype=float)
    domain = Box(((-5, 5), (-5, 5), (-5, 5)))
    constraints = resolve_separator_observations(
        pts,
        [(0, 1, 0.5)],
        measurement='fraction',
        domain=domain,
    )
    fit = fit_weights_from_separators(pts, constraints)
    diag = match_realized_pairs(
        pts,
        domain=domain,
        radii=fit.radii,
        constraints=constraints,
        return_boundary_measure=True,
    )

    assert bool(diag.realized[0]) is True
    assert diag.boundary_measure is not None
    assert np.isfinite(diag.boundary_measure[0])
    assert diag.boundary_measure[0] > 0.0


def test_match_realized_pairs_can_return_tessellation_diagnostics():
    from pyvoro2 import Box
    from pyvoro2.inverse.separator import (
        fit_weights_from_separators,
        match_realized_pairs,
        resolve_separator_observations,
    )

    pts = np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]], dtype=float)
    domain = Box(((-5, 5), (-5, 5), (-5, 5)))
    constraints = resolve_separator_observations(
        pts,
        [(0, 1, 0.5)],
        measurement='fraction',
        domain=domain,
    )
    fit = fit_weights_from_separators(pts, constraints)
    diag = match_realized_pairs(
        pts,
        domain=domain,
        radii=fit.radii,
        constraints=constraints,
        return_tessellation_diagnostics=True,
    )

    assert diag.tessellation_diagnostics is not None
    assert diag.tessellation_diagnostics.n_cells_returned == 2
    assert diag.tessellation_diagnostics.ok is True


def test_match_realized_pairs_reports_periodic_wrong_shift():
    from pyvoro2 import PeriodicCell
    from pyvoro2.inverse.separator import (
        fit_weights_from_separators,
        match_realized_pairs,
        resolve_separator_observations,
    )

    cell = PeriodicCell(vectors=((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)))
    pts = np.array([[0.1, 0.5, 0.5], [0.9, 0.5, 0.5]], dtype=float)
    constraints = resolve_separator_observations(
        pts,
        [(0, 1, 0.5, (1, 0, 0))],
        measurement='fraction',
        domain=cell,
        image='given_only',
    )
    fit = fit_weights_from_separators(pts, constraints)
    diag = match_realized_pairs(
        pts,
        domain=cell,
        radii=fit.radii,
        constraints=constraints,
    )

    assert bool(diag.realized[0]) is True
    assert bool(diag.realized_same_shift[0]) is False
    assert bool(diag.realized_other_shift[0]) is True
    assert (-1, 0, 0) in diag.realized_shifts[0]
    assert (1, 0, 0) not in diag.realized_shifts[0]
    assert diag.unaccounted_pairs == tuple()


def test_realized_pair_diagnostics_export_records():
    from pyvoro2 import Box
    from pyvoro2.inverse.separator import (
        match_realized_pairs,
        resolve_separator_observations,
    )

    pts = np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]], dtype=float)
    box = Box(((-5.0, 5.0), (-5.0, 5.0), (-5.0, 5.0)))
    constraints = resolve_separator_observations(
        pts,
        [(11, 22, 0.5)],
        ids=[11, 22],
        index_mode='id',
        measurement='fraction',
        domain=box,
    )

    realized = match_realized_pairs(
        pts, domain=box, radii=np.array([1.0, 1.0]), constraints=constraints
    )
    rows = realized.to_records(constraints, use_ids=True)
    assert len(rows) == 1
    assert rows[0]['site_i'] == 11
    assert rows[0]['site_j'] == 22
    assert rows[0]['realized'] is True


def test_match_realized_pairs_supports_planar_measure_and_diag() -> None:
    import pyvoro2.planar as pv2
    from pyvoro2.inverse.separator import (
        fit_weights_from_separators,
        match_realized_pairs,
        resolve_separator_observations,
    )

    pts = np.array([[0.0, 0.0], [2.0, 0.0]], dtype=float)
    domain = pv2.Box(((-5.0, 5.0), (-5.0, 5.0)))
    constraints = resolve_separator_observations(
        pts,
        [(0, 1, 0.5)],
        measurement='fraction',
        domain=domain,
    )
    fit = fit_weights_from_separators(pts, constraints)
    diag = match_realized_pairs(
        pts,
        domain=domain,
        radii=fit.radii,
        constraints=constraints,
        return_boundary_measure=True,
        return_tessellation_diagnostics=True,
    )

    assert bool(diag.realized[0]) is True
    assert diag.boundary_measure is not None
    assert np.isfinite(diag.boundary_measure[0])
    assert diag.boundary_measure[0] > 0.0
    assert diag.tessellation_diagnostics is not None
    assert diag.tessellation_diagnostics.n_cells_returned == 2
    assert diag.tessellation_diagnostics.ok is True


def test_match_realized_pairs_supports_planar_periodic_wrong_shift() -> None:
    import pyvoro2.planar as pv2
    from pyvoro2.inverse.separator import (
        fit_weights_from_separators,
        match_realized_pairs,
        resolve_separator_observations,
    )

    cell = pv2.RectangularCell(((0.0, 1.0), (0.0, 1.0)), periodic=(True, True))
    pts = np.array([[0.1, 0.5], [0.9, 0.5]], dtype=float)
    constraints = resolve_separator_observations(
        pts,
        [(0, 1, 0.5, (1, 0))],
        measurement='fraction',
        domain=cell,
        image='given_only',
    )
    fit = fit_weights_from_separators(pts, constraints)
    diag = match_realized_pairs(
        pts,
        domain=cell,
        radii=fit.radii,
        constraints=constraints,
    )

    assert bool(diag.realized[0]) is True
    assert bool(diag.realized_same_shift[0]) is False
    assert bool(diag.realized_other_shift[0]) is True
    assert (-1, 0) in diag.realized_shifts[0]
    assert (1, 0) not in diag.realized_shifts[0]
    assert diag.unaccounted_pairs == tuple()


def test_match_realized_pairs_reports_unaccounted_realized_pairs_in_3d():
    from pyvoro2 import Box
    from pyvoro2.inverse.separator import (
        fit_weights_from_separators,
        match_realized_pairs,
        resolve_separator_observations,
    )

    pts = np.array(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]],
        dtype=float,
    )
    box = Box(((-5.0, 5.0), (-5.0, 5.0), (-5.0, 5.0)))
    constraints = resolve_separator_observations(
        pts,
        [(0, 2, 0.5)],
        measurement='fraction',
        domain=box,
    )
    fit = fit_weights_from_separators(pts, constraints)
    diag = match_realized_pairs(
        pts,
        domain=box,
        radii=fit.radii,
        constraints=constraints,
        return_boundary_measure=True,
        unaccounted_pair_check='warn',
    )

    assert {(pair.site_i, pair.site_j) for pair in diag.unaccounted_pairs} == {
        (0, 1),
        (1, 2),
    }
    assert all(
        pair.boundary_measure is not None
        for pair in diag.unaccounted_pairs
    )
    assert any('candidate-absent' in msg for msg in diag.warnings)


def test_match_realized_pairs_reports_unaccounted_realized_pairs_in_planar_box(
) -> None:
    import pyvoro2.planar as pv2
    from pyvoro2.inverse.separator import (
        fit_weights_from_separators,
        match_realized_pairs,
        resolve_separator_observations,
    )

    pts = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]], dtype=float)
    box = pv2.Box(((-5.0, 5.0), (-5.0, 5.0)))
    constraints = resolve_separator_observations(
        pts,
        [(0, 2, 0.5)],
        measurement='fraction',
        domain=box,
    )
    fit = fit_weights_from_separators(pts, constraints)
    diag = match_realized_pairs(
        pts,
        domain=box,
        radii=fit.radii,
        constraints=constraints,
        return_boundary_measure=True,
        unaccounted_pair_check='diagnose',
    )

    assert {(pair.site_i, pair.site_j) for pair in diag.unaccounted_pairs} == {
        (0, 1),
        (1, 2),
    }
    assert diag.warnings == tuple()
