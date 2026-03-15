import numpy as np


def test_infeasible_hard_constraints_return_conflict_witness():
    from pyvoro2 import FixedValue, FitModel, fit_power_weights

    pts = np.array(
        [[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [4.0, 0.0, 0.0]],
        dtype=float,
    )
    res = fit_power_weights(
        pts,
        [(0, 1, 0.0), (1, 2, 0.0), (0, 2, 0.0)],
        measurement='position',
        model=FitModel(feasible=FixedValue(0.0)),
        solver='admm',
    )

    assert res.status == 'infeasible_hard_constraints'
    assert res.hard_feasible is False
    assert res.weights is None
    assert res.conflict is not None
    assert res.infeasible_constraints == (0, 1, 2)
    assert res.conflict.constraint_indices == (0, 1, 2)
    assert res.conflict.component_nodes == (0, 1, 2)
    assert set(res.conflict.cycle_nodes) == {0, 1, 2}
    assert len(res.conflict.terms) >= 3
    assert any(term.relation == '>=' for term in res.conflict.terms)
    assert any(term.relation == '<=' for term in res.conflict.terms)
    assert 'constraint rows [0, 1, 2]' in res.conflict.message
    assert any('constraint rows [0, 1, 2]' in msg for msg in res.warnings)


def test_feasible_fit_has_no_conflict_witness():
    from pyvoro2 import FitModel, Interval, fit_power_weights

    pts = np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]], dtype=float)
    res = fit_power_weights(
        pts,
        [(0, 1, 0.5)],
        measurement='fraction',
        model=FitModel(feasible=Interval(0.0, 1.0)),
        solver='admm',
        max_iter=2000,
    )

    assert res.status in {'optimal', 'max_iter'}
    assert res.hard_feasible is True
    assert res.conflict is None
