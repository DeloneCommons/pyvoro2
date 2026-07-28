import numpy as np
import pytest


def test_infeasible_hard_constraints_return_conflict_witness():
    from pyvoro2.inverse.separator import (
        FixedValue,
        FitModel,
        fit_weights_from_separators,
    )

    pts = np.array(
        [[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [4.0, 0.0, 0.0]],
        dtype=float,
    )
    res = fit_weights_from_separators(
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
    assert res.is_infeasible is True
    assert res.conflicting_constraint_indices == (0, 1, 2)
    assert res.conflict.constraint_indices == (0, 1, 2)
    assert res.conflict.component_nodes == (0, 1, 2)
    assert set(res.conflict.cycle_nodes) == {0, 1, 2}
    assert len(res.conflict.terms) >= 3
    assert any(term.relation == '>=' for term in res.conflict.terms)
    assert any(term.relation == '<=' for term in res.conflict.terms)
    assert 'constraint rows [0, 1, 2]' in res.conflict.message
    assert any('constraint rows [0, 1, 2]' in msg for msg in res.warnings)


def test_feasible_fit_has_no_conflict_witness():
    from pyvoro2.inverse.separator import (
        FitModel,
        Interval,
        fit_weights_from_separators,
    )

    pts = np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]], dtype=float)
    res = fit_weights_from_separators(
        pts,
        [(0, 1, 0.5)],
        measurement='fraction',
        model=FitModel(feasible=Interval(0.0, 1.0)),
        solver='admm',
        admm_max_iter=2000,
    )

    assert res.status in {'optimal', 'max_iter'}
    assert res.hard_feasible is True
    assert res.conflict is None


def test_conflict_and_fit_records_are_exportable():
    from pyvoro2.inverse.separator import (
        FixedValue,
        FitModel,
        fit_weights_from_separators,
        resolve_separator_observations,
    )

    pts = np.array(
        [[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [4.0, 0.0, 0.0]],
        dtype=float,
    )
    constraints = [(0, 1, 0.0), (1, 2, 0.0), (0, 2, 0.0)]
    res = fit_weights_from_separators(
        pts,
        constraints,
        measurement='position',
        model=FitModel(feasible=FixedValue(0.0)),
        solver='admm',
    )

    rows = res.conflict.to_records()
    assert len(rows) >= 3
    assert set(rows[0]) == {
        'constraint_index',
        'site_i',
        'site_j',
        'relation',
        'bound_value',
    }

    resolved = resolve_separator_observations(
        pts,
        constraints,
        measurement='position',
    )
    fit_rows = res.to_records(resolved)
    assert len(fit_rows) == 3
    assert fit_rows[0]['measurement'] == 'position'
    assert fit_rows[0]['predicted'] is None

    with pytest.raises(ValueError, match=r'ids\[0\] must be an integer'):
        res.conflict.to_records(ids=np.array(['10', '20', '30']))


def test_conflict_records_validate_external_ids_once(monkeypatch) -> None:
    import pyvoro2.inverse.separator.types as types_module
    from pyvoro2.inverse.separator import (
        HardConstraintConflict,
        HardConstraintConflictTerm,
    )

    original_validator = types_module._validated_ids_array
    validation_count = 0

    def counting_validator(ids, n_points=None):
        nonlocal validation_count
        validation_count += 1
        return original_validator(ids, n_points)

    monkeypatch.setattr(
        types_module,
        '_validated_ids_array',
        counting_validator,
    )
    terms = (
        HardConstraintConflictTerm(0, 0, 1, '<=', 0.0),
        HardConstraintConflictTerm(1, 1, 2, '>=', 1.0),
    )
    conflict = HardConstraintConflict(
        component_nodes=(0, 1, 2),
        cycle_nodes=(0, 1, 2),
        terms=terms,
        message='test conflict',
    )

    records = conflict.to_records(ids=[10, 20, 30])

    assert [(record['site_i'], record['site_j']) for record in records] == [
        (10, 20),
        (20, 30),
    ]
    assert validation_count == 1
