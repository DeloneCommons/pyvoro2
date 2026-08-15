from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace
import json
import warnings

import numpy as np
import pytest

import pyvoro2
import pyvoro2.inverse.separator.active as active_module
import pyvoro2.inverse.separator.problem as problem_module
import pyvoro2.inverse.separator.solver as solver_module
from pyvoro2.inverse.separator import (
    ActiveSetOptions,
    ExponentialBoundaryPenalty,
    FitModel,
    FixedValue,
    HuberLoss,
    Interval,
    L2Regularization,
    ReciprocalBoundaryPenalty,
    SoftIntervalPenalty,
    build_power_fit_problem,
    build_power_fit_result,
    build_fit_report,
    dumps_report_json,
    fit_weights_from_separators,
    match_realized_pairs,
    resolve_separator_observations,
    solve_self_consistent_power_weights,
)


class _Truthy:
    def __bool__(self) -> bool:
        return True


class _ExplodingArray:
    def __array__(self, dtype=None, copy=None):
        raise AssertionError('array coercion must not run for an invalid string mode')


class EqualAny:
    def __init__(self) -> None:
        self.comparisons = 0

    def __eq__(self, other: object) -> bool:
        self.comparisons += 1
        return True

    def __ne__(self, other: object) -> bool:
        self.comparisons += 1
        return False


_STRING_IMPOSTOR_FACTORIES = (
    pytest.param(lambda value: np.array(value), id='zero_dim_array'),
    pytest.param(lambda value: np.array([value]), id='one_element_array'),
    pytest.param(
        lambda value: np.array([value, value]),
        id='multi_element_array',
    ),
    pytest.param(
        lambda value: np.array(value, dtype=object),
        id='object_array',
    ),
    pytest.param(lambda value: value.encode(), id='bytes'),
    pytest.param(lambda value: bytearray(value.encode()), id='bytearray'),
    pytest.param(lambda value: 1, id='integer'),
    pytest.param(lambda value: True, id='boolean'),
    pytest.param(lambda value: None, id='none'),
    pytest.param(lambda value: EqualAny(), id='equal_any'),
)


_NON_NONE_STRING_IMPOSTOR_FACTORIES = _STRING_IMPOSTOR_FACTORIES[:-2] + (
    _STRING_IMPOSTOR_FACTORIES[-1],
)


def _points() -> np.ndarray:
    return np.array([[0.25, 0.5], [1.75, 0.5]], dtype=np.float64)


def _box() -> pyvoro2.planar.Box:
    return pyvoro2.planar.Box(((0.0, 2.0), (0.0, 1.0)))


def _resolved():
    return resolve_separator_observations(
        _points(),
        [(10, 20, 0.5, (0, 0))],
        ids=[10, 20],
        index_mode='id',
        domain=pyvoro2.planar.RectangularCell(
            ((0.0, 2.0), (0.0, 1.0)),
            periodic=(True, True),
        ),
        image='given_only',
    )


@pytest.mark.parametrize('value', [1, np.int32(2), np.uint64(3)])
def test_positive_solver_counts_accept_exact_integer_scalars(value: object) -> None:
    result = fit_weights_from_separators(
        _points(),
        [(0, 1, 0.5)],
        admm_max_iter=value,
    )
    assert result.status == 'optimal'


@pytest.mark.parametrize(
    'invalid',
    [
        True,
        np.bool_(False),
        1.0,
        1.5,
        '1',
        1 + 0j,
        np.array(1),
        0,
        -1,
    ],
)
def test_positive_solver_counts_reject_non_positive_indices(
    invalid: object,
) -> None:
    with pytest.raises(ValueError, match='admm_max_iter'):
        fit_weights_from_separators(
            _points(),
            [(0, 1, 0.5)],
            admm_max_iter=invalid,
        )


@pytest.mark.parametrize('value', [0, np.int32(1), np.uint64(2)])
def test_image_search_accepts_nonnegative_exact_integers(value: object) -> None:
    observations = resolve_separator_observations(
        _points(),
        [(0, 1, 0.5)],
        image_search=value,
    )
    assert observations.n_constraints == 1


@pytest.mark.parametrize(
    'invalid',
    [True, np.bool_(False), 0.0, 1.0, '1', 1 + 0j, np.array(1), -1],
)
def test_image_search_rejects_non_indices(invalid: object) -> None:
    with pytest.raises(ValueError, match='image_search'):
        resolve_separator_observations(
            _points(),
            [(0, 1, 0.5)],
            image_search=invalid,
        )


@pytest.mark.parametrize('component', [0, np.int32(-1), np.uint64(1)])
def test_periodic_shift_components_accept_exact_int64_values(
    component: object,
) -> None:
    observations = resolve_separator_observations(
        _points(),
        [(0, 1, 0.5, (component, 0))],
        domain=pyvoro2.planar.RectangularCell(
            ((0, 2), (0, 1)),
            periodic=(True, True),
        ),
        image='given_only',
    )
    assert int(observations.shifts[0, 0]) == int(component)


@pytest.mark.parametrize(
    'invalid',
    [True, np.bool_(False), 0.0, 1.5, '1', 1 + 0j, np.array(1), 2**63],
)
def test_periodic_shift_components_reject_lossy_or_overflow_values(
    invalid: object,
) -> None:
    with pytest.raises(ValueError, match='shift'):
        resolve_separator_observations(
            _points(),
            [(0, 1, 0.5, (invalid, 0))],
            domain=pyvoro2.planar.RectangularCell(
                ((0, 2), (0, 1)),
                periodic=(True, True),
            ),
            image='given_only',
        )


@pytest.mark.parametrize(
    'invalid',
    [True, np.bool_(False), 1.0, '1', 1 + 0j, np.array(1), -1, 2**63],
)
def test_direct_observation_indices_are_exact_and_bounded(invalid: object) -> None:
    observations = _resolved()
    with pytest.raises(ValueError, match='input_index'):
        replace(observations, input_index=[invalid])
    with pytest.raises(ValueError, match=r'\.i'):
        replace(observations, i=[invalid])


@pytest.mark.parametrize(
    ('field', 'invalid'),
    [
        ('add_after', 0),
        ('drop_after', -1),
        ('max_iter', 1.0),
        ('cycle_window', True),
        ('relax', 0.0),
        ('relax', 1.01),
        ('weight_step_tol', -1.0),
    ],
)
def test_active_set_options_enforce_exact_counts_and_ranges(
    field: str,
    invalid: object,
) -> None:
    with pytest.raises(ValueError, match=f'ActiveSetOptions.{field}'):
        ActiveSetOptions(**{field: invalid})


def test_active_set_options_store_python_scalars() -> None:
    options = ActiveSetOptions(
        add_after=np.int32(1),
        drop_after=np.uint64(2),
        max_iter=np.int64(3),
        cycle_window=np.uint32(4),
        relax=np.float32(0.5),
        weight_step_tol=np.float32(1e-6),
    )

    assert type(options.add_after) is int
    assert type(options.drop_after) is int
    assert type(options.max_iter) is int
    assert type(options.cycle_window) is int
    assert type(options.relax) is float
    assert type(options.weight_step_tol) is float


@pytest.mark.parametrize('invalid', [0, 1, 'true', '', _Truthy()])
def test_inverse_public_flags_reject_non_boolean_values(invalid: object) -> None:
    with pytest.raises(ValueError, match='allow_empty.*Boolean'):
        resolve_separator_observations(
            _points(),
            [(0, 1, 0.5)],
            allow_empty=invalid,
        )
    with pytest.raises(ValueError, match='return_boundary_measure.*Boolean'):
        match_realized_pairs(
            _points(),
            domain=_box(),
            constraints=_resolved(),
            weights=np.zeros(2),
            return_boundary_measure=invalid,
        )
    with pytest.raises(ValueError, match='return_history.*Boolean'):
        solve_self_consistent_power_weights(
            _points(),
            [(0, 1, 0.5)],
            domain=_box(),
            return_history=invalid,
        )


@pytest.mark.parametrize(
    'mask',
    [
        [1],
        [0],
        ['true'],
        np.array([1], dtype=np.int8),
        np.array([True, 1], dtype=object),
    ],
)
def test_observation_subset_requires_an_exact_boolean_mask(mask: object) -> None:
    with pytest.raises(ValueError, match='mask.*Boolean'):
        _resolved().subset(mask)


def test_observation_subset_accepts_boolean_scalars_and_owns_mask_result() -> None:
    observations = _resolved()
    mask = np.array([np.bool_(True)], dtype=np.bool_)
    subset = observations.subset(mask)
    mask[0] = False

    assert subset.n_constraints == 1
    assert subset.explicit_shift.flags.writeable is False


def test_direct_observation_boolean_masks_are_strict() -> None:
    observations = _resolved()
    with pytest.raises(ValueError, match='explicit_shift.*Boolean'):
        replace(observations, explicit_shift=np.array([1], dtype=np.int8))


def test_active0_requires_an_exact_boolean_mask() -> None:
    with pytest.raises(ValueError, match='active0.*Boolean'):
        solve_self_consistent_power_weights(
            _points(),
            [(0, 1, 0.5)],
            domain=_box(),
            active0=np.array([1], dtype=np.int8),
        )


@pytest.mark.parametrize('nonfinite', [np.nan, np.inf, -np.inf])
@pytest.mark.parametrize(
    'factory',
    [
        lambda value: HuberLoss(delta=value),
        lambda value: Interval(lower=value, upper=1.0),
        lambda value: FixedValue(value=value),
        lambda value: SoftIntervalPenalty(0.0, 1.0, value),
        lambda value: ExponentialBoundaryPenalty(tau=value),
        lambda value: ReciprocalBoundaryPenalty(epsilon=value),
        lambda value: L2Regularization(strength=value),
        lambda value: ActiveSetOptions(weight_step_tol=value),
    ],
)
def test_inverse_model_and_active_nan_inf_matrix(
    factory,
    nonfinite: float,
) -> None:
    with pytest.raises(ValueError, match='finite'):
        factory(nonfinite)


@pytest.mark.parametrize('invalid', [True, '1', 1 + 0j])
def test_inverse_model_scalars_reject_non_real_categories(invalid: object) -> None:
    with pytest.raises(ValueError, match='HuberLoss.delta.*real numeric'):
        HuberLoss(delta=invalid)


def test_model_scalars_are_canonical_python_floats() -> None:
    model = FitModel(
        mismatch=HuberLoss(np.float32(0.5)),
        feasible=Interval(np.int32(0), np.float32(1)),
        penalties=[SoftIntervalPenalty(0, 1, np.float32(2))],
        regularization=L2Regularization(np.float32(0.25)),
    )

    assert type(model.mismatch.delta) is float
    assert type(model.feasible.lower) is float
    assert type(model.feasible.upper) is float
    assert type(model.penalties[0].strength) is float
    assert type(model.regularization.strength) is float
    assert type(model.penalties) is tuple


@pytest.mark.parametrize('nonfinite', [np.nan, np.inf, -np.inf])
def test_l2_reference_and_confidence_nan_inf_matrix(nonfinite: float) -> None:
    with pytest.raises(ValueError, match='reference.*finite'):
        L2Regularization(reference=np.array([0.0, nonfinite]))
    with pytest.raises(ValueError, match='confidence.*finite'):
        resolve_separator_observations(
            _points(),
            [(0, 1, 0.5)],
            confidence=[nonfinite],
        )


@pytest.mark.parametrize(
    'reference',
    [
        [True, False],
        ['0', '1'],
        [0 + 0j, 1 + 0j],
        [[0.0, 1.0]],
    ],
)
def test_l2_reference_rejects_wrong_kind_or_shape(reference: object) -> None:
    with pytest.raises(ValueError, match='reference'):
        L2Regularization(reference=reference)


def test_l2_reference_is_owned_read_only_even_at_zero_strength() -> None:
    reference = np.array([1.0, 2.0])
    regularization = L2Regularization(strength=0.0, reference=reference)

    reference[:] = -1

    np.testing.assert_array_equal(regularization.reference, [1.0, 2.0])
    assert regularization.reference.flags.c_contiguous
    assert regularization.reference.flags.owndata
    assert regularization.reference.flags.writeable is False


def test_fit_model_owns_penalty_sequence() -> None:
    penalties = [SoftIntervalPenalty(0.0, 1.0, 1.0)]
    model = FitModel(penalties=penalties)
    penalties.clear()

    assert len(model.penalties) == 1
    assert isinstance(model.penalties, tuple)

    generator_model = FitModel(
        penalties=(
            penalty
            for penalty in [SoftIntervalPenalty(0.0, 1.0, 2.0)]
        )
    )
    assert isinstance(generator_model.penalties, tuple)
    assert len(generator_model.penalties) == 1


@pytest.mark.parametrize('nonfinite', [np.nan, np.inf, -np.inf])
@pytest.mark.parametrize(
    'kwargs',
    [
        {'admm_rho': None},
        {'admm_abs_tol': None},
        {'admm_rel_tol': None},
        {'r_min': None},
        {'weight_shift': None},
    ],
)
def test_solver_nan_inf_matrix(kwargs: dict[str, object], nonfinite: float) -> None:
    name = next(iter(kwargs))
    values = {name: nonfinite}
    with pytest.raises(ValueError, match=name):
        fit_weights_from_separators(
            _points(),
            [(0, 1, 0.5)],
            **values,
        )


def test_representation_option_conflict_precedes_all_numerical_work(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    problem = build_power_fit_problem(_resolved())

    def fail_if_called(*args, **kwargs):
        raise AssertionError('numerical work was entered')

    monkeypatch.setattr(
        solver_module,
        '_fit_power_weights_resolved',
        fail_if_called,
    )
    with pytest.raises(ValueError, match='at most one of r_min and weight_shift'):
        fit_weights_from_separators(
            _points(),
            [(0, 1, 0.5)],
            r_min=1.0,
            weight_shift=0.0,
        )

    monkeypatch.setattr(
        active_module,
        'fit_weights_from_separators',
        fail_if_called,
    )
    with pytest.raises(ValueError, match='at most one of r_min and weight_shift'):
        solve_self_consistent_power_weights(
            _points(),
            [(0, 1, 0.5)],
            domain=_box(),
            r_min=1.0,
            weight_shift=0.0,
        )

    monkeypatch.setattr(
        problem_module,
        '_validated_weight_vector',
        fail_if_called,
    )
    with pytest.raises(ValueError, match='at most one of r_min and weight_shift'):
        build_power_fit_result(
            problem,
            np.zeros(2),
            r_min=1.0,
            weight_shift=0.0,
        )


@pytest.mark.parametrize(
    ('points', 'measurement', 'target'),
    [
        (np.array([[-1e308, 0.0], [1e308, 0.0]]), 'fraction', 0.5),
        (np.array([[-1e200, 0.0], [1e200, 0.0]]), 'fraction', 0.5),
        (np.array([[0.0, 0.0], [1e154, 0.0]]), 'fraction', 1e308),
        (np.array([[0.0, 0.0], [1e-160, 0.0]]), 'position', 1e308),
    ],
)
def test_nonrepresentable_connector_geometry_has_stable_warning_free_error(
    points: np.ndarray,
    measurement: str,
    target: float,
) -> None:
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        with np.errstate(all='raise'):
            with pytest.raises(
                ValueError,
                match='derived separator connector.*finite',
            ):
                resolve_separator_observations(
                    points,
                    [(0, 1, target)],
                    measurement=measurement,
                )

    assert caught == []


def test_valid_extreme_connector_geometry_remains_supported_without_warnings(
) -> None:
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        with np.errstate(all='raise'):
            observations = resolve_separator_observations(
                np.array([[0.0, 0.0], [1e154, 0.0]]),
                [(0, 1, 1e308)],
                measurement='position',
            )

    assert caught == []
    assert observations.distance[0] == pytest.approx(1e154)
    assert observations.target_fraction[0] == pytest.approx(1e154)


@pytest.mark.parametrize(
    'name',
    [
        'fit_admm_rho',
        'fit_admm_abs_tol',
        'fit_admm_rel_tol',
        'r_min',
        'weight_shift',
    ],
)
@pytest.mark.parametrize('nonfinite', [np.nan, np.inf, -np.inf])
def test_active_solver_nan_inf_matrix(name: str, nonfinite: float) -> None:
    with pytest.raises(ValueError, match=name):
        solve_self_consistent_power_weights(
            _points(),
            [(0, 1, 0.5)],
            domain=_box(),
            **{name: nonfinite},
        )


@pytest.mark.parametrize(
    'invalid',
    [True, np.bool_(False), 1.0, '1', 0, -1],
)
def test_active_fit_iteration_count_is_a_positive_exact_integer(
    invalid: object,
) -> None:
    with pytest.raises(ValueError, match='fit_admm_max_iter'):
        solve_self_consistent_power_weights(
            _points(),
            [(0, 1, 0.5)],
            domain=_box(),
            fit_admm_max_iter=invalid,
        )


def test_separator_observations_own_all_retained_arrays() -> None:
    source = _resolved()
    names = (
        'i',
        'j',
        'shifts',
        'target',
        'confidence',
        'distance',
        'distance2',
        'delta',
        'target_fraction',
        'target_position',
        'input_index',
        'explicit_shift',
        'ids',
    )
    caller_arrays = {
        name: np.array(getattr(source, name), copy=True)
        for name in names
    }
    observations = replace(source, **caller_arrays)
    snapshots = {
        name: np.array(getattr(observations, name), copy=True)
        for name in names
    }

    for values in caller_arrays.values():
        if values.dtype.kind == 'b':
            values[:] = ~values
        else:
            values[...] = 0

    for name, expected in snapshots.items():
        actual = getattr(observations, name)
        np.testing.assert_array_equal(actual, expected)
        assert actual.flags.c_contiguous
        assert actual.flags.owndata
        assert actual.flags.writeable is False


def test_result_builder_options_are_exact_before_packaging() -> None:
    problem = build_power_fit_problem(_resolved())
    with pytest.raises(ValueError, match='canonicalize_gauge.*Boolean'):
        build_power_fit_result(
            problem,
            np.zeros(2),
            canonicalize_gauge=1,
        )
    with pytest.raises(ValueError, match='n_iter'):
        build_power_fit_result(problem, np.zeros(2), n_iter=1.0)


@pytest.mark.parametrize('invalid', [0, 1, 'true', _Truthy()])
def test_report_flags_are_exact_booleans(invalid: object) -> None:
    observations = _resolved()
    fit = fit_weights_from_separators(_points(), observations)
    with pytest.raises(ValueError, match='use_ids.*Boolean'):
        build_fit_report(fit, observations, use_ids=invalid)
    with pytest.raises(ValueError, match='sort_keys.*Boolean'):
        dumps_report_json({}, sort_keys=invalid)


@pytest.mark.parametrize(
    'invalid',
    [True, np.bool_(False), 2.0, '2', 2 + 0j, np.array(2)],
)
def test_report_indent_is_an_exact_integer(invalid: object) -> None:
    with pytest.raises(ValueError, match='indent.*exact integer'):
        dumps_report_json({}, indent=invalid)


def test_direct_problem_arrays_are_strict_owned_and_read_only() -> None:
    source = build_power_fit_problem(_resolved())
    alpha = np.array(source.alpha, copy=True)
    mask = np.array(source.offset_identifying_constraint_mask, copy=True)
    problem = replace(
        source,
        alpha=alpha,
        offset_identifying_constraint_mask=mask,
    )

    alpha[:] = 99
    mask[:] = ~mask

    np.testing.assert_array_equal(problem.alpha, source.alpha)
    np.testing.assert_array_equal(
        problem.offset_identifying_constraint_mask,
        source.offset_identifying_constraint_mask,
    )
    assert problem.alpha.flags.owndata
    assert problem.alpha.flags.writeable is False
    assert problem.offset_identifying_constraint_mask.flags.owndata
    assert problem.offset_identifying_constraint_mask.flags.writeable is False
    with pytest.raises(ValueError, match='offset_identifying.*Boolean'):
        replace(
            source,
            offset_identifying_constraint_mask=np.array([1], dtype=np.int8),
        )
    with pytest.raises(ValueError, match='z_obs.*finite'):
        replace(source, z_obs=np.array([np.inf]))


def test_reciprocal_penalty_relations_remain_strict() -> None:
    with pytest.raises(ValueError, match='0 < epsilon < margin'):
        ReciprocalBoundaryPenalty(margin=0.1, epsilon=0.1)
    with pytest.raises(ValueError, match='margin is too large'):
        ReciprocalBoundaryPenalty(lower=0.0, upper=0.1, margin=0.1)


@pytest.mark.parametrize('make_invalid', _STRING_IMPOSTOR_FACTORIES)
def test_direct_observation_measurement_rejects_non_string_categories_first(
    make_invalid: Callable[[str], object],
) -> None:
    invalid = make_invalid('fraction')

    with pytest.raises(ValueError, match='measurement'):
        replace(
            _resolved(),
            measurement=invalid,
            i=_ExplodingArray(),
        )

    if isinstance(invalid, EqualAny):
        assert invalid.comparisons == 0


@pytest.mark.parametrize('make_invalid', _STRING_IMPOSTOR_FACTORIES)
@pytest.mark.parametrize(
    ('field', 'allowed'),
    [
        ('measurement', 'fraction'),
        ('index_mode', 'index'),
        ('image', 'nearest'),
    ],
)
def test_resolver_string_modes_reject_before_point_and_graph_work(
    make_invalid: Callable[[str], object],
    field: str,
    allowed: str,
) -> None:
    invalid = make_invalid(allowed)

    with pytest.raises(ValueError, match=field):
        resolve_separator_observations(
            _ExplodingArray(),
            [],
            **{field: invalid},
        )

    if isinstance(invalid, EqualAny):
        assert invalid.comparisons == 0


@pytest.mark.parametrize('make_invalid', _STRING_IMPOSTOR_FACTORIES)
@pytest.mark.parametrize(
    ('field', 'allowed'),
    [
        ('measurement', 'fraction'),
        ('index_mode', 'index'),
        ('image', 'nearest'),
        ('solver', 'direct'),
        ('linear_backend', 'dense'),
        ('connectivity_check', 'warn'),
    ],
)
def test_fit_string_modes_reject_before_point_or_solver_work(
    make_invalid: Callable[[str], object],
    field: str,
    allowed: str,
) -> None:
    invalid = make_invalid(allowed)

    with pytest.raises(ValueError, match=field):
        fit_weights_from_separators(
            _ExplodingArray(),
            [],
            **{field: invalid},
        )

    if isinstance(invalid, EqualAny):
        assert invalid.comparisons == 0


@pytest.mark.parametrize('make_invalid', _STRING_IMPOSTOR_FACTORIES)
@pytest.mark.parametrize(
    ('field', 'allowed'),
    [
        ('tessellation_check', 'diagnose'),
        ('unaccounted_pair_check', 'diagnose'),
    ],
)
def test_realization_string_policies_reject_before_point_or_native_work(
    make_invalid: Callable[[str], object],
    field: str,
    allowed: str,
) -> None:
    invalid = make_invalid(allowed)

    with pytest.raises(ValueError, match=field):
        match_realized_pairs(
            _ExplodingArray(),
            domain=_box(),
            constraints=_resolved(),
            weights=np.zeros(2),
            **{field: invalid},
        )

    if isinstance(invalid, EqualAny):
        assert invalid.comparisons == 0


@pytest.mark.parametrize('make_invalid', _STRING_IMPOSTOR_FACTORIES)
@pytest.mark.parametrize(
    ('field', 'allowed'),
    [
        ('measurement', 'fraction'),
        ('index_mode', 'index'),
        ('image', 'nearest'),
        ('fit_solver', 'direct'),
        ('fit_linear_backend', 'dense'),
        ('connectivity_check', 'warn'),
        ('unaccounted_pair_check', 'warn'),
        ('tessellation_check', 'diagnose'),
    ],
)
def test_active_set_string_modes_reject_before_point_or_solver_work(
    make_invalid: Callable[[str], object],
    field: str,
    allowed: str,
) -> None:
    invalid = make_invalid(allowed)

    with pytest.raises(ValueError, match=field):
        solve_self_consistent_power_weights(
            _ExplodingArray(),
            [],
            domain=_box(),
            **{field: invalid},
        )

    if isinstance(invalid, EqualAny):
        assert invalid.comparisons == 0


@pytest.mark.parametrize('make_invalid', _STRING_IMPOSTOR_FACTORIES)
@pytest.mark.parametrize('field', ['solver', 'status'])
def test_result_builder_required_strings_reject_before_weight_packaging(
    make_invalid: Callable[[str], object],
    field: str,
) -> None:
    problem = build_power_fit_problem(_resolved())
    invalid = make_invalid('external')

    with pytest.raises(ValueError, match=field):
        build_power_fit_result(
            problem,
            _ExplodingArray(),
            **{field: invalid},
        )

    if isinstance(invalid, EqualAny):
        assert invalid.comparisons == 0


@pytest.mark.parametrize(
    'make_invalid',
    _NON_NONE_STRING_IMPOSTOR_FACTORIES,
)
@pytest.mark.parametrize('field', ['linear_backend', 'status_detail'])
def test_result_builder_optional_strings_reject_before_weight_packaging(
    make_invalid: Callable[[str], object],
    field: str,
) -> None:
    problem = build_power_fit_problem(_resolved())
    invalid = make_invalid('metadata')

    with pytest.raises(ValueError, match=field):
        build_power_fit_result(
            problem,
            _ExplodingArray(),
            **{field: invalid},
        )

    if isinstance(invalid, EqualAny):
        assert invalid.comparisons == 0


def test_numpy_string_scalars_canonicalize_across_inverse_boundaries() -> None:
    points = _points()
    domain = _box()
    observations = resolve_separator_observations(
        points,
        [(0, 1, 0.5)],
        measurement=np.str_('fraction'),
        domain=domain,
        index_mode=np.str_('index'),
        image=np.str_('nearest'),
    )
    assert type(observations.measurement) is str

    fit = fit_weights_from_separators(
        points,
        observations,
        measurement=np.str_('fraction'),
        index_mode=np.str_('index'),
        image=np.str_('nearest'),
        solver=np.str_('direct'),
        linear_backend=np.str_('dense'),
        connectivity_check=np.str_('warn'),
    )
    assert type(fit.measurement) is str
    assert type(fit.solver) is str
    assert type(fit.linear_backend) is str

    realized = match_realized_pairs(
        points,
        domain=domain,
        constraints=observations,
        weights=fit.weights,
        tessellation_check=np.str_('none'),
        unaccounted_pair_check=np.str_('none'),
    )
    assert all(type(message) is str for message in realized.warnings)

    active = solve_self_consistent_power_weights(
        points,
        observations,
        domain=domain,
        measurement=np.str_('fraction'),
        index_mode=np.str_('index'),
        image=np.str_('nearest'),
        fit_solver=np.str_('direct'),
        fit_linear_backend=np.str_('dense'),
        connectivity_check=np.str_('none'),
        unaccounted_pair_check=np.str_('none'),
        tessellation_check=np.str_('none'),
        options=ActiveSetOptions(max_iter=2),
    )
    assert type(active.termination) is str
    assert type(active.fit.solver) is str
    assert type(active.fit.linear_backend) is str


def test_result_builder_free_form_numpy_strings_are_json_friendly() -> None:
    problem = build_power_fit_problem(_resolved())
    result = build_power_fit_result(
        problem,
        np.zeros(2),
        solver=np.str_('external-custom'),
        linear_backend=np.str_('custom-backend'),
        status=np.str_('candidate'),
        status_detail=np.str_('caller supplied'),
        converged=False,
    )

    assert type(result.solver) is str
    assert type(result.linear_backend) is str
    assert type(result.status) is str
    assert type(result.status_detail) is str

    report = build_fit_report(result, problem.constraints)
    payload = json.loads(dumps_report_json(report))
    assert payload['summary']['solver'] == 'external-custom'
    assert payload['summary']['linear_backend'] == 'custom-backend'
    assert payload['summary']['status'] == 'candidate'
    assert payload['summary']['status_detail'] == 'caller supplied'
    assert type(payload['summary']['solver']) is str
    assert type(payload['summary']['status']) is str


def test_result_builder_optional_string_none_contract_is_unchanged() -> None:
    result = build_power_fit_result(
        build_power_fit_problem(_resolved()),
        np.zeros(2),
        linear_backend=None,
        status_detail=None,
    )

    assert result.linear_backend is None
    assert result.status_detail is None


def test_inverse_string_choice_sets_remain_exact_and_case_sensitive() -> None:
    with pytest.raises(ValueError, match='measurement'):
        resolve_separator_observations(
            _points(),
            [(0, 1, 0.5)],
            measurement='Fraction',
        )
    with pytest.raises(ValueError, match='solver'):
        fit_weights_from_separators(
            _points(),
            [(0, 1, 0.5)],
            solver='Direct',
        )
    with pytest.raises(ValueError, match='tessellation_check'):
        match_realized_pairs(
            _points(),
            domain=_box(),
            constraints=_resolved(),
            weights=np.zeros(2),
            tessellation_check='Diagnose',
        )


@pytest.mark.parametrize('make_invalid', _STRING_IMPOSTOR_FACTORIES)
def test_sparse_format_rejects_non_string_categories_before_scipy_conversion(
    make_invalid: Callable[[str], object],
) -> None:
    graph = build_power_fit_problem(_resolved()).observation_graph
    invalid = make_invalid('csr')

    with pytest.raises(ValueError, match='format'):
        graph.incidence_sparse(format=invalid)

    if isinstance(invalid, EqualAny):
        assert invalid.comparisons == 0


def test_sparse_format_accepts_numpy_string_scalar_without_changing_choice() -> None:
    graph = build_power_fit_problem(_resolved()).observation_graph
    matrix = graph.incidence_sparse(format=np.str_('csr'))

    assert matrix.format == 'csr'
