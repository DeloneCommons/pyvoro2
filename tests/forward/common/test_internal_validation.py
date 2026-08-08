from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest

from pyvoro2._internal.inputs import (
    checked_int64_add,
    owned_readonly_array,
    require_internal_id_range,
    require_planar_ghost_site_id_range,
    require_query_index_range,
)
from pyvoro2._internal.validation import (
    CPP_INT_MAX,
    require_bool,
    require_bool_mask,
    require_bool_tuple,
    require_finite_real,
    require_index,
    require_nonnegative_finite_real,
    require_nonnegative_index,
    require_optional_bool,
    require_ordered_bounds,
    require_positive_finite_real,
    require_positive_index,
    require_real_in_interval,
    require_optional_string,
    require_string,
    require_string_choice,
    require_string_tuple,
)


@dataclass(frozen=True)
class IndexValue:
    value: int

    def __index__(self) -> int:
        return self.value


class EqualAny:
    def __init__(self) -> None:
        self.comparisons = 0

    def __eq__(self, other: object) -> bool:
        self.comparisons += 1
        return True

    def __ne__(self, other: object) -> bool:
        self.comparisons += 1
        return False


class StringSubclass(str):
    pass


class TupleSubclass(tuple):
    pass


@pytest.mark.parametrize(
    'value',
    [1, np.int32(2), np.uint64(3), IndexValue(4)],
)
def test_require_positive_index_accepts_exact_index_scalars(value: object) -> None:
    assert require_positive_index(value, name='count', maximum=CPP_INT_MAX) >= 1


@pytest.mark.parametrize(
    'value',
    [True, np.bool_(False), 1.0, 1.5, '1', 1 + 0j, np.array(1)],
)
def test_require_index_rejects_non_exact_integer_scalars(value: object) -> None:
    with pytest.raises(ValueError, match='count.*exact integer'):
        require_index(value, name='count')


def test_index_helpers_enforce_sign_and_destination_range() -> None:
    assert require_index(-2, name='offset') == -2
    assert require_nonnegative_index(0, name='offset') == 0
    assert require_positive_index(1, name='count') == 1

    with pytest.raises(ValueError, match='count.*positive'):
        require_positive_index(0, name='count')
    with pytest.raises(ValueError, match='count.*positive'):
        require_positive_index(-1, name='count')
    with pytest.raises(ValueError, match='offset.*non-negative'):
        require_nonnegative_index(-1, name='offset')
    with pytest.raises(ValueError, match='count.*destination range'):
        require_positive_index(
            CPP_INT_MAX + 1,
            name='count',
            maximum=CPP_INT_MAX,
        )


def test_internal_id_range_is_checked_before_int32_id_construction() -> None:
    require_internal_id_range(CPP_INT_MAX + 1)
    with pytest.raises(ValueError, match=r'points length.*C\+\+ int range'):
        require_internal_id_range(CPP_INT_MAX + 2)


def test_ghost_query_index_range_has_exact_cpp_int_boundary() -> None:
    require_query_index_range(CPP_INT_MAX + 1)
    with pytest.raises(
        ValueError,
        match=r'queries length.*query indices.*destination range',
    ):
        require_query_index_range(CPP_INT_MAX + 2)


def test_planar_ghost_site_range_reserves_cpp_int_max() -> None:
    require_planar_ghost_site_id_range(CPP_INT_MAX)
    with pytest.raises(
        ValueError,
        match=r'points/site length.*destination range.*reserved.*ghost ID',
    ):
        require_planar_ghost_site_id_range(CPP_INT_MAX + 1)


def test_exact_boolean_helpers_accept_only_python_and_numpy_booleans() -> None:
    assert require_bool(True, name='flag') is True
    assert require_bool(np.bool_(False), name='flag') is False
    assert require_optional_bool(None, name='flag') is None
    assert require_optional_bool(np.bool_(True), name='flag') is True
    assert require_bool_tuple(
        (True, np.bool_(False)),
        name='flags',
        length=2,
    ) == (True, False)

    for value in (0, 1, 'False', None):
        with pytest.raises(ValueError, match='flag.*Boolean'):
            require_bool(value, name='flag')
    with pytest.raises(ValueError, match='flags.*length-2'):
        require_bool_tuple((True,), name='flags', length=2)
    with pytest.raises(ValueError, match=r'flags\[1\].*Boolean'):
        require_bool_tuple((True, 1), name='flags', length=2)


def test_scalar_string_helpers_canonicalize_only_scalar_string_categories() -> None:
    assert type(require_string('mode', name='value')) is str
    assert type(require_string(np.str_('mode'), name='value')) is str
    assert require_optional_string(None, name='detail') is None
    assert type(require_optional_string(np.str_('detail'), name='detail')) is str
    assert require_string_choice(
        np.str_('warn'),
        name='mode',
        choices=('warn', 'raise'),
    ) == 'warn'
    assert require_string_tuple(
        (np.str_('first'), 'second'),
        name='messages',
    ) == ('first', 'second')


def test_require_string_tuple_preserves_exact_canonical_tuple_identity() -> None:
    empty = ()
    warnings = ('first', 'second')

    assert require_string_tuple(empty, name='messages') is empty
    assert require_string_tuple(warnings, name='messages') is warnings


def test_require_string_tuple_canonicalizes_string_and_tuple_subclasses() -> None:
    np_strings = (np.str_('first'), np.str_('second'))
    result = require_string_tuple(np_strings, name='messages')

    assert result == ('first', 'second')
    assert type(result) is tuple
    assert all(type(item) is str for item in result)

    subclass_values = TupleSubclass(
        (StringSubclass('first'), StringSubclass('second'))
    )
    result = require_string_tuple(subclass_values, name='messages')

    assert result == ('first', 'second')
    assert type(result) is tuple
    assert all(type(item) is str for item in result)
    assert result is not subclass_values


def test_require_string_tuple_rejects_without_comparing_arbitrary_objects() -> None:
    value = EqualAny()

    with pytest.raises(ValueError, match=r'messages\[0\].*scalar string'):
        require_string_tuple((value,), name='messages')

    assert value.comparisons == 0


@pytest.mark.parametrize(
    'value',
    [
        np.array('warn'),
        np.array(['warn']),
        np.array(['warn', 'warn']),
        np.array('warn', dtype=object),
        b'warn',
        bytearray(b'warn'),
        1,
        True,
        None,
    ],
)
def test_scalar_string_helpers_reject_non_string_categories(value: object) -> None:
    with pytest.raises(
        ValueError,
        match="mode.*'warn'.*'raise'.*scalar string",
    ):
        require_string_choice(
            value,
            name='mode',
            choices=('warn', 'raise'),
        )


def test_scalar_string_choice_rejects_without_comparing_arbitrary_objects() -> None:
    value = EqualAny()

    with pytest.raises(ValueError, match='mode.*scalar string'):
        require_string_choice(
            value,
            name='mode',
            choices=('warn', 'raise'),
        )

    assert value.comparisons == 0


def test_scalar_string_choice_preserves_exact_case_sensitive_choices() -> None:
    with pytest.raises(ValueError, match="mode.*'warn'.*'raise'"):
        require_string_choice(
            'WARN',
            name='mode',
            choices=('warn', 'raise'),
        )


def test_bool_mask_is_exact_owned_and_read_only() -> None:
    source = np.array([True, False], dtype=np.bool_)
    mask = require_bool_mask(source, name='mask', length=2)

    assert mask.tolist() == [True, False]
    assert mask.flags.owndata
    assert not mask.flags.writeable
    source[0] = False
    assert mask.tolist() == [True, False]

    for values in ([1, 0], ['True', 'False'], [True, 0]):
        with pytest.raises(ValueError, match='mask.*Boolean'):
            require_bool_mask(values, name='mask', length=2)
    with pytest.raises(ValueError, match='mask.*shape'):
        require_bool_mask([True], name='mask', length=2)


@pytest.mark.parametrize(
    'value',
    [1, -2, 0.5, np.int64(3), np.float32(1.25)],
)
def test_require_finite_real_accepts_python_and_numpy_reals(value: object) -> None:
    assert np.isfinite(require_finite_real(value, name='value'))


@pytest.mark.parametrize(
    'value',
    [True, np.bool_(False), 1 + 0j, np.complex128(1), '1.0', np.array(1.0)],
)
def test_require_finite_real_rejects_non_real_scalar_kinds(value: object) -> None:
    with pytest.raises(ValueError, match='value.*real numeric scalar'):
        require_finite_real(value, name='value')


@pytest.mark.parametrize('value', [np.nan, np.inf, -np.inf])
def test_require_finite_real_rejects_nonfinite_values(value: float) -> None:
    with pytest.raises(ValueError, match='value.*finite'):
        require_finite_real(value, name='value')


def test_finite_real_range_helpers_have_explicit_boundaries() -> None:
    assert require_positive_finite_real(0.5, name='scale') == 0.5
    assert require_nonnegative_finite_real(0.0, name='tol') == 0.0
    assert require_real_in_interval(
        1.0,
        name='fraction',
        lower=0.0,
        upper=1.0,
    ) == 1.0

    with pytest.raises(ValueError, match='scale.*positive'):
        require_positive_finite_real(0.0, name='scale')
    with pytest.raises(ValueError, match='tol.*non-negative'):
        require_nonnegative_finite_real(-1.0, name='tol')
    with pytest.raises(ValueError, match=r'fraction.*\[0.0, 1.0\]'):
        require_real_in_interval(
            1.5,
            name='fraction',
            lower=0.0,
            upper=1.0,
        )


def test_require_ordered_bounds_accepts_integer_values() -> None:
    bounds = require_ordered_bounds(
        [[0, 2], [-1, 3]],
        name='bounds',
        dim=2,
    )
    assert bounds == ((0.0, 2.0), (-1.0, 3.0))


@pytest.mark.parametrize(
    ('values', 'message'),
    [
        ([[0.0, 1.0]], 'shape'),
        ([[False, 1.0], [0.0, 1.0]], 'real numeric'),
        ([['0', '1'], ['0', '1']], 'real numeric'),
        ([[0 + 0j, 1 + 0j], [0, 1]], 'real numeric'),
        ([[0.0, np.nan], [0.0, 1.0]], 'finite'),
        ([[0.0, 0.0], [0.0, 1.0]], 'strictly ordered'),
        ([[-np.finfo(float).max, np.finfo(float).max], [0, 1]], 'finite'),
    ],
)
def test_require_ordered_bounds_rejects_invalid_values(
    values: object,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        require_ordered_bounds(values, name='bounds', dim=2)


def test_owned_readonly_array_detaches_from_caller_storage() -> None:
    source = np.array([1.0, 2.0])
    owned = owned_readonly_array(source, dtype=np.float64)
    source[0] = 10.0

    assert owned.tolist() == [1.0, 2.0]
    assert owned.flags.owndata
    assert not owned.flags.writeable


def test_checked_int64_add_rejects_overflow_before_arithmetic() -> None:
    maximum = np.iinfo(np.int64).max
    minimum = np.iinfo(np.int64).min

    with pytest.raises(ValueError, match='sum.*signed int64'):
        checked_int64_add(
            np.array([maximum], dtype=np.int64),
            np.array([1], dtype=np.int64),
            name='sum',
        )
    with pytest.raises(ValueError, match='sum.*signed int64'):
        checked_int64_add(
            np.array([minimum], dtype=np.int64),
            np.array([-1], dtype=np.int64),
            name='sum',
        )
