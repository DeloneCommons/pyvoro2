"""Audit regressions for original real categories and binary64 canonicalization."""
from fractions import Fraction
import warnings

import numpy as np
import pytest

from pyvoro2 import PeriodicCell
from pyvoro2._internal.inputs import (
    coerce_finite_matrix,
    coerce_finite_vector,
    coerce_point_array,
    coerce_real_vector,
)
from pyvoro2._internal.validation import (
    _as_original_array,
    _finite_float64_array,
    _is_real_numeric,
    _real_float64_array,
    require_finite_real,
    require_ordered_bounds,
)


METHODS = ('cart_to_fractional', 'fractional_to_cart',
           'wrap_fractional', 'wrap_cart')
UNITS = ('Y', 'M', 'W', 'D', 'h', 'm', 's', 'ms', 'us', 'ns', 'ps', 'fs', 'as')


def temporal_input(kind, unit, form):
    array = np.array([[1, 2, 3]], dtype=f'{kind}[{unit}]')
    if form == 'array':
        return array
    if form == 'nested-array':
        return [array[0]]
    if form == 'mixed-list':
        return [[array[0, 0], 0, 0.]]
    if form == 'mixed-tuple':
        return ((array[0, 0], np.int64(0), 0.),)
    result = np.empty((1, 3), dtype=object)
    result[0] = [array[0, 0], 0, 0.]
    return result


@pytest.mark.parametrize('method', METHODS)
@pytest.mark.parametrize('kind', ['datetime64', 'timedelta64'])
@pytest.mark.parametrize('unit', ['ns', 's'])
@pytest.mark.parametrize('form', ['array', 'nested-array', 'mixed-list',
                                  'mixed-tuple', 'object-array'])
def test_user_coordinates_reject_original_temporal_categories(method, kind, unit, form):
    operation = getattr(PeriodicCell(np.eye(3)), method)
    with pytest.raises(ValueError, match='real numeric'):
        operation(temporal_input(kind, unit, form))


@pytest.mark.parametrize('kind', ['datetime64', 'timedelta64'])
@pytest.mark.parametrize('unit', UNITS)
def test_shared_real_validators_reject_every_temporal_resolution(kind, unit):
    values = np.array([1, 2, 3], dtype=f'{kind}[{unit}]')
    assert not _is_real_numeric(values[0])
    with pytest.raises(ValueError, match='real numeric'):
        require_finite_real(values[0], name='scalar')
    with pytest.raises(ValueError, match='real numeric'):
        coerce_finite_vector(values, name='vector', n=3)
    with pytest.raises(ValueError, match='real numeric'):
        coerce_real_vector(values, name='derived vector', n=3)
    with pytest.raises(ValueError, match='real numeric'):
        coerce_finite_matrix(values[None, :], name='matrix', shape=(1, 3))
    with pytest.raises(ValueError, match='real numeric'):
        _finite_float64_array(values, name='array')
    with pytest.raises(ValueError, match='real numeric'):
        require_ordered_bounds(values[:2][None, :], name='bounds', dim=1)


@pytest.mark.parametrize('kind', ['datetime64', 'timedelta64'])
def test_empty_temporal_arrays_do_not_become_numeric_empty_batches(kind):
    values = np.empty((0, 3), dtype=f'{kind}[ns]')
    with pytest.raises(ValueError, match='real numeric'):
        coerce_point_array(values, name='points', dim=3)


@pytest.mark.parametrize('method', METHODS)
@pytest.mark.parametrize('as_object', [False, True])
def test_real_numeric_mixed_inputs_and_empty_batches_remain_supported(
    method, as_object,
):
    values = [[np.int64(2**53 + 1), np.float32(.25), Fraction(1, 2)]]
    if as_object:
        values = np.array(values, dtype=object)
    operation = getattr(PeriodicCell(np.eye(3)), method)
    result = operation(values)
    first = 0. if method.startswith('wrap') else float(2**53)
    assert result.tolist() == [[first, .25, .5]]
    assert result.dtype == np.float64 and result.flags.owndata
    empty = operation(np.empty((0, 3), dtype=object if as_object else np.float64))
    assert empty.shape == (0, 3) and empty.dtype == np.float64
    assert _is_real_numeric(Fraction(1, 2))
    assert require_finite_real(np.float32(.25), name='scalar') == .25


@pytest.mark.parametrize('method', METHODS)
@pytest.mark.parametrize('value', [
    True, np.bool_(False), 1j, '1', b'1', None, np.nan, np.inf, -np.inf,
])
def test_existing_prohibited_coordinate_categories_stay_rejected(method, value):
    with pytest.raises(ValueError):
        getattr(PeriodicCell(np.eye(3)), method)([[value, 0, 0.]])


def test_object_conversion_still_preserves_mixed_python_categories():
    original = _as_original_array([[False, 1., np.int64(2)]], name='points')
    assert type(original[0, 0]) is bool
    assert type(original[0, 1]) is float
    assert isinstance(original[0, 2], np.int64)
    with pytest.raises(ValueError, match='real numeric'):
        _finite_float64_array(original, name='points')


EXTENDED_RANGE = np.finfo(np.longdouble).maxexp > np.finfo(np.float64).maxexp


@pytest.mark.skipif(not EXTENDED_RANGE, reason='longdouble has no extended range')
@pytest.mark.parametrize('method', METHODS)
@pytest.mark.parametrize('mode', ['raise', 'warn'])
@pytest.mark.parametrize('form', ['array', 'object-array', 'list'])
def test_canonical_input_overflow_is_a_warning_free_value_error(method, mode, form):
    values = [[np.longdouble('1e400'), 0, 0]]
    if form != 'list':
        dtype = object if form == 'object-array' else np.longdouble
        values = np.array(values, dtype=dtype)
    operation = getattr(PeriodicCell(np.eye(3)), method)
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        with np.errstate(all=mode):
            policy = np.geterr()
            with pytest.raises(ValueError, match='finite'):
                operation(values)
            assert np.geterr() == policy


@pytest.mark.skipif(not EXTENDED_RANGE, reason='longdouble has no extended range')
@pytest.mark.parametrize('method', METHODS)
@pytest.mark.parametrize('mode', ['raise', 'warn'])
@pytest.mark.parametrize(('source', 'expected'), [('1e-320', 1e-320), ('1e-400', 0.)])
def test_canonical_subnormal_or_zero_is_independent_of_error_settings(
    method, mode, source, expected,
):
    values = np.array([[np.longdouble(source), .25, .5]], dtype=np.longdouble)
    assert np.isfinite(values[0, 0]) and values[0, 0] > 0
    operation = getattr(PeriodicCell(np.eye(3)), method)
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        with np.errstate(all=mode):
            policy = np.geterr()
            result = operation(values)
            assert np.geterr() == policy
    assert result.tolist() == [[expected, .25, .5]]


@pytest.mark.parametrize('nonfinite', [np.nan, np.inf, -np.inf])
def test_finite_and_derived_array_validators_keep_distinct_nonfinite_policies(
    nonfinite,
):
    values = np.array([nonfinite], dtype=np.longdouble)
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        with np.errstate(all='raise'):
            with pytest.raises(ValueError, match='finite'):
                coerce_finite_vector(values, name='finite', n=1)
            derived = coerce_real_vector(values, name='derived', n=1)
            raw_derived = _real_float64_array(values, name='derived')
    if np.isnan(nonfinite):
        assert np.isnan(derived[0]) and np.isnan(raw_derived[0])
    else:
        assert derived[0] == nonfinite and raw_derived[0] == nonfinite


@pytest.mark.skipif(not EXTENDED_RANGE, reason='longdouble has no extended range')
def test_shared_conversion_retains_derived_overflow_for_its_consumer():
    values = np.array([np.longdouble('1e400')])
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        with np.errstate(all='raise'):
            assert coerce_real_vector(values, name='derived', n=1).tolist() == [np.inf]
            with pytest.raises(ValueError, match='finite'):
                coerce_finite_vector(values, name='finite', n=1)
