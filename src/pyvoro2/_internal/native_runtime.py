"""Non-mutating FP entry and guarded foreign numeric conversion.

Package setup retains the tiny raw-control adapter; geometry backends remain
lazy. Forward entry loads its owning backend before numeric preparation, then
checks the current thread after foreign callbacks. Mutable FP state is not cached.
"""
from __future__ import annotations

import operator
import sys
import warnings
from functools import wraps
from importlib import import_module

import numpy as np
from .ghost_failure import GhostFailure
from .locate_failure import LocateFailure


# Importing Python/NumPy itself belongs to interpreter/package setup. Retain
# this tiny adapter during that setup, keeping the geometry backends lazy.
# Pure-Python use from a source-only checkout remains possible without it.
try:
    _early_guard = import_module('pyvoro2._fpguard')
except ImportError:
    _early_guard = None
else:
    _early_guard._require_runtime_environment()
    from .native_qualification import register_native
    register_native(_early_guard)


class RuntimeFPError(RuntimeError):
    """Private refusal that must not be relabelled as a malformed input."""


def check_before_native_import():
    if _early_guard is None:
        raise RuntimeFPError('preloaded native FP control adapter is unavailable')
    try:
        _early_guard._require_runtime_environment()
    except ValueError as exc:
        raise RuntimeFPError(str(exc)) from exc


def check_environment() -> None:
    """Check a loaded native adapter, without loading one for pure algebra."""
    for name in ('pyvoro2._core', 'pyvoro2._core2d'):
        module = sys.modules.get(name)
        if module is not None:
            try:
                module._require_runtime_environment()
            except ValueError as exc:
                raise RuntimeFPError(str(exc)) from exc
            return


def checked_call(function, /, *args, **kwargs):
    """Recheck even when foreign code raises before returning a value."""
    check_environment()
    try:
        return function(*args, **kwargs)
    finally:
        check_environment()


def checked_warn(message, category=None, *, stacklevel=1, source=None):
    """Treat user-installed warning delivery as a foreign-state boundary."""
    checked_call(warnings.warn, message, category, stacklevel=stacklevel + 2,
                 source=source)


def _attribute(value, name, default=None):
    return checked_call(getattr, value, name, default)


def checked_float(value):
    """Separate foreign __index__ from subsequent integer-to-float rounding."""
    check_environment()
    method = _attribute(type(value), '__float__')
    if method is not None:
        result = checked_call(method, value)
        if not isinstance(result, float):
            raise TypeError('__float__ returned a non-float')
        return checked_call(float, result)
    index = _attribute(type(value), '__index__')
    if index is not None:
        integer = checked_call(operator.index, value)
        return checked_call(float, integer)
    # Native NumPy forcecast accepts numeric strings. Public strict-kind
    # validators still reject them before reaching this conversion helper.
    return checked_call(float, value)


def checked_index(value):
    return checked_call(operator.index, value)


def checked_tuple(values):
    """Observe iterator acquisition and every next() return separately."""
    iterator = checked_call(iter, values)
    result = []
    while True:
        try:
            result.append(checked_call(next, iterator))
        except StopIteration:
            return tuple(result)


def _plain_interface(value):
    """Freeze array-interface metadata before NumPy reads it internally."""
    kind = type(value)
    if value is None or kind in (str, bytes, int, bool, float):
        return value
    if isinstance(value, str):
        return str.__str__(value)
    if isinstance(value, dict):
        return {_plain_interface(k): _plain_interface(v)
                for k, v in checked_tuple(checked_call(value.items))}
    if isinstance(value, (list, tuple)):
        return tuple(_plain_interface(v) for v in checked_tuple(value))
    if _attribute(type(value), '__index__') is not None:
        return checked_index(value)
    return value


class _Interface:
    """Keep the original buffer owner alive behind inert protocol metadata."""

    def __init__(self, owner, name, metadata):
        self.owner = owner
        setattr(self, name, metadata)


def _base_array(value):
    # Explicit base descriptor avoids ndarray-subclass callbacks/properties.
    return np.ndarray.view(value, np.ndarray)


def _array_protocol(value):
    """Resolve each foreign protocol before NumPy performs any conversion."""
    if isinstance(value, np.ndarray):
        return _base_array(value)
    struct = _attribute(value, '__array_struct__')
    if struct is not None:
        proxy = _Interface(value, '__array_struct__', struct)
        return checked_call(np.asarray, proxy)
    interface = _attribute(value, '__array_interface__')
    if interface is not None:
        proxy = _Interface(value, '__array_interface__',
                           _plain_interface(interface))
        return checked_call(np.asarray, proxy)
    method = _attribute(value, '__array__')
    if method is not None:
        result = checked_call(method, np.dtype(object))
        if not isinstance(result, np.ndarray):
            raise ValueError('__array__ must return an ndarray')
        return _base_array(result)
    # Buffer acquisition can itself call an exporter. Establish the controls
    # again before NumPy sees the callback-free memoryview it returned.
    try:
        buffer = checked_call(memoryview, value)
    except TypeError:
        return None
    return checked_call(np.asarray, buffer)


def _object_tree(value, *, nested=False):
    """Shape and references without letting NumPy call foreign protocols."""
    check_environment()
    array = _array_protocol(value)
    if array is not None:
        if nested and array.ndim == 0:
            # A zero-dimensional array in a scalar position retains its
            # category for strict public validators. Root arrays and nested
            # nonzero-dimensional row arrays still supply their shapes.
            return (), [value]
        # Only a base ndarray reaches this conversion; no user callback is
        # nested inside its numeric-to-object conversion.
        objects = checked_call(np.asarray, array, dtype=object)
        return objects.shape, list(objects.flat)
    if isinstance(value, (str, bytes)) or np.isscalar(value):
        return (), [value]
    getitem = _attribute(type(value), '__getitem__')
    length = _attribute(type(value), '__len__')
    if getitem is None or length is None:
        return (), [value]
    count = checked_call(len, value)
    children = []
    raw = []
    for index in range(count):
        child = checked_call(operator.getitem, value, index)
        raw.append(child)
        children.append(_object_tree(child, nested=True))
    if not children:
        return (0,), []
    shape = children[0][0]
    if any(item[0] != shape for item in children[1:]):
        # Preserve NumPy's object-array ragged category for strict public
        # shape/kind validation, without a second protocol invocation.
        return (count,), raw
    leaves = []
    for _, items in children:
        leaves.extend(items)
    return (count,) + shape, leaves


def original_array(values):
    """Preserve scalar categories while guarding all protocol returns."""
    check_environment()
    shape, leaves = _object_tree(values)
    result = np.empty(len(leaves), dtype=object)
    for index, value in enumerate(leaves):
        result[index] = value
    return result.reshape(shape)


def numeric_array(values, *, dtype):
    """Convert scalar by scalar when Python numeric callbacks are possible."""
    check_environment()
    if isinstance(values, np.ndarray):
        base = _base_array(values)
        if base.dtype.kind != 'O':
            return checked_call(np.asarray, base, dtype=dtype, order='C')
        objects = base
    else:
        objects = original_array(values)
    kind = np.dtype(dtype).kind
    convert = (checked_float if kind == 'f' else
               checked_index if kind in 'iu' else
               lambda x: checked_call(bool, x))
    converted = [convert(item) for item in objects.flat]
    check_environment()
    return np.asarray(converted, dtype=dtype).reshape(objects.shape)


def prepare_native_argument(value, kind):
    """Return callback-free values for the hidden typed pybind closure."""
    # Direct imports of a qualification extension need the same common raw
    # inspector during foreign callbacks, even without a public API call.
    from pyvoro2 import _core
    _core._require_runtime_environment()
    if kind.endswith('_array'):
        dtypes = {'float_array': np.float64, 'int_array': np.int32,
                  'bool_array': np.bool_}
        return numeric_array(value, dtype=dtypes[kind])
    if kind.endswith('_sequence'):
        scalar = kind.removesuffix('_sequence')

        def convert(item):
            if isinstance(item, (list, tuple, np.ndarray)):
                return tuple(convert(v) for v in checked_tuple(item))
            return prepare_native_argument(item, scalar)
        return tuple(convert(v) for v in checked_tuple(value))
    if kind == 'float':
        return checked_float(value)
    if kind == 'int':
        return checked_index(value)
    if kind == 'bool':
        return checked_call(bool, value)
    if kind == 'object':
        return value
    raise ValueError('unknown guarded native argument kind: ' + kind)


def admit_native_entry(module, name, canonical_args):
    """Compose artifact requirements before entering native numeric work."""
    dimension = 2 if module.__name__.endswith('_core2d') else 3
    from .native_qualification import (
        NativeQualificationError, register_native, require_native,
    )
    component = None
    if name.startswith('_observe_ghost_'):
        if len(canonical_args[-3]):
            component = 'wp7-spatial'
    elif name.startswith('_observe_'):
        component = 'wp5-spatial'
    elif name.startswith('_compute_') and name.endswith('_witness'):
        component = 'wp6-planar' if dimension == 2 else 'wp5-spatial'
    elif name.startswith('_ghost_') and name.endswith('_witness'):
        queries = canonical_args[-2 if 'power' in name else -1]
        if len(queries):
            component = 'wp7-planar' if dimension == 2 else 'wp7-spatial'
    elif name.startswith('locate_') or name.startswith('_locate_'):
        # The final canonical boolean is return_source. ID-only calls do not
        # consume enclosure certificates. Empty-query safety remains owned by
        # the native/public zero-work branch.
        if canonical_args and canonical_args[-1] is True:
            queries = canonical_args[-2]
            if len(queries):
                component = 'wp8-planar' if dimension == 2 else 'wp8-spatial'
    if component is not None:
        try:
            register_native(module)
            require_native(module, component)
        except NativeQualificationError as exc:
            if component.startswith('wp7-'):
                raise ValueError('ghost_native:native:None:GHOST_NATIVE_UNSUPPORTED:'
                                 + str(exc)) from exc
            if component.startswith('wp8-'):
                raise ValueError('locate_native:profile:None:LOCATE_NATIVE_UNSUPPORTED:'
                                 + str(exc)) from exc
            if dimension == 2:
                raise RuntimeError('planar_certification:profile:' + exc.reason +
                                   ': ' + exc.detail) from exc
            raise


def _public_refusal(error, operation, dimension):
    """Construct structured refusal without inspecting or converting inputs."""
    if operation == 'locate':
        raise LocateFailure('LOCATE_NATIVE_UNSUPPORTED', str(error),
                            stage='profile', reason='runtime_fp') from error
    if operation == 'ghost':
        raise GhostFailure('GHOST_NATIVE_UNSUPPORTED', str(error),
                           stage='native', dimension=dimension,
                           reason='runtime_fp') from error
    module = import_module('pyvoro2.planar.diagnostics' if dimension == 2
                           else 'pyvoro2.diagnostics')
    measure, boundary = ('area', 'edge') if dimension == 2 else ('volume', 'face')
    code = ('WP6_PROFILE_UNSUPPORTED' if dimension == 2
            else 'WP5_UNSUPPORTED_FP_PROFILE')
    issue = module.TessellationIssue(
        code, 'error', str(error), ({'stage': 'profile', 'reason': 'runtime_fp'},),
    )
    diagnostic = module.TessellationDiagnostics(**{
        'domain_' + measure: np.nan, 'sum_cell_' + measure: np.nan,
        measure + '_ratio': np.nan, measure + '_gap': np.nan,
        measure + '_overlap': np.nan, 'n_sites_expected': 0,
        'n_cells_returned': 0, 'missing_ids': (), 'empty_ids': (),
        boundary + '_shift_available': False, 'reciprocity_checked': False,
        'n_' + boundary + 's_total': 0, 'n_' + boundary + 's_orphan': 0,
        'n_' + boundary + 's_mismatched': 0, 'issues': (issue,),
        'ok_' + measure: False, 'ok_reciprocity': False, 'ok': False,
    })
    raise module.TessellationError(code + ': ' + str(error), diagnostic) from error


def guarded_public(operation, dimension):
    """Enter the current thread's FP envelope before any public coercion."""
    def decorate(function):
        @wraps(function)
        def call(*args, **kwargs):
            try:
                check_before_native_import()
                import_module('pyvoro2._core2d' if dimension == 2 else 'pyvoro2._core')
                check_environment()
                return function(*args, **kwargs)
            except RuntimeFPError as exc:
                _public_refusal(exc, operation, dimension)
        return call
    return decorate
