"""A foreign conversion must not poison the next package-owned FP operation."""

from __future__ import annotations

import subprocess
import sys
import textwrap

import pytest
import numpy as np


def test_guarded_array_preserves_existing_buffer_input():
    from pyvoro2._internal.native_runtime import original_array, numeric_array
    source = np.array([[.25, .5, .75]])
    assert original_array(memoryview(source)).shape == source.shape
    assert np.array_equal(numeric_array(memoryview(source), dtype=np.float64),
                          source)


@pytest.mark.skipif(sys.platform != 'linux', reason='GNU fenv adversarial controls')
@pytest.mark.parametrize('dimension', [2, 3])
@pytest.mark.parametrize('operation', ['compute', 'locate', 'ghost_cells'])
@pytest.mark.parametrize('preload', [False, True])
@pytest.mark.parametrize('trap', [1, 4, 8, 16, 32])
def test_public_entry_refuses_before_input_callbacks(dimension, operation,
                                                     preload, trap):
    script = r'''
        import ctypes
        import pyvoro2
        import pyvoro2.planar
        if PRELOAD:
            from pyvoro2 import _core, _core2d
        api = pyvoro2.planar if DIM == 2 else pyvoro2
        libc = ctypes.CDLL(None)
        saved = ctypes.create_string_buffer(256)
        assert libc.fegetenv(saved) == 0
        touched = []
        class Inputs:
            def __array__(self, dtype=None, copy=None):
                touched.append(True)
                raise AssertionError('input callback before hostile-state refusal')
        method = getattr(api, OP)
        kwargs = {'domain': None}
        if OP != 'compute':
            kwargs['queries'] = Inputs()
        refused = False
        try:
            libc.feenableexcept(TRAP)
            try:
                method(Inputs(), **kwargs)
            except (ValueError, RuntimeError) as error:
                refused = 'UNSUPPORTED' in str(error)
        finally:
            assert libc.fesetenv(saved) == 0
        assert not touched
        assert refused, 'public entry did not preserve structured refusal'
    '''
    result = subprocess.run(
        [sys.executable, '-c', f'DIM={dimension}; OP={operation!r}; '
         f'PRELOAD={preload}; TRAP={trap}\n' +
         textwrap.dedent(script)], capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr


@pytest.mark.skipif(sys.platform != 'linux', reason='GNU fenv adversarial controls')
@pytest.mark.parametrize('case', ['scalar', 'index', 'array', 'object_array'])
def test_foreign_conversion_state_change_is_refused(case):
    # If guards are removed, conversion succeeds under FE_UPWARD. All state
    # changes stay inside this child; the test runner never changes its fenv.
    script = r'''
        import ctypes
        import numpy as np
        from pyvoro2 import _core
        from pyvoro2._internal import validation as v
        libc = ctypes.CDLL(None)
        before = libc.fegetround()
        class Scalar(float):
            def __float__(self):
                assert libc.fesetround(0x800) == 0
                return 1.0
        class Index:
            def __index__(self):
                assert libc.fesetround(0x800) == 0
                return 1
        class Array:
            def __array__(self, dtype=None, copy=None):
                assert libc.fesetround(0x800) == 0
                return original
        original = np.array([1.0])
        objects = np.empty(2, dtype=object)
        objects[:] = [Scalar(1), 2.0]
        actions = {
            'scalar': lambda: v.require_finite_real(Scalar(1), name='value'),
            'index': lambda: v.require_index(Index(), name='value'),
            'array': lambda: v._as_original_array(Array(), name='value'),
            'object_array': lambda: v._real_float64_array(objects, name='value'),
        }
        refused = False
        try:
            try:
                actions[CASE]()
            except (RuntimeError, ValueError):
                refused = True
            assert libc.fegetround() == 0x800, 'guard normalized caller state'
        finally:
            assert libc.fesetround(before) == 0
        assert refused, 'foreign conversion changed FP state without refusal'
    '''
    result = subprocess.run(
        [sys.executable, '-c', 'CASE = ' + repr(case) + '\n' +
         textwrap.dedent(script)], capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr


@pytest.mark.skipif(sys.platform != 'linux', reason='GNU fenv adversarial controls')
def test_array_protocol_trap_is_refused_before_numpy_cast():
    # A callback returns an existing f32 signalling NaN under unmasked invalid.
    # Guarding only after np.asarray(..., dtype=object) is too late: NumPy's
    # conversion itself raises SIGFPE. The expected refusal occurs after the
    # callback and before NumPy conversion. The child restores all saved fenv.
    script = r'''
        import ctypes
        import numpy as np
        from pyvoro2 import _core
        from pyvoro2._internal.validation import _as_original_array
        libc = ctypes.CDLL(None)
        saved = ctypes.create_string_buffer(256)
        assert libc.fegetenv(saved) == 0
        original = np.array([0x7f800001], dtype=np.uint32).view(np.float32)
        class Array:
            def __array__(self, dtype=None, copy=None):
                libc.feenableexcept(1)
                return original
        refused = False
        try:
            try:
                _as_original_array(Array(), name='points')
            except (RuntimeError, ValueError):
                refused = True
        finally:
            assert libc.fesetenv(saved) == 0
        assert refused, 'unsafe array coercion was admitted'
    '''
    result = subprocess.run(
        [sys.executable, '-c', textwrap.dedent(script)],
        capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr
