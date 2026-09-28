"""Warning callbacks cannot poison protected floating-point continuation."""
from __future__ import annotations

import subprocess
import sys
import textwrap

import pytest


@pytest.mark.skipif(sys.platform != 'linux', reason='GNU fenv trap controls')
@pytest.mark.parametrize('dimension', [2, 3])
@pytest.mark.parametrize('warning', ['persistent', 'query', 'scale'])
@pytest.mark.parametrize('callback_raises', [False, True])
def test_warning_callback_state_is_checked(dimension, warning, callback_raises):
    script = r'''
        import ctypes
        import warnings
        import pyvoro2
        import pyvoro2.planar
        api = pyvoro2.planar if DIM == 2 else pyvoro2
        box = api.Box(((.1, 1.3),) * DIM)
        points = [[.2] * DIM, [.2001] * DIM]
        kwargs = dict(domain=box, duplicate_check='warn',
                      duplicate_threshold=.001)
        if KIND == 'query':
            points = points[:1]
            kwargs['queries'] = [[.2001] * DIM]
            method = api.ghost_cells
        elif KIND == 'scale':
            points = [[.00002] * DIM]
            kwargs['domain'] = api.Box(((0., .0001),) * DIM)
            kwargs['duplicate_check'] = 'off'
            method = api.compute
        else:
            method = api.compute
        libc = ctypes.CDLL(None)
        saved = ctypes.create_string_buffer(256)
        assert libc.fegetenv(saved) == 0
        invoked = []
        def poison(*args, **kwargs):
            invoked.append(True)
            libc.feenableexcept(32)
            if RAISES:
                raise RuntimeError('warning callback raised')
        warnings.showwarning = poison
        warnings.simplefilter('always')
        refused = False
        try:
            try:
                method(points, **kwargs)
            except (ValueError, RuntimeError) as error:
                refused = 'UNSUPPORTED' in str(error)
            assert libc.fegetexcept() & 32, 'caller controls were normalized'
        finally:
            assert libc.fesetenv(saved) == 0
        assert invoked, 'warning callback was not exercised'
        assert refused, 'warning callback escaped structured runtime refusal'
    '''
    result = subprocess.run(
        [sys.executable, '-c', f'DIM={dimension}; KIND={warning!r}; '
         f'RAISES={callback_raises}\n' + textwrap.dedent(script)],
        capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr
