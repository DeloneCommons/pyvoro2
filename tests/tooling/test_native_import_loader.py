"""Qualification tools preserve canonical identity for exact production paths."""
from __future__ import annotations

from pathlib import Path
import subprocess
import sys
import textwrap

import pytest


ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize('tool,short,label', [
    ('check_planar_witness.py', '_core2d', 'production'),
    ('check_ghost_planar.py', '_core2d', 'production'),
    ('check_locate_source.py', '_core', 'pyvoro2._core'),
    ('check_locate_source.py', '_core2d', 'pyvoro2._core2d'),
])
def test_tool_loads_exact_production_path_and_refuses_other_live_copy(
        tmp_path, tool, short, label):
    script = r'''
        import importlib.util
        from pathlib import Path
        import shutil
        import sys
        import pyvoro2
        from pyvoro2._internal import native_qualification as q

        root, directory, tool, short, label = sys.argv[1:]
        sys.path.insert(0, str(Path(root) / 'tools/native'))
        tool_spec = importlib.util.spec_from_file_location(
            'qualification_tool', Path(root) / 'tools/native' / tool)
        helper = importlib.util.module_from_spec(tool_spec)
        tool_spec.loader.exec_module(helper)
        name = 'pyvoro2.' + short
        source = Path(importlib.util.find_spec(name).origin)
        requested = Path(directory) / source.name
        shutil.copyfile(source, requested)
        assert requested.parent not in map(Path, sys.path)
        module = helper.load(requested, label)
        assert module.__name__ == name
        assert sys.modules[name] is module
        assert getattr(pyvoro2, short) is module
        assert Path(module.__file__).resolve() == requested.resolve()
        assert module in q._verifier().registrations
        assert helper.load(requested, label) is module
        try:
            helper.load(source, label)
        except ValueError as error:
            assert 'different path' in str(error), error
        else:
            raise AssertionError('another live production path was accepted')
    '''
    result = subprocess.run(
        [sys.executable, '-c', textwrap.dedent(script), str(ROOT), str(tmp_path),
         tool, short, label], capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, (result.returncode, result.stdout, result.stderr)


@pytest.mark.parametrize('short', ['_core', '_core2d'])
def test_failed_production_import_is_not_published_and_can_retry(short):
    script = r'''
        import importlib.util
        from pathlib import Path
        import sys
        import pyvoro2
        from pyvoro2._internal import native_qualification as q

        root, short = sys.argv[1:]
        sys.path.insert(0, str(Path(root) / 'tools/native'))
        from qualification.native_import import load_production_module
        name = 'pyvoro2.' + short
        path = importlib.util.find_spec(name).origin
        original = q.register_native

        def fail(module):
            raise RuntimeError('injected import registration failure')

        q.register_native = fail
        try:
            load_production_module(path, name)
        except ImportError as error:
            assert 'injected import registration failure' in str(error.__cause__)
        else:
            raise AssertionError('failed native registration was accepted')
        finally:
            q.register_native = original
        assert name not in sys.modules
        assert not hasattr(pyvoro2, short)
        module = load_production_module(path, name)
        assert sys.modules[name] is module
        assert getattr(pyvoro2, short) is module
        assert module in q._verifier().registrations
    '''
    result = subprocess.run(
        [sys.executable, '-c', textwrap.dedent(script), str(ROOT), short],
        capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, (result.returncode, result.stdout, result.stderr)
