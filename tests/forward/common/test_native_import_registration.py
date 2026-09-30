"""Native lifetime identity is frozen by real imports before foreign callbacks."""
from __future__ import annotations

import subprocess
import sys
import textwrap

import pytest


def _child(script, *arguments):
    result = subprocess.run(
        [sys.executable, '-c', textwrap.dedent(script), *map(str, arguments)],
        capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, (result.returncode, result.stdout, result.stderr)


@pytest.mark.parametrize('short', ['_core', '_core2d'])
def test_direct_import_registers_native_lifetime_before_return(short):
    _child('''
        import importlib
        from pathlib import Path
        import sys
        import pyvoro2
        from pyvoro2._internal import native_qualification as q

        name = 'pyvoro2.' + sys.argv[1]
        assert 'pyvoro2._core' not in sys.modules
        assert 'pyvoro2._core2d' not in sys.modules
        original = q.register_native
        seen = []

        def register(module):
            assert sys.modules[name] is module
            assert module.__spec__.name == name
            assert Path(module.__spec__.origin) == Path(module.__file__)
            original(module)
            seen.append(module)

        q.register_native = register
        module = importlib.import_module(name)
        assert seen == [module]
        assert module in q._verifier().registrations
        assert not any(item.startswith('_planar_candidate_') for item in dir(module))
    ''', short)


@pytest.mark.parametrize('dimension', [2, 3])
def test_public_lazy_import_registers_before_first_foreign_conversion(dimension):
    _child('''
        import sys
        import pyvoro2
        from pyvoro2._internal import native_qualification as q

        dimension = int(sys.argv[1])
        api = pyvoro2.planar if dimension == 2 else pyvoro2
        name = 'pyvoro2._core2d' if dimension == 2 else 'pyvoro2._core'
        assert name not in sys.modules
        domain = api.Box(((0., 1.),) * dimension)

        class Observed(Exception):
            pass

        class Points:
            def __array__(self, *args, **kwargs):
                assert sys.modules[name] in q._verifier().registrations
                raise Observed('registered before numeric conversion')

        try:
            api.compute(Points(), domain=domain)
        except Observed:
            pass
        else:
            raise AssertionError('foreign conversion was not reached')
    ''', dimension)


@pytest.mark.skipif(sys.platform == 'win32', reason='Windows locks loaded DLL files')
@pytest.mark.parametrize('short,component', [
    ('_core', 'wp5-spatial'), ('_core2d', 'wp6-planar'),
])
def test_replacement_before_first_admission_refuses_without_linux_mapping(
        tmp_path, short, component):
    _child('''
        import importlib.util
        from pathlib import Path
        import shutil
        import sys
        import pyvoro2
        from pyvoro2._internal import native_admission
        from pyvoro2._internal import native_qualification as q

        name, component, directory = sys.argv[1:]
        source = Path(importlib.util.find_spec(name).origin)
        path = Path(directory) / source.name
        shutil.copyfile(source, path)
        # Exercise the import-lifetime check used on reviewed platforms that
        # do not have Linux's additional /proc loader-mapping predicate.
        q._require_mapping = lambda path, identity: None
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
        original_identity = module._qualification_identity()
        replacement = path.with_suffix('.replacement')
        replacement.write_bytes(path.read_bytes() + b'changed installed payload')
        replacement.replace(path)
        assert module._qualification_identity() == original_identity
        try:
            native_admission.require_component(component)
        except q.NativeQualificationError as error:
            assert error.reason == 'payload_mismatch', error
        else:
            raise AssertionError('changed import lifetime was admitted')
    ''', 'pyvoro2.' + short, component, tmp_path)


@pytest.mark.parametrize('short', ['_core', '_core2d'])
def test_production_module_cannot_skip_registration_through_alias(short):
    _child('''
        import importlib.util
        import sys
        import pyvoro2
        from pyvoro2._internal import native_qualification as q

        short = sys.argv[1]
        path = importlib.util.find_spec('pyvoro2.' + short).origin
        name = 'unregistered_alias.' + short
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        try:
            spec.loader.exec_module(module)
        except ImportError as error:
            assert isinstance(error.__cause__, q.NativeQualificationError), error
            assert error.__cause__.reason == 'payload_mismatch'
        else:
            raise AssertionError('production alias skipped import registration')
        finally:
            del sys.modules[name]
    ''', short)
