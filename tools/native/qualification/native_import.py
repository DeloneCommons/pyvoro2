"""Load exact production payloads with their canonical native import identity."""
from __future__ import annotations

import importlib.util
from pathlib import Path
import sys


def load_production_module(path, name):
    if name not in ('pyvoro2._core', 'pyvoro2._core2d'):
        raise ValueError('unknown production native module')
    # Package setup retains the installation's early FP guard and verifier.
    # The requested payload may be outside sys.path in a controlled build tree;
    # its explicit path grants no qualification or artifact admission.
    import pyvoro2
    from pyvoro2._internal.native_runtime import check_before_native_import
    from pyvoro2._internal.native_qualification import register_native

    check_before_native_import()
    path = Path(path).resolve(strict=True)
    existing = sys.modules.get(name)
    if existing is not None:
        if Path(existing.__file__).resolve() != path:
            raise ValueError(
                'production native module already loaded from a different path')
        try:
            register_native(existing)
        finally:
            check_before_native_import()
        return existing
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        try:
            spec.loader.exec_module(module)
        finally:
            check_before_native_import()
    except BaseException:
        if sys.modules.get(name) is module:
            del sys.modules[name]
        raise
    setattr(pyvoro2, name.rsplit('.', 1)[1], module)
    return module
