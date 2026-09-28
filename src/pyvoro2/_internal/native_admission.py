"""Shared artifact admission and current-thread checks for certificate routes.

The native module's self-description is consistency data. Only the external
qualification verifier decides whether the imported artifact may produce a
component's evidence; runtime control registers are checked on every entry.
"""
from __future__ import annotations

from importlib import import_module

from . import native_qualification
from .native_runtime import RuntimeFPError, check_before_native_import


def require_environment(module=None):
    """Refuse incompatible controls without changing controls or sticky flags."""
    try:
        if module is None:
            check_before_native_import()
        else:
            module._require_runtime_environment()
    except (RuntimeFPError, ValueError) as exc:
        raise native_qualification.NativeQualificationError(
            'runtime_environment', str(exc),
        ) from exc


def require_component(component):
    """Admit the actual loaded module, then recheck its executing-thread state."""
    require_environment()
    name = 'pyvoro2._core2d' if component.endswith('-planar') else 'pyvoro2._core'
    try:
        module = import_module(name)
    finally:
        require_environment()
    require_environment(module)
    try:
        native_qualification.register_native(module)
        native_qualification.require_native(module, component)
    finally:
        require_environment(module)
    return module
