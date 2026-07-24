"""Private-helper ownership and obsolete-module absence contracts."""

from __future__ import annotations

import ast
import importlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import pytest

import pyvoro2
import pyvoro2.inverse as inverse
import pyvoro2.inverse.separator as separator


PACKAGE_ROOT = Path(__file__).resolve().parents[2] / 'src' / 'pyvoro2'
INTERNAL_ROOT = PACKAGE_ROOT / '_internal'
INTERNAL_MODULES = (
    'pyvoro2._internal.cell_output',
    'pyvoro2._internal.inputs',
    'pyvoro2._internal.power_input',
    'pyvoro2._internal.weight_transforms',
    'pyvoro2._internal.spatial.domain_geometry',
    'pyvoro2._internal.spatial.domain_utils',
    'pyvoro2._internal.spatial.face_shifts',
    'pyvoro2._internal.planar.domain_geometry',
    'pyvoro2._internal.planar.edge_shifts',
)
OBSOLETE_MODULES = (
    'pyvoro2._cell_output',
    'pyvoro2._inputs',
    'pyvoro2._power_input',
    'pyvoro2._weight_transforms',
    'pyvoro2._domain_geometry',
    'pyvoro2._face_shifts3d',
    'pyvoro2._util',
    'pyvoro2.planar._domain_geometry',
    'pyvoro2.planar._edge_shifts2d',
)


def _module_name(path: Path) -> str:
    relative = path.relative_to(PACKAGE_ROOT).with_suffix('')
    parts = relative.parts
    if parts[-1] == '__init__':
        parts = parts[:-1]
    return '.'.join(('pyvoro2', *parts))


def _internal_dependencies(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding='utf-8'), filename=str(path))
    module_name = _module_name(path)
    package_name = (
        module_name
        if path.name == '__init__.py'
        else module_name.rpartition('.')[0]
    )
    dependencies: set[str] = set()
    for node in ast.walk(tree):
        names: tuple[str, ...]
        if isinstance(node, ast.Import):
            names = tuple(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            imported = '.' * node.level + (node.module or '')
            names = (importlib.util.resolve_name(imported, package_name),)
        else:
            continue
        dependencies.update(
            name
            for name in names
            if name == 'pyvoro2._internal'
            or name.startswith('pyvoro2._internal.')
        )
    return dependencies


def test_internal_modules_have_concrete_import_paths() -> None:
    for module_name in INTERNAL_MODULES:
        assert importlib.import_module(module_name).__name__ == module_name


def test_obsolete_private_module_paths_are_absent() -> None:
    for module_name in OBSOLETE_MODULES:
        spec = importlib.util.find_spec(module_name)
        # A pre-move scikit-build editable finder can retain a stale spec until
        # the environment is reinstalled. It must point only to the deleted
        # source path, and importing it must still fail. Clean-wheel validation
        # requires the stricter no-spec behavior.
        assert spec is None or (
            spec.origin is not None
            and not Path(spec.origin).exists()
        )
        with pytest.raises(
            (ModuleNotFoundError, FileNotFoundError),
        ) as exc_info:
            importlib.import_module(module_name)
        if isinstance(exc_info.value, ModuleNotFoundError):
            assert exc_info.value.name == module_name


def test_internal_initializers_do_not_reexport_helpers() -> None:
    for relative in (
        Path('__init__.py'),
        Path('spatial/__init__.py'),
        Path('planar/__init__.py'),
    ):
        path = INTERNAL_ROOT / relative
        tree = ast.parse(path.read_text(encoding='utf-8'), filename=str(path))
        assert not any(
            isinstance(node, (ast.Import, ast.ImportFrom))
            for node in ast.walk(tree)
        )


def test_internal_dependency_graph_is_acyclic_and_has_clean_direction() -> None:
    paths = sorted(INTERNAL_ROOT.rglob('*.py'))
    graph = {
        _module_name(path): _internal_dependencies(path)
        for path in paths
    }
    for path in paths:
        source = path.read_text(encoding='utf-8')
        assert 'powerfit' not in source
        assert 'pyvoro2._core' not in source
        assert 'pyvoro2._core2d' not in source

    pending = {module: set(dependencies) for module, dependencies in graph.items()}
    resolved: set[str] = set()
    while pending:
        ready = {
            module
            for module, dependencies in pending.items()
            if dependencies <= resolved
        }
        assert ready, f'internal import cycle: {pending}'
        resolved.update(ready)
        pending = {
            module: dependencies
            for module, dependencies in pending.items()
            if module not in ready
        }


def test_public_weight_transform_exports_remain_identical() -> None:
    transforms = importlib.import_module('pyvoro2._internal.weight_transforms')
    for public_module in (pyvoro2, inverse, separator):
        assert public_module.weights_to_radii is transforms.weights_to_radii
        assert public_module.radii_to_weights is transforms.radii_to_weights


def test_internal_module_imports_keep_native_extensions_lazy() -> None:
    code = """
import importlib
import importlib.abc
import json
import sys

module_names = json.loads(sys.argv[1])
native_extensions = {'pyvoro2._core', 'pyvoro2._core2d'}
attempted = []

class NativeImportRecorder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname in native_extensions:
            attempted.append(fullname)
        return None

sys.meta_path.insert(0, NativeImportRecorder())
for module_name in module_names:
    importlib.import_module(module_name)
print(json.dumps({
    'attempted': attempted,
    'loaded': sorted(native_extensions.intersection(sys.modules)),
}))
"""
    completed = subprocess.run(
        [sys.executable, '-c', code, json.dumps(INTERNAL_MODULES)],
        cwd=PACKAGE_ROOT.parents[1],
        check=True,
        capture_output=True,
        text=True,
    )
    assert json.loads(completed.stdout) == {
        'attempted': [],
        'loaded': [],
    }
